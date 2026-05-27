"""train_steerer_v4.py — V4 steerer with 21-channel SuperpositionSteererV3 + GPU features.

Proper channel alignment (6+7+8=21) with per-group MLPs.
GPU-vectorized KV cache, Topic prior, and POS features.
Warm-start from V2 best_b model, fresh V4 steerer.
"""
import sys, os, time, math, argparse, importlib.util, pickle
from pathlib import Path
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import defaultdict
import math as _math
from torch.utils.data import Dataset, DataLoader

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

_spec = importlib.util.spec_from_file_location('scaled', str(REPO / 'hybrid/train_scaled_neural_lm.py'))
_mod = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_mod); DeepCausalLM = _mod.DeepCausalLM
import sys as _sys; _sys.path.insert(0, str(REPO))  # restore after exec_module
from hybrid.superposition_steerer_v3 import SuperpositionSteererV3
_sys.path.insert(0, str(REPO))
from hybrid.gpu_channels import GPUFeatureComputer
from hybrid.backends import TrainableSurface, ZeroQPartitionedBackend, trainable_parameters

V = 50257
PUNCT_IDS = {0,2,3,4,5,7,8,9,10,11,12,13,14,25,26,27,28,29,30,31,58,60,61,90,91,92,93,198,220}
MODEL_CONFIGS = {
    '124m': dict(d_model=768, n_layers=12, n_heads=12, d_ff=3072, max_len=128),
    '500m': dict(d_model=1408, n_layers=18, n_heads=16, d_ff=5632, max_len=128),
    '1b':    dict(d_model=2048, n_layers=24, n_heads=16, d_ff=8192, max_len=128),
    '4b':    dict(d_model=3072, n_layers=40, n_heads=24, d_ff=12288, max_len=128),
    '700m':  dict(d_model=1536, n_layers=22, n_heads=16, d_ff=6144, max_len=512),
}

class StreamingSteererDatasetV4(Dataset):
    """Pre-computes CPU n-gram features in parallel background workers."""
    def __init__(self, train_ids, seq_len, V):
        self.train_ids = train_ids
        self.seq_len = seq_len
        self.N = len(train_ids)
        self.V = V

    def __len__(self):
        return 1000000

    def __getitem__(self, idx):
        start = torch.randint(0, max(1, self.N - self.seq_len - 1), (1,)).item()
        x = self.train_ids[start:start + self.seq_len]
        y = self.train_ids[start + 1:start + self.seq_len + 1]
        ctx = x.tolist()
        ch = FastNgramFeatures(self.V)
        w_cpu = compute_cpu_features(ctx, ch)
        return x, y, w_cpu

class StreamingTokenDataset(Dataset):
    def __init__(self, train_ids, seq_len):
        self.train_ids = train_ids
        self.seq_len = seq_len
        self.N = len(train_ids)

    def __len__(self):
        return 1000000

    def __getitem__(self, idx):
        start = torch.randint(0, max(1, self.N - self.seq_len - 1), (1,)).item()
        x = self.train_ids[start:start + self.seq_len]
        y = self.train_ids[start + 1:start + self.seq_len + 1]
        return x, y

class FastNgramFeatures:
    """Highly optimized CPU-side n-gram features (O(1) per token)."""
    def __init__(self, V):
        self.V = V; self._u = -_math.log(V); self.reset()

    def update(self, tid):
        tid = int(tid); self._step += 1; self._ctx.append(tid); self._ctx = self._ctx[-128:]
        if self._step % 10 == 0: self._uni *= 0.999; self._uni_total *= 0.999
        if tid < self.V: self._uni[tid] += 1.0; self._uni_total += 1.0
        if len(self._ctx) >= 2:
            p, c = self._ctx[-2], self._ctx[-1]; self._bi[(p,c)] = self._bi.get((p,c), 0)+1; self._bit[p] = self._bit.get(p, 0)+1
        if len(self._ctx) >= 3:
            p2, p1, c = self._ctx[-3], self._ctx[-2], self._ctx[-1]
            self._tri[(p2,p1,c)] = self._tri.get((p2,p1,c), 0)+1; self._trit[(p2,p1)] = self._trit.get((p2,p1), 0)+1
        if len(self._ctx) >= 2:
            self._skip2[(self._ctx[-2], tid)] = self._skip2.get((self._ctx[-2], tid), 0)+1; self._skip2t[self._ctx[-2]] = self._skip2t.get(self._ctx[-2], 0)+1
        if len(self._ctx) >= 3:
            self._skip3[(self._ctx[-3], tid)] = self._skip3.get((self._ctx[-3], tid), 0)+1; self._skip3t[self._ctx[-3]] = self._skip3t.get(self._ctx[-3], 0)+1
        self._seen[tid].append(self._step)

    def get_features(self, tid):
        tid = int(tid); ctx = self._ctx; u = self._u
        d = self._uni_total + 0.001 * self.V
        ul = _math.log(max((self._uni[tid] + 0.001) / d, 1e-7)) if d > 0 and tid < self.V else u
        bl = u
        if len(ctx) >= 1:
            tot = self._bit.get(ctx[-1], 0); db = tot + 0.001 * self.V
            bl = _math.log(max((self._bi.get((ctx[-1], tid), 0) + 0.001) / db, 1e-7)) if db > 0 else u
        tl = u
        if len(ctx) >= 2:
            ck = (ctx[-2], ctx[-1]); tot = self._trit.get(ck, 0); dt = tot + 0.001 * self.V
            tl = _math.log(max((self._tri.get((ctx[-2], ctx[-1], tid), 0) + 0.001) / dt, 1e-7)) if dt > 0 else u
        s2 = u
        if len(ctx) >= 2:
            tot = self._skip2t.get(ctx[-2], 0); ds = tot + 0.001 * self.V
            s2 = _math.log(max((self._skip2.get((ctx[-2], tid), 0) + 0.001) / ds, 1e-7)) if ds > 0 else u
        s3 = u
        if len(ctx) >= 3:
            tot = self._skip3t.get(ctx[-3], 0); ds = tot + 0.001 * self.V
            s3 = _math.log(max((self._skip3.get((ctx[-3], tid), 0) + 0.001) / ds, 1e-7)) if ds > 0 else u
        pos = self._seen.get(tid, []); gap = 128 if not pos else min(128, self._step - pos[-1])
        rl = _math.log(max(1.0 / max(gap, 1), 1e-7))
        ent = float(-ul / _math.log(self.V)) if ul < 0 else 1.0
        return [float(ul), float(bl), float(bl), float(tl), float(tl), float(s2), float(s3), float(rl), float(ent)]

    def reset(self):
        self._uni = np.zeros(self.V, dtype=np.float32); self._uni_total = 0.0
        self._bi = {}; self._bit = {}; self._tri = {}; self._trit = {}
        self._skip2 = {}; self._skip2t = {}; self._skip3 = {}; self._skip3t = {}
        self._seen = defaultdict(list); self._ctx = []; self._step = 0

def compute_cpu_features(tokens, channels):
    """Compute (T, 9) n-gram features on CPU for a sequence."""
    channels.reset()
    feats = []
    for t, tid in enumerate(tokens):
        if t > 0: channels.update(tokens[t-1])
        feats.append(channels.get_features(int(tid)))
    if not feats: return torch.zeros(1, 9)
    return torch.tensor(np.array(feats, dtype=np.float32))

def load_neural_lm(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    s = ckpt['state_dict']
    d_model = s['pos_emb.weight'].shape[-1]
    vocab = s['head_bias'].shape[0]
    d_ff = s['encoder.layers.0.linear1.weight'].shape[0]
    n_layers = len([k for k in s if k.startswith('encoder.layers.') and k.endswith('.norm1.weight')])
    n_heads = s['encoder.layers.0.self_attn.in_proj_weight'].shape[0] // (3 * d_model)
    max_len = s['pos_emb.weight'].shape[0]
    model = DeepCausalLM(vocab=vocab, d_model=d_model, n_layers=n_layers, n_heads=n_heads, d_ff=d_ff, max_len=max_len, dropout=0.0)
    model.load_state_dict(s)
    if device != 'cpu':
        model = model.to(device)
    return model, d_model

def build_fresh_lm(model_config, seq_len, device):
    cfg = dict(MODEL_CONFIGS[model_config])
    cfg['max_len'] = max(int(cfg['max_len']), int(seq_len))
    model = DeepCausalLM(vocab=V, dropout=0.0, **cfg)
    if device != 'cpu':
        model = model.to(device)
    return model, cfg['d_model']

def v4_zeroq_surface(model):
    resident = {'head_bias', 'tok_emb.weight', 'pos_emb.weight', 'ln_f.weight', 'ln_f.bias'}
    names = []
    for name, _ in model.named_parameters():
        if name in resident or '.self_attn.' in name or '.norm' in name:
            names.append(name)
    return TrainableSurface.from_names(names)

def v4_trainable_resident_names(model):
    resident = {'head_bias', 'tok_emb.weight', 'pos_emb.weight', 'ln_f.weight', 'ln_f.bias'}
    return {name for name, _ in model.named_parameters() if name in resident}

def select_v4_optimizer_params(model):
    trainable_names = v4_trainable_resident_names(model)
    params = []
    for name, param in model.named_parameters():
        keep_trainable = name in trainable_names
        param.requires_grad = keep_trainable
        if keep_trainable:
            params.append(param)
    return params

def maybe_init_dist_for_zeroq(device):
    if dist.is_available() and not dist.is_initialized():
        dist.init_process_group(backend='nccl')
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    if device.type == 'cuda':
        torch.cuda.set_device(local_rank)
    return torch.device(f'cuda:{local_rank}') if device.type == 'cuda' else device

class CompiledPriorLogitHead(nn.Module):
    def __init__(self, feature_dim, vocab_size, init_scale=1e-4):
        super().__init__()
        self.norm = nn.LayerNorm(feature_dim)
        self.proj = nn.Linear(feature_dim, vocab_size, bias=False)
        self.scale = nn.Parameter(torch.tensor(0.01))
        nn.init.normal_(self.proj.weight, mean=0.0, std=init_scale)

    def forward(self, features, dtype=None):
        prior_logits = self.scale * self.proj(self.norm(features.float()))
        return prior_logits if dtype is None else prior_logits.to(dtype=dtype)

def compute_compiled_features(x, w_cpu, gpu_fc):
    w_gpu = gpu_fc.compute_features(x)
    w_gpu[:, :, 0:9] = w_cpu[:, :, :9]
    return w_gpu

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--neural-ckpt', type=str, default='artifacts/c4_v2_768_x30/best.pt')
    p.add_argument('--resume-model', type=str, default='artifacts/steerer_v2/steerer_best_b.pt')
    p.add_argument('--from-scratch', action='store_true',
                   help='Instantiate a fresh DeepCausalLM instead of loading --resume-model.')
    p.add_argument('--model-config', choices=sorted(MODEL_CONFIGS), default='124m',
                   help='Fresh model size used with --from-scratch.')
    p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--steps', type=int, default=500)
    p.add_argument('--batch', type=int, default=8)
    p.add_argument('--seq-len', type=int, default=128)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--out-dir', type=str, default='artifacts/steerer_v4')
    p.add_argument('--device', type=str, default='cuda')
    p.add_argument('--backend', choices=['dense', 'zeroq'], default='dense')
    p.add_argument('--zeroq-path', type=str, default='/home/drawson/ZeroQ')
    p.add_argument('--compute-in-4bit', action='store_true')
    p.add_argument('--injection', choices=['residual', 'logit', 'none'], default='residual',
                   help='Compiled-prior injection method: residual hooks, output-logit head, or disabled.')
    p.add_argument('--prior-head-lr', type=float, default=None,
                   help='LR for --injection logit head; defaults to --lr.')
    p.add_argument('--log-every', type=int, default=50,
                   help='Print train-step progress every N steps; set 0 to disable.')
    p.add_argument('--eval-tokens', type=int, default=8192,
                   help='Number of validation tokens to score per epoch; full validation is too slow for frequent 700M checks.')
    p.add_argument('--resume-training-ckpt', type=str, default=None,
                   help='Resume model/injection/optimizer state from a train_steerer_v4 checkpoint.')
    p.add_argument('--no-steerer', action='store_true',
                   help='Alias for --injection none.')
    args = p.parse_args()
    if args.no_steerer:
        args.injection = 'none'
    use_compiled = args.injection != 'none'
    use_residual = args.injection == 'residual'
    use_logit_prior = args.injection == 'logit'

    device = torch.device(args.device)
    if args.backend == 'zeroq':
        device = maybe_init_dist_for_zeroq(device)
    out_dir = REPO / args.out_dir; out_dir.mkdir(parents=True, exist_ok=True)

    print('=' * 60)
    print(' STEERER V4 (21ch V3 Steerer + GPU Features)')
    print('=' * 60)

    print('[load] Data...')
    train_ids = torch.load(REPO/'artifacts/wikitext_gpt2/train_ids.pt', weights_only=False).long()
    val_ids = torch.load(REPO/'artifacts/wikitext_gpt2/validation_ids.pt', weights_only=False).long()
    print(f'  Train: {len(train_ids):,}  Val: {len(val_ids):,}')

    if not use_compiled:
        word_topics = pos_tags = ppmi_emb = None
        print('[load] V3 compiled priors skipped (--injection none)')
    else:
        print('[load] V3 compiled priors...')
        priors_dir = REPO / 'artifacts/compiled_priors_v3'
        word_topics = torch.load(priors_dir / 'word_topics.pt', map_location='cpu', weights_only=False)
        with open(priors_dir / 'pos_stats.pkl', 'rb') as f:
            pos_stats = pickle.load(f)
        token_to_tag = pos_stats.get('token_to_tag', {})
        tag_to_idx = pos_stats.get('tag_to_idx', {'WORD': 0, 'PUNCT': 1, 'NUM': 2})
        pos_tags = {int(k): tag_to_idx.get(v, 0) for k, v in token_to_tag.items()}
        ppmi_emb = torch.randn(V, 256, dtype=torch.float32) * 0.01
        print(f'  Topics: {word_topics.shape}, POS tags: {len(pos_tags)} tokens mapped')

    if args.from_scratch:
        print(f'[build] Fresh {args.model_config} DeepCausalLM...')
        model, d_model = build_fresh_lm(args.model_config, args.seq_len,
                                         'cpu' if args.backend == 'zeroq' else device)
    else:
        print('[load] Warm-start model from V2...')
        model, d_model = load_neural_lm(REPO / args.resume_model,
                                         'cpu' if args.backend == 'zeroq' else device)
    if args.backend == 'zeroq':
        print('[zeroq] Partitioning frozen backbone; resident=trainable embeddings/head/ln_f...')
        backend = ZeroQPartitionedBackend(
            device=device,
            zeroq_path=args.zeroq_path,
            compute_in_4bit=args.compute_in_4bit,
        )
        handle = backend.prepare(model, v4_zeroq_surface(model))
        model = handle.model
        model_params = select_v4_optimizer_params(model)
        print(f'  zeroq stats={handle.memory_stats()} resident_model_params={sum(p.numel() for p in model_params):,}')
    else:
        for p in model.parameters(): p.requires_grad = True
        model_params = list(model.parameters())
    print(f'  {sum(p.numel() for p in model.parameters()):,} params  d_model={d_model}')

    steerer = None
    prior_head = None
    if use_residual:
        print('[build] 21-channel SuperpositionSteererV3...')
        steerer = SuperpositionSteererV3(d_model=d_model, init_scale=0.01, noise_scale=0.05)
        steerer = steerer.to(device)
        n_hooks = steerer.register_hooks(model)
        s_params = sum(p.numel() for p in steerer.parameters())
        print(f'  {n_hooks} hooks, {s_params:,} params  (LR={args.lr})')
        print(f'    local: 6ch → layers [0,1,2]')
        print(f'    mid:   7ch → layers [4,5,6]')
        print(f'    global: 8ch → layers [8,9,10]')

    elif use_logit_prior:
        print('[build] 21-channel output-logit compiled prior head...')
        prior_head = CompiledPriorLogitHead(feature_dim=21, vocab_size=V).to(device)
        p_params = sum(p.numel() for p in prior_head.parameters())
        print(f'  {p_params:,} params  scale_init={prior_head.scale.item():.4f}  LR={args.prior_head_lr or args.lr}')
    else:
        print('[build] compiled-prior injection disabled')

    if use_compiled:
        print('[build] GPU Feature Computer...')
        gpu_fc = GPUFeatureComputer(
            V=V, punct_ids=PUNCT_IDS, topic_matrix=word_topics,
            pos_tags=pos_tags, ppmi_embeddings=ppmi_emb, device=device)
        print(f'  21 channels computed on GPU in parallel')
    else:
        gpu_fc = None

    opt_groups = [{'params': model_params, 'lr': 3e-5}]
    if steerer is not None:
        opt_groups.append({'params': steerer.parameters(), 'lr': args.lr})
    if prior_head is not None:
        opt_groups.append({'params': prior_head.parameters(), 'lr': args.prior_head_lr or args.lr})
    opt = torch.optim.AdamW(opt_groups, weight_decay=0.1)

    N = len(train_ids)
    best_eval_b = float('inf'); best_eval_s = float('inf')
    start_epoch = 1

    if args.resume_training_ckpt:
        ckpt_path = REPO / args.resume_training_ckpt
        print(f'[resume] Loading training checkpoint {ckpt_path}...')
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['state_dict'])
        if steerer is not None and ckpt.get('steerer_state') is not None:
            steerer.load_state_dict(ckpt['steerer_state'])
        if prior_head is not None and ckpt.get('prior_head_state') is not None:
            prior_head.load_state_dict(ckpt['prior_head_state'])
        if ckpt.get('opt_state') is not None:
            opt.load_state_dict(ckpt['opt_state'])
        best_eval_b = float(ckpt.get('eval_b', best_eval_b))
        best_eval_s = float(ckpt.get('eval_s', best_eval_s))
        start_epoch = int(ckpt.get('epoch', 0)) + 1
        print(f'  resumed at epoch={start_epoch} best_s={best_eval_s:.1f} best_b={best_eval_b:.1f}')

    # DataLoader for parallel CPU feature pre-computation
    train_dataset = (StreamingSteererDatasetV4(train_ids=train_ids, seq_len=args.seq_len, V=V)
                     if use_compiled else StreamingTokenDataset(train_ids=train_ids, seq_len=args.seq_len))
    train_loader = DataLoader(train_dataset, batch_size=args.batch,
                              num_workers=4, pin_memory=True, drop_last=True)
    loader_iter = iter(train_loader)

    for ep in range(start_epoch, args.epochs + 1):
        model.train()
        if steerer is not None:
            steerer.train()
        if prior_head is not None:
            prior_head.train()
        total_loss = 0.0; t0 = time.time()

        for step in range(args.steps):
            # Parallel CPU pre-fetch: n-gram features computed in background workers
            batch = next(loader_iter)
            if not use_compiled:
                x, y = batch
                w_cpu = None
            else:
                x, y, w_cpu = batch
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            if w_cpu is not None:
                w_cpu = w_cpu.to(device, non_blocking=True)

            w_gpu = None
            if use_compiled:
                w_gpu = compute_compiled_features(x, w_cpu, gpu_fc)
            if steerer is not None:
                steerer.set_weights(w_gpu)

            logits = model(x)
            if prior_head is not None:
                logits = logits + prior_head(w_gpu, dtype=logits.dtype)
            loss = F.cross_entropy(logits.reshape(-1, model.head_bias.shape[0]), y.reshape(-1))
            if steerer is not None:
                loss = loss + 0.001 * steerer.orthogonal_penalty()

            opt.zero_grad(); loss.backward()
            clip_params = list(model.parameters())
            if steerer is not None:
                clip_params += list(steerer.parameters())
            if prior_head is not None:
                clip_params += list(prior_head.parameters())
            torch.nn.utils.clip_grad_norm_(clip_params, 1.0)
            opt.step()
            total_loss += loss.item()
            if args.log_every > 0 and (step + 1) % args.log_every == 0:
                avg_so_far = total_loss / (step + 1)
                print(f'    epoch={ep:2d} step={step + 1:4d}/{args.steps} '
                      f'loss={avg_so_far:.4f} ppl={math.exp(avg_so_far):.1f} '
                      f'time={time.time() - t0:.0f}s', flush=True)

        # Eval (steered + baseline)
        model.eval()
        if steerer is not None:
            steerer.eval()
        if prior_head is not None:
            prior_head.eval()
        cpu_ch = FastNgramFeatures(V)
        with torch.no_grad():
            es_nll, es_n = 0.0, 0
            eval_limit = min(len(val_ids) - 1, max(1, args.eval_tokens))
            if use_compiled:
                for s in range(0, eval_limit, 64):
                    cl = min(64, eval_limit - s)
                    if cl <= 0: continue
                    inp = val_ids[s:s+cl].unsqueeze(0).to(device)
                    tgt = val_ids[s+1:s+cl+1].unsqueeze(0).to(device)
                    ctx = val_ids[s:s+cl].tolist()
                    w_cpu = compute_cpu_features(ctx, cpu_ch)
                    w_e = compute_compiled_features(inp, w_cpu.unsqueeze(0).to(device), gpu_fc)
                    if steerer is not None:
                        steerer.set_weights(w_e)
                    l = model(inp)
                    if prior_head is not None:
                        l = l + prior_head(w_e, dtype=l.dtype)
                    es_nll += F.cross_entropy(l.reshape(-1, model.head_bias.shape[0]), tgt.reshape(-1), reduction='sum').item()
                    es_n += cl
                eval_s = math.exp(es_nll / max(es_n, 1))
                if steerer is not None:
                    steerer._current_weights = None
            else:
                eval_s = float('inf')
            eb_nll, eb_n = 0.0, 0
            for s in range(0, eval_limit, 128):
                cl = min(128, eval_limit - s)
                if cl <= 0: continue
                inp = val_ids[s:s+cl].unsqueeze(0).to(device)
                tgt = val_ids[s+1:s+cl+1].unsqueeze(0).to(device)
                l = model(inp)
                eb_nll += F.cross_entropy(l.reshape(-1, model.head_bias.shape[0]), tgt.reshape(-1), reduction='sum').item()
                eb_n += cl
            eval_b = math.exp(eb_nll / max(eb_n, 1))

        avg_loss = total_loss / args.steps; elapsed = time.time() - t0
        new_best = []
        if eval_b < best_eval_b:
            best_eval_b = eval_b
            new_best.append('b')
        if use_compiled and eval_s < best_eval_s:
            best_eval_s = eval_s
            new_best.append('s')
        status = ''.join(new_best) or '-'
        winner = 's' if use_compiled and eval_s < eval_b else 'b'
        eval_gap = eval_s - eval_b if use_compiled else float('nan')
        if 'b' in new_best:
            torch.save({'state_dict': model.state_dict(),
                        'steerer_state': steerer.state_dict() if steerer is not None else None,
                        'prior_head_state': prior_head.state_dict() if prior_head is not None else None,
                        'injection': args.injection,
                        'eval_s': eval_s, 'eval_b': eval_b, 'epoch': ep, 'opt_state': opt.state_dict()},
                       out_dir / 'steerer_best_b.pt')
        if 's' in new_best:
            torch.save({'state_dict': model.state_dict(),
                        'steerer_state': steerer.state_dict() if steerer is not None else None,
                        'prior_head_state': prior_head.state_dict() if prior_head is not None else None,
                        'injection': args.injection,
                        'eval_s': eval_s, 'eval_b': eval_b, 'epoch': ep, 'opt_state': opt.state_dict()},
                       out_dir / 'steerer_best_s.pt')

        print(f'  epoch={ep:2d}  loss={avg_loss:.4f}  ppl={math.exp(avg_loss):.1f}  '
              f'eval_s={eval_s:.1f}  eval_b={eval_b:.1f}  '
              f'best_s={best_eval_s:.1f}  best_b={best_eval_b:.1f}  '
              f'winner={winner}  gap_s-b={eval_gap:+.1f}  new=[{status}]  '
              f'time={elapsed:.0f}s', flush=True)

    print(f'\nDone. Best eval_b: {best_eval_b:.1f}  Best eval_s: {best_eval_s:.1f}')

if __name__ == '__main__':
    main()
