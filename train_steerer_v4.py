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
    '2b':    dict(d_model=2560, n_layers=24, n_heads=16, d_ff=10240, max_len=128),
    '4b':    dict(d_model=3072, n_layers=40, n_heads=24, d_ff=12288, max_len=128),
    '700m':  dict(d_model=1536, n_layers=22, n_heads=16, d_ff=6144, max_len=512),
}

class StreamingDatasetC4(torch.utils.data.IterableDataset):
    """Mixes WikiText with C4 data at a configurable ratio."""

    def __init__(self, wt_train_ids, seq_len, vocab_size, c4_ratio=0.85, seed=42):
        super().__init__()
        self.wt_train_ids = wt_train_ids
        self.seq_len = seq_len
        self.vocab_size = int(vocab_size)
        self.c4_ratio = float(c4_ratio)
        self.seed = int(seed)

    def _local_c4_files(self):
        import glob, json, os
        roots = [
            os.environ.get('HF_DATASETS_CACHE', ''),
            os.path.join(os.environ.get('HF_HOME', ''), 'datasets'),
            os.path.expanduser('~/deepseek_experiments/artifacts/hf_cache/datasets'),
        ]
        files, seen = [], set()
        for root in roots:
            if not root: continue
            dl = os.path.join(os.path.expanduser(root), 'downloads')
            for meta_path in glob.glob(os.path.join(dl, '*.json')):
                data_path = meta_path[:-5]
                if data_path in seen or not os.path.exists(data_path): continue
                try:
                    with open(meta_path) as fh: meta = json.load(fh)
                except Exception: continue
                if '/en/c4-train.' not in str(meta.get('url', '')): continue
                seen.add(data_path); files.append(data_path)
        return sorted(files)

    def _iter_local_c4_texts(self, files, rng, worker_id, num_workers):
        import gzip, json

        shuffled = list(files)
        rng.shuffle(shuffled)
        worker_files = shuffled[worker_id::max(1, num_workers)] or shuffled

        while True:
            rng.shuffle(worker_files)
            for path in worker_files:
                try:
                    with gzip.open(path, 'rt', encoding='utf-8') as fh:
                        lines = []
                        for line in fh:
                            lines.append(line)
                            if len(lines) >= 2000:
                                rng.shuffle(lines)
                                for l in lines:
                                    try: text = (json.loads(l).get('text') or '').strip()
                                    except: continue
                                    if text: yield text
                                lines = []
                        if lines:
                            rng.shuffle(lines)
                            for l in lines:
                                try: text = (json.loads(l).get('text') or '').strip()
                                except: continue
                                if text: yield text
                except OSError:
                    continue

    def __iter__(self):
        import random
        from torch.utils.data import get_worker_info
        from transformers import AutoTokenizer

        worker = get_worker_info()
        wid = worker.id if worker else 0
        nw = worker.num_workers if worker else 1
        rng = random.Random(self.seed + wid * 1009)
        torch_gen = torch.Generator().manual_seed(self.seed + wid * 9176)
        tokenizer = AutoTokenizer.from_pretrained('gpt2')

        c4_files = self._local_c4_files()
        channels = FastNgramFeatures(self.vocab_size)
        if not c4_files:
            c4_files = []

        c4_iter = self._iter_local_c4_texts(c4_files, rng, wid, nw) if c4_files else None
        c4_buf = []

        def refill_c4():
            nonlocal c4_buf, c4_iter
            while len(c4_buf) < self.seq_len * 8:
                if c4_iter is None:
                    break
                try:
                    text = next(c4_iter)
                except StopIteration:
                    c4_iter = self._iter_local_c4_texts(c4_files, rng, wid, nw)
                    continue
                ids = tokenizer.encode(text[:2000], add_special_tokens=False)
                if ids:
                    c4_buf.extend(ids)

        refill_c4()

        while True:
            use_c4 = rng.random() < self.c4_ratio and c4_files
            if use_c4:
                if len(c4_buf) < self.seq_len + 2:
                    refill_c4()
                start = rng.randint(0, max(0, len(c4_buf) - self.seq_len - 2))
                ids = c4_buf[start:start + self.seq_len + 1]
                if len(ids) < self.seq_len + 1:
                    refill_c4()
                    continue
                x = torch.tensor(ids[:-1], dtype=torch.long)
                y = torch.tensor(ids[1:], dtype=torch.long)
                ch = FastNgramFeatures(self.vocab_size)
                w_cpu = compute_cpu_features(ids[:-1], ch)
            else:
                max_start = max(1, len(self.wt_train_ids) - self.seq_len - 1)
                start = torch.randint(0, max_start, (1,), generator=torch_gen).item()
                x = self.wt_train_ids[start:start + self.seq_len]
                y = self.wt_train_ids[start + 1:start + self.seq_len + 1]
                ch = FastNgramFeatures(self.vocab_size)
                w_cpu = compute_cpu_features(x.tolist(), ch)
            yield x, y, w_cpu


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
    model.load_state_dict(s); model = model.to(device)
    return model, d_model

def build_fresh_lm(model_config, seq_len, device):
    cfg = dict(MODEL_CONFIGS[model_config])
    cfg['max_len'] = max(int(cfg['max_len']), int(seq_len))
    model = DeepCausalLM(vocab=V, dropout=0.0, **cfg)
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


def zeroq_save_checkpoint(state_dict: dict, path, handle):
    """Gather ZeroQ weights to FP32, save, then release."""
    if handle is not None and hasattr(handle, 'wrapper') and hasattr(handle.wrapper, 'start_gather'):
        handle.wrapper.start_gather()
        handle.wrapper.wait_gather()
    torch.save(state_dict, path)
    if handle is not None and hasattr(handle, 'wrapper') and hasattr(handle.wrapper, 'release'):
        handle.wrapper.release()


def calibrate_steering_controls(model, steerer, gpu_fc, train_loader, device, epochs):
    """UNTESTED: freeze model + steerer weights, train only alpha/beta/gamma.

    Runs N epochs on the training dataset with LBFGS, searching for injection
    scaling that minimizes PPL with the frozen model.  The steerer weights
    themselves are not updated — only per-layer gammas, group betas, and alpha.
    """
    if steerer is None:
        return
    control_params = list(steerer.steering_control_parameters())
    if not control_params:
        print('[calibrate] no steering control parameters found, skipping')
        return
    print(f'[calibrate] training {sum(p.numel() for p in control_params)} control params for {epochs} epochs')

    model_prev = {p: p.requires_grad for p in model.parameters()}
    steerer_prev = {p: p.requires_grad for p in steerer.parameters()}
    for p in model.parameters():
        p.requires_grad = False
    for p in steerer.parameters():
        p.requires_grad = p in control_params

    model.eval()
    steerer.eval()
    opt = torch.optim.LBFGS(control_params, lr=0.1, max_iter=10, line_search_fn='strong_wolfe')
    cpu_ch = FastNgramFeatures(V)

    for ep in range(epochs):
        loader_iter = iter(train_loader)
        total_loss = 0.0
        steps = 0
        for _ in range(min(50, len(train_loader))):
            batch = next(loader_iter)
            x, y, w_cpu = batch
            x = x.to(device)
            y = y.to(device)
            w_cpu = w_cpu.to(device)

            def closure():
                w_gpu = compute_compiled_features(x, w_cpu, gpu_fc)
                if steerer.semantic_dim > 0:
                    sem = compute_semantic_channels(model, steerer, x)
                    w_gpu = torch.cat([w_gpu, sem], dim=-1)
                steerer.set_weights(w_gpu)
                logits = model(x)
                loss = F.cross_entropy(logits.reshape(-1, V), y.reshape(-1))
                loss = loss + 0.001 * steerer.orthogonal_penalty()
                opt.zero_grad()
                loss.backward()
                return loss

            loss = opt.step(closure)
            total_loss += loss.item()
            steps += 1

        avg_loss = total_loss / max(steps, 1)
        print(f'  calibrate epoch={ep + 1} loss={avg_loss:.4f} ppl={math.exp(avg_loss):.1f}', flush=True)

    for p, req in model_prev.items():
        p.requires_grad = req
    for p, req in steerer_prev.items():
        p.requires_grad = req
    print(f'[calibrate] done  alpha={steerer.alpha.item():.4f}', flush=True)

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


def compute_semantic_channels(model, steerer, x, capture_layers=(3, 7, 11)):
    layers = model.encoder.layers
    captured = {}

    def make_cap_hook(idx):
        def hook_fn(module, input, output):
            h = output[0] if isinstance(output, tuple) else output
            if torch.is_tensor(h):
                captured[idx] = steerer._semantic_encoder(h.float())
        return hook_fn

    cap_hooks = []
    for idx in capture_layers:
        if idx < len(layers):
            cap_hooks.append(layers[idx].register_forward_hook(make_cap_hook(idx)))

    saved_req = {p: p.requires_grad for p in model.parameters()}
    for p in model.parameters():
        p.requires_grad = False
    steerer.remove_hooks()
    model(x)
    steerer.register_hooks(model)
    for p, req in saved_req.items():
        p.requires_grad = req

    for h in cap_hooks:
        h.remove()

    if not captured:
        return torch.zeros(x.shape[0], x.shape[1], steerer.semantic_dim, device=x.device)

    stacked = torch.stack([captured[idx] for idx in sorted(captured.keys())], dim=0)
    return stacked.mean(dim=0)


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
    p.add_argument('--model-lr', type=float, default=3e-5,
                   help='Learning rate for the base model (steerer gets --lr).')
    p.add_argument('--semantic-dim', type=int, default=0,
                   help='Number of semantic channels (0 = disabled, 16 = enabled).')
    p.add_argument('--calibrate', type=int, default=0,
                   help='UNTESTED: train steerer alphas/betas/gammas for N epochs with frozen model.')
    p.add_argument('--data-mode', choices=['wikitext', 'c4-mix'], default='wikitext')
    p.add_argument('--c4-ratio', type=float, default=0.85,
                   help='Fraction of C4 examples in c4-mix mode.')
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
        model, d_model = build_fresh_lm(args.model_config, args.seq_len, device)
    else:
        print('[load] Warm-start model from V2...')
        model, d_model = load_neural_lm(REPO / args.resume_model, device)

    if not args.from_scratch or args.backend != 'zeroq':
        model_params = list(model.parameters())
        if args.backend != 'zeroq':
            for p in model.parameters(): p.requires_grad = True
    else:
        model_params = list(model.parameters())
        for p in model.parameters(): p.requires_grad = True

    opt_groups = [{'params': model_params, 'lr': args.model_lr}]
    opt = torch.optim.AdamW(opt_groups, weight_decay=0.1)

    best_eval_b = float('inf'); best_eval_s = float('inf')
    start_epoch = 1

    if args.resume_training_ckpt:
        ckpt_path = REPO / args.resume_training_ckpt
        print(f'[resume] Loading training checkpoint {ckpt_path}...')
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['state_dict'])
        if ckpt.get('opt_state') is not None:
            try:
                opt.load_state_dict(ckpt['opt_state'])
                for pg in opt.param_groups:
                    pg['lr'] = pg.get('initial_lr', args.model_lr) if pg['lr'] > args.model_lr else pg['lr']
            except (ValueError, RuntimeError) as e:
                print(f'  [resume] opt mismatch, using fresh optimizer')
        best_eval_b = float(ckpt.get('eval_b', best_eval_b))
        best_eval_s = float(ckpt.get('eval_s', best_eval_s))
        start_epoch = int(ckpt.get('epoch', 0)) + 1
        print(f'  resumed at epoch={start_epoch} best_s={best_eval_s:.1f} best_b={best_eval_b:.1f}')

    handle = None
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
    print(f'  {sum(p.numel() for p in model.parameters()):,} params  d_model={d_model}')

    steerer = None
    prior_head = None
    if use_residual:
        print('[build] 21-channel SuperpositionSteererV3...')
        steerer = SuperpositionSteererV3(d_model=d_model, init_scale=0.01, noise_scale=0.05,
                                          semantic_dim=args.semantic_dim)
        steerer = steerer.to(device)
        n_hooks = steerer.register_hooks(model)
        s_params = sum(p.numel() for p in steerer.parameters())
        print(f'  {n_hooks} hooks, {s_params:,} params  (LR={args.lr})')
    elif use_logit_prior:
        prior_head = CompiledPriorLogitHead(feature_dim=21, vocab_size=V).to(device)
    else:
        print('[build] compiled-prior injection disabled')

    opt_groups = [{'params': model_params, 'lr': args.model_lr}]
    if steerer is not None:
        opt_groups.append({'params': steerer.parameters(), 'lr': args.lr})
    if prior_head is not None:
        opt_groups.append({'params': prior_head.parameters(), 'lr': args.prior_head_lr or args.lr})
    opt = torch.optim.AdamW(opt_groups, weight_decay=0.1)

    if use_compiled:
        print('[build] GPU Feature Computer...')
        gpu_fc = GPUFeatureComputer(
            V=V, punct_ids=PUNCT_IDS, topic_matrix=word_topics,
            pos_tags=pos_tags, ppmi_embeddings=ppmi_emb, device=device)
        print(f'  21 channels computed on GPU in parallel')
    else:
        gpu_fc = None

    # DataLoader for parallel CPU feature pre-computation
    train_dataset = (StreamingDatasetC4(wt_train_ids=train_ids, seq_len=args.seq_len, vocab_size=V,
                                          c4_ratio=args.c4_ratio)
                     if use_compiled and args.data_mode == 'c4-mix' else
                     StreamingSteererDatasetV4(train_ids=train_ids, seq_len=args.seq_len, V=V)
                     if use_compiled else StreamingTokenDataset(train_ids=train_ids, seq_len=args.seq_len))
    train_loader = DataLoader(train_dataset, batch_size=args.batch,
                              num_workers=4 if args.data_mode != 'c4-mix' else 2,
                              pin_memory=True, drop_last=True)

    if args.calibrate > 0 and steerer is not None:
        calibrate_steering_controls(model, steerer, gpu_fc, train_loader, device, args.calibrate)

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
            zeroq_save_checkpoint({'state_dict': model.state_dict(),
                        'steerer_state': steerer.state_dict() if steerer is not None else None,
                        'prior_head_state': prior_head.state_dict() if prior_head is not None else None,
                        'injection': args.injection,
                        'eval_s': eval_s, 'eval_b': eval_b, 'epoch': ep, 'opt_state': opt.state_dict()},
                       out_dir / 'steerer_best_b.pt', handle)
        if 's' in new_best:
            zeroq_save_checkpoint({'state_dict': model.state_dict(),
                        'steerer_state': steerer.state_dict() if steerer is not None else None,
                        'prior_head_state': prior_head.state_dict() if prior_head is not None else None,
                        'injection': args.injection,
                        'eval_s': eval_s, 'eval_b': eval_b, 'epoch': ep, 'opt_state': opt.state_dict()},
                        out_dir / 'steerer_best_s.pt', handle)

        print(f'  epoch={ep:2d}  loss={avg_loss:.4f}  ppl={math.exp(avg_loss):.1f}  '
              f'eval_s={eval_s:.1f}  eval_b={eval_b:.1f}  '
              f'best_s={best_eval_s:.1f}  best_b={best_eval_b:.1f}  '
              f'winner={winner}  gap_s-b={eval_gap:+.1f}  new=[{status}]  '
              f'time={elapsed:.0f}s', flush=True)

    print(f'\nDone. Best eval_b: {best_eval_b:.1f}  Best eval_s: {best_eval_s:.1f}')

if __name__ == '__main__':
    main()
