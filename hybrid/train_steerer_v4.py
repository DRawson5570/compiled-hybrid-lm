"""train_steerer_v4.py — V4 steerer with 21-channel SuperpositionSteererV3 + GPU features.

Proper channel alignment (6+7+8=21) with per-group MLPs.
GPU-vectorized KV cache, Topic prior, and POS features.
Warm-start from V2 best_b model, fresh V4 steerer.
"""
import sys, time, math, argparse, importlib.util, pickle
from pathlib import Path
import torch
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

V = 50257
PUNCT_IDS = {0,2,3,4,5,7,8,9,10,11,12,13,14,25,26,27,28,29,30,31,58,60,61,90,91,92,93,198,220}

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

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--neural-ckpt', type=str, default='artifacts/c4_v2_768_x30/best.pt')
    p.add_argument('--resume-model', type=str, default='artifacts/steerer_v2/steerer_best_b.pt')
    p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--steps', type=int, default=500)
    p.add_argument('--batch', type=int, default=8)
    p.add_argument('--seq-len', type=int, default=128)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--out-dir', type=str, default='artifacts/steerer_v4')
    p.add_argument('--device', type=str, default='cuda')
    args = p.parse_args()

    device = torch.device(args.device)
    out_dir = REPO / args.out_dir; out_dir.mkdir(parents=True, exist_ok=True)

    print('=' * 60)
    print(' STEERER V4 (21ch V3 Steerer + GPU Features)')
    print('=' * 60)

    print('[load] Data...')
    train_ids = torch.load(REPO/'artifacts/wikitext_gpt2/train_ids.pt', weights_only=False).long()
    val_ids = torch.load(REPO/'artifacts/wikitext_gpt2/validation_ids.pt', weights_only=False).long()
    print(f'  Train: {len(train_ids):,}  Val: {len(val_ids):,}')

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

    print('[load] Warm-start model from V2...')
    model, d_model = load_neural_lm(REPO / args.resume_model, device)
    for p in model.parameters(): p.requires_grad = True
    print(f'  {sum(p.numel() for p in model.parameters()):,} params  d_model={d_model}')

    print('[build] 21-channel SuperpositionSteererV3...')
    steerer = SuperpositionSteererV3(d_model=d_model, init_scale=0.01, noise_scale=0.05)
    steerer = steerer.to(device)
    n_hooks = steerer.register_hooks(model)
    s_params = sum(p.numel() for p in steerer.parameters())
    print(f'  {n_hooks} hooks, {s_params:,} params  (LR={args.lr})')
    print(f'    local: 6ch → layers [0,1,2]')
    print(f'    mid:   7ch → layers [4,5,6]')
    print(f'    global: 8ch → layers [8,9,10]')

    print('[build] GPU Feature Computer...')
    gpu_fc = GPUFeatureComputer(
        V=V, punct_ids=PUNCT_IDS, topic_matrix=word_topics,
        pos_tags=pos_tags, ppmi_embeddings=ppmi_emb, device=device)
    print(f'  21 channels computed on GPU in parallel')

    opt = torch.optim.AdamW([
        {'params': model.parameters(), 'lr': 3e-5},
        {'params': steerer.parameters(), 'lr': args.lr},
    ], weight_decay=0.1)

    N = len(train_ids)
    best_eval_b = float('inf'); best_eval_s = float('inf')

    # DataLoader for parallel CPU feature pre-computation
    train_dataset = StreamingSteererDatasetV4(
        train_ids=train_ids, seq_len=args.seq_len, V=V)
    train_loader = DataLoader(train_dataset, batch_size=args.batch,
                              num_workers=4, pin_memory=True, drop_last=True)
    loader_iter = iter(train_loader)

    for ep in range(1, args.epochs + 1):
        model.train(); steerer.train()
        total_loss = 0.0; t0 = time.time()

        for step in range(args.steps):
            # Parallel CPU pre-fetch: n-gram features computed in background workers
            x, y, w_cpu = next(loader_iter)
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            w_cpu = w_cpu.to(device, non_blocking=True)

            # GPU parallel features (channels 0, 8-20)
            w_gpu = gpu_fc.compute_features(x)

            # Merge CPU n-gram features into GPU tensor
            w_gpu[:, :, 0:9] = w_cpu[:, :, :9]

            steerer.set_weights(w_gpu)

            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, model.head_bias.shape[0]), y.reshape(-1))
            loss = loss + 0.001 * steerer.orthogonal_penalty()

            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(steerer.parameters()), 1.0)
            opt.step()
            total_loss += loss.item()

        # Eval (steered + baseline)
        model.eval(); steerer.eval()
        cpu_ch = FastNgramFeatures(V)
        with torch.no_grad():
            es_nll, es_n = 0.0, 0
            for s in range(0, min(len(val_ids) - 1, 5000), 64):
                cl = min(64, len(val_ids) - s - 1)
                if cl <= 0: continue
                inp = val_ids[s:s+cl].unsqueeze(0).to(device)
                tgt = val_ids[s+1:s+cl+1].unsqueeze(0).to(device)
                w_e = gpu_fc.compute_features(inp)
                ctx = val_ids[s:s+cl].tolist()
                w_cpu = compute_cpu_features(ctx, cpu_ch)
                w_e[0, :w_cpu.shape[0], 0:9] = w_cpu[:, :9].to(device)
                steerer.set_weights(w_e)
                l = model(inp)
                es_nll += F.cross_entropy(l.reshape(-1, model.head_bias.shape[0]), tgt.reshape(-1), reduction='sum').item()
                es_n += cl
            eval_s = math.exp(es_nll / max(es_n, 1))

            steerer._current_weights = None
            eb_nll, eb_n = 0.0, 0
            for s in range(0, len(val_ids) - 1, 128):
                cl = min(128, len(val_ids) - s - 1)
                if cl <= 0: continue
                inp = val_ids[s:s+cl].unsqueeze(0).to(device)
                tgt = val_ids[s+1:s+cl+1].unsqueeze(0).to(device)
                l = model(inp)
                eb_nll += F.cross_entropy(l.reshape(-1, model.head_bias.shape[0]), tgt.reshape(-1), reduction='sum').item()
                eb_n += cl
            eval_b = math.exp(eb_nll / max(eb_n, 1))

        avg_loss = total_loss / args.steps; elapsed = time.time() - t0
        status = ''
        if eval_b < best_eval_b: best_eval_b = eval_b; status += 'b'
        if eval_s < best_eval_s: best_eval_s = eval_s; status += 's'
        if 'b' in status:
            torch.save({'state_dict': model.state_dict(), 'steerer_state': steerer.state_dict(),
                        'eval_s': eval_s, 'eval_b': eval_b, 'epoch': ep, 'opt_state': opt.state_dict()},
                       out_dir / 'steerer_best_b.pt')
        if 's' in status:
            torch.save({'state_dict': model.state_dict(), 'steerer_state': steerer.state_dict(),
                        'eval_s': eval_s, 'eval_b': eval_b, 'epoch': ep, 'opt_state': opt.state_dict()},
                       out_dir / 'steerer_best_s.pt')

        print(f'  epoch={ep:2d}  loss={avg_loss:.4f}  ppl={math.exp(avg_loss):.1f}  '
              f'eval_s={eval_s:.1f}  eval_b={eval_b:.1f}  best_b={best_eval_b:.1f}  '
              f'[{status}]  time={elapsed:.0f}s', flush=True)

    print(f'\nDone. Best eval_b: {best_eval_b:.1f}  Best eval_s: {best_eval_s:.1f}')

if __name__ == '__main__':
    main()
