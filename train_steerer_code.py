"""train_steerer_code.py — Train a code-domain steering cartridge on Python data.
Frozen 124M base model, steerer-only training. Production mode.
"""
import sys
from hybrid.config import REPO_ROOT, time, math, argparse, importlib.util
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
import torch.nn.functional as F

DEEPSEEK = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(DEEPSEEK))
import sys as _sys; _sys.path.insert(0, str(DEEPSEEK))

_spec = importlib.util.spec_from_file_location('scaled', str(DEEPSEEK / 'hybrid/train_scaled_neural_lm.py'))
_mod = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_mod); DeepCausalLM = _mod.DeepCausalLM
import sys as _sys; _sys.path.insert(0, str(DEEPSEEK))
from hybrid.superposition_steerer import SuperpositionSteerer

V = 50257; C_ACTIVE = 9

class CodeChannelFeatures:
    def __init__(self):
        self.V = V
        self._uni = np.zeros(V, dtype=np.float32)
        self._bi = {}; self._bit = {}
        self._tri = {}; self._trit = {}
        self._sp = defaultdict(list)
        self._ctx = []; self._step = 0
        self._uniform = -math.log(V)

    def update(self, token):
        tid = int(token); self._step += 1
        self._ctx.append(tid); self._ctx = self._ctx[-128:]
        self._uni *= 0.999
        if tid < self.V: self._uni[tid] += 1
        if len(self._ctx) >= 2:
            p, c = self._ctx[-2], self._ctx[-1]
            self._bi[(p, c)] = self._bi.get((p, c), 0) + 1
            self._bit[p] = self._bit.get(p, 0) + 1
        if len(self._ctx) >= 3:
            p2, p1, c = self._ctx[-3], self._ctx[-2], self._ctx[-1]
            self._tri[(p2, p1, c)] = self._tri.get((p2, p1, c), 0) + 1
            self._trit[(p2, p1)] = self._trit.get((p2, p1), 0) + 1
        self._sp[tid].append(self._step)

    def get_features(self, target):
        tid = int(target); ct = self._ctx; u = self._uniform
        d = self._uni.sum() + 0.001 * V
        ul = math.log(max((self._uni[tid] + 0.001) / d, 1e-7)) if d > 0 and tid < V else u
        bl = u; tl = u; sl = u
        if len(ct) >= 1:
            tot = self._bit.get(ct[-1], 0); db = tot + 0.001 * V
            bl = math.log(max((self._bi.get((ct[-1], tid), 0) + 0.001) / db, 1e-7)) if db > 0 else u
        if len(ct) >= 2:
            ck = (ct[-2], ct[-1]); tot = self._trit.get(ck, 0); dt = tot + 0.001 * V
            tl = math.log(max((self._tri.get((ct[-2], ct[-1], tid), 0) + 0.001) / dt, 1e-7)) if dt > 0 else u
            sk = ct[-2]; tot = self._bit.get(sk, 0); ds = tot + 0.001 * V
            sl = math.log(max((self._bi.get((sk, tid), 0) + 0.001) / ds, 1e-7)) if ds > 0 else u
        pos = self._sp.get(tid, []); gap = 128 if not pos else min(128, self._step - pos[-1])
        rl = math.log(max(1.0 / max(gap, 1), 1e-7))
        return [ul, bl, tl, sl, rl, 0.0, 0.0, 0.0, 0.0]

    def reset(self):
        self._uni = np.zeros(V, dtype=np.float32)
        self._bi = {}; self._bit = {}
        self._tri = {}; self._trit = {}
        self._sp = defaultdict(list)
        self._ctx = []; self._step = 0

def compute_weights(tokens, channels, device, window_avg=16):
    channels.reset()
    feats = []
    for t, tid in enumerate(tokens):
        if t > 0: channels.update(tokens[t - 1])
        feats.append(channels.get_features(int(tid)))
    if not feats:
        return torch.zeros(1, C_ACTIVE, device=device)
    arr = np.array(feats, dtype=np.float32)
    if window_avg and window_avg > 1:
        smoothed = np.zeros_like(arr)
        for i in range(arr.shape[0]):
            start = max(0, i - window_avg + 1)
            smoothed[i] = arr[start:i + 1].mean(axis=0)
        arr = smoothed
    return torch.tensor(arr, device=device)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--base-model', type=str, default='artifacts/steerer_stream/steerer_best_b.pt',
                   help='Frozen base model checkpoint')
    p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--steps', type=int, default=500)
    p.add_argument('--batch', type=int, default=4)
    p.add_argument('--seq-len', type=int, default=128)
    p.add_argument('--lr', type=float, default=1e-2)
    p.add_argument('--out-dir', type=str, default='artifacts/steerer_code')
    p.add_argument('--device', type=str, default='cuda')
    args = p.parse_args()

    device = torch.device(args.device)
    out_dir = Path(DEEPSEEK / args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    print('=' * 60)
    print(' CODE CARTRIDGE TRAINING (frozen base model)')
    print('=' * 60)

    print('[load] Code data...')
    data_dir = DEEPSEEK / 'artifacts/code_steerer'
    train_ids = torch.load(data_dir / 'train_ids.pt', weights_only=False).long()
    val_ids = torch.load(data_dir / 'validation_ids.pt', weights_only=False).long()
    print(f'  Train: {len(train_ids):,}  Val: {len(val_ids):,}')

    print('[load] Frozen base model...')
    base_ckpt = torch.load(DEEPSEEK / args.base_model, map_location=device, weights_only=False)
    s = base_ckpt['state_dict']
    d_model = s['pos_emb.weight'].shape[-1]; vocab = s['head_bias'].shape[0]
    d_ff = s['encoder.layers.0.linear1.weight'].shape[0]
    n_layers = len([k for k in s if k.startswith('encoder.layers.') and k.endswith('.norm1.weight')])
    n_heads = s['encoder.layers.0.self_attn.in_proj_weight'].shape[0] // (3 * d_model)
    max_len = s['pos_emb.weight'].shape[0]
    model = DeepCausalLM(vocab=vocab, d_model=d_model, n_layers=n_layers,
                         n_heads=n_heads, d_ff=d_ff, max_len=max_len, dropout=0.0)
    model.load_state_dict(s)
    for p in model.parameters(): p.requires_grad = False
    model = model.to(device); model.eval()
    print(f'  {sum(p.numel() for p in model.parameters()):,} params (FROZEN)')

    print('[build] Fresh code steerer...')
    steerer = SuperpositionSteerer(num_channels=C_ACTIVE, d_model=d_model,
                                    inject_layers=[0, 4, 8], init_scale=0.01)
    steerer = steerer.to(device)
    steerer.register_hooks(model)
    print(f'  Steerer: 3 hooks, {sum(p.numel() for p in steerer.parameters()):,} params')

    channels = CodeChannelFeatures()
    opt = torch.optim.AdamW(steerer.parameters(), lr=args.lr, weight_decay=0.1)

    N = len(train_ids)
    best_eval_s = float('inf')

    for ep in range(1, args.epochs + 1):
        steerer.train()
        total_loss = 0.0; t0 = time.time()

        for step in range(args.steps):
            starts = torch.randint(0, max(1, N - args.seq_len - 1), (args.batch,))
            x = torch.stack([train_ids[s:s+args.seq_len] for s in starts]).to(device)
            y = torch.stack([train_ids[s+1:s+args.seq_len+1] for s in starts]).to(device)

            batch_w = []
            for b in range(args.batch):
                ctx = train_ids[starts[b]:starts[b]+args.seq_len].tolist()
                batch_w.append(compute_weights(ctx, channels, device))
            w = torch.stack(batch_w, dim=0)
            steerer.set_weights(w)

            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, vocab), y.reshape(-1))
            loss = loss + 0.001 * steerer.orthogonal_penalty()

            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(steerer.parameters(), 1.0)
            opt.step()
            total_loss += loss.item()

        # Eval (steered)
        model.eval(); steerer.eval()
        with torch.no_grad():
            eval_nll, eval_n = 0.0, 0
            for s in range(0, min(len(val_ids) - 1, 5000), 64):
                cl = min(64, len(val_ids) - s - 1)
                if cl <= 0: continue
                ctx = val_ids[s:s+cl].tolist()
                w_e = compute_weights(ctx, channels, device).unsqueeze(0)
                steerer.set_weights(w_e)
                inp = val_ids[s:s+cl].unsqueeze(0).to(device)
                tgt = val_ids[s+1:s+cl+1].unsqueeze(0).to(device)
                l = model(inp)
                loss_v = F.cross_entropy(l.reshape(-1, vocab), tgt.reshape(-1), reduction='sum')
                eval_nll += loss_v.item(); eval_n += cl
        eval_ppl = math.exp(eval_nll / max(eval_n, 1))

        avg_loss = total_loss / args.steps; elapsed = time.time() - t0
        status = ''
        if eval_ppl < best_eval_s:
            best_eval_s = eval_ppl
            torch.save({'steerer_state': steerer.state_dict(),
                        'eval_s': eval_ppl, 'epoch': ep, 'opt_state': opt.state_dict()},
                       out_dir / 'code_cartridge.pt')
            status = 'SAVED'

        print(f'  epoch={ep:2d}  loss={avg_loss:.4f}  ppl={math.exp(avg_loss):.1f}  '
              f'eval_s={eval_ppl:.1f}  best={best_eval_s:.1f}  {status}  time={elapsed:.0f}s',
              flush=True)

    print(f'\nDone. Best code cartridge eval_s: {best_eval_s:.1f}')

if __name__ == '__main__':
    main()
