"""train_steerer_bpe8000.py — SuperpositionSteerer for BPE-8000 (11.6M params).
Per-position injection + Gemini upgrades: RMS norm, noise injection, ortho penalty.
"""
import sys
from hybrid.config import REPO_ROOT, time, math, argparse, importlib.util
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

DEEPSEEK = Path(__file__).resolve().parent.parent
LLM = Path('/home/drawson/llm_decoupling')
sys.path.insert(0, str(DEEPSEEK))
sys.path.append(str(LLM))

from hybrid.superposition_steerer import SuperpositionSteerer
from compile_wiki_lm_v13 import load_setup, load_or_build_tokens


def load_model_ckpt(ckpt_path, device):
    _spec = importlib.util.spec_from_file_location(
        'bpe8000', str(DEEPSEEK / 'hybrid/train_hybrid_bpe8000.py'))
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    BPE8000LM = _mod.BPE8000LM
    
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt['state_dict']
    d_model = state['pos_emb.weight'].shape[-1]
    max_len = state['pos_emb.weight'].shape[0]
    vocab = state['head_bias'].shape[0]
    d_ff = state['encoder.layers.0.linear1.weight'].shape[0]
    n_layers = len([k for k in state if k.startswith('encoder.layers.') and
                    k.endswith('.norm1.weight')])
    n_heads = state['encoder.layers.0.self_attn.in_proj_weight'].shape[0] // (3 * d_model)
    
    model = BPE8000LM(vocab=vocab, d_model=d_model, n_layers=n_layers,
                      n_heads=n_heads, d_ff=d_ff, max_len=max_len, dropout=0.0)
    model.load_state_dict(state)
    model = model.to(device)
    return model, d_model


class BPE8000ChannelFeatures:
    def __init__(self, V=8000, emb=None):
        self.V = V
        self.emb = emb
        self._uni_counts = np.zeros(V, dtype=np.float32)
        self._bi_cache = {}
        self._bi_totals = {}
        self._tri_cache = {}
        self._tri_totals = {}
        self._global_uni = None
        self._seen_positions = defaultdict(list)
        self._context = []
        self._step = 0
    
    def set_global_uni(self, train_ids):
        counts = np.bincount(train_ids.numpy().astype(np.int64), minlength=self.V).astype(np.float64)
        self._global_uni = np.log(np.maximum((counts + 0.1) / (counts.sum() + 0.1 * self.V), 1e-7))
    
    def update(self, token: int):
        tid = int(token)
        self._step += 1
        self._context.append(tid)
        self._context = self._context[-8:]
        self._uni_counts *= 0.999
        if tid < self.V: self._uni_counts[tid] += 1
        if len(self._context) >= 2:
            prev, curr = self._context[-2], self._context[-1]
            key = (prev, curr)
            self._bi_cache[key] = self._bi_cache.get(key, 0) + 1
            self._bi_totals[prev] = self._bi_totals.get(prev, 0) + 1
        if len(self._context) >= 3:
            p2, p1, c = self._context[-3], self._context[-2], self._context[-1]
            key = (p2, p1, c)
            ctx_key = (p2, p1)
            self._tri_cache[key] = self._tri_cache.get(key, 0) + 1
            self._tri_totals[ctx_key] = self._tri_totals.get(ctx_key, 0) + 1
        if self._step % 10 == 0:
            for cch, tt in [(self._bi_cache, self._bi_totals), (self._tri_cache, self._tri_totals)]:
                for k in list(cch):
                    cch[k] *= 0.999
                    if cch[k] < 1e-6: del cch[k]
                for k in list(tt):
                    tt[k] *= 0.999
                    if tt[k] < 1e-6: del tt[k]
        self._seen_positions[tid].append(self._step)
    
    def get_features(self, target: int) -> list[float]:
        tid = int(target); ctx = self._context; V = self.V; uniform = -math.log(V)
        
        d = self._uni_counts.sum() + 0.001 * V
        uni_lp = math.log(max((self._uni_counts[tid] + 0.001) / d, 1e-7)) if d > 0 and tid < V else uniform
        
        bi_lp = uniform
        if len(ctx) >= 1:
            total = self._bi_totals.get(ctx[-1], 0); db = total + 0.001 * V
            if db > 0: bi_lp = math.log(max((self._bi_cache.get((ctx[-1], tid), 0) + 0.001) / db, 1e-7))
        
        tri_lp = uniform
        if len(ctx) >= 2:
            ck = (ctx[-2], ctx[-1]); total = self._tri_totals.get(ck, 0); dt = total + 0.001 * V
            if dt > 0: tri_lp = math.log(max((self._tri_cache.get((ctx[-2], ctx[-1], tid), 0) + 0.001) / dt, 1e-7))
        
        skip_lp = uniform
        if len(ctx) >= 2:
            sk = ctx[-2]; total = self._bi_totals.get(sk, 0); ds = total + 0.001 * V
            if ds > 0: skip_lp = math.log(max((self._bi_cache.get((sk, tid), 0) + 0.001) / ds, 1e-7))
        
        positions = self._seen_positions.get(tid, [])
        gap = 128 if not positions else min(128, self._step - positions[-1])
        rec_lp = math.log(max(1.0 / max(gap, 1), 1e-7))
        
        ppmi_cos = 0.0; ppmi_max_cos = 0.0; ppmi_norm = 0.0
        if self.emb is not None and len(ctx) >= 1:
            te = self.emb[tid].float().numpy()
            ce = np.stack([self.emb[t].float().numpy() for t in ctx[-4:]])
            ca = ce.mean(axis=0); cn = np.linalg.norm(ca); tn = np.linalg.norm(te)
            if cn > 0 and tn > 0: ppmi_cos = float(np.dot(ca, te) / (cn * tn))
            for ct in ctx[-4:]:
                ce_t = self.emb[ct].float().numpy(); cn2 = np.linalg.norm(ce_t)
                if cn2 > 0 and tn > 0: ppmi_max_cos = max(ppmi_max_cos, float(np.dot(ce_t, te) / (cn2 * tn)))
            ppmi_norm = float(tn)
        
        global_uni = 0.0
        if self._global_uni is not None:
            global_uni = float(self._global_uni[tid])
        
        return [uni_lp, bi_lp, tri_lp, skip_lp, rec_lp,
                ppmi_cos, ppmi_max_cos, ppmi_norm, global_uni]
    
    def reset(self):
        self._uni_counts = np.zeros(self.V, dtype=np.float32)
        self._bi_cache = {}; self._bi_totals = {}
        self._tri_cache = {}; self._tri_totals = {}
        self._seen_positions = defaultdict(list)
        self._context = []; self._step = 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--neural-ckpt', type=str, default='artifacts/hybrid_256_l12_x50/best.pt')
    p.add_argument('--resume', type=str, default=None)
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--steps-per-epoch', type=int, default=500)
    p.add_argument('--batch', type=int, default=8)
    p.add_argument('--lr', type=float, default=1e-2)
    p.add_argument('--out-dir', type=str, default='artifacts/steerer_bpe8000')
    p.add_argument('--device', type=str, default='cuda')
    args = p.parse_args()

    device = torch.device(args.device)
    out_dir = Path(DEEPSEEK / args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print('=' * 60)
    print(' BPE-8000 STEERER TRAINING')
    print('=' * 60)

    print('[1/4] Loading BPE-8000 setup...')
    bpe, vocab, tok2id, bpe_to_lm, emb, V, d_emb = load_setup()
    print(f'  V={V}, emb={emb.shape}')
    
    print('[2/4] Loading tokenized data...')
    train_ids = load_or_build_tokens(bpe, bpe_to_lm, V).long()
    split = int(len(train_ids) * 0.9)
    val_ids = train_ids[split:split + 50000]
    train_ids = train_ids[:split]
    print(f'  Train: {len(train_ids):,}  Val: {len(val_ids):,}')

    print('[3/4] Loading neural LM...')
    model, d_model = load_model_ckpt(str(Path(DEEPSEEK / args.neural_ckpt)), device)
    for p in model.parameters():
        p.requires_grad = True
    n_params = sum(p.numel() for p in model.parameters())
    print(f'  {n_params:,} params (trainable), d_model={d_model}')

    C_ACTIVE = 9
    steerer = SuperpositionSteerer(num_channels=C_ACTIVE, d_model=d_model,
                                    inject_layers=[0, 4, 8], init_scale=0.01)
    steerer = steerer.to(device)

    channels = BPE8000ChannelFeatures(V=V, emb=emb)
    channels.set_global_uni(train_ids)

    opt = torch.optim.AdamW([
        {'params': model.parameters(), 'lr': 1e-4},
        {'params': steerer.parameters(), 'lr': args.lr},
    ], weight_decay=0.1)

    start_epoch = 0
    if args.resume:
        resume_ckpt = torch.load(Path(DEEPSEEK / args.resume), map_location=device, weights_only=False)
        steerer.load_state_dict(resume_ckpt['steerer_state'])
        start_epoch = resume_ckpt.get('epoch', 0)
        if 'opt_state' in resume_ckpt:
            opt.load_state_dict(resume_ckpt['opt_state'])
        print(f'[RESUME] epoch {start_epoch}, eval {resume_ckpt.get("eval_ppl", resume_ckpt.get("eval_s","?"))}')

    n_hooks = steerer.register_hooks(model)
    print(f'  Steerer: {n_hooks} hooks, {sum(p.numel() for p in steerer.parameters()):,} params ({C_ACTIVE} channels)')

    N = len(train_ids)
    best_eval_ppl = float('inf')
    best_eval_b = float('inf')

    for ep in range(start_epoch + 1, args.epochs + 1):
        model.train()
        steerer.train()
        total_loss = 0.0
        t0 = time.time()
        channels.reset()

        for step in range(args.steps_per_epoch):
            seq_len = 64
            max_start = N - seq_len - 1
            starts = torch.randint(0, max(1, max_start), (args.batch,))
            x = torch.stack([train_ids[s:s+seq_len] for s in starts]).to(device)
            y = torch.stack([train_ids[s+1:s+seq_len+1] for s in starts]).to(device)

            batch_w = []
            for b in range(args.batch):
                channels.reset()
                ctx = train_ids[starts[b]:starts[b]+seq_len].tolist()
                for tid in ctx[:-1]:
                    channels.update(tid)
                seq_w = []
                for t in ctx[1:min(len(ctx), 65)]:
                    seq_w.append(channels.get_features(t))
                if not seq_w:
                    seq_w = [[0.0] * C_ACTIVE]
                batch_w.append(seq_w)
            w = torch.tensor(batch_w, dtype=torch.float32, device=device)  # (B, T, C)
            steerer.set_weights(w)

            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, V), y.reshape(-1))
            loss = loss + 0.001 * steerer.orthogonal_penalty()

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(steerer.parameters()), 1.0)
            opt.step()
            total_loss += loss.item()

        # Eval: baseline (steerer OFF) + steered (steerer ON)
        model.eval(); steerer.eval()

        def eval_ppl_fn(use_steerer):
            channels.reset()
            nll, n = 0.0, 0
            for s in range(0, len(val_ids) - 1, 64):
                cl = min(64, len(val_ids) - s - 1)
                if cl <= 0: continue
                inp = val_ids[s:s+cl].unsqueeze(0).to(device)
                tgt = val_ids[s+1:s+cl+1].unsqueeze(0).to(device)
                if use_steerer:
                    channels.reset()
                    ctx = val_ids[s:s+cl].tolist()
                    for tid in ctx[:-1]:
                        channels.update(tid)
                    seq_w = [channels.get_features(t) for t in ctx[1:cl+1]]
                    if not seq_w:
                        seq_w = [[0.0] * C_ACTIVE]
                    w_t = torch.tensor(seq_w, dtype=torch.float32, device=device).unsqueeze(0)
                    steerer.set_weights(w_t)
                else:
                    steerer._current_weights = None
                l = model(inp)
                loss_v = F.cross_entropy(l.reshape(-1, V), tgt.reshape(-1), reduction='sum')
                nll += loss_v.item(); n += cl
            return math.exp(nll / max(n, 1))

        with torch.no_grad():
            eval_steered = eval_ppl_fn(use_steerer=True)
            eval_baseline = eval_ppl_fn(use_steerer=False)

        avg_loss = total_loss / args.steps_per_epoch
        elapsed = time.time() - t0
        status = ''
        if eval_baseline < best_eval_b:
            best_eval_b = eval_baseline
            status += 'b'
        if eval_steered < best_eval_ppl:
            best_eval_ppl = eval_steered
            status += 's'
        if 'b' in status:
            torch.save({'state_dict': model.state_dict(),
                        'steerer_state': steerer.state_dict(),
                        'eval_s': eval_steered, 'eval_b': eval_baseline,
                        'epoch': ep, 'opt_state': opt.state_dict()},
                       out_dir / 'steerer_best_b.pt')
        if 's' in status:
            torch.save({'state_dict': model.state_dict(),
                        'steerer_state': steerer.state_dict(),
                        'eval_s': eval_steered, 'eval_b': eval_baseline,
                        'epoch': ep, 'opt_state': opt.state_dict()},
                       out_dir / 'steerer_best_s.pt')
        print(f'  epoch={ep:2d}  loss={avg_loss:.4f}  ppl={math.exp(avg_loss):.1f}  '
              f'eval_s={eval_steered:.1f}  eval_b={eval_baseline:.1f}  '
              f'best_b={best_eval_b:.1f}  [{status}]  time={elapsed:.0f}s', flush=True)

    print(f'\nDone. Best standalone eval_b: {best_eval_b:.1f}  Best steered eval_s: {best_eval_ppl:.1f}')


if __name__ == '__main__':
    main()
