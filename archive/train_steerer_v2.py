"""train_steerer_v2.py — Train MLPSuperpositionSteerer with 15-channel streaming
features on full train set.

Computes per-channel scalar features on-the-fly (O(1) per token, not O(V)).
Trains steerer + neural LM jointly. Evaluates on held-out val set.
Uses layer-targeted injection with per-group MLPs.
"""
from __future__ import annotations

import sys
from hybrid.config import REPO_ROOT, time, math, argparse, importlib.util
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

DEEPSEEK = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(DEEPSEEK))

from hybrid.superposition_steerer_v2 import MLPSuperpositionSteerer


CHANNEL_NAMES_V2 = [
    "unigram",
    "bigram_fast",
    "bigram_slow",
    "trigram_fast",
    "trigram_slow",
    "skip2",
    "skip3",
    "recency",
    "builder_entropy",
    "shape",
    "unigram_global",
    "PPMI_cosine",
    "PPMI_max",
    "PPMI_norm",
    "bigram_contrast",
]
N_CHANNELS = len(CHANNEL_NAMES_V2)
SHAPE_IDX = 9
ACTIVE_CHANNELS = [i for i in range(N_CHANNELS) if i != SHAPE_IDX]

GROUP_CHANNELS = {
    'local': [0, 1, 2],
    'mid': [3, 4, 5, 6],
    'global': [7, 8, 10, 11, 12, 13, 14],
}
GROUP_TO_LAYERS = {
    'local': [0, 1],
    'mid': [3, 4, 5, 6],
    'global': [8, 9, 10],
}


def build_ppmi_stats(train_ids: torch.Tensor, V: int,
                     max_tokens: int = 500000) -> dict:
    """Build sparse PPMI co-occurrence statistics from training data.

    Returns dict with:
      ppmi: dict (ctx, tgt) -> PPMI value for observed pairs
      ppmi_norm: dict token -> sqrt(sum PPMI(token, k)^2) for row norms
      total_bigrams: int
    """
    use_tokens = train_ids[:min(len(train_ids), max_tokens)].long().numpy()
    T = len(use_tokens)
    pair_counts: dict[tuple, float] = {}
    unigram_counts = np.zeros(V, dtype=np.float64)

    for t in range(1, T):
        ctx = int(use_tokens[t - 1])
        tgt = int(use_tokens[t])
        unigram_counts[ctx] += 1
        unigram_counts[tgt] += 1
        key = (ctx, tgt)
        pair_counts[key] = pair_counts.get(key, 0) + 1

    total_unigrams = unigram_counts.sum()
    total_pairs = sum(pair_counts.values())
    unigram_probs = unigram_counts / max(total_unigrams, 1)

    ppmi: dict[tuple, float] = {}
    ppmi_sq_sum: dict[int, float] = defaultdict(float)

    for (ctx, tgt), count in pair_counts.items():
        p_joint = count / max(total_pairs, 1)
        p_ctx = unigram_probs[ctx]
        p_tgt = unigram_probs[tgt]
        if p_joint > 0 and p_ctx > 0 and p_tgt > 0:
            pmi = math.log(p_joint / (p_ctx * p_tgt))
            val = max(pmi, 0.0)
            if val > 1e-8:
                ppmi[(ctx, tgt)] = val
                ppmi_sq_sum[ctx] += val * val

    ppmi_norm = {t: math.sqrt(s) for t, s in ppmi_sq_sum.items()}

    return {
        'ppmi': ppmi,
        'ppmi_norm': ppmi_norm,
        'total_bigrams': total_pairs,
    }


class StreamingChannelFeaturesV2:
    """Computes 15 per-channel scalar features on-the-fly."""

    def __init__(self, V: int = 50257, ppmi_stats: dict | None = None):
        self.V = V
        self._uniform = -math.log(V)

        self._uni_counts = np.zeros(V, dtype=np.float32)
        self._bi_cache: dict[tuple, float] = {}
        self._bi_totals: dict[int, float] = {}
        self._bi_slow_cache: dict[tuple, float] = {}
        self._bi_slow_totals: dict[int, float] = {}
        self._tri_cache: dict[tuple, float] = {}
        self._tri_totals: dict[tuple, float] = {}
        self._tri_slow_cache: dict[tuple, float] = {}
        self._tri_slow_totals: dict[tuple, float] = {}
        self._skip2_cache: dict[tuple, float] = {}
        self._skip2_totals: dict[int, float] = {}
        self._skip3_cache: dict[tuple, float] = {}
        self._skip3_totals: dict[int, float] = {}

        self._context: list[int] = []
        self._step = 0
        self._seen_positions: dict[int, list[int]] = defaultdict(list)

        if ppmi_stats is not None:
            self._ppmi = ppmi_stats['ppmi']
            self._ppmi_norm = ppmi_stats['ppmi_norm']
        else:
            self._ppmi = {}
            self._ppmi_norm = {}

        self._global_uni = None
        self._decay_step = 0

    def set_global_unigram(self, train_ids: torch.Tensor):
        counts = np.bincount(
            train_ids.numpy().astype(np.int64), minlength=self.V
        ).astype(np.float64)
        total = counts.sum()
        self._global_uni = np.log(
            np.maximum((counts + 0.1) / (total + 0.1 * self.V), 1e-7)
        ).astype(np.float32)

    def update(self, token: int):
        tid = int(token)
        self._step += 1
        self._context.append(tid)
        self._context = self._context[-4:]

        if tid < self.V:
            self._uni_counts[tid] += 1

        if len(self._context) >= 2:
            prev, curr = self._context[-2], self._context[-1]
            key = (prev, curr)
            self._bi_cache[key] = self._bi_cache.get(key, 0) + 1
            self._bi_totals[prev] = self._bi_totals.get(prev, 0) + 1

        if len(self._context) >= 3:
            p2, p1, curr = self._context[-3], self._context[-2], self._context[-1]
            key = (p2, p1, curr)
            ctx_key = (p2, p1)
            self._tri_cache[key] = self._tri_cache.get(key, 0) + 1
            self._tri_totals[ctx_key] = self._tri_totals.get(ctx_key, 0) + 1
            self._skip2_cache[(p2, curr)] = self._skip2_cache.get((p2, curr), 0) + 1
            self._skip2_totals[p2] = self._skip2_totals.get(p2, 0) + 1

        if len(self._context) >= 4:
            p3 = self._context[-4]
            curr = self._context[-1]
            self._skip3_cache[(p3, curr)] = self._skip3_cache.get((p3, curr), 0) + 1
            self._skip3_totals[p3] = self._skip3_totals.get(p3, 0) + 1

        self._seen_positions[tid].append(self._step)

        self._decay_step += 1
        if self._decay_step % 16 == 0:
            _decay = 0.93
            for cache, totals in [
                (self._bi_slow_cache, self._bi_slow_totals),
                (self._tri_slow_cache, self._tri_slow_totals),
            ]:
                for k in list(cache):
                    cache[k] *= _decay
                    if cache[k] < 1e-6:
                        del cache[k]
                for k in list(totals):
                    totals[k] *= _decay
                    if totals[k] < 1e-6:
                        del totals[k]

    def _smoothed_lp(self, count: float, total: float, V: int) -> float:
        denom = total + 0.001 * V
        if denom <= 0:
            return self._uniform
        return math.log(max((count + 0.001) / denom, 1e-30))

    def get_features(self, target: int) -> list[float]:
        tid = int(target)
        ctx = self._context
        V = self.V

        feats: list[float] = []

        d = self._uni_counts.sum() + 0.001 * V
        if d > 0 and tid < V:
            uni_lp = math.log(max((self._uni_counts[tid] + 0.001) / d, 1e-30))
        else:
            uni_lp = self._uniform
        feats.append(uni_lp)

        if len(ctx) >= 1:
            prev = ctx[-1]
            total = self._bi_totals.get(prev, 0)
            count = self._bi_cache.get((prev, tid), 0) if tid < V else 0
            bi_fast_lp = self._smoothed_lp(count, total, V)
        else:
            bi_fast_lp = self._uniform
        feats.append(bi_fast_lp)

        if len(ctx) >= 1:
            prev = ctx[-1]
            total = self._bi_slow_totals.get(prev, 0)
            count = self._bi_slow_cache.get((prev, tid), 0) if tid < V else 0
            bi_slow_lp = self._smoothed_lp(count, total, V)
        else:
            bi_slow_lp = self._uniform
        feats.append(bi_slow_lp)

        if len(ctx) >= 2:
            ck = (ctx[-2], ctx[-1])
            total = self._tri_totals.get(ck, 0)
            count = self._tri_cache.get((ctx[-2], ctx[-1], tid), 0) if tid < V else 0
            tri_fast_lp = self._smoothed_lp(count, total, V)
        else:
            tri_fast_lp = self._uniform
        feats.append(tri_fast_lp)

        if len(ctx) >= 2:
            ck = (ctx[-2], ctx[-1])
            total = self._tri_slow_totals.get(ck, 0)
            count = self._tri_slow_cache.get((ctx[-2], ctx[-1], tid), 0) if tid < V else 0
            tri_slow_lp = self._smoothed_lp(count, total, V)
        else:
            tri_slow_lp = self._uniform
        feats.append(tri_slow_lp)

        if len(ctx) >= 2:
            p2 = ctx[-2]
            total = self._skip2_totals.get(p2, 0)
            count = self._skip2_cache.get((p2, tid), 0) if tid < V else 0
            skip2_lp = self._smoothed_lp(count, total, V)
        else:
            skip2_lp = self._uniform
        feats.append(skip2_lp)

        if len(ctx) >= 3:
            p3 = ctx[-3]
            total = self._skip3_totals.get(p3, 0)
            count = self._skip3_cache.get((p3, tid), 0) if tid < V else 0
            skip3_lp = self._smoothed_lp(count, total, V)
        else:
            skip3_lp = self._uniform
        feats.append(skip3_lp)

        positions = self._seen_positions.get(tid, [])
        gap = 128 if not positions else min(128, self._step - positions[-1])
        rec_lp = math.log(max(1.0 / max(gap, 1), 1e-30))
        feats.append(rec_lp)

        d = self._uni_counts.sum() + 0.001 * V
        if d > 0:
            probs = (self._uni_counts + 0.001) / d
            valid = probs > 0
            entropy = -np.sum(probs[valid] * np.log(probs[valid]))
            ent_norm = float(entropy / math.log(V)) if entropy > 0 else 1.0
        else:
            ent_norm = 1.0
        feats.append(ent_norm)

        feats.append(0.0)

        if self._global_uni is not None and tid < V:
            gu_lp = float(self._global_uni[tid])
        else:
            gu_lp = 0.0
        feats.append(gu_lp)

        ppmi_cos = 0.0
        ppmi_max = 0.0
        ppmi_norm_feat = 0.0
        if self._ppmi and len(ctx) >= 1:
            prev = ctx[-1]
            ppmi_val = self._ppmi.get((prev, tid), 0.0)
            norm_ctx = self._ppmi_norm.get(prev, 0.0)
            norm_tgt = self._ppmi_norm.get(tid, 0.0)
            denom = max(norm_ctx * norm_tgt, 1e-8)
            ppmi_cos = ppmi_val / denom if denom > 1e-8 else 0.0

            for ct in ctx[-4:]:
                v = self._ppmi.get((ct, tid), 0.0)
                if v > ppmi_max:
                    ppmi_max = v

            ppmi_norm_feat = norm_tgt
        feats.append(ppmi_cos)
        feats.append(ppmi_max)
        feats.append(ppmi_norm_feat)

        if len(ctx) >= 1:
            prev = ctx[-1]
            uni_p = max((self._uni_counts[tid] + 0.001) / (self._uni_counts.sum() + 0.001 * V), 1e-7) if tid < V else 1e-7
            bi_total = self._bi_totals.get(prev, 0) + 0.001 * V
            bi_p = max((self._bi_cache.get((prev, tid), 0) + 0.001) / bi_total, 1e-7) if bi_total > 0 else 1e-7
            contrast = math.log(max(bi_p / max(uni_p, 1e-7), 1e-30))
        else:
            contrast = 0.0
        feats.append(contrast)

        return feats

    def reset(self):
        self._uni_counts = np.zeros(self.V, dtype=np.float32)
        self._bi_cache = {}
        self._bi_totals = {}
        self._bi_slow_cache = {}
        self._bi_slow_totals = {}
        self._tri_cache = {}
        self._tri_totals = {}
        self._tri_slow_cache = {}
        self._tri_slow_totals = {}
        self._skip2_cache = {}
        self._skip2_totals = {}
        self._skip3_cache = {}
        self._skip3_totals = {}
        self._context = []
        self._step = 0
        self._seen_positions = defaultdict(list)
        self._decay_step = 0


def compute_channel_weights_v2(tokens, channels, device, window_avg=16):
    """Compute per-token channel features with optional window averaging.

    Returns (T, 15) tensor on device. Channel 9 (shape) is set to 0.0.
    """
    channels.reset()
    feats_per_pos = []
    for t, tid in enumerate(tokens):
        if t > 0:
            channels.update(tokens[t - 1])
        fs = channels.get_features(int(tid))
        feats_per_pos.append(fs)
    if not feats_per_pos:
        return torch.zeros(1, N_CHANNELS, device=device)
    feat_arr = np.array(feats_per_pos, dtype=np.float32)
    if window_avg is not None and window_avg > 1:
        smoothed = np.zeros_like(feat_arr)
        for i in range(feat_arr.shape[0]):
            start = max(0, i - window_avg + 1)
            smoothed[i] = feat_arr[start:i + 1].mean(axis=0)
        feat_arr = smoothed
    return torch.tensor(feat_arr, device=device)


def load_neural_lm(ckpt_path, device):
    _spec = importlib.util.spec_from_file_location(
        'train_scaled', str(DEEPSEEK / 'hybrid/train_scaled_neural_lm.py'))
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    DeepCausalLM = _mod.DeepCausalLM

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt['state_dict']
    d_model = state['pos_emb.weight'].shape[-1]
    max_len = state['pos_emb.weight'].shape[0]
    if 'tok_emb.weight' in state:
        vocab = state['tok_emb.weight'].shape[0]
    else:
        vocab = state['head_bias'].shape[0]

    if 'encoder.layers.0.linear1.weight' in state:
        d_ff = state['encoder.layers.0.linear1.weight'].shape[0]
        n_layers = len([k for k in state if k.startswith('encoder.layers.') and
                        k.endswith('.norm1.weight')])
        n_heads = state['encoder.layers.0.self_attn.in_proj_weight'].shape[0] // (3 * d_model)
    else:
        d_ff = state['layers.0.ff_1.weight'].shape[0]
        n_layers = len([k for k in state if k.startswith('layers.') and
                        k.endswith('.sa_norm.weight')])
        n_heads = state['layers.0.sa_q.weight'].shape[0] // d_model

    model = DeepCausalLM(vocab=vocab, d_model=d_model, n_layers=n_layers,
                         n_heads=n_heads, d_ff=d_ff, max_len=max_len, dropout=0.0)
    model.load_state_dict(state)
    model = model.to(device)
    return model, d_model


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--neural-ckpt', type=str,
                   default='artifacts/c4_v2_768_x30/best.pt')
    p.add_argument('--resume', type=str, default=None,
                   help='Resume from steerer checkpoint')
    p.add_argument('--resume-model', type=str, default=None,
                   help='Load model weights from v1 checkpoint, keep v2 steerer fresh')
    p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--steps-per-epoch', type=int, default=500)
    p.add_argument('--batch', type=int, default=8)
    p.add_argument('--seq-len', type=int, default=128)
    p.add_argument('--lr', type=float, default=1e-2)
    p.add_argument('--out-dir', type=str, default='artifacts/steerer_v2')
    p.add_argument('--device', type=str, default='cuda')
    args = p.parse_args()

    device = torch.device(args.device)
    out_dir = Path(DEEPSEEK / args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    V = 50257

    print('=' * 60)
    print(' STEERER V2 TRAINING (MLP gatekeeper + layer-targeted)')
    print('=' * 60)

    train_ids = torch.load(
        DEEPSEEK / 'artifacts/wikitext_gpt2/train_ids.pt', weights_only=False
    ).long()
    val_ids = torch.load(
        DEEPSEEK / 'artifacts/wikitext_gpt2/validation_ids.pt', weights_only=False
    ).long()
    print(f'Train: {len(train_ids):,}  Val: {len(val_ids):,}')

    print('[load] Neural LM...')
    model, d_model = load_neural_lm(DEEPSEEK / args.neural_ckpt, device)
    for p in model.parameters():
        p.requires_grad = True
    n_params = sum(p.numel() for p in model.parameters())
    print(f'  {n_params:,} params (trainable)  d_model={d_model}')

    print('[build] PPMI co-occurrence stats from training data...')
    t0_ppmi = time.time()
    ppmi_stats = build_ppmi_stats(train_ids, V, max_tokens=500000)
    print(f'  {len(ppmi_stats["ppmi"]):,} token pairs  '
          f'({time.time() - t0_ppmi:.1f}s)')

    steerer = MLPSuperpositionSteerer(
        num_channels=N_CHANNELS,
        d_model=d_model,
        group_channels=GROUP_CHANNELS,
        group_to_layers=GROUP_TO_LAYERS,
        init_scale=0.01,
        hidden_dim=32,
    )
    steerer = steerer.to(device)

    start_epoch = 0
    best_eval_b = float('inf')
    best_eval_s = float('inf')

    if args.resume_model:
        print(f'[WARM-START] Loading model weights from {args.resume_model}...')
        warm_ckpt = torch.load(
            Path(DEEPSEEK / args.resume_model), map_location=device, weights_only=False)
        model.load_state_dict(warm_ckpt['state_dict'])
        print(f'  Model loaded. V2 steerer initialized fresh.')

    if args.resume:
        resume_ckpt = torch.load(
            Path(DEEPSEEK / args.resume), map_location=device, weights_only=False)
        model.load_state_dict(resume_ckpt['state_dict'])
        steerer.load_state_dict(resume_ckpt['steerer_state'], strict=False)
        start_epoch = resume_ckpt.get('epoch', 0)
        best_eval_b = resume_ckpt.get('best_eval_b', float('inf'))
        print(f'[RESUME] epoch {start_epoch}, best_eval_b {best_eval_b:.1f}')

    n_hooks = steerer.register_hooks(model)
    s_params = sum(p.numel() for p in steerer.parameters())
    print(f'  Steerer: {n_hooks} hooks, {s_params:,} params')
    for gname, gch in GROUP_CHANNELS.items():
        layers = GROUP_TO_LAYERS[gname]
        print(f'    {gname}: channels {gch} -> layers {layers}')

    channels = StreamingChannelFeaturesV2(V=V, ppmi_stats=ppmi_stats)
    channels.set_global_unigram(train_ids)

    opt = torch.optim.AdamW([
        {'params': model.parameters(), 'lr': 3e-5},
        {'params': steerer.parameters(), 'lr': args.lr},
    ], weight_decay=0.1)

    N = len(train_ids)
    model_max_len = model.pos_emb.weight.shape[0] - 1

    for ep in range(start_epoch + 1, args.epochs + 1):
        model.train()
        steerer.train()
        total_loss = 0.0
        t0 = time.time()

        for step in range(args.steps_per_epoch):
            starts = torch.randint(0, max(1, N - args.seq_len - 1),
                                   (args.batch,))
            x = torch.stack([
                train_ids[s:s + args.seq_len] for s in starts
            ]).to(device)
            y = torch.stack([
                train_ids[s + 1:s + args.seq_len + 1] for s in starts
            ]).to(device)

            all_weights = []
            for b in range(args.batch):
                ctx_tokens = train_ids[starts[b]:starts[b] + args.seq_len].tolist()
                w = compute_channel_weights_v2(ctx_tokens, channels, device,
                                               window_avg=16)
                all_weights.append(w)
            weights = torch.stack(all_weights, dim=0)
            steerer.set_weights(weights)

            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, V), y.reshape(-1))
            loss = loss + 0.001 * steerer.orthogonal_penalty()

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(steerer.parameters()), 1.0)
            opt.step()
            total_loss += loss.item()

        model.eval()
        steerer.eval()

        eval_chunk_len = min(args.seq_len, model_max_len, len(val_ids) - 1)
        eval_tokens = val_ids[:eval_chunk_len].tolist()

        steerer._current_weights = None
        with torch.no_grad():
            eval_b_nll, eval_b_n = 0.0, 0
            for s in range(0, len(val_ids) - 1, 64):
                cl = min(64, len(val_ids) - s - 1)
                if cl <= 0:
                    continue
                inp = val_ids[s:s + cl].unsqueeze(0).to(device)
                tgt = val_ids[s + 1:s + cl + 1].unsqueeze(0).to(device)
                l = model(inp)
                loss_v = F.cross_entropy(l.reshape(-1, V), tgt.reshape(-1),
                                         reduction='sum')
                eval_b_nll += loss_v.item()
                eval_b_n += cl
        eval_b = math.exp(eval_b_nll / max(eval_b_n, 1))

        w_eval = compute_channel_weights_v2(eval_tokens, channels, device,
                                            window_avg=16)
        w_eval = w_eval.unsqueeze(0)
        steerer.set_weights(w_eval)
        with torch.no_grad():
            inp_s = val_ids[:eval_chunk_len].unsqueeze(0).to(device)
            tgt_s = val_ids[1:eval_chunk_len + 1].unsqueeze(0).to(device)
            l_s = model(inp_s)
            loss_s = F.cross_entropy(l_s.reshape(-1, V), tgt_s.reshape(-1),
                                     reduction='sum')
            eval_s = math.exp(loss_s.item() / eval_chunk_len)

        avg_loss = total_loss / args.steps_per_epoch
        elapsed = time.time() - t0
        status = ''
        if eval_b < best_eval_b:
            best_eval_b = eval_b
            status += 'b'
        if eval_s < best_eval_s:
            best_eval_s = eval_s
            status += 's'
        if 'b' in status:
            torch.save({
                'state_dict': model.state_dict(),
                'steerer_state': steerer.state_dict(),
                'opt_state': opt.state_dict(),
                'eval_b': eval_b,
                'eval_s': eval_s,
                'best_eval_b': best_eval_b,
                'best_eval_s': best_eval_s,
                'epoch': ep,
            }, out_dir / 'steerer_best_b.pt')
        if 's' in status:
            torch.save({
                'state_dict': model.state_dict(),
                'steerer_state': steerer.state_dict(),
                'opt_state': opt.state_dict(),
                'eval_b': eval_b,
                'eval_s': eval_s,
                'best_eval_b': best_eval_b,
                'best_eval_s': best_eval_s,
                'epoch': ep,
            }, out_dir / 'steerer_best_s.pt')
        print(f'  epoch={ep:3d}  loss={avg_loss:.4f}  ppl={math.exp(avg_loss):.1f}  '
              f'eval_s={eval_s:.1f}  eval_b={eval_b:.1f}  '
              f'best_b={best_eval_b:.1f}  [{status}]  '
              f'time={elapsed:.0f}s', flush=True)

    print(f'\nDone. Best eval_b: {best_eval_b:.1f}  Best eval_s: {best_eval_s:.1f}')


if __name__ == '__main__':
    main()
