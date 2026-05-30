"""dump_gpt2_channels_v3.py — Rich compiled channels using full builder inventory.

Uses GPT2CompiledChannelBuilder's unigram/bigram/trigram/skip2/skip3 counts
plus streaming decay caches, shape, and recency. Produces denser channel data
for steerer training.
"""
from __future__ import annotations

import argparse, math, sys, time
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np
import torch

DEEPSEEK = Path('/home/drawson/deepseek_experiments')
sys.path.insert(0, str(DEEPSEEK))
from hybrid.compiled_features import GPT2CompiledChannelBuilder

# 15 channels: builder's full inventory + decay caches + shape + recency
CHANNEL_NAMES = [
    "uni", "bi", "tri", "skip2", "skip3",   # builder counts
    "dc_uni_f", "dc_uni_s",                   # decay unigram fast/slow
    "dc_bi_f", "dc_bi_s",                     # decay bigram fast/slow
    "dc_tri_f", "dc_tri_s",                   # decay trigram fast/slow
    "shape", "recency",                       # shape transitions + recency
    "builder_entropy",                        # builder entropy norm
]
V = 50257


def builder_logp(builder, channel, context, token):
    """Get Laplace-smoothed log-prob from builder counts."""
    alpha = builder.cfg.alpha
    if channel == "uni":
        total = builder.total_tokens
        return math.log(max((builder.unigram.get(token, 0) + alpha) /
                           (total + alpha * V), 1e-30))
    counts_map = {
        "bi": (builder.bigram, 0),
        "skip2": (builder.skip2, 0),
        "skip3": (builder.skip3, 0),
        "tri": (builder.trigram, 1),
    }
    if channel not in counts_map:
        return -math.log(V)
    cnt, ctx_idx = counts_map[channel]
    if ctx_idx == 0:  # single-key context
        ctx_counts = cnt.get(context[-1] if context else None, Counter())
    else:  # tuple-key context
        if len(context) >= 2:
            ctx_counts = cnt.get((context[-2], context[-1]), Counter())
        else:
            return -math.log(V)
    total = sum(ctx_counts.values())
    return math.log(max((ctx_counts.get(token, 0) + alpha) /
                       (total + alpha * V), 1e-30))


def build_decay_lp(ids: np.ndarray, order: int, alpha: float) -> np.ndarray:
    T = len(ids)
    lp = np.full(T, -math.log(V), dtype=np.float32)
    if order == 1:
        counts = np.zeros(V, dtype=np.float32)
        for t in range(T):
            target = int(ids[t])
            d = counts.sum() + alpha * V
            if d > 0:
                lp[t] = math.log(max((counts[target] + alpha) / d, 1e-30))
            counts[target] += 1
            counts *= (1 - alpha)
    elif order == 2:
        cache = {}
        for t in range(1, T):
            ctx, target = int(ids[t-1]), int(ids[t])
            key = (ctx, target)
            d = sum(v for k, v in cache.items() if k[0] == ctx) + alpha * V
            if d > 0:
                lp[t] = math.log(max((cache.get(key, 0) + alpha) / d, 1e-30))
            cache[key] = cache.get(key, 0) + 1
            for k in list(cache):
                cache[k] *= (1 - alpha)
                if cache[k] < 1e-6:
                    del cache[k]
    elif order == 3:
        cache = {}
        for t in range(2, T):
            ctx = (int(ids[t-2]), int(ids[t-1]))
            target = int(ids[t])
            key = ctx + (target,)
            d = sum(v for k, v in cache.items() if k[:2] == ctx) + alpha * V
            if d > 0:
                lp[t] = math.log(max((cache.get(key, 0) + alpha) / d, 1e-30))
            cache[key] = cache.get(key, 0) + 1
            for k in list(cache):
                cache[k] *= (1 - alpha)
                if cache[k] < 1e-6:
                    del cache[k]
    return lp


def build_shape_lp(ids: np.ndarray, tokenizer, trans: np.ndarray) -> np.ndarray:
    shape_map = {}
    def get_shape(tid):
        if tid not in shape_map:
            s = tokenizer.decode([int(tid)])
            if s.isupper(): shape_map[tid] = 0
            elif s and s[0].isupper(): shape_map[tid] = 1
            elif s.isdigit(): shape_map[tid] = 2
            elif s and all(c.isalpha() for c in s): shape_map[tid] = 3
            else: shape_map[tid] = 4
        return shape_map[tid]
    T = len(ids)
    shapes = np.array([get_shape(int(t)) for t in ids], dtype=np.int32)
    lp = np.full(T, -math.log(5), dtype=np.float32)
    for t in range(1, T):
        ps, cs = shapes[t-1], shapes[t]
        lp[t] = math.log(max(float(trans[ps, cs] / trans[ps].sum()), 1e-30))
    return lp


def build_recency_lp(ids: np.ndarray, window: int = 128) -> np.ndarray:
    T = len(ids)
    lp = np.full(T, -math.log(window), dtype=np.float32)
    positions = {}
    for t in range(T):
        token = int(ids[t])
        prev = positions.get(token, [])
        last = prev[-1] if prev else None
        gap = window if last is None else min(window, t - last)
        lp[t] = math.log(max(1.0 / gap, 1e-30))
        prev.append(t)
        if len(prev) > window:
            prev.pop(0)
        positions[token] = prev
    return lp


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--builder', type=str, default='artifacts/compiled_builder_50m.pt')
    p.add_argument('--data-dir', type=str, default='artifacts/wikitext_gpt2')
    p.add_argument('--out-dir', type=str, default='artifacts/gpt2_channels_v3')
    p.add_argument('--val-tokens', type=int, default=50000)
    p.add_argument('--eval-tokens', type=int, default=100000)
    args = p.parse_args()

    out_dir = Path(DEEPSEEK / args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(DEEPSEEK / args.data_dir)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained('gpt2')

    print('=' * 60)
    print(' GPT-2 BPE CHANNEL DUMP v3 (rich builder channels)')
    print('=' * 60)

    print(f'[load] Builder from {args.builder}...')
    builder = GPT2CompiledChannelBuilder.load(args.builder)
    print(f'  {builder.total_tokens:,} tokens, V={builder.cfg.vocab_size}')

    # Shape transitions from train set
    train_ids = torch.load(data_dir / 'train_ids.pt', weights_only=False).long().numpy()
    shape_map = {}
    def get_shape(tid):
        if tid not in shape_map:
            s = tok.decode([int(tid)])
            if s.isupper(): shape_map[tid] = 0
            elif s and s[0].isupper(): shape_map[tid] = 1
            elif s.isdigit(): shape_map[tid] = 2
            elif s and all(c.isalpha() for c in s): shape_map[tid] = 3
            else: shape_map[tid] = 4
        return shape_map[tid]
    
    n_sample = min(5_000_000, len(train_ids))
    sample = train_ids[:n_sample]
    shapes = np.array([get_shape(int(t)) for t in sample], dtype=np.int32)
    trans = np.ones((5, 5), dtype=np.float32)
    for t in range(1, len(shapes)):
        trans[shapes[t-1], shapes[t]] += 1
    print(f'  Shape transitions from {n_sample:,} tokens')

    val_ids = torch.load(data_dir / 'validation_ids.pt', weights_only=False).long().numpy()
    test_ids = torch.load(data_dir / 'test_ids.pt', weights_only=False).long().numpy()
    if len(val_ids) > args.val_tokens:
        val_ids = val_ids[:args.val_tokens]
    if len(test_ids) > args.eval_tokens:
        test_ids = test_ids[:args.eval_tokens]
    print(f'  Val: {len(val_ids):,}  Eval: {len(test_ids):,}')

    for split_name, split_ids in [('val', val_ids), ('eval', test_ids)]:
        print(f'\n[{split_name}] Computing {len(CHANNEL_NAMES)} channels...')
        t0 = time.time()
        T = len(split_ids)
        C = len(CHANNEL_NAMES)

        lps = np.zeros((T, C), dtype=np.float32)

        # Builder channels: uni, bi, tri, skip2, skip3
        print('  builder channels...', flush=True)
        for ci, ch in enumerate(["uni", "bi", "tri", "skip2", "skip3"]):
            context = []
            for t in range(T):
                tid = int(split_ids[t])
                lps[t, ci] = builder_logp(builder, ch, context, tid)
                context.append(tid)
                if len(context) > 3:
                    context.pop(0)

        # Decay unigram fast/slow
        print('  decay uni...', flush=True)
        lps[:, 5] = build_decay_lp(split_ids, 1, 0.001)
        lps[:, 6] = build_decay_lp(split_ids, 1, 0.0001)

        # Decay bigram fast/slow
        print('  decay bi...', flush=True)
        lps[:, 7] = build_decay_lp(split_ids, 2, 0.001)
        lps[:, 8] = build_decay_lp(split_ids, 2, 0.0001)

        # Decay trigram fast/slow
        print('  decay tri...', flush=True)
        lps[:, 9] = build_decay_lp(split_ids, 3, 0.001)
        lps[:, 10] = build_decay_lp(split_ids, 3, 0.0001)

        # Shape
        print('  shape...', flush=True)
        lps[:, 11] = build_shape_lp(split_ids, tok, trans)

        # Recency
        print('  recency...', flush=True)
        lps[:, 12] = build_recency_lp(split_ids)

        # Builder entropy (at position, use the builder features)
        print('  builder features...', flush=True)
        ids_t = torch.from_numpy(split_ids.astype(np.int64))
        features = builder.build_features(ids_t)
        if features.shape[1] > 10:
            # Use entropy norm as channel 13
            for t in range(min(T, features.shape[0])):
                lps[t, 13] = float(features[t, 10])  # entropy norm

        # Save
        print('  saving...', flush=True)
        np.savez_compressed(out_dir / f'{split_name}.npz',
                            log_p_targets=lps,
                            targets=split_ids[1:].astype(np.int64),
                            observed=split_ids[:-1].astype(np.int64),
                            channel_names=np.array(CHANNEL_NAMES))
        print(f'  Saved {out_dir / f"{split_name}.npz"} ({time.time()-t0:.0f}s)')

    print(f'\nDone. Channels at {out_dir}/')


if __name__ == '__main__':
    main()
