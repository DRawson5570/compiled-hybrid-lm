"""dump_gpt2_channels_v2.py — Efficient compiled channels for GPT-2 BPE.

Uses only O(T) or pre-built channels. No O(V^2) count tables.

Channels:
  compiled_lp — GPT2CompiledChannelBuilder (pre-built, 420MB, serialized)
  tri_f/s     — Decayed trigram cache (fast/slow)
  bi_f/s      — Decayed bigram cache (fast/slow)  
  uc_f/s      — Decayed unigram cache (fast/slow)
  shape       — Word shape transition
  uni_lp      — Unigram log-prob (Laplace)
  recency     — Local token recency

All streaming or O(1) memory. Produces val.npz / eval.npz for blender training.
"""
from __future__ import annotations

import argparse, math, sys, time, json
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch

DEEPSEEK = Path('/home/drawson/deepseek_experiments')
sys.path.insert(0, str(DEEPSEEK))
from hybrid.compiled_features import GPT2CompiledChannelBuilder

CHANNEL_NAMES = ["compiled", "tri_f", "tri_s", "bi_f", "bi_s",
                  "uc_f", "uc_s", "shape", "uni", "recency"]
V = 50257


def build_decay_cache_observed(ids: np.ndarray, order: int, alpha: float) -> np.ndarray:
    """Per-position log-prob of the OBSERVED token from cache at previous step."""
    T = len(ids)
    lp = np.full(T, -math.log(V), dtype=np.float32)
    if order == 1:
        counts = np.zeros(V, dtype=np.float32)
        for t in range(1, T):
            target = int(ids[t])
            d = counts.sum() + alpha * V
            if d > 0:
                lp[t] = math.log(max((counts[target] + alpha) / d, 1e-30))
            prev = int(ids[t-1])
            counts[prev] += 1
            counts *= (1 - alpha)
    elif order == 2:
        cache = {}
        for t in range(2, T):
            ctx, target = int(ids[t-2]), int(ids[t])
            key = (ctx, target)
            d = sum(v for k, v in cache.items() if k[0] == ctx) + alpha * V
            if d > 0:
                lp[t] = math.log(max((cache.get(key, 0) + alpha) / d, 1e-30))
            prev_ctx, prev_tok = int(ids[t-2]), int(ids[t-1])
            pkey = (prev_ctx, prev_tok)
            cache[pkey] = cache.get(pkey, 0) + 1
            for k in list(cache):
                cache[k] *= (1 - alpha)
                if cache[k] < 1e-6:
                    del cache[k]
    elif order == 3:
        cache = {}
        for t in range(3, T):
            ctx, target = (int(ids[t-3]), int(ids[t-2])), int(ids[t])
            key = ctx + (target,)
            d = sum(v for k, v in cache.items() if k[:2] == ctx) + alpha * V
            if d > 0:
                lp[t] = math.log(max((cache.get(key, 0) + alpha) / d, 1e-30))
            prev = (int(ids[t-3]), int(ids[t-2]), int(ids[t-1]))
            cache[prev] = cache.get(prev, 0) + 1
            for k in list(cache):
                cache[k] *= (1 - alpha)
                if cache[k] < 1e-6:
                    del cache[k]
    return lp


def build_decay_cache(ids: np.ndarray, order: int, alpha: float) -> np.ndarray:
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
            ctx, target = (int(ids[t-2]), int(ids[t-1])), int(ids[t])
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


def build_shape_channel(ids: np.ndarray, tokenizer, trans_matrix: np.ndarray | None = None,
                        shape_cache: dict | None = None) -> np.ndarray:
    if shape_cache is None:
        shape_cache = {}
    def get_shape(tid):
        if tid not in shape_cache:
            s = tokenizer.decode([int(tid)])
            if s.isupper(): shape_cache[tid] = 0
            elif s and s[0].isupper(): shape_cache[tid] = 1
            elif s.isdigit(): shape_cache[tid] = 2
            elif all(c.isalpha() for c in s): shape_cache[tid] = 3
            else: shape_cache[tid] = 4
        return shape_cache[tid]
    T = len(ids)
    shapes = np.array([get_shape(int(t)) for t in ids], dtype=np.int32)
    if trans_matrix is not None:
        trans = trans_matrix.copy()
    else:
        trans = np.ones((5, 5), dtype=np.float32)
    lp = np.full(T, -math.log(5), dtype=np.float32)
    for t in range(1, T):
        ps, cs = shapes[t-1], shapes[t]
        d = trans[ps].sum()
        lp[t] = math.log(max(float(trans[ps, cs] / d), 1e-30))
    return lp


def compute_shape_transitions(train_ids: np.ndarray, tokenizer, n_samples: int = 10_000_000) -> np.ndarray:
    """Pre-compute shape transition counts from a sample of training data."""
    import time
    # Build shape cache: only decode each unique token once
    shape_cache = {}
    def get_shape(tid):
        if tid not in shape_cache:
            s = tokenizer.decode([int(tid)])
            if s.isupper(): shape_cache[tid] = 0
            elif s and s[0].isupper(): shape_cache[tid] = 1
            elif s.isdigit(): shape_cache[tid] = 2
            elif all(c.isalpha() for c in s): shape_cache[tid] = 3
            else: shape_cache[tid] = 4
        return shape_cache[tid]

    trans = np.ones((5, 5), dtype=np.float32)
    ids_sample = train_ids[:n_samples]
    shapes = np.array([get_shape(int(t)) for t in ids_sample], dtype=np.int32)
    for t in range(1, len(shapes)):
        trans[shapes[t-1], shapes[t]] += 1
    print(f'  {len(shape_cache):,} unique tokens decoded')
    return trans


def build_compiled_channel(split_ids: np.ndarray, builder: GPT2CompiledChannelBuilder) -> np.ndarray:
    """Per-position log-prob from the pre-built compiled builder."""
    T = len(split_ids)
    ids_t = torch.from_numpy(split_ids.astype(np.int64))
    # Build all features at once with full history
    features = builder.build_features(ids_t)
    lp = np.full(T, -math.log(V), dtype=np.float32)
    for t in range(T):
        lp[t] = float(features[t, 0])  # uni_logp_token
    return lp


def build_recency_channel(split_ids: np.ndarray, window: int = 128) -> np.ndarray:
    T = len(split_ids)
    lp = np.full(T, -math.log(V), dtype=np.float32)
    positions = {}
    for t in range(T):
        token = int(split_ids[t])
        prev = positions.get(token, [])
        last = prev[-1] if prev else None
        gap = window if last is None else min(window, t - last)
        lp[t] = math.log(max(1.0 / gap, 1e-30))
        prev.append(t)
        if len(prev) > window:
            prev.pop(0)
        positions[token] = prev
    return lp


def summarize(channel_lps_target: list[np.ndarray],
              channel_lps_observed: list[np.ndarray],
              targets: np.ndarray,
              observed: np.ndarray, channel_names: list[str]) -> dict:
    T = len(targets)
    C = len(channel_lps_target)
    log_p_targets = np.zeros((T, C), dtype=np.float32)
    log_p_observed = np.zeros((T, C), dtype=np.float32)
    log_p_lag1 = np.zeros((T, C), dtype=np.float32)
    entropy = np.zeros((T, C), dtype=np.float32)
    max_log_prob = np.zeros((T, C), dtype=np.float32)
    top1_id = np.zeros((T, C), dtype=np.int32)
    topk_log_probs = np.zeros((T, C, 3), dtype=np.float32)

    lag1 = np.concatenate([[observed[0]], observed[:-1]])
    for c in range(C):
        lp_t = channel_lps_target[c]
        lp_o = channel_lps_observed[c]
        log_p_targets[:, c] = lp_t
        log_p_observed[:, c] = lp_o
        log_p_lag1[:, c] = lp_o  # lag1 = observed at previous position
        pe = np.exp(np.clip(lp_t, -50, 0))
        entropy[:, c] = -pe * np.log(np.clip(pe, 1e-30, 1)) - (1-pe) * np.log(np.clip(1-pe, 1e-30, 1))
        max_log_prob[:, c] = lp_t
        topk_log_probs[:, c, 0] = lp_t
        topk_log_probs[:, c, 1:] = -1e9

    return {
        "log_p_targets": log_p_targets.astype(np.float32),
        "log_p_observed": log_p_observed.astype(np.float32),
        "log_p_lag1": log_p_lag1.astype(np.float32),
        "entropy": entropy.astype(np.float32),
        "max_log_prob": max_log_prob.astype(np.float32),
        "top1_id": top1_id,
        "topk_log_probs": topk_log_probs,
        "targets": targets.astype(np.int64),
        "observed": observed.astype(np.int64),
        "channel_names": np.array(channel_names),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', type=str, default='artifacts/wikitext_gpt2')
    p.add_argument('--compiled-builder', type=str, default='artifacts/compiled_builder_50m.pt')
    p.add_argument('--out-dir', type=str, default='artifacts/gpt2_channels_v2')
    p.add_argument('--val-tokens', type=int, default=30000)
    p.add_argument('--eval-tokens', type=int, default=100000)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained('gpt2')

    print('=' * 60)
    print(' GPT-2 BPE CHANNEL DUMP v2 (efficient)')
    print('=' * 60)

    # Load pre-built compiled builder
    print(f'[load] Compiled builder from {args.compiled_builder}...')
    builder = GPT2CompiledChannelBuilder.load(args.compiled_builder)
    print(f'  Loaded: {len(builder.unigram):,} unigrams')

    # Build unigram and shape transitions from train set
    train_ids = torch.load(data_dir / 'train_ids.pt', weights_only=False).long().numpy()
    uni_counts = np.bincount(train_ids.astype(np.int64), minlength=V).astype(np.float32)
    uni_lp_global = np.log(np.maximum((uni_counts + 0.1) / (uni_counts.sum() + 0.1 * V), 1e-30))

    print('[shape] Computing shape transitions from train set...')
    shape_trans = compute_shape_transitions(train_ids, tok)

    # Load val/eval data
    val_ids = torch.load(data_dir / 'validation_ids.pt', weights_only=False).long().numpy()
    test_ids = torch.load(data_dir / 'test_ids.pt', weights_only=False).long().numpy()
    if len(val_ids) > args.val_tokens:
        val_ids = val_ids[:args.val_tokens]
    if len(test_ids) > args.eval_tokens:
        test_ids = test_ids[:args.eval_tokens]
    print(f'Val: {len(val_ids):,}  Eval: {len(test_ids):,}')

    for split_name, split_ids in [('val', val_ids), ('eval', test_ids)]:
        print(f'\n[{split_name}] Computing {len(CHANNEL_NAMES)} channels...')
        t0 = time.time()
        T = len(split_ids)

        channels_target = []
        channels_observed = []

        # compiled — target log-prob for next token, observed for current
        print('  compiled...', flush=True)
        ids_t = torch.from_numpy(split_ids.astype(np.int64))
        features = builder.build_features(ids_t)
        compiled_lp_target = np.full(T, -math.log(V), dtype=np.float32)
        compiled_lp_observed = np.full(T, -math.log(V), dtype=np.float32)
        for t in range(T):
            compiled_lp_target[t] = float(features[t, 0])
        for t in range(1, T):
            compiled_lp_observed[t] = float(features[t-1, 0])  # prob of current token from previous context
        channels_target.append(compiled_lp_target)
        channels_observed.append(compiled_lp_observed)

        # Decay caches
        for order, name in [(3, 'tri'), (2, 'bi'), (1, 'uni')]:
            lp_t = build_decay_cache(split_ids, order, 0.001)
            lp_o = build_decay_cache_observed(split_ids, order, 0.001)
            channels_target.append(lp_t)
            channels_observed.append(lp_o)
            lp_t2 = build_decay_cache(split_ids, order, 0.0001)
            lp_o2 = build_decay_cache_observed(split_ids, order, 0.0001)
            channels_target.append(lp_t2)
            channels_observed.append(lp_o2)
            print(f'  {name}_f/s', flush=True)

        # Shape
        print('  shape...', flush=True)
        lp_s = build_shape_channel(split_ids, tok, shape_trans)
        channels_target.append(lp_s)
        channels_observed.append(lp_s)

        # Unigram
        print('  uni...', flush=True)
        uni_lp = np.array([uni_lp_global[int(t)] for t in split_ids], dtype=np.float32)
        channels_target.append(uni_lp)
        channels_observed.append(uni_lp)  # unigram is context-independent

        # Recency
        print('  recency...', flush=True)
        lp_r = build_recency_channel(split_ids)
        channels_target.append(lp_r)
        channels_observed.append(lp_r)  # recency is position-dependent but same for target/observed

        # Summarize
        targets = split_ids[1:]
        observed_arr = split_ids[:-1]
        channels_target_padded = [np.concatenate([ch[:len(targets)], np.zeros(max(0, len(targets)-len(ch)), dtype=np.float32)])
                                   for ch in channels_target]
        channels_observed_padded = [np.concatenate([ch[:len(targets)], np.zeros(max(0, len(targets)-len(ch)), dtype=np.float32)])
                                     for ch in channels_observed]

        summary = summarize(channels_target_padded, channels_observed_padded,
                            targets, observed_arr, CHANNEL_NAMES)
        np.savez_compressed(out_dir / f'{split_name}.npz', **summary)
        print(f'  Saved {out_dir / f"{split_name}.npz"} ({time.time()-t0:.0f}s)', flush=True)

    print(f'\nDone. Channels at {out_dir}/')


if __name__ == '__main__':
    main()
