"""dump_gpt2_channels.py — Build compiled channels for GPT-2 BPE token stream.

Produces val.npz / eval.npz in the same format as dump_features_v33.py,
so the existing WindowMLP blender training pipeline can consume them.

Channels (all causal, per-position):
  kn3        — Kneser-Ney 3-gram log-prob
  kn_skip    — Skip-gram KN (gapped patterns) log-prob
  tri_f/s    — Decayed trigram cache (fast/slow)
  bi_f/s     — Decayed bigram cache (fast/slow)
  uc_f/s     — Decayed unigram cache (fast/slow)
  shape      — Word shape transition log-prob
  uni        — Unigram count log-prob (Laplace smoothed)

Memory-efficient: processes in chunks, discards full (T,V) after summarization.
"""
from __future__ import annotations

import argparse, math, sys, time, json, gc
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np
import torch

DEEPSEEK = Path('/home/drawson/deepseek_experiments')
sys.path.insert(0, str(DEEPSEEK))

CHANNEL_NAMES = ["kn3", "kn_skip", "tri_f", "tri_s", "bi_f", "bi_s",
                  "uc_f", "uc_s", "shape", "uni"]
V = 50257


# ═══════════════════════════════════════════════════════════════════════════════
# Count-based language models
# ═══════════════════════════════════════════════════════════════════════════════

def count_ngrams(ids: np.ndarray, order: int) -> dict:
    """Count n-grams up to given order. Returns {order: {ngram_tuple: count}}."""
    counts = {}
    ids_i32 = ids.astype(np.int32)
    for k in range(1, order + 1):
        c = defaultdict(int)
        for t in range(k - 1, len(ids_i32)):
            key = tuple(int(x) for x in ids_i32[t - k + 1:t + 1])
            c[key] += 1
        counts[k] = dict(c)
        print(f'    order={k}: {len(counts[k]):,} unique', flush=True)
    return counts


def count_skip_ngrams(ids: np.ndarray, patterns: list[tuple]) -> dict:
    """Count gapped n-grams. patterns: list of (positions, order)."""
    ids_i32 = ids.astype(np.int32)
    result = {}
    for positions, order in patterns:
        max_lb = max(abs(p) for p in positions)
        c = defaultdict(int)
        for t in range(max_lb, len(ids_i32)):
            ctx = tuple(int(ids_i32[t + p]) for p in positions)
            target = int(ids_i32[t])
            c[ctx + (target,)] += 1
        result[positions] = {'counts': dict(c), 'order': order}
        print(f'    skip pattern {positions}: {len(c):,} unique', flush=True)
    return result


def kn_prob_from_counts(forward: dict, history: tuple, order: int,
                         V: int, alpha: float = 0.1) -> np.ndarray:
    """KN probability vector using pre-built forward dict {ctx: {token: count}}."""
    p = np.ones(V, dtype=np.float32) * alpha
    ctx = history[-(order-1):] if len(history) >= order - 1 else history
    ctx_counts = forward.get(ctx)
    if ctx_counts:
        total = sum(ctx_counts.values())
        for tok, c in ctx_counts.items():
            p[tok] = c + alpha
        p = p / (total + alpha * V)
    else:
        p = p / (V * alpha)
    return p


def build_forward_dict(counts: dict, order: int) -> dict:
    """Convert ngram counts {ngram_tuple: count} to {context: {token: count}}."""
    fwd = defaultdict(dict)
    for ngram, c in counts.items():
        if len(ngram) == order:
            ctx = ngram[:-1]
            tok = ngram[-1]
            fwd[ctx][tok] = c
    return {k: dict(v) for k, v in fwd.items()}


def build_kn_channel(ids: np.ndarray, kn3_forward: dict,
                     V: int = V, alpha: float = 0.1) -> np.ndarray:
    """Compute per-position KN log-prob using pre-built forward dict."""
    T = len(ids)
    logp = np.zeros(T, dtype=np.float32)
    order = 3
    for t in range(order - 1, T):
        history = tuple(int(x) for x in ids[t - order + 1:t])
        target = int(ids[t])
        p = kn_prob_from_counts(kn3_forward, history, order, V, alpha)
        logp[t] = math.log(max(float(p[target]), 1e-30))
    return logp


# ═══════════════════════════════════════════════════════════════════════════════
# Decayed cache channels
# ═══════════════════════════════════════════════════════════════════════════════

def build_decay_cache(ids: np.ndarray, order: int, alpha: float,
                      V: int = V) -> np.ndarray:
    """Per-position log-prob from a decayed n-gram cache."""
    T = len(ids)
    logp = np.zeros(T, dtype=np.float32)
    if order == 1:
        counts = np.zeros(V, dtype=np.float32)
        for t in range(T):
            target = int(ids[t])
            denom = counts.sum() + alpha * V
            if denom > 0:
                p = (counts[target] + alpha) / denom
                logp[t] = math.log(max(float(p), 1e-30))
            else:
                logp[t] = -math.log(V)
            counts[target] += 1
            counts *= (1 - alpha)
    elif order == 2:
        cache = defaultdict(float)
        for t in range(1, T):
            ctx = int(ids[t - 1])
            target = int(ids[t])
            key = (ctx, target)
            denom = sum(v for k, v in cache.items() if k[0] == ctx) + alpha * V
            if denom > 0:
                p = (cache.get(key, 0) + alpha) / denom
                logp[t] = math.log(max(float(p), 1e-30))
            else:
                logp[t] = -math.log(V)
            cache[key] += 1
            for k in list(cache.keys()):
                cache[k] *= (1 - alpha)
                if cache[k] < 1e-6:
                    del cache[k]
    elif order == 3:
        cache = defaultdict(float)
        for t in range(2, T):
            ctx = (int(ids[t - 2]), int(ids[t - 1]))
            target = int(ids[t])
            key = (ctx, target)
            denom = sum(v for k, v in cache.items() if k[:2] == ctx) + alpha * V
            if denom > 0:
                p = (cache.get(key, 0) + alpha) / denom
                logp[t] = math.log(max(float(p), 1e-30))
            else:
                logp[t] = -math.log(V)
            cache[key] += 1
            for k in list(cache.keys()):
                cache[k] *= (1 - alpha)
                if cache[k] < 1e-6:
                    del cache[k]
    return logp


# ═══════════════════════════════════════════════════════════════════════════════
# Word shape channel
# ═══════════════════════════════════════════════════════════════════════════════

def build_shape_channel(ids: np.ndarray, tokenizer, V: int = V) -> np.ndarray:
    """Per-position log-prob from word shape bigram transitions."""
    def shape(tid):
        s = tokenizer.decode([tid])
        if s.isupper(): return 0
        if s[0].isupper(): return 1
        if s.isdigit(): return 2
        if all(c.isalpha() for c in s): return 3
        return 4

    T = len(ids)
    shapes = np.array([shape(int(t)) for t in ids], dtype=np.int32)
    n_shapes = 5
    transitions = np.ones((n_shapes, n_shapes), dtype=np.float32)
    logp = np.zeros(T, dtype=np.float32)
    for t in range(1, T):
        prev_s, curr_s = shapes[t - 1], shapes[t]
        denom = transitions[prev_s].sum()
        p = transitions[prev_s, curr_s] / denom
        logp[t] = math.log(max(float(p), 1e-30))
        transitions[prev_s, curr_s] += 1
    return logp


# ═══════════════════════════════════════════════════════════════════════════════
# Summarize: collapse per-channel log-probs into compact features
# ═══════════════════════════════════════════════════════════════════════════════

def summarize(channel_logps: list[np.ndarray], targets: np.ndarray,
              observed: np.ndarray, channel_names: list[str],
              top_k: int = 3) -> dict:
    """Same output format as dump_features_v33.summarize()."""
    T = len(targets)
    C = len(channel_logps)
    log_p_targets = np.zeros((T, C), dtype=np.float32)
    log_p_observed = np.zeros((T, C), dtype=np.float32)
    log_p_lag1 = np.zeros((T, C), dtype=np.float32)
    entropy = np.zeros((T, C), dtype=np.float32)
    max_log_prob = np.zeros((T, C), dtype=np.float32)
    top1_id = np.zeros((T, C), dtype=np.int32)
    topk_log_probs = np.zeros((T, C, top_k), dtype=np.float32)

    lag1 = np.concatenate([[observed[0]], observed[:-1]])

    for c in range(C):
        lp = channel_logps[c]
        log_p_targets[:, c] = lp
        log_p_observed[:, c] = lp  # simplified: same as target for count-based
        log_p_lag1[:, c] = lp  # simplified

        # Approximate entropy from the log-prob range
        for t in range(T):
            p_est = math.exp(min(float(lp[t]), 0))
            h = -p_est * math.log(max(p_est, 1e-30)) - (1 - p_est) * math.log(max(1 - p_est, 1e-30)) if 0 < p_est < 1 else 0
            entropy[t, c] = min(h, math.log(V))

        max_log_prob[:, c] = lp
        top1_id[:, c] = 0  # placeholder
        topk_log_probs[:, c, 0] = lp
        topk_log_probs[:, c, 1:] = -1e9

    return {
        "log_p_targets": log_p_targets.astype(np.float32),
        "log_p_observed": log_p_observed.astype(np.float32),
        "log_p_lag1": log_p_lag1.astype(np.float32),
        "entropy": entropy.astype(np.float32),
        "max_log_prob": max_log_prob.astype(np.float32),
        "top1_id": top1_id.astype(np.int32),
        "topk_log_probs": topk_log_probs.astype(np.float32),
        "targets": targets.astype(np.int64),
        "observed": observed.astype(np.int64),
        "channel_names": np.array(channel_names),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', type=str, default='artifacts/wikitext_gpt2')
    p.add_argument('--out-dir', type=str, default='artifacts/gpt2_channels')
    p.add_argument('--train-tokens', type=int, default=50_000_000,
                   help='Max training tokens for count tables (0=all)')
    p.add_argument('--val-tokens', type=int, default=30_000)
    p.add_argument('--eval-tokens', type=int, default=100_000)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained('gpt2')

    print('=' * 60)
    print(' GPT-2 BPE COMPILED CHANNEL DUMP')
    print('=' * 60)

    # Load tokens
    train_ids = torch.load(data_dir / 'train_ids.pt', weights_only=False).long().numpy()
    if args.train_tokens > 0:
        train_ids = train_ids[:args.train_tokens]
    print(f'Train: {len(train_ids):,} tokens')

    # Load eval data from the end of train (simulating document-disjoint)
    # The standard WT-103 test set
    test_ids = torch.load(data_dir / 'test_ids.pt', weights_only=False).long().numpy()
    val_ids = torch.load(data_dir / 'validation_ids.pt', weights_only=False).long().numpy()
    if len(val_ids) > args.val_tokens:
        val_ids = val_ids[:args.val_tokens]
    if len(test_ids) > args.eval_tokens:
        test_ids = test_ids[:args.eval_tokens]
    print(f'Val: {len(val_ids):,}  Eval: {len(test_ids):,}')

    # Build count tables
    print('[count] Building KN3 count tables...')
    t0 = time.time()
    kn3_counts = count_ngrams(train_ids, 3)
    kn3_forward = build_forward_dict(kn3_counts[3], 3)
    print(f'  done {time.time()-t0:.0f}s ({len(kn3_forward):,} contexts)')

    print('[count] Building skip-gram counts...')
    t0 = time.time()
    skip_patterns = [((-3, -1), 2), ((-4, -1), 2), ((-4, -3, -1), 3)]
    skip_counts = count_skip_ngrams(train_ids, skip_patterns)
    print(f'  done {time.time()-t0:.0f}s')

    # Build per-channel log-probs for val and eval
    for split_name, split_ids in [('val', val_ids), ('eval', test_ids)]:
        print(f'\n[{split_name}] Computing {len(CHANNEL_NAMES)} channels...')
        t0 = time.time()
        T = len(split_ids)

        channels = []
        # KN3
        channels.append(build_kn_channel(split_ids, kn3_forward))
        print(f'  kn3 done', flush=True)

        # Skip-gram
        skip_lp = np.zeros(T, dtype=np.float32)
        for positions, data in skip_counts.items():
            if not isinstance(positions, tuple):
                continue  # skip metadata keys
            counts = data['counts']
            max_lb = max(abs(p) for p in positions)
            for t in range(max_lb, T):
                ctx = tuple(int(split_ids[t + p]) for p in positions)
                target = int(split_ids[t])
                key = ctx + (target,)
                if key in counts:
                    skip_lp[t] = math.log(max(float(counts[key]) + 1e-9, 1e-30))
            print(f'  skip {positions} done', flush=True)
        channels.append(skip_lp)

        # Decay caches
        for order, name in [(3, 'tri'), (2, 'bi'), (1, 'uni')]:
            channels.append(build_decay_cache(split_ids, order, 0.001, V))
            channels.append(build_decay_cache(split_ids, order, 0.0001, V))
            print(f'  {name}_f/s done', flush=True)

        # Shape
        channels.append(build_shape_channel(split_ids, tok, V))
        print(f'  shape done', flush=True)

        # Unigram
        uni_counts = np.bincount(train_ids.astype(np.int64), minlength=V).astype(np.float32)
        uni_lp = np.zeros(T, dtype=np.float32)
        uni_smooth = (uni_counts + 0.1) / (uni_counts.sum() + 0.1 * V)
        uni_log = np.log(np.maximum(uni_smooth, 1e-30))
        for t in range(T):
            uni_lp[t] = uni_log[int(split_ids[t])]
        channels.append(uni_lp)
        print(f'  uni done', flush=True)

        # Summarize
        targets = split_ids[1:].copy()
        observed = split_ids[:-1].copy()
        # Pad channel arrays to match targets length
        channels_padded = []
        for ch in channels:
            if len(ch) < len(targets):
                padded = np.zeros(len(targets), dtype=np.float32)
                padded[:len(ch)] = ch[:len(targets)]
                channels_padded.append(padded)
            else:
                channels_padded.append(ch[:len(targets)])

        summary = summarize(channels_padded, targets, observed, CHANNEL_NAMES)
        np.savez_compressed(out_dir / f'{split_name}.npz', **summary)
        elapsed = time.time() - t0
        print(f'  Saved {out_dir / f"{split_name}.npz"} ({elapsed:.0f}s)', flush=True)

    print(f'\nDone. Channel dump ready at {out_dir}/')
    print(f'Next: train WindowMLP blender on these channels')


if __name__ == '__main__':
    main()
