"""
compile_wiki_lm_v25.py — 4-way blend: KN5 ⊕ v14-mixture ⊕ trigram-induction ⊕ unigram-induction
==================================================================================================

Builds on:
  v23: Modified KN5 5-gram, PPL=88.89 heldout (global n-gram statistics)
  v14: Sparse-mixture cluster LM, K_cl=65536 (positional / semantic residual)
  v21: Trigram + unigram induction with sliding window (in-context recency)

KN5 is GLOBAL — it does not see the most recent in-context occurrences.
v21's trigram/unigram induction captures very-recent (windowed) statistics
that KN5 cannot. They should be complementary.

We precompute KN5 and v14-mixture log-probs as (N-K_pos, V) arrays,
stream through eval_ids to build sliding trigram + unigram induction
log-probs, then sweep a 4-way weight simplex grid and evaluate per-token.

Blend (linear, in prob space, then renormalize):
    p = w_kn * p_kn + w_mix * p_mix + w_tri * p_tri + w_uni * p_uni
"""
from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO = Path("/home/drawson/llm_decoupling")
sys.path.insert(0, str(REPO))

from compile_wiki_lm_v13 import load_setup, load_or_build_tokens, build_residual, parse_size, DEVICE
from compile_wiki_lm_v14 import SparseMixtureClusterLM
from compile_wiki_lm_v23 import ModifiedKNGram
from compile_wiki_lm_v24 import compute_log_p_mix, compute_log_p_kn

ARTIFACT = Path("/home/drawson/deepseek_experiments/artifacts/compiled_wiki_lm_v26")
ARTIFACT.mkdir(parents=True, exist_ok=True)


@torch.no_grad()
def build_induction_log_probs(ids_np: np.ndarray, V: int, K_pos: int,
                               window: int, alpha_tri: float, alpha_uni: float) -> tuple[np.ndarray, np.ndarray]:
    """Stream ids; build sliding-window trigram & unigram induction.

    Returns (log_p_tri, log_p_uni) of shape (N - K_pos, V), aligned so
    row i predicts ids[K_pos + i] given history through ids[K_pos + i - 1].

    Both are dense float32 numpy arrays.
    """
    N = len(ids_np)
    T = N - K_pos
    # On-GPU running tables
    Bu = torch.zeros(V, V, device=DEVICE, dtype=torch.float32)
    Ru = torch.zeros(V, device=DEVICE, dtype=torch.float32)
    Bt: dict[tuple, torch.Tensor] = {}
    Rt: dict[tuple, float] = {}
    history = deque()

    log_p_tri = np.zeros((T, V), dtype=np.float32)
    log_p_uni = np.zeros((T, V), dtype=np.float32)

    aV_u = alpha_uni * V
    aV_t = alpha_tri * V
    log_uniform = math.log(1.0 / V)

    cursor = max(K_pos, 1)
    t0 = time.time()
    # produce rows for t = K_pos .. N-1 (T rows). Need ids[t-1] and ids[t-2] for trigram.
    # We don't predict the LAST token (no target), so output is T rows but eval only uses T-1?
    # Actually we predict ids[t] from history; output row index = t - K_pos.
    # For trigram induction key (a=ids[t-2], b=ids[t-1]).
    for t in range(K_pos, N):
        i = t - K_pos
        # b = ids[t-1] (most recent context token); for trigram key (a, b) need a=ids[t-2]
        b = int(ids_np[t - 1])
        # unigram prob row
        p_uni_row = (Bu[b] + alpha_uni) / (Ru[b] + aV_u)
        lp_uni = torch.log(p_uni_row.clamp_min(1e-30)).cpu().numpy()
        log_p_uni[i] = lp_uni
        # trigram
        if t >= 2:
            a = int(ids_np[t - 2])
            key = (a, b)
            row_t = Bt.get(key)
            if row_t is not None:
                p_tri_row = (row_t + alpha_tri) / (Rt[key] + aV_t)
                log_p_tri[i] = torch.log(p_tri_row.clamp_min(1e-30)).cpu().numpy()
            else:
                log_p_tri[i].fill(log_uniform)
        else:
            log_p_tri[i].fill(log_uniform)

        # update with the OBSERVED transition (b=ids[t-1] -> c=ids[t])
        c = int(ids_np[t])
        Bu[b, c] += 1.0
        Ru[b] += 1.0
        
        a_val = None
        key_val = None
        if t >= 2:
            a_val = int(ids_np[t - 2])
            key_val = (a_val, b)
            if key_val in Bt:
                Bt[key_val][c] += 1.0
                Rt[key_val] += 1.0
            else:
                v = torch.zeros(V, device=DEVICE, dtype=torch.float32)
                v[c] = 1.0
                Bt[key_val] = v
                Rt[key_val] = 1.0
        
        history.append((a_val, b, c, key_val))

        # Sliding window decrement
        if len(history) > window:
            old_a, old_b, old_c, old_key = history.popleft()
            Bu[old_b, old_c] -= 1.0
            Ru[old_b] -= 1.0
            if old_key is not None:
                row_t = Bt.get(old_key)
                if row_t is not None:
                    row_t[old_c] -= 1.0
                    Rt[old_key] -= 1.0
                    if Rt[old_key] <= 0:
                        del Bt[old_key]
                        del Rt[old_key]

        if (i + 1) % 10000 == 0:
            print(f"    [ind] {i+1}/{T} ({time.time() - t0:.1f}s, tri-keys={len(Bt)})")

    return log_p_tri, log_p_uni


def eval_blend4(log_p_kn, log_p_mix, log_p_tri, log_p_uni, targets,
                w_kn, w_mix, w_tri, w_uni, compute_topk=False):
    """Linear prob-space blend: P = sum_k w_k * P_k (no global renorm — components are already normalized).

    NLL is computed using ONLY the target column gather (fast):
        log P[target] = log( sum_k w_k * exp(log_p_k[t, target[t]]) )

    Top1/top5 are computed only when compute_topk=True (requires full V).
    """
    s = w_kn + w_mix + w_tri + w_uni
    if s <= 0:
        return {"ppl": float("inf"), "top1": 0.0, "top5": 0.0, "n": int(len(targets))}
    if abs(s - 1.0) > 1e-6:
        w_kn, w_mix, w_tri, w_uni = w_kn/s, w_mix/s, w_tri/s, w_uni/s

    n = len(targets)
    idx = np.arange(n)
    # gather log p at target for each component (T,)
    lpk_t = log_p_kn[idx, targets]
    lpm_t = log_p_mix[idx, targets]
    lpt_t = log_p_tri[idx, targets]
    lpu_t = log_p_uni[idx, targets]
    parts = []
    for w, lp_t in [(w_kn, lpk_t), (w_mix, lpm_t), (w_tri, lpt_t), (w_uni, lpu_t)]:
        if w <= 0:
            continue
        parts.append(math.log(w) + lp_t)
    stack = np.stack(parts, axis=0)  # (k, T)
    m = stack.max(axis=0)
    log_p_target = m + np.log(np.exp(stack - m[None]).sum(axis=0))
    nll = -log_p_target.sum()
    ppl = math.exp(nll / n)

    if not compute_topk:
        return {"ppl": ppl, "top1": None, "top5": None, "n": int(n)}

    # full top-k pass — only for final eval
    c1 = 0
    c5 = 0
    BATCH = 4096
    parts_full_cfgs = [(w, lp) for w, lp in
                       [(w_kn, log_p_kn), (w_mix, log_p_mix),
                        (w_tri, log_p_tri), (w_uni, log_p_uni)] if w > 0]
    for st in range(0, n, BATCH):
        e = min(st + BATCH, n)
        bparts = [math.log(w) + lp[st:e] for w, lp in parts_full_cfgs]
        sk = np.stack(bparts, axis=0)
        mm = sk.max(axis=0)
        lp_batch = mm + np.log(np.exp(sk - mm[None]).sum(axis=0))
        am = np.argmax(lp_batch, axis=1)
        tgt = targets[st:e]
        c1 += (am == tgt).sum()
        top5 = np.argpartition(-lp_batch, 5, axis=1)[:, :5]
        c5 += (top5 == tgt[:, None]).any(axis=1).sum()
    return {"ppl": ppl, "top1": float(c1) / n, "top5": float(c5) / n, "n": int(n)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--kn-pickle", type=str, required=True)
    p.add_argument("--counts-file", type=str, required=True)
    p.add_argument("--K-pos", type=int, default=2)
    p.add_argument("--top-M", type=int, default=16)
    p.add_argument("--tau", type=float, default=0.05)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--alpha-mix", type=float, default=0.01)
    p.add_argument("--window", type=int, default=8192)
    p.add_argument("--alpha-tri", type=float, default=1e-5)
    p.add_argument("--alpha-uni", type=float, default=1e-5)
    p.add_argument("--train-tokens", type=str, default="22M")
    p.add_argument("--val-tokens", type=str, default="30K")
    p.add_argument("--eval-tokens", type=str, default="100K")
    p.add_argument("--tag", type=str, default="default")
    args = p.parse_args()

    train_n = parse_size(args.train_tokens)
    val_n = parse_size(args.val_tokens)
    eval_n = parse_size(args.eval_tokens)

    bpe, vocab, tok2id, bpe_to_lm, emb, V, d = load_setup()
    emb_dev = emb.to(DEVICE)
    ids = load_or_build_tokens(bpe, bpe_to_lm, V)
    ids_np = ids.numpy().astype(np.int32)
    T = len(ids)
    if train_n + val_n + eval_n > T:
        train_n = max(T - val_n - eval_n, T // 2)
    val_ids_t = ids[train_n:train_n + val_n]
    val_ids_n = ids_np[train_n:train_n + val_n]
    eval_ids_t = ids[-eval_n:]
    eval_ids_n = ids_np[-eval_n:]
    print(f"[v25] K_pos={args.K_pos}  V={V}  window={args.window}")
    print(f"[split] train={train_n:,}  val={val_n:,}  eval={eval_n:,}")

    print(f"[load] KN  {args.kn_pickle}")
    with open(args.kn_pickle, "rb") as f:
        kn = pickle.load(f)
    print(f"[load] counts  {args.counts_file}")
    blob = torch.load(args.counts_file, map_location=DEVICE, weights_only=False)
    mu = blob["mu"].to(DEVICE)
    counts = blob["counts"].to(DEVICE)
    assert blob["K_pos"] == args.K_pos and blob["V"] == V
    model = SparseMixtureClusterLM.from_counts(mu, counts, alpha=args.alpha_mix,
                                                V=V, K_pos=args.K_pos, d_emb=d)
    print(f"[mix] K_cl={mu.shape[0]}  tau={args.tau} gamma={args.gamma}")

    def prepare(ids_t, ids_n, label, w_size):
        print(f"\n[{label}] computing 4 component log-prob tables with window={w_size}")
        t0 = time.time()
        log_p_mix = compute_log_p_mix(ids_t, emb_dev, model, args.K_pos,
                                       args.top_M, args.tau, args.gamma).numpy()
        print(f"  mix done ({time.time() - t0:.1f}s, shape={log_p_mix.shape})")
        t0 = time.time()
        log_p_kn = compute_log_p_kn(kn, ids_n, args.K_pos)
        print(f"  KN done ({time.time() - t0:.1f}s)")
        t0 = time.time()
        log_p_tri, log_p_uni = build_induction_log_probs(
            ids_n, V, args.K_pos, w_size, args.alpha_tri, args.alpha_uni)
        print(f"  tri/uni done ({time.time() - t0:.1f}s)")
        targets = ids_n[args.K_pos:]
        return log_p_kn, log_p_mix, log_p_tri, log_p_uni, targets

    # VAL - sweep window size on val
    print(f"\n[val] computing 2 global/positional components (mix, KN) for validation")
    t0 = time.time()
    log_p_mix_v = compute_log_p_mix(val_ids_t, emb_dev, model, args.K_pos,
                                   args.top_M, args.tau, args.gamma).numpy()
    print(f"  mix done ({time.time() - t0:.1f}s)")
    t0 = time.time()
    log_p_kn_v = compute_log_p_kn(kn, val_ids_n, args.K_pos)
    print(f"  KN done ({time.time() - t0:.1f}s)")
    targets_v = val_ids_n[args.K_pos:]

    windows_to_sweep = [2048, 4096, 8192, 16384, 32768, 65536]
    print(f"\n[sweep] Sweeping windows over {windows_to_sweep}")
    best_val_ppl_overall = float("inf")
    best_w_size = args.window
    best_val_w_weights = None
    sorted_v_best = None

    for w_size in windows_to_sweep:
        t0 = time.time()
        log_p_tri_v, log_p_uni_v = build_induction_log_probs(
            val_ids_n, V, args.K_pos, w_size, args.alpha_tri, args.alpha_uni)
        print(f"  tri/uni with window={w_size} done ({time.time() - t0:.1f}s)")

        grid = []
        # baselines
        grid.append((1.0, 0.0, 0.0, 0.0))
        grid.append((0.92, 0.08, 0.0, 0.0))
        # add small tri and uni mass
        for w_tri in [0.0, 0.02, 0.05, 0.08, 0.12, 0.18, 0.25]:
            for w_uni in [0.0, 0.02, 0.05, 0.08, 0.12]:
                for w_mix_frac in [0.02, 0.05, 0.08, 0.12]:
                    w_other = w_tri + w_uni + w_mix_frac
                    if w_other >= 1.0:
                        continue
                    w_kn = 1.0 - w_other
                    grid.append((w_kn, w_mix_frac, w_tri, w_uni))
        seen = set()
        val_results = {}
        best_ppl_for_w = float("inf")
        for w in grid:
            key = tuple(round(x, 4) for x in w)
            if key in seen:
                continue
            seen.add(key)
            r = eval_blend4(log_p_kn_v, log_p_mix_v, log_p_tri_v, log_p_uni_v, targets_v, *w)
            val_results[str(key)] = {**r, "w": w}
            if r["ppl"] < best_ppl_for_w:
                best_ppl_for_w = r["ppl"]

        sorted_v = sorted(val_results.items(), key=lambda kv: kv[1]["ppl"])
        print(f"    window={w_size:5d}: Best Val PPL={best_ppl_for_w:.4f} with w={sorted_v[0][1]['w']}")
        if best_ppl_for_w < best_val_ppl_overall:
            best_val_ppl_overall = best_ppl_for_w
            best_w_size = w_size
            best_val_w_weights = sorted_v[0][1]["w"]
            sorted_v_best = sorted_v

    best_w = best_val_w_weights
    print(f"\n[sweep] -> Best Window={best_w_size} Best Val PPL={best_val_ppl_overall:.2f} with w={best_w}")

    # free val memory
    if "log_p_kn_v" in locals():
        del log_p_kn_v
    if "log_p_mix_v" in locals():
        del log_p_mix_v
    if "targets_v" in locals():
        del targets_v
    import gc
    gc.collect()

    # HELDOUT using the discovered best window
    log_p_kn_e, log_p_mix_e, log_p_tri_e, log_p_uni_e, targets_e = prepare(eval_ids_t, eval_ids_n, "eval", best_w_size)

    print(f"\n[eval] HELDOUT — evaluating top-5 val configs + KN-only baseline")
    eval_results = {}
    for label, w in [("KN-only", (1.0, 0.0, 0.0, 0.0)),
                     ("v24-best (kn0.92 mix0.08)", (0.92, 0.08, 0.0, 0.0)),
                     ("val-best", best_w)]:
        r = eval_blend4(log_p_kn_e, log_p_mix_e, log_p_tri_e, log_p_uni_e, targets_e, *w, compute_topk=True)
        print(f"  {label}  w={w}  PPL={r['ppl']:.2f}  top1={r['top1']*100:.2f}%  top5={r['top5']*100:.2f}%")
        eval_results[label] = {**r, "w": w}

    # also eval the top-5 val configs as backup
    for k, vr in sorted_v_best[:5]:
        w = vr["w"]
        if tuple(round(x, 4) for x in w) in [tuple(round(x, 4) for x in best_w)]:
            continue
        r = eval_blend4(log_p_kn_e, log_p_mix_e, log_p_tri_e, log_p_uni_e, targets_e, *w, compute_topk=True)
        label = f"alt_w=(kn{w[0]:.2f},mix{w[1]:.2f},tri{w[2]:.2f},uni{w[3]:.2f})"
        print(f"  {label}  PPL={r['ppl']:.2f}  top1={r['top1']*100:.2f}%")
        eval_results[label] = {**r, "w": w}

    out = {
        "model": "v26 4-way blend KN5 + v14-mix + trigram-induction + unigram-induction with strict sliding window sweep",
        "best_window": best_w_size, "alpha_tri": args.alpha_tri, "alpha_uni": args.alpha_uni,
        "K_pos": args.K_pos, "V": V, "tau": args.tau, "gamma": args.gamma,
        "best_val_w": best_w,
        "eval_heldout": eval_results,
        "val_top10": [(k, r["ppl"], r["w"]) for k, r in sorted_v_best[:10]],
    }
    out_path = ARTIFACT / f"eval_results_{args.tag}.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\n[save] -> {out_path}")


if __name__ == "__main__":
    main()
