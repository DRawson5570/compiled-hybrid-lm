"""
compile_wiki_lm_v27.py — Multi-window decayed induction ensemble
============================================================

Hypothesis: local repetition patterns operate on different time scales.
Fast-dynamics capture conversational/recent entity burstiness, whereas slow-dynamics
capture article-level semantic recurrence.
We run parallel unigram decay tables:
  1. Fast unigram decay: lam_uni_fast (e.g. 0.005, half-life ~140 tokens)
  2. Slow unigram decay: lam_uni_slow (e.g. 0.0005, half-life ~1400 tokens)
And a decayed trigram table (lam_tri = 0.001) with sliding window size of 8192.
We blend them with global KN5 and positional-mixture v14 into a 5-way blend:
  P = w_kn * P_kn + w_mix * P_mix + w_tri * P_tri + w_ufast * P_uni_fast + w_uslow * P_uni_slow
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

ARTIFACT = Path("/home/drawson/deepseek_experiments/artifacts/compiled_wiki_lm_v27")
ARTIFACT.mkdir(parents=True, exist_ok=True)


@torch.no_grad()
def build_multi_induction_log_probs(
    ids_np: np.ndarray, V: int, K_pos: int, window: int,
    lam_tri: float, lam_uni_fast: float, lam_uni_slow: float,
    alpha_tri: float, alpha_uni_fast: float, alpha_uni_slow: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stream ids; build sliding-window decayed trigram and dual unigram induction.

    Applies lazy decay update formulas:
      score_t = score_last * exp(-lam * delta_t)

    Returns (log_p_tri, log_p_uni_fast, log_p_uni_slow) arrays of shape (N - K_pos, V).
    """
    N = len(ids_np)
    T = N - K_pos

    # 1. Decayed Unigram - Fast
    Bu_fast = torch.zeros(V, V, device=DEVICE, dtype=torch.float32)
    Ru_fast = torch.zeros(V, device=DEVICE, dtype=torch.float32)
    last_u_fast = torch.zeros(V, V, device=DEVICE, dtype=torch.int32)
    last_Ru_fast = torch.zeros(V, device=DEVICE, dtype=torch.int32)

    # 2. Decayed Unigram - Slow
    Bu_slow = torch.zeros(V, V, device=DEVICE, dtype=torch.float32)
    Ru_slow = torch.zeros(V, device=DEVICE, dtype=torch.float32)
    last_u_slow = torch.zeros(V, V, device=DEVICE, dtype=torch.int32)
    last_Ru_slow = torch.zeros(V, device=DEVICE, dtype=torch.int32)

    # 3. Decayed Trigram
    Bt: dict[tuple, torch.Tensor] = {}
    Rt: dict[tuple, float] = {}
    last_t: dict[tuple, torch.Tensor] = {}
    last_Rt: dict[tuple, int] = {}
    
    history = deque()

    log_p_tri = np.zeros((T, V), dtype=np.float32)
    log_p_uni_fast = np.zeros((T, V), dtype=np.float32)
    log_p_uni_slow = np.zeros((T, V), dtype=np.float32)

    aV_ufast = alpha_uni_fast * V
    aV_uslow = alpha_uni_slow * V
    aV_t = alpha_tri * V
    log_uniform = math.log(1.0 / V)

    t0 = time.time()
    for t in range(K_pos, N):
        i = t - K_pos
        b = int(ids_np[t - 1])

        # A. Gather Decayed Unigram - Fast
        # Lazy sync for the row 'b' fast unigram
        p_fast_num = Bu_fast[b].clone()
        fast_dt = t - last_u_fast[b]
        p_fast_num = p_fast_num * torch.exp(-lam_uni_fast * fast_dt)
        fast_R_dt = t - last_Ru_fast[b]
        fast_den = Ru_fast[b] * math.exp(-lam_uni_fast * fast_R_dt)
        p_fast_row = (p_fast_num + alpha_uni_fast) / (fast_den + aV_ufast)
        log_p_uni_fast[i] = torch.log(p_fast_row.clamp_min(1e-30)).cpu().numpy()

        # B. Gather Decayed Unigram - Slow
        p_slow_num = Bu_slow[b].clone()
        slow_dt = t - last_u_slow[b]
        p_slow_num = p_slow_num * torch.exp(-lam_uni_slow * slow_dt)
        slow_R_dt = t - last_Ru_slow[b]
        slow_den = Ru_slow[b] * math.exp(-lam_uni_slow * slow_R_dt)
        p_slow_row = (p_slow_num + alpha_uni_slow) / (slow_den + aV_uslow)
        log_p_uni_slow[i] = torch.log(p_slow_row.clamp_min(1e-30)).cpu().numpy()

        # C. Gather Decayed Trigram
        if t >= 2:
            a = int(ids_np[t - 2])
            key = (a, b)
            row_t = Bt.get(key)
            if row_t is not None:
                tri_num = row_t.clone()
                tri_last = last_t[key]
                tri_dt = t - tri_last
                tri_num = tri_num * torch.exp(-lam_tri * tri_dt)
                tri_R_dt = t - last_Rt[key]
                tri_den = Rt[key] * math.exp(-lam_tri * tri_R_dt)
                p_tri_row = (tri_num + alpha_tri) / (tri_den + aV_t)
                log_p_tri[i] = torch.log(p_tri_row.clamp_min(1e-30)).cpu().numpy()
            else:
                log_p_tri[i].fill(log_uniform)
        else:
            log_p_tri[i].fill(log_uniform)

        # D. Update counts with the observed transition (b -> c)
        c = int(ids_np[t])

        # Unigram Fast Update
        Bu_fast[b] = Bu_fast[b] * torch.exp(-lam_uni_fast * (t - last_u_fast[b]))
        Bu_fast[b, c] += 1.0
        last_u_fast[b] = t
        
        Ru_fast[b] = Ru_fast[b] * math.exp(-lam_uni_fast * (t - last_Ru_fast[b])) + 1.0
        last_Ru_fast[b] = t

        # Unigram Slow Update
        Bu_slow[b] = Bu_slow[b] * torch.exp(-lam_uni_slow * (t - last_u_slow[b]))
        Bu_slow[b, c] += 1.0
        last_u_slow[b] = t
        
        Ru_slow[b] = Ru_slow[b] * math.exp(-lam_uni_slow * (t - last_Ru_slow[b])) + 1.0
        last_Ru_slow[b] = t

        # Trigram Update
        a_val = None
        key_val = None
        if t >= 2:
            a_val = int(ids_np[t - 2])
            key_val = (a_val, b)
            if key_val in Bt:
                Bt[key_val] = Bt[key_val] * torch.exp(-lam_tri * (t - last_t[key_val]))
                Bt[key_val][c] += 1.0
                last_t[key_val][c] = t
                
                Rt[key_val] = Rt[key_val] * math.exp(-lam_tri * (t - last_Rt[key_val])) + 1.0
                last_Rt[key_val] = t
            else:
                v = torch.zeros(V, device=DEVICE, dtype=torch.float32)
                v[c] = 1.0
                Bt[key_val] = v
                last_arr = torch.zeros(V, device=DEVICE, dtype=torch.int32)
                last_arr[c] = t
                last_t[key_val] = last_arr
                
                Rt[key_val] = 1.0
                last_Rt[key_val] = t

        history.append((a_val, b, c, key_val))

        # Sliding Window Eviction
        if len(history) > window:
            old_a, old_b, old_c, old_key = history.popleft()
            # Unigram Fast Eviction
            fast_evict_t = t - window
            Bu_fast[old_b] = Bu_fast[old_b] * torch.exp(-lam_uni_fast * (t - last_u_fast[old_b]))
            Bu_fast[old_b, old_c] -= math.exp(-lam_uni_fast * (t - fast_evict_t))
            Bu_fast[old_b, old_c] = max(0.0, Bu_fast[old_b, old_c])
            last_u_fast[old_b] = t
            
            Ru_fast[old_b] = Ru_fast[old_b] * math.exp(-lam_uni_fast * (t - last_Ru_fast[old_b])) - math.exp(-lam_uni_fast * (t - fast_evict_t))
            Ru_fast[old_b] = max(0.0, Ru_fast[old_b])
            last_Ru_fast[old_b] = t

            # Unigram Slow Eviction
            slow_evict_t = t - window
            Bu_slow[old_b] = Bu_slow[old_b] * torch.exp(-lam_uni_slow * (t - last_u_slow[old_b]))
            Bu_slow[old_b, old_c] -= math.exp(-lam_uni_slow * (t - slow_evict_t))
            Bu_slow[old_b, old_c] = max(0.0, Bu_slow[old_b, old_c])
            last_u_slow[old_b] = t
            
            Ru_slow[old_b] = Ru_slow[old_b] * math.exp(-lam_uni_slow * (t - last_Ru_slow[old_b])) - math.exp(-lam_uni_slow * (t - slow_evict_t))
            Ru_slow[old_b] = max(0.0, Ru_slow[old_b])
            last_Ru_slow[old_b] = t

            # Trigram Eviction
            if old_key is not None:
                row_t = Bt.get(old_key)
                if row_t is not None:
                    tri_evict_t = t - window
                    row_t = row_t * torch.exp(-lam_tri * (t - last_t[old_key]))
                    row_t[old_c] -= math.exp(-lam_tri * (t - tri_evict_t))
                    row_t[old_c] = max(0.0, row_t[old_c])
                    Bt[old_key] = row_t
                    last_t[old_key][old_c] = t
                    
                    Rt[old_key] = Rt[old_key] * math.exp(-lam_tri * (t - last_Rt[old_key])) - math.exp(-lam_tri * (t - tri_evict_t))
                    Rt[old_key] = max(0.0, Rt[old_key])
                    last_Rt[old_key] = t
                    
                    if Rt[old_key] <= 0:
                        del Bt[old_key]
                        del Rt[old_key]
                        del last_t[old_key]
                        del last_Rt[old_key]

        if (i + 1) % 10000 == 0:
            print(f"    [ind] {i+1}/{T} ({time.time() - t0:.1f}s, tri-keys={len(Bt)})")

    return log_p_tri, log_p_uni_fast, log_p_uni_slow


def eval_blend5(log_p_kn, log_p_mix, log_p_tri, log_p_uni_fast, log_p_uni_slow, targets,
                w_kn, w_mix, w_tri, w_ufast, w_uslow, compute_topk=False):
    """Linear prob-space blend: P = sum_k w_k * P_k (no global renorm — components already normalized)."""
    s = w_kn + w_mix + w_tri + w_ufast + w_uslow
    if s <= 0:
        return {"ppl": float("inf"), "top1": 0.0, "top5": 0.0, "n": int(len(targets))}
    if abs(s - 1.0) > 1e-6:
        w_kn, w_mix, w_tri, w_ufast, w_uslow = w_kn/s, w_mix/s, w_tri/s, w_ufast/s, w_uslow/s

    n = len(targets)
    idx = np.arange(n)
    lpk_t = log_p_kn[idx, targets]
    lpm_t = log_p_mix[idx, targets]
    lpt_t = log_p_tri[idx, targets]
    lpf_t = log_p_uni_fast[idx, targets]
    lps_t = log_p_uni_slow[idx, targets]

    parts = []
    for w, lp_t in [(w_kn, lpk_t), (w_mix, lpm_t), (w_tri, lpt_t), (w_ufast, lpf_t), (w_uslow, lps_t)]:
        if w <= 0:
            continue
        parts.append(math.log(w) + lp_t)
    stack = np.stack(parts, axis=0)
    m = stack.max(axis=0)
    log_p_target = m + np.log(np.exp(stack - m[None]).sum(axis=0))
    nll = -log_p_target.sum()
    ppl = math.exp(nll / n)

    if not compute_topk:
        return {"ppl": ppl, "top1": None, "top5": None, "n": int(n)}

    c1 = 0
    c5 = 0
    BATCH = 4096
    parts_full_cfgs = [(w, lp) for w, lp in
                       [(w_kn, log_p_kn), (w_mix, log_p_mix),
                        (w_tri, log_p_tri), (w_ufast, log_p_uni_fast), (w_uslow, log_p_uni_slow)] if w > 0]
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
    p.add_argument("--alpha-uni-fast", type=float, default=1e-5)
    p.add_argument("--alpha-uni-slow", type=float, default=1e-5)
    p.add_argument("--lam-tri", type=float, default=0.001)
    p.add_argument("--lam-uni-fast", type=float, default=0.005)
    p.add_argument("--lam-uni-slow", type=float, default=0.0005)
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

    print(f"[v27] Dual Decayed Unigram Induction + Decayed Trigram")
    print(f"  lam_tri={args.lam_tri} lam_uni_fast={args.lam_uni_fast} lam_uni_slow={args.lam_uni_slow}")
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

    def prepare(ids_t, ids_n, label):
        print(f"\n[{label}] computing 5 component log-prob tables")
        t0 = time.time()
        log_p_mix = compute_log_p_mix(ids_t, emb_dev, model, args.K_pos,
                                       args.top_M, args.tau, args.gamma).numpy()
        print(f"  mix done ({time.time() - t0:.1f}s, shape={log_p_mix.shape})")
        t0 = time.time()
        log_p_kn = compute_log_p_kn(kn, ids_n, args.K_pos)
        print(f"  KN done ({time.time() - t0:.1f}s)")
        t0 = time.time()
        log_p_tri, log_p_uni_f, log_p_uni_s = build_multi_induction_log_probs(
            ids_n, V, args.K_pos, args.window, args.lam_tri, args.lam_uni_fast, args.lam_uni_slow,
            args.alpha_tri, args.alpha_uni_fast, args.alpha_uni_slow
        )
        print(f"  tri/uni-fast/uni-slow done ({time.time() - t0:.1f}s)")
        targets = ids_n[args.K_pos:]
        return log_p_kn, log_p_mix, log_p_tri, log_p_uni_f, log_p_uni_s, targets

    # VAL
    log_p_kn_v, log_p_mix_v, log_p_tri_v, log_p_uni_f_v, log_p_uni_s_v, targets_v = prepare(val_ids_t, val_ids_n, "val")

    print(f"\n[val] coarse 5-way simplex sweep")
    grid = []
    # baselines
    grid.append((1.0, 0.0, 0.0, 0.0, 0.0))
    grid.append((0.92, 0.08, 0.0, 0.0, 0.0))
    # search around optimal weights
    for w_tri in [0.0, 0.03, 0.05, 0.08]:
        for w_ufast in [0.0, 0.03, 0.06, 0.10]:
            for w_uslow in [0.0, 0.05, 0.10, 0.15]:
                for w_mix_frac in [0.02, 0.05, 0.08]:
                    w_other = w_tri + w_ufast + w_uslow + w_mix_frac
                    if w_other >= 1.0:
                        continue
                    w_kn = 1.0 - w_other
                    grid.append((w_kn, w_mix_frac, w_tri, w_ufast, w_uslow))
                    
    seen = set()
    val_results = {}
    best = (float("inf"), None)
    for w in grid:
        key = tuple(round(x, 4) for x in w)
        if key in seen:
            continue
        seen.add(key)
        r = eval_blend5(log_p_kn_v, log_p_mix_v, log_p_tri_v, log_p_uni_f_v, log_p_uni_s_v, targets_v, *w)
        val_results[str(key)] = {**r, "w": w}
        if r["ppl"] < best[0]:
            best = (r["ppl"], w)
    sorted_v = sorted(val_results.items(), key=lambda kv: kv[1]["ppl"])
    print(f"  top-10 val configs:")
    for k, r in sorted_v[:10]:
        w = r["w"]
        print(f"    w=(kn{w[0]:.2f},mix{w[1]:.2f},tri{w[2]:.2f},ufast{w[3]:.2f},uslow{w[4]:.2f})  "
              f"PPL={r['ppl']:7.2f}")
    best_w = sorted_v[0][1]["w"]
    print(f"\n[val] best w={best_w}  PPL={best[0]:.2f}")

    # free val memory
    del log_p_kn_v, log_p_mix_v, log_p_tri_v, log_p_uni_f_v, log_p_uni_s_v, targets_v

    # HELDOUT
    log_p_kn_e, log_p_mix_e, log_p_tri_e, log_p_uni_f_e, log_p_uni_s_e, targets_e = prepare(eval_ids_t, eval_ids_n, "eval")

    print(f"\n[eval] HELDOUT — evaluating search best configurations")
    eval_results = {}
    for label, w in [("KN-only", (1.0, 0.0, 0.0, 0.0, 0.0)),
                     ("v26-best (kn0.79, mix0.05, tri0.05, uni0.12)", (0.78, 0.05, 0.05, 0.0, 0.12)),
                     ("val-best", best_w)]:
        r = eval_blend5(log_p_kn_e, log_p_mix_e, log_p_tri_e, log_p_uni_f_e, log_p_uni_s_e, targets_e, *w, compute_topk=True)
        print(f"  {label}  w={w}  PPL={r['ppl']:.2f}  top1={r['top1']*100:.2f}%  top5={r['top5']*100:.2f}%")
        eval_results[label] = {**r, "w": w}

    for k, vr in sorted_v[:5]:
        w = vr["w"]
        if tuple(round(x, 4) for x in w) in [tuple(round(x, 4) for x in best_w)]:
            continue
        r = eval_blend5(log_p_kn_e, log_p_mix_e, log_p_tri_e, log_p_uni_f_e, log_p_uni_s_e, targets_e, *w, compute_topk=True)
        label = f"alt_w=(kn{w[0]:.2f},mix{w[1]:.2f},tri{w[2]:.2f},ufast{w[3]:.2f},uslow{w[4]:.2f})"
        print(f"  {label}  PPL={r['ppl']:.2f}  top1={r['top1']*100:.2f}%")
        eval_results[label] = {**r, "w": w}

    out = {
        "model": "v27 5-way blend with Dual Decayed Unigram Induction",
        "window": args.window, "alpha_tri": args.alpha_tri, "alpha_uni_fast": args.alpha_uni_fast, "alpha_uni_slow": args.alpha_uni_slow,
        "lam_tri": args.lam_tri, "lam_uni_fast": args.lam_uni_fast, "lam_uni_slow": args.lam_uni_slow,
        "best_val_w": best_w,
        "eval_heldout": eval_results,
        "val_top10": [(k, r["ppl"], r["w"]) for k, r in sorted_v[:10]],
    }
    out_path = ARTIFACT / f"eval_results_pair_{args.tag}.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\n[save] -> {out_path}")


if __name__ == "__main__":
    main()
