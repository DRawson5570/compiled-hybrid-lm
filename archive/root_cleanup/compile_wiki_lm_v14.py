"""
compile_wiki_lm_v14.py — Top-M sparse routing + larger context
==============================================================

Extension of v13 (mixture-of-cluster-LMs) with two compounding upgrades:

1. **Top-M sparse routing**: at inference, route a token only to the M
   nearest clusters (M << K_clusters). The remaining clusters get zero
   weight. This sharpens the mixture (less mass spread over irrelevant
   clusters) and eliminates the K_clusters × V matmul bottleneck —
   instead we do M × V per token.

2. **Wider K_pos**: v13 used K=3. Now sweepable up to K=8 since the
   residual cost is amortised in cluster lookup, not in a downstream
   ridge head.

The math is otherwise identical to v13. Calibration sweeps M, τ, γ, α
on a held-out validation slice (NEVER train-tail — that calibration trap
is fully recorded in EXPERIMENT_LOG #295).

Optional: --load-counts <path> to reuse the (mu, counts) tensors from a
previous v13 run instead of recomputing — k-means + counts dominate the
non-eval wall-clock at large K_cl.

Outputs:
    artifacts/compiled_wiki_lm_v14/compiled_lm_<tag>.pt
    artifacts/compiled_wiki_lm_v14/eval_results_<tag>.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# Reuse v13 building blocks
sys.path.insert(0, str(Path(__file__).parent))
from compile_wiki_lm_v13 import (
    load_setup, load_or_build_tokens, build_residual,
    kmeans_lloyd, accumulate_cluster_counts, collect_residuals,
    parse_size, DEVICE,
)

REPO = Path("/home/drawson/llm_decoupling")
ARTIFACT = REPO / "artifacts/compiled_wiki_lm_v14"
ARTIFACT.mkdir(parents=True, exist_ok=True)


class SparseMixtureClusterLM:
    """v13 mixture LM with top-M cluster routing at inference."""

    def __init__(self, mu: torch.Tensor, log_p_cluster: torch.Tensor,
                 log_p_uni: torch.Tensor, K_pos: int, V: int, d_emb: int):
        self.mu = mu                          # (K_cl, d_res)
        self.log_p_cluster = log_p_cluster    # (K_cl, V)
        self.log_p_uni = log_p_uni            # (V,)
        self.K_pos = K_pos
        self.V = V
        self.d_emb = d_emb
        self._mu_sq = (mu * mu).sum(dim=1)

    @classmethod
    def from_counts(cls, mu, counts, alpha, V, K_pos, d_emb):
        # Memory-conservative: build log_p chunk-by-chunk over clusters
        # so we never hold counts + sm + p + log_p simultaneously
        # (at K_cl=65536 each is 2 GB → OOM on a 10 GB card).
        K_cl = counts.size(0)
        device = counts.device
        # global unigram (cheap)
        uni = counts.sum(dim=0)
        uni_p = (uni + alpha) / (uni.sum() + alpha * V)
        log_uni = torch.log(uni_p.clamp_min(1e-30))
        # per-cluster log probs in fp16 to halve memory (2 GB → 1 GB at K_cl=65536)
        log_p = torch.empty((K_cl, V), device=device, dtype=torch.float16)
        chunk = max(1, 4096)
        for s in range(0, K_cl, chunk):
            e = min(s + chunk, K_cl)
            sm = counts[s:e].to(torch.float32) + alpha
            p = sm / sm.sum(dim=1, keepdim=True)
            log_p[s:e] = torch.log(p.clamp_min(1e-30)).to(torch.float16)
            del sm, p
        return cls(mu, log_p, log_uni, K_pos, V, d_emb)

    def log_probs(self, R: torch.Tensor, tau: float, gamma: float,
                  top_M: int) -> torch.Tensor:
        """R: (B, d_res). top_M: number of nearest clusters to route to."""
        d2 = (R * R).sum(dim=1, keepdim=True) - 2 * R @ self.mu.t() + self._mu_sq[None]
        # nearest M
        neg_d2_topM, idx_topM = torch.topk(-d2, k=top_M, dim=1)   # (B, M)
        # softmax over those M (ignore rest)
        log_pi = F.log_softmax(neg_d2_topM / tau, dim=-1)         # (B, M)
        # gather per-cluster log distributions for selected clusters
        # log_p_cluster[idx_topM]: (B, M, V) — at top_M=64, B=256, V=8000:
        # 256*64*8000*4 = 512 MB. Manageable but tight; chunk if needed.
        B = R.size(0)
        out = torch.empty(B, self.V, device=R.device, dtype=R.dtype)
        # Chunk by B
        bchunk = max(1, min(B, 256_000_000 // max(top_M * self.V * 4, 1)))
        for s in range(0, B, bchunk):
            e = min(s + bchunk, B)
            lp_sel = self.log_p_cluster[idx_topM[s:e]].to(R.dtype)  # (b, M, V), upcast fp16→fp32
            block = log_pi[s:e].unsqueeze(2) + lp_sel             # (b, M, V)
            log_mix = torch.logsumexp(block, dim=1)               # (b, V)
            out[s:e] = log_mix
        if gamma >= 1.0 - 1e-9:
            return out
        log_g = math.log(gamma); log_1mg = math.log(1.0 - gamma)
        a = log_g + out
        b_ = log_1mg + self.log_p_uni[None].expand_as(out)
        m = torch.maximum(a, b_)
        return m + torch.log(torch.exp(a - m) + torch.exp(b_ - m))


def fast_ppl(R, Y, model, tau, gamma, top_M, inner_batch=256):
    nll_sum = 0.0
    top1 = 0
    top5 = 0
    count = 0
    for i in range(0, R.size(0), inner_batch):
        Rb = R[i:i + inner_batch]
        Yb = Y[i:i + inner_batch]
        logp = model.log_probs(Rb, tau=tau, gamma=gamma, top_M=top_M)
        nll_sum += -logp.gather(1, Yb.unsqueeze(1)).squeeze(1).sum().item()
        top5_idx = logp.topk(5, dim=-1).indices
        top1 += (top5_idx[:, 0] == Yb).sum().item()
        top5 += (top5_idx == Yb.unsqueeze(1)).any(dim=1).sum().item()
        count += Rb.size(0)
    avg = nll_sum / count
    return {"count": count, "ppl": math.exp(avg), "avg_nll": avg,
            "top1": top1 / count, "top5": top5 / count,
            "tau": tau, "gamma": gamma, "top_M": top_M}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--K", type=int, default=3)
    p.add_argument("--clusters", type=int, default=4096)
    p.add_argument("--kmeans-sample", type=str, default="500K")
    p.add_argument("--kmeans-iters", type=int, default=15)
    p.add_argument("--top-M", type=int, default=64,
                   help="default top-M (also swept in calibration)")
    p.add_argument("--alpha", type=float, default=0.01)
    p.add_argument("--train-tokens", type=str, default="10M")
    p.add_argument("--val-tokens", type=str, default="100K")
    p.add_argument("--eval-tokens", type=str, default="500K")
    p.add_argument("--chunk", type=str, default="100K")
    p.add_argument("--inner-batch", type=int, default=256)
    p.add_argument("--load-counts", type=str, default=None,
                   help="path to a .pt with {mu, counts, K_pos, V}; skip k-means/counting")
    p.add_argument("--tag", type=str, default="default")
    args = p.parse_args()

    train_n = parse_size(args.train_tokens)
    val_n = parse_size(args.val_tokens)
    eval_n = parse_size(args.eval_tokens)
    chunk = parse_size(args.chunk)
    km_sample = parse_size(args.kmeans_sample)

    bpe, vocab, tok2id, bpe_to_lm, emb, V, d = load_setup()
    ids = load_or_build_tokens(bpe, bpe_to_lm, V)
    N = ids.size(0)
    if train_n + val_n + eval_n > N:
        train_n = max(N - val_n - eval_n, N // 2)
    train_ids = ids[:train_n]
    val_ids = ids[train_n:train_n + val_n]
    eval_ids = ids[-eval_n:]
    print(f"[split] train={train_n:,}  val={val_n:,}  eval={eval_n:,}")
    emb_dev = emb.to(DEVICE)
    K_pos = args.K
    d_res = (K_pos + 1) * d

    if args.load_counts and Path(args.load_counts).exists():
        print(f"[load] reusing counts from {args.load_counts}")
        blob = torch.load(args.load_counts, map_location=DEVICE, weights_only=False)
        mu = blob["mu"].to(DEVICE)
        counts = blob["counts"].to(DEVICE)
        assert blob["K_pos"] == K_pos and blob["V"] == V
    else:
        # K-means sample
        print(f"[kmeans-data] sampling {km_sample:,} residuals")
        rng = np.random.RandomState(0)
        valid_lo, valid_hi = K_pos, train_n - 1
        n_avail = valid_hi - valid_lo
        sample_n = min(km_sample, n_avail)
        sample_pos = np.sort(rng.choice(n_avail, sample_n, replace=False)) + valid_lo
        Xs = torch.empty(sample_n, d_res, device=DEVICE, dtype=torch.float32)
        cursor = K_pos
        sp_idx = 0
        while cursor < train_n - 1 and sp_idx < sample_n:
            end = min(cursor + chunk, train_n - 1)
            prefix = cursor - K_pos
            window = train_ids[prefix:end + 1].to(DEVICE, non_blocking=True)
            R_full = build_residual(window, emb_dev, K_pos)
            local_lo = K_pos
            while sp_idx < sample_n and sample_pos[sp_idx] < end:
                gpos = sample_pos[sp_idx]
                lp = local_lo + (gpos - cursor)
                Xs[sp_idx] = R_full[lp]
                sp_idx += 1
            cursor = end
        mu = kmeans_lloyd(Xs, K=args.clusters, n_iter=args.kmeans_iters, seed=0)
        del Xs
        counts = accumulate_cluster_counts(train_ids, emb_dev, K_pos, mu, V,
                                           chunk=chunk)
        cache_path = ARTIFACT / f"counts_{args.tag}.pt"
        torch.save({"mu": mu.cpu(), "counts": counts.cpu(),
                    "K_pos": K_pos, "V": V, "clusters": args.clusters,
                    "train_tokens": train_n}, cache_path)
        print(f"[save] counts cache -> {cache_path}")

    # Cal residuals
    print(f"\n[cal] precomputing residuals on {val_n:,} val tokens")
    cal_R, cal_Y = collect_residuals(val_ids, emb_dev, K_pos, chunk=chunk)
    print(f"  R={tuple(cal_R.shape)}")

    alphas = sorted({args.alpha, 0.01, 0.1, 1.0})
    taus = [0.03, 0.1, 0.3, 1.0]
    Ms = sorted({args.top_M, 16, 64, 256})
    best = {"ppl": float("inf"), "alpha": args.alpha,
            "tau": 1.0, "gamma": 1.0, "top_M": args.top_M}
    print(f"[cal] sweeping α∈{alphas} τ∈{taus} M∈{Ms}")
    for alpha in alphas:
        model = SparseMixtureClusterLM.from_counts(mu, counts, alpha=alpha,
                                                    V=V, K_pos=K_pos, d_emb=d)
        for M in Ms:
            for tau in taus:
                r = fast_ppl(cal_R, cal_Y, model, tau=tau, gamma=1.0,
                             top_M=M, inner_batch=args.inner_batch)
                print(f"  α={alpha} τ={tau} M={M} γ=1 → "
                      f"PPL={r['ppl']:.2f}  top1={r['top1']*100:.2f}%")
                if r["ppl"] < best["ppl"]:
                    best = {"ppl": r["ppl"], "alpha": alpha, "tau": tau,
                            "gamma": 1.0, "top_M": M}
        del model
        torch.cuda.empty_cache()
    # gamma sweep at best (rebuild model with best alpha)
    bm = SparseMixtureClusterLM.from_counts(mu, counts, alpha=best["alpha"],
                                             V=V, K_pos=K_pos, d_emb=d)
    for gamma in [0.7, 0.85, 0.95, 0.99]:
        r = fast_ppl(cal_R, cal_Y, bm, tau=best["tau"], gamma=gamma,
                     top_M=best["top_M"], inner_batch=args.inner_batch)
        print(f"  α={best['alpha']} τ={best['tau']} M={best['top_M']} γ={gamma}"
              f" → PPL={r['ppl']:.2f}")
        if r["ppl"] < best["ppl"]:
            best = {**best, "gamma": gamma}
    best_model = bm
    print(f"[cal] best α={best['alpha']} τ={best['tau']} M={best['top_M']} "
          f"γ={best['gamma']} PPL={best['ppl']:.2f}")
    model = best_model
    del cal_R, cal_Y

    # Final evals
    print("\n[eval] in-distribution")
    R_in, Y_in = collect_residuals(train_ids[:eval_n], emb_dev, K_pos, chunk=chunk)
    sanity = fast_ppl(R_in, Y_in, model, tau=best["tau"], gamma=best["gamma"],
                       top_M=best["top_M"], inner_batch=args.inner_batch)
    print(f"  train(in): PPL={sanity['ppl']:.2f}  top1={sanity['top1']*100:.2f}%  "
          f"top5={sanity['top5']*100:.2f}%")
    del R_in, Y_in

    print("[eval] heldout")
    R_h, Y_h = collect_residuals(eval_ids, emb_dev, K_pos, chunk=chunk)
    held = fast_ppl(R_h, Y_h, model, tau=best["tau"], gamma=best["gamma"],
                    top_M=best["top_M"], inner_batch=args.inner_batch)
    print(f"  heldout: PPL={held['ppl']:.2f}  top1={held['top1']*100:.2f}%  "
          f"top5={held['top5']*100:.2f}%")
    del R_h, Y_h

    out = ARTIFACT / f"compiled_lm_{args.tag}.pt"
    torch.save({
        "K_pos": K_pos, "clusters": args.clusters,
        "alpha": best["alpha"], "top_M": best["top_M"],
        "V": V, "d_emb": d, "d_res": d_res,
        "mu": model.mu.cpu(),
        "log_p_cluster": model.log_p_cluster.cpu(),
        "log_p_uni": model.log_p_uni.cpu(),
        "best_tau": best["tau"], "best_gamma": best["gamma"],
        "train_tokens": train_n,
    }, str(out))
    print(f"[save] -> {out}")

    results = {
        "model": "Compiled Wikitext LM v14 (sparse top-M mixture)",
        "K_pos": K_pos, "clusters": args.clusters,
        "best_alpha": best["alpha"], "best_top_M": best["top_M"],
        "best_tau": best["tau"], "best_gamma": best["gamma"],
        "train_tokens": train_n, "eval_tokens": eval_n,
        "V": V, "d_emb": d, "d_res": d_res,
        "in_distribution": sanity, "heldout": held,
    }
    rp = ARTIFACT / f"eval_results_{args.tag}.json"
    with open(rp, "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
