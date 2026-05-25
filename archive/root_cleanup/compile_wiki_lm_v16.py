"""
compile_wiki_lm_v16.py — Stacked compiled mixture (two FFN layers)
===================================================================

Reuses v14's first compiled mixture-of-cluster-LM as **layer 1**, then:

1. **Layer-1 FFN output (Δ1)**: for each cluster k of layer 1, compute the
   *expected next-token embedding* under that cluster's empirical
   distribution:

       Δ_k = Σ_y p_k(y) · emb[y]    ∈ R^d_emb        (one per cluster)

   Then for a residual r0, the layer-1 FFN output is the soft mixture:

       Δ1(r0) = Σ_k π_k(r0) · Δ_k    where π_k = softmax_top_M(-d²/τ)

   This is the closed-form key-value FFN from Geva et al.: keys = cluster
   centroids, values = mean target-embedding per cluster. It maps a
   positional residual to a context-conditional shift in *embedding space*
   (R^d_emb), exactly the role of the trained FFN's output projection.

2. **Layer-2 input residual r1**: concatenate the original positional
   residual with the layer-1 FFN output:

       r1 = concat(r0, Δ1(r0))     ∈ R^(d_res0 + d_emb)

   Carries the lexical positional pattern AND the layer-1 semantic
   resolution into the next layer.

3. **Layer-2 mixture**: cluster the layer-1 output residuals r1 into a new
   set of K_cl2 centroids, then build a second mixture-of-cluster-LM over
   *next-token targets* using exactly the v13/v14 recipe but with r1
   replacing r0.

4. **Final inference**: layer-2 mixture predicts log p(y | r1). Optionally,
   product-of-experts blend with layer 1.

The compile path is fully closed-form: no SGD, no learned parameters
beyond the calibration scalars (α, τ, γ, top_M).

Usage:
    python compile_wiki_lm_v16.py \\
        --K 2 --clusters1 65536 --clusters2 65536 \\
        --load-counts1 artifacts/compiled_wiki_lm_v14/counts_k2_c64k.pt \\
        --train-tokens 22M --val-tokens 100K --eval-tokens 500K \\
        --top-M1 16 --top-M2 16 --tag stacked_k2_c64k_c64k
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

sys.path.insert(0, str(Path(__file__).parent))
from compile_wiki_lm_v13 import (
    load_setup, load_or_build_tokens, build_residual,
    kmeans_lloyd, accumulate_cluster_counts, collect_residuals,
    parse_size, DEVICE,
)
from compile_wiki_lm_v14 import SparseMixtureClusterLM, fast_ppl

REPO = Path("/home/drawson/llm_decoupling")
ARTIFACT = REPO / "artifacts/compiled_wiki_lm_v16"
ARTIFACT.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Layer-1 FFN value computation (Δ_k = E_{y~p_k}[emb[y]])
# =============================================================================

def compute_layer1_values(counts: torch.Tensor, emb: torch.Tensor,
                          alpha: float) -> torch.Tensor:
    """Δ_k = Σ_y p_k(y) · emb[y].

    counts: (K_cl, V) cluster→target raw counts (or smoothed).
    emb: (V, d_emb).
    Returns: (K_cl, d_emb) FFN value vectors.

    Chunked over clusters to stay in memory at large K_cl × V × d_emb.
    """
    K_cl, V = counts.shape
    d = emb.size(1)
    out = torch.empty(K_cl, d, device=counts.device, dtype=torch.float32)
    chunk = max(1, 16 * 1024 * 1024 // (V * 4))  # ~64 MB per chunk
    chunk = max(64, min(chunk, K_cl))
    for s in range(0, K_cl, chunk):
        e = min(s + chunk, K_cl)
        sm = counts[s:e].to(torch.float32) + alpha
        p = sm / sm.sum(dim=1, keepdim=True)            # (b, V)
        out[s:e] = p @ emb                              # (b, d)
        del sm, p
    return out


# =============================================================================
# Stacked residual builder: r1 = concat(r0, Δ1(r0))
# =============================================================================

class Layer1FFN:
    """Soft k-NN FFN: input r0 → output Δ1 ∈ R^d_emb via cluster mixture.

    π_k(r0) = softmax_top_M(-‖r0 − μ_k‖² / τ)
    Δ1(r0)  = Σ_k π_k(r0) · Δ_k
    """

    def __init__(self, mu: torch.Tensor, deltas: torch.Tensor, tau: float,
                 top_M: int):
        self.mu = mu                  # (K_cl, d_res0)
        self.deltas = deltas          # (K_cl, d_emb)
        self.tau = tau
        self.top_M = top_M
        self._mu_sq = (mu * mu).sum(dim=1)

    def forward(self, R0: torch.Tensor) -> torch.Tensor:
        # Chunk over R0 batch to bound peak memory of the (B, K_cl) distance matrix
        # at large K_cl (= 65k → 8192 × 65536 × 4 = 2 GB without chunking).
        B = R0.size(0)
        d_emb = self.deltas.size(1)
        # Target peak ≈ 256 MB for the d2 matrix → sub = 256MB / (K_cl * 4)
        K_cl = self.mu.size(0)
        sub = max(64, min(B, max(1, (256 * 1024 * 1024) // (K_cl * 4))))
        out = torch.empty(B, d_emb, device=R0.device, dtype=R0.dtype)
        for s in range(0, B, sub):
            e = min(s + sub, B)
            Rb = R0[s:e]
            d2 = (Rb * Rb).sum(dim=1, keepdim=True) - 2 * Rb @ self.mu.t() + self._mu_sq[None]
            neg_d2_topM, idx_topM = torch.topk(-d2, k=self.top_M, dim=1)
            pi = F.softmax(neg_d2_topM / self.tau, dim=-1)
            delta_sel = self.deltas[idx_topM]
            out[s:e] = (pi.unsqueeze(2) * delta_sel).sum(dim=1)
            del d2, neg_d2_topM, idx_topM, pi, delta_sel
        return out


def collect_stacked_residuals(ids: torch.Tensor, emb_dev: torch.Tensor,
                              K_pos: int, ffn1: Layer1FFN,
                              chunk: int = 100_000) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute r1 = concat(r0, Δ1(r0)) for every valid position in ids.

    Returns (R1, Y) where R1 is (N_valid, d_res0 + d_emb).
    """
    N = ids.size(0)
    d_emb = emb_dev.size(1)
    d_res0 = (K_pos + 1) * d_emb
    d_res1 = d_res0 + d_emb
    total = max(0, N - K_pos - 1)
    R1 = torch.empty(total, d_res1, device=DEVICE, dtype=torch.float32)
    Y = torch.empty(total, device=DEVICE, dtype=torch.long)
    out_idx = 0
    cursor = K_pos
    t0 = time.time()
    while cursor < N - 1:
        end = min(cursor + chunk, N - 1)
        prefix = cursor - K_pos
        window = ids[prefix:end + 1].to(DEVICE, non_blocking=True)
        R_full = build_residual(window, emb_dev, K_pos)
        local_lo = K_pos
        local_hi = K_pos + (end - cursor)
        R0_chunk = R_full[local_lo:local_hi]
        # Apply layer-1 FFN in sub-batches to bound memory at large K_cl
        sub = 8192
        D1 = torch.empty(R0_chunk.size(0), d_emb, device=DEVICE, dtype=torch.float32)
        for s in range(0, R0_chunk.size(0), sub):
            e = min(s + sub, R0_chunk.size(0))
            D1[s:e] = ffn1.forward(R0_chunk[s:e])
        R1_chunk = torch.cat([R0_chunk, D1], dim=1)
        b = R1_chunk.size(0)
        R1[out_idx:out_idx + b] = R1_chunk
        Y[out_idx:out_idx + b] = window[local_lo + 1:local_hi + 1].long()
        out_idx += b
        del R_full, R0_chunk, D1, R1_chunk
        cursor = end
    print(f"  collected {out_idx:,} stacked residuals in {time.time()-t0:.1f}s")
    return R1[:out_idx], Y[:out_idx]


def accumulate_cluster_counts_stacked(ids: torch.Tensor, emb_dev: torch.Tensor,
                                       K_pos: int, ffn1: Layer1FFN,
                                       mu2: torch.Tensor, V: int,
                                       chunk: int = 100_000) -> torch.Tensor:
    """Stream-accumulate (K_cl2, V) cluster→target counts in stacked-residual space."""
    K_cl2 = mu2.size(0)
    d_res1 = mu2.size(1)
    mu_sq = (mu2 * mu2).sum(dim=1)
    counts = torch.zeros(K_cl2, V, device=DEVICE, dtype=torch.float32)
    N = ids.size(0)
    cursor = K_pos
    t0 = time.time()
    last_report = t0
    total_done = 0
    sub = 8192
    while cursor < N - 1:
        end = min(cursor + chunk, N - 1)
        prefix = cursor - K_pos
        window = ids[prefix:end + 1].to(DEVICE, non_blocking=True)
        R_full = build_residual(window, emb_dev, K_pos)
        local_lo = K_pos
        local_hi = K_pos + (end - cursor)
        R0_chunk = R_full[local_lo:local_hi]
        Y_chunk = window[local_lo + 1:local_hi + 1].long()
        # build R1 and immediately route to nearest cluster
        for s in range(0, R0_chunk.size(0), sub):
            e = min(s + sub, R0_chunk.size(0))
            d1 = ffn1.forward(R0_chunk[s:e])
            r1 = torch.cat([R0_chunk[s:e], d1], dim=1)
            d2 = (r1 * r1).sum(dim=1, keepdim=True) - 2 * r1 @ mu2.t() + mu_sq[None]
            assign = d2.argmin(dim=1)                      # (b,)
            yb = Y_chunk[s:e]
            # one-hot scatter-add via index_put
            counts.index_put_((assign, yb),
                              torch.ones_like(assign, dtype=torch.float32),
                              accumulate=True)
            del d1, r1, d2, assign
        total_done += R0_chunk.size(0)
        del R_full, R0_chunk, Y_chunk
        cursor = end
        now = time.time()
        if now - last_report > 5:
            rate = total_done / max(now - t0, 1e-6)
            print(f"  ... counts {total_done:,}/{N - K_pos - 1:,} "
                  f"({rate/1e6:.2f}M/s, {now - t0:.1f}s)")
            last_report = now
    print(f"[counts2] done in {time.time()-t0:.1f}s")
    return counts


# =============================================================================
# Main
# =============================================================================

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--K", type=int, default=2)
    p.add_argument("--clusters1", type=int, default=65536,
                   help="layer-1 cluster count (must match --load-counts1)")
    p.add_argument("--clusters2", type=int, default=65536,
                   help="layer-2 cluster count")
    p.add_argument("--kmeans-sample", type=str, default="400K")
    p.add_argument("--kmeans-iters", type=int, default=12)
    p.add_argument("--top-M1", type=int, default=16,
                   help="layer-1 top-M for the FFN soft k-NN")
    p.add_argument("--top-M2", type=int, default=16)
    p.add_argument("--alpha1", type=float, default=0.01)
    p.add_argument("--alpha2", type=float, default=0.01)
    p.add_argument("--tau1", type=float, default=0.3,
                   help="layer-1 FFN temperature (fixed, from v14 calibration)")
    p.add_argument("--train-tokens", type=str, default="22M")
    p.add_argument("--val-tokens", type=str, default="100K")
    p.add_argument("--eval-tokens", type=str, default="300K")
    p.add_argument("--chunk", type=str, default="100K")
    p.add_argument("--inner-batch", type=int, default=256)
    p.add_argument("--load-counts1", type=str, required=True,
                   help="path to v14 counts cache (mu+counts for layer 1)")
    p.add_argument("--load-counts2", type=str, default=None,
                   help="optional path to a stacked v16 layer-2 counts cache")
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
    d_res0 = (K_pos + 1) * d
    d_res1 = d_res0 + d

    # --- Layer 1 from cache ---
    print(f"\n[layer1] loading counts from {args.load_counts1}")
    blob1 = torch.load(args.load_counts1, map_location=DEVICE, weights_only=False)
    mu1 = blob1["mu"].to(DEVICE).to(torch.float32)
    counts1 = blob1["counts"].to(DEVICE)
    assert blob1["K_pos"] == K_pos, f"K_pos mismatch: cache={blob1['K_pos']}, args={K_pos}"
    assert mu1.size(0) == args.clusters1, f"cluster count mismatch: {mu1.size(0)} vs {args.clusters1}"
    print(f"  layer1: K_cl={mu1.size(0)}, d_res0={mu1.size(1)}, V={counts1.size(1)}")

    # Layer-1 FFN values: Δ_k = Σ_y p_k(y) · emb[y]
    print("[layer1] computing FFN values Δ_k = E_p[emb]")
    t0 = time.time()
    deltas1 = compute_layer1_values(counts1, emb_dev, alpha=args.alpha1)
    print(f"  Δ shape={tuple(deltas1.shape)}, ||Δ||_avg={deltas1.norm(dim=1).mean().item():.4f}, "
          f"in {time.time()-t0:.1f}s")
    # Layer-1 counts no longer needed (deltas captured everything we use).
    del counts1
    torch.cuda.empty_cache()

    ffn1 = Layer1FFN(mu1, deltas1, tau=args.tau1, top_M=args.top_M1)

    # --- Layer 2: cluster the stacked residuals ---
    if args.load_counts2 and Path(args.load_counts2).exists():
        print(f"\n[layer2] loading cached layer-2 counts from {args.load_counts2}")
        blob2 = torch.load(args.load_counts2, map_location=DEVICE, weights_only=False)
        mu2 = blob2["mu2"].to(DEVICE).to(torch.float32)
        counts2 = blob2["counts2"].to(DEVICE)
    else:
        print(f"\n[layer2] k-means sample: {km_sample:,} stacked residuals")
        # Sample positions
        rng = np.random.RandomState(0)
        valid_lo, valid_hi = K_pos, train_n - 1
        n_avail = valid_hi - valid_lo
        sample_n = min(km_sample, n_avail)
        sample_pos = np.sort(rng.choice(n_avail, sample_n, replace=False)) + valid_lo
        Xs = torch.empty(sample_n, d_res1, device=DEVICE, dtype=torch.float32)
        cursor = K_pos
        sp_idx = 0
        sub = 8192
        t0 = time.time()
        while cursor < train_n - 1 and sp_idx < sample_n:
            end = min(cursor + chunk, train_n - 1)
            prefix = cursor - K_pos
            window = train_ids[prefix:end + 1].to(DEVICE, non_blocking=True)
            R_full = build_residual(window, emb_dev, K_pos)
            local_lo = K_pos
            R0_chunk = R_full[local_lo:K_pos + (end - cursor)]
            # Compute Δ1 for the whole chunk in sub-batches, then pick our samples
            D1 = torch.empty(R0_chunk.size(0), d, device=DEVICE, dtype=torch.float32)
            for s in range(0, R0_chunk.size(0), sub):
                e = min(s + sub, R0_chunk.size(0))
                D1[s:e] = ffn1.forward(R0_chunk[s:e])
            while sp_idx < sample_n and sample_pos[sp_idx] < end:
                gpos = sample_pos[sp_idx]
                lp = gpos - cursor   # index inside R0_chunk
                Xs[sp_idx] = torch.cat([R0_chunk[lp], D1[lp]])
                sp_idx += 1
            del R_full, R0_chunk, D1
            cursor = end
        print(f"  sampled {sp_idx:,} stacked residuals in {time.time()-t0:.1f}s")

        print(f"\n[layer2] k-means K={args.clusters2} iters={args.kmeans_iters}")
        mu2 = kmeans_lloyd(Xs, K=args.clusters2, n_iter=args.kmeans_iters, seed=0)
        del Xs

        print(f"\n[layer2] streaming counts2 over {train_n:,} train tokens")
        counts2 = accumulate_cluster_counts_stacked(train_ids, emb_dev, K_pos,
                                                    ffn1, mu2, V, chunk=chunk)
        cache2 = ARTIFACT / f"counts2_{args.tag}.pt"
        torch.save({"mu2": mu2.cpu(), "counts2": counts2.cpu(),
                    "K_pos": K_pos, "V": V,
                    "clusters1": args.clusters1, "clusters2": args.clusters2,
                    "top_M1": args.top_M1, "alpha1": args.alpha1,
                    "tau1": args.tau1, "train_tokens": train_n}, cache2)
        print(f"[save] layer-2 counts cache -> {cache2}")

    # --- Calibration on val ---
    print(f"\n[cal] precomputing stacked residuals on {val_n:,} val tokens")
    cal_R, cal_Y = collect_stacked_residuals(val_ids, emb_dev, K_pos, ffn1,
                                               chunk=chunk)
    print(f"  R1={tuple(cal_R.shape)}")

    alphas = sorted({args.alpha2, 0.01, 0.1})
    taus = [0.1, 0.3, 1.0]
    Ms = sorted({args.top_M2, 16, 64})
    best = {"ppl": float("inf"), "alpha": args.alpha2, "tau": 1.0,
            "gamma": 1.0, "top_M": args.top_M2}
    print(f"[cal] sweeping α∈{alphas} τ∈{taus} M∈{Ms}")
    for alpha in alphas:
        m2 = SparseMixtureClusterLM.from_counts(mu2, counts2, alpha=alpha,
                                                  V=V, K_pos=K_pos, d_emb=d)
        for M in Ms:
            for tau in taus:
                r = fast_ppl(cal_R, cal_Y, m2, tau=tau, gamma=1.0,
                             top_M=M, inner_batch=args.inner_batch)
                print(f"  α={alpha} τ={tau} M={M} γ=1 → "
                      f"PPL={r['ppl']:.2f}  top1={r['top1']*100:.2f}%")
                if r["ppl"] < best["ppl"]:
                    best = {"ppl": r["ppl"], "alpha": alpha, "tau": tau,
                            "gamma": 1.0, "top_M": M}
        del m2
        torch.cuda.empty_cache()
    bm = SparseMixtureClusterLM.from_counts(mu2, counts2, alpha=best["alpha"],
                                              V=V, K_pos=K_pos, d_emb=d)
    for gamma in [0.85, 0.95, 0.99]:
        r = fast_ppl(cal_R, cal_Y, bm, tau=best["tau"], gamma=gamma,
                     top_M=best["top_M"], inner_batch=args.inner_batch)
        print(f"  γ={gamma} → PPL={r['ppl']:.2f}")
        if r["ppl"] < best["ppl"]:
            best = {**best, "gamma": gamma}
    print(f"[cal] best α={best['alpha']} τ={best['tau']} M={best['top_M']} "
          f"γ={best['gamma']} PPL={best['ppl']:.2f}")
    model = bm
    del cal_R, cal_Y

    # --- Eval ---
    print("\n[eval] in-distribution")
    R_in, Y_in = collect_stacked_residuals(train_ids[:eval_n], emb_dev, K_pos,
                                             ffn1, chunk=chunk)
    sanity = fast_ppl(R_in, Y_in, model, tau=best["tau"], gamma=best["gamma"],
                       top_M=best["top_M"], inner_batch=args.inner_batch)
    print(f"  train(in): PPL={sanity['ppl']:.2f}  top1={sanity['top1']*100:.2f}%  "
          f"top5={sanity['top5']*100:.2f}%")
    del R_in, Y_in

    print("[eval] heldout")
    R_h, Y_h = collect_stacked_residuals(eval_ids, emb_dev, K_pos, ffn1,
                                           chunk=chunk)
    held = fast_ppl(R_h, Y_h, model, tau=best["tau"], gamma=best["gamma"],
                    top_M=best["top_M"], inner_batch=args.inner_batch)
    print(f"  heldout: PPL={held['ppl']:.2f}  top1={held['top1']*100:.2f}%  "
          f"top5={held['top5']*100:.2f}%")
    del R_h, Y_h

    results = {
        "model": "Compiled Wikitext LM v16 (stacked mixture, two FFN layers)",
        "K_pos": K_pos,
        "clusters1": args.clusters1, "clusters2": args.clusters2,
        "top_M1": args.top_M1, "top_M2": best["top_M"],
        "alpha1": args.alpha1, "alpha2": best["alpha"],
        "tau1": args.tau1, "tau2": best["tau"], "gamma2": best["gamma"],
        "train_tokens": train_n, "eval_tokens": eval_n,
        "V": V, "d_emb": d, "d_res0": d_res0, "d_res1": d_res1,
        "in_distribution": sanity, "heldout": held,
    }
    rp = ARTIFACT / f"eval_results_{args.tag}.json"
    with open(rp, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[save] -> {rp}")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
