"""
compile_wiki_lm_v17.py — Context-Augmented Mixture LM
=======================================================

The v14 bottleneck: positional residuals alone (K-shift concat) can't
distinguish "bank near river" from "bank near deposit" — same tokens in
the positional window produce identical residuals.

Fix: add PPMI-weighted context to the residual BEFORE it enters the
mixture routing. The cluster centroids and empirical distributions are
built on the augmented residual, so routing naturally separates different
contexts for the same positional signature.

Architecture:
    1. Same PPMI+SVD embeddings as v11-v16 (V=8000, d=256).
    2. Positional residual: r_pos = concat([emb[t], ..., emb[t-K_pos]])
    3. Context term: for each token t, compute a PPMI-weighted context
       vector ctx[t] from a wider window (±W_context):
         ctx[t] = mean(emb[w] * PPMI_cooc(t, w)) for w in [t-W, t+W]
       Where PPMI_cooc is a fast approximation: cosine similarity of
       PPMI embeddings (already encode co-occurrence).
    4. Augmented residual: r_aug = concat([r_pos, ctx[t]])
    5. k-means clusters over r_aug.
    6. Same mixture-of-cluster-LMs head as v14 (sparse top-M routing).

Everything is count-based. No gradient descent anywhere.

Context embedding computation (PPMI-based, not naive mean):
    - "river" and "bank" have high cosine similarity in PPMI space (co-occur).
    - "deposit" and "bank" also co-occur, but in a different way.
    - The context vector for "bank" will differ depending on whether
      nearby tokens are more "river-like" or "deposit-like" in PPMI space.
    - This is captured by: ctx[t] = Σ_w cos_sim(emb[t], emb[w]) · emb[w] / Σ_cos
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

# Reuse v13/v14 building blocks
sys.path.insert(0, str(Path(__file__).parent))
from compile_wiki_lm_v13 import (
    load_setup, load_or_build_tokens, kmeans_lloyd,
    accumulate_cluster_counts, collect_residuals,
    parse_size, DEVICE,
)

REPO = Path("/home/drawson/llm_decoupling")
ARTIFACT = REPO / "artifacts/compiled_wiki_lm_v17"
ARTIFACT.mkdir(parents=True, exist_ok=True)


def build_context_augmented_residual(
    emb: torch.Tensor, ids: torch.Tensor,
    K_pos: int, W_ctx: int, top_k_ctx: int,
) -> torch.Tensor:
    """
    Vectorized build, processed in GPU-friendly chunks.
    """
    N = ids.shape[0]
    d = emb.shape[1]
    win_size = 2 * W_ctx + 1
    chunk_tok = 100000  # process 100K tokens at a time
    d_res = (K_pos + 2) * d
    
    print(f"  [build] {N} tokens, K_pos={K_pos}, W_ctx={W_ctx}, top_k={top_k_ctx}, d_res={d_res}", flush=True)
    t0 = time.time()
    
    all_r = []
    for chunk_start in range(0, N, chunk_tok):
        chunk_end = min(chunk_start + chunk_tok, N)
        n_chunk = chunk_end - chunk_start
        
        # Load only this chunk's embeddings to GPU
        chunk_ids = ids[chunk_start:chunk_end]
        chunk_emb = emb[chunk_ids]  # (n_chunk, d)
        
        # ---- Positional residual for this chunk ----
        # Need to see K_pos tokens before the chunk
        r_parts = []
        for k in range(K_pos + 1):
            if k == 0:
                r_parts.append(chunk_emb)
            else:
                # Look back k tokens; pad with first token of available range
                indices = torch.arange(chunk_start - k, chunk_end - k, device=ids.device)
                indices[indices < chunk_start] = chunk_start  # pad
                r_parts.append(emb[ids[indices]])
        r_pos = torch.cat(r_parts, dim=1)  # (n_chunk, (K_pos+1)*d)
        
        # ---- Context vector for this chunk ----
        # Load context window embeddings: need W_ctx tokens before and after the chunk
        ctx_start = max(0, chunk_start - W_ctx)
        ctx_end = min(N, chunk_end + W_ctx)
        ctx_ids = ids[ctx_start:ctx_end]
        ctx_emb = emb[ctx_ids].to(DEVICE)  # (ctx_len, d)
        
        # For each position in the chunk, find its window in the context range
        pos_in_ctx = torch.arange(chunk_start - ctx_start, chunk_end - ctx_start, device=DEVICE)  # (n_chunk,)
        
        # Window indices: causal — only past + center, no future
        off = torch.arange(-W_ctx, 1, device=DEVICE)  # [-W_ctx, ..., -1, 0]
        win_idx = pos_in_ctx.unsqueeze(1) + off.unsqueeze(0)  # (n_chunk, win_size)
        win_idx = win_idx.clamp(0, ctx_emb.shape[0] - 1)  # boundary pad
        
        # Gather window embeddings: raw (not normed — no similarity computation needed)
        win_emb = ctx_emb[win_idx]  # (n_chunk, win_size, d)
        
        # ---- Position-weighted contrastive context ----
        # ctx = weighted mean of past tokens, weight ∝ 1/distance
        # Uses raw embeddings (unnormalized) since we build residuals
        if top_k_ctx > 0:
            # Build position weights: 1/(|offset|) for past tokens
            pos_weights = 1.0 / (torch.arange(1, W_ctx + 1, device=DEVICE).float())  # (W_ctx,)
            pos_weights = pos_weights / pos_weights.sum()  # normalize to sum=1
            
            # Gather past token embeddings: (n_chunk, W_ctx, d)
            past_embs = win_emb[:, :W_ctx]  # first W_ctx positions are [-W, ..., -1]
            
            # Weighted mean over past context
            ctx_vecs = (pos_weights.view(1, -1, 1) * past_embs).sum(dim=1)  # (n_chunk, d)
        else:
            ctx_vecs = chunk_emb
        
        # ---- Augmented residual — contrastive context ----
        # ctx_contrastive = ctx - emb[t] encodes what's different about surroundings
        r_aug = torch.cat([r_pos, ctx_vecs - chunk_emb], dim=1)
        all_r.append(r_aug.cpu())
        
        # Free GPU memory
        del chunk_emb, ctx_emb, ctx_vecs, r_pos, r_aug
        
        if (chunk_start // chunk_tok) % 10 == 0:
            print(f"    chunk {chunk_start//chunk_tok}: {chunk_end}/{N} ({time.time()-t0:.1f}s)", flush=True)
    
    print(f"    build done: {time.time()-t0:.1f}s", flush=True)
    return torch.cat(all_r, dim=0)


class SparseMixtureClusterLM:
    """Same as v14 but built on augmented residuals."""

    def __init__(self, mu: torch.Tensor, log_p_cluster: torch.Tensor,
                 log_p_uni: torch.Tensor, K_pos: int, V: int, d_emb: int,
                 top_M: int = 16):
        self.mu = mu                          # (K_cl, d_res)
        self.log_p_cluster = log_p_cluster    # (K_cl, V)
        self.log_p_uni = log_p_uni            # (V,)
        self.K_pos = K_pos
        self.V = V
        self.d_emb = d_emb
        self.top_M = top_M
        self._mu_sq = (mu * mu).sum(dim=1)

    @classmethod
    def from_counts(cls, mu, counts, alpha, V, K_pos, d_emb, top_M=16):
        K_cl = counts.size(0)
        device = counts.device
        uni = counts.sum(dim=0)
        uni_p = (uni + alpha) / (uni.sum() + alpha * V)
        log_uni = torch.log(uni_p.clamp_min(1e-30))
        log_p = torch.empty((K_cl, V), device=device, dtype=torch.float16)
        for k in range(K_cl):
            ck = counts[k]
            pk = (ck + alpha) / (ck.sum() + alpha * V)
            log_p[k] = torch.log(pk.clamp_min(1e-30)).to(torch.float16)
        return cls(mu, log_p, log_uni, K_pos, V, d_emb, top_M)

    @torch.no_grad()
    def forward(self, r: torch.Tensor, gamma: float, tau: float) -> torch.Tensor:
        """r: (B, d_res) -> logprobs: (B,)  [cross-entropy per token]"""
        device = r.device
        mu_sq = self._mu_sq.to(device)
        mu = self.mu.to(device)
        log_p_cluster = self.log_p_cluster.to(device)
        log_p_uni = self.log_p_uni.to(device)
        
        r_sq = (r * r).sum(dim=1, keepdim=True)  # (B, 1)
        d2 = r_sq + mu_sq.unsqueeze(0) - 2 * (r @ mu.T)  # (B, K_cl)
        
        if self.top_M and self.top_M < mu.shape[0]:
            _, top_idx = d2.topk(self.top_M, dim=1, largest=False)
            d2_top = d2.gather(1, top_idx)
            log_pi = F.log_softmax(-d2_top / tau, dim=1)  # log-probability
            log_p_top = log_p_cluster[top_idx]  # (B, M, V)
            log_mix = torch.logsumexp(
                log_pi.unsqueeze(2) + log_p_top, dim=1
            ).float()  # (B, V)
        else:
            log_pi = F.log_softmax(-d2 / tau, dim=1)  # log-probability
            log_mix = torch.logsumexp(
                log_pi.unsqueeze(2) + log_p_cluster.unsqueeze(0), dim=1
            ).float()
        
        if gamma < 1.0:
            log_p = torch.logaddexp(
                math.log(gamma) + log_mix,
                math.log(1 - gamma) + log_p_uni.float().unsqueeze(0)
            )
        else:
            log_p = log_mix
        
        return log_p

    def eval_ppl(self, r: torch.Tensor, ids_next: torch.Tensor, gamma: float, tau: float):
        """r: (N, d_res), ids_next: (N,) → PPL, top-1, top-5. Chunked to limit GPU memory."""
        total_nll = 0.0
        total_correct_1 = 0
        total_correct_5 = 0
        N = r.shape[0]
        chunk = 5000  # small chunks to keep GPU memory manageable
        for start in range(0, N, chunk):
            end = min(start + chunk, N)
            r_chunk = r[start:end]
            targets = ids_next[start:end]
            device = self.mu.device
            r_chunk = r_chunk.to(device)
            targets = targets.to(device)
            
            log_p = self.forward(r_chunk, gamma, tau)  # (chunk, V)
            nll = F.nll_loss(log_p, targets, reduction='sum').item()
            total_nll += nll
            
            _, top_idx = log_p.topk(5, dim=1)
            total_correct_1 += (top_idx[:, 0] == targets).float().sum().item()
            total_correct_5 += (top_idx == targets.unsqueeze(1)).any(dim=1).float().sum().item()
        
        ppl = math.exp(total_nll / N)
        top1 = total_correct_1 / N
        top5 = total_correct_5 / N
        return ppl, top1, top5


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-tokens", type=str, default="20M",
                        help="Training tokens (e.g. 20M, 50M)")
    parser.add_argument("--val-tokens", type=str, default="200K")
    parser.add_argument("--eval-tokens", type=str, default="300K")
    parser.add_argument("--K-pos", type=int, default=2,
                        help="Positional context width")
    parser.add_argument("--W-ctx", type=int, default=10,
                        help="Context window width for PPMI-weighted context")
    parser.add_argument("--top-k-ctx", type=int, default=5,
                        help="Top-k context tokens to use in weighting")
    parser.add_argument("--K-clusters", type=int, default=8192,
                        help="Number of k-means clusters")
    parser.add_argument("--top-M", type=int, default=16,
                        help="Sparse routing: top-M clusters at inference")
    parser.add_argument("--alpha", type=float, default=0.01,
                        help="Laplace smoothing")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    train_n = parse_size(args.train_tokens)
    val_n = parse_size(args.val_tokens)
    eval_n = parse_size(args.eval_tokens)
    
    # Load embeddings and data
    bpe, vocab, tok2id, bpe_to_lm, emb, V, d = load_setup()
    emb = emb.to(DEVICE)
    
    # Load tokens
    ids = load_or_build_tokens(bpe, bpe_to_lm, V)
    total_needed = train_n + val_n + eval_n
    assert ids.shape[0] >= total_needed, f"only {ids.shape[0]} tokens, need {total_needed}"
    ids = ids[:total_needed].to(DEVICE)  # only needed slice on GPU
    
    # Split: train, val, eval (all contiguous, no overlap between splits)
    i_train = ids[:train_n]
    i_val = ids[train_n:train_n + val_n]
    i_eval = ids[train_n + val_n:train_n + val_n + eval_n]
    print(f"[data] train={i_train.shape[0]}, val={i_val.shape[0]}, eval={i_eval.shape[0]}")
    
    # Build augmented residuals
    print(f"[build] Building context-augmented residuals (K_pos={args.K_pos}, W_ctx={args.W_ctx})...")
    t0 = time.time()
    r_train = build_context_augmented_residual(emb, i_train, args.K_pos, args.W_ctx, args.top_k_ctx)
    r_val = build_context_augmented_residual(emb, i_val, args.K_pos, args.W_ctx, args.top_k_ctx)
    r_eval = build_context_augmented_residual(emb, i_eval, args.K_pos, args.W_ctx, args.top_k_ctx)
    d_res = r_train.shape[1]
    print(f"  d_res={d_res} (positional slots={args.K_pos+1}, +context={d})")
    print(f"  build time: {time.time()-t0:.1f}s")
    
    # K-means clustering on training residuals
    print(f"[kmeans] Clustering {args.K_clusters} centroids...")
    t0 = time.time()
    sample_idx = torch.randperm(len(r_train))[:min(30000, len(r_train))]  # smaller sample for large K_cl
    r_sample = r_train[sample_idx].to(DEVICE)
    mu = kmeans_lloyd(r_sample, args.K_clusters, n_iter=15, seed=args.seed)
    print(f"  k-means time: {time.time()-t0:.1f}s")
    
    # Accumulate cluster counts using augmented residuals
    print(f"[counts] Accumulating per-cluster token counts (d_res={d_res}, K_cl={args.K_clusters})...", flush=True)
    t0 = time.time()
    mu_sq = (mu * mu).sum(dim=1)  # (K_cl,)
    counts = torch.zeros(args.K_clusters, V, dtype=torch.float32, device=DEVICE)
    
    chunk = 50000
    for start in range(0, len(r_train) - 1, chunk):
        end = min(start + chunk, len(r_train) - 1)
        r_chunk = r_train[start:end].to(DEVICE)
        # Assign each residual to nearest centroid
        d2 = (r_chunk * r_chunk).sum(dim=1, keepdim=True) + mu_sq.unsqueeze(0) - 2 * (r_chunk @ mu.T)
        _, assignments = d2.min(dim=1)  # (chunk,)
        # Increment counts: counts[cluster, next_token] += 1
        targets = i_train[1 + start:1 + end]
        counts.index_put_((assignments, targets), torch.ones_like(assignments, dtype=torch.float32), accumulate=True)
        
        if start % (chunk * 10) == 0:
            print(f"    {start}/{len(r_train)} ({time.time()-t0:.1f}s)", flush=True)
    
    print(f"  counts done: {time.time()-t0:.1f}s, shape={tuple(counts.shape)}", flush=True)
    
    # Build LM
    print(f"[build] Building mixture LM...")
    lm = SparseMixtureClusterLM.from_counts(mu, counts, args.alpha, V, args.K_pos, d_emb=d, top_M=args.top_M)
    
    # Calibrate τ, γ on validation set
    print(f"[calib] Sweeping τ, γ on validation ({i_val.shape[0]} tokens)...", flush=True)
    best_ppl = float('inf')
    best_params = (0.3, 1.0)
    for tau in [0.01, 0.05, 0.1, 0.3, 0.5, 1.0, 2.0]:
        for gamma in [0.9, 0.95, 0.99, 1.0]:
            ppl, _, _ = lm.eval_ppl(r_val[:-1], i_val[1:], gamma, tau)
            if ppl < best_ppl:
                best_ppl = ppl
                best_params = (tau, gamma)
    tau, gamma = best_params
    print(f"  best τ={tau}, γ={gamma}, val PPL={best_ppl:.2f}")
    
    # Final evaluation
    print(f"[eval] Evaluating on held-out ({i_eval.shape[0]} tokens)...", flush=True)
    ppl, top1, top5 = lm.eval_ppl(r_eval[:-1], i_eval[1:], gamma, tau)
    
    # Train PPL
    ppl_train, t1_train, t5_train = lm.eval_ppl(r_train[:-1], i_train[1:], gamma, tau)
    
    print(f"\n{'='*60}")
    print(f"  v17 — Context-Augmented Mixture LM")
    print(f"  K_pos={args.K_pos}, W_ctx={args.W_ctx}, K_clusters={args.K_clusters}")
    print(f"  d_res={d_res}, top-M={args.top_M}")
    print(f"  Train PPL: {ppl_train:.2f} (top-1: {t1_train*100:.2f}%, top-5: {t5_train*100:.2f}%)")
    print(f"  Heldout PPL: {ppl:.2f} (top-1: {top1*100:.2f}%, top-5: {top5*100:.2f}%)")
    print(f"  τ={tau}, γ={gamma}, α={args.alpha}")
    print(f"{'='*60}")
    
    # Save
    tag = f"k{args.K_pos}_c{args.K_clusters}_w{args.W_ctx}"
    result = {
        "K_pos": args.K_pos, "W_ctx": args.W_ctx, "K_clusters": args.K_clusters,
        "d_res": d_res, "tau": tau, "gamma": gamma, "alpha": args.alpha,
        "train_ppl": ppl_train, "train_top1": t1_train, "train_top5": t5_train,
        "heldout_ppl": ppl, "heldout_top1": top1, "heldout_top5": top5,
        "train_tokens": train_n, "val_tokens": val_n, "eval_tokens": eval_n,
    }
    with open(ARTIFACT / f"eval_results_{tag}.json", "w") as f:
        json.dump(result, f, indent=2)
    
    torch.save({
        "mu": mu.cpu(), "counts": counts.cpu(),
        "log_p_cluster": lm.log_p_cluster.cpu(),
        "log_p_uni": lm.log_p_uni.cpu(),
        "K_pos": args.K_pos, "V": V, "d_emb": d, "top_M": args.top_M,
        "tau": tau, "gamma": gamma, "alpha": args.alpha,
    }, ARTIFACT / f"compiled_lm_{tag}.pt")
    
    print(f"  saved to {ARTIFACT}")


if __name__ == "__main__":
    main()
