"""
compile_wiki_lm_v13.py — Mixture-of-Cluster LM (no ridge head, no linear head)
=============================================================================

Lesson from v12: a ridge LM head is a linear function of the residual r.
Adding more *linear* features (attention mixer, FFN with unit-norm values)
that are themselves linear functions of nearby embeddings is REDUNDANT —
the ridge head can already produce any linear combination of features it
sees, so adding more linear features doesn't lower training MSE much, and
held-out PPL stays flat (v11=2368, v12=2369, ~identical).

The fix: replace the linear LM head with a real nonlinearity that the ridge
head CANNOT replicate. Concretely, an explicit Gaussian mixture over
per-cluster empirical next-token distributions:

    P(y | r) = Σ_k π_k(r) · p_k(y)
        with π_k(r) = softmax(-||r - μ_k||² / τ)
        and  p_k(y) = (count_k(y) + α) / (count_k(·) + α·V)

This is the classic "key-value memory" view of an FFN (Geva et al. 2021),
but realised as the *whole* LM head — not a residual contribution. The
softmax over distances is a true nonlinearity (it can express "if r looks
like cluster 7 use distribution_7, else use distribution_3"), and the
per-cluster distributions p_k(y) are real probability vectors, not raw
logits that need a temperature.

Compilation steps (no SGD, all closed-form / counting):
    1. Build the residual r[t] = concat([emb[t], emb[t-1], ..., emb[t-K]]).
       This is the same v11 attention forward — the K-shift trick.
    2. Cluster a sample of r[t] using mini-batch k-means (deterministic,
       no SGD, just nearest-centroid + mean updates).
    3. For each token t in the training corpus, hard-assign r[t] to its
       nearest cluster k* and accumulate counts: cluster_token_counts[k*, ids[t+1]] += 1.
    4. Laplace-smooth to get p_k(y).
    5. (Optional) global backoff:
           P(y | r) = γ · mixture(y | r) + (1-γ) · unigram(y)
       Lets the model handle clusters with few hits without going to zero.

Inference:
    For r[t]:
        d_k = ||r[t] - μ_k||²
        π   = softmax(-d/τ)
        p   = π @ p_cluster        # (V,)
        logp = log(γ p + (1-γ) p_uni)

There's a SINGLE knob: τ (softness of cluster routing). γ is sweepable too.

Memory: K_clusters × V × fp16 + K_clusters × d_res × fp32.
At K=4096, V=8000, d_res=1024 that's 65 MB + 16 MB. Trivial.

Architectural status:
    * Every weight comes from a deterministic algorithm on the corpus.
    * No backprop. No SGD. No pretrained model.
    * K-means initialises with k-means++ on a sample.
    * Even the temperature τ is selected by simple grid search on a small
      train slice (one scalar = identical in spirit to a learned bias).

Outputs:
    artifacts/compiled_wiki_lm_v13/compiled_lm_<tag>.pt
    artifacts/compiled_wiki_lm_v13/eval_results_<tag>.json
"""
from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

REPO = Path("/home/drawson/llm_decoupling")
ARTIFACT = REPO / "artifacts/compiled_wiki_lm_v13"
ARTIFACT.mkdir(parents=True, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

V5_STATE = REPO / "artifacts/compiled_wiki_lm_v5/compiled_lm.pt"
V5_META = REPO / "artifacts/compiled_wiki_lm_v5/meta.pkl"
BPE_PATH = REPO / "artifacts/bpe_wiki/tokenizer.json"
CORPUS = REPO / "corpora/wikitext103.txt"
V11_TOKEN_CACHE = REPO / "artifacts/compiled_wiki_lm_v11/cache_lm_ids.pt"


def parse_size(s: str) -> int:
    s = s.strip().lower()
    if s.endswith("k"): return int(float(s[:-1]) * 1e3)
    if s.endswith("m"): return int(float(s[:-1]) * 1e6)
    if s.endswith("g") or s.endswith("b"): return int(float(s[:-1]) * 1e9)
    return int(s)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_setup():
    print("[setup] Loading BPE tokenizer, vocab, PPMI embeddings ...")
    bpe = Tokenizer.from_file(str(BPE_PATH))
    meta = pickle.load(open(V5_META, "rb"))
    vocab = meta["vocab"]
    tok2id = meta["tok2id"]
    bpe_to_lm = meta["bpe_to_lm"]
    state = torch.load(str(V5_STATE), map_location="cpu", weights_only=False)
    emb = state["emb"].float()
    V, d = emb.shape
    print(f"  V={V}, d={d}")
    return bpe, vocab, tok2id, bpe_to_lm, emb, V, d


def load_or_build_tokens(bpe: Tokenizer, bpe_to_lm: dict, V: int) -> torch.Tensor:
    if V11_TOKEN_CACHE.exists():
        print(f"[tok] Reusing v11 token cache: {V11_TOKEN_CACHE}")
        ids = torch.load(str(V11_TOKEN_CACHE), weights_only=False)
        print(f"  loaded {len(ids):,} LM tokens.")
        return ids
    raise FileNotFoundError(
        f"Expected v11 token cache at {V11_TOKEN_CACHE}. Run v11 first.")


# ---------------------------------------------------------------------------
# Residual builder (same as v11 attention forward)
# ---------------------------------------------------------------------------

def build_residual(ids_window: torch.Tensor, emb: torch.Tensor,
                   K: int) -> torch.Tensor:
    """ids_window: (W,) LongTensor. emb: (V, d). Returns (W, (K+1)*d).
    Positions < K have zero-padded prefix slots."""
    X = emb[ids_window]                 # (W, d)
    slots = [X]
    for k in range(1, K + 1):
        s = torch.zeros_like(X)
        if k < X.size(0):
            s[k:] = X[:-k]
        slots.append(s)
    return torch.cat(slots, dim=-1)     # (W, (K+1)*d)


# ---------------------------------------------------------------------------
# K-means (no SGD — Lloyd's algorithm)
# ---------------------------------------------------------------------------

def kmeans_plusplus_init(X: torch.Tensor, K: int, seed: int = 0) -> torch.Tensor:
    """k-means++ initialisation on GPU. X: (N, d). Returns (K, d) centroids."""
    g = torch.Generator(device=X.device).manual_seed(seed)
    N, d = X.shape
    centroids = torch.empty(K, d, device=X.device, dtype=X.dtype)
    first = torch.randint(0, N, (1,), generator=g, device=X.device).item()
    centroids[0] = X[first]
    # Chunked distance computation to avoid materialising (N, d) deltas
    min_dist2 = torch.empty(N, device=X.device, dtype=X.dtype)
    chunk = max(1, min(N, 50_000_000 // max(d, 1)))
    for s in range(0, N, chunk):
        e = min(s + chunk, N)
        min_dist2[s:e] = ((X[s:e] - centroids[0])**2).sum(dim=1)
    for k in range(1, K):
        probs = min_dist2 / min_dist2.sum().clamp_min(1e-12)
        idx = torch.multinomial(probs, 1, generator=g).item()
        centroids[k] = X[idx]
        for s in range(0, N, chunk):
            e = min(s + chunk, N)
            new_d2 = ((X[s:e] - centroids[k])**2).sum(dim=1)
            torch.minimum(min_dist2[s:e], new_d2, out=min_dist2[s:e])
    return centroids


def kmeans_lloyd(X: torch.Tensor, K: int, n_iter: int = 20,
                 batch: int = 8192, seed: int = 0) -> torch.Tensor:
    """Lloyd's algorithm on (N, d). Returns (K, d) centroids."""
    N, d = X.shape
    print(f"[kmeans] N={N:,} d={d} K={K} iters={n_iter}")
    t0 = time.time()
    mu = kmeans_plusplus_init(X, K, seed=seed)
    print(f"  init in {time.time()-t0:.1f}s")
    for it in range(n_iter):
        t_it = time.time()
        # assign all points to nearest centroid (chunked)
        assigns = torch.empty(N, dtype=torch.long, device=X.device)
        mu_sq = (mu * mu).sum(dim=1)                    # (K,)
        for s in range(0, N, batch):
            e = min(s + batch, N)
            Xb = X[s:e]                                  # (b, d)
            d2 = (Xb * Xb).sum(dim=1, keepdim=True) - 2 * Xb @ mu.t() + mu_sq[None]
            assigns[s:e] = d2.argmin(dim=1)
        # update centroids: mean of assigned points
        new_mu = torch.zeros_like(mu)
        counts = torch.zeros(K, device=X.device, dtype=torch.long)
        new_mu.index_add_(0, assigns, X)
        counts.index_add_(0, assigns, torch.ones(N, device=X.device, dtype=torch.long))
        empty = counts == 0
        n_empty = int(empty.sum().item())
        if n_empty > 0:
            # random reinit of empty clusters from data points
            g = torch.Generator(device=X.device).manual_seed(seed + 1000 + it)
            ridx = torch.randint(0, N, (n_empty,), generator=g, device=X.device)
            new_mu[empty] = X[ridx]
            counts[empty] = 1
        new_mu /= counts.unsqueeze(1).to(new_mu.dtype)
        shift = (new_mu - mu).norm(dim=1).mean().item()
        mu = new_mu
        print(f"  iter {it}: shift={shift:.4f}  empty={n_empty}  "
              f"({time.time()-t_it:.1f}s)")
        if shift < 1e-4:
            break
    print(f"[kmeans] total {time.time()-t0:.1f}s")
    return mu


# ---------------------------------------------------------------------------
# Cluster-conditional next-token distributions
# ---------------------------------------------------------------------------

def accumulate_cluster_counts(ids: torch.Tensor, emb: torch.Tensor, K_pos: int,
                              mu: torch.Tensor, V: int,
                              chunk: int = 100_000) -> torch.Tensor:
    """Stream over ids: for each position t in [K_pos, N-2], assign r[t] to
    nearest centroid k* and increment counts[k*, ids[t+1]].

    Returns counts: (K_clusters, V) fp32.
    """
    N = ids.size(0)
    K_cl = mu.size(0)
    mu_sq = (mu * mu).sum(dim=1)                        # (K_cl,)
    counts = torch.zeros(K_cl, V, dtype=torch.float32, device=DEVICE)

    print(f"[counts] streaming over N={N:,} ids, K_cl={K_cl}, chunk={chunk:,}")
    t0 = time.time()
    cursor = K_pos
    while cursor < N - 1:
        end = min(cursor + chunk, N - 1)
        prefix = cursor - K_pos
        window = ids[prefix:end + 1].to(DEVICE, non_blocking=True)
        R_full = build_residual(window, emb, K_pos)        # (Wsz, d_res)
        local_lo = K_pos
        local_hi = K_pos + (end - cursor)
        R = R_full[local_lo:local_hi]                       # (cnt, d_res)
        Y = window[local_lo + 1:local_hi + 1].long()        # (cnt,)
        # nearest centroid: argmin ||R - mu||²  (chunked over R rows to
        # avoid materialising the full (cnt, K_cl) distance matrix)
        cnt = R.size(0)
        sub = max(1, min(cnt, 500_000_000 // max(K_cl * 4, 1)))
        assigns = torch.empty(cnt, dtype=torch.long, device=DEVICE)
        for s in range(0, cnt, sub):
            e = min(s + sub, cnt)
            Rb = R[s:e]
            d2 = (Rb * Rb).sum(dim=1, keepdim=True) - 2 * Rb @ mu.t() + mu_sq[None]
            assigns[s:e] = d2.argmin(dim=1)
            del d2
        # accumulate counts via scatter-add on flat index
        flat = assigns * V + Y
        counts.view(-1).index_add_(0, flat,
                                   torch.ones(flat.size(0), device=DEVICE))
        cursor = end
        if cursor % (chunk * 5) == 0 or cursor >= N - 1:
            rate = (cursor - K_pos) / max(time.time() - t0, 1e-6)
            print(f"  ... {cursor:,}/{N:,} ({rate/1e6:.2f}M/s)")
    print(f"[counts] done in {time.time()-t0:.1f}s. total={counts.sum().item():.0f}")
    return counts


# ---------------------------------------------------------------------------
# Mixture LM
# ---------------------------------------------------------------------------

class MixtureClusterLM:
    """The compiled LM: cluster centroids + cluster-conditional distributions.

        P(y | r) = γ · Σ_k π_k(r) · p_k(y) + (1-γ) · p_uni(y)
        π_k(r) = softmax(-||r-μ_k||² / τ)
    """

    def __init__(self, mu: torch.Tensor, log_p_cluster: torch.Tensor,
                 log_p_uni: torch.Tensor, K_pos: int, V: int, d_emb: int):
        self.mu = mu                          # (K_cl, d_res)
        self.log_p_cluster = log_p_cluster    # (K_cl, V) fp32 log-probs
        self.log_p_uni = log_p_uni            # (V,) fp32
        self.K_pos = K_pos
        self.V = V
        self.d_emb = d_emb
        self._mu_sq = (mu * mu).sum(dim=1)   # (K_cl,)

    @classmethod
    def from_counts(cls, mu: torch.Tensor, counts: torch.Tensor,
                    alpha: float, V: int, K_pos: int, d_emb: int):
        # Laplace-smoothed per-cluster distribution
        sm = counts + alpha
        p = sm / sm.sum(dim=1, keepdim=True)
        log_p = torch.log(p.clamp_min(1e-30))
        # unigram from total counts (also laplace)
        uni = counts.sum(dim=0)
        uni_p = (uni + alpha) / (uni.sum() + alpha * V)
        log_uni = torch.log(uni_p.clamp_min(1e-30))
        return cls(mu, log_p, log_uni, K_pos, V, d_emb)

    def log_probs(self, R: torch.Tensor, tau: float, gamma: float) -> torch.Tensor:
        """R: (B, d_res). Returns (B, V) log-probs."""
        # squared distance to each centroid
        d2 = (R * R).sum(dim=1, keepdim=True) - 2 * R @ self.mu.t() + self._mu_sq[None]
        # routing log-weights
        log_pi = F.log_softmax(-d2 / tau, dim=-1)        # (B, K_cl)
        # log-mixture: logsumexp_k log_pi_k + log_p_k(y)
        # = logsumexp over K of (log_pi[:,k,None] + log_p_cluster[k,:])
        # implement as: M = log_pi.unsqueeze(2) + log_p_cluster.unsqueeze(0)
        # but K_cl*V*B is too big — do in chunks over k blocks
        B, K_cl = log_pi.shape
        # output buffer
        running_max = None
        running_sum = None
        # block size: B * kchunk * V * 4 bytes must stay under ~500 MB
        kchunk = max(1, min(256, 1 + (500_000_000 // max(B * self.V * 4, 1))))
        for ks in range(0, K_cl, kchunk):
            ke = min(ks + kchunk, K_cl)
            block = log_pi[:, ks:ke].unsqueeze(2) + self.log_p_cluster[ks:ke].unsqueeze(0)
            # block: (B, ke-ks, V)
            block_max = block.max(dim=1).values   # (B, V)
            block_sum = torch.exp(block - block_max.unsqueeze(1)).sum(dim=1)  # (B, V)
            if running_max is None:
                running_max = block_max
                running_sum = block_sum
            else:
                new_max = torch.maximum(running_max, block_max)
                running_sum = (running_sum * torch.exp(running_max - new_max)
                               + block_sum * torch.exp(block_max - new_max))
                running_max = new_max
        log_mix = running_max + torch.log(running_sum.clamp_min(1e-30))
        # blend with unigram backoff in probability space:
        # log( gamma * exp(log_mix) + (1-gamma) * exp(log_uni) )
        if gamma >= 1.0 - 1e-9:
            return log_mix
        log_g = math.log(gamma)
        log_1mg = math.log(1.0 - gamma)
        a = log_g + log_mix
        b = log_1mg + self.log_p_uni[None].expand_as(log_mix)
        m = torch.maximum(a, b)
        return m + torch.log(torch.exp(a - m) + torch.exp(b - m))


def collect_residuals(ids: torch.Tensor, emb: torch.Tensor, K_pos: int,
                      chunk: int = 100_000) -> tuple[torch.Tensor, torch.Tensor]:
    """Materialise (R, Y) for all valid positions. Used for fast calibration."""
    N = ids.size(0)
    Rs = []
    Ys = []
    cursor = K_pos
    while cursor < N - 1:
        end = min(cursor + chunk, N - 1)
        prefix = cursor - K_pos
        window = ids[prefix:end + 1].to(DEVICE, non_blocking=True)
        R_full = build_residual(window, emb, K_pos)
        local_lo = K_pos
        local_hi = K_pos + (end - cursor)
        Rs.append(R_full[local_lo:local_hi].clone())
        Ys.append(window[local_lo + 1:local_hi + 1].long().clone())
        cursor = end
    return torch.cat(Rs, dim=0), torch.cat(Ys, dim=0)


def fast_ppl(R: torch.Tensor, Y: torch.Tensor, model: MixtureClusterLM,
             tau: float, gamma: float, inner_batch: int = 256) -> dict:
    """Evaluate a (R, Y) cache against a model."""
    nll_sum = 0.0
    top1 = 0
    top5 = 0
    count = 0
    for i in range(0, R.size(0), inner_batch):
        Rb = R[i:i + inner_batch]
        Yb = Y[i:i + inner_batch]
        logp = model.log_probs(Rb, tau=tau, gamma=gamma)
        nll_sum += -logp.gather(1, Yb.unsqueeze(1)).squeeze(1).sum().item()
        top5_idx = logp.topk(5, dim=-1).indices
        top1 += (top5_idx[:, 0] == Yb).sum().item()
        top5 += (top5_idx == Yb.unsqueeze(1)).any(dim=1).sum().item()
        count += Rb.size(0)
    avg = nll_sum / count
    return {"count": count, "ppl": math.exp(avg), "avg_nll": avg,
            "top1": top1 / count, "top5": top5 / count,
            "tau": tau, "gamma": gamma}


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(model: MixtureClusterLM, ids: torch.Tensor, emb: torch.Tensor,
             chunk: int = 100_000, inner_batch: int = 1024,
             tau: float = 1.0, gamma: float = 1.0,
             label: str = "eval") -> dict:
    K_pos = model.K_pos
    N = ids.size(0)
    nll_sum = 0.0
    top1 = 0
    top5 = 0
    count = 0
    cursor = K_pos
    t0 = time.time()
    while cursor < N - 1:
        end = min(cursor + chunk, N - 1)
        prefix = cursor - K_pos
        window = ids[prefix:end + 1].to(DEVICE, non_blocking=True)
        R_full = build_residual(window, emb, K_pos)
        local_lo = K_pos
        local_hi = K_pos + (end - cursor)
        R = R_full[local_lo:local_hi]
        Y = window[local_lo + 1:local_hi + 1].long()
        for i in range(0, R.size(0), inner_batch):
            Rb = R[i:i + inner_batch]
            Yb = Y[i:i + inner_batch]
            logp = model.log_probs(Rb, tau=tau, gamma=gamma)    # (b, V)
            nll = -logp.gather(1, Yb.unsqueeze(1)).squeeze(1)
            nll_sum += nll.sum().item()
            top5_idx = logp.topk(5, dim=-1).indices
            top1 += (top5_idx[:, 0] == Yb).sum().item()
            top5 += (top5_idx == Yb.unsqueeze(1)).any(dim=1).sum().item()
            count += Rb.size(0)
            del logp, nll
        cursor = end
    avg_nll = nll_sum / count
    ppl = math.exp(avg_nll)
    print(f"  {label}: τ={tau} γ={gamma}  tokens={count:,}  "
          f"PPL={ppl:,.2f}  top1={top1/count*100:.2f}%  "
          f"top5={top5/count*100:.2f}%  ({time.time()-t0:.1f}s)")
    return {"count": count, "ppl": ppl,
            "top1": top1 / count, "top5": top5 / count,
            "avg_nll": avg_nll, "tau": tau, "gamma": gamma}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--K", type=int, default=3, help="positional lookback")
    p.add_argument("--clusters", type=int, default=4096)
    p.add_argument("--kmeans-sample", type=str, default="500K")
    p.add_argument("--kmeans-iters", type=int, default=15)
    p.add_argument("--alpha", type=float, default=0.01,
                   help="Laplace smoothing for per-cluster distributions")
    p.add_argument("--train-tokens", type=str, default="5M")
    p.add_argument("--val-tokens", type=str, default="200K",
                   help="held-out validation slice for α/τ/γ calibration")
    p.add_argument("--eval-tokens", type=str, default="500K")
    p.add_argument("--chunk", type=str, default="100K")
    p.add_argument("--inner-batch", type=int, default=1024)
    p.add_argument("--tag", type=str, default="default")
    args = p.parse_args()

    train_n = parse_size(args.train_tokens)
    eval_n = parse_size(args.eval_tokens)
    chunk = parse_size(args.chunk)
    km_sample = parse_size(args.kmeans_sample)

    bpe, vocab, tok2id, bpe_to_lm, emb, V, d = load_setup()
    ids = load_or_build_tokens(bpe, bpe_to_lm, V)
    N = ids.size(0)
    val_n = parse_size(args.val_tokens)
    if train_n + val_n + eval_n > N:
        print(f"WARN: train+val+eval > corpus; shrinking train")
        train_n = max(N - val_n - eval_n, N // 2)
    # Split: [train_n compile] | [val_n calibration] | ... | [eval_n heldout]
    train_ids = ids[:train_n]
    val_ids = ids[train_n: train_n + val_n]
    eval_ids = ids[-eval_n:]
    print(f"[split] train={train_n:,}  val={val_n:,}  eval={eval_n:,}")

    emb_dev = emb.to(DEVICE)
    K_pos = args.K
    d_res = (K_pos + 1) * d

    # --- 1) Sample residuals for k-means
    print(f"\n[kmeans-data] sampling {km_sample:,} residuals from train")
    rng = np.random.RandomState(0)
    valid_lo, valid_hi = K_pos, train_n - 1
    n_avail = valid_hi - valid_lo
    sample_n = min(km_sample, n_avail)
    sample_pos = np.sort(rng.choice(n_avail, sample_n, replace=False)) + valid_lo
    # Build the sampled residuals chunkwise — pull each contiguous run together
    # to avoid one-token-at-a-time GPU calls.
    Xs = torch.empty(sample_n, d_res, device=DEVICE, dtype=torch.float32)
    # Build the full train residual on the fly in chunks and gather
    cursor = K_pos
    sp_idx = 0
    chunk_pos = chunk
    while cursor < train_n - 1 and sp_idx < sample_n:
        end = min(cursor + chunk_pos, train_n - 1)
        prefix = cursor - K_pos
        window = train_ids[prefix:end + 1].to(DEVICE, non_blocking=True)
        R_full = build_residual(window, emb_dev, K_pos)
        local_lo = K_pos
        local_hi = K_pos + (end - cursor)
        # global positions in [cursor, end). Find which sample positions fall here.
        while sp_idx < sample_n and sample_pos[sp_idx] < end:
            gpos = sample_pos[sp_idx]
            lp = local_lo + (gpos - cursor)
            Xs[sp_idx] = R_full[lp]
            sp_idx += 1
        cursor = end
    print(f"  collected {sp_idx:,} residual samples")

    # --- 2) K-means
    mu = kmeans_lloyd(Xs, K=args.clusters, n_iter=args.kmeans_iters, seed=0)
    del Xs

    # --- 3) Accumulate cluster-conditional counts on FULL train
    counts = accumulate_cluster_counts(train_ids, emb_dev, K_pos, mu, V,
                                        chunk=chunk)

    # --- 4) Build mixture LM (sweep alpha to find best smoothing on VAL set)
    print(f"\n[cal] precomputing residuals on {val_n:,} val tokens (held out from train) ...")
    cal_R, cal_Y = collect_residuals(val_ids, emb_dev, K_pos, chunk=chunk)
    print(f"  R={tuple(cal_R.shape)}  Y={tuple(cal_Y.shape)}")
    alphas_to_try = [a for a in [args.alpha, 0.001, 0.01, 0.1, 1.0]]
    seen = set(); alphas_to_try = [a for a in alphas_to_try if not (a in seen or seen.add(a))]
    print(f"[cal] sweeping α∈{alphas_to_try}")
    best = {"ppl": float("inf"), "tau": 1.0, "gamma": 1.0, "alpha": args.alpha}
    best_model = None
    for alpha in alphas_to_try:
        model = MixtureClusterLM.from_counts(mu, counts, alpha=alpha,
                                             V=V, K_pos=K_pos, d_emb=d)
        for tau in [0.01, 0.03, 0.1, 0.3, 1.0]:
            r = fast_ppl(cal_R, cal_Y, model, tau=tau, gamma=1.0,
                         inner_batch=args.inner_batch)
            print(f"  α={alpha} τ={tau} γ=1 → PPL={r['ppl']:.2f} "
                  f"top1={r['top1']*100:.2f}%")
            if r["ppl"] < best["ppl"]:
                best = {"ppl": r["ppl"], "tau": tau, "gamma": 1.0, "alpha": alpha}
                best_model = model
    # gamma sweep at best
    print(f"[cal] gamma sweep at α={best['alpha']} τ={best['tau']}")
    bm = MixtureClusterLM.from_counts(mu, counts, alpha=best["alpha"],
                                      V=V, K_pos=K_pos, d_emb=d)
    for gamma in [0.5, 0.7, 0.85, 0.95, 0.99]:
        r = fast_ppl(cal_R, cal_Y, bm, tau=best["tau"], gamma=gamma,
                     inner_batch=args.inner_batch)
        print(f"  α={best['alpha']} τ={best['tau']} γ={gamma} → "
              f"PPL={r['ppl']:.2f}")
        if r["ppl"] < best["ppl"]:
            best = {**best, "gamma": gamma}
            best_model = bm
    print(f"[cal] best α={best['alpha']} τ={best['tau']} γ={best['gamma']} "
          f"PPL={best['ppl']:.2f}")
    model = best_model
    del counts, cal_R, cal_Y

    sanity = evaluate(model, train_ids[:eval_n], emb_dev, chunk=chunk,
                      inner_batch=args.inner_batch,
                      tau=best["tau"], gamma=best["gamma"], label="train(in)")
    held = evaluate(model, eval_ids, emb_dev, chunk=chunk,
                    inner_batch=args.inner_batch,
                    tau=best["tau"], gamma=best["gamma"], label="heldout")

    # --- 6) Save
    out = ARTIFACT / f"compiled_lm_{args.tag}.pt"
    torch.save({
        "K_pos": K_pos, "clusters": args.clusters, "alpha": best["alpha"],
        "V": V, "d_emb": d, "d_res": d_res,
        "mu": model.mu.cpu(),
        "log_p_cluster": model.log_p_cluster.cpu(),
        "log_p_uni": model.log_p_uni.cpu(),
        "best_tau": best["tau"], "best_gamma": best["gamma"],
        "best_alpha": best["alpha"],
        "train_tokens": train_n, "eval_tokens": eval_n,
    }, str(out))
    print(f"[save] -> {out}")

    results = {
        "model": "Compiled Wikitext LM v13 (mixture of cluster LMs)",
        "K_pos": K_pos, "clusters": args.clusters,
        "best_alpha": best["alpha"],
        "train_tokens": train_n, "eval_tokens": eval_n,
        "V": V, "d_emb": d, "d_res": d_res,
        "best_tau": best["tau"], "best_gamma": best["gamma"],
        "in_distribution": sanity, "heldout": held,
    }
    rp = ARTIFACT / f"eval_results_{args.tag}.json"
    with open(rp, "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
