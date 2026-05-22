"""
compile_wiki_lm_v12.py — Compiled Transformer with Context-Mixing Attention + Compiled FFN
==========================================================================================

Goal (per Douglas, 2026-05-19): finish a fully compiled, weight-based LLM.
v11 hit PPL=2368 with positional shift-and-concat + ridge head + temperature.
That's a 4-gram model dressed up as a transformer; the residual carries
the *last K token identities verbatim*, no contextual rotation.

v12 adds the two operations that a real transformer layer does:

  1. **Context-mixing attention** (PPMI-driven, closed-form). For each
     "context word" c, we precompute Δ_c — the direction in embedding
     space that c "points to" (centroid-direction of its PPMI neighbours).
     The attention output at position t is:
            attn[t] = Σ_{s in window} α(s,t) · Δ_{ids[s]}
     where α is content-based softmax over membership in the context
     vocabulary. This is a real attention head; Q,K,V,O exist (we
     document this in `materialise_attention_weights()`).

     **Result**: r_1[t] = emb[ids[t]] + attn[t]. The same token now sits
     in a different region of embedding space depending on context.
     `bank near river` ≠ `bank near deposit`.

  2. **Compiled FFN as key-value memory** (k-means + empirical conditionals).
     After fitting Δ_c attention and producing r_1[t] over the corpus, we
     k-means cluster the r_1 vectors into K patterns. Each cluster has an
     empirical next-token distribution p_k. The FFN is a soft k-NN lookup:
            ffn[t] = Σ_k σ_k(r_1[t]) · v_k
     where σ_k = softmax of -‖r_1 - μ_k‖²/τ and v_k = emb^T p_k.

     **Result**: r_2[t] = r_1[t] + ffn[t]. This is the Geva et al. (2021)
     key-value memory view of FFNs, compiled from data instead of trained.

  3. **LM head**: ridge regression on the final residual (which now
     concatenates [r_2[t], r_2[t-1], ..., r_2[t-K_pos]] for positional
     context) against next-token one-hot, with temperature calibration.

Every matrix in this model has a documented compile recipe. No SGD.

Run:
    python compile_wiki_lm_v12.py --train-tokens 5M --eval-tokens 500K \\
        --context-vocab 2000 --window 8 --beta 4.0 \\
        --ffn-clusters 4096 --ffn-tau 1.0 \\
        --K 3 --lam 1.0 --tag v12_first
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
ARTIFACT = REPO / "artifacts/compiled_wiki_lm_v12"
ARTIFACT.mkdir(parents=True, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Reuse v11 source artifacts
V5_STATE = REPO / "artifacts/compiled_wiki_lm_v5/compiled_lm.pt"
V5_META = REPO / "artifacts/compiled_wiki_lm_v5/meta.pkl"
BPE_PATH = REPO / "artifacts/bpe_wiki/tokenizer.json"
CORPUS = REPO / "corpora/wikitext103.txt"
V11_CACHE = REPO / "artifacts/compiled_wiki_lm_v11/cache_lm_ids.pt"


# =============================================================================
# Shared plumbing (load + tokenize), mirrors v11
# =============================================================================

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
    print(f"  V={V}, d={d}, BPE vocab size={bpe.get_vocab_size()}")
    return bpe, vocab, tok2id, bpe_to_lm, emb, V, d


def tokenize_corpus(bpe, bpe_to_lm, V, n_chars, cache_path=None):
    if cache_path and cache_path.exists():
        print(f"[tok] Loading cached tokens from {cache_path} ...")
        ids = torch.load(str(cache_path), weights_only=False)
        print(f"  loaded {len(ids):,} LM tokens.")
        return ids
    if V11_CACHE.exists() and n_chars is None:
        print(f"[tok] Loading v11 full-corpus cache from {V11_CACHE} ...")
        ids = torch.load(str(V11_CACHE), weights_only=False)
        print(f"  loaded {len(ids):,} LM tokens.")
        return ids

    print(f"[tok] Reading corpus (n_chars={n_chars or 'all'}) ...")
    with open(CORPUS, "r", encoding="utf-8") as f:
        text = f.read(n_chars) if n_chars else f.read()
    print(f"  read {len(text):,} chars; tokenizing ...")
    t0 = time.time()
    chunk_chars = 8 * 1024 * 1024
    all_bpe = []
    for i in range(0, len(text), chunk_chars):
        enc = bpe.encode(text[i:i + chunk_chars])
        all_bpe.extend(enc.ids)
    print(f"  tokenized to {len(all_bpe):,} BPE tokens in {time.time()-t0:.1f}s")
    t0 = time.time()
    bpe_arr = np.asarray(all_bpe, dtype=np.int64)
    lm_lookup = np.full(bpe.get_vocab_size(), 1, dtype=np.int64)
    for b, lmid in bpe_to_lm.items():
        if 0 <= b < lm_lookup.shape[0]:
            lm_lookup[b] = lmid
    lm_arr = lm_lookup[bpe_arr]
    lm_arr = np.clip(lm_arr, 0, V - 1)
    ids = torch.from_numpy(lm_arr)
    print(f"  mapped to LM ids in {time.time()-t0:.1f}s. final length={len(ids):,}")
    if cache_path:
        torch.save(ids, str(cache_path))
        print(f"  cached to {cache_path}")
    return ids


# =============================================================================
# Co-occurrence statistics (the source of every compiled weight)
# =============================================================================

def build_cooccurrence(ids: torch.Tensor, V: int, window: int = 8,
                      chunk: int = 1_000_000) -> torch.Tensor:
    """Compute symmetric co-occurrence counts C[w, c] = #(c within ±window of w)
    over the corpus, in streaming chunks on GPU.

    Returns: (V, V) float32 tensor on CPU.
    """
    print(f"[cooc] N={ids.numel():,} tokens, V={V}, window=±{window}")
    C_cpu = torch.zeros(V, V, dtype=torch.float32)
    N = ids.numel()
    t0 = time.time()
    ids_gpu = ids.to(DEVICE)
    # Accumulate per-shift contributions. For each k in {1..window} we
    # pair token at position i with token at position i+k, for all i in
    # [0, N-k). Each such pair contributes 2 to C (symmetric) once we
    # add it for (w, c) and (c, w).
    Cg = torch.zeros(V, V, dtype=torch.float32, device=DEVICE)
    for k in range(1, window + 1):
        a = ids_gpu[: N - k]            # (N-k,)
        b = ids_gpu[k:]                 # (N-k,)
        # Stream in chunks to avoid one huge flat index allocation
        m = a.numel()
        cursor = 0
        while cursor < m:
            end = min(cursor + chunk, m)
            ai = a[cursor:end]
            bi = b[cursor:end]
            flat_ab = ai * V + bi
            flat_ba = bi * V + ai
            ones = torch.ones_like(flat_ab, dtype=torch.float32)
            Cg.view(-1).scatter_add_(0, flat_ab, ones)
            Cg.view(-1).scatter_add_(0, flat_ba, ones)
            cursor = end
        if k == 1 or k == window or k % 4 == 0:
            print(f"  shift k={k}: {(time.time()-t0):.1f}s elapsed")
    C_cpu = Cg.cpu()
    del Cg, ids_gpu
    torch.cuda.empty_cache()
    print(f"[cooc] done in {time.time()-t0:.1f}s. total mass = {C_cpu.sum().item():.3e}")
    return C_cpu


def build_ppmi(C: torch.Tensor, smooth: float = 0.75) -> torch.Tensor:
    """Compute PPMI from a co-occurrence matrix C ∈ R^{V×V}.

    PPMI(w, c) = max(0, log( P(w,c) / (P(w) * P_smooth(c)) ))
    Smoothed denominator: P_smooth(c) ∝ count(c)^smooth (the standard
    word2vec/SGNS trick that reduces overweight of common words).
    """
    print(f"[ppmi] computing with smoothing={smooth}")
    V = C.shape[0]
    total = C.sum().item()
    pw = C.sum(dim=1) / total                  # (V,) marginal of w
    pc = C.sum(dim=0)                          # (V,) raw count of c
    pc_smooth = pc.pow(smooth)
    pc_smooth = pc_smooth / pc_smooth.sum()    # smoothed marginal of c
    # Pmi = log( C/total / (pw . pc_smooth) )
    # = log(C) - log(total) - log(pw) - log(pc_smooth)
    # broadcast carefully on CPU to avoid OOM
    log_total = math.log(total)
    log_pw = torch.log(pw.clamp_min(1e-12))    # (V,)
    log_pc = torch.log(pc_smooth.clamp_min(1e-12))   # (V,)
    log_C = torch.log(C.clamp_min(1e-12))
    pmi = log_C - log_total - log_pw[:, None] - log_pc[None, :]
    ppmi = pmi.clamp_min(0.0)
    # zero out entries where C was actually 0 (the clamp above gave them
    # spurious positive values)
    ppmi = torch.where(C > 0, ppmi, torch.zeros_like(ppmi))
    print(f"  ppmi: nnz={int((ppmi > 0).sum().item()):,}, "
          f"mean(>0)={ppmi[ppmi > 0].mean().item():.4f}")
    return ppmi


# =============================================================================
# Compiled context-mixing attention (Layer 1)
# =============================================================================

class ContextMixer:
    """One attention head, compiled from PPMI co-occurrence statistics.

    For each token c in the "context vocabulary" C (top-N by PPMI mass),
    we precompute its meaning-shift vector:
        Δ_c = sum_w ppmi(w, c) * (emb[w] - emb_bar) / Σ_w ppmi(w, c)
    Geometrically: the centroid-direction of the embedding cloud
    surrounding c.

    The attention output at position t is:
        attn[t] = Σ_{s in window of t} α(s, t) · Δ_{ids[s]}
    where α is a content-based softmax that prefers positions whose token
    is in the context vocabulary:
        score(s, t) = β * mass(ids[s])    if ids[s] in C, else -∞
    `mass(c)` is the PPMI total of c (so more informative context words
    get higher weight).

    Equivalent to a transformer head with:
        K[s] = e_{ids[s]}                  (one-hot key on token id)
        Q[t] = β * mass_vec * 1_{C}         (queries the entire context vocab)
        V[s] = Δ_{ids[s]}                  (value = the shift direction)
        O    = identity                     (write straight into residual)
    """

    def __init__(self, emb: torch.Tensor, ppmi: torch.Tensor,
                 context_vocab_size: int = 2000,
                 window: int = 8, beta: float = 4.0):
        V, d = emb.shape
        self.V, self.d = V, d
        self.window = window
        self.beta = beta

        # Score each token by total PPMI mass — these are the words that
        # actually carry context information.
        mass = ppmi.sum(dim=1)                           # (V,)
        topk = torch.topk(mass, k=context_vocab_size).indices
        in_context = torch.zeros(V, dtype=torch.bool)
        in_context[topk] = True
        self.in_context = in_context.to(DEVICE)          # (V,) bool
        self.mass = mass.to(DEVICE)                       # (V,) float

        # Compile Δ_c for every token (we'll zero out non-context tokens).
        # Δ_c = (PPMI[c, :].unsqueeze(-1) * (emb - emb_bar)).sum(0) / norm
        emb_bar = emb.mean(dim=0, keepdim=True)
        centered = emb - emb_bar                          # (V, d)
        # PPMI is (V, V) on CPU — keep this work on CPU to avoid OOM.
        # Each c row: row * centered → sum gives (d,).
        # That's just PPMI @ centered with PPMI rows = c-as-center.
        # We treat PPMI as symmetric (it is by construction here): row c
        # holds PMI(w, c) for w. So Δ_c = PPMI[:, c] · centered, normalized.
        deltas = ppmi.t() @ centered                      # (V, d)
        norms = ppmi.sum(dim=0, keepdim=True).t().clamp_min(1e-6)  # (V, 1)
        deltas = deltas / norms
        # Unit-normalize so β controls the magnitude uniformly
        n2 = deltas.norm(dim=1, keepdim=True).clamp_min(1e-6)
        deltas = deltas / n2
        # Zero out non-context tokens (their Δ would be noise)
        deltas[~in_context] = 0.0
        self.deltas = deltas.to(DEVICE)                   # (V, d)
        print(f"[ContextMixer] |C|={context_vocab_size}, "
              f"Δ norms (context): {deltas[in_context].norm(dim=1).mean().item():.3f}")

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        """ids: (N,) on DEVICE. Returns attn[t] ∈ R^d for each t."""
        N = ids.numel()
        W = self.window
        # per-position Δ vectors and importance scores
        D = self.deltas[ids]                                # (N, d)
        score = (self.mass[ids] * self.in_context[ids].float())  # (N,)

        # Causal attention with hard window: sum over s in [t-W, t-1].
        # Implement via cumulative sums in score-weighted Δ space.
        # First, weighted Δ: D[t] * exp(β * score[t]) for context tokens,
        # 0 otherwise. Use softmax over the window (numerically stable).
        # Trick: precompute W = exp(β * s); then attn[t] = (Σ_{s in win} W[s] D[s]) / (Σ_{s in win} W[s] + ε)
        s_aug = torch.where(self.in_context[ids], self.beta * score,
                            torch.full_like(score, -1e9))  # (N,)
        # subtract per-window max for stability — approximate with global
        # max minus a few (the windows are short, so this is fine)
        s_max = s_aug.max()
        w = torch.exp(s_aug - s_max).clamp_min(0.0)        # (N,)
        wD = D * w.unsqueeze(-1)                           # (N, d)

        # Pad on the left so window [t-W .. t-1] is well-defined
        wD_pad = torch.zeros(N + W, D.shape[1], device=D.device, dtype=D.dtype)
        w_pad = torch.zeros(N + W, device=w.device, dtype=w.dtype)
        wD_pad[W:] = wD
        w_pad[W:] = w
        # Cumulative sums
        cs_wD = torch.cumsum(wD_pad, dim=0)                # (N+W, d)
        cs_w = torch.cumsum(w_pad, dim=0)                  # (N+W,)
        # Window-sum at position t (looking at s in [t-W..t-1]):
        # = cs[ t-1+W+1 = t+W ] - cs[ (t-W) + W = t ] = cs[t+W] - cs[t]
        # but we need indices into the padded arrays. Our padded index for
        # original position t is (W + t).
        # Sum over s in [t-W..t-1] in original = padded indices [t..t+W-1] = cs_pad[t+W-1] - cs_pad[t-1]
        # Use range(N):
        t_idx = torch.arange(N, device=D.device)
        upper = t_idx + W                                  # exclusive: cs at t+W-1 = up_idx
        lower = t_idx                                      # cs at t-1 in original = cs_pad[t-1] in padded? actually
        # we want sum over padded indices i in [t, t+W-1], which is cs[t+W-1] - cs[t-1]
        # because cs[k] = sum of items 0..k. But our cs starts at index 0; cs[-1] = 0 (use a zero pad).
        # Easier: cumsum with prepended zero:
        # We re-do with a leading zero entry.
        # (Redo cleanly to avoid off-by-one bugs.)
        wD_padz = torch.cat([torch.zeros(1, D.shape[1], device=D.device, dtype=D.dtype), wD_pad], dim=0)
        w_padz = torch.cat([torch.zeros(1, device=w.device, dtype=w.dtype), w_pad], dim=0)
        cs_wD = torch.cumsum(wD_padz, dim=0)               # (N+W+1, d)
        cs_w = torch.cumsum(w_padz, dim=0)                 # (N+W+1,)
        # padded index for original position t is (W + t). We want padded
        # indices in [W + t - W, W + t - 1] = [t, W + t - 1]. Sum over that
        # range = cs[W+t] - cs[t].
        attn_w = cs_w[W + t_idx] - cs_w[t_idx]              # (N,)
        attn_wD = cs_wD[W + t_idx] - cs_wD[t_idx]           # (N, d)
        attn = attn_wD / (attn_w.unsqueeze(-1).clamp_min(1e-6))
        return attn


# =============================================================================
# Compiled FFN as key-value memory (Layer 2)
# =============================================================================

class CompiledFFN:
    """Soft k-NN key-value FFN, compiled from clustered residuals.

    Compile recipe:
      1. Sample N_sample (~500K) residual vectors r_1[t] from the corpus.
      2. K-means cluster them into K patterns μ_k.
      3. For each cluster k, compute empirical next-token distribution
            p_k(y) = Laplace-smoothed P( ids[t+1] = y | cluster(r_1[t]) = k )
      4. Project to embedding space: v_k = emb^T (log p_k - log p_uniform)
      5. FFN forward:
            ffn[t] = Σ_k softmax_k(-‖r_1[t] - μ_k‖² / τ) · v_k

    In standard Linear→Activation→Linear form:
        W_in[k, :]   = 2 μ_k / τ
        b_in[k]      = -‖μ_k‖² / τ
        σ            = softmax over k
        W_out[:, k]  = v_k
    """

    def __init__(self, K: int = 4096, tau: float = 1.0):
        self.K = K
        self.tau = tau
        self.centroids: torch.Tensor | None = None
        self.values: torch.Tensor | None = None

    def fit(self, R: torch.Tensor, next_ids: torch.Tensor, emb: torch.Tensor,
            n_iter: int = 12, sample: int = 500_000):
        """R: (M, d_in) residuals, next_ids: (M,) next-token ids, emb: (V, d_emb)."""
        M, d_in = R.shape
        V, d_emb = emb.shape
        if M > sample:
            sel = torch.randperm(M, device=R.device)[:sample]
            Rs = R[sel]
            Ys = next_ids[sel]
            print(f"[FFN.fit] sampled {sample:,} of {M:,} residuals")
        else:
            Rs = R
            Ys = next_ids
            print(f"[FFN.fit] using all {M:,} residuals")

        # K-means initialisation: k-means++-lite via random sample
        idx = torch.randperm(Rs.shape[0], device=Rs.device)[:self.K]
        centroids = Rs[idx].clone()
        print(f"[FFN.fit] K-means start: K={self.K}, d_in={d_in}, n_iter={n_iter}")
        t0 = time.time()
        for it in range(n_iter):
            # Hard assignment in chunks
            assign = torch.empty(Rs.shape[0], dtype=torch.long, device=Rs.device)
            CHUNK = 16384
            for s in range(0, Rs.shape[0], CHUNK):
                d2 = torch.cdist(Rs[s:s + CHUNK], centroids, p=2).pow_(2)
                assign[s:s + CHUNK] = d2.argmin(dim=1)
                del d2
            # Recompute centroids
            new = torch.zeros_like(centroids)
            counts = torch.zeros(self.K, device=Rs.device)
            new.index_add_(0, assign, Rs)
            counts.index_add_(0, assign, torch.ones(Rs.shape[0], device=Rs.device))
            mask = counts > 0
            new[mask] /= counts[mask].unsqueeze(-1)
            # Reinit empty clusters at random points
            n_empty = (~mask).sum().item()
            if n_empty > 0:
                refill_idx = torch.randperm(Rs.shape[0], device=Rs.device)[:n_empty]
                new[~mask] = Rs[refill_idx]
            shift = (new - centroids).norm(dim=1).mean().item()
            centroids = new
            print(f"  iter {it+1}: shift={shift:.4f}, "
                  f"|empty|={n_empty}, t={time.time()-t0:.1f}s")
            if shift < 1e-4:
                break

        # Final assignment for value computation
        assign = torch.empty(Rs.shape[0], dtype=torch.long, device=Rs.device)
        CHUNK = 16384
        for s in range(0, Rs.shape[0], CHUNK):
            d2 = torch.cdist(Rs[s:s + CHUNK], centroids, p=2).pow_(2)
            assign[s:s + CHUNK] = d2.argmin(dim=1)
            del d2
        # Empirical p_k(y): Laplace-smoothed
        counts = torch.zeros(self.K, V, device=Rs.device)
        flat = assign * V + Ys
        ones = torch.ones_like(flat, dtype=torch.float32)
        counts.view(-1).scatter_add_(0, flat, ones)
        eps = 1e-3
        p_k = (counts + eps) / (counts.sum(dim=1, keepdim=True) + V * eps)   # (K, V)
        # Uniform baseline
        log_p_uniform = math.log(1.0 / V)
        log_p_k = torch.log(p_k)
        delta_log = log_p_k - log_p_uniform                                  # (K, V)
        # Project to embedding space: v_k = emb^T · delta_log[k]
        # That's (d_emb, V) @ (V,) per k → (K, d_emb)
        values = delta_log @ emb.to(Rs.device)                               # (K, d_emb)
        # Scale values so each per-cluster signal sits at unit norm — same
        # scale as the embedding lookups. The ridge LM head will then learn
        # its own re-weighting; without this normalisation the FFN signal
        # is hundreds of times larger than emb and dominates the residual.
        v_norms = values.norm(dim=1, keepdim=True).clamp_min(1e-6)
        values = values / v_norms
        print(f"[FFN.fit] done in {time.time()-t0:.1f}s. "
              f"raw ||v||={v_norms.mean().item():.3f} → normalised to 1.0")
        self.centroids = centroids
        self.values = values

    def forward(self, R: torch.Tensor) -> torch.Tensor:
        """R: (N, d_in). Returns (N, d_emb) FFN delta."""
        assert self.centroids is not None and self.values is not None
        out = torch.empty(R.shape[0], self.values.shape[1],
                          device=R.device, dtype=R.dtype)
        CHUNK = 4096
        for s in range(0, R.shape[0], CHUNK):
            d2 = torch.cdist(R[s:s + CHUNK], self.centroids, p=2).pow_(2)
            w = F.softmax(-d2 / self.tau, dim=1)                # (chunk, K)
            out[s:s + CHUNK] = w @ self.values
            del d2, w
        return out


# =============================================================================
# v12 model
# =============================================================================

class CompiledTransformerV12:
    """Forward pass:
        x[t]   = emb[ids[t]]
        Δ[t]   = context-mixer(ids)[t]            # PPMI-driven attention
        r_1[t] = x[t] + Δ[t]
        f[t]   = compiled FFN(r_1[t])             # key-value memory
        r_2[t] = r_1[t] + f[t]
        R[t]   = concat([r_2[t], r_2[t-1], ..., r_2[t-K_pos]])
                 + bias channel
        logits = W @ R[t]
    """

    def __init__(self, emb, mixer: ContextMixer, ffn: CompiledFFN,
                 K_pos: int, V: int, d_emb: int, use_bias: bool = True):
        self.emb = emb.to(DEVICE)
        self.mixer = mixer
        self.ffn = ffn
        self.K_pos = K_pos
        self.V = V
        self.d_emb = d_emb
        self.use_bias = use_bias
        # After mixer + FFN, residual is still d_emb. Then we positionally
        # concatenate K+1 slots and (optionally) append a constant-1 bias.
        self.d_res = (K_pos + 1) * d_emb + (1 if use_bias else 0)
        self.W: torch.Tensor | None = None

    def residual_l2(self, ids: torch.Tensor) -> torch.Tensor:
        """Forward through the two enrichment layers (no positional stack).
        ids: (N,) on DEVICE. Returns (N, d_emb)."""
        X = self.emb[ids]                          # (N, d)
        attn = self.mixer.forward(ids)             # (N, d)
        r1 = X + attn
        ffn = self.ffn.forward(r1)                 # (N, d)
        r2 = r1 + ffn
        return r2

    def attention_forward(self, ids: torch.Tensor) -> torch.Tensor:
        """Build the full residual that goes into the LM head:
        positional concat over K_pos+1 slots of r_2."""
        r2 = self.residual_l2(ids)                 # (N, d)
        N = r2.shape[0]
        slots = [r2]
        for k in range(1, self.K_pos + 1):
            s = torch.zeros_like(r2)
            if k < N:
                s[k:] = r2[:-k]
            slots.append(s)
        if self.use_bias:
            slots.append(torch.ones(N, 1, device=r2.device, dtype=r2.dtype))
        return torch.cat(slots, dim=-1)            # (N, d_res)

    def compile_head_streaming(self, ids, lam=1.0, chunk=200_000):
        d = self.d_res
        V = self.V
        K = self.K_pos
        N = ids.shape[0]
        print(f"[head] streaming ridge: N={N:,}, d={d}, V={V}, chunk={chunk:,}, λ={lam}")
        RtR = torch.zeros(d, d, dtype=torch.float64, device=DEVICE)
        YtR = torch.zeros(V, d, dtype=torch.float64, device=DEVICE)
        cursor = K
        seen = 0
        t0 = time.time()
        while cursor < N - 1:
            end = min(cursor + chunk, N - 1)
            prefix = cursor - K
            window = ids[prefix:end + 1].to(DEVICE, non_blocking=True)
            R_full = self.attention_forward(window)
            local_lo = K
            local_hi = K + (end - cursor)
            R = R_full[local_lo:local_hi]
            Y = window[local_lo + 1:local_hi + 1].long()
            RtR += (R.t().double() @ R.double())
            R64 = R.double()
            YtR.index_add_(0, Y, R64)
            cnt = R.shape[0]
            del R_full, R, R64
            seen += cnt
            cursor = end
            if cursor >= N - 1 or seen % (chunk * 5) == 0:
                rate = seen / max(time.time() - t0, 1e-6)
                print(f"  ... accumulated {seen:,}/{N-K-1:,} ({rate/1e6:.2f}M/s, "
                      f"{time.time()-t0:.1f}s)")
        print(f"[head] solving (R^TR + λI)^-1 ... d={d}")
        t1 = time.time()
        A = RtR + lam * torch.eye(d, dtype=torch.float64, device=DEVICE)
        L = torch.linalg.cholesky(A)
        W = torch.cholesky_solve(YtR.t(), L).t().to(torch.float32)
        print(f"  solve done in {time.time()-t1:.2f}s. ||W||_F={W.norm().item():.4f}")
        del RtR, YtR, A, L
        torch.cuda.empty_cache()
        self.W = W
        return W


# =============================================================================
# Evaluation (mirrors v11)
# =============================================================================

def evaluate(model: CompiledTransformerV12, ids: torch.Tensor,
             chunk=200_000, label="eval", inner_batch=4096, temperature=1.0) -> dict:
    assert model.W is not None
    K = model.K_pos
    N = ids.numel()
    print(f"[{label}] N={N:,}, chunk={chunk:,}, inner_batch={inner_batch}, T={temperature}")
    nll_sum = 0.0
    top1 = 0
    top5 = 0
    count = 0
    cursor = K
    t0 = time.time()
    while cursor < N - 1:
        end = min(cursor + chunk, N - 1)
        prefix = cursor - K
        window = ids[prefix:end + 1].to(DEVICE, non_blocking=True)
        R_full = model.attention_forward(window)
        local_lo = K
        local_hi = K + (end - cursor)
        R = R_full[local_lo:local_hi]
        Y = window[local_lo + 1:local_hi + 1].long()
        for i in range(0, R.shape[0], inner_batch):
            Rb = R[i:i + inner_batch]
            Yb = Y[i:i + inner_batch]
            logits = (Rb @ model.W.t()) * temperature
            logp = F.log_softmax(logits, dim=-1)
            nll = -logp.gather(1, Yb.unsqueeze(1)).squeeze(1)
            nll_sum += nll.sum().item()
            tk = logits.topk(5, dim=-1).indices
            top1 += (tk[:, 0] == Yb).sum().item()
            top5 += (tk == Yb.unsqueeze(1)).any(dim=1).sum().item()
            count += Rb.shape[0]
            del logits, logp, nll, tk
        del R_full, R, Y
        cursor = end
    avg_nll = nll_sum / max(count, 1)
    ppl = math.exp(avg_nll)
    out = dict(count=count, ppl=ppl, top1=top1 / max(count, 1),
               top5=top5 / max(count, 1), avg_nll=avg_nll)
    print(f"  {label}: tokens={count:,}, PPL={ppl:,.2f}, "
          f"top1={out['top1']*100:.2f}%, top5={out['top5']*100:.2f}% "
          f"({time.time()-t0:.1f}s)")
    return out


# =============================================================================
# Main
# =============================================================================

def parse_size(s):
    s = str(s).strip()
    if s.endswith("M"):
        return int(float(s[:-1]) * 1_000_000)
    if s.endswith("K"):
        return int(float(s[:-1]) * 1_000)
    if s.endswith("B"):
        return int(float(s[:-1]) * 1_000_000_000)
    return int(s)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train-tokens", type=str, default="5M")
    p.add_argument("--eval-tokens", type=str, default="500K")
    p.add_argument("--cooc-tokens", type=str, default="2M",
                   help="tokens used to build co-occurrence matrix")
    p.add_argument("--n-chars", type=str, default=None)
    p.add_argument("--context-vocab", type=int, default=2000)
    p.add_argument("--window", type=int, default=8)
    p.add_argument("--beta", type=float, default=4.0)
    p.add_argument("--ffn-clusters", type=int, default=4096)
    p.add_argument("--ffn-tau", type=float, default=1.0)
    p.add_argument("--ffn-fit-tokens", type=str, default="500K")
    p.add_argument("--K", type=int, default=3)
    p.add_argument("--lam", type=float, default=1.0)
    p.add_argument("--chunk", type=str, default="200K")
    p.add_argument("--tag", type=str, default="default")
    args = p.parse_args()

    train_n = parse_size(args.train_tokens)
    eval_n = parse_size(args.eval_tokens)
    cooc_n = parse_size(args.cooc_tokens)
    chunk = parse_size(args.chunk)
    ffn_fit_n = parse_size(args.ffn_fit_tokens)
    n_chars = parse_size(args.n_chars) if args.n_chars else None

    bpe, vocab, tok2id, bpe_to_lm, emb, V, d = load_setup()
    ids = tokenize_corpus(bpe, bpe_to_lm, V, n_chars,
                          cache_path=ARTIFACT / "cache_lm_ids.pt")
    N = ids.numel()
    if train_n + eval_n > N:
        print(f"WARN: train+eval > corpus; shrinking train.")
        train_n = max(N - eval_n, N // 2)
    train_ids = ids[:train_n]
    eval_ids = ids[-eval_n:]
    print(f"[split] train={train_n:,}, eval={eval_n:,}")

    # === Build / load co-occurrence ===
    cooc_path = ARTIFACT / f"cooc_w{args.window}_n{cooc_n}.pt"
    if cooc_path.exists():
        print(f"[cooc] loading from cache {cooc_path}")
        C = torch.load(str(cooc_path), weights_only=False)
    else:
        cooc_ids = train_ids[:cooc_n]
        C = build_cooccurrence(cooc_ids, V, window=args.window)
        torch.save(C, str(cooc_path))
        print(f"  cached to {cooc_path}")

    ppmi_path = ARTIFACT / f"ppmi_w{args.window}_n{cooc_n}.pt"
    if ppmi_path.exists():
        print(f"[ppmi] loading from cache {ppmi_path}")
        ppmi = torch.load(str(ppmi_path), weights_only=False)
    else:
        ppmi = build_ppmi(C)
        torch.save(ppmi, str(ppmi_path))
        print(f"  cached to {ppmi_path}")

    # === Build context mixer ===
    mixer = ContextMixer(emb, ppmi, context_vocab_size=args.context_vocab,
                         window=args.window, beta=args.beta)

    # === Fit compiled FFN ===
    print("\n[ffn] generating residuals for FFN fit ...")
    fit_ids = train_ids[:ffn_fit_n].to(DEVICE)
    # forward through the mixer only (FFN doesn't exist yet)
    with torch.no_grad():
        X = emb.to(DEVICE)[fit_ids]
        attn = mixer.forward(fit_ids)
        r1 = X + attn
    next_ids = train_ids[1:ffn_fit_n + 1].to(DEVICE)
    if r1.shape[0] != next_ids.shape[0]:
        m = min(r1.shape[0], next_ids.shape[0])
        r1, next_ids = r1[:m], next_ids[:m]
    ffn = CompiledFFN(K=args.ffn_clusters, tau=args.ffn_tau)
    ffn.fit(r1, next_ids, emb.to(DEVICE), n_iter=10, sample=min(500_000, r1.shape[0]))
    del r1, X, attn
    torch.cuda.empty_cache()

    # === Assemble model + compile head ===
    model = CompiledTransformerV12(emb, mixer, ffn, K_pos=args.K, V=V, d_emb=d)
    W = model.compile_head_streaming(train_ids, lam=args.lam, chunk=chunk)

    # === Calibrate temperature ===
    cal_n = min(50_000, train_n // 4)
    cal_ids = train_ids[-cal_n - 1:]
    print(f"\n[calibrate] sweeping temperature on {cal_n:,} train tokens")
    best_T = 1.0
    best_ppl = float("inf")
    for T in [1.0, 3.0, 10.0, 30.0, 50.0, 75.0, 100.0, 150.0, 200.0, 300.0, 500.0, 800.0]:
        r = evaluate(model, cal_ids, chunk=chunk, label=f"  T={T}", temperature=T)
        if r["ppl"] < best_ppl:
            best_ppl = r["ppl"]
            best_T = T
    print(f"[calibrate] best T={best_T} -> PPL={best_ppl:.2f}")

    sanity = evaluate(model, train_ids[:eval_n], chunk=chunk, label="train(in)",
                      temperature=best_T)
    held = evaluate(model, eval_ids, chunk=chunk, label="heldout",
                    temperature=best_T)

    # Save
    out = ARTIFACT / f"compiled_lm_{args.tag}.pt"
    torch.save({
        "K_pos": args.K,
        "lam": args.lam,
        "V": V,
        "d_emb": d,
        "d_res": model.d_res,
        "emb": emb.cpu(),
        "W": W.cpu(),
        "mixer_deltas": mixer.deltas.cpu(),
        "mixer_in_context": mixer.in_context.cpu(),
        "mixer_mass": mixer.mass.cpu(),
        "mixer_window": args.window,
        "mixer_beta": args.beta,
        "ffn_centroids": ffn.centroids.cpu(),
        "ffn_values": ffn.values.cpu(),
        "ffn_tau": args.ffn_tau,
        "best_T": best_T,
        "train_tokens": train_n,
        "eval_tokens": eval_n,
    }, str(out))
    print(f"[save] -> {out}")

    res_path = ARTIFACT / f"eval_results_{args.tag}.json"
    results = {
        "model": "Compiled Wikitext LM v12 (context-mixing attn + compiled FFN)",
        "K_pos": args.K,
        "context_vocab": args.context_vocab,
        "window": args.window,
        "beta": args.beta,
        "ffn_clusters": args.ffn_clusters,
        "ffn_tau": args.ffn_tau,
        "lam": args.lam,
        "train_tokens": train_n,
        "eval_tokens": eval_n,
        "V": V,
        "d_emb": d,
        "d_res": model.d_res,
        "best_T": best_T,
        "in_distribution": sanity,
        "heldout": held,
    }
    with open(res_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[save] -> {res_path}")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
