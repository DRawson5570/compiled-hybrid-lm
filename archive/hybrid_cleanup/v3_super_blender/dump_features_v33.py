"""hybrid/v3_super_blender/dump_features_v33.py

Recompute all 21 v33 channels on a token slice and dump compact per-position
arrays suitable for training active super blenders.

Usage:
    python hybrid/v3_super_blender/dump_features_v33.py \\
        --kn-pickle artifacts/compiled_wiki_lm_v23/kn7_22m.pkl \\
        --counts-file artifacts/compiled_wiki_lm_v14/counts_k2_c64k.pt \\
        --out-dir hybrid/v3_super_blender/data_real \\
        --val-tokens 30K --eval-tokens 100K
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, "/home/drawson/llm_decoupling")

from compile_wiki_lm_v13 import (
    load_setup, load_or_build_tokens, build_residual, parse_size, DEVICE,
)
from compile_wiki_lm_v14 import SparseMixtureClusterLM
from compile_wiki_lm_v23 import ModifiedKNGram  # noqa: F401
from compile_wiki_lm_v24 import compute_log_p_kn
from compile_wiki_lm_v33 import (
    build_v33_induction_log_probs,
    compute_log_p_attn_unigram,
    compute_log_p_attn_residual_sliced,
    build_ppmi_semantic_log_probs_all,
    build_knn_log_probs_sliced,
    build_shape_transition_probs_v33,
    token_shape,
)

CHANNEL_NAMES = [
    "kn", "mix",
    "tri_f", "tri_s", "bi_f", "bi_s", "uc_f", "uc_s",
    "attn_uf", "attn_us", "attn_ug",
    "attn_rf1", "attn_rs1",
    "attn_rf2", "attn_rs2", "attn_rg2",
    "attn_rf3", "attn_rs3",
    "ppmi", "knn", "shape"
]
C = len(CHANNEL_NAMES)


def compute_mix_log_probs_low_mem(
    ids_t: torch.Tensor,
    emb_dev_t: torch.Tensor,
    model_t: SparseMixtureClusterLM,
    K_pos: int,
    top_M: int,
    tau: float,
    gamma: float
) -> torch.Tensor:
    """Memory-efficient implementation of SparseMixtureClusterLM logits."""
    N_t = ids_t.shape[0]
    V_t = model_t.V
    r = build_residual(ids_t.to(emb_dev_t.device).long(), emb_dev_t, K_pos)
    start_t = K_pos - 1
    end_t = N_t - 1
    T_local = end_t - start_t

    # Allocation of returned structure
    out_log_probs = torch.zeros(T_local, V_t, dtype=torch.float32, device="cpu")

    mu_sq = (model_t.mu * model_t.mu).sum(dim=1)
    chunk_size = 350

    log_p_cluster_float = model_t.log_p_cluster.to("cpu").float()
    log_p_uni_float = model_t.log_p_uni.to("cpu").float().unsqueeze(0)

    for s in range(start_t, end_t, chunk_size):
        e = min(s + chunk_size, end_t)
        r_c = r[s:e].to(DEVICE)
        r_sq = (r_c * r_c).sum(dim=1, keepdim=True)
        d2 = r_sq + mu_sq.unsqueeze(0) - 2 * (r_c @ model_t.mu.T)
        
        if top_M and top_M < model_t.mu.shape[0]:
            _, idx = d2.topk(top_M, dim=1, largest=False)
            d2_top = d2.gather(1, idx)
            log_pi = F.log_softmax(-d2_top / tau, dim=1)
            # Gather-index model_t.log_p_cluster on CPU to avoid allocating massive tensors on GPU
            log_p_top = log_p_cluster_float[idx.cpu()].to(DEVICE)
            log_mix = torch.logsumexp(log_pi.unsqueeze(2) + log_p_top, dim=1)
        else:
            log_pi = F.log_softmax(-d2 / tau, dim=1)
            log_mix = torch.logsumexp(log_pi.unsqueeze(2) + log_p_cluster_float.to(DEVICE).unsqueeze(0), dim=1)

        if gamma < 1.0:
            log_p_chunk = torch.logaddexp(
                math.log(gamma) + log_mix,
                math.log(1.0 - gamma) + log_p_uni_float.to(DEVICE),
            )
        else:
            log_p_chunk = log_mix

        out_log_probs[s - start_t : e - start_t] = log_p_chunk.cpu()

    return out_log_probs


def compute_all_channels(ids_t: torch.Tensor, ids_n: np.ndarray, V: int, d: int,
                         emb_dev: torch.Tensor, kn, model: SparseMixtureClusterLM,
                         ppmi_probs: torch.Tensor, shape_probs: torch.Tensor, token_shapes: np.ndarray,
                         args, label: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute 21 channels and extract per-position stats.
    
    Returns:
        log_p_observed: (T, C)
        log_p_lag1: (T, C)
        entropy: (T, C)
        max_log_prob: (T, C)
        log_p_targets: (T, C)
    """
    N = len(ids_n)
    T = N - args.K_pos
    targets = ids_n[args.K_pos:]
    context_tokens = ids_n[args.K_pos - 1 : len(ids_n) - 1]

    # Initialize feature tables
    log_p_observed = np.zeros((T, C), dtype=np.float32)
    log_p_lag1 = np.zeros((T, C), dtype=np.float32)
    entropy = np.zeros((T, C), dtype=np.float32)
    max_log_prob = np.zeros((T, C), dtype=np.float32)
    log_p_targets = np.zeros((T, C), dtype=np.float32)

    # Temporary directory for sequential virtualization
    mmap_dir = Path("/tmp/mmap_dump_v33")
    mmap_dir.mkdir(parents=True, exist_ok=True)

    def extract_stats(log_probs: np.ndarray, channel_idx: int):
        """Extract summary stats from (T, V) log probs."""
        # 1. log_p_observed: log p(y_t | context)
        log_p_observed[:, channel_idx] = log_probs[np.arange(T), targets]
        
        # 2. log_p_lag1: log p(y_{t-1} | context) -> shift by 1
        lag1_ids = ids_n[args.K_pos - 1 : len(ids_n) - 1]
        log_p_lag1[:, channel_idx] = log_probs[np.arange(T), lag1_ids]
        
        # 3. Entropy: -sum(p * log p)
        # Avoid direct float32 exponentiation over huge arrays to save memory; chunk step by step
        chunk = 2000
        for s in range(0, T, chunk):
            e = min(s + chunk, T)
            p = np.exp(log_probs[s:e])
            entropy[s:e, channel_idx] = -np.sum(p * log_probs[s:e], axis=1)
            max_log_prob[s:e, channel_idx] = np.max(log_probs[s:e], axis=1)

        # 4. log_p_targets: same as log_p_observed
        log_p_targets[:, channel_idx] = log_p_observed[:, channel_idx]

    # 1. Cluster mixture
    t0 = time.time()
    l_mix = compute_mix_log_probs_low_mem(ids_t, emb_dev, model, args.K_pos, args.top_M, args.tau, args.gamma).numpy()
    print(f"  (1/21) mix done ({time.time() - t0:.1f}s)")
    extract_stats(l_mix, 1)
    del l_mix
    gc.collect()

    # 2. Global KN7
    t0 = time.time()
    l_kn = compute_log_p_kn(kn, ids_n, args.K_pos)
    print(f"  (2/21) KN done ({time.time() - t0:.1f}s)")
    extract_stats(l_kn, 0)
    del l_kn
    gc.collect()

    # 3. Dynamic caches
    t0 = time.time()
    lp_trif, lp_tris, lp_bif, lp_bis, lp_ucf, lp_ucs = build_v33_induction_log_probs(
        ids_n, V, args.K_pos, args.window,
        args.lam_tri_fast, args.lam_tri_slow,
        args.lam_bi_fast, args.lam_bi_slow,
        args.lam_ucache_fast, args.lam_ucache_slow,
        args.alpha_tri_fast, args.alpha_tri_slow,
        args.alpha_bi_fast, args.alpha_bi_slow,
        args.alpha_ucache_fast, args.alpha_ucache_slow,
    )
    print(f"  (3-8/21) decay caches done ({time.time() - t0:.1f}s)")
    extract_stats(lp_trif, 2)
    extract_stats(lp_tris, 3)
    extract_stats(lp_bif, 4)
    extract_stats(lp_bis, 5)
    extract_stats(lp_ucf, 6)
    extract_stats(lp_ucs, 7)
    del lp_trif, lp_tris, lp_bif, lp_bis, lp_ucf, lp_ucs
    gc.collect()

    # 4. Unigram Attention Caches (uf, us, ug)
    t0 = time.time()
    lp_attn_uf = compute_log_p_attn_unigram(ids_t, emb_dev, args.W_attn_uf, args.beta_attn_uf, args.theta_attn_uf, args.alpha_attn_uf, args.K_pos).numpy()
    extract_stats(lp_attn_uf, 8)
    del lp_attn_uf

    lp_attn_us = compute_log_p_attn_unigram(ids_t, emb_dev, args.W_attn_us, args.beta_attn_us, args.theta_attn_us, args.alpha_attn_us, args.K_pos).numpy()
    extract_stats(lp_attn_us, 9)
    del lp_attn_us

    lp_attn_ug = compute_log_p_attn_unigram(ids_t, emb_dev, args.W_attn_ug, args.beta_attn_ug, args.theta_attn_ug, args.alpha_attn_ug, args.K_pos).numpy()
    extract_stats(lp_attn_ug, 10)
    del lp_attn_ug
    print(f"  (9-11/21) unigram attention done ({time.time() - t0:.1f}s)")
    gc.collect()

    # 5. Core Phrase Residual construction
    t0 = time.time()
    r_full = build_residual(ids_t.to(DEVICE).long(), emb_dev, K=3)

    # 6. Residual Attention Caches
    print(f"  computing multi-scale state attention caches...")
    lp_attn_rf1 = compute_log_p_attn_residual_sliced(ids_t, r_full, d, K=1, W_attn=args.W_attn_rf1, beta=args.beta_attn_rf1, theta=args.theta_attn_rf1, alpha_attn=args.alpha_attn_rf1, K_pos=args.K_pos, V=V).numpy()
    extract_stats(lp_attn_rf1, 11)
    del lp_attn_rf1

    lp_attn_rs1 = compute_log_p_attn_residual_sliced(ids_t, r_full, d, K=1, W_attn=args.W_attn_rs1, beta=args.beta_attn_rs1, theta=args.theta_attn_rs1, alpha_attn=args.alpha_attn_rs1, K_pos=args.K_pos, V=V).numpy()
    extract_stats(lp_attn_rs1, 12)
    del lp_attn_rs1

    lp_attn_rf2 = compute_log_p_attn_residual_sliced(ids_t, r_full, d, K=2, W_attn=args.W_attn_rf2, beta=args.beta_attn_rf2, theta=args.theta_attn_rf2, alpha_attn=args.alpha_attn_rf2, K_pos=args.K_pos, V=V).numpy()
    extract_stats(lp_attn_rf2, 13)
    del lp_attn_rf2

    lp_attn_rs2 = compute_log_p_attn_residual_sliced(ids_t, r_full, d, K=2, W_attn=args.W_attn_rs2, beta=args.beta_attn_rs2, theta=args.theta_attn_rs2, alpha_attn=args.alpha_attn_rs2, K_pos=args.K_pos, V=V).numpy()
    extract_stats(lp_attn_rs2, 14)
    del lp_attn_rs2

    lp_attn_rg2 = compute_log_p_attn_residual_sliced(ids_t, r_full, d, K=2, W_attn=args.W_attn_rg2, beta=args.beta_attn_rg2, theta=args.theta_attn_rg2, alpha_attn=args.alpha_attn_rg2, K_pos=args.K_pos, V=V).numpy()
    extract_stats(lp_attn_rg2, 15)
    del lp_attn_rg2

    lp_attn_rf3 = compute_log_p_attn_residual_sliced(ids_t, r_full, d, K=3, W_attn=args.W_attn_rf3, beta=args.beta_attn_rf3, theta=args.theta_attn_rf3, alpha_attn=args.alpha_attn_rf3, K_pos=args.K_pos, V=V).numpy()
    extract_stats(lp_attn_rf3, 16)
    del lp_attn_rf3

    lp_attn_rs3 = compute_log_p_attn_residual_sliced(ids_t, r_full, d, K=3, W_attn=args.W_attn_rs3, beta=args.beta_attn_rs3, theta=args.theta_attn_rs3, alpha_attn=args.alpha_attn_rs3, K_pos=args.K_pos, V=V).numpy()
    extract_stats(lp_attn_rs3, 17)
    del lp_attn_rs3
    print(f"  (12-18/21) multi-scale state attention done ({time.time() - t0:.1f}s)")

    del r_full
    gc.collect()

    # 7. PPMI Semantic Channel
    t0 = time.time()
    l_ppmi = ppmi_probs[context_tokens].numpy()
    print(f"  (19/21) PPMI semantic channel done ({time.time() - t0:.1f}s)")
    extract_stats(l_ppmi, 18)
    del l_ppmi
    gc.collect()

    # 8. KNN Context Retrieval Channel
    t0 = time.time()
    l_knn_full = build_knn_log_probs_sliced(ids_n, emb_dev, V, k=args.knn_k, temperature=args.knn_temp)
    l_knn = l_knn_full[args.K_pos:].numpy()
    print(f"  (20/21) KNN retrieval channel done ({time.time() - t0:.1f}s)")
    extract_stats(l_knn, 19)
    del l_knn_full, l_knn
    gc.collect()

    # 9. Word Shape Transition Channel
    t0 = time.time()
    prev_token_shapes = token_shapes[context_tokens]
    l_shape = shape_probs[prev_token_shapes].numpy()
    print(f"  (21/21) Word Shape transition channel done ({time.time() - t0:.1f}s)")
    extract_stats(l_shape, 20)
    del l_shape
    gc.collect()

    return log_p_observed, log_p_lag1, entropy, max_log_prob, log_p_targets


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--kn-pickle", type=str, required=True)
    p.add_argument("--counts-file", type=str, required=True)
    p.add_argument("--out-dir", type=str, default="hybrid/v3_super_blender/data_real_v33")
    p.add_argument("--val-tokens", type=str, default="30K")
    p.add_argument("--eval-tokens", type=str, default="100K")
    
    p.add_argument("--K-pos", type=int, default=2)
    p.add_argument("--top-M", type=int, default=16)
    p.add_argument("--tau", type=float, default=0.05)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--alpha-mix", type=float, default=0.01)
    p.add_argument("--window", type=int, default=8192)

    p.add_argument("--alpha-tri-fast", type=float, default=1e-5)
    p.add_argument("--alpha-tri-slow", type=float, default=1e-5)
    p.add_argument("--alpha-bi-fast", type=float, default=1e-5)
    p.add_argument("--alpha-bi-slow", type=float, default=1e-5)
    p.add_argument("--alpha-ucache-fast", type=float, default=1e-5)
    p.add_argument("--alpha-ucache-slow", type=float, default=1e-5)
    p.add_argument("--lam-tri-fast", type=float, default=0.002)
    p.add_argument("--lam-tri-slow", type=float, default=0.0002)
    p.add_argument("--lam-bi-fast", type=float, default=0.005)
    p.add_argument("--lam-bi-slow", type=float, default=0.0005)
    p.add_argument("--lam-ucache-fast", type=float, default=0.002)
    p.add_argument("--lam-ucache-slow", type=float, default=0.0002)

    p.add_argument("--W-attn-uf", type=int, default=2000)
    p.add_argument("--beta-attn-uf", type=float, default=14.0)
    p.add_argument("--theta-attn-uf", type=float, default=0.02)
    p.add_argument("--alpha-attn-uf", type=float, default=1e-5)

    p.add_argument("--W-attn-us", type=int, default=8000)
    p.add_argument("--beta-attn-us", type=float, default=8.0)
    p.add_argument("--theta-attn-us", type=float, default=0.002)
    p.add_argument("--alpha-attn-us", type=float, default=1e-5)

    p.add_argument("--W-attn-ug", type=int, default=16384)
    p.add_argument("--beta-attn-ug", type=float, default=10.0)
    p.add_argument("--theta-attn-ug", type=float, default=0.0)
    p.add_argument("--alpha-attn-ug", type=float, default=1e-5)

    p.add_argument("--W-attn-rf1", type=int, default=2000)
    p.add_argument("--beta-attn-rf1", type=float, default=12.0)
    p.add_argument("--theta-attn-rf1", type=float, default=0.01)
    p.add_argument("--alpha-attn-rf1", type=float, default=1e-5)

    p.add_argument("--W-attn-rs1", type=int, default=8000)
    p.add_argument("--beta-attn-rs1", type=float, default=8.0)
    p.add_argument("--theta-attn-rs1", type=float, default=0.004)
    p.add_argument("--alpha-attn-rs1", type=float, default=1e-5)

    p.add_argument("--W-attn-rf2", type=int, default=2000)
    p.add_argument("--beta-attn-rf2", type=float, default=14.0)
    p.add_argument("--theta-attn-rf2", type=float, default=0.02)
    p.add_argument("--alpha-attn-rf2", type=float, default=1e-5)

    p.add_argument("--W-attn-rs2", type=int, default=8000)
    p.add_argument("--beta-attn-rs2", type=float, default=10.0)
    p.add_argument("--theta-attn-rs2", type=float, default=0.004)
    p.add_argument("--alpha-attn-rs2", type=float, default=1e-5)

    p.add_argument("--W-attn-rg2", type=int, default=16384)
    p.add_argument("--beta-attn-rg2", type=float, default=14.0)
    p.add_argument("--theta-attn-rg2", type=float, default=0.0)
    p.add_argument("--alpha-attn-rg2", type=float, default=1e-5)

    p.add_argument("--W-attn-rf3", type=int, default=2000)
    p.add_argument("--beta-attn-rf3", type=float, default=20.0)
    p.add_argument("--theta-attn-rf3", type=float, default=0.04)
    p.add_argument("--alpha-attn-rf3", type=float, default=1e-5)

    p.add_argument("--W-attn-rs3", type=int, default=8000)
    p.add_argument("--beta-attn-rs3", type=float, default=14.0)
    p.add_argument("--theta-attn-rs3", type=float, default=0.004)
    p.add_argument("--alpha-attn-rs3", type=float, default=1e-5)

    p.add_argument("--ppmi-gamma", type=float, default=1.0)
    p.add_argument("--knn-k", type=int, default=16)
    p.add_argument("--knn-temp", type=float, default=0.1)

    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    bpe, vocab, tok2id, bpe_to_lm, emb, V, d = load_setup()
    emb_dev = emb.to(DEVICE).float()

    print("[load] Kneser-Ney ...")
    with open(args.kn_pickle, "rb") as f:
        kn = pickle.load(f)

    print("[load] counts ...")
    blob = torch.load(args.counts_file, map_location=DEVICE, weights_only=False)
    mu = blob["mu"].to(DEVICE)
    counts = blob["counts"].to(DEVICE)
    model = SparseMixtureClusterLM.from_counts(mu, counts, alpha=args.alpha_mix, V=V, K_pos=args.K_pos, d_emb=d)

    # Load data slices
    n_val_toks = parse_size(args.val_tokens)
    n_eval_toks = parse_size(args.eval_tokens)

    # Re-use token loaders to get contiguous slices
    tokens_raw = load_or_build_tokens(bpe, bpe_to_lm, V).numpy().astype(np.int32)
    train_split = 22_000_000
    
    val_raw = tokens_raw[train_split : train_split + n_val_toks]
    eval_raw = tokens_raw[train_split + n_val_toks : train_split + n_val_toks + n_eval_toks]

    print(f"Loaded slices: val={len(val_raw):,} | eval={len(eval_raw):,}")

    # Prebuild shape transitions
    print("[load] building word shapes...")
    shape_probs, token_shapes_t = build_shape_transition_probs_v33(tokens_raw[:train_split], bpe, V)
    shape_probs = shape_probs.float().cpu()
    token_shapes = token_shapes_t.numpy()

    # Prebuild PPMI transitions
    print("[load] building PPMI semantic matrix...")
    ppmi_probs = build_ppmi_semantic_log_probs_all(emb_dev, args.ppmi_gamma).cpu()

    for ids_n, label in [(val_raw, "val"), (eval_raw, "eval")]:
        ids_t = torch.from_numpy(ids_n).long().to(DEVICE)
        
        log_p_observed, log_p_lag1, entropy, max_log_prob, log_p_targets = compute_all_channels(
            ids_t, ids_n, V, d, emb_dev, kn, model, ppmi_probs, shape_probs, token_shapes, args, label
        )

        out_path = out_dir / f"{label}.npz"
        np.savez_compressed(
            out_path,
            log_p_observed=log_p_observed,
            log_p_lag1=log_p_lag1,
            entropy=entropy,
            max_log_prob=max_log_prob,
            observed=ids_n[args.K_pos:],
            log_p_targets=log_p_targets,
            channel_names=CHANNEL_NAMES
        )
        print(f"[{label}] -> saved {out_path} ({out_path.stat().st_size / (1024*1024):.2f} MB)")


if __name__ == "__main__":
    main()
