"""hybrid/v3_super_blender/dump_features_v32.py

Recompute the 18 v32 channels on a token slice and dump compact per-position
arrays suitable for training active super blenders.

Usage:
    python hybrid/v3_super_blender/dump_features_v32.py \\
        --kn-pickle artifacts/compiled_wiki_lm_v23/kn7_22m.pkl \\
        --counts-file artifacts/compiled_wiki_lm_v14/counts_k2_c64k.pt \\
        --out-dir hybrid/v3_super_blender/data_real \\
        --val-tokens 30K --eval-tokens 100K
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

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, "/home/drawson/llm_decoupling")

from compile_wiki_lm_v13 import (
    load_setup, load_or_build_tokens, build_residual, parse_size, DEVICE,
)
from compile_wiki_lm_v14 import SparseMixtureClusterLM
from compile_wiki_lm_v23 import ModifiedKNGram  # noqa: F401
from compile_wiki_lm_v24 import compute_log_p_kn
from compile_wiki_lm_v32 import (
    build_v32_induction_log_probs,
    compute_log_p_attn_unigram,
    compute_log_p_attn_residual_sliced,
)

CHANNEL_NAMES = [
    "kn", "mix",
    "tri_f", "tri_s", "bi_f", "bi_s", "uc_f", "uc_s",
    "attn_uf", "attn_us", "attn_ug",
    "attn_rf1", "attn_rs1",
    "attn_rf2", "attn_rs2", "attn_rg2",
    "attn_rf3", "attn_rs3"
]
C = len(CHANNEL_NAMES)


def compute_mix_log_probs(ids_t: torch.Tensor, emb_dev: torch.Tensor,
                          model: SparseMixtureClusterLM, K_pos: int,
                          top_M: int, tau: float, gamma: float) -> np.ndarray:
    V_t = model.V
    N_t = ids_t.shape[0]
    r = build_residual(ids_t.to(emb_dev.device).long(), emb_dev, K_pos)
    out_chunks = []
    start_t = K_pos - 1
    end_t = N_t - 1
    mu_sq = (model.mu * model.mu).sum(dim=1)
    chunk_size = 400
    for s in range(start_t, end_t, chunk_size):
        e = min(s + chunk_size, end_t)
        r_c = r[s:e].to(DEVICE)
        r_sq = (r_c * r_c).sum(dim=1, keepdim=True)
        d2 = r_sq + mu_sq.unsqueeze(0) - 2 * (r_c @ model.mu.T)
        if top_M and top_M < model.mu.shape[0]:
            _, idx = d2.topk(top_M, dim=1, largest=False)
            d2_top = d2.gather(1, idx)
            log_pi = F.log_softmax(-d2_top / tau, dim=1)
            log_p_top = model.log_p_cluster[idx].float()
            log_mix = torch.logsumexp(log_pi.unsqueeze(2) + log_p_top, dim=1)
        else:
            log_pi = F.log_softmax(-d2 / tau, dim=1)
            log_mix = torch.logsumexp(log_pi.unsqueeze(2) + model.log_p_cluster.float().unsqueeze(0), dim=1)
        if gamma < 1.0:
            log_p = torch.logaddexp(
                math.log(gamma) + log_mix,
                math.log(1 - gamma) + model.log_p_uni.float().unsqueeze(0),
            )
        else:
            log_p = log_mix
        out_chunks.append(log_p.cpu())
    return torch.cat(out_chunks, dim=0).numpy()


def compute_all_channels(ids_t: torch.Tensor, ids_n: np.ndarray, V: int, d: int,
                         emb_dev: torch.Tensor, kn, model: SparseMixtureClusterLM,
                         args, label: str) -> tuple[list[np.ndarray], np.ndarray]:
    print(f"\n[{label}] computing 18 channels over {len(ids_n):,} tokens")

    # 1. Cluster mixture
    t0 = time.time()
    log_p_mix = compute_mix_log_probs(ids_t, emb_dev, model, args.K_pos,
                                       args.top_M, args.tau, args.gamma)
    print(f"  (1/18) mix done ({time.time()-t0:.1f}s)")

    # 2. Global KN7
    t0 = time.time()
    log_p_kn = compute_log_p_kn(kn, ids_n, args.K_pos)
    print(f"  (2/18) KN done ({time.time()-t0:.1f}s)")

    # 3. Dynamic caches
    t0 = time.time()
    lp_trif, lp_tris, lp_bif, lp_bis, lp_ucf, lp_ucs = build_v32_induction_log_probs(
        ids_n, V, args.K_pos, args.window,
        args.lam_tri_fast, args.lam_tri_slow,
        args.lam_bi_fast, args.lam_bi_slow,
        args.lam_ucache_fast, args.lam_ucache_slow,
        args.alpha_tri_fast, args.alpha_tri_slow,
        args.alpha_bi_fast, args.alpha_bi_slow,
        args.alpha_ucache_fast, args.alpha_ucache_slow,
    )
    print(f"  (3-8/18) decay caches done ({time.time()-t0:.1f}s)")

    # 4. Unigram Attention Caches (uf, us, ug)
    t0 = time.time()
    print(f"  computing unigram attention caches...")
    lp_attn_uf = compute_log_p_attn_unigram(
        ids_t, emb_dev, args.W_attn_uf, args.beta_attn_uf, args.theta_attn_uf, args.alpha_attn_uf, args.K_pos
    ).numpy()
    lp_attn_us = compute_log_p_attn_unigram(
        ids_t, emb_dev, args.W_attn_us, args.beta_attn_us, args.theta_attn_us, args.alpha_attn_us, args.K_pos
    ).numpy()
    lp_attn_ug = compute_log_p_attn_unigram(
        ids_t, emb_dev, args.W_attn_ug, args.beta_attn_ug, args.theta_attn_ug, args.alpha_attn_ug, args.K_pos
    ).numpy()
    print(f"  (9-11/18) unigram attention done ({time.time()-t0:.1f}s)")

    # 5. Core Phrase Residual construction
    t0 = time.time()
    print(f"  building core phrase residuals...")
    r_full = build_residual(ids_t.to(DEVICE).long(), emb_dev, K=3)

    # 6. Residual Attention Caches
    print(f"  computing multi-scale state attention caches...")
    
    # Residual K_pos=1 (Bigram)
    lp_attn_rf1 = compute_log_p_attn_residual_sliced(
        ids_t, r_full, d, K=1, W_attn=args.W_attn_rf1, beta=args.beta_attn_rf1, theta=args.theta_attn_rf1, alpha_attn=args.alpha_attn_rf1, K_pos=args.K_pos, V=V
    ).numpy()
    lp_attn_rs1 = compute_log_p_attn_residual_sliced(
        ids_t, r_full, d, K=1, W_attn=args.W_attn_rs1, beta=args.beta_attn_rs1, theta=args.theta_attn_rs1, alpha_attn=args.alpha_attn_rs1, K_pos=args.K_pos, V=V
    ).numpy()
    
    # Residual K_pos=2 (Trigram)
    lp_attn_rf2 = compute_log_p_attn_residual_sliced(
        ids_t, r_full, d, K=2, W_attn=args.W_attn_rf2, beta=args.beta_attn_rf2, theta=args.theta_attn_rf2, alpha_attn=args.alpha_attn_rf2, K_pos=args.K_pos, V=V
    ).numpy()
    lp_attn_rs2 = compute_log_p_attn_residual_sliced(
        ids_t, r_full, d, K=2, W_attn=args.W_attn_rs2, beta=args.beta_attn_rs2, theta=args.theta_attn_rs2, alpha_attn=args.alpha_attn_rs2, K_pos=args.K_pos, V=V
    ).numpy()
    lp_attn_rg2 = compute_log_p_attn_residual_sliced(
        ids_t, r_full, d, K=2, W_attn=args.W_attn_rg2, beta=args.beta_attn_rg2, theta=args.theta_attn_rg2, alpha_attn=args.alpha_attn_rg2, K_pos=args.K_pos, V=V
    ).numpy()
    
    # Residual K_pos=3 (Fourgram)
    lp_attn_rf3 = compute_log_p_attn_residual_sliced(
        ids_t, r_full, d, K=3, W_attn=args.W_attn_rf3, beta=args.beta_attn_rf3, theta=args.theta_attn_rf3, alpha_attn=args.alpha_attn_rf3, K_pos=args.K_pos, V=V
    ).numpy()
    lp_attn_rs3 = compute_log_p_attn_residual_sliced(
        ids_t, r_full, d, K=3, W_attn=args.W_attn_rs3, beta=args.beta_attn_rs3, theta=args.theta_attn_rs3, alpha_attn=args.alpha_attn_rs3, K_pos=args.K_pos, V=V
    ).numpy()
    print(f"  (12-18/18) multi-scale state attention done ({time.time()-t0:.1f}s)")

    targets = ids_n[args.K_pos:]
    channels = [
        log_p_kn, log_p_mix,
        lp_trif, lp_tris, lp_bif, lp_bis, lp_ucf, lp_ucs,
        lp_attn_uf, lp_attn_us, lp_attn_ug,
        lp_attn_rf1, lp_attn_rs1,
        lp_attn_rf2, lp_attn_rs2, lp_attn_rg2,
        lp_attn_rf3, lp_attn_rs3
    ]

    T = log_p_kn.shape[0]
    for i, ch in enumerate(channels):
        assert ch.shape == (T, V), f"channel {CHANNEL_NAMES[i]} shape {ch.shape} != ({T}, {V})"
    assert targets.shape == (T,)
    return channels, targets.astype(np.int64)


def summarize(channels: list[np.ndarray], targets: np.ndarray, ids_n: np.ndarray, K_pos: int, top_k: int = 3):
    T, V = channels[0].shape
    C = len(channels)
    log_p_targets = np.zeros((T, C), dtype=np.float32)
    log_p_observed = np.zeros((T, C), dtype=np.float32)
    log_p_lag1 = np.zeros((T, C), dtype=np.float32)
    entropy = np.zeros((T, C), dtype=np.float32)
    max_log_prob = np.zeros((T, C), dtype=np.float32)
    top1_id = np.zeros((T, C), dtype=np.int32)
    topk_log_probs = np.zeros((T, C, top_k), dtype=np.float32)

    observed = ids_n[K_pos - 1 : K_pos - 1 + T].astype(np.int64)
    lag1 = np.concatenate([[observed[0]], observed[:-1]]).astype(np.int64)

    idx = np.arange(T)
    for c, lp in enumerate(channels):
        log_p_targets[:, c] = lp[idx, targets]
        log_p_observed[:, c] = lp[idx, observed]
        log_p_lag1[:, c] = lp[idx, lag1]
        chunk = 4096
        for s in range(0, T, chunk):
            e = min(s + chunk, T)
            lp_c = lp[s:e]
            p_c = np.exp(lp_c)
            entropy[s:e, c] = -(p_c * lp_c).sum(axis=1)
            part = np.argpartition(-lp_c, top_k, axis=1)[:, :top_k]
            rows = np.arange(e - s)[:, None]
            topk_vals = lp_c[rows, part]
            order = np.argsort(-topk_vals, axis=1)
            topk_sorted = np.take_along_axis(topk_vals, order, axis=1)
            topk_log_probs[s:e, c, :] = topk_sorted
            max_log_prob[s:e, c] = topk_sorted[:, 0]
            top1_id[s:e, c] = np.take_along_axis(part, order, axis=1)[:, 0].astype(np.int32)

    return {
        "log_p_targets": log_p_targets,
        "log_p_observed": log_p_observed,
        "log_p_lag1": log_p_lag1,
        "entropy": entropy,
        "max_log_prob": max_log_prob,
        "top1_id": top1_id,
        "topk_log_probs": topk_log_probs,
        "targets": targets,
        "observed": observed,
    }


def dump(slice_name: str, ids_t: torch.Tensor, ids_n: np.ndarray, V: int, d: int,
         emb_dev: torch.Tensor, kn, model, args, out_dir: Path):
    channels, targets = compute_all_channels(
        ids_t, ids_n, V, d, emb_dev, kn, model, args, slice_name,
    )
    summary = summarize(channels, targets, ids_n, args.K_pos)
    del channels
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_dir / f"{slice_name}.npz",
        **summary,
        channel_names=np.array(CHANNEL_NAMES),
    )
    print(f"[{slice_name}] dumped to {out_dir / f'{slice_name}.npz'}")
    print(f"  T={len(targets):,}  C={len(CHANNEL_NAMES)}  V={V}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--kn-pickle", type=str, required=True)
    p.add_argument("--counts-file", type=str, required=True)
    p.add_argument("--out-dir", type=str, default="hybrid/v3_super_blender/data_real")
    p.add_argument("--train-tokens", type=str, default="22M")
    p.add_argument("--val-tokens", type=str, default="30K")
    p.add_argument("--eval-tokens", type=str, default="100K")
    p.add_argument("--smoke", action="store_true")

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

    # Unigram Attention Caches
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

    # Residual Attention Caches (K=1, Bigram space)
    p.add_argument("--W-attn-rf1", type=int, default=2000)
    p.add_argument("--beta-attn-rf1", type=float, default=16.0)
    p.add_argument("--theta-attn-rf1", type=float, default=0.02)
    p.add_argument("--alpha-attn-rf1", type=float, default=1e-5)

    p.add_argument("--W-attn-rs1", type=int, default=8000)
    p.add_argument("--beta-attn-rs1", type=float, default=10.0)
    p.add_argument("--theta-attn-rs1", type=float, default=0.002)
    p.add_argument("--alpha-attn-rs1", type=float, default=1e-5)

    # Residual Attention Caches (K=2, Trigram space)
    p.add_argument("--W-attn-rf2", type=int, default=2000)
    p.add_argument("--beta-attn-rf2", type=float, default=18.0)
    p.add_argument("--theta-attn-rf2", type=float, default=0.03)
    p.add_argument("--alpha-attn-rf2", type=float, default=1e-5)

    p.add_argument("--W-attn-rs2", type=int, default=8000)
    p.add_argument("--beta-attn-rs2", type=float, default=12.0)
    p.add_argument("--theta-attn-rs2", type=float, default=0.003)
    p.add_argument("--alpha-attn-rs2", type=float, default=1e-5)

    p.add_argument("--W-attn-rg2", type=int, default=16384)
    p.add_argument("--beta-attn-rg2", type=float, default=14.0)
    p.add_argument("--theta-attn-rg2", type=float, default=0.0)
    p.add_argument("--alpha-attn-rg2", type=float, default=1e-5)

    # Residual Attention Caches (K=3, Fourgram space)
    p.add_argument("--W-attn-rf3", type=int, default=2000)
    p.add_argument("--beta-attn-rf3", type=float, default=20.0)
    p.add_argument("--theta-attn-rf3", type=float, default=0.04)
    p.add_argument("--alpha-attn-rf3", type=float, default=1e-5)

    p.add_argument("--W-attn-rs3", type=int, default=8000)
    p.add_argument("--beta-attn-rs3", type=float, default=14.0)
    p.add_argument("--theta-attn-rs3", type=float, default=0.004)
    p.add_argument("--alpha-attn-rs3", type=float, default=1e-5)

    args = p.parse_args()

    train_n = parse_size(args.train_tokens)
    val_n = parse_size(args.val_tokens) if not args.smoke else 1000
    eval_n = parse_size(args.eval_tokens) if not args.smoke else 2000

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

    print(f"[load] KN  {args.kn_pickle}")
    with open(args.kn_pickle, "rb") as f:
        kn = pickle.load(f)

    blob = torch.load(args.counts_file, map_location=DEVICE, weights_only=False)
    mu = blob["mu"].to(DEVICE)
    counts = blob["counts"].to(DEVICE)
    model = SparseMixtureClusterLM.from_counts(mu, counts, alpha=args.alpha_mix,
                                                V=V, K_pos=args.K_pos, d_emb=d)

    out_dir = Path(args.out_dir)
    dump("val", val_ids_t, val_ids_n, V, d, emb_dev, kn, model, args, out_dir)
    dump("eval", eval_ids_t, eval_ids_n, V, d, emb_dev, kn, model, args, out_dir)


if __name__ == "__main__":
    main()
