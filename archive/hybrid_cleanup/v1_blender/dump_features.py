"""hybrid/v1_blender/dump_features.py

Recompute the 12 v31 channels on a token slice and dump compact per-position
arrays suitable for training a tiny mixture blender:

  features.npy       (T, F)   float32  -- per-position input features (no leak)
  log_p_targets.npy  (T, C)   float32  -- per-channel log-prob on TRUE next token
  log_p_observed.npy (T, C)   float32  -- per-channel log-prob on observed x_t (feature)
  entropy.npy        (T, C)   float32  -- per-channel entropy (feature)
  max_prob.npy       (T, C)   float32  -- per-channel max prob (feature)
  targets.npy        (T,)     int64    -- the true next-token ids
  observed.npy       (T,)     int64    -- the observed current tokens x_t
  meta.json                              -- channel names, V, K_pos, source slice info

A second pass for the heldout slice produces the same files with _eval suffix.

Usage:
    python hybrid/v1_blender/dump_features.py \\
        --kn-pickle artifacts/compiled_wiki_lm_v23/kn7_22m.pkl \\
        --counts-file artifacts/compiled_wiki_lm_v14/counts_k2_c64k.pt \\
        --out-dir hybrid/v1_blender/data \\
        --val-tokens 30K --eval-tokens 100K

Use --smoke for a tiny (1K val + 2K eval) end-to-end pipeline test.
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

from compile_wiki_lm_v13 import (
    load_setup, load_or_build_tokens, build_residual, parse_size, DEVICE,
)
from compile_wiki_lm_v14 import SparseMixtureClusterLM
from compile_wiki_lm_v23 import ModifiedKNGram  # noqa: F401 -- required for pickle.load
from compile_wiki_lm_v24 import compute_log_p_kn
from compile_wiki_lm_v31 import (
    build_v31_induction_log_probs,
    compute_log_p_attn_unigram,
    compute_log_p_attn_residual,
)

CHANNEL_NAMES = [
    "kn", "mix", "tri_f", "tri_s", "bi_f", "bi_s",
    "uc_f", "uc_s", "att_uf", "att_us", "att_rf", "att_rs",
]
C = len(CHANNEL_NAMES)


def compute_mix_log_probs(ids_t: torch.Tensor, emb_dev: torch.Tensor,
                          model: SparseMixtureClusterLM, K_pos: int,
                          top_M: int, tau: float, gamma: float) -> np.ndarray:
    """Port of v31.main's compute_log_p_mix_low_mem (it was nested in main)."""
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


def compute_all_channels(ids_t: torch.Tensor, ids_n: np.ndarray, V: int,
                         emb_dev: torch.Tensor, kn, model: SparseMixtureClusterLM,
                         args, label: str) -> tuple[list[np.ndarray], np.ndarray]:
    print(f"\n[{label}] computing 12 channels over {len(ids_n):,} tokens")

    t0 = time.time()
    log_p_mix = compute_mix_log_probs(ids_t, emb_dev, model, args.K_pos,
                                       args.top_M, args.tau, args.gamma)
    print(f"  mix     {log_p_mix.shape} ({time.time()-t0:.1f}s)")

    t0 = time.time()
    log_p_kn = compute_log_p_kn(kn, ids_n, args.K_pos)
    print(f"  kn      {log_p_kn.shape} ({time.time()-t0:.1f}s)")

    t0 = time.time()
    lp_trif, lp_tris, lp_bif, lp_bis, lp_ucf, lp_ucs = build_v31_induction_log_probs(
        ids_n, V, args.K_pos, args.window,
        args.lam_tri_fast, args.lam_tri_slow,
        args.lam_bi_fast, args.lam_bi_slow,
        args.lam_ucache_fast, args.lam_ucache_slow,
        args.alpha_tri_fast, args.alpha_tri_slow,
        args.alpha_bi_fast, args.alpha_bi_slow,
        args.alpha_ucache_fast, args.alpha_ucache_slow,
    )
    print(f"  decay caches done ({time.time()-t0:.1f}s)")

    t0 = time.time()
    print(f"  attn unigram (uf, us) ...")
    lp_attn_uf = compute_log_p_attn_unigram(
        ids_t, emb_dev, args.W_attn_uf, args.beta_attn_uf,
        args.theta_attn_uf, args.alpha_attn_uf, args.K_pos,
    ).numpy()
    lp_attn_us = compute_log_p_attn_unigram(
        ids_t, emb_dev, args.W_attn_us, args.beta_attn_us,
        args.theta_attn_us, args.alpha_attn_us, args.K_pos,
    ).numpy()
    print(f"  attn residual (rf, rs) ...")
    lp_attn_rf = compute_log_p_attn_residual(
        ids_t, emb_dev, args.W_attn_rf, args.beta_attn_rf,
        args.theta_attn_rf, args.alpha_attn_rf, args.K_pos,
    ).numpy()
    lp_attn_rs = compute_log_p_attn_residual(
        ids_t, emb_dev, args.W_attn_rs, args.beta_attn_rs,
        args.theta_attn_rs, args.alpha_attn_rs, args.K_pos,
    ).numpy()
    print(f"  attention done ({time.time()-t0:.1f}s)")

    targets = ids_n[args.K_pos:]
    channels = [log_p_kn, log_p_mix, lp_trif, lp_tris, lp_bif, lp_bis,
                lp_ucf, lp_ucs, lp_attn_uf, lp_attn_us, lp_attn_rf, lp_attn_rs]

    # All channels should share T = len(ids) - K_pos.  Sanity check.
    T = log_p_kn.shape[0]
    for i, ch in enumerate(channels):
        assert ch.shape == (T, V), f"channel {CHANNEL_NAMES[i]} shape {ch.shape} != ({T}, {V})"
    assert targets.shape == (T,)
    return channels, targets.astype(np.int64)


def summarize(channels: list[np.ndarray], targets: np.ndarray, ids_n: np.ndarray, K_pos: int, top_k: int = 3):
    """Collapse the 12 (T,V) arrays into compact per-position summary stats.

    Returns dict of float32 arrays:
        log_p_targets   (T, C)  log p_c(y_t)
        log_p_observed  (T, C)  log p_c(x_t)   (lag-0 self-consistency, no leak)
        log_p_lag1      (T, C)  log p_c(x_{t-1}) (lag-1 self-consistency, no leak)
        entropy         (T, C)  H(p_c)
        max_log_prob    (T, C)  max_v log p_c(v)
        top1_id         (T, C)  argmax_v p_c(v)        (int32)
        topk_log_probs  (T, C, K) top-K log-probs per channel (sorted desc) -- distribution shape
    Targets:
        targets   (T,)
        observed  (T,)
    """
    T, V = channels[0].shape
    C = len(channels)
    log_p_targets = np.zeros((T, C), dtype=np.float32)
    log_p_observed = np.zeros((T, C), dtype=np.float32)
    log_p_lag1 = np.zeros((T, C), dtype=np.float32)
    entropy = np.zeros((T, C), dtype=np.float32)
    max_log_prob = np.zeros((T, C), dtype=np.float32)
    top1_id = np.zeros((T, C), dtype=np.int32)
    topk_log_probs = np.zeros((T, C, top_k), dtype=np.float32)

    # Observed token at position t is ids_n[K_pos + t] (the same as target shifted by -1)
    # Actually: channel index t in (T, V) corresponds to predicting ids_n[K_pos + t]
    # The "observed current" token at decode step t is ids_n[K_pos + t - 1].
    # The lag-1 observed is ids_n[K_pos + t - 2].
    observed = ids_n[K_pos - 1 : K_pos - 1 + T].astype(np.int64)
    lag1 = np.concatenate([[observed[0]], observed[:-1]]).astype(np.int64)

    idx = np.arange(T)
    for c, lp in enumerate(channels):
        log_p_targets[:, c] = lp[idx, targets]
        log_p_observed[:, c] = lp[idx, observed]
        log_p_lag1[:, c] = lp[idx, lag1]
        # Per-chunk: entropy, max, top-K
        chunk = 4096
        for s in range(0, T, chunk):
            e = min(s + chunk, T)
            lp_c = lp[s:e]
            p_c = np.exp(lp_c)
            entropy[s:e, c] = -(p_c * lp_c).sum(axis=1)
            # top-K (sorted desc)
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


def dump(slice_name: str, ids_t: torch.Tensor, ids_n: np.ndarray, V: int,
         emb_dev: torch.Tensor, kn, model, args, out_dir: Path):
    channels, targets = compute_all_channels(
        ids_t, ids_n, V, emb_dev, kn, model, args, slice_name,
    )
    summary = summarize(channels, targets, ids_n, args.K_pos)
    del channels  # free GPU/CPU memory before saving
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
    p.add_argument("--out-dir", type=str, default="hybrid/v1_blender/data")
    p.add_argument("--train-tokens", type=str, default="22M")
    p.add_argument("--val-tokens", type=str, default="30K")
    p.add_argument("--eval-tokens", type=str, default="100K")
    p.add_argument("--smoke", action="store_true",
                   help="Tiny slice (val=1K, eval=2K) for end-to-end pipeline test.")

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

    p.add_argument("--W-attn-uf", type=int, default=1000)
    p.add_argument("--beta-attn-uf", type=float, default=14.0)
    p.add_argument("--theta-attn-uf", type=float, default=0.02)
    p.add_argument("--alpha-attn-uf", type=float, default=1e-5)
    p.add_argument("--W-attn-us", type=int, default=4000)
    p.add_argument("--beta-attn-us", type=float, default=8.0)
    p.add_argument("--theta-attn-us", type=float, default=0.002)
    p.add_argument("--alpha-attn-us", type=float, default=1e-5)
    p.add_argument("--W-attn-rf", type=int, default=1000)
    p.add_argument("--beta-attn-rf", type=float, default=18.0)
    p.add_argument("--theta-attn-rf", type=float, default=0.03)
    p.add_argument("--alpha-attn-rf", type=float, default=1e-5)
    p.add_argument("--W-attn-rs", type=int, default=4000)
    p.add_argument("--beta-attn-rs", type=float, default=12.0)
    p.add_argument("--theta-attn-rs", type=float, default=0.003)
    p.add_argument("--alpha-attn-rs", type=float, default=1e-5)
    args = p.parse_args()

    if args.smoke:
        args.val_tokens = "1K"
        args.eval_tokens = "2K"
        # Shorter attention windows in smoke mode so tokenization+attn finishes fast.
        args.W_attn_uf = 200
        args.W_attn_us = 400
        args.W_attn_rf = 200
        args.W_attn_rs = 400

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

    print(f"[hybrid v1] dumping per-position features")
    print(f"[split] train_n={train_n:,}  val={val_n:,}  eval={eval_n:,}")

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

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "channel_names": CHANNEL_NAMES,
        "V": int(V),
        "K_pos": int(args.K_pos),
        "train_n": int(train_n),
        "val_n": int(val_n),
        "eval_n": int(eval_n),
        "smoke": bool(args.smoke),
    }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    dump("val", val_ids_t, val_ids_n, V, emb_dev, kn, model, args, out_dir)
    dump("eval", eval_ids_t, eval_ids_n, V, emb_dev, kn, model, args, out_dir)
    print("[done] features dumped")


if __name__ == "__main__":
    main()
