"""hybrid/v1_blender/train.py

Train TinyBlender on dumped val features.  Reports val NLL/PPL during training.

Usage:
    python hybrid/v1_blender/train.py \\
        --data-dir hybrid/v1_blender/data \\
        --out hybrid/v1_blender/data/blender.pt \\
        --epochs 200 --batch 4096 --lr 1e-3
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
import torch.nn as nn

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from compile_wiki_lm_v13 import load_setup
from hybrid.v1_blender.blender_model import (
    TinyBlender, build_feature_matrix, mixture_nll,
)


def load_slice(path: Path):
    npz = np.load(path, allow_pickle=True)
    out = {
        "log_p_targets": torch.from_numpy(npz["log_p_targets"]),
        "log_p_observed": torch.from_numpy(npz["log_p_observed"]),
        "log_p_lag1": torch.from_numpy(npz["log_p_lag1"]),
        "entropy": torch.from_numpy(npz["entropy"]),
        "max_log_prob": torch.from_numpy(npz["max_log_prob"]),
        "targets": torch.from_numpy(npz["targets"]),
        "observed": torch.from_numpy(npz["observed"]),
    }
    if "topk_log_probs" in npz.files:
        out["topk_log_probs"] = torch.from_numpy(npz["topk_log_probs"])
    return out


def evaluate(model: TinyBlender, features: torch.Tensor, log_p_targets: torch.Tensor,
             batch: int = 8192) -> tuple[float, float]:
    """Returns (mean_nll, ppl)."""
    model.eval()
    nll_sum = 0.0
    n = features.shape[0]
    with torch.no_grad():
        for s in range(0, n, batch):
            e = min(s + batch, n)
            log_w = model(features[s:e])
            nll = mixture_nll(log_w, log_p_targets[s:e])
            nll_sum += nll.sum().item()
    mean_nll = nll_sum / n
    return mean_nll, math.exp(mean_nll)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=str, default="hybrid/v1_blender/data")
    p.add_argument("--out", type=str, default="hybrid/v1_blender/data/blender.pt")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch", type=int, default=4096)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--val-frac", type=float, default=0.2,
                   help="Fraction of val slice held out as in-training val.")
    p.add_argument("--no-embedding", action="store_true",
                   help="Drop the PPMI+SVD token embedding from features (smaller model).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    data_dir = Path(args.data_dir)
    print(f"[load] {data_dir / 'val.npz'}")
    val = load_slice(data_dir / "val.npz")

    print("[load] PPMI+SVD embedding ...")
    _bpe, _vocab, _tok2id, _bpe_to_lm, emb, V, d = load_setup()
    emb = emb.float()

    use_emb = not args.no_embedding
    features = build_feature_matrix(
        val["log_p_observed"], val["log_p_lag1"],
        val["entropy"], val["max_log_prob"],
        emb, val["observed"], use_embedding=use_emb,
        topk_log_probs=val.get("topk_log_probs"),
    )
    print(f"[features] shape={tuple(features.shape)}  C={val['log_p_targets'].shape[1]}  V={V}")

    # Train/val-of-val split
    T = features.shape[0]
    n_val = max(1, int(T * args.val_frac))
    idx = torch.randperm(T)
    train_idx = idx[n_val:]
    inval_idx = idx[:n_val]

    device = torch.device(args.device)
    features = features.to(device)
    log_p_targets = val["log_p_targets"].to(device)

    n_channels = log_p_targets.shape[1]
    model = TinyBlender(features.shape[1], n_channels, hidden=args.hidden,
                        dropout=args.dropout).to(device)

    # Sanity: with uniform mixing weights (the init), the loss equals
    #   -log(1/C * sum_c P_c(y))
    # which should be close to log(C) above the best single channel's NLL.
    init_nll, init_ppl = evaluate(model, features[inval_idx], log_p_targets[inval_idx])
    print(f"[init] uniform-mix val nll={init_nll:.4f}  ppl={init_ppl:.2f}")

    # Per-channel reference (set w to one-hot on each channel)
    print("[ref ] per-channel NLL on inval slice:")
    with torch.no_grad():
        for c in range(n_channels):
            log_w = torch.full_like(log_p_targets[inval_idx], -1e30)
            log_w[:, c] = 0.0
            nll = mixture_nll(log_w, log_p_targets[inval_idx]).mean().item()
            print(f"  ch{c:2d}  ppl={math.exp(nll):8.2f}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_ppl = float("inf")
    best_state = None
    log = []
    t0 = time.time()
    for epoch in range(args.epochs):
        model.train()
        perm = train_idx[torch.randperm(train_idx.shape[0])]
        ep_nll = 0.0
        ep_n = 0
        for s in range(0, perm.shape[0], args.batch):
            e = min(s + args.batch, perm.shape[0])
            b = perm[s:e]
            log_w = model(features[b])
            nll = mixture_nll(log_w, log_p_targets[b])
            loss = nll.mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ep_nll += nll.sum().item()
            ep_n += b.shape[0]
        train_nll = ep_nll / ep_n
        v_nll, v_ppl = evaluate(model, features[inval_idx], log_p_targets[inval_idx])
        log.append({"epoch": epoch, "train_nll": train_nll, "val_nll": v_nll, "val_ppl": v_ppl})
        if v_ppl < best_ppl:
            best_ppl = v_ppl
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if epoch % max(1, args.epochs // 20) == 0 or epoch == args.epochs - 1:
            print(f"  ep {epoch:4d}  train_ppl={math.exp(train_nll):7.2f}  "
                  f"val_ppl={v_ppl:7.2f}  best={best_ppl:7.2f}  "
                  f"({time.time()-t0:.1f}s)")

    print(f"[done] best in-val ppl={best_ppl:.2f}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": best_state,
        "args": vars(args),
        "use_embedding": use_emb,
        "in_dim": features.shape[1],
        "n_channels": n_channels,
        "log": log,
        "best_val_ppl": best_ppl,
        "V": int(V),
    }, str(out_path))
    print(f"[save] {out_path}")


if __name__ == "__main__":
    main()
