"""hybrid/v1_blender/sweep_big.py

Extended capacity sweep for the 500K-token training slice (data_big).
Tests larger hidden sizes and a 3-layer depth variant.
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
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from compile_wiki_lm_v13 import load_setup
from hybrid.v1_blender.blender_model import build_feature_matrix, mixture_nll
from hybrid.v1_blender.train import load_slice


class DeepBlender(nn.Module):
    """N-layer MLP with GELU + dropout; final layer zero-init for uniform start."""
    def __init__(self, in_dim: int, n_channels: int, hidden: int, depth: int = 2,
                 dropout: float = 0.0):
        super().__init__()
        layers = []
        d_in = in_dim
        for _ in range(depth):
            layers += [nn.Linear(d_in, hidden), nn.GELU(), nn.Dropout(dropout)]
            d_in = hidden
        head = nn.Linear(hidden, n_channels)
        nn.init.zeros_(head.weight)
        nn.init.zeros_(head.bias)
        layers.append(head)
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return F.log_softmax(self.net(x), dim=-1)


@torch.no_grad()
def eval_ppl(model, feats, log_p_targets, batch=8192):
    model.eval()
    nll_sum = 0.0
    n = feats.shape[0]
    for s in range(0, n, batch):
        e = min(s + batch, n)
        nll = mixture_nll(model(feats[s:e]), log_p_targets[s:e])
        nll_sum += nll.sum().item()
    return math.exp(nll_sum / n)


def train_one(feats_train, log_p_train, feats_inval, log_p_inval,
              hidden, depth, dropout, lr, wd, epochs, batch, n_channels, in_dim,
              device, patience=15):
    torch.manual_seed(0)
    model = DeepBlender(in_dim, n_channels, hidden=hidden, depth=depth,
                        dropout=dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    best_ppl = float("inf")
    best_state = None
    no_improve = 0
    n_train = feats_train.shape[0]
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n_train, device=device)
        for s in range(0, n_train, batch):
            b = perm[s:s + batch]
            loss = mixture_nll(model(feats_train[b]), log_p_train[b]).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        ppl = eval_ppl(model, feats_inval, log_p_inval)
        if ppl < best_ppl - 1e-3:
            best_ppl = ppl
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break
    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    return model, best_ppl, ep + 1


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="hybrid/v1_blender/data_big")
    p.add_argument("--out", default="hybrid/v1_blender/data_big/sweep_big_capacity.json")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    data_dir = Path(args.data_dir)
    val = load_slice(data_dir / "val.npz")
    ev = load_slice(data_dir / "eval.npz")
    _, _, _, _, emb, V, d = load_setup()
    emb = emb.float()
    feats_val = build_feature_matrix(
        val["log_p_observed"], val["log_p_lag1"], val["entropy"], val["max_log_prob"],
        emb, val["observed"], use_embedding=True,
        topk_log_probs=val.get("topk_log_probs"))
    feats_eval = build_feature_matrix(
        ev["log_p_observed"], ev["log_p_lag1"], ev["entropy"], ev["max_log_prob"],
        emb, ev["observed"], use_embedding=True,
        topk_log_probs=ev.get("topk_log_probs"))
    log_p_targets_val = val["log_p_targets"]
    log_p_targets_eval = ev["log_p_targets"]
    in_dim = feats_val.shape[1]
    C = log_p_targets_val.shape[1]

    device = torch.device(args.device)
    feats_val = feats_val.to(device); feats_eval = feats_eval.to(device)
    log_p_targets_val = log_p_targets_val.to(device)
    log_p_targets_eval = log_p_targets_eval.to(device)

    T = feats_val.shape[0]
    torch.manual_seed(0)
    perm = torch.randperm(T, device=device)
    n_in = max(1, int(T * 0.2))
    inval_idx = perm[:n_in]; train_idx = perm[n_in:]
    feats_train = feats_val[train_idx]; log_p_train = log_p_targets_val[train_idx]
    feats_inval = feats_val[inval_idx]; log_p_inval = log_p_targets_val[inval_idx]

    configs = []
    # Big-capacity grid
    for hidden in [256, 512, 1024, 2048]:
        for depth in [2, 3]:
            for dropout in [0.1, 0.2, 0.3]:
                for wd in [1e-4, 1e-3]:
                    configs.append({"hidden": hidden, "depth": depth,
                                    "dropout": dropout, "lr": 1e-3,
                                    "weight_decay": wd})
    print(f"[sweep_big_capacity] {len(configs)} configs, in_dim={in_dim}, C={C}, "
          f"n_train={feats_train.shape[0]:,}")
    results = []
    best = {"heldout_ppl": float("inf")}
    t0 = time.time()
    for i, cfg in enumerate(configs):
        try:
            model, inval_ppl, n_ep = train_one(
                feats_train, log_p_train, feats_inval, log_p_inval,
                hidden=cfg["hidden"], depth=cfg["depth"], dropout=cfg["dropout"],
                lr=cfg["lr"], wd=cfg["weight_decay"], epochs=120, batch=8192,
                n_channels=C, in_dim=in_dim, device=device)
            heldout_ppl = eval_ppl(model, feats_eval, log_p_targets_eval)
            row = {**cfg, "inval_ppl": inval_ppl, "heldout_ppl": heldout_ppl,
                   "n_epochs": n_ep}
            if heldout_ppl < best["heldout_ppl"]:
                best = row
            print(f"  [{i+1:2d}/{len(configs)}] h={cfg['hidden']:4d} d={cfg['depth']} "
                  f"dr={cfg['dropout']:.1f} wd={cfg['weight_decay']:.0e} "
                  f"ep={n_ep:3d} -> inval {inval_ppl:6.2f} | "
                  f"heldout {heldout_ppl:6.2f}  ({time.time()-t0:.0f}s)")
        except torch.cuda.OutOfMemoryError:
            row = {**cfg, "error": "OOM"}
            print(f"  [{i+1:2d}/{len(configs)}] OOM h={cfg['hidden']} d={cfg['depth']}")
            torch.cuda.empty_cache()
        results.append(row)
    print(f"\n[best] heldout PPL = {best['heldout_ppl']:.3f} with {best}")
    with open(args.out, "w") as f:
        json.dump({"results": results, "best": best}, f, indent=2)
    print(f"[save] {args.out}")


if __name__ == "__main__":
    main()
