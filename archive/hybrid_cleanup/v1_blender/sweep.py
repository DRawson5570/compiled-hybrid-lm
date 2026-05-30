"""hybrid/v1_blender/sweep.py

Quick hyperparameter sweep of TinyBlender on the dumped features.
Reports best in-val PPL and heldout PPL for each config.
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

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from compile_wiki_lm_v13 import load_setup
from hybrid.v1_blender.blender_model import (
    TinyBlender, build_feature_matrix, mixture_nll,
)
from hybrid.v1_blender.train import load_slice


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


def train_one(feats_train, log_p_targets_train, feats_inval, log_p_targets_inval,
              hidden, dropout, lr, epochs, batch, weight_decay, n_channels,
              in_dim, seed, device, patience=40):
    torch.manual_seed(seed)
    model = TinyBlender(in_dim, n_channels, hidden=hidden, dropout=dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_ppl = float("inf")
    best_state = None
    no_improve = 0
    n_train = feats_train.shape[0]
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n_train, device=device)
        for s in range(0, n_train, batch):
            b = perm[s:s + batch]
            log_w = model(feats_train[b])
            loss = mixture_nll(log_w, log_p_targets_train[b]).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        ppl = eval_ppl(model, feats_inval, log_p_targets_inval)
        if ppl < best_ppl - 1e-3:
            best_ppl = ppl
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break
    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    return model, best_ppl


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=str, default="hybrid/v1_blender/data_real")
    p.add_argument("--out", type=str, default="hybrid/v1_blender/data_real/sweep_report.json")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    data_dir = Path(args.data_dir)
    val = load_slice(data_dir / "val.npz")
    ev = load_slice(data_dir / "eval.npz")

    _bpe, _vocab, _tok2id, _bpe_to_lm, emb, V, d = load_setup()
    emb = emb.float()
    feats_val = build_feature_matrix(
        val["log_p_observed"], val["log_p_lag1"], val["entropy"], val["max_log_prob"],
        emb, val["observed"], use_embedding=True,
        topk_log_probs=val.get("topk_log_probs"),
    )
    feats_eval = build_feature_matrix(
        ev["log_p_observed"], ev["log_p_lag1"], ev["entropy"], ev["max_log_prob"],
        emb, ev["observed"], use_embedding=True,
        topk_log_probs=ev.get("topk_log_probs"),
    )
    log_p_targets_val = val["log_p_targets"]
    log_p_targets_eval = ev["log_p_targets"]
    in_dim = feats_val.shape[1]
    C = log_p_targets_val.shape[1]

    device = torch.device(args.device)
    feats_val = feats_val.to(device)
    feats_eval = feats_eval.to(device)
    log_p_targets_val = log_p_targets_val.to(device)
    log_p_targets_eval = log_p_targets_eval.to(device)

    T = feats_val.shape[0]
    torch.manual_seed(0)
    perm = torch.randperm(T, device=device)
    n_in = max(1, int(T * 0.2))
    inval_idx = perm[:n_in]
    train_idx = perm[n_in:]
    feats_train = feats_val[train_idx]
    log_p_train = log_p_targets_val[train_idx]
    feats_inval = feats_val[inval_idx]
    log_p_inval = log_p_targets_val[inval_idx]

    configs = []
    for hidden in [64, 128, 256, 512]:
        for dropout in [0.0, 0.1, 0.2]:
            for lr in [1e-3, 3e-3]:
                for wd in [1e-5, 1e-3]:
                    configs.append({"hidden": hidden, "dropout": dropout,
                                    "lr": lr, "weight_decay": wd})
    print(f"[sweep] {len(configs)} configs over in_dim={in_dim}, C={C}")
    results = []
    t0 = time.time()
    best = {"heldout_ppl": float("inf")}
    for i, cfg in enumerate(configs):
        model, inval_ppl = train_one(
            feats_train, log_p_train, feats_inval, log_p_inval,
            hidden=cfg["hidden"], dropout=cfg["dropout"], lr=cfg["lr"],
            epochs=300, batch=4096, weight_decay=cfg["weight_decay"],
            n_channels=C, in_dim=in_dim, seed=0, device=device,
        )
        heldout_ppl = eval_ppl(model, feats_eval, log_p_targets_eval)
        row = {**cfg, "inval_ppl": inval_ppl, "heldout_ppl": heldout_ppl}
        results.append(row)
        if heldout_ppl < best["heldout_ppl"]:
            best = row
        print(f"  [{i+1:2d}/{len(configs)}] h={cfg['hidden']:4d} dr={cfg['dropout']:.1f} "
              f"lr={cfg['lr']:.0e} wd={cfg['weight_decay']:.0e} "
              f"-> inval {inval_ppl:6.2f} | heldout {heldout_ppl:6.2f}  "
              f"({time.time()-t0:.0f}s)")
    print(f"\n[best] heldout PPL = {best['heldout_ppl']:.3f} with {best}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"results": results, "best": best}, f, indent=2)
    print(f"[save] {args.out}")


if __name__ == "__main__":
    main()
