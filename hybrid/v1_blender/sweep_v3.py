"""hybrid/v1_blender/sweep_v3.py

Sweep DeepBlender on v3 data: v2 extended-past features + 13th NN channel.

Reads {data-dir}/val.npz and eval.npz produced by add_nn_channel.py.
Same grid as sweep_v2 but C=13.

Usage (pe2):
    CUDA_VISIBLE_DEVICES=3 nohup python -u hybrid/v1_blender/sweep_v3.py \\
        --data-dir hybrid/v1_blender/data_big_nn \\
        --out hybrid/v1_blender/data_big_nn/sweep_v3.json \\
        > /tmp/hybrid_sw_v3.log 2>&1 &
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from compile_wiki_lm_v13 import load_setup
from hybrid.v1_blender.features_v2 import build_feature_matrix_v2
from hybrid.v1_blender.sweep_big import DeepBlender, eval_ppl, train_one  # noqa: F401
from hybrid.v1_blender.sweep_v2 import build_feats
from hybrid.v1_blender.train import load_slice


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="hybrid/v1_blender/data_big_nn")
    p.add_argument("--out", default="hybrid/v1_blender/data_big_nn/sweep_v3.json")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    data_dir = Path(args.data_dir)
    print(f"[load] {data_dir}")
    val = load_slice(data_dir / "val.npz")
    ev = load_slice(data_dir / "eval.npz")
    _, _, _, _, emb, V, d = load_setup()
    emb = emb.float()
    device = torch.device(args.device)

    log_p_targets_val = val["log_p_targets"].to(device)
    log_p_targets_eval = ev["log_p_targets"].to(device)
    C = log_p_targets_val.shape[1]
    print(f"  C={C} (expecting 13: 12 compiled + 1 nn)")

    feats_cache: dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor]] = {}

    def get_feats(wm: int, ww: int):
        key = (wm, ww)
        if key not in feats_cache:
            print(f"  building features for win_mean={wm} win_won={ww}")
            fv = build_feats(val, emb, wm, ww).to(device)
            fe = build_feats(ev, emb, wm, ww).to(device)
            feats_cache[key] = (fv, fe)
        fv, fe = feats_cache[key]
        return fv, fe, fv.shape[1]

    T = val["log_p_targets"].shape[0]
    torch.manual_seed(0)
    perm = torch.randperm(T, device=device)
    n_in = max(1, int(T * 0.2))
    inval_idx = perm[:n_in]
    train_idx = perm[n_in:]

    # Tighter grid focused on the v2 winners; we expect the NN channel to
    # benefit from slightly more capacity since there is now a richer signal.
    configs = []
    for wm, ww in [(4, 8), (8, 16), (8, 32), (16, 32)]:
        for hidden in [512, 768, 1024]:
            for depth in [3]:
                for dropout in [0.2, 0.3]:
                    for wd in [1e-4, 1e-3]:
                        configs.append({
                            "hidden": hidden, "depth": depth, "dropout": dropout,
                            "lr": 1e-3, "weight_decay": wd,
                            "win_mean": wm, "win_won": ww,
                        })
    print(f"[sweep_v3] {len(configs)} configs, C={C}, n_train={train_idx.shape[0]:,}")

    results = []
    best = {"heldout_ppl": float("inf")}
    t0 = time.time()
    for i, cfg in enumerate(configs):
        try:
            feats_val_dev, feats_eval_dev, in_dim = get_feats(cfg["win_mean"], cfg["win_won"])
            feats_train = feats_val_dev[train_idx]
            log_p_train = log_p_targets_val[train_idx]
            feats_inval = feats_val_dev[inval_idx]
            log_p_inval = log_p_targets_val[inval_idx]
            model, inval_ppl, n_ep = train_one(
                feats_train, log_p_train, feats_inval, log_p_inval,
                hidden=cfg["hidden"], depth=cfg["depth"], dropout=cfg["dropout"],
                lr=cfg["lr"], wd=cfg["weight_decay"],
                epochs=120, batch=8192, n_channels=C, in_dim=in_dim,
                device=device, patience=20,
            )
            heldout_ppl = eval_ppl(model, feats_eval_dev, log_p_targets_eval)
            row = {**cfg, "in_dim": in_dim,
                   "inval_ppl": float(inval_ppl),
                   "heldout_ppl": float(heldout_ppl),
                   "n_epochs": int(n_ep)}
            if heldout_ppl < best["heldout_ppl"]:
                best = row
            print(f"  [{i+1:3d}/{len(configs)}] wm={cfg['win_mean']:2d} ww={cfg['win_won']:2d} "
                  f"h={cfg['hidden']:4d} d={cfg['depth']} dr={cfg['dropout']:.1f} "
                  f"wd={cfg['weight_decay']:.0e} ep={n_ep:3d} -> inval {inval_ppl:6.2f} | "
                  f"heldout {heldout_ppl:6.2f}  best={best['heldout_ppl']:.2f}  "
                  f"({time.time()-t0:.0f}s)")
        except torch.cuda.OutOfMemoryError:
            row = {**cfg, "error": "OOM"}
            print(f"  [{i+1:3d}/{len(configs)}] OOM h={cfg['hidden']} d={cfg['depth']}")
            torch.cuda.empty_cache()
        results.append(row)
    print(f"\n[best] heldout PPL = {best['heldout_ppl']:.3f} with {best}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"results": results, "best": best}, f, indent=2)
    print(f"[save] {args.out}")


if __name__ == "__main__":
    main()
