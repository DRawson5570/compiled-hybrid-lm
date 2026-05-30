"""hybrid/v1_blender/sweep_t1.py

First contextual-transformer-mixer sweep.  Same v3 features (data_big_nn)
+ NN channel.  Replaces position-wise MLP with TransformerBlender.

Usage (pe2):
    CUDA_VISIBLE_DEVICES=2 nohup python -u hybrid/v1_blender/sweep_t1.py \\
        --data-dir hybrid/v1_blender/data_big_nn \\
        --out hybrid/v1_blender/data_big_nn/sweep_t1.json \\
        > /tmp/hybrid_sw_t1.log 2>&1 &
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
from hybrid.v1_blender.sweep_v2 import build_feats
from hybrid.v1_blender.train import load_slice
from hybrid.v1_blender.transformer_blender import (
    TBConfig,
    TransformerBlender,
    train_transformer_blender,
)
from hybrid.v1_blender.blender_model import mixture_nll
import math


@torch.no_grad()
def eval_ppl_stream(model: TransformerBlender, feats: torch.Tensor,
                    log_p: torch.Tensor, ctx: int) -> float:
    model.eval()
    T = feats.shape[0]
    stride = max(1, ctx // 2)
    nll_sum = 0.0
    n_pos = 0
    for s in range(0, T, stride):
        e = min(s + ctx, T)
        win_f = feats[s:e].unsqueeze(0)
        win_p = log_p[s:e]
        log_w = model(win_f)[0]
        if s == 0:
            keep_start = 0
        else:
            keep_start = ctx - stride
        keep_len = (e - s) - keep_start
        if keep_len <= 0:
            continue
        nll = mixture_nll(log_w[keep_start:keep_start + keep_len],
                          win_p[keep_start:keep_start + keep_len])
        nll_sum += nll.sum().item()
        n_pos += keep_len
        if e == T:
            break
    return math.exp(nll_sum / max(1, n_pos))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="hybrid/v1_blender/data_big_nn")
    p.add_argument("--out", default="hybrid/v1_blender/data_big_nn/sweep_t1.json")
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
    print(f"  C={C}  T_train={log_p_targets_val.shape[0]:,}  T_eval={log_p_targets_eval.shape[0]:,}")

    # Use the sweep_v2 winner's feature window so the transformer mixer sees
    # the same per-position feature vectors that won at the MLP stage.
    wm, ww = 4, 8
    print(f"[features] win_mean={wm} win_won={ww}")
    feats_val = build_feats(val, emb, wm, ww).to(device)
    feats_eval = build_feats(ev, emb, wm, ww).to(device)
    in_dim = feats_val.shape[1]
    print(f"  in_dim={in_dim}")

    # Compact grid (transformer is heavier than MLP — keep it tight first).
    configs = []
    for d_model in (128, 192):
        for n_layers in (2, 3):
            for ctx in (128, 256):
                for dropout in (0.1, 0.2):
                    configs.append({
                        "d_model": d_model,
                        "n_heads": 4,
                        "d_ff": d_model * 4,
                        "n_layers": n_layers,
                        "ctx": ctx,
                        "dropout": dropout,
                        "lr": 3e-4,
                        "wd": 1e-3,
                        "batch": 16,
                        "steps": 2000,
                        "warmup": 100,
                        "eval_every": 100,
                        "patience": 6,
                    })
    print(f"[sweep_t1] {len(configs)} configs")

    results = []
    best = {"heldout_ppl": float("inf")}
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    for i, cfg_d in enumerate(configs):
        t0 = time.time()
        cfg = TBConfig(in_dim=in_dim, n_channels=C,
                       d_model=cfg_d["d_model"], n_heads=cfg_d["n_heads"],
                       d_ff=cfg_d["d_ff"], n_layers=cfg_d["n_layers"],
                       ctx=cfg_d["ctx"], dropout=cfg_d["dropout"])
        n_params = sum(p.numel() for p in TransformerBlender(cfg).parameters())
        model, inval_ppl, n_steps = train_transformer_blender(
            feats_val, log_p_targets_val, feats_val, log_p_targets_val, cfg,
            steps=cfg_d["steps"], batch=cfg_d["batch"], lr=cfg_d["lr"],
            wd=cfg_d["wd"], warmup=cfg_d["warmup"],
            eval_every=cfg_d["eval_every"], patience=cfg_d["patience"],
            device=device, log_fn=lambda *_a, **_k: None,
        )
        held_ppl = eval_ppl_stream(model, feats_eval, log_p_targets_eval, ctx=cfg.ctx)
        elapsed = time.time() - t_start
        row = {**cfg_d, "in_dim": in_dim, "n_params": n_params,
               "n_steps": n_steps, "inval_ppl": inval_ppl,
               "heldout_ppl": held_ppl}
        results.append(row)
        if held_ppl < best["heldout_ppl"]:
            best = row
        print(f"  [{i+1:2d}/{len(configs)}] d={cfg.d_model} L={cfg.n_layers} "
              f"ctx={cfg.ctx} dr={cfg.dropout} P={n_params/1e6:.2f}M "
              f"step={n_steps:4d} -> inval {inval_ppl:.3f} | heldout {held_ppl:.3f}  "
              f"best={best['heldout_ppl']:.3f}  ({elapsed:.0f}s)")
        out_path.write_text(json.dumps({"results": results, "best": best}, indent=2))
        del model
        torch.cuda.empty_cache()

    print(f"\n[best] heldout PPL = {best['heldout_ppl']:.3f}")
    print(f"  cfg: {best}")
    print(f"[save] {out_path}")


if __name__ == "__main__":
    main()
