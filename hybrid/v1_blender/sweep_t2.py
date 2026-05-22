"""hybrid/v1_blender/sweep_t2.py

Bigger contextual-transformer-mixer sweep on the 19-channel feature set
(data_big_nn_plus2).  Scaled up from sweep_t1 (#324) which was
capacity-limited at every axis.  Also fixes the inval-eval bottleneck:
sweep_t1 streamed the full 500K inval slice every 100 steps; here we
use a fixed 20K subset for early stopping.

Distributed across multiple GPUs via `--shard-idx K --shard-count N`
(each shard runs configs[K::N] and writes its own json).

Usage (one shard per GPU):
    CUDA_VISIBLE_DEVICES=0 python -u hybrid/v1_blender/sweep_t2.py \
        --shard-idx 0 --shard-count 5 \
        --out hybrid/v1_blender/data_big_nn_plus2/sweep_t2_shard0.json
"""
from __future__ import annotations

import argparse
import json
import math
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
        keep_start = 0 if s == 0 else ctx - stride
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


def build_grid():
    """36 configs: d ∈ {256,384,512} × L ∈ {3,4,6} × ctx ∈ {256,512} × dr ∈ {0.1,0.2}."""
    configs = []
    for d_model in (256, 384, 512):
        for n_layers in (3, 4, 6):
            for ctx in (256, 512):
                for dropout in (0.1, 0.2):
                    configs.append({
                        "d_model": d_model,
                        "n_heads": 8 if d_model >= 384 else 4,
                        "d_ff": d_model * 4,
                        "n_layers": n_layers,
                        "ctx": ctx,
                        "dropout": dropout,
                        "lr": 3e-4,
                        "wd": 1e-3,
                        "batch": 8 if (d_model >= 384 and ctx >= 512) else 16,
                        "steps": 4000,
                        "warmup": 200,
                        "eval_every": 200,
                        "patience": 8,
                    })
    return configs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="hybrid/v1_blender/data_big_nn_plus2")
    p.add_argument("--out", default="hybrid/v1_blender/data_big_nn_plus2/sweep_t2.json")
    p.add_argument("--shard-idx", type=int, default=0)
    p.add_argument("--shard-count", type=int, default=1)
    p.add_argument("--inval-positions", type=int, default=20000,
                   help="Number of positions from end of train slice used for early-stop inval.")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    data_dir = Path(args.data_dir)
    print(f"[load] {data_dir}  shard {args.shard_idx}/{args.shard_count}")
    val = load_slice(data_dir / "val.npz")
    ev = load_slice(data_dir / "eval.npz")
    _, _, _, _, emb, V, d = load_setup()
    emb = emb.float()
    device = torch.device(args.device)

    log_p_targets_val = val["log_p_targets"].to(device)
    log_p_targets_eval = ev["log_p_targets"].to(device)
    C = log_p_targets_val.shape[1]
    print(f"  C={C}  T_train={log_p_targets_val.shape[0]:,}  T_eval={log_p_targets_eval.shape[0]:,}")

    wm, ww = 4, 8
    print(f"[features] win_mean={wm} win_won={ww}")
    feats_val = build_feats(val, emb, wm, ww).to(device)
    feats_eval = build_feats(ev, emb, wm, ww).to(device)
    in_dim = feats_val.shape[1]
    print(f"  in_dim={in_dim}")

    inval_n = min(args.inval_positions, feats_val.shape[0])
    feats_inval = feats_val[-inval_n:].contiguous()
    log_p_inval = log_p_targets_val[-inval_n:].contiguous()
    print(f"[inval] last {inval_n:,} positions of train slice (early-stop only)")

    all_configs = build_grid()
    my_configs = all_configs[args.shard_idx::args.shard_count]
    print(f"[sweep_t2] total={len(all_configs)} shard={len(my_configs)}")

    results = []
    best = {"heldout_ppl": float("inf")}
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    for i, cfg_d in enumerate(my_configs):
        t0 = time.time()
        cfg = TBConfig(in_dim=in_dim, n_channels=C,
                       d_model=cfg_d["d_model"], n_heads=cfg_d["n_heads"],
                       d_ff=cfg_d["d_ff"], n_layers=cfg_d["n_layers"],
                       ctx=cfg_d["ctx"], dropout=cfg_d["dropout"])
        n_params = sum(p.numel() for p in TransformerBlender(cfg).parameters())
        try:
            model, inval_ppl, n_steps = train_transformer_blender(
                feats_val, log_p_targets_val, feats_inval, log_p_inval, cfg,
                steps=cfg_d["steps"], batch=cfg_d["batch"], lr=cfg_d["lr"],
                wd=cfg_d["wd"], warmup=cfg_d["warmup"],
                eval_every=cfg_d["eval_every"], patience=cfg_d["patience"],
                device=device, log_fn=lambda *_a, **_k: None,
            )
            held_ppl = eval_ppl_stream(model, feats_eval, log_p_targets_eval, ctx=cfg.ctx)
        except torch.cuda.OutOfMemoryError as exc:
            print(f"  [{i+1:2d}/{len(my_configs)}] OOM d={cfg.d_model} L={cfg.n_layers} ctx={cfg.ctx} bs={cfg_d['batch']}: {exc}")
            torch.cuda.empty_cache()
            row = {**cfg_d, "in_dim": in_dim, "n_params": n_params,
                   "n_steps": 0, "inval_ppl": float("nan"),
                   "heldout_ppl": float("nan"), "error": "OOM"}
            results.append(row)
            out_path.write_text(json.dumps({"results": results, "best": best}, indent=2))
            continue
        elapsed = time.time() - t_start
        cfg_elapsed = time.time() - t0
        row = {**cfg_d, "in_dim": in_dim, "n_params": n_params,
               "n_steps": n_steps, "inval_ppl": inval_ppl,
               "heldout_ppl": held_ppl, "cfg_seconds": cfg_elapsed}
        results.append(row)
        if held_ppl < best["heldout_ppl"]:
            best = row
        print(f"  [{i+1:2d}/{len(my_configs)}] d={cfg.d_model} L={cfg.n_layers} "
              f"ctx={cfg.ctx} dr={cfg.dropout} P={n_params/1e6:.2f}M "
              f"step={n_steps:4d} -> inval {inval_ppl:.3f} | heldout {held_ppl:.3f}  "
              f"best={best['heldout_ppl']:.3f}  ({cfg_elapsed:.0f}s/cfg, {elapsed:.0f}s)",
              flush=True)
        out_path.write_text(json.dumps({"results": results, "best": best,
                                         "shard_idx": args.shard_idx,
                                         "shard_count": args.shard_count}, indent=2))
        del model
        torch.cuda.empty_cache()

    print(f"\n[shard {args.shard_idx} done] best heldout PPL = {best['heldout_ppl']:.3f}")
    print(f"  cfg: {best}")
    print(f"[save] {out_path}")


if __name__ == "__main__":
    main()
