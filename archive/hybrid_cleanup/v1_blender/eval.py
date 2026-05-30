"""hybrid/v1_blender/eval.py

Load a trained TinyBlender and report mixture NLL/PPL on the eval (heldout)
slice.  Also reports two reference baselines:

  - uniform mixing (w = 1/C)
  - best single channel
  - oracle per-token min-NLL  (lower bound, not achievable by any fixed mixer)

PPL is computed from the same per-channel log_p_target tensor that the trainer
used, so it is directly comparable to v31's heldout PPL=38.83.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
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


def _ppl(nll_sum: float, n: int) -> float:
    return math.exp(nll_sum / n)


@torch.no_grad()
def evaluate(model: TinyBlender, features: torch.Tensor,
             log_p_targets: torch.Tensor, batch: int = 8192) -> float:
    model.eval()
    nll_sum = 0.0
    n = features.shape[0]
    for s in range(0, n, batch):
        e = min(s + batch, n)
        log_w = model(features[s:e])
        nll = mixture_nll(log_w, log_p_targets[s:e])
        nll_sum += nll.sum().item()
    return _ppl(nll_sum, n)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--blender", type=str, default="hybrid/v1_blender/data/blender.pt")
    p.add_argument("--data-dir", type=str, default="hybrid/v1_blender/data")
    p.add_argument("--slice", type=str, default="eval", choices=["val", "eval"])
    p.add_argument("--out", type=str, default="hybrid/v1_blender/data/eval_report.json")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    data_dir = Path(args.data_dir)
    sl = load_slice(data_dir / f"{args.slice}.npz")
    log_p_targets = sl["log_p_targets"]
    T, C = log_p_targets.shape
    print(f"[load] slice={args.slice}  T={T:,}  C={C}")

    print("[load] embedding ...")
    _bpe, _vocab, _tok2id, _bpe_to_lm, emb, V, d = load_setup()
    emb = emb.float()

    print(f"[load] blender {args.blender}")
    ckpt = torch.load(args.blender, map_location="cpu", weights_only=False)
    use_emb = ckpt["use_embedding"]
    in_dim = ckpt["in_dim"]
    model = TinyBlender(in_dim, ckpt["n_channels"],
                        hidden=ckpt["args"]["hidden"],
                        dropout=0.0)
    model.load_state_dict(ckpt["state_dict"])

    features = build_feature_matrix(
        sl["log_p_observed"], sl["log_p_lag1"], sl["entropy"],
        sl["max_log_prob"], emb, sl["observed"], use_embedding=use_emb,
        topk_log_probs=sl.get("topk_log_probs"),
    )
    assert features.shape == (T, in_dim), (features.shape, in_dim)

    device = torch.device(args.device)
    features = features.to(device)
    log_p_targets_d = log_p_targets.to(device)
    model = model.to(device)

    learned_ppl = evaluate(model, features, log_p_targets_d)
    print(f"[trained blender]      PPL = {learned_ppl:8.3f}")

    # Uniform baseline
    log_w_uniform = torch.full((1, C), -math.log(C), device=device).expand(T, C)
    uniform_nll = mixture_nll(log_w_uniform, log_p_targets_d).mean().item()
    uniform_ppl = math.exp(uniform_nll)
    print(f"[uniform mix]          PPL = {uniform_ppl:8.3f}")

    # Best single channel
    print("[per-channel single-channel PPL]")
    best_single = float("inf")
    best_single_idx = -1
    chan_names = list(np.load(data_dir / f"{args.slice}.npz", allow_pickle=True)["channel_names"])
    per_channel = {}
    for c in range(C):
        log_w = torch.full((T, C), -1e30, device=device)
        log_w[:, c] = 0.0
        nll = mixture_nll(log_w, log_p_targets_d).mean().item()
        ppl = math.exp(nll)
        per_channel[chan_names[c]] = ppl
        print(f"  {chan_names[c]:10s}  PPL = {ppl:8.3f}")
        if ppl < best_single:
            best_single = ppl
            best_single_idx = c
    print(f"[best single]    {chan_names[best_single_idx]:10s}  PPL = {best_single:8.3f}")

    # Oracle per-token best channel (lower bound — picks the best channel per token)
    oracle_nll = (-log_p_targets_d.max(dim=1).values).mean().item()
    oracle_ppl = math.exp(oracle_nll)
    print(f"[oracle per-token]   PPL = {oracle_ppl:8.3f}  (unreachable lower bound)")

    report = {
        "slice": args.slice,
        "T": int(T),
        "C": int(C),
        "trained_blender_ppl": learned_ppl,
        "uniform_mix_ppl": uniform_ppl,
        "best_single_channel": chan_names[best_single_idx],
        "best_single_ppl": best_single,
        "oracle_per_token_ppl": oracle_ppl,
        "per_channel_ppl": per_channel,
        "blender_args": ckpt["args"],
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[save] {args.out}")


if __name__ == "__main__":
    main()
