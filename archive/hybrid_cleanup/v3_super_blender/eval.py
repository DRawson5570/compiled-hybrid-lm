"""hybrid/v3_super_blender/eval.py

Evaluate trained sequence-aware blenders on the 100K heldout eval dataset.
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
from hybrid.v1_blender.blender_model import build_feature_matrix, mixture_nll
from hybrid.v1_blender.train import load_slice
from hybrid.v3_super_blender.model import GRUBlender, LookbackMLPBlender, CausalConvBlender, WindowMLPBlender


def _ppl(nll_sum: float, n: int) -> float:
    return math.exp(nll_sum / n)


@torch.no_grad()
def evaluate(model, features: torch.Tensor, log_p_targets: torch.Tensor, model_type: str, device: torch.device) -> float:
    model.eval()
    T, F = features.shape
    
    if model_type in ["lookback_mlp", "window_mlp"]:
        print(f"[eval] Pre-building lookback features for eval slice ({model_type}) ...")
        features_win = model.build_windowed_features(features).to(device)
        nll_sum = 0.0
        batch = 8192
        for s in range(0, T, batch):
            e = min(s + batch, T)
            log_w = model(features_win[s:e], is_already_windowed=True)
            nll = mixture_nll(log_w, log_p_targets[s:e])
            nll_sum += nll.sum().item()
        return _ppl(nll_sum, T)
    elif model_type == "gru":
        # Chunk-by-chunk with state forwarding to avoid cuDNN errors and memory limits
        h = None
        log_w_parts = []
        chunk_len = 2048
        for s in range(0, T, chunk_len):
            e = min(s + chunk_len, T)
            feat_chunk = features[s:e].unsqueeze(0).to(device)
            log_w_chunk, h = model(feat_chunk, h)
            log_w_parts.append(log_w_chunk.squeeze(0).cpu())
        log_w = torch.cat(log_w_parts, dim=0).to(device)
        nll = mixture_nll(log_w, log_p_targets)
        return _ppl(nll.sum().item(), T)
    else:
        # Causal Conv: feed the entire contiguous sequence in a single forward pass
        features_d = features.unsqueeze(0).to(device)  # (1, T, F)
        log_w = model(features_d).squeeze(0)  # (T, C)
        nll = mixture_nll(log_w, log_p_targets)
        return _ppl(nll.sum().item(), T)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--blender", type=str, required=True,
                   help="Path to the model checkpoint. e.g. hybrid/v3_super_blender/saved_models/blender_causal_conv.pt")
    p.add_argument("--data-dir", type=str, default="hybrid/v1_blender/data_real")
    p.add_argument("--slice", type=str, default="eval", choices=["val", "eval"])
    p.add_argument("--out", type=str, default="hybrid/v3_super_blender/data_real/eval_report_{model_type}.json")
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
    model_type = ckpt["model_type"]
    use_emb = ckpt["use_embedding"]
    in_dim = ckpt["in_dim"]
    
    ckpt_args = ckpt["args"]
    
    if model_type == "lookback_mlp":
        model = LookbackMLPBlender(
            single_step_dim=in_dim,
            n_channels=C,
            lookback_window=ckpt_args["lookback"],
            hidden=ckpt_args["hidden"],
            num_layers=ckpt_args["layers"],
            dropout=0.0
        )
    elif model_type == "window_mlp":
        model = WindowMLPBlender(
            single_step_dim=in_dim,
            n_channels=C,
            lookback_window=ckpt_args["lookback"],
            hidden=ckpt_args["hidden"],
            dropout=0.0
        )
    elif model_type == "gru":
        model = GRUBlender(
            in_dim=in_dim,
            n_channels=C,
            hidden=ckpt_args["hidden"],
            num_layers=ckpt_args["layers"],
            dropout=0.0
        )
    elif model_type == "causal_conv":
        model = CausalConvBlender(
            in_dim=in_dim,
            n_channels=C,
            channels=ckpt_args["hidden"],
            kernel_size=3,
            num_layers=ckpt_args["layers"],
            dropout=0.0
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

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

    learned_ppl = evaluate(model, features, log_p_targets_d, model_type, device)
    print(f"[{model_type} blender]      PPL = {learned_ppl:8.3f}")

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

    # Oracle per-token best channel
    oracle_nll = (-log_p_targets_d.max(dim=1).values).mean().item()
    oracle_ppl = math.exp(oracle_nll)
    print(f"[oracle per-token]   PPL = {oracle_ppl:8.3f}  (unreachable lower bound)")

    report = {
        "slice": args.slice,
        "T": int(T),
        "C": int(C),
        "model_type": model_type,
        "trained_blender_ppl": learned_ppl,
        "uniform_mix_ppl": uniform_ppl,
        "best_single_channel": chan_names[best_single_idx],
        "best_single_ppl": best_single,
        "oracle_per_token_ppl": oracle_ppl,
        "per_channel_ppl": per_channel,
        "blender_args": ckpt["args"],
    }
    
    out_file = args.out.replace("{model_type}", model_type)
    out_path = Path(out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[save] {out_path}")


if __name__ == "__main__":
    main()
