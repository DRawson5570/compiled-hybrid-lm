"""hybrid/v3_super_blender/train.py

Train sequence-aware blenders on wikitext-103 features and report heldout validation PPL.
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
from hybrid.v1_blender.blender_model import build_feature_matrix, mixture_nll
from hybrid.v1_blender.train import load_slice
from hybrid.v3_super_blender.model import GRUBlender, LookbackMLPBlender, CausalConvBlender, WindowMLPBlender


def _ppl(nll_sum: float, n: int) -> float:
    return math.exp(nll_sum / n)


def evaluate_mlp(model, features_win: torch.Tensor, log_p_targets: torch.Tensor, batch: int = 8192) -> tuple[float, float]:
    model.eval()
    nll_sum = 0.0
    n = features_win.shape[0]
    with torch.no_grad():
        for s in range(0, n, batch):
            e = min(s + batch, n)
            log_w = model(features_win[s:e], is_already_windowed=True)
            nll = mixture_nll(log_w, log_p_targets[s:e])
            nll_sum += nll.sum().item()
    mean_nll = nll_sum / n
    return mean_nll, math.exp(mean_nll)


def evaluate_seq(model, features: torch.Tensor, log_p_targets: torch.Tensor) -> tuple[float, float]:
    model.eval()
    with torch.no_grad():
        # Pass the whole continuous sequence forward
        if isinstance(model, GRUBlender):
            log_w, _ = model(features)
        else:
            log_w = model(features)
        nll = mixture_nll(log_w, log_p_targets)
        mean_nll = nll.mean().item()
    return mean_nll, math.exp(mean_nll)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-type", type=str, default="lookback_mlp",
                   choices=["lookback_mlp", "window_mlp", "gru", "causal_conv"])
    p.add_argument("--data-dir", type=str, default="hybrid/v1_blender/data_real")
    p.add_argument("--out-dir", type=str, default="hybrid/v3_super_blender/saved_models")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch", type=int, default=1024)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--lookback", type=int, default=16,
                   help="Lookback window size for lookback_mlp.")
    p.add_argument("--seq-len", type=int, default=256,
                   help="Sequence chunk length for training GRU/Conv.")
    p.add_argument("--stride", type=int, default=64,
                   help="Sequence extraction stride for GRU/Conv.")
    p.add_argument("--val-frac", type=float, default=0.2,
                   help="Fraction of val slice held out as contiguous validation.")
    p.add_argument("--split-type", type=str, default="random",
                   choices=["contiguous", "random"],
                   help="Type of splitting for validation framework.")
    p.add_argument("--no-embedding", action="store_true")
    p.add_argument("--seed", type=int, default=42)
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
    print(f"[features] shape={tuple(features.shape)}  C={val['log_p_targets'].shape[1]}")

    T = features.shape[0]
    n_val = max(1, int(T * args.val_frac))
    
    if args.split_type == "random" and args.model_type in ["lookback_mlp", "window_mlp"]:
        print(f"[split] Using random timestep splitting for {args.model_type} ...")
        idx = torch.randperm(T)
        train_idx = idx[n_val:]
        inval_idx = idx[:n_val]
        train_len = train_idx.shape[0]
        
        # Sliced below during windowed handling
        train_feat = None
        inval_feat = None
        train_log_p = None
        inval_log_p = None
    else:
        print(f"[split] Using contiguous sequence splitting ...")
        train_len = T - n_val
        train_idx = torch.arange(train_len)
        inval_idx = torch.arange(train_len, T)
        
        train_feat = features[:train_len]
        inval_feat = features[train_len:]
        train_log_p = val["log_p_targets"][:train_len]
        inval_log_p = val["log_p_targets"][train_len:]

    print(f"[split] train_steps={train_len:,} | inval_steps={n_val:,}")

    device = torch.device(args.device)
    n_channels = val["log_p_targets"].shape[1]

    # Initialize model
    if args.model_type == "lookback_mlp":
        model = LookbackMLPBlender(
            single_step_dim=features.shape[1],
            n_channels=n_channels,
            lookback_window=args.lookback,
            hidden=args.hidden,
            num_layers=args.layers,
            dropout=args.dropout
        )
    elif args.model_type == "window_mlp":
        model = WindowMLPBlender(
            single_step_dim=features.shape[1],
            n_channels=n_channels,
            lookback_window=args.lookback,
            hidden=args.hidden,
            dropout=args.dropout
        )
    elif args.model_type == "gru":
        model = GRUBlender(
            in_dim=features.shape[1],
            n_channels=n_channels,
            hidden=args.hidden,
            num_layers=args.layers,
            dropout=args.dropout
        )
    elif args.model_type == "causal_conv":
        model = CausalConvBlender(
            in_dim=features.shape[1],
            n_channels=n_channels,
            channels=args.hidden,
            kernel_size=3,
            num_layers=args.layers,
            dropout=args.dropout
        )
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")

    model = model.to(device)
    print(f"[model] {model.__class__.__name__} parameters: {sum(p.numel() for p in model.parameters()):,}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Prepare training features
    if args.model_type in ["lookback_mlp", "window_mlp"]:
        # We can build windowed features for the entire val sequence, and then slice it
        print(f"[prepare] Pre-building windowed features with lookback={args.lookback} ...")
        features_win = model.build_windowed_features(features).to(device)
        train_feat_win = features_win[train_idx]
        inval_feat_win = features_win[inval_idx]
        
        train_log_p_d = val["log_p_targets"][train_idx].to(device)
        inval_log_p_d = val["log_p_targets"][inval_idx].to(device)
        
        # Initial evaluation
        init_nll, init_ppl = evaluate_mlp(model, inval_feat_win, inval_log_p_d)
        print(f"[init] inval nll={init_nll:.4f}  ppl={init_ppl:.2f}")
    else:
        # For sequence models, we prepare sequence batch chunks from the contiguous train prefix
        print(f"[prepare] Structuring sequence chunks of len={args.seq_len}, stride={args.stride} ...")
        # Extract slices of length seq_len
        chunks_feat = []
        chunks_target = []
        for s in range(0, train_len - args.seq_len + 1, args.stride):
            e = s + args.seq_len
            chunks_feat.append(train_feat[s:e])
            chunks_target.append(train_log_p[s:e])
        
        chunks_feat_t = torch.stack(chunks_feat).to(device)  # (N_seq, SeqLen, F)
        chunks_target_t = torch.stack(chunks_target).to(device)  # (N_seq, SeqLen, C)
        print(f"  Total of {chunks_feat_t.shape[0]} sequence training chunks.")
        
        train_feat_d = train_feat.to(device)
        inval_feat_d = inval_feat.to(device)
        train_log_p_d = train_log_p.to(device)
        inval_log_p_d = inval_log_p.to(device)
        
        init_nll, init_ppl = evaluate_seq(model, inval_feat_d, inval_log_p_d)
        print(f"[init] inval nll={init_nll:.4f}  ppl={init_ppl:.2f}")

    best_ppl = float("inf")
    best_state = None
    t0 = time.time()
    
    # Train loop
    for epoch in range(args.epochs):
        model.train()
        
        if args.model_type in ["lookback_mlp", "window_mlp"]:
            # Direct random batches over timesteps
            perm = torch.randperm(train_len)
            ep_nll = 0.0
            ep_n = 0
            for s in range(0, train_len, args.batch):
                e = min(s + args.batch, train_len)
                b = perm[s:e]
                log_w = model(train_feat_win[b], is_already_windowed=True)
                loss = mixture_nll(log_w, train_log_p_d[b]).mean()
                
                opt.zero_grad()
                loss.backward()
                opt.step()
                
                ep_nll += loss.item() * (e - s)
                ep_n += (e - s)
            mean_train_nll = ep_nll / ep_n
            # Evaluate
            val_nll, val_ppl = evaluate_mlp(model, inval_feat_win, inval_log_p_d)
        else:
            # Batch sequences
            perm = torch.randperm(chunks_feat_t.shape[0])
            ep_nll = 0.0
            ep_n = 0
            for s in range(0, chunks_feat_t.shape[0], args.batch):
                e = min(s + args.batch, chunks_feat_t.shape[0])
                b_idx = perm[s:e]
                
                batch_feat = chunks_feat_t[b_idx]  # (B_seq, SeqLen, F)
                batch_target = chunks_target_t[b_idx]  # (B_seq, SeqLen, C)
                
                if isinstance(model, GRUBlender):
                    log_w, _ = model(batch_feat)
                else:
                    log_w = model(batch_feat)
                
                # loss is sum or mean over sequence steps and batches
                # flatten SeqLen and B_seq to compute mixture_nll
                loss = mixture_nll(log_w.reshape(-1, n_channels), batch_target.reshape(-1, n_channels)).mean()
                
                opt.zero_grad()
                loss.backward()
                opt.step()
                
                ep_nll += loss.item() * batch_feat.shape[0] * args.seq_len
                ep_n += batch_feat.shape[0] * args.seq_len
            mean_train_nll = ep_nll / ep_n
            # Evaluate
            val_nll, val_ppl = evaluate_seq(model, inval_feat_d, inval_log_p_d)
            
        if val_ppl < best_ppl:
            best_ppl = val_ppl
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
        
        if (epoch + 1) % 10 == 0 or epoch == 0 or epoch == args.epochs - 1:
            dt = time.time() - t0
            print(f"Epoch {epoch+1:3d}/{args.epochs} | train_nll={mean_train_nll:.4f} | inval_nll={val_nll:.4f} in_val_ppl={val_ppl:.3f} | best={best_ppl:.3f} | {dt:.1f}s")

    # Save best
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"blender_{args.model_type}.pt"
    
    save_data = {
        "model_type": args.model_type,
        "state_dict": best_state,
        "use_embedding": use_emb,
        "in_dim": features.shape[1],
        "n_channels": n_channels,
        "args": vars(args),
    }
    torch.save(save_data, out_path)
    print(f"Saved best model checkpoint to {out_path} (best inval ppl: {best_ppl:.3f})")


if __name__ == "__main__":
    main()
