"""hybrid/v4_fused_blender/train_fused.py

Generates a large-scale fused capability dataset, dumps 18-channel features,
and trains/evaluates all four sequence-aware CMI blenders.
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

# Import capability definitions
from hybrid.v2_capabilities.channels import (
    InstructChannel, ReasonerChannel, CoderChannel, ToolChannel
)
from hybrid.v2_capabilities.dataset import (
    tok2id, id2tok, V, get_ppmi_embeddings
)
from hybrid.v1_blender.blender_model import (
    build_feature_matrix, mixture_nll
)
from hybrid.v3_super_blender.model import (
    WindowMLPBlender, LookbackMLPBlender, GRUBlender, CausalConvBlender
)
from hybrid.v4_fused_blender.generator import interleave_capabilities_with_wikitext

def load_wikitext_tokens() -> list[str]:
    """Loads actual wikitext raw tokens."""
    wiki_path = REPO / "wikitext103.txt"
    if not wiki_path.exists():
        # Clean dummy fallback
        return ["the", "dog", "is", "a", "cat"] * 1000
    with open(wiki_path, "r", encoding="utf-8") as f:
        text = f.read()
    return text.split()

def evaluate_model(model, features, log_p_targets, targets, sequence_boundaries, model_type, channels, emb):
    """Performs evaluation of a sequence blender, calculating metrics across tasks and visualizing."""
    model.eval()
    C = log_p_targets.shape[1]
    channel_names = ["InstructChannel", "ReasonerChannel", "CoderChannel", "ToolChannel"]
    
    # Predict over the entire dataset
    with torch.no_grad():
        if model_type in ["lookback_mlp", "window_mlp"]:
            log_w = model(features, is_already_windowed=False) # (Total_T, C)
        elif model_type == "gru":
            log_w, _ = model(features.unsqueeze(0)) # (1, Total_T, C)
            log_w = log_w.squeeze(0)
        elif model_type == "causal_conv":
            log_w = model(features.unsqueeze(0)).squeeze(0) # (Total_T, C)

    # Calculate global metrics
    loss = mixture_nll(log_w, log_p_targets)
    avg_nll = loss.mean().item()
    global_ppl = math.exp(avg_nll)
    
    # Next token prediction logic
    correct = 0
    w_weights = log_w.exp() # (Total_T, C)
    
    # Category-specific metrics tracking
    task_metrics = {i: {"nll_sum": 0.0, "count": 0, "correct": 0} for i in range(4)}
    
    for start, end, ctx, target in sequence_boundaries:
        # Determine task type based on keywords in context
        ctx_str = " ".join(ctx)
        if "translate" in ctx_str or "explain" in ctx_str:
            task_type = 0 # Instruct
        elif "larger" in ctx_str:
            task_type = 1 # Reasoner
        elif "def" in ctx_str or "import" in ctx_str:
            task_type = 2 # Coder
        elif "What is" in ctx_str:
            task_type = 3 # Tool
        else:
            task_type = 0
            
        # Predict once per sequence
        tokens_full = ctx + [target]
        ids_full = torch.tensor([tok2id[token] for token in tokens_full], device=features.device)
        
        p_outputs_idx = []
        for c in channels:
            p_outputs_idx.append(c.forward(ids_full))
            
        for idx in range(start, end):
            if idx >= len(targets):
                continue
            w_idx = w_weights[idx]
            
            t_step = idx - start
            y_tgt = targets[idx].item()
            
            blended_prob = torch.zeros(V, device=features.device)
            for c_idx in range(C):
                blended_prob += w_idx[c_idx] * p_outputs_idx[c_idx][t_step].exp()
            blended_log_p = blended_prob.log()
            
            prediction = blended_log_p.argmax().item()
            is_correct = (prediction == y_tgt)
            if is_correct:
                correct += 1
                task_metrics[task_type]["correct"] += 1
                
            pos_nll = loss[idx].item()
            task_metrics[task_type]["nll_sum"] += pos_nll
            task_metrics[task_type]["count"] += 1

    global_acc = correct / len(targets) if len(targets) > 0 else 0.0
    
    # Format and display task results
    task_names = ["Instruction Following", "Transitive Reasoning", "Code Generation", "Interactive Tool Use"]
    print(f"\n" + "-"*60)
    print(f" BLENDER CAPABILITY REPORT: {model_type.upper()}")
    print("-"*60)
    print(f"Global Val Perplexity : {global_ppl:.4f}")
    print(f"Global Val Accuracy   : {global_acc * 100:.2f}%")
    print("-"*60)
    
    for i, name in enumerate(task_names):
        count = task_metrics[i]["count"]
        if count > 0:
            nll = task_metrics[i]["nll_sum"] / count
            ppl = math.exp(nll)
            acc = task_metrics[i]["correct"] / count
            print(f"{name:<25} | PPL: {ppl:.3f} | Acc: {acc * 100:.2f}% ({task_metrics[i]['correct']}/{count})")
        else:
            print(f"{name:<25} | No events found.")
            
    # Visualize routing allocation for representative examples of each task
    print("-"*60)
    print("Representative Routing Decisions (Validation Set):")
    for i, name in enumerate(task_names):
        # Find first sequence of this task category in validation set
        seq = None
        for start, end, ctx, target in sequence_boundaries:
            if start >= len(targets):
                continue
            ctx_str = " ".join(ctx)
            if i == 0 and ("translate" in ctx_str or "explain" in ctx_str):
                seq = (start, end, ctx, target)
                break
            elif i == 1 and ("larger" in ctx_str):
                seq = (start, end, ctx, target)
                break
            elif i == 2 and ("def" in ctx_str or "import" in ctx_str):
                seq = (start, end, ctx, target)
                break
            elif i == 3 and ("What is" in ctx_str):
                seq = (start, end, ctx, target)
                break
                
        if seq:
            start, end, ctx, target = seq
            idx = min(end - 1, len(targets) - 1)
            w_idx = w_weights[idx]
            print(f"\n  [Task: {name}]")
            print(f"  Context: {' '.join(ctx)}")
            print(f"  Target : {target}")
            print(f"  Routing Weights:")
            for c_idx, cl_name in enumerate(channel_names):
                bar = "#" * int(w_idx[c_idx].item() * 15)
                print(f"    - {cl_name:<17}: {w_idx[c_idx].item():.4f}  {bar}")
                
    return {
        "global_ppl": global_ppl,
        "global_acc": global_acc,
        "task_metrics": task_metrics
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--train-size", type=int, default=1200, help="Number of training context-target window pairs")
    parser.add_argument("--val-size", type=int, default=400, help="Number of validation context-target window pairs")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate for AdamW")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Target device")
    parser.add_argument("--eval-only", action="store_true", help="Bypass training and run evaluation on checkpoint saved models")
    parser.add_argument("--models-dir", type=str, default="hybrid/v4_fused_blender/saved_models", help="Directory where model checkpoints are saved/loaded")
    args = parser.parse_args()

    print("="*80)
    print(" PHASE II CHANNEL FUSION: LARGE-SCALE MULTI-TASK PIPELINE")
    print("="*80)

    # 1. Setup PPMI & Capability Channels
    emb = get_ppmi_embeddings()  # (V, d)
    print(f"Loaded embeddings: {emb.shape}")

    v2_instruct = InstructChannel(tok2id, id2tok, emb)
    v2_reasoner = ReasonerChannel(tok2id, id2tok)
    v2_coder = CoderChannel(tok2id, id2tok)
    v2_tool = ToolChannel(tok2id, id2tok)
    channels = [v2_instruct, v2_reasoner, v2_coder, v2_tool]
    channel_names = ["InstructChannel", "ReasonerChannel", "CoderChannel", "ToolChannel"]

    # 2. Interleave WikiText with capability queries
    raw_wiki = load_wikitext_tokens()
    print(f"Loaded {len(raw_wiki):,} raw WikiText-103 tokens.")

    # Request up to 30K tokens of wikitext to make a rich dataset
    fused_tokens = interleave_capabilities_with_wikitext(raw_wiki[:30000], tok2id)
    print(f"Generated large-scale fused tokens stream of size {len(fused_tokens):,}.")

    # Generate sequential datasets pairs
    dataset_pairs = []
    window_sz = 15
    for idx in range(0, len(fused_tokens) - window_sz - 1, 5):
        context = fused_tokens[idx:idx + window_sz]
        target = fused_tokens[idx + window_sz]
        dataset_pairs.append((context, target))

    print(f"Formed {len(dataset_pairs):,} sliding context-target window examples.")

    total_requested = args.train_size + args.val_size
    if len(dataset_pairs) < total_requested:
        print(f"Scaling down train/val size because dataset only has {len(dataset_pairs)} examples.")
        train_pairs = dataset_pairs[:int(len(dataset_pairs)*0.75)]
        val_pairs = dataset_pairs[int(len(dataset_pairs)*0.75):]
    else:
        train_pairs = dataset_pairs[:args.train_size]
        val_pairs = dataset_pairs[args.train_size:total_requested]

    print(f"Train subset size: {len(train_pairs):,} | Val subset size: {len(val_pairs):,}")

    # 3. Extract features
    from hybrid.v2_capabilities.train_super_blenders_v2 import build_dataset_features
    
    print("\nExtracting features for training set...")
    train_features, log_p_targets_train, targets_train, seq_bounds_train = build_dataset_features(
        train_pairs, channels, emb, use_embedding=True
    )
    
    print("Extracting features for validation set...")
    val_features, log_p_targets_val, targets_val, seq_bounds_val = build_dataset_features(
        val_pairs, channels, emb, use_embedding=True
    )

    T_tr, F_dim = train_features.shape
    C = log_p_targets_train.shape[1]
    print(f"Extracted Train joint dataset shape: T={T_tr}, Feature Dim F={F_dim}, Channels C={C}")

    # Build sequence blenders
    device = torch.device(args.device)
    print(f"Using training device: {device}")
    
    # Cast tensors to device
    train_features = train_features.to(device)
    log_p_targets_train = log_p_targets_train.to(device)
    
    val_features = val_features.to(device)
    log_p_targets_val = log_p_targets_val.to(device)
    targets_val = targets_val.to(device)

    shifted_val_seq_bounds = []
    for start, end, ctx, target in seq_bounds_val:
        shifted_val_seq_bounds.append((start, end, ctx, target))

    models = {
        "window_mlp": WindowMLPBlender(
            single_step_dim=F_dim, n_channels=C, lookback_window=4, hidden=64, dropout=0.0
        ).to(device),
        "lookback_mlp": LookbackMLPBlender(
            single_step_dim=F_dim, n_channels=C, lookback_window=4, hidden=64, num_layers=2, dropout=0.0
        ).to(device),
        "gru": GRUBlender(
            in_dim=F_dim, n_channels=C, hidden=64, num_layers=1, dropout=0.0
        ).to(device),
        "causal_conv": CausalConvBlender(
            in_dim=F_dim, n_channels=C, channels=64, kernel_size=3, num_layers=2, dropout=0.0
        ).to(device)
    }

    saved_models_dir = Path(args.models_dir)
    saved_models_dir.mkdir(parents=True, exist_ok=True)

    # Train each model
    for name, model in models.items():
        save_path = saved_models_dir / f"blender_{name}.pt"
        if args.eval_only:
            print(f"\nEvaluating pretrained Blender: {name.upper()}")
            if save_path.exists():
                model.load_state_dict(torch.load(save_path, map_location=device))
                print(f"  Loaded trained parameters from {save_path}")
            else:
                print(f"  [Warning] Checkpoint {save_path} not found! Scanning from base/dummy weights.")
        else:
            print(f"\nTraining Blender: {name.upper()}")
            opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
            epochs = args.epochs
            for epoch in range(1, epochs + 1):
                model.train()
                opt.zero_grad()
                
                if name in ["lookback_mlp", "window_mlp"]:
                    log_w = model(train_features, is_already_windowed=False)
                elif name == "gru":
                    log_w, _ = model(train_features.unsqueeze(0))
                    log_w = log_w.squeeze(0)
                elif name == "causal_conv":
                    log_w = model(train_features.unsqueeze(0)).squeeze(0)
                    
                loss = mixture_nll(log_w, log_p_targets_train).mean()
                loss.backward()
                opt.step()
                
                if epoch % 50 == 0 or epoch == 1:
                    print(f"  Epoch {epoch:3d}/{epochs} | Mixture NLL: {loss.item():.5f} | Train PPL: {math.exp(loss.item()):.4f}")

            # Save model checkpoint
            torch.save(model.state_dict(), save_path)
            print(f"  Saved trained parameters to {save_path}")

        # Run Validation Evaluation & capability breakdown
        evaluate_model(
            model, val_features, log_p_targets_val, targets_val,
            shifted_val_seq_bounds, name, channels, emb
        )

    print("\n" + "="*80)
    print(" FUSED LARGE-SCALE TRAINING WORKFLOW SUCCESSFULLY RUN!")
    print("="*80)

if __name__ == "__main__":
    main()
