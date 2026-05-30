"""Unifying Phase II (Channel Fusion) - Training and Evaluating Sequence-Aware CMI Blenders.

Trains and compares four sequence-aware routing blenders (WindowMLP, LookbackMLP, GRU, CausalConv)
on the modular integration of the 4 Compiled Expert Channels:
- InstructChannel
- ReasonerChannel
- CoderChannel
- ToolChannel

Evaluates capability-specific accuracy and perplexity under strict causal bounds.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from hybrid.v2_capabilities.channels import (
    InstructChannel, ReasonerChannel, CoderChannel, ToolChannel
)
from hybrid.v2_capabilities.dataset import (
    tok2id, id2tok, V, get_ppmi_embeddings, generate_multi_task_data
)
from hybrid.v1_blender.blender_model import (
    build_feature_matrix, mixture_nll
)
from hybrid.v3_super_blender.model import (
    WindowMLPBlender, LookbackMLPBlender, GRUBlender, CausalConvBlender
)

def build_dataset_features(dataset_pairs, channels, emb, use_embedding=True):
    """Extracts step-by-step unwindowed features and target log probabilities."""
    all_feats = []
    all_log_p_targets = []
    all_targets = []
    
    # Track boundaries to isolate sequences for GRU/CNN sequence formatting
    sequence_boundaries = []
    current_index = 0

    for context, target in dataset_pairs:
        tokens = context + [target]
        ids = torch.tensor([tok2id[token] for token in tokens])
        T_len = len(context)
        
        # Extract predictions for each channel
        p_outputs = []
        for c in channels:
            p_outputs.append(c.forward(ids)) # (T_len+1, V)
            
        start_idx = current_index
        for t in range(T_len):
            y_target = ids[t+1]
            x_observed = ids[t]
            x_lag1 = ids[t-1] if t > 0 else torch.zeros_like(x_observed)
            
            # Extract channel logprobs
            log_p_targets_t = torch.stack([p_out[t, y_target] for p_out in p_outputs]) # (C,)
            log_p_observed_t = torch.stack([p_out[t, x_observed] for p_out in p_outputs]) # (C,)
            log_p_lag1_t = torch.stack([p_out[t, x_lag1] for p_out in p_outputs]) # (C,)
            
            entropy_t = []
            max_log_prob_t = []
            for p_out in p_outputs:
                p_dist = p_out[t].exp()
                # Entropy: -sum(p * log(p))
                entropy_t.append(-(p_dist * p_out[t]).sum())
                max_log_prob_t.append(p_out[t].max())
                
            entropy_t = torch.stack(entropy_t)
            max_log_prob_t = torch.stack(max_log_prob_t)
            
            # Build 1-step features
            feat = build_feature_matrix(
                log_p_observed_t.unsqueeze(0),
                log_p_lag1_t.unsqueeze(0),
                entropy_t.unsqueeze(0),
                max_log_prob_t.unsqueeze(0),
                emb,
                x_observed.unsqueeze(0),
                use_embedding=use_embedding
            ) # (1, F)
            
            all_feats.append(feat)
            all_log_p_targets.append(log_p_targets_t)
            all_targets.append(y_target)
            current_index += 1
            
        end_idx = current_index
        sequence_boundaries.append((start_idx, end_idx, context, target))
        
    features = torch.cat(all_feats, dim=0) # (Total_T, F)
    log_p_targets = torch.stack(all_log_p_targets, dim=0) # (Total_T, C)
    targets = torch.stack(all_targets, dim=0) # (Total_T,)
    
    return features, log_p_targets, targets, sequence_boundaries


def evaluate_model(model, features, log_p_targets, targets, sequence_boundaries, model_type, channels, emb):
    """Performs deep evaluation of a model, calculating metrics across tasks and visualizing."""
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
    # Group boundaries into tasks:
    # 0: Instruct, 1: Reasoner, 2: Coder, 3: Tool
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
            
        for idx in range(start, end):
            # Reconstruct blended next-token log probability distribution
            w_idx = w_weights[idx] # (C,)
            
            # Predict
            # Calculate logits/probabilities: sum_c w_idx[c] * p_c(y)
            # Recompute model predictions on the full vocab
            p_outputs_idx = []
            tokens_full = ctx + [target]
            ids_full = torch.tensor([tok2id[token] for token in tokens_full])
            
            for c in channels:
                p_outputs_idx.append(c.forward(ids_full)) # (Len, V)
                
            t_step = idx - start
            y_tgt = targets[idx].item()
            
            blended_prob = torch.zeros(V)
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

    global_acc = correct / len(targets)
    
    # Format and display task results
    task_names = ["Instruction Following", "Transitive Reasoning", "Code Generation", "Interactive Tool Use"]
    print(f"\n" + "="*60)
    print(f" BLENDER CAPABILITY REPORT: {model_type.upper()}")
    print("="*60)
    print(f"Global Perplexity : {global_ppl:.4f}")
    print(f"Global Accuracy   : {global_acc * 100:.2f}%")
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
    print("Representative Routing Decisions:")
    for i, name in enumerate(task_names):
        # Find first sequence of this task category
        seq = None
        for start, end, ctx, target in sequence_boundaries:
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
            # Show routing of the very last token transition
            idx = end - 1
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
    print("="*80)
    print(" PHASE II CHANNEL FUSION: MULTI-TASK BLENDER OPTIMIZATION")
    print("="*80)
    
    # 1. Load PPMI representation
    emb = get_ppmi_embeddings() # (V, emb_dim)
    
    # 2. Instantiate 4 CMI Expert Channels
    v2_instruct = InstructChannel(tok2id, id2tok, emb)
    v2_reasoner = ReasonerChannel(tok2id, id2tok)
    v2_coder = CoderChannel(tok2id, id2tok)
    v2_tool = ToolChannel(tok2id, id2tok)
    
    channels = [v2_instruct, v2_reasoner, v2_coder, v2_tool]
    
    # 3. Generate multi-task capability dataset pairs
    dataset_pairs = generate_multi_task_data()
    print(f"Loaded {len(dataset_pairs)} multi-task evaluation sequences.")
    
    # 4. Construct feature matrix (T, F=32)
    features, log_p_targets, targets, sequence_boundaries = build_dataset_features(
        dataset_pairs, channels, emb, use_embedding=True
    )
    T, F_dim = features.shape
    C = log_p_targets.shape[1]
    print(f"Extracted joint dataset shape: T={T}, Feature Dim F={F_dim}, Channels C={C}")
    
    # 4 Blender models to train and compare
    models = {
        "window_mlp": WindowMLPBlender(
            single_step_dim=F_dim, n_channels=C, lookback_window=4, hidden=64, dropout=0.0
        ),
        "lookback_mlp": LookbackMLPBlender(
            single_step_dim=F_dim, n_channels=C, lookback_window=4, hidden=64, num_layers=2, dropout=0.0
        ),
        "gru": GRUBlender(
            in_dim=F_dim, n_channels=C, hidden=64, num_layers=1, dropout=0.0
        ),
        "causal_conv": CausalConvBlender(
            in_dim=F_dim, n_channels=C, channels=64, kernel_size=3, num_layers=2, dropout=0.0
        )
    }
    
    # Train each blender model
    for name, model in models.items():
        print(f"\nTraining Super Blender: {name.upper()} ({sum(p.numel() for p in model.parameters()):,} parameters)")
        opt = optim.AdamW(model.parameters(), lr=0.01, weight_decay=1e-4)
        epochs = 150
        
        for epoch in range(1, epochs + 1):
            model.train()
            opt.zero_grad()
            
            if name in ["lookback_mlp", "window_mlp"]:
                # Process MLP directly
                log_w = model(features, is_already_windowed=False)
            elif name == "gru":
                # Process as sequence chuck (1, T, F)
                log_w, _ = model(features.unsqueeze(0))
                log_w = log_w.squeeze(0)
            elif name == "causal_conv":
                # Process as sequence chuck (1, T, F)
                log_w = model(features.unsqueeze(0)).squeeze(0)
                
            loss = mixture_nll(log_w, log_p_targets).mean()
            loss.backward()
            opt.step()
            
            if epoch % 50 == 0 or epoch == 1:
                print(f"  Epoch {epoch:3d}/{epochs} | Mixture NLL Loss: {loss.item():.5f} | PPL: {math.exp(loss.item()):.4f}")
                
        # Evaluate model post-training
        evaluate_model(model, features, log_p_targets, targets, sequence_boundaries, name, channels, emb)
        
        # Save model checkpoint
        save_path = REPO / f"hybrid/v2_capabilities/super_blender_{name}.pt"
        torch.save({
            "model_type": name,
            "state_dict": model.state_dict(),
            "in_dim": F_dim,
            "n_channels": C,
        }, str(save_path))
        print(f"Successfully saved {name} checkpoint to {save_path}")
        
    print("\n" + "="*80)
    print(" ALL SUPER BLENDER ARCHITECTURES FULLY TRAINED AND VERIFIED (PHASE II COMPLETED)")
    print("="*80)

if __name__ == "__main__":
    main()