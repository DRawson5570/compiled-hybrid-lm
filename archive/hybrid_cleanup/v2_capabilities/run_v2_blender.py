"""Main run/wrapper script for hybrid v2 blender.

Generates multi-task capability datasets, runs predictions through the 4 compiled 
capability channels, dumps features, trains a TinyBlender MLP to dynamically 
route between capabilities, and evaluates performance.
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
    TinyBlender, build_feature_matrix, mixture_nll
)

def evaluate_and_explain(model: TinyBlender, dataset_pairs, emb: torch.Tensor,
                         v2_instruct, v2_reasoner, v2_coder, v2_tool):
    """Evaluates the trained blender, visualizes routing decisions, and computes overall PPL."""
    model.eval()
    
    channels = [v2_instruct, v2_reasoner, v2_coder, v2_tool]
    channel_names = ["InstructChannel", "ReasonerChannel", "CoderChannel", "ToolChannel"]
    
    print("\n" + "="*80)
    # Highlighted Routing Analysis
    print(" DYNAMIC ROUTING DECISIONS VISUALIZATION")
    print("="*80)
    
    nll_sum = 0.0
    total_tokens = 0
    correct_predictions = 0
    
    # We analyze routing for representative examples in each category
    with torch.no_grad():
        for pair_idx, (context, target) in enumerate(dataset_pairs):
            tokens = context + [target]
            ids = torch.tensor([tok2id[token] for token in tokens])
            T_len = len(context)
            
            # Predict for each position in sequence
            p_outputs = []
            for c in channels:
                # Get log probs for the full sequence
                p_outputs.append(c.forward(ids)) # (T_len+1, V)
                
            # For each position up to T_len, we predict the next token
            for t in range(T_len):
                y_target = ids[t+1]
                x_observed = ids[t]
                x_lag1 = ids[t-1] if t > 0 else torch.zeros_like(x_observed)
                
                # Assemble stats for each channel
                log_p_targets_t = torch.stack([p_out[t, y_target] for p_out in p_outputs]) # (C,)
                log_p_observed_t = torch.stack([p_out[t, x_observed] for p_out in p_outputs]) # (C,)
                log_p_lag1_t = torch.stack([p_out[t, x_lag1] for p_out in p_outputs]) # (C,)
                
                entropy_t = []
                max_log_prob_t = []
                for p_out in p_outputs:
                    p_dist = p_out[t].exp()
                    entropy_t.append(-(p_dist * p_out[t]).sum())
                    max_log_prob_t.append(p_out[t].max())
                    
                entropy_t = torch.stack(entropy_t)
                max_log_prob_t = torch.stack(max_log_prob_t)
                
                # Build feature vector
                feat = build_feature_matrix(
                    log_p_observed_t.unsqueeze(0),
                    log_p_lag1_t.unsqueeze(0),
                    entropy_t.unsqueeze(0),
                    max_log_prob_t.unsqueeze(0),
                    emb,
                    x_observed.unsqueeze(0),
                    use_embedding=True
                ) # (1, F)
                
                # Blender routing weights
                log_w = model(feat) # (1, C)
                w = log_w.exp().squeeze(0) # (C,)
                
                # Compute blended distribution
                blended_log_p = torch.zeros(V)
                # P_mix = sum_c w_c * P_c
                for c_idx in range(4):
                    blended_log_p += w[c_idx] * p_outputs[c_idx][t].exp()
                blended_log_p = blended_log_p.log()
                
                prediction = blended_log_p.argmax().item()
                if prediction == y_target.item():
                    correct_predictions += 1
                
                pos_nll = mixture_nll(log_w, log_p_targets_t.unsqueeze(0)).item()
                nll_sum += pos_nll
                total_tokens += 1
                
                # Let's visualize the routing at the final token transition of distinct templates
                if t == T_len - 1:
                    print(f"\nContext: {' '.join(context)}")
                    print(f"Target:  {target}")
                    print("Router Allocation:")
                    for c_idx, name in enumerate(channel_names):
                        bar = "#" * int(w[c_idx].item() * 20)
                        print(f"  - {name:<17}: {w[c_idx].item():.4f}  {bar}")
                    predicted_tok = id2tok[prediction]
                    print(f"Blended Next-Token Prediction: '{predicted_tok}' (Correct: {prediction == y_target.item()})")
                    
    avg_nll = nll_sum / total_tokens
    final_ppl = math.exp(avg_nll)
    accuracy = correct_predictions / total_tokens
    
    print("\n" + "="*80)
    print(" SUMMARY METRICS ON BLENDED EXPERIMENT SUITE")
    print("="*80)
    print(f"Total evaluated transitions: {total_tokens}")
    print(f"Next-Token Prediction Accuracy: {accuracy*100:.2f}%")
    print(f"Average Position-wise NLL:   {avg_nll:.5f}")
    print(f"Blended Suite Perplexity:     {final_ppl:.4f}")
    print("="*80 + "\n")
    
    return {
        "accuracy": accuracy,
        "nll": avg_nll,
        "perplexity": final_ppl
    }


def main():
    # 1. Load setup
    emb = get_ppmi_embeddings() # SVD embeddings, size (V, d)
    
    # 2. Instantiate 4 compiled capability channels
    v2_instruct = InstructChannel(tok2id, id2tok, emb)
    v2_reasoner = ReasonerChannel(tok2id, id2tok)
    v2_coder = CoderChannel(tok2id, id2tok)
    v2_tool = ToolChannel(tok2id, id2tok)
    
    channels = [v2_instruct, v2_reasoner, v2_coder, v2_tool]
    
    # 3. Generate multi-task pairs
    dataset_pairs = generate_multi_task_data()
    print(f"Generated {len(dataset_pairs)} multi-task evaluation sequences.")
    
    # 4. Extract token-wise feature matrices and target logprobs
    all_feats = []
    all_log_p_targets = []
    
    for context, target in dataset_pairs:
        tokens = context + [target]
        ids = torch.tensor([tok2id[token] for token in tokens])
        T_len = len(context)
        
        # Collect predictions from each channel
        p_outputs = []
        for c in channels:
            p_outputs.append(c.forward(ids)) # (T_len+1, V)
            
        for t in range(T_len):
            y_target = ids[t+1]
            x_observed = ids[t]
            x_lag1 = ids[t-1] if t > 0 else torch.zeros_like(x_observed)
            
            # Channel scores on target, observed, lag1, entropy & max log prob
            log_p_targets_t = torch.stack([p_out[t, y_target] for p_out in p_outputs]) # (C,)
            log_p_observed_t = torch.stack([p_out[t, x_observed] for p_out in p_outputs]) # (C,)
            log_p_lag1_t = torch.stack([p_out[t, x_lag1] for p_out in p_outputs]) # (C,)
            
            entropy_t = []
            max_log_prob_t = []
            for p_out in p_outputs:
                p_dist = p_out[t].exp()
                entropy_t.append(-(p_dist * p_out[t]).sum())
                max_log_prob_t.append(p_out[t].max())
                
            entropy_t = torch.stack(entropy_t)
            max_log_prob_t = torch.stack(max_log_prob_t)
            
            # Feature representation at position t
            feat = build_feature_matrix(
                log_p_observed_t.unsqueeze(0),
                log_p_lag1_t.unsqueeze(0),
                entropy_t.unsqueeze(0),
                max_log_prob_t.unsqueeze(0),
                emb,
                x_observed.unsqueeze(0),
                use_embedding=True
            ) # (1, F)
            
            all_feats.append(feat)
            all_log_p_targets.append(log_p_targets_t)
            
    # Concatenate all rows into unified matrices
    features = torch.cat(all_feats, dim=0) # (Total_T, F)
    log_p_targets = torch.stack(all_log_p_targets, dim=0) # (Total_T, C)
    
    print(f"Assembled feature matrix shape: {features.shape}")
    print(f"Assembled target matrix shape:  {log_p_targets.shape}")
    
    # 5. Initialize Twin-Layer TinyBlender MLP
    # Feature count (F): 4 stats * 4 channels + 16 (emb dim) = 32
    in_dim = features.shape[1]
    model = TinyBlender(in_dim=in_dim, n_channels=4, hidden=64, dropout=0.0)
    
    # 6. Train using mixture NLL minimization
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
    epochs = 120
    
    print("\nTraning TinyBlender routing optimizer...")
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        
        log_w = model(features) # (Total_T, C)
        loss_t = mixture_nll(log_w, log_p_targets)
        loss = loss_t.mean()
        
        loss.backward()
        optimizer.step()
        
        if epoch % 20 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{epochs} | Mixture Loss (NLL): {loss.item():.5f} | Joint PPL: {math.exp(loss.item()):.4f}")
            
    # 7. Evaluate and report results
    evaluate_and_explain(model, dataset_pairs, emb, v2_instruct, v2_reasoner, v2_coder, v2_tool)
    
    # Save model artifact
    save_path = REPO / "hybrid/v2_capabilities/blender_v2.pt"
    torch.save(model.state_dict(), str(save_path))
    print(f"Saved trained TinyBlender routing model to {save_path}")

if __name__ == "__main__":
    main()
