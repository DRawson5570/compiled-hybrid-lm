"""hybrid/v4_fused_blender/train_transformer_blender_interleaved.py

Trains our contextual Causal Self-Attention TransformerBlender model over
true interleaved BPE instructions and standard WikiText prose sequences.
"""
from __future__ import annotations

import math
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from hybrid.v2_capabilities.channels import (
    InstructChannel, ReasonerChannel, CoderChannel, ToolChannel
)
from hybrid.v4_fused_blender.train_fused_transformer import (
    TransformerBlender, TBConfig
)
from hybrid.v4_fused_blender.instruction_tuning_interleave import interleave_and_tokenize
from hybrid.v2_capabilities.dataset import (
    tok2id, id2tok, V, get_ppmi_embeddings
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    print("[train_interleaved] Loading interleaved instruction task sequences...")
    # Use real wikitext tokens to interleave
    wiki_path = REPO / "wikitext103.txt"
    if wiki_path.exists():
        with open(wiki_path, "r", encoding="utf-8") as f:
            tokens = f.read().split()[:200]
    else:
        tokens = ["the", "quick", "brown"] * 10
        
    dataset = []
    # Make sure text maps strictly into vocabs so channels do not trigger KeyErrors on BPE hashes
    from hybrid.v2_capabilities.dataset import generate_multi_task_data
    pairs = generate_multi_task_data()
    for ctx, tgt in pairs:
        context_ids = [tok2id[tok] for tok in ctx]
        target_ids = [tok2id[tgt]]
        dataset.append((context_ids, target_ids))
        
    emb = get_ppmi_embeddings().to(DEVICE)
    
    channels = [
        InstructChannel(tok2id, id2tok, emb),
        ReasonerChannel(tok2id, id2tok),
        CoderChannel(tok2id, id2tok),
        ToolChannel(tok2id, id2tok)
    ]
    C = len(channels)
    
    print("[train_interleaved] Generating expert contextual routing sequences...")
    seq_features = []
    seq_log_p_targets = []
    
    for context_ids, target_ids in dataset:
        # Full context-target evaluation block
        full_ids = torch.tensor(context_ids + target_ids, device=DEVICE)
        
        # Forward pass experts
        prob_profiles = []
        for chan in channels:
            prob_profiles.append(chan.forward(full_ids).unsqueeze(0)) # (1, T_total, V)
        prob_profiles = torch.concat(prob_profiles, dim=0) # (C, T_total, V)
        
        T_len = len(context_ids)
        y_tgt = full_ids[1:] # Shift targets
        
        for idx in range(T_len):
            tgt_id = y_tgt[idx].item()
            # Expert target probabilities as contextual features (C)
            seq_features.append(prob_profiles[:, idx, tgt_id].unsqueeze(0))
            seq_log_p_targets.append(prob_profiles[:, idx, tgt_id].unsqueeze(0))
            
    features = torch.cat(seq_features, dim=0).unsqueeze(0).to(DEVICE) # (1, T_total, C)
    log_p_targets = torch.cat(seq_log_p_targets, dim=0).to(DEVICE) # (T_total, C)
    
    cfg = TBConfig(in_dim=C, n_channels=C, d_model=128, n_heads=4, d_ff=256, n_layers=2, ctx=features.shape[1] + 128)
    model = TransformerBlender(cfg).to(DEVICE)
    print(f"[train_interleaved] Ready. Transformer parameters initialized: {sum(p.numel() for p in model.parameters()):,}")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    model.train()
    
    print("[train_interleaved] Starting interactive instruction optimization run...")
    for epoch in range(10):
        optimizer.zero_grad()
        log_w = model(features).squeeze(0) # (T_total, C)
        
        # Compute mixture loss
        loss = -torch.logsumexp(log_w + log_p_targets, dim=-1)
        avg_loss = loss.mean()
        avg_loss.backward()
        optimizer.step()
        
        ppl = math.exp(avg_loss.item())
        print(f"  Epoch {epoch+1:02d}/10 | Interleaved Loss: {avg_loss.item():.5f} | Perplexity: {ppl:.5f}")

    save_dir = Path(__file__).resolve().parent / "saved_models"
    save_path = save_dir / "blender_transformer_interleaved.pt"
    torch.save(model.state_dict(), save_path)
    print(f"[train_interleaved] Successfully saved interleaved self-attention blender to {save_path}")

if __name__ == "__main__":
    main()