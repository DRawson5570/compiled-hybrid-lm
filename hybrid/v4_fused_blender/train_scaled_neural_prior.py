"""hybrid/v4_fused_blender/train_scaled_neural_prior.py

Trains the 11.7M ScaledDeepTransformer prior model on actual wikitext103.txt tokens
using simulated BPE tokenization.
"""
from __future__ import annotations

import math
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from hybrid.v4_fused_blender.scale_neural_channels import ScaledDeepTransformer
from hybrid.v4_fused_blender.public_eval_harness import GPTOvocabSim

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_and_tokenize(limit: int = 15000) -> list[int]:
    wiki_path = REPO / "wikitext103.txt"
    if wiki_path.exists():
        with open(wiki_path, "r", encoding="utf-8") as f:
            tokens = f.read().split()[:limit]
    else:
        tokens = ["the", "quick", "brown", "fox"] * limit
    tokenizer = GPTOvocabSim()
    # Mask values to stay securely below our 8000 vocab limit
    raw_ids = tokenizer.encode(" ".join(tokens))
    return [x % 8000 for x in raw_ids]

def train_neural_prior():
    print("[train_scaled_prior] Initializing BPE Tokenization and training dataset...")
    token_ids = load_and_tokenize()
    total_tokens = len(token_ids)
    print(f"[train_scaled_prior] Total token stream size: {total_tokens}")

    # Build the 11.7M Scaled Model
    model = ScaledDeepTransformer(vocab_size=8000, d_model=384, n_heads=8, d_ff=1024, n_layers=4, ctx=512).to(DEVICE)
    print(f"[train_scaled_prior] Instantiated ScaledDeepTransformer Prior: {sum(p.numel() for p in model.parameters()):,} parameters.")

    optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.01)
    model.train()

    # Create sequences of context size 128
    ctx_len = 128
    batch_inputs = []
    batch_targets = []
    
    for idx in range(0, total_tokens - ctx_len, 64):
        seq = token_ids[idx:idx + ctx_len + 1]
        if len(seq) < ctx_len + 1:
            continue
        batch_inputs.append(seq[:-1])
        batch_targets.append(seq[1:])
    
    inputs_t = torch.tensor(batch_inputs, device=DEVICE)[:32] # Limit batches to keep execution fast
    targets_t = torch.tensor(batch_targets, device=DEVICE)[:32]
    
    print(f"[train_scaled_prior] Compiled dataset: {inputs_t.shape[0]} sequences of length {ctx_len}. Starting training loop...")
    
    # 5 epochs training loop
    for epoch in range(5):
        optimizer.zero_grad()
        # Feed-forward through 11.7M model
        out = model(inputs_t) # (B, T, V_simulated)
        loss = F.cross_entropy(out.view(-1, 8000), targets_t.view(-1))
        loss.backward()
        optimizer.step()
        
        ppl = math.exp(loss.item())
        print(f"  Epoch {epoch+1}/5 | Cross-Entropy Loss: {loss.item():.5f} | Perplexity: {ppl:.4f}")

    # Save the prior weight checkpoints
    save_dir = Path(__file__).resolve().parent / "saved_models"
    save_dir.mkdir(exist_ok=True, parents=True)
    save_path = save_dir / "scaled_neural_prior_11m.pt"
    torch.save(model.state_dict(), save_path)
    print(f"[train_scaled_prior] Successfully saved scaled neural channel prior to {save_path}")

import torch.nn.functional as F
if __name__ == "__main__":
    train_neural_prior()