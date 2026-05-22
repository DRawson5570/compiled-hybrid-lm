"""hybrid/v4_fused_blender/generate_fused.py

Automated Generation Client mapping user prompt contexts, evaluating
step-by-step channel routing weights, and producing interactive completions.
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

# Import capability definitions
from hybrid.v2_capabilities.channels import (
    InstructChannel, ReasonerChannel, CoderChannel, ToolChannel
)
from hybrid.v2_capabilities.dataset import (
    tok2id, id2tok, V, get_ppmi_embeddings
)
from hybrid.v4_fused_blender.train_fused_transformer import (
    TransformerBlender, TBConfig
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class UnifiedAutoGenerator:
    def __init__(self, model_path: Path):
        self.emb = get_ppmi_embeddings().to(DEVICE)
        self.channels = [
            InstructChannel(tok2id, id2tok, self.emb),
            ReasonerChannel(tok2id, id2tok),
            CoderChannel(tok2id, id2tok),
            ToolChannel(tok2id, id2tok)
        ]
        self.C = len(self.channels)
        
        # Load trained transformer model configuration
        cfg = TBConfig(in_dim=self.C, n_channels=self.C, d_model=128, n_heads=4, d_ff=256, n_layers=2, ctx=416)
        self.blender = TransformerBlender(cfg).to(DEVICE)
        
        if model_path.exists():
            print(f"[generate_fused] Loading pre-trained Self-Attention weights from {model_path}...")
            self.blender.load_state_dict(torch.load(model_path, map_location=DEVICE))
        else:
            print("[generate_fused] WARNING: Saved weights not found. Running with direct initialization weights.")
            
        self.blender.eval()

    def generate(self, prompt: str, max_new_tokens: int = 8, temperature: float = 0.7, top_p: float = 0.9) -> str:
        tokens = prompt.split()
        # Fallback for out of vocabulary words
        token_ids = [tok2id.get(tok, tok2id["<UNK>"]) for tok in tokens]
        
        print(f"\n[generate_fused] Coding Context prompt: '{prompt}'")
        print(f"[generate_fused] Vectorized IDs: {token_ids}")

        generated = []
        for _ in range(max_new_tokens):
            curr_len = len(token_ids)
            ids_tensor = torch.tensor(token_ids, device=DEVICE)
            
            # Forward-pass each expert channel
            channel_probs = []
            for chan in self.channels:
                chan_probs = chan.forward(ids_tensor) # (curr_len, V)
                channel_probs.append(chan_probs.unsqueeze(0)) # (1, curr_len, V)
                
            channel_probs = torch.cat(channel_probs, dim=0) # (C, curr_len, V)
            
            # Predict step features (C)
            # Find probability vectors corresponding to input history sequences
            step_feats = []
            for idx in range(curr_len):
                # Retrieve log probs of current tokens predicting themselves in history
                t_id = token_ids[idx]
                step_feats.append(channel_probs[:, idx, t_id].unsqueeze(0))
                
            # Shape: (1, curr_len, C)
            features = torch.cat(step_feats, dim=0).unsqueeze(0).to(DEVICE)
            
            # Feed feature sequence to Causal Transformer Blender to predict routing weights
            with torch.no_grad():
                log_w = self.blender(features).squeeze(0) # (curr_len, C)
                # Weights for the latest target prediction step
                w_weights = log_w[-1].exp() # (C)
                
            # Merge probabilities of specialized expert distributions
            latest_prob_dist = torch.zeros(V, device=DEVICE)
            latest_chan_log_p = channel_probs[:, -1, :] # (C, V)
            
            for c_idx in range(self.C):
                latest_prob_dist += w_weights[c_idx] * latest_chan_log_p[c_idx].exp()
                
            logits = latest_prob_dist.log()
            
            # Temperature Scaling
            if temperature > 0.0:
                logits = logits / temperature
                
            # Top-p Nuclear truncation
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            
            indices_to_remove = cumulative_probs > top_p
            # Keep at least the highest probability candidate
            indices_to_remove[..., 1:] = indices_to_remove[..., :-1].clone()
            indices_to_remove[..., 0] = 0
            
            logits[sorted_indices[indices_to_remove]] = -float("Inf")
            probs = F.softmax(logits, dim=-1)
            
            next_token_id = torch.multinomial(probs, num_samples=1).item()
            token_ids.append(next_token_id)
            
            out_tok = id2tok.get(next_token_id, "<UNK>")
            generated.append(out_tok)
            
            channels_info = ", ".join([f"Ch{j}:{w_weights[j]:.3f}" for j in range(self.C)])
            print(f"  -> Step Generated token: {out_tok:<12} | Routing weight profile: [{channels_info}]")
            
            if out_tok in ["<PAD>", "."]:
                break
                
        return " ".join(generated)


def main():
    model_path = Path(__file__).resolve().parent / "saved_models" / "blender_transformer.pt"
    client = UnifiedAutoGenerator(model_path)
    
    print("\n[generate_fused] Initiating sample capability trials:")
    
    # Task 1: Rule-following instruction translation
    client.generate("translate cat to french", max_new_tokens=4, temperature=0.2)
    
    # Task 2: Code instruction template context
    client.generate("def get_sum ( a , b )", max_new_tokens=5, temperature=0.1)


if __name__ == "__main__":
    main()