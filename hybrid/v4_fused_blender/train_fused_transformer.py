"""hybrid/v4_fused_blender/train_fused_transformer.py

Trains and evaluates our newly validated Causal Self-Attention TransformerBlender
model on the interleaving multi-capability dataset task.
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
    tok2id, id2tok, V, get_ppmi_embeddings, generate_multi_task_data
)
from hybrid.v1_blender.blender_model import (
    mixture_nll
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Reuse the Causal Transformer architecture validated in our local tests
class TBConfig:
    def __init__(self, in_dim: int, n_channels: int, d_model: int = 128, n_heads: int = 4, d_ff: int = 256, n_layers: int = 2, ctx: int = 256, dropout: float = 0.1):
        self.in_dim = in_dim
        self.n_channels = n_channels
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.n_layers = n_layers
        self.ctx = ctx
        self.dropout = dropout


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        qkv = self.qkv(x).view(B, T, 3, self.n_heads, self.d_head)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)  # (B, H, T, dh)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # causal SDPA
        out = F.scaled_dot_product_attention(
            q, k, v, is_causal=True,
            dropout_p=self.drop.p if self.training else 0.0
        )
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.proj(out)


class TransformerBlock(nn.Module):
    def __init__(self, cfg: TBConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg.d_model, cfg.n_heads, cfg.dropout)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.ff = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_ff),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_ff, cfg.d_model),
        )
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop(self.attn(self.ln1(x)))
        x = x + self.drop(self.ff(self.ln2(x)))
        return x


class TransformerBlender(nn.Module):
    """Contextual Causal Transformer Mixer over compile next-token probabilities."""
    def __init__(self, cfg: TBConfig):
        super().__init__()
        self.cfg = cfg
        self.in_proj = nn.Linear(cfg.in_dim, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.ctx, cfg.d_model)
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.n_channels)
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, F_in = x.shape
        assert T <= self.cfg.ctx, f"SeqLen {T} matches ctx bound {self.cfg.ctx}"
        pos = torch.arange(T, device=x.device)
        h = self.in_proj(x) + self.pos_emb(pos)[None, :, :]
        for blk in self.blocks:
            h = blk(h)
        h = self.ln_f(h)
        logits = self.head(h)
        return F.log_softmax(logits, dim=-1)


def main():
    print("[train_fused_transformer] Preparing capability-fused dataset blocks...")
    pairs = generate_multi_task_data()
    emb = get_ppmi_embeddings().to(DEVICE)
    
    # Instantiate 4 specialized experts
    channels = [
        InstructChannel(tok2id, id2tok, emb),
        ReasonerChannel(tok2id, id2tok),
        CoderChannel(tok2id, id2tok),
        ToolChannel(tok2id, id2tok)
    ]
    C = len(channels)
    
    # Construct task features
    print(f"[train_fused_transformer] Compiling statistics over {len(pairs)} contextual paths...")
    seq_features = []
    seq_log_p_targets = []
    seq_targets = []
    
    for context, target in pairs:
        tokens_full = context + [target]
        ids_full = torch.tensor([tok2id[token] for token in tokens_full], device=DEVICE)
        
        # Expert next-token prediction profiles (C, T, V)
        prob_profiles = []
        for chan in channels:
            prob_profiles.append(chan.forward(ids_full).unsqueeze(0)) # (1, T, V)
            
        prob_profiles = torch.cat(prob_profiles, dim=0) # (C, T, V)
        T_len = len(context)
        
        # Target representation
        y_tgt = ids_full[1:] # T targets
        for idx in range(T_len):
            target_tok_id = y_tgt[idx].item()
            # Expert target log probs as contextual blender input (C)
            tgt_log_probs = prob_profiles[:, idx, target_tok_id] # (C)
            seq_features.append(tgt_log_probs.unsqueeze(0))
            seq_log_p_targets.append(prob_profiles[:, idx, target_tok_id].unsqueeze(0))
            seq_targets.append(target_tok_id)

    features = torch.cat(seq_features, dim=0).unsqueeze(0).to(DEVICE) # (1, Total_T, C)
    log_p_targets = torch.cat(seq_log_p_targets, dim=0).to(DEVICE) # (Total_T, C)
    targets = torch.tensor(seq_targets, device=DEVICE)
    
    Total_T = features.shape[1]
    print(f"[train_fused_transformer] Complete. Features size: {features.shape}, target space size: {targets.shape}")

    # Build Blender Config
    cfg = TBConfig(in_dim=C, n_channels=C, d_model=128, n_heads=4, d_ff=256, n_layers=2, ctx=Total_T + 128)
    model = TransformerBlender(cfg).to(DEVICE)
    print(f"[train_fused_transformer] Active Contextual Self-Attention Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train the Model
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    
    model.train()
    print("[train_fused_transformer] Starting alignment optimization sweeps...")
    for epoch in range(15):
        optimizer.zero_grad()
        log_w = model(features).squeeze(0) # (Total_T, C)
        
        loss = mixture_nll(log_w, log_p_targets)
        avg_loss = loss.mean()
        avg_loss.backward()
        optimizer.step()
        
        ppl = math.exp(avg_loss.item())
        print(f"  Epoch {epoch+1:02d}/15 | Global mixture NLL: {avg_loss.item():.5f} | Perplexity: {ppl:.5f}")
        
    # Save the Transformer Blender model
    save_dir = Path(__file__).resolve().parent / "saved_models"
    save_dir.mkdir(exist_ok=True, parents=True)
    save_path = save_dir / "blender_transformer.pt"
    torch.save(model.state_dict(), save_path)
    print(f"[train_fused_transformer] Saved contextual self-attention blender weights to {save_path}")


if __name__ == "__main__":
    main()