"""Causal LM that consumes compiled-channel features as model inputs.

This is the code path for hybrid architecture #1 from HYBRID_STRATEGY.md:
compiled feature vectors are projected into the transformer's hidden stream
alongside token and position embeddings.  The compiled features must be causal
features for the same input positions; this module enforces alignment but does
not construct the features itself.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class CompiledFeatureTransformerConfig:
    vocab_size: int
    feature_dim: int
    d_model: int = 256
    n_layers: int = 6
    n_heads: int = 8
    d_ff: int = 1024
    max_seq_len: int = 256
    dropout: float = 0.1
    feature_dropout: float = 0.0


class CompiledFeatureTransformer(nn.Module):
    """Decoder-only transformer with additive compiled-feature conditioning.

    Args:
        input_ids: ``(B, T)`` token IDs.
        compiled_features: ``(B, T, F)`` causal feature vectors aligned with
            ``input_ids``.  Future-target features must not be included.

    Returns:
        ``(B, T, V)`` next-token logits.
    """

    def __init__(self, cfg: CompiledFeatureTransformerConfig):
        super().__init__()
        if cfg.d_model % cfg.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.cfg = cfg
        self.vocab = cfg.vocab_size

        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.feature_proj = nn.Sequential(
            nn.LayerNorm(cfg.feature_dim),
            nn.Linear(cfg.feature_dim, cfg.d_model),
        )
        self.feature_gate = nn.Parameter(torch.tensor(1.0))
        self.dropout = nn.Dropout(cfg.dropout)
        self.feature_dropout = nn.Dropout(cfg.feature_dropout)

        layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.d_ff,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=cfg.n_layers)
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head_bias = nn.Parameter(torch.zeros(cfg.vocab_size))

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.token_emb.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.pos_emb.weight, mean=0.0, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, input_ids: torch.Tensor, compiled_features: torch.Tensor) -> torch.Tensor:
        if input_ids.ndim != 2:
            raise ValueError(f"input_ids must have shape (B, T), got {tuple(input_ids.shape)}")
        if compiled_features.ndim != 3:
            raise ValueError(
                f"compiled_features must have shape (B, T, F), got {tuple(compiled_features.shape)}"
            )
        batch, seq_len = input_ids.shape
        if compiled_features.shape[:2] != (batch, seq_len):
            raise ValueError(
                "compiled feature positions must align with tokens: "
                f"tokens={(batch, seq_len)} features={tuple(compiled_features.shape[:2])}"
            )
        if compiled_features.shape[-1] != self.cfg.feature_dim:
            raise ValueError(
                f"expected feature_dim={self.cfg.feature_dim}, got {compiled_features.shape[-1]}"
            )
        if seq_len > self.cfg.max_seq_len:
            raise ValueError(f"sequence length {seq_len} exceeds max_seq_len={self.cfg.max_seq_len}")

        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        token_h = self.token_emb(input_ids)
        pos_h = self.pos_emb(positions)
        feature_h = self.feature_proj(self.feature_dropout(compiled_features.float()))
        hidden = token_h + pos_h + self.feature_gate * feature_h
        hidden = self.dropout(hidden)

        mask = nn.Transformer.generate_square_subsequent_mask(seq_len, device=input_ids.device)
        hidden = self.transformer(hidden, mask=mask, is_causal=True)
        hidden = self.ln_f(hidden)
        return hidden @ self.token_emb.weight.T + self.head_bias

    @torch.no_grad()
    def next_token_log_probs(self, input_ids: torch.Tensor, compiled_features: torch.Tensor) -> torch.Tensor:
        """Return log-probabilities for the next token after the final position."""
        logits = self(input_ids, compiled_features)
        return F.log_softmax(logits[:, -1], dim=-1)
