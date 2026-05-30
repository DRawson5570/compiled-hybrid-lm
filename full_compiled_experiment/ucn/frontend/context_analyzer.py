from __future__ import annotations

import torch
import torch.nn as nn


class ContextAnalyzer(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_latent: int = 128,
        n_layers: int = 2,
        use_pooling: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_latent = d_latent
        self.n_layers = n_layers
        self.use_pooling = use_pooling

        self.input_proj = nn.Linear(d_model, d_latent)

        self.gru = nn.GRU(
            input_size=d_latent,
            hidden_size=d_latent,
            num_layers=n_layers,
            batch_first=True,
            dropout=0.1 if n_layers > 1 else 0.0,
        )

        self.output_norm = nn.LayerNorm(d_latent)

    def forward(
        self, x: torch.Tensor
    ) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(0)

        b, seq, d = x.shape

        x_proj = self.input_proj(x.float())

        gru_out, h_n = self.gru(x_proj)

        if self.use_pooling:
            z = gru_out.mean(dim=1)
        else:
            z = h_n[-1]

        z = self.output_norm(z)
        return z
