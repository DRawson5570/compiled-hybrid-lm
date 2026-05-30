from __future__ import annotations

import torch
import torch.nn as nn


class ParameterGenerator(nn.Module):
    def __init__(self, d_latent: int, n_params: int):
        super().__init__()
        self.d_latent = d_latent
        self.n_params = n_params

        self.net = nn.Sequential(
            nn.Linear(d_latent, d_latent * 2),
            nn.ReLU(),
            nn.Linear(d_latent * 2, d_latent),
            nn.ReLU(),
            nn.Linear(d_latent, n_params),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        raw = self.net(z.float())

        params = torch.sigmoid(raw)
        return params
