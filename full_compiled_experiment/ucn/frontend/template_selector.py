from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemplateSelector(nn.Module):
    def __init__(self, d_latent: int, n_templates: int):
        super().__init__()
        self.d_latent = d_latent
        self.n_templates = n_templates

        self.classifier = nn.Sequential(
            nn.Linear(d_latent, d_latent * 2),
            nn.ReLU(),
            nn.Linear(d_latent * 2, n_templates),
        )

    def forward(
        self, z: torch.Tensor, temperature: float = 1.0
    ) -> torch.Tensor:
        logits = self.classifier(z.float()) / max(temperature, 1e-6)
        return logits

    def sample_gumbel(
        self, z: torch.Tensor, temperature: float = 1.0, hard: bool = False
    ) -> torch.Tensor:
        logits = self.forward(z, temperature)
        return F.gumbel_softmax(logits, tau=temperature, hard=hard, dim=-1)

    def predict_template(self, z: torch.Tensor) -> torch.Tensor:
        logits = self.forward(z)
        return logits.argmax(dim=-1)
