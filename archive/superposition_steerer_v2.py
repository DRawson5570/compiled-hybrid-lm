"""superposition_steerer_v2.py — MLP-based superposition steerer with layer-targeted
channel groups.

Upgrades over v1:
  - MLP gatekeeper: tiny 2-layer MLP per channel group instead of linear projection
  - Layer-targeted partition: different channel groups inject at different layers
  - Per-group MLPs: each group learns domain-specific non-linear interactions

Architecture:
  Each channel group gets its own MLP: Linear(group_size, 32) -> GELU -> Linear(32, d_model)
  Groups inject at pre-specified transformer layers via forward hooks.
  RMS normalization and noise injection preserved from v1 hook_fn.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class MLPSuperpositionSteerer(nn.Module):
    """Learnable per-channel-group MLP steering injected into transformer layers.

    Args:
        num_channels: Total number of compiled channels
        d_model: Model hidden dimension
        group_channels: Dict mapping group name -> list of channel indices
        group_to_layers: Dict mapping group name -> list of layer indices to inject at
        init_scale: Scale of random initialization for MLP weights
        hidden_dim: Hidden dimension of per-group MLPs
    """

    def __init__(self, num_channels: int, d_model: int,
                 group_channels: dict[str, list[int]],
                 group_to_layers: dict[str, list[int]],
                 init_scale: float = 0.01,
                 hidden_dim: int = 32):
        super().__init__()
        self.num_channels = num_channels
        self.d_model = d_model
        self.group_channels = group_channels
        self.group_to_layers = group_to_layers
        self.init_scale = init_scale
        self.hidden_dim = hidden_dim

        for group_name in group_channels:
            if group_name not in group_to_layers:
                raise ValueError(
                    f"Group '{group_name}' has channels but no layer mapping in group_to_layers")

        self.group_mlps = nn.ModuleDict()
        for group_name, ch_indices in group_channels.items():
            in_dim = len(ch_indices)
            mlp = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, d_model),
            )
            for module in mlp:
                if isinstance(module, nn.Linear):
                    nn.init.normal_(module.weight, std=init_scale / (module.in_features ** 0.5))
                    nn.init.zeros_(module.bias)
            self.group_mlps[group_name] = mlp

        self.gamma = nn.Parameter(torch.ones(1) * 0.1)
        self.temperature = nn.Parameter(torch.ones(1))
        self.noise_scale = 0.01

        self._current_weights: torch.Tensor | None = None
        self._hooks: list = []

    def set_weights(self, batched_weights: torch.Tensor):
        """Set per-channel weights from compiled channel features.

        batched_weights: shape (B, T, C) or (T, C) or (C,)
        For (B, T, C): per-token per-batch steering
        For (T, C): same weights across batch (broadcast)
        For (C,): single weight vector for all positions (legacy)
        """
        temp = self.temperature.abs()
        if batched_weights.dim() == 1:
            w = batched_weights / temp
            self._current_weights = torch.softmax(w, dim=0).unsqueeze(0).unsqueeze(0)
        elif batched_weights.dim() == 2:
            w = batched_weights / temp
            self._current_weights = torch.softmax(w, dim=-1).unsqueeze(0)
        else:
            w = batched_weights / temp
            self._current_weights = torch.softmax(w, dim=-1)

    def _make_hook_fn(self, group_name: str):
        """Create a forward hook function for a specific channel group."""
        group_indices = self.group_channels[group_name]
        mlp = self.group_mlps[group_name]

        def hook_fn(module, input, output):
            if self._current_weights is None:
                return output

            w = self._current_weights.to(output.device)
            w_group = w[..., group_indices]

            offset = mlp(w_group)

            if offset.dim() == 2:
                offset = offset.unsqueeze(0)
            if offset.shape[0] == 1 and output.shape[0] > 1:
                offset = offset.expand(output.shape[0], -1, -1)

            if offset.shape[1] != output.shape[1]:
                if offset.shape[1] < output.shape[1]:
                    pad = torch.zeros(offset.shape[0],
                                      output.shape[1] - offset.shape[1],
                                      offset.shape[2], device=offset.device)
                    offset = torch.cat([offset, pad], dim=1)
                else:
                    offset = offset[:, :output.shape[1], :]

            if self.training and self.noise_scale > 0:
                offset = offset + torch.randn_like(offset) * self.noise_scale

            hidden_rms = output.pow(2).mean(dim=-1, keepdim=True).sqrt()
            offset_rms = offset.pow(2).mean(dim=-1, keepdim=True).sqrt().clamp(min=1e-8)
            offset = offset * (hidden_rms / offset_rms)

            return output + self.gamma.abs() * offset

        return hook_fn

    def register_hooks(self, model) -> int:
        """Register forward hooks on transformer layers per group.

        Returns number of hooks registered.
        """
        self.remove_hooks()

        if hasattr(model, 'encoder') and hasattr(model.encoder, 'layers'):
            layers = model.encoder.layers
        elif hasattr(model, 'layers'):
            layers = model.layers
        else:
            return 0

        for group_name, layer_indices in self.group_to_layers.items():
            hook_fn = self._make_hook_fn(group_name)
            for layer_idx in layer_indices:
                if layer_idx < len(layers):
                    hook = layers[layer_idx].register_forward_hook(hook_fn)
                    self._hooks.append(hook)

        return len(self._hooks)

    def remove_hooks(self):
        for hook in self._hooks:
            hook.remove()
        self._hooks = []

    def orthogonal_penalty(self) -> torch.Tensor:
        """Decorrelation penalty on MLP output weight columns.

        For each group's last Linear layer (d_model x hidden_dim),
        normalize columns and penalize ||W.T @ W - I||^2.
        Returns scalar loss averaged across groups.
        """
        if not self.group_mlps:
            return torch.tensor(0.0, device=self.gamma.device)
        total_penalty = 0.0
        for mlp in self.group_mlps.values():
            last_linear = mlp[-1]
            W = last_linear.weight
            W_norm = W / (W.norm(dim=0, keepdim=True) + 1e-8)
            gram = W_norm.T @ W_norm
            eye = torch.eye(self.hidden_dim, device=W.device)
            total_penalty = total_penalty + ((gram - eye) ** 2).mean()
        return total_penalty / len(self.group_mlps)

    def forward(self, channel_weights: torch.Tensor):
        """Convenience: set weights (no single offset in v2)."""
        self.set_weights(channel_weights)
        return None
