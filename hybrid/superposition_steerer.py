"""superposition_steerer.py — Activation-level compiled channel injection.

Maps compiled channel log-probabilities to activation offsets in the
transformer residual stream. Registers forward hooks to inject at specific
layers. Supports three modes:
  - output:  blend at logit level (current default)
  - superposition: inject as activation offsets mid-computation
  - both:  apply both paths

Architecture:
  Each compiled channel gets a learnable d_model-dimensional steering vector.
  At injection time: offset = Σ_c w_c(ctx) · steer_vector[c]
  where w_c is derived from the channel's log-prob (or blender weights).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class SuperpositionSteerer(nn.Module):
    """Learnable per-channel steering vectors injected into transformer layers.

    Args:
        num_channels: Number of compiled channels
        d_model: Model hidden dimension
        inject_layers: Which transformer encoder layers to inject at (0-indexed)
        init_scale: Scale of random initialization for steering vectors
    """

    def __init__(self, num_channels: int, d_model: int,
                 inject_layers: list[int] | None = None,
                 init_scale: float = 0.01):
        super().__init__()
        self.num_channels = num_channels
        self.d_model = d_model
        self.inject_layers = inject_layers or [0, 4, 8]
        self.init_scale = init_scale

        # Per-channel learnable steering vectors
        self.steer_vectors = nn.Parameter(
            torch.randn(num_channels, d_model) * init_scale / (d_model ** 0.5)
        )
        # Learned softmax temperature (starts at 1.0 — no change initially)
        self.temperature = nn.Parameter(torch.ones(1))
        # Learned injection gate (starts at 0.1 — gentle initial influence)
        self.gamma = nn.Parameter(torch.ones(1) * 0.1)
        # Training noise scale (0 = disabled)
        self.noise_scale = 0.01

        # Current context-dependent weights (updated each step from compiled channels)
        self._current_weights: torch.Tensor | None = None
        self._current_offsets: torch.Tensor | None = None  # per-position offsets for injection
        self._hooks = []

    def set_weights(self, batched_weights: torch.Tensor):
        """Set per-channel weights from compiled channel features.
        
        batched_weights: shape (B, T, C) or (T, C) or (C,)
        For (B, T, C): per-token per-batch steering
        For (T, C): same weights across batch (broadcast)
        For (C,): single weight vector for all positions (legacy)
        """
        if batched_weights.dim() == 1:
            # Legacy: single weight vector
            w = batched_weights / self.temperature.abs()
            self._current_weights = torch.softmax(w, dim=0).unsqueeze(0).unsqueeze(0)  # (1,1,C)
        elif batched_weights.dim() == 2:
            w = batched_weights / self.temperature.abs()
            self._current_weights = torch.softmax(w, dim=-1).unsqueeze(0)  # (1,T,C)
        else:
            w = batched_weights / self.temperature.abs()
            self._current_weights = torch.softmax(w, dim=-1)  # (B,T,C)
        self._current_offsets = None  # clear per-position offsets mode

    def compute_offset(self) -> torch.Tensor:
        """Compute activation offset: Σ_c w_c · steer_vector[c]."""
        if self._current_weights is None:
            return torch.zeros(self.d_model)
        return (self._current_weights.unsqueeze(0) @ self.steer_vectors).squeeze(0)

    def compute_per_position_offsets(self, per_pos_weights: torch.Tensor) -> torch.Tensor:
        """Compute per-position offsets from (T, C) weight matrix.
        Returns (T, d_model) offset tensor."""
        return per_pos_weights @ self.steer_vectors  # (T, C) @ (C, d) -> (T, d)

    def set_per_position_offsets(self, offsets: torch.Tensor):
        """Set pre-computed per-position offsets for hook injection."""
        self._current_offsets = offsets
        self._current_weights = None  # clear single-offset mode

    def _hook_fn(self, module, input, output):
        """Forward hook: add RMS-normalized steering offset to residual stream.
        Supports (B,T,C) per-token weights via einsum."""
        if self._current_weights is None:
            return output
        
        w = self._current_weights.to(output.device)
        
        # Expand weights to match batch/time dims of output
        if w.dim() == 2:
            w = w.unsqueeze(0)
        if w.dim() == 3 and w.shape[0] == 1 and output.shape[0] > 1:
            w = w.expand(output.shape[0], -1, -1)
        
        if w.shape[1] != output.shape[1]:
            if w.shape[1] < output.shape[1]:
                pad = torch.zeros(w.shape[0], output.shape[1] - w.shape[1], w.shape[2],
                                  device=w.device)
                w = torch.cat([w, pad], dim=1)
            else:
                w = w[:, :output.shape[1], :]
        
        # Training noise: add jitter to weights for robustness
        if self.training and self.noise_scale > 0:
            w = w + torch.randn_like(w) * self.noise_scale
        
        offset = torch.einsum('btc,cd->btd', w, self.steer_vectors.to(w.device))
        
        # RMS normalization: match steering offset scale to hidden state scale
        hidden_rms = output.pow(2).mean(dim=-1, keepdim=True).sqrt()
        offset_rms = offset.pow(2).mean(dim=-1, keepdim=True).sqrt().clamp(min=1e-8)
        offset = offset * (hidden_rms / offset_rms)
        
        return output + self.gamma.abs() * offset

    def register_hooks(self, model) -> int:
        """Register forward hooks on transformer encoder layers.
        Returns number of hooks registered.
        """
        self.remove_hooks()

        # Try to find encoder layers
        if hasattr(model, 'encoder') and hasattr(model.encoder, 'layers'):
            layers = model.encoder.layers
        elif hasattr(model, 'layers'):
            layers = model.layers
        else:
            return 0

        for idx in self.inject_layers:
            if idx < len(layers):
                hook = layers[idx].register_forward_hook(self._hook_fn)
                self._hooks.append(hook)

        return len(self._hooks)

    def remove_hooks(self):
        for hook in self._hooks:
            hook.remove()
        self._hooks = []

    def orthogonal_penalty(self) -> torch.Tensor:
        """Decorrelation penalty: || V V^T - I ||^2.
        Returns scalar loss to add to training objective.
        """
        V = self.steer_vectors  # (C, d)
        V_norm = V / (V.norm(dim=1, keepdim=True) + 1e-8)
        gram = V_norm @ V_norm.T  # (C, C)
        eye = torch.eye(self.num_channels, device=V.device)
        return ((gram - eye) ** 2).mean()

    def forward(self, channel_lps: list[torch.Tensor]):
        """Convience: set weights and return offset."""
        self.set_weights(channel_lps)
        return self.compute_offset()
