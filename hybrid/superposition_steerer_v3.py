"""superposition_steerer_v3.py — 21-Channel Multi-Timescale MLP Steerer.

Aligned to handle: 6 local + 7 mid + 8 global = 21 channels.
Per-group MLPs with proper slicing. Layer-targeted injection.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SuperpositionSteererV3(nn.Module):
    """Upgraded 21-Channel MLP Steerer with Multi-Timescale Layer Routing.

    Channel Indices:
      [0:6]   Local Group (6):      uni, bi_fast, bi_slow, tri_fast, tri_slow, skip2
      [6:13]  Mid Group (7):        skip3, recency, entropy, shape, global_uni, ppmi_cos, ppmi_max
      [13:21] Global Group (8):     ppmi_norm, punct_density, repetition, unique_ratio, topic, KV, POS, spare
    """

    def __init__(self, d_model: int = 768, inject_layers: list[int] | None = None,
                 init_scale: float = 0.01, noise_scale: float = 0.05):
        super().__init__()
        self.num_channels = 21
        self.d_model = d_model
        self.noise_scale = noise_scale

        self.layer_routing = {
            0: 'local', 1: 'local', 2: 'local',
            4: 'mid', 5: 'mid', 6: 'mid',
            8: 'global', 9: 'global', 10: 'global'
        }
        self.inject_layers = inject_layers or list(self.layer_routing.keys())

        # Per-group steering vectors
        self.steer_local = nn.Parameter(torch.randn(6, d_model) * init_scale / (d_model ** 0.5))
        self.steer_mid = nn.Parameter(torch.randn(7, d_model) * init_scale / (d_model ** 0.5))
        self.steer_global = nn.Parameter(torch.randn(8, d_model) * init_scale / (d_model ** 0.5))

        # Non-linear Gating MLPs (expanded for 21 channels)
        self.local_mlp = nn.Sequential(
            nn.Linear(6, 12), nn.GELU(), nn.Linear(12, 6))
        self.mid_mlp = nn.Sequential(
            nn.Linear(7, 14), nn.GELU(), nn.Linear(14, 7))
        self.global_mlp = nn.Sequential(
            nn.Linear(8, 16), nn.GELU(), nn.Linear(16, 8))

        # Learned per-layer injection scalars
        self.gammas = nn.ParameterDict({
            str(layer): nn.Parameter(torch.tensor(0.01)) for layer in self.inject_layers
        })

        self._current_weights: torch.Tensor | None = None
        self._hooks = []

    def set_weights(self, weights: torch.Tensor):
        if self.training and self.noise_scale > 0:
            noise = torch.randn_like(weights) * self.noise_scale
            weights = weights + noise
        if weights.ndim == 2:
            weights = weights.unsqueeze(0)
        self._current_weights = weights

    def orthogonal_penalty(self) -> torch.Tensor:
        all_vectors = torch.cat(
            [self.steer_local, self.steer_mid, self.steer_global], dim=0)
        norm_vectors = F.normalize(all_vectors, p=2, dim=-1)
        correlation = torch.matmul(norm_vectors, norm_vectors.T)
        identity = torch.eye(21, device=correlation.device)
        return torch.mean((correlation - identity) ** 2)

    def _steer_layer(self, h: torch.Tensor, layer_idx: int) -> torch.Tensor:
        if self._current_weights is None:
            return h

        group = self.layer_routing[layer_idx]
        w = self._current_weights.to(h.device)

        if w.shape[0] == 1 and h.shape[0] > 1:
            w = w.expand(h.shape[0], -1, -1)

        if w.shape[1] != h.shape[1]:
            if w.shape[1] < h.shape[1]:
                pad = torch.zeros(w.shape[0], h.shape[1] - w.shape[1], w.shape[2], device=w.device)
                w = torch.cat([w, pad], dim=1)
            else:
                w = w[:, :h.shape[1], :]

        temp = 2.0
        if group == 'local':
            gated_w = self.local_mlp(w[:, :, 0:6])
            w_soft = torch.softmax(gated_w / temp, dim=-1)
            offset = torch.einsum('btc, cd -> btd', w_soft, self.steer_local)
        elif group == 'mid':
            gated_w = self.mid_mlp(w[:, :, 6:13])
            w_soft = torch.softmax(gated_w / temp, dim=-1)
            offset = torch.einsum('btc, cd -> btd', w_soft, self.steer_mid)
        else:  # global: channels 13:21
            gated_w = self.global_mlp(w[:, :, 13:21])
            w_soft = torch.softmax(gated_w / temp, dim=-1)
            offset = torch.einsum('btc, cd -> btd', w_soft, self.steer_global)

        h_rms = h.pow(2).mean(dim=-1, keepdim=True).sqrt()
        o_rms = offset.pow(2).mean(dim=-1, keepdim=True).sqrt().clamp(min=1e-8)
        normalized_offset = offset * (h_rms / o_rms)

        gamma = self.gammas[str(layer_idx)].abs()
        return h + (gamma * normalized_offset)

    def register_hooks(self, model) -> int:
        self.remove_hooks()
        layers = model.encoder.layers if hasattr(model, 'encoder') else model.layers

        def make_hook(l_idx):
            def hook_fn(module, input, output):
                return self._steer_layer(output, l_idx)
            return hook_fn

        for idx in self.inject_layers:
            if idx < len(layers):
                hook = layers[idx].register_forward_hook(make_hook(idx))
                self._hooks.append(hook)
        return len(self._hooks)

    def remove_hooks(self):
        for hook in self._hooks:
            hook.remove()
        self._hooks = []


class FeatureConditionedAdapterSteerer(nn.Module):
    """Higher-capacity residual adapter for task cartridges.

    Unlike the compact 21-vector superposition steerer, this adapter conditions
    on both the current hidden state and the 21 compiled channel features. It
    still exposes the same `_steer_layer` ABI, so it composes through
    `SteererCartridgeRack` as an additive hot-swappable cartridge.
    """

    def __init__(self, d_model: int = 768, inject_layers: list[int] | None = None,
                 bottleneck: int = 64, init_scale: float = 0.01, noise_scale: float = 0.02):
        super().__init__()
        self.num_channels = 21
        self.d_model = d_model
        self.bottleneck = bottleneck
        self.noise_scale = noise_scale
        self.inject_layers = inject_layers or [0, 1, 2, 4, 5, 6, 8, 9, 10]

        self.norms = nn.ModuleDict({str(layer): nn.LayerNorm(d_model) for layer in self.inject_layers})
        self.down = nn.ModuleDict({str(layer): nn.Linear(d_model, bottleneck) for layer in self.inject_layers})
        self.feature = nn.ModuleDict({str(layer): nn.Linear(self.num_channels, bottleneck) for layer in self.inject_layers})
        self.up = nn.ModuleDict({str(layer): nn.Linear(bottleneck, d_model) for layer in self.inject_layers})
        self.gammas = nn.ParameterDict({
            str(layer): nn.Parameter(torch.tensor(0.05)) for layer in self.inject_layers
        })
        self._current_weights: torch.Tensor | None = None
        self._hooks = []

        for layer in self.inject_layers:
            nn.init.normal_(self.down[str(layer)].weight, mean=0.0, std=init_scale / (d_model ** 0.5))
            nn.init.zeros_(self.down[str(layer)].bias)
            nn.init.normal_(self.feature[str(layer)].weight, mean=0.0, std=init_scale)
            nn.init.zeros_(self.feature[str(layer)].bias)
            nn.init.normal_(self.up[str(layer)].weight, mean=0.0, std=init_scale / (bottleneck ** 0.5))
            nn.init.zeros_(self.up[str(layer)].bias)

    def set_weights(self, weights: torch.Tensor):
        if self.training and self.noise_scale > 0:
            weights = weights + torch.randn_like(weights) * self.noise_scale
        if weights.ndim == 2:
            weights = weights.unsqueeze(0)
        self._current_weights = weights

    def orthogonal_penalty(self) -> torch.Tensor:
        penalties = []
        for layer in self.inject_layers:
            weight = self.up[str(layer)].weight
            norm_weight = F.normalize(weight, p=2, dim=0)
            correlation = torch.matmul(norm_weight.T, norm_weight)
            identity = torch.eye(correlation.shape[0], device=correlation.device)
            penalties.append(torch.mean((correlation - identity) ** 2))
        return torch.stack(penalties).mean()

    def _aligned_weights(self, h: torch.Tensor) -> torch.Tensor:
        if self._current_weights is None:
            return torch.zeros(h.shape[0], h.shape[1], self.num_channels, device=h.device, dtype=h.dtype)
        weights = self._current_weights.to(device=h.device, dtype=h.dtype)
        if weights.shape[0] == 1 and h.shape[0] > 1:
            weights = weights.expand(h.shape[0], -1, -1)
        if weights.shape[1] != h.shape[1]:
            if weights.shape[1] < h.shape[1]:
                pad = torch.zeros(weights.shape[0], h.shape[1] - weights.shape[1], weights.shape[2],
                                  device=weights.device, dtype=weights.dtype)
                weights = torch.cat([weights, pad], dim=1)
            else:
                weights = weights[:, :h.shape[1], :]
        return weights[:, :, :self.num_channels]

    def _steer_layer(self, h: torch.Tensor, layer_idx: int) -> torch.Tensor:
        key = str(layer_idx)
        if key not in self.up:
            return h
        weights = self._aligned_weights(h)
        hidden_part = self.down[key](self.norms[key](h))
        feature_part = self.feature[key](weights)
        delta = self.up[key](F.gelu(hidden_part + feature_part))
        return h + self.gammas[key].abs() * delta

    def register_hooks(self, model) -> int:
        self.remove_hooks()
        layers = model.encoder.layers if hasattr(model, 'encoder') else model.layers

        def make_hook(l_idx):
            def hook_fn(module, input, output):
                return self._steer_layer(output, l_idx)
            return hook_fn

        for idx in self.inject_layers:
            if idx < len(layers):
                hook = layers[idx].register_forward_hook(make_hook(idx))
                self._hooks.append(hook)
        return len(self._hooks)

    def remove_hooks(self):
        for hook in self._hooks:
            hook.remove()
        self._hooks = []
