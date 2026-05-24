"""dynamic_gating.py — Self-attenuating steering injection (Gemini V4 upgrade).

Instead of a static learned scalar gamma, the transformer's hidden state at
each layer predicts its own gating scalar per position:

    γ_{l,t} = σ(w_l^T · h_{l,t})

This lets the model shut off steering when confident and amplify it when uncertain.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DynamicGatingSteerer(nn.Module):
    """Steerer with per-layer, per-position dynamic gating.

    Each injected layer has its own learned gating vector w_l.
    Before injection, the hidden state h_{l,t} predicts γ_{l,t} via sigmoid.
    """

    def __init__(self, num_channels: int, d_model: int,
                 inject_layers=None, init_scale=0.01):
        super().__init__()
        self.num_channels = num_channels
        self.d_model = d_model
        self.inject_layers = inject_layers or [0, 4, 8]

        # Per-channel steering vectors
        self.steer_vectors = nn.Parameter(
            torch.randn(num_channels, d_model) * init_scale / (d_model ** 0.5))

        # Learned softmax temperature
        self.temperature = nn.Parameter(torch.ones(1) * 2.0)

        # Dynamic gating: per-injected-layer learned projection
        self.layer_gates = nn.ParameterDict({
            str(l): nn.Parameter(torch.zeros(d_model) * init_scale)
            for l in self.inject_layers
        })

        # Global base gamma (multiplied by dynamic gate)
        self.gamma = nn.Parameter(torch.tensor(0.01))

        self._current_weights = None
        self._hooks = []

    def set_weights(self, weights: torch.Tensor):
        if self.training:
            weights = weights + torch.randn_like(weights) * 0.01
        w = weights / self.temperature.abs()
        self._current_weights = torch.softmax(w, dim=-1)

    def _make_hook(self, layer_idx):
        """Create a hook function for a specific layer with dynamic gating."""
        gate_param = self.layer_gates[str(layer_idx)]

        def hook_fn(module, input, output):
            if self._current_weights is None:
                return output

            w = self._current_weights.to(output.device)
            B, T, d = output.shape

            # Expand weights
            if w.ndim == 2:
                w = w.unsqueeze(0).expand(B, -1, -1)
            if w.shape[1] != T:
                if w.shape[1] < T:
                    pad = torch.zeros(B, T - w.shape[1], w.shape[2], device=w.device)
                    w = torch.cat([w, pad], dim=1)
                else:
                    w = w[:, :T, :]

            # Compute steering offset
            offset = torch.einsum('btc,cd->btd', w, self.steer_vectors.to(w.device))

            # RMS normalization
            h_norm = output.norm(p=2, dim=-1, keepdim=True)
            o_norm = offset.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-8)
            offset = offset * (h_norm / o_norm)

            # Dynamic gating: γ_{l,t} = σ(w_l^T · h_{l,t})
            dynamic_gate = torch.sigmoid(
                torch.einsum('btd,d->bt', output, gate_param.to(output.device)))
            dynamic_gate = dynamic_gate.unsqueeze(-1)  # (B, T, 1)

            return output + self.gamma.abs() * dynamic_gate * offset

        return hook_fn

    def register_hooks(self, model) -> int:
        self.remove_hooks()
        layers = model.encoder.layers if hasattr(model, 'encoder') else model.layers

        for idx in self.inject_layers:
            if idx < len(layers):
                hook = layers[idx].register_forward_hook(self._make_hook(idx))
                self._hooks.append(hook)

        return len(self._hooks)

    def remove_hooks(self):
        for hook in self._hooks:
            hook.remove()
        self._hooks = []

    def orthogonal_penalty(self) -> torch.Tensor:
        V = self.steer_vectors
        V_norm = F.normalize(V, p=2, dim=-1)
        gram = V_norm @ V_norm.T
        eye = torch.eye(self.num_channels, device=V.device)
        return ((gram - eye) ** 2).mean()
