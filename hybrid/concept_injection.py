"""concept_injection.py — Vocab-space projection steering (Gemini V4 upgrade).

Instead of scalar weights × static steer vectors, projects each channel's
full V-dimensional probability distribution into the residual stream via
the model's own token embedding matrix.

    o_{c,t} = softmax(p_{c,t} / T) · W_E
    offset_t = Σ γ_c · o_{c,t}

This is literal concept injection — "France" activations are injected
when the trigram prior predicts "France", not just a "Wiki-style" signal.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConceptInjectionSteerer(nn.Module):
    """Steerer that projects channel distributions through token embeddings.

    Each channel produces a V-dimensional log-prob distribution. Instead of
    collapsing to a scalar weight, we softmax and multiply by the model's
    token embedding matrix to get a d_model-dimensional semantic offset.
    """

    def __init__(self, num_channels: int, d_model: int, vocab_size: int,
                 tok_emb: torch.Tensor, inject_layers=None,
                 temperature: float = 0.5):
        super().__init__()
        self.num_channels = num_channels
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.inject_layers = inject_layers or [0, 4, 8]

        # Register token embedding as fixed lookup (not trained)
        self.register_buffer('tok_emb', tok_emb)  # (V, d)

        # Per-channel learned scalar gates (which channels to trust)
        self.channel_gates = nn.Parameter(torch.ones(num_channels) * 0.01)

        # Learned temperature
        self.temperature = nn.Parameter(torch.tensor(temperature))

        # Overall injection strength
        self.gamma = nn.Parameter(torch.tensor(0.01))

        self._current_distributions: torch.Tensor | None = None  # (B, T, C, V)
        self._hooks = []

    def set_channel_distributions(self, channel_logprobs: torch.Tensor):
        """Set full V-dimensional channel log-probability distributions.

        Args:
            channel_logprobs: (B, T, C, V) — per-batch, per-position,
                              per-channel, per-vocab log-probabilities.
        """
        # Temperature-scaled softmax over vocabulary
        scaled = channel_logprobs / self.temperature.abs()
        self._current_distributions = torch.softmax(scaled, dim=-1)  # (B,T,C,V)

    def compute_offset(self, batch_size, seq_len, device):
        """Compute concept-injected offset: Σ γ_c · softmax(p_c/T) · W_E.

        Returns:
            offset: (B, T, d_model) activation offset
        """
        if self._current_distributions is None:
            return torch.zeros(batch_size, seq_len, self.d_model, device=device)

        dist = self._current_distributions.to(device)  # (B, T, C, V)
        B, T, C, V = dist.shape

        # Project each channel's distribution through embedding
        # dist: (B, T, C, V), tok_emb: (V, d)
        # → (B, T, C, d)
        channel_offsets = torch.einsum('btcv,vd->btcd', dist, self.tok_emb)

        # Weight by learned channel gates
        gates = self.channel_gates.abs()  # (C,)
        gated = channel_offsets * gates.view(1, 1, C, 1)  # (B, T, C, d)

        # Sum over channels
        offset = gated.sum(dim=2)  # (B, T, d)

        return offset

    def _hook_fn(self, module, input, output):
        if self._current_distributions is None:
            return output

        B, T, d = output.shape
        offset = self.compute_offset(B, T, output.device)

        # RMS normalization
        h_norm = output.norm(p=2, dim=-1, keepdim=True)
        o_norm = offset.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-8)
        offset = offset * (h_norm / o_norm)

        return output + self.gamma.abs() * offset

    def register_hooks(self, model) -> int:
        self.remove_hooks()
        layers = model.encoder.layers if hasattr(model, 'encoder') else model.layers
        for idx in self.inject_layers:
            if idx < len(layers):
                hook = layers[idx].register_forward_hook(self._hook_fn)
                self._hooks.append(hook)
        return len(self._hooks)

    def remove_hooks(self):
        for hook in self._hooks:
            hook.remove()
        self._hooks = []
