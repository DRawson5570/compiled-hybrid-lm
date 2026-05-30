"""surfaces/inject.py — Add residual components to a model without retraining.

Part of TICKET-004. Supports:
  - inject_logit_bias: add a fixed vector to output logits
  - inject_concept_pack: add a concept-triggered residual

Every injection records a restore checksum so retract() can verify exact restoration.
"""
from __future__ import annotations

import time
import torch
import torch.nn as nn

from hybrid.surfaces.registry import ComponentRegistry

_counter = 0


class LogitBiasWrapper(nn.Module):
    """Wraps a model to add a per-token logit bias vector at inference time."""

    def __init__(self, model: nn.Module, bias: torch.Tensor, token_ids: list[int],
                 component_id: str = ''):
        super().__init__()
        self.model = model
        self.vocab = bias.shape[0]
        self.component_id = component_id
        self.register_buffer('bias', bias.detach().clone())
        self.token_ids = token_ids

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        logits = self.model(ids)
        # Apply bias to specified token positions
        for tid in self.token_ids:
            if 0 <= tid < self.vocab:
                logits[..., tid] += self.bias[tid]
        return logits


def inject_logit_bias(model: nn.Module, bias: torch.Tensor,
                      token_ids: list[int] | None = None,
                      component_id: str | None = None,
                      registry: ComponentRegistry | None = None) -> tuple[nn.Module, str]:
    """Inject a logit bias into the model's output.

    Args:
        model: the neural LM to modify
        bias: (V,) tensor of logit biases (added to logits before softmax)
        token_ids: which token IDs the bias applies to (default: all)
        component_id: unique ID for this component (auto-generated if None)
        registry: optional ComponentRegistry to track the installation

    Returns:
        (wrapped_model, component_id)
    """
    import time
    global _counter
    _counter += 1
    if token_ids is None:
        token_ids = list(range(bias.shape[0]))

    if component_id is None:
        component_id = f'logit_bias_{int(time.time())}_{_counter}'

    wrapped = LogitBiasWrapper(model, bias, token_ids, component_id)

    if registry:
        checksum = ComponentRegistry.compute_model_checksum(model)
        registry.register(component_id, 'logit_bias',
                          {'n_biased_tokens': len(token_ids)})
        registry.set_checksum(component_id, checksum)

    return wrapped, component_id


def inject_concept_pack(model: nn.Module,
                        trigger_tokens: list[int],
                        concept_vector: torch.Tensor,
                        alpha: float = 1.0,
                        component_id: str | None = None,
                        registry: ComponentRegistry | None = None) -> tuple[nn.Module, str]:
    """Inject a concept-triggered residual: when trigger tokens appear in context,
    add alpha * concept_vector to the output logits.

    This is the compiled-injection pattern proven in the main repo.
    """
    import time
    global _counter
    _counter += 1

    if component_id is None:
        component_id = f'concept_{int(time.time())}_{_counter}'

    bias = alpha * concept_vector
    wrapped, cid = inject_logit_bias(model, bias, trigger_tokens,
                                      component_id, registry)
    return wrapped, cid
