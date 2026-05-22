"""surfaces/retract.py — Remove an injected component and restore exact model state.

Part of TICKET-004. Unwraps LogitBiasWrapper by component_id and verifies
parameter checksum matches the pre-installation snapshot.
"""
from __future__ import annotations

import torch.nn as nn

from hybrid.surfaces.inject import LogitBiasWrapper
from hybrid.surfaces.registry import ComponentRegistry


def retract(wrapped_model: nn.Module, component_id: str,
            registry: ComponentRegistry | None = None) -> tuple[nn.Module, bool]:
    """Remove an injected component and restore the model without that wrapper.

    Walks the wrapper chain to find the target component_id, removes it,
    and re-links the chain.  Verifies checksum if registry is provided.
    """
    # Build the wrapper chain as a list
    chain = []
    current = wrapped_model
    while isinstance(current, LogitBiasWrapper):
        chain.append(current)
        current = current.model

    if not chain:
        if registry:
            registry.deactivate(component_id)
        return wrapped_model, True

    # Find and remove the target wrapper
    target_idx = None
    for i, w in enumerate(chain):
        if w.component_id == component_id:
            target_idx = i
            break

    if target_idx is None:
        # Component not found — maybe already retracted
        if registry:
            registry.deactivate(component_id)
        return wrapped_model, False

    # Remove the target wrapper from the chain
    removed = chain.pop(target_idx)

    # Rebuild chain: link remaining wrappers to the innermost model
    result = current  # innermost (original) model
    for w in reversed(chain):
        w.model = result
        result = w

    # Verify checksum
    verified = True
    if registry:
        entry = registry.get(component_id)
        if entry and entry.get('restore_checksum'):
            current_checksum = ComponentRegistry.compute_model_checksum(result)
            verified = (current_checksum == entry['restore_checksum'])
        registry.deactivate(component_id)

    return result, verified
