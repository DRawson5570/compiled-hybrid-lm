"""surfaces/compose.py — Apply multiple components and report interaction order.

Part of TICKET-004. Stacks multiple inject() calls and tracks the composition
order in the ComponentRegistry.
"""
from __future__ import annotations

import torch.nn as nn

from hybrid.surfaces.registry import ComponentRegistry
from hybrid.surfaces.inject import inject_logit_bias


def compose(model: nn.Module,
            components: list[dict],
            registry: ComponentRegistry | None = None) -> tuple[nn.Module, list[str]]:
    """Apply multiple components sequentially.

    Args:
        model: base model to modify
        components: list of {'type': 'logit_bias', 'bias': ..., 'token_ids': ...}
        registry: optional ComponentRegistry

    Returns:
        (composed_model, list_of_component_ids) in installation order
    """
    if registry is None:
        registry = ComponentRegistry()

    current = model
    installed = []

    for comp in components:
        comp_type = comp.get('type', 'logit_bias')
        if comp_type == 'logit_bias':
            current, cid = inject_logit_bias(
                current,
                bias=comp['bias'],
                token_ids=comp.get('token_ids'),
                component_id=comp.get('component_id'),
                registry=registry,
            )
            installed.append(cid)
        else:
            raise ValueError(f'Unknown component type: {comp_type}')

    return current, installed
