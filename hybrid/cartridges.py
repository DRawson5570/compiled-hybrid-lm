"""Cartridge manifests and residual-stream composition helpers.

This module keeps the deployable cartridge unit separate from the base model.
Multiple compatible steerers can be mounted beside each other and composed as
weighted residual-stream deltas through one hook rack.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import torch
import torch.nn as nn


class CartridgeRole(str, Enum):
    """Deployment roles for hot-swappable cartridge packages."""

    SUPERPOSITION_STEERER = 'superposition_steerer'
    DOMAIN_CAPABILITY = 'domain_capability'
    TASK_CAPABILITY = 'task_capability'
    CONCEPT_INJECTION = 'concept_injection'


@dataclass(frozen=True)
class CartridgeManifest:
    """Compatibility metadata for a hot-swappable cartridge."""

    cartridge_id: str
    role: CartridgeRole | str
    base_model_id: str
    tokenizer_id: str
    channel_schema: str = 'cmi-21ch-v3'
    steerer_class: str = 'SuperpositionSteererV3'
    inject_layers: tuple[int, ...] = (0, 1, 2, 4, 5, 6, 8, 9, 10)
    composition_space: str = 'residual_stream:additive:v1'
    parameter_count: int | None = None
    source_corpus: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def compatible_with(self, other: 'CartridgeManifest') -> bool:
        """Return True when two cartridges can be additively composed."""
        return (
            self.base_model_id == other.base_model_id
            and self.tokenizer_id == other.tokenizer_id
            and self.channel_schema == other.channel_schema
            and self.composition_space == other.composition_space
        )


@dataclass
class MountedCartridge:
    """A steerer instance plus deployment metadata and mix weight."""

    manifest: CartridgeManifest
    steerer: nn.Module
    weight: float = 1.0
    active: bool = True


class SteererCartridgeRack:
    """Mount and compose multiple superposition steering cartridges.

    The rack registers one hook per target layer. For each active mounted
    steerer, it computes that steer's layer delta against the original hidden
    state and adds the weighted sum. This preserves separate cartridges while
    avoiding hook-order coupling.
    """

    def __init__(self):
        self._mounted: OrderedDict[str, MountedCartridge] = OrderedDict()
        self._hooks: list[Any] = []

    def mount(self, manifest: CartridgeManifest, steerer: nn.Module,
              weight: float = 1.0, active: bool = True) -> str:
        if self._mounted:
            first = next(iter(self._mounted.values())).manifest
            if not first.compatible_with(manifest):
                raise ValueError(
                    f'Cartridge {manifest.cartridge_id!r} is incompatible with '
                    f'{first.cartridge_id!r}'
                )
        if not hasattr(steerer, '_steer_layer'):
            raise TypeError('mounted steerers must expose _steer_layer(h, layer_idx)')

        self._mounted[manifest.cartridge_id] = MountedCartridge(
            manifest=manifest,
            steerer=steerer,
            weight=float(weight),
            active=active,
        )
        return manifest.cartridge_id

    def unmount(self, cartridge_id: str) -> MountedCartridge:
        return self._mounted.pop(cartridge_id)

    def activate(self, cartridge_id: str, active: bool = True):
        self._mounted[cartridge_id].active = active

    def set_weight(self, cartridge_id: str, weight: float):
        self._mounted[cartridge_id].weight = float(weight)

    def set_weights(self, weights: torch.Tensor, cartridge_id: str | None = None):
        """Set compiled channel features on one cartridge or all mounted steerers."""
        targets = (
            [self._mounted[cartridge_id]]
            if cartridge_id is not None else self._mounted.values()
        )
        for mounted in targets:
            if hasattr(mounted.steerer, 'set_weights'):
                mounted.steerer.set_weights(weights)

    def list_active(self) -> list[str]:
        return [cid for cid, mounted in self._mounted.items() if mounted.active]

    def manifests(self) -> list[CartridgeManifest]:
        return [mounted.manifest for mounted in self._mounted.values()]

    def remove_hooks(self):
        for hook in self._hooks:
            hook.remove()
        self._hooks = []

    def register_hooks(self, model: nn.Module) -> int:
        """Register additive composition hooks on the model's transformer layers."""
        self.remove_hooks()
        layers = model.encoder.layers if hasattr(model, 'encoder') else model.layers
        target_layers = sorted({
            layer
            for mounted in self._mounted.values()
            for layer in mounted.manifest.inject_layers
        })

        def make_hook(layer_idx: int):
            def hook_fn(module, inputs, output):
                hidden = output[0] if isinstance(output, tuple) else output
                if not torch.is_tensor(hidden):
                    return output
                total_delta = torch.zeros_like(hidden)
                for mounted in self._mounted.values():
                    if not mounted.active or layer_idx not in mounted.manifest.inject_layers:
                        continue
                    steered = mounted.steerer._steer_layer(hidden, layer_idx)
                    total_delta = total_delta + mounted.weight * (steered - hidden)
                result = (hidden + total_delta).to(dtype=hidden.dtype)
                if isinstance(output, tuple):
                    return (result,) + output[1:]
                return result
            return hook_fn

        for layer_idx in target_layers:
            if layer_idx < len(layers):
                self._hooks.append(layers[layer_idx].register_forward_hook(make_hook(layer_idx)))
        return len(self._hooks)