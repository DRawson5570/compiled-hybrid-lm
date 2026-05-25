from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

DEEPSEEK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DEEPSEEK.parent))

from hybrid.cartridges import CartridgeManifest, CartridgeRole, SteererCartridgeRack


class AddOneLayer(nn.Module):
    def forward(self, x):
        return x


class DummyTransformer(nn.Module):
    def __init__(self, n_layers: int = 2):
        super().__init__()
        self.layers = nn.ModuleList(AddOneLayer() for _ in range(n_layers))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class TupleLayer(nn.Module):
    def forward(self, x):
        return x, 'cache'


class TupleTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([TupleLayer()])

    def forward(self, x):
        hidden, cache = self.layers[0](x)
        return hidden, cache


class FakeSteerer(nn.Module):
    def __init__(self, delta: float):
        super().__init__()
        self.delta = delta
        self.seen_weights = None

    def set_weights(self, weights: torch.Tensor):
        self.seen_weights = weights

    def _steer_layer(self, h: torch.Tensor, layer_idx: int) -> torch.Tensor:
        return h + self.delta * (layer_idx + 1)


def manifest(cartridge_id: str, role: CartridgeRole | str,
             base_model_id: str = 'c4-124m') -> CartridgeManifest:
    return CartridgeManifest(
        cartridge_id=cartridge_id,
        role=role,
        base_model_id=base_model_id,
        tokenizer_id='gpt2-bpe',
        inject_layers=(0,),
    )


def test_manifest_compatibility_checks_composition_surface():
    steerer = manifest('v4-steerer', CartridgeRole.SUPERPOSITION_STEERER)
    code = manifest('code', CartridgeRole.DOMAIN_CAPABILITY)
    other_base = manifest('medical', CartridgeRole.DOMAIN_CAPABILITY, base_model_id='other')

    assert steerer.compatible_with(code)
    assert not steerer.compatible_with(other_base)


def test_rack_composes_independent_steerer_and_capability_cartridge():
    model = DummyTransformer(n_layers=1)
    rack = SteererCartridgeRack()
    rack.mount(manifest('v4-steerer', CartridgeRole.SUPERPOSITION_STEERER), FakeSteerer(1.0), weight=0.5)
    rack.mount(manifest('code', CartridgeRole.DOMAIN_CAPABILITY), FakeSteerer(3.0), weight=2.0)

    assert rack.register_hooks(model) == 1
    x = torch.zeros(2, 4, 8)
    y = model(x)

    assert torch.allclose(y, torch.full_like(y, 6.5))

    rack.remove_hooks()
    assert torch.allclose(model(x), x)


def test_rack_rejects_incompatible_cartridge():
    rack = SteererCartridgeRack()
    rack.mount(manifest('v4-steerer', CartridgeRole.SUPERPOSITION_STEERER), FakeSteerer(1.0))

    with pytest.raises(ValueError):
        rack.mount(manifest('bad-base', CartridgeRole.DOMAIN_CAPABILITY, base_model_id='wrong'), FakeSteerer(1.0))


def test_rack_sets_channel_weights_for_all_or_one_cartridge():
    rack = SteererCartridgeRack()
    steer_a = FakeSteerer(1.0)
    steer_b = FakeSteerer(2.0)
    rack.mount(manifest('a', CartridgeRole.SUPERPOSITION_STEERER), steer_a)
    rack.mount(manifest('b', CartridgeRole.DOMAIN_CAPABILITY), steer_b)

    first = torch.ones(1, 2, 21)
    second = torch.zeros(1, 2, 21)
    rack.set_weights(first)
    rack.set_weights(second, cartridge_id='b')

    assert steer_a.seen_weights is first
    assert steer_b.seen_weights is second


def test_rack_preserves_tuple_outputs_and_hidden_dtype():
    model = TupleTransformer()
    rack = SteererCartridgeRack()
    rack.mount(manifest('qwen-task', CartridgeRole.TASK_CAPABILITY), FakeSteerer(1.0), weight=0.25)

    assert rack.register_hooks(model) == 1
    x = torch.zeros(1, 2, 4, dtype=torch.float16)
    hidden, cache = model(x)

    assert cache == 'cache'
    assert hidden.dtype == torch.float16
    assert torch.allclose(hidden, torch.full_like(hidden, 0.25))