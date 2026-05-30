"""CUDA probe for side-by-side cartridge rack composition.

Run manually on a CUDA host:
    CUDA_VISIBLE_DEVICES=1 python hybrid/tests/probe_cartridge_cuda.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from hybrid.cartridges import CartridgeManifest, CartridgeRole, SteererCartridgeRack
from hybrid.superposition_steerer_v3 import SuperpositionSteererV3


class IdentityLayer(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class OneLayerModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([IdentityLayer()])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


def build_steerer(seed: int, d_model: int, device: torch.device) -> SuperpositionSteererV3:
    torch.manual_seed(seed)
    steerer = SuperpositionSteererV3(
        d_model=d_model,
        inject_layers=[0],
        init_scale=0.02,
        noise_scale=0.0,
    ).to(device)
    steerer.eval()
    return steerer


def manifest(cartridge_id: str, role: CartridgeRole) -> CartridgeManifest:
    return CartridgeManifest(
        cartridge_id=cartridge_id,
        role=role,
        base_model_id='probe-c4-124m',
        tokenizer_id='gpt2-bpe',
        channel_schema='cmi-21ch-v3',
        inject_layers=(0,),
    )


def run_probe() -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for this probe')

    device = torch.device('cuda')
    torch.manual_seed(20260524)
    d_model = 64
    batch = 2
    seq_len = 8
    alpha = 0.75
    beta = 0.40

    x = torch.randn(batch, seq_len, d_model, device=device)
    weights = torch.randn(batch, seq_len, 21, device=device)

    base_model = OneLayerModel().to(device).eval()
    base = base_model(x)

    superposition = build_steerer(seed=11, d_model=d_model, device=device)
    capability = build_steerer(seed=29, d_model=d_model, device=device)

    superposition.set_weights(weights)
    capability.set_weights(weights)

    rack_superposition = SteererCartridgeRack()
    rack_superposition.mount(
        manifest('probe-superposition', CartridgeRole.SUPERPOSITION_STEERER),
        superposition,
        weight=1.0,
    )
    rack_superposition.register_hooks(base_model)
    superposition_only = base_model(x)
    rack_superposition.remove_hooks()

    rack_capability = SteererCartridgeRack()
    rack_capability.mount(
        manifest('probe-python-capability', CartridgeRole.DOMAIN_CAPABILITY),
        capability,
        weight=1.0,
    )
    rack_capability.register_hooks(base_model)
    capability_only = base_model(x)
    rack_capability.remove_hooks()

    rack_combined = SteererCartridgeRack()
    rack_combined.mount(
        manifest('probe-superposition', CartridgeRole.SUPERPOSITION_STEERER),
        superposition,
        weight=alpha,
    )
    rack_combined.mount(
        manifest('probe-python-capability', CartridgeRole.DOMAIN_CAPABILITY),
        capability,
        weight=beta,
    )
    rack_combined.register_hooks(base_model)
    combined = base_model(x)
    rack_combined.remove_hooks()

    expected = base + alpha * (superposition_only - base) + beta * (capability_only - base)
    max_abs_error = (combined - expected).abs().max().item()
    superposition_delta = (superposition_only - base).norm().item()
    capability_delta = (capability_only - base).norm().item()
    combined_delta = (combined - base).norm().item()

    assert superposition_delta > 0.0, 'superposition cartridge produced no delta'
    assert capability_delta > 0.0, 'capability cartridge produced no delta'
    assert combined_delta > 0.0, 'combined cartridges produced no delta'
    assert max_abs_error < 1e-5, f'combined output mismatch: {max_abs_error}'

    return {
        'device': torch.cuda.get_device_name(0),
        'dtype': str(x.dtype),
        'batch': batch,
        'seq_len': seq_len,
        'd_model': d_model,
        'alpha': alpha,
        'beta': beta,
        'active_combined': ['probe-superposition', 'probe-python-capability'],
        'superposition_delta_norm': superposition_delta,
        'capability_delta_norm': capability_delta,
        'combined_delta_norm': combined_delta,
        'max_abs_error_vs_linear_expected': max_abs_error,
        'passed': True,
    }


if __name__ == '__main__':
    print(json.dumps(run_probe(), indent=2, sort_keys=True))
