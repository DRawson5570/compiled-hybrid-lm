"""test_hybrid_surfaces.py — Acceptance tests for inject/retract/compose API.

Part of TICKET-004. Verifies:
  - Install component A, eval, retract, verify exact restore
  - Install A then B, eval, retract A, verify B still active
  - Install A then B then retract B, verify final state == A-only state
"""
from __future__ import annotations

import sys
from pathlib import Path

DEEPSEEK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DEEPSEEK.parent))  # ~/deepseek_experiments

import numpy as np
import torch
import torch.nn as nn

DEEPSEEK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DEEPSEEK))

from hybrid.surfaces.inject import inject_logit_bias
from hybrid.surfaces.retract import retract
from hybrid.surfaces.compose import compose
from hybrid.surfaces.registry import ComponentRegistry


class DummyModel(nn.Module):
    """Simple model for testing — returns identity-like logits."""
    def __init__(self, vocab: int = 100, d_model: int = 16):
        super().__init__()
        self.vocab = vocab
        self.embed = nn.Embedding(vocab, d_model)
        self.head = nn.Linear(d_model, vocab)

    def forward(self, x):
        return self.head(self.embed(x))


def test_inject_retract_restore():
    """Install component A, eval, retract, verify exact restore."""
    model = DummyModel(vocab=100)
    original_state = {k: v.clone() for k, v in model.state_dict().items()}

    registry = ComponentRegistry()
    bias = torch.zeros(100)
    bias[42] = 5.0  # boost token 42

    wrapped, cid = inject_logit_bias(model, bias, registry=registry)
    assert cid is not None
    assert len(registry.list_active()) == 1

    # Model is wrapped — verify bias is applied
    x = torch.randint(0, 100, (1, 10))
    logits_wrapped = wrapped(x)
    logits_original = model(x)
    assert (logits_wrapped[..., 42] - logits_original[..., 42]).abs().max() > 1.0, \
        'Bias should change token 42 logits'

    # Retract
    unwrapped, verified = retract(wrapped, cid, registry)
    assert verified
    assert len(registry.list_active()) == 0

    # Verify parameter restoration
    for key in original_state:
        assert torch.allclose(original_state[key], unwrapped.state_dict()[key], atol=1e-6), \
            f'Parameter {key} not restored'


def test_compose_a_then_b_retract_a():
    """Install A then B, eval, retract A, verify B still active."""
    model = DummyModel(vocab=100)
    registry = ComponentRegistry()

    bias_a = torch.zeros(100)
    bias_a[10] = 3.0
    bias_b = torch.zeros(100)
    bias_b[20] = 7.0

    wrapped, ids = compose(model, [
        {'type': 'logit_bias', 'bias': bias_a, 'token_ids': list(range(100))},
        {'type': 'logit_bias', 'bias': bias_b, 'token_ids': list(range(100))},
    ], registry)

    assert len(ids) == 2
    assert len(registry.list_active()) == 2

    # Retract first component (bias_a)
    unwrapped, verified = retract(wrapped, ids[0], registry)
    assert len(registry.list_active()) == 1  # B still active

    # Verify B's effect remains
    x = torch.randint(0, 100, (1, 10))
    logits_after_retract_a = unwrapped(x)
    logits_original = model(x)
    assert (logits_after_retract_a[..., 20] - logits_original[..., 20]).abs().max() > 1.0, \
        'Bias B should still be active after retracting A'


def test_compose_a_b_retract_b():
    """Install A then B, retract B, verify final state == A-only state."""
    model = DummyModel(vocab=100)
    registry = ComponentRegistry()

    bias_a = torch.zeros(100)
    bias_a[10] = 3.0
    bias_b = torch.zeros(100)
    bias_b[20] = 7.0

    wrapped, ids = compose(model, [
        {'type': 'logit_bias', 'bias': bias_a},
        {'type': 'logit_bias', 'bias': bias_b},
    ], registry)

    # Retract B
    unwrapped, _ = retract(wrapped, ids[1], registry)
    assert len(registry.list_active()) == 1

    # Verify only A's effect remains
    x = torch.randint(0, 100, (1, 10))
    logits_final = unwrapped(x)
    logits_original = model(x)

    # Token 10 should be boosted (A), token 20 should NOT be boosted (B retracted)
    assert (logits_final[..., 10] - logits_original[..., 10]).abs().max() > 1.0
    assert (logits_final[..., 20] - logits_original[..., 20]).abs().max() < 1e-3


def test_registry_snapshot():
    """Registry snapshot contains all installed components."""
    registry = ComponentRegistry()
    model = DummyModel(vocab=100)
    bias = torch.ones(100)

    _, cid = inject_logit_bias(model, bias, registry=registry)
    snap = registry.snapshot()
    assert cid in snap['components']
    assert snap['active_count'] == 1


def test_registry_compute_checksum():
    """Model checksum is deterministic and changes on modification."""
    model = DummyModel(vocab=100)
    cs1 = ComponentRegistry.compute_model_checksum(model)
    cs2 = ComponentRegistry.compute_model_checksum(model)
    assert cs1 == cs2, 'Checksum should be deterministic'

    # Modify a parameter
    model.head.weight.data[0, 0] += 1.0
    cs3 = ComponentRegistry.compute_model_checksum(model)
    assert cs1 != cs3, 'Checksum should change on modification'


if __name__ == '__main__':
    test_inject_retract_restore()
    print('PASS: test_inject_retract_restore')
    test_compose_a_then_b_retract_a()
    print('PASS: test_compose_a_then_b_retract_a')
    test_compose_a_b_retract_b()
    print('PASS: test_compose_a_b_retract_b')
    test_registry_snapshot()
    print('PASS: test_registry_snapshot')
    test_registry_compute_checksum()
    print('PASS: test_registry_compute_checksum')
    print('\nAll hybrid surfaces tests passed.')
