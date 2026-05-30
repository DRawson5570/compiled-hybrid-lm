"""Tests for hybrid.v1_blender.transformer_blender."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from hybrid.v1_blender.transformer_blender import (
    TBConfig,
    TransformerBlender,
)


def test_shapes_and_logsoftmax():
    cfg = TBConfig(in_dim=24, n_channels=5, d_model=32, n_heads=4, d_ff=64,
                   n_layers=2, ctx=16, dropout=0.0)
    model = TransformerBlender(cfg).eval()
    x = torch.randn(3, 16, 24)
    log_w = model(x)
    assert log_w.shape == (3, 16, 5)
    # log-softmax along channel dim => exp-sum to 1 per position.
    sums = log_w.exp().sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


def test_uniform_at_init():
    cfg = TBConfig(in_dim=8, n_channels=4, d_model=16, n_heads=2, d_ff=32,
                   n_layers=1, ctx=8, dropout=0.0)
    model = TransformerBlender(cfg).eval()
    x = torch.randn(2, 8, 8)
    log_w = model(x)
    expected = math.log(1.0 / 4)
    assert torch.allclose(log_w, torch.full_like(log_w, expected), atol=1e-5)


def test_causal_no_future_leak():
    """Changing a future-position feature must not change earlier outputs."""
    cfg = TBConfig(in_dim=6, n_channels=3, d_model=16, n_heads=2, d_ff=32,
                   n_layers=2, ctx=8, dropout=0.0)
    model = TransformerBlender(cfg).eval()
    # Break uniform-at-init: scramble the head so causality is observable.
    with torch.no_grad():
        model.head.weight.copy_(torch.randn_like(model.head.weight) * 0.5)
        model.head.bias.copy_(torch.randn_like(model.head.bias) * 0.5)
    x = torch.randn(1, 8, 6)
    out_a = model(x)
    # Perturb only the last position.
    x2 = x.clone()
    x2[0, -1] = torch.randn_like(x2[0, -1])
    out_b = model(x2)
    # All but the last position must be bit-exact (eager attention, eval).
    diff = (out_a[0, :-1] - out_b[0, :-1]).abs().max().item()
    assert diff < 1e-6, diff
    # The last position should differ.
    last_diff = (out_a[0, -1] - out_b[0, -1]).abs().max().item()
    assert last_diff > 1e-6, last_diff


def test_can_overfit_tiny():
    """One window of features should be learnable: the model should learn to
    pick the channel that has the highest log_p_target at each position."""
    torch.manual_seed(0)
    cfg = TBConfig(in_dim=12, n_channels=4, d_model=32, n_heads=4, d_ff=64,
                   n_layers=2, ctx=16, dropout=0.0)
    model = TransformerBlender(cfg)
    x = torch.randn(1, 16, 12)
    # log_p_targets shape (1, T, C) — channel `t % C` is the perfect oracle
    # at position t (log p = 0); other channels have log p = log(1/8).
    T, C = 16, 4
    log_p = torch.full((1, T, C), math.log(1.0 / 8))
    for t in range(T):
        log_p[0, t, t % C] = 0.0
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    from hybrid.v1_blender.blender_model import mixture_nll
    final = None
    for _ in range(400):
        log_w = model(x)
        loss = mixture_nll(log_w.reshape(-1, C), log_p.reshape(-1, C)).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        final = loss.item()
    # Perfect mixture would put weight 1 on the oracle channel -> loss = 0.
    assert final < 0.05, final


if __name__ == "__main__":
    test_shapes_and_logsoftmax(); print("shapes ok")
    test_uniform_at_init(); print("uniform ok")
    test_causal_no_future_leak(); print("no-leak ok")
    test_can_overfit_tiny(); print("overfit ok")
    print("all pass")
