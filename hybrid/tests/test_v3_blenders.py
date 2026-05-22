"""Unit and regression tests for sequence-aware v3 blenders.

Verifies shapes, initialization, state propagation, and causality constraints.
"""
from __future__ import annotations

import math
from pathlib import Path
import sys

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from hybrid.v1_blender.blender_model import mixture_nll
from hybrid.v3_super_blender.model import (
    GRUBlender, WindowMLPBlender, LookbackMLPBlender, CausalConvBlender
)


def test_gru_blender_shapes_and_state():
    # In dim 24, 7 channels, hidden 16, 2 layers
    model = GRUBlender(in_dim=24, n_channels=7, hidden=16, num_layers=2)
    model.eval()

    # Flat sequence: shape (SeqLen, in_dim)
    x_flat = torch.randn(50, 24)
    with torch.no_grad():
        log_w_flat, h_n_flat = model(x_flat)
    
    assert log_w_flat.shape == (50, 7)
    assert h_n_flat.shape == (2, 1, 16)
    assert torch.allclose(log_w_flat.exp().sum(dim=-1), torch.ones(50), atol=1e-5)

    # Batched sequence: shape (B, SeqLen, in_dim)
    x_batch = torch.randn(4, 50, 24)
    h0 = torch.zeros(2, 4, 16)
    with torch.no_grad():
        log_w_batch, h_n_batch = model(x_batch, h0)
    
    assert log_w_batch.shape == (4, 50, 7)
    assert h_n_batch.shape == (2, 4, 16)
    assert torch.allclose(log_w_batch.exp().sum(dim=-1), torch.ones(4, 50), atol=1e-5)


def test_window_mlp_blender():
    model = WindowMLPBlender(single_step_dim=6, n_channels=4, lookback_window=10, hidden=32)
    model.eval()

    # Input: shape (T, single_step_dim)
    x = torch.randn(30, 6)
    with torch.no_grad():
        log_w = model(x)
    
    assert log_w.shape == (30, 4)
    # Check that predictions sum to 1 in probability space
    assert torch.allclose(log_w.exp().sum(dim=-1), torch.ones(30), atol=1e-5)


def test_lookback_mlp_blender_resnet():
    model = LookbackMLPBlender(single_step_dim=6, n_channels=4, lookback_window=10, hidden=32, num_layers=2)
    model.eval()

    # Input: shape (T, single_step_dim)
    x = torch.randn(30, 6)
    with torch.no_grad():
        log_w = model(x)
    
    assert log_w.shape == (30, 4)
    assert torch.allclose(log_w.exp().sum(dim=-1), torch.ones(30), atol=1e-5)


def test_causal_conv_blender():
    model = CausalConvBlender(in_dim=12, n_channels=5, channels=16, kernel_size=3, num_layers=3)
    model.eval()

    # Sequential flat input: (T, in_dim)
    x_flat = torch.randn(40, 12)
    with torch.no_grad():
        log_w_flat = model(x_flat)
        
    assert log_w_flat.shape == (40, 5)
    assert torch.allclose(log_w_flat.exp().sum(dim=-1), torch.ones(40), atol=1e-5)

    # Batched input: (B, T, in_dim)
    x_batch = torch.randn(3, 40, 12)
    with torch.no_grad():
        log_w_batch = model(x_batch)
        
    assert log_w_batch.shape == (3, 40, 5)
    assert torch.allclose(log_w_batch.exp().sum(dim=-1), torch.ones(3, 40), atol=1e-5)


def test_zero_initialization_uniformity():
    # Verify that init_uniform=True works for all models and forces near-uniform output initially
    models = [
        GRUBlender(in_dim=10, n_channels=8, init_uniform=True),
        WindowMLPBlender(single_step_dim=5, n_channels=8, lookback_window=4, init_uniform=True),
        LookbackMLPBlender(single_step_dim=5, n_channels=8, lookback_window=4, init_uniform=True),
        CausalConvBlender(in_dim=10, n_channels=8, init_uniform=True)
    ]

    for model in models:
        model.eval()
        if isinstance(model, (WindowMLPBlender, LookbackMLPBlender)):
            x = torch.randn(10, 5)
            log_w = model(x)
        elif isinstance(model, GRUBlender):
            x = torch.randn(10, 10)
            log_w, _ = model(x)
        else:
            x = torch.randn(10, 10)
            log_w = model(x)

        # Uniform logprob for 8 channels is -log(8) ≈ -2.0794
        expected = torch.full_like(log_w, -math.log(8))
        assert torch.allclose(log_w, expected, atol=1e-5)


def test_causality_regression():
    # Verify strict causality: changes in steps t >= 15 do not affect outputs at t < 15
    T = 40
    C = 6
    in_dim = 8
    
    models = [
        ("lookback_mlp", LookbackMLPBlender(in_dim, C, lookback_window=8)),
        ("window_mlp", WindowMLPBlender(in_dim, C, lookback_window=8)),
        ("gru", GRUBlender(in_dim, C)),
        ("causal_conv", CausalConvBlender(in_dim, C))
    ]

    x_full = torch.randn(T, in_dim)
    t_cutoff = 20
    x_perturbed = x_full.clone()
    x_perturbed[t_cutoff:] = torch.randn_like(x_perturbed[t_cutoff:])

    for name, model in models:
        model.eval()
        with torch.no_grad():
            if name in ["lookback_mlp", "window_mlp"]:
                out_full = model(x_full)
                out_perturbed = model(x_perturbed)
            elif name == "gru":
                out_full, _ = model(x_full)
                out_perturbed, _ = model(x_perturbed)
            elif name == "causal_conv":
                out_full = model(x_full)
                out_perturbed = model(x_perturbed)

        # Check maximum discrepancy prior to t_cutoff
        diff = torch.abs(out_full[:t_cutoff] - out_perturbed[:t_cutoff]).max().item()
        assert diff < 1e-5, f"Causality leak in {name}! Max diff < {t_cutoff}: {diff:.3e}"
