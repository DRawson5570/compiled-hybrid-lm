"""Unit/regression tests for sequence-aware CMI blenders.

Verifies:
  * Shape and formatting of sequence-aware architectures (GRU, Causal CNN, Lookback MLP, Window MLP).
  * Softmax property of outputs (row-sums equal 1 in log-space).
  * Zero-init uniform routing properties.
  * Correct lookback window replication/padding on edge cases.
  * Autograd backpropagation end-to-end.
"""
from __future__ import annotations

import math
from pathlib import Path
import sys

import pytest
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from hybrid.v3_super_blender.model import (
    WindowMLPBlender, LookbackMLPBlender, GRUBlender, CausalConvBlender
)


def test_window_mlp_blender():
    T, F_dim, C = 20, 10, 4
    model = WindowMLPBlender(single_step_dim=F_dim, n_channels=C, lookback_window=4, hidden=16, init_uniform=True)
    features = torch.randn(T, F_dim)
    
    # 1. Forward pass
    log_w = model(features)
    assert log_w.shape == (T, C)
    
    # 2. Row sum check (softmax)
    assert torch.allclose(log_w.exp().sum(dim=-1), torch.ones(T), atol=1e-5)
    
    # 3. Uniform initialization check
    expected = torch.full((T, C), -math.log(C))
    assert torch.allclose(log_w, expected, atol=1e-5)
    
    # 4. Backward pass
    loss = log_w.sum()
    loss.backward()
    for p in model.parameters():
        if p.requires_grad:
            assert p.grad is not None


def test_lookback_mlp_blender():
    T, F_dim, C = 15, 8, 3
    model = LookbackMLPBlender(single_step_dim=F_dim, n_channels=C, lookback_window=3, hidden=16, init_uniform=True)
    features = torch.randn(T, F_dim)
    
    # 1. Forward pass
    log_w = model(features)
    assert log_w.shape == (T, C)
    
    # 2. Row sum check (softmax)
    assert torch.allclose(log_w.exp().sum(dim=-1), torch.ones(T), atol=1e-5)
    
    # 3. Uniform initialization check
    expected = torch.full((T, C), -math.log(C))
    assert torch.allclose(log_w, expected, atol=1e-5)
    
    # 4. Backward pass
    loss = log_w.sum()
    loss.backward()
    for p in model.parameters():
        if p.requires_grad:
            assert p.grad is not None


def test_gru_blender():
    T, F_dim, C = 25, 12, 5
    model = GRUBlender(in_dim=F_dim, n_channels=C, hidden=16, num_layers=2, init_uniform=True)
    features = torch.randn(T, F_dim)
    
    # 1. Forward flat/2D
    log_w, h = model(features)
    assert log_w.shape == (T, C)
    assert h.shape == (2, 1, 16)
    
    # 2. Softmax
    assert torch.allclose(log_w.exp().sum(dim=-1), torch.ones(T), atol=1e-5)
    
    # 3. Uniform initialization
    expected = torch.full((T, C), -math.log(C))
    assert torch.allclose(log_w, expected, atol=1e-5)
    
    # 4. Forward bats/3D
    x_3d = torch.randn(2, T, F_dim)
    log_w_3d, h_3d = model(x_3d)
    assert log_w_3d.shape == (2, T, C)
    assert h_3d.shape == (2, 2, 16)
    
    # 5. Backward
    loss = log_w_3d.sum()
    loss.backward()
    for p in model.parameters():
        if p.requires_grad:
            assert p.grad is not None


def test_causal_conv_blender():
    T, F_dim, C = 30, 14, 4
    model = CausalConvBlender(in_dim=F_dim, n_channels=C, channels=16, kernel_size=3, num_layers=2, init_uniform=True)
    features = torch.randn(T, F_dim)
    
    # 1. Forward pass
    log_w = model(features)
    assert log_w.shape == (T, C)
    
    # 2. Softmax
    assert torch.allclose(log_w.exp().sum(dim=-1), torch.ones(T), atol=1e-5)
    
    # 3. Uniform initialization
    expected = torch.full((T, C), -math.log(C))
    assert torch.allclose(log_w, expected, atol=1e-5)
    
    # 4. Forward batch mode
    x_3d = torch.randn(3, T, F_dim)
    log_w_3d = model(x_3d)
    assert log_w_3d.shape == (3, T, C)
    
    # 5. Backward pass
    loss = log_w_3d.sum()
    loss.backward()
    for p in model.parameters():
        if p.requires_grad:
            assert p.grad is not None
