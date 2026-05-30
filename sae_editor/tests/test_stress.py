"""Tier 1: Numerical stress tests at realistic model scales."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from sae_editor.recompiler import (
    build_dense_map,
    compute_null_space_rank,
    orthogonal_projection,
    verify_dense_map,
)
from sae_editor.splicer import SafetensorsSplicer


class TestLargeScaleDenseMap:
    def test_large_N_768(self):
        N, d = 500, 768
        keys = torch.randn(N, d)
        values = torch.randn(N, d)
        W_down, W_up = build_dense_map(keys, values, eps=1e-4)
        recon = verify_dense_map(keys, W_down, W_up)
        max_err = (recon - values).norm(dim=-1).max().item()
        assert max_err < 1e-2, f"Large N reconstruction error too high: {max_err:.6f}"

    def test_large_d_model_4096(self):
        N, d = 10, 4096
        keys = torch.randn(N, d)
        values = torch.randn(N, d)
        W_down, W_up = build_dense_map(keys, values)
        assert W_down.shape == (d, N)
        assert W_up.shape == (N, d)
        recon = verify_dense_map(keys, W_down, W_up)
        assert torch.allclose(recon, values, atol=1e-4)

    @pytest.mark.parametrize("N", [1, 10, 100, 500])
    def test_reconstruction_fidelity_N(self, N):
        d = 768
        keys = torch.randn(N, d)
        values = torch.randn(N, d)
        W_down, W_up = build_dense_map(keys, values, eps=1e-3)
        recon = verify_dense_map(keys, W_down, W_up)
        max_err = (recon - values).norm(dim=-1).max().item()
        assert max_err < 1e-2, f"N={N}, d={d}: max_err={max_err:.6f}"

    def test_ill_conditioned_near_duplicates(self):
        N, d = 50, 256
        keys = torch.randn(N, d)
        keys[1] = keys[0] + 1e-5 * torch.randn(d)
        keys[3] = keys[2] + 1e-5 * torch.randn(d)
        values = torch.randn(N, d)
        W_down, W_up = build_dense_map(keys, values, eps=1e-2)
        assert not torch.isnan(W_down).any()
        assert not torch.isinf(W_down).any()
        recon = verify_dense_map(keys, W_down, W_up)
        assert not torch.isnan(recon).any()

    def test_eps_prevents_nan_with_identical_keys(self):
        N, d = 5, 128
        keys = torch.randn(N - 1, d)
        keys = torch.cat([keys, keys[0:1].clone()], dim=0)
        values = torch.randn(N, d)

        W_down, W_up = build_dense_map(keys, values, eps=1e-3)
        assert not torch.isnan(W_down).any()
        assert not torch.isinf(W_down).any()
        assert not torch.isnan(W_up).any()
        assert not torch.isinf(W_up).any()


class TestLargeScaleCrosstalk:
    def test_many_protected_features(self):
        d = 768
        m = 200
        U = torch.randn(d, m)
        W = torch.randn(d, 32)
        W_proj = orthogonal_projection(W, U, eps=1e-4)
        residual = U.T @ W_proj
        max_leak = residual.abs().max().item()
        assert max_leak < 1e-3, f"Crosstalk leak with {m} features: {max_leak:.6f}"

    def test_null_space_tracks_capacity(self):
        d = 256
        for m in [10, 50, 100, 200]:
            U = torch.randn(d, m)
            rank = compute_null_space_rank(U)
            assert rank <= d - m
            assert rank >= d - m - 10, f"m={m}: rank={rank}, expected ~{d - m}"

    def test_idempotent_large_scale(self):
        d = 512
        m = 128
        U = torch.randn(d, m)
        W = torch.randn(d, 64)
        W1 = orthogonal_projection(W, U, eps=1e-4)
        W2 = orthogonal_projection(W1, U, eps=1e-4)
        diff = (W1 - W2).abs().max().item()
        assert diff < 1e-5, f"Idempotent check failed at d={d}, m={m}: diff={diff:.6f}"


class TestDtypeRoundTrip:
    def test_fp16_keys_float32_output(self):
        keys = torch.randn(3, 128, dtype=torch.float16)
        values = torch.randn(3, 128, dtype=torch.float16)
        W_down, W_up = build_dense_map(keys, values)
        assert W_down.dtype == torch.float32
        assert W_up.dtype == torch.float32

    def test_bf16_round_trip(self):
        keys = torch.randn(3, 64, dtype=torch.bfloat16)
        values = torch.randn(3, 64, dtype=torch.bfloat16)
        W_down, W_up = build_dense_map(keys, values)
        recon = keys.float() @ W_down @ W_up
        assert torch.allclose(recon, values.float(), atol=1e-3)

    def test_splice_f32_preserves_dtype(self, temp_safetensors):
        import numpy as np

        tensor_name = "model.layers.0.mlp.up_proj.weight"
        with SafetensorsSplicer(temp_safetensors) as spl:
            dtype = spl.get_tensor_dtype(tensor_name)
            assert dtype == "F32"

        element_count = 4 * 8
        array = np.zeros(element_count, dtype="float32")
        new_bytes = array.tobytes()

        with SafetensorsSplicer(temp_safetensors) as spl:
            spl.splice_tensor(tensor_name, new_bytes, verify_shape=True)

        from safetensors import safe_open
        with safe_open(temp_safetensors, framework="pt") as f:
            loaded = f.get_tensor(tensor_name)
            assert loaded.dtype == torch.float32
