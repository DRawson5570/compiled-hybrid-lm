"""Test recompiler: analytical matrix construction and crosstalk prevention."""

from __future__ import annotations

import pytest
import torch

from sae_editor.recompiler import (
    RecompilerEngine,
    build_dense_map,
    compute_null_space_rank,
    orthogonal_projection,
    verify_dense_map,
)


class TestBuildDenseMap:
    def test_single_pair_identity(self):
        d = 8
        key = torch.randn(1, d)
        value = torch.randn(1, d)
        W_down, W_up = build_dense_map(key, value)

        assert W_down.shape == (d, 1)
        assert W_up.shape == (1, d)
        recon = verify_dense_map(key, W_down, W_up)
        assert torch.allclose(recon, value, atol=1e-4)

    def test_multiple_pairs_exact(self):
        N, d_in, d_out = 4, 16, 32
        keys = torch.randn(N, d_in)
        values = torch.randn(N, d_out)
        W_down, W_up = build_dense_map(keys, values)

        assert W_down.shape == (d_in, N)
        assert W_up.shape == (N, d_out)
        recon = verify_dense_map(keys, W_down, W_up)
        assert torch.allclose(recon, values, atol=1e-4)

    def test_orthonormal_keys(self):
        d = 8
        N = 8
        keys = torch.eye(d)[:N]
        values = torch.randn(N, d)
        W_down, W_up = build_dense_map(keys, values)

        recon = verify_dense_map(keys, W_down, W_up)
        assert torch.allclose(recon, values, atol=1e-4)

    def test_regularization_prevents_singularity(self):
        N, d = 3, 8
        keys = torch.randn(N, d)
        keys[1] = keys[0]
        values = torch.randn(N, d)

        W_down, W_up = build_dense_map(keys, values, eps=1e-3)
        assert not torch.isnan(W_down).any()
        assert not torch.isinf(W_down).any()

    def test_dtype_consistency(self):
        keys = torch.randn(3, 8, dtype=torch.float16)
        values = torch.randn(3, 8, dtype=torch.float16)
        W_down, W_up = build_dense_map(keys, values)

        assert W_down.dtype == torch.float32
        assert W_up.dtype == torch.float32

    def test_shape_mismatch_raises(self):
        keys = torch.randn(3, 8)
        values = torch.randn(5, 8)
        with pytest.raises(ValueError):
            build_dense_map(keys, values)


class TestOrthogonalProjection:
    def test_projection_kills_original_features(self):
        d = 16
        m = 4
        U = torch.randn(d, m)
        W = torch.randn(d, 8)

        W_proj = orthogonal_projection(W, U)

        assert torch.allclose(U.T @ W_proj, torch.zeros(m, 8), atol=1e-4)

    def test_projection_idempotent(self):
        d = 16
        m = 4
        U = torch.randn(d, m)
        W = torch.randn(d, 8)

        W1 = orthogonal_projection(W, U)
        W2 = orthogonal_projection(W1, U)

        assert torch.allclose(W1, W2, atol=1e-5)

    def test_null_space_rank(self):
        d = 32
        m = 10
        U = torch.randn(d, m)
        rank = compute_null_space_rank(U)
        assert rank == d - m

    def test_regularization_stability(self):
        d = 32
        U = torch.randn(d, 8)
        U[:, 4] = U[:, 0] + 1e-7
        W = torch.randn(d, 8)

        W_proj = orthogonal_projection(W, U, eps=1e-3)
        assert not torch.isnan(W_proj).any()


class TestRecompilerEngine:
    def test_compile_simple(self):
        engine = RecompilerEngine(eps=1e-6)
        keys = torch.randn(3, 16)
        values = torch.randn(3, 16)

        result = engine.compile(keys, values)
        assert "W_down" in result
        assert "W_up" in result

        recon = engine.verify(keys, result["W_down"], result["W_up"])
        assert torch.allclose(recon, values, atol=1e-4)

    def test_compile_with_crosstalk(self):
        engine = RecompilerEngine(eps=1e-6)
        d = 16
        U = torch.randn(d, 4)
        keys = torch.randn(3, d)
        values = torch.randn(3, d)

        result = engine.compile(keys, values, original_features=U)
        assert torch.allclose(U.T @ result["W_down"], torch.zeros(4, 3), atol=1e-4)

    def test_compile_from_pairs(self):
        engine = RecompilerEngine()
        pairs = [(torch.randn(8), torch.randn(8)) for _ in range(3)]
        result = engine.compile_from_pairs(pairs)

        keys = torch.stack([k for k, v in pairs])
        values = torch.stack([v for k, v in pairs])
        recon = engine.verify(keys, result["W_down"], result["W_up"])
        assert torch.allclose(recon, values, atol=1e-4)

    def test_france_to_paris_scenario(self):
        """Verification walkthrough from NRTCS_SPEC.md Section 6."""
        d = 8

        france_key = torch.tensor([0.9, -0.1, 0.1, 0.1, 0.0, 0.0, 0.0, 0.0])
        london_value = torch.tensor([0.1, 0.9, 0.1, 0.1, 0.0, 0.0, 0.0, 0.0])
        paris_value = torch.tensor([0.1, 0.1, 0.9, 0.1, 0.0, 0.0, 0.0, 0.0])

        engine = RecompilerEngine(eps=1e-6)

        result = engine.compile(france_key.unsqueeze(0), paris_value.unsqueeze(0))
        recon = engine.verify(
            france_key.unsqueeze(0), result["W_down"], result["W_up"]
        )

        assert torch.allclose(recon.squeeze(0), paris_value, atol=1e-4)
        assert not torch.allclose(recon.squeeze(0), london_value, atol=0.1)
