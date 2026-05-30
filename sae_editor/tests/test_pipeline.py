"""Integration tests for the full NRTCS round-trip pipeline."""

from __future__ import annotations

import os
import tempfile

import pytest
import torch
from safetensors.torch import save_file

from sae_editor.pipeline import NRTCSPipeline
from sae_editor.recompiler import RecompilerEngine


class TestNRTCSPipeline:
    @pytest.fixture
    def pipeline(self):
        return NRTCSPipeline(eps=1e-6)

    @pytest.fixture
    def sample_edits(self):
        d_in, d_out = 8, 8
        keys = torch.tensor([
            [0.9, -0.1, 0.1, 0.1, 0.0, 0.0, 0.0, 0.0],
            [-0.2, 0.8, 0.3, -0.1, 0.0, 0.0, 0.0, 0.0],
        ])
        values = torch.tensor([
            [0.1, 0.1, 0.9, 0.1, 0.0, 0.0, 0.0, 0.0],
            [0.8, -0.3, 0.1, 0.4, 0.0, 0.0, 0.0, 0.0],
        ])
        return {0: {"keys": keys, "values": values}}

    @pytest.fixture
    def sample_features(self):
        U = torch.tensor([
            [0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ]).T
        return {0: U}

    def test_compile_dense_map(self, pipeline, sample_edits):
        edit = sample_edits[0]
        result = pipeline.compile_dense_map(edit["keys"], edit["values"])
        assert "W_down" in result
        assert "W_up" in result
        assert result["W_down"].dtype == torch.float32

    def test_compile_dense_map_with_crosstalk(self, pipeline, sample_edits, sample_features):
        edit = sample_edits[0]
        result = pipeline.compile_dense_map(
            edit["keys"], edit["values"], sample_features[0]
        )
        assert torch.allclose(
            sample_features[0].T @ result["W_down"],
            torch.zeros(2, 2),
            atol=1e-4,
        )

    def test_compile_from_uvm_edits(self, pipeline, sample_edits):
        patches = pipeline.compile_from_uvm_edits(sample_edits)
        assert 0 in patches
        assert "W_down" in patches[0]
        assert "W_up" in patches[0]

    def test_compile_from_uvm_edits_multilayer(self, pipeline):
        edits = {
            0: {"keys": torch.randn(2, 8), "values": torch.randn(2, 8)},
            2: {"keys": torch.randn(3, 8), "values": torch.randn(3, 8)},
            5: {"keys": torch.randn(1, 8), "values": torch.randn(1, 8)},
        }
        patches = pipeline.compile_from_uvm_edits(edits)
        assert set(patches.keys()) == {0, 2, 5}

        for layer_idx, patch in patches.items():
            edit = edits[layer_idx]
            keys = edit["keys"]
            values = edit["values"]
            recon = keys @ patch["W_down"] @ patch["W_up"]
            assert torch.allclose(recon, values, atol=1e-4), f"Layer {layer_idx} failed"

    def test_verify_compilation_perfect(self, pipeline, sample_edits):
        patches = pipeline.compile_from_uvm_edits(sample_edits)
        results = pipeline.verify_compilation(sample_edits, patches)
        assert results[0]["mean_cosine"] > 0.999
        assert results[0]["mean_error"] < 1e-3

    def test_verify_compilation_auto_compile(self, pipeline, sample_edits):
        results = pipeline.verify_compilation(sample_edits)
        assert results[0]["mean_cosine"] > 0.999

    def test_splice_patches_roundtrip(self, pipeline, sample_edits):
        tensors = {
            "model.layers.0.mlp.down_proj.weight": torch.randn(8, 2),
            "model.layers.0.mlp.up_proj.weight": torch.randn(2, 8),
            "model.layers.2.mlp.down_proj.weight": torch.randn(8, 2),
            "model.layers.2.mlp.up_proj.weight": torch.randn(2, 8),
        }
        fd, path = tempfile.mkstemp(suffix=".safetensors")
        os.close(fd)
        save_file(tensors, path)

        try:
            edits = {
                0: {"keys": torch.randn(2, 8), "values": torch.randn(2, 8)},
                2: {"keys": torch.randn(2, 8), "values": torch.randn(2, 8)},
            }
            patches = pipeline.compile_from_uvm_edits(edits)
            pipeline.splice_patches(path, patches)

            from safetensors import safe_open
            with safe_open(path, framework="pt") as f:
                for layer_idx in [0, 2]:
                    loaded_down = f.get_tensor(f"model.layers.{layer_idx}.mlp.down_proj.weight")
                    expected_down = patches[layer_idx]["W_down"].to(dtype=loaded_down.dtype)
                    assert torch.allclose(loaded_down, expected_down, atol=1e-4)
        finally:
            os.unlink(path)

    def test_full_round_trip(self, pipeline, sample_edits):
        tensors = {
            "model.layers.0.mlp.down_proj.weight": torch.randn(8, 2),
            "model.layers.0.mlp.up_proj.weight": torch.randn(2, 8),
        }
        fd, path = tempfile.mkstemp(suffix=".safetensors")
        os.close(fd)
        save_file(tensors, path)

        try:
            patches = pipeline.round_trip(path, sample_edits)

            from safetensors import safe_open
            with safe_open(path, framework="pt") as f:
                loaded = f.get_tensor("model.layers.0.mlp.down_proj.weight")

            expected = patches[0]["W_down"].to(dtype=loaded.dtype)
            assert torch.allclose(loaded, expected, atol=1e-4)
        finally:
            os.unlink(path)

    def test_france_paris_walkthrough(self, pipeline):
        """Concrete verification walkthrough from NRTCS_SPEC.md Section 6."""
        d = 8
        france_key = torch.tensor([[0.9, -0.1, 0.1, 0.1, 0.0, 0.0, 0.0, 0.0]])
        paris_value = torch.tensor([[0.1, 0.1, 0.9, 0.1, 0.0, 0.0, 0.0, 0.0]])
        london_value = torch.tensor([[0.1, 0.9, 0.1, 0.1, 0.0, 0.0, 0.0, 0.0]])

        result_no_proj = pipeline.compile_dense_map(france_key, paris_value)
        recon_no_proj = france_key @ result_no_proj["W_down"] @ result_no_proj["W_up"]
        assert torch.allclose(recon_no_proj, paris_value, atol=1e-4), "Should recover Paris"
        assert not torch.allclose(recon_no_proj, london_value, atol=0.1), "Should NOT recover London"

        other_features = torch.tensor([
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        ]).T

        result_proj = pipeline.compile_dense_map(
            france_key, paris_value, original_features=other_features
        )
        recon_proj = france_key @ result_proj["W_down"] @ result_proj["W_up"]
        assert torch.allclose(recon_proj, paris_value, atol=1e-4), "Should recover Paris even with projection"

        assert torch.allclose(
            other_features.T @ result_proj["W_down"],
            torch.zeros(2, 1),
            atol=1e-4,
        ), "Crosstalk prevented"

    def test_reconstruction_fidelity_under_crosstalk(self, pipeline):
        """Patched mappings survive crosstalk prevention when keys are
        orthogonal to protected features."""
        d = 16
        N = 4
        m = 6

        U = torch.zeros(d, m)
        for i in range(m):
            U[i, i] = 1.0

        keys = torch.randn(N, d)
        keys[:, :m] = 0.0
        values = torch.randn(N, d)

        patches = pipeline.compile_dense_map(keys, values, original_features=U)
        W_down = patches["W_down"]
        W_up = patches["W_up"]

        recon = keys @ W_down @ W_up
        assert torch.allclose(recon, values, atol=1e-4)

        assert torch.allclose(U.T @ W_down, torch.zeros(m, N), atol=1e-4)
