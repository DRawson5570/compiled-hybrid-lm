"""Wave 7: Transfer, compaction, activation mitigation, CLI tests."""

from __future__ import annotations

import tempfile

import pytest
import torch

from sae_editor.recompiler import (
    compact_features,
    decompact_features,
    pre_activation_scale,
)
from sae_editor.transfer import project_features, transfer_edit
from sae_editor.architectures import GPT2, QWEN2


class TestFeatureCompaction:
    def test_compact_reduces_rank(self):
        W_down = torch.randn(128, 50)
        W_comp, basis = compact_features(W_down, n_components=10)
        assert W_comp.shape == (128, 10)
        assert basis.shape[0] == 10

    def test_decompact_recovers_approximate(self):
        W_down = torch.randn(128, 20)
        W_comp, basis = compact_features(W_down, n_components=10)
        recovered = decompact_features(W_comp, basis)
        assert recovered.shape == W_down.shape
        cosine = torch.nn.functional.cosine_similarity(
            recovered.flatten(), W_down.flatten(), dim=0
        )
        assert cosine > 0.5, f"Recovery cosine: {cosine:.4f}"

    def test_compact_all_components_exact(self):
        W_down = torch.randn(64, 16)
        W_comp, basis = compact_features(W_down, n_components=16)
        recovered = decompact_features(W_comp, basis)
        assert torch.allclose(recovered, W_down, atol=1e-4)


class TestPreActivationScale:
    def test_scale_reduces_max_abs(self):
        keys = torch.randn(10, 128) * 5.0
        scaled = pre_activation_scale(keys, target_range=2.0)
        assert scaled.abs().max() <= 2.1

    def test_scale_preserves_small_keys(self):
        keys = torch.randn(10, 128) * 0.5
        scaled = pre_activation_scale(keys, target_range=2.0)
        assert torch.allclose(scaled, keys, atol=1e-6)


class TestCrossArchitectureTransfer:
    def test_project_features_preserves_shape(self):
        features = torch.randn(5, 768)
        projected = project_features(features, from_d_model=768, to_d_model=1536)
        assert projected.shape == (5, 1536)

    def test_project_features_same_dim_is_identity(self):
        features = torch.randn(3, 128)
        projected = project_features(features, from_d_model=128, to_d_model=128)
        assert torch.allclose(features, projected, atol=1e-6)

    def test_transfer_edit_produces_valid_structure(self):
        edits = {0: {"keys": torch.randn(2, 768), "values": torch.randn(2, 768)}}
        result = transfer_edit(edits, from_d_model=768, to_d_model=1536)
        assert 0 in result
        assert result[0]["keys"].shape == (2, 1536)
        assert result[0]["values"].shape == (2, 1536)
