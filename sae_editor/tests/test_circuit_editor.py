"""Wave 4: Circuit editor tests."""

from __future__ import annotations

import pytest
import torch

from sae_editor.circuit_editor import CircuitEditor
from sae_editor.decompiler import NRTCSDecompiler


class TestCircuitEditor:
    @pytest.fixture
    def editor(self, synthetic_model, synthetic_sae_factory):
        sae = synthetic_sae_factory(d_model=64, n_features=16)
        decomp = NRTCSDecompiler(
            model=synthetic_model, tokenizer=None,
            saes={0: sae}, threshold=0.0, device="cpu",
        )
        return CircuitEditor(decomp)

    def test_find_feature_returns_dict(self, editor):
        features = editor.find_feature_activating_on(["hello world"])
        assert isinstance(features, dict)
        assert 0 in features

    def test_extract_feature_vector_shape(self, editor):
        vec = editor.extract_feature_vector(layer=0, feature_idx=3)
        assert vec.shape == (64,)

    def test_extract_value_vector_for_text(self, editor):
        vec = editor.extract_value_vector_for_text("hello world", layer=0)
        assert vec.shape == (64,)

    def test_create_edit_from_texts_structure(self, editor):
        edit = editor.create_edit_from_texts("hello", "world", layer=0, top_k=2)
        if 0 in edit:
            assert "keys" in edit[0]
            assert "values" in edit[0]
            assert edit[0]["keys"].ndim == 2
            assert edit[0]["values"].ndim == 2

    def test_find_feature_empty_text(self, editor):
        features = editor.find_feature_activating_on([""])
        assert isinstance(features, dict)
        assert 0 in features
        assert isinstance(features[0], list)

    def test_verify_edit_structure(self, editor):
        keys = torch.randn(2, 64)
        values = torch.randn(2, 64)
        edit = {0: {"keys": keys, "values": values}}
        result = editor.verify_edit(edit)
        assert isinstance(result, bool)
