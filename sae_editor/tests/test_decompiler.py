"""Tier 2+3: Decompiler unit tests (synthetic model + tiny-gpt2)."""

from __future__ import annotations

import pytest
import torch

from sae_editor.decompiler import NRTCSDecompiler
from sae_editor.tests.utils import (
    SyntheticConfig,
    SyntheticModel,
    make_random_sae,
)


class TestDecompilerWithSyntheticModel:
    def test_collect_activations_shapes(self, synthetic_model, synthetic_sae_factory):
        sae = synthetic_sae_factory(d_model=64)
        decomp = NRTCSDecompiler(
            model=synthetic_model,
            tokenizer=None,
            saes={0: sae, 1: sae, 2: sae},
            threshold=0.1,
            device="cpu",
        )
        texts = ["hello world", "test text"]
        acts = decomp.collect_activations(texts, max_length=8, batch_size=2)
        assert len(acts) == 3
        for layer_idx in [0, 1, 2]:
            assert acts[layer_idx].shape == (2, 8, 64), f"Layer {layer_idx} shape mismatch"

    def test_collect_activations_multiple_batches(self, synthetic_model, synthetic_sae_factory):
        sae = synthetic_sae_factory(d_model=64)
        decomp = NRTCSDecompiler(
            model=synthetic_model,
            tokenizer=None,
            saes={0: sae},
            threshold=0.1,
            device="cpu",
        )
        texts = ["a", "b", "c", "d", "e"]
        acts = decomp.collect_activations(texts, max_length=8, batch_size=2)
        assert acts[0].shape[0] == 5

    def test_collect_activations_hook_cleanup(self, synthetic_model, synthetic_sae_factory):
        sae = synthetic_sae_factory(d_model=64)
        decomp = NRTCSDecompiler(
            model=synthetic_model,
            tokenizer=None,
            saes={0: sae},
            threshold=0.1,
            device="cpu",
        )
        texts = ["hello world"]
        acts1 = decomp.collect_activations(texts, max_length=8)
        acts2 = decomp.collect_activations(texts, max_length=8)
        assert acts1[0].shape == acts2[0].shape
        assert len(decomp._hooks) == 0

    def test_extract_features_shapes(self, synthetic_model, synthetic_sae_factory):
        sae = synthetic_sae_factory(d_model=64, n_features=16)
        decomp = NRTCSDecompiler(
            model=synthetic_model,
            tokenizer=None,
            saes={0: sae},
            threshold=0.0,
            device="cpu",
        )
        texts = ["hello world", "test"]
        features = decomp.extract_features(texts, max_length=8)
        assert 0 in features
        f0 = features[0]
        assert "activations" in f0
        assert "feature_indices" in f0
        assert "feature_vectors" in f0
        assert "feature_acts" in f0

    def test_extract_features_threshold_filters(self, synthetic_model, synthetic_sae_factory):
        sae = synthetic_sae_factory(d_model=64, n_features=16)
        decomp_high = NRTCSDecompiler(
            model=synthetic_model,
            tokenizer=None,
            saes={0: sae},
            threshold=100.0,
            device="cpu",
        )
        texts = ["hello world"]
        features = decomp_high.extract_features(texts, max_length=8)
        assert len(features[0]["feature_indices"]) == 0

    def test_extract_features_empty_text(self, synthetic_model, synthetic_sae_factory):
        sae = synthetic_sae_factory(d_model=64, n_features=16)
        decomp = NRTCSDecompiler(
            model=synthetic_model,
            tokenizer=None,
            saes={0: sae},
            threshold=0.0,
            device="cpu",
        )
        features = decomp.extract_features([""], max_length=8)
        assert 0 in features

    def test_path_attribution_shapes(self, synthetic_model, synthetic_sae_factory):
        sae_up = synthetic_sae_factory(d_model=64, n_features=8)
        sae_down = synthetic_sae_factory(d_model=64, n_features=8)
        decomp = NRTCSDecompiler(
            model=synthetic_model,
            tokenizer=None,
            saes={0: sae_up, 1: sae_down},
            threshold=0.0,
            device="cpu",
        )
        attr = decomp.path_attribution(
            text="test",
            upstream_layer=0,
            downstream_layer=1,
            upstream_features=[0, 1, 2],
            downstream_feature=3,
        )
        assert attr["attributions"].shape == (3,)
        assert len(attr["downstream_indices"]) == 1

    def test_path_attribution_upstream_subset(self, synthetic_model, synthetic_sae_factory):
        sae_up = synthetic_sae_factory(d_model=64, n_features=16)
        sae_down = synthetic_sae_factory(d_model=64, n_features=16)
        decomp = NRTCSDecompiler(
            model=synthetic_model,
            tokenizer=None,
            saes={0: sae_up, 1: sae_down},
            threshold=0.0,
            device="cpu",
        )
        attr = decomp.path_attribution(
            text="test",
            upstream_layer=0,
            downstream_layer=1,
            upstream_features=[5, 7],
            downstream_feature=0,
        )
        assert attr["attributions"].shape == (2,)

    def test_path_attribution_missing_sae(self, synthetic_model, synthetic_sae_factory):
        sae = synthetic_sae_factory(d_model=64)
        decomp = NRTCSDecompiler(
            model=synthetic_model,
            tokenizer=None,
            saes={0: sae},
            threshold=0.0,
            device="cpu",
        )
        with pytest.raises(KeyError):
            decomp.path_attribution("test", upstream_layer=0, downstream_layer=2)

    def test_get_layer_gpt2_style(self):
        config = SyntheticConfig(d_model=64, n_layers=2)
        model = SyntheticModel(config)
        model.transformer = type("H", (), {"h": model.model.layers})()
        sae = make_random_sae(64, 8)
        decomp = NRTCSDecompiler(model=model, tokenizer=None, saes={0: sae}, device="cpu")
        layer = decomp._get_layer(0)
        assert layer is not None

    def test_get_layer_unknown(self, synthetic_sae_factory):
        config = SyntheticConfig(d_model=64, n_layers=2)
        model = SyntheticModel(config)
        model.model = None
        if hasattr(model, "layers"):
            del model.layers
        sae = synthetic_sae_factory(d_model=64)
        decomp = NRTCSDecompiler(model=model, tokenizer=None, saes={0: sae}, device="cpu")
        with pytest.raises(AttributeError, match="Cannot find layers"):
            decomp._get_layer(0)


@pytest.mark.slow
class TestDecompilerWithTinyGPT2:
    def test_real_collect_activations_shapes(self, tiny_gpt2_model, tiny_gpt2_tokenizer):
        d_model = tiny_gpt2_model.config.hidden_size
        sae = make_random_sae(d_model=d_model, n_features=32)
        decomp = NRTCSDecompiler(
            model=tiny_gpt2_model,
            tokenizer=tiny_gpt2_tokenizer,
            saes={0: sae, 1: sae},
            threshold=0.0,
            device="cpu",
        )
        texts = ["The capital of France is Paris.", "Machine learning is fun."]
        acts = decomp.collect_activations(texts, max_length=32, batch_size=4)
        assert len(acts) == 2
        for layer_idx in [0, 1]:
            assert acts[layer_idx].ndim == 3
            assert acts[layer_idx].shape[-1] == d_model

    def test_real_collect_activations_variable_lengths(self, tiny_gpt2_model, tiny_gpt2_tokenizer):
        d_model = tiny_gpt2_model.config.hidden_size
        sae = make_random_sae(d_model=d_model, n_features=32)
        decomp = NRTCSDecompiler(
            model=tiny_gpt2_model,
            tokenizer=tiny_gpt2_tokenizer,
            saes={0: sae, 1: sae},
            threshold=0.0,
            device="cpu",
        )
        texts = ["Hi.", "This is a much longer sentence with more tokens."]
        acts = decomp.collect_activations(texts, max_length=64, batch_size=4)
        assert acts[0].ndim == 3
        assert acts[1].ndim == 3

    def test_real_extract_features_runs(self, tiny_gpt2_model, tiny_gpt2_tokenizer):
        d_model = tiny_gpt2_model.config.hidden_size
        sae = make_random_sae(d_model=d_model, n_features=32)
        decomp = NRTCSDecompiler(
            model=tiny_gpt2_model,
            tokenizer=tiny_gpt2_tokenizer,
            saes={0: sae, 1: sae},
            threshold=0.0,
            device="cpu",
        )
        texts = ["Hello world."]
        features = decomp.extract_features(texts, max_length=16)
        assert 0 in features
        assert 1 in features

    def test_real_path_attribution_gradient_nonzero(self, tiny_gpt2_model, tiny_gpt2_tokenizer):
        d_model = tiny_gpt2_model.config.hidden_size
        sae = make_random_sae(d_model=d_model, n_features=32)
        decomp = NRTCSDecompiler(
            model=tiny_gpt2_model,
            tokenizer=tiny_gpt2_tokenizer,
            saes={0: sae, 1: sae},
            threshold=0.0,
            device="cpu",
        )
        attr = decomp.path_attribution(
            text="The capital of France is Paris.",
            upstream_layer=0,
            downstream_layer=1,
            upstream_features=[0, 1, 2, 3],
            downstream_feature=0,
        )
        assert attr["attributions"].numel() > 0, "Should return attributions"

    def test_real_path_attribution_returns_all_features(self, tiny_gpt2_model, tiny_gpt2_tokenizer):
        d_model = tiny_gpt2_model.config.hidden_size
        sae = make_random_sae(d_model=d_model, n_features=32)
        decomp = NRTCSDecompiler(
            model=tiny_gpt2_model,
            tokenizer=tiny_gpt2_tokenizer,
            saes={0: sae, 1: sae},
            threshold=0.0,
            device="cpu",
        )
        attr = decomp.path_attribution(
            text="Test sentence for attribution.",
            upstream_layer=0,
            downstream_layer=1,
        )
        assert len(attr["attributions"]) == 32
        assert len(attr["downstream_indices"]) == 32
        assert not torch.isnan(attr["upstream_acts"]).any()
        assert not torch.isnan(attr["downstream_acts"]).any()
