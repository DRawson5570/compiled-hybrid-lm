"""Wave 3: SAE training pipeline tests."""

from __future__ import annotations

import os
import tempfile

import pytest
import torch

from sae_editor.sae_training import SAERegistry, SAETrainingPipeline
from sae_editor.tests.utils import SyntheticConfig, SyntheticModel, make_random_sae


class TestSAETrainingPipeline:
    def test_collect_layer_activations_shapes(self, synthetic_model):
        pipeline = SAETrainingPipeline()
        acts = pipeline.collect_layer_activations(
            model=synthetic_model,
            tokenizer=None,
            texts=["hello world", "test"],
            layers=[0, 1],
            max_length=8,
        )
        assert len(acts) == 2
        assert acts[0].ndim == 3

    def test_collect_multiple_batches(self, synthetic_model):
        pipeline = SAETrainingPipeline()
        texts = [f"text {i}" for i in range(10)]
        acts = pipeline.collect_layer_activations(
            model=synthetic_model,
            tokenizer=None,
            texts=texts,
            layers=[0],
            batch_size=3,
        )
        assert acts[0].shape[0] == 10


class TestSAERegistry:
    def test_save_load_round_trip(self):
        saes = {0: make_random_sae(64, 16), 2: make_random_sae(64, 16)}
        with tempfile.TemporaryDirectory() as tmpdir:
            SAERegistry.save(saes, tmpdir)
            loaded = SAERegistry.load(tmpdir, d_model=64, n_features=16)

            assert list(loaded.keys()) == [0, 2]
            for layer in [0, 2]:
                assert torch.allclose(
                    loaded[layer].encoder.weight, saes[layer].encoder.weight, atol=1e-6
                )

    def test_load_empty_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            loaded = SAERegistry.load(tmpdir, d_model=64, n_features=16)
            assert loaded == {}

    def test_create_decompiler(self, synthetic_model):
        saes = {0: make_random_sae(64, 16)}
        with tempfile.TemporaryDirectory() as tmpdir:
            SAERegistry.save(saes, tmpdir)
            decomp = SAERegistry.create_decompiler(
                model=synthetic_model,
                tokenizer=None,
                path_prefix=tmpdir,
                d_model=64,
                n_features=16,
            )
            assert 0 in decomp.saes
            assert decomp.saes[0].n_features == 16


@pytest.mark.slow
class TestSAETrainingSmoke:
    def test_train_one_sae_tiny_gpt2(self, tiny_gpt2_model, tiny_gpt2_tokenizer):
        pipeline = SAETrainingPipeline()
        texts = [
            "The capital of France is Paris.",
            "Machine learning is fun.",
            "Hello world.",
        ] * 5

        saes = pipeline.train_all(
            model=tiny_gpt2_model,
            tokenizer=tiny_gpt2_tokenizer,
            texts=texts,
            layers=[0],
            n_features=8,
            steps=50,
            lr=1e-3,
            batch_size=16,
            device="cpu",
        )
        assert 0 in saes
        features = saes[0].get_features()
        assert len(features) > 0
