"""Tests for train_steerer_v4.py — model building, dataset, steerer."""
from __future__ import annotations
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from hybrid.train_steerer_v4 import (
    MODEL_CONFIGS,
    StreamingTokenDataset,
    StreamingSteererDatasetV4,
    v4_zeroq_surface,
    v4_trainable_resident_names,
    select_v4_optimizer_params,
    build_fresh_lm,
    V,
)
import hybrid.train_scaled_neural_lm  # noqa — ensures DeepCausalLM import works


class TestModelConfigs:
    def test_all_configs_present(self):
        assert '124m' in MODEL_CONFIGS
        assert '500m' in MODEL_CONFIGS
        assert '1b' in MODEL_CONFIGS
        assert '2b' in MODEL_CONFIGS
        assert '4b' in MODEL_CONFIGS
        assert '700m' in MODEL_CONFIGS

    def test_124m_params(self):
        cfg = MODEL_CONFIGS['124m']
        assert cfg['d_model'] == 768
        assert cfg['n_layers'] == 12
        assert cfg['n_heads'] == 12
        assert cfg['d_ff'] == 3072

    def test_500m_params(self):
        cfg = MODEL_CONFIGS['500m']
        assert cfg['d_model'] == 1408
        assert cfg['n_layers'] == 18
        assert cfg['d_ff'] == 5632

    def test_2b_params(self):
        cfg = MODEL_CONFIGS['2b']
        assert cfg['d_model'] == 2560
        assert cfg['n_layers'] == 24

    def test_4b_params(self):
        cfg = MODEL_CONFIGS['4b']
        assert cfg['d_model'] == 3072
        assert cfg['n_layers'] == 40
        assert cfg['n_heads'] == 24
        assert cfg['d_ff'] == 12288


class TestBuildFreshLM:
    def test_build_4b_dense(self):
        model, d_model = build_fresh_lm('4b', 128, 'cpu')
        assert d_model == 3072
        n = sum(p.numel() for p in model.parameters())
        assert 4_500_000_000 < n < 5_000_000_000  # DeepCausalLM counts with nn.TransformerEncoder


class TestStreamingTokenDataset:
    @pytest.fixture
    def ids(self):
        return torch.randint(0, 50000, (10000,))

    def test_len(self, ids):
        ds = StreamingTokenDataset(ids, 128)
        assert len(ds) == 1000000  # matches StreamingSteererDatasetV4 pattern

    def test_getitem_shape(self, ids):
        ds = StreamingTokenDataset(ids, 128)
        x, y = ds[0]
        assert x.shape == (128,)
        assert y.shape == (128,)
        assert x.dtype == torch.int64
        # y should be x shifted by 1
        assert torch.equal(y[:-1], x[1:])


class TestStreamingSteererDatasetV4:
    @pytest.fixture
    def ids(self):
        return torch.randint(0, 50000, (10000,))

    def test_len(self, ids):
        ds = StreamingSteererDatasetV4(ids, 128, V)
        assert len(ds) == 1000000

    def test_getitem_shape(self, ids):
        ds = StreamingSteererDatasetV4(ids, 128, V)
        x, y, w_cpu = ds[0]
        assert x.shape == (128,)
        assert y.shape == (128,)
        assert w_cpu.shape == (128, 9)
        assert w_cpu.dtype == torch.float32
        assert torch.equal(y[:-1], x[1:])


class TestZeroQSurface:
    def test_surface_selects_correct_names(self):
        model, _ = build_fresh_lm('124m', 128, 'cpu')
        surface = v4_zeroq_surface(model)
        names = set(surface.parameter_names)
        assert 'head_bias' in names
        assert 'tok_emb.weight' in names
        assert 'pos_emb.weight' in names
        assert 'ln_f.weight' in names
        # FFN weights should NOT be in the surface
        assert not any('linear1' in n or 'linear2' in n for n in names)

    def test_surface_nonempty(self):
        model, _ = build_fresh_lm('124m', 128, 'cpu')
        surface = v4_zeroq_surface(model)
        assert len(surface.parameter_names) > 0


class TestTrainableResidentNames:
    def test_includes_head_bias(self):
        model, _ = build_fresh_lm('124m', 128, 'cpu')
        names = v4_trainable_resident_names(model)
        assert 'head_bias' in names
        assert 'tok_emb.weight' in names
