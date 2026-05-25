from __future__ import annotations

import sys
import types

import pytest
import torch
import torch.nn as nn

from hybrid.backends import (
    DenseTorchBackend,
    TrainableSurface,
    ZeroQPartitionedBackend,
    set_trainable_surface,
)


class TinyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(3, 3), nn.Linear(3, 3)])
        self.ln_f = nn.LayerNorm(3)
        self.head_bias = nn.Parameter(torch.zeros(3))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.ln_f(x) + self.head_bias


def test_trainable_surface_freezes_everything_except_named_params():
    model = TinyBackbone()

    names = set_trainable_surface(model, TrainableSurface.head_bias())

    assert names == ('head_bias',)
    assert model.head_bias.requires_grad
    assert not model.layers[0].weight.requires_grad
    assert not model.ln_f.weight.requires_grad


def test_trainable_surface_rejects_missing_parameter():
    with pytest.raises(ValueError):
        set_trainable_surface(TinyBackbone(), TrainableSurface.from_names(['missing.weight']))


def test_dense_backend_keeps_cartridge_abi_model_shape_and_trainable_surface():
    model = TinyBackbone()

    handle = DenseTorchBackend(device='cpu').prepare(model, TrainableSurface.head_bias())
    out = handle.model(torch.ones(2, 4, 3))

    assert handle.backend == 'dense'
    assert handle.device == torch.device('cpu')
    assert handle.trainable_parameter_names == ('head_bias',)
    assert out.shape == (2, 4, 3)
    assert [name for name, param in model.named_parameters() if param.requires_grad] == ['head_bias']


def test_zeroq_backend_streams_only_frozen_params_and_leaves_surface_trainable(monkeypatch):
    class FakeStatus:
        NOT_AVAILABLE = 'not_available'
        PARTITIONED = 'partitioned'

    class FakeConfig:
        frozen_only = True
        partition_trainable = False

    class FakeZeroQParam:
        def __init__(self, name, param):
            self.name = name
            self.param = param
            self.status = FakeStatus.NOT_AVAILABLE
            self.partitioned_from_device = None

        def partition_from_full_precision(self, weight):
            self.partitioned_from_device = weight.device.type
            self.status = FakeStatus.PARTITIONED

    class FakeCoordinator:
        def __init__(self, config):
            self.config = config
            self._params = {}

        def get_memory_stats(self):
            return {'num_params': float(len(self._params))}

    class FakeWrapper:
        def __init__(self, model, coordinator, trainable_only=False):
            self.model = model
            self.coordinator = coordinator
            for name, param in model.named_parameters():
                if not param.requires_grad:
                    coordinator._params[name] = FakeZeroQParam(name, param)

        def partition(self):
            raise AssertionError('streaming partition should be used')

        def get_memory_stats(self):
            return self.coordinator.get_memory_stats()

    src_mod = types.ModuleType('src')
    config_mod = types.ModuleType('src.config')
    config_mod.MAXWELL_CONFIG = FakeConfig()
    coordinator_mod = types.ModuleType('src.coordinator')
    coordinator_mod.ZeroQCoordinator = FakeCoordinator
    coordinator_mod.ZeroQModuleWrapper = FakeWrapper
    coordinator_mod.ZeroQParamStatus = FakeStatus

    monkeypatch.setitem(sys.modules, 'src', src_mod)
    monkeypatch.setitem(sys.modules, 'src.config', config_mod)
    monkeypatch.setitem(sys.modules, 'src.coordinator', coordinator_mod)

    model = TinyBackbone()
    handle = ZeroQPartitionedBackend(device='cpu').prepare(model, TrainableSurface.head_bias())

    assert handle.backend == 'zeroq'
    assert handle.trainable_parameter_names == ('head_bias',)
    assert model.head_bias.requires_grad
    assert not model.layers[0].weight.requires_grad
    assert 'head_bias' not in handle.coordinator._params
    assert set(handle.coordinator._params) == {
        'layers.0.weight', 'layers.0.bias',
        'layers.1.weight', 'layers.1.bias',
        'ln_f.weight', 'ln_f.bias',
    }
    assert all(param.status == FakeStatus.PARTITIONED for param in handle.coordinator._params.values())
    assert all(param.partitioned_from_device == 'cpu' for param in handle.coordinator._params.values())
    assert handle.memory_stats() == {'num_params': 6.0}