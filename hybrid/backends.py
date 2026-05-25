"""Runtime backends for CMI Hybrid models.

The cartridge ABI should not know whether the frozen backbone is dense PyTorch
or a ZeroQ-partitioned model. This module keeps that execution choice below the
cartridge rack and exposes a small preparation surface for training scripts.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any, Iterable

import torch
import torch.distributed as dist
import torch.nn as nn


@dataclass(frozen=True)
class BackendHandle:
    """Prepared backend state returned to training/runtime code."""

    backend: str
    model: nn.Module
    device: torch.device
    trainable_parameter_names: tuple[str, ...]
    coordinator: Any | None = None
    wrapper: Any | None = None

    def memory_stats(self) -> dict[str, float]:
        if self.wrapper is None or not hasattr(self.wrapper, 'get_memory_stats'):
            return {}
        return dict(self.wrapper.get_memory_stats())


@dataclass(frozen=True)
class TrainableSurface:
    """Names the small trainable surface riding on a frozen backbone."""

    parameter_names: tuple[str, ...]

    @classmethod
    def from_names(cls, names: Iterable[str]) -> 'TrainableSurface':
        cleaned = tuple(dict.fromkeys(str(name) for name in names))
        if not cleaned:
            raise ValueError('trainable surface must include at least one parameter name')
        return cls(cleaned)

    @classmethod
    def head_bias(cls) -> 'TrainableSurface':
        return cls(('head_bias',))


def set_trainable_surface(
    model: nn.Module,
    surface: TrainableSurface,
    *,
    allow_missing: bool = False,
) -> tuple[str, ...]:
    """Freeze all model parameters except the named trainable surface."""

    wanted = set(surface.parameter_names)
    found: list[str] = []
    for name, param in model.named_parameters():
        is_trainable = name in wanted
        param.requires_grad = is_trainable
        if is_trainable:
            found.append(name)

    missing = tuple(name for name in surface.parameter_names if name not in found)
    if missing and not allow_missing:
        raise ValueError(f'model is missing trainable surface parameters: {missing}')
    return tuple(found)


def trainable_parameters(model: nn.Module) -> list[nn.Parameter]:
    return [param for param in model.parameters() if param.requires_grad]


def allreduce_trainable_grads(model: nn.Module, world_size: int | None = None) -> None:
    """Average gradients for manually synchronized distributed trainable surfaces."""

    if not dist.is_available() or not dist.is_initialized():
        return
    actual_world_size = world_size or dist.get_world_size()
    for param in model.parameters():
        if not param.requires_grad or param.grad is None:
            continue
        dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
        param.grad.div_(actual_world_size)


@dataclass(frozen=True)
class DenseTorchBackend:
    """Standard PyTorch backend for models that fit directly on one device."""

    device: str | torch.device = 'cpu'

    def prepare(self, model: nn.Module, surface: TrainableSurface) -> BackendHandle:
        device = torch.device(self.device)
        names = set_trainable_surface(model, surface)
        model.to(device)
        return BackendHandle(
            backend='dense',
            model=model,
            device=device,
            trainable_parameter_names=names,
        )


@dataclass(frozen=True)
class ZeroQPartitionedBackend:
    """ZeroQ backend for huge frozen or mostly-frozen CMI backbones.

    ZeroQ is optional and loaded only when this backend is prepared. Frozen
    parameters are quantized/sharded by ZeroQ hooks; named trainable parameters
    stay materialized for cartridge/head/adapter optimization.
    """

    device: str | torch.device
    zeroq_path: str | Path | None = None
    config_name: str = 'MAXWELL_CONFIG'
    stream_partition: bool = True

    def prepare(self, model: nn.Module, surface: TrainableSurface) -> BackendHandle:
        device = torch.device(self.device)
        names = set_trainable_surface(model, surface)
        config, coordinator_cls, wrapper_cls, status_cls = self._load_zeroq()

        coordinator = coordinator_cls(config)
        wrapper = wrapper_cls(model, coordinator, trainable_only=False)

        if self.stream_partition:
            self._stream_partition(coordinator, status_cls, device)
        else:
            wrapper.partition()

        # Re-assert the requested surface after partition setup, then materialize
        # only those parameters. Frozen params are owned by ZeroQ shards/hooks.
        names = set_trainable_surface(model, surface)
        for param in model.parameters():
            if param.requires_grad:
                param.data = param.data.to(device=device)

        return BackendHandle(
            backend='zeroq',
            model=model,
            device=device,
            trainable_parameter_names=names,
            coordinator=coordinator,
            wrapper=wrapper,
        )

    def _load_zeroq(self):
        if self.zeroq_path is not None:
            path = str(Path(self.zeroq_path).expanduser())
            if path not in sys.path:
                sys.path.insert(0, path)

        try:
            from src import config as zeroq_config
            from src.coordinator import ZeroQCoordinator, ZeroQModuleWrapper, ZeroQParamStatus
        except Exception as exc:  # pragma: no cover - exact import failure depends on host env
            raise RuntimeError(
                'ZeroQ backend requested, but ZeroQ could not be imported. '
                'Pass zeroq_path=Path("~/ZeroQ") or install ZeroQ on PYTHONPATH.'
            ) from exc

        try:
            config = getattr(zeroq_config, self.config_name)
        except AttributeError as exc:
            raise ValueError(f'ZeroQ config {self.config_name!r} was not found') from exc
        return config, ZeroQCoordinator, ZeroQModuleWrapper, ZeroQParamStatus

    @staticmethod
    def _stream_partition(coordinator: Any, status_cls: Any, device: torch.device) -> None:
        params = getattr(coordinator, '_params', None)
        if params is None:
            raise RuntimeError('ZeroQ coordinator does not expose _params for streaming partition')

        not_available = getattr(status_cls, 'NOT_AVAILABLE')
        for zq_param in params.values():
            if getattr(zq_param, 'status') != not_available:
                continue
            full_precision = zq_param.param.data.to(device=device)
            zq_param.partition_from_full_precision(full_precision)


__all__ = [
    'BackendHandle',
    'DenseTorchBackend',
    'TrainableSurface',
    'ZeroQPartitionedBackend',
    'allreduce_trainable_grads',
    'set_trainable_surface',
    'trainable_parameters',
]