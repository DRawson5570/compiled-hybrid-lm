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
    grad_process_group: Any | None = None

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
    def frozen(cls) -> 'TrainableSurface':
        """Freeze every backbone parameter.

        Cartridge-only trainers optimize parameters outside the backbone, so the
        backend still needs a first-class way to partition a fully frozen model.
        """
        return cls(())

    @classmethod
    def head_bias(cls) -> 'TrainableSurface':
        return cls(('head_bias',))

    @classmethod
    def head_bias_and_embeddings(cls) -> 'TrainableSurface':
        """Include embeddings for weight-tied output projection."""
        return cls(('head_bias', 'tok_emb.weight', 'pos_emb.weight'))


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


def allreduce_trainable_grads(
    model: nn.Module,
    world_size: int | None = None,
    process_group: Any | None = None,
) -> None:
    """Average gradients for manually synchronized distributed trainable surfaces."""

    if not dist.is_available() or not dist.is_initialized():
        return
    actual_world_size = world_size or dist.get_world_size(process_group)
    for param in model.parameters():
        if not param.requires_grad or param.grad is None:
            continue
        if process_group is not None and param.grad.is_cuda:
            grad_cpu = param.grad.detach().to('cpu')
            dist.all_reduce(grad_cpu, op=dist.ReduceOp.SUM, group=process_group)
            grad_cpu.div_(actual_world_size)
            param.grad.copy_(grad_cpu.to(param.grad.device))
        else:
            dist.all_reduce(param.grad, op=dist.ReduceOp.SUM, group=process_group)
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
    compute_in_4bit: bool = False

    def prepare(self, model: nn.Module, surface: TrainableSurface) -> BackendHandle:
        device = torch.device(self.device)
        names = set_trainable_surface(model, surface)
        config, coordinator_cls, wrapper_cls, status_cls = self._load_zeroq()

        process_group = None
        if dist.is_available() and dist.is_initialized() and device.type == 'cuda':
            major, _ = torch.cuda.get_device_capability(device)
            if major < 6:
                process_group = dist.new_group(backend='gloo')

        if process_group is None:
            coordinator = coordinator_cls(config)
        else:
            try:
                coordinator = coordinator_cls(config, process_group=process_group)
            except TypeError:
                coordinator = coordinator_cls(config)
                coordinator.process_group = process_group
        wrapper = wrapper_cls(model, coordinator, trainable_only=False)

        if self.stream_partition:
            self._stream_partition(coordinator, status_cls, device)
        else:
            wrapper.partition()

        if self.compute_in_4bit:
            _convert_linear_modules_to_4bit_compute(model, coordinator, wrapper, device, process_group)
            _force_python_encoder_forward(model)

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
            grad_process_group=process_group,
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
            # Never partition params that the surface wants trainable (head_bias, etc.)
            if zq_param.param.requires_grad:
                continue
            full_precision = zq_param.param.data.to(device=device)
            zq_param.partition_from_full_precision(full_precision)


def _convert_linear_modules_to_4bit_compute(
    model: nn.Module,
    coordinator: Any,
    wrapper: Any,
    device: torch.device,
    process_group: Any | None = None,
) -> int:
    """Replace partitioned Linear modules with native bitsandbytes Linear4bit modules.

    ZeroQ's default hooks gather and release every frozen Linear weight on every
    forward/backward pass. On M40/SYS topology that communication dominates the
    run. This path gathers the already-quantized shards once, installs bnb's
    fused 4-bit matmul weight object, and removes the converted Linear modules
    from the live module tree so their ZeroQ hooks no longer fire.
    """

    try:
        import bitsandbytes as bnb
        from bitsandbytes.functional import QuantState
    except Exception as exc:  # pragma: no cover - host dependency varies
        raise RuntimeError(
            'compute_in_4bit=True requires bitsandbytes with Linear4bit support'
        ) from exc

    module_param_ids = getattr(wrapper, '_module_param_ids', None)
    params = getattr(coordinator, '_params', None)
    if module_param_ids is None or params is None:
        raise RuntimeError('ZeroQ wrapper/coordinator does not expose module parameter mappings')

    parent_lookup = {
        id(child): (parent, child_name)
        for parent in model.modules()
        for child_name, child in parent.named_children()
    }
    converted = 0
    for module, param_ids in list(module_param_ids.items()):
        if not isinstance(module, nn.Linear):
            continue
        parent_record = parent_lookup.get(id(module))
        if parent_record is None:
            continue

        weight_zq = _find_zeroq_param(params, param_ids, 'weight')
        if weight_zq is None:
            continue
        bias_zq = _find_zeroq_param(params, param_ids, 'bias') if module.bias is not None else None

        packed, quant_state, quant_meta = _assemble_4bit_weight(weight_zq, QuantState, device, process_group)
        param_4bit = bnb.nn.Params4bit(
            packed.contiguous().view(-1, 1),
            requires_grad=False,
            quant_state=quant_state,
            quant_type=quant_meta.get('quant_type', 'nf4'),
            blocksize=quant_meta.get('blocksize', 64),
        )
        new_linear = bnb.nn.Linear4bit(
            module.in_features,
            module.out_features,
            bias=module.bias is not None,
            compute_dtype=torch.float16,
            compress_statistics=False,
            quant_type=quant_meta.get('quant_type', 'nf4'),
        ).to(device)
        new_linear.weight = param_4bit
        if bias_zq is not None:
            coordinator.fetch_params([getattr(bias_zq, 'param_id')], async_op=False)
            new_linear.bias = torch.nn.Parameter(
                bias_zq.param.detach().to(device=device),
                requires_grad=False,
            )
        elif module.bias is not None and module.bias.numel() == module.out_features:
            new_linear.bias = torch.nn.Parameter(
                module.bias.detach().to(device=device),
                requires_grad=False,
            )

        parent, child_name = parent_record
        setattr(parent, child_name, new_linear)
        converted += 1

    if converted == 0:
        raise RuntimeError('compute_in_4bit=True did not find any partitioned Linear modules to convert')
    return converted


def _force_python_encoder_forward(model: nn.Module):
    """Patch encoder layers to use pure-Python forward instead of C++ fused path.

    The C++ fused ``torch._transformer_encoder_layer_fwd`` accesses Linear weights
    directly and cannot handle bitsandbytes Linear4bit dequantization.  Replacing
    each layer's forward with the manual ``_sa_block / _ff_block`` path ensures
    the Python Linear4bit.forward method is called, which handles dtype casting.
    """
    layers = getattr(model, 'encoder', None)
    if layers is None or not hasattr(layers, 'layers'):
        return

    for layer in layers.layers:
        if hasattr(layer, '_sa_block') and hasattr(layer, '_ff_block'):
            _sa = layer._sa_block
            _ff = layer._ff_block
            _n1 = layer.norm1
            _n2 = layer.norm2
            _dr1 = layer.dropout1
            _dr2 = layer.dropout2

            def make_py_forward(sa, ff, n1, n2, dr1, dr2):
                def py_forward(self, src, src_mask=None, src_key_padding_mask=None, is_causal=False):
                    x = src
                    x = n1(x + dr1(sa(x, src_mask, src_key_padding_mask, is_causal)))
                    x = n2(x + dr2(ff(x)))
                    return x
                return py_forward

            layer.forward = make_py_forward(_sa, _ff, _n1, _n2, _dr1, _dr2).__get__(layer)


def _find_zeroq_param(params: dict[Any, Any], param_ids: Iterable[Any], param_name: str) -> Any | None:
    for param_id in param_ids:
        zq_param = params.get(param_id)
        if zq_param is not None and getattr(zq_param, 'param_name', None) == param_name:
            return zq_param
    return None


def _assemble_4bit_weight(
    zq_param: Any,
    quant_state_cls: Any,
    device: torch.device,
    process_group: Any | None = None,
) -> tuple[torch.Tensor, Any, dict[str, Any]]:
    quant_meta = getattr(zq_param, '_quant_meta', None)
    if quant_meta is None:
        raise RuntimeError(f'Missing ZeroQ quant metadata for param_id={getattr(zq_param, "param_id", "?")}')
    local_packed = getattr(zq_param, 'local_packed', None)
    local_absmax = getattr(zq_param, 'local_absmax', None)
    if local_packed is None or local_absmax is None:
        raise RuntimeError(f'Missing ZeroQ local shards for param_id={getattr(zq_param, "param_id", "?")}')

    packed = _gather_partitioned_vector(
        local_packed,
        int(getattr(zq_param, 'packed_per_rank', local_packed.numel())),
        int(getattr(zq_param, '_packed_remainder', 0)),
        int(getattr(zq_param, '_packed_total', local_packed.numel())),
        device,
        process_group,
    )
    absmax = _gather_partitioned_vector(
        local_absmax,
        int(getattr(zq_param, 'absmax_per_rank', local_absmax.numel())),
        int(getattr(zq_param, '_absmax_remainder', 0)),
        int(getattr(zq_param, '_absmax_total', local_absmax.numel())),
        device,
        process_group,
    )
    out_dtype = quant_meta.get('dtype', getattr(zq_param, 'original_dtype', torch.float16))
    if device.type == 'cuda' and out_dtype == torch.bfloat16:
        out_dtype = torch.float16
    quant_state = quant_state_cls(
        absmax=absmax,
        shape=quant_meta['shape'],
        dtype=out_dtype,
        blocksize=quant_meta['blocksize'],
        code=quant_meta.get('code'),
        quant_type=quant_meta['quant_type'],
    )
    return packed, quant_state, quant_meta


def _gather_partitioned_vector(
    local: torch.Tensor,
    per_rank: int,
    remainder: int,
    total: int,
    device: torch.device,
    process_group: Any | None = None,
) -> torch.Tensor:
    if dist.is_available() and dist.is_initialized():
        world_size = dist.get_world_size(process_group)
    else:
        world_size = 1
    if world_size == 1:
        return local.contiguous().view(-1)[:total].to(device=device)

    stride = per_rank + remainder
    send = torch.zeros(stride, dtype=local.dtype, device='cpu')
    flat_local = local.detach().contiguous().view(-1).to('cpu')
    send[:flat_local.numel()] = flat_local
    gathered = [torch.empty_like(send) for _ in range(world_size)]
    dist.all_gather(gathered, send, group=process_group)

    pieces: list[torch.Tensor] = []
    for rank, shard in enumerate(gathered):
        shard_len = per_rank + (remainder if rank == world_size - 1 else 0)
        if shard_len > 0:
            pieces.append(shard[:shard_len])
    return torch.cat(pieces, dim=0)[:total].contiguous().to(device=device)


__all__ = [
    'BackendHandle',
    'DenseTorchBackend',
    'TrainableSurface',
    'ZeroQPartitionedBackend',
    'allreduce_trainable_grads',
    'set_trainable_surface',
    'trainable_parameters',
]