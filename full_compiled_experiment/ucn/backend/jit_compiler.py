from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Tuple

import torch

from .cache import L1StructuralCache, L2SemanticCache
from .codegen.reference import ReferenceBackend
from .optimizer import optimize


class KernelHandle:
    def __init__(self, compiled_fn: Callable, params: Dict[str, torch.Tensor]):
        self.compiled_fn = compiled_fn
        self.params = params


class JITCompiler:
    def __init__(
        self,
        stdlib_weights: Dict[str, Any] | None = None,
        device: str = "cuda",
        dtype: torch.dtype = torch.float32,
        use_triton: bool = False,
    ):
        self.stdlib_weights = stdlib_weights or {}
        self.device = device
        self.dtype = dtype
        self.l1_cache = L1StructuralCache()
        self.l2_cache = L2SemanticCache()
        self.use_triton = use_triton

        self.reference = ReferenceBackend(
            stdlib_weights=self.stdlib_weights,
            device=device,
            dtype=dtype,
        )

        self.triton_backend = None
        if use_triton:
            from .codegen.triton_backend import TritonBackend
            self.triton_backend = TritonBackend(
                stdlib_weights=self.stdlib_weights,
                device=device,
            )

    def compile(self, program) -> KernelHandle:
        cached = self.l1_cache.get(program)
        if cached is not None:
            return cached

        program = optimize(program)

        kernel = self._make_kernel(program)
        params = self._extract_params(program)

        handle = KernelHandle(kernel, params)
        self.l1_cache.put(program, handle)
        return handle

    def _make_kernel(self, program) -> Callable:
        if self.triton_backend is not None:
            try:
                return self.triton_backend.compile(program)
            except Exception:
                pass

        def reference_kernel(
            inputs: Dict[str, torch.Tensor],
            batch_size: int | None = None,
        ) -> Dict[str, torch.Tensor]:
            return self.reference.execute(program, inputs, batch_size)

        return reference_kernel

    def _extract_params(self, program) -> Dict[str, torch.Tensor]:
        params = {}
        for stmt in program.statements:
            expr = stmt.expr
            from ..dsl.ast import Transform
            if isinstance(expr, Transform):
                if expr.matrix.ref_type == "stdlib":
                    name = expr.matrix.name
                    if name in self.stdlib_weights:
                        for k, v in self.stdlib_weights[name].items():
                            if isinstance(v, torch.Tensor):
                                params[f"{name}/{k}"] = v
        return params

    def execute(
        self,
        handle: KernelHandle,
        inputs: Dict[str, torch.Tensor],
        batch_size: int | None = None,
    ) -> Dict[str, torch.Tensor]:
        return handle.compiled_fn(inputs, batch_size)

    def compile_and_execute(
        self,
        program,
        inputs: Dict[str, torch.Tensor],
        batch_size: int | None = None,
        context_z: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        handle = self.compile(program)
        return self.execute(handle, inputs, batch_size)

    def clear_caches(self):
        self.l1_cache.clear()
        self.l2_cache.clear()
