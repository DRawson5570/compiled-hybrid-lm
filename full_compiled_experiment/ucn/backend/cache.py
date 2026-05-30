from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch


def murmur3_hash(data: str | bytes) -> str:
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()[:16]


def ast_structure_hash(program) -> str:
    from ..dsl.ast import (
        Activate,
        GatherContext,
        Mix,
        Program,
        Project,
        QueryMemory,
        Residual,
        Rotate,
        Statement,
        Transform,
    )

    parts = []
    for stmt in program.statements:
        expr = stmt.expr
        if isinstance(expr, Mix):
            parts.append(f"mix_{len(expr.inputs)}")
        elif isinstance(expr, Project):
            ss = expr.subspace
            parts.append(f"project_{ss.start}_{ss.end}")
        elif isinstance(expr, Transform):
            parts.append(f"transform_{expr.matrix.ref_type}_{expr.matrix.name}")
        elif isinstance(expr, QueryMemory):
            parts.append(f"query_{expr.db.partition}_{expr.top_k}")
        elif isinstance(expr, GatherContext):
            parts.append(f"gather_{expr.top_k}_{int(expr.causal)}")

    return murmur3_hash("_".join(parts))


class L1StructuralCache:
    def __init__(self, max_entries: int = 1024):
        self.max_entries = max_entries
        self._cache: Dict[str, Any] = {}
        self._access_order: list[str] = []

    def get(self, program) -> Optional[Any]:
        key = ast_structure_hash(program)
        if key in self._cache:
            self._access_order.remove(key)
            self._access_order.append(key)
            return self._cache[key]
        return None

    def put(self, program, compiled_kernel: Any):
        key = ast_structure_hash(program)
        if key in self._cache:
            self._access_order.remove(key)
        elif len(self._cache) >= self.max_entries:
            oldest = self._access_order.pop(0)
            del self._cache[oldest]
        self._cache[key] = compiled_kernel
        self._access_order.append(key)

    def clear(self):
        self._cache.clear()
        self._access_order.clear()


class L2SemanticCache:
    def __init__(
        self,
        max_entries: int = 4096,
        n_bits: int = 256,
        epsilon: float = 0.05,
    ):
        self.max_entries = max_entries
        self.n_bits = n_bits
        self.epsilon = epsilon
        self._projections: Optional[torch.Tensor] = None
        self._cache: Dict[str, Tuple[Any, Any]] = {}
        self._access_order: list[str] = []

    def _init_projections(self, d_latent: int):
        if self._projections is None or self._projections.shape[1] != d_latent:
            self._projections = torch.randn(self.n_bits, d_latent)

    def get(self, z: torch.Tensor) -> Optional[Tuple[Any, Any]]:
        if z is None or self._projections is None:
            return None

        z_flat = z.detach().reshape(-1).to(torch.float32)
        self._init_projections(z_flat.shape[0])
        proj = self._projections.to(z_flat.device)
        bits = ((z_flat @ proj.T) > 0).int()
        key = "".join(str(b.item()) for b in bits)

        for cached_key, (cached_z, cached_value) in self._cache.items():
            if self._bits_distance(key, cached_key) <= self.n_bits * self.epsilon:
                self._access_order.remove(cached_key)
                self._access_order.append(cached_key)
                return cached_value

        return None

    def put(self, z: torch.Tensor, value: Tuple[Any, Any]):
        if z is None or self._projections is None:
            return

        z_flat = z.detach().reshape(-1).to(torch.float32)
        proj = self._projections.to(z_flat.device)
        bits = ((z_flat @ proj.T) > 0).int()
        key = "".join(str(b.item()) for b in bits)

        if key in self._cache:
            self._access_order.remove(key)
        elif len(self._cache) >= self.max_entries:
            oldest = self._access_order.pop(0)
            del self._cache[oldest]

        self._cache[key] = (z.detach().clone(), value)
        self._access_order.append(key)

    def clear(self):
        self._cache.clear()
        self._access_order.clear()

    @staticmethod
    def _bits_distance(a: str, b: str) -> int:
        return sum(c1 != c2 for c1, c2 in zip(a, b))
