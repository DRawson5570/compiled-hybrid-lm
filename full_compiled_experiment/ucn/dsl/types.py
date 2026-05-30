from __future__ import annotations

import torch


class Vector:
    def __init__(self, dim: int, data: torch.Tensor | None = None):
        self.dim = dim
        self.data = data

    def __repr__(self):
        return f"Vector[{self.dim}]"


class Subspace:
    def __init__(self, size: int, parent_dim: int, indices: list[int] | None = None):
        self.size = size
        self.parent_dim = parent_dim
        self.indices = indices or list(range(size))

    def __repr__(self):
        return f"Subspace[{self.size}/{self.parent_dim}]"


class Scalar:
    def __init__(self, value: float = 0.0):
        self.value = value

    def __repr__(self):
        return f"Scalar({self.value})"


class Matrix:
    def __init__(self, rows: int, cols: int, weight: torch.Tensor | None = None):
        self.rows = rows
        self.cols = cols
        self.weight = weight

    def __repr__(self):
        return f"Matrix[{self.rows}x{self.cols}]"
