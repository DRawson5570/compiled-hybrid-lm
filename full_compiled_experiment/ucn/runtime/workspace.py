from __future__ import annotations

from typing import Dict, Optional

import torch


class TensorWorkspace:
    def __init__(
        self,
        max_tokens: int = 512,
        d_model: int = 1536,
        device: str = "cuda",
        dtype: torch.dtype = torch.float32,
        max_registers: int = 64,
    ):
        self.max_tokens = max_tokens
        self.d_model = d_model
        self.device = device
        self.dtype = dtype
        self.max_registers = max_registers
        self._tensors: Dict[str, torch.Tensor] = {}
        self._liveness: Dict[str, int] = {}

    def allocate(
        self,
        name: str,
        shape: Optional[tuple[int, ...]] = None,
    ) -> torch.Tensor:
        if shape is None:
            shape = (self.d_model,)

        if len(self._tensors) >= self.max_registers:
            self._evict_dead()

        tensor = torch.zeros(shape, device=self.device, dtype=self.dtype)
        self._tensors[name] = tensor
        self._liveness[name] = 0
        return tensor

    def get(self, name: str) -> torch.Tensor:
        if name not in self._tensors:
            raise KeyError(f"Tensor '{name}' not found in workspace")
        self._liveness[name] = 0
        return self._tensors[name]

    def set(self, name: str, tensor: torch.Tensor):
        self._tensors[name] = tensor.to(device=self.device, dtype=self.dtype)
        self._liveness[name] = 0

    def free(self, name: str):
        if name in self._tensors:
            del self._tensors[name]
            del self._liveness[name]

    def _evict_dead(self):
        to_evict = []
        for name in self._tensors:
            self._liveness[name] += 1
            if self._liveness[name] > 3:
                to_evict.append(name)

        for name in to_evict:
            if name in self._tensors:
                del self._tensors[name]
                del self._liveness[name]

    def clear(self):
        self._tensors.clear()
        self._liveness.clear()
