from __future__ import annotations

import json
import mmap
import struct
from typing import Any, BinaryIO


class SafetensorsSplicer:
    """Phase IV: Binary Splicing Engine.

    Inline tensor replacement in safetensors files via memory mapping.
    Replaces tensor payloads without rewriting the entire file,
    provided shapes match.

    Based on the safetensors format:
      [8 bytes: header_len (little-endian u64)]
      [header_len bytes: JSON metadata header]
      [concatenated tensor payloads, offsets specified in header]
    """

    def __init__(self, path: str):
        self.path = path
        self._f: BinaryIO | None = None
        self._mm: mmap.mmap | None = None
        self._header: dict[str, Any] | None = None
        self._header_len: int = 0

    def open(self):
        if self._f is not None:
            return
        self._f = open(self.path, "r+b")
        self._mm = mmap.mmap(self._f.fileno(), 0)
        self._parse_header()

    def close(self):
        if self._mm is not None:
            self._mm.close()
            self._mm = None
        if self._f is not None:
            self._f.close()
            self._f = None

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def _parse_header(self):
        header_bytes = self._mm[0:8]
        if len(header_bytes) < 8:
            raise ValueError(f"File too short for safetensors header: {len(header_bytes)} bytes")
        self._header_len = struct.unpack("<Q", header_bytes)[0]
        header_json = self._mm[8 : 8 + self._header_len]
        self._header = json.loads(header_json.decode("utf-8"))

    @property
    def header(self) -> dict[str, Any]:
        if self._header is None:
            raise RuntimeError("Not opened. Use open() or context manager.")
        return self._header

    @property
    def tensor_names(self) -> list[str]:
        return list(self.header.keys())

    def get_tensor_info(self, name: str) -> dict[str, Any]:
        return self.header[name]

    def get_tensor_shape(self, name: str) -> list[int]:
        return self.header[name]["shape"]

    def get_tensor_dtype(self, name: str) -> str:
        return self.header[name]["dtype"]

    def get_tensor_offsets(self, name: str) -> list[int]:
        return self.header[name]["data_offsets"]

    def read_tensor(self, name: str) -> bytes:
        start, end = self.get_tensor_offsets(name)
        return bytes(self._mm[self._payload_base + start : self._payload_base + end])

    def splice_tensor(
        self,
        name: str,
        new_data: bytes,
        verify_shape: bool = True,
    ):
        """Inline-replace a tensor's payload.

        Args:
            name:        Tensor name in the safetensors header
            new_data:    Raw bytes of the replacement tensor
            verify_shape: Assert that the new data matches the original tensor size

        Raises:
            AssertionError if verify_shape=True and sizes don't match
            KeyError if tensor name not found in header
        """
        if self._mm is None:
            raise RuntimeError("Not opened. Use open() or context manager.")

        if name not in self._header:
            raise KeyError(f"Tensor '{name}' not found in safetensors header")

        tensor_meta = self._header[name]
        start, end = tensor_meta["data_offsets"]

        if verify_shape:
            expected_size = end - start
            actual_size = len(new_data)
            assert actual_size == expected_size, (
                f"Size mismatch for '{name}': "
                f"expected {expected_size} bytes, got {actual_size} bytes"
            )

        self._mm[self._payload_base + start : self._payload_base + end] = new_data
        self._mm.flush()

    def splice_mlp(
        self,
        layer: int,
        W_down: object,
        W_up: object,
        model_name: str = "model.layers.{layer}.mlp",
        arch=None,
    ):
        """Convenience: splice MLP down_proj and up_proj for a specific layer.

        Args:
            layer:      Layer index
            W_down:     New down_proj weight (torch.Tensor or numpy array)
            W_up:       New up_proj weight (torch.Tensor or numpy array)
            model_name: Format string for layer prefix (old API, falls back
                       to {prefix}.down_proj.weight / up_proj.weight).
            arch:       ArchitectureSpec (new API, uses spec's mlp_down/up names).
                       If provided, model_name is ignored.

        Uses the original tensor dtypes and shapes for validation.
        """
        if arch is not None:
            down_name = arch.mlp_down_name(layer)
            up_name = arch.mlp_up_name(layer)
        else:
            prefix = model_name.format(layer=layer)
            down_name = f"{prefix}.down_proj.weight"
            up_name = f"{prefix}.up_proj.weight"

        self._splice_tensor_from_array(down_name, W_down)
        self._splice_tensor_from_array(up_name, W_up)

    def _splice_tensor_from_array(self, name: str, array):
        """Convert a tensor/array to raw bytes in the target tensor's dtype and splice."""
        import torch
        import numpy as np

        original_dtype = self.get_tensor_dtype(name)
        original_shape = self.get_tensor_shape(name)

        if isinstance(array, torch.Tensor):
            array = array.detach().cpu()
            if list(array.shape) != original_shape:
                raise ValueError(
                    f"Shape mismatch for '{name}': "
                    f"expected {original_shape}, got {list(array.shape)}"
                )
            np_array = array.numpy()
        else:
            np_array = np.asarray(array)
            if list(np_array.shape) != original_shape:
                raise ValueError(
                    f"Shape mismatch for '{name}': "
                    f"expected {original_shape}, got {list(np_array.shape)}"
                )

        import warnings

        dtype_map = {
            "F32": "float32",
            "F16": "float16",
            "BF16": "bfloat16",
            "F64": "float64",
            "I32": "int32",
            "I64": "int64",
            "I16": "int16",
            "I8": "int8",
            "U8": "uint8",
            "BOOL": "bool",
        }
        target_dtype = dtype_map.get(original_dtype, "float32")

        if original_dtype in ("F16", "BF16") and np_array.dtype != np.dtype(target_dtype):
            warnings.warn(
                f"Converting {np_array.dtype} data to {target_dtype} for '{name}'. "
                "This may lose precision. Consider providing data already in the target dtype.",
                stacklevel=2,
            )

        try:
            np_array = np_array.astype(target_dtype)
        except TypeError:
            if target_dtype == "bfloat16":
                np_fp32 = np_array.astype("float32")
                np_array = np_fp32.view(np.uint16).astype(np.uint16)
                np_array = np_array.view(np.float16)

        new_data = np_array.tobytes()
        self.splice_tensor(name, new_data, verify_shape=True)

    @property
    def _payload_base(self) -> int:
        return 8 + self._header_len


def splice_tensor(
    safetensors_path: str,
    tensor_name: str,
    new_data: bytes,
) -> None:
    """Convenience function: splice one tensor in one shot.

    Opens the file, splices the tensor, closes.

    Args:
        safetensors_path: Path to .safetensors file
        tensor_name:      Name of tensor to replace
        new_data:         Raw bytes for replacement
    """
    with SafetensorsSplicer(safetensors_path) as spl:
        spl.splice_tensor(tensor_name, new_data)
