from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import torch


@dataclass
class ArchitectureSpec:
    """Complete tensor name and shape map for one model architecture."""

    name: str

    layer_prefix: str
    layer_access_path: str

    mlp_down_suffix: str
    mlp_up_suffix: str
    mlp_gate_suffix: str | None = None

    attn_q_suffix: str | None = None
    attn_k_suffix: str | None = None
    attn_v_suffix: str | None = None
    attn_o_suffix: str | None = None

    mlp_type: Literal["simple", "gated"] = "simple"
    attn_type: Literal["separate", "fused"] = "separate"

    has_gqa: bool = False
    has_rope: bool = False

    def mlp_down_name(self, layer: int) -> str:
        return f"{self.layer_prefix.format(layer=layer)}.{self.mlp_down_suffix}"

    def mlp_up_name(self, layer: int) -> str:
        return f"{self.layer_prefix.format(layer=layer)}.{self.mlp_up_suffix}"

    def mlp_gate_name(self, layer: int) -> str | None:
        if self.mlp_gate_suffix is None:
            return None
        return f"{self.layer_prefix.format(layer=layer)}.{self.mlp_gate_suffix}"

    def mlp_tensor_names(self, layer: int) -> list[str]:
        names = [self.mlp_down_name(layer), self.mlp_up_name(layer)]
        if self.mlp_gate_suffix is not None:
            names.append(self.mlp_gate_name(layer))
        return names

    def attn_q_name(self, layer: int) -> str | None:
        if self.attn_q_suffix is None:
            return None
        return f"{self.layer_prefix.format(layer=layer)}.{self.attn_q_suffix}"

    def attn_k_name(self, layer: int) -> str | None:
        if self.attn_k_suffix is None:
            return None
        return f"{self.layer_prefix.format(layer=layer)}.{self.attn_k_suffix}"

    def attn_v_name(self, layer: int) -> str | None:
        if self.attn_v_suffix is None:
            return None
        return f"{self.layer_prefix.format(layer=layer)}.{self.attn_v_suffix}"

    def attn_o_name(self, layer: int) -> str | None:
        if self.attn_o_suffix is None:
            return None
        return f"{self.layer_prefix.format(layer=layer)}.{self.attn_o_suffix}"

    def attn_tensor_names(self, layer: int) -> list[str]:
        names = []
        for fn in [self.attn_q_name, self.attn_k_name, self.attn_v_name, self.attn_o_name]:
            n = fn(layer)
            if n is not None:
                names.append(n)
        return names

    def all_tensor_names(self, layer: int) -> list[str]:
        return self.mlp_tensor_names(layer) + self.attn_tensor_names(layer)

    @classmethod
    def detect_from_keys(cls, keys: list[str]) -> ArchitectureSpec:
        has_transformer = any("transformer.h." in k for k in keys)
        if has_transformer:
            return GPT2

        has_separate_attn = any("self_attn.q_proj" in k or "self_attn.k_proj" in k for k in keys)
        has_mlp_layers = any("model.layers." in k for k in keys)
        if has_separate_attn or has_mlp_layers:
            return QWEN2

        has_bare_layers = any(k.startswith("layers.") and k.endswith(".weight") and "model." not in k for k in keys)
        if has_bare_layers:
            return DEEPSEEK_CUSTOM

        return QWEN2

    @classmethod
    def detect(cls, safetensors_path: str) -> ArchitectureSpec:
        from safetensors import safe_open
        with safe_open(safetensors_path, framework="pt") as f:
            keys = list(f.keys())
        return cls.detect_from_keys(keys)

    @classmethod
    def from_model_name(cls, name: str) -> ArchitectureSpec:
        name_lower = name.lower()
        if "qwen" in name_lower:
            return QWEN2
        if "gpt" in name_lower:
            return GPT2
        if "deepseek" in name_lower:
            return CMI_DEEPSEEK
        if "llama" in name_lower:
            return LLAMA3
        if "mistral" in name_lower:
            return MISTRAL
        return QWEN2


QWEN2 = ArchitectureSpec(
    name="qwen2",
    layer_prefix="model.layers.{layer}",
    layer_access_path="model.layers",
    mlp_down_suffix="mlp.down_proj.weight",
    mlp_up_suffix="mlp.up_proj.weight",
    mlp_gate_suffix="mlp.gate_proj.weight",
    attn_q_suffix="self_attn.q_proj.weight",
    attn_k_suffix="self_attn.k_proj.weight",
    attn_v_suffix="self_attn.v_proj.weight",
    attn_o_suffix="self_attn.o_proj.weight",
    mlp_type="gated",
    attn_type="separate",
    has_gqa=True,
    has_rope=True,
)

GPT2 = ArchitectureSpec(
    name="gpt2",
    layer_prefix="transformer.h.{layer}",
    layer_access_path="transformer.h",
    mlp_down_suffix="mlp.c_fc.weight",
    mlp_up_suffix="mlp.c_proj.weight",
    mlp_gate_suffix=None,
    attn_q_suffix=None,
    attn_k_suffix=None,
    attn_v_suffix=None,
    attn_o_suffix="attn.c_proj.weight",
    mlp_type="simple",
    attn_type="fused",
    has_gqa=False,
    has_rope=False,
)

LLAMA3 = ArchitectureSpec(
    name="llama3",
    layer_prefix="model.layers.{layer}",
    layer_access_path="model.layers",
    mlp_down_suffix="mlp.down_proj.weight",
    mlp_up_suffix="mlp.up_proj.weight",
    mlp_gate_suffix="mlp.gate_proj.weight",
    attn_q_suffix="self_attn.q_proj.weight",
    attn_k_suffix="self_attn.k_proj.weight",
    attn_v_suffix="self_attn.v_proj.weight",
    attn_o_suffix="self_attn.o_proj.weight",
    mlp_type="gated",
    attn_type="separate",
    has_gqa=True,
    has_rope=True,
)

MISTRAL = ArchitectureSpec(
    name="mistral",
    layer_prefix="model.layers.{layer}",
    layer_access_path="model.layers",
    mlp_down_suffix="mlp.down_proj.weight",
    mlp_up_suffix="mlp.up_proj.weight",
    mlp_gate_suffix="mlp.gate_proj.weight",
    attn_q_suffix="self_attn.q_proj.weight",
    attn_k_suffix="self_attn.k_proj.weight",
    attn_v_suffix="self_attn.v_proj.weight",
    attn_o_suffix="self_attn.o_proj.weight",
    mlp_type="gated",
    attn_type="separate",
    has_gqa=True,
    has_rope=True,
)

DEEPSEEK_CUSTOM = ArchitectureSpec(
    name="deepseek-custom",
    layer_prefix="layers.{layer}",
    layer_access_path="layers",
    mlp_down_suffix="ffn2.weight",
    mlp_up_suffix="ffn1.weight",
    mlp_gate_suffix=None,
    attn_q_suffix="q_proj.weight",
    attn_k_suffix="k_proj.weight",
    attn_v_suffix="v_proj.weight",
    attn_o_suffix="o_proj.weight",
    mlp_type="simple",
    attn_type="separate",
    has_gqa=False,
    has_rope=False,
)

CMI_DEEPSEEK = ArchitectureSpec(
    name="cmi-deepseek",
    layer_prefix="model.layers.{layer}",
    layer_access_path="model.layers",
    mlp_down_suffix="mlp.down_proj.weight",
    mlp_up_suffix="mlp.up_proj.weight",
    mlp_gate_suffix="mlp.gate_proj.weight",
    attn_q_suffix="self_attn.q_proj.weight",
    attn_k_suffix="self_attn.k_proj.weight",
    attn_v_suffix="self_attn.v_proj.weight",
    attn_o_suffix="self_attn.o_proj.weight",
    mlp_type="gated",
    attn_type="separate",
    has_gqa=True,
    has_rope=True,
)
