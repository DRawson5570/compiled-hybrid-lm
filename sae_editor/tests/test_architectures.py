"""Wave 1: Architecture abstraction tests."""

from __future__ import annotations

import os
import tempfile

import pytest
import torch
from safetensors.torch import save_file

from sae_editor.architectures import (
    ArchitectureSpec,
    CMI_DEEPSEEK,
    DEEPSEEK_CUSTOM,
    GPT2,
    LLAMA3,
    MISTRAL,
    QWEN2,
)


class TestArchitectureSpec:
    @pytest.mark.parametrize("arch", [QWEN2, GPT2, LLAMA3, MISTRAL, DEEPSEEK_CUSTOM])
    def test_name_construction_layer_0(self, arch):
        assert isinstance(arch.mlp_down_name(0), str)
        assert isinstance(arch.mlp_up_name(0), str)

    @pytest.mark.parametrize("arch", [QWEN2, LLAMA3, MISTRAL, CMI_DEEPSEEK])
    def test_name_construction_layer_N(self, arch):
        assert "layers.5" in arch.mlp_down_name(5)
        assert "layers.5" in arch.mlp_up_name(5)

    def test_gpt2_uses_c_fc_c_proj(self):
        assert "c_fc.weight" in GPT2.mlp_down_name(0)
        assert "c_proj.weight" in GPT2.mlp_up_name(0)

    def test_qwen_uses_down_up_proj(self):
        assert "down_proj.weight" in QWEN2.mlp_down_name(0)
        assert "up_proj.weight" in QWEN2.mlp_up_name(0)

    def test_mlp_gate_only_for_gated(self):
        assert QWEN2.mlp_gate_name(0) is not None
        assert GPT2.mlp_gate_name(0) is None

    def test_attn_separate_has_qkv_names(self):
        assert QWEN2.attn_q_name(0) is not None
        assert QWEN2.attn_k_name(0) is not None
        assert QWEN2.attn_v_name(0) is not None
        assert QWEN2.attn_o_name(0) is not None

    def test_attn_fused_has_no_qkv(self):
        assert GPT2.attn_q_name(0) is None
        assert GPT2.attn_k_name(0) is None
        assert GPT2.attn_v_name(0) is None

    def test_all_tensor_names_includes_both_mlp_and_attn(self):
        names = QWEN2.all_tensor_names(0)
        assert any("mlp" in n for n in names)
        assert any("attn" in n for n in names)


class TestDetectFromKeys:
    def test_detect_qwen(self):
        keys = ["model.layers.0.mlp.down_proj.weight", "model.layers.1.self_attn.q_proj.weight"]
        arch = ArchitectureSpec.detect_from_keys(keys)
        assert arch.name == "qwen2"

    def test_detect_gpt2(self):
        keys = ["transformer.h.0.mlp.c_fc.weight", "transformer.h.1.attn.c_attn.weight"]
        arch = ArchitectureSpec.detect_from_keys(keys)
        assert arch.name == "gpt2"

    def test_detect_deepseek_custom(self):
        keys = ["layers.0.ffn1.weight", "layers.5.q_proj.weight"]
        arch = ArchitectureSpec.detect_from_keys(keys)
        assert arch.name == "deepseek-custom"

    def test_detect_empty_falls_back(self):
        arch = ArchitectureSpec.detect_from_keys(["unknown.tensor"])
        assert arch.name in ["qwen2", "gpt2"]


class TestDetectFromFile:
    def test_detect_gpt2_from_file(self, tiny_gpt2_model):
        fd, path = tempfile.mkstemp(suffix=".safetensors")
        os.close(fd)
        save_file({k: v.cpu().clone() for k, v in tiny_gpt2_model.state_dict().items()}, path)
        try:
            arch = ArchitectureSpec.detect(path)
            assert arch.name == "gpt2"
        finally:
            os.unlink(path)


class TestModelNameDetection:
    def test_from_model_name_qwen(self):
        arch = ArchitectureSpec.from_model_name("Qwen/Qwen2.5-1.5B")
        assert arch.name == "qwen2"

    def test_from_model_name_gpt2(self):
        arch = ArchitectureSpec.from_model_name("sshleifer/tiny-gpt2")
        assert arch.name == "gpt2"

    def test_from_model_name_llama(self):
        arch = ArchitectureSpec.from_model_name("meta-llama/Llama-3-8B")
        assert arch.name == "llama3"

    def test_from_model_name_mistral(self):
        arch = ArchitectureSpec.from_model_name("mistralai/Mistral-7B")
        assert arch.name == "mistral"


class TestSpliceWithArchSpec:
    def test_splice_mlp_with_qwen_arch(self, temp_safetensors):
        from sae_editor.splicer import SafetensorsSplicer

        original = torch.randn(8, 4)
        new = torch.randn(8, 4)

        with SafetensorsSplicer(temp_safetensors) as spl:
            spl.splice_mlp(layer=0, W_down=new, W_up=torch.randn(4, 8), arch=QWEN2)

        from safetensors import safe_open
        with safe_open(temp_safetensors, framework="pt") as f:
            loaded = f.get_tensor("model.layers.0.mlp.down_proj.weight")
        assert torch.allclose(loaded, new, atol=1e-4)

    def test_splice_mlp_backward_compat_string(self, temp_safetensors):
        from sae_editor.splicer import SafetensorsSplicer

        new = torch.randn(8, 4)
        with SafetensorsSplicer(temp_safetensors) as spl:
            spl.splice_mlp(layer=0, W_down=new, W_up=torch.randn(4, 8),
                          model_name="model.layers.{layer}.mlp")

        from safetensors import safe_open
        with safe_open(temp_safetensors, framework="pt") as f:
            loaded = f.get_tensor("model.layers.0.mlp.down_proj.weight")
        assert torch.allclose(loaded, new, atol=1e-4)
