"""Wave 2: Attention support tests."""

from __future__ import annotations

import os
import tempfile

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from sae_editor.architectures import GPT2, QWEN2
from sae_editor.attention import AttentionExtractor, AttentionSplicer


@pytest.mark.slow
class TestAttentionExtractorQwen:
    @pytest.fixture(scope="class")
    def qwen_dummy(self):
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen2.5-1.5B",
            torch_dtype=torch.float32,
            trust_remote_code=True,
            attn_implementation="eager",
        )
        model.eval()
        yield model
        del model

    def test_extract_qwen_attention_shapes(self, qwen_dummy):
        extractor = AttentionExtractor(arch=QWEN2)
        weights, meta = extractor.extract(qwen_dummy, layer=0)
        assert "W_q" in weights
        assert "W_k" in weights
        assert "W_v" in weights
        assert "W_o" in weights
        assert meta["n_heads"] == qwen_dummy.config.num_attention_heads
        assert meta["has_gqa"] is True

    def test_extract_qwen_metadata(self, qwen_dummy):
        extractor = AttentionExtractor(arch=QWEN2)
        _, meta = extractor.extract(qwen_dummy, layer=0)
        assert meta["n_heads"] == 12
        assert meta["n_kv_heads"] == 2
        assert meta["head_dim"] == 128

    def test_splice_attention_qwen_identity(self, qwen_dummy):
        extractor = AttentionExtractor(arch=QWEN2)
        weights, meta = extractor.extract(qwen_dummy, layer=0)

        fd, path = tempfile.mkstemp(suffix=".safetensors")
        os.close(fd)
        save_file({k: v.cpu().clone() for k, v in qwen_dummy.state_dict().items()}, path)
        try:
            splicer = AttentionSplicer(arch=QWEN2)
            splicer.splice(path, layer=0, weights=weights)

            with safe_open(path, framework="pt") as f:
                loaded_q = f.get_tensor("model.layers.0.self_attn.q_proj.weight")
                assert torch.equal(loaded_q, weights["W_q"])
        finally:
            os.unlink(path)

    def test_splice_attention_qwen_changes(self, qwen_dummy):
        extractor = AttentionExtractor(arch=QWEN2)
        weights, meta = extractor.extract(qwen_dummy, layer=0)

        scrambled = {k: torch.randn_like(v) for k, v in weights.items()}

        fd, path = tempfile.mkstemp(suffix=".safetensors")
        os.close(fd)
        save_file({k: v.cpu().clone() for k, v in qwen_dummy.state_dict().items()}, path)
        try:
            splicer = AttentionSplicer(arch=QWEN2)
            splicer.splice(path, layer=0, weights=scrambled)

            with safe_open(path, framework="pt") as f:
                loaded_q = f.get_tensor("model.layers.0.self_attn.q_proj.weight")
                assert not torch.equal(loaded_q, weights["W_q"])
                assert torch.equal(loaded_q, scrambled["W_q"])
        finally:
            os.unlink(path)

    def test_attention_transplant(self, qwen_dummy):
        fd, path = tempfile.mkstemp(suffix=".safetensors")
        os.close(fd)
        save_file({k: v.cpu().clone() for k, v in qwen_dummy.state_dict().items()}, path)
        try:
            with safe_open(path, framework="pt") as f:
                original_l0 = f.get_tensor("model.layers.0.self_attn.q_proj.weight").clone()
                original_l2 = f.get_tensor("model.layers.2.self_attn.q_proj.weight").clone()

            splicer = AttentionSplicer(arch=QWEN2)
            splicer.transplant(path, source_layer=0, target_layer=2)

            with safe_open(path, framework="pt") as f:
                l0_after = f.get_tensor("model.layers.0.self_attn.q_proj.weight")
                l2_after = f.get_tensor("model.layers.2.self_attn.q_proj.weight")

            assert torch.equal(l0_after, original_l0), "Source should be unchanged"
            assert torch.equal(l2_after, original_l0), "Target should match source"
        finally:
            os.unlink(path)


@pytest.mark.slow
class TestAttentionExtractorGPT2:
    @pytest.fixture(scope="class")
    def gpt2_dummy(self):
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            "gpt2",
            torch_dtype=torch.float32,
            attn_implementation="eager",
        )
        model.eval()
        yield model
        del model

    def test_extract_gpt2_fused_attention_splits(self, gpt2_dummy):
        extractor = AttentionExtractor(arch=GPT2)
        weights, meta = extractor.extract(gpt2_dummy, layer=0)

        assert "W_q" in weights
        assert "W_k" in weights
        assert "W_v" in weights
        assert "W_o" in weights
        d_model = gpt2_dummy.config.hidden_size
        assert weights["W_q"].shape[0] == d_model

    def test_splice_gpt2_attention_identity(self, gpt2_dummy):
        extractor = AttentionExtractor(arch=GPT2)
        weights, meta = extractor.extract(gpt2_dummy, layer=0)

        fd, path = tempfile.mkstemp(suffix=".safetensors")
        os.close(fd)
        save_file({k: v.cpu().clone() for k, v in gpt2_dummy.state_dict().items()}, path)
        try:
            splicer = AttentionSplicer(arch=GPT2)
            splicer.splice(path, layer=0, weights=weights)

            with safe_open(path, framework="pt") as f:
                loaded = f.get_tensor("transformer.h.0.attn.c_attn.weight")
                catted = torch.cat([weights["W_q"], weights["W_k"], weights["W_v"]], dim=0)
                assert torch.equal(loaded, catted)
        finally:
            os.unlink(path)
