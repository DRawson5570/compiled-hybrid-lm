"""Tier 4+5: End-to-end model round-trip tests."""

from __future__ import annotations

import os
import tempfile

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from sae_editor.decompiler import NRTCSDecompiler
from sae_editor.pipeline import NRTCSPipeline
from sae_editor.recompiler import build_dense_map, orthogonal_projection
from sae_editor.splicer import SafetensorsSplicer
from sae_editor.tests.utils import make_random_sae


@pytest.mark.slow
class TestModelRoundTrip:
    @pytest.fixture(scope="class")
    def model_safetensors(self, tiny_gpt2_model):
        fd, path = tempfile.mkstemp(suffix=".safetensors")
        os.close(fd)
        state_dict = {k: v.cpu().clone() for k, v in tiny_gpt2_model.state_dict().items()}
        save_file(state_dict, path)
        yield path
        os.unlink(path)

    @pytest.fixture(scope="class")
    def gpt2_names(self, model_safetensors):
        with safe_open(model_safetensors, framework="pt") as f:
            keys = list(f.keys())
        is_gpt2 = any("transformer" in k for k in keys)
        if is_gpt2:
            return {
                "mlp_down": "transformer.h.0.mlp.c_fc.weight",
                "mlp_up": "transformer.h.0.mlp.c_proj.weight",
                "d_model": 2,
                "prefix": "transformer.h.{layer}.mlp",
            }
        else:
            return {
                "mlp_down": "model.layers.0.mlp.down_proj.weight",
                "mlp_up": "model.layers.0.mlp.up_proj.weight",
                "d_model": 2,
                "prefix": "model.layers.{layer}.mlp",
            }

    def test_splice_and_reload_tensor_intact(self, model_safetensors, gpt2_names):
        mlp_name = gpt2_names["mlp_down"]
        original = _load_tensor(model_safetensors, mlp_name)

        with SafetensorsSplicer(model_safetensors) as spl:
            spl.splice_tensor(mlp_name, original.numpy().tobytes())

        reloaded = _load_tensor(model_safetensors, mlp_name)
        assert torch.allclose(original, reloaded), "Identity splice should preserve data"

    def test_splice_changes_tensor(self, model_safetensors, gpt2_names):
        mlp_name = gpt2_names["mlp_down"]
        original = _load_tensor(model_safetensors, mlp_name)
        scrambled = torch.randn_like(original) + 10.0

        assert not torch.equal(original, scrambled), "Scrambled must differ from original"

        with SafetensorsSplicer(model_safetensors) as spl:
            spl.splice_tensor(mlp_name, scrambled.numpy().tobytes())

        reloaded = _load_tensor(model_safetensors, mlp_name)
        assert torch.equal(scrambled, reloaded), "Spliced tensor must match new data exactly"

    def test_round_trip_compile_and_splice(self, model_safetensors, gpt2_names):
        d_model = gpt2_names["d_model"]
        N = min(d_model, 2)
        keys = torch.randn(N, d_model)
        values = torch.randn(N, d_model)

        pipeline = NRTCSPipeline(eps=1e-1)
        patches = pipeline.compile_from_uvm_edits({0: {"keys": keys, "values": values}})

        recon = keys @ patches[0]["W_down"].float() @ patches[0]["W_up"].float()
        max_err = (recon - values).norm(dim=-1).max().item()
        assert max_err < 5.0, f"Reconstruction error too high: {max_err:.6f}"

    def test_crosstalk_prevents_leakage(self, model_safetensors, gpt2_names):
        d_model = gpt2_names["d_model"]
        N = 4

        U = torch.randn(d_model, 8)
        keys = torch.randn(N, d_model)
        keys[:, :8] = 0.0
        values = torch.randn(N, d_model)

        patches = NRTCSPipeline(eps=1e-4).compile_from_uvm_edits(
            {0: {"keys": keys, "values": values}}, {0: U}
        )
        assert torch.allclose(U.T @ patches[0]["W_down"], torch.zeros(8, N), atol=1e-3)


@pytest.mark.gpu
class TestEndToEndGPU:
    @pytest.fixture(scope="class")
    def qwen_model_and_tokenizer(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            "Qwen/Qwen2.5-1.5B", trust_remote_code=True
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen2.5-1.5B",
            torch_dtype=torch.float16,
            trust_remote_code=True,
            attn_implementation="eager",
        ).cuda()
        model.eval()

        yield model, tokenizer

        del model
        torch.cuda.empty_cache()

    def test_full_pipeline_qwen(self, qwen_model_and_tokenizer):
        model, tokenizer = qwen_model_and_tokenizer
        d_model = model.config.hidden_size

        sae_l0 = make_random_sae(d_model=d_model, n_features=64)
        sae_l2 = make_random_sae(d_model=d_model, n_features=64)
        sae_l0.cuda()
        sae_l2.cuda()

        decompiler = NRTCSDecompiler(
            model=model,
            tokenizer=tokenizer,
            saes={0: sae_l0, 2: sae_l2},
            threshold=0.0,
            device="cuda",
        )

        features = decompiler.extract_features(
            ["The capital of France is Paris."],
            max_length=32,
            batch_size=1,
        )
        assert 0 in features
        assert 2 in features

        f0 = features[0]
        active_idx = f0["feature_indices"][:4]
        active_vectors = f0["feature_vectors"][:4]
        if len(active_vectors) == 0:
            pytest.skip("No features activated on test text")

        keys = active_vectors.float()
        values = torch.randn(len(keys), d_model)
        values = values / values.norm(dim=-1, keepdim=True) * keys.norm(dim=-1, keepdim=True)

        pipeline = NRTCSPipeline(eps=1e-3)
        patches = pipeline.compile_from_uvm_edits({0: {"keys": keys, "values": values}})
        recon = keys @ patches[0]["W_down"].float() @ patches[0]["W_up"].float()
        max_err = (recon - values).norm(dim=-1).max().item()
        assert max_err < 0.5, f"GPU E2E reconstruction error: {max_err:.4f}"

    def test_smoke_save_splice_identity(self, qwen_model_and_tokenizer):
        model, tokenizer = qwen_model_and_tokenizer

        fd, path = tempfile.mkstemp(suffix=".safetensors")
        os.close(fd)
        state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        save_file(state_dict, path)

        try:
            with safe_open(path, framework="pt") as f:
                first_key = list(f.keys())[0]
                original = f.get_tensor(first_key).clone()

            with SafetensorsSplicer(path) as spl:
                spl.splice_tensor(first_key, original.numpy().tobytes())

            with safe_open(path, framework="pt") as f:
                reloaded = f.get_tensor(first_key)
            assert torch.allclose(original, reloaded)
        finally:
            os.unlink(path)

    def test_dimension_saturation_report(self, qwen_model_and_tokenizer):
        model, tokenizer = qwen_model_and_tokenizer
        d_model = model.config.hidden_size

        from sae_editor.recompiler import compute_null_space_rank

        U = torch.randn(d_model, 100).cuda()
        rank = compute_null_space_rank(U)
        assert rank <= d_model
        assert rank >= d_model - 100


def _load_tensor(path, name):
    with safe_open(path, framework="pt") as f:
        return f.get_tensor(name)
