"""E2E confirmation suite: every NRTCS feature against Qwen2.5-0.5B."""

from __future__ import annotations

import os
import tempfile

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from sae_editor.architectures import ArchitectureSpec, QWEN2
from sae_editor.attention import AttentionExtractor, AttentionSplicer
from sae_editor.decompiler import NRTCSDecompiler
from sae_editor.pipeline import NRTCSPipeline
from sae_editor.preview import PreviewResult, MultiLayerPreviewResult
from sae_editor.recompiler import (
    build_dense_map,
    compact_features,
    decompact_features,
    orthogonal_projection,
)
from sae_editor.sae_training import SAERegistry, SAETrainingPipeline
from sae_editor.splicer import SafetensorsSplicer
from sae_editor.transfer import project_features

MODEL_NAME = "Qwen/Qwen2.5-0.5B"


@pytest.mark.gpu
class TestE2EConfirm:
    @pytest.fixture(scope="class")
    def qwen_model(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, trust_remote_code=True,
            torch_dtype=torch.float16, attn_implementation="eager",
        ).cuda()
        model.eval()
        yield model
        del model
        torch.cuda.empty_cache()

    @pytest.fixture(scope="class")
    def qwen_tokenizer(self):
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    @pytest.fixture(scope="class")
    def safetensors_copy(self, qwen_model):
        fd, path = tempfile.mkstemp(suffix=".safetensors")
        os.close(fd)
        state = {k: v.cpu().clone() for k, v in qwen_model.state_dict().items()}
        save_file(state, path)
        yield path
        os.unlink(path)

    @pytest.fixture(scope="class")
    def d_model(self, qwen_model):
        return qwen_model.config.hidden_size

    @pytest.fixture(scope="class")
    def trained_saes(self, qwen_model, qwen_tokenizer):
        pipeline = SAETrainingPipeline()
        texts = [
            "The capital of France is Paris.",
            "Machine learning is a field of artificial intelligence.",
            "Who are you? I am an AI assistant.",
            "Hello world, this is a test.",
        ] * 3

        saes = pipeline.train_all(
            model=qwen_model, tokenizer=qwen_tokenizer,
            texts=texts, layers=[0, 5],
            n_features=64, steps=100, lr=1e-3,
            batch_size=32, device="cuda",
        )
        yield saes
        for sae in saes.values():
            del sae
        torch.cuda.empty_cache()

    # Architecture detection
    def test_arch_detect_from_file(self, safetensors_copy):
        arch = ArchitectureSpec.detect(safetensors_copy)
        assert arch.name == "qwen2"
        assert arch.layer_prefix == "model.layers.{layer}"

    def test_arch_detect_from_model_name(self):
        arch = ArchitectureSpec.from_model_name(MODEL_NAME)
        assert arch.name == "qwen2"

    # Decompiler
    def test_extract_features(self, qwen_model, qwen_tokenizer, trained_saes):
        decompiler = NRTCSDecompiler(
            model=qwen_model, tokenizer=qwen_tokenizer,
            saes=trained_saes, threshold=0.0, device="cuda",
        )
        features = decompiler.extract_features(["Who are you?"], max_length=32, batch_size=1)
        assert 0 in features
        assert 5 in features
        for layer in [0, 5]:
            assert features[layer]["activations"].ndim == 3

    def test_path_attribution(self, qwen_model, qwen_tokenizer, trained_saes):
        decompiler = NRTCSDecompiler(
            model=qwen_model, tokenizer=qwen_tokenizer,
            saes=trained_saes, threshold=0.0, device="cuda",
        )
        attr = decompiler.path_attribution(
            text="Who are you?",
            upstream_layer=0, downstream_layer=5,
            upstream_features=[0, 1, 2], downstream_feature=0,
        )
        assert "attributions" in attr
        assert "upstream_acts" in attr
        assert "downstream_acts" in attr

    # Recompiler at Qwen scale
    def test_build_dense_map_qwen_scale(self, d_model):
        N = 4
        keys = torch.randn(N, d_model)
        values = torch.randn(N, d_model)
        W_down, W_up = build_dense_map(keys, values, eps=1e-3)
        recon = keys @ W_down @ W_up
        err = (recon - values).norm(dim=-1).max().item()
        assert err < 1e-2, f"Qwen-scale reconstruction error: {err:.6f}"

    def test_crosstalk_at_qwen_scale(self, d_model):
        N = 4
        m = 20
        keys = torch.randn(N, d_model)
        values = torch.randn(N, d_model)
        U = torch.randn(d_model, m)
        W_down, _ = build_dense_map(keys, values, eps=1e-3)
        W_proj = orthogonal_projection(W_down, U, eps=1e-3)
        leak = (U.T @ W_proj).abs().max().item()
        assert leak < 1e-2, f"Crosstalk leak at d={d_model}, m={m}: {leak:.6f}"

    # Splicer + reload
    def test_splice_identity(self, safetensors_copy):
        mlp_name = "model.layers.0.mlp.down_proj.weight"
        with safe_open(safetensors_copy, framework="pt") as f:
            original = f.get_tensor(mlp_name).clone()

        with SafetensorsSplicer(safetensors_copy) as spl:
            spl.splice_tensor(mlp_name, original.numpy().tobytes())

        with safe_open(safetensors_copy, framework="pt") as f:
            reloaded = f.get_tensor(mlp_name)
        assert torch.equal(original, reloaded)

    def test_splice_changes(self, safetensors_copy):
        mlp_name = "model.layers.1.mlp.down_proj.weight"
        with safe_open(safetensors_copy, framework="pt") as f:
            original = f.get_tensor(mlp_name).clone()

        scrambled = torch.randn_like(original) * 10.0
        with SafetensorsSplicer(safetensors_copy) as spl:
            spl.splice_tensor(mlp_name, scrambled.numpy().tobytes())

        with safe_open(safetensors_copy, framework="pt") as f:
            reloaded = f.get_tensor(mlp_name)
        assert torch.equal(scrambled, reloaded)
        assert not torch.equal(original, reloaded)

    def test_splice_reload_forward_pass(self, safetensors_copy, qwen_tokenizer, qwen_model):
        from transformers import AutoModelForCausalLM

        reloaded = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, trust_remote_code=True,
            torch_dtype=torch.float16, attn_implementation="eager",
        )
        reloaded.load_state_dict(load_file(safetensors_copy), strict=False)
        reloaded.eval()

        inputs = qwen_tokenizer("Hello world", return_tensors="pt")
        with torch.no_grad():
            out = reloaded(**inputs)
        assert out.logits.ndim == 3
        del reloaded
        torch.cuda.empty_cache()

    # Preview
    def test_preview_single_at_qwen_scale(self, qwen_model, qwen_tokenizer, d_model):
        pipeline = NRTCSPipeline(eps=1e-3)
        result = pipeline.preview_single(
            layer=0, keys=torch.randn(2, d_model), values=torch.randn(2, d_model),
            model=qwen_model, tokenizer=qwen_tokenizer,
            prompts=["Hello world."], strength=1.0,
        )
        assert isinstance(result, PreviewResult)
        assert not result.cosine_shift != result.cosine_shift  # nan check
        assert result.offset_l2 >= 0

    def test_preview_compare_at_qwen(self, qwen_model, qwen_tokenizer, d_model):
        pipeline = NRTCSPipeline(eps=1e-3)
        edits = {0: {"keys": torch.randn(2, d_model), "values": torch.randn(2, d_model)}}
        results = pipeline.compare(edits, qwen_model, qwen_tokenizer,
                                   ["Hello."], strengths=[0.1, 0.5, 1.0])
        assert len(results) == 3

    def test_preview_no_nan_at_qwen(self, qwen_model, qwen_tokenizer, d_model):
        pipeline = NRTCSPipeline(eps=1e-3)
        result = pipeline.preview_single(
            layer=0, keys=torch.randn(3, d_model), values=torch.randn(3, d_model),
            model=qwen_model, tokenizer=qwen_tokenizer,
            prompts=["Test prompt."], strength=2.0,
        )
        assert result.cosine_shift == result.cosine_shift

    # Attention
    def test_attention_extract_qwen(self, qwen_model):
        extractor = AttentionExtractor(arch=QWEN2)
        weights, meta = extractor.extract(qwen_model, layer=0)
        assert "W_q" in weights
        assert weights["W_q"].shape[0] == qwen_model.config.hidden_size
        assert meta["n_heads"] == qwen_model.config.num_attention_heads
        assert meta["has_gqa"] is True

    def test_attention_splice_qwen(self, qwen_model, safetensors_copy):
        extractor = AttentionExtractor(arch=QWEN2)
        weights, _ = extractor.extract(qwen_model, layer=0)

        w_q_original = weights["W_q"].clone()
        splicer = AttentionSplicer(arch=QWEN2)
        splicer.splice(safetensors_copy, layer=0, weights=weights)

        with safe_open(safetensors_copy, framework="pt") as f:
            loaded = f.get_tensor("model.layers.0.self_attn.q_proj.weight")
        assert torch.equal(loaded.cpu(), w_q_original.cpu())

    # SAE Registry
    def test_sae_registry_round_trip(self, trained_saes):
        with tempfile.TemporaryDirectory() as tmpdir:
            SAERegistry.save(trained_saes, tmpdir)
            d_model = list(trained_saes.values())[0].encoder.in_features
            n_features = list(trained_saes.values())[0].encoder.out_features
            loaded = SAERegistry.load(tmpdir, d_model=d_model, n_features=n_features)
            assert set(loaded.keys()) == set(trained_saes.keys())

    # Circuit Editor
    def test_circuit_editor_qwen(self, qwen_model, qwen_tokenizer, trained_saes):
        from sae_editor.circuit_editor import CircuitEditor

        decompiler = NRTCSDecompiler(
            model=qwen_model, tokenizer=qwen_tokenizer,
            saes=trained_saes, threshold=0.0, device="cuda",
        )
        editor = CircuitEditor(decompiler)
        features = editor.find_feature_activating_on(["Who are you?"], top_k=3)
        assert isinstance(features, dict)
        assert len(features) > 0

    # DSL round-trip
    def test_dsl_round_trip_qwen(self, d_model):
        from sae_editor.dsl.nrtcs_parser import parse_nrtcs, serialize_nrtcs

        edits = {
            3: {"keys": torch.randn(2, d_model), "values": torch.randn(2, d_model)}
        }
        text = serialize_nrtcs(edits)
        parsed = parse_nrtcs(text)
        assert 3 in parsed

    # Transfer
    def test_transfer_project_qwen(self, d_model):
        features = torch.randn(5, d_model)
        projected = project_features(features, d_model, 1536, seed=42)
        assert projected.shape == (5, 1536)

    # Compaction at Qwen scale
    def test_compaction_qwen(self, d_model):
        W_down = torch.randn(d_model, 32)
        W_comp, basis = compact_features(W_down, n_components=8)
        recovered = decompact_features(W_comp, basis)
        cosine = torch.nn.functional.cosine_similarity(
            recovered.flatten(), W_down.flatten(), dim=0
        )
        assert cosine > 0.3, f"Compaction recovery cosine at d={d_model}: {cosine:.4f}"
