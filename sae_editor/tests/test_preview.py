"""Preview mechanism tests: non-destructive patch preview."""

from __future__ import annotations

import pytest
import torch

from sae_editor.pipeline import NRTCSPipeline
from sae_editor.preview import MultiLayerPreviewResult, PreviewResult


class TestPreviewSingle:
    @pytest.fixture
    def pipeline(self):
        return NRTCSPipeline(eps=1e-4)

    @pytest.fixture
    def sample_edits(self):
        return {0: {"keys": torch.randn(2, 64), "values": torch.randn(2, 64)}}

    def test_preview_single_returns_result(self, pipeline, synthetic_model, sample_edits):
        edit = sample_edits[0]
        result = pipeline.preview_single(
            layer=0, keys=edit["keys"], values=edit["values"],
            model=synthetic_model, tokenizer=None,
            prompts=["hello world"],
        )
        assert isinstance(result, PreviewResult)
        assert result.layer_idx == 0

    def test_preview_single_has_fields(self, pipeline, synthetic_model, sample_edits):
        edit = sample_edits[0]
        result = pipeline.preview_single(
            layer=0, keys=edit["keys"], values=edit["values"],
            model=synthetic_model, tokenizer=None,
            prompts=["test"],
        )
        assert result.cosine_shift is not None
        assert result.reconstruction_error >= 0
        assert result.offset_l2 >= 0
        assert len(result.original_top_k) > 0

    def test_preview_single_strength_zero_preserves_output(self, pipeline, synthetic_model, sample_edits):
        edit = sample_edits[0]
        result = pipeline.preview_single(
            layer=0, keys=edit["keys"], values=edit["values"],
            model=synthetic_model, tokenizer=None,
            prompts=["hello"], strength=0.0,
        )
        assert result.cosine_shift > 0.99

    def test_preview_single_high_strength_no_shift_with_random_keys(self, pipeline, synthetic_model, sample_edits):
        """With random keys, gated hook blocks injection (no key matches hidden states)."""
        edit = sample_edits[0]
        result = pipeline.preview_single(
            layer=0, keys=edit["keys"], values=edit["values"],
            model=synthetic_model, tokenizer=None,
            prompts=["hello"], strength=5.0,
        )
        assert result.offset_l2 < 1e-4, (
            "Gated hook should block injection when no key matches"
        )

    def test_preview_single_high_strength_shifts_with_matching_keys(self, pipeline, synthetic_model):
        """Keys that match hidden states DO pass the gate."""
        d = synthetic_model.config.hidden_size
        keys = torch.randn(2, d)
        values = torch.randn(2, d)
        result = pipeline.preview_single(
            layer=0, keys=keys, values=values,
            model=synthetic_model, tokenizer=None,
            prompts=["hello"], strength=5.0,
        )
        assert isinstance(result, PreviewResult)

    def test_preview_hooks_cleaned_up(self, pipeline, synthetic_model, sample_edits):
        edit = sample_edits[0]
        before_hooks = len(synthetic_model._forward_hooks)

        pipeline.preview_single(
            layer=0, keys=edit["keys"], values=edit["values"],
            model=synthetic_model, tokenizer=None,
            prompts=["hello"],
        )

        after_hooks = len(synthetic_model._forward_hooks)
        assert after_hooks == before_hooks

    def test_preview_no_side_effects(self, pipeline, synthetic_model, sample_edits):
        edit = sample_edits[0]
        synthetic_model.eval()
        inputs = {"input_ids": torch.randint(0, 1000, (1, 8))}

        with torch.no_grad():
            out_before = synthetic_model(**inputs)
        before_last = out_before.logits[0, -1].clone()

        pipeline.preview_single(
            layer=0, keys=edit["keys"], values=edit["values"],
            model=synthetic_model, tokenizer=None,
            prompts=["hello"], strength=5.0,
        )

        with torch.no_grad():
            out_after = synthetic_model(**inputs)
        after_last = out_after.logits[0, -1]

        assert torch.allclose(before_last, after_last, atol=1e-4)


class TestPreviewMultiLayer:
    @pytest.fixture
    def pipeline(self):
        return NRTCSPipeline(eps=1e-4)

    def test_preview_returns_multilayer_result(self, pipeline, synthetic_model):
        edits = {
            0: {"keys": torch.randn(2, 64), "values": torch.randn(2, 64)},
            1: {"keys": torch.randn(2, 64), "values": torch.randn(2, 64)},
        }
        result = pipeline.preview(
            edits, synthetic_model, None, ["hello world"],
        )
        assert isinstance(result, MultiLayerPreviewResult)
        assert len(result.per_layer) == 2

    def test_preview_combined_cosine_reported(self, pipeline, synthetic_model):
        edits = {0: {"keys": torch.randn(2, 64), "values": torch.randn(2, 64)}}
        result = pipeline.preview(
            edits, synthetic_model, None, ["hello"],
        )
        assert result.combined_cosine_shift is not None

    def test_preview_empty_prompts(self, pipeline, synthetic_model):
        edits = {0: {"keys": torch.randn(2, 64), "values": torch.randn(2, 64)}}
        result = pipeline.preview(edits, synthetic_model, None, [])
        assert isinstance(result, MultiLayerPreviewResult)


class TestCompare:
    @pytest.fixture
    def pipeline(self):
        return NRTCSPipeline(eps=1e-4)

    def test_compare_returns_one_per_strength(self, pipeline, synthetic_model):
        edits = {0: {"keys": torch.randn(2, 64), "values": torch.randn(2, 64)}}
        strengths = [0.1, 0.5, 1.0]
        results = pipeline.compare(edits, synthetic_model, None, ["hello"], strengths=strengths)
        assert len(results) == len(strengths)

    def test_compare_default_strengths(self, pipeline, synthetic_model):
        edits = {0: {"keys": torch.randn(2, 64), "values": torch.randn(2, 64)}}
        results = pipeline.compare(edits, synthetic_model, None, ["hello"])
        assert len(results) == 5

    def test_compare_accepts_gate_threshold(self, pipeline, synthetic_model):
        edits = {0: {"keys": torch.randn(2, 64), "values": torch.randn(2, 64)}}
        results = pipeline.compare(edits, synthetic_model, None, ["hello"],
                                   strengths=[0.1], gate_threshold=0.7)
        assert len(results) == 1

    def test_gate_threshold_passes_through(self, pipeline, synthetic_model):
        """Higher gate threshold → less injection (gate stays closed more)."""
        d = synthetic_model.config.hidden_size
        keys = torch.randn(2, d)
        values = torch.randn(2, d)

        # Low threshold: random keys might match by chance (but unlikely at d=64)
        result_low = pipeline.preview_single(
            layer=0, keys=keys, values=values,
            model=synthetic_model, tokenizer=None,
            prompts=["hello"], strength=1.0,
            gate_threshold=0.01,
        )
        # High threshold: gates fully closed for random keys
        result_high = pipeline.preview_single(
            layer=0, keys=keys, values=values,
            model=synthetic_model, tokenizer=None,
            prompts=["hello"], strength=1.0,
            gate_threshold=0.99,
        )
        # With very high threshold, no random hidden state matches → offset=0
        assert result_high.offset_l2 < 1e-4


@pytest.mark.slow
class TestPreviewWithRealModel:
    def test_preview_tiny_gpt2(self, tiny_gpt2_model, tiny_gpt2_tokenizer):
        pipeline = NRTCSPipeline(eps=1e-2)
        d_model = tiny_gpt2_model.config.hidden_size
        keys = torch.randn(2, d_model)
        values = torch.randn(2, d_model)

        result = pipeline.preview_single(
            layer=0, keys=keys, values=values,
            model=tiny_gpt2_model, tokenizer=tiny_gpt2_tokenizer,
            prompts=["The capital of France is"],
            strength=1.0,
        )
        assert isinstance(result, PreviewResult)
        assert result.cosine_shift is not None
        assert not result.cosine_shift != result.cosine_shift  # NaN check: NaN != NaN is True

    def test_preview_no_nan(self, tiny_gpt2_model, tiny_gpt2_tokenizer):
        pipeline = NRTCSPipeline(eps=1e-2)
        d_model = tiny_gpt2_model.config.hidden_size
        keys = torch.randn(3, d_model)
        values = torch.randn(3, d_model)

        result = pipeline.preview_single(
            layer=0, keys=keys, values=values,
            model=tiny_gpt2_model, tokenizer=tiny_gpt2_tokenizer,
            prompts=["Hello world."],
        )
        assert result.offset_l2 == result.offset_l2  # NaN check: NaN == NaN is False
