from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from hybrid.compiled_features import (
    CompiledFeatureTransformer,
    CompiledFeatureTransformerConfig,
    GPT2_COMPILED_FEATURE_DIM,
    GPT2CompiledChannelBuilder,
    CompiledFeatureRuntime,
    build_token_stat_features,
    build_token_stat_features_for_span,
    iter_compiled_feature_batches,
    iter_span_compiled_feature_batches,
)


def test_compiled_feature_transformer_shapes():
    cfg = CompiledFeatureTransformerConfig(
        vocab_size=97,
        feature_dim=10,
        d_model=32,
        n_layers=2,
        n_heads=4,
        d_ff=64,
        max_seq_len=16,
        dropout=0.0,
    )
    model = CompiledFeatureTransformer(cfg).eval()
    ids = torch.randint(0, cfg.vocab_size, (3, 12))
    features = torch.randn(3, 12, cfg.feature_dim)
    logits = model(ids, features)
    assert logits.shape == (3, 12, cfg.vocab_size)


def test_compiled_feature_transformer_uses_features():
    torch.manual_seed(0)
    cfg = CompiledFeatureTransformerConfig(
        vocab_size=31,
        feature_dim=5,
        d_model=16,
        n_layers=1,
        n_heads=4,
        d_ff=32,
        max_seq_len=8,
        dropout=0.0,
    )
    model = CompiledFeatureTransformer(cfg).eval()
    ids = torch.randint(0, cfg.vocab_size, (1, 8))
    a = torch.zeros(1, 8, cfg.feature_dim)
    b = a.clone()
    b[:, -1, 0] = 10.0
    diff = (model(ids, a)[:, -1] - model(ids, b)[:, -1]).abs().max().item()
    assert diff > 1e-6


def test_compiled_feature_transformer_no_future_feature_leak():
    torch.manual_seed(0)
    cfg = CompiledFeatureTransformerConfig(
        vocab_size=41,
        feature_dim=6,
        d_model=24,
        n_layers=2,
        n_heads=4,
        d_ff=48,
        max_seq_len=10,
        dropout=0.0,
    )
    model = CompiledFeatureTransformer(cfg).eval()
    ids = torch.randint(0, cfg.vocab_size, (1, 10))
    features = torch.randn(1, 10, cfg.feature_dim)
    out_a = model(ids, features)
    changed = features.clone()
    changed[:, -1] = torch.randn_like(changed[:, -1]) * 20.0
    out_b = model(ids, changed)
    earlier_diff = (out_a[:, :-1] - out_b[:, :-1]).abs().max().item()
    last_diff = (out_a[:, -1] - out_b[:, -1]).abs().max().item()
    assert earlier_diff < 1e-6
    assert last_diff > 1e-6


def test_gpt2_feature_adapter_is_causal_and_batches_align():
    ids = torch.tensor([5, 7, 5, 9, 5, 7, 11, 5, 50256, 5, 7, 13])
    features = build_token_stat_features(ids, vocab_size=50257, window=4)
    assert features.shape == (len(ids), 10)
    assert features[0, 2].item() == 0.0
    assert features[2, 2].item() > 0.0

    gen = torch.Generator().manual_seed(123)
    batcher = iter_compiled_feature_batches(ids, batch_size=2, seq_len=4, generator=gen)
    batch = next(batcher)
    assert batch.input_ids.shape == (2, 4)
    assert batch.target_ids.shape == (2, 4)
    assert batch.compiled_features.shape == (2, 4, 10)
    assert torch.equal(batch.input_ids[:, 1:], batch.target_ids[:, :-1])


def test_span_feature_builder_matches_full_prefix_with_enough_history():
    ids = torch.tensor([5, 7, 5, 9, 5, 7, 11, 5, 50256, 5, 7, 13])
    full = build_token_stat_features(ids, vocab_size=50257, window=4)
    span = build_token_stat_features_for_span(
        ids,
        start=4,
        length=5,
        history=10,
        vocab_size=50257,
        window=4,
    )
    assert torch.allclose(span, full[4:9])


def test_span_batcher_aligns_tokens_and_features():
    ids = torch.tensor([1, 2, 1, 3, 1, 2, 4, 1, 5, 9, 2, 6, 1])
    gen = torch.Generator().manual_seed(456)
    batcher = iter_span_compiled_feature_batches(
        ids,
        batch_size=3,
        seq_len=4,
        history=6,
        window=4,
        generator=gen,
    )
    batch = next(batcher)
    assert batch.input_ids.shape == (3, 4)
    assert batch.target_ids.shape == (3, 4)
    assert batch.compiled_features.shape == (3, 4, 10)
    assert torch.equal(batch.input_ids[:, 1:], batch.target_ids[:, :-1])


def test_compiled_feature_runtime_recomputes_generation_features():
    torch.manual_seed(0)
    cfg = CompiledFeatureTransformerConfig(
        vocab_size=19,
        feature_dim=10,
        d_model=16,
        n_layers=1,
        n_heads=4,
        d_ff=32,
        max_seq_len=8,
        dropout=0.0,
    )
    model = CompiledFeatureTransformer(cfg).eval()
    runtime = CompiledFeatureRuntime(model, history=8, window=4)
    ids = torch.tensor([[1, 2, 1, 3]])
    logits = runtime(ids)
    direct_features = build_token_stat_features_for_span(ids[0], start=0, length=4, history=8, window=4)
    direct = model(ids, direct_features.unsqueeze(0))
    assert torch.allclose(logits, direct)


def test_gpt2_compiled_channels_are_causal_and_span_aligned():
    train_ids = torch.tensor([1, 2, 3, 1, 2, 4, 1, 2, 3, 5, 1, 7, 3, 1])
    builder = GPT2CompiledChannelBuilder.from_ids(train_ids)
    full = builder.build_features(train_ids)
    span = builder.build_features_for_span(train_ids, start=4, length=6, history=20)
    assert full.shape == (train_ids.numel(), GPT2_COMPILED_FEATURE_DIM)
    assert torch.allclose(span, full[4:10])
    assert torch.isfinite(full).all()

    changed_future = train_ids.clone()
    changed_future[-1] = 42
    original_prefix = builder.build_features_for_span(train_ids, start=0, length=8, history=20)
    changed_prefix = builder.build_features_for_span(changed_future, start=0, length=8, history=20)
    assert torch.allclose(original_prefix, changed_prefix)


def test_compiled_feature_runtime_uses_compiled_ngram_builder():
    torch.manual_seed(0)
    ids = torch.tensor([1, 2, 3, 1, 2, 4, 1, 2])
    builder = GPT2CompiledChannelBuilder.from_ids(ids)
    cfg = CompiledFeatureTransformerConfig(
        vocab_size=50,
        feature_dim=GPT2_COMPILED_FEATURE_DIM,
        d_model=16,
        n_layers=1,
        n_heads=4,
        d_ff=32,
        max_seq_len=8,
        dropout=0.0,
    )
    model = CompiledFeatureTransformer(cfg).eval()
    runtime = CompiledFeatureRuntime(model, history=8, window=4, compiled_builder=builder)
    prompt = ids[:5].unsqueeze(0)
    logits = runtime(prompt)
    direct_features = builder.build_features_for_span(prompt[0], start=0, length=5, history=8)
    direct = model(prompt, direct_features.unsqueeze(0))
    assert torch.allclose(logits, direct)


def test_gpt2_compiled_channel_artifact_roundtrip(tmp_path):
    ids = torch.tensor([1, 2, 3, 1, 2, 4, 1, 2, 3, 5, 1, 7, 3, 1])
    builder = GPT2CompiledChannelBuilder.from_ids(ids)
    path = tmp_path / "compiled_channels.pt"
    builder.save(path)
    loaded = GPT2CompiledChannelBuilder.load(path)
    prompt = torch.tensor([1, 2, 3, 1, 2])
    assert loaded.total_tokens == builder.total_tokens
    assert loaded.channel_names == builder.channel_names
    assert torch.allclose(loaded.build_features(prompt), builder.build_features(prompt))


def test_gpt2_compiled_channel_summary_cache_preserves_features():
    ids = torch.tensor([1, 2, 3, 1, 2, 4, 1, 2, 3, 5, 1, 7, 3, 1])
    builder = GPT2CompiledChannelBuilder.from_ids(ids)
    first = builder.build_features_for_span(ids, start=2, length=8, history=20)
    assert builder._total_cache
    assert builder._entropy_norm_cache
    assert builder._max_logp_cache

    cached_total_keys = set(builder._total_cache)
    cached_entropy_keys = set(builder._entropy_norm_cache)
    cached_max_keys = set(builder._max_logp_cache)
    second = builder.build_features_for_span(ids, start=2, length=8, history=20)
    assert torch.allclose(second, first)
    assert cached_total_keys.issubset(builder._total_cache)
    assert cached_entropy_keys.issubset(builder._entropy_norm_cache)
    assert cached_max_keys.issubset(builder._max_logp_cache)


def test_compiled_feature_runtime_appends_cached_features():
    torch.manual_seed(0)
    ids = torch.tensor([1, 2, 3, 1, 2, 4, 1, 2])
    builder = GPT2CompiledChannelBuilder.from_ids(ids)
    cfg = CompiledFeatureTransformerConfig(
        vocab_size=50,
        feature_dim=GPT2_COMPILED_FEATURE_DIM,
        d_model=16,
        n_layers=1,
        n_heads=4,
        d_ff=32,
        max_seq_len=8,
        dropout=0.0,
    )
    model = CompiledFeatureTransformer(cfg).eval()
    runtime = CompiledFeatureRuntime(model, history=8, window=4, compiled_builder=builder)
    prompt = ids[:5].unsqueeze(0)
    extended = ids[:6].unsqueeze(0)
    _ = runtime(prompt)
    cached_prefix = runtime._cached_features.clone()
    logits = runtime(extended)
    expected_features = builder.build_features_for_span(extended[0], start=0, length=6, history=8)
    expected = model(extended, expected_features.unsqueeze(0))
    assert runtime._cached_features.shape == (6, GPT2_COMPILED_FEATURE_DIM)
    assert torch.allclose(runtime._cached_features[:-1, :-1], cached_prefix[:, :-1])
    assert torch.allclose(logits, expected)
