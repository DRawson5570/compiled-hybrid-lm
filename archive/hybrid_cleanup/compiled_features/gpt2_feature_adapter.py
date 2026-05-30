"""GPT-2-tokenized compiled-feature adapter interfaces.

The real 21-channel compiler is currently BPE-8000.  This module gives the
GPT-2 path a stable, causal feature interface that training code can depend on
while the full compiled stack is ported to V=50257.  The provided token-stat
features are intentionally weak baselines, not a substitute for real channels.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterator

import torch


@dataclass(frozen=True)
class CompiledFeatureBatch:
    input_ids: torch.Tensor
    target_ids: torch.Tensor
    compiled_features: torch.Tensor


FeatureBuilder = Callable[[torch.Tensor], torch.Tensor]


def _token_shape_flags(ids: torch.Tensor) -> torch.Tensor:
    """Cheap tokenizer-agnostic token shape flags from integer IDs."""
    ids_f = ids.float()
    return torch.stack(
        [
            (ids == 0).float(),
            (ids == 50256).float(),
            (ids < 256).float(),
            ((ids >= 256) & (ids < 2048)).float(),
            (ids_f.remainder(2) == 0).float(),
        ],
        dim=-1,
    )


def build_token_stat_features(ids: torch.Tensor, vocab_size: int = 50257, window: int = 128) -> torch.Tensor:
    """Build causal GPT-2-compatible token-stat features.

    For each position ``t``, features depend only on ``ids[:t+1]``.  Returned
    shape is ``(T, 10)``:
    ``token_id_norm, position_norm, count_so_far_norm, window_count_norm,
    gap_norm`` plus five token-shape flags.
    """
    if ids.ndim != 1:
        raise ValueError(f"ids must be 1D, got {tuple(ids.shape)}")
    if vocab_size <= 0:
        raise ValueError("vocab_size must be positive")
    if window <= 0:
        raise ValueError("window must be positive")

    ids = ids.long().cpu()
    total = ids.numel()
    counts: dict[int, int] = {}
    recent_positions: dict[int, list[int]] = {}
    rows = []

    denom_vocab = max(vocab_size - 1, 1)
    denom_pos = max(total - 1, 1)
    for pos, token in enumerate(ids.tolist()):
        prior_count = counts.get(token, 0)
        positions = recent_positions.get(token, [])
        last_pos = positions[-1] if positions else None
        gap = window if last_pos is None else min(window, pos - last_pos)
        window_start = max(0, pos - window + 1)
        window_count = sum(1 for p in positions if p >= window_start)

        base = torch.tensor(
            [
                token / denom_vocab,
                pos / denom_pos,
                prior_count / max(pos, 1),
                window_count / window,
                gap / window,
            ],
            dtype=torch.float32,
        )
        shape = _token_shape_flags(torch.tensor([token], dtype=torch.long))[0]
        rows.append(torch.cat([base, shape], dim=0))

        counts[token] = prior_count + 1
        positions.append(pos)
        if len(positions) > window:
            del positions[:-window]
        recent_positions[token] = positions

    return torch.stack(rows, dim=0) if rows else torch.empty(0, 10, dtype=torch.float32)


def build_token_stat_features_for_span(
    ids: torch.Tensor,
    *,
    start: int,
    length: int,
    history: int = 512,
    vocab_size: int = 50257,
    window: int = 128,
) -> torch.Tensor:
    """Build causal token-stat features for a span with bounded prior context.

    The returned rows correspond to ``ids[start:start + length]``. Statistics
    are warmed with up to ``history`` tokens before ``start`` and never inspect
    tokens after the row being emitted. This is the sampling-friendly training
    path; `build_token_stat_features` remains the exact full-prefix builder.
    """
    if ids.ndim != 1:
        raise ValueError(f"ids must be 1D, got {tuple(ids.shape)}")
    if start < 0 or length < 0 or start + length > ids.numel():
        raise ValueError("span is outside ids")
    if history < 0:
        raise ValueError("history must be non-negative")

    ids = ids.long().cpu()
    prefix_start = max(0, start - history)
    counts: dict[int, int] = {}
    recent_positions: dict[int, list[int]] = {}

    for pos in range(prefix_start, start):
        token = int(ids[pos].item())
        counts[token] = counts.get(token, 0) + 1
        positions = recent_positions.get(token, [])
        positions.append(pos)
        if len(positions) > window:
            del positions[:-window]
        recent_positions[token] = positions

    rows = []
    denom_vocab = max(vocab_size - 1, 1)
    denom_pos = max(ids.numel() - 1, 1)
    for pos in range(start, start + length):
        token = int(ids[pos].item())
        prior_count = counts.get(token, 0)
        positions = recent_positions.get(token, [])
        last_pos = positions[-1] if positions else None
        gap = window if last_pos is None else min(window, pos - last_pos)
        window_start = max(0, pos - window + 1)
        window_count = sum(1 for prior_pos in positions if prior_pos >= window_start)

        base = torch.tensor(
            [
                token / denom_vocab,
                pos / denom_pos,
                prior_count / max(pos, 1),
                window_count / window,
                gap / window,
            ],
            dtype=torch.float32,
        )
        shape = _token_shape_flags(torch.tensor([token], dtype=torch.long))[0]
        rows.append(torch.cat([base, shape], dim=0))

        counts[token] = prior_count + 1
        positions.append(pos)
        if len(positions) > window:
            del positions[:-window]
        recent_positions[token] = positions

    return torch.stack(rows, dim=0) if rows else torch.empty(0, 10, dtype=torch.float32)


def iter_span_compiled_feature_batches(
    ids: torch.Tensor,
    *,
    batch_size: int,
    seq_len: int,
    history: int = 512,
    vocab_size: int = 50257,
    window: int = 128,
    device: torch.device | str | None = None,
    generator: torch.Generator | None = None,
) -> Iterator[CompiledFeatureBatch]:
    """Yield random LM batches with bounded-history causal feature rows.

    This avoids materializing feature vectors for the full corpus before
    training. It is still causal: each feature row is computed from a bounded
    prefix ending at that row.
    """
    if ids.ndim != 1:
        raise ValueError(f"ids must be 1D, got {tuple(ids.shape)}")
    if batch_size <= 0 or seq_len <= 0:
        raise ValueError("batch_size and seq_len must be positive")
    if ids.numel() < seq_len + 1:
        raise ValueError("ids must contain at least seq_len + 1 tokens")

    ids = ids.long().cpu()
    target_device = torch.device(device) if device is not None else ids.device
    max_start = ids.numel() - seq_len - 1
    offsets = torch.arange(seq_len + 1)
    while True:
        starts = torch.randint(0, max_start + 1, (batch_size,), generator=generator)
        token_idx = starts.unsqueeze(1) + offsets.unsqueeze(0)
        spans = ids[token_idx]
        feature_rows = [
            build_token_stat_features_for_span(
                ids,
                start=int(start.item()),
                length=seq_len,
                history=history,
                vocab_size=vocab_size,
                window=window,
            )
            for start in starts
        ]
        yield CompiledFeatureBatch(
            input_ids=spans[:, :-1].to(target_device),
            target_ids=spans[:, 1:].to(target_device),
            compiled_features=torch.stack(feature_rows, dim=0).to(target_device),
        )


def iter_compiled_feature_batches(
    ids: torch.Tensor,
    *,
    batch_size: int,
    seq_len: int,
    feature_builder: FeatureBuilder = build_token_stat_features,
    device: torch.device | str | None = None,
    generator: torch.Generator | None = None,
) -> Iterator[CompiledFeatureBatch]:
    """Yield random causal LM batches with aligned compiled features."""
    if ids.ndim != 1:
        raise ValueError(f"ids must be 1D, got {tuple(ids.shape)}")
    if batch_size <= 0 or seq_len <= 0:
        raise ValueError("batch_size and seq_len must be positive")
    if ids.numel() < seq_len + 1:
        raise ValueError("ids must contain at least seq_len + 1 tokens")

    ids = ids.long().cpu()
    all_features = feature_builder(ids)
    if all_features.shape[0] != ids.numel():
        raise ValueError("feature_builder must return one feature row per token")

    target_device = torch.device(device) if device is not None else ids.device
    max_start = ids.numel() - seq_len - 1
    offsets = torch.arange(seq_len + 1)
    while True:
        starts = torch.randint(0, max_start + 1, (batch_size,), generator=generator)
        token_idx = starts.unsqueeze(1) + offsets.unsqueeze(0)
        spans = ids[token_idx]
        feature_idx = token_idx[:, :-1]
        yield CompiledFeatureBatch(
            input_ids=spans[:, :-1].to(target_device),
            target_ids=spans[:, 1:].to(target_device),
            compiled_features=all_features[feature_idx].to(target_device),
        )
