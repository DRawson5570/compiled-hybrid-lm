"""Deterministic multi-corpus token mixer.

The mixer is token-level and deliberately independent of any specific dataset
library.  Callers can feed WikiText, code, math, instruction, or web token IDs
as tensors and receive reproducible fixed-length chunks for training.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Sequence

import torch


@dataclass(frozen=True)
class CorpusStream:
    name: str
    token_ids: torch.Tensor
    weight: int = 1

    def validate(self) -> None:
        if not self.name:
            raise ValueError("corpus name cannot be empty")
        if self.token_ids.ndim != 1:
            raise ValueError(f"{self.name}: token_ids must be 1D")
        if self.token_ids.numel() == 0:
            raise ValueError(f"{self.name}: token_ids cannot be empty")
        if self.weight <= 0:
            raise ValueError(f"{self.name}: weight must be positive")


@dataclass(frozen=True)
class MixedChunk:
    input_ids: torch.Tensor
    target_ids: torch.Tensor
    source_names: tuple[str, ...]


def build_weighted_schedule(streams: Sequence[CorpusStream], *, seed: int = 42) -> list[int]:
    """Build a deterministic weighted stream schedule."""
    if not streams:
        raise ValueError("at least one stream is required")
    for stream in streams:
        stream.validate()
    schedule = []
    for idx, stream in enumerate(streams):
        schedule.extend([idx] * stream.weight)
    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(schedule), generator=generator).tolist()
    return [schedule[i] for i in perm]


def mixed_token_stream(streams: Sequence[CorpusStream], *, seed: int = 42) -> Iterator[tuple[int, str]]:
    """Yield ``(token_id, source_name)`` forever from weighted corpus streams."""
    schedule = build_weighted_schedule(streams, seed=seed)
    positions = [0 for _ in streams]
    schedule_pos = 0

    while True:
        stream_idx = schedule[schedule_pos]
        stream = streams[stream_idx]
        token_pos = positions[stream_idx] % stream.token_ids.numel()
        yield int(stream.token_ids[token_pos].item()), stream.name
        positions[stream_idx] += 1
        schedule_pos = (schedule_pos + 1) % len(schedule)


def iter_mixed_chunks(
    streams: Sequence[CorpusStream],
    *,
    seq_len: int,
    seed: int = 42,
    device: torch.device | str | None = None,
) -> Iterator[MixedChunk]:
    """Yield fixed-length causal LM chunks from mixed corpora."""
    if seq_len <= 0:
        raise ValueError("seq_len must be positive")
    target_device = torch.device(device) if device is not None else torch.device("cpu")
    token_iter = mixed_token_stream(streams, seed=seed)

    while True:
        tokens = []
        names = []
        for _ in range(seq_len + 1):
            token, name = next(token_iter)
            tokens.append(token)
            names.append(name)
        ids = torch.tensor(tokens, dtype=torch.long, device=target_device)
        yield MixedChunk(
            input_ids=ids[:-1],
            target_ids=ids[1:],
            source_names=tuple(names[:-1]),
        )


def materialize_mixed_tokens(
    streams: Sequence[CorpusStream],
    *,
    n_tokens: int,
    seed: int = 42,
) -> torch.Tensor:
    """Materialize a deterministic mixed token prefix."""
    if n_tokens < 0:
        raise ValueError("n_tokens must be non-negative")
    token_iter = mixed_token_stream(streams, seed=seed)
    return torch.tensor([next(token_iter)[0] for _ in range(n_tokens)], dtype=torch.long)
