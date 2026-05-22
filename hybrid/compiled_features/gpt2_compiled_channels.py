"""GPT-2 vocabulary compiled-channel feature builder.

This module is the GPT-2-tokenized replacement for the weak token-stat adapter.
It compiles corpus counts once, then emits compact causal feature rows aligned
with token spans. The rows are target-independent summaries of compiled channel
state at each input position, so they can be used during training and generation.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from math import log
from typing import DefaultDict

import torch

GPT2_COMPILED_CHANNEL_NAMES = [
    "uni_logp_token",
    "bi_logp_token",
    "tri_logp_token",
    "skip2_logp_token",
    "skip3_logp_token",
    "uni_entropy_norm",
    "bi_entropy_norm",
    "tri_entropy_norm",
    "skip2_entropy_norm",
    "skip3_entropy_norm",
    "uni_max_logp",
    "bi_max_logp",
    "tri_max_logp",
    "skip2_max_logp",
    "skip3_max_logp",
    "bi_context_seen",
    "tri_context_seen",
    "skip2_context_seen",
    "skip3_context_seen",
    "local_recency",
    "position_norm",
]
GPT2_COMPILED_FEATURE_DIM = len(GPT2_COMPILED_CHANNEL_NAMES)


@dataclass(frozen=True)
class GPT2CompiledChannelConfig:
    vocab_size: int = 50257
    alpha: float = 0.1
    max_train_tokens: int = 0
    recency_window: int = 128


class GPT2CompiledChannelBuilder:
    """Compiled count channels for GPT-2 token IDs.

    Channels are unigram, adjacent bigram/trigram, and two skip transitions.
    Counts are compiled from the training split and then reused for val/test or
    generation. Feature rows are causal relative to the emitted span: row ``t``
    only uses tokens before or at that input token plus the compiled artifact.
    """

    feature_dim = GPT2_COMPILED_FEATURE_DIM
    channel_names = tuple(GPT2_COMPILED_CHANNEL_NAMES)

    def __init__(self, cfg: GPT2CompiledChannelConfig | None = None):
        self.cfg = cfg or GPT2CompiledChannelConfig()
        if self.cfg.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if self.cfg.alpha <= 0:
            raise ValueError("alpha must be positive")
        self.unigram: Counter[int] = Counter()
        self.bigram: DefaultDict[int, Counter[int]] = defaultdict(Counter)
        self.trigram: DefaultDict[tuple[int, int], Counter[int]] = defaultdict(Counter)
        self.skip2: DefaultDict[int, Counter[int]] = defaultdict(Counter)
        self.skip3: DefaultDict[int, Counter[int]] = defaultdict(Counter)
        self.total_tokens = 0

    @classmethod
    def from_ids(cls, ids: torch.Tensor, cfg: GPT2CompiledChannelConfig | None = None) -> "GPT2CompiledChannelBuilder":
        builder = cls(cfg)
        builder.fit(ids)
        return builder

    def fit(self, ids: torch.Tensor) -> "GPT2CompiledChannelBuilder":
        if ids.ndim != 1:
            raise ValueError(f"ids must be 1D, got {tuple(ids.shape)}")
        ids_list = ids.long().cpu().tolist()
        if self.cfg.max_train_tokens > 0:
            ids_list = ids_list[: self.cfg.max_train_tokens]
        for pos, token in enumerate(ids_list):
            token = int(token)
            self.unigram[token] += 1
            if pos >= 1:
                self.bigram[int(ids_list[pos - 1])][token] += 1
            if pos >= 2:
                self.trigram[(int(ids_list[pos - 2]), int(ids_list[pos - 1]))][token] += 1
                self.skip2[int(ids_list[pos - 2])][token] += 1
            if pos >= 3:
                self.skip3[int(ids_list[pos - 3])][token] += 1
        self.total_tokens = len(ids_list)
        return self

    def build_features(self, ids: torch.Tensor) -> torch.Tensor:
        return self.build_features_for_span(ids, start=0, length=ids.numel(), history=ids.numel())

    def build_features_for_span(
        self,
        ids: torch.Tensor,
        *,
        start: int,
        length: int,
        history: int = 512,
    ) -> torch.Tensor:
        if ids.ndim != 1:
            raise ValueError(f"ids must be 1D, got {tuple(ids.shape)}")
        if start < 0 or length < 0 or start + length > ids.numel():
            raise ValueError("span is outside ids")
        if history < 0:
            raise ValueError("history must be non-negative")

        ids = ids.long().cpu()
        prefix_start = max(0, start - history)
        local_counts: Counter[int] = Counter(int(tok) for tok in ids[prefix_start:start].tolist())
        rows = []
        denom_pos = max(ids.numel() - 1, 1)
        for pos in range(start, start + length):
            token = int(ids[pos].item())
            prev1 = int(ids[pos - 1].item()) if pos >= 1 else None
            prev2 = int(ids[pos - 2].item()) if pos >= 2 else None
            prev3 = int(ids[pos - 3].item()) if pos >= 3 else None

            bi_counts = self.bigram.get(prev1, Counter()) if prev1 is not None else Counter()
            tri_counts = self.trigram.get((prev2, prev1), Counter()) if prev1 is not None and prev2 is not None else Counter()
            skip2_counts = self.skip2.get(prev2, Counter()) if prev2 is not None else Counter()
            skip3_counts = self.skip3.get(prev3, Counter()) if prev3 is not None else Counter()

            rows.append(torch.tensor([
                self._logp(self.unigram, token),
                self._logp(bi_counts, token),
                self._logp(tri_counts, token),
                self._logp(skip2_counts, token),
                self._logp(skip3_counts, token),
                self._entropy_norm(self.unigram),
                self._entropy_norm(bi_counts),
                self._entropy_norm(tri_counts),
                self._entropy_norm(skip2_counts),
                self._entropy_norm(skip3_counts),
                self._max_logp(self.unigram),
                self._max_logp(bi_counts),
                self._max_logp(tri_counts),
                self._max_logp(skip2_counts),
                self._max_logp(skip3_counts),
                1.0 if bi_counts else 0.0,
                1.0 if tri_counts else 0.0,
                1.0 if skip2_counts else 0.0,
                1.0 if skip3_counts else 0.0,
                min(local_counts.get(token, 0), self.cfg.recency_window) / max(self.cfg.recency_window, 1),
                pos / denom_pos,
            ], dtype=torch.float32))

            local_counts[token] += 1
            if pos - self.cfg.recency_window >= prefix_start:
                leaving = int(ids[pos - self.cfg.recency_window].item())
                local_counts[leaving] -= 1
                if local_counts[leaving] <= 0:
                    del local_counts[leaving]

        return torch.stack(rows, dim=0) if rows else torch.empty(0, self.feature_dim, dtype=torch.float32)

    def _logp(self, counts: Counter[int], token: int) -> float:
        total = sum(counts.values())
        return log((counts.get(token, 0) + self.cfg.alpha) / (total + self.cfg.alpha * self.cfg.vocab_size))

    def _max_logp(self, counts: Counter[int]) -> float:
        total = sum(counts.values())
        max_count = max(counts.values()) if counts else 0
        return log((max_count + self.cfg.alpha) / (total + self.cfg.alpha * self.cfg.vocab_size))

    def _entropy_norm(self, counts: Counter[int]) -> float:
        total = sum(counts.values())
        if total <= 0:
            return 1.0
        denom = total + self.cfg.alpha * self.cfg.vocab_size
        entropy = 0.0
        for count in counts.values():
            prob = (count + self.cfg.alpha) / denom
            entropy -= prob * log(prob)
        unseen_mass = (self.cfg.vocab_size - len(counts)) * self.cfg.alpha / denom
        if unseen_mass > 0.0:
            unseen_prob = self.cfg.alpha / denom
            entropy -= unseen_mass * log(unseen_prob)
        return entropy / log(self.cfg.vocab_size)
