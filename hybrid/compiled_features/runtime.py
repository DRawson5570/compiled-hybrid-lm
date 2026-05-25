"""Runtime feature adapter for compiled-feature transformer generation."""
from __future__ import annotations

import torch

from .feature_transformer import CompiledFeatureTransformer
from .gpt2_compiled_channels import GPT2CompiledChannelBuilder
from .gpt2_feature_adapter import build_token_stat_features_for_span


class CompiledFeatureRuntime:
    """Callable wrapper that supplies causal compiled features during decode."""

    def __init__(
        self,
        model: CompiledFeatureTransformer,
        *,
        history: int,
        window: int,
        compiled_builder: GPT2CompiledChannelBuilder | None = None,
    ):
        self.model = model
        self.history = history
        self.window = window
        self.compiled_builder = compiled_builder
        self._cached_input: torch.Tensor | None = None
        self._cached_features: torch.Tensor | None = None

    def __call__(self, input_ids: torch.Tensor) -> torch.Tensor:
        if input_ids.ndim != 2:
            raise ValueError(f"input_ids must have shape (B, T), got {tuple(input_ids.shape)}")
        device = input_ids.device
        cpu_ids = input_ids.detach().cpu()
        if self.compiled_builder is not None and cpu_ids.shape[0] == 1:
            features = self._compiled_features_cached(cpu_ids[0]).unsqueeze(0).to(device)
            return self.model(input_ids, features)

        rows = []
        for row in cpu_ids:
            if self.compiled_builder is None:
                features = build_token_stat_features_for_span(
                    row,
                    start=0,
                    length=row.numel(),
                    history=self.history,
                    window=self.window,
                )
            else:
                features = self.compiled_builder.build_features_for_span(
                    row,
                    start=0,
                    length=row.numel(),
                    history=self.history,
                )
            rows.append(features)
        features = torch.stack(rows, dim=0).to(device)
        return self.model(input_ids, features)

    def _compiled_features_cached(self, row: torch.Tensor) -> torch.Tensor:
        assert self.compiled_builder is not None
        can_append = (
            self._cached_input is not None
            and self._cached_features is not None
            and row.numel() >= self._cached_input.numel()
            and torch.equal(row[: self._cached_input.numel()], self._cached_input)
        )
        if can_append:
            old_len = self._cached_input.numel()
            if row.numel() > old_len:
                new_features = self.compiled_builder.build_features_for_span(
                    row,
                    start=old_len,
                    length=row.numel() - old_len,
                    history=self.history,
                )
                self._cached_features = torch.cat([self._cached_features, new_features], dim=0)
                self._cached_input = row.clone()
            self._refresh_position_column(row.numel())
            return self._cached_features

        self._cached_input = row.clone()
        self._cached_features = self.compiled_builder.build_features_for_span(
            row,
            start=0,
            length=row.numel(),
            history=self.history,
        )
        return self._cached_features

    def _refresh_position_column(self, seq_len: int) -> None:
        if self._cached_features is None:
            return
        denom = max(seq_len - 1, 1)
        self._cached_features[:, -1] = torch.arange(seq_len, dtype=torch.float32) / denom