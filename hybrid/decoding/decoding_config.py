"""Deterministic autoregressive decoding controls."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import torch
import torch.nn.functional as F


class LogitModel(Protocol):
    def __call__(self, input_ids: torch.Tensor, *args, **kwargs) -> torch.Tensor: ...


@dataclass(frozen=True)
class DecodingConfig:
    max_new_tokens: int = 32
    temperature: float = 0.0
    top_k: int | None = None
    seed: int = 42
    eos_token_id: int | None = None
    deterministic_algorithms: bool = True

    def validate(self) -> None:
        if self.max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative")
        if self.temperature < 0:
            raise ValueError("temperature must be non-negative")
        if self.top_k is not None and self.top_k <= 0:
            raise ValueError("top_k must be positive when set")


def set_global_determinism(seed: int, *, deterministic_algorithms: bool = True) -> torch.Generator:
    """Set Python-independent torch/numpy determinism and return a generator."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic_algorithms, warn_only=True)
    return torch.Generator().manual_seed(seed)


def _sample_from_logits(logits: torch.Tensor, cfg: DecodingConfig, generator: torch.Generator) -> torch.Tensor:
    if cfg.temperature == 0.0:
        return logits.argmax(dim=-1)

    scaled = logits / cfg.temperature
    if cfg.top_k is not None and cfg.top_k < scaled.shape[-1]:
        top_vals, top_idx = scaled.topk(cfg.top_k, dim=-1)
        masked = torch.full_like(scaled, -torch.inf)
        scaled = masked.scatter(dim=-1, index=top_idx, src=top_vals)
    probs = F.softmax(scaled, dim=-1)
    return torch.multinomial(probs, num_samples=1, generator=generator).squeeze(-1)


@torch.no_grad()
def deterministic_generate(
    model: LogitModel,
    input_ids: torch.Tensor,
    cfg: DecodingConfig,
    *model_args,
    **model_kwargs,
) -> torch.Tensor:
    """Generate tokens reproducibly for a model returning ``(B, T, V)`` logits."""
    cfg.validate()
    if input_ids.ndim != 2:
        raise ValueError(f"input_ids must have shape (B, T), got {tuple(input_ids.shape)}")
    generator = set_global_determinism(cfg.seed, deterministic_algorithms=cfg.deterministic_algorithms)
    generated = input_ids.clone()

    for _ in range(cfg.max_new_tokens):
        logits = model(generated, *model_args, **model_kwargs)
        if logits.ndim != 3:
            raise ValueError(f"model must return (B, T, V) logits, got {tuple(logits.shape)}")
        next_token = _sample_from_logits(logits[:, -1], cfg, generator).to(generated.device)
        generated = torch.cat([generated, next_token[:, None]], dim=1)
        if cfg.eos_token_id is not None and bool((next_token == cfg.eos_token_id).all()):
            break
    return generated
