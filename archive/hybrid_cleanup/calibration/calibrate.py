"""Calibration metrics for model logits and probabilities."""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class TemperatureSearchResult:
    temperature: float
    nll: float


def apply_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    if temperature <= 0 or not math.isfinite(float(temperature)):
        raise ValueError("temperature must be a finite positive value")
    return logits / float(temperature)


def expected_calibration_error(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    n_bins: int = 15,
) -> torch.Tensor:
    """Compute top-label Expected Calibration Error."""
    if n_bins <= 0:
        raise ValueError("n_bins must be positive")
    if logits.ndim != 2:
        raise ValueError(f"logits must have shape (N, V), got {tuple(logits.shape)}")
    targets = targets.reshape(-1).to(logits.device)
    if targets.numel() != logits.shape[0]:
        raise ValueError("targets length must match logits batch dimension")

    probs = F.softmax(logits.float(), dim=-1)
    conf, pred = probs.max(dim=-1)
    correct = (pred == targets).float()

    ece = torch.zeros((), dtype=torch.float32, device=logits.device)
    bin_edges = torch.linspace(0.0, 1.0, n_bins + 1, device=logits.device)
    for idx in range(n_bins):
        lo, hi = bin_edges[idx], bin_edges[idx + 1]
        if idx == 0:
            mask = (conf >= lo) & (conf <= hi)
        else:
            mask = (conf > lo) & (conf <= hi)
        if mask.any():
            bin_acc = correct[mask].mean()
            bin_conf = conf[mask].mean()
            ece = ece + mask.float().mean() * (bin_acc - bin_conf).abs()
    return ece


def brier_score(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Compute multiclass Brier score from logits."""
    if logits.ndim != 2:
        raise ValueError(f"logits must have shape (N, V), got {tuple(logits.shape)}")
    targets = targets.reshape(-1).to(logits.device)
    if targets.numel() != logits.shape[0]:
        raise ValueError("targets length must match logits batch dimension")
    probs = F.softmax(logits.float(), dim=-1)
    one_hot = F.one_hot(targets, num_classes=logits.shape[-1]).float()
    return ((probs - one_hot) ** 2).sum(dim=-1).mean()


def find_best_temperature(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    candidates: torch.Tensor | None = None,
) -> TemperatureSearchResult:
    """Grid-search a scalar temperature that minimizes NLL."""
    if candidates is None:
        candidates = torch.linspace(0.5, 3.0, 51, device=logits.device)
    else:
        candidates = candidates.to(logits.device).float()
    if candidates.numel() == 0:
        raise ValueError("temperature candidates cannot be empty")

    targets = targets.reshape(-1).to(logits.device)
    best_temp = None
    best_nll = float("inf")
    for temp in candidates:
        temp_f = float(temp.item())
        if temp_f <= 0 or not math.isfinite(temp_f):
            continue
        nll = F.cross_entropy(apply_temperature(logits, temp_f), targets).item()
        if nll < best_nll:
            best_nll = nll
            best_temp = temp_f
    if best_temp is None:
        raise ValueError("no valid positive temperature candidates")
    return TemperatureSearchResult(temperature=best_temp, nll=best_nll)
