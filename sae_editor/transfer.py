from __future__ import annotations

import torch
import torch.nn as nn


def project_features(
    features: torch.Tensor,
    from_d_model: int,
    to_d_model: int,
    seed: int = 42,
) -> torch.Tensor:
    """Project feature vectors between d_model spaces.

    Uses a fixed random Gaussian projection (Johnson-Lindenstrauss transform)
    that approximately preserves pairwise distances.

    Args:
        features:     (N, d) or (d,) tensor
        from_d_model: Source dimension
        to_d_model:   Target dimension
        seed:         Random seed for deterministic projection
    """
    if from_d_model == to_d_model:
        return features.clone()

    generator = torch.Generator(device=features.device)
    generator.manual_seed(seed)
    projection = torch.randn(
        from_d_model, to_d_model,
        generator=generator, device=features.device,
    ) / (from_d_model ** 0.5)

    if features.ndim == 1:
        return features.to(projection.device) @ projection
    return features.to(projection.device) @ projection


def transfer_edit(
    edits: dict[int, dict[str, torch.Tensor]],
    from_d_model: int,
    to_d_model: int,
    seed: int = 42,
) -> dict[int, dict[str, torch.Tensor]]:
    result = {}
    for layer_idx, edit in edits.items():
        keys = project_features(edit["keys"], from_d_model, to_d_model, seed=seed + layer_idx)
        values = project_features(edit["values"], from_d_model, to_d_model, seed=seed + layer_idx)
        result[layer_idx] = {"keys": keys, "values": values}
    return result
