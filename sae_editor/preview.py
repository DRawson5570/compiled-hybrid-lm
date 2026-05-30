from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class PreviewResult:
    """Structured result of a non-destructive single-layer patch preview."""

    layer_idx: int
    prompt: str
    original_top_k: list[tuple[str, float]]
    patched_top_k: list[tuple[str, float]]
    cosine_shift: float
    max_token_shift: list[tuple[str, float]]
    reconstruction_error: float
    offset_l2: float
    strength: float = 1.0


@dataclass
class MultiLayerPreviewResult:
    """Accumulated preview results across multiple patched layers."""

    per_layer: dict[int, PreviewResult] = field(default_factory=dict)
    combined_top_k: list[tuple[str, float]] = field(default_factory=list)
    combined_cosine_shift: float = 0.0
    prompts: list[str] = field(default_factory=list)
    original_text: str = ""
    patched_text: str = ""

    @property
    def summary(self) -> str:
        lines = [f"Preview: {len(self.prompts)} prompts, {len(self.per_layer)} layers"]
        for layer_idx, result in self.per_layer.items():
            lines.append(
                f"  Layer {layer_idx}: cosine={result.cosine_shift:.4f}, "
                f"recon_err={result.reconstruction_error:.6f}, "
                f"offset_l2={result.offset_l2:.4f}"
            )
        if self.combined_cosine_shift:
            lines.append(f"  Combined: cosine={self.combined_cosine_shift:.4f}")
        return "\n".join(lines)


def _make_preview_hook(W_down, W_up, keys, strength, model_dtype, gate_threshold=0.3):
    """Similarity-gated preview hook.

    Only injects delta at positions where cosine_similarity(h, k_i) > threshold
    for at least one key k_i. Delta is soft-weighted by max_cos (higher
    similarity = larger injection). This prevents corrupting the residual
    stream at token positions that have nothing to do with the keys.

    Args:
        W_down:         (d_in, N) compiled down-projection
        W_up:           (N, d_out) compiled up-projection
        keys:           (N, d_in) original key vectors
        strength:        Multiplier on injected delta
        model_dtype:     Model's weight dtype for casting
        gate_threshold:  Cosine similarity threshold (0-1). Only positions
                        with max(cos(h, k_i)) > threshold get injection.
    """
    W_down_local = W_down.to(dtype=torch.float32)
    W_up_local = W_up.to(dtype=torch.float32)
    keys_local = keys.to(dtype=torch.float32)
    keys_norm = keys_local / keys_local.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    delta_l2_values = []

    def hook_fn(module, input, output):
        hidden = output[0] if isinstance(output, tuple) else output
        device = hidden.device
        h_f32 = hidden.to(dtype=torch.float32)

        h_norm = h_f32 / h_f32.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        cos_sim = torch.einsum("btd,nd->btn", h_norm, keys_norm.to(device=device))
        max_cos, _ = cos_sim.max(dim=-1)
        gate = (max_cos > gate_threshold).float().unsqueeze(-1)

        if gate.sum() == 0:
            delta_l2_values.append(0.0)
            if isinstance(output, tuple):
                return output
            return hidden

        W_down_f32 = W_down_local.to(device=device)
        W_up_f32 = W_up_local.to(device=device)
        delta = h_f32 @ W_down_f32 @ W_up_f32

        h_rms = h_f32.pow(2).mean(dim=-1, keepdim=True).sqrt().clamp(min=1e-8)
        d_rms = delta.pow(2).mean(dim=-1, keepdim=True).sqrt().clamp(min=1e-8)
        delta_normalized = delta * (h_rms / d_rms)

        gated_delta = gate * max_cos.unsqueeze(-1) * delta_normalized * strength
        delta_l2_values.append(gated_delta.norm(dim=-1).mean().item())

        modified = hidden + gated_delta.to(dtype=model_dtype)

        if isinstance(output, tuple):
            return (modified,) + output[1:]
        return modified

    return hook_fn, delta_l2_values


def _get_layer_for_arch(model, layer_idx: int, arch):
    if arch is not None:
        path = arch.layer_access_path
        parts = path.split(".")
        obj = model
        for part in parts:
            obj = getattr(obj, part)
        return obj[layer_idx]

    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers[layer_idx]
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h[layer_idx]
    if hasattr(model, "layers"):
        return model.layers[layer_idx]
    raise AttributeError(f"Cannot find layers in model of type {type(model)}")


def _top_k_tokens(logits: torch.Tensor, tokenizer, k: int = 10) -> list[tuple[str, float]]:
    probs = torch.nn.functional.softmax(logits.float(), dim=-1)
    topk = torch.topk(probs, k=min(k, probs.shape[-1]))
    if tokenizer is not None:
        return [(tokenizer.decode([int(idx)]), float(val))
                for idx, val in zip(topk.indices, topk.values)]
    return [(f"token_{int(idx)}", float(val))
            for idx, val in zip(topk.indices, topk.values)]


def _token_shift(original_probs, patched_probs, tokenizer, k: int = 10) -> list[tuple[str, float]]:
    diff = (patched_probs - original_probs).abs()
    top_indices = torch.topk(diff.flatten(), k=min(k, diff.numel())).indices
    results = []
    for idx in top_indices:
        token = f"token_{int(idx.item())}"
        if tokenizer is not None:
            token = tokenizer.decode([int(idx.item())])
        results.append((token, float(diff.flatten()[idx].item())))
    return results
