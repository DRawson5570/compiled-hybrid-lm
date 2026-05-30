from __future__ import annotations

from typing import Dict, List, Tuple

import torch


def test_feature_on_prompt(
    feature_vec: torch.Tensor,
    model,
    tokenizer,
    prompt: str,
    layer: int,
    scale: float = 1.0,
    device: str = "cuda",
) -> Dict[str, float]:
    if isinstance(model, torch.nn.Module):
        hf_model = model
    else:
        hf_model = model.model

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    feature_vec = feature_vec.to(device=device, dtype=torch.float32)

    original_logits = _get_logits(hf_model, inputs)
    original_top5 = _top_k_tokens(original_logits, tokenizer, k=5)

    def inject_hook(module, input, output):
        modified = output[0] + scale * feature_vec.unsqueeze(0).unsqueeze(0)
        modified = modified.to(dtype=output[0].dtype)
        return (modified,) + output[1:]

    target_layer = hf_model.model.layers[layer]
    handle = target_layer.register_forward_hook(inject_hook)

    with torch.no_grad():
        out = hf_model(**inputs)
        modified_logits = out.logits[0, -1].float()

    handle.remove()

    modified_top5 = _top_k_tokens(modified_logits, tokenizer, k=5)

    shift = float(torch.nn.functional.cosine_similarity(
        modified_logits.unsqueeze(0), original_logits.unsqueeze(0), dim=-1
    ).item())

    return {
        "prompt": prompt,
        "scale": scale,
        "original_top1": original_top5[0] if original_top5 else "",
        "modified_top1": modified_top5[0] if modified_top5 else "",
        "cosine_shift": shift,
        "original_top5": original_top5,
        "modified_top5": modified_top5,
    }


def feature_intervention(
    sae,
    model,
    tokenizer,
    prompt: str,
    feature_idx: int,
    layer: int,
    feature_scale: float = 3.0,
    device: str = "cuda",
) -> Dict[str, float]:
    feature_vec = sae.decoder.weight[:, feature_idx].detach().clone()

    return test_feature_on_prompt(
        feature_vec=feature_vec,
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        layer=layer,
        scale=feature_scale,
        device=device,
    )


def _get_logits(model, inputs) -> torch.Tensor:
    with torch.no_grad():
        out = model(**inputs)
        return out.logits[0, -1].float()


def _top_k_tokens(logits: torch.Tensor, tokenizer, k: int = 5) -> List[str]:
    topk = torch.topk(logits.float(), k=k).indices
    return [tokenizer.decode([int(t)]) for t in topk]


def measure_intervention_impact(
    sae,
    collector,
    feature_idx: int,
    layer: int,
    prompt: str,
    scales: List[float] | None = None,
) -> Dict[str, List]:
    if scales is None:
        scales = [-3.0, -1.0, 0.0, 1.0, 3.0, 10.0]

    feature_vec = sae.decoder.weight[:, feature_idx].detach().clone()
    model = collector.model
    tokenizer = collector.tokenizer

    results = {"scale": [], "cosine_shift": [], "top1": []}
    for scale in scales:
        result = test_feature_on_prompt(
            feature_vec, model, tokenizer, prompt, layer, scale=scale
        )
        results["scale"].append(scale)
        results["cosine_shift"].append(result["cosine_shift"])
        results["top1"].append(result["modified_top1"])

    return results
