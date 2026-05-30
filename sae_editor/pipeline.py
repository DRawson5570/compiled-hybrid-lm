from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn

from sae_editor.recompiler import RecompilerEngine
from sae_editor.splicer import SafetensorsSplicer


class NRTCSPipeline:
    """Neurosymbolic Round-Trip Compilation Stack — full pipeline orchestrator.

    Wires together all four phases:
      1. Decompile  (C2S): SAE feature extraction + circuit analysis
      2. Refactor   (UVM): Symbolic edits to the decompiled representation
      3. Recompile  (S2C): Analytical matrix construction + crosstalk prevention
      4. Splice     (Binary): Inline safetensors patching

    Usage:
        pipeline = NRTCSPipeline(decompiler)
        pipeline.compile(uvm_edits)
    """

    def __init__(
        self,
        recompiler: RecompilerEngine | None = None,
        eps: float = 1e-6,
    ):
        self.recompiler = recompiler or RecompilerEngine(eps=eps)

    def compile_dense_map(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        original_features: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compile a dense_map specification into W_down, W_up.

        Args:
            keys:   (N, d_in) key vectors
            values: (N, d_out) value vectors
            original_features: (d_in, m) features to protect from crosstalk

        Returns:
            {"W_down": (d_in, N), "W_up": (N, d_out)} compiled tensors (float32)
        """
        return self.recompiler.compile(keys, values, original_features)

    def compile_from_uvm_edits(
        self,
        edits: dict[int, dict[str, torch.Tensor]],
        original_features: dict[int, torch.Tensor] | None = None,
    ) -> dict[int, dict[str, torch.Tensor]]:
        """Compile UVM-DSL style per-layer edits into weight patches.

        Args:
            edits: Dict mapping layer_idx -> {
                "keys": (N, d_in) tensor,
                "values": (N, d_out) tensor,
            }
            original_features: Dict mapping layer_idx -> (d_in, m) features

        Returns:
            Dict mapping layer_idx -> {"W_down": ..., "W_up": ...}
        """
        results = {}
        for layer_idx, edit in edits.items():
            feats = None
            if original_features and layer_idx in original_features:
                feats = original_features[layer_idx]
            results[layer_idx] = self.recompiler.compile(
                edit["keys"], edit["values"], feats
            )
        return results

    def splice_patches(
        self,
        safetensors_path: str,
        patches: dict[int, dict[str, torch.Tensor]],
        model_prefix: str = "model.layers.{layer}.mlp",
        arch=None,
    ) -> None:
        """Splice compiled weight patches into a safetensors file.

        Args:
            safetensors_path: Path to target .safetensors file
            patches:          Dict from compile_from_uvm_edits
            model_prefix:     Format string for tensor names (old API)
            arch:             ArchitectureSpec (new API, takes precedence)
        """
        with SafetensorsSplicer(safetensors_path) as spl:
            for layer_idx, patch in patches.items():
                spl.splice_mlp(
                    layer=layer_idx,
                    W_down=patch["W_down"],
                    W_up=patch["W_up"],
                    model_name=model_prefix,
                    arch=arch,
                )

    def round_trip(
        self,
        safetensors_path: str,
        edits: dict[int, dict[str, torch.Tensor]],
        original_features: dict[int, torch.Tensor] | None = None,
        model_prefix: str = "model.layers.{layer}.mlp",
        arch=None,
    ) -> dict[int, dict[str, torch.Tensor]]:
        """Full round-trip: compile edits and splice into safetensors.

        Args:
            safetensors_path: Target .safetensors file to patch
            edits:            Per-layer key-value pairs
            original_features: Per-layer original features for crosstalk prevention
            model_prefix:     Tensor name format (old API)
            arch:             ArchitectureSpec (new API, takes precedence)

        Returns:
            The compiled patches dict (same as compile_from_uvm_edits)
        """
        patches = self.compile_from_uvm_edits(edits, original_features)
        self.splice_patches(safetensors_path, patches, model_prefix, arch=arch)
        return patches

    def verify_compilation(
        self,
        edits: dict[int, dict[str, torch.Tensor]],
        patches: dict[int, dict[str, torch.Tensor]] | None = None,
    ) -> dict[int, dict[str, torch.Tensor]]:
        """Verify that key @ W_down @ W_up recovers the values.

        If patches not provided, compiles them first.
        Returns per-layer reconstruction error.
        """
        if patches is None:
            patches = self.compile_from_uvm_edits(edits)

        results = {}
        for layer_idx, patch in patches.items():
            edit = edits[layer_idx]
            keys = edit["keys"].to(dtype=torch.float32)
            values = edit["values"].to(dtype=torch.float32)
            W_down = patch["W_down"].to(dtype=torch.float32, device=keys.device)
            W_up = patch["W_up"].to(dtype=torch.float32, device=keys.device)

            recon = keys @ W_down @ W_up
            err = (recon - values).norm(dim=-1)
            cosine = torch.nn.functional.cosine_similarity(recon, values, dim=-1)

            results[layer_idx] = {
                "max_error": err.max().item(),
                "mean_error": err.mean().item(),
                "min_cosine": cosine.min().item(),
                "mean_cosine": cosine.mean().item(),
            }
        return results

    def preview_single(
        self,
        layer: int,
        keys: torch.Tensor,
        values: torch.Tensor,
        model: nn.Module,
        tokenizer,
        prompts: list[str],
        strength: float = 1.0,
        top_k: int = 10,
        arch=None,
        gate_threshold: float = 0.3,
    ) -> "PreviewResult":
        from sae_editor.preview import (
            PreviewResult,
            _get_layer_for_arch,
            _make_preview_hook,
            _top_k_tokens,
            _token_shift,
        )

        compiled = self.recompiler.compile(keys, values)
        W_down = compiled["W_down"]
        W_up = compiled["W_up"]
        recon_err = (keys.float() @ W_down.float() @ W_up.float()
                     - values.float()).norm(dim=-1).mean().item()

        model_dtype = next(model.parameters()).dtype
        device = next(model.parameters()).device
        prompt = prompts[0] if prompts else ""
        if tokenizer is not None:
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
        else:
            seq_len = max(len(prompt.split()) + 2, 8) if prompt else 8
            inputs = {"input_ids": torch.randint(0, 1000, (1, seq_len), device=device)}

        with torch.no_grad():
            original_out = model(**inputs)
        original_logits = original_out.logits[0, -1].float()

        target_layer = _get_layer_for_arch(model, layer, arch)
        hook_fn, delta_l2_values = _make_preview_hook(
            W_down, W_up, keys, strength, model_dtype, gate_threshold
        )
        handle = target_layer.register_forward_hook(hook_fn)

        with torch.no_grad():
            patched_out = model(**inputs)
        patched_logits = patched_out.logits[0, -1].float()

        handle.remove()

        cosine = float(torch.nn.functional.cosine_similarity(
            original_logits.unsqueeze(0), patched_logits.unsqueeze(0), dim=-1
        ).item())

        original_probs = torch.nn.functional.softmax(original_logits, dim=-1)
        patched_probs = torch.nn.functional.softmax(patched_logits, dim=-1)

        return PreviewResult(
            layer_idx=layer,
            prompt=prompt,
            original_top_k=_top_k_tokens(original_logits, tokenizer, k=top_k),
            patched_top_k=_top_k_tokens(patched_logits, tokenizer, k=top_k),
            cosine_shift=cosine,
            max_token_shift=_token_shift(original_probs, patched_probs, tokenizer, k=top_k),
            reconstruction_error=recon_err,
            offset_l2=sum(delta_l2_values) / max(len(delta_l2_values), 1),
            strength=strength,
        )

    def preview(
        self,
        edits: dict[int, dict[str, torch.Tensor]],
        model: nn.Module,
        tokenizer,
        prompts: list[str],
        strength: float = 1.0,
        top_k: int = 10,
        max_new_tokens: int = 0,
        arch=None,
        gate_threshold: float = 0.3,
    ) -> "MultiLayerPreviewResult":
        from sae_editor.preview import (
            MultiLayerPreviewResult,
            PreviewResult,
            _get_layer_for_arch,
            _make_preview_hook,
            _top_k_tokens,
        )

        patches = self.compile_from_uvm_edits(edits)
        verification = self.verify_compilation(edits, patches)

        model_dtype = next(model.parameters()).dtype
        device = next(model.parameters()).device

        prompt = prompts[0] if prompts else ""
        if tokenizer is not None:
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
        else:
            seq_len = max(len(prompt.split()) + 2, 8) if prompt else 8
            inputs = {"input_ids": torch.randint(0, 1000, (1, seq_len), device=device)}

        with torch.no_grad():
            original_out = model(**inputs)
        original_logits = original_out.logits[0, -1].float()

        handles = []
        all_delta_values = []
        for layer_idx, patch in patches.items():
            target_layer = _get_layer_for_arch(model, layer_idx, arch)
            layer_keys = edits[layer_idx]["keys"]
            hook_fn, delta_l2 = _make_preview_hook(
                patch["W_down"], patch["W_up"], layer_keys,
                strength, model_dtype, gate_threshold
            )
            handle = target_layer.register_forward_hook(hook_fn)
            handles.append(handle)
            all_delta_values.append((layer_idx, delta_l2))

        with torch.no_grad():
            patched_out = model(**inputs)
        patched_logits = patched_out.logits[0, -1].float()

        for handle in handles:
            handle.remove()

        combined_cosine = float(torch.nn.functional.cosine_similarity(
            original_logits.unsqueeze(0), patched_logits.unsqueeze(0), dim=-1
        ).item())

        result = MultiLayerPreviewResult(
            combined_top_k=_top_k_tokens(patched_logits, tokenizer, k=top_k),
            combined_cosine_shift=combined_cosine,
            prompts=list(prompts),
        )

        for layer_idx, patch in patches.items():
            verif_data = verification.get(layer_idx, {})
            delta_l2 = 0.0
            for idx, dl2 in all_delta_values:
                if idx == layer_idx:
                    delta_l2 = sum(dl2) / max(len(dl2), 1)
                    break
            result.per_layer[layer_idx] = PreviewResult(
                layer_idx=layer_idx,
                prompt=prompt,
                original_top_k=_top_k_tokens(original_logits, tokenizer, k=top_k),
                patched_top_k=[],
                cosine_shift=0.0,
                max_token_shift=[],
                reconstruction_error=verif_data.get("mean_error", 0.0),
                offset_l2=delta_l2,
                strength=strength,
            )

        if max_new_tokens > 0 and prompts and tokenizer is not None:
            with torch.no_grad():
                orig_gen = model.generate(
                    **inputs, max_new_tokens=max_new_tokens, do_sample=False
                )
            gen_handles = []
            for layer_idx, patch in patches.items():
                target_layer = _get_layer_for_arch(model, layer_idx, arch)
                layer_keys = edits[layer_idx]["keys"]
                hook_fn, _ = _make_preview_hook(
                    patch["W_down"], patch["W_up"], layer_keys,
                    strength, model_dtype, gate_threshold
                )
                gen_handles.append(target_layer.register_forward_hook(hook_fn))
            with torch.no_grad():
                patched_gen = model.generate(
                    **inputs, max_new_tokens=max_new_tokens, do_sample=False
                )
            for h in gen_handles:
                h.remove()

            result.original_text = tokenizer.decode(orig_gen[0], skip_special_tokens=True)
            result.patched_text = tokenizer.decode(patched_gen[0], skip_special_tokens=True)

        return result

    def compare(
        self,
        edits: dict[int, dict[str, torch.Tensor]],
        model: nn.Module,
        tokenizer,
        prompts: list[str],
        strengths: list[float] | None = None,
        arch=None,
        gate_threshold: float = 0.3,
    ) -> list["MultiLayerPreviewResult"]:
        if strengths is None:
            strengths = [0.1, 0.5, 1.0, 2.0, 5.0]

        results = []
        for s in strengths:
            result = self.preview(
                edits=edits, model=model, tokenizer=tokenizer,
                prompts=prompts, strength=s, arch=arch,
                gate_threshold=gate_threshold,
            )
            results.append(result)
        return results
