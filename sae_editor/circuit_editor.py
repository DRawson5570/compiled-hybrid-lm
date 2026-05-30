from __future__ import annotations

from typing import Dict, List

import torch


class CircuitEditor:
    """Bridge from decompiled features to recompilable key-value edits."""

    def __init__(self, decompiler):
        self.decompiler = decompiler
        self.d_model = decompiler.d_model

    def find_feature_activating_on(
        self, texts: List[str], top_k: int = 5
    ) -> Dict[int, List[int]]:
        features = self.decompiler.extract_features(texts)
        result = {}
        for layer_idx, fdata in features.items():
            acts = fdata["feature_acts"]
            if acts.numel() == 0:
                result[layer_idx] = []
                continue
            mean_acts = acts.mean(dim=(0, 1))
            sorted_idx = mean_acts.argsort(descending=True)
            top = sorted_idx[: min(top_k, len(sorted_idx))]
            result[layer_idx] = fdata["feature_indices"][top].tolist()
        return result

    def extract_feature_vector(self, layer: int, feature_idx: int) -> torch.Tensor:
        sae = self.decompiler.saes[layer]
        return sae.decoder.weight[:, feature_idx].detach().clone()

    def extract_value_vector_for_text(self, text: str, layer: int) -> torch.Tensor:
        if self.decompiler.tokenizer is not None:
            inputs = self.decompiler.tokenizer(
                text, return_tensors="pt"
            ).to(self.decompiler.device)
        else:
            inputs = {
                "input_ids": torch.randint(
                    0, 1000, (1, max(len(text.split()), 1) + 4),
                    device=self.decompiler.device,
                )
            }

        hidden = None

        def hook(module, input, output):
            nonlocal hidden
            hidden = output[0] if isinstance(output, tuple) else output

        target_layer = self.decompiler._get_layer(layer)
        handle = target_layer.register_forward_hook(hook)

        with torch.no_grad():
            self.decompiler.model(**inputs)

        handle.remove()

        if hidden is None:
            raise RuntimeError(f"Failed to capture activation at layer {layer}")

        return hidden[0, -1, :].detach().cpu()

    def create_edit_from_texts(
        self, source_text: str, target_text: str,
        layer: int, top_k: int = 1,
    ) -> Dict[int, Dict[str, torch.Tensor]]:
        features = self.find_feature_activating_on([source_text], top_k=top_k)
        feature_indices = features.get(layer, [])
        if not feature_indices:
            return {}

        key_vecs = []
        for f_idx in feature_indices:
            kv = self.extract_feature_vector(layer, f_idx)
            key_vecs.append(kv)

        value_vec = self.extract_value_vector_for_text(target_text, layer)

        keys = torch.stack(key_vecs, dim=0)
        values = value_vec.unsqueeze(0).expand(len(key_vecs), -1)

        return {layer: {"keys": keys, "values": values}}

    def verify_edit(
        self,
        edits: Dict[int, Dict[str, torch.Tensor]],
        cos_threshold: float = 0.9,
    ) -> bool:
        from sae_editor.pipeline import NRTCSPipeline

        pipeline = NRTCSPipeline()
        patches = pipeline.compile_from_uvm_edits(edits)
        results = pipeline.verify_compilation(edits, patches)

        for layer_idx, v in results.items():
            if v["mean_cosine"] < cos_threshold:
                return False

        return True
