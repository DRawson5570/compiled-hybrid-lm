from __future__ import annotations

import os
from typing import Dict, List

import torch
import torch.nn as nn


class SAETrainingPipeline:
    """Train SAEs on model layers for use with NRTCSDecompiler."""

    def __init__(self, arch=None):
        if arch is None:
            from sae_editor.architectures import ArchitectureSpec
            self.arch = ArchitectureSpec.from_model_name("qwen")
        else:
            self.arch = arch

    def collect_layer_activations(
        self,
        model: nn.Module,
        tokenizer,
        texts: List[str],
        layers: List[int],
        max_length: int = 128,
        batch_size: int = 4,
    ) -> Dict[int, torch.Tensor]:
        from sae_editor.decompiler import NRTCSDecompiler
        from sae_editor.tests.utils import make_random_sae

        d_model = model.config.hidden_size
        saes = {l: make_random_sae(d_model=d_model, n_features=1) for l in layers}
        decompiler = NRTCSDecompiler(
            model=model, tokenizer=tokenizer, saes=saes,
            threshold=0.0, device=str(next(model.parameters()).device),
        )
        return decompiler.collect_activations(texts, max_length, batch_size)

    def train_all(
        self,
        model: nn.Module,
        tokenizer,
        texts: List[str],
        layers: List[int],
        n_features: int = 256,
        steps: int = 2000,
        lr: float = 1e-3,
        batch_size: int = 256,
        device: str = "cuda",
    ) -> Dict[int, nn.Module]:
        from full_compiled_experiment.ucn.decompile.sae import (
            SparseAutoencoder,
            train_sae,
            normalize_decoder,
        )

        d_model = model.config.hidden_size

        activations_by_layer = self.collect_layer_activations(
            model=model,
            tokenizer=tokenizer,
            texts=texts,
            layers=layers,
            max_length=128,
            batch_size=4,
        )

        saes = {}
        for layer in layers:
            acts = activations_by_layer[layer]
            acts_flat = acts.reshape(-1, d_model).to(dtype=torch.float32, device=device)

            sae = SparseAutoencoder(
                d_model=d_model,
                n_features=n_features,
                l1_lambda=1e-3,
            )

            train_batch = max(min(batch_size, acts_flat.shape[0]), 16)
            train_sae(
                sae=sae,
                activations=acts_flat,
                steps=steps,
                lr=lr,
                batch_size=train_batch,
                device=device,
                verbose=False,
            )

            normalize_decoder(sae)
            saes[layer] = sae

        return saes


class SAERegistry:
    """Persist and load trained SAEs with metadata."""

    @staticmethod
    def save(saes: Dict[int, nn.Module], path_prefix: str):
        os.makedirs(path_prefix, exist_ok=True)
        for layer_idx, sae in saes.items():
            path = os.path.join(path_prefix, f"layer_{layer_idx}.pt")
            torch.save(sae.state_dict(), path)

    @staticmethod
    def load(path_prefix: str, d_model: int, n_features: int) -> Dict[int, nn.Module]:
        from full_compiled_experiment.ucn.decompile.sae import SparseAutoencoder

        saes = {}
        for fname in sorted(os.listdir(path_prefix)):
            if fname.startswith("layer_") and fname.endswith(".pt"):
                layer_idx = int(fname.split("_")[1].split(".")[0])
                sae = SparseAutoencoder(
                    d_model=d_model, n_features=n_features, l1_lambda=1e-3
                )
                path = os.path.join(path_prefix, fname)
                sae.load_state_dict(torch.load(path, weights_only=True))
                sae.eval()
                saes[layer_idx] = sae

        return saes

    @staticmethod
    def create_decompiler(
        model, tokenizer, path_prefix: str,
        d_model: int, n_features: int, threshold: float = 0.1,
    ):
        from sae_editor.decompiler import NRTCSDecompiler

        saes = SAERegistry.load(path_prefix, d_model, n_features)
        device = str(next(model.parameters()).device)
        return NRTCSDecompiler(
            model=model, tokenizer=tokenizer, saes=saes,
            threshold=threshold, device=device,
        )
