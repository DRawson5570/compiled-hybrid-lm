from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn as nn


class NRTCSDecompiler:
    """Phase I: Continuous-to-Symbolic (C2S) Decompiler.

    Extracts feature activations from a model using pretrained Sparse Autoencoders
    and performs path attribution patching to trace causal circuits.

    The decompiler reads model activations layer-by-layer, runs each layer's SAE
    to identify which features fire above threshold τ, and computes gradient-based
    attributions to link upstream features to downstream effects.
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        saes: Dict[int, nn.Module],
        threshold: float = 0.1,
        device: str = "cuda",
    ):
        """
        Args:
            model:     HF transformer model
            tokenizer: Corresponding tokenizer
            saes:      Dict mapping layer_idx -> pretrained SparseAutoencoder
            threshold: Activation threshold τ (features with h > τ are kept)
            device:    Device for computation
        """
        self.model = model
        self.tokenizer = tokenizer
        self.saes = saes
        self.threshold = threshold
        self.device = device

        for sae in self.saes.values():
            sae.to(self.device)
            sae.eval()

        self._hooks = []
        self._saved_activations: Dict[int, torch.Tensor] = {}

    @property
    def d_model(self) -> int:
        return self.model.config.hidden_size

    def collect_activations(
        self,
        texts: List[str],
        max_length: int = 128,
        batch_size: int = 4,
    ) -> Dict[int, torch.Tensor]:
        """Collect residual stream activations for specified layers.

        Returns dict mapping layer_idx -> (batch, seq, d_model) tensor.
        """
        self._clear_hooks()
        collected: Dict[int, List[torch.Tensor]] = {}

        for layer_idx in self.saes.keys():
            layer = self._get_layer(layer_idx)
            handle = layer.register_forward_hook(self._make_hook(layer_idx, collected))
            self._hooks.append(handle)

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            if self.tokenizer is not None:
                inputs = self.tokenizer(
                    batch,
                    return_tensors="pt",
                    padding="max_length",
                    truncation=True,
                    max_length=max_length,
                ).to(self.device)
            else:
                batch_size_actual = len(batch)
                seq_len = min(max_length, 32)
                inputs = {
                    "input_ids": torch.randint(
                        0, 1000, (batch_size_actual, seq_len),
                        device=self.device,
                    )
                }

            with torch.no_grad():
                self.model(**inputs)

        self._clear_hooks()

        result = {}
        for layer_idx, chunks in collected.items():
            result[layer_idx] = torch.cat(chunks, dim=0)
        return result

    def extract_features(
        self,
        texts: List[str],
        max_length: int = 128,
        batch_size: int = 4,
    ) -> Dict[int, Dict[int, torch.Tensor]]:
        """Run SAEs on activations and extract features above threshold.

        Returns:
            Dict mapping layer_idx -> {
                "activations": (B, T, d_model) raw activations,
                "feature_indices": (K,) tensor of feature indices,
                "feature_vectors": (K, d_model) decoder direction vectors,
                "feature_acts": (B, T, K) activation strengths,
            }
        """
        activations = self.collect_activations(texts, max_length, batch_size)
        result = {}

        for layer_idx, acts in activations.items():
            sae = self.saes[layer_idx]
            sae.to(self.device)

            acts_flat = acts.reshape(-1, self.d_model).to(
                dtype=torch.float32, device=self.device
            )

            with torch.no_grad():
                h = sae.encode(acts_flat)

            h_reshaped = h.reshape(acts.shape[0], acts.shape[1], -1)

            fire_mask = h_reshaped > self.threshold
            any_fire = fire_mask.any(dim=(0, 1))

            active_indices = any_fire.nonzero(as_tuple=True)[0].cpu()
            active_vectors = sae.decoder.weight[:, active_indices].T.cpu()
            active_acts = h_reshaped[:, :, active_indices].cpu()

            result[layer_idx] = {
                "activations": acts.cpu(),
                "feature_indices": active_indices,
                "feature_vectors": active_vectors,
                "feature_acts": active_acts,
            }

        return result

    def path_attribution(
        self,
        text: str,
        upstream_layer: int,
        downstream_layer: int,
        upstream_features: List[int] | None = None,
        downstream_feature: int | None = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute path attribution A(x_j -> y) between upstream and downstream features.

        A(x_j -> y) = x_j · grad(y with respect to x_j)

        Where x_j is the activation of upstream feature j and y is the
        downstream feature activation.

        The gradient is computed through the model layers between upstream
        and downstream, so model parameters must retain grad tracking.
        This requires enough memory to hold activations for the forward pass
        (typically fine for single-text analysis on models up to ~7B).

        Args:
            text:              Single input text
            upstream_layer:    Layer index for upstream features
            downstream_layer:  Layer index for downstream features
            upstream_features: List of upstream feature indices to attribute,
                              or None to use all features with non-zero SAE decoder
            downstream_feature: Target downstream feature index, or None to
                               attribute to all downstream features

        Returns:
            dict with keys: "attributions" (N_up, N_down), "upstream_indices",
            "downstream_indices", "upstream_acts", "downstream_acts"
        """
        if upstream_layer not in self.saes:
            raise KeyError(f"No SAE for upstream layer {upstream_layer}")
        if downstream_layer not in self.saes:
            raise KeyError(f"No SAE for downstream layer {downstream_layer}")

        sae_up = self.saes[upstream_layer]
        sae_down = self.saes[downstream_layer]

        if self.tokenizer is not None:
            inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        else:
            seq_len = min(len(text.split()) + 2, 32) if text else 8
            inputs = {
                "input_ids": torch.randint(0, 1000, (1, seq_len), device=self.device)
            }

        upstream_hidden = None
        downstream_hidden = None

        def upstream_hook(module, input, output):
            nonlocal upstream_hidden
            upstream_hidden = output[0] if isinstance(output, tuple) else output

        def downstream_hook(module, input, output):
            nonlocal downstream_hidden
            downstream_hidden = output[0] if isinstance(output, tuple) else output

        layer_up = self._get_layer(upstream_layer)
        layer_down = self._get_layer(downstream_layer)

        handle_up = layer_up.register_forward_hook(upstream_hook)
        handle_down = layer_down.register_forward_hook(downstream_hook)

        self.model(**inputs)

        handle_up.remove()
        handle_down.remove()

        if upstream_hidden is None or downstream_hidden is None:
            raise RuntimeError("Failed to capture activations")

        upstream_hidden = upstream_hidden.to(dtype=torch.float32)
        downstream_hidden = downstream_hidden.to(dtype=torch.float32)

        h_up_flat = sae_up.encode(upstream_hidden.reshape(-1, self.d_model))
        h_down_raw = sae_down.encode(downstream_hidden.reshape(-1, self.d_model))

        y = h_down_raw[:, downstream_feature].sum() if downstream_feature is not None else h_down_raw.sum()

        grad = torch.autograd.grad(y, upstream_hidden, retain_graph=False, allow_unused=True)[0]
        if grad is None:
            return {
                "attributions": torch.zeros(0),
                "upstream_indices": torch.zeros(0, dtype=torch.long),
                "downstream_indices": torch.zeros(0, dtype=torch.long),
                "upstream_acts": h_up_flat.detach().cpu(),
                "downstream_acts": h_down_raw.detach().cpu(),
            }

        grad = grad.reshape(-1, self.d_model)

        if upstream_features is None:
            upstream_features = list(range(sae_up.n_features))

        attributions = []
        for feat_idx in upstream_features:
            w_dec = sae_up.decoder.weight[:, feat_idx].detach().to(
                dtype=torch.float32, device=self.device
            )
            hidden_flat = upstream_hidden.reshape(-1, self.d_model).to(dtype=torch.float32)
            h_i_raw = hidden_flat @ w_dec
            attr = (h_i_raw.detach() * (grad.float() @ w_dec)).sum()
            attributions.append(attr.item())

        downstream_indices = (
            [downstream_feature]
            if downstream_feature is not None
            else list(range(sae_down.n_features))
        )

        return {
            "attributions": torch.tensor(attributions),
            "upstream_indices": torch.tensor(upstream_features),
            "downstream_indices": torch.tensor(downstream_indices),
            "upstream_acts": h_up_flat.detach().cpu(),
            "downstream_acts": h_down_raw.detach().cpu(),
        }

    def _get_layer(self, layer_idx: int):
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            return self.model.model.layers[layer_idx]
        if hasattr(self.model, "transformer") and hasattr(self.model.transformer, "h"):
            return self.model.transformer.h[layer_idx]
        if hasattr(self.model, "layers"):
            return self.model.layers[layer_idx]
        raise AttributeError(f"Cannot find layers in model of type {type(self.model)}")

    def _make_hook(self, layer_idx: int, collected: Dict[int, List[torch.Tensor]]):
        def hook(module, input, output):
            if layer_idx not in collected:
                collected[layer_idx] = []
            if isinstance(output, tuple):
                val = output[0]
            else:
                val = output
            collected[layer_idx].append(val.detach().cpu())

        return hook

    def _clear_hooks(self):
        for handle in self._hooks:
            handle.remove()
        self._hooks.clear()
