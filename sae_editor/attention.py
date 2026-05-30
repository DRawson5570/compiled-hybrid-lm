from __future__ import annotations

import torch
import torch.nn as nn


class AttentionExtractor:
    """Extract attention weights from a loaded model into a structured dict."""

    def __init__(self, arch=None):
        if arch is None:
            from sae_editor.architectures import ArchitectureSpec
            self.arch = ArchitectureSpec.from_model_name("qwen")
        else:
            self.arch = arch

    def extract(self, model, layer: int) -> tuple[dict[str, torch.Tensor], dict]:
        """Returns ({W_q, W_k, W_v, W_o, biases...}, metadata_dict).

        Metadata: n_heads, n_kv_heads, head_dim, has_gqa, has_rope.
        For fused QKV (GPT-2): splits c_attn into Q/K/V slices.
        """
        weights = {}
        metadata = {}

        if self.arch.attn_type == "fused":
            weights, metadata = self._extract_fused(model, layer)
        else:
            weights, metadata = self._extract_separate(model, layer)

        return weights, metadata

    def _extract_separate(self, model, layer: int):
        attn = self._get_attn_module(model, layer)
        config = self._get_model_config(model)

        head_dim = getattr(config, "head_dim", None) or (config.hidden_size // config.num_attention_heads)
        n_heads = config.num_attention_heads
        n_kv_heads = getattr(config, "num_key_value_heads", n_heads)
        has_gqa = n_heads != n_kv_heads

        weights = {}

        for proj_name in ["q_proj", "k_proj", "v_proj"]:
            proj = getattr(attn, proj_name, None)
            if proj is not None:
                weights[f"W_{proj_name[0]}"] = proj.weight.data.detach().clone()
                if proj.bias is not None:
                    weights[f"b_{proj_name[0]}"] = proj.bias.data.detach().clone()

        o_proj = getattr(attn, "o_proj", None)
        if o_proj is not None:
            weights["W_o"] = o_proj.weight.data.detach().clone()
            if o_proj.bias is not None:
                weights["b_o"] = o_proj.bias.data.detach().clone()

        metadata = {
            "n_heads": n_heads,
            "n_kv_heads": n_kv_heads,
            "head_dim": head_dim,
            "has_gqa": has_gqa,
            "has_rope": self.arch.has_rope,
        }

        return weights, metadata

    def _extract_fused(self, model, layer: int):
        attn = self._get_attn_module(model, layer)
        config = self._get_model_config(model)

        d_model = config.hidden_size
        n_heads = config.num_attention_heads
        head_dim = d_model // n_heads

        c_attn = getattr(attn, "c_attn", None)
        if c_attn is not None:
            c_attn_weight = c_attn.weight.data.detach().clone()
            c_attn_bias = c_attn.bias.data.detach().clone() if c_attn.bias is not None else None

            W_q = c_attn_weight[:d_model, :]
            W_k = c_attn_weight[d_model:2*d_model, :]
            W_v = c_attn_weight[2*d_model:, :]

            weights = {"W_q": W_q, "W_k": W_k, "W_v": W_v}
            if c_attn_bias is not None:
                weights["b_q"] = c_attn_bias[:d_model]
                weights["b_k"] = c_attn_bias[d_model:2*d_model]
                weights["b_v"] = c_attn_bias[2*d_model:]
        else:
            weights = {}

        c_proj = getattr(attn, "c_proj", None)
        if c_proj is not None:
            weights["W_o"] = c_proj.weight.data.detach().clone()
            if c_proj.bias is not None:
                weights["b_o"] = c_proj.bias.data.detach().clone()

        metadata = {
            "n_heads": n_heads,
            "n_kv_heads": n_heads,
            "head_dim": head_dim,
            "has_gqa": False,
            "has_rope": False,
        }

        return weights, metadata

    def _get_attn_module(self, model, layer: int):
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            return model.model.layers[layer].self_attn
        if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
            return model.transformer.h[layer].attn
        raise AttributeError(f"Cannot find attention module at layer {layer}")

    def _get_model_config(self, model):
        return model.config


class AttentionSplicer:
    """Splice attention weights into a safetensors file."""

    def __init__(self, arch=None):
        if arch is None:
            from sae_editor.architectures import ArchitectureSpec
            self.arch = ArchitectureSpec.from_model_name("qwen")
        else:
            self.arch = arch

    def splice(self, safetensors_path: str, layer: int,
               weights: dict[str, torch.Tensor]):
        """Write attention weights to safetensors.

        For fused architectures (GPT-2): concatenates Q/K/V into c_attn.
        For separate architectures (Qwen): writes q_proj, k_proj, v_proj, o_proj.
        """
        from sae_editor.splicer import SafetensorsSplicer

        with SafetensorsSplicer(safetensors_path) as spl:
            if self.arch.attn_type == "fused":
                self._splice_fused(spl, layer, weights)
            else:
                self._splice_separate(spl, layer, weights)

    def transplant(self, safetensors_path: str,
                   source_layer: int, target_layer: int):
        """Copy attention weights from source_layer to target_layer."""
        from safetensors import safe_open

        weights = {}
        with safe_open(safetensors_path, framework="pt") as f:
            for name in f.keys():
                if self.arch.layer_prefix.format(layer=source_layer) in name:
                    key = name.replace(
                        self.arch.layer_prefix.format(layer=source_layer),
                        self.arch.layer_prefix.format(layer=target_layer),
                    )
                    weights[key] = f.get_tensor(name).clone()

        from sae_editor.splicer import SafetensorsSplicer
        with SafetensorsSplicer(safetensors_path) as spl:
            for name, tensor in weights.items():
                spl.splice_tensor(name, tensor.numpy().tobytes())

    def _splice_separate(self, spl, layer, weights):
        proj_map = {"W_q": "q", "W_k": "k", "W_v": "v", "W_o": "o"}
        missing = []
        for wkey, pkey in proj_map.items():
            suffix = f"self_attn.{pkey}_proj.weight"
            name = f"{self.arch.layer_prefix.format(layer=layer)}.{suffix}"
            if wkey in weights:
                spl._splice_tensor_from_array(name, weights[wkey])
            else:
                missing.append(wkey)
        if missing:
            import warnings
            warnings.warn(
                f"Attention splice missing weights: {missing}. "
                "These projections were not updated.",
                stacklevel=2,
            )

    def _splice_fused(self, spl, layer, weights):
        if all(k in weights for k in ["W_q", "W_k", "W_v"]):
            c_attn_weight = torch.cat([
                weights["W_q"], weights["W_k"], weights["W_v"]
            ], dim=0)
            name = f"{self.arch.layer_prefix.format(layer=layer)}.attn.c_attn.weight"
            spl._splice_tensor_from_array(name, c_attn_weight)

        if "W_o" in weights:
            name = f"{self.arch.layer_prefix.format(layer=layer)}.attn.c_proj.weight"
            spl._splice_tensor_from_array(name, weights["W_o"])
