from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class QwenActivationCollector:
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-1.5B",
        layers: Optional[List[int]] = None,
        device: str = "cuda",
    ):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            attn_implementation="eager",
        ).to(self.device)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        self.d_model = self.model.config.hidden_size
        self.n_layers = self.model.config.num_hidden_layers
        self.n_heads = self.model.config.num_attention_heads
        self.head_dim = self.d_model // self.n_heads
        self.n_kv_heads = self.model.config.num_key_value_heads

        if layers is None:
            self.layers = list(range(self.n_layers))
        else:
            self.layers = layers

        self._hooks = []
        self._activations: Dict[str, torch.Tensor] = {}

    def collect_residual_stream(
        self,
        texts: List[str],
        max_length: int = 128,
        batch_size: int = 4,
    ) -> Dict[int, torch.Tensor]:
        self._clear_hooks()

        residual: Dict[int, List[torch.Tensor]] = {}

        def make_hook(layer_idx):
            def hook(module, input, output):
                if layer_idx not in residual:
                    residual[layer_idx] = []
                residual[layer_idx].append(output[0].detach().cpu())
            return hook

        for layer_idx in self.layers:
            layer = self.model.model.layers[layer_idx]
            handle = layer.register_forward_hook(make_hook(layer_idx))
            self._hooks.append(handle)

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            ).to(self.device)

            with torch.no_grad():
                self.model(**inputs, output_attentions=False)

        self._clear_hooks()
        result = {}
        for k, v_list in sorted(residual.items()):
            if not v_list:
                continue
            if len(v_list) == 1:
                result[k] = v_list[0]
            else:
                result[k] = torch.cat([t.reshape(-1, t.shape[-1]) for t in v_list], dim=0)
        return result

    def collect_attention_outputs(
        self,
        texts: List[str],
        layer: int,
        max_length: int = 128,
        batch_size: int = 4,
    ) -> List[torch.Tensor]:
        self._clear_hooks()

        outputs = []

        def hook(module, input, output):
            outputs.append(output[0].detach().cpu())

        target = self.model.model.layers[layer].self_attn
        handle = target.register_forward_hook(hook)
        self._hooks.append(handle)

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            ).to(self.device)

            with torch.no_grad():
                self.model(**inputs, output_hidden_states=False)

        self._clear_hooks()
        return outputs

    def collect_headwise_attention(
        self,
        texts: List[str],
        max_length: int = 128,
    ) -> Dict[Tuple[int, int], List[torch.Tensor]]:
        self._clear_hooks()

        patterns: Dict[Tuple[int, int], List[torch.Tensor]] = {}

        for layer_idx in self.layers:
            for head_idx in range(self.n_heads):
                patterns[(layer_idx, head_idx)] = []

        def make_attention_hook(layer_idx):
            def hook(module, input, output):
                if len(output) >= 2 and output[1] is not None:
                    attn_weights = output[1].detach().cpu()
                    for head_idx in range(self.n_heads):
                        patterns[(layer_idx, head_idx)].append(
                            attn_weights[:, head_idx, :, :]
                        )
            return hook

        for layer_idx in self.layers:
            layer = self.model.model.layers[layer_idx]
            handle = layer.self_attn.register_forward_hook(make_attention_hook(layer_idx))
            self._hooks.append(handle)

        for i in range(0, len(texts), 1):
            text = texts[i]
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
            ).to(self.device)

            with torch.no_grad():
                self.model(**inputs, output_attentions=True)

        self._clear_hooks()
        return patterns

    def collect_attention_from_layer(
        self,
        texts: List[str],
        layers: List[int],
        max_length: int = 128,
    ) -> Dict[int, List[torch.Tensor]]:
        self._clear_hooks()

        attn_per_layer: Dict[int, List[torch.Tensor]] = {
            layer: [] for layer in layers
        }

        def make_hook(layer_idx):
            def hook(module, input, output):
                if len(output) >= 2 and output[1] is not None:
                    attn_per_layer[layer_idx].append(
                        output[1].detach().cpu()
                    )
            return hook

        for layer_idx in layers:
            layer = self.model.model.layers[layer_idx]
            handle = layer.self_attn.register_forward_hook(make_hook(layer_idx))
            self._hooks.append(handle)

        for text in texts:
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
            ).to(self.device)

            with torch.no_grad():
                self.model(**inputs, output_attentions=True)

        self._clear_hooks()
        return attn_per_layer

    def collect_mlp_activations(
        self,
        texts: List[str],
        layers: List[int],
        max_length: int = 128,
    ) -> Dict[int, List[torch.Tensor]]:
        self._clear_hooks()

        mlp_acts: Dict[int, List[torch.Tensor]] = {
            layer: [] for layer in layers
        }

        def make_hook(layer_idx):
            def hook(module, input, output):
                mlp_acts[layer_idx].append(output.detach().cpu())
            return hook

        for layer_idx in layers:
            layer = self.model.model.layers[layer_idx]
            handle = layer.mlp.register_forward_hook(make_hook(layer_idx))
            self._hooks.append(handle)

        for text in texts:
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
            ).to(self.device)

            with torch.no_grad():
                self.model(**inputs)

        self._clear_hooks()
        return mlp_acts

    def collect_all_layer_attention(
        self,
        texts: List[str],
        max_length: int = 128,
    ) -> Dict[int, torch.Tensor]:
        self._clear_hooks()

        all_attn: Dict[int, torch.Tensor] = {}

        def make_hook(layer_idx):
            def hook(module, input, output):
                all_attn[layer_idx] = output[0].detach().cpu()
            return hook

        for layer_idx in range(self.n_layers):
            layer = self.model.model.layers[layer_idx]
            handle = layer.self_attn.register_forward_hook(make_hook(layer_idx))
            self._hooks.append(handle)

        for text in texts:
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
            ).to(self.device)

            with torch.no_grad():
                self.model(**inputs)

        self._clear_hooks()
        return all_attn

    def run_model_with_output_hidden(
        self,
        text: str,
        max_length: int = 128,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        ).to(self.device)

        with torch.no_grad():
            out = self.model(**inputs, output_hidden_states=True, use_cache=False)
            logits = out.logits[0].cpu()
            hidden_states = [h[0].cpu() for h in out.hidden_states]

        return logits, hidden_states

    def _clear_hooks(self):
        for handle in self._hooks:
            handle.remove()
        self._hooks.clear()
