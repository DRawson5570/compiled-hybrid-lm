"""
Gap 3: Build multi-layer stdlib from Qwen2.5-1.5B.

Extracts attention + MLP primitives for layers [0,4,8,12,16,20,24].
Saves unified stdlib.uvm for multi-layer execution.
"""

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ucn.decompile.source_model import QwenActivationCollector
from ucn.decompile.mlp_decomposer import extract_mlp_keys_values
from ucn.stdlib.loader import save_stdlib_json, save_weight_tensor
from ucn.stdlib.schema import BehaviorMeta, MathDef, PrimitiveEntry


def extract_layer_attention_weights(model, layer):
    attn = model.model.layers[layer].self_attn
    weights = {}
    for name, param in attn.named_parameters():
        weights[name] = param.data.clone().float()
    return weights


def extract_rotary_embeddings(model, seq_len=256):
    from transformers.models.qwen2.modeling_qwen2 import Qwen2RotaryEmbedding
    head_dim = model.config.hidden_size // model.config.num_attention_heads
    rotary = model.model.rotary_emb
    dummy = torch.zeros(1, seq_len, head_dim)
    pos_ids = torch.arange(seq_len).unsqueeze(0)
    cos, sin = rotary(dummy, pos_ids)
    return cos, sin


def build_entry_attention(layer, attn_weights, cos, sin, weights_dir):
    n_heads = 12
    n_kv_heads = 2
    head_dim = 128

    weight_files = {}
    for name, t in attn_weights.items():
        fname = f"L{layer}_attn_{name.replace('.', '_')}.pt"
        save_weight_tensor(t, weights_dir / fname)
        weight_files[name] = fname

    cos_name = f"rope_cos.pt"
    sin_name = f"rope_sin.pt"
    if cos is not None:
        save_weight_tensor(cos, weights_dir / cos_name)
        save_weight_tensor(sin, weights_dir / sin_name)

    entry = PrimitiveEntry(
        primitive_id=f"PRM_ATTN_L{layer}",
        symbolic_name=f"full_attention_layer_{layer}",
        type="operator_circuit",
        source_layers=[layer],
        math_def=MathDef(
            operator_type="multihead_attention",
            u_uri=None,
        ),
        behavior=BehaviorMeta(
            description=f"Full multi-head attention for layer {layer}",
            trigger_conditions=["all_tokens"],
        ),
    )
    entry.weight_data = {
        "n_heads": n_heads,
        "n_kv_heads": n_kv_heads,
        "head_dim": head_dim,
        "weight_files": weight_files,
    }
    return entry


def build_entry_mlp(layer, kv_data, weights_dir):
    fname_keys = f"L{layer}_mlp_keys.pt"
    fname_values = f"L{layer}_mlp_values.pt"
    save_weight_tensor(kv_data["keys"], weights_dir / fname_keys)
    save_weight_tensor(kv_data["values"], weights_dir / fname_values)

    entry = PrimitiveEntry(
        primitive_id=f"PRM_MLP_L{layer}",
        symbolic_name=f"mlp_kv_layer_{layer}",
        type="operator_circuit",
        source_layers=[layer],
        math_def=MathDef(
            operator_type="key_value_lookup",
            u_uri=f"weights/{fname_keys}",
        ),
        behavior=BehaviorMeta(
            description=f"MLP key-value memory for layer {layer} ({kv_data.get('method', 'unknown')})",
            trigger_conditions=["all_tokens"],
        ),
    )
    entry.weight_data = {
        "n_neurons": kv_data.get("n_neurons", 0),
        "n_keys": kv_data.get("n_keys", 0),
        "method": kv_data.get("method", "unknown"),
        "keys_uri": fname_keys,
        "values_uri": fname_values,
    }
    return entry


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(__file__).resolve().parent.parent / "artifacts" / "multilayer_stdlib"
    out_dir.mkdir(parents=True, exist_ok=True)
    weights_dir = out_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)

    layers = [0, 4, 8, 12, 16, 20, 24]
    method = "clustered"
    n_clusters = 1024

    print(f"Loading Qwen2.5-1.5B...")
    collector = QwenActivationCollector(
        model_name="Qwen/Qwen2.5-1.5B",
        layers=layers,
        device=device,
    )

    cos, sin = extract_rotary_embeddings(collector.model)

    entries = []

    for layer in layers:
        print(f"\nLayer {layer}:")

        attn_weights = extract_layer_attention_weights(collector.model, layer)
        entry_attn = build_entry_attention(layer, attn_weights, cos, sin, weights_dir)
        entries.append(entry_attn)
        print(f"  Attention: {len(attn_weights)} weight tensors")

        kv_data = extract_mlp_keys_values(
            collector.model, layer, method=method, n_clusters=n_clusters
        )
        entry_mlp = build_entry_mlp(layer, kv_data, weights_dir)
        entries.append(entry_mlp)
        print(f"  MLP: {kv_data['n_keys']} key-value pairs ({kv_data['method']})")

    stdlib_path = out_dir / "stdlib.uvm"
    save_stdlib_json(entries, stdlib_path)

    metadata = {
        "model": "Qwen2.5-1.5B",
        "layers": layers,
        "mlp_method": method,
        "n_clusters": n_clusters,
        "n_entries": len(entries),
        "weights_dir": str(weights_dir),
    }
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nMulti-layer stdlib built: {len(entries)} entries")
    print(f"Saved to {stdlib_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
