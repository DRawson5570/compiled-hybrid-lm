"""
Phase 2: Find a copy head in Qwen2.5-1.5B, train SAE, extract feature vectors.

This script:
1. Loads Qwen2.5-1.5B
2. Probes attention patterns to find heads that attend to previous position
3. Trains a Sparse Autoencoder on residual stream activations from the top copy-head layer
4. Extracts feature vectors and saves them in stdlib format
"""

import json
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ucn.decompile.source_model import QwenActivationCollector
from ucn.decompile.copy_head_finder import find_copy_heads, measure_copy_fidelity
from ucn.decompile.sae import (
    SparseAutoencoder,
    extract_feature_vectors,
    train_sae_on_layer_activations,
    train_sae,
)
from ucn.stdlib.loader import save_stdlib_json, save_weight_tensor
from ucn.stdlib.schema import BehaviorMeta, MathDef, PrimitiveEntry


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    out_dir = Path(__file__).resolve().parent.parent / "artifacts" / "copy_head_extraction"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading Qwen2.5-1.5B...")
    collector = QwenActivationCollector(
        model_name="Qwen/Qwen2.5-1.5B",
        layers=[0, 4, 8, 12, 16, 20, 24],
        device=device,
    )
    print(f"  d_model={collector.d_model}, n_layers={collector.n_layers}, n_heads={collector.n_heads}")

    print("\n--- Step 1: Find copy heads ---")
    candidates = find_copy_heads(collector, n_top=10, max_texts=20, max_length=64)

    if not candidates:
        print("ERROR: No copy head candidates found. Checking attention patterns...")
        text = "The cat sat on the mat and looked around."
        attn = collector.collect_attention_from_layer([text], layers=[0, 4, 8, 12])
        for layer_idx, patterns in attn.items():
            if patterns:
                p = patterns[0]
                print(f"  Layer {layer_idx}: attention shape={p.shape}")
        return 1

    print("\nTop copy head candidates:")
    print(f"{'Layer':>6} {'Head':>6} {'Prev Attn':>12} {'Copy Str':>12}")
    print("-" * 42)
    for c in candidates:
        print(f"{c.layer:>6} {c.head:>6} {c.prev_token_attention:>12.4f} {c.copy_strength:>12.4f}")

    best = candidates[0]
    print(f"\nBest copy head: layer={best.layer}, head={best.head}")

    fidelity = measure_copy_fidelity(collector, best.layer, best.head, n_texts=10)
    print(f"  Fidelity: prev_attn={fidelity['prev_attention']:.4f}, diag_attn={fidelity['diag_attention']:.4f}")

    print("\n--- Step 2: Collect activations for SAE ---")
    texts = _get_eval_texts()
    all_residual = collector.collect_residual_stream(texts, max_length=128, batch_size=1)
    acts = all_residual.get(best.layer)
    if acts is None:
        print(f"ERROR: No activations for layer {best.layer}")
        return 1

    acts_flat = acts.reshape(-1, acts.shape[-1]).to(dtype=torch.float32)
    print(f"  Got {acts_flat.shape[0]} activation vectors of dim {acts_flat.shape[1]}")

    d_model = collector.d_model
    n_features = 256
    n_samples = min(acts_flat.shape[0], 500)
    acts_sample = acts_flat[:n_samples]

    print(f"  Using {n_samples} samples, d_model={d_model}, n_features={n_features}")

    if torch.isnan(acts_sample).any():
        print("  WARNING: activations contain NaN, filtering...")
        acts_sample = acts_sample[~torch.isnan(acts_sample).any(dim=-1)]
        n_samples = acts_sample.shape[0]
        print(f"  After filtering: {n_samples} vectors")

    if torch.isinf(acts_sample).any():
        print("  WARNING: activations contain Inf, filtering...")
        acts_sample = acts_sample[~torch.isinf(acts_sample).any(dim=-1)]
        n_samples = acts_sample.shape[0]

    sae = SparseAutoencoder(
        d_model=d_model,
        n_features=n_features,
        l1_lambda=1e-4,
    )

    history = train_sae(
        sae,
        acts_sample,
        steps=2000,
        lr=3e-4,
        batch_size=min(64, n_samples),
        device=device,
    )

    print("\n--- Step 3: Extract feature vectors ---")
    features = extract_feature_vectors(sae)

    active_features = {k: v for k, v in features.items() if v.norm() > 0.01}
    print(f"  Total features: {len(features)}, active: {len(active_features)}")

    top_features = sorted(active_features.items(), key=lambda x: x[1].norm(), reverse=True)[:100]
    print(f"  Using top 100 features by norm (range: {top_features[-1][1].norm():.4f} - {top_features[0][1].norm():.4f})")

    print("\n--- Step 4: Save stdlib artifacts ---")
    entries = []

    weights_dir = out_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)

    for idx, (feat_idx, vec) in enumerate(top_features):
        vec = vec.detach().cpu().to(torch.float32)
        vec = vec / vec.norm()

        prim_id = f"PRM_COPY_L{best.layer}_F{feat_idx:04d}"
        entry = PrimitiveEntry(
            primitive_id=prim_id,
            symbolic_name=f"copy_head_L{best.layer}_feature_{feat_idx}",
            type="latent_feature",
            source_layers=[best.layer],
            math_def=MathDef(
                operator_type="direction_vector",
                vector_uri=f"weights/{prim_id}_vec.pt",
            ),
            behavior=BehaviorMeta(
                description=f"Latent feature {feat_idx} from copy head at layer {best.layer}, head {best.head}",
                trigger_conditions=[],
            ),
        )

        save_weight_tensor(vec, weights_dir / f"{prim_id}_vec.pt")

        entries.append(entry)

    stdlib_path = out_dir / "stdlib.uvm"
    save_stdlib_json(entries, stdlib_path)

    copy_head_data = {
        "layer": best.layer,
        "head": best.head,
        "d_model": d_model,
        "n_layers": collector.n_layers,
        "n_heads": collector.n_heads,
        "copy_strength": best.copy_strength,
        "prev_attention": best.prev_token_attention,
        "fidelity": fidelity,
        "sae_config": {
            "d_model": d_model,
            "n_features": n_features,
            "l1_lambda": 1e-3,
        },
        "n_extracted_features": len(entries),
        "stdlib_path": str(stdlib_path),
        "weights_dir": str(weights_dir),
    }

    with open(out_dir / "copy_head_info.json", "w") as f:
        json.dump(copy_head_data, f, indent=2)

    print(f"\nSaved to {out_dir}:")
    print(f"  {len(entries)} primitives in {stdlib_path}")
    print(f"  {len(entries) * 2} weight files in {weights_dir}")
    print(f"  copy_head_info.json")
    print("\nPhase 2 complete!")

    return 0


def _get_eval_texts() -> list[str]:
    return [
        "The cat sat on the mat and looked around the room with curiosity.",
        "Machine learning enables computers to learn from data without explicit programming.",
        "The capital of France is Paris, a city known for its art and culture.",
        "Python is a high-level programming language used for web development and data science.",
        "The quick brown fox jumps over the lazy dog near the river bank.",
        "Neural networks consist of layers of interconnected nodes that process information.",
        "The Earth orbits the Sun at an average distance of about 93 million miles.",
        "Shakespeare wrote many famous plays including Hamlet and Romeo and Juliet.",
        "Water boils at 100 degrees Celsius and freezes at 0 degrees Celsius under normal pressure.",
        "The human brain contains approximately 86 billion neurons connected by synapses.",
        "Einstein developed the theory of relativity which revolutionized modern physics.",
        "The Amazon rainforest produces about 20 percent of the world oxygen supply.",
        "Deep learning models require large amounts of data and computational resources for training.",
        "The Great Wall of China is over 13000 miles long and was built over many centuries.",
        "Photosynthesis is the process by which plants convert sunlight into chemical energy.",
        "JavaScript is commonly used for front end web development alongside HTML and CSS.",
        "The speed of light in a vacuum is approximately 299792458 meters per second.",
        "Climate change poses significant risks to ecosystems and human societies around the world.",
        "DNA contains the genetic instructions for the development and function of living organisms.",
        "Blockchain technology enables decentralized and secure digital transactions without intermediaries.",
    ]


if __name__ == "__main__":
    sys.exit(main())
