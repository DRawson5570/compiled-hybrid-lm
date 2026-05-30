# NRTCS User Manual

## Neurosymbolic Round-Trip Compilation Stack

**Version:** 1.0 | **Status:** Operational

---

## 1. What is NRTCS?

NRTCS is a closed-loop pipeline for **decompiling, editing, and recompiling neural network weights without gradient-based training**. It lets you fix factual errors, inject safety gates, or remap associations in a pretrained model by editing the weights directly.

### Four phases

```
safetensors → [1. Decompile] → UVM-DSL → [2. Edit] → UVM-DSL → [3. Recompile] → weights → [4. Splice] → patched .safetensors
```

| Phase | Name | What it does |
|-------|------|--------------|
| 1 | **Decompiler (C2S)** | Reads model weights + activations, runs SAEs, extracts features and circuits |
| 2 | **Refactoring (UVM)** | You edit the symbolic representation (via Python or DSL) |
| 3 | **Recompiler (S2C)** | Compiles edited key-value pairs into FFN weight matrices with crosstalk prevention |
| 4 | **Splicer (Binary)** | Inline-patches `.safetensors` files via mmap — no file rewrite needed |

---

## 2. Quick Start

### Install

The package lives at `hybrid/sae_editor/`. It depends on:

```
torch, safetensors, numpy
```

And optionally reuses from `hybrid/full_compiled_experiment/ucn/` for SAE-based decompilation.

```bash
# Verify everything works
cd ~/deepseek_experiments/hybrid
python -m pytest sae_editor/tests/ -v

# 34 passed
```

### Hello World: Fix a single key-value mapping

```python
import torch
from sae_editor import NRTCSPipeline

# Create keys and values (shape: N × d)
france_key = torch.tensor([[0.9, -0.1, 0.1, 0.1, 0.0, 0.0, 0.0, 0.0]])
paris_value = torch.tensor([[0.1, 0.1, 0.9, 0.1, 0.0, 0.0, 0.0, 0.0]])

pipeline = NRTCSPipeline()
result = pipeline.compile_dense_map(france_key, paris_value)

# Verify reconstruction
recon = france_key @ result["W_down"] @ result["W_up"]
print(recon)  # Should match paris_value (within numerical precision)
```

---

## 3. Library API Reference

### 3.1 Recompiler (`sae_editor.recompiler`)

The core phase that turns symbolic specifications into numerical weights.

```python
from sae_editor.recompiler import build_dense_map, orthogonal_projection, RecompilerEngine

W_down, W_up = build_dense_map(keys, values, eps=1e-6)
# W_down: (d_in, N)   W_up: (N, d_out)
# Guarantees: keys @ W_down @ W_up ≈ values

W_protected = orthogonal_projection(W_compiled, original_features, eps=1e-6)
# Projects W_compiled into subspace orthogonal to original_features
# Guarantees: original_features.T @ W_protected ≈ 0

engine = RecompilerEngine(eps=1e-6)
result = engine.compile(keys, values, original_features=None)
# result = {"W_down": tensor, "W_up": tensor}
```

**Math:**
- `W_down = K^T @ (K@K^T + εI)^-1`  — Cholesky decomposition for numerical stability
- `W_up = V`                           — raw values
- `P_perp = I - U(U^T U + εI)^-1 U^T` — orthogonal projector
- All computation in `float32`

**Key constraint:** If you use `original_features`, the keys MUST be orthogonal to those features. Otherwise, the orthogonal projection destroys the key signal and reconstruction fails.

### 3.2 Splicer (`sae_editor.splicer`)

Inline `.safetensors` tensor patching via memory mapping.

```python
from sae_editor.splicer import SafetensorsSplicer, splice_tensor

# Single-shot convenience:
splice_tensor("model.safetensors", "model.layers.0.mlp.down_proj.weight", raw_bytes)

# Detailed usage:
with SafetensorsSplicer("model.safetensors") as spl:
    # List all tensors
    print(spl.tensor_names)

    # Inspect tensor
    info = spl.get_tensor_info("model.layers.0.mlp.down_proj.weight")
    # info = {"dtype": "F32", "shape": [768, 3072], "data_offsets": [128, 9437184]}

    # Read tensor as raw bytes
    data = spl.read_tensor("model.layers.0.mlp.down_proj.weight")

    # Replace tensor (must match size)
    spl.splice_tensor("model.layers.0.mlp.down_proj.weight", new_bytes)

    # Convenience: splice MLP weights from torch tensors
    spl.splice_mlp(
        layer=0,
        W_down=torch.randn(768, 3072),   # Must match original shape
        W_up=torch.randn(3072, 768),     # Must match original shape
        model_name="model.layers.{layer}.mlp",
    )
```

**Safetensors format:**
```
[8 bytes: u64 header_len LE]
[header_len bytes: JSON]
{
  "tensor_name": {
    "dtype": "F32",
    "shape": [d1, d2],
    "data_offsets": [start, end]
  }
}
[tensor payloads concatenated]
```

### 3.3 Decompiler (`sae_editor.decompiler`)

Extracts feature activations and circuits from a loaded model using SAEs.

```python
from sae_editor.decompiler import NRTCSDecompiler

decompiler = NRTCSDecompiler(
    model=hf_model,
    tokenizer=tokenizer,
    saes={0: sae_l0, 2: sae_l2, 5: sae_l5},  # layer_idx → SAE
    threshold=0.1,  # τ: features with activation > 0.1 are kept
    device="cuda",
)

# Extract features from text
features = decompiler.extract_features(
    texts=["The capital of France is"],
    max_length=128,
)
# features[layer_idx] = {
#     "activations": (B, T, d_model),
#     "feature_indices": (K,),
#     "feature_vectors": (K, d_model),
#     "feature_acts": (B, T, K),
# }

# Compute path attributions between layers
attr = decompiler.path_attribution(
    text="The capital of France is",
    upstream_layer=0,
    downstream_layer=2,
    upstream_features=[3, 7, 12],  # or None for all
    downstream_feature=5,           # or None for all
)
# attr = {
#     "attributions": (N_up,),
#     "upstream_indices": ...,
#     "downstream_indices": ...,
#     "upstream_acts": ...,
#     "downstream_acts": ...,
# }

# Raw activations (no SAE)
activations = decompiler.collect_activations(texts, max_length=128)
# {layer_idx: (B, T, d_model)}
```

**Memory note:** `path_attribution` runs the model forward pass with gradient tracking (no `no_grad()`), so it requires full activation memory. For single-text analysis on models up to ~7B, this is manageable. For larger models, use `extract_features` (no gradient tracking) instead.

### 3.4 Pipeline (`sae_editor.pipeline`)

Orchestrates recompiler + splicer into a single round-trip.

```python
from sae_editor.pipeline import NRTCSPipeline

pipeline = NRTCSPipeline(eps=1e-6)

# Compile edits → per-layer weight patches
edits = {
    0: {"keys": keys_0, "values": values_0},
    2: {"keys": keys_2, "values": values_2},
}
patches = pipeline.compile_from_uvm_edits(edits)
# patches[layer_idx] = {"W_down": ..., "W_up": ...}

# Splice into safetensors
pipeline.splice_patches("model.safetensors", patches)

# Full round-trip in one call
pipeline.round_trip("model.safetensors", edits)

# Verify reconstruction quality
results = pipeline.verify_compilation(edits, patches)
# results[layer_idx] = {
#     "max_error": 1.2e-5,
#     "mean_error": 3e-6,
#     "min_cosine": 0.9999,
#     "mean_cosine": 0.99999,
# }
```

---

## 4. CLI Reference

```bash
cd ~/deepseek_experiments/hybrid
export PYTHONPATH="$PWD:$PYTHONPATH"
python -m sae_editor.cli <command> [options]
```

### `recompile` — Phase 3: compile edits to weight patches

```bash
python -m sae_editor.cli recompile edits.pt --output patches/ --eps 1e-6
```

Input: `edits.pt` — a dict saved with `torch.save()`:
```python
# Single-layer format (auto-wrapped to layer 0):
{"keys": tensor, "values": tensor}

# Multi-layer format:
{0: {"keys": tensor, "values": tensor}, 2: {...}, 5: {...}}
```

Output: Directory `patches/layer_0/W_down.pt`, `patches/layer_0/W_up.pt`, etc.

### `splice` — Phase 4: inspect or patch safetensors

```bash
# List all MLP tensors
python -m sae_editor.cli splice model.safetensors --tensor-name mlp

# Patch from pre-compiled weight directory
python -m sae_editor.cli splice model.safetensors --tensor-name mlp --patch-path patches/
```

### `roundtrip` — Full compile + splice

```bash
python -m sae_editor.cli roundtrip edits.pt model.safetensors --features-path original_features.pt
```

### `decompile` — Phase 1 (stub)

Currently a stub. Use the `NRTCSDecompiler` class programmatically.

---

## 5. Recipes

### Recipe 1: Fix a factual error (France → Paris)

The spec's concrete walkthrough — fix "capital of France = London" → "Paris".

```python
import torch
from sae_editor.pipeline import NRTCSPipeline

d = 768  # match your model's d_model

# Step 1: Decompile (find the faulty key/value vectors in layer 14)
#   Run NRTCSDecompiler on "The capital of France is" → identify
#   the SAE features that encode "France" → "London" mapping.
#   For this example, we'll use synthetic vectors:

france_key = torch.tensor([[0.9, -0.1, 0.1, 0.1] + [0.0]*(d-4)])
paris_value = torch.tensor([[0.1, 0.1, 0.9, 0.1] + [0.0]*(d-4)])

# Step 2: Edit — we change London value to Paris value (done above)

# Step 3: Recompile
pipeline = NRTCSPipeline()
edits = {14: {"keys": france_key, "values": paris_value}}

# Step 3a: Without crosstalk prevention (simplest)
patches = pipeline.compile_from_uvm_edits(edits)

# Step 3b: With crosstalk prevention (protect existing features)
#   First, pass the France key through the SAE on original activations
#   to get original_features = U (other active features in layer 14).
#   Make sure the France key is orthogonal to U.
original_features = torch.randn(d, 20)  # 20 other features in layer 14
# Zero out the France key dimensions in U so they're orthogonal:
original_features[:4, :] = 0.0

patches_protected = pipeline.compile_from_uvm_edits(edits, {14: original_features})

# Step 4: Splice
pipeline.splice_patches("model.safetensors", patches_protected)

# Step 5: Verify reconstruction quality
results = pipeline.verify_compilation(edits, patches_protected)
print(f"Layer 14 mean cosine: {results[14]['mean_cosine']}")
# Should be > 0.9999
```

### Recipe 2: Splice new weights into a model without PyTorch

```python
import numpy as np
from sae_editor.splicer import SafetensorsSplicer

with SafetensorsSplicer("model.safetensors") as spl:
    new_weights = np.ones((768, 3072), dtype="float32")
    new_bytes = new_weights.tobytes()
    spl.splice_tensor("model.layers.0.mlp.down_proj.weight", new_bytes)
```

### Recipe 3: Build a dense associative memory

```python
import torch
from sae_editor.recompiler import build_dense_map

N, d_in, d_out = 50, 768, 768

# Random key-value store
keys = torch.randn(N, d_in)
values = torch.randn(N, d_out)

W_down, W_up = build_dense_map(keys, values, eps=1e-4)

# Look up all items
reconstructed = keys @ W_down @ W_up  # shape (50, 768)
error = (reconstructed - values).norm(dim=-1)
print(f"Max error: {error.max():.6f}, Mean: {error.mean():.6f}")
```

---

## 6. Numerical Considerations

### Dtype rules (from AGENTS.md)

- **All matrix construction is in float32**. Inputs are auto-cast.
- **Cholesky decomposition** is used for matrix inversion — requires positive definiteness, ensured by `eps * I` regularization.
- **Orthogonal projection** can destroy signal if keys overlap with protected features. If `eps` is too small and features are nearly collinear, Cholesky may fail.

### Tuning `eps`

| Scenario | Recommended `eps` |
|----------|-------------------|
| Well-conditioned keys (orthonormal) | 1e-6 |
| Random keys (typical) | 1e-4 |
| Near-duplicate keys | 1e-2 |
| Very large d_in (>4096) | 1e-3 (floating point accumulation) |

### Memory

- `build_dense_map` memory: O(N² + N·d). For N=100, d=4096: ~16 MB.
- `orthogonal_projection` memory: O(d²). For d=4096: ~64 MB.
- `path_attribution` memory: Full model forward pass activations (varies by model).

---

## 7. Testing

```bash
# All tests (34 total)
python -m pytest sae_editor/tests/ -v

# Categories
python -m pytest sae_editor/tests/test_recompiler.py -v   # 14 tests
python -m pytest sae_editor/tests/test_splicer.py -v       # 10 tests
python -m pytest sae_editor/tests/test_pipeline.py -v      # 10 tests

# With coverage
python -m pytest sae_editor/tests/ --cov=sae_editor --cov-report=term
```

---

## 8. Dependency Map

```
sae_editor/
├── recompiler.py    → torch (standalone)
├── splicer.py       → safetensors, mmap (standalone)
├── pipeline.py      → recompiler, splicer
├── decompiler.py    → torch (standalone, uses SAEs from full_compiled_experiment)
├── cli.py           → pipeline, recompiler, splicer
└── tests/           → recompiler, splicer, pipeline (+ safetensors.torch for test fixtures)
```

The recompiler and splicer have **zero internal dependencies** — they can be used independently in any project that has `torch` and `safetensors`.

---

## 9. Error Reference

| Error | Cause | Fix |
|-------|-------|-----|
| `ValueError: Mismatch: N keys vs M values` | Different number of keys and values | Each key needs exactly one value |
| `RuntimeError: mat1 and mat2 shapes cannot be multiplied` | Shape mismatch in verification | Check W_down/W_up shapes: (d_in, N) and (N, d_out) |
| `RuntimeError: ... cholesky ...` | Gram matrix not positive definite | Increase `eps` |
| `KeyError: Tensor '...' not found in safetensors header` | Wrong tensor name | Use `spl.tensor_names` to list valid names |
| `AssertionError: Size mismatch` | New tensor is different byte size | Match the original tensor's byte count exactly |
| `ValueError: Shape mismatch` | New tensor has wrong shape but right byte count | Match shape exactly (not just element count) |
| `RuntimeError: Not opened` | Used `SafetensorsSplicer` without `open()` or context manager | Use `with SafetensorsSplicer(path) as spl:` |
