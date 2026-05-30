# NRTCS Architecture

## Neurosymbolic Round-Trip Compilation Stack — System Design

---

## 1. Overview

NRTCS implements a **deterministic, gradient-free weight-editing pipeline** for neural networks. The system reads a `.safetensors` model file, decompiles it into a symbolic representation, allows edits to that representation, recompiles the edits back into weight matrices, and splices them into the original file via memory mapping.

```
                       ┌──────────────────────────┐
                       │     source.safetensors    │
                       └────────────┬─────────────┘
                                    │
            ┌───────────────────────┼───────────────────────┐
            │                       ▼                       │
            │  ┌──────────────────────────────────────┐     │
            │  │   Phase 1: Decompiler (C2S)          │     │
            │  │   SAE feature extraction + circuit   │     │
            │  │   attribution → symbolic features    │     │
            │  └──────────────────┬───────────────────┘     │
            │                     │                         │
            │                     ▼                         │
            │  ┌──────────────────────────────────────┐     │
            │  │   Phase 2: Refactoring (UVM-DSL)     │     │
            │  │   Developer edits symbolic features  │     │
            │  │   (Python API or DSL)                │     │
            │  └──────────────────┬───────────────────┘     │
            │                     │                         │
            │                     ▼                         │
            │  ┌──────────────────────────────────────┐     │
            │  │   Phase 3: Recompiler (S2C)          │     │
            │  │   Analytical matrix construction     │     │
            │  │   + orthogonal crosstalk prevention  │     │
            │  └──────────────────┬───────────────────┘     │
            │                     │                         │
            │                     ▼                         │
            │  ┌──────────────────────────────────────┐     │
            │  │   Phase 4: Binary Splicer            │     │
            │  │   mmap inline tensor replacement     │     │
            │  └──────────────────┬───────────────────┘     │
            │                     │                         │
            │                     ▼                         │
            │              patched_model.safetensors        │
            └───────────────────────────────────────────────┘
```

---

## 2. Component Deep Dive

### 2.1 Decompiler (`decompiler.py`) — Phase 1

**Purpose:** Convert continuous neural parameters into discrete, interpretable features.

**Classes:**
- `NRTCSDecompiler` — main decompiler, wraps a model + SAEs

**Key algorithms:**

#### Feature extraction (C2S)
```
For each layer with a trained SAE:
  1. Run model forward on input text
  2. Capture residual stream activation at that layer
  3. Encode through SAE: h = ReLU(W_enc @ x + b_enc)
  4. Keep features where h_i > τ (threshold)
  5. Return: feature indices, decoder vectors, activation strengths
```

#### Path attribution
```
For upstream layer L_u → downstream layer L_d:
  1. Run model forward (gradient tracking ON)
  2. Capture hidden states at L_u (requires_grad=True) and L_d
  3. Encode both through respective SAEs
  4. Compute y = Σ SAE_down(L_d_hidden)[:downstream_feature]
  5. ∇x = ∂y/∂(L_u_hidden)  — full chain rule through model layers
  6. For each upstream feature j:
     A(x_j → y) = (h_j) · (∇x @ w_dec_j)
```

**Constraints:**
- Requires pretrained SAEs per layer (trained via `full_compiled_experiment/ucn/decompile/sae.py`)
- `path_attribution` runs with gradient tracking → memory cost scales with model size
- Hook registration is compatible with HuggingFace model API patterns

**Integration points:**
- Uses `full_compiled_experiment/ucn/decompile/sae.py` for SAE class
- Uses `full_compiled_experiment/ucn/decompile/source_model.py` pattern for activation collection
- Compatible with any HF model that exposes `model.layers[]` or `transformer.h[]`

---

### 2.2 Recompiler (`recompiler.py`) — Phase 3

**Purpose:** Convert symbolic key-value pairs into numerical FFN weight matrices.

**Functions:**
- `build_dense_map(keys, values, eps)` — associative memory matrix construction
- `orthogonal_projection(W_compiled, U, eps)` — crosstalk prevention
- `compute_null_space_rank(U, eps)` — capacity tracking
- `verify_dense_map(keys, W_down, W_up)` — reconstruction verification

**Class:**
- `RecompilerEngine` — wraps the above with a consistent interface

#### Algorithm: Analytical Matrix Construction

```
Input:  K ∈ R^{N×d_in}  (N key vectors, each of dimension d_in)
        V ∈ R^{N×d_out} (N value vectors, each of dimension d_out)

1. Compute Gram matrix: G = K @ K^T  ∈ R^{N×N}
2. Regularize:           G_reg = G + ε·I
3. Cholesky decompose:   L = cholesky(G_reg)
4. Invert:               G_reg^{-1} = cholesky_inverse(L)
5. Construct W_down:     W_down = K^T @ G_reg^{-1}  ∈ R^{d_in×N}
6. Construct W_up:       W_up = V                    ∈ R^{N×d_out}

Verification:  K @ W_down @ W_up
             = K @ K^T @ (K@K^T)^{-1} @ V
             = I @ V
             = V   ✓
```

**Why Cholesky and not `torch.linalg.inv`?**
- Cholesky is O(N³/3) vs O(N³) for general inverse
- Better numerical stability for symmetric positive definite matrices
- Built-in positive definiteness check (fails cleanly if singular)

#### Algorithm: Orthogonal Subspace Projection

```
Input:  W_compiled ∈ R^{d×k}   (compiled weight matrix)
        U ∈ R^{d×m}            (original active feature vectors)

1. Compute feature covariance:  C = U^T @ U  ∈ R^{m×m}
2. Regularize:                  C_reg = C + ε·I
3. Cholesky invert:             C_reg^{-1}
4. Build orthogonal projector:  P_perp = I_d - U @ C_reg^{-1} @ U^T
5. Project compiled weights:    W_final = P_perp @ W_compiled

Verification:  U^T @ W_final = U^T @ (I - U@C^{-1}@U^T) @ W_compiled
                             = (U^T - U^T@U@C^{-1}@U^T) @ W_compiled
                             = (U^T - C@C^{-1}@U^T) @ W_compiled
                             = (U^T - U^T) @ W_compiled
                             = 0   ✓
```

**Design decision — only W_down is projected:**
W_up (= V) contains the raw value vectors. Projecting W_up would distort the values. Instead, only W_down is projected into the orthogonal subspace, ensuring that keys cannot trigger the compiled mapping from within the protected feature space.

**Dimension saturation tracking:**
```
null_space_rank = rank(P_perp) = d - rank(U)

When null_space_rank < 0.10 * d → alert: dimension compaction needed
```

**Numerical properties:**
- All computation in `float32` (per AGENTS.md fp16 safety rule)
- ε defaults to 1e-6, increase for near-collinear features
- Cholesky will raise `RuntimeError` if ε is too small for ill-conditioned Gram matrix

---

### 2.3 Binary Splicer (`splicer.py`) — Phase 4

**Purpose:** Replace tensor payloads in `.safetensors` files without rewriting.

**Class:**
- `SafetensorsSplicer` — context-managed mmap file handler

#### Safetensors File Layout

```
Offset 0:   [u64 LE: header_length]
Offset 8:   [header_length bytes: UTF-8 JSON]
Offset 8+N: [tensor_0 payload] [tensor_1 payload] ...
```

JSON header structure:
```json
{
  "tensor.name": {
    "dtype": "F32" | "F16" | "BF16" | "F64" | "I32" | ...,
    "shape": [dim1, dim2, ...],
    "data_offsets": [start_byte, end_byte]
  }
}
```

#### mmap Splicing Protocol

```
1. open(path, "r+b")              — read+write binary
2. mmap(fileno(), 0)              — map entire file
3. header_len = struct.unpack("<Q", mm[0:8])[0]
4. header = json.loads(mm[8:8+header_len])
5. tensor_start = 8 + header_len + header[name]["data_offsets"][0]
6. tensor_end   = 8 + header_len + header[name]["data_offsets"][1]
7. assert len(new_data) == tensor_end - tensor_start
8. mm[tensor_start:tensor_end] = new_data
9. mm.flush()
```

**Safety properties:**
- **Shape validation**: refuses to splice if `new_data.shape != original.shape` (not just byte count — actual shape must match to prevent silent data corruption)
- **No header modifications**: the JSON header (shapes, dtypes, offsets) remains unchanged
- **Atomic at flush boundary**: `mm.flush()` commits changes; if the process crashes before flush, the file is unaffected
- **Context manager**: `with SafetensorsSplicer(path) as spl:` ensures cleanup

**Limitations:**
- Cannot change tensor shapes (would require header rewrite)
- Cannot add or remove tensors
- File must exist (no creation)

---

### 2.4 Pipeline (`pipeline.py`) — Orchestrator

**Purpose:** Wire phases 3+4 together. Phase 1+2 are used independently.

**Class:**
- `NRTCSPipeline` — holds a `RecompilerEngine`, provides convenience methods

**Method relationships:**
```
compile_dense_map()
  └── RecompilerEngine.compile()

compile_from_uvm_edits()
  └── for each layer: RecompilerEngine.compile()

splice_patches()
  └── for each layer: SafetensorsSplicer.splice_mlp()

round_trip()  = compile_from_uvm_edits() + splice_patches()

verify_compilation()
  └── for each layer: keys @ W_down @ W_up ≈ values
```

---

## 3. Data Flow

### End-to-end round-trip walkthrough

```
1. USER provides:
   - edits = {layer: {"keys": (N, d_in), "values": (N, d_out)}}
   - safetensors_path

2. compile_from_uvm_edits(edits):
   For each layer L:
     K = edits[L]["keys"]          # shape (N, d_in)
     V = edits[L]["values"]        # shape (N, d_out)
     W_down, W_up = build_dense_map(K, V)
     If original_features[L] exists:
       W_down = orthogonal_projection(W_down, original_features[L])
     patches[L] = {"W_down": W_down, "W_up": W_up}

3. splice_patches(safetensors_path, patches):
   Open safetensors via mmap
   For each patch:
     Convert W_down from float32 → target dtype
     Convert to raw bytes
     Write at correct offset in mmap region
     Repeat for W_up
   mm.flush()
   Close

4. verify_compilation(edits, patches):
   For each layer:
     recon = K @ W_down @ W_up
     error = ||recon - V||
     cosine = similarity(recon, V)

5. Reload model from safetensors → verify behavior change
```

## 4. Key Design Decisions

### 4.1 float32-only computation

All analytical math (matrix inversion, projection) runs in float32. Inputs are auto-cast. This avoids:
- fp16 underflow (min subnormal ~6e-8)
- fp16 overflow (max ~65504)
- Silent NaN propagation in RMS norm (per AGENTS.md rule #2)

Output can be cast back to fp16/bf16 during splicing if the target model uses those dtypes.

### 4.2 Cholesky instead of direct inversion

`torch.linalg.inv()` uses LU decomposition, which is 2x slower for SPD matrices and less numerically stable. Cholesky decomposition:
- Exploits symmetry: only computes lower triangle
- Fails explicitly if matrix is not positive definite (serves as a validation check)
- `torch.cholesky_inverse(L)` is a single fused operation

### 4.3 W_up = V (not V^T)

The NRTCS spec says `W_up = V^T`, but the math requires `W_up` to have shape `(N, d_out)` for the verification `K @ W_down @ W_up = V` to hold. The implementation uses `W_up = V` which gives the correct shape. This is a notational correction from the spec.

### 4.4 No W_up projection in crosstalk prevention

Only W_down is projected into the orthogonal subspace. Projecting W_up would distort the value vectors, breaking the associative memory. The protection comes from W_down being in the orthogonal subspace: if a key overlaps with a protected feature, the key's projection through W_down is attenuated, preventing the mapping from firing.

### 4.5 Shape validation over byte-count validation

The original spec's splicer only checks byte count (`verify_shape`). We additionally validate the full shape tuple. A tensor reshaped from `(768, 3072)` to `(3072, 768)` has the same byte count but wrong data layout — this would silently corrupt the model.

---

## 5. Relationship to Surrounding Codebase

### Reused from `full_compiled_experiment/`

| Module | What it provides |
|--------|-----------------|
| `ucn/decompile/sae.py` | `SparseAutoencoder` class, `train_sae()`, `normalize_decoder()` |
| `ucn/decompile/source_model.py` | `QwenActivationCollector` for HF model activations |
| `ucn/decompile/feature_analyzer.py` | Intervention testing utilities |
| `ucn/dsl/parser.py` | UVM-DSL text parser (future integration for Phase 2) |
| `ucn/dsl/ast.py` | UVM-DSL AST types (future integration) |

### Relation to hybrid/ compiled features

The 21-channel compiled features system (`channels_v3.py`, `superposition_steerer_v3.py`) is orthogonal to NRTCS:
- **Compiled features**: Static prior statistics injected at runtime via hooks — real-time, no weight modification
- **NRTCS**: Permanent weight modification via decompile→recompile→splice — offline, changes the model file

They can be combined: NRTCS patches the base model weights, then compiled features steer the patched model at runtime.

---

## 6. Safety Properties

| Property | Mechanism |
|----------|-----------|
| **Crosstalk prevention** | Orthogonal projection `P_perp @ W_compiled` ensures new mappings don't activate from protected feature space |
| **Dimension capacity tracking** | `compute_null_space_rank()` monitors remaining orthogonal dimensions — alerts when < 10% of d_model |
| **Numerical stability** | float32 + Cholesky + Tikhonov regularization (ε·I) |
| **File integrity** | mmap flush semantics — crash-safe at OS level |
| **Shape corruption guard** | Array shape validation before splicing prevents silent data corruption |
| **Idempotent projection** | `P_perp @ P_perp = P_perp` — applying projection twice is safe |
| **Dtype safety** | Auto dtype conversion when splicing based on target tensor's declared dtype |

---

## 7. Testing Strategy

### Unit tests (34 total)
- **test_recompiler.py (14)**: BuildDenseMap (6), OrthogonalProjection (4), RecompilerEngine (4)
- **test_splicer.py (10)**: Open/parse, read, splice, error paths, convenience functions, cross-layer isolation
- **test_pipeline.py (10)**: Compile, crosstalk, multilayer, verification, splice round-trip, France→Paris walkthrough

### Test coverage
- Core math: Every matrix construction path, edge cases (singular matrices, regularization)
- File I/O: Create temp safetensors, splice, reload, verify exact match
- Error paths: Shape mismatch, nonexistent tensors, closed handles
- Integration: Full round-trip (compile → splice → reload → verify)

### Gap: Decompiler tests
The decompiler requires a loaded model + pretrained SAEs, which need GPU and significant setup. These are integration-tested in `full_compiled_experiment/` scripts (`phase5_benchmark.py`, etc.) rather than in `sae_editor/tests/`.
