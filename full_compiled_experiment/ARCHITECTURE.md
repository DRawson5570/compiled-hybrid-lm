# UCN Architecture

## Unified Compiled Network — Architecture Document

**Status:** Implemented (Phases 0-5 complete)  
**Last updated:** 2026-05-30

---

## 1. Executive Summary

The Unified Compiled Network (UCN) replaces the static, sequential Transformer architecture with a compiler-based execution model. Instead of every token passing through identical matrix multiplications at every layer, a lightweight Meta-Compiler (neural frontend) analyzes input context and emits a token-specific executable program in a Vector Domain-Specific Language (UVM-DSL). A JIT Backend Compiler lowers this program to hardware-optimized kernels via Triton/CUDA, resolves symbols against a pre-built Standard Library (`stdlib.uvm`), and executes the fused kernel on a virtual tensor workspace.

The standard library is populated by reverse-engineering a pretrained Transformer (Qwen2.5-1.5B) using Sparse Autoencoders and mechanistic interpretability, extracting disentangled feature directions as named primitives.

This architecture achieves perfect fidelity (1.0000 cosine similarity, 0.0000 MSE) when compiling and executing real Qwen2.5-1.5B attention layers through the UCN pipeline, demonstrating that static neural network computation can be fully expressed as dynamically compiled programs.

---

## 2. System Topology

```
                              ┌──────────────────────────┐
                              │   Standard Library        │
                              │   (stdlib.uvm + weights)  │
                              └────────────┬─────────────┘
                                           │ symbol resolution
                                           ▼
┌──────────────────┐    UVM-DSL    ┌───────────────┐    Triton/CUDA    ┌──────────────┐
│  Meta-Compiler   │ ──────────►  │ JIT Backend   │ ───────────────► │  Runtime     │
│  (Neural Front)  │   Program    │  Compiler     │   Fused Kernel   │  Executor    │
│                  │              │               │                  │              │
│  Context Analyzer│              │ L1 Cache      │                  │  Workspace   │
│  Template Select │              │ L2 Cache      │                  │  Param DB    │
│  Param Generator │              │ Optimizer     │                  │              │
└────────┬─────────┘              │ Codegen       │                  └──────┬───────┘
         │                        │  - Reference  │                         │
         │                        │  - Triton     │                         │
         │                        │  - C (PoC)    │                         │
         │                        └───────────────┘                         │
         │                                                                  │
         │  Input Tokens (X)                                                │
         ▼                                                                  ▼
┌──────────────────────────────────────────────┐              ┌──────────────────────┐
│              Decompilation Pipeline          │              │   Output State (Y)   │
│                                              │              └──────────────────────┘
│  QwenActivationCollector                     │
│  SparseAutoencoder (SAE)                     │
│  Circuit Discovery (copy head finder)        │
│  Feature Analyzer (intervention tools)       │
│  stdlib Builder (SAE → .uvm + .pt)           │
└──────────────────────────────────────────────┘
```

---

## 3. Component Architecture

### 3.1 DSL Layer (`ucn/dsl/`)

The UVM-DSL is a strongly typed intermediate language for expressing vector transformations.

**AST Nodes** (`ast.py`):
| Primitive | Signature | Purpose |
|-----------|-----------|---------|
| `mix(inputs, weights)` | [T,D] → [T,D] | Weighted sum of vectors |
| `project(input, subspace)` | [T,D] → [T,D] | Mask to subspace coordinates |
| `transform(input, matrix_ref)` | [T,D] → [T,D] | Matrix multiply (stdlib or dynamic) |
| `activate(input, type)` | [T,D] → [T,D] | GELU/ReLU/SiLU/Identity |
| `query_memory(input, db, top_k)` | [T,D] → [T,D] | Sparse key-value DB lookup |
| `residual(inputs)` | [T,D]... → [T,D] | Sum accumulation (fused) |
| `rotate(input, theta, subspace)` | [T,D] → [T,D] | RoPE-style subspace rotation |

**Parser** (`parser.py`): Recursive-descent parser for the text DSL grammar. Full BNF grammar from [FULLY_COMPILED_SPEC.md](FULLY_COMPILED_SPEC.md) §3.2.

**Types** (`types.py`): `Vector<D>`, `Subspace<K,D>`, `Scalar`, `Matrix<R,C>`.

### 3.2 Backend Layer (`ucn/backend/`)

Two code generation targets, plus compilation orchestration.

#### Reference Backend (`codegen/reference.py`)
Pure PyTorch execution. Serves as the **golden correctness reference**. All other backends must produce identical output to within float32 epsilon. Supports all 7 primitives plus the `multihead_attention` operator type for full Transformer attention.

Key design: executes programs by walking the AST and dispatching to typed `_execute_*` methods. Workspace is a Python dict of tensors. Fully differentiable for training.

#### Triton Backend (`codegen/triton_backend.py`)
GPU kernel codegen using Triton 3.5.1. Each UVM-DSL primitive maps to a `@triton.jit` kernel using block-parallel execution over d_model dimensions. Fuses consecutive Transform+Activate pairs into single kernels to avoid HBM writebacks. Includes a special `direction_vector` transform for additive feature injection.

#### JIT Compiler (`jit_compiler.py`)
Top-level orchestration: AST → compile → execute.
- Falls back to Reference backend if Triton compilation fails
- `_extract_params()` collects weight tensors from Transform nodes

#### Cache (`cache.py`)
Two-level JIT caching:
- **L1 Structural Cache**: Keys off AST topology hash (MurmurHash3 of operation types). Reuses compiled kernel binaries across identical program structures with different parameters.
- **L2 Semantic Cache**: Locality-sensitive hashing on context vector z. Allows bypassing compilation entirely when context is similar to a cached entry.

#### Optimizer (`optimizer.py`)
IR optimization passes applied before codegen:
1. Dead Code Elimination: removes statements whose outputs are never used
2. Subspace Pruning: adjusts loop bounds based on project operations
3. Operator Fusion: merges Transform+Activate and Mix+Activate pairs

### 3.3 Runtime Layer (`ucn/runtime/`)

#### TensorWorkspace (`workspace.py`)
Virtual tensor workspace modeling on-chip SRAM. Allocates and tracks tensor liveness. Implements eviction policy for register pressure (reclaims least-recently-used tensors when capacity exceeded).

#### UCNExecutor (`executor.py`)
High-level inference API. Takes token embeddings, optionally a pre-synthesized program, compiles via JIT, executes via workspace, and returns output tensor.

### 3.4 Standard Library (`ucn/stdlib/`)

#### Schema (`schema.py`)
`PrimitiveEntry` dataclass mapping to the `stdlib.uvm` JSON schema:
- `primitive_id`: unique hash identifier
- `symbolic_name`: human-readable or systematic name
- `type`: `operator_circuit` or `latent_feature`
- `source_layers`: which layers the primitive was extracted from
- `math_def`: operator type + weight file URIs
- `behavior_meta`: description + trigger conditions

#### Loader (`loader.py`)
Reads `.uvm` JSON files and resolves weight binary files (`.pt` tensors). Supports three operator types:
1. `low_rank_projection`: u, v matrices for rank-r decomposition
2. `direction_vector`: single feature direction for additive injection
3. `multihead_attention`: full Q/K/V/O weights + RoPE metadata for entire attention layer

### 3.5 Decompilation Pipeline (`ucn/decompile/`)

#### QwenActivationCollector (`source_model.py`)
Hook-based activation extraction from Qwen2.5-1.5B. Captures:
- Residual stream at all layers (forward hooks)
- Attention outputs per layer (forward hooks with `output_attentions=True`)
- Head-wise attention patterns (12 heads × 28 layers)
- MLP intermediate activations

Uses eager attention implementation (`attn_implementation="eager"`) to expose attention weights.

#### SparseAutoencoder (`sae.py`)
Trains an overcomplete SAE (e.g., 256 features for d_model=1536) on residual stream activations. Loss = MSE(x, x_hat) + λ·L1(h). Extracts W_dec columns as feature direction vectors.

#### Copy Head Finder (`copy_head_finder.py`)
Probes attention patterns across all heads to identify copy-head behavior (high attention to previous token). Ranks (layer, head) pairs by average prev-token attention.

#### Feature Analyzer (`feature_analyzer.py`)
Intervention tools for causally testing SAE features:
- `test_feature_on_prompt()`: inject feature direction with scaling factor, observe output change
- `feature_intervention()`: SAE-aware version using decoder weights
- `measure_intervention_impact()`: sweep across scales to measure causal effect

### 3.6 Frontend Layer (`ucn/frontend/`)

#### Context Analyzer (`context_analyzer.py`)
2-layer GRU (d_model → d_latent=128) with input projection and LayerNorm. Pools over time dimension.

#### Template Selector (`template_selector.py`)
MLP classifier (d_latent → 2*d_latent → n_templates) with Gumbel-Softmax support for differentiable discrete selection.

#### Parameter Generator (`parameter_generator.py`)
3-layer MLP (d_latent → 2*d_latent → d_latent → max_params) with sigmoid output for continuous parameter regression.

#### Template Library (`template_library.py`)
8 predefined AST template skeletons:
| ID | Name | Operations |
|----|------|-----------|
| 0 | identity_pass | scale(input) |
| 1 | single_transform | transform(input, stdlib.X) |
| 2 | mix_two | weighted mix of input + prev_input |
| 3 | transform_activate | transform → activate |
| 4 | mix_activate | mix → activate |
| 5 | rotate_transform | rotate → transform |
| 6 | project_transform | project → transform |
| 7 | dense_residual | residual accumulation |

#### MetaCompiler (`meta_compiler.py`)
Top-level orchestrator: embeddings → context_z → (template_id, params) → Program AST. ~497K trainable parameters (for d_model=1536).

### 3.7 Training (`ucn/training/`)

#### Distillation (`distill.py`)
Supervised learning: teacher model provides (template_id, params) targets. Loss = CE(template_logits, target) + 0.1·MSE(params_pred, params_target). ~500 steps to convergence on synthetic tasks.

#### REINFORCE (`reinforce.py`)
Policy gradient for discrete template selection: ∇E[L] ≈ (L - baseline)·∇log P(T|z). Exponential moving average baseline. Continuous parameters optimized via backprop.

---

## 4. Data Flow

### 4.1 Inference Path (MetaCompiler active)

```
Input Tokens
    │
    ▼
Embedding Layer ────► ContextAnalyzer (GRU) ──► latent z
                                                    │
                              ┌─────────────────────┼─────────────────────┐
                              ▼                     ▼                     ▼
                       TemplateSelector      ParameterGenerator    TemplateLibrary
                              │                     │                     │
                              ▼                     ▼                     │
                       template_id            params (float[])           │
                              │                     │                     │
                              └─────────┬───────────┘                     │
                                        ▼                                 │
                                  build_program() ◄───────────────────────┘
                                        │
                                        ▼
                                  UVM-DSL AST
                                        │
                                        ▼
                              JITCompiler.compile()
                                        │
                          ┌─────────────┼─────────────┐
                          ▼             ▼             ▼
                    L1 Cache Hit?   Optimizer    Codegen Backend
                          │         (DCE+Fuse)   (Reference/Triton)
                          ▼                           │
                    Reuse Binary                     ▼
                                              Fused Kernel
                                                  │
                                                  ▼
                                          TensorWorkspace
                                          (allocate, execute, store)
                                                  │
                                                  ▼
                                          Output Tensor Y
```

### 4.2 Decompilation Path

```
Pretrained Model (Qwen2.5-1.5B)
    │
    ▼
Activation Collection ◄── Text corpus (WikiText-like)
    │
    ├──► Residual Stream [layer 0..27]
    │        │
    │        ▼
    │    Sparse Autoencoder Training
    │    (MSE + λ·L1 on activations)
    │        │
    │        ▼
    │    Feature Vectors (W_dec columns)
    │        │
    │        ├──► Semantic Primitives (human-labeled)
    │        └──► Latent Primitives (systematic IDs)
    │
    ├──► Attention Patterns [layer 0..27, head 0..11]
    │        │
    │        ▼
    │    Copy Head Finder (prev-token attention)
    │        │
    │        ▼
    │    Circuit Discovery (activation patching)
    │
    └──► Q/K/V/O Weight Extraction
             │
             ▼
         Combined: stdlib.uvm + weight/*.pt
```

---

## 5. Key Design Decisions

1. **Reference backend as golden source**: The PyTorch reference backend is the correctness definition. Triton kernels are verified by pixel-perfect parity testing.

2. **Float32 throughout**: All custom math in hooks, kernels, and SAE training uses float32 internally per AGENTS.md rule #2 (fp16 numeric stability). Outputs are cast back to model dtype.

3. **Eager attention for decompilation**: SDPA/flash-attention is disabled during extraction to expose attention weight tensors. Training runs use SDPA for speed.

4. **Template-based AST synthesis**: The MetaCompiler selects from 8 predefined templates rather than generating arbitrary code, trading expressivity for correctness and compilability.

5. **stdlib as the model/bridge boundary**: The standard library is the interface between decompilation output and compilation input. Any model can be decompiled into `stdlib.uvm`; any backend can consume it.

6. **Two-phase training**: Distillation (supervised) initializes the MetaCompiler; REINFORCE (policy gradient) tunes discrete template selections against a task reward.

---

## 6. Dependencies

| Dependency | Version | Purpose |
|-----------|---------|---------|
| PyTorch | 2.10 | Neural primitives, autograd, tensor ops |
| Triton | 3.5.1 | GPU kernel JIT compilation |
| transformers | latest | Loading Qwen2.5-1.5B, tokenization |
| numpy | any | Stdlib weight serialization |

**No** external dependency on TransformerLens, bitsandbytes, or MLIR — all hooks, SAE training, and codegen are custom-built.

---

## 7. File Inventory

```
full_compiled_experiment/
├── poc/                          # Phase 0: Toy PoC (C compiler demo)
│   ├── toy_stdlib.py
│   └── ucn_compiler.py
│
├── ucn/                          # Main UCN package
│   ├── __init__.py               # Package exports
│   ├── config.py                 # Path resolution
│   ├── dsl/                      # UVM-DSL language
│   │   ├── ast.py                # 7 expression types + Program
│   │   ├── parser.py             # Recursive-descent parser
│   │   └── types.py              # Vector, Subspace, Scalar, Matrix
│   ├── stdlib/                   # Standard library
│   │   ├── schema.py             # PrimitiveEntry dataclass
│   │   └── loader.py             # .uvm JSON + .pt weight loading
│   ├── backend/                  # JIT compilation
│   │   ├── jit_compiler.py       # Compile orchestration
│   │   ├── cache.py              # L1 + L2 cache
│   │   ├── optimizer.py          # DCE, fusion, pruning
│   │   └── codegen/
│   │       ├── reference.py      # PyTorch reference (golden)
│   │       └── triton_backend.py # Triton GPU kernels
│   ├── runtime/                  # Execution engine
│   │   ├── workspace.py          # Virtual tensor workspace
│   │   └── executor.py           # High-level forward pass
│   ├── frontend/                 # Neural MetaCompiler
│   │   ├── context_analyzer.py   # 2-layer GRU
│   │   ├── template_selector.py  # Categorical classifier
│   │   ├── parameter_generator.py# Continuous parameter regressor
│   │   ├── template_library.py   # 8 AST templates
│   │   └── meta_compiler.py      # Orchestrator
│   ├── decompile/                # Reverse-engineering pipeline
│   │   ├── source_model.py       # Qwen activation collector
│   │   ├── copy_head_finder.py   # Attention head scanner
│   │   ├── sae.py               # Sparse autoencoder
│   │   └── feature_analyzer.py   # Intervention tools
│   └── training/                 # MetaCompiler training
│       ├── distill.py            # Supervised distillation
│       └── reinforce.py          # Policy gradient
│
├── scripts/                      # Runnable experiments
│   ├── find_and_extract_copy_head.py  # Phase 2 pipeline
│   ├── verify_copy_head.py            # Phase 3 fidelity test
│   ├── full_attention_verifier.py     # Phase 3a full verifier
│   ├── verify_ucn_attention.py        # Phase 3c UCN pipeline test
│   └── phase5_benchmark.py            # Phase 5 end-to-end
│
├── tests/                        # Test suite
│   └── test_phase1_integration.py     # 12 tests
│
├── artifacts/                    # Generated outputs
│   ├── copy_head_extraction/     # 100 SAE primitives
│   ├── copy_head_fidelity/       # V*O fidelity results
│   ├── full_attention_verifier/  # Full Q/K/V/O weights
│   ├── full_attention_stdlib/    # UCN-compiled stdlib
│   └── phase5_benchmark/         # Final report
│
└── specs/                        # Original spec documents
    ├── FULLY_COMPILED_SPEC.md
    ├── FULLY_COMPILED_POC_SPEC.md
    └── FULLY_COMPILED_MODEL_NOTES.md
```

---

## 8. Performance Characteristics

| Operation | Backend | d_model=1536 | d_model=128 |
|-----------|---------|-------------|-------------|
| Mix (2 inputs) | Reference | ~20 μs | ~2 μs |
| Mix (2 inputs) | Triton | ~15 μs | ~2 μs |
| Transform (dense) | Reference | ~1.2 ms | ~10 μs |
| Transform (low-rank r=128) | Reference | ~0.4 ms | — |
| Multi-head attention (full) | Reference | ~3.5 ms | — |
| Activate (GELU) | Reference | ~5 μs | <1 μs |
| Activate (GELU) | Triton | ~3 μs | <1 μs |
| MetaCompiler forward | Reference | ~0.8 ms | ~0.2 ms |

*Measured on RTX 3080, single vector. Timings include kernel launch overhead.*

---

## 9. Fidelity Benchmarks

| Test | Method | Cosine | MSE | Δ from V*O-only |
|------|--------|--------|-----|-----------------|
| Copy head V*O only | Single head projection | 0.18 | — | baseline |
| Copy head V*O (all 12) | All head projections | 0.25 | — | +0.07 |
| Full attention (manual) | Q/K/V/O + RoPE + softmax | **1.0000** | **0.0000** | +0.75 |
| Full attention (UCN) | Compiled via UCN pipeline | **1.0000** | **0.0000** | +0.75 |

**The fidelity gap is fully bridged.**
