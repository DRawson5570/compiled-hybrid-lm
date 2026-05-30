# UCN — Operator Manual

## Unified Compiled Network — Complete Usage Guide

**Target audience:** AI agents and human researchers  
**Last updated:** 2026-05-30  
**Status:** Phases 0-5 complete, all tests passing

---

## Quickstart (3 minutes)

```bash
# 1. Verify environment
cd ~/deepseek_experiments/hybrid/full_compiled_experiment
python3 -c "import torch, triton; print('OK')"

# 2. Run PoC toy compiler
cd poc
python3 ucn_compiler.py && gcc -O3 model_pipeline.c -o ucn_poc_bin && ./ucn_poc_bin

# 3. Run all tests
cd ..
python3 tests/test_phase1_integration.py
# Expected: 12/12 PASS

# 4. Verify Qwen is available
python3 -c "from transformers import AutoModelForCausalLM; \
  m = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-1.5B', trust_remote_code=True); \
  print('Qwen OK')"
```

---

## Core Concepts

### What UCN does
UCN is a **compiler for neural network computation**. Instead of executing static matrix multiplications, it:
1. Analyzes input context via a lightweight neural network (MetaCompiler)
2. Emits a UVM-DSL program describing what operations to perform
3. JIT-compiles that program to GPU kernels via Triton
4. Executes the fused kernel on a virtual tensor workspace

### Key terms
- **UVM-DSL**: Unified Vector Manipulation Domain-Specific Language. 7 primitive operations (mix, project, transform, activate, query_memory, residual, rotate).
- **stdlib.uvm**: JSON database of extracted feature vectors and attention weights from a pretrained model.
- **Primitive**: A named, executable operation in the standard library. Stored as low-rank matrices, direction vectors, or full attention weights.
- **MetaCompiler**: Neural frontend (~497K params) that maps embeddings → UVM-DSL programs.
- **JITCompiler**: Backend that compiles UVM-DSL ASTs → fused GPU kernels (Reference or Triton backends).
- **SAE**: Sparse Autoencoder. Trained on residual stream activations to extract disentangled feature directions.
- **Decompilation**: The process of extracting primitives from a pretrained model's weights and activations.

---

## CLI & Entry Points

### Phase 0: Toy PoC
```bash
cd poc
python3 ucn_compiler.py       # Python → model_pipeline.c
gcc -O3 model_pipeline.c -o ucn_poc_bin  # C → binary
./ucn_poc_bin                  # Execute
```
Purpose: Verify the compilation pipeline concept (d_model=8, 2 tokens, CPU-only C generation).

### Phase 2: Decompile Qwen2.5-1.5B
```bash
python3 scripts/find_and_extract_copy_head.py
```
What it does:
1. Loads Qwen2.5-1.5B (float16, eager attention)
2. Probes attention patterns → identifies copy heads (layer 0, head 8 at ~72% prev-token attention)
3. Collects residual stream activations from target layer
4. Trains SAE (256 features, MSE→0.0003, sparsity→0.76)
5. Extracts top 100 feature vectors → saves to `artifacts/copy_head_extraction/stdlib.uvm`

### Phase 3: Fidelity Test

#### V*O projection (original, limited)
```bash
python3 scripts/verify_copy_head.py
```
Extracts V and O weights for the copy head, compiles as UCN program, measures cosine similarity vs real attention output. **Average: 0.18 cosine (V*O only, insufficient).**

#### Full attention (bridges gap)
```bash
python3 scripts/full_attention_verifier.py
```
Extracts ALL Q/K/V/O weights + biases + RoPE metadata from layer 0. Manually recomputes full multi-head attention. Compares against actual model output. **Average: 1.0000 cosine, 0.0000 MSE — perfect fidelity.**

#### UCN-compiled full attention (bridges gap via pipeline)
```bash
python3 scripts/verify_ucn_attention.py
```
Saves full attention weights as stdlib primitive. Executes via UCN ReferenceBackend. **Average: 1.0000 cosine — gap fully bridged.**

### Phase 5: End-to-End Benchmark
```bash
python3 scripts/phase5_benchmark.py
```
Synthetic benchmark: MetaCompiler trained via distillation, UVM-DSL programs synthesized from context, executed through UCN pipeline as a readme, compared with ground truth. Produces `artifacts/phase5_benchmark/final_report.json`.

---

## Using the UCN Programmatically

### Building a UVM-DSL Program (Python API)
```python
from ucn.dsl.ast import Program, Mix, Activate, ActivateType, Transform, MatrixRef

# Create a program that mixes two inputs with GELU activation
program = Program()
program.add_stmt("temp", Mix(["x0", "x1"], [0.7, 0.3]))
program.add_stmt("output", Activate("temp", ActivateType.GELU))
```

### Building with the Text Parser
```python
from ucn.dsl.parser import parse_program

source = """
y1 = mix([x0, x1], [0.75, 0.25]);
y2 = activate(y1, gelu)
"""
program = parse_program(source)
```

### Executing via Reference Backend
```python
import torch
from ucn.backend.codegen.reference import ReferenceBackend

# Load stdlib weights (from Phase 3 extraction)
stdlib = {
    "copy_head_L0_H8": {
        "operator_type": "low_rank_projection",
        "u": torch.load("artifacts/copy_head_fidelity/weights/copy_head_v_u.pt"),
        "v": torch.load("artifacts/copy_head_fidelity/weights/copy_head_o_v.pt"),
    }
}

backend = ReferenceBackend(stdlib_weights=stdlib, device="cpu", dtype=torch.float32)

program = Program()
program.add_stmt("y", Transform("x", MatrixRef("stdlib", "copy_head_L0_H8")))

x = torch.randn(1536)  # Qwen2.5-1.5B d_model
outputs = backend.execute(program, {"x": x})
result = outputs["y"]  # shape: [1536]
```

### Executing via JIT Compiler with Triton
```python
from ucn.backend.jit_compiler import JITCompiler

compiler = JITCompiler(
    stdlib_weights=stdlib,
    device="cuda",
    dtype=torch.float32,
    use_triton=True,  # GPU-accelerated
)

x_cuda = torch.randn(1536, device="cuda")
outputs = compiler.compile_and_execute(program, {"x": x_cuda})
```

### Using the Executor (high-level API)
```python
from ucn.runtime.executor import UCNExecutor

executor = UCNExecutor(
    d_model=1536,
    stdlib_weights=stdlib,
    device="cuda",
    use_triton=True,
)

embeddings = torch.randn(1, 8, 1536, device="cuda")  # [B, T, D]
output = executor.forward(embeddings, program=program)
```

### Extracting Activations from Qwen2.5-1.5B
```python
from ucn.decompile.source_model import QwenActivationCollector

collector = QwenActivationCollector(
    model_name="Qwen/Qwen2.5-1.5B",
    layers=[0, 4, 8, 12, 16, 20, 24],
    device="cuda",
)

texts = [
    "The cat sat on the mat.",
    "Machine learning is a field of AI.",
]

# Collect residual stream at each hooked layer
residual = collector.collect_residual_stream(texts, max_length=128)
layer0_acts = residual[0]  # shape: [total_tokens, 1536]

# Collect attention outputs
attn_data = collector.collect_attention_from_layer(texts, layers=[0])
layer0_attn = attn_data[0]  # list of attention pattern tensors

# Run model with full hidden states
logits, hidden_states = collector.run_model_with_output_hidden("Hello world")
```

### Training a Sparse Autoencoder
```python
import torch
from ucn.decompile.sae import SparseAutoencoder, train_sae, normalize_decoder

sae = SparseAutoencoder(d_model=1536, n_features=256, l1_lambda=1e-4)

# activations: flat tensor of shape [N, 1536] from collector
history = train_sae(sae, activations, steps=2000, lr=1e-3, batch_size=128, device="cuda")
normalize_decoder(sae)

# Extract feature vectors
features = sae.get_features()  # Dict[int, Tensor[1536]]
for feat_idx, vec in features.items():
    print(f"Feature {feat_idx}: norm={vec.norm():.4f}")
```

### Using the MetaCompiler
```python
from ucn.frontend.meta_compiler import MetaCompiler

mc = MetaCompiler(d_model=1536, n_templates=8, max_params=4, d_latent=128, device="cuda")

# Single batch
embeddings = torch.randn(1, 8, 1536, device="cuda")
program = mc.synthesize(embeddings, stdlib_names=["copy_head", "induction_head"])
# program is a UVM-DSL AST

# Training the MetaCompiler
from ucn.training.distill import train_meta_compiler_supervised
train_data = [(emb, 1, torch.tensor([0.5])) for emb in ...]  # [(tensor, template_id, params)]
history = train_meta_compiler_supervised(mc, train_data, steps=500, lr=1e-3)
```

### Building a Standard Library
```python
from ucn.stdlib.schema import MathDef, BehaviorMeta, PrimitiveEntry
from ucn.stdlib.loader import save_stdlib_json, save_weight_tensor

entry = PrimitiveEntry(
    primitive_id="PRM_0xA1B2",
    symbolic_name="copy_head_L0_H8",
    type="operator_circuit",
    source_layers=[0],
    math_def=MathDef(
        operator_type="low_rank_projection",
        rank=128,
        u_uri="weights/copy_head_u.pt",
        v_uri="weights/copy_head_v.pt",
    ),
    behavior=BehaviorMeta(
        description="Copy head at layer 0: attends to previous token",
        trigger_conditions=["sequence_tokens"],
    ),
)

save_weight_tensor(u_tensor, "/path/to/weights/copy_head_u.pt")
save_weight_tensor(v_tensor, "/path/to/weights/copy_head_v.pt")
save_stdlib_json([entry], "/path/to/stdlib.uvm")
```

---

## Configuration

### `ucn/config.py`
```python
REPO_ROOT    # Path to full_compiled_experiment/ (computed)
ARTIFACTS    # ~/deepseek_experiments/artifacts/ucn/
STDLIB_DIR   # REPO_ROOT/stdlib_weights/
CACHE_DIR    # ARTIFACTS/jit_cache/
DEFAULT_DTYPE  # "float32"
DEFAULT_DEVICE # "cuda"
```

### Model Selection
All scripts default to `Qwen/Qwen2.5-1.5B`. To use a different model:
```python
collector = QwenActivationCollector(model_name="Qwen/Qwen2.5-3B", ...)
# Or for any HuggingFace model:
collector = QwenActivationCollector(model_name="meta-llama/Llama-2-7b-hf", ...)
```

### GPU vs CPU
All scripts auto-detect:
```python
device = "cuda" if torch.cuda.is_available() else "cpu"
```
Override with `device="cpu"` for debugging or resource-constrained environments.

---

## Running Tests

```bash
# Full integration test suite (12 tests, ~10 seconds)
python3 tests/test_phase1_integration.py

# What each test covers:
# AST construction         — Python API → Program
# DSL parser               — Text → AST (simple)
# Full DSL parser          — Text → AST (all primitives)
# Reference backend mix    — mix op execution
# Reference backend activate — GELU execution
# Reference backend transform — low-rank projection
# Reference backend rotate — rotation with bounds checking
# Reference backend residual — sum accumulation
# JIT compiler with caching — compile + execute + cache hit
# UCNExecutor forward      — high-level forward pass
# Triton backend parity    — Triton output == Reference output
# Triton activate parity   — Triton GELU == PyTorch GELU
```

To add a new test, create `tests/test_*.py` and follow the pattern of `test_phase1_integration.py`: import UCN modules, construct test data, compare against expected output.

---

## Common Recipes

### Recipe 1: Decompile a new attention head from a different layer
```bash
python3 -c "
from ucn.decompile.source_model import QwenActivationCollector
from ucn.decompile.copy_head_finder import find_copy_heads, measure_copy_fidelity

collector = QwenActivationCollector(layers=list(range(28)), device='cuda')
candidates = find_copy_heads(collector, n_top=20)
for c in candidates:
    f = measure_copy_fidelity(collector, c.layer, c.head)
    print(f'L{c.layer:2d} H{c.head:2d} prev_attn={c.prev_token_attention:.4f} fid_prev={f[\"prev_attention\"]:.4f}')
"
```

### Recipe 2: Train SAE on a specific layer
```python
from ucn.decompile.source_model import QwenActivationCollector
from ucn.decompile.sae import SparseAutoencoder, train_sae, normalize_decoder

collector = QwenActivationCollector(layers=[12], device="cuda")
texts = ["..." * 50]  # 50+ diverse sentences
residual = collector.collect_residual_stream(texts, max_length=128)
acts = residual[12].reshape(-1, 1536).float()

sae = SparseAutoencoder(d_model=1536, n_features=512, l1_lambda=1e-4)
train_sae(sae, acts, steps=2000, lr=1e-3, batch_size=128, device="cuda")
normalize_decoder(sae)
features = sae.get_features()
# features now contains disentangled direction vectors for layer 12
```

### Recipe 3: Measure compiled vs real output
```python
# After building a stdlib entry:
program = Program()
program.add_stmt("y", Transform("x", MatrixRef("stdlib", "my_feature")))
backend = ReferenceBackend(stdlib_weights=stdlib, device="cpu", dtype=torch.float32)

# Run through UCN
ucn_output = backend.execute(program, {"x": real_input})["y"]

# Compare
cos = F.cosine_similarity(ucn_output.flatten(), real_output.flatten(), dim=0)
mse = F.mse_loss(ucn_output, real_output)
```

### Recipe 4: Add a new template to the MetaCompiler
```python
from ucn.frontend.template_library import TemplateDef, TemplateLibrary

# Define a new template
new_template = TemplateDef(
    template_id=8,
    name="double_transform",
    description="Sequential transform through two primitives",
    n_params=2,  # two matrix indices
)

# Extend the library
lib = TemplateLibrary()
lib.templates.append(new_template)

# Add corresponding build logic in build_program():
# elif template_id == 8:
#     ...
```

---

## Debugging & Troubleshooting

### NaN in SAE training
**Symptom:** `loss=nan, mse=nan` after first training step.  
**Cause:** Float16 activations from the model mixing with float32 SAE.  
**Fix:** Cast activations to float32 before training:
```python
acts_flat = acts.reshape(-1, acts.shape[-1]).to(dtype=torch.float32)
```

### "sdpa attention does not support output_attentions"
**Symptom:** Attention hooks return no data.  
**Fix:** Force eager attention implementation:
```python
model = AutoModelForCausalLM.from_pretrained(..., attn_implementation="eager")
```

### Hook returns empty input tuple
**Symptom:** `IndexError: tuple index out of range` in forward_pre_hook.  
**Cause:** Qwen2.5 attention uses keyword arguments, not positional.  
**Fix:** Use `register_forward_pre_hook(hook, with_kwargs=True)` and access `kwargs['hidden_states']`.

### Triton kernel compilation error
**Symptom:** `CompilationError` with `AttributeError`.  
**Cause:** Triton API version mismatch (e.g., `tl.extra.fast_gelu` removed in 3.5+).  
**Fix:** Use manual implementations: GELU via `0.5 * x * (1 + erf(x * 0.7071))`, ReLU via `tl.maximum(x, 0)`.

### Low fidelity in compiled output
**Symptom:** Cosine similarity < 0.9 after compilation.  
**Checklist:**
1. Are biases included? Q/K/V projections in Qwen have bias terms.
2. Is RoPE applied? Q and K need position-dependent rotation.
3. Are KV heads properly repeated? GQA: 2 KV heads → 12 query heads.
4. Is the causal mask correct? `torch.triu(-inf, diagonal=1)`.
5. Is dtype consistent? All intermediate computation in float32.

---

## Artifacts Reference

| Path | Contents | Size |
|------|----------|------|
| `poc/model_pipeline.c` | Generated C code from toy compiler | ~3KB |
| `artifacts/copy_head_extraction/stdlib.uvm` | 100 SAE feature primitives (JSON) | ~50KB |
| `artifacts/copy_head_extraction/weights/` | 100 weight files (.pt) | ~60MB |
| `artifacts/copy_head_fidelity/stdlib.uvm` | Copy head V/O weights | ~2KB |
| `artifacts/copy_head_fidelity/weights/` | V and O weight tensors | ~1.5MB |
| `artifacts/copy_head_fidelity/fidelity_results.json` | Cosine/MSE per prompt | ~2KB |
| `artifacts/full_attention_verifier/` | Q/K/V/O weights + biases + RoPE | ~12MB |
| `artifacts/full_attention_stdlib/ucn_fidelity_report.json` | UCN pipeline fidelity report | ~1KB |
| `artifacts/phase5_benchmark/final_report.json` | End-to-end benchmark summary | ~2KB |

---

## Environment & Hardware

### Required
- Python 3.12+, PyTorch 2.10+, CUDA 12.6+
- Triton 3.5.1+
- transformers 4.x (`trust_remote_code=True` for Qwen2.5)
- RTX 3080 or equivalent (10GB VRAM recommended for fp32 Qwen2.5-1.5B)

### Optional
- NVIDIA M40 24GB (pe2) for larger models or multi-GPU via ZeroQ
- bitsandbytes (for future 4-bit quantization support)
- NVIDIA A100/H100 for real-time compilation latency targets

### Memory Budget
| Component | VRAM (fp32) |
|-----------|-------------|
| Qwen2.5-1.5B (eager) | ~5.8 GB |
| SAE (d=1536, f=256) | ~1.6 MB |
| MetaCompiler (497K params) | ~2 MB |
| stdlib (100 primitives) | ~60 MB |
| Activation buffers (single prompt) | ~50 MB |
| **Total (RTX 3080, 10GB)** | ~6 GB |

---

## Known Limitations

1. **Triton rotate kernel only supports contiguous subspaces.** For arbitrary index sets, use the Reference backend.
2. **Multi-batch fusion not implemented.** Each batch element gets its own kernel launch.
3. **L2 semantic cache is instantiated but not yet populated.** Only L1 structural cache is active.
4. **Template library parameter scaling assumes d_model=1536.** Adjust `int(p * 1536)` in `build_program()` for different model sizes.
5. **Full attention primitive requires eager mode**, which is slower than SDPA/flash-attention. For production inference, the `multihead_attention` operator type would need a flash-attention-compatible implementation.
6. **REINFORCE training is functional but untuned**. Baseline and learning rate defaults need per-task adjustment.

---

## Extending the System

### Adding a new UVM-DSL primitive
1. Add the class to `ucn/dsl/ast.py`
2. Add a parsing rule to `ucn/dsl/parser.py`
3. Implement `_execute_*` in `ucn/backend/codegen/reference.py`
4. Optionally implement a Triton kernel in `ucn/backend/codegen/triton_backend.py`
5. Add test cases to `tests/test_phase1_integration.py`

### Adding a new backend
1. Create `ucn/backend/codegen/new_backend.py`
2. Implement the same interface as `ReferenceBackend.execute()`
3. Register with `JITCompiler` in `_make_kernel()`
4. Verify parity against `ReferenceBackend` output

### Decompiling a different model
1. Create or extend `QwenActivationCollector` for the new model architecture
2. Adjust hook injection points to match the model's module hierarchy
3. Verify that `collect_residual_stream` and `collect_attention_from_layer` return correct shapes
4. Run `find_and_extract_copy_head.py` with the new model name
