# NRTCS — Project Purpose Quick Reference

**Neurosymbolic Round-Trip Compilation Stack**

---

## One Sentence

A **compiler for neural network weights** — decompile, edit, recompile, and splice model parameters without gradient training.

---

## The Four Phases

```
safetensors  ──►  1. Decompile (C2S)    ──►  symbolic features
                      SAE feature extraction
                      + circuit attribution

symbolic features ──►  2. Refactor (UVM) ──►  patched symbols
                      Human or agent edits

patched symbols ──►  3. Recompile (S2C)  ──►  weight matrices
                      W_down = K^T(KK^T)^-1
                      W_up   = V
                      + orthogonal crosstalk prevention

weight matrices ──►  4. Splice (Binary)  ──►  patched .safetensors
                      mmap inline tensor replacement
```

---

## What It Can Do Today

| Capability | Status |
|-----------|--------|
| **FFN weight patching** — decompile → edit key-value pairs → recompile FFN matrices → splice | Production-ready |
| **Crosstalk prevention** — orthogonal projection ensures patches don't degrade unrelated behavior | Production-ready |
| **Safetensors splicing** — mmap-based inline tensor replacement without file rewrite | Production-ready |
| **SAE-based decompilation** — extract features from any layer above threshold τ | Production-ready |
| **Path attribution** — gradient-based circuit tracing between layers | Production-ready |
| **Full round-trip** — compile → splice → verify reconstruction in one call | Production-ready |
| **Dimension tracking** — `compute_null_space_rank()` monitors remaining capacity | Production-ready |
| **Model loading/compatibility** — tested on GPT-2 (tiny, small), Qwen 2.5-1.5B | Production-ready |

---

## What It Cannot Do Yet

| Limitation | Why | Status |
|-----------|-----|--------|
| **Attention head splicing** | Recompiler targets FFN `down_proj`/`up_proj` shapes; attention weights (Q/K/V/O) not wired | Bridge exists (UCN backend compiles full attention at cosine=1.0), not integrated |
| **Add new tensors** | mmap splicer requires existing tensors with matching shapes | Architectural constraint |
| **Change tensor shapes** | Would require header rewrite | Could be added |
| **Multiple architecture support** | `splice_mlp` assumes `{prefix}.down_proj.weight` naming; GPT-2 uses `c_fc`/`c_proj` | Overridable via `model_name` parameter |
| **Trained SAE requirements** | Decompiler needs pretrained SAEs per layer; untrained random SAEs work for testing but not for real feature extraction | Inherited from SAE training pipeline |

### The Attention Gap — Specifics

The decompiler already captures activations from any layer (attention output included in the residual stream). The UCN backend in `full_compiled_experiment/ucn/backend/reference.py` already compiles full multi-head attention (Q/K/V/O projections + RoPE + GQA + causal softmax + output projection) with **cosine similarity = 1.000000** against the original model.

What's missing to close the gap:
1. A mapping from decompiled attention features → UCN `multihead_attention` template parameters
2. A splicer path that writes into Q/K/V/O weight tensors (different names than MLP)
3. Shape compatibility layer between UCN-compiled attention and safetensors tensor layout

---

## Key Numbers

| Metric | Value |
|--------|-------|
| Tests | **73** (61 fast, 9 slow with tiny-gpt2, 3 GPU with Qwen 1.5B) |
| Source modules | 7 (`__init__`, `recompiler`, `splicer`, `decompiler`, `pipeline`, `cli`) |
| Doc files | 4 (`NRTCS_SPEC`, `ARCHITECTURE`, `MANUAL`, `USE_CASES`) |
| Dependencies | `torch`, `safetensors`, `numpy` |
| Tested models | `sshleifer/tiny-gpt2`, GPT-2 small, Qwen/Qwen2.5-1.5B |
| Max tested scale | N=500 keys, d=768, m=200 protected features |
| Numerical dtype | All math in float32 (AGENTS.md rule #2) |
| Matrix inversion | Cholesky decomposition (O(N³/3), SPD check built-in) |

---

## Quick Start (5 lines)

```python
import torch
from sae_editor import NRTCSPipeline

pipeline = NRTCSPipeline()
result = pipeline.compile_dense_map(
    keys=torch.randn(10, 768),      # 10 key vectors, d_model=768
    values=torch.randn(10, 768),    # 10 value vectors
)
# result["W_down"]: (768, 10), result["W_up"]: (10, 768)
# Guarantee: keys @ W_down @ W_up ≈ values (atol < 1e-4)
```

---

## Relationship to Surrounding Project

```
compiled-hybrid-lm/
├── hybrid/
│   ├── sae_editor/                    ◄── THIS PROJECT
│   │   ├── recompiler.py             (NWR: analytical matrix construction)
│   │   ├── splicer.py                (NWR: mmap safetensors patching)
│   │   ├── decompiler.py             (reuses UCN SAEs)
│   │   └── pipeline.py               (orchestrator)
│   │
│   ├── full_compiled_experiment/ucn/  ◄── sibling: UCN compiler
│   │   ├── backend/reference.py      (full attention compilation)
│   │   ├── decompile/sae.py          (SAE training)
│   │   └── stdlib/                   (primitive library schema)
│   │
│   ├── compiled_features/             ◄── runtime: 21-channel n-gram features
│   ├── channels_v3.py                 ◄── runtime: Witten-Bell compiled channels
│   ├── superposition_steerer_v3.py    ◄── runtime: activation steering hooks
│   └── cartridges.py                  ◄── runtime: hot-swappable cartridge rack
```

**Division of labor:**
- **NRTCS** = permanent weight modification (offline, file-level)
- **Compiled features + steerers** = runtime behavior modulation (online, hook-based)
- **Together** = patched base model + runtime steering → deployed system

---

## Run the Tests

```bash
cd ~/deepseek_experiments/hybrid

# Fast (CPU, <1s)
pytest sae_editor/tests/ -m "not slow and not gpu" -v

# All CPU (fast + tiny-gpt2, ~5s)  
pytest sae_editor/tests/ -m "not gpu" -v

# Everything (+ GPU Qwen 1.5B, ~15s, needs CUDA)
pytest sae_editor/tests/ -v
```
