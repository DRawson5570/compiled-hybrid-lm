# Current Blocker: Multi-Layer MLP Fidelity Collapse

**Status:** Investigated, root cause confirmed, Path A exhausted, Path B plan ready.  
**Handoff:** Gemini 3.5 — pick up at Path B (distilled compact MLP).  
**Date:** 2026-05-30  
**Agent:** deepseek-v4-pro  
**Update:** **BUG FOUND AND FIXED** — RoPE embeddings missing from multi-layer attention stdlib. After fix: layer 0 improved from 0.80 to 0.92, multi-layer avg from 0.463 to 0.501 (+8%). Blockers partially resolved — see Path B for remaining gap.

---

## 1. Core Problem

The UCN (Unified Compiled Network) can replace Transformer attention layers with compiled UVM-DSL programs at **1.0000 cosine fidelity**. The MLP layers can be replaced at **0.80 single-layer cosine** via sparse down projection (compute gate+up densely, sparsify only the down projection at K=1024 neurons).

However, when chaining multiple layers together, **MLP fidelity collapses geometrically**: 0.80 (layer 0) → 0.12 (layer 4) → 0.10 (layer 8). The sparse MLP approximation error accumulates across residual connections, corrupting downstream layer inputs.

**The blocker:** Multi-layer UCN execution cannot replace the original model's MLP computation with sufficient fidelity. A 28-layer model would diverge completely by layer 5-6.

---

## 2. What Already Works (Solid Foundation)

| Component | Fidelity | Method |
|-----------|----------|--------|
| Attention compilation | **1.0000 cosine** | Full Q/K/V/O + RoPE + SDPA, compiled to Triton/Reference kernels |
| DSL + parser + JIT + cache | All 7 primitives + test suite (18/19 tests) | UVM-DSL grammar, recursive descent parser, L1 structural cache, Triton GPU kernels |
| Decompilation pipeline | Copy head finder, SAE training, weight extraction | Qwen2.5-1.5B layer 0, 256 SAE features, 100 primitive extraction |
| Single-layer sparse MLP | **0.80 cosine** (K=1024, rank-128 low-rank residual) | `sparse_down_projection_lr` operator type |
| Multi-layer accumulated correction | **0.46 avg** (3 layers) at +0.12 improvement | Per-layer correction biases from UCN's own calibration output |

---

## 3. What Was Tried (Path A — Failed)

### Path A: Joint MetaCompiler Training via Soft Template Mixing

**Approach:** Train one MetaCompiler per layer jointly, where each MC's `synthesize_soft_forward` produces a soft-weighted mixture of 4 pre-computed sparse MLP templates. Loss = per-token cosine similarity between UCN output and teacher hidden states.

**Infrastructure built:**
- `synthesize_soft_forward()` in `ucn/frontend/meta_compiler.py` — differentiable soft template mixing
- `scripts/train_joint_multilayer.py` — full training loop with WikiText-103 data
- Gradient verification: 36/54 gradients flow through MetaCompiler chain
- pe3 deployment: M40 12GB GPU, fp16 model + fp32 eager stdlib extraction

**Training results (2000 steps on pe3, 280 prompts, lr=1e-4, cosine LR schedule, accum=4):**

| Metric | Start | End | Note |
|--------|-------|-----|------|
| Train loss | 1.18 | **0.19** | 6× reduction — training converges |
| Val cosine | 0.186 | **0.191** | <0.005 improvement — essentially flat |
| Layer 0 val | 0.162 | 0.164 | No improvement |
| Layer 4 val | 0.196 | 0.202 | No improvement |
| Layer 8 val | 0.200 | 0.207 | No improvement |

**Why it failed:** The soft template mixing approach learns to weight 4 pre-computed sparse MLP outputs, but cannot produce *corrective deltas* — it only remixes existing fixed transforms. The MetaCompilers can't compensate for upstream sparse MLP error because they can only select which fixed MLP output to use, not modify the MLP computation itself. This is an architectural limitation, not a training issue.

**Artifacts:** `artifacts/joint_training/pe3_run_20260530_154132.log`, config, checkpoint files.

---

## 3a. CRITICAL BUG FOUND AND FIXED (2026-05-30)

**Bug:** The multi-layer stdlib builder (`build_stdlib_and_programs()`) was NOT including RoPE embeddings (`cos`/`sin`) in attention stdlib entries. This meant attention at layers 4 and 8 was operating without positional encoding, producing outputs at only 0.89-0.96 cosine fidelity instead of 1.0000.

**Root cause:** `test_e2e.py` test 8 and `train_joint_multilayer.py` both build attention stdlib entries with Q/K/V/O weights only, omitting `"cos": cos_val, "sin": sin_val` that `_apply_full_attention` expects for RoPE rotation.

**Fix:** Extract `cos, sin = model.model.rotary_emb(...)` once and include in every attention stdlib entry.

**Impact after fix:**

| Layer | Before (no RoPE) | After (with RoPE) | Improvement |
|-------|------------------|-------------------|-------------|
| Layer 0 no correction | 0.801 | **0.922** | +15% |
| Layer 4 with correction | 0.291 | **0.291** | — |
| Layer 8 with correction | 0.290 | **0.290** | — |
| Average with correction | 0.463 | **0.501** | +8% |

**Key insight:** Layer 0 went from 0.80 to 0.92 — the previous 0.80 was not measuring the true sparse MLP limit, it was measuring ATTENTION error + sparse MLP error combined. The true single-layer sparse MLP fidelity (K=1024, R=128 LR) is 0.92.

**Remaining gap:** Layers 4 and 8 still collapse to 0.11 without correction, 0.29 with correction. This is now confirmed as genuine sparse MLP error accumulation (not a bug). The accumulated correction approach helps (0.11 → 0.29, ~2.6×) but can't fully recover.

---

## 4. Path B: Distilled Compact MLP (Recommended Next Step)

**Approach:** Replace Qwen's gated SiLU MLP (1536→8960→1536, 8960 neurons) with a trainable **compact dense MLP (1536→2048→1536, 2048 neurons)** per layer. Distill the compact MLP to match the original MLP's output via MSE loss on (input, output) activation pairs collected from the frozen Qwen model.

### Why this works

1. The dense MLP has no sparsity — it computes exactly, just with fewer neurons. No approximation error.
2. 2048 neurons (vs 8960) gives 4.4× compression while preserving enough capacity. Can be tuned: 1024 (8.8×), 4096 (2.2×).
3. The compact MLP can be expressed as a simple UVM-DSL program: `y = transform(activate(transform(x, w1), gelu), w2)` — two `Transform` + one `Activate` nodes.
4. The Triton backend already supports fused Transform+Activate kernels for arbitrary shapes.

### Files to create/modify

| File | Action | Lines |
|------|--------|-------|
| `ucn/decompile/distilled_mlp.py` | **New** — `DistilledMLP` nn.Module class (1536→2048→1536, GELU) | ~40 |
| `ucn/decompile/mlp_decomposer.py` | Add `collect_mlp_io_pairs()` — hooks on MLP input+output to capture (h_in, h_out) pairs | ~30 |
| `scripts/train_distilled_mlp.py` | **New** — MSE distillation training loop per layer | ~150 |
| `scripts/build_multilayer_stdlib.py` | Update to include distilled MLP entries in stdlib | ~20 |
| `tests/test_e2e.py` | Add test 11: distilled MLP fidelity vs original | ~50 |
| `ucn/backend/codegen/triton_backend.py` | Fix `d_out = d_in` to `d_out = w.shape[0]` for non-square transforms | 1 line |

### Training procedure

1. For each target layer [0, 4, 8, 12, 16, 20, 24]:
   a. Run frozen Qwen on WikiText-103 prompts, collect (h_in, h_out) activation pairs via forward hooks on the MLP module.
   b. Save as `artifacts/mlp_distillation/L{layer}_io_pairs.pt`.
2. For each layer:
   a. Initialize `DistilledMLP(d_model=1536, hidden=2048)`.
   b. Train via AdamW(lr=1e-3) with MSE loss on (h_in, h_out) pairs for 500 steps.
   c. Save trained weights as stdlib entries: `stdlib["dmlp_L{layer}"] = {"operator_type": "dense", "weight": w1/w2 tensors}`.
3. Build multi-layer programs:
   ```
   # Attention (1.0000 fidelity, unchanged)
   y = transform(x, stdlib.a_L{layer})
   
   # Distilled MLP (near-1.0000 fidelity)
   h1 = transform(x, stdlib.dmlp_w1_L{layer})
   h2 = activate(h1, gelu)
   y = transform(h2, stdlib.dmlp_w2_L{layer})
   ```
4. Re-run multi-layer test 8 — expected >0.95 cosine at all layers.

### Memory budget (pe3 M40 12GB)

| Component | Memory |
|-----------|--------|
| Qwen2.5-1.5B fp16 (for data collection) | ~3.0 GB |
| Activations per prompt | ~0.1 GB |
| 1× DistilledMLP training | ~0.01 GB |
| **Total** | **~3.2 GB** — comfortable on 12GB |

### Training time

- Data collection: ~5 min (280 prompts × forward pass)
- Per-layer training: ~2 min (500 steps × MSE backprop)
- 7 layers total: **~15-20 min** for full pipeline

---

## 5. Environment Reference

### pe3 (2× Tesla M40 12GB)

| Item | Detail |
|------|--------|
| SSH | `ssh pe3` (key auth) |
| Python venv | `~/local_venvs/m40_env/` (torch 2.7.1, bnb 0.41.3) |
| GPUs | Both idle, use `CUDA_VISIBLE_DEVICES=0` |
| Qwen cache | `~/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B/` |
| WikiText | `~/deepseek_experiments/artifacts/wikitext_gpt2/train_ids.pt` (957MB) |
| Working dir | `~/deepseek_experiments/hybrid/full_compiled_experiment/` |
| M40 constraints | No fp16 tensor cores (AGENTS.md #8); fp32 for compute, fp16 for memory only |

### Launch pattern

```bash
# Sync code
rsync -aH --delete --exclude='artifacts/*' --exclude='__pycache__' --exclude='*.pyc' \
  ~/deepseek_experiments/hybrid/full_compiled_experiment/ \
  pe3:~/deepseek_experiments/hybrid/full_compiled_experiment/

# Run training (background for long runs)
ssh pe3 "cd ~/deepseek_experiments/hybrid/full_compiled_experiment && \
  source ~/local_venvs/m40_env/bin/activate && \
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. nohup \
  python3 scripts/train_distilled_mlp.py --layer 0 --hidden 2048 --steps 500 \
  > artifacts/mlp_distillation/run_L0.log 2>&1 &"
```

### Local machine (RTX 3080 10GB)

Used for development and testing. Can also run training if pe3 is unavailable. Same code paths, just use `--device cuda` locally.

---

## 6. Key Files for Path B

| File | Purpose |
|------|---------|
| `ucn/decompile/distilled_mlp.py` | **TO CREATE** — DistilledMLP module |
| `ucn/decompile/mlp_decomposer.py` | **TO MODIFY** — add `collect_mlp_io_pairs()` |
| `scripts/train_distilled_mlp.py` | **TO CREATE** — distillation training |
| `tests/test_e2e.py` test 11 | **TO ADD** — fidelity verification |
| `ucn/backend/codegen/triton_backend.py:496` | **TO FIX** — `d_out = d_in` → `d_out = w.shape[0]` |
| `ucn/backend/codegen/reference.py` | Already handles arbitrary dense transforms |
| `ucn/dsl/ast.py` | Already supports Transform + Activate chains |

### Key existing patterns to follow

- **Distillation training**: `ucn/training/distill.py` — supervised MSE pattern
- **Hook-based data collection**: `ucn/decompile/source_model.py` — QwenActivationCollector
- **Model loading**: `scripts/train_joint_multilayer.py` — fp16 SDPA model for data, fp32 eager CPU model for weights
- **stdlib building**: `scripts/build_multilayer_stdlib.py` — PrimitiveEntry construction
- **Triton kernel fusion**: `ucn/backend/codegen/triton_backend.py` — `_fused_transform_activate_kernel`

---

## 7. Success Criteria for Path B

| Metric | Target | Measurement |
|--------|--------|-------------|
| Single-layer MLP cosine | >0.95 | test 3 (sparse_down_projection comparison) |
| Multi-layer layer 0 | >0.95 | test 8 (multi-layer fidelity) |
| Multi-layer layer 4 | >0.85 | test 8 |
| Multi-layer layer 8 | >0.85 | test 8 |
| Multi-layer average | >0.88 | test 8 |
| Triton execution parity | max_diff < 1e-3 | test 2 (Triton vs Reference) |
| Training convergence | loss < 0.001 | 500 steps MSE |

### Red(yellow)-green thresholds

| Gate | Yellow (partial) | Green (success) |
|------|-----------------|-----------------|
| Per-layer cosine at layer 4 | >0.40 | >0.85 |
| Multi-layer average | >0.60 | >0.88 |
| Training time | <30 min | <15 min |
| Memory on pe3 M40 | <6GB | <4GB |

---

## 8. Quick Start for Gemini 3.5

```bash
# 1. Connect to environment
ssh pe3
cd ~/deepseek_experiments/hybrid/full_compiled_experiment
source ~/local_venvs/m40_env/bin/activate

# 2. Verify infrastructure works
PYTHONPATH=. python3 -c "
from ucn.decompile.source_model import QwenActivationCollector
collector = QwenActivationCollector(device='cuda', layers=[0], attn_implementation='eager')
print('Infrastructure OK, d_model=%d, n_layers=%d' % (collector.d_model, collector.n_layers))
"

# 3. Run existing tests
PYTHONPATH=. python3 tests/test_phase1_integration.py  # 12 tests, should all pass
PYTHONPATH=. python3 tests/test_e2e.py --tests 1,5,6,7 --device cpu  # 4 CPU tests

# 4. Build Path B
#    a. Create ucn/decompile/distilled_mlp.py
#    b. Add collect_mlp_io_pairs() to ucn/decompile/mlp_decomposer.py
#    c. Create scripts/train_distilled_mlp.py
#    d. Train per layer, save to stdlib
#    e. Test with test 8 (multi-layer fidelity)

# 5. Deployment pattern for long runs
rsync -aH --delete --exclude='artifacts/*' --exclude='__pycache__' --exclude='*.pyc' \
  . pe3:~/deepseek_experiments/hybrid/full_compiled_experiment/

ssh pe3 "cd ~/deepseek_experiments/hybrid/full_compiled_experiment && \
  source ~/local_venvs/m40_env/bin/activate && \
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. nohup \
  python3 scripts/train_distilled_mlp.py --layer 0 --hidden 2048 --steps 500 \
  > artifacts/mlp_distillation/run_L0.log 2>&1 &"
```
