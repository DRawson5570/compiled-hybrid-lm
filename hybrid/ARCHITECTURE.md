# compiled-hybrid-lm: Architecture

This document explains how compiled-hybrid-lm achieves frontier-quality language models on consumer hardware at 100× data efficiency. It is intended for researchers and engineers who want to understand, reproduce, or extend the system.

## Table of Contents

0. [Capability Tracks](#capability-tracks)
1. [Overview](#1-overview)
2. [Compiled Priors Pipeline](#2-compiled-priors-pipeline)
3. [Superposition Steering](#3-superposition-steering)
4. [Co-Training Dynamics](#4-co-training-dynamics)
5. [ZeroQ 4-Bit Distributed Training](#5-zeroq-4-bit-distributed-training)
6. [Cartridge System](#6-cartridge-system)
7. [Memory and Throughput](#7-memory-and-throughput)
8. [Key Design Decisions](#8-key-design-decisions)

---

## Capability Tracks

The system supports three capability attachment tracks. They are complementary,
but they answer different engineering questions.

| Track | Question | Mechanism | Best fit |
| --- | --- | --- | --- |
| Frozen base plus routed runtime cartridges | Can a fixed model load task-specific capability artifacts on demand? | Keep the base model frozen, mount cartridge artifacts, and use a learned router or explicit control plane to activate one cartridge or a small gated set. | Modular products, hot-swappable capabilities, ablations, rollback, and benchmark comparisons against the same base model. |
| Training-time integrated cartridges | Can capability be compiled, co-trained, or baked into the model artifact itself? | Train steering surfaces, adapters, compiled-channel heads, residual priors, or cartridge-like modules during model construction/refinement. | Faster inference, simpler deployment, model-native behavior, and capabilities that should always be present. |
| Agentic tooling and skill loading | Can a model or agent decide which external capability to use while solving a task? | Let the surrounding agent loop discover, load, call, and unload tools, skills, cartridges, or compiled artifacts as needed. | Open-ended workflows, large capability libraries, and tasks where the needed capability is not known before runtime. |

The first track is routed activation among external artifacts. The second track
is baking or co-training capability into the artifact being shipped. The third
track is a higher-level agentic control plane that can discover and use
capabilities dynamically. Reports should name the track being evaluated so a
runtime-router result is not confused with training-time integration or tool use.

---

## 1. Overview

### The Thesis

Pure SGD on massive clusters spends billions of tokens discovering statistical structure that can be pre-computed. Compiled-hybrid-lm replaces those billions of SGD tokens with 21 pre-computed statistical channels injected as activation offsets through a trainable 17K–65K-parameter "cartridge."

**Result:** 4.7B-param transformer, 95% GPU utilization, 87.7 tok/s on 2× Tesla M40 (2015), matched GPT-2 PPL at 3,400× fewer training tokens.

### System Diagram

```
┌──────────────────────────────────────────────────────────────┐
│                   COMPILED PRIORS (pre-computed)              │
│  Witten-Bell n-grams  │  Topic vectors  │  KV cache  │  POS  │
│         ↓                ↓              ↓          ↓        │
│    21-channel feature vector per token position              │
└──────────────────────────────────┬───────────────────────────┘
                                   │ set_weights()
                                   ▼
┌──────────────────────────────────────────────────────────────┐
│              SUPERPOSITION STEERER (65K params)               │
│  per-group MLP gatekeepers  │  RMS-normalized injection      │
│  local[0:6] → mid[6:13] → global[13:21]                     │
└──────────────────────────────────┬───────────────────────────┘
                                   │ forward hooks at 9 layers
                                   ▼
┌──────────────────────────────────────────────────────────────┐
│              FROZEN/MOSTLY-FROZEN TRANSFORMER                 │
│  DeepSeekForCausalLM  │  GPT-2 BPE (V=50257)                 │
│  d=3072, L=40, heads=24  │  explicit Q/K/V/O Linear layers  │
└──────────────────────────────────┬───────────────────────────┘
                                   │ logits
                                   ▼
┌──────────────────────────────────────────────────────────────┐
│                      OUTPUT                                   │
│  logits + head_bias  │  weight-tied embedding projection      │
│  eval_s (steered)  │  eval_b (baseline, no steerer)          │
└──────────────────────────────────────────────────────────────┘
```

---

## 2. Compiled Priors Pipeline

The 21 channels are computed from 119M tokens of WikiText-103, tokenized with GPT-2 BPE (V=50257).

### Channel Groups

| Group | Channels | Indices | Description |
|-------|----------|---------|-------------|
| Local | 6 | 0–5 | unigram, bigram fast/slow, trigram fast/slow, skip-2 |
| Mid | 7 | 6–12 | skip-3, recency, entropy, shape, global unigram, PPMI cos/max |
| Global | 8 | 13–20 | PPMI norm, punct density, repetition, unique ratio, topic, KV, POS, spare |

### Runtime Computation

Channels are not pre-dumped to disk. They are computed live per batch:

- **GPU channels (9–20):** Vectorized tensor operations in `gpu_channels.py`. Topic tracking via weighted decay accumulation, KV retrieval via causal cosine similarity, punct density via cumulative sum, repetition via roll-and-compare. Zero Python loops.
- **CPU channels (0–8):** O(1) streaming n-gram statistics in `FastNgramFeatures`. Uses running scalar sums instead of O(V²) count tables — a 50K² table would be 2.5B entries. Pre-fetched by background DataLoader workers.

### Witten-Bell Smoothing

N-gram probabilities use Witten-Bell smoothing rather than Kneser-Ney:

```
P_WB(w|c) = (c(c,w) + λ·P(w)) / (c(c) + λ)
λ = T(c) / (T(c) + N(c))
```

Where `T(c)` is the number of distinct tokens following context `c`, and `N(c)` is the total count. Witten-Bell is chosen because it provides stable estimates with small context counts — critical when operating on 119M tokens rather than billions.

### Topic Vectors

K=50 topic dimensions, trained via SVD on co-occurrence statistics. A running topic vector is maintained via exponential decay (γ=0.95) over the token stream. The channel value is the dot product of the current token's topic embedding with the running topic vector — high values indicate topical coherence.

### KV Semantic Cache

A causal cosine-similarity retrieval mechanism. Each token's PPMI embedding is compared against all previous tokens within a 128-token window. The maximum cosine similarity becomes the KV channel value. Provides a local semantic coherence signal without external databases.

---

## 3. Superposition Steering

### Architecture

`SuperpositionSteererV3` (`superposition_steerer_v3.py`) is a 65K-parameter module that injects 21-channel features as activation offsets at 9 transformer layer positions.

**Per-group MLP gatekeepers:**

```
local[0:6]  → Linear(6→12) → GELU → Linear(12→6) → softmax
mid[6:13]   → Linear(7→14) → GELU → Linear(14→7) → softmax
global[13:21] → Linear(8→16) → GELU → Linear(16→8) → softmax
```

Each group has a learned steering matrix (`steer_local`: [6, d_model], `steer_mid`: [7, d_model], `steer_global`: [8, d_model]). The softmax output weights the steering vectors, and the weighted sum is the per-token offset:

```
offset = einsum('btc, cd → btd', softmax(MLP(weights)), steer_matrix)
```

### RMS-Normalized Injection

The offset magnitude is normalized to match the hidden state's RMS to prevent scale drift:

```
h_rms = sqrt(mean(h²))
o_rms = sqrt(mean(offset²))
normalized_offset = offset × (h_rms / max(o_rms, 1e-8))
h' = h + γ · normalized_offset
```

γ is a learned per-layer scalar (initialized at 0.01). **Critical**: all RMS math must happen in float32, not float16. fp16 `pow(2)` overflows or underflows silently (range ±65504, min subnormal ~6e-8). The steerer code casts to float32 before RMS computation and casts back after.

### Layer Routing

Hooks are installed at 9 positions, each assigned to one of three groups:

| Layers | Group | Role |
|--------|-------|------|
| 0, 1, 2 | Local | Inject local n-gram statistics early |
| 4, 5, 6 | Mid | Inject mid-range features at intermediate depth |
| 8, 9, 10 | Global | Inject topic/KV/POS features near output |

### Hook Mechanism

Forward hooks fire after each target layer's output. The steerer reads `_current_weights` (set by `steerer.set_weights(features)` before the forward pass), computes the per-group offset, and modifies the hidden state:

```python
def _steer_layer(self, h, layer_idx):
    group = self.layer_routing[layer_idx]
    w = self._current_weights
    if group == 'local':
        offset = einsum(softmax(local_mlp(w[:,:,0:6])), steer_local)
    # ... mid, global similarly
    h_float = h.float(); offset_float = offset.float()  # fp32 safety
    h_rms = h_float.pow(2).mean(-1, keepdim=True).sqrt()
    o_rms = offset_float.pow(2).mean(-1, keepdim=True).sqrt().clamp(min=1e-8)
    return h + (gamma * offset_float * (h_rms / o_rms)).to(dtype=h.dtype)
```

---

## 4. Co-Training Dynamics

### Why Co-Training

The model and steerer are trained simultaneously. This is not a post-hoc injection — the model learns to incorporate the steerer's signals, and the steerer learns to produce useful offsets in the model's activation space.

**Frozen model + trainable steerer alone does not work.** The gradient from the output must propagate backward through all frozen layers to reach the steerer hooks. With 28–40 frozen layers, this gradient is attenuated to near-zero. The steerer receives no useful training signal and cannot improve accuracy. This was verified experimentally (see self-improvement-research/FAILURE_ANALYSIS.md).

### Gradient Flow

```
loss → LM head → layer L → ... → layer 0 → input
  ↑                                    ↑
  └── steerer hooks modify activations ─┘
  ↑                                    ↑
  └──── gradient flows back through ────┘
       trainable model + steerer params
```

The model uses a moderate LR (3e-4) while the steerer uses a higher LR (1e-3). Both use AdamW with weight decay 0.1. An orthogonal penalty (0.001×) encourages the 21 steering vectors to span distinct directions:

```python
orthogonal_penalty = mean((normalize(steer_vectors) @ normalize(steer_vectors).T - I)²)
```

### Training vs Evaluation

Two PPL numbers are tracked per epoch:

- **eval_s (steered):** Model + steerer with compiled channel features active. Measures the full system.
- **eval_b (baseline):** Model only, no steerer. Measures what the model learned independently.

The gap between eval_s and eval_b represents the steerer's contribution. A large gap (eval_s << eval_b) means the steerer is providing significant signal.

---

## 5. ZeroQ 4-Bit Distributed Training

### Problem: Model Doesn't Fit on One GPU

A 4.7B-param model in fp16 is ~9.4GB. With optimizer states and activations, this exceeds even a 24GB GPU. ZeroQ solves this via quantization + sharding.

The distributed trainer also exposes a `700m` DeepSeek/CMI configuration for
single RTX 3080 runs: `d_model=1536`, `n_layers=22`, `n_heads=16`, `d_ff=6144`,
and GPT-2 BPE vocabulary, yielding `701,192,785` parameters before ZeroQ
partitioning.

### Standard Path (Gather/Release)

The original ZeroQ approach installs hooks on every `nn.Linear` layer:

```
forward: gather 4-bit shards → dequantize to fp16 → compute → release
backward: gather → compute gradients → all-reduce → release
```

On SYS-topology GPUs (PCIe across NUMA nodes), the per-layer `dist.all_gather` serializes and dominates runtime. 240 Linear layers × ~300 MiB/s NCCL = 87% idle time.

### 4-Bit Compute Mode (Current)

After streaming partition, each `nn.Linear` is converted to `bnb.nn.Linear4bit` with `Params4bit`:

```python
# Gather full 4-bit weight ONCE per layer (not per forward pass):
coordinator.fetch_params([param_id])
packed_2d = assembled_packed.clone().view(-1, 1)
param_4bit = bnb.nn.Params4bit(packed_2d, quant_state=gathered_state, ...)
new_linear = bnb.nn.Linear4bit(in_features, out_features, ...)
new_linear.weight = param_4bit
# Remove ZeroQ hooks — Linear4bit handles matmul natively
wrapper.remove_hooks()
```

The `bnb.nn.Linear4bit.forward()` uses a fused `matmul_4bit` CUDA kernel that operates directly on the 4-bit packed representation. No dequantization, no gather, no NCCL per layer. The weight is always 4-bit and always live on GPU.

**Result:** 8.3× faster, 95% GPU utilization, 20°C cooler.

### Streaming Partition

Models are built on CPU, then weights are processed one at a time:

```
for each Linear.weight on CPU:
    move to GPU (one tensor, ~100MB max)
    quantize to 4-bit NF4 (blocksize=64, nf4 quant type)
    shard across ranks via NCCL all-gather
    store packed shard + absmax
    free full-precision GPU copy
```

Peak GPU memory is the largest single tensor (~116MB for ffn1), not the full model (~9.4GB). This enables loading models that are larger than any individual GPU.

### Config

```python
ZeroQConfig(
    compute_dtype=torch.float32,  # Maxwell has no fp16 tensor cores
    double_quant=True,
    blocksize=64,
    async_gather=True,
    frozen_only=True,             # only quantize frozen params
    compute_in_4bit=True,         # 4-bit compute mode
)
```

### Bitsandbytes Version Lock

Maxwell (SM 5.2) requires **bitsandbytes == 0.41.3** with **triton == 3.3.1**. Bitsandbytes 0.46.1+ dropped Maxwell support entirely. The version must be pinned — any other combination will fail to import.

---

## 6. Cartridge System

### GPT-2-Large ZeroQ Assistant Lane

The web-demo assistant lane can run a frozen Hugging Face GPT-2-family substrate
under ZeroQ on a single RTX 3080 while training only a CMI task cartridge. The
runtime entry points are:

- `hybrid/train_gpt2_zeroq_chat.py` — trains a `FeatureConditionedAdapterSteerer`
    against assistant-only chat loss while the GPT-2-large substrate remains
    frozen and ZeroQ-partitioned.
- `hybrid/gpt2_zeroq_assistant.py` — loads the frozen ZeroQ substrate plus an
    optional cartridge and can generate baseline-vs-cartridge side-by-side JSON.
- `hybrid/eval_gpt2_zeroq_assistant.py` — scores the same side-by-side runtime
    on the assistant gate used for the small-base cartridge lane.

GPT-2 ties `lm_head.weight` to `transformer.wte.weight`, so the ZeroQ GPT-2 lane
keeps token and position embeddings resident on the GPU as frozen parameters and
partitions the remaining backbone weights. This avoids output-projection device
mismatch while preserving the cartridge-only optimization contract.

The current selected demo candidate is
`artifacts/gpt2_large_zeroq_chat_3080_prod_v5_repair/latest_chat_cartridge.pt`.
It scores `23/24` on the side-by-side assistant gate versus raw GPT-2-large at
`6/24`, with clear wins on greeting, facts, story generation, science, project
definitions, health lists, coding, arithmetic, safety refusal, writing,
workflows, calibration, and hot-swapping. The remaining miss is creator-name
anchoring in the exact "I am your creator, Douglas" turn, so it is a strong
side-by-side demo candidate rather than the end state for the 4B assistant lane.

### Website Benchmark Export

`hybrid/export_benchmark_demo.py` exports a website-ready JSON payload for the
popular benchmark comparison: compiled-hybrid baseline with cartridge injection
disabled versus the same checkpoint with `SuperpositionSteererV3` enabled. The
default source is `artifacts/steerer_v4/steerer_best_s.pt`, which records
WikiText-103 validation `eval_b=37.2111` and `eval_s=28.2080`. The exporter
fails by default if the cartridge does not improve baseline perplexity, so stale
or bad checkpoints do not quietly become demo copy.

### CartridgeManifest

Each cartridge carries metadata for compatibility checking:

```python
@dataclass(frozen=True)
class CartridgeManifest:
    cartridge_id: str          # e.g., "wiki-capability"
    role: CartridgeRole        # SUPERPOSITION_STEERER, DOMAIN_CAPABILITY, TASK_CAPABILITY
    base_model_id: str         # e.g., "c4-124m-v4"
    tokenizer_id: str          # e.g., "gpt2-bpe"
    channel_schema: str        # "cmi-21ch-v3"
    inject_layers: tuple       # (0, 1, 2, 4, 5, 6, 8, 9, 10)
    steerer_class: str         # "SuperpositionSteererV3"
    parameter_count: int       # 16,796 for V4
```

### SteererCartridgeRack

Multiple steerers can be mounted simultaneously. For general steerers or deliberately blended domains, they can be composed additively. For task cartridges, the validated production mode is gated activation: mount every cartridge, let the router select the compatible cartridge for the prompt, activate only that cartridge, and then run the rack in `chain` mode.

```python
rack = SteererCartridgeRack()
rack.mount(wiki_manifest, wiki_steerer)
rack.mount(chat_manifest, chat_steerer)
rack.register_hooks(model)

# Mode switching at runtime:
rack.activate('wiki-capability', True)
rack.activate('chat-capability', False)
```

The rack registers one hook per target layer. For each active steerer, it computes that steerer's layer delta independently and sums them. This preserves separate cartridges while avoiding hook-order coupling.

### Hot-Swap

Cartridges can be loaded, unloaded, and composed at runtime without restarting the model. A single frozen C4 base model can serve Wikipedia, code, chat, or any trained domain by swapping a 70KB file.

### Qwen Learned-Router Validation

The Qwen cartridge rack has two validated deployment tracks as of 2026-05-25:

| Track | Runtime shape | What trains | Runtime router? | Validation |
|---|---|---|---|---|
| Loadable cartridge rack | Frozen `Qwen/Qwen2.5-1.5B` + mounted adapter cartridges | Individual cartridges plus a small learned router head | Yes | pe3 `learned_router_gated_chain_eval.json`: private_facts 53/60; arithmetic, code_labels, safety_labels, instruction_format 100%; no saved-score regressions |
| Baked native LoRA | Frozen `Qwen/Qwen2.5-1.5B` + native LoRA adapter | LoRA matrices only; base weights frozen | No | pe3 `baked_lora_native_300`: eval_loss 4.1611 -> 0.0739; bounded generation eval 34/40, heldout 3/4 |

The loadable rack uses a learned router, not keyword routing. The router artifact type is `qwen_embedding_linear_v1`: frozen Qwen embeds the prompt by mean-pooling the final hidden state, and a trained linear head selects from the mounted cartridge IDs. The router is a control-plane module; it does not modify Qwen weights. At generation time the runtime uses:

```
prompt -> frozen Qwen embedding -> learned router head -> activate selected cartridge -> gated-chain generation
```

This gating is required for safe composition. Naive all-active composition was tested and is unsafe for task cartridges; it caused severe interference, including private-fact collapse to 0/60. The deterministic prompt router remains only as a fallback/proof harness. Product evaluation should use the learned router artifact at `learned_router/qwen_learned_router.pt`.

The rack was also re-tested one suite at a time through the trained router with `--composition-mode gated-chain`, `--skip-baseline`, and `--max-tokens 8`. The reports live under `learned_router_one_by_one/` and confirm the router selects the expected cartridge for each suite: private_facts 53/60, arithmetic 32/32, code_labels 24/24, safety_labels 24/24, instruction_format 24/24, all with `saved_score_regression=false`.

The baked native-LoRA track is separate. It distills the suite behavior into one reloadable adapter artifact, saved as `adapter.pt` plus `adapter_config.json`, and can be loaded with `QwenBakedLoraRunner.from_adapter(...)`. This path removes the runtime router, but it also removes cartridge-level hot-swapping, selective activation, and per-cartridge introspection. It is a deployment/packaging path, not a replacement for the modular router architecture.

---

## 7. Memory and Throughput

### 4.7B Model, 2× M40 24GB

| Mode | VRAM/GPU | Throughput | GPU Util | Note |
|------|----------|------------|----------|------|
| 4-bit compute | 5.4 GB | 87.7 tok/s | 95% | Current production |
| Gather/release | 17.7 GB | 2.86 tok/s | 13% | Legacy, NCCL-bound |
| No quantization | OOM | — | — | Won't fit |

### Memory Breakdown (4-bit compute)

| Component | Per GPU |
|-----------|---------|
| 4-bit weights (Linear layers) | 1.2 GB |
| Embeddings (fp32, unpartitioned) | 540 MB |
| LayerNorm + biases | <10 MB |
| Steerer (65K params) | <1 MB |
| Activations (batch=6, seq=64) | ~3.5 GB |
| **Total** | **~5.4 GB** |

### Throughput Scaling

Throughput scales sub-linearly with batch size due to the 21-channel feature computation overhead. At batch=1, the CPU FastNgramFeatures dominates. At batch=6, the GPU is 95% utilized. Estimated saturation around batch=8–12 on the same hardware.

### Projected Cluster Capacity (5× M40 24GB)

| Model Size | 4-bit Weight/GPU | Feasibility |
|------------|------------------|-------------|
| 4.7B | 1.2 GB | Running |
| 10B | 2.5 GB | Batch ≤ 4 |
| 20B | 5.0 GB | Batch=1 |
| 30B | 7.5 GB | With checkpointing |
| 35B | 8.8 GB | Activation-bound |

---

## 8. Key Design Decisions

### Why explicit Q/K/V/O Linear layers (not fused MultiheadAttention)?

Fused `nn.MultiheadAttention` stores Q/K/V as a single `in_proj_weight` tensor. ZeroQ quantizes it as one blob, but the gather/release hooks interact poorly with the internal attention logic. Separate `nn.Linear` per projection means each weight is independently quantizable and the hooks fire on predictable module boundaries. This costs 3× the parameters for attention projections but enables the 4-bit compute path.

### Why Witten-Bell over Kneser-Ney?

Kneser-Ney requires tracking lower-order context frequencies, which doubles the memory and compute budget. On 119M tokens (not billions), Witten-Bell provides stable estimates with fewer parameters. The empirical difference on WikiText is <2% PPL.

### Why weight-tied embeddings?

The output projection `logits = tok_emb.weight.T @ hidden + head_bias` means the embedding matrix must stay materialized (unpartitioned) at all times — it's needed both at input (embedding lookup) and output (projection). At V=50257, d=3072, this is 540MB fp32. Including it in the trainable surface prevents ZeroQ from releasing it mid-forward-pass.

### Why float32 RMS normalization in hooks?

fp16 range is ±65504. `h.pow(2)` on a hidden state with norm ~10 produces values that can overflow (>65504) or underflow (<6e-8). Casting to float32 before `pow(2).mean().sqrt()` eliminates this entirely. The cast-back overhead is negligible compared to the matmul cost. Not catching this caused the original self-improvement cartridge work to silently produce NaN gradients.

### Why co-training and not frozen model + trainable cartridge?

A frozen 4.7B model cannot be meaningfully steered by a 65K-param offset module. The gradient from the loss must propagate through 40 frozen layers to reach the steerer hooks — attenuated to effectively zero. The model must co-adapt with the steerer. This was the key finding from the self-improvement failure analysis.

### Why LoRA/QLoRA for External Models (Qwen, Llama, etc.)

Our custom models co-train with the steerer from scratch — the model learns to accept activation offsets during formative training. External models like Qwen were trained without a steerer. Freeze them + attach steerer = gradient attenuated to zero through frozen layers.

LoRA/QLoRA solves this by adding small trainable adapters as **signal bridges** — they conduct the steerer's gradient through the frozen model without modifying original weights. The adapters don't need to be large; they just provide a low-resistance path for the gradient. This is independent of model size — even a 1.5B model needs LoRA if it wasn't co-trained with a steerer.
