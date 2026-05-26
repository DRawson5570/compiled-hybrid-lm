# Hybrid LLM — Complete Manual

Author: Douglas Rawson
Date: 2026-05-25
Repo: `github.com/DRawson5570/compiled-hybrid-lm`

---

## Table of Contents

1. [What This Is](#1-what-this-is)
2. [Quick Start](#2-quick-start)
3. [Architecture Overview](#3-architecture-overview)
4. [Hardware Requirements](#4-hardware-requirements)
5. [Setup](#5-setup)
6. [Data Pipeline](#6-data-pipeline)
7. [Compiled Channels](#7-compiled-channels)
8. [Superposition Steering](#8-superposition-steering)
9. [ZeroQ Distributed Training](#9-zeroq-distributed-training)
10. [HF Model Wrapper](#10-hf-model-wrapper)
11. [Training](#11-training)
12. [Evaluation](#12-evaluation)
13. [Generation](#13-generation)
14. [Results](#14-results)
15. [Troubleshooting](#15-troubleshooting)

---

## 1. What This Is

A hybrid language model that combines **compiled** (deterministic, count-based) and **learned** (SGD-trained) components. The compiled substrate captures corpus statistics that transformers waste billions of tokens rediscovering. The learned substrate handles what counting can't: composition, long-range dependencies, and context-dependent reasoning.

The result: a model that beats GPT-2 Small (PPL=29.0) with ~1/100th the training data.

---

## 2. Quick Start

Prove the pipeline works in 10 minutes on any machine with a GPU:

```bash
# Clone and install
git clone https://github.com/DRawson5570/compiled-hybrid-lm.git
cd compiled-hybrid-lm
pip install torch transformers

# Run the validation script
python3 hybrid/quickstart.py
```

This loads a pre-built 124M C4 base model, runs a WikiText steerer cartridge, and evaluates PPL. Expected output: eval_s < 35, eval_b < 50.

For a larger test with ZeroQ 4-bit distributed training:

```bash
# Single GPU, tiny model (verifies infrastructure):
torchrun --nproc_per_node=1 hybrid/train_4b_distributed.py \
  --backend dense --model-config test --epochs 1 --steps 1 --batch 1

# Multi-GPU, 4-bit compute (verifies ZeroQ):
torchrun --nproc_per_node=2 hybrid/train_4b_distributed.py \
  --backend zeroq --model-config test --compute-in-4bit \
  --epochs 1 --steps 1 --batch 1 --zeroq-path ~/ZeroQ
```

---

## 3. Architecture Overview

```
                    ┌──────────────────────────────────┐
                    │         Compiled Channels         │
                    │  (KN n-gram, attention caches,    │
                    │   cluster mixture, PPMI, KNN,     │
                    │   word shape, recency)            │
                    │          V=8000 or V=50257        │
                    └──────────────┬───────────────────┘
                                   │ per-position log-probs
                    ┌──────────────▼───────────────────┐
                    │        WindowMLP Blender         │
                    │  (1.3M params, learns to mix     │
                    │   21 channels contextually)      │
                    └──────────────┬───────────────────┘
                                   │ blended log-prob
                    ┌──────────────▼───────────────────┐
                    │        Hybrid Ensemble           │
                    │  p(y) = α·p_compiled + (1-α)·p_neural │
                    └──────────────────────────────────┘
                                   │
                    ┌──────────────┴───────────────────┐
                    │                                   │
          ┌─────────▼──────────┐          ┌─────────────▼──────────┐
          │   Neural LM        │          │   Compiled Artifacts   │
          │  (DeepCausalLM)    │          │   - KN count tables    │
          │  12-36 layers      │          │   - PPMI embeddings    │
          │  d_model=256-768   │          │   - Attention caches   │
          │  PPMI init (V=8000)│          │   - Cluster centroids  │
          └────────────────────┘          └────────────────────────┘
```

### Key Code Paths

| Path | File | Purpose |
|---|---|---|
| BPE-8000 hybrid | `train_hybrid_bpe8000.py` | Train neural LM in BPE-8000 space, blend with v33 compiled channels |
| GPT-2 BPE hybrid | `dump_gpt2_channels_v2.py` → `train_gpt2_blender.py` | Build GPT-2 compiled channels, train blender, blend with neural LM |
| C4 multi-corpus | `train_c4_v2.py` | Train neural LM on C4+WikiText interleaved stream |
| Generation | `generate_hybrid.py` | Autoregressive sampler with temperature, top-p, repetition penalty |
| Surfaces API | `surfaces/{inject,retract,compose,provenance}.py` | Component injection/retraction with provenance tracking |
| Public benchmarks | `capability_pipeline.py` | MMLU, HellaSwag, GSM8K, HumanEval, IFEval |

### Tokenizers

- **BPE-8000**: Custom 8K-vocab tokenizer at `~/llm_decoupling/artifacts/bpe_wiki/tokenizer.json`. Used by the compiled v33 pipeline and BPE-8000 neural LM.
- **GPT-2 BPE**: Standard HuggingFace `gpt2` tokenizer (V=50257). Used for public benchmark comparison.

### Vocabulary Bridge

The compiled model lives in BPE-8000 space. The GPT-2 BPE compiled channels are count-based approximations rebuilt for V=50257. They don't match the full 21-channel v33 pipeline but are sufficient to demonstrate the thesis.

---

## 4. Hardware Requirements

| Component | Minimum | Recommended |
|---|---|---|
| GPU | 10 GB VRAM (RTX 3080) | 24 GB (M40, 2× for parallel) |
| RAM | 32 GB | 64 GB |
| Disk | 100 GB | 500 GB (C4 cached on SSD) |
| Network | Internet for dataset download | Fast connection for C4 streaming |

This project was developed on:
- **pe2/pe3**: PowerEdge servers with Tesla M40 24GB GPUs
- **Local workstation**: RTX 3080 10GB, 125 GB RAM, 2TB NVMe

---

## 5. Setup

### 4.1 Clone and Dependencies

```bash
git clone https://github.com/DRawson5570/compiled-hybrid-lm.git
cd compiled-hybrid-lm

# Python venv
python3 -m venv .venv && source .venv/bin/activate
pip install torch numpy transformers datasets tokenizers

# Link to existing compiled artifacts (from ~/llm_decoupling)
# The training scripts expect these paths:
#   ~/llm_decoupling/artifacts/compiled_wiki_lm_v11/cache_lm_ids.pt
#   ~/llm_decoupling/artifacts/compiled_wiki_lm_v23/kn5_22m.pkl
#   ~/llm_decoupling/artifacts/compiled_wiki_lm_v5/compiled_lm.pt
#   ~/llm_decoupling/artifacts/bpe_wiki/tokenizer.json
```

### 4.2 Remote Hosts (pe2, pe3)

```bash
# Sync code
rsync -az ~/deepseek_experiments/ pe2:~/deepseek_experiments/

# Venv on remotes: ~/local_venvs/m40_env/
# Run with: CUDA_VISIBLE_DEVICES=0 ~/local_venvs/m40_env/bin/python3
```

### 4.3 C4 Dataset (optional, for GPT-2 BPE training)

```bash
# Set cache location (use SSD if available)
export HF_HOME=/media/drawson/SSD-PGU3/hf_cache
export HF_DATASETS_CACHE=/media/drawson/SSD-PGU3/hf_cache/datasets

# Pre-cache (takes ~5 hours, ~305 GB compressed)
python3 cache_c4.py

# Or stream on-the-fly (no disk space needed, slower per epoch)
```

---

## 6. Data Pipeline

### 5.1 WikiText-103 (BPE-8000)

The main repo provides pre-tokenized BPE-8000 data:

```python
from compile_wiki_lm_v13 import load_or_build_tokens
ids = load_or_build_tokens(None, None, None)  # loads from cache_lm_ids.pt
# ~22.7M tokens, V=8000
# Split: train=0..22M, val=22M..22.03M, eval=22.03M..22.13M
```

### 5.2 WikiText-103 (GPT-2 BPE)

```bash
# Tokenize with GPT-2 BPE
python3 hybrid/tokenize_wikitext_gpt2.py
# Output: artifacts/wikitext_gpt2/{train,validation,test}_ids.pt
# train=119.7M tokens, val=251K, test=287K
# Uses HuggingFace splits (document-disjoint)
```

### 5.3 C4 (GPT-2 BPE, streaming)

```python
from datasets import load_dataset
c4 = load_dataset('allenai/c4', 'en', split='train', streaming=True,
                  trust_remote_code=True)
# ~364M documents, ~750 GB of text
# Streaming mode: reads from local cache if available, downloads otherwise
```

---

## 7. Compiled Channels

Rebuilt for GPT-2 BPE (V=50257). Uses efficient O(T) streaming channels.

```bash
# Build priors (one-time):
python3 build_v3_priors.py
# → artifacts/compiled_priors_v3/word_topics.pt, pos_stats.pkl

# Verify:
python3 quickstart.py
```

The 21 compiled channels are: 6 local (uni, bi_fast, bi_slow, tri_fast, tri_slow, skip2), 7 mid (skip3, recency, entropy, shape, global_uni, ppmi_cos, ppmi_max), 8 global (ppmi_norm, punct_density, repetition, unique_ratio, topic, KV, POS, spare).

Channel features are computed per batch at runtime via `gpu_channels.py` (GPU-vectorized) and `FastNgramFeatures` (CPU O(1) n-gram statistics). No pre-computed log-probability dumps needed — features are live-computed from training data during the forward pass.

---

## 8. Superposition Steering

The SuperpositionSteererV3 (`superposition_steerer_v3.py`) injects 21-channel compiled features as activation offsets at specific transformer layers via forward hooks.

### Architecture

- **21 channels**: 6 local + 7 mid + 8 global (n-gram, topic, KV, POS, register)
- **Per-group MLP gatekeepers**: learn to weight channel contributions per layer
- **RMS-normalized injection**: offsets scaled to match hidden state RMS (fp32-safe)
- **Layer routing**: 9 hooks on layers [0,1,2,4,5,6,8,9,10] (configurable)

### Co-training

```bash
cd hybrid
python3 -u train_steerer_v4.py \
  --neural-ckpt artifacts/c4_v2_768_x30/best.pt \
  --resume-model artifacts/c4_v2_768_x30/best.pt \
  --epochs 200 --steps 500 --batch 8
```

### Domain Cartridges

Hot-swappable, linearly composable:
- **Domain**: WikiText, Python, or custom corpus
- **Chat**: Conversational via `chat_cartridge.py`
- **Size**: ~17K–65K parameters, ~70KB on disk

```bash
python3 chat_cartridge.py --mode chat \
  --base-model artifacts/steerer_v4/steerer_best_b.pt \
  --chat-cartridge artifacts/steerer_chat/chat_cartridge.pt
```

### Qwen Cartridge Rack: Learned Router Track

Use this track when cartridges must remain modular, hot-swappable, auditable, and independently replaceable. Qwen stays frozen. The individual adapter cartridges are mounted into a rack, and a learned control-plane router selects which cartridge is active for each prompt.

This is not keyword routing. The validated router artifact is `qwen_embedding_linear_v1`: frozen Qwen embeds each prompt, then a trained linear head selects one of the mounted cartridge IDs. The deterministic prompt router is only a fallback/proof harness.

```bash
cd ~/deepseek_experiments

# Train or rebuild the learned router from frozen-Qwen prompt embeddings.
CUDA_VISIBLE_DEVICES=0 ~/local_venvs/m40_env/bin/python -m hybrid.cartridge_harness.cli train-router \
  --model Qwen/Qwen2.5-1.5B \
  --device cuda \
  --out-dir artifacts/qwen_cartridge_rack_full_20260525_171513/learned_router \
  --epochs 300 \
  --lr 0.003

# Evaluate all mounted cartridges through the learned router and gated-chain runtime.
CUDA_VISIBLE_DEVICES=0 ~/local_venvs/m40_env/bin/python -m hybrid.cartridge_harness.cli eval-loaded-rack \
  --model Qwen/Qwen2.5-1.5B \
  --device cuda \
  --out-dir artifacts/qwen_cartridge_rack_full_20260525_171513 \
  --router-path artifacts/qwen_cartridge_rack_full_20260525_171513/learned_router/qwen_learned_router.pt \
  --composition-mode gated-chain \
  --report artifacts/qwen_cartridge_rack_full_20260525_171513/learned_router_gated_chain_eval.json
```

Expected validated pe3 result for the current rack:

| Suite | Result |
|---|---:|
| private_facts | 53/60 |
| arithmetic | 32/32 |
| code_labels | 24/24 |
| safety_labels | 24/24 |
| instruction_format | 24/24 |

All suites should report `saved_score_regression=false`. If a task cartridge rack regresses under all-active composition, do not treat that as a rack failure; all-active task composition is a known unsafe diagnostic mode. Use `--composition-mode gated-chain` with the learned router for product evaluation.

### Qwen Baked Adapter: Native LoRA Track

Use this track when deployment wants one fused adapter and does not need runtime cartridge selection. Qwen base weights still remain frozen, but native LoRA matrices are trained inside the model and saved as a reloadable adapter artifact.

This path avoids PEFT/bitsandbytes on pe3. The pe3 environment has broken PEFT/bitsandbytes Triton imports for this use case, so the harness uses native LoRA wrappers and saves `adapter.pt` plus `adapter_config.json`.

```bash
cd ~/deepseek_experiments

CUDA_VISIBLE_DEVICES=0 ~/local_venvs/m40_env/bin/python -m hybrid.cartridge_harness.cli train-baked-lora \
  --model Qwen/Qwen2.5-1.5B \
  --device cuda \
  --out-dir artifacts/qwen_cartridge_rack_full_20260525_171513/baked_lora_native_300 \
  --steps 300 \
  --eval-every 50 \
  --lr 0.0002 \
  --lora-r 16 \
  --lora-alpha 32 \
  --lora-dropout 0.05 \
  --final-eval-limit 40
```

Expected validated pe3 result for the current baked artifact:

| Metric | Value |
|---|---:|
| Final eval loss | 0.0739 |
| Bounded generation eval | 34/40 |
| Heldout in bounded eval | 3/4 |
| Best adapter | `artifacts/qwen_cartridge_rack_full_20260525_171513/baked_lora_native_300/best_adapter` |

Reload a baked adapter for inference:

```python
from hybrid.cartridge_harness.qwen import QwenBakedLoraRunner

runner = QwenBakedLoraRunner.from_adapter(
    "artifacts/qwen_cartridge_rack_full_20260525_171513/baked_lora_native_300/best_adapter",
    device="cuda",
)
print(runner.generate("Project Atlas access code:", max_tokens=8))
```

On pe3, this reload smoke loaded 196 native LoRA modules and generated `LUMEN-506` from the saved artifact.

---

## 9. ZeroQ Distributed Training

4-bit quantized ZeRO-3 via `backends.py` and `train_4b_distributed.py`.

### 4-bit Compute Mode

Converts `nn.Linear` → `bnb.nn.Linear4bit` after partition. Fused `matmul_4bit` kernel eliminates per-layer gather/release — 8.3× faster on Maxwell GPUs.

### Launch

```bash
# 4.7B model, 2 GPUs, steerer enabled:
torchrun --nproc_per_node=2 --nnodes=1 --node_rank=0 \
    --master_addr=localhost --master_port=29500 \
    hybrid/train_4b_distributed.py \
    --backend zeroq --model-config 4b \
    --train-surface cmi_steerer --compute-in-4bit \
    --epochs 100 --steps 50 --batch 6 \
    --zeroq-path ~/ZeroQ
```

### Results

| Model | GPUs | VRAM/GPU | Throughput | GPU Util |
|-------|------|----------|------------|----------|
| 4.7B (4-bit compute) | 2× M40 24GB | 5.4 GB | 87.7 tok/s | 95% |
| 4.7B (gather/release) | 2× M40 24GB | 17.7 GB | 2.86 tok/s | 13% |

### Bitsandbytes Lock

M40 requires bitsandbytes == 0.41.3 + triton == 3.3.1. **Must pin both.** 0.46.1+ dropped Maxwell.

---

## 10. HF Model Wrapper

`hf_deepseek.py` registers DeepSeekForCausalLM with HuggingFace.

| Config | d_model | L | Params |
|--------|---------|---|--------|
| test | 192 | 2 | 10M |
| 3b | 2688 | 32 | 2.9B |
| 4b | 3072 | 40 | 4.7B |

Explicit `nn.Linear` layers for Q/K/V/O — every weight independently quantizable.

---

## 11. Training

### Compiled Priors

```bash
python3 build_v3_priors.py  # → artifacts/compiled_priors_v3/
```

### Resuming

```bash
./resume_pe2_4b_zeroq_4bit.sh --epochs 50 --batch 4        # resume
./resume_pe2_4b_zeroq_4bit.sh --fresh --epochs 100 --batch 6  # fresh
```

---

## 12. Evaluation

### PPL on WikiText-103

```bash
# BPE-8000 (automatic with train_hybrid_bpe8000.py)
# GPT-2 BPE (automatic with train_gpt2_neural_lm.py or as in train_gpt2_blender.py)

# Standalone eval:
python3 hybrid/cross_tree_eval.py --out artifacts/cross_tree_results.json
```

### Public Benchmarks

```bash
python3 hybrid/capability_pipeline.py benchmark \
  --ckpt artifacts/hybrid_gpt2/gpt2_lm_best.pt --quick
# Evaluates: MMLU, HellaSwag, GSM8K, HumanEval, IFEval
# Uses real HuggingFace datasets, real model forward passes
```

---

## 13. Generation

```bash
# Production chat (with steerer cartridge):
python3 chat_cartridge.py --mode chat \
  --base-model artifacts/steerer_v4/steerer_best_b.pt \
  --chat-cartridge artifacts/steerer_chat/chat_cartridge.pt

# Raw generation (no cartridge):
python3 chat_gpt2.py
```

Features: temperature, top-p, top-k, repetition penalty, stop markers, n-gram repetition detection, multi-turn conversation history.

---

## 14. Results

### V4 Co-trained Steerer (124M params, GPT-2 BPE)

| Metric | Value |
|--------|-------|
| eval_b (standalone) | 35.6 |
| eval_s (steered) | 28.2 |
| Training time | 4.5 hours on RTX 3080 |
| Training tokens | 154M (3,400× fewer than GPT-2) |

### ZeroQ 4-bit (4.7B params, in progress)

| Metric | Epoch 1 | Epoch 23 |
|--------|---------|----------|
| eval_s | 44,081 | 6,027 |
| Throughput | 87.7 tok/s at batch=6 |
| GPU util | 95% on 2× M40 |
| VRAM | 5.4 GB per GPU |

### Domain Cartridge (frozen C4 base)

| Cartridge | eval_s |
|-----------|--------|
| WikiText | 28.3 |
| Published | huggingface.co/draw5570/compiled-hybrid-lm |

### Qwen Cartridge Tracks (pe3, 2026-05-25)

| Track | Artifact | Result |
|---|---|---|
| Learned router + gated rack | `artifacts/qwen_cartridge_rack_full_20260525_171513/learned_router/qwen_learned_router.pt` | private_facts 53/60; all other suites 100%; no saved-score regressions |
| Baked native LoRA | `artifacts/qwen_cartridge_rack_full_20260525_171513/baked_lora_native_300/best_adapter` | eval_loss 0.0739; bounded generation eval 34/40; heldout 3/4 |

The learned-router track is the modular cartridge architecture. The baked native-LoRA track is a fused deployment artifact. Both were tested; choose based on whether runtime cartridge control is required.

---

## 15. Troubleshooting

### "CUDA out of memory"

Reduce batch/seq: `--batch 1 --seq-len 32`. Or use `--backend zeroq --compute-in-4bit` for 4-bit mode.

### "bitsandbytes import fails"

Must pin: `bitsandbytes==0.41.3` and `triton==3.3.1`. Maxwell (M40) is the only supported architecture for 4-bit compute.

### "ZeroQ partition_from_full_precision not found"

Update `~/ZeroQ/src/coordinator.py` from the latest `github.com/DRawson5570/ZeroQ`.

### "Module not found"

```bash
pip install -r requirements.txt
# Ensure repo root is on PYTHONPATH or run from repo directory
```

### "Training stalls at data loading"

Verify artifacts exist: `ls artifacts/wikitext_gpt2/train_ids.pt`. If missing, run `build_v3_priors.py` and tokenize WikiText first.

---

*Manual last updated: 2026-05-25*
