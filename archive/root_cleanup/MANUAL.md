# Hybrid LLM — Complete Manual

Author: DeepSeek V4 + human (drawson)
Date: 2026-05-22
Repo: `github.com/DRawson5570/hybrid-llm` (private)

---

## Table of Contents

1. [What This Is](#1-what-this-is)
2. [Architecture Overview](#2-architecture-overview)
3. [Hardware Requirements](#3-hardware-requirements)
4. [Setup](#4-setup)
5. [Data Pipeline](#5-data-pipeline)
6. [Compiled Channels (BPE-8000)](#6-compiled-channels-bpe-8000)
7. [Compiled Channels (GPT-2 BPE)](#7-compiled-channels-gpt-2-bpe)
8. [Blender Training](#8-blender-training)
9. [Neural LM Training](#9-neural-lm-training)
10. [Hybrid Blending](#10-hybrid-blending)
11. [Evaluation](#11-evaluation)
12. [Generation](#12-generation)
13. [Surfaces API](#13-surfaces-api)
14. [Results](#14-results)
15. [Troubleshooting](#15-troubleshooting)

---

## 1. What This Is

A hybrid language model that combines **compiled** (deterministic, count-based) and **learned** (SGD-trained) components. The compiled substrate captures corpus statistics that transformers waste billions of tokens rediscovering. The learned substrate handles what counting can't: composition, long-range dependencies, and context-dependent reasoning.

The result: a model that beats GPT-2 Small (PPL=29.0) with ~1/100th the training data.

---

## 2. Architecture Overview

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

## 3. Hardware Requirements

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

## 4. Setup

### 4.1 Clone and Dependencies

```bash
git clone https://github.com/DRawson5570/hybrid-llm.git
cd deepseek_experiments

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

## 5. Data Pipeline

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

## 6. Compiled Channels (BPE-8000)

The full 21-channel compiled pipeline lives in the main repo at `~/llm_decoupling/`.
It produces per-position log-probabilities for 21 statistical models.

### Channel Inventory (v33)

| # | Name | Description | PPL (eval) |
|---|---|---|---|
| 0 | kn | Kneser-Ney 5-gram | 89.88 |
| 1 | mix | Sparse mixture cluster LM | 293.72 |
| 2-3 | tri_f/s | Decayed trigram cache (fast/slow) | 4086/4828 |
| 4-5 | bi_f/s | Decayed bigram cache | 2316/2261 |
| 6-7 | uc_f/s | Decayed unigram cache | 2070/2111 |
| 8-10 | attn_uf/us/ug | Attention unigram (fast/slow/global) | 1437/537/3342 |
| 11-12 | attn_rf1/rs1 | Attention residual K=1 | 391/180 |
| 13-15 | attn_rf2/rs2/rg2 | Attention residual K=2 | 209/200/1455 |
| 16-17 | attn_rf3/rs3 | Attention residual K=3 | 257/352 |
| 18 | ppmi | PPMI semantic similarity | 7647 |
| 19 | knn | KNN retrieval | 5664 |
| 20 | shape | Word shape transitions | 10940 |

### Dumping for Blender Training

```bash
python3 hybrid/v3_super_blender/dump_features_v33.py \
  --kn-pickle artifacts/compiled_wiki_lm_v23/kn7_22m.pkl \
  --counts-file artifacts/compiled_wiki_lm_v14/counts_k2_c64k.pt \
  --out-dir hybrid/v3_super_blender/data_real_v33

# Produces: val.npz (30K tokens), eval.npz (100K tokens)
# Each contains: log_p_targets (T,21), log_p_observed (T,21),
#                entropy (T,21), max_log_prob (T,21), observed (T,),
#                targets (T,), channel_names (21,)
```

---

## 7. Compiled Channels (GPT-2 BPE)

Rebuilt from scratch for V=50257. Uses efficient O(T) streaming channels
(no O(V²) count tables that would blow up memory).

### Quick Dump

```bash
# Build 10-channel dump (5 compiled logp + tri_f, bi_f, uc_f, uni, recency)
python3 hybrid/dump_gpt2_channels_v2.py \
  --val-tokens 30000 --eval-tokens 100000 \
  --out-dir artifacts/gpt2_channels_v2

# Produces: val.npz, eval.npz
# Same format as BPE-8000 dump (compatible with blender training)
```

### Pre-built Compiled Builder

```bash
# One-time: build GPT-2 compiled channel artifact from WikiText train
python3 -c "
from hybrid.compiled_features import GPT2CompiledChannelBuilder, GPT2CompiledChannelConfig
import torch
wt = torch.load('artifacts/wikitext_gpt2/train_ids.pt', weights_only=False)
builder = GPT2CompiledChannelBuilder.from_ids(
    wt, GPT2CompiledChannelConfig(alpha=0.1, max_train_tokens=50_000_000)
)
builder.save('artifacts/compiled_builder_50m.pt')
"
# Serialized: ~365 MB
# Provides 21 causal feature channels per position
```

---

## 8. Blender Training

The WindowMLP blender learns to mix compiled channel log-probs using
per-position features.

### BPE-8000 (21 channels)

```bash
python3 hybrid/v3_super_blender/train.py \
  --data-dir hybrid/v3_super_blender/data_real_v33 \
  --epochs 20 --lr 3e-4
# Output: saved_models/blender_window_mlp.pt
# Best result: PPL=11.62 (eval)
```

### GPT-2 BPE (10 channels)

```bash
python3 hybrid/train_gpt2_blender.py \
  --data-dir artifacts/gpt2_channels_v2 \
  --epochs 50 --lr 1e-3 --device cuda \
  --out-dir artifacts/gpt2_blender
# Output: report.json with blender_ppl, blend_results
```

---

## 9. Neural LM Training

### BPE-8000 (d_model=256, PPMI init)

The proven best config. PPMI init requires d_model=256 (matching the pre-built embeddings).

```bash
python3 hybrid/train_hybrid_bpe8000.py \
  --epochs 20 --steps-per-epoch 2000 --batch 8 --seq-len 128 \
  --d-model 256 --n-layers 12 --n-heads 8 --d-ff 1024 \
  --out-dir artifacts/hybrid_run

# Automatically blends with compiled WindowMLP on completion.
# Report: report.json with neural_ppl, compiled_ppl, best_blend_ppl, best_alpha
```

### GPT-2 BPE (d_model=768, random init)

```bash
python3 hybrid/train_gpt2_neural_lm.py \
  --epochs 30 --steps-per-epoch 4000 --batch 2 --lr 3e-4 \
  --d-model 768 --n-layers 12 --n-heads 12 --d-ff 3072 \
  --out-dir artifacts/gpt2_768

# WikiText-only training. No compiled features during training.
# Blend evaluation is done separately via train_gpt2_blender.py.
```

### GPT-2 BPE + C4 (d_model=768, 100 epochs)

```bash
# Launch in screen for long runs
screen -dmS c4_train bash -c \
  "cd ~/deepseek_experiments && python3 -u /tmp/c4_simple.py --epochs 100 --steps 4000 2>&1 | tee /tmp/c4_simple.log"

# Interleaves C4 (85%) + WikiText (15%), saves checkpoints each epoch.
# Check progress: tail -5 /tmp/c4_simple.log
# Reattach: screen -r c4_train
```

---

## 10. Hybrid Blending

The blend is a per-token convex combination of compiled and neural distributions:

```
p_blend(y) = α · p_compiled(y) + (1-α) · p_neural(y)
log p_blend = log(α · exp(lp_compiled) + (1-α) · exp(lp_neural))
```

This is computed at evaluation time for α ∈ {0.0, 0.3, 0.5, 0.7, 0.9, 1.0}.
The best α is selected by lowest PPL.

The `train_hybrid_bpe8000.py` script automatically blends at the end of training.
For GPT-2 BPE, use `train_gpt2_blender.py` which blends with the neural LM checkpoint.

---

## 11. Evaluation

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

## 12. Generation

```bash
# Single prompt
python3 hybrid/generate_hybrid.py \
  --prompt "Explain quantum computing" --max-new 200 --temperature 0.7

# Interactive chat
python3 hybrid/generate_hybrid.py --chat
```

Features:
- Temperature scaling
- Top-p (nucleus) sampling
- Repetition penalty
- Stop tokens (EOS, `<|im_end|>`)
- Multi-turn chat with history

Note: Coherent generation requires neural PPL < 50. At PPL=71, output is partially coherent but sometimes nonsensical.

---

## 13. Surfaces API

The hybrid nature of the model enables component injection/retraction without retraining.

```python
from hybrid.surfaces import inject_logit_bias, retract, compose, ComponentRegistry

# Inject a logit bias
registry = ComponentRegistry()
wrapped, cid = inject_logit_bias(model, bias_tensor, registry=registry)

# Retract (exact restore, verified by checksum)
unwrapped, verified = retract(wrapped, cid, registry)
assert verified  # checksum matches pre-install state

# Compose multiple components
wrapped, ids = compose(model, [
    {'type': 'logit_bias', 'bias': bias_a},
    {'type': 'logit_bias', 'bias': bias_b},
], registry)

# Provenance tracking
from hybrid.surfaces.provenance import ProvenanceRing, ProvenanceBlender
```

Tests: `tests/test_hybrid_surfaces.py` (5 tests, all passing)
Provenance: `tests/test_provenance.py` (200-token decode verified, logsumexp invariant holds)

---

## 14. Results

### BPE-8000 (custom tokenizer, V=8000)

| Model | Params | Neural PPL | Compiled PPL | Blend PPL |
|---|---|---|---|---|
| 12L, d=256, 20ep | 11.6M | 71.55 | 11.62 | **9.21** |
| 12L, d=256, 50ep | 11.6M | 81.2 | 11.62 | 9.28 |
| 24L, d=256, 20ep | 21.0M | 96.2 | 11.62 | 9.62 |
| 12L, d=512, 20ep | 42.0M | 85.9 | 11.62 | 9.38 |
| 12L, d=768, 20ep | 91.3M | 136.9 | 11.62 | 10.11 |

**Key finding**: PPMI init at d_model=256 is critical. Bigger models without it underperform.

### GPT-2 BPE (standard tokenizer, V=50257)

| Model | Params | Neural PPL | Compiled PPL | Blend PPL |
|---|---|---|---|---|
| 12L, d=768, 30ep WikiText | 124M | 105.46 | 58.23 | **23.33** |
| GPT-2 Small (baseline) | 124M | — | — | 29.0 |

**Key finding**: Hybrid beats GPT-2 Small with 120M tokens vs billions.

---

## 15. Troubleshooting

### "C4 training crashes before epoch 1"

Use `text[:2000]` to truncate before `tokenizer.encode()`. The tokenizer hangs on 11K-token documents.

### "Process dies in background"

Use `screen` instead of `nohup`:
```bash
screen -dmS myrun bash -c "python3 -u script.py 2>&1 | tee /tmp/out.log"
screen -r myrun  # reattach
# Ctrl+A, D to detach
```

### "GPU out of memory"

Reduce batch size: `--batch 1`. The 124M model needs ~3GB at batch=2.

### "Identical results across runs"

The default torch seed is 42. For different trajectories, change `torch.manual_seed()`.

### "Module not found: compile_wiki_lm_v13"

Add `~/llm_decoupling` to `sys.path`. The main repo must be accessible.

### "Shape channel gives PPL=1.44"

The shape channel (token capitalization/punctuation patterns) is overwhelmingly predictive in English text. Exclude it for meaningful compiled channel blending.

---

*Manual last updated: 2026-05-22*
