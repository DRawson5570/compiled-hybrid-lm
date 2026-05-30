# CMI Infrastructure

## Hardware Topology

| Host | GPUs | VRAM/GPU | Role |
|---|---|---|---|
| 3080 (local) | 1× RTX 3080 | 10GB | V4 steerer training (primary) |
| pe2 | 5× Tesla M40 | 24GB each | 340M base model training (GPU 0) |
| pe3 | 2× Tesla M40 | 12GB each | V1 steerer (completed) |

**Remote access:** `ssh pe2`, `ssh pe3`. Python venv at `~/local_venvs/m40_env/`. GPUs managed via `nvidia-smi`.

M40s lack tensor cores (Maxwell architecture, FP32 compute only). 3080 has Ampere tensor cores — ~2.5x faster per epoch for same workload.

## Software Stack

### Core Components

```
┌─────────────────────────────────────────────────┐
│                 Training Loop                     │
│  train_steerer_v4.py (V4, 21ch + GPU + O(1))     │
│  train_steerer_code.py (code cartridge)          │
│  train_340m.py (base model pretraining)          │
├─────────────────────────────────────────────────┤
│              Steerer Architecture                 │
│  superposition_steerer.py (V1 linear, 9ch)       │
│  superposition_steerer_v3.py (V3, 21ch MLP)      │
│  cartridges.py (manifest + multi-cartridge rack) │
│  backends.py (dense + optional ZeroQ substrates)  │
│  dynamic_gating.py (per-layer self-attenuating)  │
│  concept_injection.py (vocab-space projection)    │
├─────────────────────────────────────────────────┤
│           Compiled Channel Features               │
│  channels_v3.py (FullV3ChannelFeatures, 21ch)    │
│  gpu_channels.py (GPUFeatureComputer, parallel)   │
│  smoothing.py (Witten-Bell, KN-interpolated)     │
├─────────────────────────────────────────────────┤
│              Model Definitions                    │
│  train_scaled_neural_lm.py (DeepCausalLM)        │
├─────────────────────────────────────────────────┤
│              Compiled Priors                      │
│  build_v3_priors.py (topic matrix, POS stats)    │
│  compiled_priors_v3.py (TopicVector, KV cache)   │
│  build_code_dataset.py (Python tokenization)     │
│  config.py (shared path resolution)              │
│  data_filter.py (prior-based quality gate)       │
├─────────────────────────────────────────────────┤
│                 Inference                         │
│  chat_gpt2.py (production chat, steerer ON)      │
│  quickstart.py (10-min validation)               │
│  cartridge_harness/ (owned cartridge research)    │
```

### Data Pipeline

```
WikiText-103 / C4 → GPT-2 BPE Tokenizer → train_ids.pt (119M tokens)
                                   → validation_ids.pt (251K tokens)
                        artifacts/wikitext_gpt2/

Python stdlib (70K .py files) → GPT-2 BPE Tokenizer → code_steerer/train_ids.pt
                                              artifacts/code_steerer/
```

### Checkpoint Architecture

All checkpoints saved to `artifacts/<experiment>/`:

| File | Contents | Use |
|---|---|---|
| `steerer_best_b.pt` | Model state + steerer state at best eval_b | Standalone model |
| `steerer_best_s.pt` | Model state + steerer state at best eval_s | Hybrid system |
| `best.pt` | Model state at best eval (base training) | Pretrained base |
| `code_cartridge.pt` | Steerer state only (frozen model) | Code domain |
| `*.cartridge.pt` | Steerer state + `CartridgeManifest` metadata | Hot-swappable cartridge |

Checkpoint format:
```python
{
    'state_dict': model.state_dict(),       # neural LM weights
    'steerer_state': steerer.state_dict(),  # cartridge weights
    'opt_state': opt.state_dict(),          # optimizer for resume
    'eval_s': float, 'eval_b': float,       # dual metrics
    'epoch': int, 'gamma': float,           # training state
}
```

## Steering Architecture

### Cartridge Rack and ABI

`hybrid.cartridges` defines the runtime ABI for independent cartridge loading:

- `CartridgeManifest`: compatibility metadata for base model, tokenizer, channel schema, injection layers, and additive composition space.
- `CartridgeRole`: distinguishes `superposition_steerer`, `domain_capability`, `task_capability`, and `concept_injection` packages.
- `SteererCartridgeRack`: mounts multiple compatible steerers, registers one hook set on the model, and sums weighted residual deltas.

This keeps the general superposition steerer hot-swappable beside domain/capability cartridges. A session can load a V4 21-channel steerer, then layer a WikiText cartridge and a Python cartridge with separate weights without mutating the frozen base model.

### Backend Substrate

`hybrid.backends` keeps execution mechanics below the cartridge ABI:

- `DenseTorchBackend` freezes the requested backbone, materializes it on one device, and exposes only a named trainable surface.
- `ZeroQPartitionedBackend` loads `~/ZeroQ` on demand, streams frozen parameters through 4-bit partitioning, and materializes only the requested trainable surface. Its default compatibility mode uses ZeroQ gather/release hooks; `compute_in_4bit=True` converts frozen `nn.Linear` modules to native bitsandbytes `Linear4bit` after partitioning so large M40 runs avoid per-layer gather/release traffic.
- `TrainableSurface` makes the trainable boundary explicit; current large-backbone runs use `head_bias` plus a separate compiled-prior steerer cartridge, with adapters/LoRA-style surfaces as the next extension.

Because cartridges attach through model layer hooks, the same `SteererCartridgeRack` contract works whether the substrate is dense PyTorch or ZeroQ-partitioned.

### V3 (SuperpositionSteererV3 — 21 channels)

Current production architecture. 21 channels (6 local + 7 mid + 8 global), per-group MLP gatekeepers, layer-targeted injection. Witten-Bell smoothing on n-gram channels. GPU-vectorized topic vector and KV semantic cache. O(1) FastNgramFeatures for CPU-side n-grams. DataLoader with 4 background workers for parallel pre-fetch.

16,796 params. Layer mapping: local→[0,1,2], mid→[4,5,6], global→[8,9,10].

### V1 (SuperpositionSteerer)

9 channels, linear projection. Inject at layers [0, 4, 8]. 2,306 params. Archived — kept for backward compatibility and quickstart.

### V2 (MLPSuperpositionSteerer)

14 channels, per-group MLPs, layer-targeted injection. 76,578 params. Archived — superseded by V3.

## Compiled Prior Engine

### Channel Inventory

| Index | Name | Type | Description |
|---|---|---|---|
| 0 | unigram | Global stat | Decayed unigram log-prob |
| 1 | bigram_fast | Local stat | P(token | prev token) |
| 2 | bigram_slow | Local stat | Decayed bigram over longer window |
| 3 | trigram_fast | Local stat | P(token | prev 2 tokens) |
| 4 | trigram_slow | Local stat | Decayed trigram |
| 5 | skip-2 | Gapped stat | P(token | token 2 steps back) |
| 6 | skip-3 | Gapped stat | P(token | token 3 steps back) |
| 7 | recency | Temporal | Gap since last occurrence |
| 8 | builder_entropy | Global stat | Unigram entropy (normalized) |
| 9 | shape | Surface | Word shape (upper/lower/digit) |
| 10-14 | PPMI stubs | Semantic | PPMI cosine, max, norm (precomputed) |
| 15 | punct_density | Register | Punctuation/word ratio |
| 16 | repetition_score | Register | Adjacent token repeats |
| 17 | unique_token_ratio | Register | Vocabulary diversity |
| 18 | topic_logp | Semantic | Probability under running topic vector |
| 19 | kv_max_sim | Retrieval | Max cosine sim in KV cache |
| 20 | pos_transition | Syntax | POS bigram transition probability |

### Smoothing Pipeline

Witten-Bell replaces Laplace for unigram, bigram, trigram:
- `P(unseen) = U/(N+U)` where U = unique items
- `P(seen) = count/(N+U)`

Dynamically adapts: repetitive contexts get near-zero unseen mass; diverse contexts get smooth backoff.

### Precomputation Pipeline

`build_v3_priors.py` runs offline:
1. Build sparse PPMI stats from first 500K tokens
2. Cluster token co-occurrence signatures → K=50 topics (MiniBatchKMeans)
3. NLTK POS-tag first 2M tokens → compile transition matrices
4. Save `word_topics.pt` + `pos_stats.pkl` to `artifacts/compiled_priors_v3/`

## Training Pipeline

### Owned Cartridge Harness

`hybrid/cartridge_harness/` is the in-repo self-improvement harness for cartridge research/building. It replaces ad hoc scripts under the external Life-Harness checkout for our core loop:

1. Build a task suite (`TaskExample`) with train and held-out splits.
2. Evaluate the frozen baseline with strict row-level scoring.
3. Train only a mounted cartridge through `SteererCartridgeRack`.
4. Re-evaluate, compare fail-to-pass improvements and regressions, and write `summary.json` plus the cartridge artifact under `artifacts/`.

Current CLI:

```bash
python -m hybrid.cartridge_harness.cli private-facts \
    --model Qwen/Qwen2.5-1.5B \
    --device cuda \
    --out-dir artifacts/owned_private_fact_cartridge
```

The harness may use external benchmarks as task sources, but owned cartridge artifacts, manifests, summaries, and regressions are produced by this package.

### Modes

| Mode | Model | Steerer | Use Case |
|---|---|---|---|
| Base pretraining | Trainable | None | Building foundation model (340M) |
| Co-training | Trainable | Trainable | Model absorbs compiled prior (V1/V2) |
| Cartridge-only | Frozen | Trainable | Domain specialization (code) |

### Co-training (Production Path)

1. Load C4-pretrained base model + fresh steerer
2. Train jointly: model lr=3e-5, steerer lr=1e-2
3. Track eval_b (standalone) and eval_s (steered)
4. Save split checkpoints on best_b and best_s
5. Model learns to produce steerer offsets internally (compiled prior absorption)

### Cartridge-Only (Cartridge Production)

1. Load frozen model from best_b checkpoint
2. Create fresh steerer (76K params)
3. Train only steerer on domain-specific data (Python code, medical text, etc.)
4. Save cartridge as `code_cartridge.pt` (76K params, ~300KB)
5. Ship base model + cartridges separately

### Training Arguments

| Arg | V4 | 340M | Code |
|---|---|---|---|
| epochs | 200 | 400 | 200 |
| steps/epoch | 500 | 2000 | 500 |
| batch | 8 | 2 | 4 |
| seq_len | 128 | 128 | 128 |
| model lr | 3e-5 | 3e-4 | frozen |
| steerer lr | 1e-3 | — | 1e-2 |

## Inference Pipeline

### Production Chat (chat_gpt2.py)

```
User prompt
    → GPT-2 BPE tokenize
    → LiveChannelFeatures.update() on each token
    → For each generation step:
        → compute per-position channel features (context only)
        → steerer.set_weights() (raw log-probs, temperature softmax)
        → model.forward() (layers [0,4,8] receive steering offset)
        → sample next token
        → channels.update(next_token)
    → GPT-2 BPE decode → output
```

Loads `steerer_best_s.pt` — best steered checkpoint with joint model+steerer weights. Steerer active at inference (production mode).

### Cartridge Swapping

```python
from hybrid.cartridges import CartridgeManifest, CartridgeRole, SteererCartridgeRack

rack = SteererCartridgeRack()
rack.mount(
    CartridgeManifest('v4-steerer', CartridgeRole.SUPERPOSITION_STEERER,
                      base_model_id='c4-124m', tokenizer_id='gpt2-bpe'),
    v4_steerer,
    weight=1.0,
)
rack.mount(
    CartridgeManifest('python-code', CartridgeRole.DOMAIN_CAPABILITY,
                      base_model_id='c4-124m', tokenizer_id='gpt2-bpe'),
    code_steerer,
    weight=0.4,
)
rack.register_hooks(model)

# Per generation step:
rack.set_weights(live_compiled_channel_features)
```

Zero latency — change rack weights, active flags, or mounted steerer state. No base-model weight merging, no CUDA reload.

## Communication & Monitoring

### Telegram Bot

`tg.py`: Send/read Telegram messages for experiment monitoring.
- `python3 ~/tg.py send "message"` — send update
- `python3 ~/tg.py read` — check for new messages
- Bot token + chat ID stored in `~/.clawdbot/credentials/`

### Screen Sessions

| Session | Host | Experiment |
|---|---|---|
| steerer_v4 | 3080 local | V4 steerer training (primary) |
| train_340m | pe2 | 340M base training |

### Log Files

| Log | Host | Content |
|---|---|---|
| /tmp/steerer_v2_3080.log | local | V2 training epochs |
| /tmp/steerer_gpt2_3080.log | local | V1 training (completed) |
| /tmp/train_340m.log | pe2 | 340M training |
| /tmp/steerer_v1_pe3.log | pe3 | V1 training |
| /tmp/build_code.log | local | Code dataset tokenization |
| /tmp/build_v3_priors.log | local | V3 prior precomputation |

## Experiment History

### What Works

1. **Natural co-training**: Model absorbs compiled prior through gradient descent. 152→41 PPL standalone on GPT-2 BPE 124M.
2. **21-channel V3 steerer**: Per-group MLPs with proper channel slicing. 16,796 params, 9 hooks.
3. **GPU-vectorized features**: Topic vector and KV cache computed in parallel on GPU.
4. **O(1) CPU n-grams**: Running scalar sum replaces O(V) array scans. DataLoader with 4 workers for parallel pre-fetch.
5. **Split saves**: Separate best_b.pt (standalone) and best_s.pt (hybrid) checkpoints.
6. **Cartridge paradigm**: Frozen base model + trainable cartridge = domain-specific LM.
7. **Code cartridge dataset**: 625M tokens of Python code, 5.3GB, ready for training.

### What Doesn't Work

1. **Gamma annealing**: Decaying gamma starves the model. Fixed gamma via optimizer-trainable per-layer gammas.
2. **BPE-8000**: 11.6M params too small to absorb compiled prior (6% gain vs 60%+ for GPT-2 BPE).
3. **Output blending**: Proven (20.22 PPL) but coarser than activation superposition. Archived as fallback.
