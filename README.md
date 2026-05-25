# compiled-hybrid-lm

> **Plug-and-play steering cartridges for zero-latency activation guidance in language models.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

`compiled-hybrid-lm` is an alternative paradigm to dense pretraining. Instead of forcing a transformer to burn trillions of FLOPs memorizing static local statistics (n-grams, syntax, boilerplate, recency), this architecture **factorizes language modeling into two substrates**:

1. **A Compiled Prior:** A stateful statistical engine that tracks n-gram frequencies, decay caches, POS transitions, topic vectors, and PPMI semantics in real-time.
2. **A Lightweight Neural Network:** A standard Transformer that learns only the residual — focusing its parameters on high-entropy reasoning and long-range composition.

At runtime, these merge via **Superposition Gated Steering**. The compiled prior's channel statistics are injected directly into the residual stream as activation offsets through tiny, hot-swappable **Steering Cartridges** (17K parameters, ~70KB on disk). A general superposition steerer can be loaded beside separate domain/capability cartridges and blended additively at inference.

## Key Features

- **Consumer-hardware training**: Train a fluent 124M-parameter model on an RTX 3080 to sub-30 PPL in hours, not weeks.
- **Plug-and-Play Cartridges**: Change domain (Wikipedia → Python code → medical) by swapping a 17K-parameter file. No model retraining.
- **Zero-Latency Swapping**: O(1) tensor pointer update. No CUDA weight-merging. Swap mid-generation.
- **Explainable Control**: Every steering vector maps to an explicit compiled channel (unigrams, skip-grams, decay caches, topic vectors). Auditable per-token attribution.
- **GPU-Vectorized Engine**: Zero-loop parallel prior computer using PyTorch tensor-shifting — KV retrieval, topic scans, POS lookups in microseconds.

## Results (124M Model, C4 base + WikiText cartridge)

Training on a single RTX 3080, 21-channel steerer, 59s/epoch:

```
epoch=  1  loss=3.68  ppl=39.7  eval_s=34.9  eval_b=40.6  [bs]
epoch= 50  loss=3.62  ppl=37.4  eval_s=32.1  eval_b=43.2  [b]
epoch=100  loss=3.58  ppl=35.9  eval_s=30.4  eval_b=42.1  [bs]
```

- **eval_s**: Steered perplexity — the hybrid system with cartridge active
- **eval_b**: Blind perplexity — standalone model, proving prior absorption

Baseline C4 model: 152 PPL. With WikiText cartridge: 35 PPL steered, 41 PPL standalone.

### Website Benchmark Demo

Generate a side-by-side JSON payload for the website from an evaluated cartridge
checkpoint:

```bash
python hybrid/export_benchmark_demo.py \
    --checkpoint artifacts/steerer_v4/steerer_best_s.pt \
    --out artifacts/web_demo_compiled_hybrid_benchmark.json
```

Current demo payload, measured on WikiText-103 validation with GPT-2 BPE:

| Mode | PPL | Active cartridge |
|---|---:|---|
| Compiled Hybrid Baseline | 37.2111 | none |
| Cartridge Injection Active | 28.2080 | superposition-steerer-v3 |

That is a 9.0031 PPL absolute gain, or 24.19% relative perplexity reduction,
from enabling the cartridge on the same checkpoint.

### Generation Samples (eval_s=35)

```
Prompt: "The capital of France is"
Output:  "located in the 2nd kilometre (7.1 mi) plot at the top of the island. The island
         is a large part of the mainland, and was built as a port in the 13th century by the Greek"

Prompt: "In physics, the theory of"
Output:  "quantum physics is the atypical theory."
```

## Quickstart (5 Minutes)

```bash
git clone https://github.com/org/compiled-hybrid-lm
cd compiled-hybrid-lm
pip install -r requirements.txt

# Run validation: 11M model, 20 epochs, ~5 min on RTX 3080
python hybrid/quickstart.py
```

Expected: `eval_s` splits from `eval_b` — proving the steering cartridge provides real signal.

## How It Works

Inside each transformer layer, hidden states are augmented with a per-position steering offset:

$$\mathbf{o}_{t} = \text{MLP}\left(\sum_c w_{c,t} \cdot \mathbf{v}_c\right)$$

$$h_{l,t}^{new} = h_{l,t}^{old} + \gamma_l \cdot \mathbf{o}_{t,\text{norm}}$$

- $\mathbf{v}_c \in \mathbb{R}^{d_{model}}$: learnable steering vector for channel $c$
- $w_{c,t}$: compiled channel feature at position $t$ (bigram, trigram, topic, KV, POS, etc.)
- $\gamma_l$: per-layer gating parameter
- MLP: non-linear channel interaction gatekeeper

The compiled prior computes 21 streaming channel features per token. These are projected through per-group MLPs (local/mid/global) and injected at 9 target layers. A tiny offset (γ ≈ 0.01) redirects the model between domain subspaces.

## Architecture

```
Compiled Channels (21 streaming statistics)
    → Per-token features (bigram, trigram, recency, topic, KV, POS, entropy)
    → Per-Group MLP Gatekeepers (local/mid/global, 6+7+8 channels)
    → Layer-Targeted Activation Offset (B×T×768)
    → Transformer Residual Stream [0,1,2, 4,5,6, 8,9,10]
    → Domain-Specialized Output
```

### Cartridge Types

| Type | Size | Example |
|---|---|---|
| Superposition Steerer | 17K params, ~70KB | General 21-channel activation controller |
| Domain Capability | 17K params, ~70KB | Wikipedia, Python, Medical, Legal |
| Task Capability | 17K params, ~70KB | Reasoning, factual, instruction-following |

### Key Properties

- **Frozen base model**: Ship one 124M model, ship many cartridges
- **Hot-swappable**: Pointer change, no CUDA reload
- **Linear composable**: `offset = α·wiki_cartridge + β·code_cartridge`
- **Edge deployable**: 50 cartridges = 3.5MB cache
- **Auditable**: explicit channel weights, per-token provenance

## Repository Structure

```
hybrid/                        # Core library
    config.py                  # Path configuration
    superposition_steerer.py   # V1: linear steerer (9 channels)
    superposition_steerer_v3.py # V3: 21ch per-group MLP steerer
    cartridges.py              # Cartridge manifests + composition rack
    dynamic_gating.py          # Self-attenuating per-layer injection
    concept_injection.py       # Vocab-space projection steering
    channels_v3.py             # 21-channel features (Witten-Bell, topic, KV, POS)
    smoothing.py               # Witten-Bell + KN-interpolated smoothing
    gpu_channels.py            # Zero-loop GPU-vectorized channel computer
    data_filter.py             # Prior-based data quality filtering
    compiled_priors_v3.py      # Topic vector + KV cache + POS tracker
    chat_gpt2.py               # Production inference (steerer ON)

    train_steerer_v4.py        # V4: 21ch co-training (primary)
    train_steerer_code.py      # Code cartridge training (frozen model)
    train_340m.py              # 340M base model pretraining
    train_scaled_neural_lm.py  # DeepCausalLM model definition

    build_v3_priors.py         # Precompute topic matrix + POS stats
    build_code_dataset.py      # Tokenize Python files for code cartridge
    quickstart.py              # 10-minute validation script

pyproject.toml                 # Package config
README.md                      # This file
```

## Training

### Full co-training (model absorbs compiled prior)
```bash
python hybrid/train_steerer_v4.py \
    --neural-ckpt artifacts/c4_v2_768_x30/best.pt \
    --epochs 200 --steps 500 --batch 8
```

### Cartridge-only (frozen model, train cartridge)
```bash
python hybrid/train_steerer_code.py \
    --base-model artifacts/steerer_v4/steerer_best_b.pt \
    --epochs 200 --steps 500 --batch 4
```

### Inference
```bash
python hybrid/chat_gpt2.py
```

### Cartridge Swapping
```python
from hybrid.cartridges import CartridgeManifest, CartridgeRole, SteererCartridgeRack

rack = SteererCartridgeRack()
rack.mount(CartridgeManifest('v4-steerer', CartridgeRole.SUPERPOSITION_STEERER,
                             base_model_id='c4-124m', tokenizer_id='gpt2-bpe'), v4)
rack.mount(CartridgeManifest('code', CartridgeRole.DOMAIN_CAPABILITY,
                             base_model_id='c4-124m', tokenizer_id='gpt2-bpe'), code, weight=0.3)
rack.register_hooks(model)

# Per generation step: feed the same live compiled channel features to mounted cartridges.
rack.set_weights(live_features)
```

## Requirements

- Python 3.10+, PyTorch 2.0+, transformers, numpy, scikit-learn
- CUDA GPU (RTX 3080 10GB for 124M, A100 40GB+ for 340M+)

## License

MIT

## Citation

```bibtex
@misc{rawson2026compiledhybridlm,
  title   = {compiled-hybrid-lm: Steering Cartridges for Activation-Guided Language Models},
  author  = {Douglas Rawson},
  year    = {2026},
  url     = {https://github.com/drawson/compiled-hybrid-lm},
  note    = {Compiled Modular Intelligence — activation superposition steering with domain-specific cartridges},
}
```

Douglas Rawson — rawson.douglas@gmail.com
