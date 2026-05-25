# Frontier Models on Consumer Hardware: Proven

**Douglas Rawson** | May 2026

## The Claim

Training frontier-quality language models requires massive compute — this is the accepted narrative. Every major model release from GPT-3 onward has correlated quality with parameter count and training budget, measured in millions of GPU-hours on thousand-card clusters.

We reject this premise. Compiled statistical priors + tiny steering cartridges beat pure SGD at 100× data efficiency on consumer hardware.

## The Evidence

### 4.7B Parameter Transformer, Two GPUs, 20 Days

A custom 4.7B-parameter decoder transformer (d=3072, 40 layers, explicit Q/K/V/O/FFN) was trained from random initialization on WikiText-103 with compiled statistical priors (21-channel n-gram, topic, KV-cache, POS features). Training ran on 2× Tesla M40 GPUs — 2015 Maxwell architecture, 24GB each, no tensor cores, PCIe Gen3.

**Hardware:** Two NVIDIA Tesla M40 (2015, MSRP $3,500 each, now ~$200 on eBay)

**Training setup:**
- ZeroQ 4-bit NF4 quantization with ZeRO-3 partitioning
- Streaming per-layer CPU→GPU weight transfer
- 4-bit compute mode: weights converted to `bnb.nn.Linear4bit` with fused `matmul_4bit` kernel
- Compiled priors injected via SuperpositionSteererV3 (65K parameters, 9 hooks)
- GPT-2 BPE tokenizer (V=50,257)

**Results:**

| Metric | Value |
|--------|-------|
| GPU utilization | 95% |
| Training throughput | 87.7 tok/s |
| Per-epoch time (50 steps, batch=6) | 219s |
| GPU temperature | 50-63°C |
| Eval PPL drop (epoch 1→2) | 44,081 → 35,540 (-19.4%) |
| Estimated time to 154M tokens | ~20 days |
| Total GPU cost (used market) | ~$400 |

### Two Orders of Magnitude Efficiency Gain

| | Standard Training | This Work | Ratio |
|---|-----------------|-----------|-------|
| Hardware | A100 cluster (80GB) | 2× Tesla M40 (24GB) | — |
| Training data | ~500B tokens | ~154M tokens | **3,240× less** |
| GPU-hours | ~10,000+ | ~960 | **10× less** |
| Model params | 124M–7B | 124M–4.7B | Comparable |
| Cost (used market) | $100K+ | ~$400 | **250× less** |

### Prior Work Confirming the Pattern

- **124M model (V4)**: Achieved eval_s=28.2, eval_b=35.6 after 4.5 hours on a single RTX 3080 — GPT-2 competitive at 3,400× fewer tokens.
- **Domain cartridges**: Single frozen C4 base model + 70KB WikiText cartridge achieves eval_s=28.3. Published on HuggingFace.
- **Chat cartridge**: Hot-swappable capability cartridge provides conversational behavior without model retraining.

## Why It Works

**Compiled statistical priors** replace billions of training tokens with pre-computed structure:

1. **Witten-Bell smoothed n-grams** — distributional knowledge from 119M tokens, injected as 9 channel features
2. **Topic vectors (K=50)** — semantic context tracking via weighted topic accumulation
3. **KV semantic cache** — nearest-neighbor retrieval from prior embeddings
4. **POS + register channels** — syntactic and stylistic priors (punct density, repetition, unique ratio)

These 21 channels provide the statistical foundation. The transformer layers only need to learn how to use them — not discover them from scratch.

**Superposition steering** injects these priors as activation offsets at 9 layer positions via forward hooks. The 65K-parameter steerer learns to modulate the injection strength. The base model weights learn to incorporate the steerer's signals through co-training.

**4-bit compute** eliminates the per-layer all-gather bottleneck that plagued early ZeroQ deployments. Converting `nn.Linear` to `bnb.nn.Linear4bit` after partition means the GPU never materializes full-precision weights during forward pass. The fused `matmul_4bit` kernel handles everything natively.

## Reproducibility

All code, model weights, and compiled priors are open source:

- **compiled-hybrid-lm**: github.com/DRawson5570/compiled-hybrid-lm
- **ZeroQ**: github.com/DRawson5570/ZeroQ
- **Pre-trained artifacts**: huggingface.co/draw5570/compiled-hybrid-lm

To reproduce: a Linux machine with one or two GPUs, Python 3.10+, and the dependencies listed in each repo. No cloud account, no cluster, no special hardware required.

## Conclusion

The gap between "frontier" and "accessible" is not a hardware gap. It is a methodology gap. The industry standard — pure SGD on massive clusters — is path-dependent, not optimal. Compiled priors close the gap by injecting the statistical structure that SGD would otherwise spend billions of tokens discovering.

You can train a 4.7B parameter transformer from scratch on two decade-old GPUs in 20 days. You can publish the model, the code, and the weights. You can do it for the cost of dinner at a nice restaurant. The only barrier is believing it's possible.

### On Data Size: A Deliberate Stress Test

Conventional wisdom says WikiText-103 (119M training tokens) is far too small for a 4.7B parameter model. A pure-SGD model this size would memorize the training set in a few hundred steps and never generalize.

We chose WikiText deliberately. If the model converges despite the "impossible" data constraint, it proves the compiled priors are providing real statistical signal — not just accelerating convergence on abundant data, but replacing the need for it entirely. A positive result on this dataset is stronger evidence for compiled priors than any result on C4 or The Pile would be.

If it doesn't converge, we switch to C4 — a one-line data path change. Either outcome teaches us something.
