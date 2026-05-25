# Hybrid Compile + SGD Strategy

**Status:** ✅ **Validated 2026-05-20 — Hybrid v1 lands HELDOUT PPL=20.22, crossing the GPT-2 boundary on the first experiment.** See EXPERIMENT_LOG **#310**.

## North Star

> **Build frontier-quality LLMs on consumer hardware with much smaller datasets.**

The fully-compiled, SGD-free track (current PPL push toward < 29.0 on WT103) is the
**research breakthrough**. The compile-then-SGD hybrid is the **product**.

If the pure-compiled track plateaus before crossing the GPT-2 boundary, we do NOT
treat that as failure. We take the compiled model as far as it will go, then close
the remaining gap with a small amount of SGD on top. The compiled prior gives SGD
an enormously warm start, which is exactly what unlocks consumer-hardware training
and small-dataset training.

## Why this is the right escape hatch

- Compiled methods cheaply encode statistics SGD would otherwise burn millions of
  tokens learning: n-gram backoff, semantic neighborhoods, recency, cluster
  structure, attention retrieval.
- **SGD is incredibly inefficient at discovering rare useful structure.** Most
  pretraining FLOPs in current frontier models are spent rediscovering corpus
  statistics that are analytically extractable. A compiler that pre-builds the
  prior turns those FLOPs into refinement FLOPs, not discovery FLOPs.
- SGD's comparative advantage is long-range structure, compositionality, and
  contextual interactions between channels — exactly where linear / Dirichlet
  blending hits a wall.
- A 7B model from scratch costs ~$100K of compute. The same model fine-tuned on
  top of a strong compiled prior should need 1–2 orders of magnitude less data
  and compute, because the loss landscape starts near-optimal for the easy stuff.

## A different scaling law

Current transformers scale by brute-force optimization against bandwidth,
memory, training-cost, and data-quality walls. This project trades brute force
for **explicit information organization**. That is a fundamentally different
curve, not the same curve climbed faster:

```
Standard:    Random Weights → 10^25 FLOPs → Hopefully Intelligence
This path:   Corpus Statistics
              → Compiled Semantic Graph
              → Multi-scale Retrieval + Routing
              → Small Differentiable Refinement Network
              → Frontier-like Behavior
```

If retrieval + compilation scale more favorably than dense parameter
memorization (likely true once you account for bandwidth and energy), the
hybrid path wins on the metrics that matter for consumer hardware: total FLOPs,
required dataset size, convergence time, parameter count, energy cost.

## The compiled artifact has five product surfaces

Even if the pure-compiled track stops short of our PPL goal, the compiled
artifact is sellable on its own as any of:

1. **Initialization prior** for transformer pretraining (warm-start that
   collapses the easy part of the loss landscape before SGD starts).
2. **Latent-space generator** — semantic neighborhoods + cluster structure
   exposed as a service to other models.
3. **Routing oracle** — per-token expert / channel selection signals for MoE
   training and inference.
4. **Synthetic pretrainer** — emit teacher distributions for distillation into
   a small differentiable network, replacing raw next-token CE.
5. **Differentiable dataset compressor** — the compiled artifact *is* a
   lossy compression of the training corpus into structural priors; sellable to
   any team that needs to train on small data.

This matters because it means the project's commercial value is not gated on
crossing GPT-2 PPL with zero SGD. Even at 38 PPL compiled-only + lightweight
SGD refinement getting us to 20–25, we have a real, defensible product story.

## Hybrid architectures worth trying (in priority order)

1. **Compiled channels as input features (probably the winner).**
   Feed the per-token outputs of every compiled channel (KN log-prob,
   attention-cache log-prob, cluster mixture, residual-attention cache, etc.) as
   auxiliary input features to a small transformer alongside its embeddings. The
   transformer learns a **contextual** blender — e.g. "trust the attention cache
   when context is repetitive, trust KN when it's novel." Replaces our Dirichlet
   random search with a learned, context-aware gating function.

2. **Compiled-as-residual-prior.**
   Freeze the compiled distribution `p_c(x | ctx)`. Train a small Δ-head whose
   logits are *added* to `log p_c`. SGD only learns the residual. Tiny parameter
   count, fast convergence, surgical.

3. **Compiled-as-teacher distillation.**
   Use the compiled blend as a teacher distribution for KL distillation into a
   small transformer. Every position has a calibrated full distribution rather
   than a one-hot target — much richer signal than standard next-token CE.

4. **Compiled init for embedding + unembedding matrices.**
   PPMI+SVD embeddings already encode strong semantic structure. Use them as
   init; SGD only refines. Pythia/GPT-2 spend a lot of capacity rediscovering
   this.

## Implementation notes

- Some compiled channels (attention caches, residual-attention caches) are
  **inference-time computations**, not parameters. They remain as runtime
  features or get amortized into the transformer's attention weights via
  distillation. Default plan: keep them as runtime — they're cheap and they
  generalize.
- The marketing claim shifts from "no SGD" purity to **"frontier capability at
  consumer-hardware scale via compile-then-finetune."** This is a more defensible
  and more useful claim. The compiled side of the system is still the novel
  contribution, just packaged with a small SGD finishing pass.

## When to invoke this fallback

- Pure-compiled WT103 heldout PPL stalls above ~33 with no obvious orthogonal
  channel left to add, OR
- A target downstream task (instruction following, code, math) requires
  compositional behavior that no count-based or attention-cache method can
  express, OR
- A user / product milestone needs a deliverable now and we've already taken
  pure-compiled as far as it's going to go in the current research window.

When invoked, start with hybrid architecture **#1** (compiled channels as input
features). It preserves everything we've built and is the most additive.

## v2: Activation Superposition Steering (Production Architecture)

*Last updated by Qwen3.6 at 2026-05-24 08:30 local*

The current and primary integration point is **activation superposition**. Compiled channel features are injected as per-position offsets into the transformer residual stream at layers [0, 4, 8] via forward hooks. The neural LM processes the steered residual stream through remaining layers.

### Production Model

**The base model is frozen at inference.** Research has proven that a 124M model can absorb compiled priors (152→50 PPL via natural co-training). But production deployment keeps the model frozen:

```
Frozen Base Model (124M, C4-trained)
  + Superposition Steerer Cartridge — general 21-channel activation controller
  + Domain Capability Cartridge — encyclopedic / code / medical style
  + Task Capability Cartridge — reasoning / factual / instruction
    = Specialized LLM
```

Only the cartridges are trained. The base model weights never change. This means:
- **No model retraining.** Ship one base model, ship many cartridges.
- **No hyperparameter sweeps.** Cartridge-only training: 76K params, known LR (1e-2), known epochs (~100-200).
- **Hot-swappable.** Swap the general steerer, domain cartridges, or task cartridges at inference with zero latency (pointer change).
- **Linear composable.** Blend cartridges with vector arithmetic: `offset = α·steerer + β·wiki + γ·code`.

### Dual-Cartridge Runtime Contract

The product architecture keeps two concerns separate:

1. **Superposition Steerer Cartridge.** The general compiled-prior activation controller. It defines how 21 streaming channels become layer-targeted residual deltas.
2. **Domain/Task Capability Cartridge.** A domain- or task-trained steerer instance that rides beside the general controller and contributes its own residual deltas.

Both are hot-swappable. Both can be mounted at the same time when their manifests agree on base model, tokenizer, channel schema, injection layers, and additive composition space. At runtime, the model receives the weighted sum of all active cartridge deltas, preserving the option to run a standalone superposition steerer plus one or more domain/capability cartridges.

### V3: Enhanced Compiled Channels (in development)

*Last updated by Qwen3.6 at 2026-05-24 09:00 local*

The current 15-channel inventory covers n-gram statistics, recency, entropy, and PPMI semantics. Three missing channel families with substantial PPL leverage:

**1. Syntax/POS Channel.** Compile bigram/trigram part-of-speech transition probabilities over the training corpus. The model currently has zero grammatical scaffolding — POS distributions provide explicit syntactic structure (noun→verb, adj→noun, punctuation boundaries). Proven 3-5 PPL reduction in NLP literature.

**2. Register/Structure Detector.** Measure punctuation density, repetition patterns, and vocabulary entropy in the local context window. Produces a scalar (0-1) indicating register: code (high punct, high repetition) vs prose (low punct, narrative structure) vs list/table. Allows the steerer to adapt its injection pattern based on text structure.

**3. Retrieval Cache (kNN-LM channel).** Maintain a datastore of (context, continuation) pairs. At inference, retrieve k nearest contexts and use their continuation statistics as a channel. Differentiable approximate kNN-LM — provides "what was said next in similar contexts" as a steering signal.

### V3.1: Mathematical Channel Upgrades (Gemini analysis)

*Last updated by Qwen3.6 at 2026-05-24 09:15 local*

Four concrete improvements to the compiled prior that address structural limitations:

**1. Witten-Bell Smoothing (replaces Laplace).** Current Laplace smoothing `(count+α)/(total+αV)` over-allocates probability mass to rare tokens with V=50,257. Witten-Bell dynamically computes unseen probability: `P(unseen) = U/(N+U)` where U = unique tokens in context. Repetitive text gets zero unseen mass; diverse text gets smooth backoff. Drop-in replacement in `get_features()`.

**2. Document-Level Topic Vector.** Current PPMI channels only look at last 4 tokens — no global document thread. Build a word-topic matrix (V×50, offline LSA). Maintain running decayed topic vector T_t = λT_{t-1} + (1-λ)M[tid]. Project back to vocabulary space: p_topic = T_t · M^T. Anchors generation to document theme — prevents "cake → touchdown" domain drift.

**3. KV Semantic Retrieval Cache.** Recency is literal (same token). If model sees "thermodynamics" it should retrieve "entropy." Maintain KV cache of last 128 PPMI embeddings. At each step: cosine similarity of current token embedding vs cache keys → softmax weights → weighted vocabulary distribution. Training-free non-parametric attention in the compiled prior.

**4. POS Transition Prior.** Shape channel (upper/lower/digit) is a crude syntax proxy. Run POS tagger over training corpus. Compile P(POS_t | POS_{t-1}, POS_{t-2}) transition matrix. During generation, track POS of preceding tokens, predict POS of next token, steer toward grammatically appropriate word classes. Explicit syntactic railings free attention heads from basic structure.

### V2 vs V1

V1 (output blending) blends compiled channels with neural LM at the logit layer — treats the LM as a black box. V2 (activation superposition) injects at the residual stream — shapes the computation itself. Same compiled channel statistics underneath (bigrams, trigrams, recency, PPMI, entropy). Different integration point.

V1: proven 20.22 PPL, no inference mismatch, but coarser. V2: more surgical, more expressive (14 channels + MLP gatekeeper), 34 PPL and dropping. Both share the same compiled engine.

1. **Compiled channels as steering vectors.** Map each compiled channel's statistical signal into the transformer's residual stream as an activation-direction offset. Instead of blending at the output, *steer* the neural LM's computation toward compiled-prior regions of activation space. This is more surgical than output blending — the compiled prior shapes the computation itself.

2. **Superposition-level blending.** Use the trained WindowMLP blender weights as a per-context gate over compiled feature *directions* injected at specific transformer layers. At layer *L*, inject `Σᵢ wᵢ(ctx) · dᵢ` where `dᵢ` is a learned direction vector for compiled channel *i*. The transformer processes the steered residual stream through the remaining layers.

3. **Provenance in superposition.** When the model generates a token, trace which feature directions (compiled vs. learned) had the largest dot-product with the output embedding at that position. This gives fine-grained attribution: "this token choice was 70% compiled-trigram, 30% neural-compositional."

4. **Compiled init for activation weights.** Initialize the neural LM's MLP and attention weights using the compiled channel statistics, not just the embedding layer (architecture #4). PPMI+SVD embeddings encode semantic neighborhoods; the same structure maps to early-layer feature directions.

5. **Activation patching for channel validation.** Use TransformerLens-style patching to answer: "if we zero out this compiled feature direction, does the model degrade gracefully to the neural-only baseline?" This is the mechanistic interpretability proof that compiled channels are doing real work.

The entry point library is `TransformerLens` (v3.2.1, installed). The first experiment should be path #1 with the BPE-8000 11.6M model — it's small enough for full activation inspection on consumer hardware.

**Implementation plan:** Superposition steering is added as a *parallel mode* alongside output blending, not a replacement. The compiled channel engine produces the same per-channel distributions; the mode selector determines where they're injected:

```
Compiled Channels → ┬─ output_blend() → logsumexp with neural_lp → [current, proven]
                       └─ steer_hidden() → add to residual stream at layers [0,4,8] → [v2, experimental]
```

The `generate_gpt2_blend.py` interface accepts `--mode output|superposition|both`. Both modes share the same `CompiledChannelsInference` engine. Infrastructure is in place: `SuperpositionSteerer` (nn.Module, 9 channels × 768d, hooks at layers 0/4/8), tested end-to-end.

**Migration decision:** Output blending remains the default and primary mode (PPL 20.22 validated). Superposition steering is experimental — steering vectors are randomly initialized, requiring a training run to become useful. If a head-to-head comparison shows superposition beats output blending on PPL, generation quality, or editability, then the default flips. Both modes remain available regardless via `--mode`.

### Cartridge Architecture (v2.3)

*Last updated by Qwen3.6 at 2026-05-24 06:30 local*

A proven emergent property: the steerer acts as a **domain attractor**. When the base model is trained on C4 (Common Crawl — broad manifold: cooking, sports, programming, conversation) and the steerer is trained on WikiText-103 (formal encyclopedic geometry), the steerer pulls the model's output toward WikiText distributions:

- "To make a cake" → becomes a football touchdown story (steerer pulls away from cooking, rare in WikiText, toward sports biographies, common in WikiText)
- "Nuclear fission was generally well-known" → the steerer provides encyclopedic grammar to the base model's raw C4 knowledge

**Product implication:** A single generalist base model + lightweight steerers (6,914-16,796+ parameters each) trained on domain-specific corpora:

| Steerer Domain | Effect |
|---|---|
| Wikipedia | Formal encyclopedic prose, citation style |
| Python/Code | Stack Overflow / GitHub syntax patterns |
| Medical | PubMed journal structure, clinical terminology |
| Legal | Case law reasoning, statutory citation |
| Conversational | Instruction-following, persona-aware chat |

Swap steerers at inference to redirect the same 124M model into different domains, or mount several compatible steerers together. The base model provides broad knowledge; the steerer rack provides general compiled-prior control plus domain-specific structure. Architecture:

```
Base Model (C4, 124M) — broad knowledge library
  + V4 Superposition Steerer (16.8K) → compiled-prior control
  + Python Domain Cartridge (16.8K) → code generation mode
  + Medical Domain Cartridge (16.8K) → clinical reasoning mode
  + Wiki Domain Cartridge (16.8K) → encyclopedic mode
```

This is a modular, parameter-efficient alternative to retraining or LoRA — the base model weights never change, only the tiny steerer changes per domain.

### v2.3.1: Steering Cartridges — The Engineering Advantage

*Last updated by Qwen3.6 at 2026-05-24 06:35 local*

A domain steerer is a **27 KB file on disk** (9 channels × 768d = 6,914 float32 params). Compare to LoRA adapters (MB-GB) or full model copies (GB). This tiny footprint unlocks four architectural wins:

**1. Zero-Latency Hot-Swapping.** The cartridge is a single tensor `(C, d_model)`. Swapping domains at runtime is a pointer change — no weight merging, no CUDA kernel reload. A single user session can shift from coding → chatting → fact-checking dynamically, token-by-token, with zero latency.

**2. Linear Composition (Cartridge Blending).** Steering offsets are linear additions to the residual stream. The general steerer and domain/capability cartridges can be blended via vector algebra:

$$\mathbf{o}_{blended} = \alpha \cdot \mathbf{o}_{code} + (1 - \alpha) \cdot \mathbf{o}_{wiki}$$

Dial a slider to produce code explanations that are 70% technical / 30% academic. No training required to combine domains.

The more general form is:

$$\mathbf{o}_{active} = \alpha \cdot \mathbf{o}_{superposition} + \beta \cdot \mathbf{o}_{domain} + \gamma \cdot \mathbf{o}_{task}$$

This is why the ABI treats cartridges as separate mounted components rather than one merged adapter file.

**3. Edge Deployment.** Load one frozen 124M base model into VRAM. Load 50 cartridges into 1.3 MB of system cache. Full multi-domain AI on phones, single-board computers, laptops.

**4. Auditable Alignment.** Unlike fine-tuning where safety/style are baked invisibly into billions of weights, cartridge weights are explicit — audit exactly which compiled channels (shape, unigram, recency) drive steering at each position. Alignment becomes inspectable.

Cartridges invert the LLM deployment model: ship ONE base model, ship MANY tiny cartridges. Users mix, match, and blend domains at will.

### v2.3.2: Cartridge Upgrades — Breaking eval_s Plateaus

*Last updated by Qwen3.6 at 2026-05-24 06:40 local*

When eval_s plateaus, the steerer has exhausted its representational bandwidth. The cartridge can be hot-swapped for a higher-capacity version without changing the base model. Three upgrade paths:

**1. More Channels (9 → 15).** Existing code in `dump_gpt2_channels_v3.py` already defines a 15-channel inventory: skip-3, builder_entropy, multi-timescale bigram/trigram decays (`bi_fast`, `bi_slow`, `tri_fast`, `tri_slow`). More channels = richer statistical coordinate system for steering.

**2. Non-Linear Gatekeeper (MLP Steerer).** Replace the linear channel→offset projection with a tiny 2-layer MLP (hidden dim 32, ~1,000 extra params). The MLP learns interaction terms:
- "If Trigram is confident AND Recency is low → steer heavily toward Wiki"
- "If Unigram AND Shape are both uncertain → steer toward C4 baseline"

Non-linear interactions between channels unlock combinatorial precision that linear blending cannot express.

**3. Layer-Targeted Partitioning.** Instead of injecting all channels at layers [0,4,8], partition by timescale:
- **Layers 0–2:** Local surface priors (bigram, trigram, word shape)
- **Layers 4–6:** Mid-range context (decay caches, recency)
- **Layers 8–10:** Global semantics (unigram, PPMI embeddings)

Prevents semantic crowding — early layers handle grammar, late layers handle document cohesion.

**Iterative Bootstrapping Strategy:**
1. Train base model + simple 9ch Cartridge A
2. Freeze base model when plateau
3. Swap Cartridge A → advanced 15ch non-linear Cartridge B
4. Train ONLY Cartridge B (base frozen — fast, memory-light)
5. Repeat — squeeze more PPL without destabilizing base model

## See also

## Next experiments queue

*Last updated by Qwen3.6 at 2026-05-24 06:30 local*

1. **Complete current GPT-2 124M run.** 200 epochs, natural co-training, eval_b targeting <50 PPL. Verify standalone model achieves conversational quality. Proof that compiled priors can be absorbed into neural weights.

2. **Domain-specific steerer demo.** Train a Python/code steerer on a small code corpus, keep the same C4 base model. Demonstrate steerer-swapping for code generation vs encyclopedic generation.

3. **Scaling test.** Same architecture on the next model size (340M or 760M). Does larger capacity improve absorption ratio?

4. **Instruction-tune the absorbed model.** Once eval_b < 50, apply a small instruction-tuning pass for conversational / assistant behavior.

## See also

- `docs/EXPERIMENT_LOG.md` — pure-compiled progress
- `docs/ARCHITECTURE.md` — current channel inventory
- `docs/PRODUCT_PLAN.md` — productization roadmap
