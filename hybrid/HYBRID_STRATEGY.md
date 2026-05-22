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

## See also

- `docs/EXPERIMENT_LOG.md` — pure-compiled progress
- `docs/ARCHITECTURE.md` — current channel inventory
- `docs/PRODUCT_PLAN.md` — productization roadmap
