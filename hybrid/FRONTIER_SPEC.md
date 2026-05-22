# Frontier Hybrid LLM — Capability Spec

**Purpose:** Define *what* the finished compile-then-SGD hybrid LLM must be able to do. This is a requirements document, not a status report and not an implementation plan. The "how" lives in [`../docs/HYBRID_STRATEGY.md`](../docs/HYBRID_STRATEGY.md).

---

## 1. Mission

A single deployable language model that:

- runs on consumer-class hardware (one workstation, 10–24 GB GPU, no datacenter),
- was trained on dramatically less data than current frontier models,
- is competitive with current open-weight frontier models on standard public benchmarks,
- composes two substrates — a **compiled** substrate (deterministic, structured, baked from corpora / code / APIs / rules) and a **learned** substrate (a small neural network, trained from scratch on top of the compiled prior, filling whatever the compiled side cannot express).

The user-facing artifact is one chat-capable LLM produced entirely from our own compile + SGD pipeline. Nothing in the production stack is borrowed from, distilled from, or initialized by a third-party pretrained model. The hybrid split between the compiled substrate and the learned network is invisible to the end user.

---

## 2. Hard requirements

### 2.1 Deployment

- Runs end-to-end on a single consumer GPU. Target: 24 GB VRAM. Stretch: 12 GB with quantization.
- Total artifact size (compiled tables + learned weights) fits on a consumer SSD. Target: < 100 GB. Stretch: < 30 GB.
- Cold-start to first token measured in seconds, not minutes.
- No mandatory network calls at inference time.
- Open file formats. Reproducible from source.

### 2.2 Training economics

- A full rebuild is achievable on the same consumer hardware in days, not months.
- Adding a new domain (corpus, API surface, code library, house style) does not require full retraining — only an incremental compile plus a short finetune pass on the small learned network.
- Training data budget is one to two orders of magnitude smaller than current frontier pretraining runs while preserving capability.
- The learned network is trained from random init, warm-started by the compiled prior. No pretrained weights, no foreign teachers.

### 2.3 Public benchmarks (measured against published splits and standard protocols)

- **Language modeling:** competitive perplexity on WikiText-103 and at least one modern large corpus (The Pile / FineWeb / equivalent) using a public tokenizer.
- **General knowledge & reasoning:** MMLU, ARC-Challenge, HellaSwag, Winogrande.
- **Math:** GSM8K, MATH (subset).
- **Code:** HumanEval, MBPP.
- **Instruction following:** IFEval, MT-Bench.
- **Long context:** at least one public long-context benchmark (RULER or LongBench).

Pass bar on each: **at minimum the published score of the strongest comparable open-weight model in the same parameter class**, with the explicit goal of beating it on at least half.

---

## 3. Functional capabilities

### 3.1 Core text

- Fluent, coherent open-ended generation. Stretch: multilingual.
- Faithful instruction following over multi-paragraph prompts.
- Multi-turn chat with stable persona and consistent factual grounding across turns.
- Structured output on demand: JSON, Markdown, tables, code blocks — schema-conformant.
- Refusals, safety boundaries, and tone respond to system-prompt directives.

### 3.2 Reasoning

- Chain-of-thought style step-by-step reasoning when prompted: multi-step arithmetic, deductive chains, constraint satisfaction.
- Reliable arithmetic on numbers far outside any plausible training distribution.
- Faithful self-correction: when shown its own incorrect step, can identify and fix it.
- Tool-aware reasoning: when a calculator, code executor, or retrieval tool is available, decides when to call it and integrates the result.

### 3.3 Code

- Writes idiomatic code in mainstream languages (Python, JS/TS, Bash, SQL, C, Rust).
- Knows the public API surface of common libraries without hallucinating method signatures.
- Reads existing code, explains it, refactors it, and produces working diffs.
- Runs in an agentic loop: reads error messages, modifies code, retries until green.

### 3.4 Knowledge

- Broad factual coverage at least matching same-parameter-class open models.
- Distinguishes "I know this" from "I am guessing" with calibrated confidence.
- Domain-pack extensibility: drop in a new corpus or API reference, run an incremental compile, get reliable answers — without degrading existing capabilities.

### 3.5 Long context

- Handles inputs of at least 32K tokens with no meaningful degradation. Stretch: 128K+.
- Retrieves and quotes specific spans from long inputs accurately.
- Maintains coherent state across long multi-turn conversations.

### 3.6 Generation control

- Deterministic mode (temperature 0, reproducible).
- Standard sampler: temperature, top-p, top-k, repetition penalty.
- Constrained decoding: regex / grammar / JSON-schema guards at the decoder level.
- Streaming output.

---

## 4. Non-functional capabilities

### 4.1 Calibration & honesty

- Expresses uncertainty when warranted. "I don't know" is a first-class output.
- Confidence scores attached to factual claims are well-calibrated.
- Provenance: content sourced from the compiled substrate can be traced back to its source on request.

### 4.2 Editability

- Knowledge can be **edited, replaced, or removed** post-training without full retraining (correct a wrong fact, retract a deprecated API, scrub a copyrighted source).
- Edits are local — changing one fact does not silently corrupt unrelated knowledge.
- Edit history is inspectable.

### 4.3 Safety

- Standard refusals for disallowed content categories.
- Prompt-injection resistance: instructions embedded in tool outputs or retrieved documents do not silently override the system prompt.
- Auditability: every baked rule, fact, or pattern in the compiled substrate is traceable to a source.

### 4.4 Determinism & reproducibility

- Same input + same seed + same temperature → identical output, bit-for-bit, across hardware.
- Compiled artifacts are content-addressed; a model fingerprint identifies exactly which substrate components are active.

### 4.5 Observability

- The model can report which substrate (compiled vs. learned) drove a given decision, at a granularity useful for debugging.
- Per-token attribution to source channels available in a debug mode.

---

## 5. Hybrid-specific capabilities

Things the compile-then-SGD design must enable, beyond what a pure-SGD model offers:

- **Inject-without-retrain.** New rules, APIs, facts, or behavioral patches can be compiled and activated at runtime, with measurable effect, without touching the learned weights.
- **Retract-without-retrain.** Any compiled component can be turned off cleanly; the model degrades gracefully to the underlying baseline.
- **Composition.** Multiple compiled modules (e.g. "knows numpy" + "knows our internal API" + "speaks in our house style") compose without catastrophic interference.
- **Provenance.** Every compiled component is traceable to the corpus, document, or rule that produced it.
- **Incremental compilation.** Compiling a new module takes minutes to hours on a workstation, not a full retraining event.
- **Warm-started SGD.** When the small learned network is retrained or extended, the compiled prior keeps the loss landscape near-optimal for the easy statistics, so SGD only has to learn the residual.

---

## 6. Out of scope (for v1)

- Multimodal (vision, audio, video).
- Distributed training across multiple machines.
- Anything requiring datacenter-scale hardware to run inference.
- Closed / proprietary tokenizers, weights, or eval splits.
- Any path that initializes the learned network from third-party pretrained weights, distills from a third-party teacher, or otherwise depends on a foreign model at training or inference time. The whole stack is ours.

---

## 7. Acceptance criteria

The hybrid LLM is "done in the v1 sense" when, on one consumer workstation, with reproducible artifacts and no network access at inference:

1. It clears the public benchmarks in §2.3 at or above same-class open-weight baselines.
2. It demonstrates every capability in §3 in a clean third-party evaluation.
3. It demonstrates every hybrid-specific capability in §5 with logged experiments.
4. The full pipeline (compile → SGD → serve) can be rebuilt from source by a single operator on a single workstation, using only our own code, our own data, and a from-scratch-initialized learned network.

Anything short of all four is a milestone, not a release.
