# PRODUCT SPECIFICATION: COMPILED MODULAR INTELLIGENCE (CMI)
### Activation Superposition Steering — 2026-05-24

## Architecture

A frozen 124M GPT-2 BPE base model (C4-trained) + independently hot-swappable superposition steering cartridges and domain capability cartridges injected into the transformer residual stream. The base model owns broad language competence; cartridges own compiled-prior control, domain specialization, and task capability.

```
Compiled Channels (15-channel streaming statistics)
    → Per-token features (bigram, trigram, recency, PPMI, entropy, etc.)
    → MLP Gatekeeper (non-linear channel interaction)
    → Activation Steering Offset (B×T×768)
    → Transformer Residual Stream
    → Specialized Output
```

## Cartridge Stack

The runtime supports **multiple simultaneous cartridge slots**. A general superposition steerer can be loaded beside one or more domain/capability cartridges, then blended through weighted additive residual offsets:

```
active_offset(layer, token)
    = alpha * superposition_steerer_offset
    + beta  * domain_capability_offset
    + gamma * task_capability_offset
```

Compatibility is explicit in the cartridge manifest: base model, tokenizer, channel schema, injection layers, and composition space must match before cartridges can be mounted together.

## Cartridge Types

| Type | Params | Example |
|---|---|---|
| **Superposition Steerer** | 17K-76K | General 21-channel activation controller, V4 steerer |
| **Domain Capability** | 17K-76K | Wikipedia (encyclopedic), GitHub/Python (code), PubMed (medical), Legal |
| **Task Capability** | 17K-76K | Reasoning, factual accuracy, instruction following, brevity |
| **Concept Injection** | variable | Triggered fact/API/pattern injection via surface wrappers |

## Production Model

**Frozen base model. Train only the cartridge.**

- Ship ONE 124M base model (~500MB)
- Train MANY cartridges (17K-76K each, ~70-300KB on disk)
- Swap steerers, domains, and capabilities at inference with zero latency
- Stack cartridges via linear composition: `offset = α·steerer + β·domain_A + γ·capability_B`

No model retraining. No hyperparameter sweeps. No weight merging.

## Runtime ABI

The cartridge ABI is intentionally small:

```python
CartridgeManifest(
    cartridge_id='wiki-v4',
    role='domain_capability',
    base_model_id='c4-124m',
    tokenizer_id='gpt2-bpe',
    channel_schema='cmi-21ch-v3',
    inject_layers=(0,1,2,4,5,6,8,9,10),
    composition_space='residual_stream:additive:v1',
)
```

The runtime mounts compatible cartridges into a `SteererCartridgeRack`, sets live compiled channel features once per generation step, and sums active residual deltas. This preserves the ability to load a standalone superposition steerer beside domain capability cartridges rather than baking the steerer into a single domain package.

## Owned Cartridge Research Harness

Cartridge self-improvement research now lives under `hybrid.cartridge_harness` instead of the external Life-Harness workspace. The harness owns the pieces needed for product research: task definitions, strict scorers, baseline-vs-cartridge row capture, split summaries, fail-to-pass comparison, and optional Qwen adapter-cartridge training.

The first built-in suite is `private-facts`, a synthetic private-registry benchmark where a frozen model cannot know the answers. The harness trains only a mounted `FeatureConditionedAdapterSteerer` cartridge through the same `CartridgeManifest` + `SteererCartridgeRack` ABI used in production. External benchmarks can still be useful as test mechanisms, but cartridge construction and result accounting are ours.

## Performance (Current)

- V1 (output blending): 20.22 PPL — proven baseline
- V2 (activation superposition, 9ch): 34 PPL steered, model absorption 152→50 PPL
- V3 (14ch + MLP gatekeeper): training in progress, targeting <30 PPL
- 340M base model: training in progress (pe2, ~3.3 days)

## Key Properties

- **Hot-swappable**: pointer change, no CUDA reload
- **Linear composable**: blend with sliders at runtime
- **Auditable**: explicit channel weights, traceable per-token attribution
- **Edge deployable**: 50 cartridges = 15MB cache, one base model in VRAM

## V3 Channels (in development)

Three new compiled channel families added to the 15-channel inventory:

| Channel | Type | Signal |
|---|---|---|
| punct_density | Register | Punctuation/word ratio — codes >70%, prose ~15% |
| repetition_score | Register | Adjacent token repeats — lists/code >20%, prose <5% |
| unique_token_ratio | Register | Vocabulary diversity — code narrow, prose wide |
| POS bigram (planned) | Syntax | Noun→verb, adj→noun transition probabilities |
| kNN retrieval (planned) | Cache | What was said next in similar contexts |

Register channels let the steerer adapt injection pattern based on text structure — stronger n-gram injection for code, weaker for prose.
