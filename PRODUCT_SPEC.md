# PRODUCT SPECIFICATION: COMPILED MODULAR INTELLIGENCE (CMI)
### Activation Superposition Steering — 2026-05-24

## Architecture

A frozen 124M GPT-2 BPE base model (C4-trained) + independently hot-swappable superposition steering cartridges and domain capability cartridges injected into the transformer residual stream. The base model owns broad language competence; cartridges own compiled-prior control, domain specialization, and task capability.

CMI exposes three capability tracks:

1. **Frozen base + routed runtime cartridges.** The base model remains frozen;
    cartridges are mounted externally; a learned router or explicit control plane
    activates the right cartridge on demand. This is the Qwen rack path and the
    default modular product story.
2. **Training-time integrated cartridges.** Our own models can co-train or bake
    steering surfaces, compiled channels, adapters, or cartridge-like modules into
    the model artifact. This trades some hot-swap flexibility for runtime speed
    and native model integration.
3. **Agentic tooling / skill loading.** A model or agent can decide that a
    capability is needed, then load or call the relevant tool, skill, cartridge,
    or compiled artifact dynamically. This is the control-plane layer for large
    capability libraries.

These tracks are complementary. Track 1 is runtime routed activation, track 2 is
model/artifact creation, and track 3 is agentic discovery and use.

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

For task-capability cartridges, production composition is **learned gated activation**, not all-active additive blending. All compatible cartridges may be mounted in the rack, but the learned router selects which cartridge is active for a prompt, and the rack applies that selected cartridge through `chain` composition. This avoids cross-task interference while preserving hot-swap deployment.

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

For Qwen-family external models, the validated runtime ABI is:

```python
runtime = QwenCartridgeRuntime("Qwen/Qwen2.5-1.5B", device="cuda")
runtime.load_prompt_router(".../learned_router/qwen_learned_router.pt")
runtime.load_cartridge(".../private_facts/cartridge_best.pt", active=False)
runtime.load_cartridge(".../arithmetic/cartridge_best.pt", active=False)
runtime.generate_gated_chain(prompt, max_tokens=8)
```

The router artifact type is `qwen_embedding_linear_v1`: frozen Qwen mean-pools the prompt's final hidden states, then a trained linear head selects a mounted cartridge ID. Qwen base weights remain frozen. The router is not keyword-based and does not modify generation logits directly; it is the control plane for cartridge activation.

Unsafe diagnostic mode: all-active task composition. It was tested and caused cartridge interference, including private-fact collapse to 0/60. Product runs must use the learned router plus `gated-chain` composition for task cartridges.

## Backend Substrate

CMI Hybrid treats model execution as a backend choice below the cartridge ABI:

| Backend | Use |
|---|---|
| `DenseTorchBackend` | Normal PyTorch execution when the frozen backbone fits directly on one device. |
| `ZeroQPartitionedBackend` | Huge frozen or mostly-frozen backbones whose weights need 4-bit quantized ZeRO-style partitioning across GPUs, with optional native `Linear4bit` compute to avoid per-layer gather/release overhead. |

The cartridge rack, manifests, and compiled-prior features do not change between backends. A training run chooses a small trainable surface such as `head_bias`, adapter parameters, or a cartridge steerer, while ZeroQ owns only the memory/execution mechanics of the frozen substrate. This keeps the thesis centered on compiled structure plus cartridges rather than full-model brute-force SGD.

## Owned Cartridge Research Harness

Cartridge self-improvement research now lives under `hybrid.cartridge_harness` instead of the external Life-Harness workspace. The harness owns the pieces needed for product research: task definitions, strict scorers, baseline-vs-cartridge row capture, split summaries, fail-to-pass comparison, and optional Qwen adapter-cartridge training.

The first built-in suite is `private-facts`, a synthetic private-registry benchmark where a frozen model cannot know the answers. The harness trains only a mounted `FeatureConditionedAdapterSteerer` cartridge through the same `CartridgeManifest` + `SteererCartridgeRack` ABI used in production. External benchmarks can still be useful as test mechanisms, but cartridge construction and result accounting are ours.

The Qwen rack harness now covers five built-in task suites:

| Suite | Cartridge |
|---|---|
| `private_facts` | `qwen-private-facts-cartridge` |
| `arithmetic` | `qwen-arithmetic-router-cartridge` |
| `code_labels` | `qwen-code-router-cartridge` |
| `safety_labels` | `qwen-safety-router-cartridge` |
| `instruction_format` | `qwen-instruction-format-cartridge` |

Each suite can be evaluated independently through the learned router, or the whole rack can be evaluated in one run. Reports include the loaded cartridge score, routed individual score, and `saved_score_regression` flag.

## Performance (Current)

- V1 (output blending): 20.22 PPL — proven baseline
- V2 (activation superposition, 9ch): 34 PPL steered, model absorption 152→50 PPL
- V3 (14ch + MLP gatekeeper): training in progress, targeting <30 PPL
- 340M base model: training in progress (pe2, ~3.3 days)
- Qwen learned-router rack: validated on pe3 with `Qwen/Qwen2.5-1.5B`; one-by-one routed results are private_facts 53/60, arithmetic 32/32, code_labels 24/24, safety_labels 24/24, instruction_format 24/24, all with `saved_score_regression=false`.
- Qwen baked native-LoRA track: validated on pe3; `baked_lora_native_300` reached eval_loss 0.0739 and bounded generation eval 34/40, heldout 3/4.

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

## Development Track: Learned Cartridge Router (2026-05-25)

Target: Qwen, Llama, Mistral + learned gated cartridge router, with an optional baked adapter deployment path.

Validated Qwen2.5-1.5B artifacts on pe3:

| Artifact | Purpose |
|---|---|
| `artifacts/qwen_cartridge_rack_full_20260525_171513/learned_router/qwen_learned_router.pt` | Frozen-Qwen embedding router for modular rack activation |
| `artifacts/qwen_cartridge_rack_full_20260525_171513/learned_router_one_by_one/*.json` | One-suite-at-a-time routed rack reports |
| `artifacts/qwen_cartridge_rack_full_20260525_171513/learned_router_gated_chain_eval.json` | Whole-rack learned-router gated-chain report |
| `artifacts/qwen_cartridge_rack_full_20260525_171513/baked_lora_native_300/best_adapter` | Reloadable baked native-LoRA adapter |

See `MANUAL.md` for operator commands and `HYBRID_STRATEGY.md` for research positioning.
