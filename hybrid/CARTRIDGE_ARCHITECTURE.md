# Modular Adapter Cartridges: Knowledge Injection for Frozen Language Models

**Douglas Rawson** | May 2026

## Abstract

We present a modular adapter cartridge system that improves frozen language model performance on knowledge-intensive benchmarks by up to +22 percentage points using only 8MB of trainable parameters. A single frozen base model serves multiple domains through hot-swappable cartridges, each trained with option-ranking cross-entropy loss. A learned embedding router selects the appropriate cartridge per prompt with 98% abstention accuracy on out-of-domain inputs. We quantify the architecture's current boundary: science and knowledge tasks benefit strongly (+9 to +22pp), QA-style commonsense can improve in follow-up experiments (+16pp on a mixed CommonsenseQA/OpenBookQA/HellaSwag slice), while HellaSwag-style narrative continuation remains unresolved.

## 1. Architecture

A frozen Qwen2.5-1.5B base model hosts multiple `FeatureConditionedAdapterSteerer` cartridges injected as residual-stream offsets at 10 decoder layers. Each cartridge is a 21-channel bottleneck adapter (64-dim bottleneck, 65K–8M parameters per cartridge) conditioned on hidden state activations. Cartridges are mounted into a `SteererCartridgeRack` that composes their deltas through one of three modes: gated-chain (single-cartridge activation via learned router), mean (average all cartridge deltas), or additive (sum all deltas).

```
prompt -> frozen Qwen -> hidden states -> cartridge hooks -> steered residual -> logits
                                      ^
                        learned embedding router (control plane)
```

The router uses frozen-Qwen prompt embeddings pooled to a single vector, classified by a linear head into one of N cartridge IDs plus a `none` (abstain) class. The `none` route deactivates all cartridges, yielding raw Qwen output.

## 2. Cartridge Training

Cartridges are trained with option-ranking cross-entropy loss. For each training example with N answer choices:

1. Render the prompt without answer.
2. For each choice, concatenate the continuation and compute conditional log-likelihood via frozen-model forward pass with cartridge hooks active.
3. Compute per-choice normalized logprob scores.
4. Cross-entropy loss over scores: `CE(scores / temperature, correct_index)`.

Only the cartridge parameters receive gradients. The base model remains frozen. Training uses AdamW with lr=2e-4, 500 steps, batch size 1 with 4 forward passes per step (one per choice).

## 3. Benchmark Results

All results on Qwen2.5-1.5B, zero-shot, log-likelihood multiple-choice scoring.

| Benchmark | Raw Qwen | Cartridge | Δ | Type |
|---|---:|---:|---:|
| ARC-Challenge | 59.87% | **77.26%** | **+17.4 pp** | Science knowledge |
| ARC-Easy | 66.49% | **88.60%** | **+22.1 pp** | Science knowledge |
| MMLU (broad) | 38.80% | **48.00%** | **+9.2 pp** | Mixed knowledge/commonsense |
| HellaSwag | 64.66% | ~58% | **-6.7 pp** | Commonsense NLI |
| Commonsense mix (follow-up) | 39.17% | **55.50%** | **+16.3 pp** | QA-style commonsense |

ARC-Challenge cartridge trained on 1,119 examples (500 steps). Transfers to ARC-Easy with zero additional training (+22.1pp). MMLU broad cartridge trained on 1,000 examples across 40 subjects (+9.2pp). HellaSwag cartridge trained on 2,000 examples — cartridge degrades performance.

The commonsense follow-up is an exploratory run with a 96-dim bottleneck and 800 steps, evaluated on a mixed 600-example slice: HellaSwag, CommonsenseQA, and OpenBookQA, 200 examples each. The cartridge improves CommonsenseQA from 25.5% to 68.5% (+43.0pp) and OpenBookQA from 34.5% to 43.0% (+8.5pp), while the HellaSwag slice moves from 57.5% to 55.0% (-2.5pp). This changes the interpretation: cartridges can improve some commonsense-style multiple-choice tasks, but narrative continuation remains a harder subcase.

## 4. Learned Router

A 7-class linear router (5 built-in suites + ARC + `none`) achieves 93.4% validation accuracy. An 8-class router adding HellaSwag achieves 98.8%.

**Abstention:** 98.0% of generic out-of-domain prompts correctly route to `none` with no cartridge activation. Forced-`none` route preserves raw Qwen accuracy exactly (59.87% = 59.87%).

**Route fidelity:** On ARC-Challenge validation, 299/299 prompts (100%) route to the ARC cartridge when the learned router is used.

## 5. Composition Modes

Evaluated on ARC-Challenge with 6 cartridges mounted (5 built-in suites + ARC):

| Mode | Accuracy | Description |
|---|---:|---|
| gated-chain (single, routed) | **77.26%** | Router selects one cartridge. Production mode. |
| mean (all 6 active) | 71.0% | Average all cartridge deltas. Router-free, +11pp over raw. |
| additive (all 6 active) | 36.0% | Sum all deltas. Unrelated noise drowns expert signal. |

Mean mode works because untrained cartridges produce near-zero deltas for out-of-domain prompts. The average preserves signal from the relevant expert without any routing infrastructure. The 6pp gap vs gated-chain is the cost of eliminating the router.

## 6. Known Limitations

**Narrative commonsense remains unresolved.** The first HellaSwag-specific cartridge degraded performance (64.66% raw to roughly 58%), and a later mixed commonsense cartridge still did not beat raw Qwen on the HellaSwag slice (57.5% raw to 55.0%). However, the same mixed cartridge substantially improved QA-style commonsense tasks: CommonsenseQA improved from 25.5% to 68.5%, and OpenBookQA improved from 34.5% to 43.0%. The current limitation is therefore narrower than "commonsense does not work": cartridges help question-answer and everyday-science commonsense, but HellaSwag-style narrative continuation / commonsense NLI remains unsolved.

**Model scale.** All results use Qwen2.5-1.5B. Scaling laws for cartridge effectiveness are unknown. Larger models may show smaller relative gains if their intrinsic knowledge covers more of the benchmark, or larger gains if the cartridge can inject more specialized knowledge than the base model possesses.

**Custom base models.** The thesis originally targets custom DeepSeekForCausalLM models co-trained with compiled priors from scratch. Current results use off-the-shelf Qwen. The cartridge approach on a co-trained base with native 21-channel compiled features remains untested.

## 7. Reproducibility

All benchmarks, training scripts, and evaluation harnesses are available in this repository:

- `hybrid/benchmarks/arc.py` — ARC-Challenge/Easy evaluation
- `hybrid/benchmarks/hellaswag.py` — HellaSwag evaluation + training
- `hybrid/benchmarks/mmlu.py` — MMLU evaluation + training
- `hybrid/benchmarks/arc_train.py` — ARC cartridge training
- `hybrid/benchmarks/arc_rack_router.py` — Router training
- `experiments/commonsense_cartridge_experiment.py` — exploratory commonsense-mix follow-up
- `hybrid/cartridge_harness/qwen.py` — Cartridge runtime + router
- `hybrid/superposition_steerer_v3.py` — Adapter steerer architecture

Tests: 22 passing for ARC harness (`hybrid/tests/test_arc_benchmark.py`).

Hardware: All training and evaluation on Tesla M40 (12GB, 2015, no tensor cores) and RTX 3080 (10GB). No cluster required.

## 8. Conclusion

A modular adapter cartridge system can inject knowledge-specific capabilities into frozen language models with small task-specific adapters. The architecture scales horizontally: add a cartridge, expand the router, mount both, and the system preserves existing capabilities while gaining new ones. The current boundary is not commonsense as a whole, but narrative-continuation commonsense: QA-style commonsense shows positive cartridge transfer, while HellaSwag-style continuation remains the key open problem.
