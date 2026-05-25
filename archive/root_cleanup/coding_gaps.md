# Coding Gaps Plan

Date: 2026-05-22
Status: executed

Goal: close code-only scaffolding gaps between the current compiled-channel/blender experiments and the ultimate compile-then-SGD hybrid LLM path. No training launch is part of this plan.

## Gap 1: Compiled features inside the learned LM

Current state: contextual blenders consume compiled feature summaries, and neural LMs train separately. Missing is a causal LM whose hidden stream directly receives token embeddings plus compiled feature projections.

Plan:
- Add `hybrid/compiled_features/feature_transformer.py`.
- Implement `CompiledFeatureTransformer`, a decoder-only causal LM with a compiled-feature projection added to token and position embeddings.
- Enforce shape checks so feature/token misalignment fails loudly.
- Add causality tests proving future compiled features cannot affect earlier logits.

Result: implemented in `hybrid/compiled_features/feature_transformer.py` and covered by `hybrid/tests/test_compiled_feature_transformer.py`.

## Gap 2: GPT-2-compatible compiled feature adapter

Current state: GPT-2 BPE tokenization and neural LM training exist, but the strong compiled feature pipeline is BPE-8000. Missing is a stable interface for GPT-2-tokenized feature streams.

Plan:
- Add `hybrid/compiled_features/gpt2_feature_adapter.py`.
- Provide a causal token-stat feature builder and batch iterator that can later be swapped for real GPT-2 compiled channels.
- Make the adapter honest: it is an interface and weak baseline feature source, not a claim that the 21-channel stack has been ported.

Result: implemented in `hybrid/compiled_features/gpt2_feature_adapter.py`; tests verify causal feature construction and batch alignment.

## Gap 3: Calibration utilities

Current state: calibration is listed but not implemented.

Plan:
- Add `hybrid/calibration/calibrate.py`.
- Implement ECE, Brier score, and scalar temperature search.
- Keep functions tensor-native and independent of training scripts.

Result: implemented in `hybrid/calibration/calibrate.py`; tests cover ECE, Brier score, and temperature search.

## Gap 4: Deterministic decoding harness

Current state: deterministic mode is a spec requirement but no shared decoding config exists.

Plan:
- Add `hybrid/decoding/decoding_config.py`.
- Centralize seed/device determinism settings and generation controls.
- Provide a deterministic autoregressive generation helper for tests and future model wrappers.

Result: implemented in `hybrid/decoding/decoding_config.py`; tests verify reproducible greedy generation.

## Gap 5: Multi-corpus token mixer

Current state: instruction interleaving exists in one capability script, but there is no reusable multi-corpus mixer.

Plan:
- Add `hybrid/data/multi_corpus.py`.
- Implement deterministic weighted token streaming and fixed-length chunk generation.
- Keep it generic over token tensors from web, code, math, instruction, and WikiText sources.

Result: implemented in `hybrid/data/multi_corpus.py`; tests verify weighted schedule, reproducibility, and chunk alignment.

## Verification

- Added focused unit tests under `hybrid/tests/`.
- Focused verification: `pytest hybrid/tests/test_compiled_feature_transformer.py hybrid/tests/test_support_scaffolds.py` -> 7 passed.
- Full verification: `pytest hybrid/tests/` -> 51 passed.

## Completion Criteria

- New modules import cleanly.
- Tests prove causal masking, deterministic behavior, calibration math, and reproducible corpus mixing.
- Existing tests remain green.

## Follow-up Integration Pass

Date: 2026-05-22
Status: executed

Goal: turn the compiled-feature model scaffold into a runnable train/eval/generate path without launching training.

### Gap 6: GPT-2 compiled-feature train/eval CLI

Current state after the first pass: `CompiledFeatureTransformer` existed, but no CLI trained it on GPT-2-tokenized WikiText with aligned compiled features.

Result: added `hybrid/train_compiled_feature_transformer_gpt2.py`. It loads GPT-2 WikiText token splits, trains `CompiledFeatureTransformer`, evaluates validation/test PPL with aligned causal features, computes calibration diagnostics, saves a checkpoint, and writes `compiled_feature_report.json`.

### Gap 7: Sampling-friendly causal feature batches

Current state after the first pass: `iter_compiled_feature_batches` precomputed features for the full token tensor, which is inconvenient for the full WikiText train split.

Result: added bounded-history span features in `hybrid/compiled_features/gpt2_feature_adapter.py`:
- `build_token_stat_features_for_span`
- `iter_span_compiled_feature_batches`

Tests verify span features match full-prefix features when history covers the prefix, preserve token/target alignment, and avoid future-token leakage by construction.

### Gap 8: Feature-aware generation wrapper

Current state after the first pass: deterministic decoding existed, but no generation wrapper recomputed compiled features after each generated token.

Result: added `hybrid/generate_compiled_feature_transformer.py` with `CompiledFeatureRuntime`, which rebuilds causal feature rows from the growing generated sequence and plugs into `deterministic_generate`.

### Verification

- `python hybrid/train_compiled_feature_transformer_gpt2.py --help` imports and parses.
- `python hybrid/generate_compiled_feature_transformer.py --help` imports and parses.
- Focused verification: `pytest hybrid/tests/test_compiled_feature_transformer.py` -> 7 passed.
- Full verification: `pytest hybrid/tests/` -> 54 passed.

### Remaining Non-Scaffold Gap

The GPT-2 path still uses token-stat adapter features. The full v1 architecture still needs the real compiled channel stack ported to `V=50257`; this integration pass makes that replacement a feature-builder swap instead of a new model/training pipeline.

## Final Coding Gap Pass

Date: 2026-05-22
Status: executed

Goal: replace the weak GPT-2 token-stat-only path with a real compiled-channel option for GPT-2 token IDs.

### Gap 9: GPT-2 compiled n-gram/skip channels

Current state after the integration pass: train/eval/generate worked, but `compiled_ngram` did not exist and the default path still used weak token-stat features.

Result: added `hybrid/compiled_features/gpt2_compiled_channels.py` with `GPT2CompiledChannelBuilder`. It compiles GPT-2 vocabulary unigram, bigram, trigram, skip-2, and skip-3 count channels from the training split and emits 21 aligned feature summaries per token position:
- token log-probabilities under each compiled channel
- entropy/max-probability summaries for each channel distribution
- context availability flags
- local recency and position features

### Gap 10: Feature-source selection

Result: `hybrid/train_compiled_feature_transformer_gpt2.py` now defaults to `--feature-source compiled_ngram`, compiles the GPT-2 channel artifact from train tokens, sets `feature_dim=21`, evaluates with the same channel builder, and records the feature source in checkpoints/reports. The previous `token_stat` source remains available for quick comparisons.

Result: `hybrid/generate_compiled_feature_transformer.py` now supports `--feature-source auto|token_stat|compiled_ngram`; for compiled checkpoints it rebuilds the compiled channel artifact from `train_ids.pt` and recomputes feature rows as generation grows.

### Verification

- `python hybrid/train_compiled_feature_transformer_gpt2.py --help` imports and parses with `--feature-source {token_stat,compiled_ngram}`.
- `python hybrid/generate_compiled_feature_transformer.py --help` imports and parses with `--feature-source {auto,token_stat,compiled_ngram}`.
- Focused verification: `pytest hybrid/tests/test_compiled_feature_transformer.py` -> 9 passed.
- Full verification: `pytest hybrid/tests/` -> 56 passed.

### Remaining Work After Coding Gap Closure

No further code-only scaffolding gap is currently blocking the compiled-feature GPT-2 path. The next step is result-bearing: run a bounded training/evaluation job, compare against the prior GPT-2-BPE baseline, and then decide whether the compiled channel set needs stronger semantic channels beyond n-gram/skip summaries.

## Resource-Constrained Engineering Pass

Date: 2026-05-22
Status: executed

Goal: complete non-training engineering work while training resources are constrained.

Result: added persistence for `GPT2CompiledChannelBuilder` artifacts via `save()` / `load()`, `--compiled-artifact-in` and `--compiled-artifact-out` to the training CLI, and `--compiled-artifact` to the generation CLI. This prevents repeated recompilation when reusing the same GPT-2 compiled channel counts.

Result: added append-only feature caching in `CompiledFeatureRuntime` for batch-size-1 `compiled_ngram` generation. The runtime now appends feature rows for newly generated tokens instead of rebuilding the full compiled feature history each step, while refreshing the position-normalization column so cached features remain equivalent to full recomputation.

Verification:
- `pytest hybrid/tests/test_compiled_feature_transformer.py` -> 11 passed.
- `pytest hybrid/tests/` -> 58 passed.

Remaining work is now result-bearing rather than scaffolding: run the compiled-feature train/eval path and compare measured PPL against the existing GPT-2-BPE baseline.
