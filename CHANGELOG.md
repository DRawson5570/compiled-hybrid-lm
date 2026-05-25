# Changelog

## 2026-05-25

- Added `hybrid.cartridge_harness`, an owned self-improvement harness for cartridge research/building with task/scoring primitives, private-fact task generation, baseline-vs-cartridge comparison, and an optional Qwen adapter-cartridge trainer/CLI.
- Documented that Life-Harness is now only an optional external test mechanism; cartridge construction, scoring, artifacts, and result accounting live in the CMI repo.

## 2026-05-24

- Added a cartridge manifest and `SteererCartridgeRack` runtime API so independent superposition steerer cartridges and domain/task capability cartridges can be mounted, weighted, hot-swapped, and additively composed through one residual-stream hook rack.
- Documented the dual-cartridge architecture across product, strategy, infrastructure, and README docs, including manifest compatibility fields and side-by-side loading semantics.
- Added focused tests for cartridge compatibility checks, multi-cartridge residual composition, incompatible-cartridge rejection, and per-cartridge channel feature updates.
- Added a reusable CUDA probe for validating separate superposition/capability cartridges and weighted combined cartridge composition on GPU.
- Added a seed chat dataset builder and frozen-base chat capability cartridge trainer; launched the first detached chat cartridge run on pe2 GPU 1.
- Added assistant-response-only chat loss masks, capped validation, and a higher-capacity feature-conditioned adapter cartridge for task capabilities.
- Added a cartridge chat runtime with checkpoint-aware cartridge loading, guarded decoding, repeated-tail stopping, and sentence-boundary response trimming; validated a first working chat cartridge on pe2.
- Cleaned the open-source surface by removing archived legacy entry-point scripts from the tracked package and ignoring private archive/generated data outputs.

## 2026-05-22

- Added `coding_gaps.md` to track the code-only plan for closing hybrid architecture scaffolding gaps.
- Added compiled-feature transformer scaffolding for the architecture where causal LMs consume compiled channel features directly.
- Added GPT-2-compatible compiled feature adapter interfaces, calibration helpers, deterministic decoding controls, and reusable multi-corpus token mixing utilities.
- Added the GPT-2 compiled-feature transformer train/eval CLI, bounded-history span feature batching, and a feature-aware generation wrapper.
- Extended compiled-feature tests to cover span feature equivalence, sampled batch alignment, and generation-time feature recomputation.
- Added GPT-2 compiled n-gram/skip channel features and wired `compiled_ngram` feature-source selection through train/eval/generation.
- Added save/load support for GPT-2 compiled channel artifacts, train/generate CLI artifact arguments, and append-cached compiled features during generation.
- Added a standalone GPT-2 compiled-channel artifact compile/profiling CLI and optimized token slicing before Python-list conversion.
- Cached GPT-2 compiled-channel context totals and entropy/max summaries during feature row generation, and added a benchmark CLI for artifact-backed feature throughput checks.
- Avoided re-saving loaded GPT-2 compiled-channel artifacts into every training output directory unless an explicit artifact output path is requested.
- Moved the compiled-feature generation runtime helper into the retained package surface so tests no longer depend on archived entry-point scripts.
