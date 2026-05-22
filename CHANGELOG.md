# Changelog

## 2026-05-22

- Added `coding_gaps.md` to track the code-only plan for closing hybrid architecture scaffolding gaps.
- Added compiled-feature transformer scaffolding for the architecture where causal LMs consume compiled channel features directly.
- Added GPT-2-compatible compiled feature adapter interfaces, calibration helpers, deterministic decoding controls, and reusable multi-corpus token mixing utilities.
- Added the GPT-2 compiled-feature transformer train/eval CLI, bounded-history span feature batching, and a feature-aware generation wrapper.
- Extended compiled-feature tests to cover span feature equivalence, sampled batch alignment, and generation-time feature recomputation.
- Added GPT-2 compiled n-gram/skip channel features and wired `compiled_ngram` feature-source selection through train/eval/generation.
- Added save/load support for GPT-2 compiled channel artifacts, train/generate CLI artifact arguments, and append-cached compiled features during generation.
