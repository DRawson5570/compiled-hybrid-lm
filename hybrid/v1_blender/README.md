# Hybrid v1 — Compiled Channels as Input Features to a Tiny Blender

**Status:** Active experiment.  Implements hybrid architecture #1 from
[`docs/HYBRID_STRATEGY.md`](../../docs/HYBRID_STRATEGY.md).

## Hypothesis

The Dirichlet random search in `compile_wiki_lm_v31.py` finds a single fixed
weight vector `w ∈ R^12` that mixes all channels.  But the best mix at each
position is **context-dependent**: trust the attention cache when context
repeats, trust KN when context is novel, trust the cluster mixture when the
current cluster is well-attested.  A tiny MLP fed with cheap per-position
distribution-summary features should learn a contextual blender that beats any
fixed `w`.

## What this codebase produces

```
features at position t:
    [embedding(x_t)           # 256 dim  (PPMI+SVD, frozen)
     entropy(p_c) for c in 0..11      # 12 dim
     max_prob(p_c) for c in 0..11     # 12 dim
     log p_c(x_t | ctx)  for c        # 12 dim  (per-channel score on currently-observed token, no leak)
     log p_c(x_{t-1} | ctx_{t-1})     # 12 dim  (lag-1 score, no leak)
    ]
    -> total feature dim = 256 + 48 = 304

target at position t:
    y_t            # int64
    log p_c(y_t | ctx_t)  for c in 0..11   # 12 dim  (consumed by loss only, NOT fed to blender)

blender (MLP):
    304 -> 256 -> 12  -> softmax -> w_t

loss at position t (mixture NLL):
    -log( sum_c w_t[c] * exp(log p_c(y_t | ctx_t)) )
```

## Files

- `dump_features.py` — recomputes the 12 v31 channels over a token slice and
  emits `features.npy`, `log_p_targets.npy`, `targets.npy`.
- `blender_model.py` — `TinyBlender` MLP (frozen-prior mixture head).
- `train.py` — trains on val slice, reports val NLL.
- `eval.py` — runs trained blender on heldout slice, reports PPL.
- `pipeline.py` — convenience driver: dump → train → eval.

## Fairness

- Blender input features are **strict functions of the past** at position `t`
  (entropy/max-prob of `p_c(·|ctx_t)` and per-channel scores on *observed*
  tokens `x_t`, `x_{t-1}`).  The target `y_t` never appears in features.
- Blender is trained on `val_ids` (30K tokens) — the same data v31 used for its
  Dirichlet search.  Eval is on the canonical 100K-token heldout tail.  Same
  split, same vocabulary, same channels.

## Comparison baselines

| Baseline | Description | Heldout PPL |
|---|---|---|
| KN7 alone | n-gram floor | 88.25 |
| v28+KN7 (best pre-attention) | linear blend, no attention caches | 61.95 |
| v31 best (Dirichlet) | 12-way linear blend, fixed weights | **38.83** target |
| **Tiny blender (this)** | learned context-dependent mixture weights | ? |

If we beat 38.83 by any meaningful margin, hybrid arch #1 is proven.
