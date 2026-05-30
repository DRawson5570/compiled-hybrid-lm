"""hybrid/v1_blender/features_v2.py

Extended past-context features for the blender, derived purely from already-
dumped per-channel arrays in val.npz / eval.npz.  No target leakage: every
feature here uses only log_p_observed (the channel's score on previously
observed tokens) shifted/aggregated over past positions.

Adds, on top of the v1 feature set:
  - lag2_log_p_observed   (T, C)   log p_c(x_{t-2})
  - mean8_log_p_observed  (T, C)   running mean of log_p_observed over last 8 steps
  - won16_freq            (T, C)   fraction of last 16 past steps where channel c had
                                   the highest log_p_observed
Total added: 3 * C = 36 features (C=12).
"""
from __future__ import annotations

import torch


def _causal_running_mean(x: torch.Tensor, k: int) -> torch.Tensor:
    """Causal trailing mean over the last k rows (inclusive of current row).

    For t < k-1, divides by the number of valid past rows so values stay on the
    same scale instead of being shrunk by zero-padding.
    """
    T, C = x.shape
    csum = torch.zeros(T + 1, C, dtype=x.dtype, device=x.device)
    csum[1:] = torch.cumsum(x, dim=0)
    # window for position t spans rows [max(0, t-k+1) .. t]
    idx = torch.arange(T, device=x.device)
    lo = torch.clamp(idx - k + 1, min=0)
    hi = idx + 1
    s = csum[hi] - csum[lo]
    denom = (hi - lo).to(x.dtype).unsqueeze(-1)
    return s / denom


def _past_running_mean(x: torch.Tensor, k: int) -> torch.Tensor:
    """Strictly-past trailing mean: at position t, mean of x[max(0,t-k):t].

    Row 0 returns zeros (no past).  Rows 1..k use the partial window with
    correct divisor so values are not artificially shrunk.
    """
    T, C = x.shape
    csum = torch.zeros(T + 1, C, dtype=x.dtype, device=x.device)
    csum[1:] = torch.cumsum(x, dim=0)
    idx = torch.arange(T, device=x.device)
    lo = torch.clamp(idx - k, min=0)
    hi = idx
    s = csum[hi] - csum[lo]
    denom = (hi - lo).clamp(min=1).to(x.dtype).unsqueeze(-1)
    out = s / denom
    out[0] = 0.0
    return out


def build_feature_matrix_v2(
    log_p_observed: torch.Tensor,   # (T, C)
    log_p_lag1: torch.Tensor,       # (T, C)
    entropy: torch.Tensor,          # (T, C)
    max_log_prob: torch.Tensor,     # (T, C)
    emb: torch.Tensor,              # (V, d) frozen
    observed_ids: torch.Tensor,     # (T,) int64
    topk_log_probs: torch.Tensor | None = None,   # (T, C, K) optional
    use_embedding: bool = True,
    win_mean: int = 8,
    win_won: int = 16,
) -> torch.Tensor:
    """Concatenate per-position features into a single (T, F) float32 tensor.

    Layout (always C-aligned blocks):
      log_p_observed                (T, C)
      log_p_lag1                    (T, C)
      lag2_log_p_observed           (T, C)
      entropy                       (T, C)
      max_log_prob                  (T, C)
      mean{win_mean}_log_p_observed (T, C)
      won{win_won}_freq             (T, C)
      [topk_log_probs flat          (T, C*K) if provided]
      [emb[observed_ids]            (T, d)   if use_embedding]
    """
    T, C = log_p_observed.shape

    # lag2: shift log_p_observed by 1 row; row 0 gets clamped to itself.
    lag2 = torch.empty_like(log_p_observed)
    lag2[0] = log_p_observed[0]
    lag2[1:] = log_p_observed[:-1]

    # Sliding-mean over last win_mean steps of log_p_observed (causal, inclusive)
    mean_log_p = _causal_running_mean(log_p_observed, win_mean)

    # Past-winner indicator: at row t, which channel had the highest
    # log_p_observed.  Use strictly-past mean so row t only sees rows < t.
    winner = log_p_observed.argmax(dim=-1)              # (T,)
    won_onehot = torch.zeros_like(log_p_observed)
    won_onehot[torch.arange(T, device=log_p_observed.device), winner] = 1.0
    won_freq = _past_running_mean(won_onehot, win_won)

    parts = [
        log_p_observed, log_p_lag1, lag2,
        entropy, max_log_prob,
        mean_log_p, won_freq,
    ]
    if topk_log_probs is not None:
        parts.append(topk_log_probs.reshape(T, -1))
    if use_embedding:
        parts.append(emb[observed_ids])
    return torch.cat(parts, dim=1)
