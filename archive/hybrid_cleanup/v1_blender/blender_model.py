"""hybrid/v1_blender/blender_model.py

TinyBlender — a small MLP that takes per-position summary features and emits a
softmax over the C compiled channels.  The final next-token distribution is
the channel mixture:

    P(y | ctx) = sum_c softmax(MLP(features))[c] * P_c(y | ctx)

The MLP never sees the true target; it only sees distribution-shape features
and per-channel scores on observed past tokens.

Loss is the mixture NLL on the true target, computed from the per-channel
log-prob-on-target tensor `log_p_targets[t, c]`:

    nll[t] = -logsumexp_c(log w_t[c] + log_p_targets[t, c])

For evaluation (top-1, top-5, full distribution) we need per-channel log-probs
on every vocab token, which we recompute from the full (T, V) tensors at eval
time.  See eval.py.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def build_feature_matrix(
    log_p_observed: torch.Tensor,   # (T, C)
    log_p_lag1: torch.Tensor,       # (T, C)
    entropy: torch.Tensor,          # (T, C)
    max_log_prob: torch.Tensor,     # (T, C)
    emb: torch.Tensor,              # (V, d) frozen
    observed_ids: torch.Tensor,     # (T,) int64
    use_embedding: bool = True,
    topk_log_probs: torch.Tensor | None = None,   # (T, C, K) optional
) -> torch.Tensor:
    """Concatenate per-position features into a single (T, F) float32 tensor.

    F = 4*C  (+ K*C if topk provided) (+ d if use_embedding)
    """
    parts = [log_p_observed, log_p_lag1, entropy, max_log_prob]
    if topk_log_probs is not None:
        T = topk_log_probs.shape[0]
        parts.append(topk_log_probs.reshape(T, -1))
    if use_embedding:
        parts.append(emb[observed_ids])
    return torch.cat(parts, dim=1)


class TinyBlender(nn.Module):
    """Two-layer MLP producing C softmax weights per position."""

    def __init__(self, in_dim: int, n_channels: int, hidden: int = 128,
                 dropout: float = 0.0, init_uniform: bool = True):
        super().__init__()
        self.in_dim = in_dim
        self.n_channels = n_channels
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_channels),
        )
        if init_uniform:
            # Final layer biased to uniform mixture; head learns deviations.
            nn.init.zeros_(self.net[-1].weight)
            nn.init.zeros_(self.net[-1].bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Returns log-mixing-weights log_w of shape (T, C) with log_softmax along C."""
        logits = self.net(features)
        return F.log_softmax(logits, dim=-1)


def mixture_nll(log_w: torch.Tensor, log_p_targets: torch.Tensor) -> torch.Tensor:
    """Mixture negative log-likelihood per position.

        nll[t] = -logsumexp_c(log_w[t, c] + log_p_targets[t, c])

    Args:
        log_w:         (T, C) log mixing weights (each row log-softmaxed)
        log_p_targets: (T, C) log p_c(y_t | ctx_t)
    Returns:
        (T,) float per-position NLL
    """
    assert log_w.shape == log_p_targets.shape, (log_w.shape, log_p_targets.shape)
    return -torch.logsumexp(log_w + log_p_targets, dim=-1)
