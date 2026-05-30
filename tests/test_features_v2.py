"""Unit tests for hybrid/v1_blender/features_v2.py."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from hybrid.v1_blender.features_v2 import (
    _causal_running_mean, build_feature_matrix_v2,
)


def test_causal_running_mean_simple():
    x = torch.tensor([[1.0], [2.0], [3.0], [4.0], [5.0]])
    m = _causal_running_mean(x, k=3)
    # t=0: mean([1]) = 1
    # t=1: mean([1,2]) = 1.5
    # t=2: mean([1,2,3]) = 2
    # t=3: mean([2,3,4]) = 3
    # t=4: mean([3,4,5]) = 4
    expected = torch.tensor([[1.0], [1.5], [2.0], [3.0], [4.0]])
    assert torch.allclose(m, expected), (m, expected)


def test_causal_running_mean_k_geq_T():
    x = torch.tensor([[1.0], [2.0], [3.0]])
    m = _causal_running_mean(x, k=10)
    expected = torch.tensor([[1.0], [1.5], [2.0]])
    assert torch.allclose(m, expected)


def test_causal_running_mean_no_leak():
    # changing row t must not change rows 0..t-1
    torch.manual_seed(0)
    x = torch.randn(20, 4)
    m0 = _causal_running_mean(x, k=5).clone()
    x[10] += 100.0
    m1 = _causal_running_mean(x, k=5)
    assert torch.allclose(m0[:10], m1[:10]), "leak: past rows changed when row 10 changed"
    assert not torch.allclose(m0[10:], m1[10:]), "future rows didn't update"


def test_build_feature_matrix_v2_shape_and_no_leak():
    T, C, V, d, K = 50, 12, 100, 8, 3
    torch.manual_seed(0)
    log_p_observed = torch.randn(T, C)
    log_p_lag1 = torch.randn(T, C)
    entropy = torch.rand(T, C)
    max_log_prob = -torch.rand(T, C)
    emb = torch.randn(V, d)
    observed_ids = torch.randint(0, V, (T,))
    topk = torch.randn(T, C, K)

    feats = build_feature_matrix_v2(
        log_p_observed, log_p_lag1, entropy, max_log_prob,
        emb, observed_ids, topk_log_probs=topk, use_embedding=True,
        win_mean=8, win_won=16,
    )
    # 7 C-aligned blocks + C*K topk + d embed
    expected_F = 7 * C + C * K + d
    assert feats.shape == (T, expected_F), (feats.shape, expected_F)

    # No-target-leak smoke check: changing future rows of log_p_observed must
    # not change feature row 0.
    feats0 = feats[0].clone()
    log_p_observed[10:] += 1000.0
    feats_new = build_feature_matrix_v2(
        log_p_observed, log_p_lag1, entropy, max_log_prob,
        emb, observed_ids, topk_log_probs=topk, use_embedding=True,
        win_mean=8, win_won=16,
    )
    assert torch.allclose(feats0, feats_new[0]), "feature row 0 leaked from future"


def test_won_freq_in_unit_interval():
    T, C = 100, 5
    torch.manual_seed(1)
    log_p_observed = torch.randn(T, C)
    feats = build_feature_matrix_v2(
        log_p_observed, log_p_observed.clone(), torch.rand(T, C),
        -torch.rand(T, C), torch.zeros(1, 1), torch.zeros(T, dtype=torch.long),
        use_embedding=False, win_mean=8, win_won=16,
    )
    # won_freq is the 7th C-block (index 6) of 7 stat blocks
    won = feats[:, 6 * C : 7 * C]
    assert (won >= 0).all() and (won <= 1).all(), (won.min().item(), won.max().item())
    # rows where past window contains anything should sum to 1; row 0 sums to 0
    assert won[0].sum().item() == 0.0
    for t in range(1, T):
        assert abs(won[t].sum().item() - 1.0) < 1e-5, (t, won[t].sum().item())


if __name__ == "__main__":
    test_causal_running_mean_simple()
    test_causal_running_mean_k_geq_T()
    test_causal_running_mean_no_leak()
    test_build_feature_matrix_v2_shape_and_no_leak()
    test_won_freq_in_unit_interval()
    print("OK: 5/5 features_v2 tests passed")
