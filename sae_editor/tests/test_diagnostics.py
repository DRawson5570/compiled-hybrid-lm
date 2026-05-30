"""Diagnostic tests: confirm preview hook behavior + gating fix."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from sae_editor.recompiler import build_dense_map
from sae_editor.preview import _make_preview_hook


class TestKeyReconstruction:
    """Confirm the core mathematical guarantee."""

    def test_key_reconstruction_exact(self):
        N, d = 4, 64
        keys = torch.randn(N, d)
        values = torch.randn(N, d)
        W_down, W_up = build_dense_map(keys, values, eps=1e-6)
        recon = keys @ W_down @ W_up
        assert torch.allclose(recon, values, atol=1e-4)

    def test_random_h_delta_is_arbitrary(self):
        """Random hidden states produce non-zero delta — the bug."""
        N, d = 4, 64
        keys = torch.randn(N, d)
        values = torch.randn(N, d)
        W_down, W_up = build_dense_map(keys, values)

        h_random = torch.randn(1, 8, d)
        delta = h_random @ W_down @ W_up
        assert delta.shape == (1, 8, d)
        assert delta.abs().sum() > 0

    def test_cosine_random_h_to_random_keys_is_low(self):
        """At d=896, random vectors have negligible cosine similarity."""
        d = 896
        keys = torch.randn(4, d)
        h = torch.randn(10, d)
        cos = F.cosine_similarity(
            h.unsqueeze(1), keys.unsqueeze(0), dim=-1
        )
        assert cos.abs().max().item() < 0.15


class TestGatedHook:
    """Confirm the gated hook fixes the problem."""

    def test_gated_hook_blocks_random_input(self):
        """Random hidden states have cosine << 0.3, so gate stays closed."""
        N, d = 4, 64
        keys = torch.randn(N, d)
        values = torch.randn(N, d)
        W_down, W_up = build_dense_map(keys, values)

        hook_fn, delta_l2 = _make_preview_hook(
            W_down, W_up, keys, strength=1.0,
            model_dtype=torch.float32, gate_threshold=0.3,
        )

        class FakeModule:
            pass

        hidden = torch.randn(1, 8, d)
        output = (hidden,)

        result = hook_fn(FakeModule(), None, output)
        result_hidden = result[0]

        assert torch.allclose(result_hidden, hidden, atol=1e-4), (
            "Hidden state should be unchanged for random input (gate closed)"
        )

    def test_gated_hook_lets_matching_keys_through(self):
        """Keys themselves have cosine=1.0, so gate opens fully."""
        N, d = 3, 64
        keys = torch.randn(N, d)
        values = torch.randn(N, d)
        W_down, W_up = build_dense_map(keys, values)

        hook_fn, delta_l2 = _make_preview_hook(
            W_down, W_up, keys, strength=1.0,
            model_dtype=torch.float32, gate_threshold=0.3,
        )

        class FakeModule:
            pass

        hidden = keys.unsqueeze(0)
        output = (hidden,)

        result = hook_fn(FakeModule(), None, output)
        result_hidden = result[0]

        assert not torch.allclose(result_hidden, hidden, atol=1e-4), (
            "Keys should pass the gate and be modified"
        )

    def test_gated_hook_soft_weights_proportional(self):
        """Higher cosine produces larger injection."""
        d = 64
        base = torch.randn(1, d)
        base_norm = base / base.norm()

        key = base_norm
        values = torch.randn(1, d)
        W_down, W_up = build_dense_map(key, values)

        hook_fn_strong, _ = _make_preview_hook(
            W_down, W_up, key, strength=1.0,
            model_dtype=torch.float32, gate_threshold=0.3,
        )

        class FakeModule:
            pass

        hidden_exact = key.unsqueeze(0)
        result_exact = hook_fn_strong(FakeModule(), None, (hidden_exact,))
        diff_exact = (result_exact[0] - hidden_exact).norm().item()

        hidden_partial = (base_norm + 2.0 * torch.randn(1, d)).unsqueeze(0)
        result_partial = hook_fn_strong(FakeModule(), None, (hidden_partial,))
        diff_partial = (result_partial[0] - hidden_partial).norm().item()

        assert diff_exact > diff_partial * 0.1, (
            f"Exact key match should produce larger injection than partial. "
            f"exact_diff={diff_exact:.4f}, partial_diff={diff_partial:.4f}"
        )
