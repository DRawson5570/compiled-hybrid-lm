"""surfaces/provenance.py — Per-token channel attribution tracking.

Part of TICKET-005.  Hooks into the blender forward pass and records which
compiled channels contributed to each token's prediction.  Streaming-safe
via a fixed-size ring buffer.

Acceptance: sum(contributions) ≈ final_logp within 1e-4 per position.
"""
from __future__ import annotations

import json
import math
from collections import deque
from typing import Optional

import numpy as np
import torch


class ProvenanceRing:
    """Fixed-size ring buffer storing per-position channel contributions.

    Each entry: {'position': int, 'target_token': int, 'final_logp': float,
                  'channels': [(name, logp_contribution), ...]}
    """

    def __init__(self, capacity: int = 16384):
        self._buffer: deque[dict] = deque(maxlen=capacity)
        self._position_offset: int = 0

    def set_offset(self, offset: int):
        """Set the global token position of the first entry in the buffer."""
        self._position_offset = offset

    def record(self, global_position: int, target_token: int,
               final_logp: float, channel_contributions: list[tuple[str, float]]):
        """Record provenance for one position."""
        self._buffer.append({
            'position': global_position,
            'target_token': target_token,
            'final_logp': round(final_logp, 6),
            'channels': [(name, round(contrib, 6)) for name, contrib in channel_contributions],
        })

    def provenance(self, token_idx: int, top_k: int = 5) -> list[tuple[str, float]] | None:
        """Return top-k contributing channels for a token position, or None."""
        for entry in self._buffer:
            if entry['position'] == token_idx:
                return sorted(entry['channels'], key=lambda x: -x[1])[:top_k]
        return None

    def to_list(self) -> list[dict]:
        """Export full buffer as list of dicts."""
        return list(self._buffer)

    def dump_json(self, path: str):
        """Write buffer to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_list(), f, indent=2)

    def verify_invariant(self, atol: float = 1e-4) -> bool:
        """Verify logsumexp(all_channel_contributions) ≈ final_logp per position."""
        for entry in self._buffer:
            contribs = [c for _, c in entry['channels']]
            if not contribs:
                continue
            # logsumexp of contributions should equal final_logp
            computed = float(torch.logsumexp(torch.tensor(contribs), dim=0).item())
            if abs(computed - entry['final_logp']) > atol:
                return False
        return True

    def __len__(self) -> int:
        return len(self._buffer)


class ProvenanceBlender:
    """Wraps a compiled blender to record per-position channel provenance.

    For each forward pass, records:
      - Which channels contributed most to the final log-prob
      - Their individual log-prob contributions
      - The final blended log-prob for the true target token
    """

    def __init__(self, blender, channel_names: list[str],
                 log_p_targets: torch.Tensor,
                 provenance_ring: ProvenanceRing,
                 position_offset: int = 0):
        """
        Args:
            blender: model with forward(features) -> log_w (T, C) mixing weights
            channel_names: list of channel name strings
            log_p_targets: (T, C) per-channel log-prob for TRUE target tokens
            provenance_ring: ring buffer for recording
            position_offset: global token offset for this slice
        """
        self.blender = blender
        self.channel_names = channel_names
        self.log_p_targets = log_p_targets
        self.provenance = provenance_ring
        self.position_offset = position_offset
        self.provenance.set_offset(position_offset)

    @torch.no_grad()
    def forward(self, features: torch.Tensor, targets: torch.Tensor,
                is_already_windowed: bool = False) -> torch.Tensor:
        """Forward pass with provenance recording.

        Args:
            features: (T, F) per-position feature vectors
            targets: (T,) true next-token IDs
            is_already_windowed: passed through to blender

        Returns:
            log_w: (T, C) log-softmax mixing weights
        """
        if is_already_windowed:
            log_w = self.blender(features, is_already_windowed=True)
        else:
            win = self.blender.build_windowed_features(features)
            log_w = self.blender(win, is_already_windowed=True)

        T = log_w.shape[0]
        C = log_w.shape[1]

        # Compute per-channel contribution to final log-prob
        # contribution[t, c] = log_w[t, c] + log_p_targets[t, c]
        contributions = log_w + self.log_p_targets[:T].to(log_w.device)

        # Final blended log-prob: logsumexp over channels
        final_logp = torch.logsumexp(contributions, dim=-1)  # (T,)

        # Record provenance for each position
        contrib_np = contributions.cpu().numpy()
        final_np = final_logp.cpu().numpy()
        target_np = targets[:T].cpu().numpy()

        for t in range(T):
            global_pos = self.position_offset + t

            channel_contribs = contrib_np[t]  # (C,)

            # Record ALL channels with their log-prob contributions
            recorded = []
            for ci in range(C):
                recorded.append((self.channel_names[ci], float(channel_contribs[ci])))

            self.provenance.record(
                global_position=global_pos,
                target_token=int(target_np[t]),
                final_logp=float(final_np[t]),
                channel_contributions=recorded,
            )

        return log_w

    def provenance(self, token_idx: int) -> list[tuple[str, float]] | None:
        """Query provenance for a specific token position."""
        return self.provenance.provenance(token_idx)


def verify_provenance_invariant(blender_wrapper, eval_features, eval_targets,
                                max_tokens: int = 200, atol: float = 1e-4) -> bool:
    """Run a 200-token decode and verify sum(contributions) ≈ final_logp."""
    ring = ProvenanceRing(capacity=max_tokens * 2)
    wrapper = ProvenanceBlender(
        blender_wrapper.blender,
        blender_wrapper.channel_names,
        blender_wrapper.log_p_targets,
        ring,
        position_offset=0,
    )

    # Run forward pass on a slice
    feat_slice = eval_features[:max_tokens]
    tgt_slice = eval_targets[:max_tokens]
    wrapper.forward(feat_slice, tgt_slice, is_already_windowed=False)

    # Verify invariant
    ok = ring.verify_invariant(top_k=5, atol=atol)
    if not ok:
        print('FAIL: provenance sum does not match final_logp')
        return False

    print(f'PASS: provenance invariant holds on {len(ring)} positions (atol={atol})')
    return True
