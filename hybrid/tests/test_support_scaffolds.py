from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from hybrid.calibration import brier_score, expected_calibration_error, find_best_temperature
from hybrid.data.multi_corpus import CorpusStream, build_weighted_schedule, iter_mixed_chunks, materialize_mixed_tokens
from hybrid.decoding import DecodingConfig, deterministic_generate


class ToyNextTokenModel(nn.Module):
    def __init__(self, vocab: int):
        super().__init__()
        self.vocab = vocab

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        batch, seq_len = input_ids.shape
        logits = torch.zeros(batch, seq_len, self.vocab)
        next_ids = (input_ids + 1) % self.vocab
        logits.scatter_(2, next_ids.unsqueeze(-1), 5.0)
        return logits


def test_calibration_metrics_and_temperature_search():
    logits = torch.tensor([
        [4.0, 1.0, 0.0],
        [0.0, 3.0, 0.5],
        [0.5, 0.0, 2.5],
        [3.0, 1.0, 0.0],
    ])
    targets = torch.tensor([0, 1, 2, 1])
    ece = expected_calibration_error(logits, targets, n_bins=5)
    brier = brier_score(logits, targets)
    result = find_best_temperature(logits, targets, candidates=torch.tensor([0.5, 1.0, 2.0]))
    assert 0.0 <= ece.item() <= 1.0
    assert brier.item() >= 0.0
    assert result.temperature in {0.5, 1.0, 2.0}
    assert result.nll > 0.0


def test_deterministic_greedy_generation():
    model = ToyNextTokenModel(vocab=10)
    prompt = torch.tensor([[1, 2, 3]])
    cfg = DecodingConfig(max_new_tokens=4, temperature=0.0, seed=99)
    a = deterministic_generate(model, prompt, cfg)
    b = deterministic_generate(model, prompt, cfg)
    assert torch.equal(a, b)
    assert a.tolist() == [[1, 2, 3, 4, 5, 6, 7]]


def test_multi_corpus_mixer_is_weighted_and_reproducible():
    streams = [
        CorpusStream("wiki", torch.tensor([1, 2, 3]), weight=2),
        CorpusStream("code", torch.tensor([10, 11]), weight=1),
    ]
    schedule = build_weighted_schedule(streams, seed=7)
    assert len(schedule) == 3
    assert schedule.count(0) == 2
    assert schedule.count(1) == 1

    a = materialize_mixed_tokens(streams, n_tokens=12, seed=7)
    b = materialize_mixed_tokens(streams, n_tokens=12, seed=7)
    assert torch.equal(a, b)

    chunker = iter_mixed_chunks(streams, seq_len=5, seed=7)
    chunk = next(chunker)
    assert chunk.input_ids.shape == (5,)
    assert chunk.target_ids.shape == (5,)
    assert len(chunk.source_names) == 5
    assert torch.equal(chunk.input_ids[1:], chunk.target_ids[:-1])
