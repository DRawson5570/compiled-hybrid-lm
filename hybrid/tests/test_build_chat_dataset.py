from __future__ import annotations

import sys
from pathlib import Path

DEEPSEEK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DEEPSEEK.parent))

from hybrid.build_chat_dataset import (
    BASIC_ASSISTANT_EXAMPLES,
    FOCUSED_CHAT_EXAMPLES,
    GREETING_EXAMPLES,
    IDENTITY_AND_FACT_EXAMPLES,
    PRODUCTION_ASSISTANT_EXAMPLES,
    SEED_EXAMPLES,
    build_examples,
    encode_transcript,
    generate_examples,
)
from transformers import AutoTokenizer


def test_generate_examples_exposes_anchor_repeats():
    examples = generate_examples(rounds=2, anchor_repeat=3, focused_repeat=5)

    expected = (
        len(SEED_EXAMPLES)
        + len(BASIC_ASSISTANT_EXAMPLES)
        + len(PRODUCTION_ASSISTANT_EXAMPLES)
        + 3 * len(GREETING_EXAMPLES)
        + 5 * len(FOCUSED_CHAT_EXAMPLES)
        + 5 * len(IDENTITY_AND_FACT_EXAMPLES)
        + 2 * 3
    )
    assert len(examples) == expected
    assert examples.count(GREETING_EXAMPLES[0]) == 3 + 1
    assert examples.count(FOCUSED_CHAT_EXAMPLES[0]) == 5 + 1


def test_build_examples_shuffles_deterministically_without_losing_rows():
    original = build_examples(rounds=3, alpaca_count=0, anchor_repeat=2, focused_repeat=2, shuffle_seed=None)
    shuffled_a = build_examples(rounds=3, alpaca_count=0, anchor_repeat=2, focused_repeat=2, shuffle_seed=123)
    shuffled_b = build_examples(rounds=3, alpaca_count=0, anchor_repeat=2, focused_repeat=2, shuffle_seed=123)

    assert shuffled_a == shuffled_b
    assert shuffled_a != original
    assert sorted(shuffled_a) == sorted(original)


def test_encode_transcript_trains_explicit_assistant_eos():
    tokenizer = AutoTokenizer.from_pretrained('gpt2')

    ids, mask = encode_transcript('Hello', 'Hello. What would you like to work on today?', tokenizer)

    assert ids[-1] == tokenizer.eos_token_id
    assert mask[-1] == 1
    assert mask[0] == 0
