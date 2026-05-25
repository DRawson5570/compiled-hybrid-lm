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
    INSTRUCTION_FOLLOWING_EXAMPLES,
    MULTITURN_EXAMPLES,
    PRODUCTION_ASSISTANT_EXAMPLES,
    SEED_EXAMPLES,
    build_examples,
    encode_dialogue,
    encode_split,
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
        + len(INSTRUCTION_FOLLOWING_EXAMPLES)
        + len(MULTITURN_EXAMPLES)
        + 3 * len(GREETING_EXAMPLES)
        + 5 * len(FOCUSED_CHAT_EXAMPLES)
        + 5 * len(IDENTITY_AND_FACT_EXAMPLES)
        + 5 * len(INSTRUCTION_FOLLOWING_EXAMPLES)
        + 5 * len(MULTITURN_EXAMPLES)
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
    assert sorted(shuffled_a, key=repr) == sorted(original, key=repr)


def test_encode_transcript_trains_explicit_assistant_eos():
    tokenizer = AutoTokenizer.from_pretrained('gpt2')

    ids, mask = encode_transcript('Hello', 'Hello. What would you like to work on today?', tokenizer)

    assert ids[-1] == tokenizer.eos_token_id
    assert mask[-1] == 1
    assert mask[0] == 0


def test_encode_dialogue_trains_each_assistant_turn_eos():
    tokenizer = AutoTokenizer.from_pretrained('gpt2')

    ids, mask = encode_dialogue((
        ('What is the capital of France?', 'The capital of France is Paris.'),
        ('Germany?', 'If you mean the capital of Germany, it is Berlin.'),
    ), tokenizer)

    assert ids.count(tokenizer.eos_token_id) == 2
    eos_positions = [idx for idx, token_id in enumerate(ids) if token_id == tokenizer.eos_token_id]
    assert all(mask[idx] == 1 for idx in eos_positions)
    assert mask[0] == 0


def test_encode_split_preserves_example_boundaries():
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    examples = [
        ('Hello', 'Hello. What would you like to work on today?'),
        ('Tell me a story.', 'Once there was a little robot that fixed radios.'),
    ]

    train_ids, train_mask, val_ids, val_mask, train_examples, val_examples = encode_split(
        examples,
        tokenizer,
        val_fraction=0.5,
    )

    assert len(train_examples) == 1
    assert len(val_examples) == 1
    assert len(train_ids) == len(train_examples[0]['ids'])
    assert len(val_ids) == len(val_examples[0]['ids'])
    assert int(train_mask.sum().item()) == int(train_examples[0]['mask'].sum().item())
    assert int(val_mask.sum().item()) == int(val_examples[0]['mask'].sum().item())
