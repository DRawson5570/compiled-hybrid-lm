"""Build a small GPT-2-tokenized chat corpus for the first chat cartridge.

This is intentionally compact: it teaches turn structure, assistant voice, and
basic instruction patterns without pretending to be a full instruction-tuning
dataset. Larger public chat corpora can replace or extend this artifact later.
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import torch
from transformers import AutoTokenizer


SYSTEM = (
    'You are CMI, a concise and helpful assistant. Answer directly, keep the '
    'conversation coherent, and ask a brief clarifying question when needed.'
)


SEED_EXAMPLES: list[tuple[str, str]] = [
    ('Hello!', 'Hello. What would you like to work on today?'),
    ('Hello', 'Hello. What would you like to work on today?'),
    ('Hi', 'Hi. What would you like to work on today?'),
    ('Hey there', 'Hello. I am ready to help.'),
    ('Who are you?', 'I am CMI, a small cartridge-guided language model built to be helpful and direct.'),
    ('What are you?', 'I am CMI, a small cartridge-guided language model built to be helpful and direct.'),
    ('Explain activation steering in one paragraph.', 'Activation steering adds a small learned offset to the model residual stream so the same base model can move into a useful domain or behavior without changing its core weights.'),
    ('What is a cartridge in this project?', 'A cartridge is a small hot-swappable steering module. It changes model behavior by adding learned residual-stream deltas while the base model stays frozen.'),
    ('What is a chat cartridge?', 'A chat cartridge is a task capability cartridge that steers the frozen base model toward assistant-style responses. It can be loaded beside the general superposition steerer without merging weights into the base model.'),
    ('Explain what a chat cartridge is in two sentences.', 'A chat cartridge is a small task-specific adapter that nudges the frozen language model toward conversational behavior. It is separate from the general superposition steerer, so it can be loaded, disabled, or replaced independently.'),
    ('Give me three next steps for debugging a failing test.', 'First, reproduce the failure with the smallest command. Second, read the assertion and nearby code. Third, make one targeted change and rerun the focused test.'),
    ('Write a polite short email asking for a meeting.', 'Subject: Meeting Request\n\nHi,\n\nWould you be available for a brief meeting this week to discuss the next steps?\n\nBest,\n'),
    ('Summarize: The model improved eval_s from 31.3 to 29.4 while eval_b stayed near 37.9.', 'The active steered model crossed below 30 perplexity, while the blind base stayed near 37.9. That shows a strong cartridge effect.'),
    ('What should I do if a number looks wrong?', 'Pause and sanity-check it before reporting. Compare it to expected ranges, inspect the data path, and verify the calculation.'),
    ('Explain overfitting simply.', 'Overfitting happens when a model memorizes training examples too closely and performs worse on new examples.'),
    ('What is the difference between eval_s and eval_b?', 'eval_s measures the model with steering active. eval_b measures the same model with steering disabled, so it shows the blind base behavior.'),
    ('Make this clearer: the run is not done but looks good.', 'The run is still in progress, but the trend is healthy so far.'),
    ('List two risks in a deployment plan.', 'Two risks are configuration drift between environments and missing rollback steps if the deployment fails.'),
    ('How do I train a capability cartridge?', 'Freeze the base model, mount the compatible cartridge, train only the cartridge weights on task data, and validate the steered and blind paths separately.'),
    ('Answer in one sentence: why keep the base frozen?', 'Keeping the base frozen lets many small cartridges specialize one shared model without retraining or duplicating it.'),
    ('What is a good response when you are unsure?', 'State what you know, say what is uncertain, and suggest the next check that would resolve it.'),
    ('Turn this into a checklist: inspect, edit, test, commit.', '- Inspect the relevant code.\n- Make the focused edit.\n- Run the tests.\n- Commit the finished change.'),
    ('Explain why generated artifacts should not be committed.', 'Generated artifacts are usually large, reproducible, and machine-specific. Keeping them out of Git makes the repository smaller and cleaner.'),
    ('What is the next step after a sub-30 eval_s?', 'Package the checkpoint, build a chat runtime, and train a chat capability cartridge on instruction-style turns.'),
    ('Give a calm status update.', 'The core change is implemented and the focused tests pass. I am checking the broader suite now.'),
    ('What does hot-swappable mean here?', 'It means a cartridge can be loaded, disabled, replaced, or reweighted at runtime without merging weights into the base model.'),
    ('I am confused about superposition steering.', 'That is a reasonable place to pause. For superposition steering, the useful move is to separate what is known from what still needs a direct check.'),
    ('Give me three practical next steps for testing this model.', 'First, define the behavior you expect. Second, run fixed prompts in base, superposition, and chat modes. Third, save the outputs and compare them before changing the cartridge.'),
]


TOPICS = [
    'compiled priors', 'unit tests', 'Python code', 'model checkpoints', 'GPU training',
    'chat cartridges', 'documentation', 'debugging', 'data cleaning', 'open-source releases',
]


GREETING_EXAMPLES = [
    ('Hello', 'Hello. What would you like to work on today?'),
    ('Hello!', 'Hello. What would you like to work on today?'),
    ('Hi', 'Hi. What would you like to work on today?'),
    ('Hey', 'Hello. I am ready to help.'),
]


FOCUSED_CHAT_EXAMPLES = [
    ('What is a chat cartridge?', 'A chat cartridge is a task capability cartridge that steers the frozen base model toward assistant-style responses. It can be loaded beside the general superposition steerer without merging weights into the base model.'),
    ('Explain what a chat cartridge is in two sentences.', 'A chat cartridge is a small task-specific adapter that nudges the frozen language model toward conversational behavior. It is separate from the general superposition steerer, so it can be loaded, disabled, or replaced independently.'),
    ('Give me three practical next steps for testing this model.', 'First, define the behavior you expect. Second, run fixed prompts in base, superposition, and chat modes. Third, save the outputs and compare them before changing the cartridge.'),
    ('I am confused about superposition steering.', 'That is a reasonable place to pause. For superposition steering, the useful move is to separate what is known from what still needs a direct check.'),
]


def encode_transcript(user: str, assistant: str, tokenizer) -> tuple[list[int], list[int]]:
    prefix = f'System:\n{SYSTEM}\n\nUser:\n{user}\n\nAssistant:\n'
    response = f'{assistant}\n\n'
    prefix_ids = tokenizer.encode(prefix)
    response_ids = tokenizer.encode(response)
    ids = prefix_ids + response_ids
    # Mask predicts the current token. The trainer shifts it by one so loss is
    # charged only for assistant response tokens, not system/user scaffolding.
    mask = [0] * len(prefix_ids) + [1] * len(response_ids)
    return ids, mask


def generate_examples(rounds: int, anchor_repeat: int = 24, focused_repeat: int = 24) -> list[tuple[str, str]]:
    examples = list(SEED_EXAMPLES)
    for _ in range(anchor_repeat):
        examples.extend(GREETING_EXAMPLES)
    for _ in range(focused_repeat):
        examples.extend(FOCUSED_CHAT_EXAMPLES)
    for i in range(rounds):
        topic = TOPICS[i % len(TOPICS)]
        examples.append((
            f'Explain {topic} in two sentences.',
            f'{topic.capitalize()} matters because it changes how the system behaves in practice. Start with the simplest useful explanation, then verify it with a concrete check.',
        ))
        examples.append((
            f'Give me a practical plan for {topic}.',
            f'First, define the goal for {topic}. Next, gather the smallest relevant evidence. Then make a focused change, test it, and record the result.',
        ))
        examples.append((
            f'I am confused about {topic}.',
            f'That is a reasonable place to pause. For {topic}, the useful move is to separate what is known from what still needs a direct check.',
        ))
    return examples


def build_examples(rounds: int, alpaca_count: int, anchor_repeat: int,
                   focused_repeat: int, shuffle_seed: int | None) -> list[tuple[str, str]]:
    examples = generate_examples(rounds, anchor_repeat, focused_repeat)
    examples.extend(load_alpaca_examples(alpaca_count))
    if shuffle_seed is not None:
        random.Random(shuffle_seed).shuffle(examples)
    return examples


def load_alpaca_examples(count: int) -> list[tuple[str, str]]:
    if count <= 0:
        return []
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit('Install datasets or run with --alpaca-count 0') from exc

    dataset = load_dataset('yahma/alpaca-cleaned', split=f'train[:{count}]')
    examples: list[tuple[str, str]] = []
    for row in dataset:
        instruction = str(row.get('instruction') or '').strip()
        extra_input = str(row.get('input') or '').strip()
        response = str(row.get('output') or '').strip()
        if not instruction or not response:
            continue
        user = f'{instruction}\n\n{extra_input}' if extra_input else instruction
        examples.append((user, response))
    return examples


def encode_split(examples: list[tuple[str, str]], tokenizer, val_fraction: float):
    split = max(1, int(len(examples) * (1.0 - val_fraction)))

    def encode_many(items: list[tuple[str, str]]):
        ids: list[int] = []
        mask: list[int] = []
        for user, assistant in items:
            item_ids, item_mask = encode_transcript(user, assistant, tokenizer)
            ids.extend(item_ids)
            mask.extend(item_mask)
        return torch.tensor(ids, dtype=torch.long), torch.tensor(mask, dtype=torch.float32)

    train_ids, train_mask = encode_many(examples[:split])
    val_ids, val_mask = encode_many(examples[split:])
    return train_ids, train_mask, val_ids, val_mask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out-dir', type=str, default='artifacts/chat_steerer')
    parser.add_argument('--rounds', type=int, default=80)
    parser.add_argument('--alpaca-count', type=int, default=0)
    parser.add_argument('--anchor-repeat', type=int, default=24)
    parser.add_argument('--focused-repeat', type=int, default=24)
    parser.add_argument('--shuffle-seed', type=int, default=None)
    parser.add_argument('--val-fraction', type=float, default=0.15)
    args = parser.parse_args()

    out_dir = REPO_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    examples = build_examples(
        rounds=args.rounds,
        alpaca_count=args.alpaca_count,
        anchor_repeat=args.anchor_repeat,
        focused_repeat=args.focused_repeat,
        shuffle_seed=args.shuffle_seed,
    )
    train_ids, train_mask, val_ids, val_mask = encode_split(examples, tokenizer, args.val_fraction)

    torch.save(train_ids, out_dir / 'train_ids.pt')
    torch.save(train_mask, out_dir / 'train_loss_mask.pt')
    torch.save(val_ids, out_dir / 'validation_ids.pt')
    torch.save(val_mask, out_dir / 'validation_loss_mask.pt')
    (out_dir / 'README.txt').write_text(
        f'Chat cartridge seed corpus\ntrain_tokens={len(train_ids)}\nval_tokens={len(val_ids)}\nexamples={len(examples)}\nsynthetic_rounds={args.rounds}\nalpaca_count={args.alpaca_count}\nanchor_repeat={args.anchor_repeat}\nfocused_repeat={args.focused_repeat}\nshuffle_seed={args.shuffle_seed}\nassistant_loss_only=1\n',
        encoding='utf-8',
    )

    print(f'wrote {out_dir}')
    print(f'train_tokens={len(train_ids):,} val_tokens={len(val_ids):,} examples={len(examples):,}')
    print(f'train_loss_tokens={int(train_mask.sum().item()):,} val_loss_tokens={int(val_mask.sum().item()):,}')


if __name__ == '__main__':
    main()