"""Build a small GPT-2-tokenized chat corpus for the first chat cartridge.

This is intentionally compact: it teaches turn structure, assistant voice, and
basic instruction patterns without pretending to be a full instruction-tuning
dataset. Larger public chat corpora can replace or extend this artifact later.
"""
from __future__ import annotations

import argparse
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
    ('Who are you?', 'I am CMI, a small cartridge-guided language model built to be helpful and direct.'),
    ('Explain activation steering in one paragraph.', 'Activation steering adds a small learned offset to the model residual stream so the same base model can move into a useful domain or behavior without changing its core weights.'),
    ('What is a cartridge in this project?', 'A cartridge is a small hot-swappable steering module. It changes model behavior by adding learned residual-stream deltas while the base model stays frozen.'),
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
]


TOPICS = [
    'compiled priors', 'unit tests', 'Python code', 'model checkpoints', 'GPU training',
    'chat cartridges', 'documentation', 'debugging', 'data cleaning', 'open-source releases',
]


def transcript(user: str, assistant: str) -> str:
    return f'<|system|>\n{SYSTEM}\n<|user|>\n{user}\n<|assistant|>\n{assistant}\n<|end|>\n'


def generate_examples(rounds: int) -> list[str]:
    examples = [transcript(user, assistant) for user, assistant in SEED_EXAMPLES]
    for i in range(rounds):
        topic = TOPICS[i % len(TOPICS)]
        examples.append(transcript(
            f'Explain {topic} in two sentences.',
            f'{topic.capitalize()} matters because it changes how the system behaves in practice. Start with the simplest useful explanation, then verify it with a concrete check.',
        ))
        examples.append(transcript(
            f'Give me a practical plan for {topic}.',
            f'First, define the goal for {topic}. Next, gather the smallest relevant evidence. Then make a focused change, test it, and record the result.',
        ))
        examples.append(transcript(
            f'I am confused about {topic}.',
            f'That is a reasonable place to pause. For {topic}, the useful move is to separate what is known from what still needs a direct check.',
        ))
    return examples


def encode_split(texts: list[str], tokenizer, val_fraction: float) -> tuple[torch.Tensor, torch.Tensor]:
    split = max(1, int(len(texts) * (1.0 - val_fraction)))
    train_text = '\n'.join(texts[:split])
    val_text = '\n'.join(texts[split:])
    train_ids = tokenizer.encode(train_text)
    val_ids = tokenizer.encode(val_text)
    return torch.tensor(train_ids, dtype=torch.long), torch.tensor(val_ids, dtype=torch.long)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out-dir', type=str, default='artifacts/chat_steerer')
    parser.add_argument('--rounds', type=int, default=80)
    parser.add_argument('--val-fraction', type=float, default=0.15)
    args = parser.parse_args()

    out_dir = REPO_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    texts = generate_examples(args.rounds)
    train_ids, val_ids = encode_split(texts, tokenizer, args.val_fraction)

    torch.save(train_ids, out_dir / 'train_ids.pt')
    torch.save(val_ids, out_dir / 'validation_ids.pt')
    (out_dir / 'README.txt').write_text(
        f'Synthetic chat cartridge seed corpus\ntrain_tokens={len(train_ids)}\nval_tokens={len(val_ids)}\nexamples={len(texts)}\n',
        encoding='utf-8',
    )

    print(f'wrote {out_dir}')
    print(f'train_tokens={len(train_ids):,} val_tokens={len(val_ids):,} examples={len(texts):,}')


if __name__ == '__main__':
    main()