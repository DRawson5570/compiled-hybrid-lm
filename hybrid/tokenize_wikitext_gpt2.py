"""tokenize_wikitext_gpt2.py — Tokenize WikiText-103 with real GPT-2 BPE tokenizer.

Produces:
  artifacts/wikitext_gpt2/train_ids.pt  — tokenized training set
  artifacts/wikitext_gpt2/val_ids.pt    — tokenized validation set
  artifacts/wikitext_gpt2/test_ids.pt   — tokenized test set

Each is a 1D torch.LongTensor of GPT-2 BPE token IDs.
Articles are separated by the EOS token (50256) so document boundaries
are explicit and recoverable.
"""
from __future__ import annotations

import argparse, sys, time
from pathlib import Path

import torch
from transformers import AutoTokenizer
from datasets import load_dataset

OUT_DIR = Path('/home/drawson/deepseek_experiments/artifacts/wikitext_gpt2')
OUT_DIR.mkdir(parents=True, exist_ok=True)


def tokenize_split(dataset, tokenizer, name: str, add_eos: bool = True) -> torch.Tensor:
    """Tokenize all articles in a split, optionally adding EOS between articles."""
    all_ids = []
    eos = tokenizer.eos_token_id
    n_articles = len(dataset)
    t0 = time.time()

    for i, article in enumerate(dataset):
        text = article['text']
        if not text.strip():
            # Empty article (section break) — add EOS only
            if add_eos and all_ids:
                all_ids.append(eos)
            continue

        ids = tokenizer.encode(text)
        if ids:
            all_ids.extend(ids)
            if add_eos:
                all_ids.append(eos)

        if (i + 1) % 50000 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            print(f'  [{name}] {i+1:,}/{n_articles:,} articles '
                  f'({rate:.0f} art/s, {len(all_ids):,} tokens)', flush=True)

    elapsed = time.time() - t0
    ids_t = torch.tensor(all_ids, dtype=torch.long)
    print(f'  [{name}] Done: {len(all_ids):,} tokens in {elapsed:.0f}s '
          f'({len(all_ids)/elapsed:.0f} tok/s)', flush=True)
    return ids_t


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--splits', nargs='+', default=['train', 'validation', 'test'])
    p.add_argument('--out-dir', type=str, default=str(OUT_DIR))
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print('[load] GPT-2 tokenizer...')
    tok = AutoTokenizer.from_pretrained('gpt2')
    print(f'  vocab_size={tok.vocab_size}  eos={tok.eos_token_id}')

    for split in args.splits:
        path = out_dir / f'{split}_ids.pt'
        if path.exists():
            ids_t = torch.load(path, weights_only=False)
            print(f'[{split}] Already cached: {len(ids_t):,} tokens')
            continue

        print(f'[{split}] Loading WikiText-103...')
        ds = load_dataset('wikitext', 'wikitext-103-raw-v1', split=split)
        print(f'[{split}] Tokenizing {len(ds):,} articles...')
        ids_t = tokenize_split(ds, tok, split, add_eos=True)
        torch.save(ids_t, path)
        print(f'[{split}] Saved to {path} ({path.stat().st_size / 1e6:.1f} MB)')
        print()

    # Print summary
    print('=' * 60)
    for split in args.splits:
        path = out_dir / f'{split}_ids.pt'
        if path.exists():
            ids_t = torch.load(path, weights_only=False)
            print(f'  {split:12s}: {len(ids_t):>12,} tokens  '
                  f'unique={ids_t.unique().numel():>6,}')
    print('=' * 60)


if __name__ == '__main__':
    main()
