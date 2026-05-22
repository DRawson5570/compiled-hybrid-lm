"""split_corpus_by_article.py — Document-disjoint wikitext corpus splitter.

Uses a deterministic RNG seed to shuffle the original articles of WikiText-103 (or WT-2),
partitioning them into train/val/eval splits of articles so there is zero article overlap
between splits (i.e. document-disjoint splits).

Outputs token-id tensors as .pt files and writes split_manifest.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer


def get_file_sha256(filepath: Path) -> str:
    """Computes SHA-256 hash of a file on disk."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def get_tokenizer_hash(tokenizer) -> str:
    """Computes a SHA-256 hash representing the tokenizer's vocabulary."""
    try:
        vocab_dict = tokenizer.get_vocab()
        vocab_str = json.dumps(vocab_dict, sort_keys=True)
        return hashlib.sha256(vocab_str.encode("utf-8")).hexdigest()
    except Exception:
        return "unknown"


def ensure_corpus_on_disk(path: Path) -> None:
    """Bootstraps the raw combined wikitext-103 if not present on disk."""
    if path.exists():
        return
    print(f"[bootstrap] Corpus file {path} not found.")
    print("Downloading 'wikitext-103-raw-v1' from HuggingFace to bootstrap this file...")
    path.parent.mkdir(parents=True, exist_ok=True)
    from datasets import load_dataset
    with open(path, "w", encoding="utf-8") as f:
        for split in ["train", "validation", "test"]:
            print(f"  Streaming {split} split raw rows...")
            ds = load_dataset("wikitext", "wikitext-103-raw-v1", split=split)
            for row in ds:
                # Keep exact spacing and newlines
                f.write(row["text"])
    print(f"[bootstrap] Combined WikiText-103 corpus saved to {path}")


def parse_articles(lines: list[str]) -> list[list[str]]:
    """Groups raw lines into articles based on standard WikiText boundaries."""
    articles = []
    current_article = []
    
    for line in lines:
        stripped = line.strip()
        # Main headers: match '= Title =' but not sub-headers like '= = Subheader = =' or deeper
        if stripped.startswith("= ") and stripped.endswith(" =") and not stripped.startswith("= = "):
            if current_article:
                articles.append(current_article)
            current_article = [line]
        else:
            if current_article:
                current_article.append(line)
            else:
                if stripped:
                    current_article = [line]
                    
    if current_article:
        articles.append(current_article)
        
    return articles


def get_ranges(ids: list[int]) -> list[list[int]]:
    """Converts a sorted list of integer IDs into contiguous inclusive ranges [start, end]."""
    if not ids:
        return []
    sorted_ids = sorted(ids)
    ranges = []
    start = sorted_ids[0]
    prev = sorted_ids[0]
    for x in sorted_ids[1:]:
        if x == prev + 1:
            prev = x
        else:
            ranges.append([start, prev])
            start = x
            prev = x
    ranges.append([start, prev])
    return ranges


def main() -> None:
    parser = argparse.ArgumentParser(description="Deterministic document-disjoint corpus splitter")
    parser.add_argument("--corpus", type=str, required=True, help="Path to raw WikiText text file")
    parser.add_argument("--out", type=str, required=True, help="Output directory folder")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for article-shuffling")
    parser.add_argument("--ratios", type=float, nargs=3, default=[0.90, 0.05, 0.05],
                        help="Train, Validation, and Evaluation splits ratios")
    parser.add_argument("--tokenizer", type=str, default="gpt2", help="Pretrained tokenizer to use")
    args = parser.parse_args()

    # Validate ratios
    if abs(sum(args.ratios) - 1.0) > 1e-6:
        print(f"Error: ratios {args.ratios} must sum to 1.0", file=sys.stderr)
        sys.exit(1)

    corpus_path = Path(args.corpus)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Ensure/load the corpus file
    ensure_corpus_on_disk(corpus_path)
    corpus_sha = get_file_sha256(corpus_path)
    print(f"Corpus SHA-256: {corpus_sha}")

    # Read lines of the corpus
    with open(corpus_path, "r", encoding="utf-8") as f:
        all_lines = f.readlines()

    # 2. Parse articles from lines
    print(f"Parsing articles from {len(all_lines):,} raw lines...")
    articles = parse_articles(all_lines)
    n_articles = len(articles)
    print(f"Successfully identified {n_articles:,} articles.")

    if n_articles < 3:
        print(f"Error: Too few articles found ({n_articles}) to perform disjoint splitting.", file=sys.stderr)
        sys.exit(1)

    # 3. Deterministically partition articles using the seed
    indices = list(range(n_articles))
    rng = random.Random(args.seed)
    rng.shuffle(indices)

    ratio_train, ratio_val, ratio_eval = args.ratios
    n_train = int(round(n_articles * ratio_train))
    n_val = int(round(n_articles * ratio_val))
    n_eval = n_articles - n_train - n_val

    train_indices = sorted(indices[:n_train])
    val_indices = sorted(indices[n_train:n_train+n_val])
    eval_indices = sorted(indices[n_train+n_val:])

    print(f"Split distribution (seed={args.seed}):")
    print(f"  Train:      {n_train:,} articles ({ratio_train:.1%})")
    print(f"  Validation: {n_val:,} articles ({ratio_val:.1%})")
    print(f"  Evaluation: {n_eval:,} articles ({ratio_eval:.1%})")

    # 4. Tokenize each split
    print(f"Loading '{args.tokenizer}' tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    tok_hash = get_tokenizer_hash(tokenizer)
    eos_id = tokenizer.eos_token_id if hasattr(tokenizer, "eos_token_id") else None

    # Helper function to tokenize a split list of article indices
    def tokenize_and_build(split_indices: list[int], name: str) -> tuple[torch.Tensor, int]:
        all_ids = []
        t0 = time.time()
        for count, idx in enumerate(split_indices):
            # Each article is a list of lines
            article_lines = articles[idx]
            text = "".join(article_lines)
            ids = tokenizer.encode(text)
            if ids:
                all_ids.extend(ids)
                if eos_id is not None:
                    all_ids.append(eos_id)
            if (count + 1) % 5000 == 0:
                print(f"  [{name}] {count+1:,}/{len(split_indices):,} articles processed...")
        
        ids_tensor = torch.tensor(all_ids, dtype=torch.long)
        elapsed = time.time() - t0
        print(f"  [{name}] Done in {elapsed:.1f}s — {len(ids_tensor):,} tokens")
        return ids_tensor, len(ids_tensor)

    print("Tokenizing train split...")
    train_t, n_train_tokens = tokenize_and_build(train_indices, "train")
    print("Tokenizing validation split...")
    val_t, n_val_tokens = tokenize_and_build(val_indices, "val")
    print("Tokenizing evaluation split...")
    eval_t, n_eval_tokens = tokenize_and_build(eval_indices, "eval")

    # 5. Save .pt files
    train_path = out_dir / "train_ids.pt"
    val_path = out_dir / "val_ids.pt"
    eval_path = out_dir / "eval_ids.pt"

    torch.save(train_t, train_path)
    torch.save(val_t, val_path)
    torch.save(eval_t, eval_path)

    print(f"Saved tensor splits to:")
    print(f"  {train_path} ({train_path.stat().st_size / 1e6:.2f} MB)")
    print(f"  {val_path} ({val_path.stat().st_size / 1e6:.2f} MB)")
    print(f"  {eval_path} ({eval_path.stat().st_size / 1e6:.2f} MB)")

    # 6. Save split_manifest.json
    manifest = {
        "corpus_sha256": corpus_sha,
        "seed": args.seed,
        "tokenizer": args.tokenizer,
        "tokenizer_hash": tok_hash,
        "n_articles_per_split": {
            "train": n_train,
            "validation": n_val,
            "evaluation": n_eval
        },
        "n_tokens_per_split": {
            "train": n_train_tokens,
            "validation": n_val_tokens,
            "evaluation": n_eval_tokens
        },
        "article_id_ranges": {
            "train": get_ranges(train_indices),
            "validation": get_ranges(val_indices),
            "evaluation": get_ranges(eval_indices)
        }
    }

    manifest_path = out_dir / "split_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Saved split manifest to: {manifest_path}")


if __name__ == "__main__":
    main()
