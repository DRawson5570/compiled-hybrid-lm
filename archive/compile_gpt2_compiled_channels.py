"""Compile GPT-2 n-gram/skip channel artifacts without training.

This is the CPU/RAM-heavy preflight step for the compiled-feature transformer
path. It turns tokenized WikiText GPT-2 train IDs into a reusable compiled
channel artifact and writes a small profiling report.
"""
from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from hybrid.compiled_features import GPT2CompiledChannelBuilder, GPT2CompiledChannelConfig


def peak_rss_gib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ids", type=Path, default=Path("artifacts/wikitext_gpt2/train_ids.pt"))
    parser.add_argument("--out", type=Path, default=Path("artifacts/compiled_feature_gpt2/compiled_ngram_channels.pt"))
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--max-train-tokens", type=int, default=0)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--recency-window", type=int, default=128)
    args = parser.parse_args()

    if not args.ids.exists():
        raise FileNotFoundError(f"missing token file: {args.ids}")

    report_path = args.report or args.out.with_suffix(".report.json")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    started = time.time()
    print(f"[load] {args.ids}", flush=True)
    ids = torch.load(args.ids, weights_only=False).long().cpu()
    token_count = int(ids.numel())
    if args.max_train_tokens > 0:
        print(f"[slice] first {args.max_train_tokens:,} / {token_count:,} tokens", flush=True)
    else:
        print(f"[slice] full {token_count:,} tokens", flush=True)

    cfg = GPT2CompiledChannelConfig(
        alpha=args.alpha,
        max_train_tokens=args.max_train_tokens,
        recency_window=args.recency_window,
    )
    print("[compile] building GPT-2 compiled ngram/skip channels", flush=True)
    compile_started = time.time()
    builder = GPT2CompiledChannelBuilder.from_ids(ids, cfg)
    compile_seconds = time.time() - compile_started

    print(f"[save] {args.out}", flush=True)
    builder.save(args.out)
    artifact_bytes = args.out.stat().st_size

    report = {
        "ids": str(args.ids),
        "artifact": str(args.out),
        "token_count_input": token_count,
        "token_count_compiled": builder.total_tokens,
        "feature_dim": builder.feature_dim,
        "channel_names": list(builder.channel_names),
        "alpha": args.alpha,
        "recency_window": args.recency_window,
        "max_train_tokens": args.max_train_tokens,
        "compile_seconds": compile_seconds,
        "total_seconds": time.time() - started,
        "peak_rss_gib": peak_rss_gib(),
        "artifact_bytes": artifact_bytes,
    }
    with open(report_path, "w") as handle:
        json.dump(report, handle, indent=2)
    print(f"[report] {report_path}", flush=True)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()