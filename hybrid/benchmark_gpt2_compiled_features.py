"""Benchmark GPT-2 compiled-channel feature row generation.

This is a CPU/RAM preflight for the compiled-feature transformer training path.
It loads an existing compiled channel artifact, samples token spans, emits causal
feature rows, and writes a compact throughput report.
"""
from __future__ import annotations

import argparse
import json
import random
import resource
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from hybrid.compiled_features import GPT2CompiledChannelBuilder


def peak_rss_gib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ids", type=Path, default=Path("artifacts/wikitext_gpt2/train_ids.pt"))
    parser.add_argument("--artifact", type=Path, default=Path("artifacts/compiled_feature_gpt2/compiled_ngram_channels.pt"))
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--spans", type=int, default=128)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--history", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not args.ids.exists():
        raise FileNotFoundError(f"missing token file: {args.ids}")
    if not args.artifact.exists():
        raise FileNotFoundError(f"missing compiled artifact: {args.artifact}")
    if args.spans <= 0 or args.seq_len <= 0:
        raise ValueError("--spans and --seq-len must be positive")

    report_path = args.report or args.artifact.with_suffix(".feature_benchmark.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)

    started = time.time()
    print(f"[load ids] {args.ids}", flush=True)
    ids = torch.load(args.ids, weights_only=False).long().cpu()
    max_start = ids.numel() - args.seq_len
    if max_start < 0:
        raise ValueError("token file is shorter than --seq-len")

    print(f"[load artifact] {args.artifact}", flush=True)
    builder = GPT2CompiledChannelBuilder.load(args.artifact)

    rng = random.Random(args.seed)
    starts = [rng.randint(0, max_start) for _ in range(args.spans)]

    print(f"[benchmark] spans={args.spans:,} seq_len={args.seq_len:,} history={args.history:,}", flush=True)
    bench_started = time.time()
    checksum = 0.0
    for start in starts:
        features = builder.build_features_for_span(ids, start=start, length=args.seq_len, history=args.history)
        checksum += float(features[:, 0].sum().item())
    feature_seconds = time.time() - bench_started
    feature_rows = args.spans * args.seq_len
    report = {
        "ids": str(args.ids),
        "artifact": str(args.artifact),
        "token_count": int(ids.numel()),
        "artifact_tokens": int(builder.total_tokens),
        "feature_dim": int(builder.feature_dim),
        "spans": args.spans,
        "seq_len": args.seq_len,
        "history": args.history,
        "feature_rows": feature_rows,
        "feature_seconds": feature_seconds,
        "rows_per_second": feature_rows / max(feature_seconds, 1e-12),
        "checksum": checksum,
        "total_seconds": time.time() - started,
        "peak_rss_gib": peak_rss_gib(),
    }
    with open(report_path, "w") as handle:
        json.dump(report, handle, indent=2)
    print(f"[report] {report_path}", flush=True)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()