"""Train CompiledFeatureTransformer on GPT-2-tokenized WikiText.

This is the first end-to-end training entry point for HYBRID_STRATEGY.md
architecture #1: token embeddings plus causal compiled-feature inputs inside the
learned LM. The default feature source is the GPT-2 token-stat adapter, which is
an honest weak baseline until the full compiled channel stack is ported to
V=50257.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from hybrid.calibration import brier_score, expected_calibration_error, find_best_temperature
from hybrid.compiled_features import (
    CompiledFeatureTransformer,
    CompiledFeatureTransformerConfig,
    GPT2_COMPILED_FEATURE_DIM,
    GPT2CompiledChannelBuilder,
    GPT2CompiledChannelConfig,
    build_token_stat_features_for_span,
    iter_span_compiled_feature_batches,
)


@torch.no_grad()
def eval_sliding_window(
    model: CompiledFeatureTransformer,
    ids: torch.Tensor,
    *,
    seq_len: int,
    device: torch.device,
    history: int,
    window: int,
    compiled_builder: GPT2CompiledChannelBuilder | None = None,
    max_eval_tokens: int = 0,
    calibration_tokens: int = 4096,
) -> dict:
    """Evaluate causal NLL/PPL with aligned compiled features."""
    model.eval()
    ids = ids.long().cpu()
    if max_eval_tokens > 0:
        ids = ids[:max_eval_tokens]

    total_nll = 0.0
    total_tokens = 0
    calibration_logits = []
    calibration_targets = []

    for start in range(0, max(0, ids.numel() - 1), seq_len):
        chunk_len = min(seq_len, ids.numel() - start - 1)
        if chunk_len <= 0:
            continue
        input_ids = ids[start:start + chunk_len].unsqueeze(0).to(device)
        target_ids = ids[start + 1:start + chunk_len + 1].unsqueeze(0).to(device)
        if compiled_builder is None:
            features = build_token_stat_features_for_span(
                ids,
                start=start,
                length=chunk_len,
                history=history,
                window=window,
            )
        else:
            features = compiled_builder.build_features_for_span(
                ids,
                start=start,
                length=chunk_len,
                history=history,
            )
        features = features.unsqueeze(0).to(device)
        logits = model(input_ids, features)
        loss = F.cross_entropy(logits.reshape(-1, model.vocab), target_ids.reshape(-1), reduction="sum")
        total_nll += float(loss.item())
        total_tokens += chunk_len

        if sum(t.numel() for t in calibration_targets) < calibration_tokens:
            flat_logits = logits.reshape(-1, model.vocab).detach().cpu()
            flat_targets = target_ids.reshape(-1).detach().cpu()
            remaining = calibration_tokens - sum(t.numel() for t in calibration_targets)
            calibration_logits.append(flat_logits[:remaining])
            calibration_targets.append(flat_targets[:remaining])

    avg_nll = total_nll / max(total_tokens, 1)
    report = {
        "nll": avg_nll,
        "ppl": math.exp(avg_nll) if avg_nll < 100 else float("inf"),
        "tokens": total_tokens,
    }

    if calibration_logits and calibration_targets:
        logits_c = torch.cat(calibration_logits, dim=0)
        targets_c = torch.cat(calibration_targets, dim=0)
        temp = find_best_temperature(logits_c, targets_c)
        report.update({
            "ece": float(expected_calibration_error(logits_c, targets_c).item()),
            "brier": float(brier_score(logits_c, targets_c).item()),
            "best_temperature": temp.temperature,
            "temperature_nll": temp.nll,
            "calibration_tokens": int(targets_c.numel()),
        })
    return report


def load_ids(data_dir: Path, split: str) -> torch.Tensor:
    path = data_dir / f"{split}_ids.pt"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}; run hybrid/tokenize_wikitext_gpt2.py first")
    return torch.load(path, weights_only=False).long()


def serializable_args(args: argparse.Namespace) -> dict:
    return {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}


def iter_builder_batches(
    ids: torch.Tensor,
    builder: GPT2CompiledChannelBuilder,
    *,
    batch_size: int,
    seq_len: int,
    history: int,
    device: torch.device,
    generator: torch.Generator,
):
    ids = ids.long().cpu()
    max_start = ids.numel() - seq_len - 1
    offsets = torch.arange(seq_len + 1)
    while True:
        starts = torch.randint(0, max_start + 1, (batch_size,), generator=generator)
        token_idx = starts.unsqueeze(1) + offsets.unsqueeze(0)
        spans = ids[token_idx]
        features = [
            builder.build_features_for_span(ids, start=int(start.item()), length=seq_len, history=history)
            for start in starts
        ]
        yield spans[:, :-1].to(device), spans[:, 1:].to(device), torch.stack(features, dim=0).to(device)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("artifacts/wikitext_gpt2"))
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/compiled_feature_gpt2"))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--steps-per-epoch", type=int, default=1000)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--history", type=int, default=512)
    parser.add_argument("--feature-window", type=int, default=128)
    parser.add_argument("--feature-source", choices=["token_stat", "compiled_ngram"], default="compiled_ngram")
    parser.add_argument("--compile-max-train-tokens", type=int, default=0)
    parser.add_argument("--compile-alpha", type=float, default=0.1)
    parser.add_argument("--compiled-artifact-in", type=Path, default=None)
    parser.add_argument("--compiled-artifact-out", type=Path, default=None)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--n-layers", type=int, default=6)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--d-ff", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--feature-dropout", type=float, default=0.0)
    parser.add_argument("--max-train-tokens", type=int, default=0)
    parser.add_argument("--max-eval-tokens", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    generator = torch.Generator().manual_seed(args.seed)
    device = torch.device(args.device)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print(" COMPILED-FEATURE TRANSFORMER — GPT-2 BPE")
    print("=" * 72)
    print("[1/5] Loading tokenized WikiText-103 GPT-2 splits...")
    train_ids = load_ids(args.data_dir, "train")
    val_ids = load_ids(args.data_dir, "validation")
    test_ids = load_ids(args.data_dir, "test")
    if args.max_train_tokens > 0:
        train_ids = train_ids[:args.max_train_tokens]
    print(f"  train={train_ids.numel():,} val={val_ids.numel():,} test={test_ids.numel():,}")

    print("[2/5] Building model...")
    compiled_builder = None
    if args.feature_source == "compiled_ngram":
        loaded_compiled_artifact = False
        if args.compiled_artifact_in is not None and args.compiled_artifact_in.exists():
            print(f"  loading compiled GPT-2 channel artifact from {args.compiled_artifact_in}...")
            compiled_builder = GPT2CompiledChannelBuilder.load(args.compiled_artifact_in)
            loaded_compiled_artifact = True
        else:
            print("  compiling GPT-2 ngram/skip channel artifact from train split...")
            compiled_builder = GPT2CompiledChannelBuilder.from_ids(
                train_ids,
                GPT2CompiledChannelConfig(
                    alpha=args.compile_alpha,
                    max_train_tokens=args.compile_max_train_tokens,
                    recency_window=args.feature_window,
                ),
            )
        artifact_out = args.compiled_artifact_out
        if artifact_out is None and not loaded_compiled_artifact:
            artifact_out = args.out_dir / "compiled_ngram_channels.pt"
        if artifact_out is not None:
            compiled_builder.save(artifact_out)
            print(f"  compiled artifact saved to {artifact_out}")
        else:
            print("  using loaded compiled artifact without re-saving it")
        feature_dim = GPT2_COMPILED_FEATURE_DIM
        feature_note = "GPT-2 compiled ngram/skip channel summaries"
    else:
        feature_dim = 10
        feature_note = "bounded-history token-stat adapter; weak baseline"

    cfg = CompiledFeatureTransformerConfig(
        vocab_size=50257,
        feature_dim=feature_dim,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
        max_seq_len=args.seq_len,
        dropout=args.dropout,
        feature_dropout=args.feature_dropout,
    )
    model = CompiledFeatureTransformer(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  params={n_params:,} feature_dim={cfg.feature_dim} feature_source={args.feature_source} device={device}")

    print("[3/5] Training...")
    opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1, betas=(0.9, 0.95))
    total_steps = args.epochs * args.steps_per_epoch
    scheduler = optim.lr_scheduler.OneCycleLR(
        opt,
        max_lr=args.lr,
        total_steps=total_steps,
        pct_start=min(args.warmup_steps / max(total_steps, 1), 0.4),
    )
    if compiled_builder is None:
        batches = iter_span_compiled_feature_batches(
            train_ids,
            batch_size=args.batch,
            seq_len=args.seq_len,
            history=args.history,
            window=args.feature_window,
            device=device,
            generator=generator,
        )
    else:
        batches = iter_builder_batches(
            train_ids,
            compiled_builder,
            batch_size=args.batch,
            seq_len=args.seq_len,
            history=args.history,
            device=device,
            generator=generator,
        )

    best_val_ppl = float("inf")
    train_log = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()
        for _ in range(args.steps_per_epoch):
            batch = next(batches)
            if compiled_builder is None:
                input_ids, target_ids, compiled_features = batch.input_ids, batch.target_ids, batch.compiled_features
            else:
                input_ids, target_ids, compiled_features = batch
            logits = model(input_ids, compiled_features)
            loss = F.cross_entropy(logits.reshape(-1, model.vocab), target_ids.reshape(-1))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            scheduler.step()
            epoch_loss += float(loss.item())

        val_report = eval_sliding_window(
            model,
            val_ids,
            seq_len=args.seq_len,
            device=device,
            history=args.history,
            window=args.feature_window,
            compiled_builder=compiled_builder,
            max_eval_tokens=args.max_eval_tokens,
        )
        row = {
            "epoch": epoch,
            "train_loss": epoch_loss / args.steps_per_epoch,
            "val_ppl": val_report["ppl"],
            "val_nll": val_report["nll"],
            "seconds": time.time() - t0,
        }
        train_log.append(row)
        print(
            f"  epoch={epoch:02d} train_loss={row['train_loss']:.4f} "
            f"val_ppl={row['val_ppl']:.2f} lr={scheduler.get_last_lr()[0]:.2e} "
            f"time={row['seconds']:.0f}s",
            flush=True,
        )
        if val_report["ppl"] < best_val_ppl:
            best_val_ppl = val_report["ppl"]
            torch.save(
                {
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "config": cfg.__dict__,
                    "args": serializable_args(args),
                    "feature_source": args.feature_source,
                    "val_report": val_report,
                },
                args.out_dir / "compiled_feature_transformer_best.pt",
            )

    print("[4/5] Evaluating best checkpoint on test split...")
    ckpt = torch.load(args.out_dir / "compiled_feature_transformer_best.pt", map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    test_report = eval_sliding_window(
        model,
        test_ids,
        seq_len=args.seq_len,
        device=device,
        history=args.history,
        window=args.feature_window,
        compiled_builder=compiled_builder,
        max_eval_tokens=args.max_eval_tokens,
    )

    print("[5/5] Saving report...")
    report = {
        "model": "CompiledFeatureTransformer",
        "tokenizer": "GPT-2 BPE",
        "feature_source": feature_note,
        "params": n_params,
        "config": cfg.__dict__,
        "args": serializable_args(args),
        "train_log": train_log,
        "best_val_ppl": best_val_ppl,
        "test_report": test_report,
        "split": "WikiText-103 train/validation/test GPT-2 tokenized splits",
        "baseline_notes": {
            "gpt2_small_wt103_ppl_approx": 29.0,
            "status": "integration path; numbers are not a full compiled-channel milestone until real GPT-2 compiled channels replace token-stat features",
        },
    }
    with open(args.out_dir / "compiled_feature_report.json", "w") as handle:
        json.dump(report, handle, indent=2)
    print(f"  test_ppl={test_report['ppl']:.2f} test_nll={test_report['nll']:.4f}")
    print(f"  report={args.out_dir / 'compiled_feature_report.json'}")


if __name__ == "__main__":
    main()
