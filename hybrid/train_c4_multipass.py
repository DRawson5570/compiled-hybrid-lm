"""train_c4_multipass.py — Train CompiledFeatureTransformer on local C4 + WikiText.

Reads C4 parquet files from the SSD on /media/drawson/SSD-PGU3.
Uses compiled features from WikiText-fitted channel builder (architecture #1).
Cosine-with-restarts LR schedule for sustained multi-pass training.
Evaluates on standard WikiText-103 test set (GPT-2 BPE).
"""
from __future__ import annotations

import argparse, json, math, os, sys, time
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from transformers import AutoTokenizer

os.environ.setdefault('HF_HOME', os.path.expanduser('~/.cache/huggingface'))
os.environ.setdefault('HF_HUB_CACHE', os.path.expanduser('~/.cache/huggingface/hub'))
os.environ.setdefault('HF_DATASETS_CACHE', os.path.expanduser('~/.cache/huggingface/datasets'))

DEEPSEEK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DEEPSEEK))

from hybrid.compiled_features import (
    CompiledFeatureTransformer,
    CompiledFeatureTransformerConfig,
    GPT2_COMPILED_FEATURE_DIM,
    GPT2CompiledChannelBuilder,
    GPT2CompiledChannelConfig,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Data pipeline
# ═══════════════════════════════════════════════════════════════════════════════

def iter_c4_batches(
    tokenizer,
    compiled_builder: GPT2CompiledChannelBuilder,
    batch_size: int,
    seq_len: int,
    history: int,
    device: torch.device,
    generator: torch.Generator,
    wiki_ids: torch.Tensor | None = None,
    wiki_weight: float = 0.15,
) -> Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Yield (input_ids, target_ids, compiled_features) from C4 + WikiText.

    Uses streaming mode to read cached shards from SSD and download
    missing shards on-the-fly. Interleaves WikiText at wiki_weight fraction.
    """
    from datasets import load_dataset

    # Streaming reads from local cache when available, downloads when not
    ds = load_dataset('allenai/c4', 'en', split='train', streaming=True,
                      trust_remote_code=True)
    ds = ds.shuffle(seed=42, buffer_size=10000)
    print(f'  C4: streaming from SSD cache + network fallback')

    wiki_len = len(wiki_ids) if wiki_ids is not None else 0
    token_buffer = []
    total_tokens = 0

    def _fresh_stream():
        return iter(ds.shuffle(seed=hash(str(time.time())) % 2**32, buffer_size=10000))

    example_iter = _fresh_stream()

    while True:
        if wiki_ids is not None and torch.rand(1, generator=generator).item() < wiki_weight:
            start = torch.randint(0, max(1, wiki_len - seq_len * 8), (1,), generator=generator).item()
            length = torch.randint(seq_len, seq_len * 8, (1,), generator=generator).item()
            length = min(length, wiki_len - start)
            chunk = wiki_ids[start:start + length].tolist()
        else:
            try:
                example = next(example_iter)
                text = example['text']
                if not text or not text.strip():
                    continue
                chunk = tokenizer.encode(text)
            except StopIteration:
                example_iter = _fresh_stream()
                continue

        if not chunk:
            continue
        token_buffer.extend(chunk)
        total_tokens += len(chunk)

        # Emit batches when buffer is full enough
        while len(token_buffer) >= seq_len * batch_size * 2:
            buffer_t = torch.tensor(token_buffer, dtype=torch.long)
            max_start = len(token_buffer) - seq_len - 1

            # Sample random spans
            starts = torch.randint(0, max_start, (batch_size,), generator=generator)
            offsets = torch.arange(seq_len + 1)
            token_idx = starts.unsqueeze(1) + offsets.unsqueeze(0)
            spans = buffer_t[token_idx]

            # Build compiled features
            features = []
            for b in range(batch_size):
                feat = compiled_builder.build_features_for_span(
                    buffer_t, start=int(starts[b].item()),
                    length=seq_len, history=history,
                )
                features.append(feat)

            # Discard consumed prefix, keep history window
            consumed = int(starts.max().item()) + seq_len + 1
            token_buffer = token_buffer[max(0, consumed - history):]

            yield (
                spans[:, :-1].to(device),
                spans[:, 1:].to(device),
                torch.stack(features, dim=0).to(device),
            )

        if total_tokens % 10_000_000 == 0:
            print(f'    [data] {total_tokens/1e6:.0f}M tokens processed', flush=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Evaluation
# ═══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def eval_sliding_window(model, ids, compiled_builder, seq_len, history, device):
    model.eval()
    ids = ids.long().cpu()
    total_nll = 0.0
    total_tokens = 0

    for start in range(0, max(0, ids.numel() - 1), seq_len):
        chunk_len = min(seq_len, ids.numel() - start - 1)
        if chunk_len <= 0:
            continue
        input_ids = ids[start:start + chunk_len].unsqueeze(0).to(device)
        target_ids = ids[start + 1:start + chunk_len + 1].unsqueeze(0).to(device)
        features = compiled_builder.build_features_for_span(
            ids, start=start, length=chunk_len, history=history,
        ).unsqueeze(0).to(device)
        logits = model(input_ids, features)
        loss = F.cross_entropy(
            logits.reshape(-1, model.vocab), target_ids.reshape(-1), reduction='sum'
        )
        total_nll += float(loss.item())
        total_tokens += chunk_len

    avg_nll = total_nll / max(total_tokens, 1)
    return {'nll': avg_nll, 'ppl': math.exp(avg_nll), 'tokens': total_tokens}


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--epochs', type=int, default=5)
    p.add_argument('--steps-per-epoch', type=int, default=8000)
    p.add_argument('--batch', type=int, default=8)
    p.add_argument('--seq-len', type=int, default=128)
    p.add_argument('--history', type=int, default=512)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--lr-min', type=float, default=1e-5)
    p.add_argument('--warmup-steps', type=int, default=1000)
    p.add_argument('--restart-every-epochs', type=int, default=3)
    p.add_argument('--d-model', type=int, default=256)
    p.add_argument('--n-layers', type=int, default=6)
    p.add_argument('--n-heads', type=int, default=8)
    p.add_argument('--d-ff', type=int, default=1024)
    p.add_argument('--dropout', type=float, default=0.1)
    p.add_argument('--compile-max-train-tokens', type=int, default=10_000_000)
    p.add_argument('--wiki-weight', type=float, default=0.15)
    p.add_argument('--out-dir', type=str, default='artifacts/c4_multipass')
    p.add_argument('--device', type=str,
                   default='cuda' if torch.cuda.is_available() else 'cpu')
    args = p.parse_args()

    device = torch.device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(42)
    np.random.seed(42)
    gen = torch.Generator().manual_seed(42)

    print('=' * 60)
    print(' C4 MULTI-PASS TRAINING — Architecture #1')
    print('=' * 60)

    # ── Load tokenizer ──
    print('[1/5] Loading tokenizer and WikiText splits...')
    tok = AutoTokenizer.from_pretrained('gpt2')
    V = 50257
    wt_test = torch.load(DEEPSEEK / 'artifacts/wikitext_gpt2/test_ids.pt', weights_only=False).long()
    wt_val = torch.load(DEEPSEEK / 'artifacts/wikitext_gpt2/validation_ids.pt', weights_only=False).long()
    wt_train = torch.load(DEEPSEEK / 'artifacts/wikitext_gpt2/train_ids.pt', weights_only=False).long()
    print(f'  V={V}  test={len(wt_test):,}  val={len(wt_val):,}  wiki_train={len(wt_train):,}')

    # ── Fit compiled builder on WikiText ──
    print(f'[2/5] Fitting compiled builder on {args.compile_max_train_tokens/1e6:.0f}M WikiText tokens...')
    t0 = time.time()
    compiled_builder = GPT2CompiledChannelBuilder.from_ids(
        wt_train,
        GPT2CompiledChannelConfig(
            alpha=0.1, recency_window=args.history,
            max_train_tokens=args.compile_max_train_tokens,
        ),
    )
    print(f'  Fitted in {time.time()-t0:.0f}s  feature_dim={GPT2_COMPILED_FEATURE_DIM}')

    # ── Build model ──
    print('[3/5] Building CompiledFeatureTransformer...')
    cfg = CompiledFeatureTransformerConfig(
        vocab_size=V, feature_dim=GPT2_COMPILED_FEATURE_DIM,
        d_model=args.d_model, n_layers=args.n_layers, n_heads=args.n_heads,
        d_ff=args.d_ff, max_seq_len=args.seq_len, dropout=args.dropout,
    )
    model = CompiledFeatureTransformer(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'  params={n_params:,}  d_model={args.d_model}  n_layers={args.n_layers}')

    # ── Set up data pipeline ──
    print('[4/5] Setting up C4 streaming pipeline (SSD cache + network)...')
    batches = iter_c4_batches(
        tok, compiled_builder, args.batch, args.seq_len, args.history,
        device, gen, wiki_ids=wt_train, wiki_weight=args.wiki_weight,
    )

    # ── Train with cosine restarts ──
    opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1, betas=(0.9, 0.95))
    steps_per_restart = args.restart_every_epochs * args.steps_per_epoch

    best_val_ppl = float('inf')
    train_log = []

    for epoch in range(1, args.epochs + 1):
        # Cosine schedule with restarts
        cycle_pos = ((epoch - 1) * args.steps_per_epoch) % steps_per_restart
        lr = args.lr_min + 0.5 * (args.lr - args.lr_min) * (
            1 + math.cos(math.pi * cycle_pos / steps_per_restart)
        )
        for pg in opt.param_groups:
            pg['lr'] = lr

        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        for step in range(args.steps_per_epoch):
            input_ids, target_ids, features = next(batches)
            logits = model(input_ids, features)
            loss = F.cross_entropy(logits.reshape(-1, V), target_ids.reshape(-1))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            epoch_loss += float(loss.item())

        val_report = eval_sliding_window(
            model, wt_val, compiled_builder, args.seq_len, args.history, device
        )
        elapsed = time.time() - t0
        print(f'  epoch={epoch:2d}/{args.epochs}  '
              f'train_loss={epoch_loss / args.steps_per_epoch:.4f}  '
              f'val_ppl={val_report["ppl"]:.2f}  '
              f'lr={lr:.2e}  '
              f'time={elapsed:.0f}s', flush=True)

        train_log.append({'epoch': epoch, 'train_loss': epoch_loss / args.steps_per_epoch,
                          'val_ppl': val_report['ppl'], 'lr': lr})

        if val_report['ppl'] < best_val_ppl:
            best_val_ppl = val_report['ppl']
            torch.save({
                'epoch': epoch, 'state_dict': model.state_dict(),
                'config': cfg.__dict__, 'val_ppl': best_val_ppl,
            }, out_dir / 'c4_model_best.pt')

    # ── Evaluate on WT-103 test ──
    print('[5/5] Evaluating on WikiText-103 test set...')
    ckpt = torch.load(out_dir / 'c4_model_best.pt', map_location=device)
    model.load_state_dict(ckpt['state_dict'])
    test_report = eval_sliding_window(
        model, wt_test, compiled_builder, args.seq_len, args.history, device
    )

    print(f'\n  Test PPL: {test_report["ppl"]:.2f}  ({test_report["tokens"]:,} tokens)')
    print(f'  GPT-2 Small (124M, 3B tokens): PPL=29.0')
    print(f'  Ours ({n_params//1_000_000}M, C4+WikiText): PPL={test_report["ppl"]:.2f}')

    report = {
        'model': 'CompiledFeatureTransformer', 'params': n_params,
        'config': cfg.__dict__, 'train_log': train_log,
        'test_report': test_report,
        'data': 'C4 (local SSD) + WikiText-103, interleaved 85/15',
        'tokenizer': 'GPT-2 BPE (50257)',
        'lr_schedule': f'cosine restarts every {args.restart_every_epochs} epochs',
    }
    with open(out_dir / 'c4_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    print(f'  Report: {out_dir / "c4_report.json"}')


if __name__ == '__main__':
    main()
