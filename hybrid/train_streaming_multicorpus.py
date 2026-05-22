"""train_streaming_multicorpus.py — Train CompiledFeatureTransformer on streaming C4 + Pile + WikiText.

Streams corpora via HuggingFace datasets (no local download needed), tokenizes
on-the-fly with GPT-2 BPE, builds compiled features from WikiText-fitted channel
builder, and trains architecture #1 from HYBRID_STRATEGY.md.

The compiled channel builder is fitted once on WikiText-103 train (pre-tokenized).
Features for all corpora use the same compiled statistics — this tests whether
compiled n-gram patterns generalize across corpora.
"""
from __future__ import annotations

import argparse, json, math, sys, time
from pathlib import Path
from typing import Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from transformers import AutoTokenizer

DEEPSEEK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DEEPSEEK))

from hybrid.compiled_features import (
    CompiledFeatureTransformer,
    CompiledFeatureTransformerConfig,
    GPT2_COMPILED_FEATURE_DIM,
    GPT2CompiledChannelBuilder,
    GPT2CompiledChannelConfig,
)
from hybrid.calibration import expected_calibration_error, brier_score, find_best_temperature


# ═══════════════════════════════════════════════════════════════════════════════
# Streaming data pipeline
# ═══════════════════════════════════════════════════════════════════════════════

def stream_mixed_corpus(
    tokenizer,
    corpus_weights: dict[str, float],
    seq_len: int,
    buffer_size: int = 1_000_000,
    rng_seed: int = 42,
) -> Iterator[torch.Tensor]:
    """Stream and tokenize mixed corpora, yielding flat token tensors.

    Args:
        tokenizer: GPT-2 BPE tokenizer
        corpus_weights: {'c4': 0.4, 'pile': 0.4, 'wikitext': 0.2} etc.
        seq_len: sequence length (used for batching later)
        buffer_size: max tokens to accumulate before yielding a chunk
        rng_seed: random seed for corpus selection
    """
    import numpy as np
    rng = np.random.default_rng(rng_seed)
    names = list(corpus_weights.keys())
    probs = np.array([corpus_weights[n] for n in names])
    probs = probs / probs.sum()

    # Load WikiText train (pre-tokenized)
    wt_path = DEEPSEEK / 'artifacts/wikitext_gpt2/train_ids.pt'
    wt_ids = None
    if 'wikitext' in corpus_weights and wt_path.exists():
        wt_ids = torch.load(wt_path, weights_only=False).long()
        print(f'  WikiText pre-tokenized: {len(wt_ids):,} tokens')

    # Set up streaming datasets
    from datasets import load_dataset
    streams = {}
    if 'c4' in corpus_weights:
        c4_ds = load_dataset('allenai/c4', 'en', split='train', streaming=True,
                             trust_remote_code=True)
        streams['c4'] = iter(c4_ds)
    if 'pile' in corpus_weights:
        pile_ds = load_dataset('monology/pile-uncopyrighted', split='train', streaming=True)
        streams['pile'] = iter(pile_ds)

    token_buffer = []
    total_tokens = 0

    while True:
        # Pick a corpus
        name = rng.choice(names, p=probs)
        if name == 'wikitext' and wt_ids is not None:
            # Sample a random span from pre-tokenized WikiText
            start = rng.integers(0, max(1, len(wt_ids) - seq_len * 16))
            length = rng.integers(seq_len, seq_len * 16)
            length = min(length, len(wt_ids) - start)
            chunk = wt_ids[start:start + length].tolist()
        elif name in streams:
            try:
                example = next(streams[name])
                text = example['text']
                if not text or not text.strip():
                    continue
                chunk = tokenizer.encode(text)
            except StopIteration:
                # Restart the stream
                if name == 'c4':
                    ds = load_dataset('allenai/c4', 'en', split='train', streaming=True,
                                      trust_remote_code=True)
                else:
                    ds = load_dataset('monology/pile-uncopyrighted', split='train', streaming=True)
                streams[name] = iter(ds)
                continue
        else:
            continue

        if not chunk:
            continue
        token_buffer.extend(chunk)
        total_tokens += len(chunk)

        if len(token_buffer) >= buffer_size:
            yield torch.tensor(token_buffer[:buffer_size], dtype=torch.long)
            token_buffer = token_buffer[buffer_size:]

        if total_tokens % 5_000_000 == 0:
            print(f'    [stream] {total_tokens/1e6:.1f}M tokens streamed', flush=True)


def iter_batches_from_stream(
    stream: Iterator[torch.Tensor],
    compiled_builder: GPT2CompiledChannelBuilder,
    batch_size: int,
    seq_len: int,
    history: int,
    device: torch.device,
    generator: torch.Generator,
) -> Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Yield (input_ids, target_ids, compiled_features) batches from token stream.

    Maintains a rolling buffer of tokens.  When the buffer is large enough,
    samples random spans, builds compiled features, and yields batches.
    """
    buffer = []
    while True:
        # Refill buffer
        while len(buffer) < seq_len * batch_size * 4:
            try:
                chunk = next(stream)
                buffer.extend(chunk.tolist())
            except StopIteration:
                break

        if len(buffer) < seq_len + 1:
            break

        buffer_t = torch.tensor(buffer[:len(buffer)], dtype=torch.long)
        max_start = len(buffer) - seq_len - 1

        starts = torch.randint(0, max(1, max_start), (batch_size,), generator=generator)
        offsets = torch.arange(seq_len + 1)
        token_idx = starts.unsqueeze(1) + offsets.unsqueeze(0)
        spans = buffer_t[token_idx]

        # Build compiled features for each span
        features = []
        for b in range(batch_size):
            feat = compiled_builder.build_features_for_span(
                buffer_t,
                start=int(starts[b].item()),
                length=seq_len,
                history=history,
            )
            features.append(feat)

        # Discard consumed prefix (keep some history for context)
        consumed = int(starts.max().item()) + seq_len + 1
        buffer = buffer[max(0, consumed - history):]

        yield (
            spans[:, :-1].to(device),
            spans[:, 1:].to(device),
            torch.stack(features, dim=0).to(device),
        )


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
            ids, start=start, length=chunk_len, history=history
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
    p.add_argument('--steps-per-epoch', type=int, default=4000)
    p.add_argument('--batch', type=int, default=8)
    p.add_argument('--seq-len', type=int, default=128)
    p.add_argument('--history', type=int, default=512)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--warmup-steps', type=int, default=500)
    p.add_argument('--d-model', type=int, default=256)
    p.add_argument('--n-layers', type=int, default=6)
    p.add_argument('--n-heads', type=int, default=8)
    p.add_argument('--d-ff', type=int, default=1024)
    p.add_argument('--dropout', type=float, default=0.1)
    p.add_argument('--out-dir', type=str, default='artifacts/streaming_multicorpus')
    p.add_argument('--device', type=str,
                   default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--c4-weight', type=float, default=0.4)
    p.add_argument('--pile-weight', type=float, default=0.4)
    p.add_argument('--wikitext-weight', type=float, default=0.2)
    p.add_argument('--compile-max-train-tokens', type=int, default=0,
                   help='Max tokens for compiled builder fitting (0=all)')
    args = p.parse_args()

    device = torch.device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(42)
    gen = torch.Generator().manual_seed(42)

    print('=' * 60)
    print(' STREAMING MULTI-CORPUS TRAINING — Architecture #1')
    print('=' * 60)

    # ── Load tokenizer + WikiText test set ──
    print('[1/5] Loading GPT-2 tokenizer and WikiText test set...')
    tok = AutoTokenizer.from_pretrained('gpt2')
    V = 50257
    wt_test = torch.load(DEEPSEEK / 'artifacts/wikitext_gpt2/test_ids.pt', weights_only=False).long()
    wt_val = torch.load(DEEPSEEK / 'artifacts/wikitext_gpt2/validation_ids.pt', weights_only=False).long()
    print(f'  V={V}  test={len(wt_test):,}  val={len(wt_val):,}')

    # ── Fit compiled channel builder on WikiText train ──
    print('[2/5] Fitting compiled channel builder on WikiText-103 train...')
    wt_train = torch.load(DEEPSEEK / 'artifacts/wikitext_gpt2/train_ids.pt', weights_only=False).long()
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
    print(f'  params={n_params:,}  d_model={args.d_model}')

    # ── Set up streaming ──
    print('[4/5] Setting up streaming corpus pipeline...')
    corpus_weights = {
        'c4': args.c4_weight,
        'pile': args.pile_weight,
        'wikitext': args.wikitext_weight,
    }
    print(f'  Mix: c4={args.c4_weight} pile={args.pile_weight} wikitext={args.wikitext_weight}')

    token_stream = stream_mixed_corpus(tok, corpus_weights, args.seq_len)
    batches = iter_batches_from_stream(
        token_stream, compiled_builder,
        args.batch, args.seq_len, args.history, device, gen
    )

    # ── Train ──
    opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1, betas=(0.9, 0.95))
    total_steps = args.epochs * args.steps_per_epoch
    scheduler = optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=total_steps,
        pct_start=min(args.warmup_steps / max(total_steps, 1), 0.4),
    )

    best_val_ppl = float('inf')
    train_log = []

    for epoch in range(1, args.epochs + 1):
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
            scheduler.step()
            epoch_loss += float(loss.item())

        val_report = eval_sliding_window(
            model, wt_val, compiled_builder, args.seq_len, args.history, device
        )
        elapsed = time.time() - t0
        print(f'  epoch={epoch:2d}/{args.epochs}  '
              f'train_loss={epoch_loss / args.steps_per_epoch:.4f}  '
              f'val_ppl={val_report["ppl"]:.2f}  '
              f'lr={scheduler.get_last_lr()[0]:.2e}  '
              f'time={elapsed:.0f}s', flush=True)

        train_log.append({'epoch': epoch, 'train_loss': epoch_loss / args.steps_per_epoch,
                          'val_ppl': val_report['ppl']})

        if val_report['ppl'] < best_val_ppl:
            best_val_ppl = val_report['ppl']
            torch.save({
                'epoch': epoch, 'state_dict': model.state_dict(),
                'config': cfg.__dict__, 'val_ppl': best_val_ppl,
            }, out_dir / 'streaming_model_best.pt')

    # ── Evaluate on WikiText-103 test ──
    print('[5/5] Evaluating on WikiText-103 test set...')
    ckpt = torch.load(out_dir / 'streaming_model_best.pt', map_location=device)
    model.load_state_dict(ckpt['state_dict'])
    test_report = eval_sliding_window(
        model, wt_test, compiled_builder, args.seq_len, args.history, device
    )

    print(f'\n  Test PPL: {test_report["ppl"]:.2f}  ({test_report["tokens"]:,} tokens)')
    print(f'  GPT-2 Small baseline: 29.0')

    report = {
        'model': 'CompiledFeatureTransformer (streaming multi-corpus)',
        'params': n_params, 'config': cfg.__dict__,
        'corpus_weights': corpus_weights,
        'train_log': train_log,
        'test_report': test_report,
    }
    with open(out_dir / 'streaming_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    print(f'  Report: {out_dir / "streaming_report.json"}')


if __name__ == '__main__':
    main()
