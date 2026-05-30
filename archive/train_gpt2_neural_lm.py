"""train_gpt2_neural_lm.py — Train DeepCausalLM on WikiText-103 with real GPT-2 BPE.

Uses the standard WikiText-103 train/val/test splits tokenized with GPT-2 BPE.
Evaluates PPL on the standard test set using the sliding-window protocol.
This enables direct comparison against GPT-2 Small (PPL=29.0) and other public baselines.

Architecture: same DeepCausalLM but with V=50257 (GPT-2 vocab).
"""
from __future__ import annotations

import argparse, math, sys, time, json, importlib.util
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

DEEPSEEK = Path('/home/drawson/deepseek_experiments')
sys.path.insert(0, str(DEEPSEEK))

# Import model class from scaled training script
def _import_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_train_mod = _import_file('train_scaled',
                          str(DEEPSEEK / 'hybrid/train_scaled_neural_lm.py'))
DeepCausalLM = _train_mod.DeepCausalLM


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def iter_batches(ids: torch.Tensor, batch_size: int, seq_len: int, device,
                 generator: torch.Generator | None = None):
    N = len(ids)
    max_start = N - seq_len - 1
    while True:
        starts = torch.randint(0, max(1, max_start), (batch_size,),
                               generator=generator)
        idx = starts.unsqueeze(1) + torch.arange(seq_len + 1).unsqueeze(0)
        span = ids[idx].to(device, non_blocking=True)
        yield span[:, :-1], span[:, 1:]


@torch.no_grad()
def eval_sliding_window(model, ids: torch.Tensor, seq_len: int, device):
    """Standard sliding-window causal LM evaluation. Returns (avg_nll, total_tokens)."""
    model.eval()
    N = len(ids)
    V = model.vocab
    total_nll = 0.0
    total_tokens = 0

    stride = seq_len
    for start in range(0, N, stride):
        end = min(start + seq_len, N)
        chunk = ids[start:end].unsqueeze(0).to(device)
        L = chunk.shape[1]
        if L < 2:
            continue
        # For the sliding-window protocol, each position predicts the next token
        # using all previous tokens in the window.
        logits = model(chunk)
        lp = F.log_softmax(logits[0, :-1], dim=-1)  # (L-1, V)
        targets = chunk[0, 1:]  # (L-1,)
        nlls = F.nll_loss(lp, targets, reduction='none')
        total_nll += nlls.sum().item()
        total_tokens += L - 1

    return total_nll / total_tokens, total_tokens


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--batch', type=int, default=8)
    p.add_argument('--seq-len', type=int, default=128)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--d-model', type=int, default=256)
    p.add_argument('--n-layers', type=int, default=12)
    p.add_argument('--n-heads', type=int, default=8)
    p.add_argument('--d-ff', type=int, default=1024)
    p.add_argument('--dropout', type=float, default=0.1)
    p.add_argument('--steps-per-epoch', type=int, default=2000)
    p.add_argument('--warmup-steps', type=int, default=500)
    p.add_argument('--data-dir', type=str,
                   default='artifacts/wikitext_gpt2')
    p.add_argument('--out-dir', type=str, default='artifacts/hybrid_gpt2')
    p.add_argument('--device', type=str,
                   default='cuda' if torch.cuda.is_available() else 'cpu')
    args = p.parse_args()

    device = torch.device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir)

    torch.manual_seed(42)
    np.random.seed(42)
    g = torch.Generator().manual_seed(42)

    # --- Load GPT-2 BPE tokens ---
    print('=' * 60)
    print(' NEURAL LM — GPT-2 BPE (V=50257)')
    print('=' * 60)
    print('[1/5] Loading GPT-2 BPE tokenized data...')
    V = 50257

    train_ids = torch.load(data_dir / 'train_ids.pt', weights_only=False).long()
    val_ids = torch.load(data_dir / 'validation_ids.pt', weights_only=False).long()
    test_ids = torch.load(data_dir / 'test_ids.pt', weights_only=False).long()

    print(f'  Train: {len(train_ids):,} tokens')
    print(f'  Val:   {len(val_ids):,} tokens')
    print(f'  Test:  {len(test_ids):,} tokens')
    print(f'  Vocab: {V}')

    # --- Build model ---
    print('[2/5] Building DeepCausalLM (V=50257)...')
    model = DeepCausalLM(
        vocab=V, d_model=args.d_model, n_layers=args.n_layers,
        n_heads=args.n_heads, d_ff=args.d_ff, max_len=args.seq_len + 1,
        dropout=args.dropout
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'  Params: {n_params:,}  d_model={args.d_model}  n_layers={args.n_layers}')

    # --- Train ---
    print('[3/5] Training...')
    opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1,
                      betas=(0.9, 0.95))
    total_train_steps = args.epochs * args.steps_per_epoch
    warmup_frac = min(args.warmup_steps / max(total_train_steps, 1), 0.4)
    scheduler = optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=total_train_steps,
        pct_start=warmup_frac
    )
    batcher = iter_batches(train_ids, args.batch, args.seq_len, device, generator=g)

    best_val_ppl = float('inf')
    train_log = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        for step in range(args.steps_per_epoch):
            x, y = next(batcher)
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, V), y.reshape(-1))
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            scheduler.step()
            epoch_loss += loss.item()

        val_nll, val_tokens = eval_sliding_window(model, val_ids, args.seq_len, device)
        val_ppl = math.exp(val_nll)
        elapsed = time.time() - t0
        print(f'  Epoch {epoch:2d}/{args.epochs}  '
              f'train_loss={epoch_loss / args.steps_per_epoch:.4f}  '
              f'val_ppl={val_ppl:.2f}  '
              f'lr={scheduler.get_last_lr()[0]:.2e}  '
              f'time={elapsed:.0f}s', flush=True)

        train_log.append({'epoch': epoch, 'train_loss': epoch_loss / args.steps_per_epoch,
                          'val_ppl': val_ppl})

        if val_ppl < best_val_ppl:
            best_val_ppl = val_ppl
            torch.save({'epoch': epoch, 'state_dict': model.state_dict(),
                        'val_ppl': val_ppl, 'args': vars(args)},
                       out_dir / 'gpt2_lm_best.pt')

    # --- Evaluate on test set ---
    print('[4/5] Evaluating on standard WikiText-103 test set...')
    ckpt = torch.load(out_dir / 'gpt2_lm_best.pt', map_location=device)
    model.load_state_dict(ckpt['state_dict'])
    test_nll, test_tokens = eval_sliding_window(model, test_ids, args.seq_len, device)
    test_ppl = math.exp(test_nll)

    # --- Report ---
    print()
    print('=' * 60)
    print(' GPT-2 BPE RESULTS')
    print('=' * 60)
    print(f'  Model: DeepCausalLM  {n_params:,} params')
    print(f'  Tokenizer: GPT-2 BPE (V=50257)')
    print(f'  Eval protocol: Sliding-window (stride={args.seq_len})')
    print(f'  Test tokens: {test_tokens:,}')
    print(f'  Test PPL: {test_ppl:.2f}')
    print(f'  Test NLL: {test_nll:.4f}')
    print()
    print(f'  Baselines (GPT-2 BPE, WT-103):')
    print(f'    GPT-2 Small (124M):     PPL ≈ 29.0')
    print(f'    Our model ({n_params//1_000_000}M):  PPL = {test_ppl:.2f}')
    print('=' * 60)

    # --- Save report ---
    print('[5/5] Saving report...')
    report = {
        'model': 'DeepCausalLM',
        'params': n_params,
        'd_model': args.d_model, 'n_layers': args.n_layers,
        'n_heads': args.n_heads, 'd_ff': args.d_ff,
        'vocab_size': V, 'tokenizer': 'GPT-2 BPE (HuggingFace)',
        'train_tokens': len(train_ids), 'val_tokens': len(val_ids),
        'test_tokens': len(test_ids),
        'epochs': args.epochs, 'lr': args.lr,
        'best_val_ppl': best_val_ppl,
        'test_ppl': test_ppl, 'test_nll': test_nll,
        'eval_protocol': 'Standard sliding-window (stride=seq_len)',
        'split': 'Standard WikiText-103 train/val/test (document-disjoint)',
        'baselines': {
            'gpt2_small_124m': 29.0,
            'gpt2_medium_355m': 22.0,
        },
        'notes': [
            'First honest evaluation against public GPT-2 BPE baseline',
            'Document-disjoint train/val/test splits via HuggingFace WikiText-103',
            'Standard sliding-window eval protocol',
        ],
    }
    with open(out_dir / 'gpt2_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    print(f'  Report saved to {out_dir / "gpt2_report.json"}')


if __name__ == '__main__':
    main()
