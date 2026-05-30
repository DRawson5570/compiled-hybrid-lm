"""train_c4_batched.py — Train 124M neural LM on C4 with proper batching.

Reads C4 parquet files from local SSD. Interleaves WikiText at 15%.
Uses gradient accumulation for effective larger batches.
"""
from __future__ import annotations

import os, sys, math, time, json, argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from transformers import AutoTokenizer

os.environ.setdefault('HF_HOME', '/media/drawson/SSD-PGU3/hf_cache')
os.environ.setdefault('HF_DATASETS_CACHE', '/media/drawson/SSD-PGU3/hf_cache/datasets')

DEEPSEEK = Path('/home/drawson/deepseek_experiments')
sys.path.insert(0, str(DEEPSEEK))

from hybrid.train_scaled_neural_lm import DeepCausalLM


def load_c4_iter():
    from datasets import load_dataset
    ds = load_dataset('allenai/c4', 'en', split='train', streaming=True, trust_remote_code=True)
    ds = ds.shuffle(seed=42, buffer_size=10000)
    return iter(ds)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--steps-per-epoch', type=int, default=4000)
    p.add_argument('--batch', type=int, default=2)
    p.add_argument('--grad-accum', type=int, default=4)
    p.add_argument('--seq-len', type=int, default=128)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--out-dir', type=str, default='artifacts/c4_batched_768')
    p.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = p.parse_args()

    device = torch.device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(42); np.random.seed(42)
    gen = torch.Generator().manual_seed(42)

    print('=' * 60)
    print(' C4 BATCHED LM TRAINING (124M params)')
    print('=' * 60)

    # Load tokenizer and WikiText for eval
    tok = AutoTokenizer.from_pretrained('gpt2')
    V = 50257
    wt_test = torch.load(DEEPSEEK / 'artifacts/wikitext_gpt2/test_ids.pt', weights_only=False).long()
    wt_val = torch.load(DEEPSEEK / 'artifacts/wikitext_gpt2/validation_ids.pt', weights_only=False).long()
    wt_train = torch.load(DEEPSEEK / 'artifacts/wikitext_gpt2/train_ids.pt', weights_only=False).long()
    print(f'WikiText: train={len(wt_train):,} val={len(wt_val):,} test={len(wt_test):,}')

    # Load C4 from SSD
    print('Loading C4 from SSD...')
    c4_iter = load_c4_iter()
    print('C4 loaded')

    # Build model
    model = DeepCausalLM(vocab=V, d_model=768, n_layers=12, n_heads=12, d_ff=3072,
                          max_len=args.seq_len + 1, dropout=0.1).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'Params: {n_params:,}')

    opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1, betas=(0.9, 0.95))
    total_steps = args.epochs * args.steps_per_epoch
    scheduler = optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=total_steps,
                                               pct_start=min(500 / max(total_steps, 1), 0.4))

    @torch.no_grad()
    def eval_ppl(model, ids):
        model.eval()
        nll, n = 0.0, 0
        for s in range(0, max(0, len(ids) - 1), args.seq_len):
            cl = min(args.seq_len, len(ids) - s - 1)
            if cl <= 0: continue
            inp = ids[s:s + cl].unsqueeze(0).to(device)
            tgt = ids[s + 1:s + cl + 1].unsqueeze(0).to(device)
            logits = model(inp)
            loss = F.cross_entropy(logits.reshape(-1, V), tgt.reshape(-1), reduction='sum')
            nll += loss.item(); n += cl
        return nll / max(n, 1), n

    best_val_ppl = float('inf')
    token_buffer = []
    total_c4_tokens = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0; t0 = time.time()
        opt.zero_grad()

        for step in range(args.steps_per_epoch):
            # Build batch: fill buffer with C4 tokens, sample spans
            while len(token_buffer) < args.seq_len * args.batch * 2:
                r = np.random.random()
                if r < 0.15 and len(wt_train) > args.seq_len * 2:
                    s = np.random.randint(0, len(wt_train) - args.seq_len * 2)
                    token_buffer.extend(wt_train[s:s + args.seq_len * 2].tolist())
                else:
                    try:
                        ex = next(c4_iter)
                        text = ex.get('text', '')
                        if text and text.strip():
                            chunk = tok.encode(text)
                            if len(chunk) > 10:
                                token_buffer.extend(chunk[:args.seq_len * 4])
                                total_c4_tokens += len(chunk[:args.seq_len * 4])
                    except StopIteration:
                        c4_iter = load_c4_iter()
                        continue

            # Sample spans from buffer
            buf_t = torch.tensor(token_buffer, dtype=torch.long)
            max_start = len(token_buffer) - args.seq_len - 1
            if max_start < 1: continue
            starts = torch.randint(0, max_start, (args.batch,), generator=gen)
            offsets = torch.arange(args.seq_len + 1)
            idx = starts.unsqueeze(1) + offsets.unsqueeze(0)
            spans = buf_t[idx]
            inputs = spans[:, :-1].to(device)
            targets = spans[:, 1:].to(device)

            logits = model(inputs)
            loss = F.cross_entropy(logits.reshape(-1, V), targets.reshape(-1)) / args.grad_accum
            loss.backward()
            epoch_loss += loss.item() * args.grad_accum

            # Discard consumed prefix
            consumed = int(starts.max().item()) + args.seq_len + 1
            token_buffer = token_buffer[max(0, consumed - args.seq_len * 4):]

            if (step + 1) % args.grad_accum == 0:
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                scheduler.step()
                opt.zero_grad()

        val_nll, _ = eval_ppl(model, wt_val)
        val_ppl = math.exp(val_nll)
        elapsed = time.time() - t0
        print(f'epoch={epoch:2d}/{args.epochs} loss={epoch_loss / args.steps_per_epoch:.4f} '
              f'val={val_ppl:.1f} C4={total_c4_tokens/1e6:.0f}M tok time={elapsed:.0f}s', flush=True)

        if val_ppl < best_val_ppl:
            best_val_ppl = val_ppl
            torch.save({'epoch': epoch, 'state_dict': model.state_dict(), 'val_ppl': val_ppl},
                       out_dir / 'best.pt')

    # Test eval
    ckpt = torch.load(out_dir / 'best.pt', map_location=device)
    model.load_state_dict(ckpt['state_dict'])
    test_nll, test_tokens = eval_ppl(model, wt_test)
    test_ppl = math.exp(test_nll)
    print(f'\nTest PPL: {test_ppl:.2f} ({test_tokens:,} tokens)')
    print(f'GPT-2 Small: 29.0')

    json.dump({'test_ppl': test_ppl, 'best_val_ppl': best_val_ppl, 'params': n_params},
              open(out_dir / 'report.json', 'w'), indent=2)


if __name__ == '__main__':
    main()
