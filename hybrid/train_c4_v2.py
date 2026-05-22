"""train_c4_v2.py — Train 124M neural LM on C4+WikiText. Proven pattern."""
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--steps-per-epoch', type=int, default=4000)
    p.add_argument('--batch', type=int, default=2)
    p.add_argument('--seq-len', type=int, default=128)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--d-model', type=int, default=768)
    p.add_argument('--n-layers', type=int, default=12)
    p.add_argument('--n-heads', type=int, default=12)
    p.add_argument('--d-ff', type=int, default=3072)
    p.add_argument('--out-dir', type=str, default='artifacts/c4_v2')
    p.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = p.parse_args()

    device = torch.device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(42); np.random.seed(42)
    gen = torch.Generator().manual_seed(42)

    print('=' * 60)
    print(' C4 + WikiText TRAINING')
    print('=' * 60)

    tok = AutoTokenizer.from_pretrained('gpt2')
    V = 50257

    wt_test = torch.load(DEEPSEEK / 'artifacts/wikitext_gpt2/test_ids.pt', weights_only=False).long()
    wt_val = torch.load(DEEPSEEK / 'artifacts/wikitext_gpt2/validation_ids.pt', weights_only=False).long()
    wt_train = torch.load(DEEPSEEK / 'artifacts/wikitext_gpt2/train_ids.pt', weights_only=False).long()
    print(f'WikiText: train={len(wt_train):,} val={len(wt_val):,} test={len(wt_test):,}')

    from datasets import load_dataset
    c4_ds = load_dataset('allenai/c4', 'en', split='train', streaming=True, trust_remote_code=True)
    
    def fresh_c4_iter():
        return iter(c4_ds.shuffle(seed=hash(str(time.time())) % 2**32, buffer_size=10000))
    
    c4_iter = fresh_c4_iter()
    print('C4 streaming ready')

    model = DeepCausalLM(vocab=V, d_model=args.d_model, n_layers=args.n_layers,
                          n_heads=args.n_heads, d_ff=args.d_ff,
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
            nll += float(loss); n += cl
            del logits, loss
        return nll / max(n, 1), n

    best_val_ppl = float('inf')
    total_c4 = 0
    token_buffer = []  # shared buffer across batches

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0; t0 = time.time()

        for step in range(args.steps_per_epoch):
            # Refill buffer
            while len(token_buffer) < (args.seq_len + 1) * args.batch * 2:
                r = np.random.random()
                if r < 0.15 and len(wt_train) > args.seq_len * 2:
                    s = np.random.randint(0, len(wt_train) - args.seq_len * 2 - 1)
                    token_buffer.extend(wt_train[s:s + args.seq_len * 2].tolist())
                else:
                    try:
                        ex = next(c4_iter)
                        text = ex.get('text', '')
                        if text and text.strip():
                            ids = tok.encode(text[:2000])  # truncate text before tokenizing
                            if ids:
                                token_buffer.extend(ids)
                                total_c4 += len(ids)
                    except StopIteration:
                        c4_iter = fresh_c4_iter()

            if len(token_buffer) < (args.seq_len + 1) * args.batch:
                token_buffer.extend(wt_train[:args.seq_len * 8].tolist())

            # Sample batch from buffer
            buf_t = torch.tensor(token_buffer, dtype=torch.long)
            max_start = len(token_buffer) - args.seq_len - 1
            starts = torch.randint(0, max(1, max_start), (args.batch,), generator=gen)
            offsets = torch.arange(args.seq_len + 1)
            idx = starts.unsqueeze(1) + offsets.unsqueeze(0)
            spans = buf_t[idx]
            inputs = spans[:, :-1].to(device)
            targets = spans[:, 1:].to(device)

            logits = model(inputs)
            loss = F.cross_entropy(logits.reshape(-1, V), targets.reshape(-1))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            scheduler.step()
            epoch_loss += loss.detach().item()

            # Discard consumed prefix (keep some history)
            consumed = int(starts.max().item()) + args.seq_len + 1
            token_buffer = token_buffer[max(0, consumed - args.seq_len * 2):]

        # Eval on 10K tokens to prevent OOM
        val_slice = wt_val[:10000]
        torch.cuda.empty_cache()
        val_nll, val_tok = eval_ppl(model, val_slice)
        val_ppl = math.exp(val_nll)
        elapsed = time.time() - t0
        print(f'epoch={epoch:2d} loss={epoch_loss / args.steps_per_epoch:.4f} '
              f'val={val_ppl:.1f} C4={total_c4/1e6:.0f}M lr={scheduler.get_last_lr()[0]:.2e} time={elapsed:.0f}s', flush=True)

        if val_ppl < best_val_ppl:
            best_val_ppl = val_ppl
            torch.save({'epoch': epoch, 'state_dict': model.state_dict(), 'val_ppl': val_ppl},
                       out_dir / 'best.pt')

    ckpt = torch.load(out_dir / 'best.pt', map_location=device)
    model.load_state_dict(ckpt['state_dict'])
    test_nll, test_tok = eval_ppl(model, wt_test)
    test_ppl = math.exp(test_nll)
    print(f'\nTest PPL: {test_ppl:.2f} ({test_tok:,} tokens)')

    json.dump({'test_ppl': test_ppl, 'best_val_ppl': best_val_ppl, 'params': n_params,
               'total_c4_tokens': total_c4},
              open(out_dir / 'report.json', 'w'), indent=2)


if __name__ == '__main__':
    main()
