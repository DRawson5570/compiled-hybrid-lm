"""train_340m_c4.py — Train a 340M GPT-2 BPE model on WikiText/C4 data."""
import sys
from hybrid.config import REPO_ROOT, time, math, argparse
from pathlib import Path
import importlib.util

import torch
import torch.nn as nn
import torch.nn.functional as F

DEEPSEEK = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(DEEPSEEK))

_spec = importlib.util.spec_from_file_location(
    'train_scaled', str(DEEPSEEK / 'hybrid/train_scaled_neural_lm.py'))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
DeepCausalLM = _mod.DeepCausalLM


def iter_batches(ids, batch, seq_len, device, generator=None):
    N = len(ids)
    while True:
        starts = torch.randint(0, max(1, N - seq_len - 1), (batch,), generator=generator)
        x = torch.stack([ids[s:s+seq_len] for s in starts]).to(device)
        y = torch.stack([ids[s+1:s+seq_len+1] for s in starts]).to(device)
        yield x, y


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--batch', type=int, default=2)
    p.add_argument('--seq-len', type=int, default=128)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--d-model', type=int, default=1024)
    p.add_argument('--n-layers', type=int, default=24)
    p.add_argument('--n-heads', type=int, default=16)
    p.add_argument('--d-ff', type=int, default=4096)
    p.add_argument('--dropout', type=float, default=0.1)
    p.add_argument('--steps-per-epoch', type=int, default=2000)
    p.add_argument('--warmup-steps', type=int, default=500)
    p.add_argument('--out-dir', type=str, default='artifacts/c4_340m')
    p.add_argument('--device', type=str, default='cuda')
    args = p.parse_args()

    device = torch.device(args.device)
    out_dir = Path(DEEPSEEK / args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print('=' * 60)
    print(f' 340M GPT-2 BPE TRAINING')
    print(f' d={args.d_model} L={args.n_layers} h={args.n_heads} ff={args.d_ff}')
    print('=' * 60)

    print('[load] data...')
    train_ids = torch.load(
        DEEPSEEK / 'artifacts/wikitext_gpt2/train_ids.pt',
        weights_only=False).long()
    val_ids = torch.load(
        DEEPSEEK / 'artifacts/wikitext_gpt2/validation_ids.pt',
        weights_only=False).long()
    print(f'  Train: {len(train_ids):,}  Val: {len(val_ids):,}')

    V = 50257
    model = DeepCausalLM(vocab=V, d_model=args.d_model, n_layers=args.n_layers,
                         n_heads=args.n_heads, d_ff=args.d_ff,
                         max_len=args.seq_len + 1, dropout=args.dropout)
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'  {n_params:,} params ({n_params/1e6:.1f}M)')

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95),
                            weight_decay=0.1)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, steps_per_epoch=args.steps_per_epoch,
        epochs=args.epochs, pct_start=args.warmup_steps / (args.steps_per_epoch * args.epochs))

    g = torch.Generator(device='cpu')
    g.manual_seed(42)
    batcher = iter_batches(train_ids, args.batch, args.seq_len, device, generator=g)

    best_eval = float('inf')
    t_start = time.time()

    for ep in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        t0 = time.time()

        for step in range(args.steps_per_epoch):
            x, y = next(batcher)
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, V), y.reshape(-1))

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            scheduler.step()
            total_loss += loss.item()

        avg_loss = total_loss / args.steps_per_epoch
        train_ppl = math.exp(avg_loss)

        # Eval
        model.eval()
        with torch.no_grad():
            eval_nll, eval_n = 0.0, 0
            for s in range(0, len(val_ids) - 1, 128):
                cl = min(128, len(val_ids) - s - 1)
                if cl <= 0: continue
                inp = val_ids[s:s+cl].unsqueeze(0).to(device)
                tgt = val_ids[s+1:s+cl+1].unsqueeze(0).to(device)
                l = model(inp)
                loss_v = F.cross_entropy(l.reshape(-1, V), tgt.reshape(-1), reduction='sum')
                eval_nll += loss_v.item(); eval_n += cl
        eval_ppl = math.exp(eval_nll / max(eval_n, 1))

        elapsed = time.time() - t0
        status = ''
        if eval_ppl < best_eval:
            best_eval = eval_ppl
            torch.save({'state_dict': model.state_dict(),
                        'eval_ppl': eval_ppl, 'epoch': ep,
                        'opt_state': opt.state_dict()},
                       out_dir / 'best.pt')
            status = 'SAVED'

        print(f'  epoch={ep:2d}  loss={avg_loss:.4f}  ppl={train_ppl:.1f}  '
              f'eval={eval_ppl:.1f}  best={best_eval:.1f}  {status}  time={elapsed:.0f}s',
              flush=True)

    print(f'\nDone. Best eval: {best_eval:.1f}  Total: {(time.time()-t_start)/3600:.1f}h')


if __name__ == '__main__':
    main()
