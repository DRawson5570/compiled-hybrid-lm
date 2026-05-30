"""c4_train_remote.py — C4+WikiText neural LM training, portable for remotes.

Usage:
  SEED=42 CUDA_VISIBLE_DEVICES=0 python3 -u c4_train_remote.py \
    --epochs 30 --steps 4000 --batch 2 --seq-len 128 --lr 3e-4 \
    --d-model 768 --n-layers 12 --n-heads 12 --d-ff 3072 \
    --out-dir artifacts/c4_seed42 --seed 42
"""
import sys, torch, math, time, numpy as np, os, argparse
from pathlib import Path
from transformers import AutoTokenizer
from datasets import load_dataset

sys.path.insert(0, os.path.expanduser('~/deepseek_experiments'))
from hybrid.train_scaled_neural_lm import DeepCausalLM
import torch.nn.functional as F
import torch.optim as optim

p = argparse.ArgumentParser()
p.add_argument('--epochs', type=int, default=30)
p.add_argument('--steps', type=int, default=4000)
p.add_argument('--batch', type=int, default=2)
p.add_argument('--seq-len', type=int, default=128)
p.add_argument('--lr', type=float, default=3e-4)
p.add_argument('--d-model', type=int, default=768)
p.add_argument('--n-layers', type=int, default=12)
p.add_argument('--n-heads', type=int, default=12)
p.add_argument('--d-ff', type=int, default=3072)
p.add_argument('--patience', type=int, default=10)
p.add_argument('--seed', type=int, default=42)
p.add_argument('--resume', type=str, default=None)
p.add_argument('--out-dir', type=str, default='artifacts/c4_remote')
args = p.parse_args()

if 'SEED' in os.environ:
    args.seed = int(os.environ['SEED'])

tok = AutoTokenizer.from_pretrained('gpt2')
V = 50257
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
gen = torch.Generator().manual_seed(args.seed)
torch.manual_seed(args.seed)
np.random.seed(args.seed)

wt_val = torch.load(
    os.path.expanduser('~/deepseek_experiments/artifacts/wikitext_gpt2/validation_ids.pt'),
    weights_only=False).long()[:10000]
wt_train = torch.load(
    os.path.expanduser('~/deepseek_experiments/artifacts/wikitext_gpt2/train_ids.pt'),
    weights_only=False).long()[:100000]

model = DeepCausalLM(vocab=V, d_model=args.d_model, n_layers=args.n_layers,
                     n_heads=args.n_heads, d_ff=args.d_ff,
                     max_len=args.seq_len + 1).to(device)
opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1)
total_steps = args.epochs * args.steps
if args.resume:
    scheduler = optim.lr_scheduler.ConstantLR(opt, factor=1.0)
    for g in opt.param_groups:
        g['lr'] = 5e-5
else:
    scheduler = optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=total_steps,
                                               pct_start=0.02)
n_params = sum(p.numel() for p in model.parameters())
print(f'Params: {n_params:,}  seed={args.seed}  device={device}', flush=True)

start_epoch = 0
if args.resume:
    ckpt = torch.load(os.path.expanduser(f'~/deepseek_experiments/{args.resume}'),
                      map_location=device, weights_only=False)
    model.load_state_dict(ckpt['state_dict'])
    start_epoch = ckpt.get('epoch', 0)
    best_val_ppl = ckpt.get('val_ppl', float('inf'))
    patience_counter = ckpt.get('patience_counter', 0)
    total_c4 = ckpt.get('total_c4', 0)
    print(f'[RESUME] epoch={start_epoch} best_ppl={best_val_ppl:.1f}', flush=True)
else:
    best_val_ppl = float('inf')
    patience_counter = 0
    total_c4 = 0

c4_ds = load_dataset('allenai/c4', 'en', split='train', streaming=True)
c4_iter = iter(c4_ds.shuffle(seed=args.seed, buffer_size=10000))
buf = list(wt_train.tolist())

out_dir = Path(os.path.expanduser(f'~/deepseek_experiments/{args.out_dir}'))
out_dir.mkdir(parents=True, exist_ok=True)

epoch_ppls = []  # local progress tracking

for epoch in range(start_epoch + 1, args.epochs + 1):
    model.train()
    epoch_loss = 0.0
    t0 = time.time()
    for step in range(args.steps):
        while len(buf) < 2000:
            if np.random.random() < 0.15:
                s = np.random.randint(0, len(wt_train) - 256)
                buf.extend(wt_train[s:s+256].tolist())
            else:
                try:
                    ex = next(c4_iter)
                    text = ex.get('text', '')
                    if text and text.strip():
                        ids = tok.encode(text[:2000])
                        buf.extend(ids)
                        total_c4 += len(ids)
                except StopIteration:
                    c4_iter = iter(c4_ds.shuffle(seed=hash(str(time.time())) % 2**32,
                                                 buffer_size=10000))

        max_start = len(buf) - args.seq_len - 1
        if max_start < 1:
            continue
        starts = torch.randint(0, max_start, (args.batch,), generator=gen)
        inputs_list = [torch.tensor(buf[s:s+args.seq_len]) for s in starts]
        targets_list = [torch.tensor(buf[s+1:s+args.seq_len+1]) for s in starts]
        inputs = torch.stack(inputs_list).to(device)
        targets = torch.stack(targets_list).to(device)

        logits = model(inputs)
        loss = F.cross_entropy(logits.reshape(-1, V), targets.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn_utils = torch.nn.utils
        nn_utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        scheduler.step()
        epoch_loss += loss.detach().item()
        consumed = int(starts.max().item()) + args.seq_len + 1
        buf = buf[max(0, consumed - 256):]

    model.eval()
    with torch.no_grad():
        nll, n = 0.0, 0
        for s in range(0, len(wt_val) - 1, args.seq_len):
            cl = min(args.seq_len, len(wt_val) - s - 1)
            if cl <= 0:
                continue
            inp = wt_val[s:s+cl].unsqueeze(0).to(device)
            tgt = wt_val[s+1:s+cl+1].unsqueeze(0).to(device)
            logits = model(inp)
            loss_val = F.cross_entropy(logits.reshape(-1, V), tgt.reshape(-1),
                                       reduction='sum')
            nll += loss_val.item()
            n += cl
    val_ppl = math.exp(nll / max(n, 1))

    # Local progress: reset if improving over recent epochs
    if len(epoch_ppls) == 0 or val_ppl < min(epoch_ppls[-max(1, args.patience // 2):]):
        patience_counter = 0
    else:
        patience_counter += 1
    epoch_ppls.append(val_ppl)

    if val_ppl < best_val_ppl:
        best_val_ppl = val_ppl
        torch.save({'epoch': epoch, 'state_dict': model.state_dict(),
                    'val_ppl': val_ppl, 'seed': args.seed,
                    'patience_counter': patience_counter},
                   out_dir / 'best.pt')

    status = "SAVED" if patience_counter == 0 else f"waiting ({patience_counter}/{args.patience})"
    print(f'epoch={epoch:2d} loss={epoch_loss/args.steps:.4f} val={val_ppl:.1f} '
          f'best={best_val_ppl:.1f} {status} C4={total_c4/1e6:.0f}M '
          f'time={time.time()-t0:.0f}s seed={args.seed}', flush=True)

    if patience_counter >= args.patience:
        print(f'Early stopping after {epoch} epochs', flush=True)
        break

print(f'Done. Best val PPL={best_val_ppl:.1f} at seed={args.seed}', flush=True)
