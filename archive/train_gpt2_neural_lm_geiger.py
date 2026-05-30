"""train_gpt2_neural_lm_geiger.py — train_gpt2_neural_lm with:
  --resume PATH      Load checkpoint and continue training
  --geiger-tcp HOST:PORT  Connect to TCP server, read 0xFF bytes as click events
  --geiger-noise N    Scale of isotropic Gaussian noise per click (default 1e-5)
"""
from __future__ import annotations

import argparse, math, sys, time, json, importlib.util, os, select, errno, socket
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

DEEPSEEK = Path('/home/drawson/deepseek_experiments')
sys.path.insert(0, str(DEEPSEEK))

def _import_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_train_mod = _import_file('train_scaled',
                          str(DEEPSEEK / 'hybrid/train_scaled_neural_lm.py'))
DeepCausalLM = _train_mod.DeepCausalLM


def iter_batches(ids, batch_size, seq_len, device, generator=None):
    N = len(ids)
    max_start = N - seq_len - 1
    while True:
        starts = torch.randint(0, max(1, max_start), (batch_size,), generator=generator)
        idx = starts.unsqueeze(1) + torch.arange(seq_len + 1).unsqueeze(0)
        span = ids[idx].to(device, non_blocking=True)
        yield span[:, :-1], span[:, 1:]


@torch.no_grad()
def eval_sliding_window(model, ids, seq_len, device):
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
        logits = model(chunk)
        lp = F.log_softmax(logits[0, :-1], dim=-1)
        targets = chunk[0, 1:]
        nlls = F.nll_loss(lp, targets, reduction='none')
        total_nll += nlls.sum().item()
        total_tokens += L - 1
    return total_nll / total_tokens, total_tokens


class GeigerTCPReader:
    """Persistent TCP connection to geiger serial relay. Reads in background thread,
    accumulates click count. Training loop polls click_count atomically."""

    def __init__(self, host, port):
        self.host = host
        self.port = port
        self._click_count = 0
        self._buffer = b''
        self._sock = None
        self._running = False
        self._reconnect()

    def _reconnect(self):
        if self._sock:
            try: self._sock.close()
            except: pass
        while not self._running:
            try:
                self._sock = socket.socket()
                self._sock.settimeout(5.0)
                self._sock.connect((self.host, self.port))
                self._sock.setblocking(False)
                self._running = True
            except Exception:
                time.sleep(1)

    def drain(self):
        """Read all available data from socket, count 0xFF bytes. Non-blocking."""
        if not self._running:
            self._reconnect()
            return 0
        try:
            while True:
                try:
                    chunk = self._sock.recv(4096)
                except BlockingIOError:
                    break
                except (ConnectionResetError, BrokenPipeError, OSError):
                    self._running = False
                    self._reconnect()
                    break
                if not chunk:
                    self._running = False
                    self._reconnect()
                    break
                self._buffer += chunk
        except Exception:
            self._running = False
            self._reconnect()
            return 0
        clicks = self._buffer.count(b'\xff')
        self._buffer = self._buffer[-64:]  # keep trailing partial bytes
        self._click_count += clicks
        return clicks

    def total_clicks(self):
        return self._click_count


def inject_geiger_noise(model, noise_scale, clicks):
    """Add isotropic Gaussian noise to all model parameters, scaled by noise_scale * sqrt(clicks)."""
    if clicks <= 0:
        return
    scale = noise_scale * math.sqrt(clicks)
    with torch.no_grad():
        for p in model.parameters():
            p.data.add_(torch.randn_like(p) * scale)
    return scale


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
    p.add_argument('--data-dir', type=str, default='artifacts/wikitext_gpt2')
    p.add_argument('--out-dir', type=str, default='artifacts/hybrid_gpt2')
    p.add_argument('--device', type=str,
                    default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--resume', type=str, default=None,
                    help='Path to checkpoint.pt to resume from')
    p.add_argument('--seed', type=int, default=42,
                    help='Random seed (also reads SEED env var for remote runs)')
    p.add_argument('--geiger-tcp', type=str, default=None,
                    help='TCP host:port for Geiger counter serial relay')
    p.add_argument('--geiger-noise', type=float, default=1e-5,
                    help='Gaussian noise scale per sqrt(click) added to weights')
    args = p.parse_args()

    if 'SEED' in os.environ:
        args.seed = int(os.environ['SEED'])

    device = torch.device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    g = torch.Generator().manual_seed(args.seed)

    V = 50257

    print('=' * 60)
    print(' NEURAL LM — GPT-2 BPE w/ Geiger noise injection')
    print('=' * 60)

    print('[1/5] Loading GPT-2 BPE tokenized data...')
    train_ids = torch.load(data_dir / 'train_ids.pt', weights_only=False).long()
    val_ids = torch.load(data_dir / 'validation_ids.pt', weights_only=False).long()
    test_ids = torch.load(data_dir / 'test_ids.pt', weights_only=False).long()
    print(f'  Train: {len(train_ids):,} tokens')
    print(f'  Val:   {len(val_ids):,} tokens')
    print(f'  Test:  {len(test_ids):,} tokens')
    print(f'  Vocab: {V}')

    print('[2/5] Building DeepCausalLM (V=50257)...')
    model = DeepCausalLM(
        vocab=V, d_model=args.d_model, n_layers=args.n_layers,
        n_heads=args.n_heads, d_ff=args.d_ff, max_len=args.seq_len + 1,
        dropout=args.dropout
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'  Params: {n_params:,}  d_model={args.d_model}  n_layers={args.n_layers}')

    opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1,
                      betas=(0.9, 0.95))
    total_train_steps = args.epochs * args.steps_per_epoch
    warmup_frac = min(args.warmup_steps / max(total_train_steps, 1), 0.4)
    scheduler = optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=total_train_steps,
        pct_start=warmup_frac
    )

    start_epoch = 0
    geiger_clicks_total = 0

    if args.resume:
        print(f'[RESUME] Loading checkpoint: {args.resume}')
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['state_dict'])
        if 'opt_state' in ckpt:
            opt.load_state_dict(ckpt['opt_state'])
        if 'sched_state' in ckpt:
            scheduler.load_state_dict(ckpt['sched_state'])
        start_epoch = ckpt.get('epoch', 0)
        best_val_ppl = ckpt.get('val_ppl', float('inf'))
        print(f'  Resuming from epoch {start_epoch}, best val PPL={best_val_ppl:.2f}')
    else:
        best_val_ppl = float('inf')

    geiger_reader = None
    if args.geiger_tcp:
        host, port_str = args.geiger_tcp.split(':')
        geiger_reader = GeigerTCPReader(host, int(port_str))
        print(f'[GEIGER] TCP: {args.geiger_tcp}  noise_scale: {args.geiger_noise:.1e}')
    else:
        print('[GEIGER] No source configured — standard training')

    batcher = iter_batches(train_ids, args.batch, args.seq_len, device, generator=g)
    train_log = []

    print('[3/5] Training...')

    for epoch in range(start_epoch + 1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()
        epoch_clicks = 0

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

            if geiger_reader:
                clicks = geiger_reader.drain()
                if clicks > 0:
                    inject_geiger_noise(model, args.geiger_noise, clicks)
                    epoch_clicks += clicks
                    geiger_clicks_total += clicks

        val_nll, val_tokens = eval_sliding_window(model, val_ids, args.seq_len, device)
        val_ppl = math.exp(val_nll)
        elapsed = time.time() - t0
        geiger_info = f'  geiger={epoch_clicks} clicks' if geiger_reader else ''
        cur_lr = scheduler.get_last_lr()[0]
        print(f'  Epoch {epoch:2d}/{args.epochs}  '
              f'train_loss={epoch_loss / args.steps_per_epoch:.4f}  '
              f'val_ppl={val_ppl:.2f}  '
              f'lr={cur_lr:.2e}  '
              f'time={elapsed:.0f}s{geiger_info}', flush=True)

        train_log.append({'epoch': epoch, 'train_loss': epoch_loss / args.steps_per_epoch,
                          'val_ppl': val_ppl, 'geiger_clicks': epoch_clicks})

        if val_ppl < best_val_ppl:
            best_val_ppl = val_ppl
            save_dict = {
                'epoch': epoch, 'state_dict': model.state_dict(),
                'opt_state': opt.state_dict(), 'sched_state': scheduler.state_dict(),
                'val_ppl': val_ppl, 'args': vars(args),
                'geiger_clicks_total': geiger_clicks_total,
            }
            torch.save(save_dict, out_dir / 'gpt2_lm_best.pt')

    ckpt = torch.load(out_dir / 'gpt2_lm_best.pt', map_location=device)
    model.load_state_dict(ckpt['state_dict'])
    print('[4/5] Evaluating on standard WikiText-103 test set...')
    test_nll, test_tokens = eval_sliding_window(model, test_ids, args.seq_len, device)
    test_ppl = math.exp(test_nll)

    print()
    print('=' * 60)
    print(' GPT-2 BPE RESULTS (w/ Geiger noise)')
    print('=' * 60)
    print(f'  Model: DeepCausalLM  {n_params:,} params')
    print(f'  Seed: {args.seed}')
    print(f'  Geiger clicks total: {geiger_clicks_total}')
    print(f'  Test PPL: {test_ppl:.2f}')
    print(f'  Test NLL: {test_nll:.4f}')
    print('=' * 60)

    report = {
        'model': 'DeepCausalLM',
        'params': n_params,
        'seed': args.seed,
        'geiger_noise_scale': args.geiger_noise,
        'geiger_clicks_total': geiger_clicks_total,
        'best_val_ppl': best_val_ppl,
        'test_ppl': test_ppl, 'test_nll': test_nll,
    }
    with open(out_dir / 'gpt2_report.json', 'w') as f:
        json.dump(report, f, indent=2)


if __name__ == '__main__':
    main()
