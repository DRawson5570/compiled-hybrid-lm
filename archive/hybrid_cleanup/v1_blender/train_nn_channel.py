"""hybrid/v1_blender/train_nn_channel.py

Train a tiny causal transformer LM as an additional blender channel.

The compiled channels (kn n-gram, cluster mix, decayed caches, attention caches)
cover local + frequency + cache statistics but have no compositional context.
A 1-layer transformer over a 256-token window gives the blender a genuine
context-dependent next-token distribution to mix with.

Trained on a slice of the same 22M-token cache_lm_ids.pt used for the v31
channels.  Validated on the val_big slice (ids[22M:22.5M]).

Usage (pe2):
    CUDA_VISIBLE_DEVICES=1 nohup python -u hybrid/v1_blender/train_nn_channel.py \\
        --out artifacts/nn_channel/nn_channel_v1.pt \\
        > /tmp/nn_channel_train.log 2>&1 &
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))


# ----------------------------- model -------------------------------------

class TinyTransformerLM(nn.Module):
    """1-layer causal transformer LM."""
    def __init__(self, vocab: int, d_model: int = 256, n_heads: int = 4,
                 d_ff: int = 1024, ctx: int = 256, n_layers: int = 1,
                 dropout: float = 0.1):
        super().__init__()
        self.vocab = vocab
        self.d_model = d_model
        self.ctx = ctx
        self.tok_emb = nn.Embedding(vocab, d_model)
        self.pos_emb = nn.Embedding(ctx, d_model)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            _Block(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab, bias=False)
        # weight tying
        self.head.weight = self.tok_emb.weight
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        """ids: (B, T); returns logits (B, T, V)."""
        B, T = ids.shape
        assert T <= self.ctx, (T, self.ctx)
        pos = torch.arange(T, device=ids.device)
        x = self.tok_emb(ids) + self.pos_emb(pos)[None]
        x = self.drop(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        return self.head(x)


class _Block(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout,
                                          batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.shape[1]
        h = self.ln1(x)
        mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
        attn, _ = self.attn(h, h, h, attn_mask=mask, need_weights=False)
        x = x + attn
        x = x + self.ff(self.ln2(x))
        return x


# --------------------------- data loader --------------------------------

def sample_batch(ids: torch.Tensor, batch: int, ctx: int, device: torch.device,
                 generator: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample `batch` random ctx+1 windows from `ids`.

    Returns (x, y) where y is x shifted by 1.
    """
    N = ids.shape[0]
    starts = torch.randint(0, N - ctx - 1, (batch,), generator=generator)
    x = torch.stack([ids[s:s + ctx] for s in starts]).to(device, non_blocking=True).long()
    y = torch.stack([ids[s + 1:s + ctx + 1] for s in starts]).to(device, non_blocking=True).long()
    return x, y


@torch.no_grad()
def eval_ppl(model: TinyTransformerLM, ids: torch.Tensor, ctx: int, device: torch.device,
             batch: int = 32, max_chunks: int = 64) -> float:
    """Sliding-window evaluation: chunk ids into ctx-sized blocks (no overlap)
    and compute mean NLL over targets.  Limited to max_chunks for speed.
    """
    model.eval()
    N = ids.shape[0]
    n_chunks = min(max_chunks, max(1, (N - 1) // ctx))
    nll_sum = 0.0
    n_tok = 0
    for i in range(0, n_chunks, batch):
        b = min(batch, n_chunks - i)
        starts = [i_ * ctx for i_ in range(i, i + b)]
        x = torch.stack([ids[s:s + ctx] for s in starts]).to(device).long()
        y = torch.stack([ids[s + 1:s + ctx + 1] for s in starts]).to(device).long()
        logits = model(x)
        log_probs = F.log_softmax(logits, dim=-1)
        nll = -log_probs.gather(-1, y.unsqueeze(-1)).squeeze(-1)
        nll_sum += nll.sum().item()
        n_tok += y.numel()
    mean_nll = nll_sum / n_tok
    return math.exp(mean_nll)


# ------------------------------ main ------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", default="artifacts/compiled_wiki_lm_v11/cache_lm_ids.pt")
    p.add_argument("--out", default="artifacts/nn_channel/nn_channel_v1.pt")
    p.add_argument("--train-end", type=int, default=22_000_000,
                   help="Use ids[:train_end] for training (leaves 22M..22.5M = val_big, last 100K = heldout).")
    p.add_argument("--val-start", type=int, default=22_000_000)
    p.add_argument("--val-len", type=int, default=200_000,
                   help="Tokens of val_big to sample-eval during training.")
    p.add_argument("--ctx", type=int, default=256)
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--d-ff", type=int, default=1024)
    p.add_argument("--n-layers", type=int, default=1)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup", type=int, default=500)
    p.add_argument("--steps", type=int, default=10_000)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    print(f"[device] {device}  cuda={torch.cuda.is_available()}")

    print(f"[load] {args.cache}")
    ids = torch.load(args.cache).long()
    print(f"  total tokens={ids.shape[0]:,}  V={int(ids.max()) + 1}")
    V = int(ids.max()) + 1

    train_ids = ids[:args.train_end]
    val_ids = ids[args.val_start:args.val_start + args.val_len]
    print(f"  train={train_ids.shape[0]:,}  val={val_ids.shape[0]:,}")

    model = TinyTransformerLM(
        vocab=V, d_model=args.d_model, n_heads=args.n_heads,
        d_ff=args.d_ff, ctx=args.ctx, n_layers=args.n_layers,
        dropout=args.dropout,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] params={n_params/1e6:.2f}M  ctx={args.ctx}  d={args.d_model}  "
          f"heads={args.n_heads}  layers={args.n_layers}")

    # AdamW with decoupled wd; bias/LN no decay
    decay, nodecay = [], []
    for n, prm in model.named_parameters():
        if prm.dim() >= 2:
            decay.append(prm)
        else:
            nodecay.append(prm)
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": args.weight_decay},
         {"params": nodecay, "weight_decay": 0.0}],
        lr=args.lr, betas=(0.9, 0.95),
    )

    def lr_at(step: int) -> float:
        if step < args.warmup:
            return args.lr * step / args.warmup
        progress = (step - args.warmup) / max(1, args.steps - args.warmup)
        return args.lr * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    gen = torch.Generator().manual_seed(args.seed)

    t0 = time.time()
    best_ppl = float("inf")
    best_state = None
    log = []
    print(f"[train] {args.steps} steps  batch={args.batch}  lr={args.lr}")
    for step in range(args.steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        model.train()
        x, y = sample_batch(train_ids, args.batch, args.ctx, device, gen)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, V), y.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % args.eval_every == 0 or step == args.steps - 1:
            val_ppl = eval_ppl(model, val_ids, args.ctx, device,
                               batch=args.batch, max_chunks=64)
            elapsed = time.time() - t0
            sps = (step + 1) / elapsed if elapsed > 0 else 0.0
            print(f"  step {step:6d}  train_loss={loss.item():.3f}  "
                  f"val_ppl={val_ppl:7.2f}  lr={lr_at(step):.2e}  "
                  f"{sps:.1f} step/s  ({elapsed:.0f}s)")
            log.append({"step": step, "train_loss": loss.item(),
                        "val_ppl": val_ppl, "lr": lr_at(step)})
            if val_ppl < best_ppl:
                best_ppl = val_ppl
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    print(f"[done] best val_ppl={best_ppl:.2f}  ({time.time()-t0:.0f}s)")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": best_state,
        "config": {"vocab": V, "d_model": args.d_model, "n_heads": args.n_heads,
                   "d_ff": args.d_ff, "ctx": args.ctx, "n_layers": args.n_layers,
                   "dropout": args.dropout},
        "best_val_ppl": best_ppl,
        "log": log,
        "args": vars(args),
    }, str(out))
    print(f"[save] {out}")


if __name__ == "__main__":
    main()
