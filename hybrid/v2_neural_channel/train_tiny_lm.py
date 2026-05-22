"""hybrid/v2_neural_channel/train_tiny_lm.py

Train a small honest decoder-only transformer language model on the same
WikiText BPE token stream the compiled channels use.  Output is intended to
serve as an additional channel (the 13th) for the v1 blender.

Honest by construction:
  * Trained on ids[: 22M] only (the same prefix the compiled channels' KN/counts
    artifacts came from).
  * Evaluated on the v1 blender's val slice ids[22M : 22.5M] and the heldout
    slice ids[-100K:].
  * Standard next-token LM loss; no peeking, no oracle hooks, no stack
    introspection.

Default architecture: 1-layer causal Transformer, d_model=256, n_heads=4,
ffn=1024, weight-tied output head.  ~3M parameters.

Usage:
    CUDA_VISIBLE_DEVICES=2 python hybrid/v2_neural_channel/train_tiny_lm.py \\
        --out artifacts/hybrid_v2/tiny_lm_d256_l1.pt
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


class TinyCausalLM(nn.Module):
    def __init__(self, vocab: int, d_model: int = 256, n_layers: int = 1,
                 n_heads: int = 4, d_ff: int = 1024, max_len: int = 256,
                 dropout: float = 0.1):
        super().__init__()
        self.vocab = vocab
        self.d_model = d_model
        self.max_len = max_len
        self.tok_emb = nn.Embedding(vocab, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.dropout = nn.Dropout(dropout)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, activation="gelu", batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.ln_f = nn.LayerNorm(d_model)
        # weight tying
        self.head_bias = nn.Parameter(torch.zeros(vocab))
        self._init_weights()

    def _init_weights(self) -> None:
        # GPT-style: small std on embeddings and Linears so initial logits are
        # near zero -> initial CE ~ log(vocab).
        nn.init.normal_(self.tok_emb.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.pos_emb.weight, mean=0.0, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        """ids: (B, T) int64 -> logits: (B, T, V) float32"""
        B, T = ids.shape
        assert T <= self.max_len, (T, self.max_len)
        pos = torch.arange(T, device=ids.device).unsqueeze(0).expand(B, T)
        x = self.tok_emb(ids) + self.pos_emb(pos)
        x = self.dropout(x)
        mask = nn.Transformer.generate_square_subsequent_mask(T, device=ids.device)
        x = self.encoder(x, mask=mask, is_causal=True)
        x = self.ln_f(x)
        logits = x @ self.tok_emb.weight.T + self.head_bias
        return logits


def iter_batches(ids: torch.Tensor, batch: int, seq_len: int, device,
                 generator: torch.Generator | None = None):
    """Random contiguous spans of length seq_len+1 from ids.  Inputs are
    span[:-1], targets are span[1:]."""
    N = ids.shape[0]
    max_start = N - seq_len - 1
    while True:
        starts = torch.randint(0, max_start, (batch,), generator=generator)
        idx = starts.unsqueeze(1) + torch.arange(seq_len + 1).unsqueeze(0)
        span = ids[idx].to(device, non_blocking=True)
        yield span[:, :-1], span[:, 1:]


@torch.no_grad()
def eval_ppl_on_slice(model: TinyCausalLM, ids: torch.Tensor, seq_len: int,
                      device, stride: int | None = None) -> float:
    """Sliding-window eval PPL on a contiguous slice.  Uses non-overlapping
    windows by default (stride = seq_len) which is the fast and standard
    estimator; we just need it to be honest."""
    model.eval()
    if stride is None:
        stride = seq_len
    N = ids.shape[0]
    nll_sum = 0.0
    tok_count = 0
    for s in range(0, N - seq_len - 1, stride):
        e = s + seq_len + 1
        span = ids[s:e].to(device).unsqueeze(0)
        logits = model(span[:, :-1])
        loss = F.cross_entropy(logits.reshape(-1, model.vocab),
                               span[:, 1:].reshape(-1),
                               reduction="sum")
        nll_sum += loss.item()
        tok_count += seq_len
    return math.exp(nll_sum / tok_count)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ids", default="artifacts/compiled_wiki_lm_v11/cache_lm_ids.pt")
    p.add_argument("--out", default="artifacts/hybrid_v2/tiny_lm_d256_l1.pt")
    p.add_argument("--vocab", type=int, default=8000)
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--n-layers", type=int, default=1)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--d-ff", type=int, default=1024)
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--warmup", type=int, default=500)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--train-end", type=int, default=22_000_000,
                   help="Train on ids[:train_end].")
    p.add_argument("--val-start", type=int, default=22_000_000)
    p.add_argument("--val-end", type=int, default=22_500_000)
    p.add_argument("--heldout-tail", type=int, default=100_000,
                   help="Heldout slice = ids[-heldout_tail:].")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    print(f"[load] {args.ids}")
    ids = torch.load(args.ids).long()
    print(f"  total tokens: {ids.shape[0]:,}  vocab observed: [{ids.min().item()}, {ids.max().item()}]")

    train_ids = ids[: args.train_end].contiguous()
    val_ids = ids[args.val_start : args.val_end].contiguous()
    heldout_ids = ids[-args.heldout_tail:].contiguous()
    print(f"  train: {train_ids.shape[0]:,}  val: {val_ids.shape[0]:,}  heldout: {heldout_ids.shape[0]:,}")

    model = TinyCausalLM(
        vocab=args.vocab, d_model=args.d_model, n_layers=args.n_layers,
        n_heads=args.n_heads, d_ff=args.d_ff, max_len=args.seq_len,
        dropout=args.dropout,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] TinyCausalLM d={args.d_model} L={args.n_layers} H={args.n_heads} ff={args.d_ff} "
          f"seq={args.seq_len} params={n_params:,}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay,
                            betas=(0.9, 0.95))

    def lr_at(step: int) -> float:
        if step < args.warmup:
            return args.lr * (step + 1) / args.warmup
        # cosine decay to lr/10
        progress = (step - args.warmup) / max(1, args.steps - args.warmup)
        return args.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))

    gen = torch.Generator()
    gen.manual_seed(args.seed + 1)
    train_iter = iter_batches(train_ids, args.batch, args.seq_len, device, generator=gen)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")
    t0 = time.time()
    log = []
    for step in range(args.steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        model.train()
        x, y = next(train_iter)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, model.vocab), y.reshape(-1))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if (step + 1) % args.eval_every == 0 or step == 0:
            val_ppl = eval_ppl_on_slice(model, val_ids, args.seq_len, device)
            heldout_ppl = eval_ppl_on_slice(model, heldout_ids, args.seq_len, device)
            row = {"step": step + 1, "loss": float(loss.item()),
                   "val_ppl": val_ppl, "heldout_ppl": heldout_ppl,
                   "lr": lr_at(step)}
            log.append(row)
            tag = ""
            if val_ppl < best_val - 1e-3:
                best_val = val_ppl
                torch.save({
                    "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                    "args": vars(args),
                    "step": step + 1,
                    "val_ppl": val_ppl,
                    "heldout_ppl": heldout_ppl,
                }, args.out)
                tag = " * saved"
            elapsed = time.time() - t0
            print(f"  step {step+1:6d}  loss={loss.item():.4f}  "
                  f"val_ppl={val_ppl:7.2f}  heldout_ppl={heldout_ppl:7.2f}  "
                  f"lr={lr_at(step):.2e}  ({elapsed:.0f}s){tag}", flush=True)

    print(f"\n[done] best val_ppl={best_val:.2f}  out={args.out}")


if __name__ == "__main__":
    main()
