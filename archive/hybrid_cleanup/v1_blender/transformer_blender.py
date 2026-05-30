"""hybrid/v1_blender/transformer_blender.py

Small causal transformer that consumes per-position channel feature vectors
and emits per-position softmax mixing weights over channels.  Drop-in
replacement for the position-wise ``DeepBlender`` MLP, giving the gate
context beyond hand-crafted summary statistics.

The model is trained on fixed-length windows of the (T, F) feature stream.
During eval the stream is processed with overlapping windows so every
heldout position has full ctx-length left context.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TBConfig:
    in_dim: int
    n_channels: int
    d_model: int = 192
    n_heads: int = 4
    d_ff: int = 768
    n_layers: int = 2
    ctx: int = 256
    dropout: float = 0.1


class _CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        qkv = self.qkv(x).view(B, T, 3, self.n_heads, self.d_head)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)  # (B, H, T, dh)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        # scaled_dot_product_attention with is_causal=True
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True,
                                             dropout_p=self.drop.p if self.training else 0.0)
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.proj(out)


class _Block(nn.Module):
    def __init__(self, cfg: TBConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = _CausalSelfAttention(cfg.d_model, cfg.n_heads, cfg.dropout)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.ff = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_ff),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_ff, cfg.d_model),
        )
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop(self.attn(self.ln1(x)))
        x = x + self.drop(self.ff(self.ln2(x)))
        return x


class TransformerBlender(nn.Module):
    """Causal transformer feature mixer.

    Input  : (B, T, F)  per-position feature vectors
    Output : (B, T, C)  log mixing weights (already log-softmaxed)
    """

    def __init__(self, cfg: TBConfig):
        super().__init__()
        self.cfg = cfg
        self.in_proj = nn.Linear(cfg.in_dim, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.ctx, cfg.d_model)
        self.blocks = nn.ModuleList([_Block(cfg) for _ in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.n_channels)
        # Standard transformer init for all submodules first.
        self.apply(self._init_weights)
        # Then zero the head so the model is uniform-mixture-at-init regardless
        # of the input — the log_softmax of zeros is log(1/C) per channel.
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, F_in = x.shape
        assert T <= self.cfg.ctx, (T, self.cfg.ctx)
        pos = torch.arange(T, device=x.device)
        h = self.in_proj(x) + self.pos_emb(pos)[None, :, :]
        for blk in self.blocks:
            h = blk(h)
        h = self.ln_f(h)
        # Re-zero head was set above; for safety pass through head.
        logits = self.head(h)
        return F.log_softmax(logits, dim=-1)


# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------


def _sample_windows(T: int, ctx: int, n: int, device: torch.device) -> torch.Tensor:
    """Sample n window start indices in [0, T - ctx]."""
    if T <= ctx:
        return torch.zeros(n, dtype=torch.long, device=device)
    return torch.randint(0, T - ctx + 1, (n,), device=device)


def train_transformer_blender(
    feats_train: torch.Tensor,           # (T_train, F)  on device
    log_p_train: torch.Tensor,           # (T_train, C)  log p_c(y_t | ctx)
    feats_inval: torch.Tensor,           # (T_inval, F)
    log_p_inval: torch.Tensor,           # (T_inval, C)
    cfg: TBConfig,
    *,
    steps: int = 1500,
    batch: int = 16,
    lr: float = 3e-4,
    wd: float = 1e-3,
    warmup: int = 100,
    eval_every: int = 100,
    patience: int = 6,
    device: Optional[torch.device] = None,
    log_fn=print,
) -> tuple[TransformerBlender, float, int]:
    """Train a TransformerBlender on randomly sampled causal windows.

    Reports best-on-inval PPL (mixture NLL).  Early stops when no improvement
    for ``patience`` consecutive eval rounds.
    """
    if device is None:
        device = feats_train.device
    model = TransformerBlender(cfg).to(device)
    # Decoupled weight decay (skip bias / LayerNorm / pos_emb).
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim == 1 or n.endswith(".bias") or "ln" in n or "pos_emb" in n:
            no_decay.append(p)
        else:
            decay.append(p)
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": wd},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=lr, betas=(0.9, 0.95),
    )

    def lr_at(step: int) -> float:
        if step < warmup:
            return lr * (step + 1) / warmup
        progress = (step - warmup) / max(1, steps - warmup)
        return lr * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    from hybrid.v1_blender.blender_model import mixture_nll

    T_tr = feats_train.shape[0]
    best_ppl = float("inf")
    best_state = None
    no_improve = 0

    @torch.no_grad()
    def eval_ppl_stream(feats: torch.Tensor, log_p: torch.Tensor) -> float:
        # Sliding eval: stride = ctx // 2, keep predictions for the second half
        # of each window (which has at least ctx/2 left context).  First window
        # contributes positions 0..ctx-1 (cold start).
        model.eval()
        T = feats.shape[0]
        ctx = cfg.ctx
        stride = max(1, ctx // 2)
        nll_sum = 0.0
        n_pos = 0
        for s in range(0, T, stride):
            e = min(s + ctx, T)
            win_f = feats[s:e].unsqueeze(0)  # (1, L, F)
            win_p = log_p[s:e]               # (L, C)
            log_w = model(win_f)[0]          # (L, C)
            # On the first window keep all positions; on later windows keep
            # the last (e - s) - (ctx - stride) = stride positions (their left
            # context is already at least stride long).
            if s == 0:
                keep_start = 0
            else:
                keep_start = ctx - stride
            keep_len = (e - s) - keep_start
            if keep_len <= 0:
                continue
            nll = mixture_nll(log_w[keep_start:keep_start + keep_len],
                              win_p[keep_start:keep_start + keep_len])
            nll_sum += nll.sum().item()
            n_pos += keep_len
        return math.exp(nll_sum / max(1, n_pos))

    t_total0 = __import__("time").time()
    for step in range(steps):
        model.train()
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        starts = _sample_windows(T_tr, cfg.ctx, batch, device)
        # Build (B, ctx, F) and (B, ctx, C, V) — V can be large so this is the
        # memory pressure point; keep batch modest.
        idx = starts[:, None] + torch.arange(cfg.ctx, device=device)[None, :]
        win_f = feats_train[idx]          # (B, ctx, F)
        win_p = log_p_train[idx]          # (B, ctx, C)
        log_w = model(win_f)              # (B, ctx, C)
        # mixture_nll expects (N, C) and (N, C); flatten B*ctx.
        Bw, Tw, Cw = log_w.shape
        loss = mixture_nll(log_w.reshape(Bw * Tw, Cw),
                           win_p.reshape(Bw * Tw, Cw)).mean()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if (step + 1) % eval_every == 0 or step == 0:
            ppl = eval_ppl_stream(feats_inval, log_p_inval)
            if ppl < best_ppl - 1e-3:
                best_ppl = ppl
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
            elapsed = __import__("time").time() - t_total0
            log_fn(f"  step {step + 1:5d}/{steps}  loss={loss.item():.4f}  "
                   f"inval_ppl={ppl:.3f}  best={best_ppl:.3f}  "
                   f"lr={lr_at(step):.2e}  ({elapsed:.0f}s)")
            if no_improve >= patience:
                log_fn(f"  early stop at step {step + 1} (no_improve={no_improve})")
                break

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    return model, best_ppl, step + 1
