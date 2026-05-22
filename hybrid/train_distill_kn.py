"""train_distill_kn.py — Distill Kneser-Ney n-gram into a neural LM via KL divergence.

Architecture #3 from HYBRID_STRATEGY.md: the compiled KN model provides a
per-token full probability distribution as the teacher. The neural student
learns to match it via KL divergence — absorbing n-gram statistics directly
without memorizing count tables.

Uses BPE-8000 vocabulary to avoid the tokenizer mismatch with the compiled KN.
The KN model's prob_vector(history) → (8000,) gives a calibrated distribution
at every position.
"""
from __future__ import annotations

import argparse, math, pickle, sys, time, json, importlib.util
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

DEEPSEEK = Path('/home/drawson/deepseek_experiments')
LLM_DECOUPLING = Path('/home/drawson/llm_decoupling')
sys.path.insert(0, str(LLM_DECOUPLING))

from compile_wiki_lm_v13 import load_or_build_tokens
from compile_wiki_lm_v23 import ModifiedKNGram  # needed for pickle.load


# ═══════════════════════════════════════════════════════════════════════════════
# Student model — same DeepCausalLM, BPE-8000 vocab
# ═══════════════════════════════════════════════════════════════════════════════

class StudentLM(nn.Module):
    def __init__(self, vocab: int = 8000, d_model: int = 256, n_layers: int = 6,
                 n_heads: int = 8, d_ff: int = 1024, max_len: int = 128,
                 dropout: float = 0.1):
        super().__init__()
        self.vocab = vocab
        self.d_model = d_model
        self.max_len = max_len
        self.tok_emb = nn.Embedding(vocab, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.drop = nn.Dropout(dropout)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, activation='gelu', batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.ln_f = nn.LayerNorm(d_model)
        self.head_bias = nn.Parameter(torch.zeros(vocab))

        nn.init.normal_(self.tok_emb.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.pos_emb.weight, mean=0.0, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        B, T = ids.shape
        assert T <= self.max_len, (T, self.max_len)
        pos = torch.arange(T, device=ids.device).unsqueeze(0).expand(B, T)
        x = self.tok_emb(ids) + self.pos_emb(pos)
        x = self.drop(x)
        mask = nn.Transformer.generate_square_subsequent_mask(T, device=ids.device)
        x = self.encoder(x, mask=mask, is_causal=True)
        x = self.ln_f(x)
        return x @ self.tok_emb.weight.T + self.head_bias


# ═══════════════════════════════════════════════════════════════════════════════
# Teacher — Kneser-Ney n-gram model
# ═══════════════════════════════════════════════════════════════════════════════

class KNTeacher:
    """Wraps ModifiedKNGram to provide per-position (V,) probability vectors."""

    def __init__(self, kn, ids_np: np.ndarray):
        self.kn = kn
        self.ids = ids_np.astype(np.int32)
        self.N = kn.N

    def get_probs(self, positions: np.ndarray) -> np.ndarray:
        """Return (len(positions), V) float32 teacher probabilities."""
        V = self.kn.V
        probs = np.zeros((len(positions), V), dtype=np.float32)
        for i, t in enumerate(positions):
            history = tuple(int(x) for x in self.ids[max(0, t - self.N + 1):t])
            p = self.kn.prob_vector(history)
            s = p.sum()
            if s > 0:
                p = p / s
            else:
                p = np.ones(V) / V
            probs[i] = p.astype(np.float32)
        return probs


# ═══════════════════════════════════════════════════════════════════════════════
# Training
# ═══════════════════════════════════════════════════════════════════════════════

def iter_batches_bpe8000(ids: np.ndarray, batch_size: int, seq_len: int,
                         device, generator: torch.Generator):
    """Yield (input_ids, target_ids) from random spans of BPE-8000 tokens."""
    N = len(ids)
    max_start = N - seq_len - 1
    offsets = torch.arange(seq_len + 1)
    while True:
        starts = torch.randint(0, max(1, max_start), (batch_size,), generator=generator)
        idx = starts.unsqueeze(1) + offsets.unsqueeze(0)
        span = torch.from_numpy(ids[idx.numpy()]).long()
        yield span[:, :-1].to(device), span[:, 1:].to(device), starts.numpy()


@torch.no_grad()
def eval_ppl_bpe8000(model, ids_np: np.ndarray, seq_len: int, device):
    """Standard sliding-window PPL on BPE-8000 tokens."""
    model.eval()
    ids = torch.from_numpy(ids_np.astype(np.int64)).long()
    total_nll = 0.0
    total_tokens = 0

    for start in range(0, max(0, len(ids) - 1), seq_len):
        chunk_len = min(seq_len, len(ids) - start - 1)
        if chunk_len <= 0:
            continue
        inp = ids[start:start + chunk_len].unsqueeze(0).to(device)
        tgt = ids[start + 1:start + chunk_len + 1].unsqueeze(0).to(device)
        logits = model(inp)
        loss = F.cross_entropy(logits.reshape(-1, model.vocab), tgt.reshape(-1), reduction='sum')
        total_nll += float(loss.item())
        total_tokens += chunk_len

    return total_nll / max(total_tokens, 1), total_tokens


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--kn-pickle', type=str,
                   default=str(LLM_DECOUPLING / 'artifacts/compiled_wiki_lm_v23/kn5_22m.pkl'))
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--steps-per-epoch', type=int, default=2000)
    p.add_argument('--batch', type=int, default=16)
    p.add_argument('--seq-len', type=int, default=64)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--d-model', type=int, default=256)
    p.add_argument('--n-layers', type=int, default=6)
    p.add_argument('--n-heads', type=int, default=8)
    p.add_argument('--d-ff', type=int, default=1024)
    p.add_argument('--dropout', type=float, default=0.1)
    p.add_argument('--out-dir', type=str, default='artifacts/distill_kn')
    p.add_argument('--device', type=str,
                   default='cuda' if torch.cuda.is_available() else 'cpu')
    args = p.parse_args()

    device = torch.device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(42)
    np.random.seed(42)
    gen = torch.Generator().manual_seed(42)

    print('=' * 60)
    print(' DISTILLATION: KN Teacher → Neural Student')
    print('=' * 60)

    # ── Load tokens and KN model ──
    print('[1/5] Loading BPE-8000 tokens and KN teacher...')
    ids_all = load_or_build_tokens(None, None, None).long().numpy().astype(np.int32)
    V = 8000
    print(f'  {len(ids_all):,} tokens, V={V}')

    train_ids = ids_all[:22_000_000]

    # Load or build KN teacher
    kn = None
    if Path(args.kn_pickle).exists():
        try:
            with open(args.kn_pickle, 'rb') as f:
                kn = pickle.load(f)
            print(f'  KN N={kn.N} loaded from pickle')
        except Exception as e:
            print(f'  KN pickle failed ({e}), rebuilding...')
    if kn is None:
        print(f'  Building KN-5 from {22_000_000:,} training tokens...')
        from compile_wiki_lm_v23 import ModifiedKNGram
        kn = ModifiedKNGram(5, 8000)
        kn.build(train_ids[:22_000_000])

    teacher = KNTeacher(kn, ids_all)

    # Val on last 100K BPE-8000 tokens (same relative position as eval.npz)
    val_ids = ids_all[-100000:]

    # ── Build student ──
    print('[2/5] Building student model...')
    model = StudentLM(
        vocab=V, d_model=args.d_model, n_layers=args.n_layers,
        n_heads=args.n_heads, d_ff=args.d_ff,
        max_len=args.seq_len + 1, dropout=args.dropout,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'  params={n_params:,}')

    # ── Train ──
    print('[3/5] Training with KL distillation...')
    opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1, betas=(0.9, 0.95))
    total_steps = args.epochs * args.steps_per_epoch
    scheduler = optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=total_steps,
        pct_start=min(500 / max(total_steps, 1), 0.4),
    )
    batcher = iter_batches_bpe8000(train_ids, args.batch, args.seq_len, device, gen)

    best_val_ppl = float('inf')
    train_log = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        for step in range(args.steps_per_epoch):
            inputs, targets, starts = next(batcher)
            B, T = inputs.shape

            # Get teacher distributions for this batch
            teacher_probs = []
            for b in range(B):
                for t in range(T):
                    pos = int(starts[b]) + t + 1  # position of target token
                    teacher_probs.append(teacher.get_probs(np.array([pos]))[0])

            teacher_t = torch.from_numpy(np.stack(teacher_probs)).float().to(device)
            teacher_t = teacher_t.view(B, T, V)

            # Student forward
            logits = model(inputs)
            student_logp = F.log_softmax(logits, dim=-1)

            # KL divergence: sum_y teacher[y] * (log teacher[y] - log student[y])
            # = -sum_y teacher[y] * log student[y] + H(teacher)
            # H(teacher) is constant, so we minimize cross-entropy with teacher targets
            loss = -(teacher_t * student_logp).sum(dim=-1).mean()

            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            scheduler.step()
            epoch_loss += float(loss.item())

        # Evaluate
        val_nll, val_tokens = eval_ppl_bpe8000(model, val_ids, args.seq_len, device)
        val_ppl = math.exp(val_nll)
        elapsed = time.time() - t0
        print(f'  epoch={epoch:2d}/{args.epochs}  '
              f'kl_loss={epoch_loss / args.steps_per_epoch:.4f}  '
              f'val_ppl={val_ppl:.2f}  '
              f'lr={scheduler.get_last_lr()[0]:.2e}  '
              f'time={elapsed:.0f}s', flush=True)

        train_log.append({'epoch': epoch, 'kl_loss': epoch_loss / args.steps_per_epoch,
                          'val_ppl': val_ppl})

        if val_ppl < best_val_ppl:
            best_val_ppl = val_ppl
            torch.save({'epoch': epoch, 'state_dict': model.state_dict(),
                        'val_ppl': val_ppl, 'args': vars(args)},
                       out_dir / 'distill_best.pt')

    # ── Eval ──
    print('[4/5] Final evaluation...')
    ckpt = torch.load(out_dir / 'distill_best.pt', map_location=device)
    model.load_state_dict(ckpt['state_dict'])
    test_nll, test_tokens = eval_ppl_bpe8000(model, val_ids, args.seq_len, device)
    test_ppl = math.exp(test_nll)

    print(f'\n  KN teacher PPL:   89.88 (on eval slice)')
    print(f'  Student PPL:      {test_ppl:.2f}')
    print(f'  WindowMLP blend:  11.62 (21-channel, for reference)')

    print('[5/5] Saving report...')
    report = {
        'model': 'StudentLM (KN-distilled)', 'params': n_params,
        'train_log': train_log,
        'teacher_ppl': 89.88, 'student_ppl': test_ppl,
        'blend_reference_ppl': 11.62,
    }
    with open(out_dir / 'distill_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    print(f'  Report: {out_dir / "distill_report.json"}')


if __name__ == '__main__':
    main()
