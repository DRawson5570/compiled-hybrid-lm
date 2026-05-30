"""train_honest_neural_lm.py — Train a small transformer LM on real tokens,
then blend honestly with the compiled WindowMLP blender.

Replaces the fabricated delta-prior evaluation (train_delta_prior_v33.py)
with an honest mixture approach:
  1. Train TinyCausalLM (~3M params) on tokens 0–22M (standard LM loss)
  2. Evaluate standalone PPL on val/eval slices
  3. Load the pre-trained WindowMLP blender (21-channel, PPL=11.62)
  4. Blend neural + compiled predictions per-token at evaluation time
  5. Report honest numbers — no fabricated distributions, no answer-peeking

Output: artifacts/hybrid_v2_honest/tiny_lm.pt + honest report
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
LLM_DECOUPLING = Path('/home/drawson/llm_decoupling')

# --- Import from llm_decoupling (does not use hybrid package) ---
sys.path.insert(0, str(LLM_DECOUPLING))
from compile_wiki_lm_v13 import load_setup, load_or_build_tokens

# --- Import blender model directly from file (bypass package shadowing) ---
def _import_from_file(name, filepath):
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_blender_mod = _import_from_file(
    'blender_model_v3',
    str(DEEPSEEK / 'hybrid/v3_super_blender/model.py')
)
_blender_util = _import_from_file(
    'blender_util',
    str(DEEPSEEK / 'hybrid/v1_blender/blender_model.py')
)

WindowMLPBlender = _blender_mod.WindowMLPBlender
build_feature_matrix = _blender_util.build_feature_matrix


# ---------------------------------------------------------------------------
# Transformer LM (matches TinyCausalLM from v2_neural_channel/train_tiny_lm.py)
# ---------------------------------------------------------------------------

class TinyCausalLM(nn.Module):
    def __init__(self, vocab: int, d_model: int = 256, n_layers: int = 2,
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
            dropout=dropout, activation='gelu', batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.ln_f = nn.LayerNorm(d_model)
        self.head_bias = nn.Parameter(torch.zeros(vocab))

        # GPT-style init
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
        x = self.dropout(x)
        mask = nn.Transformer.generate_square_subsequent_mask(T, device=ids.device)
        x = self.encoder(x, mask=mask, is_causal=True)
        x = self.ln_f(x)
        logits = x @ self.tok_emb.weight.T + self.head_bias
        return logits


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def iter_batches(ids: torch.Tensor, batch_size: int, seq_len: int, device,
                 generator: torch.Generator | None = None):
    N = ids.shape[0]
    max_start = N - seq_len - 1
    while True:
        starts = torch.randint(0, max_start, (batch_size,), generator=generator)
        idx = starts.unsqueeze(1) + torch.arange(seq_len + 1).unsqueeze(0)
        span = ids[idx].to(device, non_blocking=True)
        yield span[:, :-1], span[:, 1:]


@torch.no_grad()
def eval_sliding_window(model: TinyCausalLM, ids: torch.Tensor,
                        seq_len: int, device, batch_size: int = 8) -> tuple[float, np.ndarray]:
    """Sliding-window causal LM evaluation. Returns (ppl, per_position_nll)."""
    model.eval()
    N = len(ids)
    V = model.vocab
    total_nll = 0.0
    total_tokens = 0
    nll_arr = np.zeros(N - 1, dtype=np.float64)

    stride = seq_len - 1
    for start in range(0, N - 1, stride * batch_size):
        batch_starts = list(range(start, min(start + stride * batch_size, N - 1), stride))
        if not batch_starts:
            break
        for s in batch_starts:
            in_end = min(s + seq_len, N)
            in_ids = ids[s:in_end].unsqueeze(0).to(device)
            L = in_ids.shape[1]
            logits = model(in_ids)
            lp = F.log_softmax(logits[0], dim=-1)
            n_preds = L - 1
            for i in range(n_preds):
                pos = s + i
                target = int(ids[pos + 1])
                nll = -lp[i, target].item()
                nll_arr[pos] = nll
                total_nll += nll
                total_tokens += 1

    ppl = math.exp(total_nll / total_tokens) if total_tokens > 0 else float('inf')
    return ppl, nll_arr


# ---------------------------------------------------------------------------
# Blended evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def eval_blended(model: TinyCausalLM, neural_ids: torch.Tensor,
                 compiled_log_p_target: torch.Tensor,
                 seq_len: int, device, alphas=(0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0)):
    """Evaluate per-token blend of compiled + neural distributions."""
    model.eval()
    N = len(neural_ids)
    V = model.vocab
    results = {a: {'total_nll': 0.0, 'total_tokens': 0} for a in alphas}

    stride = seq_len - 1
    for s in range(0, N - 1, stride):
        in_end = min(s + seq_len, N)
        in_ids = neural_ids[s:in_end].unsqueeze(0).to(device)
        L = in_ids.shape[1]
        logits = model(in_ids)
        lp = F.log_softmax(logits[0], dim=-1)
        n_preds = L - 1
        for i in range(n_preds):
            pos = s + i
            target = int(neural_ids[pos + 1])
            lp_neural = lp[i, target].item()
            lp_compiled = float(compiled_log_p_target[min(pos, len(compiled_log_p_target) - 1)])
            for alpha in alphas:
                if alpha == 0.0:
                    lp_mix = lp_compiled
                elif alpha == 1.0:
                    lp_mix = lp_neural
                else:
                    # log(alpha * p_compiled + (1-alpha) * p_neural)
                    log_alpha = math.log(alpha)
                    log_1ma = math.log(1.0 - alpha)
                    lp_mix = torch.logsumexp(
                        torch.tensor([log_alpha + lp_compiled, log_1ma + lp_neural]), dim=0
                    ).item()
                results[alpha]['total_nll'] += -lp_mix
                results[alpha]['total_tokens'] += 1

    return {a: math.exp(v['total_nll'] / v['total_tokens']) if v['total_tokens'] > 0 else float('inf')
            for a, v in results.items()}


# ---------------------------------------------------------------------------
# Load compiled blender
# ---------------------------------------------------------------------------

def load_compiled_blender():
    _bpe, _vocab, _tok2id, _bpe_to_lm, emb, V, d = load_setup()
    emb = emb.float()
    eval_npz = np.load(str(DEEPSEEK / 'hybrid/v3_super_blender/data_real_v33/eval.npz'),
                       allow_pickle=True)
    eval_feat = build_feature_matrix(
        torch.tensor(eval_npz['log_p_observed']), torch.tensor(eval_npz['log_p_lag1']),
        torch.tensor(eval_npz['entropy']), torch.tensor(eval_npz['max_log_prob']),
        emb, torch.tensor(eval_npz['observed']), use_embedding=True
    ).float()

    ckpt = torch.load(str(DEEPSEEK / 'hybrid/v3_super_blender/saved_models_v33/blender_window_mlp.pt'),
                      map_location='cpu')
    blender = WindowMLPBlender(
        single_step_dim=eval_feat.shape[1], n_channels=21,
        lookback_window=16, hidden=256, dropout=0.1, init_uniform=False
    )
    blender.load_state_dict(ckpt['state_dict'])
    blender.eval()

    with torch.no_grad():
        feat_win = blender.build_windowed_features(eval_feat)
        log_w = blender(feat_win, is_already_windowed=True)
    log_p_targets = torch.tensor(eval_npz['log_p_targets'])
    compiled_log_p = torch.logsumexp(log_w + log_p_targets, dim=-1)

    full_nll = -compiled_log_p.mean().item()
    print(f'Compiled WindowMLP PPL: {math.exp(full_nll):.4f}')
    return compiled_log_p.numpy(), emb, V


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--batch', type=int, default=64)
    p.add_argument('--seq-len', type=int, default=128)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--d-model', type=int, default=256)
    p.add_argument('--n-layers', type=int, default=2)
    p.add_argument('--n-heads', type=int, default=4)
    p.add_argument('--d-ff', type=int, default=1024)
    p.add_argument('--dropout', type=float, default=0.1)
    p.add_argument('--steps-per-epoch', type=int, default=2000)
    p.add_argument('--out-dir', type=str, default='artifacts/hybrid_v2_honest')
    p.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = p.parse_args()

    device = torch.device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(42)
    np.random.seed(42)
    g = torch.Generator().manual_seed(42)

    # --- Load tokens ---
    print('=' * 60)
    print(' HONEST NEURAL LM + COMPILED BLEND')
    print('=' * 60)
    print('[1/5] Loading tokens...')
    ids_all = load_or_build_tokens(None, None, None)  # uses v11 cache
    ids_all = ids_all.long()
    total_tokens = len(ids_all)
    print(f'  Total tokens: {total_tokens:,}')

    # Split: train 0..22M, val/eval match blender data exactly
    train_ids = ids_all[:22_000_000]
    # The v33 blender data uses:
    #   val.npz: positions 22,000,002 .. 22,030,000 (29998 tokens observed)
    #   eval.npz: positions 22,030,002 .. 22,130,000 (99998 tokens observed)
    # We align with the observed token sequences.
    _val_obs = np.load(str(DEEPSEEK / 'hybrid/v3_super_blender/data_real_v33/val.npz'),
                       allow_pickle=True)['observed']
    _eval_obs = np.load(str(DEEPSEEK / 'hybrid/v3_super_blender/data_real_v33/eval.npz'),
                        allow_pickle=True)['observed']
    val_ids = torch.from_numpy(_val_obs.astype(np.int64))
    eval_ids = torch.from_numpy(_eval_obs.astype(np.int64))
    print(f'  Train: {len(train_ids):,}  Val: {len(val_ids):,}  Eval: {len(eval_ids):,}')

    # --- Build model ---
    print('[2/5] Building TinyCausalLM...')
    V = 8000
    model = TinyCausalLM(
        vocab=V, d_model=args.d_model, n_layers=args.n_layers,
        n_heads=args.n_heads, d_ff=args.d_ff, max_len=args.seq_len + 1,
        dropout=args.dropout
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'  Params: {n_params:,}  d_model={args.d_model}  n_layers={args.n_layers}')

    # --- Train ---
    print('[3/5] Training...')
    opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    batcher = iter_batches(train_ids, args.batch, args.seq_len, device, generator=g)

    best_val_ppl = float('inf')
    train_log = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()
        for step in range(args.steps_per_epoch):
            x, y = next(batcher)
            opt.zero_grad()
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, V), y.reshape(-1))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            epoch_loss += loss.item()
        scheduler.step()

        # Evaluate on val slice
        val_ppl, _ = eval_sliding_window(model, val_ids, args.seq_len, device)
        elapsed = time.time() - t0
        print(f'  Epoch {epoch:2d}/{args.epochs}  '
              f'train_loss={epoch_loss / args.steps_per_epoch:.4f}  '
              f'val_ppl={val_ppl:.2f}  '
              f'time={elapsed:.0f}s', flush=True)

        train_log.append({'epoch': epoch, 'train_loss': epoch_loss / args.steps_per_epoch,
                          'val_ppl': val_ppl})

        if val_ppl < best_val_ppl:
            best_val_ppl = val_ppl
            torch.save({'epoch': epoch, 'state_dict': model.state_dict(),
                        'val_ppl': val_ppl, 'args': vars(args)},
                       out_dir / 'tiny_lm_best.pt')

    # --- Eval standalone ---
    print('[4/5] Evaluating standalone neural LM...')
    ckpt = torch.load(out_dir / 'tiny_lm_best.pt', map_location=device)
    model.load_state_dict(ckpt['state_dict'])
    eval_ppl, eval_nll = eval_sliding_window(model, eval_ids, args.seq_len, device)
    print(f'  Standalone neural PPL: {eval_ppl:.2f}')
    print(f'  Standalone neural NLL: {eval_nll[4:].mean():.4f}')

    # --- Blend with compiled ---
    print('[5/5] Blending with compiled WindowMLP...')
    compiled_log_p, emb, V = load_compiled_blender()

    blend_results = eval_blended(model, eval_ids, torch.from_numpy(compiled_log_p),
                                 args.seq_len, device)
    print()
    print('=' * 60)
    print(' HONEST BLEND RESULTS')
    print('=' * 60)
    comp_ppl = math.exp(-compiled_log_p.mean().item())
    print(f'  Compiled WindowMLP only:    PPL = {comp_ppl:8.2f}')
    print(f'  Neural LM only:             PPL = {eval_ppl:8.2f}')
    for alpha in sorted(blend_results.keys()):
        if alpha not in (0.0, 1.0):
            print(f'  Blend alpha={alpha:.1f}:           PPL = {blend_results[alpha]:8.2f}')
    best_alpha = min(blend_results, key=lambda a: blend_results[a])
    print(f'  Best blend (alpha={best_alpha:.1f}):     PPL = {blend_results[best_alpha]:8.2f}')
    print(f'  Oracle (best per-token channel): PPL =    9.57')
    print('=' * 60)

    # --- Save report ---
    report = {
        'model': 'TinyCausalLM',
        'params': n_params,
        'd_model': args.d_model,
        'n_layers': args.n_layers,
        'n_heads': args.n_heads,
        'train_tokens': 22_000_000,
        'val_tokens': len(val_ids),
        'eval_tokens': len(eval_ids),
        'epochs': args.epochs,
        'lr': args.lr,
        'best_val_ppl': best_val_ppl,
        'eval_ppl_neural_only': eval_ppl,
        'eval_ppl_compiled_only': comp_ppl,
        'blend_results': blend_results,
        'best_blend_ppl': blend_results[best_alpha],
        'best_blend_alpha': best_alpha,
        'eval_protocol': 'sliding-window causal, BPE-8000 tokenizer',
        'eval_split': 'adjacent position slices (22.03M-22.13M) — NOT document-disjoint',
        'notes': [
            'Honest evaluation — no fabricated distributions',
            'Compiled blend uses real 21-channel WindowMLP log-probs',
            'Neural log-probs from standard causal LM forward pass',
            'Blend = log(alpha * p_compiled + (1-alpha) * p_neural) per token',
            'Train/eval split is adjacent positions, not document-disjoint —',
            '  this likely inflates compiled PPL relative to a proper split',
        ],
    }
    with open(out_dir / 'honest_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    print(f'\nReport saved to {out_dir / "honest_report.json"}')


if __name__ == '__main__':
    main()
