"""train_scaled_neural_lm.py — Train a deeper transformer LM (11.8M params)
with PPMI+SVD embedding initialization, honest evaluation, and blending
with the compiled WindowMLP blender.

Follows ROADMAP.md Phase 3 architecture:
  - 12-layer decoder Transformer, d_model=256, n_heads=8, d_ff=1024
  - PPMI+SVD embedding transplant from compiled v5 embeddings
  - Weight tying (head shares embedding weights)
  - Trained on 22M BPE-8000 tokens, evaluated on held-out slices
  - Blended honestly with compiled 21-channel WindowMLP blender

Usage:
    python hybrid/train_scaled_neural_lm.py --epochs 20 --batch 32 --lr 3e-4
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
sys.path.insert(0, str(LLM_DECOUPLING))
from compile_wiki_lm_v13 import load_setup, load_or_build_tokens

# Import blender model directly from file
def _import_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_blender_mod = _import_file('blender_v3', str(DEEPSEEK / 'hybrid/v3_super_blender/model.py'))
_blender_util = _import_file('blender_util', str(DEEPSEEK / 'hybrid/v1_blender/blender_model.py'))
WindowMLPBlender = _blender_mod.WindowMLPBlender
build_feature_matrix = _blender_util.build_feature_matrix


# ---------------------------------------------------------------------------
# Deep Transformer LM
# ---------------------------------------------------------------------------

class DeepCausalLM(nn.Module):
    """12-layer decoder-only transformer with weight tying and Pre-LN."""
    def __init__(self, vocab: int, d_model: int = 256, n_layers: int = 12,
                 n_heads: int = 8, d_ff: int = 1024, max_len: int = 256,
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

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.tok_emb.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.pos_emb.weight, mean=0.0, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def transplant_embeddings(self, emb: torch.Tensor):
        """Initialize token embeddings from PPMI+SVD matrix."""
        with torch.no_grad():
            self.tok_emb.weight.copy_(emb.float())

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
    N = len(ids)
    max_start = N - seq_len - 1
    while True:
        starts = torch.randint(0, max_start, (batch_size,), generator=generator)
        idx = starts.unsqueeze(1) + torch.arange(seq_len + 1).unsqueeze(0)
        span = ids[idx].to(device, non_blocking=True)
        yield span[:, :-1], span[:, 1:]


@torch.no_grad()
def eval_sliding_window(model, ids: torch.Tensor, seq_len: int,
                        device, batch_size: int = 8):
    model.eval()
    N = len(ids)
    V = model.vocab
    total_nll = 0.0
    total_tokens = 0

    stride = seq_len - 1
    for s in range(0, N - 1, stride):
        in_end = min(s + seq_len, N)
        in_ids = ids[s:in_end].unsqueeze(0).to(device)
        L = in_ids.shape[1]
        logits = model(in_ids)
        lp = F.log_softmax(logits[0], dim=-1)
        n_preds = L - 1
        for i in range(n_preds):
            target = int(ids[s + i + 1])
            nll = -lp[i, target].item()
            total_nll += nll
            total_tokens += 1

    return total_nll / total_tokens, total_tokens


@torch.no_grad()
def eval_blended(model, neural_ids: torch.Tensor,
                 compiled_log_p_target: torch.Tensor,
                 seq_len: int, device, alphas=(0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0)):
    model.eval()
    N = len(neural_ids)
    results = {a: {'total_nll': 0.0, 'total_tokens': 0} for a in alphas}

    stride = seq_len - 1
    for s in range(0, N - 1, stride):
        in_end = min(s + seq_len, N)
        in_ids = neural_ids[s:in_end].unsqueeze(0).to(device)
        L = in_ids.shape[1]
        logits = model(in_ids)
        lp = F.log_softmax(logits[0], dim=-1)
        for i in range(L - 1):
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
                    log_alpha = math.log(alpha)
                    log_1ma = math.log(1.0 - alpha)
                    lp_mix = torch.logsumexp(
                        torch.tensor([log_alpha + lp_compiled, log_1ma + lp_neural]),
                        dim=0
                    ).item()
                results[alpha]['total_nll'] += -lp_mix
                results[alpha]['total_tokens'] += 1

    return {a: math.exp(v['total_nll'] / v['total_tokens'])
            for a, v in results.items()}


# ---------------------------------------------------------------------------
# Compiled blender
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

    print(f'Compiled WindowMLP PPL: {math.exp(-compiled_log_p.mean().item()):.4f}')
    return compiled_log_p.numpy(), emb, V


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--batch', type=int, default=32)
    p.add_argument('--seq-len', type=int, default=128)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--d-model', type=int, default=256)
    p.add_argument('--n-layers', type=int, default=12)
    p.add_argument('--n-heads', type=int, default=8)
    p.add_argument('--d-ff', type=int, default=1024)
    p.add_argument('--dropout', type=float, default=0.1)
    p.add_argument('--steps-per-epoch', type=int, default=2000)
    p.add_argument('--warmup-steps', type=int, default=500)
    p.add_argument('--grad-accum', type=int, default=1)
    p.add_argument('--use-ppmi-init', action='store_true', default=True)
    p.add_argument('--out-dir', type=str, default='artifacts/hybrid_v2_scaled')
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
    print(' SCALED NEURAL LM (11.8M) + COMPILED BLEND')
    print('=' * 60)
    print('[1/6] Loading tokens and PPMI embeddings...')
    ids_all = load_or_build_tokens(None, None, None).long()
    _val_obs = np.load(str(DEEPSEEK / 'hybrid/v3_super_blender/data_real_v33/val.npz'),
                       allow_pickle=True)['observed']
    _eval_obs = np.load(str(DEEPSEEK / 'hybrid/v3_super_blender/data_real_v33/eval.npz'),
                        allow_pickle=True)['observed']

    train_ids = ids_all[:22_000_000]
    val_ids = torch.from_numpy(_val_obs.astype(np.int64))
    eval_ids = torch.from_numpy(_eval_obs.astype(np.int64))

    V = 8000
    print(f'  Train: {len(train_ids):,}  Val: {len(val_ids):,}  Eval: {len(eval_ids):,}')

    # --- Load PPMI embeddings for init ---
    _bpe, _vocab, _tok2id, _bpe_to_lm, emb, _V, d = load_setup()
    emb = emb.float()
    print(f'  PPMI embeddings: {emb.shape}')

    # --- Build model ---
    print('[2/6] Building DeepCausalLM (12-layer, ~11.8M params)...')
    model = DeepCausalLM(
        vocab=V, d_model=args.d_model, n_layers=args.n_layers,
        n_heads=args.n_heads, d_ff=args.d_ff, max_len=args.seq_len + 1,
        dropout=args.dropout
    )
    if args.use_ppmi_init:
        model.transplant_embeddings(emb)
        print('  PPMI+SVD embedding transplant applied')
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'  Params: {n_params:,}')

    # --- Train ---
    print('[3/6] Training...')
    opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1,
                      betas=(0.9, 0.95))
    total_train_steps = args.epochs * args.steps_per_epoch
    warmup_frac = min(args.warmup_steps / max(total_train_steps, 1), 0.4)
    scheduler = optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr,
        total_steps=total_train_steps,
        pct_start=warmup_frac
    )
    batcher = iter_batches(train_ids, args.batch, args.seq_len, device, generator=g)

    best_val_ppl = float('inf')
    train_log = []
    total_steps = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()
        opt.zero_grad()

        for step in range(args.steps_per_epoch):
            x, y = next(batcher)
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, V), y.reshape(-1))
            loss = loss / args.grad_accum
            loss.backward()

            if (step + 1) % args.grad_accum == 0:
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                scheduler.step()
                opt.zero_grad()

            epoch_loss += loss.item() * args.grad_accum
            total_steps += 1

        val_nll, val_tokens = eval_sliding_window(model, val_ids, args.seq_len, device)
        val_ppl = math.exp(val_nll)
        elapsed = time.time() - t0
        print(f'  Epoch {epoch:2d}/{args.epochs}  '
              f'train_loss={epoch_loss / args.steps_per_epoch:.4f}  '
              f'val_ppl={val_ppl:.2f}  '
              f'lr={scheduler.get_last_lr()[0]:.2e}  '
              f'time={elapsed:.0f}s', flush=True)

        train_log.append({'epoch': epoch, 'train_loss': epoch_loss / args.steps_per_epoch,
                          'val_ppl': val_ppl})

        if val_ppl < best_val_ppl:
            best_val_ppl = val_ppl
            torch.save({'epoch': epoch, 'state_dict': model.state_dict(),
                        'val_ppl': val_ppl, 'args': vars(args)},
                       out_dir / 'scaled_lm_best.pt')

    # --- Eval standalone ---
    print('[4/6] Evaluating standalone neural LM...')
    ckpt = torch.load(out_dir / 'scaled_lm_best.pt', map_location=device)
    model.load_state_dict(ckpt['state_dict'])
    eval_nll, eval_tokens = eval_sliding_window(model, eval_ids, args.seq_len, device)
    eval_ppl = math.exp(eval_nll)
    print(f'  Standalone neural PPL: {eval_ppl:.2f}  (over {eval_tokens:,} tokens)')

    # --- Blend with compiled ---
    print('[5/6] Blending with compiled WindowMLP...')
    compiled_log_p, _, _ = load_compiled_blender()

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
    print('[6/6] Saving report...')
    report = {
        'model': 'DeepCausalLM',
        'params': n_params,
        'd_model': args.d_model, 'n_layers': args.n_layers,
        'n_heads': args.n_heads, 'd_ff': args.d_ff,
        'train_tokens': 22_000_000,
        'val_tokens': len(val_ids), 'eval_tokens': len(eval_ids),
        'epochs': args.epochs, 'lr': args.lr,
        'ppmi_init': args.use_ppmi_init,
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
            'PPMI+SVD embedding initialization applied',
            'OneCycleLR with warmup',
            '12-layer transformer, weight-tied embeddings',
        ],
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / 'scaled_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    print(f'Report saved to {out_dir / "scaled_report.json"}')


if __name__ == '__main__':
    main()
