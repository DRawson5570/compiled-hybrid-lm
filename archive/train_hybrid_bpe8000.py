"""train_hybrid_bpe8000.py — Train BPE-8000 neural LM, blend with compiled WindowMLP.

The compiled 21-channel blender lives in BPE-8000 space (PPL=11.6).
We train neural LMs in the same BPE-8000 space and blend honestly.
Pushes the hybrid thesis: compiled + neural → better than either alone.

Uses PPMI+SVD embedding init, OneCycleLR, standard next-token CE loss.
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


# ═══════════════════════════════════════════════════════════════════════════════
# Neural LM (same DeepCausalLM, V=8000)
# ═══════════════════════════════════════════════════════════════════════════════

class BPE8000LM(nn.Module):
    def __init__(self, vocab=8000, d_model=256, n_layers=6, n_heads=8,
                 d_ff=1024, max_len=256, dropout=0.1):
        super().__init__()
        self.vocab = vocab
        self.d_model = d_model
        self.max_len = max_len
        self.tok_emb = nn.Embedding(vocab, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.drop = nn.Dropout(dropout)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, activation='gelu', batch_first=True, norm_first=True,
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

    def transplant_embeddings(self, emb: torch.Tensor):
        if emb.shape[1] != self.d_model:
            return  # dimension mismatch, skip
        with torch.no_grad():
            self.tok_emb.weight.copy_(emb.float())

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        B, T = ids.shape
        assert T <= self.max_len
        pos = torch.arange(T, device=ids.device).unsqueeze(0).expand(B, T)
        x = self.tok_emb(ids) + self.pos_emb(pos)
        x = self.drop(x)
        mask = nn.Transformer.generate_square_subsequent_mask(T, device=ids.device)
        x = self.encoder(x, mask=mask, is_causal=True)
        x = self.ln_f(x)
        return x @ self.tok_emb.weight.T + self.head_bias


# ═══════════════════════════════════════════════════════════════════════════════
# Training
# ═══════════════════════════════════════════════════════════════════════════════

def iter_batches(ids, batch_size, seq_len, device, generator):
    N = len(ids)
    max_start = N - seq_len - 1
    offsets = torch.arange(seq_len + 1)
    while True:
        starts = torch.randint(0, max(1, max_start), (batch_size,), generator=generator)
        idx = starts.unsqueeze(1) + offsets.unsqueeze(0)
        span = ids[idx].to(device)
        yield span[:, :-1], span[:, 1:]


@torch.no_grad()
def eval_ppl(model, ids, seq_len, device):
    model.eval()
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


# ═══════════════════════════════════════════════════════════════════════════════
# Blended eval with compiled WindowMLP
# ═══════════════════════════════════════════════════════════════════════════════

def load_compiled_blender():
    _blender_mod = importlib.util.spec_from_file_location(
        'blender_v3', str(DEEPSEEK / 'hybrid/v3_super_blender/model.py'))
    _bm = importlib.util.module_from_spec(_blender_mod)
    _blender_mod.loader.exec_module(_bm)
    WindowMLPBlender = _bm.WindowMLPBlender

    _util = importlib.util.spec_from_file_location(
        'blender_util', str(DEEPSEEK / 'hybrid/v1_blender/blender_model.py'))
    _bu = importlib.util.module_from_spec(_util)
    _util.loader.exec_module(_bu)
    build_feature_matrix = _bu.build_feature_matrix

    _bpe, _vocab, _tok2id, _bpe_to_lm, emb, V, d = load_setup()
    emb = emb.float()

    eval_npz = np.load(str(DEEPSEEK / 'hybrid/v3_super_blender/data_real_v33/eval.npz'), allow_pickle=True)
    eval_feat = build_feature_matrix(
        torch.tensor(eval_npz['log_p_observed']), torch.tensor(eval_npz['log_p_lag1']),
        torch.tensor(eval_npz['entropy']), torch.tensor(eval_npz['max_log_prob']),
        emb, torch.tensor(eval_npz['observed']), use_embedding=True
    ).float()
    ckpt = torch.load(str(DEEPSEEK / 'hybrid/v3_super_blender/saved_models_v33/blender_window_mlp.pt'), map_location='cpu')
    blender = WindowMLPBlender(eval_feat.shape[1], 21, lookback_window=16, hidden=256, dropout=0.1, init_uniform=False)
    blender.load_state_dict(ckpt['state_dict'])
    blender.eval()

    with torch.no_grad():
        win = blender.build_windowed_features(eval_feat)
        log_w = blender(win, is_already_windowed=True)
    compiled_lp = torch.logsumexp(log_w + torch.tensor(eval_npz['log_p_targets']), dim=-1).numpy()

    eval_observed = torch.from_numpy(eval_npz['observed'].astype(np.int64))
    print(f'  Compiled WindowMLP PPL: {math.exp(-compiled_lp.mean()):.4f}')
    return compiled_lp, eval_observed


@torch.no_grad()
def eval_blended(model, neural_ids, compiled_lp, seq_len, device, alphas=(0.0, 0.3, 0.5, 0.7, 0.9, 1.0)):
    model.eval()
    N = len(neural_ids)
    results = {a: {'nll': 0.0, 'tokens': 0} for a in alphas}

    for s in range(0, N - 1, seq_len):
        in_end = min(s + seq_len, N)
        inp = neural_ids[s:in_end].unsqueeze(0).to(device)
        L = inp.shape[1]
        logits = model(inp)
        lp = F.log_softmax(logits[0], dim=-1)
        for i in range(L - 1):
            pos = s + i
            target = int(neural_ids[pos + 1])
            lp_neural = lp[i, target].item()
            lp_comp = float(compiled_lp[min(pos, len(compiled_lp) - 1)])
            for alpha in alphas:
                if alpha == 0.0:
                    lp_mix = lp_comp
                elif alpha == 1.0:
                    lp_mix = lp_neural
                else:
                    la, l1a = math.log(alpha), math.log(1 - alpha)
                    lp_mix = float(torch.logsumexp(
                        torch.tensor([la + lp_comp, l1a + lp_neural]), dim=0))
                results[alpha]['nll'] += -lp_mix
                results[alpha]['tokens'] += 1

    return {a: math.exp(v['nll'] / v['tokens']) for a, v in results.items()}


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--steps-per-epoch', type=int, default=2000)
    p.add_argument('--batch', type=int, default=16)
    p.add_argument('--seq-len', type=int, default=128)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--d-model', type=int, default=256)
    p.add_argument('--n-layers', type=int, default=12)
    p.add_argument('--n-heads', type=int, default=8)
    p.add_argument('--d-ff', type=int, default=1024)
    p.add_argument('--dropout', type=float, default=0.1)
    p.add_argument('--warmup-steps', type=int, default=500)
    p.add_argument('--out-dir', type=str, default='artifacts/hybrid_bpe8000')
    p.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = p.parse_args()

    device = torch.device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(42)
    np.random.seed(42)
    gen = torch.Generator().manual_seed(42)

    print('=' * 60)
    print(' BPE-8000 HYBRID: Neural LM + Compiled WindowMLP')
    print('=' * 60)

    # ── Load data ──
    print('[1/4] Loading BPE-8000 tokens and PPMI embeddings...')
    ids_all = load_or_build_tokens(None, None, None).long()
    V = 8000
    train_ids = ids_all[:22_000_000]
    _bpe, _vocab, _tok2id, _bpe_to_lm, emb, _V, d = load_setup()
    emb = emb.float()
    print(f'  Train: {len(train_ids):,} tokens  V={V}  PPMI emb: {emb.shape}')

    # ── Build model ──
    print(f'[2/4] Building BPE8000LM (d_model={args.d_model}, n_layers={args.n_layers})...')
    model = BPE8000LM(
        vocab=V, d_model=args.d_model, n_layers=args.n_layers,
        n_heads=args.n_heads, d_ff=args.d_ff,
        max_len=args.seq_len + 1, dropout=args.dropout,
    )
    model.transplant_embeddings(emb)
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'  Params: {n_params:,}')

    # ── Train ──
    print('[3/4] Training...')
    opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1, betas=(0.9, 0.95))
    total_steps = args.epochs * args.steps_per_epoch
    scheduler = optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=total_steps,
        pct_start=min(args.warmup_steps / max(total_steps, 1), 0.4),
    )
    batcher = iter_batches(train_ids, args.batch, args.seq_len, device, gen)

    best_val_ppl = float('inf')
    train_log = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()
        for _ in range(args.steps_per_epoch):
            x, y = next(batcher)
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, V), y.reshape(-1))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            scheduler.step()
            epoch_loss += float(loss.item())

        # Val on last 30K tokens
        val_ids = ids_all[22_000_000:22_030_000]
        val_nll, _ = eval_ppl(model, val_ids, args.seq_len, device)
        val_ppl = math.exp(val_nll)
        elapsed = time.time() - t0
        print(f'  epoch={epoch:2d}/{args.epochs}  loss={epoch_loss / args.steps_per_epoch:.4f}  '
              f'val_ppl={val_ppl:.2f}  lr={scheduler.get_last_lr()[0]:.2e}  time={elapsed:.0f}s', flush=True)
        train_log.append({'epoch': epoch, 'loss': epoch_loss / args.steps_per_epoch, 'val_ppl': val_ppl})
        if val_ppl < best_val_ppl:
            best_val_ppl = val_ppl
            torch.save({'epoch': epoch, 'state_dict': model.state_dict(), 'val_ppl': val_ppl},
                       out_dir / 'best.pt')

    # ── Blend with compiled ──
    print('[4/4] Blending with compiled WindowMLP...')
    ckpt = torch.load(out_dir / 'best.pt', map_location=device)
    model.load_state_dict(ckpt['state_dict'])
    compiled_lp, eval_obs = load_compiled_blender()

    neural_nll, _ = eval_ppl(model, eval_obs, args.seq_len, device)
    neural_ppl = math.exp(neural_nll)
    comp_ppl = math.exp(-compiled_lp.mean())
    blend_results = eval_blended(model, eval_obs, compiled_lp, args.seq_len, device)
    best_alpha = min(blend_results, key=lambda a: blend_results[a])

    print(f'\n  Compiled  only: PPL={comp_ppl:.2f}')
    print(f'  Neural    only: PPL={neural_ppl:.2f}')
    for a in sorted(blend_results):
        if a not in (0.0, 1.0):
            print(f'  Blend a={a:.1f}:  PPL={blend_results[a]:.2f}')
    print(f'  Best blend (a={best_alpha}): PPL={blend_results[best_alpha]:.2f}')

    report = {'model': 'BPE8000LM', 'params': n_params,
              'd_model': args.d_model, 'n_layers': args.n_layers,
              'neural_ppl': neural_ppl, 'compiled_ppl': comp_ppl,
              'best_blend_ppl': blend_results[best_alpha],
              'best_alpha': best_alpha, 'blend_results': blend_results}
    with open(out_dir / 'report.json', 'w') as f:
        json.dump(report, f, indent=2)
    print(f'  Report: {out_dir / "report.json"}')


if __name__ == '__main__':
    main()
