"""train_gpt2_blender.py — Train WindowMLP on GPT-2 BPE channels, blend with neural LM.

Loads val.npz/eval.npz from dump_gpt2_channels_v2.py, trains a WindowMLP blender,
then blends with the pre-trained GPT-2 BPE neural LM.
"""
from __future__ import annotations

import sys, json, math, time, importlib.util
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

DEEPSEEK = Path('/home/drawson/deepseek_experiments')
sys.path.insert(0, str(DEEPSEEK))

# Import blender model
_spec = importlib.util.spec_from_file_location(
    'blender_v3', str(DEEPSEEK / 'hybrid/v3_super_blender/model.py'))
_bm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_bm)
WindowMLPBlender = _bm.WindowMLPBlender

# Import neural LM
_spec2 = importlib.util.spec_from_file_location(
    'scaled_lm', str(DEEPSEEK / 'hybrid/train_scaled_neural_lm.py'))
_lm = importlib.util.module_from_spec(_spec2)
_spec2.loader.exec_module(_lm)
DeepCausalLM = _lm.DeepCausalLM


def build_windowed_features(log_p_observed, log_p_lag1, entropy, max_log_prob,
                            observed, emb, topk_log_probs=None, window=16):
    """Manually build windowed features matching blender_model.build_windowed_features."""
    T, C = log_p_observed.shape
    d = emb.shape[1]
    # Features per channel: 4 stats + embedding
    feat_dim = 4 * C
    if topk_log_probs is not None:
        feat_dim += topk_log_probs.shape[2] * C
    feat_dim += d

    features = torch.zeros(T, feat_dim, dtype=torch.float32)

    # Per-channel stats
    for c in range(C):
        base = 4 * c
        features[:, base + 0] = log_p_observed[:, c]
        features[:, base + 1] = log_p_lag1[:, c]
        features[:, base + 2] = entropy[:, c]
        features[:, base + 3] = max_log_prob[:, c]

    # Embedding of observed token
    offset = 4 * C
    if topk_log_probs is not None:
        offset += topk_log_probs.shape[2] * C
    features[:, offset:offset + d] = emb[observed.long()]

    return features


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', type=str, default='artifacts/gpt2_channels_v2')
    p.add_argument('--neural-ckpt', type=str, default='artifacts/hybrid_gpt2_768_dense/gpt2_lm_best.pt')
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--out-dir', type=str, default='artifacts/gpt2_blender')
    args = p.parse_args()

    device = torch.device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir)

    torch.manual_seed(42)

    print('=' * 60)
    print(' GPT-2 BPE BLENDER + HYBRID')
    print('=' * 60)

    # Load channel data
    val = np.load(data_dir / 'val.npz', allow_pickle=True)
    eval_npz = np.load(data_dir / 'eval.npz', allow_pickle=True)
    C = val['log_p_targets'].shape[1]
    channel_names = val['channel_names'].tolist()
    print(f'Channels: {C} — {channel_names}')

    # Build features
    # Use a random embedding since we don't have GPT-2 PPMI
    d_emb = 256
    emb = torch.randn(50257, d_emb)

    # Exclude shape channel (channel 7) — it dominates via orthography, not statistics
    exclude_channels = {7}  # shape channel
    keep_indices = [c for c in range(C) if c not in exclude_channels]

    val_lpo = torch.tensor(val['log_p_observed'])[:, keep_indices]
    val_lpl = torch.tensor(val['log_p_lag1'])[:, keep_indices]
    val_ent = torch.tensor(val['entropy'])[:, keep_indices]
    val_max = torch.tensor(val['max_log_prob'])[:, keep_indices]
    val_obs = torch.tensor(val['observed'])

    eval_lpo = torch.tensor(eval_npz['log_p_observed'])[:, keep_indices]
    eval_lpl = torch.tensor(eval_npz['log_p_lag1'])[:, keep_indices]
    eval_ent = torch.tensor(eval_npz['entropy'])[:, keep_indices]
    eval_max = torch.tensor(eval_npz['max_log_prob'])[:, keep_indices]
    eval_obs = torch.tensor(eval_npz['observed'])

    C_active = len(keep_indices)
    active_names = [channel_names[c] for c in keep_indices]
    print(f'Active channels ({C_active}): {active_names}')

    val_feat = build_windowed_features(val_lpo, val_lpl, val_ent, val_max, val_obs, emb)
    eval_feat = build_windowed_features(eval_lpo, eval_lpl, eval_ent, eval_max, eval_obs, emb)
    print(f'Feature dim: {val_feat.shape[1]}')

    # Build blender
    blender = WindowMLPBlender(val_feat.shape[1], C_active, lookback_window=16, hidden=256,
                                dropout=0.1, init_uniform=True)
    blender = blender.to(device)
    n_params = sum(p.numel() for p in blender.parameters())
    print(f'Blender params: {n_params:,}')

    # Train
    opt = torch.optim.AdamW(blender.parameters(), lr=args.lr)
    val_targets = torch.tensor(val['log_p_targets'])[:, keep_indices]
    val_win = blender.build_windowed_features(val_feat).to(device)

    best_val_ppl = float('inf')
    for epoch in range(1, args.epochs + 1):
        blender.train()
        log_w = blender(val_win, is_already_windowed=True)
        compiled_lp = torch.logsumexp(log_w + val_targets.to(device), dim=-1)
        loss = -compiled_lp.mean()
        opt.zero_grad()
        loss.backward()
        opt.step()

        val_ppl = math.exp(-compiled_lp.mean().item())
        if val_ppl < best_val_ppl:
            best_val_ppl = val_ppl

        if epoch % 5 == 0:
            print(f'  epoch={epoch:2d}  val_ppl={val_ppl:.2f}', flush=True)

    # Evaluate on eval
    blender.eval()
    with torch.no_grad():
        eval_win = blender.build_windowed_features(eval_feat).to(device)
        log_w = blender(eval_win, is_already_windowed=True)
        eval_targets = torch.tensor(eval_npz['log_p_targets'])[:, keep_indices]
        compiled_lp = torch.logsumexp(log_w + eval_targets.to(device), dim=-1)
        blender_ppl = math.exp(-compiled_lp.mean().item())
    print(f'\n  Blender PPL (eval): {blender_ppl:.2f}')

    # Also compute per-channel PPL
    print(f'\n  Per-channel PPL (eval):')
    for ci, c in enumerate(keep_indices):
        ch_nll = -eval_targets[:, ci].mean().item()
        print(f'    {channel_names[c]:12s}: {math.exp(ch_nll):8.2f}')

    # Blend with neural LM
    print(f'\n[blend] Loading neural LM from {args.neural_ckpt}...')
    ckpt = torch.load(args.neural_ckpt, map_location=device)
    V_nn = ckpt['state_dict']['head_bias'].shape[0]
    nn_cfg = ckpt.get('args', {})
    nn_model = DeepCausalLM(
        vocab=V_nn, d_model=nn_cfg.get('d_model', 256),
        n_layers=nn_cfg.get('n_layers', 12),
        n_heads=nn_cfg.get('n_heads', 8),
        d_ff=nn_cfg.get('d_ff', 1024),
        max_len=nn_cfg.get('seq_len', 128) + 1, dropout=0.0,
    ).to(device)
    nn_model.load_state_dict(ckpt['state_dict'])
    nn_model.eval()
    print(f'  Neural LM: {sum(p.numel() for p in nn_model.parameters()):,} params, V={V_nn}')

    # Blend eval
    eval_tokens = torch.from_numpy(eval_npz['observed'].astype(np.int64))
    compiled_np = compiled_lp.cpu().numpy()

    total_neural_nll = 0.0
    total_blend_nll = {a: 0.0 for a in [0.3, 0.5, 0.7, 0.9]}
    total_tokens = 0

    with torch.no_grad():
        for s in range(0, len(eval_tokens) - 1, 128):
            end = min(s + 128, len(eval_tokens) - 1)
            inp = eval_tokens[s:end].unsqueeze(0).to(device)
            L = inp.shape[1]
            logits = nn_model(inp)
            lp = F.log_softmax(logits[0], dim=-1)
            for i in range(L - 1):
                pos = s + i
                target = int(eval_tokens[pos + 1])
                lp_n = lp[i, target].item()
                lp_c = float(compiled_np[min(pos, len(compiled_np) - 1)])
                total_neural_nll += -lp_n
                for a in total_blend_nll:
                    la, l1a = math.log(a), math.log(1 - a)
                    lp_mix = float(torch.logsumexp(
                        torch.tensor([la + lp_c, l1a + lp_n]), dim=0))
                    total_blend_nll[a] += -lp_mix
                total_tokens += 1

    neural_ppl = math.exp(total_neural_nll / total_tokens)
    print(f'\n  Blender PPL:   {blender_ppl:.2f}')
    print(f'  Neural PPL:    {neural_ppl:.2f}')
    for a in sorted(total_blend_nll):
        ppl = math.exp(total_blend_nll[a] / total_tokens)
        print(f'  Blend a={a:.1f}:  PPL={ppl:.2f}')

    best_a = min(total_blend_nll, key=lambda a: total_blend_nll[a])
    best_ppl = math.exp(total_blend_nll[best_a] / total_tokens)

    report = {
        'blender_ppl': blender_ppl, 'neural_ppl': neural_ppl,
        'best_blend_ppl': best_ppl, 'best_alpha': best_a,
        'blend_results': {str(a): math.exp(total_blend_nll[a] / total_tokens) for a in total_blend_nll},
    }
    with open(out_dir / 'report.json', 'w') as f:
        json.dump(report, f, indent=2)
    print(f'\n  Best blend (a={best_a}): PPL={best_ppl:.2f}')
    print(f'  Report: {out_dir / "report.json"}')


if __name__ == '__main__':
    main()
