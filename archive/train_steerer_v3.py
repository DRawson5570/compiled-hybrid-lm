"""train_steerer_v3.py — Train SuperpositionSteerer with v3 rich builder channels.

Uses pre-computed channel data (dump_gpt2_channels_v3.py output).
15 channels, faster than streaming. Trains steerer + model jointly.
Evals on held-out data.
"""
from __future__ import annotations

import sys, time, math, argparse, importlib.util
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

DEEPSEEK = Path('/home/drawson/deepseek_experiments')
sys.path.insert(0, str(DEEPSEEK))

from hybrid.superposition_steerer import SuperpositionSteerer


def load_neural_lm(ckpt_path, device):
    _spec = importlib.util.spec_from_file_location(
        'train_scaled', str(DEEPSEEK / 'hybrid/train_scaled_neural_lm.py'))
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    DeepCausalLM = _mod.DeepCausalLM

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt['state_dict']
    d_model = state['pos_emb.weight'].shape[-1]
    max_len = state['pos_emb.weight'].shape[0]
    vocab = state.get('tok_emb.weight', state['head_bias']).shape[0]
    if 'encoder.layers.0.linear1.weight' in state:
        d_ff = state['encoder.layers.0.linear1.weight'].shape[0]
        n_layers = len([k for k in state if k.startswith('encoder.layers.') and
                        k.endswith('.norm1.weight')])
        n_heads = state['encoder.layers.0.self_attn.in_proj_weight'].shape[0] // (3 * d_model)
    else:
        d_ff = state['layers.0.ff_1.weight'].shape[0]
        n_layers = len([k for k in state if k.startswith('layers.') and
                        k.endswith('.sa_norm.weight')])
        n_heads = state['layers.0.sa_q.weight'].shape[0] // d_model

    model = DeepCausalLM(vocab=vocab, d_model=d_model, n_layers=n_layers,
                         n_heads=n_heads, d_ff=d_ff, max_len=max_len, dropout=0.0)
    model.load_state_dict(state)
    model = model.to(device)
    return model, d_model


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--neural-ckpt', type=str, required=True)
    p.add_argument('--train-npz', type=str, default='artifacts/gpt2_channels_v3/val.npz')
    p.add_argument('--eval-npz', type=str, default='artifacts/gpt2_channels_v3/eval.npz')
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--steps-per-epoch', type=int, default=1000)
    p.add_argument('--batch', type=int, default=4)
    p.add_argument('--lr', type=float, default=1e-2)
    p.add_argument('--out-dir', type=str, default='artifacts/steerer_v3')
    p.add_argument('--device', type=str, default='cuda')
    args = p.parse_args()

    device = torch.device(args.device)
    out_dir = Path(DEEPSEEK / args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print('=' * 60)
    print(' STEERER TRAINING v3 (15 builder channels)')
    print('=' * 60)

    print('[1/3] Loading neural LM...')
    model, d_model = load_neural_lm(args.neural_ckpt, device)
    for p in model.parameters():
        p.requires_grad = True
    n_params = sum(p.numel() for p in model.parameters())
    print(f'  {n_params:,} params (trainable)')

    print('[2/3] Loading channel data...')
    train_data = np.load(Path(DEEPSEEK / args.train_npz), allow_pickle=True)
    eval_data = np.load(Path(DEEPSEEK / args.eval_npz), allow_pickle=True)
    C = train_data['log_p_targets'].shape[1]
    channel_names = train_data['channel_names'].tolist()
    print(f'  Channels: {C} — {channel_names}')
    print(f'  Train tokens: {len(train_data["targets"]):,}')
    print(f'  Eval tokens: {len(eval_data["targets"]):,}')

    # Exclude shape channel if present
    exclude = set()
    for ci, name in enumerate(channel_names):
        if name == 'shape':
            exclude.add(ci)
    keep_idx = [c for c in range(C) if c not in exclude]
    C_active = len(keep_idx)
    print(f'  Active: {C_active} (excluded {exclude})')

    train_lps = train_data['log_p_targets'][:, keep_idx]
    train_obs = train_data['observed'].astype(np.int64)
    eval_lps = eval_data['log_p_targets'][:, keep_idx]
    eval_obs = eval_data['observed'].astype(np.int64)

    print('[3/3] Building steerer...')
    steerer = SuperpositionSteerer(num_channels=C_active, d_model=d_model,
                                    inject_layers=[0, 4, 8], init_scale=0.01)
    steerer = steerer.to(device)
    n_hooks = steerer.register_hooks(model)
    print(f'  {n_hooks} hooks at layers {steerer.inject_layers}')
    print(f'  Steerer: {sum(p.numel() for p in steerer.parameters()):,} params')

    opt = torch.optim.AdamW([
        {'params': model.parameters(), 'lr': 3e-5},
        {'params': steerer.parameters(), 'lr': args.lr},
    ], weight_decay=0.1)

    N = len(train_obs)
    best_eval_ppl = float('inf')

    for ep in range(1, args.epochs + 1):
        model.train()
        steerer.train()
        total_loss = 0.0
        t0 = time.time()

        for step in range(args.steps_per_epoch):
            starts = torch.randint(0, max(1, N - args.batch), (args.batch,))
            positions = starts.tolist()

            # Get channel weights from first batch item's position
            pos = positions[0]
            channel_lp = torch.tensor(train_lps[pos], dtype=torch.float32)
            w = channel_lp - channel_lp.mean()
            w = torch.softmax(w, dim=0).to(device)
            steerer._current_weights = w

            # Build input/target from observed tokens starting at each position
            seq_len = 64
            max_start = N - seq_len - 1
            if starts.max() + seq_len >= N:
                starts = torch.randint(0, max(1, max_start), (args.batch,))
            x = torch.stack([
                torch.from_numpy(train_obs[s:s+seq_len]) for s in starts
            ]).to(device)
            y = torch.stack([
                torch.from_numpy(train_obs[s+1:s+seq_len+1]) for s in starts
            ]).to(device)

            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(steerer.parameters()), 1.0)
            opt.step()
            total_loss += loss.item()

        # Eval (steerer OFF)
        model.eval()
        steerer.eval()
        steerer._current_weights = None
        eval_N = len(eval_obs)
        with torch.no_grad():
            eval_nll, eval_n = 0.0, 0
            for s in range(0, eval_N - 1, 64):
                cl = min(64, eval_N - s - 1)
                if cl <= 0:
                    continue
                inp = torch.from_numpy(eval_obs[s:s+cl]).unsqueeze(0).to(device)
                tgt = torch.from_numpy(eval_obs[s+1:s+cl+1]).unsqueeze(0).to(device)
                l = model(inp)
                loss_v = F.cross_entropy(l.reshape(-1, l.shape[-1]), tgt.reshape(-1),
                                         reduction='sum')
                eval_nll += loss_v.item()
                eval_n += cl
        eval_ppl = math.exp(eval_nll / max(eval_n, 1))

        avg_loss = total_loss / args.steps_per_epoch
        elapsed = time.time() - t0
        status = ''
        if eval_ppl < best_eval_ppl:
            best_eval_ppl = eval_ppl
            torch.save({
                'state_dict': model.state_dict(),
                'steerer_state': steerer.state_dict(),
                'eval_ppl': eval_ppl,
            }, out_dir / 'steerer_best.pt')
            status = 'SAVED'
        print(f'  epoch={ep:2d}  loss={avg_loss:.4f}  ppl={math.exp(avg_loss):.1f}  '
              f'eval={eval_ppl:.1f}  best={best_eval_ppl:.1f}  {status}  '
              f'time={elapsed:.0f}s', flush=True)

    print(f'\nDone. Best eval PPL: {best_eval_ppl:.1f}')


if __name__ == '__main__':
    main()
