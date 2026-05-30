"""train_steerer.py — Train SuperpositionSteerer using pre-computed channel data.

Uses val.npz from dump_gpt2_channels_v2.py for compiled channel features.
Freezes neural LM, trains only steerer vectors.
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
    vocab = state.get('tok_emb.weight', state.get('head_bias', torch.zeros(1))).shape[0]
    if 'encoder.layers.0.linear1.weight' in state:
        d_ff = state['encoder.layers.0.linear1.weight'].shape[0]
        n_layers = len([k for k in state if k.startswith('encoder.layers.') and k.endswith('.norm1.weight')])
        n_heads = state['encoder.layers.0.self_attn.in_proj_weight'].shape[0] // (3 * d_model)
    else:
        d_ff = state['layers.0.ff_1.weight'].shape[0]
        n_layers = len([k for k in state if k.startswith('layers.') and k.endswith('.sa_norm.weight')])
        n_heads = state['layers.0.sa_q.weight'].shape[0] // d_model

    model = DeepCausalLM(vocab=vocab, d_model=d_model, n_layers=n_layers,
                         n_heads=n_heads, d_ff=d_ff, max_len=max_len, dropout=0.0)
    model.load_state_dict(state)
    model = model.to(device)
    model.eval()
    return model, d_model


def compute_channel_weights(log_p_observed, log_p_lag1, entropy, max_log_prob, keep_indices):
    """Compute per-channel scalar weights from pre-computed channel features.
    Uses distribution peakedness (max - mean log-prob) as confidence signal.
    """
    confidences = []
    for ci in range(len(keep_indices)):
        lp = torch.tensor(log_p_observed[-1, ci:ci+1])  # latest position
        scalar = float(lp.max() - lp.mean())
        confidences.append(max(scalar, 0.01))
    w = torch.tensor(confidences, dtype=torch.float32)
    return w / w.sum()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--neural-ckpt', type=str, required=True)
    p.add_argument('--channel-npz', type=str, default='artifacts/gpt2_channels_v2/val.npz')
    p.add_argument('--eval-npz', type=str, default='artifacts/gpt2_channels_v2/eval.npz')
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--steps-per-epoch', type=int, default=1000)
    p.add_argument('--batch', type=int, default=4)
    p.add_argument('--seq-len', type=int, default=128)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--out-dir', type=str, default='artifacts/steerer_v1')
    p.add_argument('--device', type=str, default='cuda')
    args = p.parse_args()

    device = torch.device(args.device)
    out_dir = Path(DEEPSEEK / args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print('=' * 60)
    print(' SUPEPOSITION STEERER TRAINING (pre-computed channels)')
    print('=' * 60)

    # Load neural LM (trainable)
    print('[1/3] Loading neural LM (trainable)...')
    model, d_model = load_neural_lm(args.neural_ckpt, device)
    for p in model.parameters():
        p.requires_grad = True  # unfreeze — learn to compose with steering
    n_params = sum(p.numel() for p in model.parameters())
    print(f'  {n_params:,} params (trainable)')

    # Load pre-computed channel data
    print('[2/3] Loading channel data...')
    data = np.load(Path(DEEPSEEK / args.channel_npz), allow_pickle=True)
    all_observed = data['observed']  # token IDs
    C = data['log_p_observed'].shape[1]
    channel_names = data['channel_names'].tolist()
    exclude = {7}  # shape channel
    keep_idx = [c for c in range(C) if c not in exclude]
    C_active = len(keep_idx)
    print(f'  Channels: {C} → {C_active} active (excluded {exclude})')
    print(f'  Tokens: {len(all_observed):,}')

    log_p_observed = data['log_p_observed'][:, keep_idx]
    log_p_lag1 = data['log_p_lag1'][:, keep_idx]
    entropy = data['entropy'][:, keep_idx]
    max_log_prob = data['max_log_prob'][:, keep_idx]

    # Load eval data
    eval_data = np.load(Path(DEEPSEEK / args.eval_npz), allow_pickle=True)
    eval_observed = eval_data['observed']
    eval_T = len(eval_observed)
    print(f'  Eval tokens: {eval_T:,}')

    # Create steerer
    print('[3/3] Building steerer...')
    steerer = SuperpositionSteerer(num_channels=C_active, d_model=d_model,
                                    inject_layers=[0, 4, 8], init_scale=0.01)
    steerer = steerer.to(device)
    n_hooks = steerer.register_hooks(model)
    print(f'  {n_hooks} hooks at layers {steerer.inject_layers}')
    print(f'  Steerer: {sum(p.numel() for p in steerer.parameters()):,} params')
    print(f'  Total trainable: {n_params + sum(p.numel() for p in steerer.parameters()):,} params')

    opt = torch.optim.AdamW([
        {'params': model.parameters(), 'lr': 3e-5},  # gentle — model already trained
        {'params': steerer.parameters(), 'lr': args.lr},  # faster — steerer is new
    ], weight_decay=0.1)
    N = len(all_observed)
    best_loss = float('inf')

    for ep in range(1, args.epochs + 1):
        steerer.train()
        total_loss = 0.0
        t0 = time.time()

        for step in range(args.steps_per_epoch):
            # Sample random positions for batch
            starts = torch.randint(0, max(1, N - args.seq_len - 1),
                                   (args.batch,))
            inputs = torch.stack([
                torch.from_numpy(all_observed[s:s + args.seq_len].astype(np.int64))
                for s in starts
            ]).to(device)
            targets = torch.stack([
                torch.from_numpy(all_observed[s + 1:s + args.seq_len + 1].astype(np.int64))
                for s in starts
            ]).to(device)

            # Compute channel weights from mid-sequence position
            mid_pos = min(starts[0].item() + args.seq_len // 2, N - 1)
            weights = compute_channel_weights(
                log_p_observed, log_p_lag1, entropy, max_log_prob, keep_idx
            )
            steerer._current_weights = weights.to(device)

            # Forward through neural LM (hooks inject steering)
            logits = model(inputs)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                targets.reshape(-1)
            )

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(steerer.parameters()), 1.0)
            opt.step()
            total_loss += loss.item()

            if step % 200 == 0:
                steerer._current_weights = None  # clear for val-like step

        avg_loss = total_loss / args.steps_per_epoch
        elapsed = time.time() - t0

        # Eval on held-out set
        steerer.eval()
        model.eval()
        eval_nll = 0.0
        eval_n = 0
        with torch.no_grad():
            eval_seq_len = 64
            for s in range(0, eval_T - 1, eval_seq_len):
                cl = min(eval_seq_len, eval_T - s - 1)
                if cl <= 0:
                    continue
                inp = torch.from_numpy(eval_observed[s:s+cl].astype(np.int64)).unsqueeze(0).to(device)
                tgt = torch.from_numpy(eval_observed[s+1:s+cl+1].astype(np.int64)).unsqueeze(0).to(device)
                # Eval WITHOUT steerer — measure generalization of trained weights
                steerer._current_weights = None
                logits = model(inp)
                loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), tgt.reshape(-1), reduction='sum')
                eval_nll += loss.item()
                eval_n += cl
        eval_ppl = math.exp(eval_nll / max(eval_n, 1))
        steerer._current_weights = None
        steerer.train()
        model.train()

        status = ''
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'state_dict': model.state_dict(),
                'steerer_state': steerer.state_dict(),
                'loss': avg_loss, 'eval_ppl': eval_ppl,
            }, out_dir / 'steerer_best.pt')
            status = 'SAVED'
        print(f'  epoch={ep:2d}  loss={avg_loss:.4f}  ppl={math.exp(avg_loss):.1f}  '
              f'eval_ppl={eval_ppl:.1f}  best={best_loss:.4f}  {status}  time={elapsed:.0f}s', flush=True)

    print(f'\nDone. Best loss: {best_loss:.4f}')
    print(f'  Weights: {out_dir / "steerer_best.pt"}')


if __name__ == '__main__':
    main()
