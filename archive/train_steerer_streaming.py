"""train_steerer_streaming.py — Train SuperpositionSteerer with streaming compiled
channels on full train set.

Computes per-channel scalar features on-the-fly (O(1) per token, not O(V)).
Trains steerer + neural LM jointly. Evaluates on held-out val set.
"""
from __future__ import annotations

import sys
from hybrid.config import REPO_ROOT, time, math, argparse, importlib.util
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

DEEPSEEK = Path(__file__).resolve().parent.parent
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
    if 'tok_emb.weight' in state:
        vocab = state['tok_emb.weight'].shape[0]
    else:
        vocab = state['head_bias'].shape[0]

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


class StreamingChannelFeatures:
    """Computes per-channel scalar features on-the-fly without full V-distributions."""

    def __init__(self, V=50257):
        self.V = V
        self._uni_counts = np.zeros(V, dtype=np.float32)
        self._bi_cache = {}
        self._bi_totals = {}
        self._tri_cache = {}
        self._tri_totals = {}
        self._seen_positions = defaultdict(list)
        self._context = []
        self._step = 0
        self._uniform = -math.log(V)
        self._decay_step = 0

    def update(self, token: int):
        tid = int(token)
        self._step += 1
        self._context.append(tid)
        self._context = self._context[-3:]

        if tid < self.V:
            self._uni_counts[tid] += 1

        if len(self._context) >= 2:
            prev, curr = self._context[-2], self._context[-1]
            key = (prev, curr)
            self._bi_cache[key] = self._bi_cache.get(key, 0) + 1
            self._bi_totals[prev] = self._bi_totals.get(prev, 0) + 1

        if len(self._context) >= 3:
            p2, p1, curr = self._context
            key = (p2, p1, curr)
            ctx_key = (p2, p1)
            self._tri_cache[key] = self._tri_cache.get(key, 0) + 1
            self._tri_totals[ctx_key] = self._tri_totals.get(ctx_key, 0) + 1

        self._seen_positions[tid].append(self._step)

    def get_features(self, target: int) -> list[float]:
        tid = int(target)
        feats = []

        d = self._uni_counts.sum() + 0.001 * self.V
        if d > 0 and tid < self.V:
            uni_lp = math.log(max((self._uni_counts[tid] + 0.001) / d, 1e-30))
        else:
            uni_lp = self._uniform
        feats.append(uni_lp)

        if len(self._context) >= 2:
            ctx = tuple(self._context[-2:])
            total = self._tri_totals.get(ctx, 0)
            d = total + 0.001 * self.V
            if d > 0:
                tri_lp = math.log(max((self._tri_cache.get(tuple(list(ctx) + [tid]), 0) + 0.001) / d, 1e-30))
            else:
                tri_lp = self._uniform
        else:
            tri_lp = self._uniform
        feats.append(tri_lp)
        feats.append(tri_lp)

        if len(self._context) >= 1:
            ctx = self._context[-1]
            total = self._bi_totals.get(ctx, 0)
            d = total + 0.001 * self.V
            if d > 0 and tid < self.V:
                bi_lp = math.log(max((self._bi_cache.get((ctx, tid), 0) + 0.001) / d, 1e-30))
            else:
                bi_lp = self._uniform
        else:
            bi_lp = self._uniform
        feats.append(bi_lp)
        feats.append(bi_lp)

        d = self._uni_counts.sum() + 0.001 * self.V
        if d > 0 and tid < self.V:
            uni_decay_lp = math.log(max((self._uni_counts[tid] + 0.001) / d, 1e-30))
        else:
            uni_decay_lp = self._uniform
        feats.append(uni_decay_lp)
        feats.append(uni_decay_lp)

        feats.append(0.0)

        feats.append(uni_decay_lp)

        positions = self._seen_positions.get(tid, [])
        gap = 128 if not positions else min(128, self._step - positions[-1])
        rec_lp = math.log(max(1.0 / max(gap, 1), 1e-30))
        feats.append(rec_lp)

        return feats

    def reset(self):
        self._uni_counts = np.zeros(self.V, dtype=np.float32)
        self._bi_cache = {}
        self._bi_totals = {}
        self._tri_cache = {}
        self._tri_totals = {}
        self._seen_positions = defaultdict(list)
        self._context = []
        self._step = 0
        self._decay_step = 0


def compute_channel_weights(tokens, channels, device, window_avg=16):
    channels.reset()
    feats_per_pos = []
    for t, tid in enumerate(tokens):
        if t > 0:
            channels.update(tokens[t - 1])
        fs = channels.get_features(int(tid))
        feats_per_pos.append(fs)
    if not feats_per_pos:
        return torch.zeros(1, 9, device=device)
    feat_arr = np.array(feats_per_pos, dtype=np.float32)
    if window_avg is not None and window_avg > 1:
        smoothed = np.zeros_like(feat_arr)
        for i in range(feat_arr.shape[0]):
            start = max(0, i - window_avg + 1)
            smoothed[i] = feat_arr[start:i + 1].mean(axis=0)
        feat_arr = smoothed
    active = np.delete(feat_arr, 7, axis=1)
    return torch.tensor(active, device=device)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--neural-ckpt', type=str, required=True)
    p.add_argument('--resume', type=str, default=None, help='Resume from steerer checkpoint')
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--steps-per-epoch', type=int, default=2000)
    p.add_argument('--batch', type=int, default=2)
    p.add_argument('--seq-len', type=int, default=128)
    p.add_argument('--lr', type=float, default=1e-2)
    p.add_argument('--out-dir', type=str, default='artifacts/steerer_stream')
    p.add_argument('--device', type=str, default='cuda')
    args = p.parse_args()

    device = torch.device(args.device)
    out_dir = Path(DEEPSEEK / args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    V = 50257

    print('=' * 60)
    print(' STREAMING STEERER TRAINING')
    print('=' * 60)

    train_ids = torch.load(
        DEEPSEEK / 'artifacts/wikitext_gpt2/train_ids.pt', weights_only=False
    ).long()
    val_ids = torch.load(
        DEEPSEEK / 'artifacts/wikitext_gpt2/validation_ids.pt', weights_only=False
    ).long()
    print(f'Train: {len(train_ids):,}  Val: {len(val_ids):,}')

    print('[load] Neural LM...')
    model, d_model = load_neural_lm(args.neural_ckpt, device)
    for p in model.parameters():
        p.requires_grad = True
    n_params = sum(p.numel() for p in model.parameters())
    print(f'  {n_params:,} params (trainable)')

    steerer = SuperpositionSteerer(num_channels=9, d_model=d_model,
                                    inject_layers=[0, 4, 8], init_scale=0.01)
    steerer = steerer.to(device)

    start_epoch = 0
    best_eval_b = float('inf')
    best_eval_s = float('inf')
    if args.resume:
        resume_ckpt = torch.load(Path(DEEPSEEK / args.resume), map_location=device,
                                 weights_only=False)
        model.load_state_dict(resume_ckpt['state_dict'])
        steerer.load_state_dict(resume_ckpt['steerer_state'], strict=False)
        start_epoch = resume_ckpt.get('epoch', 0)
        best_eval_b = resume_ckpt.get('best_eval_b', float('inf'))
        print(f'[RESUME] epoch {start_epoch}, best_eval_b {best_eval_b:.1f}')

    n_hooks = steerer.register_hooks(model)
    print(f'  Steerer: {n_hooks} hooks, {sum(p.numel() for p in steerer.parameters()):,} params')

    channels = StreamingChannelFeatures(V=V)

    opt = torch.optim.AdamW([
        {'params': model.parameters(), 'lr': 3e-5},
        {'params': steerer.parameters(), 'lr': args.lr},
    ], weight_decay=0.1)

    N = len(train_ids)
    gamma_initial = steerer.gamma.item()
    model_max_len = model.pos_emb.weight.shape[0] - 1

    for ep in range(start_epoch + 1, args.epochs + 1):
        anneal_progress = min(1.0, (ep - 1) / max(args.epochs * 0.7, 1))
        gamma_curr = gamma_initial * (1.0 - anneal_progress)
        with torch.no_grad():
            steerer.gamma.copy_(torch.tensor(gamma_curr))

        model.train()
        steerer.train()
        total_loss = 0.0
        t0 = time.time()

        for step in range(args.steps_per_epoch):
            starts = torch.randint(0, max(1, N - args.seq_len - 1),
                                   (args.batch,))
            x = torch.stack([
                train_ids[s:s + args.seq_len] for s in starts
            ]).to(device)
            y = torch.stack([
                train_ids[s + 1:s + args.seq_len + 1] for s in starts
            ]).to(device)

            all_weights = []
            for b in range(args.batch):
                ctx_tokens = train_ids[starts[b]:starts[b] + args.seq_len].tolist()
                w = compute_channel_weights(ctx_tokens, channels, device, window_avg=16)
                all_weights.append(w)
            weights = torch.stack(all_weights, dim=0)
            steerer.set_weights(weights)

            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, V), y.reshape(-1))
            loss = loss + 0.001 * steerer.orthogonal_penalty()

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(steerer.parameters()), 1.0)
            opt.step()
            total_loss += loss.item()

        model.eval()
        steerer.eval()

        eval_chunk_len = min(args.seq_len, model_max_len, len(val_ids) - 1)
        eval_tokens = val_ids[:eval_chunk_len].tolist()

        steerer._current_weights = None
        with torch.no_grad():
            eval_b_nll, eval_b_n = 0.0, 0
            for s in range(0, len(val_ids) - 1, 64):
                cl = min(64, len(val_ids) - s - 1)
                if cl <= 0:
                    continue
                inp = val_ids[s:s + cl].unsqueeze(0).to(device)
                tgt = val_ids[s + 1:s + cl + 1].unsqueeze(0).to(device)
                l = model(inp)
                loss_v = F.cross_entropy(l.reshape(-1, V), tgt.reshape(-1),
                                         reduction='sum')
                eval_b_nll += loss_v.item()
                eval_b_n += cl
        eval_b = math.exp(eval_b_nll / max(eval_b_n, 1))

        w_eval = compute_channel_weights(eval_tokens, channels, device, window_avg=16)
        w_eval = w_eval.unsqueeze(0)
        steerer.set_weights(w_eval)
        with torch.no_grad():
            inp_s = val_ids[:eval_chunk_len].unsqueeze(0).to(device)
            tgt_s = val_ids[1:eval_chunk_len + 1].unsqueeze(0).to(device)
            l_s = model(inp_s)
            loss_s = F.cross_entropy(l_s.reshape(-1, V), tgt_s.reshape(-1),
                                     reduction='sum')
            eval_s = math.exp(loss_s.item() / eval_chunk_len)

        avg_loss = total_loss / args.steps_per_epoch
        elapsed = time.time() - t0
        status = ''
        if eval_b < best_eval_b:
            best_eval_b = eval_b
            status += 'b'
        if eval_s < best_eval_s:
            best_eval_s = eval_s
            status += 's'
        if 'b' in status:
            torch.save({
                'state_dict': model.state_dict(),
                'steerer_state': steerer.state_dict(),
                'opt_state': opt.state_dict(),
                'eval_b': eval_b,
                'eval_s': eval_s,
                'best_eval_b': best_eval_b,
                'best_eval_s': best_eval_s,
                'epoch': ep,
            }, out_dir / 'steerer_best_b.pt')
        if 's' in status:
            torch.save({
                'state_dict': model.state_dict(),
                'steerer_state': steerer.state_dict(),
                'opt_state': opt.state_dict(),
                'eval_b': eval_b,
                'eval_s': eval_s,
                'best_eval_b': best_eval_b,
                'best_eval_s': best_eval_s,
                'epoch': ep,
            }, out_dir / 'steerer_best_s.pt')
        print(f'  epoch={ep:2d}  loss={avg_loss:.4f}  ppl={math.exp(avg_loss):.1f}  '
              f'eval_s={eval_s:.1f}  eval_b={eval_b:.1f}  '
              f'best_b={best_eval_b:.1f}  [{status}]  '
              f'time={elapsed:.0f}s', flush=True)

    print(f'\nDone. Best eval_b: {best_eval_b:.1f}  Best eval_s: {best_eval_s:.1f}')


if __name__ == '__main__':
    main()
