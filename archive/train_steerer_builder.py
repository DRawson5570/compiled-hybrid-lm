"""train_steerer_builder.py — SuperpositionSteerer using builder counts + light streaming.

Uses GPT2CompiledChannelBuilder for static n-gram channels (O(1) lookup)
plus light streaming decay caches for unigram/bigram (no trigram decay — uses builder).
15 channels, fast per-step, no pre-dump needed.
"""
from __future__ import annotations

import sys, time, math, argparse, importlib.util
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

DEEPSEEK = Path('/home/drawson/deepseek_experiments')
sys.path.insert(0, str(DEEPSEEK))

from hybrid.superposition_steerer import SuperpositionSteerer
from hybrid.compiled_features import GPT2CompiledChannelBuilder


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


class BuilderChannelFeatures:
    """Fast per-token channel features using builder counts + light streaming."""
    
    def __init__(self, builder: GPT2CompiledChannelBuilder, V=50257):
        self.builder = builder
        self.V = V
        self.alpha = builder.cfg.alpha
        self._uni_counts = np.zeros(V, dtype=np.float32)
        self._bi_cache = {}
        self._bi_totals = {}
        self._seen_positions = defaultdict(list)
        self._context = []
        self._step = 0
        self._uniform = -math.log(V)
        
    def update(self, token: int):
        tid = int(token)
        self._step += 1
        self._context.append(tid)
        self._context = self._context[-4:]
        
        # Light unigram decay (every step, but cheap — numpy multiply)
        self._uni_counts *= 0.999
        if tid < self.V:
            self._uni_counts[tid] += 1
        
        # Light bigram decay
        if len(self._context) >= 2:
            prev, curr = self._context[-2], self._context[-1]
            key = (prev, curr)
            self._bi_cache[key] = self._bi_cache.get(key, 0) + 1
            self._bi_totals[prev] = self._bi_totals.get(prev, 0) + 1
            if self._step % 10 == 0:
                for k in list(self._bi_cache):
                    self._bi_cache[k] *= 0.999
                    if self._bi_cache[k] < 1e-6:
                        del self._bi_cache[k]
                for k in list(self._bi_totals):
                    self._bi_totals[k] *= 0.999
                    if self._bi_totals[k] < 1e-6:
                        del self._bi_totals[k]
        
        self._seen_positions[tid].append(self._step)
    
    def _builder_logp(self, counts, context, token):
        """Laplace-smoothed builder log-prob. O(1)."""
        total = sum(counts.values()) if hasattr(counts, 'values') else 0
        d = total + self.alpha * self.V
        if d <= 0:
            return self._uniform
        return math.log(max((counts.get(token, 0) + self.alpha) / d, 1e-7))
    
    def get_features(self, target: int) -> list[float]:
        """15 channel features. O(1) per channel."""
        tid = int(target)
        feats = []
        ctx = self._context
        
        # 0: builder unigram
        feats.append(self._builder_logp(self.builder.unigram, None, tid))
        
        # 1: builder bigram
        if len(ctx) >= 1:
            bi_counts = self.builder.bigram.get(ctx[-1], {})
            feats.append(self._builder_logp(bi_counts, ctx[-1], tid))
        else:
            feats.append(self._uniform)
        
        # 2: builder trigram
        if len(ctx) >= 2:
            tri_counts = self.builder.trigram.get((ctx[-2], ctx[-1]), {})
            feats.append(self._builder_logp(tri_counts, (ctx[-2], ctx[-1]), tid))
        else:
            feats.append(self._uniform)
        
        # 3: builder skip2
        if len(ctx) >= 2:
            s2_counts = self.builder.skip2.get(ctx[-2], {})
            feats.append(self._builder_logp(s2_counts, ctx[-2], tid))
        else:
            feats.append(self._uniform)
        
        # 4: builder skip3
        if len(ctx) >= 3:
            s3_counts = self.builder.skip3.get(ctx[-3], {})
            feats.append(self._builder_logp(s3_counts, ctx[-3], tid))
        else:
            feats.append(self._uniform)
        
        # 5-6: decay unigram fast/slow
        d = self._uni_counts.sum() + 0.001 * self.V
        if d > 0 and tid < self.V:
            uni_lp = math.log(max((self._uni_counts[tid] + 0.001) / d, 1e-7))
        else:
            uni_lp = self._uniform
        feats.append(uni_lp)
        feats.append(uni_lp)
        
        # 7-8: decay bigram fast/slow
        if len(ctx) >= 1:
            total = self._bi_totals.get(ctx[-1], 0)
            d_bi = total + 0.001 * self.V
            if d_bi > 0:
                bi_lp = math.log(max((self._bi_cache.get((ctx[-1], tid), 0) + 0.001) / d_bi, 1e-7))
            else:
                bi_lp = self._uniform
        else:
            bi_lp = self._uniform
        feats.append(bi_lp)
        feats.append(bi_lp)
        
        # 9-10: builder trigram (reuse, no decay trigram to avoid slow cache)
        if len(ctx) >= 2:
            tri_counts = self.builder.trigram.get((ctx[-2], ctx[-1]), {})
            tri_lp_b = self._builder_logp(tri_counts, (ctx[-2], ctx[-1]), tid)
        else:
            tri_lp_b = self._uniform
        feats.append(tri_lp_b)
        feats.append(tri_lp_b)
        
        # 11: shape (placeholder — excluded)
        feats.append(0.0)
        
        # 12: recency
        positions = self._seen_positions.get(tid, [])
        gap = 128 if not positions else min(128, self._step - positions[-1])
        feats.append(math.log(max(1.0 / max(gap, 1), 1e-7)))
        
        # 13: builder entropy (approximate — use unigram entropy)
        d_uni = sum(self.builder.unigram.values()) + self.alpha * self.V
        prob = (self.builder.unigram.get(tid, 0) + self.alpha) / d_uni
        feats.append(-prob * math.log(max(prob, 1e-7)))
        
        return feats
    
    def reset(self):
        self._uni_counts = np.zeros(self.V, dtype=np.float32)
        self._bi_cache = {}
        self._bi_totals = {}
        self._seen_positions = defaultdict(list)
        self._context = []
        self._step = 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--neural-ckpt', type=str, required=True)
    p.add_argument('--builder', type=str, default='artifacts/compiled_builder_50m.pt')
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--steps-per-epoch', type=int, default=500)
    p.add_argument('--batch', type=int, default=2)
    p.add_argument('--lr', type=float, default=1e-2)
    p.add_argument('--out-dir', type=str, default='artifacts/steerer_builder')
    p.add_argument('--device', type=str, default='cuda')
    args = p.parse_args()

    device = torch.device(args.device)
    out_dir = Path(DEEPSEEK / args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print('=' * 60)
    print(' STEERER TRAINING (builder counts + streaming)')
    print('=' * 60)

    print('[load] Builder...')
    builder = GPT2CompiledChannelBuilder.load(
        str(Path(DEEPSEEK / args.builder)))
    print(f'  {builder.total_tokens:,} tokens')

    train_ids = torch.load(
        DEEPSEEK / 'artifacts/wikitext_gpt2/train_ids.pt', weights_only=False
    ).long()
    val_ids = torch.load(
        DEEPSEEK / 'artifacts/wikitext_gpt2/validation_ids.pt', weights_only=False
    ).long()
    print(f'  Train: {len(train_ids):,}  Val: {len(val_ids):,}')

    print('[load] Neural LM...')
    model, d_model = load_neural_lm(args.neural_ckpt, device)
    for p in model.parameters():
        p.requires_grad = True
    n_params = sum(p.numel() for p in model.parameters())
    print(f'  {n_params:,} params (trainable)')

    EXCLUDE = {11}  # shape channel
    C_active = 13  # 14 total - 1 excluded
    steerer = SuperpositionSteerer(num_channels=C_active, d_model=d_model,
                                    inject_layers=[0, 4, 8], init_scale=0.01)
    steerer = steerer.to(device)
    n_hooks = steerer.register_hooks(model)
    print(f'  Steerer: {n_hooks} hooks, {sum(p.numel() for p in steerer.parameters()):,} params (14 channels)')

    channels = BuilderChannelFeatures(builder)

    opt = torch.optim.AdamW([
        {'params': model.parameters(), 'lr': 3e-5},
        {'params': steerer.parameters(), 'lr': args.lr},
    ], weight_decay=0.1)

    N = len(train_ids)
    best_eval_ppl = float('inf')

    for ep in range(1, args.epochs + 1):
        model.train()
        steerer.train()
        total_loss = 0.0
        t0 = time.time()
        channels.reset()

        for step in range(args.steps_per_epoch):
            starts = torch.randint(0, max(1, N - args.batch), (args.batch,))
            x = torch.stack([
                train_ids[s:s+64] for s in starts
            ]).to(device)
            y = torch.stack([
                train_ids[s+1:s+65] for s in starts
            ]).to(device)

            # Seed channels from first batch item context
            context = train_ids[starts[0]:starts[0]+64].tolist()
            for tid in context[:-1]:
                channels.update(tid)

            # Get features for first target token
            feats = channels.get_features(context[1])
            active = [f for i, f in enumerate(feats) if i not in EXCLUDE]
            w = torch.tensor(active, dtype=torch.float32, device=device)
            w = w - w.mean()
            w = torch.softmax(w, dim=0)
            steerer._current_weights = w

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
        with torch.no_grad():
            eval_nll, eval_n = 0.0, 0
            for s in range(0, len(val_ids) - 1, 64):
                cl = min(64, len(val_ids) - s - 1)
                if cl <= 0:
                    continue
                inp = val_ids[s:s+cl].unsqueeze(0).to(device)
                tgt = val_ids[s+1:s+cl+1].unsqueeze(0).to(device)
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
