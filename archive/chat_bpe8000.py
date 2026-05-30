"""chat_bpe8000.py — Interactive raw-text continuation with BPE-8000 + SuperpositionSteerer.
Loads joint model+steerer checkpoint for correct weight synchronization.
"""
import sys
from pathlib import Path
import importlib.util

import torch
import numpy as np
from collections import defaultdict
import math

DEEPSEEK = Path('/home/drawson/deepseek_experiments')
LLM = Path('/home/drawson/llm_decoupling')
sys.path.insert(0, str(DEEPSEEK))
sys.path.append(str(LLM))

from hybrid.superposition_steerer import SuperpositionSteerer
from compile_wiki_lm_v13 import load_setup


def build_model(vocab, d_model, n_layers, n_heads, d_ff, max_len, device):
    _spec = importlib.util.spec_from_file_location(
        'bpe8000', str(DEEPSEEK / 'hybrid/train_hybrid_bpe8000.py'))
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    BPE8000LM = _mod.BPE8000LM
    return BPE8000LM(vocab=vocab, d_model=d_model, n_layers=n_layers,
                     n_heads=n_heads, d_ff=d_ff, max_len=max_len, dropout=0.0)


class LiveChannelFeatures:
    def __init__(self, V=8000, emb=None):
        self.V = V
        self.emb = emb
        self.reset()

    def update(self, token: int):
        tid = int(token)
        self._step += 1
        self._context.append(tid)
        if len(self._context) > 256:
            self._context = self._context[-256:]
        self._uni_counts *= 0.999
        if tid < self.V:
            self._uni_counts[tid] += 1
        if len(self._context) >= 2:
            prev, curr = self._context[-2], self._context[-1]
            key = (prev, curr)
            self._bi_cache[key] = self._bi_cache.get(key, 0) + 1
            self._bi_totals[prev] = self._bi_totals.get(prev, 0) + 1
        if len(self._context) >= 3:
            p2, p1, c = self._context[-3], self._context[-2], self._context[-1]
            key = (p2, p1, c)
            ctx_key = (p2, p1)
            self._tri_cache[key] = self._tri_cache.get(key, 0) + 1
            self._tri_totals[ctx_key] = self._tri_totals.get(ctx_key, 0) + 1
        self._seen_positions[tid].append(self._step)

    def get_features(self, target: int) -> list[float]:
        tid = int(target); ctx = self._context; V = self.V; uniform = -math.log(V)
        d = self._uni_counts.sum() + 0.001 * V
        uni_lp = math.log(max((self._uni_counts[tid] + 0.001) / d, 1e-7)) if d > 0 and tid < V else uniform

        bi_lp = uniform
        if len(ctx) >= 1:
            total = self._bi_totals.get(ctx[-1], 0); db = total + 0.001 * V
            if db > 0: bi_lp = math.log(max((self._bi_cache.get((ctx[-1], tid), 0) + 0.001) / db, 1e-7))

        tri_lp = uniform
        if len(ctx) >= 2:
            ck = (ctx[-2], ctx[-1]); total = self._tri_totals.get(ck, 0); dt = total + 0.001 * V
            if dt > 0: tri_lp = math.log(max((self._tri_cache.get((ctx[-2], ctx[-1], tid), 0) + 0.001) / dt, 1e-7))

        skip_lp = uniform
        if len(ctx) >= 2:
            sk = ctx[-2]; total = self._bi_totals.get(sk, 0); ds = total + 0.001 * V
            if ds > 0: skip_lp = math.log(max((self._bi_cache.get((sk, tid), 0) + 0.001) / ds, 1e-7))

        positions = self._seen_positions.get(tid, [])
        gap = 128 if not positions else min(128, self._step - positions[-1])
        rec_lp = math.log(max(1.0 / max(gap, 1), 1e-7))

        ppmi_cos = 0.0; ppmi_max_cos = 0.0; ppmi_norm = 0.0
        if self.emb is not None and len(ctx) >= 1:
            te = self.emb[tid].float().numpy()
            ce = np.stack([self.emb[t].float().numpy() for t in ctx[-4:]])
            ca = ce.mean(axis=0); cn = np.linalg.norm(ca); tn = np.linalg.norm(te)
            if cn > 0 and tn > 0: ppmi_cos = float(np.dot(ca, te) / (cn * tn))
            for ct in ctx[-4:]:
                ce2 = self.emb[ct].float().numpy(); cn2 = np.linalg.norm(ce2)
                if cn2 > 0 and tn > 0: ppmi_max_cos = max(ppmi_max_cos, float(np.dot(ce2, te) / (cn2 * tn)))
            ppmi_norm = float(tn)

        return [uni_lp, bi_lp, tri_lp, skip_lp, rec_lp,
                ppmi_cos, ppmi_max_cos, ppmi_norm, 0.0]

    def reset(self):
        self._uni_counts = np.zeros(self.V, dtype=np.float32)
        self._bi_cache = {}
        self._bi_totals = {}
        self._tri_cache = {}
        self._tri_totals = {}
        self._seen_positions = defaultdict(list)
        self._context = []
        self._step = 0


def main():
    device = torch.device('cuda')
    C_ACTIVE = 9

    print('[1] Loading BPE-8000 setup...')
    bpe, vocab, tok2id, bpe_to_lm, emb, V, d_emb = load_setup()
    lm_to_bpe = {v: k for k, v in bpe_to_lm.items()}

    print('[2] Loading neural LM...')
    orig = torch.load(DEEPSEEK / 'artifacts/hybrid_256_l12_x50/best.pt',
                      map_location=device, weights_only=False)
    orig_sd = orig['state_dict']
    d_model = orig_sd['pos_emb.weight'].shape[-1]
    vocab_size = orig_sd['head_bias'].shape[0]
    d_ff = orig_sd['encoder.layers.0.linear1.weight'].shape[0]
    n_layers = len([k for k in orig_sd if k.startswith('encoder.layers.') and k.endswith('.norm1.weight')])
    n_heads = orig_sd['encoder.layers.0.self_attn.in_proj_weight'].shape[0] // (3 * d_model)
    max_len = orig_sd['pos_emb.weight'].shape[0]

    model = build_model(vocab=vocab_size, d_model=d_model, n_layers=n_layers,
                        n_heads=n_heads, d_ff=d_ff, max_len=max_len, device=device)
    model.load_state_dict(orig_sd)
    model = model.to(device)
    model.eval()
    print(f'  {sum(p.numel() for p in model.parameters()):,} params, d_model={d_model}')
    print(f'  Using ORIGINAL base model (~82 PPL) — steerer annealing in progress')

    print(f'\n=== BPE-8000 WikiText Continuation ===')
    print('Type a Wikipedia-style prompt. /quit to exit.\n')

    gen_len = 80
    temperature = 0.7

    while True:
        try:
            prompt = input('\nPrompt: ').strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not prompt:
            continue
        if prompt == '/quit':
            break

        pids = bpe.encode(prompt).ids
        lids = [bpe_to_lm.get(t, 0) for t in pids]
        ctx = lids[:]

        print('Model: ', end='', flush=True)
        with torch.no_grad():
            for _ in range(gen_len):
                inp = torch.tensor([ctx[-64:]], dtype=torch.long, device=device)
                logits = model(inp)
                logits = logits[0, -1] / temperature
                probs = torch.softmax(logits, dim=-1)
                next_id = torch.multinomial(probs, 1).item()

                bpe_id = lm_to_bpe.get(next_id)
                if bpe_id is not None:
                    print(bpe.decode([int(bpe_id)]), end='', flush=True)

                ctx.append(next_id)

        print()


if __name__ == '__main__':
    main()
