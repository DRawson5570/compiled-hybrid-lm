"""chat_gpt2.py — GPT-2 BPE chat with SuperpositionSteerer active (production mode)."""
import sys
from hybrid.config import REPO_ROOT, torch, numpy as np, math
from pathlib import Path; from collections import defaultdict; import importlib.util

DEEPSEEK = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(DEEPSEEK))

_spec = importlib.util.spec_from_file_location('scaled', str(DEEPSEEK / 'hybrid/train_scaled_neural_lm.py'))
_mod = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_mod); DeepCausalLM = _mod.DeepCausalLM
import sys as _sys; _sys.path.insert(0, str(DEEPSEEK))
from hybrid.superposition_steerer import SuperpositionSteerer

V = 50257; C_ACTIVE = 9

class LiveCF:
    def __init__(self):
        self._uni = np.zeros(V, dtype=np.float32)
        self._bi = {}; self._bit = {}; self._tri = {}; self._trit = {}
        self._sp = defaultdict(list); self._ctx = []; self._step = 0
        self._u = -math.log(V)

    def update(self, token):
        tid = int(token); self._step += 1
        self._ctx.append(tid); self._ctx = self._ctx[-256:]
        self._uni *= 0.999
        if tid < V: self._uni[tid] += 1
        if len(self._ctx) >= 2:
            p, c = self._ctx[-2], self._ctx[-1]; k = (p, c)
            self._bi[k] = self._bi.get(k, 0) + 1; self._bit[p] = self._bit.get(p, 0) + 1
        if len(self._ctx) >= 3:
            p2, p1, c = self._ctx[-3], self._ctx[-2], self._ctx[-1]
            self._tri[(p2, p1, c)] = self._tri.get((p2, p1, c), 0) + 1
            self._trit[(p2, p1)] = self._trit.get((p2, p1), 0) + 1
        self._sp[tid].append(self._step)

    def get_features(self, target):
        tid = int(target); ct = self._ctx; u = self._u
        d = self._uni.sum() + 0.001 * V
        ul = math.log(max((self._uni[tid] + 0.001) / d, 1e-7)) if d > 0 and tid < V else u
        bl = u; tl = u; sl = u
        if len(ct) >= 1:
            tot = self._bit.get(ct[-1], 0); db = tot + 0.001 * V
            bl = math.log(max((self._bi.get((ct[-1], tid), 0) + 0.001) / db, 1e-7)) if db > 0 else u
        if len(ct) >= 2:
            ck = (ct[-2], ct[-1]); tot = self._trit.get(ck, 0); dt = tot + 0.001 * V
            tl = math.log(max((self._tri.get((ct[-2], ct[-1], tid), 0) + 0.001) / dt, 1e-7)) if dt > 0 else u
            sk = ct[-2]; tot = self._bit.get(sk, 0); ds = tot + 0.001 * V
            sl = math.log(max((self._bi.get((sk, tid), 0) + 0.001) / ds, 1e-7)) if ds > 0 else u
        pos = self._sp.get(tid, [])
        gap = 128 if not pos else min(128, self._step - pos[-1])
        rl = math.log(max(1.0 / max(gap, 1), 1e-7))
        return [ul, bl, tl, sl, rl, 0.0, 0.0, 0.0, 0.0]


def main():
    device = torch.device('cuda')

    CKPT = DEEPSEEK / 'artifacts/steerer_stream/steerer_best_s.pt'

    print('[load] Model + Steerer...')
    j = torch.load(CKPT, map_location=device, weights_only=False)
    s = j['state_dict']
    d_model = s['pos_emb.weight'].shape[-1]
    vocab = s['head_bias'].shape[0]
    d_ff = s['encoder.layers.0.linear1.weight'].shape[0]
    n_layers = len([k for k in s if k.startswith('encoder.layers.') and k.endswith('.norm1.weight')])
    n_heads = s['encoder.layers.0.self_attn.in_proj_weight'].shape[0] // (3 * d_model)
    max_len = s['pos_emb.weight'].shape[0]

    model = DeepCausalLM(vocab=vocab, d_model=d_model, n_layers=n_layers,
                         n_heads=n_heads, d_ff=d_ff, max_len=max_len, dropout=0.0)
    model.load_state_dict(s)
    model = model.to(device); model.eval()

    steerer = SuperpositionSteerer(num_channels=C_ACTIVE, d_model=d_model,
                                    inject_layers=[0, 4, 8], init_scale=0.01)
    steerer.load_state_dict(j['steerer_state'])
    steerer = steerer.to(device); steerer.eval()
    steerer.register_hooks(model)

    channels = LiveCF()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained('gpt2')

    es = j.get('eval_s', j.get('eval_ppl', '?'))
    eb = j.get('eval_b', '?')
    ep = j.get('epoch', '?')
    print(f'  eval_s={es}  eval_b={eb}  epoch={ep}  gamma={steerer.gamma.item():.3f}')
    print(f'\n=== GPT-2 BPE Steered Chat ===')
    print('Type /quit to exit\n')

    gen_len = 120; temperature = 0.7; ctx = []

    while True:
        try:
            prompt = input('\nYou: ').strip()
        except (EOFError, KeyboardInterrupt):
            print(); break
        if not prompt: continue
        if prompt == '/quit': break

        ids = tok.encode(prompt); ctx.extend(ids)
        for tid in ids: channels.update(tid)

        print('Model: ', end='', flush=True)
        with torch.no_grad():
            for _ in range(gen_len):
                c = ctx[-64:]
                sw = [channels.get_features(t) for t in c]
                w = torch.tensor(sw, dtype=torch.float32, device=device).unsqueeze(0)
                steerer.set_weights(w)
                inp = torch.tensor([c], dtype=torch.long, device=device)
                logits = model(inp)
                probs = torch.softmax(logits[0, -1] / temperature, dim=-1)
                nid = torch.multinomial(probs, 1).item()
                ctx.append(nid); channels.update(nid)
                print(tok.decode([nid]), end='', flush=True)
        print()

    steerer.remove_hooks()
    print('Done.')


if __name__ == '__main__':
    main()
