"""generate_gpt2_blend.py — Blended generation: compiled channels + neural LM + chat.

Runs compiled n-gram/recency/shape/unigram channels at inference time,
blends with a trained DeepCausalLM neural LM, and provides interactive chat.

Usage:
  python hybrid/generate_gpt2_blend.py --ckpt artifacts/c4_v2_768_x30/best.pt --builder artifacts/compiled_builder_50m.pt --chat
  python hybrid/generate_gpt2_blend.py --ckpt artifacts/c4_v2_768_x30/best.pt --builder artifacts/compiled_builder_50m.pt --prompt "Hello"
"""
from __future__ import annotations

import sys, time, argparse, math, importlib.util, os
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

DEEPSEEK = Path('/home/drawson/deepseek_experiments')
sys.path.insert(0, str(DEEPSEEK))

from hybrid.superposition_steerer import SuperpositionSteerer

# ── Compiled channel engine ──────────────────────────────────────────────

V_GPT2 = 50257

class CompiledChannelsInference:
    """Runs compiled channels at inference time, producing full V-distributions.
    Maintains stateful decay caches. Updates on each new token."""

    def __init__(self, builder_path: str, tokenizer, train_ids_path: str):
        from hybrid.compiled_features import GPT2CompiledChannelBuilder
        self._builder_path = builder_path
        self._train_ids_path = train_ids_path
        self.builder = GPT2CompiledChannelBuilder.load(builder_path)
        self.tokenizer = tokenizer
        self.V = V_GPT2

        # Pre-compute static unigram from compiled builder (Laplace-smoothed)
        unigram_counts = np.zeros(self.V, dtype=np.float64)
        for tid, count in self.builder.unigram.items():
            if tid < self.V:
                unigram_counts[tid] = count
        uni_smooth = (unigram_counts + 0.1) / (unigram_counts.sum() + 0.1 * self.V)
        self.uni_compiled_lp = torch.tensor(np.log(np.maximum(uni_smooth, 1e-30)),
                                            dtype=torch.float32)

        # Pre-compute Laplace-smoothed unigram from train set
        train_ids = torch.load(train_ids_path, weights_only=False).long().numpy()
        train_counts = np.bincount(train_ids.astype(np.int64), minlength=self.V).astype(np.float64)
        train_smooth = (train_counts + 0.1) / (train_counts.sum() + 0.1 * self.V)
        self.uni_train_lp = torch.tensor(np.log(np.maximum(train_smooth, 1e-30)),
                                         dtype=torch.float32)

        # Pre-compute token shapes
        self._shape_cache = {}
        self._build_shape_map()
        self._shape_trans = self._compute_shape_transitions(train_ids)

        # Stateful caches
        self._uc_counts = np.zeros(self.V, dtype=np.float32)     # unigram decay cache
        self._bi_cache = {}                                       # bigram decay: (prev, token) -> count
        self._tri_cache = {}                                      # trigram decay: (prev2, prev1, token) -> count
        self._seen_positions = defaultdict(list)                  # recency: token -> [positions]
        self._context_history = []                                # recent token ids for n-gram context
        self._step = 0

    def _build_shape_map(self):
        for tid in range(self.V):
            s = self.tokenizer.decode([tid])
            if s.isupper():
                self._shape_cache[tid] = 0
            elif s and s[0].isupper():
                self._shape_cache[tid] = 1
            elif s.isdigit():
                self._shape_cache[tid] = 2
            elif s and all(c.isalpha() for c in s):
                self._shape_cache[tid] = 3
            else:
                self._shape_cache[tid] = 4

    def _compute_shape_transitions(self, train_ids, n_samples=5_000_000):
        trans = np.ones((5, 5), dtype=np.float32)
        sample = train_ids[:n_samples]
        shapes = np.array([self._shape_cache.get(int(t), 4) for t in sample], dtype=np.int32)
        for t in range(1, len(shapes)):
            trans[shapes[t-1], shapes[t]] += 1
        return trans

    def _reset_caches(self):
        self._uc_counts = np.zeros(self.V, dtype=np.float32)
        self._bi_cache = {}
        self._tri_cache = {}
        self._seen_positions = defaultdict(list)
        self._context_history = []
        self._step = 0

    def update(self, token_id: int):
        """Update all caches with a newly observed token."""
        tid = int(token_id)
        self._step += 1
        self._context_history.append(tid)

        # Unigram decay cache
        self._uc_counts *= 0.999  # alpha=0.001
        if tid < self.V:
            self._uc_counts[tid] += 1

        # Bigram decay cache
        if len(self._context_history) >= 2:
            prev = self._context_history[-2]
            key = (prev, tid)
            self._bi_cache[key] = self._bi_cache.get(key, 0) + 1
            # Decay all entries
            to_del = []
            for k in self._bi_cache:
                self._bi_cache[k] *= 0.999
                if self._bi_cache[k] < 1e-6:
                    to_del.append(k)
            for k in to_del:
                del self._bi_cache[k]

        # Trigram decay cache
        if len(self._context_history) >= 3:
            p2, p1 = self._context_history[-3], self._context_history[-2]
            key = (p2, p1, tid)
            self._tri_cache[key] = self._tri_cache.get(key, 0) + 1
            to_del = []
            for k in self._tri_cache:
                self._tri_cache[k] *= 0.999
                if self._tri_cache[k] < 1e-6:
                    to_del.append(k)
            for k in to_del:
                del self._tri_cache[k]

        # Recency
        self._seen_positions[tid].append(self._step)

    def channel_unigram_decay(self) -> torch.Tensor:
        """Full V-distribution for decayed unigram cache."""
        d = self._uc_counts.sum() + 0.001 * self.V
        if d <= 0:
            return torch.full((self.V,), -math.log(self.V), dtype=torch.float32)
        probs = (torch.from_numpy(self._uc_counts) + 0.001) / d
        return torch.log(torch.clamp(probs, min=1e-30))

    def channel_bigram_decay(self) -> torch.Tensor:
        """Full V-distribution for decayed bigram cache given last token."""
        if len(self._context_history) < 1:
            return torch.full((self.V,), -math.log(self.V), dtype=torch.float32)
        ctx = self._context_history[-1]
        lp = torch.full((self.V,), -math.log(self.V), dtype=torch.float32)
        total = sum(v for k, v in self._bi_cache.items() if k[0] == ctx)
        d = total + 0.001 * self.V
        if d <= 0:
            return lp
        for (c, t), v in self._bi_cache.items():
            if c == ctx:
                lp[t] = math.log(max((v + 0.001) / d, 1e-30))
        return lp

    def channel_trigram_decay(self) -> torch.Tensor:
        """Full V-distribution for decayed trigram cache given last two tokens."""
        if len(self._context_history) < 2:
            return torch.full((self.V,), -math.log(self.V), dtype=torch.float32)
        ctx = (self._context_history[-2], self._context_history[-1])
        lp = torch.full((self.V,), -math.log(self.V), dtype=torch.float32)
        total = sum(v for k, v in self._tri_cache.items() if k[:2] == ctx)
        d = total + 0.001 * self.V
        if d <= 0:
            return lp
        for (c2, c1, t), v in self._tri_cache.items():
            if (c2, c1) == ctx:
                lp[t] = math.log(max((v + 0.001) / d, 1e-30))
        return lp

    def channel_shape(self) -> torch.Tensor:
        """Full V-distribution for shape transition given previous token."""
        if len(self._context_history) < 1:
            return torch.full((self.V,), -math.log(5), dtype=torch.float32)
        prev = self._context_history[-1]
        prev_shape = self._shape_cache.get(prev, 4)
        lp = torch.full((self.V,), -math.log(5), dtype=torch.float32)
        trans_row = self._shape_trans[prev_shape]
        d = trans_row.sum()
        if d <= 0:
            return lp
        for tid in range(self.V):
            ts = self._shape_cache.get(tid, 4)
            lp[tid] = math.log(max(float(trans_row[ts] / d), 1e-30))
        return lp

    def channel_recency(self) -> torch.Tensor:
        """Full V-distribution for token recency."""
        window = 128
        lp = torch.full((self.V,), -math.log(window), dtype=torch.float32)
        for tid, positions in self._seen_positions.items():
            if positions:
                gap = max(1, min(window, self._step - positions[-1]))
                lp[tid] = math.log(max(1.0 / gap, 1e-30))
        return lp

    def get_all_channel_lps(self) -> list[torch.Tensor]:
        """Return list of [compiled_uni, tri_fast, tri_slow, bi_fast, bi_slow,
           uni_fast, uni_slow, shape, uni_train, recency] as full V-distributions.
        """
        return [
            self.uni_compiled_lp,          # 0: compiled
            self.channel_trigram_decay(),   # 1: tri_f
            self.channel_trigram_decay(),   # 2: tri_s (same as tri_f for now, different alpha)
            self.channel_bigram_decay(),     # 3: bi_f
            self.channel_bigram_decay(),     # 4: bi_s
            self.channel_unigram_decay(),    # 5: uc_f
            self.channel_unigram_decay(),    # 6: uc_s
            self.channel_shape(),           # 7: shape
            self.uni_train_lp,              # 8: uni (train)
            self.channel_recency(),         # 9: recency
        ]


# ── Model loading ───────────────────────────────────────────────────────

def load_neural_lm(ckpt_path: str, device):
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
    elif 'head.weight' in state:
        vocab = state['head.weight'].shape[0]
    else:
        raise KeyError("Cannot detect vocab size")

    if 'encoder.layers.0.linear1.weight' in state:
        d_ff = state['encoder.layers.0.linear1.weight'].shape[0]
        n_layers = len([k for k in state if k.startswith('encoder.layers.') and k.endswith('.norm1.weight')])
        n_heads = state['encoder.layers.0.self_attn.in_proj_weight'].shape[0] // (3 * d_model)
    elif 'layers.0.ff_1.weight' in state:
        d_ff = state['layers.0.ff_1.weight'].shape[0]
        n_layers = len([k for k in state if k.startswith('layers.') and k.endswith('.sa_norm.weight')])
        n_heads = state['layers.0.sa_q.weight'].shape[0] // d_model
    else:
        raise KeyError("Unknown checkpoint format")

    model = DeepCausalLM(vocab=vocab, d_model=d_model, n_layers=n_layers,
                         n_heads=n_heads, d_ff=d_ff, max_len=max_len, dropout=0.0)
    model.load_state_dict(state)
    model = model.to(device)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    return model, n_params, ckpt.get('epoch', '?')


# ── Generation ──────────────────────────────────────────────────────────

@torch.no_grad()
def generate_blend(neural_model, compiled: CompiledChannelsInference,
                   tokenizer, prompt: str, max_new: int = 200,
                   temperature: float = 0.7, top_p: float = 0.9,
                   repetition_penalty: float = 1.1,
                   alpha: float = 0.5, device=None,
                   steerer: SuperpositionSteerer | None = None,
                   mode: str = 'output'):
    """Generate text blending compiled channels with neural LM.

    Modes:
      - output: blend at logit level (current default)
      - superposition: inject compiled channels as activation offsets
      - both: apply both paths
    """
    prompt_ids = tokenizer.encode(prompt)
    ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    generated = list(prompt_ids)

    # Seed compiled channels with prompt
    for tid in prompt_ids[:-1]:
        compiled.update(tid)

    use_output = mode in ('output', 'both')
    use_superposition = mode in ('superposition', 'both')

    for step in range(max_new):
        # Get compiled channel distributions
        all_lps = compiled.get_all_channel_lps()
        EXCLUDE = {7}

        # Superposition: set steerer weights before forward pass
        if use_superposition and steerer is not None:
            steerer.set_weights(all_lps, exclude_channels=EXCLUDE)

        # Neural LM forward (steerer hooks fire during this pass)
        ctx = ids[:, -neural_model.max_len + 1:]
        logits = neural_model(ctx)
        next_logits = logits[0, -1, :] / max(temperature, 0.01)
        neural_lp = F.log_softmax(next_logits, dim=-1)

        # Output blending
        if use_output:
            active_lps = [lp for i, lp in enumerate(all_lps) if i not in EXCLUDE]
            compiled_lp = torch.stack(active_lps).mean(dim=0).to(device)
            la, l1a = math.log(alpha), math.log(1 - alpha)
            blend_lp = torch.logsumexp(torch.stack([
                la + compiled_lp, l1a + neural_lp
            ]), dim=0)
        else:
            # Superposition-only: use steered neural LM output directly
            blend_lp = neural_lp

        # Repetition penalty
        if repetition_penalty != 1.0:
            for token_id in set(generated[-50:]):
                if token_id < len(blend_lp):
                    if blend_lp[token_id] < 0:
                        blend_lp[token_id] *= repetition_penalty
                    else:
                        blend_lp[token_id] /= repetition_penalty

        # Top-p sampling
        if top_p < 1.0 and top_p > 0.0:
            sorted_lp, sorted_indices = torch.sort(blend_lp, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_lp, dim=-1), dim=-1)
            cutoff = cumulative_probs > top_p
            if cutoff.any():
                cutoff[1:] = cutoff[:-1].clone()
                cutoff[0] = False
                blend_lp[sorted_indices[cutoff]] = -float('inf')

        probs = F.softmax(blend_lp, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1).item()

        if next_token == tokenizer.eos_token_id:
            break
        generated.append(next_token)
        ids = torch.cat([ids, torch.tensor([[next_token]], device=device)], dim=1)

        # Update compiled channels
        compiled.update(next_token)

    return tokenizer.decode(generated)


def chat_loop(neural_model, compiled, tokenizer, device, alpha, model_info='',
              steerer=None, mode='output'):
    print("\n" + "=" * 60)
    print(f" GPT-2 BPE Chat — mode={mode}{model_info}")
    print(" Type 'quit' to exit, 'reset' to clear history")
    print("=" * 60)

    history = []
    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() == 'quit':
            break
        if user_input.lower() == 'reset':
            history = []
            print("History cleared.")
            continue

        history.append(f"User: {user_input}")
        prompt = "\n".join(history) + "\nAssistant:"

        # Reset compiled caches for fresh generation
        compiled._reset_caches()
        compiled._context_history = list(tokenizer.encode(prompt)[:-1])
        for tid in compiled._context_history:
            compiled.update(tid)

        print("\nAssistant: ", end="", flush=True)
        t0 = time.time()
        response = generate_blend(neural_model, compiled, tokenizer, prompt,
                                  max_new=256, temperature=0.7, top_p=0.9,
                                  repetition_penalty=1.1, alpha=alpha, device=device,
                                  steerer=steerer, mode=mode)
        elapsed = time.time() - t0

        assistant_part = response.split("Assistant:")[-1].strip()
        print(assistant_part)
        token_count = len(tokenizer.encode(assistant_part))
        print(f"  [{elapsed:.1f}s, {token_count} tokens, {token_count/elapsed:.1f} tok/s]")
        history.append(f"Assistant: {assistant_part}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', type=str, required=True, help='Neural LM checkpoint')
    p.add_argument('--builder', type=str, required=True, help='Compiled channel builder .pt')
    p.add_argument('--train-ids', type=str,
                    default='artifacts/wikitext_gpt2/train_ids.pt',
                    help='Path to training token IDs for unigram stats')
    p.add_argument('--prompt', type=str, default='Explain quantum computing in simple terms')
    p.add_argument('--max-new', type=int, default=200)
    p.add_argument('--temperature', type=float, default=0.7)
    p.add_argument('--top-p', type=float, default=0.9)
    p.add_argument('--repetition-penalty', type=float, default=1.1)
    p.add_argument('--alpha', type=float, default=0.5,
                    help='Blend weight for output mode: alpha*compiled + (1-alpha)*neural')
    p.add_argument('--mode', type=str, default='output',
                    choices=['output', 'superposition', 'both'],
                    help='output: logit blend | superposition: activation injection | both')
    p.add_argument('--chat', action='store_true')
    p.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = p.parse_args()

    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    print('[load] Compiled channels engine...')
    compiled = CompiledChannelsInference(args.builder, tokenizer, args.train_ids)
    print(f'  V={compiled.V}, shapes built')

    print('[load] Neural LM...')
    model, n_params, epoch = load_neural_lm(args.ckpt, device)
    model_info = f" — {n_params/1e6:.1f}M params, epoch {epoch}, mode={args.mode}"
    print(f'  {n_params:,} params, epoch {epoch}')

    steerer = None
    if args.mode in ('superposition', 'both'):
        steerer = SuperpositionSteerer(num_channels=9, d_model=768,
                                        inject_layers=[0, 4, 8])
        steerer = steerer.to(device)
        n_hooks = steerer.register_hooks(model)
        print(f'[steerer] {n_hooks} hooks registered at layers {steerer.inject_layers}')

    if args.chat:
        chat_loop(model, compiled, tokenizer, device, args.alpha, model_info,
                  steerer=steerer, mode=args.mode)
    else:
        print(f'\n[gen] "{args.prompt}"')
        output = generate_blend(model, compiled, tokenizer, args.prompt,
                                args.max_new, args.temperature, args.top_p,
                                args.repetition_penalty, args.alpha, device,
                                steerer=steerer, mode=args.mode)
        print(f'\n{output}')


if __name__ == '__main__':
    main()
