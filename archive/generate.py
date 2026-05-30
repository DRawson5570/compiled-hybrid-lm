"""generate.py — Autoregressive text generation from the trained neural LM.

Loads the DeepCausalLM checkpoint and generates text using temperature + top-p
sampling. Decodes tokens back to text via the project BPE tokenizer.

Usage:
    python hybrid/generate.py --ckpt artifacts/hybrid_v2_scaled/scaled_lm_best.pt \\
        --prompt "The history of" --max-tokens 50 --temperature 0.8
"""
from __future__ import annotations

import argparse, sys, math, importlib.util
from pathlib import Path

import torch
import torch.nn.functional as F

LLM_DECOUPLING = Path('/home/drawson/llm_decoupling')
DEEPSEEK = Path('/home/drawson/deepseek_experiments')
sys.path.insert(0, str(LLM_DECOUPLING))
sys.path.insert(0, str(DEEPSEEK))

from compile_wiki_lm_v13 import load_setup


def _import_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_model(ckpt_path: str, device):
    _train_mod = _import_file('train_scaled',
                              str(DEEPSEEK / 'hybrid/train_scaled_neural_lm.py'))
    DeepCausalLM = _train_mod.DeepCausalLM

    ckpt = torch.load(ckpt_path, map_location=device)
    args = ckpt['args']
    model = DeepCausalLM(
        vocab=8000, d_model=args.get('d_model', 256),
        n_layers=args.get('n_layers', 12),
        n_heads=args.get('n_heads', 8),
        d_ff=args.get('d_ff', 1024),
        max_len=args.get('seq_len', 128) + 1,
        dropout=0.0,
    ).to(device)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    return model, ckpt


@torch.no_grad()
def generate(model, tokenizer, prompt: str, max_tokens: int = 50,
             temperature: float = 0.8, top_p: float = 0.9,
             device=None):
    """Autoregressive generation from the neural LM."""
    if device is None:
        device = next(model.parameters()).device

    # Tokenize prompt
    prompt_ids = tokenizer.encode(prompt).ids
    ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    generated = list(prompt_ids)

    for _ in range(max_tokens):
        # Truncate to max context
        ctx = ids[:, -model.max_len:]
        logits = model(ctx)
        # Last position logits
        next_logits = logits[0, -1, :]

        # Temperature
        if temperature > 0:
            next_logits = next_logits / temperature

        # Top-p (nucleus) sampling
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
            sorted_indices_to_remove[0] = False
            indices_to_remove = sorted_indices_to_remove.scatter(
                0, sorted_indices, sorted_indices_to_remove
            )
            next_logits[indices_to_remove] = -float('inf')

        probs = F.softmax(next_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1).item()
        generated.append(next_token)
        ids = torch.cat([ids, torch.tensor([[next_token]], device=device)], dim=1)

    return tokenizer.decode(generated)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', type=str, required=True)
    p.add_argument('--prompt', type=str, default='The')
    p.add_argument('--max-tokens', type=int, default=50)
    p.add_argument('--temperature', type=float, default=0.8)
    p.add_argument('--top-p', type=float, default=0.9)
    p.add_argument('--device', type=str,
                   default='cuda' if torch.cuda.is_available() else 'cpu')
    args = p.parse_args()

    device = torch.device(args.device)
    print(f'[load] Loading model from {args.ckpt}...')
    model, ckpt = load_model(args.ckpt, device)
    epoch = ckpt.get('epoch', '?')
    val_ppl = ckpt.get('val_ppl', '?')
    n_params = sum(p.numel() for p in model.parameters())
    print(f'  Epoch: {epoch}  Val PPL: {val_ppl}  Params: {n_params:,}')

    print(f'[load] Loading BPE tokenizer...')
    bpe, vocab, tok2id, bpe_to_lm, emb, V, d = load_setup()
    print(f'  V={V}')

    print(f'[gen] Prompt: "{args.prompt}"')
    print(f'[gen] Generating {args.max_tokens} tokens (T={args.temperature}, top_p={args.top_p})...')
    output = generate(model, bpe, args.prompt, args.max_tokens,
                      args.temperature, args.top_p, device)
    print()
    print('=' * 60)
    print(output)
    print('=' * 60)


if __name__ == '__main__':
    main()
