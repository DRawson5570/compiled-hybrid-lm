"""generate_hybrid.py — Chat-capable generation from BPE-8000 hybrid model.

Features: prompt formatting, temperature + top-p + repetition penalty,
stop tokens, multi-turn chat loop.
"""
from __future__ import annotations

import sys, time, argparse
from pathlib import Path

import torch
import torch.nn.functional as F

DEEPSEEK = Path('/home/drawson/deepseek_experiments')
LLM_DECOUPLING = Path('/home/drawson/llm_decoupling')
sys.path.insert(0, str(LLM_DECOUPLING))
sys.path.insert(0, str(DEEPSEEK))

from compile_wiki_lm_v13 import load_setup


def load_model(ckpt_path: str, device):
    import importlib.util
    _spec = importlib.util.spec_from_file_location(
        'train_hybrid', str(DEEPSEEK / 'hybrid/train_hybrid_bpe8000.py'))
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    BPE8000LM = _mod.BPE8000LM

    ckpt = torch.load(ckpt_path, map_location=device)
    model = BPE8000LM(vocab=8000, d_model=256, n_layers=12,
                       n_heads=8, d_ff=1024,
                       max_len=ckpt['state_dict']['pos_emb.weight'].shape[0],
                       dropout=0.0)
    model.load_state_dict(ckpt['state_dict'])
    model = model.to(device)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    return model, n_params, ckpt.get('epoch', '?')


def build_prompt_ids(prompt: str, tokenizer) -> list[int]:
    """Format prompt with a simple instruction wrapper."""
    formatted = f"<|user|>\n{prompt}\n<|assistant|>\n"
    return tokenizer.encode(formatted).ids


@torch.no_grad()
def generate(model, tokenizer, prompt: str, max_new: int = 150,
             temperature: float = 0.7, top_p: float = 0.9,
             repetition_penalty: float = 1.1, device=None):
    """Autoregressive generation with temperature, top-p, and repetition penalty."""
    prompt_ids = build_prompt_ids(prompt, tokenizer)
    ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    generated = list(prompt_ids)
    stop_tokens = {tokenizer.token_to_id("<|endoftext|>"),
                   tokenizer.token_to_id("<|im_end|>")}

    for _ in range(max_new):
        ctx = ids[:, -model.max_len:]
        logits = model(ctx)
        next_logits = logits[0, -1, :] / max(temperature, 0.01)

        # Repetition penalty
        if repetition_penalty != 1.0:
            for token_id in set(generated[-50:]):
                if next_logits[token_id] < 0:
                    next_logits[token_id] *= repetition_penalty
                else:
                    next_logits[token_id] /= repetition_penalty

        # Top-p (nucleus) sampling
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            cutoff = cumulative_probs > top_p
            cutoff[1:] = cutoff[:-1].clone()
            cutoff[0] = False
            next_logits[sorted_indices[cutoff]] = -float('inf')

        probs = F.softmax(next_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1).item()

        if next_token in stop_tokens or next_token in (0, 2):
            break
        generated.append(next_token)
        ids = torch.cat([ids, torch.tensor([[next_token]], device=device)], dim=1)

    return tokenizer.decode(generated)


def chat_loop(model, tokenizer, device):
    """Interactive multi-turn chat."""
    print("\n" + "=" * 60)
    print(" Hybrid Chat — BPE-8000 (PPL=9.21 blend)")
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

        print("\nAssistant: ", end="", flush=True)
        t0 = time.time()
        response = generate(model, tokenizer, prompt, max_new=200,
                            temperature=0.7, top_p=0.9,
                            repetition_penalty=1.1, device=device)
        elapsed = time.time() - t0

        # Strip prompt from response
        response = response.split("<|assistant|>\n")[-1].strip()
        print(response)
        print(f"  [{elapsed:.1f}s]")
        history.append(f"Assistant: {response}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', type=str,
                   default=str(DEEPSEEK / 'artifacts/hybrid_256_l12_x50/best.pt'))
    p.add_argument('--prompt', type=str, default='Explain quantum computing in simple terms')
    p.add_argument('--max-new', type=int, default=200)
    p.add_argument('--temperature', type=float, default=0.7)
    p.add_argument('--top-p', type=float, default=0.9)
    p.add_argument('--repetition-penalty', type=float, default=1.1)
    p.add_argument('--chat', action='store_true')
    p.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = p.parse_args()

    device = torch.device(args.device)

    print('[load] Tokenizer...')
    bpe, vocab, tok2id, bpe_to_lm, emb, V, d = load_setup()
    print(f'  V={V}')

    print('[load] Model...')
    model, n_params, epoch = load_model(args.ckpt, device)
    print(f'  {n_params:,} params, epoch {epoch}')

    if args.chat:
        chat_loop(model, bpe, device)
    else:
        print(f'\n[gen] "{args.prompt}"')
        output = generate(model, bpe, args.prompt, args.max_new,
                          args.temperature, args.top_p,
                          args.repetition_penalty, device)
        print(f'\n{output}')


if __name__ == '__main__':
    main()
