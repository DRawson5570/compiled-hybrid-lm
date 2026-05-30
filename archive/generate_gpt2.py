"""generate_gpt2.py — Chat with DeepCausalLM trained on GPT-2 BPE (WikiText or C4).

Usage:
  python hybrid/generate_gpt2.py --ckpt artifacts/path/best.pt --chat
  python hybrid/generate_gpt2.py --ckpt artifacts/path/best.pt --prompt "Hello"
"""
from __future__ import annotations

import sys, time, argparse, importlib.util
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

DEEPSEEK = Path('/home/drawson/deepseek_experiments')


def load_model(ckpt_path: str, device):
    _spec = importlib.util.spec_from_file_location(
        'train_scaled', str(DEEPSEEK / 'hybrid/train_scaled_neural_lm.py'))
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    DeepCausalLM = _mod.DeepCausalLM

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt['state_dict']
    d_model = state['pos_emb.weight'].shape[-1]
    if 'tok_emb.weight' in state:
        vocab = state['tok_emb.weight'].shape[0]
    elif 'head.weight' in state:
        vocab = state['head.weight'].shape[0]
    else:
        raise KeyError("Cannot detect vocab size from checkpoint")
    max_len = state['pos_emb.weight'].shape[0]

    if 'encoder.layers.0.linear1.weight' in state:
        d_ff = state['encoder.layers.0.linear1.weight'].shape[0]
        n_layers = len([k for k in state if k.startswith('encoder.layers.') and k.endswith('.norm1.weight')])
        n_heads = state['encoder.layers.0.self_attn.in_proj_weight'].shape[0] // (3 * d_model)
    elif 'layers.0.ff_1.weight' in state:
        d_ff = state['layers.0.ff_1.weight'].shape[0]
        n_layers = len([k for k in state if k.startswith('layers.') and k.endswith('.sa_norm.weight')])
        n_heads = state['layers.0.sa_q.weight'].shape[0] // d_model
    else:
        raise KeyError("Unknown checkpoint format — cannot detect architecture")

    model = DeepCausalLM(vocab=vocab, d_model=d_model, n_layers=n_layers,
                         n_heads=n_heads, d_ff=d_ff, max_len=max_len, dropout=0.0)
    model.load_state_dict(state)
    model = model.to(device)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    return model, n_params, ckpt.get('epoch', '?')


@torch.no_grad()
def generate(model, tokenizer, prompt: str, max_new: int = 200,
             temperature: float = 0.7, top_p: float = 0.9,
             repetition_penalty: float = 1.1, device=None):
    prompt_ids = tokenizer.encode(prompt)
    ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    generated = list(prompt_ids)
    stop_tokens = {tokenizer.eos_token_id}

    for _ in range(max_new):
        ctx = ids[:, -model.max_len + 1:]
        logits = model(ctx)
        next_logits = logits[0, -1, :] / max(temperature, 0.01)

        if repetition_penalty != 1.0:
            for token_id in set(generated[-50:]):
                if next_logits[token_id] < 0:
                    next_logits[token_id] *= repetition_penalty
                else:
                    next_logits[token_id] /= repetition_penalty

        if top_p < 1.0 and top_p > 0.0:
            sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            cutoff = cumulative_probs > top_p
            if cutoff.any():
                cutoff[1:] = cutoff[:-1].clone()
                cutoff[0] = False
                next_logits[sorted_indices[cutoff]] = -float('inf')

        probs = F.softmax(next_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1).item()

        if next_token in stop_tokens:
            break
        generated.append(next_token)
        ids = torch.cat([ids, torch.tensor([[next_token]], device=device)], dim=1)

    return tokenizer.decode(generated)


def chat_loop(model, tokenizer, device, model_info: str = ''):
    print("\n" + "=" * 60)
    print(f" GPT-2 BPE Chat{model_info}")
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
        response = generate(model, tokenizer, prompt, max_new=256,
                            temperature=0.7, top_p=0.9,
                            repetition_penalty=1.1, device=device)
        elapsed = time.time() - t0

        assistant_part = response.split("Assistant:")[-1].strip()
        print(assistant_part)
        print(f"  [{elapsed:.1f}s, {len(tokenizer.encode(assistant_part))} tokens]")
        history.append(f"Assistant: {assistant_part}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', type=str, required=True)
    p.add_argument('--prompt', type=str, default='Explain quantum computing in simple terms')
    p.add_argument('--max-new', type=int, default=200)
    p.add_argument('--temperature', type=float, default=0.7)
    p.add_argument('--top-p', type=float, default=0.9)
    p.add_argument('--repetition-penalty', type=float, default=1.1)
    p.add_argument('--chat', action='store_true')
    p.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = p.parse_args()

    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    print('[load] Model...')
    model, n_params, epoch = load_model(args.ckpt, device)
    model_info = f" — {n_params/1e6:.1f}M params, epoch {epoch}"
    print(f'  {n_params:,} params, epoch {epoch}')

    if args.chat:
        chat_loop(model, tokenizer, device, model_info)
    else:
        print(f'\n[gen] "{args.prompt}"')
        output = generate(model, tokenizer, args.prompt, args.max_new,
                          args.temperature, args.top_p,
                          args.repetition_penalty, device)
        print(f'\n{output}')


if __name__ == '__main__':
    main()
