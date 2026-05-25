"""Chat with a frozen base model plus hot-swappable steering cartridges."""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import pickle
import re
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

_spec = importlib.util.spec_from_file_location('scaled', str(REPO / 'hybrid/train_scaled_neural_lm.py'))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
DeepCausalLM = _mod.DeepCausalLM
sys.path.insert(0, str(REPO))

from hybrid.cartridges import CartridgeManifest, CartridgeRole, SteererCartridgeRack
from hybrid.gpu_channels import GPUFeatureComputer
from hybrid.superposition_steerer_v3 import FeatureConditionedAdapterSteerer, SuperpositionSteererV3
from hybrid.train_steerer_v4 import FastNgramFeatures, PUNCT_IDS, compute_cpu_features

V = 50257
SYSTEM = (
    'You are CMI, a concise and helpful assistant. Answer directly, keep the '
    'conversation coherent, and ask a brief clarifying question when needed.'
)


def load_model_from_state(state: dict[str, torch.Tensor], device: torch.device):
    d_model = state['pos_emb.weight'].shape[-1]
    vocab = state['head_bias'].shape[0]
    d_ff = state['encoder.layers.0.linear1.weight'].shape[0]
    n_layers = len([k for k in state if k.startswith('encoder.layers.') and k.endswith('.norm1.weight')])
    n_heads = state['encoder.layers.0.self_attn.in_proj_weight'].shape[0] // (3 * d_model)
    max_len = state['pos_emb.weight'].shape[0]
    model = DeepCausalLM(vocab=vocab, d_model=d_model, n_layers=n_layers,
                         n_heads=n_heads, d_ff=d_ff, max_len=max_len, dropout=0.0)
    model.load_state_dict(state)
    return model.to(device).eval(), d_model


def load_feature_computer(device: torch.device) -> GPUFeatureComputer:
    priors_dir = REPO / 'artifacts/compiled_priors_v3'
    word_topics = torch.load(priors_dir / 'word_topics.pt', map_location='cpu', weights_only=False)
    with open(priors_dir / 'pos_stats.pkl', 'rb') as f:
        pos_stats = pickle.load(f)
    token_to_tag = pos_stats.get('token_to_tag', {})
    tag_to_idx = pos_stats.get('tag_to_idx', {'WORD': 0, 'PUNCT': 1, 'NUM': 2})
    pos_tags = {int(k): tag_to_idx.get(v, 0) for k, v in token_to_tag.items()}
    generator = torch.Generator(device='cpu').manual_seed(20260524)
    ppmi_emb = torch.randn(V, 256, dtype=torch.float32, generator=generator) * 0.01
    return GPUFeatureComputer(
        V=V,
        punct_ids=PUNCT_IDS,
        topic_matrix=word_topics,
        pos_tags=pos_tags,
        ppmi_embeddings=ppmi_emb,
        device=device,
    )


def compute_weights(input_ids: torch.Tensor, gpu_fc: GPUFeatureComputer, cpu_ch: FastNgramFeatures) -> torch.Tensor:
    weights = gpu_fc.compute_features(input_ids)
    for batch_idx in range(input_ids.shape[0]):
        w_cpu = compute_cpu_features(input_ids[batch_idx].detach().cpu().tolist(), cpu_ch)
        weights[batch_idx, :w_cpu.shape[0], 0:9] = w_cpu[:, :9].to(input_ids.device)
    return weights


def top_p_filter(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    if top_p >= 1.0:
        return logits
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    sorted_probs = torch.softmax(sorted_logits, dim=-1)
    cumulative = torch.cumsum(sorted_probs, dim=-1)
    remove = cumulative > top_p
    remove[1:] = remove[:-1].clone()
    remove[0] = False
    filtered = logits.clone()
    filtered[sorted_indices[remove]] = -float('inf')
    return filtered


def apply_repetition_penalty(logits: torch.Tensor, generated: list[int], penalty: float) -> torch.Tensor:
    if penalty <= 1.0 or not generated:
        return logits
    logits = logits.clone()
    for token_id in set(generated):
        if logits[token_id] < 0:
            logits[token_id] *= penalty
        else:
            logits[token_id] /= penalty
    return logits


def has_repeated_tail(token_ids: list[int], ngram: int = 8) -> bool:
    if len(token_ids) < ngram * 2:
        return False
    return token_ids[-ngram:] == token_ids[-2 * ngram:-ngram]


def trim_to_sentences(text: str, max_sentences: int) -> str:
    if max_sentences <= 0:
        return text.strip()
    matches = list(re.finditer(r'[.!?]', text))
    if len(matches) < max_sentences:
        return text.strip()
    return text[:matches[max_sentences - 1].end()].strip()


def sample_next(logits: torch.Tensor, temperature: float, top_k: int, top_p: float,
                generated: list[int] | None = None, repetition_penalty: float = 1.0) -> int:
    logits = apply_repetition_penalty(logits, generated or [], repetition_penalty)
    logits = logits.float() / max(temperature, 1e-5)
    if top_k > 0:
        values, _ = torch.topk(logits, min(top_k, logits.numel()))
        logits = torch.where(logits < values[-1], torch.full_like(logits, -float('inf')), logits)
    logits = top_p_filter(logits, top_p)
    probs = torch.softmax(logits, dim=-1)
    return int(torch.multinomial(probs, 1).item())


def build_chat_steerer_from_checkpoint(ckpt: dict, d_model: int, device: torch.device):
    steerer_class = ckpt.get('steerer_class') or ckpt.get('manifest', {}).get('steerer_class', 'SuperpositionSteererV3')
    if steerer_class == 'FeatureConditionedAdapterSteerer':
        bottleneck = int(ckpt.get('adapter_bottleneck', 64))
        return FeatureConditionedAdapterSteerer(
            d_model=d_model,
            bottleneck=bottleneck,
            init_scale=0.01,
            noise_scale=0.0,
        ).to(device).eval()
    return SuperpositionSteererV3(d_model=d_model, init_scale=0.01, noise_scale=0.0).to(device).eval()


class CartridgeChatRuntime:
    def __init__(self, base_model: str, general_steerer: str, chat_cartridge: str,
                 device: str = 'cuda', mode: str = 'chat'):
        self.device = torch.device(device)
        base_ckpt = torch.load(REPO / base_model, map_location=self.device, weights_only=False)
        self.model, d_model = load_model_from_state(base_ckpt['state_dict'], self.device)
        self.tokenizer = AutoTokenizer.from_pretrained('gpt2')
        self.gpu_fc = load_feature_computer(self.device)
        self.cpu_ch = FastNgramFeatures(V)
        self.rack = SteererCartridgeRack()

        general = SuperpositionSteererV3(d_model=d_model, init_scale=0.01, noise_scale=0.0).to(self.device).eval()
        general_ckpt = torch.load(REPO / general_steerer, map_location=self.device, weights_only=False)
        general.load_state_dict(general_ckpt['steerer_state'])

        chat_ckpt = torch.load(REPO / chat_cartridge, map_location=self.device, weights_only=False)
        chat = build_chat_steerer_from_checkpoint(chat_ckpt, d_model, self.device)
        chat.load_state_dict(chat_ckpt['steerer_state'])

        self.rack.mount(
            CartridgeManifest('general-superposition', CartridgeRole.SUPERPOSITION_STEERER,
                              base_model_id='c4-124m-v4', tokenizer_id='gpt2-bpe'),
            general,
        )
        self.rack.mount(
            CartridgeManifest('chat-capability', CartridgeRole.TASK_CAPABILITY,
                              base_model_id='c4-124m-v4', tokenizer_id='gpt2-bpe'),
            chat,
        )
        self.rack.register_hooks(self.model)
        self.set_mode(mode)

    def set_mode(self, mode: str):
        if mode not in {'base', 'superposition', 'chat'}:
            raise ValueError(f'unknown mode: {mode}')
        self.mode = mode
        self.rack.activate('general-superposition', mode in {'superposition', 'chat'})
        self.rack.activate('chat-capability', mode == 'chat')

    def format_prompt(self, user_text: str, history: list[tuple[str, str]] | None = None) -> str:
        parts = [f'System:\n{SYSTEM}\n\n']
        for user, assistant in history or []:
            parts.append(f'User:\n{user}\n\nAssistant:\n{assistant}\n\n')
        parts.append(f'User:\n{user_text}\n\nAssistant:\n')
        return ''.join(parts)

    @torch.no_grad()
    def generate(self, user_text: str, history: list[tuple[str, str]] | None = None,
                 max_new_tokens: int = 80, temperature: float = 0.7,
                 top_k: int = 40, top_p: float = 0.9, context_len: int = 128,
                 repetition_penalty: float = 1.15, stop_ngram: int = 8,
                 max_sentences: int = 0) -> str:
        prompt = self.format_prompt(user_text, history)
        ids = self.tokenizer.encode(prompt)
        generated: list[int] = []
        stop_markers = ['\n\nUser:', '\nUser:', '\n\nSystem:', '\nSystem:']

        for _ in range(max_new_tokens):
            ctx = ids[-context_len:]
            x = torch.tensor([ctx], dtype=torch.long, device=self.device)
            if self.mode != 'base':
                self.rack.set_weights(compute_weights(x, self.gpu_fc, self.cpu_ch))
            logits = self.model(x)[0, -1]
            next_id = sample_next(
                logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                generated=generated,
                repetition_penalty=repetition_penalty,
            )
            ids.append(next_id)
            generated.append(next_id)
            text = self.tokenizer.decode(generated)
            if any(marker in text for marker in stop_markers):
                for marker in stop_markers:
                    if marker in text:
                        text = text.split(marker)[0]
                return trim_to_sentences(text, max_sentences)
            trimmed = trim_to_sentences(text, max_sentences)
            if trimmed != text.strip():
                return trimmed
            if stop_ngram > 0 and has_repeated_tail(generated, stop_ngram):
                trimmed = generated[:-stop_ngram]
                return trim_to_sentences(self.tokenizer.decode(trimmed), max_sentences)
        return trim_to_sentences(self.tokenizer.decode(generated), max_sentences)


def run_prompts(runtime: CartridgeChatRuntime, prompts: list[str], args):
    rows = []
    for prompt in prompts:
        answer = runtime.generate(
            prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            context_len=args.context_len,
            repetition_penalty=args.repetition_penalty,
            stop_ngram=args.stop_ngram,
            max_sentences=args.max_sentences,
        )
        rows.append({'mode': runtime.mode, 'prompt': prompt, 'answer': answer})
        print(f'\n[{runtime.mode}] User: {prompt}\nAssistant: {answer}', flush=True)
    if args.report:
        Path(args.report).write_text(json.dumps(rows, indent=2), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-model', default='artifacts/steerer_v4/steerer_best_b.pt')
    parser.add_argument('--general-steerer', default='artifacts/steerer_v4/steerer_best_b.pt')
    parser.add_argument('--chat-cartridge', default='artifacts/steerer_chat_adapter_seed_v4/chat_cartridge.pt')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--mode', choices=['base', 'superposition', 'chat'], default='chat')
    parser.add_argument('--prompt', action='append')
    parser.add_argument('--interactive', action='store_true')
    parser.add_argument('--max-new-tokens', type=int, default=80)
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--top-k', type=int, default=40)
    parser.add_argument('--top-p', type=float, default=0.9)
    parser.add_argument('--context-len', type=int, default=128)
    parser.add_argument('--repetition-penalty', type=float, default=1.15)
    parser.add_argument('--stop-ngram', type=int, default=8)
    parser.add_argument('--max-sentences', type=int, default=2)
    parser.add_argument('--report')
    args = parser.parse_args()

    runtime = CartridgeChatRuntime(
        base_model=args.base_model,
        general_steerer=args.general_steerer,
        chat_cartridge=args.chat_cartridge,
        device=args.device,
        mode=args.mode,
    )

    prompts = args.prompt or [
        'Hello!',
        'Explain what a chat cartridge is in two sentences.',
        'Give me three practical next steps for testing this model.',
    ]
    run_prompts(runtime, prompts, args)

    if args.interactive:
        history: list[tuple[str, str]] = []
        while True:
            try:
                user_text = input('\nYou: ').strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if user_text in {'/quit', '/exit'}:
                break
            answer = runtime.generate(user_text, history=history, max_new_tokens=args.max_new_tokens,
                                      temperature=args.temperature, top_k=args.top_k, top_p=args.top_p,
                                      context_len=args.context_len,
                                      repetition_penalty=args.repetition_penalty,
                                      stop_ngram=args.stop_ngram,
                                      max_sentences=args.max_sentences)
            print(f'CMI: {answer}')
            history.append((user_text, answer))


if __name__ == '__main__':
    main()