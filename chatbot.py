"""Chatbot — production-grade chatbot for 124M DeepSeekForCausalLM.

Loads a frozen V4 base model with a general superposition steerer,
a chat cartridge, and optionally a knowledge cartridge through the
SteererCartridgeRack. Provides system-prompt formatting, multi-turn
history, repetition penalty, stop markers, and n-gram loop detection.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import pickle
import re
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

try:
    import spacy
except ImportError:
    spacy = None

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

_spec = importlib.util.spec_from_file_location(
    'scaled', str(REPO / 'hybrid/train_scaled_neural_lm.py'))
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
    "You are CMI, a concise and helpful assistant. Answer directly, keep the "
    "conversation coherent, and ask a brief clarifying question when needed."
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
        V=V, punct_ids=PUNCT_IDS, topic_matrix=word_topics,
        pos_tags=pos_tags, ppmi_embeddings=ppmi_emb, device=device,
    )


def compute_weights(input_ids: torch.Tensor, gpu_fc: GPUFeatureComputer,
                    cpu_ch: FastNgramFeatures) -> torch.Tensor:
    weights = gpu_fc.compute_features(input_ids)
    for batch_idx in range(input_ids.shape[0]):
        w_cpu = compute_cpu_features(input_ids[batch_idx].detach().cpu().tolist(), cpu_ch)
        weights[batch_idx, :w_cpu.shape[0], 0:9] = w_cpu[:, :9].to(input_ids.device)
    return weights


def build_steerer_from_checkpoint(ckpt: dict, d_model: int, device: torch.device):
    steerer_class = ckpt.get('steerer_class') or ckpt.get('manifest', {}).get('steerer_class', 'SuperpositionSteererV3')
    if steerer_class == 'FeatureConditionedAdapterSteerer':
        bottleneck = int(ckpt.get('adapter_bottleneck', 64))
        st = ckpt.get('steerer_state', {})
        feat_shape = st.get('feature.0.weight')
        extra_channels = int(ckpt.get('extra_channels', 0))
        if feat_shape is not None:
            total_ch = feat_shape.shape[-1]
            semantic_dim = max(0, total_ch - 21 - extra_channels)
        else:
            semantic_dim = 16
        return FeatureConditionedAdapterSteerer(
            d_model=d_model, bottleneck=bottleneck, init_scale=0.01, noise_scale=0.0,
            semantic_dim=semantic_dim, extra_channels=extra_channels).to(device).eval()
    return SuperpositionSteererV3(d_model=d_model, init_scale=0.01, noise_scale=0.0).to(device).eval()


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
    matches = [m for m in re.finditer(r'[.!?]', text)
               if not re.fullmatch(r'\s*\d+\.', text[text.rfind('\n', 0, m.start()) + 1:m.end()])]
    if len(matches) < max_sentences:
        return text.strip()
    return text[:matches[max_sentences - 1].end()].strip()


class ProductionChatbot:
    """Production-grade chatbot for 124M DeepSeekForCausalLM.

    Mounts general superposition steerer + chat cartridge + optional knowledge
    cartridge through the SteererCartridgeRack. Uses system prompt formatting,
    multi-turn history, repetition penalty, stop markers, and n-gram detection.
    """

    def __init__(self, base_model: str, general_steerer: str, chat_cartridge: str,
                 knowledge_cartridge: str | None = None, device: str = 'cuda',
                 composition_mode: str = 'additive'):
        self.device = torch.device(device)

        base_ckpt = torch.load(REPO / base_model, map_location=self.device, weights_only=False)
        self.model, d_model = load_model_from_state(base_ckpt['state_dict'], self.device)
        for p in self.model.parameters():
            p.requires_grad = False

        self.d_model = d_model
        self.tokenizer = AutoTokenizer.from_pretrained('gpt2')
        self.gpu_fc = load_feature_computer(self.device)
        self.cpu_ch = FastNgramFeatures(V)
        self.rack = SteererCartridgeRack(composition_mode=composition_mode)
        self.has_knowledge = False
        self.ner_dim = 0
        self._nlp = None

        st = base_ckpt.get('steerer_state', {})
        self.semantic_dim = 16 if any('steer_semantic' in k for k in st) else 0

        general = SuperpositionSteererV3(d_model=d_model, init_scale=0.01, noise_scale=0.0,
                                          semantic_dim=self.semantic_dim).to(self.device).eval()
        general_ckpt = torch.load(REPO / general_steerer, map_location=self.device, weights_only=False)
        general.load_state_dict(general_ckpt['steerer_state'], strict=False)

        chat_ckpt = torch.load(REPO / chat_cartridge, map_location=self.device, weights_only=False)
        chat = build_steerer_from_checkpoint(chat_ckpt, d_model, self.device)
        chat.load_state_dict(chat_ckpt['steerer_state'], strict=False)

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

        if knowledge_cartridge:
            self._load_knowledge_cartridge(knowledge_cartridge)

        self.rack.activate('general-superposition', True)
        self.rack.activate('chat-capability', True)

    def _load_knowledge_cartridge(self, path: str):
        ckpt = torch.load(REPO / path, map_location=self.device, weights_only=False)
        steerer = build_steerer_from_checkpoint(ckpt, self.d_model, self.device)
        steerer.load_state_dict(ckpt['steerer_state'], strict=False)
        self.ner_dim = int(ckpt.get('extra_channels', 0))
        if self.ner_dim > 0 and self._nlp is None and spacy is not None:
            self._nlp = spacy.load('en_core_web_sm')
        self.rack.mount(
            CartridgeManifest('knowledge-cartridge', CartridgeRole.TASK_CAPABILITY,
                              base_model_id='c4-124m-v4', tokenizer_id='gpt2-bpe',
                              steerer_class=ckpt.get('steerer_class', 'FeatureConditionedAdapterSteerer')),
            steerer,
        )
        self.rack.register_hooks(self.model)
        self.rack.activate('knowledge-cartridge', True)
        self.has_knowledge = True

    def format_prompt(self, user_text: str, history: list[tuple[str, str]] | None = None) -> str:
        parts = [f'System:\n{SYSTEM}\n\n']
        for user, assistant in history or []:
            parts.append(f'User:\n{user}\n\nAssistant:\n{assistant}\n\n')
        parts.append(f'User:\n{user_text}\n\nAssistant:\n')
        return ''.join(parts)

    @torch.no_grad()
    def generate(self, user_text: str, history: list[tuple[str, str]] | None = None,
                 max_new_tokens: int = 80, temperature: float = 0.3,
                 top_k: int = 40, top_p: float = 0.9, context_len: int = 128,
                 repetition_penalty: float = 1.15, stop_ngram: int = 8,
                 max_sentences: int = 3) -> tuple[str, float, int]:
        t0 = time.perf_counter()
        prompt = self.format_prompt(user_text, history)
        ids = self.tokenizer.encode(prompt)
        generated: list[int] = []
        stop_markers = ['\n\nUser:', '\nUser:', '\n\nSystem:', '\nSystem:']
        tok_count = 0

        prompt_ner = None
        if self.ner_dim > 0 and self._nlp is not None:
            from hybrid.ner_features import get_ner_features_for_ids
            prompt_ner = get_ner_features_for_ids(ids, self.tokenizer, self._nlp)

        for _ in range(max_new_tokens):
            ctx = ids[-context_len:]
            x = torch.tensor([ctx], dtype=torch.long, device=self.device)
            w = compute_weights(x, self.gpu_fc, self.cpu_ch)
            if self.ner_dim > 0 and prompt_ner is not None:
                ner_full = torch.zeros(len(ctx), self.ner_dim)
                pl = min(min(len(ids), prompt_ner.shape[0]), len(ctx))
                ner_full[:pl] = prompt_ner[:pl]
                ner = ner_full.unsqueeze(0).to(self.device, dtype=torch.float32)
                if w.shape[1] < ner.shape[1]:
                    w = torch.cat([w, torch.zeros(w.shape[0], ner.shape[1] - w.shape[1], w.shape[2], device=self.device)], dim=1)
                elif ner.shape[1] < w.shape[1]:
                    ner = torch.cat([ner, torch.zeros(1, w.shape[1] - ner.shape[1], ner.shape[2], device=self.device)], dim=1)
                w = torch.cat([w, ner], dim=-1)
            self.rack.set_weights(w)
            logits = self.model(x)[0, -1]

            logits = apply_repetition_penalty(logits, generated, repetition_penalty)
            if temperature <= 0:
                next_id = int(torch.argmax(logits).item())
            else:
                logits = logits.float() / max(temperature, 1e-5)
                if top_k > 0:
                    values, _ = torch.topk(logits, min(top_k, logits.numel()))
                    logits = torch.where(logits < values[-1], torch.full_like(logits, -float('inf')), logits)
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    sorted_probs = torch.softmax(sorted_logits, dim=-1)
                    cumulative = torch.cumsum(sorted_probs, dim=-1)
                    remove = cumulative > top_p
                    remove[1:] = remove[:-1].clone()
                    remove[0] = False
                    filtered = logits.clone()
                    filtered[sorted_indices[remove]] = -float('inf')
                    logits = filtered
                probs = torch.softmax(logits, dim=-1)
                next_id = int(torch.multinomial(probs, 1).item())

            tok_count += 1
            if next_id == self.tokenizer.eos_token_id:
                text = self.tokenizer.decode(generated)
                if text.strip():
                    elapsed = time.perf_counter() - t0
                    return trim_to_sentences(text, max_sentences), elapsed, tok_count
                continue
            ids.append(next_id)
            generated.append(next_id)
            text = self.tokenizer.decode(generated)
            for marker in stop_markers:
                if marker in text:
                    text = text.split(marker)[0]
                    elapsed = time.perf_counter() - t0
                    return trim_to_sentences(text, max_sentences), elapsed, tok_count
            trimmed = trim_to_sentences(text, max_sentences)
            if trimmed != text.strip():
                elapsed = time.perf_counter() - t0
                return trimmed, elapsed, tok_count
            if stop_ngram > 0 and has_repeated_tail(generated, stop_ngram):
                trimmed = generated[:-stop_ngram]
                elapsed = time.perf_counter() - t0
                return trim_to_sentences(self.tokenizer.decode(trimmed), max_sentences), elapsed, tok_count
        elapsed = time.perf_counter() - t0
        return trim_to_sentences(self.tokenizer.decode(generated), max_sentences), elapsed, tok_count

    def cleanup(self):
        self.rack.remove_hooks()


def main():
    parser = argparse.ArgumentParser(description='Production chatbot for DeepSeek 124M')
    parser.add_argument('--base-model', default='artifacts/steerer_v4/steerer_best_b.pt')
    parser.add_argument('--general-steerer', default='artifacts/steerer_v4/steerer_best_b.pt')
    parser.add_argument('--chat-cartridge', default='artifacts/steerer_chat_production_v5_balanced_b384/chat_cartridge.pt')
    parser.add_argument('--knowledge-cartridge', default=None)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--temperature', type=float, default=0.3)
    parser.add_argument('--max-new-tokens', type=int, default=80)
    parser.add_argument('--top-k', type=int, default=40)
    parser.add_argument('--top-p', type=float, default=0.9)
    parser.add_argument('--context-len', type=int, default=128)
    parser.add_argument('--repetition-penalty', type=float, default=1.15)
    parser.add_argument('--stop-ngram', type=int, default=8)
    parser.add_argument('--max-sentences', type=int, default=3)
    parser.add_argument('--interactive', action='store_true')
    parser.add_argument('--prompt', action='append')
    args = parser.parse_args()

    print('Loading production chatbot...', flush=True)
    bot = ProductionChatbot(
        base_model=args.base_model,
        general_steerer=args.general_steerer,
        chat_cartridge=args.chat_cartridge,
        knowledge_cartridge=args.knowledge_cartridge,
        device=args.device,
    )
    print(f'  Cartridges loaded: {bot.rack.list_active()}')
    print(f'  Knowledge: {"yes" if bot.has_knowledge else "no"}')
    print(f'  Params: {sum(p.numel() for p in bot.model.parameters()):,}')
    print(f'  Device: {bot.device}')
    print()

    if args.interactive:
        print("CMI Chatbot — type /quit to exit\n")
        total_tokens = 0
        total_time = 0.0
        turns = 0
        while True:
            try:
                user_text = input('You: ').strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if user_text in {'/quit', '/exit'}:
                break
            if not user_text:
                continue
            answer, elapsed, tok_count = bot.generate(
                user_text,
                max_new_tokens=args.max_new_tokens, temperature=args.temperature,
                top_k=args.top_k, top_p=args.top_p, context_len=args.context_len,
                repetition_penalty=args.repetition_penalty, stop_ngram=args.stop_ngram,
                max_sentences=args.max_sentences,
            )
            total_tokens += tok_count
            total_time += elapsed
            turns += 1
            tps = tok_count / max(elapsed, 0.001)
            print(f'CMI: {answer}')
            print(f'     [{tok_count} tok, {elapsed:.1f}s, {tps:.0f} tok/s]')
            print()

        if turns > 0:
            avg_tps = total_tokens / max(total_time, 0.001)
            print(f'Session: {turns} turns, {total_tokens} tokens, {total_time:.1f}s, {avg_tps:.0f} tok/s avg')

    prompts = args.prompt
    if prompts is None and not args.interactive:
        prompts = [
            'Hello! How are you?',
            'What is the capital of France?',
            'What is the capital of Japan?',
            'Explain what gravity is.',
            'What is the speed of light?',
        ]
    if prompts:
        for prompt in prompts:
            answer, elapsed, tok_count = bot.generate(
                prompt, max_new_tokens=args.max_new_tokens, temperature=args.temperature,
                top_k=args.top_k, top_p=args.top_p, context_len=args.context_len,
                repetition_penalty=args.repetition_penalty, stop_ngram=args.stop_ngram,
                max_sentences=args.max_sentences,
            )
            tps = tok_count / max(elapsed, 0.001)
            print(f'You: {prompt}')
            print(f'CMI: {answer}')
            print(f'     [{tok_count} tok, {elapsed:.1f}s, {tps:.0f} tok/s]\n')

    bot.cleanup()


if __name__ == '__main__':
    main()
