"""Side-by-side GPT-2 + ZeroQ cartridge assistant runtime."""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from hybrid.backends import TrainableSurface, ZeroQPartitionedBackend
from hybrid.cartridges import CartridgeManifest, CartridgeRole, SteererCartridgeRack
from hybrid.compiled_features.gpt2_compiled_channels import GPT2CompiledChannelBuilder
from hybrid.compiled_features.gpt2_feature_adapter import build_token_stat_features_for_span
from hybrid.superposition_steerer_v3 import FeatureConditionedAdapterSteerer


SYSTEM = (
    'You are CMI, a concise and helpful assistant. Answer directly, keep the '
    'conversation coherent, and ask a brief clarifying question when needed.'
)


NUMBER_WORDS = {'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5}


def requested_numbered_items(user_text: str) -> int:
    lowered = user_text.lower()
    for word, count in NUMBER_WORDS.items():
        if re.search(rf'\b{word}\b', lowered):
            return count
    match = re.search(r'\b([1-5])\b', lowered)
    return int(match.group(1)) if match else 0


def has_repeated_tail(token_ids: list[int], ngram: int = 8) -> bool:
    return len(token_ids) >= ngram * 2 and token_ids[-ngram:] == token_ids[-2 * ngram:-ngram]


def trim_to_sentences(text: str, max_sentences: int) -> str:
    if max_sentences <= 0:
        return text.strip()
    matches = list(re.finditer(r'[.!?]', text))
    if len(matches) < max_sentences:
        return text.strip()
    return text[:matches[max_sentences - 1].end()].strip()


def answer_can_stop(text: str, required_items: int = 0) -> bool:
    stripped = text.strip()
    if not stripped or stripped.endswith((':', ',', ';')):
        return False
    if re.search(r'(?:^|\n)\s*\d+\.\s*$', stripped):
        return False
    if required_items > 0 and re.search(r'(?:^|\n)\s*1\.', stripped):
        item_numbers = [int(match.group(1)) for match in re.finditer(r'(?:^|\n)\s*([1-5])\.', stripped)]
        if item_numbers and max(item_numbers) < required_items:
            return False
    return True


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


def sample_next(logits: torch.Tensor, temperature: float, top_k: int, top_p: float,
                generated: list[int] | None = None, repetition_penalty: float = 1.0) -> int:
    logits = apply_repetition_penalty(logits, generated or [], repetition_penalty)
    if temperature <= 0:
        return int(torch.argmax(logits).item())
    logits = logits.float() / max(temperature, 1e-5)
    if top_k > 0:
        values, _ = torch.topk(logits, min(top_k, logits.numel()))
        logits = torch.where(logits < values[-1], torch.full_like(logits, -float('inf')), logits)
    logits = top_p_filter(logits, top_p)
    probs = torch.softmax(logits, dim=-1)
    return int(torch.multinomial(probs, 1).item())


def ensure_single_rank_process_group(device: torch.device, run_dir: Path) -> None:
    if dist.is_available() and dist.is_initialized():
        return
    if not dist.is_available():
        raise RuntimeError('ZeroQ requires torch.distributed to be available')
    run_dir.mkdir(parents=True, exist_ok=True)
    init_file = run_dir / 'zeroq_single_rank_pg'
    if init_file.exists():
        init_file.unlink()
    backend = 'nccl' if device.type == 'cuda' else 'gloo'
    dist.init_process_group(
        backend=backend,
        init_method=f'file://{init_file}',
        rank=0,
        world_size=1,
    )


def gpt2_resident_surface(model) -> TrainableSurface:
    names = [
        name for name, _ in model.named_parameters()
        if name in {'transformer.wte.weight', 'transformer.wpe.weight'}
    ]
    return TrainableSurface.from_names(names) if names else TrainableSurface.frozen()


def build_feature_rows(ids: torch.Tensor, builder: GPT2CompiledChannelBuilder | None) -> torch.Tensor:
    row = ids.detach().cpu().long()
    if builder is None:
        base = build_token_stat_features_for_span(row, start=0, length=row.numel(), history=512)
        pad = torch.zeros(base.shape[0], 21 - base.shape[1], dtype=base.dtype)
        return torch.cat([base, pad], dim=-1)
    return builder.build_features_for_span(row, start=0, length=row.numel(), history=512)


class GPT2ZeroQAssistantRuntime:
    def __init__(
        self,
        *,
        model_name: str = 'gpt2-large',
        cartridge: str | None = None,
        device: str = 'cuda',
        zeroq_path: str = '~/ZeroQ',
        adapter_bottleneck: int = 128,
    ):
        self.device = torch.device(device)
        self.model_name = model_name
        run_dir = REPO / 'artifacts' / 'zeroq_runtime_pg' / re.sub(r'[^A-Za-z0-9_.-]+', '_', model_name)
        ensure_single_rank_process_group(self.device, run_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device.type == 'cuda' else torch.float32,
            low_cpu_mem_usage=True,
        )
        self.model.config.use_cache = False
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        backend = ZeroQPartitionedBackend(device=self.device, zeroq_path=zeroq_path)
        self.backend_handle = backend.prepare(self.model, gpt2_resident_surface(self.model))
        self.model = self.backend_handle.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        self.rack = SteererCartridgeRack()
        self.steerer: FeatureConditionedAdapterSteerer | None = None
        self.compiled_builder: GPT2CompiledChannelBuilder | None = None
        self.cartridge_id = 'gpt2-large-chat-capability'
        self.inject_layers = self._default_inject_layers()

        if cartridge:
            ckpt = torch.load(REPO / cartridge, map_location='cpu', weights_only=False)
            self.inject_layers = list(ckpt.get('inject_layers', self.inject_layers))
            adapter_bottleneck = int(ckpt.get('adapter_bottleneck', adapter_bottleneck))
            self.compiled_builder = GPT2CompiledChannelBuilder.from_state_dict(ckpt['compiled_builder_state'])
            self.steerer = FeatureConditionedAdapterSteerer(
                d_model=self.model.config.n_embd,
                inject_layers=self.inject_layers,
                bottleneck=adapter_bottleneck,
                init_scale=0.005,
                noise_scale=0.0,
            ).to(self.device)
            self.steerer.load_state_dict(ckpt['steerer_state'])
            self.steerer.eval()
            manifest = CartridgeManifest(
                cartridge_id=self.cartridge_id,
                role=CartridgeRole.TASK_CAPABILITY,
                base_model_id=model_name,
                tokenizer_id=model_name,
                steerer_class='FeatureConditionedAdapterSteerer',
                inject_layers=tuple(self.inject_layers),
                parameter_count=sum(param.numel() for param in self.steerer.parameters()),
                source_corpus=str(ckpt.get('data_dir', 'chat_steerer')),
                metadata={'runtime': 'gpt2_zeroq_assistant', 'zeroq': True},
            )
            self.rack.mount(manifest, self.steerer, weight=1.0, active=True)
            self.rack.register_hooks(self.model)

    def _default_inject_layers(self) -> list[int]:
        n_layer = int(getattr(self.model.config, 'n_layer', 36))
        return [idx for idx in (0, 2, 4, 8, 12, 16, 20, 24, 30) if idx < n_layer]

    def set_cartridge_enabled(self, enabled: bool):
        if self.steerer is not None:
            self.rack.activate(self.cartridge_id, enabled)

    def format_prompt(self, user_text: str, history: list[tuple[str, str]] | None = None) -> str:
        parts = [f'System:\n{SYSTEM}\n\n']
        for user, assistant in history or []:
            parts.append(f'User:\n{user}\n\nAssistant:\n{assistant}\n\n')
        parts.append(f'User:\n{user_text}\n\nAssistant:\n')
        return ''.join(parts)

    @torch.no_grad()
    def generate(
        self,
        user_text: str,
        *,
        history: list[tuple[str, str]] | None = None,
        use_cartridge: bool = True,
        max_new_tokens: int = 120,
        temperature: float = 0.0,
        top_k: int = 40,
        top_p: float = 0.9,
        context_len: int = 512,
        repetition_penalty: float = 1.12,
        stop_ngram: int = 8,
        max_sentences: int = 0,
    ) -> str:
        self.set_cartridge_enabled(use_cartridge and self.steerer is not None)
        ids = self.tokenizer.encode(self.format_prompt(user_text, history))
        generated: list[int] = []
        required_items = requested_numbered_items(user_text)
        stop_markers = [
            '\n\nUser:', '\nUser:', '\n\nUser', '\nUser',
            '\n\nSystem:', '\nSystem:', '\n\nSystem', '\nSystem',
            '\n\nAssistant:', '\nAssistant:', '\nQuestion:', '\nQ:',
        ]

        for _ in range(max_new_tokens):
            ctx = ids[-context_len:]
            x = torch.tensor([ctx], dtype=torch.long, device=self.device)
            if use_cartridge and self.steerer is not None:
                features = build_feature_rows(x[0], self.compiled_builder).unsqueeze(0).to(self.device)
                self.rack.set_weights(features)
            logits = self.model(input_ids=x, use_cache=False).logits[0, -1].float()
            next_id = sample_next(
                logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                generated=generated,
                repetition_penalty=repetition_penalty,
            )
            if next_id == self.tokenizer.eos_token_id:
                text = self.tokenizer.decode(generated)
                if answer_can_stop(text, required_items):
                    return trim_to_sentences(text, max_sentences)
                continue
            ids.append(next_id)
            generated.append(next_id)
            text = self.tokenizer.decode(generated)
            for marker in stop_markers:
                if marker in text:
                    return trim_to_sentences(text.split(marker)[0], max_sentences)
            if re.search(r'\n\s*(System|User|Assistant)\s*$', text):
                return trim_to_sentences(text, max_sentences)
            trimmed = trim_to_sentences(text, max_sentences)
            if trimmed != text.strip():
                return trimmed
            if stop_ngram > 0 and has_repeated_tail(generated, stop_ngram):
                return trim_to_sentences(self.tokenizer.decode(generated[:-stop_ngram]), max_sentences)
        return trim_to_sentences(self.tokenizer.decode(generated), max_sentences)

    def cleanup(self):
        self.rack.remove_hooks()
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-name', default='gpt2-large')
    parser.add_argument('--cartridge')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--zeroq-path', default='~/ZeroQ')
    parser.add_argument('--prompt', action='append')
    parser.add_argument('--history-json')
    parser.add_argument('--compare', action='store_true')
    parser.add_argument('--interactive', action='store_true')
    parser.add_argument('--max-new-tokens', type=int, default=120)
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--top-k', type=int, default=40)
    parser.add_argument('--top-p', type=float, default=0.9)
    parser.add_argument('--context-len', type=int, default=512)
    parser.add_argument('--report')
    args = parser.parse_args()

    runtime = GPT2ZeroQAssistantRuntime(
        model_name=args.model_name,
        cartridge=args.cartridge,
        device=args.device,
        zeroq_path=args.zeroq_path,
    )
    history = json.loads(args.history_json) if args.history_json else None
    rows = []

    def run_one(prompt: str):
        if args.compare:
            baseline = runtime.generate(
                prompt,
                history=history,
                use_cartridge=False,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                context_len=args.context_len,
            )
            cartridge = runtime.generate(
                prompt,
                history=history,
                use_cartridge=True,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                context_len=args.context_len,
            )
            print(f'\n[baseline] User: {prompt}\nAssistant: {baseline}')
            print(f'\n[cartridge] User: {prompt}\nAssistant: {cartridge}')
            rows.append({'prompt': prompt, 'baseline': baseline, 'cartridge': cartridge})
        else:
            answer = runtime.generate(
                prompt,
                history=history,
                use_cartridge=True,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                context_len=args.context_len,
            )
            print(f'\nUser: {prompt}\nAssistant: {answer}')
            rows.append({'prompt': prompt, 'answer': answer})

    for prompt in args.prompt or []:
        run_one(prompt)

    if args.interactive:
        while True:
            try:
                prompt = input('\nYou: ').strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if prompt in {'/quit', '/exit'}:
                break
            if prompt:
                run_one(prompt)

    if args.report:
        report_path = REPO / args.report
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(rows, indent=2), encoding='utf-8')
    runtime.cleanup()


if __name__ == '__main__':
    main()
