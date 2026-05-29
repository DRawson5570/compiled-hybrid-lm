"""DeepSeek cartridge rack system for deploying multiple hot-swappable steerers
on a frozen DeepCausalLM V4 model.

Runtime wraps the frozen base model + general superposition steerer and allows
mounting additional task/domain cartridges (chat, knowledge, etc.) through the
SteererCartridgeRack. A prompt router classifies prompts and selects the right
cartridge automatically.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import pickle
import sys
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

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
        semantic_dim = 16
        if feat_shape is not None and feat_shape.shape[-1] == 21:
            semantic_dim = 0
        return FeatureConditionedAdapterSteerer(
            d_model=d_model, bottleneck=bottleneck, init_scale=0.01, noise_scale=0.0,
            semantic_dim=semantic_dim).to(device).eval()
    return SuperpositionSteererV3(d_model=d_model, init_scale=0.01, noise_scale=0.0).to(device).eval()


class DeepSeekCartridgeRuntime:
    """Frozen DeepCausalLM V4 runtime with hot-swappable steering cartridges.

    Wraps the frozen base model with a SteererCartridgeRack. The general
    superposition steerer is always mounted and can be toggled on/off.
    Additional task cartridges (chat, knowledge) can be loaded and
    activated selectively.
    """

    def __init__(self, base_model_path: str, general_steerer_path: str | None = None,
                 device: str = 'cuda'):
        self.device = torch.device(device)

        base_ckpt = torch.load(REPO / base_model_path, map_location=self.device, weights_only=False)
        self.model, d_model = load_model_from_state(base_ckpt['state_dict'], self.device)
        for p in self.model.parameters():
            p.requires_grad = False

        self.d_model = d_model
        self.tokenizer = AutoTokenizer.from_pretrained('gpt2')
        self.gpu_fc = load_feature_computer(self.device)
        self.cpu_ch = FastNgramFeatures(V)
        self.rack = SteererCartridgeRack()

        if general_steerer_path is None:
            general_steerer_path = base_model_path

        self.general = SuperpositionSteererV3(d_model=d_model, init_scale=0.01, noise_scale=0.0).to(self.device).eval()
        general_ckpt = torch.load(REPO / general_steerer_path, map_location=self.device, weights_only=False)
        if 'steerer_state' in general_ckpt:
            self.general.load_state_dict(general_ckpt['steerer_state'], strict=False)

        self.rack.mount(
            CartridgeManifest(
                'general-superposition', CartridgeRole.SUPERPOSITION_STEERER,
                base_model_id='c4-124m-v4', tokenizer_id='gpt2-bpe'),
            self.general,
        )
        self.rack.register_hooks(self.model)
        self.rack.activate('general-superposition', True)

    def load_cartridge(self, path: str, cartridge_id: str | None = None,
                       role: str | CartridgeRole = 'TASK_CAPABILITY',
                       weight: float = 1.0, active: bool = True) -> str:
        ckpt = torch.load(REPO / path, map_location=self.device, weights_only=False)
        steerer = build_steerer_from_checkpoint(ckpt, self.d_model, self.device)
        if 'steerer_state' in ckpt:
            steerer.load_state_dict(ckpt['steerer_state'], strict=False)

        if cartridge_id is None:
            cartridge_id = Path(path).stem

        manifest = CartridgeManifest(
            cartridge_id=cartridge_id,
            role=role,
            base_model_id='c4-124m-v4',
            tokenizer_id='gpt2-bpe',
            steerer_class=ckpt.get('steerer_class', 'SuperpositionSteererV3'),
        )
        self.rack.mount(manifest, steerer, weight=weight, active=active)
        self.rack.register_hooks(self.model)
        return cartridge_id

    def set_mode(self, cartridge_ids: list[str] | None = None):
        if cartridge_ids is None:
            cartridge_ids = ['general-superposition']
        for cid in self.rack.list_active():
            self.rack.activate(cid, False)
        for cid in cartridge_ids:
            self.rack.activate(cid, True)

    def activate_only(self, cartridge_id: str | None):
        for cid in self.rack.list_active():
            self.rack.activate(cid, False)
        if cartridge_id is not None:
            self.rack.activate(cartridge_id, True)

    @torch.no_grad()
    def generate(self, prompt: str, max_tokens: int = 60, temperature: float = 0.7,
                 top_k: int = 40, top_p: float = 0.9, context_len: int = 128) -> str:
        ids = self.tokenizer.encode(prompt)
        generated: list[int] = []

        for _ in range(max_tokens):
            ctx = ids[-context_len:]
            x = torch.tensor([ctx], dtype=torch.long, device=self.device)
            if self.rack.list_active():
                self.rack.set_weights(compute_weights(x, self.gpu_fc, self.cpu_ch))
            logits = self.model(x)[0, -1].float()

            if temperature <= 0:
                next_id = int(torch.argmax(logits).item())
            else:
                logits = logits / max(temperature, 1e-5)
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

            if next_id == self.tokenizer.eos_token_id:
                break
            ids.append(next_id)
            generated.append(next_id)

        return self.tokenizer.decode(generated)

    @torch.no_grad()
    def prompt_embedding(self, prompt: str) -> torch.Tensor:
        ids = self.tokenizer.encode(prompt)
        x = torch.tensor([ids], dtype=torch.long, device=self.device)
        self.rack.set_weights(compute_weights(x, self.gpu_fc, self.cpu_ch))
        B, T = x.shape
        pos = torch.arange(T, device=self.device).unsqueeze(0).expand(B, T)
        h = self.model.tok_emb(x) + self.model.pos_emb(pos)
        h = self.model.dropout(h)
        mask = torch.nn.Transformer.generate_square_subsequent_mask(T, device=self.device)
        h = self.model.encoder(h, mask=mask, is_causal=True)
        h = self.model.ln_f(h)
        embedding = h[0].mean(dim=0).float().cpu()
        return embedding

    def cleanup(self):
        self.rack.remove_hooks()


class DeepSeekPromptRouter:
    """Prompts router for the DeepCausalLM cartridge rack.

    Uses hidden-state mean-pooling + cosine similarity against precomputed
    label embeddings to route prompts to the correct cartridge. Falls back
    to keyword-based heuristics when a runtime is not available.
    """

    _ROUTE_DESCRIPTIONS = {
        'chat-capability': 'conversation greeting hello hi help assistant chat talk question how what why',
        'knowledge-cartridge': 'definition define what means explain fact knowledge encyclopedia science history geography biology chemistry math physics',
        'general-superposition': 'general default',
    }

    def __init__(self, model: torch.nn.Module | None = None,
                 tokenizer: AutoTokenizer | None = None, device: str = 'cuda'):
        self.device = torch.device(device)
        self.model = model
        self.tokenizer = tokenizer
        self._label_embeddings: dict[str, torch.Tensor] | None = None
        self._head: torch.nn.Linear | None = None
        self._cartridge_ids: tuple[str, ...] = ()

    def _keyword_route(self, prompt: str, available: set[str]) -> str | None:
        text = prompt.lower()
        scores: list[tuple[int, str]] = []
        for cartridge_id, keywords in self._ROUTE_DESCRIPTIONS.items():
            if cartridge_id not in available:
                continue
            score = sum(1 for kw in keywords.split() if kw in text)
            if score:
                scores.append((score, cartridge_id))
        if not scores:
            return None
        scores.sort(reverse=True)
        return scores[0][1]

    def route(self, prompt: str, available: Iterable[str] | None = None) -> str:
        available_set = set(available or self._ROUTE_DESCRIPTIONS.keys())

        if self._head is not None and self.model is not None and self.tokenizer is not None:
            embedding = self._embed_prompt(prompt)
            selected = self._linear_route(embedding, available_set)
            if selected is not None:
                return selected

        result = self._keyword_route(prompt, available_set)
        if result is not None:
            return result
        if 'general-superposition' in available_set:
            return 'general-superposition'
        return next(iter(available_set), 'general-superposition')

    @torch.no_grad()
    def _embed_prompt(self, prompt: str) -> torch.Tensor:
        ids = self.tokenizer.encode(prompt)
        x = torch.tensor([ids], dtype=torch.long, device=self.device)
        B, T = x.shape
        pos = torch.arange(T, device=self.device).unsqueeze(0).expand(B, T)
        h = self.model.tok_emb(x) + self.model.pos_emb(pos)
        h = self.model.dropout(h)
        mask = torch.nn.Transformer.generate_square_subsequent_mask(T, device=self.device)
        h = self.model.encoder(h, mask=mask, is_causal=True)
        h = self.model.ln_f(h)
        return h[0].mean(dim=0).float().cpu()

    def _linear_route(self, embedding: torch.Tensor, available_set: set[str]) -> str | None:
        valid_ids = [self._cartridge_ids.index(cid) for cid in self._cartridge_ids if cid in available_set]
        if not valid_ids:
            return None
        inp = embedding.unsqueeze(0).to(self.device)
        logits = self._head(inp)[0]
        mask = torch.full_like(logits, -1e9)
        for idx in valid_ids:
            mask[idx] = 0
        logits = logits + mask
        best_idx = int(logits.argmax().item())
        return self._cartridge_ids[best_idx]

    def load_router(self, path: str | Path):
        payload = torch.load(Path(path), map_location=self.device, weights_only=False)
        if payload.get('router_type') != 'deepseek_embedding_linear_v1':
            raise ValueError(f"unsupported router type: {payload.get('router_type')}")
        self._cartridge_ids = tuple(payload['cartridge_ids'])
        self._head = torch.nn.Linear(int(payload['d_model']), len(self._cartridge_ids)).to(self.device)
        self._head.load_state_dict(payload['head_state'])
        self._head.eval()


def train_deepseek_router(
    *,
    runtime: DeepSeekCartridgeRuntime,
    out_dir: str | Path,
    epochs: int = 300,
    lr: float = 3e-3,
) -> dict:
    labeled_examples: list[tuple[str, str]] = [
        ('Hello! How are you?', 'chat-capability'),
        ('Hi there!', 'chat-capability'),
        ('What is the capital of France?', 'knowledge-cartridge'),
        ('Explain the theory of relativity.', 'knowledge-cartridge'),
        ('Define photosynthesis.', 'knowledge-cartridge'),
        ('Who was Marie Curie?', 'knowledge-cartridge'),
        ('What is the Pythagorean theorem?', 'knowledge-cartridge'),
        ('Tell me about the solar system.', 'knowledge-cartridge'),
        ('How does gravity work?', 'knowledge-cartridge'),
        ('What is DNA made of?', 'knowledge-cartridge'),
        ('Can you help me with something?', 'chat-capability'),
        ('Good morning!', 'chat-capability'),
        ('What does "entropy" mean in thermodynamics?', 'knowledge-cartridge'),
        ('Describe the process of cellular respiration.', 'knowledge-cartridge'),
        ('How are you doing today?', 'chat-capability'),
        ("What's the difference between a planet and a star?", 'knowledge-cartridge'),
        ('Nice to meet you.', 'chat-capability'),
        ('Explain quantum entanglement.', 'knowledge-cartridge'),
    ]

    cartridge_ids = tuple(sorted({'chat-capability', 'knowledge-cartridge', 'general-superposition'}))
    label_idx = {cid: i for i, cid in enumerate(cartridge_ids)}

    train_examples = labeled_examples[:14]
    val_examples = labeled_examples[14:]

    train_x = torch.stack([runtime.prompt_embedding(prompt) for prompt, _ in train_examples])
    train_y = torch.tensor([label_idx[label] for _, label in train_examples], dtype=torch.long)
    val_x = torch.stack([runtime.prompt_embedding(prompt) for prompt, _ in val_examples])
    val_y = torch.tensor([label_idx[label] for _, label in val_examples], dtype=torch.long)

    head = torch.nn.Linear(train_x.shape[1], len(cartridge_ids)).to(runtime.device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)

    train_x_dev = train_x.to(runtime.device)
    train_y_dev = train_y.to(runtime.device)
    best_state = None
    best_val = -1.0
    history: list[dict] = []

    for epoch in range(1, epochs + 1):
        head.train()
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(head(train_x_dev), train_y_dev)
        loss.backward()
        optimizer.step()

        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            head.eval()
            with torch.no_grad():
                train_acc = float((head(train_x.to(runtime.device)).argmax(dim=-1).cpu() == train_y).float().mean())
                val_acc = float((head(val_x.to(runtime.device)).argmax(dim=-1).cpu() == val_y).float().mean())
            history.append({'epoch': epoch, 'loss': float(loss.item()), 'train_accuracy': train_acc, 'val_accuracy': val_acc})
            if val_acc > best_val:
                best_val = val_acc
                best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}

    if best_state is None:
        best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}

    head.load_state_dict(best_state)
    head.eval()
    with torch.no_grad():
        train_acc = float((head(train_x.to(runtime.device)).argmax(dim=-1).cpu() == train_y).float().mean())
        val_acc = float((head(val_x.to(runtime.device)).argmax(dim=-1).cpu() == val_y).float().mean())

    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact = output_dir / 'deepseek_learned_router.pt'
    payload = {
        'router_type': 'deepseek_embedding_linear_v1',
        'd_model': runtime.d_model,
        'cartridge_ids': cartridge_ids,
        'head_state': best_state,
        'train_accuracy': train_acc,
        'val_accuracy': val_acc,
        'train_count': len(train_examples),
        'val_count': len(val_examples),
        'history': history,
    }
    torch.save(payload, artifact)
    report = {k: v for k, v in payload.items() if k != 'head_state'}
    report['artifact'] = str(artifact)
    (output_dir / 'deepseek_learned_router_report.json').write_text(
        json.dumps(report, indent=2), encoding='utf-8')
    return report


def main():
    parser = argparse.ArgumentParser(description='DeepSeek cartridge rack demo')
    parser.add_argument('--base-model', default='artifacts/steerer_v4/steerer_best_b.pt')
    parser.add_argument('--general-steerer', default=None)
    parser.add_argument('--chat-cartridge', default=None)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--mode', choices=['base', 'general', 'chat', 'auto'], default='chat')
    parser.add_argument('--prompt', action='append')
    parser.add_argument('--interactive', action='store_true')
    parser.add_argument('--max-tokens', type=int, default=60)
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--train-router', action='store_true')
    parser.add_argument('--router-out', default='artifacts/deepseek_router/')
    args = parser.parse_args()

    runtime = DeepSeekCartridgeRuntime(
        base_model_path=args.base_model,
        general_steerer_path=args.general_steerer or args.base_model,
        device=args.device,
    )

    if args.chat_cartridge:
        try:
            runtime.load_cartridge(args.chat_cartridge, cartridge_id='chat-capability',
                                   role=CartridgeRole.TASK_CAPABILITY)
            print(f'[loaded] chat cartridge: {args.chat_cartridge}')
        except Exception as e:
            print(f'[warn] could not load chat cartridge: {e}')

    router = DeepSeekPromptRouter(
        model=runtime.model,
        tokenizer=runtime.tokenizer,
        device=args.device,
    )

    if args.train_router:
        report = train_deepseek_router(runtime=runtime, out_dir=args.router_out)
        print(json.dumps(report, indent=2))
        router.load_router(report['artifact'])

    if args.mode == 'base':
        runtime.activate_only(None)
    elif args.mode == 'general':
        runtime.set_mode(['general-superposition'])
    elif args.mode == 'chat':
        if 'chat-capability' in runtime.rack.list_active():
            runtime.set_mode(['general-superposition', 'chat-capability'])
        else:
            runtime.set_mode(['general-superposition'])
    elif args.mode == 'auto':
        pass

    def generate_with_route(prompt: str) -> str:
        if args.mode == 'auto':
            available = list(runtime.rack.list_active())
            chosen = router.route(prompt, available)
            print(f'  [router] -> {chosen}')
            runtime.activate_only(chosen)
        return runtime.generate(prompt, max_tokens=args.max_tokens, temperature=args.temperature)

    prompts = args.prompt
    if prompts is None and not args.interactive:
        prompts = [
            'Hello! How are you?',
            'Explain what gravity is.',
            'What is the capital of France?',
            'Define photosynthesis in simple terms.',
        ]

    if prompts:
        for prompt in prompts:
            answer = generate_with_route(prompt)
            print(f'\n[{args.mode}] Prompt: {prompt}\nAssistant: {answer}', flush=True)

    if args.interactive:
        while True:
            try:
                user_text = input('\nYou: ').strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if user_text in {'/quit', '/exit'}:
                break
            answer = generate_with_route(user_text)
            print(f'CMI: {answer}')

    runtime.cleanup()


if __name__ == '__main__':
    main()
