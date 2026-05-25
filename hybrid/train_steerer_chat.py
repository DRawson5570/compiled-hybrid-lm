"""Train a chat capability cartridge beside an optional frozen superposition steerer."""
from __future__ import annotations

import argparse
import importlib.util
import math
import pickle
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

_spec = importlib.util.spec_from_file_location('scaled', str(REPO / 'hybrid/train_scaled_neural_lm.py'))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
DeepCausalLM = _mod.DeepCausalLM
sys.path.insert(0, str(REPO))

from hybrid.cartridges import CartridgeManifest, CartridgeRole, SteererCartridgeRack
from hybrid.gpu_channels import GPUFeatureComputer
from hybrid.superposition_steerer_v3 import SuperpositionSteererV3
from hybrid.train_steerer_v4 import FastNgramFeatures, PUNCT_IDS, compute_cpu_features

V = 50257


def load_model_from_state(state: dict[str, torch.Tensor], device: torch.device) -> tuple[DeepCausalLM, int, int]:
    d_model = state['pos_emb.weight'].shape[-1]
    vocab = state['head_bias'].shape[0]
    d_ff = state['encoder.layers.0.linear1.weight'].shape[0]
    n_layers = len([k for k in state if k.startswith('encoder.layers.') and k.endswith('.norm1.weight')])
    n_heads = state['encoder.layers.0.self_attn.in_proj_weight'].shape[0] // (3 * d_model)
    max_len = state['pos_emb.weight'].shape[0]
    model = DeepCausalLM(vocab=vocab, d_model=d_model, n_layers=n_layers,
                         n_heads=n_heads, d_ff=d_ff, max_len=max_len, dropout=0.0)
    model.load_state_dict(state)
    model = model.to(device)
    return model, d_model, vocab


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


def sample_batch(token_ids: torch.Tensor, batch: int, seq_len: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    max_start = max(1, len(token_ids) - seq_len - 1)
    starts = torch.randint(0, max_start, (batch,))
    x = torch.stack([token_ids[start:start + seq_len] for start in starts]).to(device)
    y = torch.stack([token_ids[start + 1:start + seq_len + 1] for start in starts]).to(device)
    return x, y


def eval_mode(model, rack, token_ids, vocab, gpu_fc, cpu_ch, device, mode: str, seq_len: int) -> float:
    rack.activate('general-superposition', mode in {'superposition', 'chat'})
    rack.activate('chat-capability', mode == 'chat')
    nll = 0.0
    count = 0
    with torch.no_grad():
        for start in range(0, len(token_ids) - 1, seq_len):
            current_len = min(seq_len, len(token_ids) - start - 1)
            if current_len <= 0:
                continue
            x = token_ids[start:start + current_len].unsqueeze(0).to(device)
            y = token_ids[start + 1:start + current_len + 1].unsqueeze(0).to(device)
            if mode != 'base':
                rack.set_weights(compute_weights(x, gpu_fc, cpu_ch))
            logits = model(x)
            nll += F.cross_entropy(logits.reshape(-1, vocab), y.reshape(-1), reduction='sum').item()
            count += current_len
    return math.exp(nll / max(count, 1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-model', type=str, default='artifacts/steerer_v4/steerer_best_b.pt')
    parser.add_argument('--general-steerer', type=str, default='artifacts/steerer_v4/steerer_best_s.pt')
    parser.add_argument('--data-dir', type=str, default='artifacts/chat_steerer')
    parser.add_argument('--out-dir', type=str, default='artifacts/steerer_chat')
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--steps', type=int, default=200)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--seq-len', type=int, default=96)
    parser.add_argument('--lr', type=float, default=3e-3)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    device = torch.device(args.device)
    out_dir = REPO / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print('=' * 72)
    print(' CHAT CAPABILITY CARTRIDGE TRAINING (frozen base + optional steerer)')
    print('=' * 72)

    train_ids = torch.load(REPO / args.data_dir / 'train_ids.pt', weights_only=False).long()
    val_ids = torch.load(REPO / args.data_dir / 'validation_ids.pt', weights_only=False).long()
    print(f'[data] train={len(train_ids):,} val={len(val_ids):,}')

    base_ckpt = torch.load(REPO / args.base_model, map_location=device, weights_only=False)
    model, d_model, vocab = load_model_from_state(base_ckpt['state_dict'], device)
    for param in model.parameters():
        param.requires_grad = False
    model.eval()
    print(f'[model] frozen params={sum(p.numel() for p in model.parameters()):,} d_model={d_model}')

    general = SuperpositionSteererV3(d_model=d_model, init_scale=0.01, noise_scale=0.0).to(device)
    general_ckpt = torch.load(REPO / args.general_steerer, map_location=device, weights_only=False)
    general.load_state_dict(general_ckpt['steerer_state'])
    for param in general.parameters():
        param.requires_grad = False
    general.eval()

    chat = SuperpositionSteererV3(d_model=d_model, init_scale=0.01, noise_scale=0.03).to(device)
    chat.train()
    print(f'[chat] trainable params={sum(p.numel() for p in chat.parameters()):,}')

    rack = SteererCartridgeRack()
    rack.mount(
        CartridgeManifest('general-superposition', CartridgeRole.SUPERPOSITION_STEERER,
                          base_model_id='c4-124m-v4', tokenizer_id='gpt2-bpe'),
        general,
        weight=1.0,
    )
    rack.mount(
        CartridgeManifest('chat-capability', CartridgeRole.TASK_CAPABILITY,
                          base_model_id='c4-124m-v4', tokenizer_id='gpt2-bpe',
                          source_corpus='synthetic-chat-seed'),
        chat,
        weight=1.0,
    )
    hooks = rack.register_hooks(model)
    print(f'[rack] hooks={hooks} active={rack.list_active()}')

    gpu_fc = load_feature_computer(device)
    cpu_ch = FastNgramFeatures(V)
    opt = torch.optim.AdamW(chat.parameters(), lr=args.lr, weight_decay=0.05)
    best_chat = float('inf')

    for epoch in range(1, args.epochs + 1):
        chat.train()
        general.eval()
        model.eval()
        rack.activate('general-superposition', True)
        rack.activate('chat-capability', True)
        total_loss = 0.0
        t0 = time.time()

        for _ in range(args.steps):
            x, y = sample_batch(train_ids, args.batch, args.seq_len, device)
            rack.set_weights(compute_weights(x, gpu_fc, cpu_ch))
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, vocab), y.reshape(-1))
            loss = loss + 0.001 * chat.orthogonal_penalty()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(chat.parameters(), 1.0)
            opt.step()
            total_loss += loss.item()

        chat.eval()
        eval_base = eval_mode(model, rack, val_ids, vocab, gpu_fc, cpu_ch, device, 'base', args.seq_len)
        eval_super = eval_mode(model, rack, val_ids, vocab, gpu_fc, cpu_ch, device, 'superposition', args.seq_len)
        eval_chat = eval_mode(model, rack, val_ids, vocab, gpu_fc, cpu_ch, device, 'chat', args.seq_len)
        avg_loss = total_loss / max(args.steps, 1)
        status = ''
        if eval_chat < best_chat:
            best_chat = eval_chat
            torch.save({
                'steerer_state': chat.state_dict(),
                'manifest': rack._mounted['chat-capability'].manifest.__dict__,
                'eval_base': eval_base,
                'eval_superposition': eval_super,
                'eval_chat': eval_chat,
                'epoch': epoch,
                'opt_state': opt.state_dict(),
            }, out_dir / 'chat_cartridge.pt')
            status = 'SAVED'

        elapsed = time.time() - t0
        print(
            f'  epoch={epoch:3d} loss={avg_loss:.4f} ppl={math.exp(avg_loss):.1f} '
            f'eval_base={eval_base:.1f} eval_super={eval_super:.1f} '
            f'eval_chat={eval_chat:.1f} best_chat={best_chat:.1f} {status} time={elapsed:.0f}s',
            flush=True,
        )

    rack.remove_hooks()
    print(f'\nDone. Best chat cartridge eval_chat={best_chat:.1f}')


if __name__ == '__main__':
    main()