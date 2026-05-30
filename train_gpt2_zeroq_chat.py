"""Train a CMI chat cartridge on a frozen ZeroQ GPT-2-family substrate."""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from hybrid.assistant_eval import DEFAULT_TASKS, score_answer, summarize
from hybrid.backends import TrainableSurface, ZeroQPartitionedBackend
from hybrid.cartridges import CartridgeManifest, CartridgeRole, SteererCartridgeRack
from hybrid.compiled_features.gpt2_compiled_channels import (
    GPT2CompiledChannelBuilder,
    GPT2CompiledChannelConfig,
)
from hybrid.gpt2_zeroq_assistant import build_feature_rows
from hybrid.gpt2_zeroq_assistant import gpt2_resident_surface
from hybrid.superposition_steerer_v3 import FeatureConditionedAdapterSteerer


def ensure_single_rank_process_group(device: torch.device, out_dir: Path) -> None:
    if dist.is_available() and dist.is_initialized():
        return
    if not dist.is_available():
        raise RuntimeError('ZeroQ requires torch.distributed to be available')
    out_dir.mkdir(parents=True, exist_ok=True)
    init_file = out_dir / 'zeroq_single_rank_pg'
    if init_file.exists():
        init_file.unlink()
    backend = 'nccl' if device.type == 'cuda' else 'gloo'
    dist.init_process_group(
        backend=backend,
        init_method=f'file://{init_file}',
        rank=0,
        world_size=1,
    )


def sample_example_batch(
    examples: list[dict[str, torch.Tensor]],
    batch: int,
    seq_len: int,
    device: torch.device,
    pad_id: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    xs: list[torch.Tensor] = []
    ys: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []
    for _ in range(batch):
        item = examples[int(torch.randint(0, len(examples), ()).item())]
        ids = item['ids'].long()
        loss_mask = item['mask'].float()
        if len(ids) <= seq_len + 1:
            pad_len = seq_len + 1 - len(ids)
            window_ids = torch.cat([ids, torch.full((pad_len,), pad_id, dtype=torch.long)])
            window_mask = torch.cat([loss_mask, torch.zeros(pad_len, dtype=torch.float32)])
        else:
            max_start = len(ids) - seq_len - 1
            loss_positions = torch.nonzero(loss_mask > 0, as_tuple=False).flatten()
            if len(loss_positions) and torch.rand(()).item() < 0.9:
                target = int(loss_positions[int(torch.randint(0, len(loss_positions), ()).item())].item())
                low = max(0, target - seq_len + 1)
                high = min(target, max_start)
                start = int(torch.randint(low, high + 1, ()).item()) if high >= low else min(target, max_start)
            else:
                start = int(torch.randint(0, max_start + 1, ()).item())
            window_ids = ids[start:start + seq_len + 1]
            window_mask = loss_mask[start:start + seq_len + 1]
        xs.append(window_ids[:-1])
        ys.append(window_ids[1:])
        masks.append(window_mask[1:])
    return torch.stack(xs).to(device), torch.stack(ys).to(device), torch.stack(masks).to(device)


def masked_ce(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    losses = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        targets.reshape(-1),
        reduction='none',
    ).reshape_as(targets)
    return (losses * mask).sum() / mask.sum().clamp(min=1.0)


def batch_features(x: torch.Tensor, builder: GPT2CompiledChannelBuilder, device: torch.device) -> torch.Tensor:
    rows = [build_feature_rows(row, builder) for row in x.detach().cpu()]
    return torch.stack(rows, dim=0).to(device)


def default_inject_layers(n_layer: int) -> list[int]:
    return [idx for idx in (0, 2, 4, 8, 12, 16, 20, 24, 30) if idx < n_layer]


def evaluate_loss(model, rack, steerer_id: str, examples, builder, batch: int, seq_len: int,
                  device: torch.device, pad_id: int, batches: int) -> float:
    if batches <= 0:
        return float('nan')
    rack.activate(steerer_id, True)
    nll = 0.0
    tokens = 0.0
    with torch.no_grad():
        for _ in range(batches):
            x, y, mask = sample_example_batch(examples, batch, seq_len, device, pad_id)
            rack.set_weights(batch_features(x, builder, device))
            logits = model(input_ids=x, use_cache=False).logits.float()
            losses = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1), reduction='none').reshape_as(y)
            nll += float((losses * mask).sum().item())
            tokens += float(mask.sum().item())
    return math.exp(nll / max(tokens, 1.0))


def run_assistant_probe(runtime_args: dict, report_path: Path, task_limit: int) -> dict:
    from hybrid.gpt2_zeroq_assistant import GPT2ZeroQAssistantRuntime

    runtime = GPT2ZeroQAssistantRuntime(**runtime_args)
    tasks = DEFAULT_TASKS[:task_limit] if task_limit > 0 else DEFAULT_TASKS
    rows = []
    for task in tasks:
        answer = runtime.generate(
            task.prompt,
            history=list(task.history),
            use_cartridge=True,
            max_new_tokens=140,
            temperature=0.0,
            max_sentences=0,
        )
        rows.append(score_answer(task, answer))
    summary = summarize(rows)
    payload = {'summary': summary, 'rows': [row.to_json() for row in rows]}
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    runtime.cleanup()
    return summary


def save_checkpoint(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    torch.save(payload, tmp)
    tmp.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-name', default='gpt2-large')
    parser.add_argument('--data-dir', default='artifacts/chat_steerer_instruction_v7_examples')
    parser.add_argument('--out-dir', default='artifacts/gpt2_large_zeroq_chat')
    parser.add_argument('--zeroq-path', default='~/ZeroQ')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--epochs', type=int, default=12)
    parser.add_argument('--steps', type=int, default=180)
    parser.add_argument('--batch', type=int, default=1)
    parser.add_argument('--seq-len', type=int, default=96)
    parser.add_argument('--lr', type=float, default=8e-5)
    parser.add_argument('--weight-decay', type=float, default=0.01)
    parser.add_argument('--adapter-bottleneck', type=int, default=128)
    parser.add_argument('--compile-tokens', type=int, default=400000)
    parser.add_argument('--eval-batches', type=int, default=12)
    parser.add_argument('--probe-every', type=int, default=0)
    parser.add_argument('--probe-task-limit', type=int, default=8)
    parser.add_argument('--resume-cartridge')
    parser.add_argument('--seed', type=int, default=20260525)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    out_dir = REPO / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    ensure_single_rank_process_group(device, out_dir)

    print('=' * 80)
    print(' GPT-2 ZEROQ CHAT CARTRIDGE TRAINING')
    print('=' * 80)
    print(f'[config] model={args.model_name} data={args.data_dir} device={device} zeroq={args.zeroq_path}')

    data_dir = REPO / args.data_dir
    train_examples = torch.load(data_dir / 'train_examples.pt', map_location='cpu', weights_only=False)
    val_examples_path = data_dir / 'validation_examples.pt'
    val_examples = torch.load(val_examples_path, map_location='cpu', weights_only=False) if val_examples_path.exists() else train_examples[:256]
    train_ids = torch.load(data_dir / 'train_ids.pt', map_location='cpu', weights_only=False).long()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    pad_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 50256
    print(f'[data] train_examples={len(train_examples):,} val_examples={len(val_examples):,} train_tokens={len(train_ids):,}')

    builder_path = out_dir / 'compiled_builder.pt'
    if builder_path.exists():
        builder = GPT2CompiledChannelBuilder.load(builder_path)
        print(f'[compiled] loaded {builder_path} tokens={builder.total_tokens:,}')
    else:
        cfg = GPT2CompiledChannelConfig(max_train_tokens=args.compile_tokens)
        builder = GPT2CompiledChannelBuilder.from_ids(train_ids, cfg)
        builder.save(builder_path)
        print(f'[compiled] built {builder_path} tokens={builder.total_tokens:,}')

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16 if device.type == 'cuda' else torch.float32,
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    model.train(False)
    for param in model.parameters():
        param.requires_grad = False
    print(f'[model] loaded params={sum(param.numel() for param in model.parameters()):,}')

    backend = ZeroQPartitionedBackend(device=device, zeroq_path=args.zeroq_path)
    t_prepare = time.time()
    handle = backend.prepare(model, gpt2_resident_surface(model))
    model = handle.model
    for param in model.parameters():
        param.requires_grad = False
    print(f'[zeroq] prepared in {time.time() - t_prepare:.1f}s stats={handle.memory_stats()} trainable={handle.trainable_parameter_names}')

    inject_layers = default_inject_layers(int(getattr(model.config, 'n_layer', 36)))
    steerer = FeatureConditionedAdapterSteerer(
        d_model=model.config.n_embd,
        inject_layers=inject_layers,
        bottleneck=args.adapter_bottleneck,
        init_scale=0.005,
        noise_scale=0.01,
    ).to(device)
    if args.resume_cartridge:
        resume = torch.load(REPO / args.resume_cartridge, map_location=device, weights_only=False)
        steerer.load_state_dict(resume['steerer_state'])
        print(f'[resume] loaded {args.resume_cartridge}')
    steerer.train()
    print(f'[cartridge] trainable={sum(param.numel() for param in steerer.parameters()):,} layers={inject_layers}')

    cartridge_id = 'gpt2-large-chat-capability'
    rack = SteererCartridgeRack()
    manifest = CartridgeManifest(
        cartridge_id=cartridge_id,
        role=CartridgeRole.TASK_CAPABILITY,
        base_model_id=args.model_name,
        tokenizer_id=args.model_name,
        steerer_class='FeatureConditionedAdapterSteerer',
        inject_layers=tuple(inject_layers),
        parameter_count=sum(param.numel() for param in steerer.parameters()),
        source_corpus=args.data_dir,
        metadata={'zeroq': True, 'compiled_features': 'gpt2-ngram-skip-v1'},
    )
    rack.mount(manifest, steerer, weight=1.0, active=True)
    hooks = rack.register_hooks(model)
    print(f'[rack] hooks={hooks}')

    opt = torch.optim.AdamW(steerer.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_probe = -1.0
    best_val = float('inf')
    history: list[dict] = []
    for epoch in range(1, args.epochs + 1):
        steerer.train()
        model.eval()
        rack.activate(cartridge_id, True)
        total_loss = 0.0
        total_tokens = 0.0
        t0 = time.time()
        for step in range(1, args.steps + 1):
            x, y, mask = sample_example_batch(train_examples, args.batch, args.seq_len, device, pad_id)
            rack.set_weights(batch_features(x, builder, device))
            logits = model(input_ids=x, use_cache=False).logits.float()
            loss = masked_ce(logits, y, mask) + 0.00005 * steerer.orthogonal_penalty()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(steerer.parameters(), 1.0)
            opt.step()
            total_loss += float(loss.detach().item())
            total_tokens += float(mask.sum().item())
            if step == 1 or step % 25 == 0:
                print(f'  step={step:4d}/{args.steps} loss={float(loss.detach().item()):.4f} mask_tokens={int(mask.sum().item())}', flush=True)

        steerer.eval()
        val_ppl = evaluate_loss(model, rack, cartridge_id, val_examples, builder, args.batch, args.seq_len, device, pad_id, args.eval_batches)
        avg_loss = total_loss / max(args.steps, 1)
        row = {
            'epoch': epoch,
            'loss': avg_loss,
            'train_ppl': math.exp(min(avg_loss, 20.0)),
            'val_ppl': val_ppl,
            'seconds': time.time() - t0,
        }
        status = ''

        ckpt_payload = {
            'steerer_state': {key: value.detach().cpu() for key, value in steerer.state_dict().items()},
            'manifest': manifest.__dict__,
            'model_name': args.model_name,
            'data_dir': args.data_dir,
            'inject_layers': inject_layers,
            'adapter_bottleneck': args.adapter_bottleneck,
            'compiled_builder_state': builder.state_dict(),
            'epoch': epoch,
            'history': history + [row],
            'optimizer_state': opt.state_dict(),
            'zeroq': {'path': args.zeroq_path, 'stats': handle.memory_stats()},
        }
        save_checkpoint(out_dir / 'latest_chat_cartridge.pt', ckpt_payload)
        if val_ppl < best_val:
            best_val = val_ppl
            save_checkpoint(out_dir / 'best_loss_chat_cartridge.pt', ckpt_payload)
            status = 'SAVED_LOSS'

        if args.probe_every > 0 and epoch % args.probe_every == 0:
            probe_ckpt = out_dir / 'latest_chat_cartridge.pt'
            probe_report = out_dir / f'assistant_probe_epoch_{epoch:03d}.json'
            summary = run_assistant_probe(
                {
                    'model_name': args.model_name,
                    'cartridge': str(probe_ckpt.relative_to(REPO)),
                    'device': str(device),
                    'zeroq_path': args.zeroq_path,
                    'adapter_bottleneck': args.adapter_bottleneck,
                },
                probe_report,
                args.probe_task_limit,
            )
            row['probe'] = summary
            accuracy = float(summary['accuracy'])
            if accuracy >= best_probe:
                best_probe = accuracy
                save_checkpoint(out_dir / 'best_probe_chat_cartridge.pt', ckpt_payload)
                status = (status + ' SAVED_PROBE').strip()

        history.append(row)
        (out_dir / 'history.json').write_text(json.dumps(history, indent=2), encoding='utf-8')
        print(
            f"epoch={epoch:3d} loss={avg_loss:.4f} train_ppl={row['train_ppl']:.1f} "
            f"val_ppl={val_ppl:.1f} best_val={best_val:.1f} {status} time={row['seconds']:.0f}s",
            flush=True,
        )

    rack.remove_hooks()
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()
    print(f'Done. best_val={best_val:.3f} best_probe={best_probe:.3f}')


if __name__ == '__main__':
    main()
