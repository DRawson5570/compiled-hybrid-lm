"""train_4b_distributed.py — CMI backend training: DeepSeek LM + compiled cartridge + ZeroQ.

Usage:
  torchrun --nproc_per_node=2 --master_addr=localhost --master_port=29500 \
           train_4b_distributed.py --backend zeroq --epochs 100 --batch 2
"""
import argparse, os, sys, socket, time, math, pickle
import torch, torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader

_here = os.path.dirname(os.path.abspath(__file__))
_repo = os.path.dirname(_here)
sys.path.insert(0, _repo)
sys.path.insert(0, _here)

import hf_deepseek
from hf_deepseek import DeepSeekConfig, DeepSeekForCausalLM
from hybrid.backends import (
    DenseTorchBackend,
    TrainableSurface,
    ZeroQPartitionedBackend,
    allreduce_trainable_grads,
    trainable_parameters,
)
from hybrid.superposition_steerer_v3 import SuperpositionSteererV3
from gpu_channels import GPUFeatureComputer
from train_steerer_v4 import StreamingSteererDatasetV4, FastNgramFeatures, compute_cpu_features, PUNCT_IDS

V = 50257
MODEL_CONFIGS = {
    "test": dict(d_model=192, n_layers=2, n_heads=6, d_ff=768, max_len=256),
    "3b": dict(d_model=2688, n_layers=32, n_heads=21, d_ff=10752, max_len=512),
}


def _init_dist():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank, torch.device(f"cuda:{local_rank}")


def _rank0_print(rank, msg):
    if rank == 0: print(msg, flush=True)


def _manual_allreduce_grads(model, world_size):
    allreduce_trainable_grads(model, world_size)


def load_priors(device):
    priors_dir = os.path.expanduser("~/deepseek_experiments/artifacts/compiled_priors_v3")
    word_topics = torch.load(os.path.join(priors_dir, "word_topics.pt"), map_location='cpu')
    with open(os.path.join(priors_dir, "pos_stats.pkl"), 'rb') as f:
        pos_stats = pickle.load(f)
    tag_to_idx = pos_stats.get('tag_to_idx', {'WORD': 0, 'PUNCT': 1, 'NUM': 2})
    token_to_tag = pos_stats.get('token_to_tag', {})
    pos_tags = {int(k): tag_to_idx.get(v, 0) for k, v in token_to_tag.items()}
    ppmi_emb = torch.randn(V, 256, dtype=torch.float32) * 0.01
    return word_topics.to(device), pos_tags, ppmi_emb.to(device)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backend", choices=["dense", "zeroq"], default="zeroq")
    p.add_argument("--model-config", choices=sorted(MODEL_CONFIGS), default="3b")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--steerer-lr", type=float, default=1e-3)
    p.add_argument("--train-surface", choices=["head_bias", "cmi_steerer"], default="cmi_steerer")
    p.add_argument("--eval-tokens", type=int, default=2000)
    p.add_argument("--zeroq-path", default="~/ZeroQ")
    args = p.parse_args()

    rank, world_size, local_rank, device = _init_dist()
    hostname = socket.gethostname()
    model_cfg = MODEL_CONFIGS[args.model_config]
    _rank0_print(rank, f"=== CMI BACKEND TRAIN === Rank {rank}/{world_size} {hostname}")
    _rank0_print(rank, f"backend={args.backend} config={args.model_config} surface={args.train_surface} d={model_cfg['d_model']} L={model_cfg['n_layers']} epochs={args.epochs} batch={args.batch}")

    # Build model on CPU; the selected backend decides whether to move densely
    # or stream frozen weights through ZeroQ quantized partitioning.
    cfg = DeepSeekConfig(**model_cfg)
    model = DeepSeekForCausalLM(cfg)
    n_params = sum(p.numel() for p in model.parameters())
    _rank0_print(rank, f"Params: {n_params:,}")

    _rank0_print(rank, f"[{args.backend}] Preparing frozen backbone...")
    t0 = time.time()
    if args.backend == "zeroq":
        backend = ZeroQPartitionedBackend(device=device, zeroq_path=args.zeroq_path)
    else:
        backend = DenseTorchBackend(device=device)
    handle = backend.prepare(model, TrainableSurface.head_bias_and_embeddings())
    # Embeddings must stay materialized for weight-tied output, but should be frozen
    # (only head_bias, ln_f, and steerer are actually trained)
    model.tok_emb.weight.requires_grad = False
    model.pos_emb.weight.requires_grad = False
    _rank0_print(rank, f"[{args.backend}] Prepared in {time.time()-t0:.1f}s stats={handle.memory_stats()}")

    steerer = None
    n_hooks = 0
    if args.train_surface == "cmi_steerer":
        _rank0_print(rank, "[build] CMI compiled-prior steerer...")
        steerer = SuperpositionSteererV3(d_model=model_cfg['d_model'], init_scale=0.01, noise_scale=0.05).to(device)
        n_hooks = steerer.register_hooks(model)
    model_trainable = sum(param.numel() for param in trainable_parameters(model))
    steerer_trainable = sum(param.numel() for param in steerer.parameters()) if steerer is not None else 0
    _rank0_print(rank, f"  hooks={n_hooks} model_trainable={model_trainable:,} steerer_trainable={steerer_trainable:,}")

    # Load compiled priors
    gpu_fc = None
    if steerer is not None:
        _rank0_print(rank, "[load] Compiled priors...")
        word_topics, pos_tags, ppmi_emb = load_priors(device)
        gpu_fc = GPUFeatureComputer(V=V, punct_ids=PUNCT_IDS, topic_matrix=word_topics,
                                    pos_tags=pos_tags, ppmi_embeddings=ppmi_emb, device=device)
    cpu_ch = FastNgramFeatures(V)
    _rank0_print(rank, "  21 channels ready" if steerer is not None else "  compiled steerer disabled for memory-safe smoke")

    # Load training data
    _rank0_print(rank, "[load] Data...")
    train_ids = torch.load(
        os.path.expanduser("~/deepseek_experiments/artifacts/wikitext_gpt2/train_ids.pt"))
    val_ids = torch.load(
        os.path.expanduser("~/deepseek_experiments/artifacts/wikitext_gpt2/validation_ids.pt"))
    _rank0_print(rank, f"  Train: {len(train_ids):,}  Val: {len(val_ids):,}")

    train_dataset = StreamingSteererDatasetV4(train_ids=train_ids, seq_len=args.seq_len, V=V)
    train_loader = DataLoader(train_dataset, batch_size=args.batch,
                              num_workers=2, pin_memory=True, drop_last=True)

    opt = torch.optim.AdamW([
        {'params': trainable_parameters(model), 'lr': args.lr},
    ] + ([{'params': steerer.parameters(), 'lr': args.steerer_lr}] if steerer is not None else []))
    best_eval_b = float('inf')

    for ep in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0; t0 = time.time()
        loader_iter = iter(train_loader)

        for step in range(args.steps):
            x, y, w_cpu = next(loader_iter)
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            w_cpu = w_cpu.to(device, non_blocking=True)

            if steerer is not None:
                w_gpu = gpu_fc.compute_features(x)
                w_gpu[:, :, 0:9] = w_cpu[:, :, :9]
                steerer.set_weights(w_gpu)
            out = model(x)
            loss = F.cross_entropy(out.logits.reshape(-1, V), y.reshape(-1))
            if steerer is not None:
                loss = loss + 0.001 * steerer.orthogonal_penalty()

            opt.zero_grad(set_to_none=True)
            loss.backward()
            _manual_allreduce_grads(model, world_size)
            if steerer is not None:
                _manual_allreduce_grads(steerer, world_size)
            opt.step()
            total_loss += loss.item()

        # Eval
        model.eval()
        if steerer is not None:
            steerer.eval()
        with torch.no_grad():
            es_nll, es_n = 0.0, 0
            cpu_ch = FastNgramFeatures(V)
            eval_limit = min(len(val_ids) - 1, max(args.eval_tokens, 1))
            for s in range(0, eval_limit, 128):
                cl = min(128, len(val_ids) - s - 1)
                if cl <= 0: continue
                inp = val_ids[s:s+cl].unsqueeze(0).to(device)
                tgt = val_ids[s+1:s+cl+1].unsqueeze(0).to(device)
                if steerer is not None:
                    w_e = gpu_fc.compute_features(inp)
                    w_cpu_eval = compute_cpu_features(val_ids[s:s+cl].tolist(), cpu_ch)
                    w_e[0, :w_cpu_eval.shape[0], 0:9] = w_cpu_eval[:, :9].to(device)
                    steerer.set_weights(w_e)
                logits = model(inp).logits
                es_nll += F.cross_entropy(logits.reshape(-1, V), tgt.reshape(-1), reduction='sum').item()
                es_n += cl
            eval_s = math.exp(es_nll / max(es_n, 1))

        avg_loss = total_loss / args.steps; elapsed = time.time() - t0
        status = ""
        if eval_s < best_eval_b:
            best_eval_b = eval_s; status = "SAVED"
            out_dir = os.path.expanduser("~/deepseek_experiments/artifacts/train_3b")
            os.makedirs(out_dir, exist_ok=True)
            torch.save({'state_dict': model.state_dict(),
                        'steerer_state': steerer.state_dict() if steerer is not None else None,
                        'backend': args.backend,
                        'model_config': args.model_config,
                        'train_surface': args.train_surface,
                        'epoch': ep,
                        'eval_s': eval_s},
                       os.path.join(out_dir, 'best.pt'))

        _rank0_print(rank, f"  epoch={ep:3d}  loss={avg_loss:.4f}  ppl={math.exp(avg_loss):.1f}  "
                     f"eval_s={eval_s:.1f}  best={best_eval_b:.1f}  [{status}]  time={elapsed:.0f}s")

    _rank0_print(rank, f"Done. Best eval_s: {best_eval_b:.1f}")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
