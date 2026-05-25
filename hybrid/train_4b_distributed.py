"""train_4b_distributed.py — CMI backend training: DeepSeek LM + compiled cartridge + ZeroQ.

Usage:
  torchrun --nproc_per_node=2 --master_addr=localhost --master_port=29500 \
           train_4b_distributed.py --backend zeroq --epochs 100 --batch 2
"""
import argparse, os, sys, socket, time, math, pickle
from collections import defaultdict
import torch, torch.distributed as dist
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader

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

V = 50257
PUNCT_IDS = {0, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 25, 26, 27, 28, 29, 30, 31, 58, 60, 61, 90, 91, 92, 93, 198, 220}
MODEL_CONFIGS = {
    "test": dict(d_model=192, n_layers=2, n_heads=6, d_ff=768, max_len=256),
    "700m": dict(d_model=1536, n_layers=22, n_heads=16, d_ff=6144, max_len=512),
    "3b": dict(d_model=2688, n_layers=32, n_heads=21, d_ff=10752, max_len=512),
    "4b": dict(d_model=3072, n_layers=40, n_heads=24, d_ff=12288, max_len=512),
}


class StreamingSteererDatasetV4(Dataset):
    def __init__(self, train_ids, seq_len, vocab_size=None, V=None):
        self.train_ids = train_ids
        self.seq_len = seq_len
        self.N = len(train_ids)
        self.vocab_size = int(vocab_size if vocab_size is not None else V)

    def __len__(self):
        return 1000000

    def __getitem__(self, idx):
        start = torch.randint(0, max(1, self.N - self.seq_len - 1), (1,)).item()
        x = self.train_ids[start:start + self.seq_len]
        y = self.train_ids[start + 1:start + self.seq_len + 1]
        channels = FastNgramFeatures(self.vocab_size)
        return x, y, compute_cpu_features(x.tolist(), channels)


class FastNgramFeatures:
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size
        self._u = -math.log(vocab_size)
        self.reset()

    def reset(self):
        self._uni = np.zeros(self.vocab_size, dtype=np.float32)
        self._uni_total = 0.0
        self._bi = {}
        self._bit = {}
        self._tri = {}
        self._trit = {}
        self._skip2 = {}
        self._skip2t = {}
        self._skip3 = {}
        self._skip3t = {}
        self._seen = defaultdict(list)
        self._ctx = []
        self._step = 0

    def update(self, token_id):
        token_id = int(token_id)
        self._step += 1
        self._ctx.append(token_id)
        self._ctx = self._ctx[-128:]
        if self._step % 10 == 0:
            self._uni *= 0.999
            self._uni_total *= 0.999
        if token_id < self.vocab_size:
            self._uni[token_id] += 1.0
            self._uni_total += 1.0
        if len(self._ctx) >= 2:
            prev, cur = self._ctx[-2], self._ctx[-1]
            self._bi[(prev, cur)] = self._bi.get((prev, cur), 0) + 1
            self._bit[prev] = self._bit.get(prev, 0) + 1
        if len(self._ctx) >= 3:
            prev2, prev1, cur = self._ctx[-3], self._ctx[-2], self._ctx[-1]
            self._tri[(prev2, prev1, cur)] = self._tri.get((prev2, prev1, cur), 0) + 1
            self._trit[(prev2, prev1)] = self._trit.get((prev2, prev1), 0) + 1
        if len(self._ctx) >= 2:
            self._skip2[(self._ctx[-2], token_id)] = self._skip2.get((self._ctx[-2], token_id), 0) + 1
            self._skip2t[self._ctx[-2]] = self._skip2t.get(self._ctx[-2], 0) + 1
        if len(self._ctx) >= 3:
            self._skip3[(self._ctx[-3], token_id)] = self._skip3.get((self._ctx[-3], token_id), 0) + 1
            self._skip3t[self._ctx[-3]] = self._skip3t.get(self._ctx[-3], 0) + 1
        self._seen[token_id].append(self._step)

    def get_features(self, token_id):
        token_id = int(token_id)
        ctx = self._ctx
        uniform = self._u
        uni_denom = self._uni_total + 0.001 * self.vocab_size
        uni_log = math.log(max((self._uni[token_id] + 0.001) / uni_denom, 1e-7)) if uni_denom > 0 and token_id < self.vocab_size else uniform
        bi_log = uniform
        if len(ctx) >= 1:
            total = self._bit.get(ctx[-1], 0)
            denom = total + 0.001 * self.vocab_size
            bi_log = math.log(max((self._bi.get((ctx[-1], token_id), 0) + 0.001) / denom, 1e-7)) if denom > 0 else uniform
        tri_log = uniform
        if len(ctx) >= 2:
            key = (ctx[-2], ctx[-1])
            total = self._trit.get(key, 0)
            denom = total + 0.001 * self.vocab_size
            tri_log = math.log(max((self._tri.get((ctx[-2], ctx[-1], token_id), 0) + 0.001) / denom, 1e-7)) if denom > 0 else uniform
        skip2_log = uniform
        if len(ctx) >= 2:
            total = self._skip2t.get(ctx[-2], 0)
            denom = total + 0.001 * self.vocab_size
            skip2_log = math.log(max((self._skip2.get((ctx[-2], token_id), 0) + 0.001) / denom, 1e-7)) if denom > 0 else uniform
        skip3_log = uniform
        if len(ctx) >= 3:
            total = self._skip3t.get(ctx[-3], 0)
            denom = total + 0.001 * self.vocab_size
            skip3_log = math.log(max((self._skip3.get((ctx[-3], token_id), 0) + 0.001) / denom, 1e-7)) if denom > 0 else uniform
        positions = self._seen.get(token_id, [])
        gap = 128 if not positions else min(128, self._step - positions[-1])
        recency_log = math.log(max(1.0 / max(gap, 1), 1e-7))
        entropy = float(-uni_log / math.log(self.vocab_size)) if uni_log < 0 else 1.0
        return [float(uni_log), float(bi_log), float(bi_log), float(tri_log), float(tri_log), float(skip2_log), float(skip3_log), float(recency_log), float(entropy)]


def compute_cpu_features(tokens, channels):
    channels.reset()
    features = []
    for idx, token_id in enumerate(tokens):
        if idx > 0:
            channels.update(tokens[idx - 1])
        features.append(channels.get_features(int(token_id)))
    if not features:
        return torch.zeros(1, 9)
    return torch.tensor(np.array(features, dtype=np.float32))


def _init_dist():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank, torch.device(f"cuda:{local_rank}")


def _rank0_print(rank, msg):
    if rank == 0: print(msg, flush=True)


def _manual_allreduce_grads(model, world_size, process_group=None):
    allreduce_trainable_grads(model, world_size, process_group=process_group)


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
    p.add_argument("--compute-in-4bit", action="store_true",
                   help="Convert frozen ZeroQ Linear layers to native bitsandbytes Linear4bit after partitioning")
    p.add_argument("--resume-checkpoint", default=None,
                   help="Resume model and steerer weights from a previous train_4b_distributed best.pt")
    args = p.parse_args()

    rank, world_size, local_rank, device = _init_dist()
    hostname = socket.gethostname()
    model_cfg = MODEL_CONFIGS[args.model_config]
    _rank0_print(rank, f"=== CMI BACKEND TRAIN === Rank {rank}/{world_size} {hostname}")
    _rank0_print(rank, f"backend={args.backend} config={args.model_config} surface={args.train_surface} d={model_cfg['d_model']} L={model_cfg['n_layers']} epochs={args.epochs} batch={args.batch} compute_in_4bit={args.compute_in_4bit}")

    # Build model on CPU; the selected backend decides whether to move densely
    # or stream frozen weights through ZeroQ quantized partitioning.
    cfg = DeepSeekConfig(**model_cfg)
    model = DeepSeekForCausalLM(cfg)
    n_params = sum(p.numel() for p in model.parameters())
    _rank0_print(rank, f"Params: {n_params:,}")

    _rank0_print(rank, f"[{args.backend}] Preparing frozen backbone...")
    t0 = time.time()
    if args.backend == "zeroq":
        backend = ZeroQPartitionedBackend(
            device=device,
            zeroq_path=args.zeroq_path,
            compute_in_4bit=args.compute_in_4bit,
        )
    else:
        backend = DenseTorchBackend(device=device)
    handle = backend.prepare(model, TrainableSurface.head_bias_and_embeddings())
    # Embeddings must stay materialized for weight-tied output, but should be frozen
    # (only head_bias, ln_f, and steerer are actually trained)
    model.tok_emb.weight.requires_grad = False
    model.pos_emb.weight.requires_grad = False
    _rank0_print(rank, f"[{args.backend}] Prepared in {time.time()-t0:.1f}s stats={handle.memory_stats()}")

    resume_ckpt = None
    resume_start_epoch = 0
    resume_best_eval_s = float('inf')
    resume_best_eval_b = float('inf')
    if args.resume_checkpoint:
        resume_path = os.path.expanduser(args.resume_checkpoint)
        _rank0_print(rank, f"[resume] Loading checkpoint: {resume_path}")
        resume_ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        resume_start_epoch = int(resume_ckpt.get('epoch', 0) or 0)
        resume_best_eval_s = float(resume_ckpt.get('eval_s', float('inf')))
        resume_best_eval_b = float(resume_ckpt.get('eval_b', float('inf')))
        state_dict = resume_ckpt.get('state_dict')
        if state_dict:
            try:
                load_result = model.load_state_dict(state_dict, strict=False)
                _rank0_print(rank, f"[resume] model loaded missing={len(load_result.missing_keys)} unexpected={len(load_result.unexpected_keys)}")
            except RuntimeError as exc:
                current_state = model.state_dict()
                compatible_state = {
                    name: value for name, value in state_dict.items()
                    if name in current_state and tuple(current_state[name].shape) == tuple(value.shape)
                }
                load_result = model.load_state_dict(compatible_state, strict=False)
                _rank0_print(rank, f"[resume] partial model load after shape mismatch: loaded={len(compatible_state)} missing={len(load_result.missing_keys)} unexpected={len(load_result.unexpected_keys)} error={exc}")

    steerer = None
    n_hooks = 0
    if args.train_surface == "cmi_steerer":
        _rank0_print(rank, "[build] CMI compiled-prior steerer...")
        steerer = SuperpositionSteererV3(d_model=model_cfg['d_model'], init_scale=0.01, noise_scale=0.05).to(device)
        if resume_ckpt is not None and resume_ckpt.get('steerer_state') is not None:
            steerer.load_state_dict(resume_ckpt['steerer_state'])
            _rank0_print(rank, "[resume] steerer loaded")
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
    best_eval_b = resume_best_eval_b
    best_eval_s = resume_best_eval_s

    for epoch_offset in range(1, args.epochs + 1):
        ep = resume_start_epoch + epoch_offset
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
            _manual_allreduce_grads(model, world_size, handle.grad_process_group)
            if steerer is not None:
                _manual_allreduce_grads(steerer, world_size, handle.grad_process_group)
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

            # Baseline eval (no steerer)
            eb_nll, eb_n = 0.0, 0
            if steerer is not None:
                steerer._current_weights = None
            eb_limit = min(len(val_ids) - 1, max(args.eval_tokens, 1))
            for s in range(0, eb_limit, 128):
                cl = min(128, len(val_ids) - s - 1)
                if cl <= 0: continue
                inp = val_ids[s:s+cl].unsqueeze(0).to(device)
                tgt = val_ids[s+1:s+cl+1].unsqueeze(0).to(device)
                logits = model(inp).logits
                eb_nll += F.cross_entropy(logits.reshape(-1, V), tgt.reshape(-1), reduction='sum').item()
                eb_n += cl
            eval_b = math.exp(eb_nll / max(eb_n, 1))

        avg_loss = total_loss / args.steps; elapsed = time.time() - t0
        status = ""
        if eval_b < best_eval_b: best_eval_b = eval_b; status += "b"
        if eval_s < best_eval_s: best_eval_s = eval_s; status += "s"
        if status:
            backend_name = f"{args.backend}_4bit" if args.compute_in_4bit else args.backend
            out_dir = os.path.expanduser(
                f"~/deepseek_experiments/artifacts/train_{args.model_config}_{args.train_surface}_{backend_name}"
            )
            os.makedirs(out_dir, exist_ok=True)
            torch.save({'state_dict': model.state_dict(),
                        'steerer_state': steerer.state_dict() if steerer is not None else None,
                        'backend': args.backend,
                        'compute_in_4bit': args.compute_in_4bit,
                        'model_config': args.model_config,
                        'train_surface': args.train_surface,
                        'epoch': ep,
                        'eval_s': eval_s,
                        'eval_b': eval_b,
                        'resume_checkpoint': args.resume_checkpoint},
                       os.path.join(out_dir, 'best.pt'))

        _rank0_print(rank, f"  epoch={ep:3d}  loss={avg_loss:.4f}  ppl={math.exp(avg_loss):.1f}  "
                     f"eval_s={eval_s:.1f}  eval_b={eval_b:.1f}  best_b={best_eval_b:.1f}  [{status}]  time={elapsed:.0f}s")

    _rank0_print(rank, f"Done. Best eval_b: {best_eval_b:.1f}  Best eval_s: {best_eval_s:.1f}")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
