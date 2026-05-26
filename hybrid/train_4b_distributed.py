"""train_4b_distributed.py — CMI backend training: DeepSeek LM + compiled cartridge + ZeroQ.

Usage:
  torchrun --nproc_per_node=2 --master_addr=localhost --master_port=29500 \
           train_4b_distributed.py --backend zeroq --epochs 100 --batch 2
"""
import argparse, os, sys, socket, time, math, pickle, random
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


class C4MixedSteererDataset(torch.utils.data.IterableDataset):
    def __init__(self, wt_train_ids, seq_len, vocab_size, c4_ratio=0.85, seed=42):
        self.wt_train_ids = wt_train_ids
        self.seq_len = seq_len
        self.vocab_size = int(vocab_size)
        self.c4_ratio = float(c4_ratio)
        self.seed = int(seed)

    def _local_c4_files(self):
        import glob, json

        roots = []
        datasets_cache = os.environ.get("HF_DATASETS_CACHE")
        if datasets_cache:
            roots.append(datasets_cache)
        hf_home = os.environ.get("HF_HOME")
        if hf_home:
            roots.append(os.path.join(hf_home, "datasets"))
        roots.append(os.path.expanduser("~/deepseek_experiments/artifacts/hf_cache/datasets"))

        files = []
        seen = set()
        for root in roots:
            downloads_dir = os.path.join(os.path.expanduser(root), "downloads")
            for meta_path in glob.glob(os.path.join(downloads_dir, "*.json")):
                data_path = meta_path[:-5]
                if data_path in seen or not os.path.exists(data_path):
                    continue
                try:
                    with open(meta_path, "r", encoding="utf-8") as handle:
                        meta = json.load(handle)
                except Exception:
                    continue
                url = str(meta.get("url", ""))
                if "/en/c4-train." not in url or not url.endswith(".json.gz"):
                    continue
                seen.add(data_path)
                files.append(data_path)
        return sorted(files)

    def _iter_local_c4_texts(self, files, rng, worker_id, num_workers):
        import gzip, json

        while True:
            shuffled = list(files)
            rng.shuffle(shuffled)
            worker_files = shuffled[worker_id::max(1, num_workers)] or shuffled
            for path in worker_files:
                try:
                    with gzip.open(path, "rt", encoding="utf-8") as handle:
                        for line in handle:
                            try:
                                text = (json.loads(line).get("text") or "").strip()
                            except json.JSONDecodeError:
                                continue
                            if text:
                                yield text
                except OSError:
                    continue

    def __iter__(self):
        from datasets import load_dataset
        from torch.utils.data import get_worker_info
        from transformers import AutoTokenizer

        worker = get_worker_info()
        worker_id = worker.id if worker is not None else 0
        num_workers = worker.num_workers if worker is not None else 1
        rng = random.Random(self.seed + worker_id * 1009)
        torch_gen = torch.Generator().manual_seed(self.seed + worker_id * 9176)
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        local_c4_files = self._local_c4_files()
        if local_c4_files:
            c4_iter = self._iter_local_c4_texts(local_c4_files, rng, worker_id, num_workers)
            c4_ds = None
        else:
            c4_ds = load_dataset("allenai/c4", "en", split="train", streaming=True)
            c4_iter = iter(c4_ds.shuffle(seed=self.seed + worker_id, buffer_size=10000))
        token_buffer = []
        channels = FastNgramFeatures(self.vocab_size)

        while True:
            while len(token_buffer) < (self.seq_len + 1) * 4:
                use_c4 = rng.random() < self.c4_ratio
                if use_c4:
                    try:
                        example = next(c4_iter)
                    except StopIteration:
                        if c4_ds is None:
                            c4_iter = self._iter_local_c4_texts(local_c4_files, rng, worker_id, num_workers)
                        else:
                            c4_iter = iter(c4_ds.shuffle(seed=rng.randrange(2**32), buffer_size=10000))
                        continue
                    text = example if isinstance(example, str) else (example.get("text") or "").strip()
                    if not text:
                        continue
                    ids = tokenizer.encode(text[:2000], add_special_tokens=False, truncation=True, max_length=1024)
                    if ids:
                        token_buffer.extend(ids)
                else:
                    max_start = max(1, len(self.wt_train_ids) - self.seq_len * 2 - 1)
                    start = torch.randint(0, max_start, (1,), generator=torch_gen).item()
                    token_buffer.extend(self.wt_train_ids[start:start + self.seq_len * 2].tolist())

            max_start = max(1, len(token_buffer) - self.seq_len - 1)
            start = torch.randint(0, max_start, (1,), generator=torch_gen).item()
            span = token_buffer[start:start + self.seq_len + 1]
            x = torch.tensor(span[:-1], dtype=torch.long)
            y = torch.tensor(span[1:], dtype=torch.long)
            consumed = start + self.seq_len + 1
            token_buffer = token_buffer[max(0, consumed - self.seq_len * 2):]
            yield x, y, compute_cpu_features(x.tolist(), channels)


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


def _checkpoint_targets_for_status(out_dir, status):
    targets = []
    if "b" in status:
        targets.append((os.path.join(out_dir, "best_b.pt"), "blind_best"))
    if "s" in status:
        targets.append((os.path.join(out_dir, "best_s.pt"), "steered_best"))
        targets.append((os.path.join(out_dir, "best.pt"), "legacy_steered_best"))
    return targets


def _save_metric_checkpoints(out_dir, status, payload):
    os.makedirs(out_dir, exist_ok=True)
    written = []
    for path, checkpoint_kind in _checkpoint_targets_for_status(out_dir, status):
        checkpoint_payload = dict(payload)
        checkpoint_payload["checkpoint_kind"] = checkpoint_kind
        torch.save(checkpoint_payload, path)
        written.append(path)
    return written


def _early_stop_metric_improved(status, metric):
    if metric == "none":
        return False
    if metric == "steered":
        return "s" in status
    if metric == "blind":
        return "b" in status
    if metric == "either":
        return bool(status)
    raise ValueError(f"unknown early-stop metric: {metric}")


def _manual_allreduce_grads(model, world_size, process_group=None):
    allreduce_trainable_grads(model, world_size, process_group=process_group)


def _surface_names_for_train_surface(model, train_surface):
    if train_surface in {"head_bias", "cmi_steerer"}:
        return ["head_bias"]
    if train_surface in {"full", "full_cmi_steerer"}:
        return [name for name, _ in model.named_parameters()]
    if train_surface in {"top1", "top1_cmi_steerer", "top2", "top2_cmi_steerer", "top4", "top4_cmi_steerer"}:
        count = int(train_surface.removeprefix("top").split("_")[0])
        n_layers = len(getattr(model, "layers"))
        first_trainable = max(0, n_layers - count)
        names = ["head_bias", "ln_f.weight", "ln_f.bias"]
        for name, _ in model.named_parameters():
            if not name.startswith("layers."):
                continue
            layer_idx = int(name.split(".", 2)[1])
            if layer_idx >= first_trainable:
                names.append(name)
        return names
    raise ValueError(f"unknown trainable surface: {train_surface}")


def _trainable_surface_for_model(model, train_surface, backend="dense"):
    names = list(_surface_names_for_train_surface(model, train_surface))
    if backend == "zeroq":
        # The token embedding is weight-tied as the output projection, so it is
        # used outside the Embedding module's ZeroQ hooks. Keep it materialized.
        names.extend(["tok_emb.weight", "pos_emb.weight"])
    return TrainableSurface.from_names(names)


def _apply_frozen_materialized_params(model, train_surface):
    desired_trainable = set(_surface_names_for_train_surface(model, train_surface))
    for name, param in model.named_parameters():
        if name not in desired_trainable and name in {"tok_emb.weight", "pos_emb.weight"}:
            param.requires_grad = False


def _uses_cmi_steerer(train_surface):
    return train_surface in {"cmi_steerer", "full_cmi_steerer", "top1_cmi_steerer", "top2_cmi_steerer", "top4_cmi_steerer"}


def _set_requires_grad(params, enabled):
    for param in params:
        param.requires_grad = enabled


def _prior_on_beats_off(eval_prior_on, eval_prior_off, min_delta=0.0):
    return math.isfinite(eval_prior_on) and math.isfinite(eval_prior_off) and eval_prior_on + min_delta < eval_prior_off


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
    p.add_argument("--train-surface", choices=["head_bias", "cmi_steerer", "full", "full_cmi_steerer", "top1", "top1_cmi_steerer", "top2", "top2_cmi_steerer", "top4", "top4_cmi_steerer"], default="cmi_steerer",
                   help="head_bias/full/topN train neural surfaces; *_cmi_steerer also trains the compiled-prior steerer. topN_cmi_steerer is the ZeroQ thesis track.")
    p.add_argument("--eval-tokens", type=int, default=2000)
    p.add_argument("--zeroq-path", default="~/ZeroQ")
    p.add_argument("--compute-in-4bit", action="store_true",
                   help="Convert frozen ZeroQ Linear layers to native bitsandbytes Linear4bit after partitioning")
    p.add_argument("--resume-checkpoint", default=None,
                   help="Resume model and steerer weights from a previous train_4b_distributed best.pt")
    p.add_argument("--data-mode", choices=["wikitext", "c4-mix"], default="wikitext")
    p.add_argument("--data-dir", default="~/deepseek_experiments/artifacts/wikitext_gpt2",
                   help="Directory containing train_ids.pt and validation_ids.pt for wikitext mode and eval")
    p.add_argument("--c4-ratio", type=float, default=0.85,
                   help="Fraction of C4 examples in c4-mix training mode; remaining examples come from data-dir train_ids.pt")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", default=None,
                   help="Checkpoint directory. Defaults to artifacts/train_<config>_<surface>_<backend>[_c4_mix]")
    p.add_argument("--early-stop-metric", choices=["none", "steered", "blind", "either"], default="none",
                   help="Metric used for patience-based early stopping. 'steered' tracks eval_s/eval_prior_on, the product path.")
    p.add_argument("--early-stop-patience", type=int, default=0,
                   help="Stop after this many epochs without improvement on --early-stop-metric. 0 disables early stopping.")
    p.add_argument("--disable-prior-after-on-plateau", type=int, default=0,
                   help="If >0, stop using the compiled prior/steerer during training after eval_prior_on has not improved for this many epochs. Eval still reports prior-on/off diagnostics.")
    p.add_argument("--freeze-model-until-prior-on-beats-off", type=float, default=None,
                   help="If set, train only the steerer until eval_prior_on beats eval_prior_off by this PPL margin, then unfreeze the neural surface.")
    p.add_argument("--prior-on-warmup-patience", type=int, default=1,
                   help="Consecutive evals where eval_prior_on beats eval_prior_off before unfreezing the neural surface.")
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

    _rank0_print(rank, f"[{args.backend}] Preparing trainable surface...")
    t0 = time.time()
    if args.backend == "zeroq":
        backend = ZeroQPartitionedBackend(
            device=device,
            zeroq_path=args.zeroq_path,
            compute_in_4bit=args.compute_in_4bit,
        )
    else:
        backend = DenseTorchBackend(device=device)
    handle = backend.prepare(model, _trainable_surface_for_model(model, args.train_surface, args.backend))
    _apply_frozen_materialized_params(model, args.train_surface)
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
        resume_best_eval_s = float(resume_ckpt.get('best_eval_s', resume_ckpt.get('eval_s', float('inf'))))
        resume_best_eval_b = float(resume_ckpt.get('best_eval_b', resume_ckpt.get('eval_b', float('inf'))))
        resume_dir = os.path.dirname(resume_path)
        for filename, key, label in (("best_s.pt", "eval_s", "steered"), ("best_b.pt", "eval_b", "blind")):
            metric_path = os.path.join(resume_dir, filename)
            if not os.path.exists(metric_path):
                continue
            try:
                metric_ckpt = torch.load(metric_path, map_location='cpu', weights_only=False)
                metric_value = float(metric_ckpt.get(key, float('inf')))
            except Exception as exc:
                _rank0_print(rank, f"[resume] could not read {label} metric checkpoint {metric_path}: {exc}")
                continue
            if label == "steered":
                resume_best_eval_s = min(resume_best_eval_s, metric_value)
            else:
                resume_best_eval_b = min(resume_best_eval_b, metric_value)
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
    if _uses_cmi_steerer(args.train_surface):
        _rank0_print(rank, "[build] CMI compiled-prior steerer...")
        steerer = SuperpositionSteererV3(d_model=model_cfg['d_model'], init_scale=0.01, noise_scale=0.05).to(device)
        if resume_ckpt is not None and resume_ckpt.get('steerer_state') is not None:
            steerer.load_state_dict(resume_ckpt['steerer_state'])
            _rank0_print(rank, "[resume] steerer loaded")
        n_hooks = steerer.register_hooks(model)
    model_trainable_params = trainable_parameters(model)
    model_trainable = sum(param.numel() for param in model_trainable_params)
    steerer_trainable = sum(param.numel() for param in steerer.parameters()) if steerer is not None else 0
    _rank0_print(rank, f"  hooks={n_hooks} model_trainable={model_trainable:,} steerer_trainable={steerer_trainable:,}")
    _rank0_print(rank, "  metrics: eval_prior_on=validation with compiled prior/steerer active; eval_prior_off=validation with it disabled")

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
    data_dir = os.path.expanduser(args.data_dir)
    train_ids = torch.load(os.path.join(data_dir, "train_ids.pt"), weights_only=False).long()
    val_ids = torch.load(os.path.join(data_dir, "validation_ids.pt"), weights_only=False).long()
    _rank0_print(rank, f"  Data mode: {args.data_mode}  data_dir={data_dir}")
    _rank0_print(rank, f"  Train/ref: {len(train_ids):,}  Val: {len(val_ids):,}")

    if args.data_mode == "c4-mix":
        train_dataset = C4MixedSteererDataset(
            wt_train_ids=train_ids,
            seq_len=args.seq_len,
            vocab_size=V,
            c4_ratio=args.c4_ratio,
            seed=args.seed + rank * 100000,
        )
    else:
        train_dataset = StreamingSteererDatasetV4(train_ids=train_ids, seq_len=args.seq_len, V=V)
    train_loader = DataLoader(train_dataset, batch_size=args.batch,
                              num_workers=2, pin_memory=True, drop_last=True)

    opt = torch.optim.AdamW([
        {'params': model_trainable_params, 'lr': args.lr},
    ] + ([{'params': steerer.parameters(), 'lr': args.steerer_lr}] if steerer is not None else []))
    best_eval_b = resume_best_eval_b
    best_eval_s = resume_best_eval_s
    last_early_stop_improvement_epoch = resume_start_epoch
    last_prior_on_improvement_epoch = resume_start_epoch
    prior_training_enabled = steerer is not None
    warmup_gate_enabled = steerer is not None and args.freeze_model_until_prior_on_beats_off is not None
    neural_training_enabled = True
    prior_warmup_win_count = 0
    if warmup_gate_enabled:
        neural_training_enabled = bool(resume_ckpt.get('neural_training_enabled', False)) if resume_ckpt is not None else False
        prior_warmup_win_count = int(resume_ckpt.get('prior_warmup_win_count', 0)) if resume_ckpt is not None else 0
        _set_requires_grad(model_trainable_params, neural_training_enabled)
        _rank0_print(
            rank,
            f"[phase] neural surface starts {'unfrozen' if neural_training_enabled else 'frozen'}; "
            f"steerer warmup gate requires eval_prior_on + {args.freeze_model_until_prior_on_beats_off:g} < eval_prior_off "
            f"for {args.prior_on_warmup_patience} eval(s)",
        )

    for epoch_offset in range(1, args.epochs + 1):
        ep = resume_start_epoch + epoch_offset
        epoch_neural_training_enabled = neural_training_enabled
        model.train()
        total_loss = 0.0; t0 = time.time()
        loader_iter = iter(train_loader)

        for step in range(args.steps):
            x, y, w_cpu = next(loader_iter)
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            w_cpu = w_cpu.to(device, non_blocking=True)

            if steerer is not None and prior_training_enabled:
                w_gpu = gpu_fc.compute_features(x)
                w_gpu[:, :, 0:9] = w_cpu[:, :, :9]
                steerer.set_weights(w_gpu)
            elif steerer is not None:
                steerer._current_weights = None
            out = model(x)
            loss = F.cross_entropy(out.logits.reshape(-1, V), y.reshape(-1))
            if steerer is not None and prior_training_enabled:
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

        avg_loss = total_loss / args.steps
        current_batch_ppl = math.exp(avg_loss)
        elapsed = time.time() - t0
        status = ""
        if eval_b < best_eval_b: best_eval_b = eval_b; status += "b"
        if eval_s < best_eval_s:
            best_eval_s = eval_s
            status += "s"
            last_prior_on_improvement_epoch = ep
        early_stop_improved = _early_stop_metric_improved(status, args.early_stop_metric)
        if early_stop_improved:
            last_early_stop_improvement_epoch = ep

        if warmup_gate_enabled and not neural_training_enabled:
            if _prior_on_beats_off(eval_s, eval_b, args.freeze_model_until_prior_on_beats_off):
                prior_warmup_win_count += 1
            else:
                prior_warmup_win_count = 0
            if prior_warmup_win_count >= max(1, args.prior_on_warmup_patience):
                neural_training_enabled = True
                _set_requires_grad(model_trainable_params, True)
                _rank0_print(
                    rank,
                    f"[phase] eval_prior_on={eval_s:.1f} beat eval_prior_off={eval_b:.1f} "
                    f"for {prior_warmup_win_count} eval(s); neural surface will train from next epoch",
                )

        if status:
            backend_name = f"{args.backend}_4bit" if args.compute_in_4bit else args.backend
            default_suffix = "_c4_mix" if args.data_mode == "c4-mix" else ""
            out_dir = os.path.expanduser(
                args.out_dir or f"~/deepseek_experiments/artifacts/train_{args.model_config}_{args.train_surface}_{backend_name}{default_suffix}"
            )
            _save_metric_checkpoints(
                out_dir,
                status,
                {'state_dict': model.state_dict(),
                 'steerer_state': steerer.state_dict() if steerer is not None else None,
                 'backend': args.backend,
                 'compute_in_4bit': args.compute_in_4bit,
                 'model_config': args.model_config,
                 'train_surface': args.train_surface,
                 'data_mode': args.data_mode,
                 'data_dir': args.data_dir,
                 'c4_ratio': args.c4_ratio,
                 'seed': args.seed,
                 'batch': args.batch,
                 'seq_len': args.seq_len,
                 'eval_tokens': args.eval_tokens,
                 'current_batch_loss': avg_loss,
                 'current_batch_ppl': current_batch_ppl,
                 'compiled_prior_active_during_train': steerer is not None,
                 'eval_prior_on': eval_s,
                 'eval_prior_off': eval_b,
                 'prior_training_enabled': prior_training_enabled,
                 'neural_training_enabled': neural_training_enabled,
                 'neural_training_enabled_this_epoch': epoch_neural_training_enabled,
                 'freeze_model_until_prior_on_beats_off': args.freeze_model_until_prior_on_beats_off,
                 'prior_on_warmup_patience': args.prior_on_warmup_patience,
                 'prior_warmup_win_count': prior_warmup_win_count,
                 'disable_prior_after_on_plateau': args.disable_prior_after_on_plateau,
                 'early_stop_metric': args.early_stop_metric,
                 'early_stop_patience': args.early_stop_patience,
                 'epoch': ep,
                 'eval_s': eval_s,
                 'eval_b': eval_b,
                 'best_eval_s': best_eval_s,
                 'best_eval_b': best_eval_b,
                 'resume_checkpoint': args.resume_checkpoint},
            )

        _rank0_print(rank, f"  epoch={ep:3d}  current_batch_loss={avg_loss:.4f}  current_batch_ppl={current_batch_ppl:.1f}  "
                 f"eval_prior_on={eval_s:.1f}  best_prior_on={best_eval_s:.1f}  "
                     f"eval_prior_off={eval_b:.1f}  best_prior_off={best_eval_b:.1f}  "
                     f"prior_train={'on' if prior_training_enabled else 'off'}  "
                     f"neural_train={'on' if epoch_neural_training_enabled else 'warmup'}  [{status}]  time={elapsed:.0f}s")

        if (
            steerer is not None
            and prior_training_enabled
            and neural_training_enabled
            and args.disable_prior_after_on_plateau > 0
            and ep - last_prior_on_improvement_epoch >= args.disable_prior_after_on_plateau
        ):
            prior_training_enabled = False
            steerer._current_weights = None
            _rank0_print(
                rank,
                f"[phase] eval_prior_on plateaued for {args.disable_prior_after_on_plateau} epochs; "
                "compiled prior/steerer disabled for subsequent training epochs",
            )

        if args.early_stop_metric != "none" and args.early_stop_patience > 0:
            stale_epochs = ep - last_early_stop_improvement_epoch
            if stale_epochs >= args.early_stop_patience:
                _rank0_print(
                    rank,
                    f"[early-stop] metric={args.early_stop_metric} patience={args.early_stop_patience} "
                    f"last_improved_epoch={last_early_stop_improvement_epoch} current_epoch={ep}",
                )
                break

    _rank0_print(rank, f"Done. Best eval_prior_off: {best_eval_b:.1f}  Best eval_prior_on: {best_eval_s:.1f}")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
