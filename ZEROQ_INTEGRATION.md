# ZeroQ Integration Guide — Compiled-Hybrid-LM

> **Updated 2026-05-25:** ZeroQ is now an optional backend under the cartridge ABI (`hybrid/backends.py`).
> Not part of the cartridge API itself. Use `--backend zeroq` on the distributed trainer.

## Architecture (v2 — GPT-5.5)

### `hybrid/backends.py`

Two backends, same cartridge ABI:

```python
from hybrid.backends import DenseTorchBackend, ZeroQPartitionedBackend, TrainableSurface

# Standard dense training (3080, single GPU)
backend = DenseTorchBackend(device="cuda")
handle = backend.prepare(model, TrainableSurface.head_bias())

# ZeroQ 4-bit sharded training (M40, multi-GPU)
backend = ZeroQPartitionedBackend(device="cuda:0", zeroq_path="~/ZeroQ")
handle = backend.prepare(model, TrainableSurface.head_bias())

# Native 4-bit compute path for performance-sensitive frozen backbones.
# This converts frozen nn.Linear modules to bnb.nn.Linear4bit after partitioning
# and avoids per-layer gather/release hooks during forward/backward.
backend = ZeroQPartitionedBackend(device="cuda:0", zeroq_path="~/ZeroQ", compute_in_4bit=True)
handle = backend.prepare(model, TrainableSurface.head_bias())
```

`TrainableSurface.head_bias()` declares only `head_bias` as trainable — everything else is frozen and quantized by ZeroQ. The steerer (SuperpositionSteererV3) is always trainable and mounts via hooks on DecoderLayer outputs.

### Streaming Partition (automatic)

The `ZeroQPartitionedBackend` handles streaming automatically:
1. `set_trainable_surface(model, surface)` — marks head_bias trainable, freezes rest
2. `ZeroQModuleWrapper` registers all frozen params
3. `_stream_partition` iterates each param: CPU→GPU→4-bit NF4→shard across ranks→free CPU copy
4. Trainable params (head_bias, LayerNorm) stay materialized on GPU
5. Optional `compute_in_4bit=True` gathers frozen Linear shards once, installs bitsandbytes `Params4bit` weights, and bypasses ZeroQ's per-layer gather/release hook path.

### Multi-GPU Launch

```bash
torchrun --nproc_per_node=2 --master_addr=localhost --master_port=29500 \
    hybrid/train_4b_distributed.py \
    --backend zeroq \
    --model-config test|700m|3b|4b \
    --epochs 100 --batch 2 \
    --zeroq-path ~/ZeroQ \
    --compute-in-4bit
```

## Key Result

**4.69B param steering-enabled model smoke-proven on 2× Tesla M40 24GB with native 4-bit compute.** The old gather/release path ran the 50-step pe2 test at about 1118s/epoch; the native 4-bit smoke completed a full forward/backward/eval/save with the actual epoch body in 5s after one-time conversion.

## What Gets Quantized

| Module Type | Quantized? | Reason |
|---|---|---|
| `nn.Linear` (frozen) | Yes (4-bit NF4) | Largest savings |
| `nn.Embedding` | No | Marked trainable in surface |
| `nn.LayerNorm` | No | Tiny, trainable |
| `head_bias` | No | Trainable surface |
| SuperpositionSteererV3 | No | Always trainable |

## Maxwell/M40 Notes (2026-05-25)

On sm_52/M40, some NCCL CUDA collectives used by ZeroQ gather/release are unsupported or too slow over SYS topology. The backend creates a Gloo process group for trainable-gradient sync on these devices, and the production 4B path should use `--compute-in-4bit` so frozen Linear layers compute through bitsandbytes instead of per-layer ZeroQ gather/release hooks.

## Original Integration (v1 — Ad-hoc)

The first working integration used manual streaming partition via `ZeroQParameter.partition_from_full_precision()` with per-tensor GPU moves. This approach is superseded by `backends.py` but the same mechanism runs under the hood.

## Configuration

### Model Sizes

| d_model | n_layers | n_heads | d_ff | Params | VRAM (ZeroQ) |
|---|---|---|---|---|---|
| 768 | 12 | 12 | 3072 | 124M | — |
| 1536 | 20 | 12 | 6144 | 644M | — |
| 2688 | 32 | 21 | 10752 | 2.9B | **1.8GB** |
| 3072 | 40 | 24 | 12288 | 4.7B | ~2.5GB est. |

### ZeroQ Config (Maxwell/M40)

```python
ZeroQConfig(
    compute_dtype=torch.float32,
    double_quant=True,
    blocksize=64,
    async_gather=True,
    frozen_only=True,
)
```

Note: M40 does fp32 math natively — no fp16 tensor cores. 4-bit weights dequantize to fp32.

## Files

| File | Purpose |
|---|---|
| `hybrid/backends.py` | Backend abstraction (DenseTorchBackend, ZeroQPartitionedBackend) |
| `hybrid/hf_deepseek.py` | HF-compatible model + DecoderLayer |
| `hybrid/train_4b_distributed.py` | Distributed trainer (supports --backend dense|zeroq) |
| `hybrid/smoke_test_4b.py` | Legacy smoke test (ad-hoc, pre-backends.py) |
| `hybrid/tests/test_backends.py` | Backend unit tests |
| `hybrid/tests/test_hf_deepseek.py` | HF model tests |
| `~/ZeroQ/src/coordinator.py` | ZeroQ core (updated on pe3 with streaming partition) |
