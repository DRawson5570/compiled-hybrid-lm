# ZeroQ: Architecture and Internals

This document explains how ZeroQ achieves 4.7B-parameter model training at 1.8GB VRAM on decade-old GPUs. It covers the partitioning algorithm, gather/release lifecycle, 4-bit compute conversion, NCCL communication patterns, and the hetero extension for mixed-VRAM clusters.

## Table of Contents

1.  [Partitioning Algorithm](#1-partitioning-algorithm)
2.  [Streaming Partition](#2-streaming-partition)
3.  [Gather/Release Lifecycle](#3-gatherrelease-lifecycle)
4.  [4-Bit Compute Mode](#4-4-bit-compute-mode)
5.  [NCCL Communication](#5-nccl-communication)
6.  [Registration Consistency](#6-registration-consistency)
7.  [Hetero Extension](#7-hetero-extension)
8.  [Memory Budget](#8-memory-budget)

---

## 1. Partitioning Algorithm

ZeroQ operates on individual `nn.Parameter` tensors. Each parameter is quantized independently — there is no cross-parameter state sharing. This allows streaming (one tensor at a time) and per-layer precision.

### Quantize Step

```python
# bitsandbytes.functional.quantize_4bit:
packed, quant_state = quantize_4bit(
    weight,                    # fp16 tensor, shape [out, in]
    blocksize=64,              # quantize in groups of 64
    quant_type="nf4",          # NormalFloat4
)

# packed:    uint8 tensor, shape [out * in / 2]     (4 bits per element)
# quant_state: QuantState(absmax, shape, blocksize, ...)
```

The `packed` tensor stores two 4-bit values per byte. The `quant_state` stores per-block absmax values needed for dequantization:

```
dequant(x) = packed[x] * absmax[block_of(x)]
```

### Shard Step

The packed tensor is split into `world_size` equal (or weighted) partitions:

```
partition_size = ceil(packed.numel() / world_size)
rank_i gets: packed[i * partition_size : (i+1) * partition_size]
```

If the tensor size is not evenly divisible by `world_size × blocksize`, padding is added. The `PartitionInfo` struct tracks original shape, padded length, and per-partition sizes.

After sharding, each rank stores only its local packed slice. The full tensor is reconstructed on demand via all-gather.

### Memory Representation

```
Per-rank state for parameter P (shape [3072, 3072], ~9.4M elements):
  local_packed:   uint8 [587K]     (9.4M / 2 / 8 ranks)
  local_absmax:   fp16 [2.3K]      (9.4M / 64 / 64 blocksize)
  Total:          ~1.2 MB per rank (vs 18.9 MB fp16 full tensor)
```

Compression ratio: 18.9 MB / 1.2 MB ≈ 15.8× per tensor. Actual system compression is ~7.1× due to unquantized embeddings and overhead.

---

## 2. Streaming Partition

The full model cannot be loaded on a single GPU before partitioning. Streaming partition solves this by processing parameters one at a time:

```
for each nn.Parameter in model (CPU):
    1. Move tensor to GPU (one tensor, max ~116MB for ffn1 weight)
    2. quantize_4bit() → packed + quant_state
    3. Split packed into world_size shards
    4. Rank i keeps shard i, discards the rest
    5. Replace module parameter with empty placeholder on GPU
    6. Free the fp16 GPU tensor
```

At no point does the full model exist on any single GPU. Peak GPU memory is the largest single parameter (~116MB for the 4.7B model), not the full 9.4GB.

### Pseudo-code

```python
def stream_partition(coordinator, device):
    for zq_param in coordinator.params:
        weight = zq_param.param.data.to(device)  # CPU → GPU
        packed, quant_state = quantize_4bit(weight, ...)
        shard_size = ceil(packed.numel() / world_size)
        local_packed = packed[rank * shard_size : (rank+1) * shard_size]
        zq_param.local_packed = local_packed
        zq_param.param.data = torch.empty(0, device=device)  # free GPU memory
        del weight  # explicit GC
```

Parameters that are marked `requires_grad=True` (head_bias, LayerNorm, embeddings) are skipped — they stay materialized and unquantized.

---

## 3. Gather/Release Lifecycle

### Standard Mode

During forward/backward, parameters are gathered on-demand via forward hooks:

```
┌─ FORWARD PASS ─────────────────────────────────────┐
│                                                     │
│  _pre_forward_hook(layer):                          │
│    all_gather(local_packed)  →  full_packed         │
│    all_gather(local_absmax)  →  full_absmax         │
│    dequantize_4bit(full_packed, full_absmax) → fp16 │
│    replace param.data with fp16 tensor              │
│                                                     │
│  layer.forward()  ← uses dequantized fp16 weight    │
│                                                     │
│  _post_forward_hook(layer):                         │
│    replace param.data with torch.empty(0)           │
│    free fp16 tensor                                 │
│                                                     │
└─────────────────────────────────────────────────────┘
```

Each layer has its own hooks. At any moment, only ~1 layer's worth of fp16 parameters is materialized.

### Backward

The same pattern applies in reverse for gradient computation. The `_pre_backward_hook` re-gathers before the backward pass, and `_post_backward_hook` releases after.

### Bottleneck

On SYS-topology GPUs (PCIe across NUMA nodes), each `all_gather` is a synchronous NCCL collective. With 240 Linear layers × 2 (forward + backward) × 2 (packed + absmax) = 960 all-gather operations per training step, the GPU spends 87% of time waiting on NCCL. This is the motivation for 4-bit compute mode.

---

## 4. 4-Bit Compute Mode

4-bit compute mode eliminates the per-layer gather/release entirely by converting `nn.Linear` to `bnb.nn.Linear4bit` after the initial partition.

### Conversion

After streaming partition, for each `nn.Linear` module:

```python
# 1. Gather the full 4-bit packed weight once
coordinator.fetch_params([param_id])  # all_gather across ranks

# 2. Build Params4bit from the assembled packed data
packed_2d = assembled_packed.view(-1, 1)  # [N] → [N, 1]
param_4bit = bnb.nn.Params4bit(
    packed_2d,
    requires_grad=False,
    quant_state=gathered_state,
    quant_type="nf4",
    blocksize=64,
    bnb_quantized=True,
)

# 3. Replace nn.Linear with bnb.nn.Linear4bit
new_linear = bnb.nn.Linear4bit(
    old_linear.in_features, old_linear.out_features,
    bias=old_linear.bias is not None,
    compute_dtype=torch.float16,
)
new_linear.weight = param_4bit
new_linear.bias = old_linear.bias

# 4. Replace in parent module
setattr(parent_module, child_name, new_linear)

# 5. Remove ZeroQ hooks — no longer needed
wrapper.remove_hooks()
```

### How Linear4bit Works

`bnb.nn.Linear4bit.forward()` calls bitsandbytes' fused `matmul_4bit` CUDA kernel:

```c
// Pseudocode for matmul_4bit:
// It takes 4-bit packed weight + absmax, dequantizes on-the-fly
// within the matmul kernel, never materializing an fp16 weight tensor.
output = matmul_4bit(input, packed_weight, absmax, bias)
```

The weight stays as `Params4bit` — an 8-bit packed tensor + quantization metadata — for the entire lifetime of the model. No gather, no release, no fp16 materialization. The GPU's tensor cores perform the dequantization inline during the matrix multiply.

### Result

| Metric | Gather/Release | 4-Bit Compute |
|--------|---------------|---------------|
| Per-step all-gathers | 960 | 1 (initial only) |
| GPU utilization | 13% | 95% |
| VRAM per GPU | 17.7 GB (cycling) | 5.4 GB (steady) |
| Throughput | 2.86 tok/s | 87.7 tok/s |
| Temperature | 84°C | 50-63°C |

The temperature reduction is because `matmul_4bit` processes smaller data types — the memory controller and ALUs handle 4-bit packed data instead of fp16, reducing thermal load by ~20°C at equivalent utilization.

---

## 5. NCCL Communication

### Topology

Two Tesla M40 GPUs on pe2 connected via SYS topology:

```
GPU0  X   SYS
GPU1 SYS   X
```

SYS means the GPUs communicate across NUMA nodes via PCIe + CPU interconnect (QPI/UPI). This is the slowest possible GPU-to-GPU path. NCCL throughput is ~300 MiB/s on this topology.

### All-Gather Pattern

For a parameter of shape [3072, 3072] (18.9 MB fp16):

```
Standard gather/release (per forward pass):
  4-bit packed: 2.4 MB × 960 all-gathers/step = 2.3 GB/step
  At 300 MiB/s: 7.8 seconds of NCCL time per step

4-bit compute (one-time):
  4-bit packed: 2.4 MB × 482 parameters = 1.2 GB total
  At 300 MiB/s: 4.0 seconds total (once at startup)
```

The 4-bit compute path does all 482 all-gathers at initialization, then never touches NCCL again for forward/backward. The 4 seconds of startup cost is amortized over thousands of training steps.

### Process Group

On Maxwell GPUs (compute capability 5.2), NCCL's default backend has issues with collective operations. ZeroQ falls back to `gloo` for the process group on Maxwell:

```python
if torch.cuda.get_device_capability(device)[0] < 6:
    process_group = dist.new_group(backend='gloo')
```

### Gradient Sync

Only trainable parameters (head_bias, LayerNorm, steerer) need gradient synchronization. Base model weights are frozen — no gradient communication needed. Manual all-reduce on ~115K trainable parameters:

```python
for p in model.parameters():
    if p.requires_grad and p.grad is not None:
        dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
        p.grad.div_(world_size)
```

---

## 6. Registration Consistency

After partition, each rank independently hashes its `param_id → shape` mapping and all-gathers the result. All ranks verify they agree on the registration order and tensor shapes before any training step runs:

```python
def verify_registration_consistency(coordinator):
    local_hash = hash((param_id, shape) for all registered params)
    all_hashes = [torch.zeros(1, dtype=torch.long) for _ in range(world_size)]
    dist.all_gather(all_hashes, local_hash)
    assert all(h == all_hashes[0] for h in all_hashes)
```

This catches ordering bugs (e.g., one rank registering parameters in a different order due to model architecture differences) before they manifest as silent NCCL deadlocks or incorrect gradient computation.

---

## 7. Hetero Extension

The `hetero/` module supports mixed-VRAM clusters. A 24GB GPU takes proportionally more of each shard than a 12GB GPU:

### Shard Weighting

```python
def discover_rank_weights():
    vram = torch.cuda.get_device_properties(rank).total_memory
    return vram / sum(all_vrams)
```

A 24GB GPU (66.7% of a 36GB pair) gets 66.7% of each parameter's shard. A 12GB GPU gets 33.3%. This automatically balances memory across heterogeneous hardware.

### Activation Reserve

```python
config.activation_reserve_mb = 4000  # Subtract 4GB from VRAM before weighting
```

This accounts for per-rank activation memory that doesn't shard (embeddings, LayerNorm, optimizer states). Smaller GPUs get proportionally more headroom.

### Variable-Length All-Gather

Since shards have different sizes on different ranks, standard `all_gather` (which requires equal-sized inputs) doesn't work. The `varlen_collectives.py` module implements variable-length all-gather using `all_to_all` to exchange shard sizes first, then `all_gather` with pre-allocated buffers sized to the maximum.

---

## 8. Memory Budget

### 4.7B Model, 2× M40 24GB, Batch=6

| Component | Per GPU | Notes |
|-----------|---------|-------|
| Linear weights (4-bit) | 1,215 MB | 482 parameters, 7.11× compression |
| Embeddings (fp32, unquantized) | 540 MB | tok_emb (50257×3072) + pos_emb (512×3072) |
| LayerNorm + biases | <10 MB | 82 LayerNorm params + 482 biases |
| Steerer (65K params) | <1 MB | SuperpositionSteererV3 |
| Optimizer (trainable only) | <1 MB | AdamW states for ~115K params |
| Activations (batch=6, seq=64) | ~3,600 MB | Attn intermediates + FFN + residuals |
| NCCL buffers | ~50 MB | All-gather scratch space |
| **Total** | **~5,400 MB** | |

### Headroom

24,000 MB available − 5,400 MB used = 18,600 MB free. This headroom enables:

- Batch scaling to ~12-16 before OOM
- Larger models (4.7B → 10B with same batch size)
- Future steerer expansions (more hooks, larger MLPs)
- Longer sequence lengths (128 → 512)

### Scaling to 5 GPUs

With 5× M40 24GB (120GB total), the same 4.7B model would use ~1.2GB per GPU for weights, leaving ~2-3× more activation headroom. A 30B model (7.5GB 4-bit weights per GPU) would fit with activation checkpointing. A 35B model (8.8GB) would be activation-bound at batch=1.

---

## Requirements

```
torch >= 2.0
bitsandbytes == 0.41.3    # MUST PIN — last Maxwell-compatible version
triton == 3.3.1            # Required by bitsandbytes 0.41.3
nccl                       # GPU communication
```

**Critical:** bitsandbytes 0.46.1+ dropped CUDA compute capability 5.2 (Maxwell). Version 0.41.3 is the last version that supports Tesla M40, GTX 9xx, and GTX 10xx GPUs. Must be pinned in all requirements files.

## References

- [ZeroQ Repository](https://github.com/DRawson5570/ZeroQ)
- [compiled-hybrid-lm Repository](https://github.com/DRawson5570/compiled-hybrid-lm)
- [bitsandbytes Documentation](https://github.com/TimDettmers/bitsandbytes)
- [NCCL Documentation](https://docs.nvidia.com/deeplearning/nccl/)
