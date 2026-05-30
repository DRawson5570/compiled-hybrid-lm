"""smoke_test_4b.py — ZeroQ 4-bit sharding with manual gather/release.

Avoids hook nesting issues with nn.MultiheadAttention by calling
coordinator.fetch_params/release_params manually per step.

Usage:
  torchrun --nproc_per_node=2 --master_addr=localhost --master_port=29500 \
           smoke_test_4b.py --model-config 4b
"""
import argparse, os, sys, socket, time
import torch, torch.distributed as dist
import torch.nn.functional as F

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)
sys.path.insert(0, os.path.expanduser("~/ZeroQ"))

import hf_deepseek
from hf_deepseek import DeepSeekConfig, DeepSeekForCausalLM

CFG_4B = dict(d_model=2688, n_layers=32, n_heads=21, d_ff=10752, max_len=512)


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
    for p in model.parameters():
        if not p.requires_grad or p.grad is None: continue
        dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
        p.grad.div_(world_size)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-config", choices=["4b", "test"], default="test")
    p.add_argument("--steps", type=int, default=5)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--seq-len", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    args = p.parse_args()

    rank, world_size, local_rank, device = _init_dist()
    hostname = socket.gethostname()
    _rank0_print(rank, f"=== ZeroQ MANUAL: DeepSeekForCausalLM ({args.model_config}) ===")
    _rank0_print(rank, f"Rank {rank}/{world_size} host={hostname} device={device}")

    if args.model_config == "4b":
        cfg = DeepSeekConfig(**CFG_4B)
    else:
        cfg = DeepSeekConfig(d_model=768, n_layers=4, n_heads=8, d_ff=2048)

    _rank0_print(rank, f"Config: d={cfg.d_model} L={cfg.n_layers}")

    # Build on CPU — ZeroQ's partition moves each tensor to GPU one at a time,
    # quantizes, and frees the full-precision copy. No peak fp32 memory issue.
    model = DeepSeekForCausalLM(cfg)  # stays on CPU
    n_params = sum(p.numel() for p in model.parameters())
    _rank0_print(rank, f"Params: {n_params:,}")

    # Mark everything frozen, then mark non-Linear params trainable to skip them
    for p in model.parameters():
        p.requires_grad = False
    model.tok_emb.weight.requires_grad = True
    model.pos_emb.weight.requires_grad = True
    for m in model.modules():
        if isinstance(m, (torch.nn.LayerNorm, torch.nn.Dropout)):
            for p in m.parameters():
                p.requires_grad = True
    _rank0_print(rank, f"[DEBUG] tok_emb req_grad={model.tok_emb.weight.requires_grad} "
                 f"pos_emb req_grad={model.pos_emb.weight.requires_grad}")

    from src.config import MAXWELL_CONFIG
    from src.coordinator import ZeroQCoordinator, ZeroQModuleWrapper

    coordinator = ZeroQCoordinator(MAXWELL_CONFIG)
    wrapper = ZeroQModuleWrapper(model, coordinator, trainable_only=False)

    wrapper = ZeroQModuleWrapper(model, coordinator, trainable_only=False)
    
    # Move each param to GPU one at a time, then partition. Avoids fp32 OOM.
    _rank0_print(rank, "[ZeroQ] Streaming param move + partition...")
    from src.coordinator import ZeroQParamStatus
    t0 = time.time()
    for pid, zqp in coordinator._params.items():
        if zqp.status == ZeroQParamStatus.NOT_AVAILABLE:
            zqp.partition_from_full_precision(zqp.param.data.to(device=device))
    _rank0_print(rank, f"[ZeroQ] Done in {time.time()-t0:.1f}s")

    # Now set actual trainable params (embeddings and LayerNorm stay unpartitioned)
    model.tok_emb.weight.requires_grad = False
    model.pos_emb.weight.requires_grad = False
    for m in model.modules():
        if isinstance(m, torch.nn.LayerNorm):
            for p in m.parameters():
                p.requires_grad = False
    model.head_bias.requires_grad = True
    model.ln_f.weight.requires_grad = True
    model.ln_f.bias.requires_grad = True

    # Move unpartitioned params to GPU (embeddings, LayerNorm, head_bias)
    for p in model.parameters():
        if p.requires_grad or not p.device.type == 'cuda':
            p.data = p.data.to(device=device)
    _rank0_print(rank, f"[ZeroQ] All params on device")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    _rank0_print(rank, f"Trainable: {trainable:,}")

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)

    # Hooks handle gather/release automatically per-module. No manual calls.
    _rank0_print(rank, f"Training {args.steps} steps (hook-based gather)...")
    for step in range(args.steps):
        x = torch.randint(0, 1000, (args.batch, args.seq_len), device=device)
        labels = x.clone()

        model.train()
        out = model(x, labels=labels)
        loss = out.loss

        opt.zero_grad(set_to_none=True)
        loss.backward()
        _manual_allreduce_grads(model, world_size)
        opt.step()

        _rank0_print(rank, f"  step={step+1}/{args.steps}  loss={loss.item():.4f}  "
                     f"mem={torch.cuda.max_memory_allocated(device)/1e9:.2f}GB")

    _rank0_print(rank, "SMOKE TEST PASSED")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
