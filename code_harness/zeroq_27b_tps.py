#!/usr/bin/env python3
"""ZeroQ 2-GPU inference benchmark for Qwen3.6-27B on pe3."""
import os, sys, time, glob, math
from pathlib import Path
import torch
import torch.distributed as dist
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from accelerate import init_empty_weights
from safetensors.torch import safe_open

sys.path.insert(0, str(Path.home() / "ZeroQ" / "src"))
from coordinator import ZeroQCoordinator, ZeroQModuleWrapper
from config import ZeroQConfig

# M40: fp16 compute for memory (Maxwell does fp32 internally anyway)
CONFIG = ZeroQConfig(
    quant_type="nf4",
    blocksize=64,
    compute_dtype=torch.float16,
    double_quant=True,
)

MODEL_DIR = Path.home() / "models" / "qwen3.6-27b-safetensors"
TOKENIZER_ID = "Qwen/Qwen3.5-4B"


def main():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    print(f"[rank {rank}/{world_size}] device={device}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    config = AutoConfig.from_pretrained(str(MODEL_DIR), trust_remote_code=True)
    if rank == 0:
        tc = config.text_config
        print(f"model_type={tc.model_type} hidden={tc.hidden_size} layers={tc.num_hidden_layers}", flush=True)

    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.float16,
                                                  trust_remote_code=True)
    model.eval()
    model.train()  # required for GC to engage during training; safe when params frozen
    for p in model.parameters():
        p.requires_grad = False

    print(f"[rank {rank}] skeleton built", flush=True)

    coordinator = ZeroQCoordinator(CONFIG)
    wrapper = ZeroQModuleWrapper(model, coordinator, trainable_only=False)
    # hooks are installed automatically in __init__

    safetensor_files = sorted(glob.glob(str(MODEL_DIR / "model*.safetensors")))
    safetensor_files = [f for f in safetensor_files if "index" not in f]
    print(f"[rank {rank}] streaming {len(safetensor_files)} shards", flush=True)
    name_to_param = dict(model.named_parameters())
    name_to_buffer = dict(model.named_buffers())
    registered = 0
    unmatched = 0
    for st_file in safetensor_files:
        with safe_open(st_file, framework="pt", device="cpu") as f:
            for key in f.keys():
                # Some model variants save with "model.language_model." prefix but
                # AutoModelForCausalLM uses just "model." — normalize the key.
                lookup_key = key.replace("language_model.", "", 1)
                buf = name_to_buffer.get(key)
                if buf is None:
                    buf = name_to_buffer.get(lookup_key)
                if buf is not None:
                    with torch.no_grad():
                        buf.data = f.get_tensor(key).to(device, non_blocking=False)
                    continue
                param = name_to_param.get(key)
                if param is None:
                    param = name_to_param.get(lookup_key)
                if param is None:
                    unmatched += 1
                    if unmatched <= 3 and rank == 0:
                        print(f"  unmatched: {key}", flush=True)
                    continue
                # Materialize embedding + lm_head directly — too large for gather/release
                if "embed_tokens" in lookup_key or "lm_head" in lookup_key:
                    weight = f.get_tensor(key).to(device, dtype=param.dtype, non_blocking=False)
                    # Set via parent module to replace meta tensor properly
                    mod_path = ".".join(key.replace("language_model.", "").split(".")[:-1])
                    param_name = key.split(".")[-1]
                    target_mod = model
                    for part in mod_path.split("."):
                        target_mod = getattr(target_mod, part)
                    with torch.no_grad():
                        setattr(target_mod, param_name, torch.nn.Parameter(weight))
                    if rank == 0:
                        print(f"  materialized: {lookup_key} ({weight.nelement()/1e6:.0f}M elems)", flush=True)
                    del weight
                    continue
                zq_param = coordinator.get_param_for_tensor(param)
                if zq_param is None:
                    continue
                if param.requires_grad:
                    continue
                weight = f.get_tensor(key)
                zq_param.partition_from_full_precision(weight)
                registered += 1
                del weight
        import gc; gc.collect()
        torch.cuda.empty_cache()
    if rank == 0:
        print(f"  unmatched keys: {unmatched}", flush=True)
    print(f"[rank {rank}] partitioned {registered} params", flush=True)

    gbytes = torch.cuda.memory_allocated(device) / 1e9
    gbfree = torch.cuda.mem_get_info(device)[0] / 1e9
    print(f"[rank {rank}] VRAM used={gbytes:.1f}GB free={gbfree:.1f}GB", flush=True)

    # Warmup
    prompt = "def fibonacci(n):\n    "
    ids = tokenizer.encode(prompt)
    x = torch.tensor([ids[-64:]], device=device)
    with torch.no_grad():
        _ = model(input_ids=x)
    torch.cuda.synchronize()

    # Benchmark
    prompts = [
        "def fibonacci(n):\n    ",
        "def binary_search(arr, target):\n    ",
    ]
    for prompt in prompts:
        ids = tokenizer.encode(prompt)
        prompt_len = len(ids)
        x = torch.tensor([ids[-64:]], device=device)
        tokens_generated = 0
        torch.cuda.synchronize()
        t0 = time.time()
        with torch.no_grad():
            for _ in range(10):
                out = model(input_ids=x)
                nid = int(out.logits[0, -1].argmax())
                if nid == tokenizer.eos_token_id:
                    break
                ids.append(nid)
                x = torch.tensor([ids[-64:]], device=device)
                tokens_generated += 1
        torch.cuda.synchronize()
        elapsed = time.time() - t0
        tps = tokens_generated / elapsed if elapsed > 0 else 0
        gen = tokenizer.decode(ids[prompt_len:prompt_len + tokens_generated])
        if rank == 0:
            print(f"  {tps:.1f} tok/s | {prompt[:40]}... => {repr(gen[:80])}", flush=True)

    if rank == 0:
        print("Done.", flush=True)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
