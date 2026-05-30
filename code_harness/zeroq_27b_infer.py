#!/usr/bin/env python3
"""ZeroQ 4-bit inference benchmark for Qwen3.6-27B on pe3 (2× M40 12GB).

Stream-loads weights from sharded safetensors, partitions with ZeroQ,
then measures single-token generation speed (tps).
"""
import os
import sys
import time
import glob
from pathlib import Path

import torch
import torch.distributed as dist
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from accelerate import init_empty_weights
from safetensors.torch import safe_open

sys.path.insert(0, str(Path.home() / "ZeroQ" / "src"))
from coordinator import ZeroQCoordinator, ZeroQModuleWrapper, MAXWELL_CONFIG

MODEL_DIR = Path.home() / "models" / "qwen3.6-27b-safetensors"
TOKENIZER_ID = "Qwen/Qwen3.5-4B"  # tokenizer compatible with Qwen3.6


def find_safetensors(model_dir):
    files = sorted(glob.glob(str(model_dir / "model*.safetensors")))
    return [f for f in files if "index" not in f]


def stream_load_and_partition(model, coordinator, safetensor_files):
    """Stream each safetensor shard, partition params through ZeroQ."""
    registered = 0
    for st_file in safetensor_files:
        with safe_open(st_file, framework="pt", device="cpu") as f:
            for key in f.keys():
                tensor = f.get_tensor(key)
                try:
                    param = dict(model.named_parameters())[key]
                except KeyError:
                    continue
                zq_param = coordinator.get_param_for_tensor(param)
                if zq_param is not None:
                    zq_param.partition_from_full_precision(tensor)
                    registered += 1
                del tensor
    print(f"  Partitioned {registered} parameters", flush=True)
    return registered


def benchmark_tps(model, tokenizer, prompt="def fibonacci(n):\n    ", n_tokens=20):
    """Measure tokens-per-second for single-sequence greedy generation."""
    ids = tokenizer.encode(prompt)
    device = next(model.parameters()).device
    x = torch.tensor([ids], device=device)
    torch.cuda.synchronize()
    t0 = time.time()
    tokens_generated = 0
    with torch.no_grad():
        for _ in range(n_tokens):
            out = model(x)
            nid = int(out.logits[0, -1].argmax())
            if nid == tokenizer.eos_token_id:
                break
            ids.append(nid)
            x = torch.tensor([ids[-512:]], device=device)
            tokens_generated += 1
    torch.cuda.synchronize()
    elapsed = time.time() - t0
    tps = tokens_generated / elapsed if elapsed > 0 else 0
    return tps, tokenizer.decode(ids[-tokens_generated:])


def main():
    safetensor_files = find_safetensors(MODEL_DIR)
    if not safetensor_files:
        print(f"No safetensors found in {MODEL_DIR}", flush=True)
        # Try to copy from partial rsync
        safetensor_files = find_safetensors(MODEL_DIR.parent / "qwen3.6-27b-safetensors")
    print(f"Found {len(safetensor_files)} shards", flush=True)

    print("Loading tokenizer...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading config...", flush=True)
    config = AutoConfig.from_pretrained(str(MODEL_DIR), trust_remote_code=True)
    print(f"  model_type={config.text_config.model_type} "
          f"hidden={config.text_config.hidden_size} "
          f"layers={config.text_config.num_hidden_layers}", flush=True)

    print("Building skeleton (init_empty_weights)...", flush=True)
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.float32,
                                                  trust_remote_code=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    print("Initializing ZeroQ coordinator...", flush=True)
    coordinator = ZeroQCoordinator(MAXWELL_CONFIG)
    wrapper = ZeroQModuleWrapper(model, coordinator, trainable_only=False)

    print("Streaming + partitioning weights...", flush=True)
    registered = stream_load_and_partition(model, coordinator, safetensor_files)
    print(f"  Total: {registered} params partitioned", flush=True)

    print(f"\nVRAM per GPU:", flush=True)
    for i in range(torch.cuda.device_count()):
        free = torch.cuda.mem_get_info(i)[0]
        total = torch.cuda.mem_get_info(i)[1]
        print(f"  GPU {i}: {free/1e9:.1f} GB free / {total/1e9:.1f} GB", flush=True)

    print("\nRunning warmup...", flush=True)
    _ = model(torch.tensor([[1]], device="cuda:0"))

    print("\n--- Speed Benchmark ---", flush=True)
    prompts = [
        "def fibonacci(n):\n    ",
        "def binary_search(arr, target):\n    ",
        "import numpy as np\n\ndef matrix_multiply(a, b):\n    ",
    ]
    for prompt in prompts:
        tps, gen = benchmark_tps(model, tokenizer, prompt, n_tokens=10)
        print(f"  {tps:.1f} tok/s  |  prompt: {prompt[:40]}...", flush=True)
        print(f"            gen: {repr(gen[:60])}", flush=True)

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
