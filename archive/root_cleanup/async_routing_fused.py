"""async_routing_fused.py

Asynchronous multi-channel CMI routing engine executing expert lookups
concurrently under Python's async/await framework.
"""
from __future__ import annotations

import asyncio
import time
import torch
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

from hybrid.v2_capabilities.dataset import tok2id, id2tok, V

async def evaluate_channel_async(channel_name: str, ids: torch.Tensor) -> dict:
    """Simulates async model evaluation with varying computational complexities."""
    t_start = time.perf_counter()
    # Add minor mock context switch delays representing CPU/GPU concurrency
    if channel_name == "CoderChannel":
        await asyncio.sleep(0.015)  # simulates syntactical context complexity
    elif channel_name == "ReasonerChannel":
        await asyncio.sleep(0.025)  # simulates multi-hop reasoning
    else:
        await asyncio.sleep(0.005)  # standard fast channel lookup

    # Construct mock output logits of vocabs
    mock_logits = torch.randn(ids.shape[0], V)
    dur = (time.perf_counter() - t_start) * 1000.0
    return {"channel": channel_name, "logits": mock_logits, "duration_ms": dur}

async def dispatch_concurrent_routing(prompt_tokens: list[str]):
    print(f"Async dispatching multi-channel routing for: {prompt_tokens}")
    ids = torch.tensor([tok2id.get(t, 0) for t in prompt_tokens])
    
    channels = ["InstructChannel", "ReasonerChannel", "CoderChannel", "ToolChannel"]
    
    t_start = time.perf_counter()
    # Execute all 4 expert channels concurrently using asyncio.gather
    tasks = [evaluate_channel_async(c, ids) for c in channels]
    results = await asyncio.gather(*tasks)
    total_dur = (time.perf_counter() - t_start) * 1000.0
    
    print("\n--- Concurrency Execution Outputs ---")
    sum_individual = 0.0
    for r in results:
        print(f"  Channel: \033[96m{r['channel']:<16}\033[0m | Task Latency: {r['duration_ms']:.2f}ms")
        sum_individual += r['duration_ms']
        
    print(f"Sum of Serial Latencies: {sum_individual:.2f}ms")
    print(f"Actual Concurrent Assembly Latency: \033[92m{total_dur:.2f}ms\033[0m (Overlap achieved!)")
    print("-" * 50)
    return results

if __name__ == "__main__":
    asyncio.run(dispatch_concurrent_routing(["translate", "dog", "to", "french"]))
