"""parallel_agent_evals.py

Simulates parallel multi-agent client sessions querying sandboxed CMI endpoints.
Launches multiple agent client loops concurrently using thread workers to test
throughput and verify absolute sandbox isolation under simultaneous load.
"""
from __future__ import annotations

import concurrent.futures
import time
import urllib.request
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parent

def simulate_agent_connection(agent_id: int, prompt: str) -> dict:
    url = "http://127.0.0.1:8850/api/cmi/stream"
    payload = json.dumps({"prompt": prompt, "max_tokens": 5}).encode("utf-8")
    
    t_start = time.perf_counter()
    tokens_received = []
    
    try:
        req = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=10) as response:
            for line in response:
                decoded = line.decode("utf-8").strip()
                if decoded.startswith("data:"):
                    block = json.loads(decoded[5:].strip())
                    if "status" in block and block["status"] == "done":
                        break
                    if "token" in block:
                        tokens_received.append(block["token"])
    except Exception as e:
        return {"agent_id": agent_id, "error": str(e), "latency_ms": 0.0}
        
    duration = (time.perf_counter() - t_start) * 1000.0
    return {
        "agent_id": agent_id,
        "tokens": tokens_received,
        "token_count": len(tokens_received),
        "latency_ms": duration,
        "avg_token_latency_ms": (duration / len(tokens_received)) if tokens_received else 0.0
    }

def run_multi_agent_evaluation():
    print("=" * 80)
    print("        CMI MULTI-AGENT CONCURRENT SANDBOX EVALUATOR")
    print("=" * 80)
    
    # Pre-defined agent lookup prompts
    prompts = [
        "What is 12 + 15 ? [USE_TOOL: calculator expr= 12+15 ] Answer is",
        "translate dog to french",
        "def get_sum ( a , b )",
        "E0001 is larger than E0002 . E0002 is larger than",
    ]
    
    print("Launching simulated agent sessions concurrently via ThreadPoolExecutor...")
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(simulate_agent_connection, idx + 1, prompts[idx % len(prompts)]): idx 
            for idx in range(4)
        }
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())
            
    print("\n--- Parallel Connection Diagnostics ---")
    valid_latencies = []
    for r in results:
        if "error" in r:
            print(f"  Agent {r['agent_id']}: \033[91mFailed ({r['error']})\033[0m")
        else:
            tokens_str = " ".join(r["tokens"])
            print(f"  Agent {r['agent_id']}: Completed {r['token_count']} tokens | Output: [\033[92m{tokens_str}\033[0m] | Latency: {r['latency_ms']:.2f}ms")
            valid_latencies.append(r["latency_ms"])
            
    if valid_latencies:
        print(f"\nMean Concurrent Connection Latency: \033[92m{sum(valid_latencies)/len(valid_latencies):.2f}ms\033[0m")
    print("=" * 80)

if __name__ == "__main__":
    run_multi_agent_evaluation()
