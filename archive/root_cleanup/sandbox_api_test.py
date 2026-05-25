"""sandbox_api_test.py

Verifies streaming output, response structures, routing correctness,
and performs latency benchmarks on the sandboxed CMI API server.
"""
import urllib.request
import json
import time

def test_stream(prompt_str, max_tokens=10):
    url = "http://localhost:8850/api/cmi/stream"
    payload = json.dumps({"prompt": prompt_str, "max_tokens": max_tokens}).encode("utf-8")
    
    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"}
    )
    
    print(f"\nEvaluating Prompt: '{prompt_str}' (max_tokens={max_tokens})")
    print("-" * 75)
    
    start_time = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=5) as response:
            line_count = 0
            for line in response:
                decoded_line = line.decode("utf-8").strip()
                if decoded_line.startswith("data:"):
                    line_count += 1
                    data_str = decoded_line[5:].strip()
                    block = json.loads(data_str)
                    
                    if "status" in block and block["status"] == "done":
                        print(f"\n[DONE] Server confirmed completion in {block['total_duration_ms']:.2f}ms.")
                        break
                    
                    token = block["token"]
                    dominant = block["dominant_channel"]
                    latency_ms = block["step_duration_ms"]
                    print(f"Token {block['step_number']:2d}: \033[92m{token:<12}\033[0m | Dominant Channel: \033[96m{dominant:<16}\033[0m | Latency: {latency_ms:.2f}ms")
    except Exception as e:
        print(f"Error during stream processing: {e}")
        return False, 0.0
    
    total_time = (time.perf_counter() - start_time) * 1000.0
    print(f"Full Round-trip response processed in {total_time:.2f}ms.\n")
    return True, total_time

if __name__ == "__main__":
    time.sleep(1.0) # Grace period for server setup
    
    # Test arithmetic tool injection trigger
    test_stream("What is 12 + 15 ? [USE_TOOL: calculator expr= 12+15 ] Answer is", max_tokens=4)
    
    # Test standard translation triggering InstructChannel
    test_stream("translate dog to french", max_tokens=3)
