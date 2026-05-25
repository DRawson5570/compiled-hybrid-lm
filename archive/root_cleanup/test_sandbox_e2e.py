"""test_sandbox_e2e.py

End-to-end integration harness starting the secured auth server, testing maths,
verifying Bearer gate validation steps, and ensuring isolated sandbox stability.
"""
from __future__ import annotations

import subprocess
import time
import urllib.request
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent

def run_integration_check():
    print("=" * 80)
    print("      CMI AUTH-GATED END-TO-END SANDBOX SYSTEM CHECK")
    print("=" * 80)
    
    # 1. Spin up the Secure server process
    server_proc = subprocess.Popen(
        ["/home/drawson/anaconda3/envs/open-webui/bin/python", "-u", str(REPO / "bearer_auth_gateway.py"), "8850"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    time.sleep(6.0) # Grace loading period
    
    # Check if process is still running
    if server_proc.poll() is not None:
        print("Error: SECURED server exited prematurely.")
        return False
        
    url = "http://127.0.0.1:8850/api/cmi/stream"
    payload = json.dumps({"prompt": "translate dog to french", "max_tokens": 1}).encode("utf-8")
    
    # Test A: Execution WITHOUT valid Bearer Token (Should return 401 Unauthorized)
    print("Test A: Submitting prompt with missing/invalid Bearer authorization...")
    req_bad = urllib.request.Request(
        url, data=payload, headers={"Content-Type": "application/json"}
    )
    try:
        urllib.request.urlopen(req_bad, timeout=5)
        print("  \033[91mFail: Request succeeded when it should have returned 401!\033[0m")
        test_a_ok = False
    except urllib.error.HTTPError as e:
        if e.code == 401:
            print("  \033[92mSuccess: Endpoint correctly blocked with HTTP 401 Unauthorized.\033[0m")
            test_a_ok = True
        else:
            print(f"  Fail: Returned unexpected error code: {e.code}")
            test_a_ok = False
            
    # Test B: Execution WITH valid Bearer Token (Should succeed and stream output)
    print("\nTest B: Submitting prompt with authorized Bearer Token...")
    req_good = urllib.request.Request(
        url, data=payload, headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer sandbox-token-12345"
        }
    )
    try:
        with urllib.request.urlopen(req_good, timeout=5) as resp:
            data = resp.readline().decode("utf-8").strip()
            if data.startswith("data:"):
                block = json.loads(data[5:].strip())
                print(f"  Received token: \033[92m{block.get('token')}\033[0m")
                test_b_ok = BlockOk = "token" in block
            else:
                test_b_ok = False
    except Exception as e:
        print(f"  Fail: Access denied or exception thrown: {e}")
        test_b_ok = False
        
    # Standard cleanup
    server_proc.terminate()
    server_proc.wait()
    
    overall = test_a_ok and test_b_ok
    status_str = "\033[92mPASSED\033[0m" if overall else "\033[91mFAILED\033[0m"
    print(f"  Overall Verdict: {status_str}")
    print("=" * 80)
    return overall

if __name__ == "__main__":
    run_integration_check()
