"""Sandbox test runner for generated Python code."""
import subprocess, tempfile, os, sys, time
from dataclasses import dataclass

@dataclass
class TestResult:
    passed: bool
    stdout: str
    elapsed: float

def run_test(code: str, test_code: str, timeout: float = 5.0) -> TestResult:
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(
            [sys.executable, "-c", test_code],
            input=code, capture_output=True, text=True,
            timeout=timeout,
            env={**os.environ, "PYTHONPATH": os.path.expanduser("~/deepseek_experiments")},
        )
        stdout = proc.stdout.strip()
        passed = "PASS" in stdout and "FAIL" not in stdout
    except subprocess.TimeoutExpired:
        stdout = "TIMEOUT"
        passed = False
    except Exception as e:
        stdout = f"ERROR: {e}"
        passed = False
    elapsed = time.perf_counter() - t0
    return TestResult(passed=passed, stdout=stdout, elapsed=elapsed)
