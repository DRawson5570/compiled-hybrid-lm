"""Phase 3: DeepSeek generates targeted training data per weakness.

For each weakness in the catalog, the teacher generates novel coding problems.
Each problem's test code is validated against its canonical solution in the sandbox.
Outputs weaknesses/{weakness_id}_training.jsonl.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path.home() / "deepseek_experiments"))
sys.path.insert(0, str(Path.home() / "code_harness"))

from teacher.deepseek_client import DeepSeekTeacher


def _run_code_test(generated_code: str, test_code: str) -> bool:
    full = f"{generated_code}\n\n{test_code}"
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False,
                                     encoding="utf-8") as f:
        f.write(full)
        tmp = f.name
    try:
        r = subprocess.run(["python3", tmp], capture_output=True, text=True, timeout=10.0)
        return r.returncode == 0 and "FAIL" not in r.stdout and "FAIL" not in r.stderr
    except Exception:
        return False
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass


def validate_example(ex: dict[str, str]) -> bool:
    if not all(k in ex for k in ("prompt", "expected", "test_code", "entry_point")):
        return False
    if not ex["expected"].strip():
        return False
    result = _run_code_test(ex["expected"], ex["test_code"])
    return result


def synthesize_for_weakness(
    teacher: DeepSeekTeacher,
    weakness: dict[str, Any],
    count: int = 20,
    output_dir: Path | str = None,
) -> list[dict[str, Any]]:
    wid = weakness["weakness_id"]
    name = weakness.get("weakness_name", wid)
    desc = weakness.get("description", "")

    print(f"\nSynthesizing {count} examples for: {name} ({wid})", flush=True)
    examples = teacher.synthesize_examples(name, desc, count=count)
    if not examples:
        print(f"  WARNING: DeepSeek returned 0 examples for {wid}. "
              f"Check API key, rate limits, or prompt compatibility.", flush=True)
        return []

    valid: list[dict[str, Any]] = []
    for i, ex in enumerate(examples):
        if not isinstance(ex, dict):
            continue
        ex["weakness_id"] = wid
        ex["id"] = f"{wid}_{i:03d}"
        ex["tier"] = 1
        ex["source"] = "teacher_synthesized"

        if validate_example(ex):
            valid.append(ex)
            print(f"  [{len(valid)}/{i+1}] ✓", end="", flush=True)
        else:
            print(f"  [{len(valid)}/{i+1}] ✗ failed validation", end="", flush=True)
        if (i + 1) % 5 == 0:
            print(flush=True)

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / f"{wid}_training.jsonl"
        with open(out_path, "w") as f:
            for ex in valid:
                f.write(json.dumps(ex) + "\n")
        print(f"\n  Wrote {len(valid)} valid examples to {out_path}", flush=True)

    return valid


def synthesize_all(teacher: DeepSeekTeacher, catalog: dict[str, Any],
                   examples_per_weakness: int = 20,
                   output_dir: Path | str = None) -> dict[str, list[dict[str, Any]]]:
    results: dict[str, list[dict[str, Any]]] = {}
    for w in catalog.get("weaknesses", []):
        wid = w["weakness_id"]
        results[wid] = synthesize_for_weakness(
            teacher, w, count=examples_per_weakness, output_dir=output_dir
        )
        time.sleep(1.0)
    return results


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalog", default=str(Path.home() / "code_harness" / "weaknesses" / "catalog.json"))
    ap.add_argument("--api-key-path", default=str(Path.home() / "api_keys" / "deepseek"))
    ap.add_argument("--output-dir", default=str(Path.home() / "code_harness" / "weaknesses"))
    ap.add_argument("--count", type=int, default=20)
    ap.add_argument("--weakness-id", default=None, help="Only synthesize for one weakness")
    args = ap.parse_args()

    catalog = json.loads(Path(args.catalog).read_text())
    teacher = DeepSeekTeacher(api_key_path=args.api_key_path)

    if args.weakness_id:
        w = next((w for w in catalog["weaknesses"] if w["weakness_id"] == args.weakness_id), None)
        if w is None:
            print(f"Weakness not found: {args.weakness_id}")
            return
        synthesize_for_weakness(teacher, w, count=args.count, output_dir=args.output_dir)
    else:
        synthesize_all(teacher, catalog, examples_per_weakness=args.count,
                       output_dir=args.output_dir)


if __name__ == "__main__":
    main()
