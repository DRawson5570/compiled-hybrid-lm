"""Phase 1: Teacher analyzes target model's benchmark failures.

Runs baseline HumanEval eval on the target model, then sends each failure
to DeepSeek for weakness classification. Clusters failures by weakness_id.
Outputs weaknesses/catalog.json.
"""
from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path.home() / "deepseek_experiments"))
sys.path.insert(0, str(Path.home() / "code_harness"))
from datasets import load_dataset

from teacher.deepseek_client import DeepSeekTeacher
from eval_harness.eval import CHAT_INSTRUCTION, extract_code, CodeEvaluator, run_test as _he_run, build_he_program


def load_humaneval_problems() -> list[dict[str, str]]:
    return [
        {
            "task_id": e["task_id"],
            "prompt": e["prompt"],
            "test": e["test"],
            "entry_point": e["entry_point"],
            "canonical_solution": e["canonical_solution"],
        }
        for e in load_dataset("openai/openai_humaneval", split="test")
    ]


def run_baseline(model_name: str, device: str, problems: list[dict],
                 max_new: int = 256, batch: int = 1) -> tuple[dict[str, bool], dict[str, str]]:
    ev = CodeEvaluator(model_name, device, use_chat_template=True)
    prompts = [ev._format_prompt(p, CHAT_INSTRUCTION) for p in problems]
    print(f"  Generating {len(problems)} completions (batch={batch})...", flush=True)
    gens_raw = ev.generate(prompts, max_new=max_new, batch=batch)
    print(f"  Evaluating...", flush=True)
    gens = {}
    results = {}
    for p, g in zip(problems, gens_raw):
        code = extract_code(g)
        gens[p["task_id"]] = code
        passed = _he_run(build_he_program(p["prompt"], code, p["entry_point"]),
                         p["test"], p["entry_point"])
        results[p["task_id"]] = passed
    ev.cleanup()
    return results, gens


def run_analysis(teacher: DeepSeekTeacher, problems: list[dict],
                 baseline_results: dict[str, bool],
                 generated_outputs: dict[str, str],
                 model_name: str,
                 output_dir: Path) -> dict[str, Any]:
    failures_by_weakness: dict[str, list[dict]] = defaultdict(list)
    weakness_info: dict[str, dict[str, str]] = {}
    all_failures: list[dict] = []

    failed = [(tid, p) for tid, p in [(p["task_id"], p) for p in problems]
              if not baseline_results.get(tid, False)]

    for i, (task_id, prob) in enumerate(failed):
        output = generated_outputs.get(task_id, "")
        analysis = teacher.analyze_failure(
            prob["prompt"], prob["canonical_solution"], output
        )
        wid = analysis.get("weakness_id", "unclassified")
        analysis["task_id"] = task_id
        analysis["prompt"] = prob["prompt"]
        analysis["expected"] = prob["canonical_solution"]
        analysis["model_output"] = output
        all_failures.append(analysis)
        failures_by_weakness[wid].append(analysis)
        if wid not in weakness_info:
            weakness_info[wid] = {
                "weakness_id": wid,
                "weakness_name": analysis.get("weakness_name", wid),
                "description": analysis.get("description", ""),
                "severity": analysis.get("severity", "medium"),
                "category": analysis.get("category", "unknown"),
            }
        print(f"  [{i+1}/{len(failed)}] {task_id} → {wid} ({analysis.get('weakness_name')})", flush=True)
        time.sleep(0.3)

    catalog = {
        "model": model_name,
        "benchmark": "humaneval",
        "total_problems": len(problems),
        "total_failures": len(failed),
        "failures": all_failures,
        "weaknesses": [
            {
                **weakness_info[wid],
                "failing_task_ids": [f["task_id"] for f in failures_by_weakness[wid]],
                "failure_count": len(failures_by_weakness[wid]),
            }
            for wid in sorted(failures_by_weakness, key=lambda w: -len(failures_by_weakness[w]))
        ],
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "catalog.json").write_text(json.dumps(catalog, indent=2))
    print(f"\nFound {len(catalog['weaknesses'])} weakness categories across {len(failed)} failures")
    print(f"Catalog written to {output_dir}/catalog.json", flush=True)
    return catalog


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--api-key-path", default=str(Path.home() / "api_keys" / "deepseek"))
    ap.add_argument("--output-dir", default=str(Path.home() / "code_harness" / "weaknesses"))
    ap.add_argument("--baseline-only", action="store_true", help="Only run baseline eval, skip teacher analysis")
    args = ap.parse_args()

    output_dir = Path(args.output_dir)
    problems = load_humaneval_problems()

    print(f"Running baseline eval on {args.model}...", flush=True)
    baseline_results, generated_outputs = run_baseline(
        args.model, args.device, problems, max_new=256, batch=1
    )
    baseline_passes = sum(baseline_results.values())
    baseline_failures = len(problems) - baseline_passes
    print(f"Baseline: {baseline_passes}/{len(problems)} passes ({baseline_passes/len(problems):.1%})", flush=True)
    print(f"Failures to analyze: {baseline_failures}", flush=True)

    if args.baseline_only or baseline_failures == 0:
        return

    print(f"\nAnalyzing {baseline_failures} failures with DeepSeek teacher...", flush=True)
    teacher = DeepSeekTeacher(api_key_path=args.api_key_path)
    run_analysis(teacher, problems, baseline_results, generated_outputs,
                 args.model, output_dir)


if __name__ == "__main__":
    main()
