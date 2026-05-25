"""Command line entry points for the owned cartridge harness."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from hybrid.cartridge_harness.private_facts import build_private_fact_tasks
from hybrid.cartridge_harness.qwen import QwenAdapterCartridgeRunner, split_tasks, train_answer_cartridge


def main() -> int:
    parser = argparse.ArgumentParser(description="Run cartridge self-improvement harness experiments.")
    sub = parser.add_subparsers(dest="command", required=True)

    private_facts = sub.add_parser("private-facts", help="Train/evaluate a private-fact adapter cartridge.")
    private_facts.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    private_facts.add_argument("--device", default="cuda")
    private_facts.add_argument("--steps", type=int, default=700)
    private_facts.add_argument("--eval-every", type=int, default=50)
    private_facts.add_argument("--bottleneck", type=int, default=64)
    private_facts.add_argument("--out-dir", default="artifacts/owned_private_fact_cartridge")

    args = parser.parse_args()
    if args.command == "private-facts":
        tasks = build_private_fact_tasks()
        train_tasks, eval_tasks = split_tasks(tasks)
        runner = QwenAdapterCartridgeRunner(args.model, args.device, args.bottleneck)
        try:
            result = train_answer_cartridge(
                runner=runner,
                train_tasks=train_tasks,
                eval_tasks=eval_tasks,
                out_dir=Path(args.out_dir),
                steps=args.steps,
                eval_every=args.eval_every,
            )
        finally:
            runner.cleanup()
        print(json.dumps({
            "artifact": result["artifact"],
            "baseline_summary": result["baseline_summary"],
            "cartridge_summary": result["cartridge_summary"],
            "improved_count": len(result["improved"]),
            "regressions": len(result["regressed"]),
        }, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())