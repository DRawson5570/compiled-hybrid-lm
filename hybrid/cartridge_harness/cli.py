"""Command line entry points for the owned cartridge harness."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from hybrid.cartridge_harness.private_facts import build_private_fact_tasks
from hybrid.cartridge_harness.qwen import QwenAdapterCartridgeRunner, split_tasks, train_answer_cartridge
from hybrid.cartridge_harness.rack_builder import assemble_rack_summary, build_rack


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

    rack = sub.add_parser("rack", help="Train/evaluate a rack of separate Qwen adapter cartridges.")
    rack.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    rack.add_argument("--device", default="cuda")
    rack.add_argument("--steps", type=int, default=300)
    rack.add_argument("--eval-every", type=int, default=50)
    rack.add_argument("--bottleneck", type=int, default=64)
    rack.add_argument("--out-dir", default="artifacts/qwen_cartridge_rack")
    rack.add_argument("--suite", action="append", dest="suites")
    rack.add_argument("--assemble-only", action="store_true")

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
    elif args.command == "rack":
        if args.assemble_only:
            summary = assemble_rack_summary(
                model=args.model,
                device=args.device,
                out_dir=Path(args.out_dir),
                suites=args.suites,
            )
        else:
            summary = build_rack(
                model=args.model,
                device=args.device,
                bottleneck=args.bottleneck,
                out_dir=Path(args.out_dir),
                suites=args.suites,
                steps=args.steps,
                eval_every=args.eval_every,
            )
        print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())