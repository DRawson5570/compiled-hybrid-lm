"""Command line entry points for the owned cartridge harness."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from hybrid.cartridge_harness.private_facts import build_private_fact_tasks
from hybrid.cartridge_harness.core import build_summary, compare_rows, evaluate_text_runner
from hybrid.cartridge_harness.qwen import (
    QwenAdapterCartridgeRunner,
    QwenCartridgeRuntime,
    split_tasks,
    train_qwen_baked_lora,
    train_qwen_embedding_router,
    train_answer_cartridge,
)
from hybrid.cartridge_harness.rack_builder import assemble_rack_summary, build_rack
from hybrid.cartridge_harness.suites import build_all_suites, get_suite


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

    loaded = sub.add_parser("eval-loaded-rack", help="Load saved Qwen cartridges and test routed individual behavior.")
    loaded.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    loaded.add_argument("--device", default="cuda")
    loaded.add_argument("--out-dir", default="artifacts/qwen_cartridge_rack")
    loaded.add_argument("--suite", action="append", dest="suites")
    loaded.add_argument("--max-tokens", type=int, default=24)
    loaded.add_argument("--report", help="Optional JSON report path. Defaults to OUT_DIR/loaded_rack_eval.json.")
    loaded.add_argument("--skip-baseline", action="store_true", help="Only evaluate active cartridges against their saved expected scores.")
    loaded.add_argument("--allow-individual-regressions", action="store_true")
    loaded.add_argument("--composition-mode", choices=("routed", "gated-chain", "additive", "mean", "chain"), default="gated-chain")
    loaded.add_argument("--test-all-active", action="store_true", help="Also diagnose naive all-active composition. Task cartridges are expected to be routed, not blindly summed.")
    loaded.add_argument("--router-path", help="Optional learned Qwen router artifact to use for routed/gated-chain evaluation.")

    router = sub.add_parser("train-router", help="Train a learned Qwen embedding router for mounted cartridges.")
    router.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    router.add_argument("--device", default="cuda")
    router.add_argument("--out-dir", default="artifacts/qwen_cartridge_router")
    router.add_argument("--epochs", type=int, default=300)
    router.add_argument("--lr", type=float, default=3e-3)
    router.add_argument("--confidence-threshold", type=float, default=0.0)
    router.add_argument("--ambiguous-margin", type=float, default=0.0)

    baked = sub.add_parser("train-baked-lora", help="Train a Qwen LoRA adapter that bakes suite behavior into the model.")
    baked.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    baked.add_argument("--device", default="cuda")
    baked.add_argument("--out-dir", default="artifacts/qwen_baked_lora")
    baked.add_argument("--steps", type=int, default=600)
    baked.add_argument("--eval-every", type=int, default=100)
    baked.add_argument("--lr", type=float, default=2e-4)
    baked.add_argument("--lora-r", type=int, default=16)
    baked.add_argument("--lora-alpha", type=int, default=32)
    baked.add_argument("--lora-dropout", type=float, default=0.05)

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
    elif args.command == "eval-loaded-rack":
        selected = build_all_suites() if not args.suites else [get_suite(suite_id) for suite_id in args.suites]
        runtime = QwenCartridgeRuntime(args.model, args.device)
        try:
            if args.router_path:
                runtime.load_prompt_router(args.router_path)
            loaded_by_suite = {}
            loaded_summaries = {}
            for suite in selected:
                loaded = runtime.load_cartridge(Path(args.out_dir) / suite.suite_id / "cartridge_best.pt", active=False)
                loaded_by_suite[suite.suite_id] = loaded.manifest.cartridge_id
                loaded_summaries[suite.suite_id] = loaded.summary

            if args.composition_mode in {"additive", "mean", "chain"}:
                runtime.set_composition_mode(args.composition_mode)
                runtime.set_all_active(True)
            report = {
                "model": args.model,
                "out_dir": args.out_dir,
                "composition_mode": args.composition_mode,
                "router_path": args.router_path,
                "suites": [],
            }
            for suite in selected:
                if args.composition_mode == "gated-chain":
                    runner = lambda prompt: runtime.generate_gated_chain(prompt, max_tokens=args.max_tokens)
                elif args.composition_mode == "routed":
                    runner = lambda prompt: runtime.generate_routed(prompt, max_tokens=args.max_tokens)
                else:
                    runner = lambda prompt: runtime.generate(prompt, max_tokens=args.max_tokens)
                baseline_rows = []
                baseline_summary = None
                if not args.skip_baseline:
                    runtime.activate_only(None)
                    baseline_rows = evaluate_text_runner(suite.tasks, runner)
                    baseline_summary = build_summary(baseline_rows)

                if args.composition_mode in {"routed", "gated-chain"}:
                    expected_route = loaded_by_suite[suite.suite_id]
                    bad_routes = [
                        task.task_id for task in suite.tasks
                        if runtime.route_prompt(task.prompt) != expected_route
                    ]
                    if bad_routes:
                        raise SystemExit(f"router mismatch for {suite.suite_id}: {bad_routes[:5]}")
                else:
                    runtime.set_all_active(True)
                individual_rows = evaluate_text_runner(suite.tasks, runner)
                individual_summary = build_summary(individual_rows)

                individual_cmp = compare_rows(baseline_rows, individual_rows) if baseline_rows else {"improved": [], "regressed": []}
                loaded_correct = loaded_summaries[suite.suite_id]["correct"]
                item = {
                    "suite_id": suite.suite_id,
                    "cartridge_id": loaded_by_suite[suite.suite_id],
                    "loaded_summary": loaded_summaries[suite.suite_id],
                    "baseline_summary": baseline_summary.to_json() if baseline_summary else None,
                    "individual_summary": individual_summary.to_json(),
                    "individual_improved_count": len(individual_cmp["improved"]),
                    "individual_regression_count": len(individual_cmp["regressed"]),
                    "saved_score_regression": individual_summary.correct < loaded_correct,
                }
                if args.test_all_active:
                    runtime.set_all_active(True)
                    all_active_rows = evaluate_text_runner(suite.tasks, runner)
                    all_active_summary = build_summary(all_active_rows)
                    all_active_cmp = compare_rows(individual_rows, all_active_rows)
                    item["all_active_summary"] = all_active_summary.to_json()
                    item["all_active_regression_count"] = len(all_active_cmp["regressed"])
                report["suites"].append(item)
                print(
                    f"[loaded] {suite.suite_id} individual="
                    f"{individual_summary.correct}/{individual_summary.total} "
                    f"saved={loaded_correct}/{loaded_summaries[suite.suite_id]['total']} "
                    f"inactive_base="
                    f"{baseline_summary.correct if baseline_summary else 'skipped'}/"
                    f"{baseline_summary.total if baseline_summary else 'skipped'} "
                    f"individual_regressions={len(individual_cmp['regressed'])}",
                    flush=True,
                )
                if not args.allow_individual_regressions and individual_cmp["regressed"]:
                    raise SystemExit(f"individual regression detected for {suite.suite_id}")
                if individual_summary.correct < loaded_correct:
                    raise SystemExit(f"saved-score regression detected for {suite.suite_id}")
            report_path = Path(args.report) if args.report else Path(args.out_dir) / "loaded_rack_eval.json"
            report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
            print(json.dumps(report, indent=2), flush=True)
        finally:
            runtime.cleanup()
    elif args.command == "train-router":
        report = train_qwen_embedding_router(
            model_name=args.model,
            device=args.device,
            out_dir=args.out_dir,
            epochs=args.epochs,
            lr=args.lr,
            confidence_threshold=args.confidence_threshold,
            ambiguous_margin=args.ambiguous_margin,
        )
        print(json.dumps(report, indent=2), flush=True)
    elif args.command == "train-baked-lora":
        report = train_qwen_baked_lora(
            model_name=args.model,
            device=args.device,
            out_dir=args.out_dir,
            steps=args.steps,
            eval_every=args.eval_every,
            lr=args.lr,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
        )
        print(json.dumps({k: v for k, v in report.items() if k != "final_rows"}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())