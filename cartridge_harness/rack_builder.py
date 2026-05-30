"""Build and evaluate a rack of separate Qwen adapter cartridges."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from hybrid.cartridge_harness.qwen import QwenAdapterCartridgeRunner, split_tasks, train_answer_cartridge
from hybrid.cartridge_harness.suites import build_all_suites, get_suite


def assemble_rack_summary(
    *,
    model: str,
    device: str,
    out_dir: Path,
    suites: list[str] | None,
) -> dict:
    selected = build_all_suites() if not suites else [get_suite(suite_id) for suite_id in suites]
    rack_items: list[dict] = []
    missing: list[str] = []
    for suite in selected:
        summary_path = out_dir / suite.suite_id / "summary.json"
        if not summary_path.exists():
            missing.append(str(summary_path))
            continue
        result = json.loads(summary_path.read_text(encoding="utf-8"))
        rack_items.append({
            "suite": asdict(suite) | {"tasks": len(suite.tasks)},
            "artifact": result["artifact"],
            "baseline_summary": result["baseline_summary"],
            "cartridge_summary": result["cartridge_summary"],
            "improved_count": len(result["improved"]),
            "regression_count": len(result["regressed"]),
        })
    if missing:
        raise FileNotFoundError("missing suite summaries: " + ", ".join(missing))
    summary = {"model": model, "device": device, "items": rack_items}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "rack_manifest.json").write_text(
        json.dumps({"model": model, "device": device, "items": rack_items}, indent=2),
        encoding="utf-8",
    )
    (out_dir / "rack_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def build_rack(
    *,
    model: str,
    device: str,
    bottleneck: int,
    out_dir: Path,
    suites: list[str] | None,
    steps: int,
    eval_every: int,
) -> dict:
    selected = build_all_suites() if not suites else [get_suite(suite_id) for suite_id in suites]
    out_dir.mkdir(parents=True, exist_ok=True)
    rack_items: list[dict] = []
    results: dict[str, dict] = {}
    for suite in selected:
        suite_dir = out_dir / suite.suite_id
        train_tasks, eval_tasks = split_tasks(suite.tasks)
        runner = QwenAdapterCartridgeRunner(
            model,
            device,
            bottleneck,
            cartridge_id=suite.cartridge_id,
            role=suite.role,
            source_corpus=f"hybrid.cartridge_harness.{suite.suite_id}",
        )
        try:
            result = train_answer_cartridge(
                runner=runner,
                train_tasks=train_tasks,
                eval_tasks=eval_tasks,
                out_dir=suite_dir,
                steps=steps,
                eval_every=eval_every,
            )
        finally:
            runner.cleanup()
        results[suite.suite_id] = result
        rack_items.append({
            "suite": asdict(suite) | {"tasks": len(suite.tasks)},
            "artifact": result["artifact"],
            "baseline_summary": result["baseline_summary"],
            "cartridge_summary": result["cartridge_summary"],
            "improved_count": len(result["improved"]),
            "regression_count": len(result["regressed"]),
        })
        (out_dir / "rack_manifest.json").write_text(
            json.dumps({"model": model, "device": device, "items": rack_items}, indent=2),
            encoding="utf-8",
        )
    return assemble_rack_summary(model=model, device=device, out_dir=out_dir, suites=suites)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a rack of separate Qwen adapter cartridges.")
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--bottleneck", type=int, default=64)
    parser.add_argument("--out-dir", default="artifacts/qwen_cartridge_rack")
    parser.add_argument("--suite", action="append", dest="suites", help="Suite id to run; repeat for multiple. Defaults to all suites.")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--assemble-only", action="store_true", help="Read suite summaries and write the rack manifest without training.")
    args = parser.parse_args()
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
