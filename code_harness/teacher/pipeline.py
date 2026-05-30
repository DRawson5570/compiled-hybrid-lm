"""Phase 6: End-to-end teacher-driven code improvement pipeline.

Orchestrates the full pipeline:
  analyze → document → synthesize → train → integrate

Usage:
    python teacher/pipeline.py --target-model Qwen/Qwen3.5-4B

Resume support:
    python teacher/pipeline.py --resume-from WEAKNESS_ID
    python teacher/pipeline.py --skip-analysis
    python teacher/pipeline.py --skip-integration
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

CODE_HARNESS = Path.home() / "code_harness"
TEACHER_DIR = CODE_HARNESS / "teacher"


def run_step(name: str, cmd: list[str], cwd: Path | str | None = None) -> bool:
    print(f"\n{'='*60}")
    print(f"  PHASE: {name}")
    print(f"{'='*60}", flush=True)
    result = subprocess.run(
        [sys.executable] + cmd,
        cwd=str(cwd) if cwd else str(CODE_HARNESS),
    )
    if result.returncode != 0:
        print(f"\n  PHASE FAILED: {name} (exit code {result.returncode})", flush=True)
        return False
    print(f"  PHASE COMPLETE: {name}", flush=True)
    return True


def main():
    ap = argparse.ArgumentParser(description="Teacher-driven code improvement pipeline")
    ap.add_argument("--target-model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--output-dir", default=str(CODE_HARNESS / "artifacts" / "code_improvement_v1"))
    ap.add_argument("--weakness-dir", default=str(CODE_HARNESS / "weaknesses"))
    ap.add_argument("--api-key-path", default=str(Path.home() / "api_keys" / "deepseek"))

    ap.add_argument("--canonical-steps", type=int, default=500)
    ap.add_argument("--rft-steps", type=int, default=500)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--synthesize-count", type=int, default=20)

    ap.add_argument("--skip-analysis", action="store_true",
                    help="Skip baseline+teacher analysis, use cached catalog.json")
    ap.add_argument("--skip-synthesis", action="store_true",
                    help="Skip training data generation, use cached JSONL files")
    ap.add_argument("--skip-training", action="store_true",
                    help="Skip cartridge training, use cached cartridge.pt files")
    ap.add_argument("--skip-integration", action="store_true",
                    help="Skip final regression test")
    ap.add_argument("--resume-from", default=None,
                    help="Weakness ID to start training from (trains this and all subsequent)")
    ap.add_argument("--single-weakness", default=None,
                    help="Train ONLY this one weakness and stop")
    ap.add_argument("--max-weaknesses", type=int, default=0,
                    help="Only train first N weaknesses (0 = all)")
    ap.add_argument("--no-gc", action="store_true",
                    help="Disable gradient checkpointing during training")
    ap.add_argument("--mode", choices=["canonical", "rft", "both"], default="both")

    args = ap.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    weakness_dir = Path(args.weakness_dir)
    cartridge_dir = output_dir / "cartridges"

    catalog_path = weakness_dir / "catalog.json"

    # ── Phase 1: Analyze ────────────────────────────────────────────────
    if not args.skip_analysis:
        if not run_step("1. Analyze — Baseline + Teacher Audit", [
            str(TEACHER_DIR / "analyze.py"),
            "--model", args.target_model,
            "--device", args.device,
            "--api-key-path", args.api_key_path,
            "--output-dir", str(weakness_dir),
        ]):
            return

    if not catalog_path.exists():
        print(f"ERROR: catalog not found at {catalog_path}")
        return

    catalog = json.loads(catalog_path.read_text())
    weaknesses = catalog.get("weaknesses", [])
    print(f"\nCatalog loaded: {len(weaknesses)} weakness(es), "
          f"{catalog.get('total_failures', '?')} failures")

    if args.max_weaknesses > 0:
        weaknesses = weaknesses[:args.max_weaknesses]
        print(f"Limiting to first {args.max_weaknesses} weakness(es)")

    # ── Phase 2: Document ───────────────────────────────────────────────
    if not args.skip_analysis:
        run_step("2. Document — Per-Weakness Reports", [
            str(TEACHER_DIR / "document.py"),
            "--catalog", str(catalog_path),
            "--output-dir", str(weakness_dir),
        ])

    # ── Phase 3: Synthesize ─────────────────────────────────────────────
    if not args.skip_synthesis:
        for w in weaknesses:
            training_file = weakness_dir / f"{w['weakness_id']}_training.jsonl"
            if training_file.exists() and training_file.stat().st_size > 0:
                print(f"  Skipping synthesis for {w['weakness_id']} — training data exists")
                continue
            run_step(f"3. Synthesize — {w['weakness_name']}", [
                str(TEACHER_DIR / "synthesize.py"),
                "--catalog", str(catalog_path),
                "--api-key-path", args.api_key_path,
                "--output-dir", str(weakness_dir),
                "--count", str(args.synthesize_count),
                "--weakness-id", w["weakness_id"],
            ])
            time.sleep(2.0)

    # ── Phase 4: Train ──────────────────────────────────────────────────
    skip_until = args.resume_from
    single_target = args.single_weakness
    gc_flag = ["--no-gc"] if args.no_gc else []
    for w in weaknesses:
        wid = w["weakness_id"]
        if single_target:
            if wid != single_target:
                print(f"  Skipping {wid} (single-weakness mode)")
                continue
        elif skip_until and wid != skip_until:
            print(f"  Skipping {wid} (before resume point)")
            continue
        skip_until = None

        cartridge_file = cartridge_dir / wid / "cartridge_best.pt"
        if args.skip_training and cartridge_file.exists():
            print(f"  Skipping training for {wid} — cartridge exists")
            continue

        run_step(f"4. Train — {w['weakness_name']}", [
            str(TEACHER_DIR / "train_weakness.py"),
            "--weakness-id", wid,
            "--model", args.target_model,
            "--device", args.device,
            "--catalog", str(catalog_path),
            "--training-dir", str(weakness_dir),
            "--output-dir", str(cartridge_dir),
            "--mode", args.mode,
            "--canonical-steps", str(args.canonical_steps),
            "--rft-steps", str(args.rft_steps),
            "--lr", str(args.lr),
        ] + gc_flag)

    # ── Phase 5: Integrate ──────────────────────────────────────────────
    if not args.skip_integration:
        run_step("5. Integration — Full Regression Test", [
            str(TEACHER_DIR / "integrate.py"),
            "--cartridge-dir", str(cartridge_dir),
            "--catalog", str(catalog_path),
            "--model", args.target_model,
            "--device", args.device,
            "--output", str(output_dir / "regression_report.json"),
        ])

    print(f"\n{'='*60}")
    print(f"  PIPELINE COMPLETE")
    print(f"  Output: {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
