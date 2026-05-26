"""ARC-Challenge benchmark evaluation harness.

Usage:
    .venv/bin/python -m hybrid.benchmarks.arc eval --mode hf-causal --report-dir artifacts/arc_baseline
    .venv/bin/python -m hybrid.benchmarks.arc eval --mode hf-causal --dry-run
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from hybrid.benchmarks.arc_data import ARCExample, load_arc_dataset
from hybrid.benchmarks.arc_prompts import PromptTemplate, get_template
from hybrid.benchmarks.arc_reports import write_reports

import json as _json_mod
from hybrid.benchmarks.arc_scoring import ChoiceScore, HFArcScorer, ScoredExample


def _resolve_device(device_arg: str) -> str:
    if device_arg == "cuda":
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_arg


def _run_eval_hf_causal(
    examples: list[ARCExample],
    template: PromptTemplate,
    model_name: str,
    device: str,
    dtype_str: str,
    report_dir: Path | None,
    batch_size: int,
    log_every: int,
) -> list[ScoredExample]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch_device = torch.device(device)
    torch_dtype = torch.float16 if dtype_str == "float16" and torch_device.type == "cuda" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
    ).to(torch_device)
    model.eval()

    scorer = HFArcScorer(model, tokenizer, torch_device, dtype=torch_dtype)
    scored: list[ScoredExample] = []

    t_start = time.perf_counter()
    for i, example in enumerate(examples):
        se = scorer.score_example(example, template)
        scored.append(se)
        if (i + 1) % log_every == 0 or i == len(examples) - 1:
            correct = sum(1 for s in scored if s.correct_norm)
            print(
                f"  [{i + 1}/{len(examples)}] acc_norm={correct / (i + 1):.4f} "
                f"({(time.perf_counter() - t_start):.1f}s)",
                flush=True,
            )

    if report_dir:
        meta = {
            "config": examples[0].config if examples else "ARC-Challenge",
            "dataset": "allenai/ai2_arc",
            "split": examples[0].split if examples else "validation",
            "model": model_name,
            "mode": "hf-causal",
            "prompt_template": template.template_id,
            "prompt_template_sha256": template.hash(),
            "duration_sec": time.perf_counter() - t_start,
            "started_at": "",
        }
        summary = write_reports(report_dir, scored, 0, meta)
        print(f"  summary: {_json_mod.dumps(summary, indent=2)}", flush=True)

    return scored


def _run_eval_qwen_rack(
    examples: list[ARCExample],
    template: PromptTemplate,
    model_name: str,
    device: str,
    report_dir: Path | None,
    out_dir: str | None,
    router_path: str | None,
    composition_mode: str,
    log_every: int,
    force_cartridge_id: str | None = None,
    force_route: str | None = None,
    arc_cartridge_path: str | None = None,
) -> list[ScoredExample]:
    import torch
    from hybrid.cartridge_harness.qwen import QwenCartridgeRuntime

    torch_device = torch.device(device)

    runtime = QwenCartridgeRuntime(model_name, device=device)
    if out_dir:
        outline = Path(out_dir)
        if outline.exists():
            cartridge_patterns = [
                "cartridge_best.pt",
                "best_cartridge.pt",
                "cartridge.pt",
            ]
            for cart_path in sorted(outline.iterdir()):
                if cart_path.is_file() and cart_path.name in cartridge_patterns:
                    runtime.load_cartridge(cart_path)
            for subdir in sorted(outline.iterdir()):
                if subdir.is_dir():
                    for cart_path in sorted(subdir.iterdir()):
                        if cart_path.is_file() and cart_path.name in cartridge_patterns:
                            runtime.load_cartridge(cart_path)
        else:
            print(f"  rack out-dir not found: {out_dir}", flush=True)
    if arc_cartridge_path:
        arc_path = Path(arc_cartridge_path)
        if arc_path.exists() and arc_path.is_file():
            runtime.load_cartridge(arc_path)
    if router_path:
        router = Path(router_path)
        if router.exists():
            runtime.load_prompt_router(router_path)
        else:
            print(f"  router path not found: {router_path}", flush=True)
    runtime.set_composition_mode(composition_mode)

    routing_mode = "learned"
    if force_route is not None:
        routing_mode = "forced"
    elif force_cartridge_id is not None:
        routing_mode = "forced"

    tokenizer = runtime.tokenizer
    model = runtime.hf_model

    scorer = HFArcScorer(model, tokenizer, torch_device)
    scored: list[ScoredExample] = []
    route_trace: list[dict] = []
    route_counts: dict[str, int] = {}

    t_start = time.perf_counter()
    for i, example in enumerate(examples):
        cartridge_id = None
        prompt = template.render_prompt(example)

        if force_route == "none":
            runtime.activate_only(None)
            route = "none"
        elif force_cartridge_id is not None:
            runtime.activate_only(force_cartridge_id)
            route = force_cartridge_id
        else:
            route = None
            try:
                cartridge_id = runtime.route_prompt(prompt)
                route = cartridge_id
            except Exception:
                pass
            if cartridge_id and cartridge_id != "none":
                runtime.activate_only(cartridge_id)
            else:
                runtime.activate_only(None)
                route = "none"

        route_counts[route or "none"] = route_counts.get(route or "none", 0) + 1

        with torch.no_grad():
            se = scorer.score_example(example, template)
        scored.append(se)

        route_trace.append({
            "id": example.id,
            "route": route,
            "selected_cartridge_id": route if route and route != "none" else None,
            "cartridge_activated": route is not None and route != "none",
        })

        if (i + 1) % log_every == 0 or i == len(examples) - 1:
            correct = sum(1 for s in scored if s.correct_norm)
            print(
                f"  [{i + 1}/{len(examples)}] acc_norm={correct / (i + 1):.4f} "
                f"({(time.perf_counter() - t_start):.1f}s)",
                flush=True,
            )

    runtime.cleanup()

    if report_dir:
        route_trace_path = report_dir / "route_trace.jsonl"
        import json as _j
        route_trace_path.write_text(
            "\n".join(_j.dumps(rr) for rr in route_trace) + "\n", encoding="utf-8"
        )

        meta = {
            "config": examples[0].config if examples else "ARC-Challenge",
            "dataset": "allenai/ai2_arc",
            "split": examples[0].split if examples else "validation",
            "model": model_name,
            "mode": "qwen-rack",
            "routing_mode": routing_mode,
            "prompt_template": template.template_id,
            "prompt_template_sha256": template.hash(),
            "router_path": router_path or "",
            "router_type": "qwen_embedding_linear_v1" if router_path else "qwen_prompt_fallback",
            "composition_mode": composition_mode,
            "mounted_cartridge_count": len(runtime.loaded),
            "mounted_cartridges": sorted(runtime.loaded.keys()),
            "route_counts": route_counts,
            "duration_sec": time.perf_counter() - t_start,
            "started_at": "",
        }
        summary = write_reports(report_dir, scored, 0, meta)
        print(f"  summary: {_json_mod.dumps(summary, indent=2)}", flush=True)

    return scored


def _run_eval_single_cartridge(
    examples: list[ARCExample],
    template: PromptTemplate,
    model_name: str,
    device: str,
    cartridge_path: str,
    report_dir: Path | None,
    log_every: int,
) -> list[ScoredExample]:
    import torch
    from hybrid.cartridge_harness.qwen import QwenCartridgeRuntime

    torch_device = torch.device(device)

    runtime = QwenCartridgeRuntime(model_name, device=device)
    runtime.load_cartridge(cartridge_path)
    runtime.activate_only(list(runtime.loaded.keys())[0])

    tokenizer = runtime.tokenizer
    model = runtime.hf_model

    scorer = HFArcScorer(model, tokenizer, torch_device)
    scored: list[ScoredExample] = []

    t_start = time.perf_counter()
    for i, example in enumerate(examples):
        with torch.no_grad():
            se = scorer.score_example(example, template)
        scored.append(se)

        if (i + 1) % log_every == 0 or i == len(examples) - 1:
            correct = sum(1 for s in scored if s.correct_norm)
            print(
                f"  [{i + 1}/{len(examples)}] acc_norm={correct / (i + 1):.4f} "
                f"({(time.perf_counter() - t_start):.1f}s)",
                flush=True,
            )

    runtime.cleanup()

    if report_dir:
        meta = {
            "config": examples[0].config if examples else "ARC-Challenge",
            "dataset": "allenai/ai2_arc",
            "split": examples[0].split if examples else "validation",
            "model": model_name,
            "mode": "qwen-single-cartridge",
            "prompt_template": template.template_id,
            "prompt_template_sha256": template.hash(),
            "cartridge_path": cartridge_path,
            "cartridge_id": sorted(runtime.loaded.keys())[0] if runtime.loaded else "unknown",
            "duration_sec": time.perf_counter() - t_start,
            "started_at": "",
        }
        summary = write_reports(report_dir, scored, 0, meta)
        print(f"  summary: {_json_mod.dumps(summary, indent=2)}", flush=True)

    return scored


def _run_eval_baked_lora(
    examples: list[ARCExample],
    template: PromptTemplate,
    adapter_dir: str,
    device: str,
    report_dir: Path | None,
    log_every: int,
) -> list[ScoredExample]:
    import torch
    from hybrid.cartridge_harness.qwen import QwenBakedLoraRunner

    torch_device = torch.device(device)

    runner = QwenBakedLoraRunner.from_adapter(adapter_dir, device=device)
    runner.model.eval()

    tokenizer = runner.tokenizer
    model = runner.model

    scorer = HFArcScorer(model, tokenizer, torch_device)
    scored: list[ScoredExample] = []

    t_start = time.perf_counter()
    for i, example in enumerate(examples):
        with torch.no_grad():
            se = scorer.score_example(example, template)
        scored.append(se)

        if (i + 1) % log_every == 0 or i == len(examples) - 1:
            correct = sum(1 for s in scored if s.correct_norm)
            print(
                f"  [{i + 1}/{len(examples)}] acc_norm={correct / (i + 1):.4f} "
                f"({(time.perf_counter() - t_start):.1f}s)",
                flush=True,
            )

    if report_dir:
        meta = {
            "config": examples[0].config if examples else "ARC-Challenge",
            "dataset": "allenai/ai2_arc",
            "split": examples[0].split if examples else "validation",
            "model": runner.model_name,
            "mode": "baked-lora",
            "prompt_template": template.template_id,
            "prompt_template_sha256": template.hash(),
            "adapter_dir": str(adapter_dir),
            "duration_sec": time.perf_counter() - t_start,
            "started_at": "",
        }
        summary = write_reports(report_dir, scored, 0, meta)
        print(f"  summary: {_json_mod.dumps(summary, indent=2)}", flush=True)

    return scored


def cmd_eval(args: argparse.Namespace) -> int:
    device = _resolve_device(args.device)
    print(f"Device: {device}", flush=True)

    valid, invalid, invalid_raw = load_arc_dataset(
        dataset_name=args.dataset,
        config=args.config,
        split=args.split,
        local_jsonl=args.local_jsonl,
        max_examples=args.max_examples,
        strict_data=args.strict_data,
    )

    if invalid:
        print(f"Invalid examples skipped: {len(invalid)}", flush=True)
        for ex in invalid[:5]:
            print(f"  - {ex.id}: {ex.is_valid()}", flush=True)

    if not valid:
        print("No valid examples to score.", flush=True)
        return 0

    print(f"Loaded {len(valid)} valid examples from {args.config}/{args.split}", flush=True)

    template = get_template(args.prompt_template)

    if args.dry_run:
        ex = valid[0]
        print(f"\n--- Dry-run first example ({ex.id}) ---", flush=True)
        print(f"Prompt:\n{template.render_prompt(ex)}", flush=True)
        for c in ex.choices:
            cont = template.render_continuation(c.text)
            print(f"  Continuation for {c.label}: {cont!r}", flush=True)
        print(f"\nAnswer key: {ex.answer_key}", flush=True)
        return 0

    report_dir = Path(args.report_dir) if args.report_dir else None
    if report_dir:
        report_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "hf-causal":
        scored = _run_eval_hf_causal(
            valid, template, args.model, device, args.dtype,
            report_dir, args.batch_size, args.log_every,
        )
    elif args.mode == "qwen-rack":
        scored = _run_eval_qwen_rack(
            valid, template, args.model, device, report_dir,
            args.out_dir, args.router_path, args.composition_mode,
            args.log_every,
            force_cartridge_id=args.force_cartridge_id,
            force_route=args.force_route,
            arc_cartridge_path=args.arc_cartridge_path,
        )
    elif args.mode == "qwen-single-cartridge":
        if not args.cartridge_path:
            print("ERROR: --cartridge-path required for mode qwen-single-cartridge", flush=True)
            return 1
        scored = _run_eval_single_cartridge(
            valid, template, args.model, device,
            args.cartridge_path, report_dir, args.log_every,
        )
    elif args.mode == "baked-lora":
        if not args.adapter_dir:
            print("ERROR: --adapter-dir required for mode baked-lora", flush=True)
            return 1
        scored = _run_eval_baked_lora(
            valid, template, args.adapter_dir, device,
            report_dir, args.log_every,
        )
    else:
        print(f"ERROR: unknown mode {args.mode!r}", flush=True)
        return 1

    correct = sum(1 for s in scored if s.correct_norm)
    total_with_answers = sum(1 for s in scored if s.example.answer_key is not None)
    print(
        f"\nFinal: accuracy_norm = {correct}/{total_with_answers} = "
        f"{correct / max(total_with_answers, 1):.4f}",
        flush=True,
    )

    return 0


def cmd_train(args: argparse.Namespace) -> int:
    import torch
    from hybrid.benchmarks.arc_train import train_arc_cartridge
    from hybrid.cartridge_harness.qwen import QwenAdapterCartridgeRunner

    device = _resolve_device(args.device)
    print(f"Device: {device}", flush=True)

    train_examples, invalid_train, _ = load_arc_dataset(
        dataset_name=args.dataset,
        config=args.config,
        split=args.train_split,
        local_jsonl=args.local_jsonl,
        strict_data=args.strict_data,
    )
    if invalid_train:
        print(f"Invalid train examples skipped: {len(invalid_train)}", flush=True)
    if not train_examples:
        print("No valid train examples.", flush=True)
        return 1
    print(f"Train examples: {len(train_examples)}", flush=True)

    val_examples, invalid_val, _ = load_arc_dataset(
        dataset_name=args.dataset,
        config=args.config,
        split=args.val_split,
        local_jsonl=args.local_jsonl,
        strict_data=args.strict_data,
    )
    if invalid_val:
        print(f"Invalid val examples skipped: {len(invalid_val)}", flush=True)
    print(f"Val examples: {len(val_examples)}", flush=True)

    runner = QwenAdapterCartridgeRunner(
        args.model,
        device=device,
        bottleneck=args.bottleneck,
        cartridge_id=f"{args.config.lower().replace('-', '_')}_{get_template(args.prompt_template).hash()[:8]}",
    )

    report = train_arc_cartridge(
        runner,
        train_examples,
        val_examples,
        Path(args.out_dir),
        template_id=args.prompt_template,
        steps=args.steps,
        lr=args.lr,
        eval_every=args.eval_every,
        temperature=args.temperature,
        lambda_margin=args.lambda_margin,
        seed=args.seed,
    )
    print(f"\nResult: best_val_acc={report['best_val_accuracy']:.4f} "
          f"final_val_acc={report['final_val_accuracy']:.4f}", flush=True)
    runner.cleanup()
    return 0


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="ARC-Challenge benchmark evaluation harness",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    eval_p = sub.add_parser("eval", help="Run ARC evaluation")
    eval_p.add_argument("--dataset", default="allenai/ai2_arc")
    eval_p.add_argument("--config", default="ARC-Challenge", choices=["ARC-Challenge", "ARC-Easy"])
    eval_p.add_argument("--split", default="validation")
    eval_p.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    eval_p.add_argument("--device", default="cuda")
    eval_p.add_argument("--dtype", default="float16")
    eval_p.add_argument("--mode", default="hf-causal",
                        choices=["hf-causal", "qwen-rack", "qwen-single-cartridge", "baked-lora"])
    eval_p.add_argument("--prompt-template", default="arc_v1")
    eval_p.add_argument("--max-examples", type=int, default=0)
    eval_p.add_argument("--seed", type=int, default=1337)
    eval_p.add_argument("--report-dir", default=None)
    eval_p.add_argument("--local-jsonl", default=None)
    eval_p.add_argument("--batch-size", type=int, default=1)
    eval_p.add_argument("--log-every", type=int, default=25)
    eval_p.add_argument("--dry-run", action="store_true")
    eval_p.add_argument("--strict-data", action="store_true")
    eval_p.add_argument("--limit-answer-tokens", type=int, default=None)
    eval_p.add_argument("--cache-dir", default=None)
    eval_p.add_argument("--out-dir", default=None,
                        help="Cartridge rack directory for qwen-rack mode")
    eval_p.add_argument("--router-path", default=None,
                        help="Learned router path for qwen-rack mode")
    eval_p.add_argument("--composition-mode", default="chain",
                        help="Composition mode for qwen-rack mode (additive, mean, chain)")
    eval_p.add_argument("--cartridge-path", default=None,
                        help="Single cartridge path for qwen-single-cartridge mode")
    eval_p.add_argument("--adapter-dir", default=None,
                        help="Baked LoRA adapter dir for baked-lora mode")
    eval_p.add_argument("--force-cartridge-id", default=None,
                        help="Diagnostic: force a specific cartridge id instead of router")
    eval_p.add_argument("--force-route", default=None,
                        help="Diagnostic: force a route (e.g. 'none') for no-op test")
    eval_p.add_argument("--arc-cartridge-path", default=None,
                        help="Path to ARC cartridge.pt for mounting in the rack")

    train_p = sub.add_parser("train-cartridge", help="Train an ARC cartridge with option-ranking loss")
    train_p.add_argument("--config", default="ARC-Challenge")
    train_p.add_argument("--train-split", default="train")
    train_p.add_argument("--val-split", default="validation")
    train_p.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    train_p.add_argument("--device", default="cuda")
    train_p.add_argument("--out-dir", required=True)
    train_p.add_argument("--steps", type=int, default=500)
    train_p.add_argument("--lr", type=float, default=2e-4)
    train_p.add_argument("--eval-every", type=int, default=50)
    train_p.add_argument("--temperature", type=float, default=1.0)
    train_p.add_argument("--lambda-margin", type=float, default=0.0)
    train_p.add_argument("--bottleneck", type=int, default=64)
    train_p.add_argument("--prompt-template", default="arc_v1")
    train_p.add_argument("--seed", type=int, default=23)
    train_p.add_argument("--dataset", default="allenai/ai2_arc")
    train_p.add_argument("--local-jsonl", default=None)
    train_p.add_argument("--strict-data", action="store_true")
    train_p.add_argument("--dtype", default="float16")

    return parser


def main() -> int:
    parser = _build_arg_parser()
    args = parser.parse_args()

    if args.command == "eval":
        return cmd_eval(args)
    elif args.command == "train-cartridge":
        return cmd_train(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
