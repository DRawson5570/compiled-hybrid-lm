"""ARC benchmark report generation."""
from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from hybrid.benchmarks.arc_scoring import ChoiceScore, ScoredExample


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _git_info() -> tuple[str | None, bool]:
    sha, dirty = None, True
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain"], text=True, stderr=subprocess.DEVNULL
        ).strip()
        dirty = bool(status)
    except Exception:
        pass
    return sha, dirty


def _score_to_dict(s: ChoiceScore) -> dict:
    return {
        "label": s.label,
        "text": s.text,
        "score_norm": s.score_norm,
        "score_sum": s.score_sum,
        "num_tokens": s.num_tokens,
    }


def write_reports(
    output_dir: Path,
    scored: list[ScoredExample],
    invalid_count: int,
    meta: dict,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    scored_with_answers = [s for s in scored if s.example.answer_key is not None]
    correct_norm = sum(1 for s in scored_with_answers if s.correct_norm)
    correct_sum = sum(1 for s in scored_with_answers if s.correct_sum)
    total = max(len(scored_with_answers), 1)
    acc_norm = correct_norm / total
    acc_sum = correct_sum / total

    correct_margins = [s.margin_norm for s in scored_with_answers if s.correct_norm and s.margin_norm != 0.0]
    answer_tokens = []
    for s in scored:
        for sc in s.scores:
            if sc.num_tokens > 0:
                answer_tokens.append(sc.num_tokens)

    git_sha, git_dirty = _git_info()
    now = _now_iso()

    summary = {
        "benchmark": meta.get("config", "ARC-Challenge"),
        "dataset": meta.get("dataset", "allenai/ai2_arc"),
        "split": meta.get("split", "validation"),
        "model": meta.get("model", ""),
        "mode": meta.get("mode", ""),
        "prompt_template": meta.get("prompt_template", ""),
        "prompt_template_sha256": meta.get("prompt_template_sha256", ""),
        "num_examples_total": len(scored) + invalid_count,
        "num_examples_scored": len(scored),
        "num_examples_invalid": invalid_count,
        "accuracy_norm": round(acc_norm, 6),
        "accuracy_sum": round(acc_sum, 6),
        "mean_correct_score_margin_norm": round(
            sum(correct_margins) / max(len(correct_margins), 1), 6
        ) if correct_margins else None,
        "median_correct_score_margin_norm": round(
            sorted(correct_margins)[len(correct_margins) // 2], 6
        ) if correct_margins else None,
        "mean_answer_tokens": round(
            sum(answer_tokens) / max(len(answer_tokens), 1), 2
        ) if answer_tokens else 0.0,
        "started_at": meta.get("started_at", ""),
        "finished_at": now,
        "duration_sec": meta.get("duration_sec", 0.0),
        "git_sha": git_sha,
        "git_dirty": git_dirty,
    }

    for key in ("router_path", "router_type", "composition_mode", "mounted_cartridges"):
        if key in meta:
            summary[key] = meta[key]

    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    with open(output_dir / "predictions.jsonl", "w", encoding="utf-8") as fh:
        for se in scored:
            row = {
                "id": se.example.id,
                "answer_key": se.example.answer_key,
                "pred_norm": se.pred_norm,
                "pred_sum": se.pred_sum,
                "correct_norm": se.correct_norm,
                "correct_sum": se.correct_sum,
                "scores": [_score_to_dict(s) for s in se.scores],
                "margin_norm": se.margin_norm,
                "question_len_chars": len(se.example.question),
                "choice_count": len(se.example.choices),
            }
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    with open(output_dir / "failures.jsonl", "w", encoding="utf-8") as fh:
        for se in scored:
            if se.correct_norm is not None and not se.correct_norm:
                fh.write(json.dumps({
                    "id": se.example.id,
                    "answer_key": se.example.answer_key,
                    "pred_norm": se.pred_norm,
                    "scores": [_score_to_dict(s) for s in se.scores],
                    "margin_norm": se.margin_norm,
                }, ensure_ascii=False) + "\n")

    env = {
        "hostname": platform.node(),
        "python_version": sys.version.split()[0],
        "torch_version": "unknown",
        "transformers_version": "unknown",
        "cuda_available": False,
        "gpu_name": "not available",
        "command_argv": sys.argv,
        "working_directory": os.getcwd(),
        "git_sha": git_sha,
        "git_dirty": git_dirty,
    }
    try:
        import torch
        env["torch_version"] = torch.__version__
        env["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            env["gpu_name"] = torch.cuda.get_device_name(0)
    except Exception:
        pass
    try:
        import transformers
        env["transformers_version"] = transformers.__version__
    except Exception:
        pass

    (output_dir / "environment.json").write_text(
        json.dumps(env, indent=2) + "\n", encoding="utf-8"
    )

    return summary
