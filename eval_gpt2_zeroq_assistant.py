"""Evaluate baseline GPT-2-family generation beside a ZeroQ cartridge assistant."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from hybrid.assistant_eval import DEFAULT_TASKS, score_answer, summarize
from hybrid.gpt2_zeroq_assistant import GPT2ZeroQAssistantRuntime


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-name', default='gpt2-large')
    parser.add_argument('--cartridge', required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--zeroq-path', default='~/ZeroQ')
    parser.add_argument('--task-limit', type=int, default=0)
    parser.add_argument('--max-new-tokens', type=int, default=140)
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--report', default='artifacts/gpt2_zeroq_assistant_eval.json')
    args = parser.parse_args()

    torch.manual_seed(20260525)
    runtime = GPT2ZeroQAssistantRuntime(
        model_name=args.model_name,
        cartridge=args.cartridge,
        device=args.device,
        zeroq_path=args.zeroq_path,
    )
    tasks = DEFAULT_TASKS[:args.task_limit] if args.task_limit > 0 else DEFAULT_TASKS
    baseline_rows = []
    cartridge_rows = []
    side_by_side = []
    for task in tasks:
        baseline = runtime.generate(
            task.prompt,
            history=list(task.history),
            use_cartridge=False,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            max_sentences=0,
        )
        cartridge = runtime.generate(
            task.prompt,
            history=list(task.history),
            use_cartridge=True,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            max_sentences=0,
        )
        baseline_row = score_answer(task, baseline)
        cartridge_row = score_answer(task, cartridge)
        baseline_rows.append(baseline_row)
        cartridge_rows.append(cartridge_row)
        side_by_side.append({
            'task_id': task.task_id,
            'category': task.category,
            'prompt': task.prompt,
            'baseline': baseline_row.to_json(),
            'cartridge': cartridge_row.to_json(),
        })
        status = 'PASS' if cartridge_row.passed else 'FAIL'
        print(f'[{status}] {task.task_id}: {cartridge}', flush=True)
        if cartridge_row.failures:
            print(f'  failures={list(cartridge_row.failures)}', flush=True)

    payload = {
        'model_name': args.model_name,
        'cartridge': args.cartridge,
        'baseline_summary': summarize(baseline_rows),
        'cartridge_summary': summarize(cartridge_rows),
        'rows': side_by_side,
    }
    report_path = REPO / args.report
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    print(json.dumps({
        'baseline_summary': payload['baseline_summary'],
        'cartridge_summary': payload['cartridge_summary'],
    }, indent=2), flush=True)
    runtime.cleanup()


if __name__ == '__main__':
    main()
