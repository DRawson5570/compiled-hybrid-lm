"""Export website-ready benchmark comparisons for CMI cartridge demos."""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


@dataclass(frozen=True)
class BenchmarkRow:
    id: str
    label: str
    metric_name: str
    metric_value: float
    metric_unit: str
    lower_is_better: bool
    description: str
    active_cartridges: list[str]


@dataclass(frozen=True)
class BenchmarkDemoPayload:
    demo_id: str
    title: str
    subtitle: str
    benchmark: str
    split: str
    tokenizer: str
    checkpoint: str
    epoch: int | None
    rows: list[BenchmarkRow]
    improvement: dict[str, float]
    sanity: dict[str, Any]
    notes: list[str]

    def to_json(self) -> dict[str, Any]:
        data = asdict(self)
        data['rows'] = [asdict(row) for row in self.rows]
        return data


def safe_float(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'checkpoint is missing numeric {name!r}') from exc
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f'checkpoint {name!r} must be positive and finite, got {result!r}')
    return result


def model_parameter_count(state_dict: dict[str, torch.Tensor] | None) -> int | None:
    if not state_dict:
        return None
    return int(sum(tensor.numel() for tensor in state_dict.values() if torch.is_tensor(tensor)))


def build_payload(
    checkpoint_path: Path,
    checkpoint: dict[str, Any],
    *,
    benchmark: str,
    split: str,
    tokenizer: str,
    demo_id: str,
    title: str,
    subtitle: str,
) -> BenchmarkDemoPayload:
    eval_b = safe_float(checkpoint.get('eval_b'), 'eval_b')
    eval_s = safe_float(checkpoint.get('eval_s'), 'eval_s')
    epoch = checkpoint.get('epoch')
    epoch_int = int(epoch) if epoch is not None else None
    improvement_abs = eval_b - eval_s
    improvement_pct = improvement_abs / eval_b * 100.0
    ratio = eval_b / eval_s
    state_params = model_parameter_count(checkpoint.get('state_dict'))
    steerer_params = model_parameter_count(checkpoint.get('steerer_state'))
    cartridge_improves = eval_s < eval_b

    rows = [
        BenchmarkRow(
            id='compiled_hybrid_baseline',
            label='Compiled Hybrid Baseline',
            metric_name='perplexity',
            metric_value=round(eval_b, 4),
            metric_unit='PPL',
            lower_is_better=True,
            description='Same neural checkpoint with cartridge injection disabled.',
            active_cartridges=[],
        ),
        BenchmarkRow(
            id='cartridge_injection',
            label='Cartridge Injection Active',
            metric_name='perplexity',
            metric_value=round(eval_s, 4),
            metric_unit='PPL',
            lower_is_better=True,
            description='Same checkpoint with the compiled-prior steering cartridge active.',
            active_cartridges=['superposition-steerer-v3'],
        ),
    ]
    notes = [
        'Lower perplexity is better.',
        'Both rows are measured from the same checkpoint; only cartridge injection changes.',
    ]
    if state_params is not None:
        notes.append(f'Backbone parameters in checkpoint: {state_params:,}.')
    if steerer_params is not None:
        notes.append(f'Cartridge parameters in checkpoint: {steerer_params:,}.')

    return BenchmarkDemoPayload(
        demo_id=demo_id,
        title=title,
        subtitle=subtitle,
        benchmark=benchmark,
        split=split,
        tokenizer=tokenizer,
        checkpoint=str(checkpoint_path),
        epoch=epoch_int,
        rows=rows,
        improvement={
            'absolute_ppl': round(improvement_abs, 4),
            'relative_percent': round(improvement_pct, 2),
            'baseline_to_cartridge_ratio': round(ratio, 4),
        },
        sanity={
            'cartridge_improves_baseline': cartridge_improves,
            'eval_s_less_than_eval_b': cartridge_improves,
            'message': 'pass' if cartridge_improves else 'warning: cartridge PPL is not lower than baseline PPL',
        },
        notes=notes,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default='artifacts/steerer_v4/steerer_best_s.pt')
    parser.add_argument('--out', default='artifacts/web_demo_compiled_hybrid_benchmark.json')
    parser.add_argument('--benchmark', default='WikiText-103')
    parser.add_argument('--split', default='validation')
    parser.add_argument('--tokenizer', default='GPT-2 BPE')
    parser.add_argument('--demo-id', default='compiled-hybrid-wikitext-cartridge')
    parser.add_argument('--title', default='Compiled Hybrid Baseline vs Cartridge Injection')
    parser.add_argument('--subtitle', default='Popular language-modeling benchmark comparison using the same checkpoint with cartridge injection disabled vs active.')
    parser.add_argument('--allow-non-improving', action='store_true')
    args = parser.parse_args()

    checkpoint_path = REPO / args.checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    payload = build_payload(
        Path(args.checkpoint),
        checkpoint,
        benchmark=args.benchmark,
        split=args.split,
        tokenizer=args.tokenizer,
        demo_id=args.demo_id,
        title=args.title,
        subtitle=args.subtitle,
    )
    if not payload.sanity['cartridge_improves_baseline'] and not args.allow_non_improving:
        raise SystemExit(payload.sanity['message'])

    out_path = REPO / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload.to_json(), indent=2), encoding='utf-8')
    print(json.dumps(payload.to_json(), indent=2), flush=True)


if __name__ == '__main__':
    main()
