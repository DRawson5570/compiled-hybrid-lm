from pathlib import Path

import torch

from hybrid.export_benchmark_demo import build_payload, model_parameter_count, safe_float


def test_safe_float_rejects_missing_or_nonpositive():
    assert safe_float('42.5', 'eval_s') == 42.5
    for bad in (None, 0, -1, float('inf')):
        try:
            safe_float(bad, 'eval_s')
        except ValueError:
            pass
        else:
            raise AssertionError(f'expected ValueError for {bad!r}')


def test_model_parameter_count_handles_checkpoint_state():
    state = {
        'a': torch.zeros(2, 3),
        'b': torch.zeros(4),
        'metadata': 'ignored',
    }
    assert model_parameter_count(state) == 10
    assert model_parameter_count(None) is None


def test_build_payload_compares_baseline_and_cartridge():
    payload = build_payload(
        Path('artifacts/demo.pt'),
        {
            'eval_b': 42.1,
            'eval_s': 30.4,
            'epoch': 100,
            'state_dict': {'weight': torch.zeros(2, 5)},
            'steerer_state': {'vectors': torch.zeros(21, 4)},
        },
        benchmark='WikiText-103',
        split='validation',
        tokenizer='GPT-2 BPE',
        demo_id='demo',
        title='Baseline vs Cartridge',
        subtitle='same checkpoint',
    )
    data = payload.to_json()
    assert data['benchmark'] == 'WikiText-103'
    assert data['epoch'] == 100
    assert data['rows'][0]['id'] == 'compiled_hybrid_baseline'
    assert data['rows'][0]['metric_value'] == 42.1
    assert data['rows'][1]['id'] == 'cartridge_injection'
    assert data['rows'][1]['metric_value'] == 30.4
    assert data['improvement']['absolute_ppl'] == 11.7
    assert data['improvement']['relative_percent'] == 27.79
    assert data['sanity']['cartridge_improves_baseline'] is True
    assert any('Backbone parameters' in note for note in data['notes'])
    assert any('Cartridge parameters' in note for note in data['notes'])
