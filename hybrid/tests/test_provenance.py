"""test_provenance.py — Acceptance test for TICKET-005 provenance tracking.

Decodes a 200-token prompt through the WindowMLP blender with provenance
wrapping, verifies sum(contributions) ≈ final_logp within 1e-4, and dumps
the provenance JSON.
"""
from __future__ import annotations

import sys, json, math, importlib.util
from pathlib import Path

import numpy as np
import torch

DEEPSEEK = None
for p in Path(__file__).resolve().parents:
    if p.name == 'deepseek_experiments':
        DEEPSEEK = p
        break
if DEEPSEEK is None:
    DEEPSEEK = Path(__file__).resolve().parents[2]

LLM_DECOUPLING = Path('/home/drawson/llm_decoupling')

# Force deepseek_experiments hybrid to load first — clear any cached imports
for k in list(sys.modules):
    if k.startswith('hybrid') or k.startswith('compile_wiki'):
        del sys.modules[k]
sys.path.insert(0, str(DEEPSEEK))
sys.path.insert(1, str(LLM_DECOUPLING))

from compile_wiki_lm_v13 import load_setup

# Import blender and feature modules directly
def _imp(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_blender_mod = _imp('blender_v3', str(DEEPSEEK / 'hybrid/v3_super_blender/model.py'))
_blender_util = _imp('blender_util', str(DEEPSEEK / 'hybrid/v1_blender/blender_model.py'))
WindowMLPBlender = _blender_mod.WindowMLPBlender
build_feature_matrix = _blender_util.build_feature_matrix

from hybrid.surfaces.provenance import ProvenanceRing, ProvenanceBlender


def test_provenance_200_tokens():
    """Run provenance on a 200-token decode and verify invariants."""
    print('=' * 60)
    print(' TICKET-005: PROVENANCE ACCEPTANCE TEST')
    print('=' * 60)

    # Load data and blender
    print('[1/4] Loading compiled blender and eval data...')
    _bpe, _vocab, _tok2id, _bpe_to_lm, emb, V, d = load_setup()
    emb = emb.float()

    eval_npz = np.load(
        str(DEEPSEEK / 'hybrid/v3_super_blender/data_real_v33/eval.npz'),
        allow_pickle=True
    )
    channel_names = eval_npz['channel_names'].tolist()
    print(f'  {len(channel_names)} channels: {channel_names}')

    eval_features = build_feature_matrix(
        torch.tensor(eval_npz['log_p_observed']),
        torch.tensor(eval_npz['log_p_lag1']),
        torch.tensor(eval_npz['entropy']),
        torch.tensor(eval_npz['max_log_prob']),
        emb, torch.tensor(eval_npz['observed']),
        use_embedding=True
    ).float()

    ckpt = torch.load(
        str(DEEPSEEK / 'hybrid/v3_super_blender/saved_models_v33/blender_window_mlp.pt'),
        map_location='cpu'
    )
    blender = WindowMLPBlender(
        single_step_dim=eval_features.shape[1], n_channels=21,
        lookback_window=16, hidden=256, dropout=0.1, init_uniform=False
    )
    blender.load_state_dict(ckpt['state_dict'])
    blender.eval()
    print(f'  Blender loaded: {sum(p.numel() for p in blender.parameters()):,} params')

    # Set up provenance ring
    print('[2/4] Setting up provenance tracker (ring buffer, capacity=4096)...')
    ring = ProvenanceRing(capacity=4096)
    log_p_targets = torch.tensor(eval_npz['log_p_targets'])
    targets = torch.tensor(eval_npz['targets'].astype(np.int64))

    wrapper = ProvenanceBlender(
        blender, channel_names, log_p_targets, ring, position_offset=0
    )

    # Run forward pass on first 200 tokens
    N = 200
    print(f'[3/4] Running blender forward pass on {N} tokens with provenance...')
    feat_slice = eval_features[:N]
    tgt_slice = targets[:N]

    with torch.no_grad():
        win = blender.build_windowed_features(feat_slice)
        log_w = wrapper.forward(feat_slice, tgt_slice, is_already_windowed=False)

    print(f'  Recorded {len(ring)} provenance entries')

    # Verify invariant: sum(top-k contributions) ≈ final_logp
    print('[4/4] Verifying provenance invariants...')
    all_ok = True
    for entry in ring.to_list():
        contribs = [c for _, c in entry['channels']]
        if contribs:
            computed = float(np.log(np.exp(contribs).sum()))
            final = entry['final_logp']
            diff = abs(computed - final)
            if diff > 1e-4:
                print(f'  FAIL at position {entry["position"]}: '
                      f'logsumexp={computed:.6f} final={final:.6f} diff={diff:.6f}')
                all_ok = False

    if all_ok:
        print(f'  PASS: logsumexp(contributions) ≈ final_logp for all {len(ring)} positions '
              f'(atol=1e-4)')

    # Dump provenance JSON
    out_path = DEEPSEEK / 'artifacts/provenance_sample.json'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ring.dump_json(str(out_path))
    print(f'  Provenance JSON saved to {out_path}')

    # Show a sample
    print('\n  Sample provenance (first 3 positions, top-5 channels):')
    for entry in ring.to_list()[:3]:
        top5 = sorted(entry['channels'], key=lambda x: -x[1])[:5]
        print(f'    pos={entry["position"]:4d}  target={entry["target_token"]:5d}  '
              f'final_logp={entry["final_logp"]:.4f}')
        for name, contrib in top5:
            print(f'      {name:10s}  contrib={contrib:8.4f}')

    # Verify the ring buffer query API
    pos0 = ring.to_list()[0]['position']
    prov0 = ring.provenance(pos0, top_k=5)
    assert prov0 is not None, 'provenance(token_idx) should return data'
    assert len(prov0) > 0, 'should have channel contributions'
    print(f'\n  provenance({pos0}) returns {len(prov0)} channels — OK')

    assert all_ok, 'Provenance invariant failed'
    print('\n' + '=' * 60)
    print(' TICKET-005: ALL TESTS PASSED')
    print('=' * 60)


if __name__ == '__main__':
    test_provenance_200_tokens()
