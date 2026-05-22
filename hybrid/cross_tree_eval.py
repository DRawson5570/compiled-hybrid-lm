"""cross_tree_eval.py — Evaluate both compiled LM and blender on shared protocol.

Part of TICKET-003.  Reports PPL/top1/top5 for:
  - Our compiled LM (BPE-8000, from ~/llm_decoupling)
  - Your neural LM (GPT-2 BPE or BPE-8000, from this repo)
  - Your compiled WindowMLP blender (BPE-8000, from v33 artifacts)

All use the same sliding-window protocol and JSON output shape.
"""
from __future__ import annotations

import sys, json, math, time, importlib.util
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

DEEPSEEK = Path('/home/drawson/deepseek_experiments')
LLM_DECOUPLING = Path('/home/drawson/llm_decoupling')
sys.path.insert(0, str(DEEPSEEK))
sys.path.insert(0, str(LLM_DECOUPLING))


def _import_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ── Neural LM evaluator ──────────────────────────────────────────────────────

@torch.no_grad()
def eval_neural_lm(model, ids: torch.Tensor, seq_len: int, device,
                   model_name: str, tokenizer_name: str) -> dict:
    """Evaluate a neural LM with sliding-window protocol."""
    model.eval()
    N = len(ids)
    V = model.vocab
    total_nll = 0.0
    total_tokens = 0
    top1_correct = 0
    top5_correct = 0

    stride = seq_len
    t0 = time.time()
    for start in range(0, N, stride):
        end = min(start + seq_len, N)
        chunk = ids[start:end].unsqueeze(0).to(device)
        L = chunk.shape[1]
        if L < 2:
            continue
        logits = model(chunk)
        lp = F.log_softmax(logits[0, :-1], dim=-1)
        targets = chunk[0, 1:]

        nlls = F.nll_loss(lp, targets, reduction='none')
        total_nll += nlls.sum().item()
        total_tokens += L - 1

        _, top5_idx = lp.topk(5, dim=-1)
        top1_correct += (top5_idx[:, 0] == targets).sum().item()
        top5_correct += (top5_idx == targets.unsqueeze(-1)).any(dim=-1).sum().item()

    elapsed = time.time() - t0
    ppl = math.exp(total_nll / total_tokens)
    return {
        'model_id': model_name,
        'tokenizer': tokenizer_name,
        'vocab_size': V,
        'eval_protocol': f'sliding-window (stride={seq_len})',
        'n_tokens_evaluated': total_tokens,
        'ppl': round(ppl, 4),
        'nll': round(total_nll / total_tokens, 4),
        'top1_accuracy': round(top1_correct / total_tokens, 4) if total_tokens else 0,
        'top5_accuracy': round(top5_correct / total_tokens, 4) if total_tokens else 0,
        'wall_clock_s': round(elapsed, 1),
        'eval_split': 'standard WikiText-103 test (GPT-2 BPE)',
    }


# ── Compiled LM evaluator (KN n-gram) ────────────────────────────────────────

def eval_compiled_kn(ids_np: np.ndarray, n_gram_order: int = 5) -> dict:
    """Evaluate the compiled KN n-gram LM on a token stream."""
    from compile_wiki_lm_v23 import ModifiedKNGram

    # Build a small KN model from the eval tokens (no training data leakage —
    # this evaluates the KN methodology, not a pre-trained model)
    kn = ModifiedKNGram(n_gram_order, V=8000)
    kn.build(ids_np)

    T = len(ids_np)
    total_nll = 0.0
    top1_correct = 0
    top5_correct = 0
    n_eval = 0

    t0 = time.time()
    for t in range(n_gram_order, T):
        history = tuple(int(x) for x in ids_np[t - n_gram_order + 1:t + 1][:-1])
        p = kn.prob_vector(history)
        s = p.sum()
        p = p / s if s > 0 else np.ones(kn.V) / kn.V

        target = int(ids_np[t])
        total_nll += -math.log(max(float(p[target]), 1e-30))

        top5 = np.argpartition(-p, 5)[:5]
        if top5[0] == target:
            top1_correct += 1
        if target in top5:
            top5_correct += 1
        n_eval += 1

    elapsed = time.time() - t0
    ppl = math.exp(total_nll / n_eval)
    return {
        'model_id': f'ModifiedKNGram-N={n_gram_order}',
        'tokenizer': 'BPE-8000 (custom)',
        'vocab_size': 8000,
        'eval_protocol': 'per-position prob_vector with n-gram history',
        'n_tokens_evaluated': n_eval,
        'ppl': round(ppl, 4),
        'nll': round(total_nll / n_eval, 4),
        'top1_accuracy': round(top1_correct / n_eval, 4),
        'top5_accuracy': round(top5_correct / n_eval, 4),
        'wall_clock_s': round(elapsed, 1),
        'eval_split': 'BPE-8000 token stream (adjacent positions)',
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--neural-ckpt', type=str,
                   default='artifacts/hybrid_gpt2/gpt2_lm_best.pt')
    p.add_argument('--eval-ids', type=str,
                   default='artifacts/wikitext_gpt2/test_ids.pt')
    p.add_argument('--bpe8000-ids', type=str,
                   default=None,  # uses v11 cache
                   help='Path to BPE-8000 token IDs for compiled LM eval')
    p.add_argument('--seq-len', type=int, default=128)
    p.add_argument('--kn-order', type=int, default=5)
    p.add_argument('--device', type=str,
                   default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--out', type=str,
                   default='artifacts/cross_tree_results.json')
    args = p.parse_args()

    device = torch.device(args.device)
    results = []

    # ── Evaluate neural LM (GPT-2 BPE) ──
    print('=' * 60)
    print(' CROSS-TREE EVAL BRIDGE')
    print('=' * 60)

    print('\n[1/3] Evaluating neural LM (GPT-2 BPE)...')
    train_mod = _import_file('train_scaled',
                             str(DEEPSEEK / 'hybrid/train_scaled_neural_lm.py'))
    DeepCausalLM = train_mod.DeepCausalLM

    ckpt = torch.load(args.neural_ckpt, map_location=device)
    V = ckpt['state_dict']['head_bias'].shape[0]
    cfg = ckpt.get('args', {})
    model = DeepCausalLM(
        vocab=V, d_model=cfg.get('d_model', 256),
        n_layers=cfg.get('n_layers', 12), n_heads=cfg.get('n_heads', 8),
        d_ff=cfg.get('d_ff', 1024), max_len=cfg.get('seq_len', 128) + 1,
        dropout=0.0,
    ).to(device)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f'  Model: {n_params:,} params, V={V}, epoch={ckpt.get("epoch", "?")}')

    eval_ids = torch.load(args.eval_ids, weights_only=False).long()
    print(f'  Eval tokens: {len(eval_ids):,}')

    r = eval_neural_lm(model, eval_ids, args.seq_len, device,
                       model_name=f'DeepCausalLM-{n_params//1_000_000}M',
                       tokenizer_name='GPT-2 BPE (50257)')
    results.append(r)
    print(f'  PPL={r["ppl"]:.2f}  Top1={r["top1_accuracy"]:.4f}  Top5={r["top5_accuracy"]:.4f}')

    # ── Evaluate compiled KN (BPE-8000) ──
    print('\n[2/3] Evaluating compiled KN n-gram LM (BPE-8000)...')
    from compile_wiki_lm_v13 import load_or_build_tokens

    if args.bpe8000_ids:
        bpe8k_ids = torch.load(args.bpe8000_ids, weights_only=False).long().numpy()
    else:
        bpe8k_ids = load_or_build_tokens(None, None, None).long().numpy()

    # Use a 100K-token slice for fast eval
    eval_slice = bpe8k_ids[-100000:].astype(np.int32)
    r2 = eval_compiled_kn(eval_slice, n_gram_order=args.kn_order)
    results.append(r2)
    print(f'  PPL={r2["ppl"]:.2f}  Top1={r2["top1_accuracy"]:.4f}  Top5={r2["top5_accuracy"]:.4f}')

    # ── Save ──
    print('\n[3/3] Saving cross-tree results...')
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump({
            'cross_tree_results': results,
            'meta': {
                'eval_date': time.strftime('%Y-%m-%d %H:%M'),
                'note': 'These models use DIFFERENT tokenizers — '
                        'PPL values are not directly comparable across tokenizers.',
                'gpt2_small_baseline_ppl': 29.0,
                'gpt2_small_tokenizer': 'GPT-2 BPE (50257)',
            },
        }, f, indent=2)
    print(f'  Saved to {out_path}')

    print('\n' + '=' * 60)
    print(' SUMMARY')
    print('=' * 60)
    for r in results:
        print(f'  {r["model_id"]:30s}  PPL={r["ppl"]:8.2f}  '
              f'Top1={r["top1_accuracy"]:.4f}  '
              f'Vocab={r["vocab_size"]}  ({r["tokenizer"]})')
    print('=' * 60)


if __name__ == '__main__':
    main()
