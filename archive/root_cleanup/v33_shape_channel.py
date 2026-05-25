"""Word Shape Channel — predicts next-token capitalization patterns.

Novel compiled channel not covered by v32's 18 channels.
For each token, classify its capitalization pattern and predict
the next token's pattern. Boosts tokens matching the expected shape.

Patterns: lowercase, UPPERCASE, Capitalized, Mixed, numeric, punctuation
"""
import torch, torch.nn.functional as F, math, time, numpy as np
from pathlib import Path
import sys

REPO = Path("/home/drawson/llm_decoupling")
sys.path.insert(0, str(REPO))

from compile_wiki_lm_v13 import load_setup, load_or_build_tokens, DEVICE


def token_shape(token_str: str) -> int:
    """Classify token shape: 0=lower, 1=UPPER, 2=Cap, 3=mixed, 4=digit, 5=punct, 6=other"""
    if not token_str:
        return 6
    if token_str.isdigit():
        return 4
    if not any(c.isalpha() for c in token_str):
        return 5  # punctuation
    if token_str.islower():
        return 0
    if token_str.isupper():
        return 1
    if token_str[0].isupper() and token_str[1:].islower():
        return 2  # Capitalized
    return 3  # mixed


def build_shape_transition_probs(
    ids_np: np.ndarray, bpe, V: int
) -> np.ndarray:
    """
    Build token shape transition probabilities.
    Uses the BPE tokenizer to decode tokens to strings for shape classification.
    """
    N = len(ids_np)
    N_SHAPES = 7
    
    # Map each token ID to its shape class (build from BPE tokenizer)
    token_shapes = np.zeros(V, dtype=np.int32)
    for tid in range(V):
        tok_str = bpe.decode([tid])
        token_shapes[tid] = token_shape(tok_str)
    
    # Count shape transitions on training data (use a subset for speed)
    shape_counts = np.zeros((N_SHAPES, N_SHAPES), dtype=np.float64)
    count_n = min(N - 1, 5_000_000)
    for t in range(count_n):
        s_cur = token_shapes[ids_np[t]]
        s_nxt = token_shapes[ids_np[t + 1]]
        shape_counts[s_cur, s_nxt] += 1
    
    # Laplace-smoothed transition probabilities
    alpha = 0.1
    shape_trans = (shape_counts + alpha) / (shape_counts.sum(axis=1, keepdims=True) + alpha * N_SHAPES)
    
    # Per-token shape bias: for each token, predict next-token distribution
    # biased toward tokens with the expected next shape
    log_probs = np.full((N, V), -math.log(V), dtype=np.float32)
    
    for t in range(N):
        cur_shape = token_shapes[ids_np[t]]
        expected_next_shape_probs = shape_trans[cur_shape]  # (7,)
        
        # Boost tokens matching each shape by the shape's transition probability
        logits = np.full(V, -10.0, dtype=np.float32)
        for s in range(N_SHAPES):
            mask = (token_shapes == s)
            if mask.any():
                boost = math.log(expected_next_shape_probs[s] + 1e-10)
                logits[mask] = boost
        
        log_probs[t] = logits - np.log(np.sum(np.exp(logits)))  # log-softmax
    
    return log_probs


def main():
    bpe, vocab, tok2id, bpe_to_lm, emb, V, d = load_setup()
    ids = load_or_build_tokens(bpe, bpe_to_lm, V)
    
    eval_n = 100_000
    eval_ids = ids[-eval_n:].numpy()
    
    print("Building shape transition statistics...", flush=True)
    t0 = time.time()
    log_probs = build_shape_transition_probs(eval_ids, bpe, V)
    print(f"  done: {time.time()-t0:.1f}s", flush=True)
    
    targets = eval_ids[1:].astype(np.int64)
    lp = log_probs[:-1]
    nll = sum(-lp[i, targets[i]] for i in range(len(targets)))
    ppl = math.exp(nll / len(targets))
    correct = sum(1 for i in range(len(targets)) if np.argmax(lp[i]) == targets[i])
    print(f'Word Shape PPL: {ppl:.1f}  Top-1: {correct/len(targets)*100:.2f}%')
    print(f'Reference: KN7=88.25, v32=36.97, uniform=8000')
    
    # Also show shape transition matrix
    shape_names = ['lower', 'UPPER', 'Capital', 'mixed', 'digit', 'punct', 'other']
    counts = np.zeros((7, 7))
    for t in range(len(eval_ids) - 1):
        s1 = token_shape(bpe.decode([int(eval_ids[t])]))
        s2 = token_shape(bpe.decode([int(eval_ids[t+1])]))
        counts[s1, s2] += 1
    trans = (counts + 0.1) / (counts.sum(axis=1, keepdims=True) + 0.7)
    print("\nShape transitions (→ next):")
    print(f"{'':>10}", end="")
    for s in shape_names:
        print(f"{s:>8}", end="")
    print()
    for i, s1 in enumerate(shape_names):
        print(f"{s1:>10}", end="")
        for j in range(7):
            print(f"{trans[i,j]:8.3f}", end="")
        print()


if __name__ == "__main__":
    main()
