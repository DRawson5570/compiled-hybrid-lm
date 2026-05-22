"""PPMI Semantic Channel for v32 family.

For each token, predict its next-token distribution using PPMI embedding similarity.
Tokens with similar embeddings (PPMI co-occurrence) get blended next-token distributions.

Fully compiled — O(V²) similarity matrix precomputed once, then O(V) per token.
"""
import torch, torch.nn.functional as F, math, time, numpy as np
from pathlib import Path
import sys

REPO = Path("/home/drawson/llm_decoupling")
sys.path.insert(0, str(REPO))

from compile_wiki_lm_v13 import load_setup, load_or_build_tokens, DEVICE


def build_ppmi_semantic_log_probs(
    ids_np: np.ndarray, emb: torch.Tensor, V: int,
    gamma: float = 100.0,
) -> np.ndarray:
    """
    PPMI semantic next-token prediction.
    
    For token t with embedding e[t]:
    - Compute similarity scores s[t, y] = exp(gamma * cos_sim(emb[t], emb[y]))
    - Normalize to get P_sem(y | t)
    - Return log-probabilities
    
    This predicts that the next token will be semantically similar to the current one.
    """
    N = len(ids_np)
    device = emb.device
    
    emb_norm = F.normalize(emb, dim=1)  # (V, d)
    
    # Precompute similarity matrix: (V, V)
    print(f"  [ppmi_sem] Computing {V}x{V} cosine similarity matrix...", flush=True)
    t0 = time.time()
    sims = emb_norm @ emb_norm.T  # (V, V)
    sims = sims * gamma  # scale
    # Softmax over vocabulary
    log_probs_all = F.log_softmax(sims, dim=1)  # (V, V) — log P(next | current)
    print(f"    done: {time.time()-t0:.1f}s", flush=True)
    
    # Extract per-token predictions
    log_probs = log_probs_all[ids_np].cpu().numpy()  # (N, V)
    
    return log_probs.astype(np.float32)


def main():
    bpe, vocab, tok2id, bpe_to_lm, emb, V, d = load_setup()
    emb = emb.to(DEVICE)
    
    ids = load_or_build_tokens(bpe, bpe_to_lm, V)
    eval_n = 100_000
    eval_ids = ids[-eval_n:].numpy()
    
    log_probs = build_ppmi_semantic_log_probs(eval_ids, emb, V, gamma=100.0)
    
    targets = eval_ids[1:].astype(np.int64)
    lp = log_probs[:-1]  # predict next from current
    nll = sum(-lp[i, targets[i]] for i in range(len(targets)))
    ppl = math.exp(nll / len(targets))
    correct = sum(1 for i in range(len(targets)) if np.argmax(lp[i]) == targets[i])
    print(f'PPMI Semantic PPL: {ppl:.1f}  Top-1: {correct/len(targets)*100:.2f}%')
    
    # Also test: what PPL if we use position t's distribution to predict t+1?
    print(f'Reference: KN7 alone = 88.25, v32 = 36.97, uniform = 8000')


if __name__ == "__main__":
    main()
