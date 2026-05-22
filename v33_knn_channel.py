"""KNN-LM retrieval channel for compiled v32 family.

Implements the Khandelwal et al. (2020) nearest-neighbor LM mechanism
as a compiled channel: for each token position, retrieve the k nearest
prior occurrences in embedding space, and blend their empirical next-token
distributions as a retrieval-based prediction.

Fully compiled — zero SGD, only cosine similarity + counting.

Adds as a 19th channel to the v32 blended family.
"""
import torch, torch.nn.functional as F, math, time, numpy as np
from pathlib import Path
import sys

REPO = Path("/home/drawson/llm_decoupling")
sys.path.insert(0, str(REPO))

from compile_wiki_lm_v13 import load_setup, load_or_build_tokens, DEVICE


def build_knn_log_probs(
    ids_np: np.ndarray, emb: torch.Tensor,
    V: int, k: int = 64, temperature: float = 1.0,
) -> np.ndarray:
    """
    Build KNN-LM retrieval log-probabilities — batched for GPU efficiency.
    Process queries in chunks of 500, searching a sliding window of prior tokens.
    """
    N = len(ids_np)
    device = emb.device
    max_window = 5000
    query_batch = 500  # process this many queries at once
    
    emb_norm = F.normalize(emb, dim=1)
    tok_norm = emb_norm[ids_np].to(device)  # (N, d) on GPU
    
    log_probs = np.full((N, V), -math.log(V), dtype=np.float32)
    
    print(f"  [knn] N={N:,}, V={V}, k={k}, window={max_window}", flush=True)
    t0 = time.time()
    
    for q_start in range(1, N, query_batch):
        q_end = min(q_start + query_batch, N)
        n_queries = q_end - q_start
        
        # Queries: (n_queries, d)
        queries = tok_norm[q_start:q_end]
        
        # For each query, search its prior window
        # We process one query batch against a shared key buffer
        window_start = max(0, q_start - max_window)
        window_end = q_start
        keys = tok_norm[window_start:window_end]  # (window, d)
        n_keys = keys.shape[0]
        
        if n_keys == 0:
            continue
        
        # All-pairs cosine similarity: (n_queries, n_keys)
        sims = queries @ keys.T  # (q, k)
        
        # Top-k per query
        k_actual = min(k, n_keys)
        top_vals, top_idx = sims.topk(k_actual, dim=1)  # (q, k)
        
        # Map to absolute positions
        abs_positions = window_start + top_idx
        next_tokens = torch.from_numpy(ids_np[abs_positions.cpu().numpy() + 1]).to(device)  # (q, k)
        
        # Temperature-scaled weights
        weights = F.softmax(top_vals / temperature, dim=1)  # (q, k)
        
        # Build per-query distributions
        for qi in range(n_queries):
            t = q_start + qi
            # Weighted empirical distribution
            dist = torch.zeros(V, device=device, dtype=torch.float32)
            nt = next_tokens[qi]
            w = weights[qi]
            dist.index_put_((nt,), w.to(torch.float32), accumulate=True)
            
            alpha = 0.01
            smoothed = (dist * k_actual + alpha) / (k_actual + alpha * V)
            log_probs[t] = torch.log(smoothed.clamp_min(1e-30)).cpu().numpy()
        
        if q_start % 5000 == 0:
            elapsed = time.time() - t0
            print(f"    {q_start:,}/{N:,} ({elapsed:.0f}s)", flush=True)
    
    print(f"    done: {time.time()-t0:.0f}s", flush=True)
    return log_probs


def main():
    """Quick test: evaluate KNN retrieval PPL on a small heldout slice."""
    bpe, vocab, tok2id, bpe_to_lm, emb, V, d = load_setup()
    emb = emb.to(DEVICE)
    
    ids = load_or_build_tokens(bpe, bpe_to_lm, V)
    print(f"Corpus: {len(ids):,} tokens")
    
    # Use a small heldout slice for testing
    start = 22_000_000
    eval_n = 20_000  # 20K tokens for quick eval
    eval_ids = ids[start:start + eval_n].numpy()
    
    log_probs = build_knn_log_probs(eval_ids, emb, V, k=64, temperature=1.0)
    
    # Evaluate PPL
    targets = eval_ids[1:].astype(np.int64)
    log_probs_eval = log_probs[1:]
    
    nll = 0.0
    for i in range(len(targets)):
        nll += -log_probs_eval[i, targets[i]]
    
    ppl = math.exp(nll / len(targets))
    print(f"\nKNN-LM retrieval PPL on 20K heldout: {ppl:.1f}")
    
    # Top-1 accuracy
    correct = 0
    for i in range(len(targets)):
        if np.argmax(log_probs_eval[i]) == targets[i]:
            correct += 1
    print(f"Top-1: {correct/len(targets)*100:.2f}%")


if __name__ == "__main__":
    main()
