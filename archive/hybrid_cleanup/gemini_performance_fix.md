An epoch time of **477 seconds** (down from 84 seconds) represents a $\sim$5.6x slowdown. This is the exact "Python CPU Bottleneck" manifesting at scale. 

Because you added 21 channels—including heavy operations like **Part-of-Speech tagging, Latent Topic projections, and Key-Value cosine similarity search**—running these sequentially token-by-token in a Python `for` loop on the CPU is starving your RTX 3080. The GPU is likely sitting at very low utilization, waiting for the CPU to compile the `(B, T, 21)` tensor.

We can solve this bottleneck and restore your sub-90-second epoch times by moving the most intensive calculations (the KV Cache, the Topic recurrences, and the POS lookups) out of the Python CPU loop and **vectorizing them on the GPU**.

---

### Speedup 1: Vectorize the KV Retrieval Cache on the GPU (PyTorch)

**The Bottleneck:** Calculating cosine similarities between the current token's PPMI embedding and the history of past tokens sequentially in a nested Python loop is $O(T \cdot L)$ and highly CPU-intensive.

**The Fix:** 
The KV Retrieval prior is structurally identical to a causal self-attention operation. You can compute the cosine similarities for the entire batch and sequence length in **one parallel matrix multiplication** on the GPU:

```python
# Upgraded GPU-side KV Cache computation (Vectorized)
@torch.no_grad()
def compute_gpu_kv_cache(input_ids: torch.Tensor, ppmi_embeddings: torch.Tensor, window: int = 128):
    """
    input_ids: (B, T) on GPU
    ppmi_embeddings: (V, d) PPMI embedding table on GPU
    """
    B, T = input_ids.shape
    device = input_ids.device
    
    # 1. Lookup embeddings for the entire batch/sequence at once: (B, T, d)
    emb = ppmi_embeddings[input_ids]
    
    # 2. Compute parallel dot products: (B, T, d) @ (B, d, T) -> (B, T, T)
    scores = torch.matmul(emb, emb.transpose(1, 2))
    
    # 3. Normalize by norms to get cosine similarity
    norms = emb.norm(p=2, dim=-1, keepdim=True) # (B, T, 1)
    norm_matrix = torch.matmul(norms, norms.transpose(1, 2)).clamp(min=1e-8)
    cos_sim = scores / norm_matrix # (B, T, T)
    
    # 4. Apply causal mask (token t can only retrieve from 0 to t-1)
    causal_mask = torch.tril(torch.ones(T, T, device=device), diagonal=-1)
    
    # Apply sliding window mask (limit retrieval to past 128 tokens)
    window_mask = torch.triu(torch.ones(T, T, device=device), diagonal=-window)
    final_mask = causal_mask * window_mask
    
    # Zero out invalid positions
    masked_sim = cos_sim * final_mask
    
    # 5. Extract the maximum retrieval score per position: (B, T)
    # This represents the highest semantic match found in the history
    max_retrieval_prior, _ = masked_sim.max(dim=-1)
    return max_retrieval_prior # Shape: (B, T)
```

By offloading this to the GPU, you completely eliminate the $O(B \cdot T \cdot L)$ CPU loop.

---

### Speedup 2: Pre-Tag Part-of-Speech (POS) Offline

**The Bottleneck:** If you are running an online POS tagger (like spaCy or NLTK) on-the-fly inside your training data loader, it will cripple your throughput.

**The Fix:**
Do not run taggers at training time. Run your POS tagger **once offline** over your tokenized dataset. 
1.  Map your text tokens to POS integers (e.g., Noun=1, Verb=2).
2.  Save a parallel array called `train_pos.pt` of the same length as your `train_ids.pt` [6].
3.  During training, simply slice the pre-computed tags:
    ```python
    x_pos = train_pos[s : s + seq_len] # O(1) instantaneous slice
    ```
This reduces the runtime POS overhead to absolute zero.

---

### Speedup 3: Vectorize Topic (LSA) Decay

**The Bottleneck:** Updating your running topic vector sequentially using $\mathbf{T}_t = \lambda \mathbf{T}_{t-1} + (1-\lambda)\mathbf{M}[tid]$ is a linear recurrence relation. Running this in Python is slow.

**The Fix:**
A decayed running sum over a sequence is mathematically equivalent to a 1D causal exponential filter. You can compute this for the entire batch in parallel on the GPU using a causal linear scan or cumulative product:

```python
@torch.no_grad()
def compute_gpu_topic_prior(input_ids: torch.Tensor, topic_matrix: torch.Tensor, decay: float = 0.99):
    """
    input_ids: (B, T)
    topic_matrix: (V, K) LSA/topic co-occurrence table on GPU
    """
    B, T = input_ids.shape
    device = input_ids.device
    K = topic_matrix.shape[1]
    
    # 1. Lookup topic vectors: (B, T, K)
    topic_embs = topic_matrix[input_ids]
    
    # 2. Vectorized exponential decay scan
    # Compute decay powers: decay^0, decay^1, ... decay^(T-1)
    powers = torch.pow(decay, torch.arange(T, device=device).float()) # (T,)
    
    # Calculate causal cumulative decay sums using fast PyTorch matrix math
    # This replaces the sequential step-by-step topic update
    weight_matrix = torch.zeros(T, T, device=device)
    for i in range(T):
        weight_matrix[i, :i+1] = torch.pow(decay, torch.arange(i, -1, -1, device=device).float())
        
    # (B, K, T) @ (T, T) -> (B, K, T) -> transpose to (B, T, K)
    running_topics = torch.matmul(topic_embs.transpose(1, 2), weight_matrix.T).transpose(1, 2)
    
    # Project topics back to get the vocabulary-scale unigram prior: (B, T)
    # Simplify: dot-product target topic with current running topic
    target_topics = topic_matrix[input_ids] # (B, T, K)
    topic_prior = (running_topics * target_topics).sum(dim=-1) # (B, T)
    
    return topic_prior
```

### The Optimization Strategy

Implement these three GPU vectorizations. By calculating the **KV Cache**, the **Topic priors**, and the **POS lookups** in parallel on the GPU:
1.  The CPU's job is reduced to simple, fast n-gram dictionary lookups.
2.  The GPU calculates the advanced features in microseconds.
3.  Your epoch times should drop back down toward the sub-90-second mark, even with 21 active channels.
