"""compiled_priors_v3.py — Topic vector + KV semantic cache + POS prior.

Precomputed structures for advanced compiled channels.
"""
import numpy as np
import math
import torch
from pathlib import Path
from collections import defaultdict

DEEPSEEK = Path('/home/drawson/deepseek_experiments')


def build_topic_matrix(embeddings, K=50, sample_size=100000):
    """Build word-topic matrix via mini-batch k-means on PPMI embeddings.
    
    Args:
        embeddings: (V, d) PPMI embedding matrix
        K: number of topics
        sample_size: tokens to sample for k-means
    
    Returns:
        topic_centers: (K, d) topic centers
        word_topics: (V, K) soft topic assignments
    """
    from sklearn.cluster import MiniBatchKMeans
    
    V, d = embeddings.shape
    idx = np.random.choice(V, min(sample_size, V), replace=False)
    samples = embeddings[idx].astype(np.float64)
    
    print(f'  KMeans: {len(samples)} samples, K={K}...')
    km = MiniBatchKMeans(n_clusters=K, random_state=42, batch_size=1024,
                         n_init=3, max_iter=100)
    km.fit(samples)
    centers = km.cluster_centers_.astype(np.float32)  # (K, d)
    
    # Compute soft assignments for all tokens
    # cos_sim between each token and each center
    norms_v = np.linalg.norm(embeddings, axis=1, keepdims=True)  # (V, 1)
    norms_c = np.linalg.norm(centers, axis=1, keepdims=True)  # (K, 1)
    normalized_v = embeddings / (norms_v + 1e-8)
    normalized_c = centers / (norms_c + 1e-8)
    
    # Compute in chunks for memory
    chunk_size = 16384
    word_topics = np.zeros((V, K), dtype=np.float32)
    for start in range(0, V, chunk_size):
        end = min(start + chunk_size, V)
        chunk = normalized_v[start:end]  # (chunk, d)
        sim = chunk @ normalized_c.T  # (chunk, K)
        sim = np.clip(sim, -1, 1)
        # Softmax per row
        sim_exp = np.exp(sim * 5.0)  # temperature for sharper assignments
        word_topics[start:end] = sim_exp / (sim_exp.sum(axis=1, keepdims=True) + 1e-8)
    
    print(f'  Built topic matrix: {V}×{K}')
    return torch.tensor(centers), torch.tensor(word_topics)


class TopicVectorTracker:
    """Running topic vector maintained during generation.
    
    T_t = λ * T_{t-1} + (1-λ) * M[tid]
    
    Projects back to vocabulary at each step for a global semantic prior.
    """
    def __init__(self, word_topics, lambd=0.95):
        """Args:
            word_topics: (V, K) soft topic assignments per token
            lambd: decay factor for topic vector
        """
        self.word_topics = word_topics  # (V, K)
        self.lambd = lambd
        self.K = word_topics.shape[1]
        self._topic_vec = np.zeros(self.K, dtype=np.float32)
        self._vocab_prior = np.zeros(word_topics.shape[0], dtype=np.float32)
    
    def update(self, token_id):
        """Update running topic vector with a new token."""
        tid = int(token_id)
        token_topic = self.word_topics[tid].numpy()  # (K,)
        self._topic_vec = self.lambd * self._topic_vec + (1 - self.lambd) * token_topic
    
    def get_topic_prior(self):
        """Get vocabulary-level topic prior: p_topic = T · M^T.
        Returns log-probabilities over vocabulary.
        """
        topic_prior = self._topic_vec @ self.word_topics.numpy().T  # (V,)
        topic_prior = topic_prior / (topic_prior.sum() + 1e-8)
        return np.log(np.maximum(topic_prior, 1e-12))
    
    def get_topic_features(self, target_token):
        """Get topic-related features for a target token.
        Returns:
            topic_logp: log probability of token under topic prior
            topic_max: max topic activation
            topic_entropy: entropy of topic vector
        """
        tid = int(target_token)
        tv = self._topic_vec
        probs = tv / (tv.sum() + 1e-8)
        probs_valid = probs[probs > 0]
        entropy = -np.sum(probs_valid * np.log(probs_valid)) / math.log(self.K) if len(probs_valid) > 0 else 1.0
        
        topic_logp = float(self._vocab_prior[tid]) if tid < len(self._vocab_prior) else self._vocab_prior[0]
        
        return topic_logp, float(np.max(tv)), float(1.0 - entropy)


class KVSemanticCache:
    """Key-Value cache of PPMI embeddings for non-parametric retrieval.
    
    Keys: PPMI embeddings of last N tokens.
    Values: token IDs.
    
    At each step: cosine similarity of current embedding vs all keys
    → softmax weights → weighted vocabulary distribution.
    """
    def __init__(self, embeddings, max_size=128):
        """Args:
            embeddings: (V, d) PPMI embedding matrix
            max_size: number of tokens in the cache window
        """
        self.embeddings = embeddings  # (V, d)
        self.max_size = max_size
        self._keys = np.zeros((max_size, embeddings.shape[1]), dtype=np.float32)
        self._values = np.zeros(max_size, dtype=np.int64)
        self._norms = np.zeros(max_size, dtype=np.float32)
        self._count = 0
        self._ptr = 0
    
    def add(self, token_id):
        tid = int(token_id)
        emb = self.embeddings[tid].float().numpy()
        idx = self._ptr
        self._keys[idx] = emb
        self._values[idx] = tid
        self._norms[idx] = np.linalg.norm(emb)
        self._ptr = (self._ptr + 1) % self.max_size
        self._count = min(self._count + 1, self.max_size)
    
    def retrieve(self, target_token):
        """Get retrieval features for a target token.
        
        Returns:
            max_sim: maximum cosine similarity with cached tokens
            weighted_logp: weighted log probability from similar contexts
        """
        if self._count == 0:
            return 0.0, 0.0
        
        tid = int(target_token)
        target_emb = self.embeddings[tid].float().numpy()
        target_norm = np.linalg.norm(target_emb)
        
        if target_norm < 1e-8:
            return 0.0, 0.0
        
        # Cosine similarity with all cached keys
        cosine_sims = (self._keys[:self._count] @ target_emb) / (
            self._norms[:self._count] * target_norm + 1e-8)
        cosine_sims = np.clip(cosine_sims, -1, 1)
        
        max_sim = float(np.max(cosine_sims))
        
        # Weighted sum of similarities as a proxy for retrieval prior
        top_k = min(8, self._count)
        top_indices = np.argpartition(cosine_sims, -top_k)[-top_k:]
        top_sims = cosine_sims[top_indices]
        
        # Weighted log prob: higher similarity = more weight
        weights = np.exp(top_sims * 3.0)
        weights = weights / (weights.sum() + 1e-8)
        
        # Compute probability that target appears near similar contexts
        match_count = 0
        for idx in top_indices:
            retrieved_tok = int(self._values[idx])
            if retrieved_tok == tid:
                match_count += 1
        
        retrieval_p = (match_count + 0.001) / (top_k + 0.001 * top_k)
        retrieval_logp = math.log(max(retrieval_p, 1e-12))
        
        return max_sim, retrieval_logp
