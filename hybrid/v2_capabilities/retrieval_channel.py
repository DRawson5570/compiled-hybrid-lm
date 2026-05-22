"""RetrievalChannel — compiled document retrieval for CMI architecture.

Adds a 5th capability channel: retrieves relevant information from stored
documents using PPMI embedding similarity, then biases token predictions
toward the retrieved content.

Mechanism:
1. Documents are pre-chunked and encoded as PPMI-weighted mean embeddings.
2. At forward() time, the input tokens encode a query.
3. Query embedding is compared to all chunk embeddings via cosine similarity.
4. Top-k matching chunks are retrieved.
5. The channel boosts log-probabilities of tokens that appear in retrieved chunks.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


class RetrievalChannel:
    """Compiled document retrieval via chunk embedding similarity."""

    def __init__(self, tok2id: dict[str, int], id2tok: dict[int, str],
                 emb: torch.Tensor, doc_texts: list[str] | None = None):
        self.tok2id = tok2id
        self.id2tok = id2tok
        self.emb = emb  # (V, d)
        self.V = len(tok2id)
        self.d = emb.shape[1]

        # Build chunk index from documents
        self.chunk_embs = []       # list of (d,) tensors
        self.chunk_token_ids = []  # list of sets of token IDs in each chunk
        self.chunk_labels = []     # display labels

        if doc_texts:
            self._index_documents(doc_texts)

    def _index_documents(self, doc_texts: list[str]):
        """Chunk documents and index by PPMI-weighted mean embeddings."""
        for doc_idx, doc in enumerate(doc_texts):
            tokens = doc.split()
            chunk_size = 8
            for c in range(0, len(tokens), chunk_size // 2):  # overlap
                chunk_tokens = tokens[c:c + chunk_size]
                if len(chunk_tokens) < 2:
                    continue

                # Get token IDs for this chunk (skip UNK tokens)
                chunk_ids = []
                for t in chunk_tokens:
                    tid = self.tok2id.get(t)
                    if tid is not None:
                        chunk_ids.append(tid)

                if not chunk_ids:
                    continue

                # Chunk embedding: mean of token embeddings
                chunk_emb = self.emb[torch.tensor(chunk_ids)].mean(dim=0)
                self.chunk_embs.append(chunk_emb)
                self.chunk_token_ids.append(set(chunk_ids))
                self.chunk_labels.append(f"doc{doc_idx}c{c // chunk_size}: {' '.join(chunk_tokens)}")

        if self.chunk_embs:
            self.chunk_embs_t = torch.stack(self.chunk_embs)  # (n_chunks, d)
        else:
            self.chunk_embs_t = torch.empty(0, self.d)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: (T,) token IDs representing a query
        Returns:
            log_probs: (T, V) — log-probabilities boosted for retrieved tokens
        """
        T = input_ids.shape[0]
        device = input_ids.device

        # Default: uniform background
        log_probs = torch.full((T, self.V), -10.0, device=device)

        if len(self.chunk_embs_t) == 0 or T == 0:
            return log_probs

        chunks_t = self.chunk_embs_t.to(device)

        for t in range(T):
            # Query embedding: mean of query token embeddings
            query_emb = self.emb[input_ids[:t + 1]].mean(dim=0)

            # Cosine similarity with all chunks
            query_norm = F.normalize(query_emb.unsqueeze(0), dim=1)
            chunk_norms = F.normalize(chunks_t, dim=1)
            similarities = (query_norm @ chunk_norms.T).squeeze(0)  # (n_chunks,)

            # Top-k retrieved chunks
            k_retrieve = min(3, len(similarities))
            top_vals, top_idx = similarities.topk(k_retrieve)

            # Boost log-probabilities for tokens in retrieved chunks
            for i in range(k_retrieve):
                chunk_id_set = self.chunk_token_ids[top_idx[i].item()]
                weight = top_vals[i].item() * 8.0  # scale by similarity
                for tid in chunk_id_set:
                    log_probs[t, tid] = max(log_probs[t, tid].item(), weight)

        return log_probs


# ─── Document corpus using existing vocabulary tokens ───

DEFAULT_DOCS = [
    # French vocabulary reference (for InstructChannel + retrieval overlap)
    "translate dog to french chien . translate cat to french chat . "
    "translate apple to french pomme . translate gravity to french gravité .",

    # Gravity facts
    "gravity is a force . mass and earth have gravity . "
    "explain gravity force mass earth attraction . value of gravity is force .",

    # Dog facts
    "a dog is larger than a cat . dog has value of force . "
    "explain dog attraction earth mass .",

    # Cat facts  
    "a cat is larger than a dog . cat has value of mass . "
    "explain cat force attraction earth .",

    # Code + Python facts
    "import numpy as np . def get_sum ( a , b ) : return a + b . "
    "np . zeros ( 10 ) is a function .",

    # Math facts
    "What is 54 + 23 ? calculator expr= 54+23 . Answer is 77 . "
    "What is 12 + 15 ? calculator expr= 12+15 . Answer is 27 .",
]
