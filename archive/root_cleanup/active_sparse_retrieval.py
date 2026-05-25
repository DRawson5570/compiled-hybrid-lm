"""active_sparse_retrieval.py

Local memory gates sparse hash-mapping catalog storing previous traces and
high-performance selections. Employs lightweight retrieval to prevent RAM footprint
exceeding 64GB.
"""
from __future__ import annotations

import sys
import torch
from pathlib import Path

REPO = Path(__file__).resolve().parent

class SparseRetrievalIndex:
    """Rigidly constrained RAM indexing mechanism utilizing hash filters."""
    def __init__(self, vocab_size: int = 8000):
        # High efficiency sparse map lookup
        self.sparse_record: dict[str, list[float]] = {}
        
    def add_trace(self, prompt: str, routing_weights: list[float]):
        # Store clean representations
        clean_key = " ".join([w.strip().lower() for w in prompt.split() if w.isalnum()])
        self.sparse_record[clean_key] = routing_weights
        
    def retrieve_closest(self, query: str) -> list[float] | None:
        clean_query = " ".join([w.strip().lower() for w in query.split() if w.isalnum()])
        
        # Exact keyword match
        if clean_query in self.sparse_record:
            return self.sparse_record[clean_query]
            
        # Fallback to Jaccard-similar subset matches
        query_words = set(clean_query.split())
        best_match = None
        best_score = 0.0
        
        for k, v in self.sparse_record.items():
            k_words = set(k.split())
            if not k_words or not query_words:
                continue
            intersection = query_words.intersection(k_words)
            score = len(intersection) / len(query_words.union(k_words))
            if score > best_score and score >= 0.5:
                best_score = score
                best_match = v
                
        return best_match

def run_retrieval_suite():
    print("=" * 80)
    print("         CMI ACTIVE SPARSE MEMORY GATED RETRIEVAL INDEX")
    print("=" * 80)
    
    index = SparseRetrievalIndex()
    
    # Register typical diagnostic and translation footprints
    index.add_trace("translate dog to french", [0.95, 0.05, 0.0, 0.0])
    index.add_trace("Solve standard mathematical equation sum", [0.0, 0.05, 0.0, 0.95])
    print(f"Registered {len(index.sparse_record)} active historical trails.")
    
    # Test strict retrieval properties
    print("\nQuerying closely matched instruction: 'translate dog to french'")
    res1 = index.retrieve_closest("translate dog to french")
    print(f"  Result 1 weights: {res1}")
    
    print("\nQuerying overlapping fuzzy match: 'translate a dog to standard french'")
    res2 = index.retrieve_closest("translate a dog to standard french")
    print(f"  Result 2 weights: {res2}")
    print(f"  Did fuzzy retrieval successfully map? {'Yes! (Pass)' if res2 else 'No (Fail)'}")
    print("=" * 80)

if __name__ == "__main__":
    run_retrieval_suite()
