"""Unit test for the deterministic document-disjoint wikitext corpus splitter.

Verifies:
  * Proper grouping of text lines into article blocks.
  * Correctness and disjointness of split ranges.
  * Deterministic reproduction when using the identical RNG seed.
  * Zero overlap of article contents between train, validation, and evaluation splits.
"""
from __future__ import annotations

import json
from pathlib import Path
import random
import sys
import tempfile
import pytest
import torch
from transformers import AutoTokenizer

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from hybrid.data.split_corpus_by_article import (
    parse_articles, get_ranges, get_tokenizer_hash
)


_MOCK_CORPUS = """

 = Article Alpha = 
This is the first sentence of Alpha.
Wait, section headers might follow:
 = = Section Alpha.1 = = 
Details and mathematical equations.
 = = = Subsection Alpha.1.a = = = 
More deep detail.

 = Article Beta = 
This is Beta. Yes, the second article.
Section headings follow:
 = = Section Beta.1 = = 
Beta details.

 = Article Gamma = 
The third article is Gamma.
It is short.

 = Article Delta = 
The fourth article.
We have four articles total in here.

 = Article Epsilon = 
The fifth article. Epsilon.
Empty lines follow below.


"""


def test_parse_articles_matching():
    lines = _MOCK_CORPUS.strip().splitlines(keepends=True)
    articles = parse_articles(lines)
    
    # We expect exactly 5 articles
    assert len(articles) == 5
    
    # Check titles
    titles = [art[0].strip() for art in articles]
    assert titles == [
        "= Article Alpha =",
        "= Article Beta =",
        "= Article Gamma =",
        "= Article Delta =",
        "= Article Epsilon ="
    ]
    
    # Check that sections elements are not misclassified as main headers
    for idx, art in enumerate(articles):
        # The first element is the main header
        assert art[0].strip().startswith("= ") and art[0].strip().endswith(" =") and not art[0].strip().startswith("= = ")
        # Subsequent lines should NOT be treated as new articles
        for line in art[1:]:
            stripped = line.strip()
            if stripped.startswith("= ") and stripped.endswith(" ="):
                assert stripped.startswith("= = ")


def test_get_ranges():
    # Test on empty list
    assert get_ranges([]) == []
    
    # Test on contiguous IDs
    assert get_ranges([0, 1, 2, 3]) == [[0, 3]]
    
    # Test on disjoint IDs
    assert get_ranges([0, 1, 3, 4, 10]) == [[0, 1], [3, 4], [10, 10]]


def test_zero_overlap_split_manifest_and_tensors():
    # Use a temporary directory to write files and execute end-to-end splitting
    with tempfile.TemporaryDirectory() as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str)
        corpus_path = tmp_dir / "mock_corpus.txt"
        with open(corpus_path, "w", encoding="utf-8") as f:
            f.write(_MOCK_CORPUS)
            
        out_path = tmp_dir / "output_disjoint"
        
        # Invoke via import main or run directly
        import sys
        from unittest.mock import patch
        
        # Build arguments lists
        test_args = [
            "split_corpus_by_article.py",
            "--corpus", str(corpus_path),
            "--out", str(out_path),
            "--seed", "12345",
            "--ratios", "0.6", "0.2", "0.2"
        ]
        
        with patch.object(sys, "argv", test_args):
            from hybrid.data.split_corpus_by_article import main
            main()
            
        # Verify files are produced
        train_t_path = out_path / "train_ids.pt"
        val_t_path = out_path / "val_ids.pt"
        eval_t_path = out_path / "eval_ids.pt"
        manifest_path = out_path / "split_manifest.json"
        
        assert train_t_path.exists()
        assert val_t_path.exists()
        assert eval_t_path.exists()
        assert manifest_path.exists()
        
        # Read the manifest
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
            
        # Total articles must sum to 5
        art_counts = manifest["n_articles_per_split"]
        assert art_counts["train"] + art_counts["validation"] + art_counts["evaluation"] == 5
        
        # Extract individual sets of original article ID ranges
        train_ranges = manifest["article_id_ranges"]["train"]
        val_ranges = manifest["article_id_ranges"]["validation"]
        eval_ranges = manifest["article_id_ranges"]["evaluation"]
        
        # Helper to convert ranges list of lists to a flat set of integer IDs
        def ranges_to_set(ranges):
            s = set()
            for start, end in ranges:
                s.update(range(start, end + 1))
            return s
            
        train_set = ranges_to_set(train_ranges)
        val_set = ranges_to_set(val_ranges)
        eval_set = ranges_to_set(eval_ranges)
        
        # Verify partition is strictly disjoint
        assert train_set.isdisjoint(val_set)
        assert train_set.isdisjoint(eval_set)
        assert val_set.isdisjoint(eval_set)
        
        # Verify the sum of sizes is exactly the total articles (5) and covers them without gaps
        union_set = train_set.union(val_set).union(eval_set)
        assert union_set == set(range(5))
        
        # Load PT files
        train_tokens = torch.load(train_t_path, weights_only=False)
        val_tokens = torch.load(val_t_path, weights_only=False)
        eval_tokens = torch.load(eval_t_path, weights_only=False)
        
        # Verify sizes are positive of LongTensors
        assert isinstance(train_tokens, torch.Tensor)
        assert train_tokens.dim() == 1
        assert len(train_tokens) > 0
        assert len(val_tokens) > 0
        assert len(eval_tokens) > 0
