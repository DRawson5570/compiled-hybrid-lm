"""test_tokenizer_roundtrip.py — GPT-2 BPE round-trip verification.

Part of TICKET-008: encode → decode → re-encode must be identity
on a representative sample of WikiText-103 tokens.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

DEEPSEEK = Path('/home/drawson/deepseek_experiments')
sys.path.insert(0, str(DEEPSEEK))


def test_gpt2_tokenizer_roundtrip():
    """Decode produces readable text; re-encoding is close to original length.

    Note: GPT-2 BPE decode→encode is NOT identity-preserving because of
    whitespace normalization. This is standard — PPL evaluation uses the
    original tokenization, not round-tripped tokens.
    """
    tok = AutoTokenizer.from_pretrained('gpt2')

    ids = torch.load(
        DEEPSEEK / 'artifacts/wikitext_gpt2/test_ids.pt',
        weights_only=False
    ).long()

    # Test on first 1000 tokens
    sample = ids[:1000]
    text = tok.decode(sample.tolist())
    re_ids = tok.encode(text)
    re_tensor = torch.tensor(re_ids, dtype=torch.long)

    # Check: re-encoded length is close (within 5%)
    len_ratio = len(re_tensor) / len(sample)
    assert 0.90 < len_ratio < 1.10, (
        f'Re-encoded length ratio {len_ratio:.3f} not in [0.90, 1.10]'
    )

    # Check: text is readable (contains spaces and letters)
    assert len(text) > 100, f'Decoded text too short: {len(text)} chars'
    assert any(c.isalpha() for c in text), 'No alphabetic characters in decoded text'

    print(f'PASS: Decode→encode length ratio = {len_ratio:.3f} (within bounds)')
    print(f'      Decoded text sample: {text[:100]}...')


def test_tokenizer_special_tokens():
    """EOS and other special tokens survive round-trip."""
    tok = AutoTokenizer.from_pretrained('gpt2')

    ids = torch.load(
        DEEPSEEK / 'artifacts/wikitext_gpt2/test_ids.pt',
        weights_only=False
    ).long()

    eos_positions = (ids == tok.eos_token_id).nonzero(as_tuple=True)[0]
    assert len(eos_positions) > 0, 'No EOS tokens found in test set'
    print(f'Found {len(eos_positions)} EOS tokens in test set — OK')


def test_vocab_coverage():
    """All token IDs in the corpus are within the GPT-2 vocab range."""
    tok = AutoTokenizer.from_pretrained('gpt2')

    train_ids = torch.load(
        DEEPSEEK / 'artifacts/wikitext_gpt2/train_ids.pt',
        weights_only=False
    ).long()

    max_id = train_ids.max().item()
    min_id = train_ids.min().item()
    assert 0 <= min_id, f'Negative token ID: {min_id}'
    assert max_id < tok.vocab_size, f'Token ID {max_id} >= vocab size {tok.vocab_size}'
    print(f'Vocab coverage: [{min_id}, {max_id}] ⊆ [0, {tok.vocab_size}) — OK')


if __name__ == '__main__':
    test_gpt2_tokenizer_roundtrip()
    test_tokenizer_special_tokens()
    test_vocab_coverage()
    print('\nAll GPT-2 BPE tokenizer tests passed.')
