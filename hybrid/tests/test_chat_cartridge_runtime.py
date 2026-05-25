from __future__ import annotations

import sys
from pathlib import Path

DEEPSEEK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DEEPSEEK.parent))

from hybrid.chat_cartridge import answer_can_stop, requested_numbered_items, trim_to_sentences


def test_answer_can_stop_rejects_dangling_numbered_list_marker():
    assert not answer_can_stop('1. Eat balanced meals.\n2.')
    assert answer_can_stop('1. Eat balanced meals.\n2. Move regularly.\n3. Sleep enough.')


def test_answer_can_stop_respects_required_numbered_items():
    assert not answer_can_stop('1. Eat balanced meals.\n2. Move regularly.', required_items=3)
    assert answer_can_stop('1. Eat balanced meals.\n2. Move regularly.\n3. Sleep enough.', required_items=3)


def test_answer_can_stop_rejects_trailing_clause_punctuation():
    assert not answer_can_stop('The next step is:')
    assert answer_can_stop('The next step is to run the focused test.')


def test_requested_numbered_items_parses_small_counts():
    assert requested_numbered_items('Give me three tips for staying healthy.') == 3
    assert requested_numbered_items('List 2 risks.') == 2
    assert requested_numbered_items('Explain gravity in simple terms.') == 0


def test_trim_to_sentences_ignores_numbered_list_markers():
    text = '1. Eat balanced meals.\n2. Move regularly.\n3. Sleep enough.'

    assert trim_to_sentences(text, 3) == text