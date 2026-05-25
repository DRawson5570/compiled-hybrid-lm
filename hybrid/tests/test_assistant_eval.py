from __future__ import annotations

from hybrid.assistant_eval import AssistantTask, numbered_count, score_answer, summarize


def test_numbered_count_counts_distinct_numbered_items():
    assert numbered_count('1. First\n2. Second\n2. Duplicate\n3. Third') == 3


def test_score_answer_requires_keywords_and_numbered_items():
    task = AssistantTask(
        task_id='debug',
        category='workflow',
        prompt='Give three debugging steps.',
        required_any=('reproduce', 'rerun'),
        numbered_items=3,
    )

    row = score_answer(task, '1. Reproduce the failure.\n2. Inspect the assertion.\n3. Rerun the focused test.')

    assert row.passed


def test_score_answer_reports_missing_requirements():
    task = AssistantTask(
        task_id='gravity',
        category='science',
        prompt='Explain gravity.',
        required_all=('mass',),
        forbidden_any=('System:',),
    )

    row = score_answer(task, 'System: Gravity is a mystery.')

    assert not row.passed
    assert 'missing:mass' in row.failures
    assert 'forbidden:System:' in row.failures


def test_summarize_groups_by_category():
    rows = [
        score_answer(AssistantTask('a', 'x', 'a', required_all=('yes',)), 'yes'),
        score_answer(AssistantTask('b', 'x', 'b', required_all=('yes',)), 'no'),
    ]

    summary = summarize(rows)

    assert summary['passed'] == 1
    assert summary['total'] == 2
    assert summary['by_category']['x'] == {'passed': 1, 'total': 2}