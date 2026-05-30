from __future__ import annotations

from typing import Callable, List

import torch

from ..dsl.ast import Program


def build_standalone_accuracy_reward(
    task_data: List[tuple],
    oracle_fn: Callable,
) -> Callable:
    """
    Build a reward function for REINFORCE training.
    Returns accuracy-like reward (0-1, higher is better) on synthetic tasks.

    task_data: list of (embeddings, ground_truth_program)
    oracle_fn: (program, ground_truth_program) -> float similarity
    """
    def reward_fn(embeddings: torch.Tensor, program: Program) -> float:
        best_score = 0.0
        for e, gt_prog in task_data:
            if torch.allclose(embeddings.reshape(-1)[:10], e.reshape(-1)[:10]):
                score = oracle_fn(program, gt_prog)
                best_score = max(best_score, score)
                break
        return best_score
    return reward_fn


def template_match_oracle(program: Program, gt_program: Program) -> float:
    """Check if template ID matches. Returns 1.0 if same, 0.0 otherwise."""
    p_types = [type(stmt.expr).__name__ for stmt in program.statements]
    gt_types = [type(stmt.expr).__name__ for stmt in gt_program.statements]
    matches = sum(1 for a, b in zip(p_types, gt_types) if a == b)
    return matches / max(len(gt_types), 1)


def build_synthetic_task_set(
    n_tasks: int = 32,
    d_model: int = 128,
    seed: int = 42,
) -> List[tuple]:
    torch.manual_seed(seed)
    from ucn.dsl.ast import Mix, Program

    tasks = []
    for i in range(n_tasks):
        emb = torch.randn(8, d_model)
        gt = Program()
        gt.add_stmt("y", Mix(["x"], [0.5 + torch.rand(1).item() * 0.5]))
        tasks.append((emb, gt))
    return tasks
