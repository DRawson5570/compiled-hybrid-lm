"""ARC benchmark scoring via conditional log-likelihood."""
from __future__ import annotations

import math
import time
from dataclasses import dataclass

import torch

from hybrid.benchmarks.arc_data import ARCExample, ARCChoice
from hybrid.benchmarks.arc_prompts import PromptTemplate


@dataclass
class ChoiceScore:
    label: str
    text: str
    score_norm: float
    score_sum: float
    num_tokens: int


@dataclass
class ScoredExample:
    example: ARCExample
    scores: list[ChoiceScore]
    pred_norm: str
    pred_sum: str
    correct_norm: bool | None
    correct_sum: bool | None
    margin_norm: float
    elapsed_sec: float


class HFArcScorer:
    """Log-likelihood scorer using any raw HuggingFace causal LM."""

    def __init__(self, model, tokenizer, device: torch.device, dtype=torch.float16):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.dtype = dtype
        self._model_was_training = model.training
        model.eval()

    def _score_continuation(self, prompt: str, continuation: str) -> tuple[float, float, int]:
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")
        full_text = prompt + continuation
        full_ids = self.tokenizer.encode(full_text, add_special_tokens=False, return_tensors="pt")
        if full_ids is None or prompt_ids is None:
            return float("-inf"), float("-inf"), 0
        full_ids = full_ids.to(self.device)
        answer_ids = full_ids[0, prompt_ids.shape[1]:]
        if answer_ids.numel() == 0:
            continuation_space = " " + continuation
            full_text = prompt + continuation_space
            full_ids = self.tokenizer.encode(full_text, add_special_tokens=False, return_tensors="pt")
            full_ids = full_ids.to(self.device)
            answer_ids = full_ids[0, prompt_ids.shape[1]:]
        if answer_ids.numel() == 0:
            return float("-inf"), float("-inf"), 0

        with torch.no_grad():
            logits = self.model(full_ids).logits.float()
            logprobs = torch.nn.functional.log_softmax(logits, dim=-1)

        total_logprob = 0.0
        for i, token_id in enumerate(answer_ids):
            token_logprob = logprobs[0, prompt_ids.shape[1] + i - 1, token_id].item()
            total_logprob += token_logprob

        num_tokens = answer_ids.numel()
        score_norm = total_logprob / max(num_tokens, 1)
        score_sum = total_logprob
        return score_norm, score_sum, num_tokens

    def score_options(self, example: ARCExample, template: PromptTemplate) -> list[ChoiceScore]:
        prompt = template.render_prompt(example)
        scores: list[ChoiceScore] = []
        for choice in example.choices:
            continuation = template.render_continuation(choice.text)
            norm, total, num_tokens = self._score_continuation(prompt, continuation)
            scores.append(ChoiceScore(
                label=choice.label,
                text=choice.text,
                score_norm=norm,
                score_sum=total,
                num_tokens=num_tokens,
            ))
        return scores

    def score_example(self, example: ARCExample, template: PromptTemplate) -> ScoredExample:
        t0 = time.perf_counter()
        scores = self.score_options(example, template)

        if all(math.isinf(s.score_norm) for s in scores):
            pred_norm = scores[0].label
            pred_sum = scores[0].label
        else:
            best_norm = max(scores, key=lambda s: s.score_norm)
            pred_norm = best_norm.label
            best_sum = max(scores, key=lambda s: s.score_sum)
            pred_sum = best_sum.label

        correct_norm = None
        correct_sum = None
        if example.answer_key is not None:
            correct_norm = pred_norm == example.answer_key
            correct_sum = pred_sum == example.answer_key

        sorted_scores = sorted(scores, key=lambda s: s.score_norm, reverse=True)
        margin_norm = 0.0
        if len(sorted_scores) >= 2:
            margin_norm = sorted_scores[0].score_norm - sorted_scores[1].score_norm

        elapsed = time.perf_counter() - t0

        return ScoredExample(
            example=example,
            scores=scores,
            pred_norm=pred_norm,
            pred_sum=pred_sum,
            correct_norm=correct_norm,
            correct_sum=correct_sum,
            margin_norm=margin_norm,
            elapsed_sec=elapsed,
        )
