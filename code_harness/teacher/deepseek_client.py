"""DeepSeek v4 Flash API client for teacher operations.

Uses the OpenAI-compatible API at api.deepseek.com with key from ~/api_keys/deepseek.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import openai


class DeepSeekTeacher:
    def __init__(self, api_key_path: str = "~/api_keys/deepseek",
                 model: str = "deepseek-chat", base_url: str = "https://api.deepseek.com",
                 max_tokens: int = 4096):
        key = Path(api_key_path).expanduser().read_text().strip()
        self.client = openai.OpenAI(api_key=key, base_url=base_url)
        self.model = model
        self.max_tokens = max_tokens
        self._rate_limit_retries = 3

    def _call(self, system: str, user: str, temperature: float = 0.0,
              response_json: bool = False) -> str:
        kwargs: dict[str, Any] = dict(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=temperature,
            max_tokens=self.max_tokens,
        )
        if response_json:
            kwargs["response_format"] = {"type": "json_object"}

        for attempt in range(self._rate_limit_retries):
            try:
                resp = self.client.chat.completions.create(**kwargs)
                return resp.choices[0].message.content or ""
            except openai.RateLimitError:
                wait = 2 ** attempt
                time.sleep(wait)
            except openai.APIError as e:
                if attempt == self._rate_limit_retries - 1:
                    raise
                time.sleep(2 ** attempt)
        return ""

    def analyze_failure(self, prompt: str, expected: str, model_output: str) -> dict[str, str]:
        system = (
            "You are auditing a smaller code generation model's failures on HumanEval. "
            "Classify each failure into a narrow, specific coding weakness category. "
            "Be precise — name the exact missing capability, not generic terms. "
            "Examples of good weakness_ids: 'fails_on_recursion', 'hallucinates_imports', "
            "'incorrect_return_type', 'off_by_one_loops', 'list_comprehension_syntax', "
            "'unclosed_file_handles', 'dict_key_error', 'type_annotation_mismatch'. "
            "Always respond with valid JSON."
        )
        user = (
            f"HUMANEVAL PROBLEM PROMPT:\n{prompt}\n\n"
            f"EXPECTED SOLUTION:\n{expected}\n\n"
            f"MODEL'S ACTUAL OUTPUT:\n{model_output}\n\n"
            "Analyze this failure. Return JSON with keys: "
            "weakness_id (snake_case, specific), weakness_name (human-readable), "
            "description (one sentence), severity (high|medium|low), category (string)."
        )
        response = self._call(system, user, temperature=0.0, response_json=True)
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            return {"weakness_id": "unclassified", "weakness_name": "Unclassified",
                    "description": response[:200], "severity": "medium", "category": "unknown"}

    def synthesize_examples(self, weakness_name: str, weakness_description: str,
                            count: int = 20) -> list[dict[str, str]]:
        system = (
            "You are generating training data for a code model that has a specific weakness. "
            "Create novel Python function-completion coding problems that specifically test "
            "and teach the target skill. Each problem must be self-contained, have a clear "
            "function signature with docstring as the prompt, a correct canonical solution, "
            "and assert-based test code using a `check(func_name)` pattern. "
            "Vary difficulty from easy to hard. Do NOT copy or closely mimic known "
            "HumanEval or MBPP benchmark problems. Always respond with valid JSON array."
        )
        user = (
            f"Weakness: {weakness_name}\n"
            f"Description: {weakness_description}\n\n"
            f"Generate exactly {count} Python coding problems targeting this specific weakness. "
            "Each problem must have: prompt (function signature + docstring to complete), "
            "expected (canonical solution code string), test_code (Python assert-based test "
            "that calls `check(func_name)` at end), entry_point (function name).\n\n"
            "Output a JSON array of objects."
        )
        response = self._call(system, user, temperature=0.7, response_json=True)
        try:
            parsed = json.loads(response)
            if isinstance(parsed, dict):
                for key in ("problems", "examples", "results", "data", "items"):
                    val = parsed.get(key)
                    if isinstance(val, list):
                        return val
                for val in parsed.values():
                    if isinstance(val, list):
                        return val
                return []
            return parsed if isinstance(parsed, list) else []
        except json.JSONDecodeError:
            return []
