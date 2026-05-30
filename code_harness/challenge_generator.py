"""Extract coding challenges from the compiled-hybrid-lm codebase.

Produces a JSONL database of challenges at three difficulty tiers.
"""
import ast, json, sys, os, re
from pathlib import Path
from dataclasses import dataclass, field, asdict

REPO = Path("/home/drawson/deepseek_experiments")


@dataclass
class Challenge:
    id: str
    tier: int          # 1=function, 2=module, 3=debug
    prompt: str        # code prefix to complete
    expected: str      # reference completion
    test_code: str     # Python code that evaluates correctness
    source_file: str   # where it came from


def extract_function_challenges() -> list[Challenge]:
    """Tier 1: function completion from docstrings."""
    challenges = []
    for py_file in sorted(REPO.rglob("hybrid/*.py")):
        if "archive" in str(py_file) or "__pycache__" in str(py_file):
            continue
        try:
            source = py_file.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue

        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if len(node.body) < 2:
                continue
            body_start = node.body[0].lineno
            body_end = node.body[-1].end_lineno
            if body_end - body_start < 2:
                continue

            lines = source.split("\n")
            # Prompt: signature + docstring
            sig_end = body_start - 1
            prompt_lines = lines[node.lineno - 1:sig_end]
            docstring = ast.get_docstring(node)
            if not docstring and len(prompt_lines) < 3:
                continue

            prompt = "\n".join(prompt_lines) + "\n"
            expected_body = "\n".join(lines[body_start - 1:body_end]) + "\n"
            test_body = "\n".join(lines[body_start - 2:body_end + 2])

            func_name = node.name
            cid = f"func_{py_file.stem}_{func_name}"

            # Simple test: does it compile and define the right function?
            test_code = f"""\
import ast, sys
code = sys.stdin.read()
try:
    tree = ast.parse(code)
    funcs = [n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
    assert "{func_name}" in funcs, f"Missing {func_name}, found {{funcs}}"
    print("PASS")
except SyntaxError as e:
    print(f"FAIL: syntax error - {{e}}")
except AssertionError as e:
    print(f"FAIL: {{e}}")
"""

            challenges.append(Challenge(
                id=cid, tier=1, prompt=prompt,
                expected=expected_body, test_code=test_code,
                source_file=str(py_file.relative_to(REPO)),
            ))
    return challenges


def extract_bugfix_challenges() -> list[Challenge]:
    """Tier 3: bugfix tasks from CHANGELOG.md."""
    changelog = REPO / "CHANGELOG.md"
    if not changelog.exists():
        return []
    text = changelog.read_text(encoding="utf-8", errors="replace")
    challenges = []

    # Find entries that mention "fix" or "crash" or "bug"
    entries = text.split("\n- ")
    for i, entry in enumerate(entries):
        if not any(kw in entry.lower() for kw in ("fix", "crash", "bug", "oom", "error")):
            continue
        first_line = entry.split("\n")[0][:120]
        cid = f"bugfix_{i}"

        prompt = f"# Bug report from CHANGELOG.md:\n# {first_line}\n\n# Write a fix for this issue:\n"
        expected = "# The fix should address the root cause described above."
        test_code = f"""\
import sys
code = sys.stdin.read()
assert len(code) > 20, "Fix is too short"
print("PASS")
"""

        challenges.append(Challenge(
            id=cid, tier=3, prompt=prompt,
            expected=expected, test_code=test_code,
            source_file="CHANGELOG.md",
        ))

    return challenges


def extract_module_challenges() -> list[Challenge]:
    """Tier 2: module implementation from small standalone files."""
    challenges = []
    small_files = [
        "hybrid/ner_features.py",
        "hybrid/cartridges.py",
    ]
    for rel_path in small_files:
        fpath = REPO / rel_path
        if not fpath.exists():
            continue
        source = fpath.read_text(encoding="utf-8", errors="replace")
        lines = source.split("\n")
        midpoint = len(lines) // 2

        prompt = "\n".join(lines[:midpoint]) + "\n"
        expected = "\n".join(lines[midpoint:])
        cid = f"module_{Path(rel_path).stem}"

        test_code = f"""\
import ast
code = sys.stdin.read()
try:
    ast.parse(code)
    print("PASS")
except SyntaxError as e:
    print(f"FAIL: {{e}}")
"""

        challenges.append(Challenge(
            id=cid, tier=2, prompt=prompt,
            expected=expected, test_code=test_code,
            source_file=rel_path,
        ))
    return challenges


def main():
    out_dir = Path("/home/drawson/code_harness/challenges")
    out_dir.mkdir(parents=True, exist_ok=True)

    all_challenges = []
    all_challenges.extend(extract_function_challenges())
    all_challenges.extend(extract_module_challenges())
    all_challenges.extend(extract_bugfix_challenges())

    # Write JSONL
    out_file = out_dir / "challenges.jsonl"
    with open(out_file, "w", encoding="utf-8") as f:
        for c in all_challenges:
            f.write(json.dumps(asdict(c)) + "\n")

    tiers = {}
    for c in all_challenges:
        tiers[c.tier] = tiers.get(c.tier, 0) + 1
    print(f"Generated {len(all_challenges)} challenges: {tiers}")
    print(f"Saved to {out_file}")


if __name__ == "__main__":
    main()
