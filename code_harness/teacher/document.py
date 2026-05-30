"""Phase 2: Generate per-weakness markdown documentation.

Reads weaknesses/catalog.json and writes a detailed .md file for each weakness.
"""
from __future__ import annotations

import json
from pathlib import Path


def _format_problem(prompt: str, max_lines: int = 10) -> str:
    lines = prompt.strip().split("\n")
    if len(lines) <= max_lines:
        return prompt.strip()
    return "\n".join(lines[:max_lines]) + "\n..."

def _format_code(code: str, max_lines: int = 15) -> str:
    lines = code.strip().split("\n")
    if len(lines) <= max_lines:
        return code.strip()
    return "\n".join(lines[:max_lines]) + "\n..."


def generate_weakness_docs(catalog_path: Path | str, output_dir: Path | str) -> list[Path]:
    catalog_path = Path(catalog_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    catalog = json.loads(catalog_path.read_text())
    weaknesses = catalog.get("weaknesses", [])

    written: list[Path] = []
    for w in weaknesses:
        wid = w["weakness_id"]
        failures = [f for f in catalog.get("failures", [])
                    if f.get("weakness_id") == wid]

        doc = f"""# Weakness: {w['weakness_name']}

**ID:** `{wid}`
**Severity:** {w.get('severity', 'unknown')}
**Category:** {w.get('category', 'unknown')}
**Failure Count:** {w.get('failure_count', 0)}/{catalog.get('total_problems', '?')}

## Description

{w.get('description', 'No description available.')}

## Failing Problems

"""
        for f in failures:
            doc += f"""### {f.get('task_id', '?')}

**Prompt:**
```python
{_format_problem(f.get('prompt', ''))}
```

**Model Output:**
```python
{_format_code(f.get('model_output', ''))}
```

**Expected:**
```python
{_format_code(f.get('expected', ''))}
```

"""

        doc += """## Training Strategy

1. **Phase A (Canonical CE):** Train on DeepSeek-synthesized examples targeting this weakness.
   Masked cross-entropy on canonical solutions — warm start with stable gradients.

2. **Phase B (RFT Refinement):** Generate candidate completions from the target model on
   the synthesized problems. Train ONLY on the model's own passing rollouts to close the
   off-policy gap between training and inference.

## Expected Outcome

After training, the cartridge should fix the failing problems listed above without
introducing regressions on problems that already pass.
"""
        out_path = output_dir / f"{wid}.md"
        out_path.write_text(doc)
        written.append(out_path)

    return written


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalog", default=str(Path.home() / "code_harness" / "weaknesses" / "catalog.json"))
    ap.add_argument("--output-dir", default=str(Path.home() / "code_harness" / "weaknesses"))
    args = ap.parse_args()

    paths = generate_weakness_docs(args.catalog, args.output_dir)
    print(f"Wrote {len(paths)} weakness docs to {args.output_dir}")


if __name__ == "__main__":
    main()
