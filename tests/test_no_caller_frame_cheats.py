"""test_no_caller_frame_cheats.py — Static AST scan for runtime inspection cheats.

Scans every .py file under hybrid/ for:
  - sys._getframe (caller introspection)
  - sys.argv inspection in model/forward-pass code
  - Any import of inspect.stack or inspect.currentframe

Fails if any are found.  Part of TICKET-006 cheat purge.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path


def _scan_file(filepath: Path) -> list[str]:
    """Return list of violation descriptions for this file, or empty."""
    violations = []
    try:
        tree = ast.parse(filepath.read_text())
    except SyntaxError:
        return []

    for node in ast.walk(tree):
        # Detect: sys._getframe() or sys._getframe(…) calls
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute):
                if (isinstance(node.func.value, ast.Name)
                        and node.func.value.id == 'sys'
                        and node.func.attr == '_getframe'):
                    violations.append(
                        f'{filepath}:{node.lineno}: sys._getframe() call'
                    )
                # Detect: inspect.currentframe() or inspect.stack()
                if (isinstance(node.func.value, ast.Name)
                        and node.func.value.id == 'inspect'
                        and node.func.attr in ('currentframe', 'stack')):
                    violations.append(
                        f'{filepath}:{node.lineno}: inspect.{node.func.attr}() call'
                    )

        # Detect: sys.argv access (in any context — overly strict but safe)
        if isinstance(node, ast.Attribute):
            if (isinstance(node.value, ast.Name)
                    and node.value.id == 'sys'
                    and node.attr == 'argv'):
                # Allow in argparse-only files (they use sys.argv in main())
                # Check if this is inside a function that looks like a CLI entry
                violations.append(
                    f'{filepath}:{node.lineno}: sys.argv access'
                )

    return violations


def test_no_sys_getframe_in_hybrid():
    """No file under hybrid/ uses sys._getframe or inspect.currentframe."""
    hybrid_dir = Path(__file__).resolve().parents[1]
    violations = []

    for py_file in hybrid_dir.rglob('*.py'):
        if '__pycache__' in str(py_file):
            continue
        if '.pytest_cache' in str(py_file):
            continue
        file_violations = _scan_file(py_file)
        # Filter: sys.argv in main() functions or after `if __name__` is OK
        file_violations = [v for v in file_violations
                           if 'sys.argv access' not in v
                           or 'capability_pipeline' not in str(py_file)
                           and 'work_ticket' not in str(py_file)]
        violations.extend(file_violations)

    if violations:
        msg = 'Found runtime inspection cheats:\n' + '\n'.join(violations)
        raise AssertionError(msg)


def test_no_clever_weights_function():
    """No function named _get_clever_weights exists in the codebase."""
    hybrid_dir = Path(__file__).resolve().parents[1]
    self_path = Path(__file__).resolve()
    for py_file in hybrid_dir.rglob('*.py'):
        if '__pycache__' in str(py_file):
            continue
        if py_file.resolve() == self_path:
            continue
        try:
            tree = ast.parse(py_file.read_text())
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                if 'clever' in node.name.lower():
                    raise AssertionError(
                        f'{py_file}:{node.lineno}: function {node.name}() — '
                        f'suspicious name'
                    )


if __name__ == '__main__':
    test_no_sys_getframe_in_hybrid()
    test_no_clever_weights_function()
    print('PASS: No runtime inspection cheats found.')
