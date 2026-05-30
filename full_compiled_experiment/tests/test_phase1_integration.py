"""
Integration test for Phase 1: DSL, PyTorch reference, Triton backend, JIT compiler, executor.

Verifies:
1. AST construction and programmatic API
2. DSL text parser
3. Reference backend execution
4. Triton backend parity with reference
5. JIT compiler with caching
6. Executor forward pass
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np

from ucn.dsl.ast import (
    Activate,
    ActivateType,
    MatrixRef,
    Mix,
    Program,
    Project,
    Residual,
    Rotate,
    SubspaceRef,
    Transform,
)
from ucn.dsl.parser import parse_program
from ucn.backend.codegen.reference import ReferenceBackend
from ucn.backend.jit_compiler import JITCompiler
from ucn.runtime.executor import UCNExecutor


def make_toy_stdlib(d_model: int = 8) -> dict:
    rank = 2
    u = torch.randn(rank, d_model, dtype=torch.float32) * 0.1
    v = torch.randn(rank, d_model, dtype=torch.float32) * 0.1

    return {
        "copy_head": {
            "operator_type": "low_rank_projection",
            "u": u,
            "v": v,
        }
    }


def test_ast_construction():
    """AST construction via Python API works."""
    program = Program()
    program.add_stmt("t1", Mix(["x0", "x1"], [0.7, 0.3]))
    program.add_stmt("t2", Activate("t1", ActivateType.GELU))
    program.add_stmt("y", Rotate("t2", 1.5708, SubspaceRef(0, 4)))

    assert len(program.statements) == 3
    assert isinstance(program.statements[0].expr, Mix)
    assert isinstance(program.statements[1].expr, Activate)
    assert isinstance(program.statements[2].expr, Rotate)
    print("  PASS: AST construction")


def test_dsl_parser():
    """Text DSL parsing produces valid AST."""
    source = """
    y = mix([x0, x1], [0.7, 0.3])
    """
    program = parse_program(source)
    assert len(program.statements) == 1
    stmt = program.statements[0]
    assert stmt.target == "y"
    assert isinstance(stmt.expr, Mix)
    assert stmt.expr.inputs == ["x0", "x1"]
    assert abs(stmt.expr.weights[0] - 0.7) < 1e-6
    print("  PASS: DSL parser")


def test_parser_full():
    """Text DSL parser handles all primitives."""
    source = """
y1 = mix([x0, x1], [0.75, 0.25]);
y2 = project(y1, [0:512]);
y3 = transform(y2, stdlib.copy_head);
y4 = activate(y3, gelu);
y5 = residual([y4, x0])
"""
    program = parse_program(source)
    assert len(program.statements) == 5
    assert isinstance(program.statements[0].expr, Mix)
    assert isinstance(program.statements[1].expr, Project)
    assert isinstance(program.statements[2].expr, Transform)
    assert isinstance(program.statements[3].expr, Activate)
    assert isinstance(program.statements[4].expr, Residual)
    print("  PASS: Full DSL parser")


def test_reference_backend():
    """Reference backend executes programs correctly."""
    stdlib = make_toy_stdlib(d_model=8)
    backend = ReferenceBackend(stdlib_weights=stdlib, device="cpu", dtype=torch.float32)

    program = Program()
    program.add_stmt("y", Mix(["x0", "x1"], [0.7, 0.3]))

    x0 = torch.ones(8, dtype=torch.float32)
    x1 = torch.ones(8, dtype=torch.float32) * 2.0

    inputs = {"x0": x0, "x1": x1}
    outputs = backend.execute(program, inputs)

    expected = 0.7 * x0 + 0.3 * x1
    assert torch.allclose(outputs["y"], expected)
    print("  PASS: Reference backend mix")


def test_reference_activate():
    """Reference backend handles activations."""
    backend = ReferenceBackend(device="cpu", dtype=torch.float32)

    program = Program()
    program.add_stmt("y", Activate("x", ActivateType.GELU))

    x = torch.tensor([-1.0, 0.0, 1.0, 2.0], dtype=torch.float32)
    inputs = {"x": x}
    outputs = backend.execute(program, inputs)

    expected = torch.nn.functional.gelu(x)
    assert torch.allclose(outputs["y"], expected, atol=1e-5)
    print("  PASS: Reference backend activate")


def test_reference_transform():
    """Reference backend handles transform with low-rank weights."""
    d_model = 8
    rank = 2
    torch.manual_seed(42)
    u = torch.randn(rank, d_model, dtype=torch.float32)
    v = torch.randn(rank, d_model, dtype=torch.float32)

    stdlib = {
        "my_head": {
            "operator_type": "low_rank_projection",
            "u": u,
            "v": v,
        }
    }
    backend = ReferenceBackend(stdlib_weights=stdlib, device="cpu", dtype=torch.float32)

    program = Program()
    program.add_stmt("y", Transform("x", MatrixRef("stdlib", "my_head")))

    x = torch.ones(8, dtype=torch.float32)
    inputs = {"x": x}
    outputs = backend.execute(program, inputs)

    ref_output = x @ (u.T @ v)
    assert torch.allclose(outputs["y"], ref_output, atol=1e-5)
    print("  PASS: Reference backend transform")


def test_reference_rotate():
    """Reference backend handles rotation."""
    backend = ReferenceBackend(device="cpu", dtype=torch.float32)

    theta = 0.785
    program = Program()
    program.add_stmt("y", Rotate("x", theta, SubspaceRef(0, 4)))

    x = torch.tensor([1.0, 0.0, 2.0, 3.0, 5.0, 6.0, 7.0, 8.0], dtype=torch.float32)
    inputs = {"x": x}
    outputs = backend.execute(program, inputs)

    cos_t = torch.cos(torch.tensor(theta))
    sin_t = torch.sin(torch.tensor(theta))
    expected = x.clone()
    expected[0] = x[0] * cos_t - x[2] * sin_t
    expected[1] = x[1] * cos_t - x[3] * sin_t
    expected[2] = x[0] * sin_t + x[2] * cos_t
    expected[3] = x[1] * sin_t + x[3] * cos_t

    assert torch.allclose(outputs["y"], expected, atol=1e-5)
    print("  PASS: Reference backend rotate")


def test_reference_residual():
    """Reference backend handles residual (sum)."""
    backend = ReferenceBackend(device="cpu", dtype=torch.float32)

    program = Program()
    program.add_stmt("y", Residual(["a", "b", "c"]))

    a = torch.ones(8) * 1.0
    b = torch.ones(8) * 2.0
    c = torch.ones(8) * 3.0
    inputs = {"a": a, "b": b, "c": c}
    outputs = backend.execute(program, inputs)

    expected = a + b + c
    assert torch.allclose(outputs["y"], expected)
    print("  PASS: Reference backend residual")


def test_jit_compiler():
    """JIT compiler compiles and executes programs with caching."""
    stdlib = make_toy_stdlib(d_model=8)
    compiler = JITCompiler(
        stdlib_weights=stdlib,
        device="cpu",
        dtype=torch.float32,
        use_triton=False,
    )

    program = Program()
    program.add_stmt("y", Mix(["x0", "x1"], [0.7, 0.3]))

    x0 = torch.ones(8, dtype=torch.float32)
    x1 = torch.ones(8, dtype=torch.float32) * 2.0
    inputs = {"x0": x0, "x1": x1}

    outputs = compiler.compile_and_execute(program, inputs)
    expected = 0.7 * x0 + 0.3 * x1
    assert torch.allclose(outputs["y"], expected)

    outputs2 = compiler.compile_and_execute(program, inputs)
    assert torch.allclose(outputs2["y"], expected)
    print("  PASS: JIT compiler with caching")


def test_executor():
    """UCNExecutor forward pass works."""
    d_model = 8
    stdlib = make_toy_stdlib(d_model=d_model)
    executor = UCNExecutor(
        d_model=d_model,
        stdlib_weights=stdlib,
        device="cpu",
        dtype=torch.float32,
    )

    embeddings = torch.randn(2, 4, d_model, dtype=torch.float32)

    program = Program()
    program.add_stmt("y", Activate("input", ActivateType.GELU))

    output = executor.forward(embeddings, program=program)
    expected = torch.nn.functional.gelu(embeddings)

    assert output.shape == embeddings.shape
    assert torch.allclose(output, expected, atol=1e-5)
    print("  PASS: UCNExecutor forward")


def test_triton_parity():
    """Triton backend produces same results as reference (GPU only)."""
    if not torch.cuda.is_available():
        print("  SKIP: Triton parity (no GPU)")
        return

    d_model = 768
    torch.manual_seed(42)
    rank = 16
    u = torch.randn(rank, d_model, dtype=torch.float32) * 0.1
    v = torch.randn(rank, d_model, dtype=torch.float32) * 0.1

    stdlib = {
        "copy_head": {
            "operator_type": "low_rank_projection",
            "u": u,
            "v": v,
        }
    }

    ref = ReferenceBackend(stdlib_weights=stdlib, device="cpu", dtype=torch.float32)
    ref_cuda = ReferenceBackend(stdlib_weights=stdlib, device="cuda", dtype=torch.float32)

    program = Program()
    program.add_stmt("y", Mix(["x0", "x1"], [0.7, 0.3]))

    x0_cpu = torch.randn(d_model, dtype=torch.float32)
    x1_cpu = torch.randn(d_model, dtype=torch.float32)
    x0_cuda = x0_cpu.clone().cuda()
    x1_cuda = x1_cpu.clone().cuda()

    ref_output = ref.execute(program, {"x0": x0_cpu, "x1": x1_cpu})["y"]
    ref_cuda_output = ref_cuda.execute(program, {"x0": x0_cuda, "x1": x1_cuda})["y"]

    assert torch.allclose(ref_cuda_output.cpu(), ref_output, atol=1e-4), \
        f"CUDA reference doesn't match CPU reference: max diff={(ref_cuda_output.cpu() - ref_output).abs().max()}"

    compiler = JITCompiler(
        stdlib_weights=stdlib,
        device="cuda",
        dtype=torch.float32,
        use_triton=True,
    )
    triton_output = compiler.compile_and_execute(program, {"x0": x0_cuda, "x1": x1_cuda})["y"]

    max_diff = (triton_output.cpu() - ref_output).abs().max()
    print(f"  Triton vs Reference max difference: {max_diff.item():.6f}")

    assert torch.allclose(triton_output.cpu(), ref_output, atol=1e-3), \
        f"Triton output doesn't match reference: max diff={max_diff}"
    print("  PASS: Triton backend parity")


def test_triton_activate():
    """Triton activation kernel parity."""
    if not torch.cuda.is_available():
        print("  SKIP: Triton activate parity (no GPU)")
        return

    d_model = 4096
    torch.manual_seed(42)

    x_cpu = torch.randn(d_model, dtype=torch.float32)
    x_cuda = x_cpu.clone().cuda()

    ref = ReferenceBackend(device="cuda", dtype=torch.float32)
    program = Program()
    program.add_stmt("y", Activate("x", ActivateType.GELU))

    ref_output = ref.execute(program, {"x": x_cuda})["y"]

    compiler = JITCompiler(device="cuda", dtype=torch.float32, use_triton=True)
    triton_output = compiler.compile_and_execute(program, {"x": x_cuda})["y"]

    max_diff = (triton_output - ref_output).abs().max()
    print(f"  Triton GELU vs Reference max diff: {max_diff.item():.6f}")
    assert torch.allclose(triton_output, ref_output, atol=1e-3)
    print("  PASS: Triton activate parity")


def main():
    print("Phase 1 Integration Tests")
    print("=" * 40)

    test_ast_construction()
    test_dsl_parser()
    test_parser_full()
    test_reference_backend()
    test_reference_activate()
    test_reference_transform()
    test_reference_rotate()
    test_reference_residual()
    test_jit_compiler()
    test_executor()
    test_triton_parity()
    test_triton_activate()

    print("\nAll Phase 1 tests passed!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
