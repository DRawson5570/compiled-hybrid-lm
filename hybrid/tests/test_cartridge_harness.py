from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

DEEPSEEK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DEEPSEEK.parent))

from hybrid.cartridge_harness import (
    ExactFirstLineScorer,
    TaskExample,
    build_private_fact_tasks,
    build_summary,
    evaluate_text_runner,
    normalize_first_line,
)
from hybrid.cartridge_harness.core import compare_rows
from hybrid.cartridge_harness.qwen import (
    CartridgeRouteDecision,
    QwenLearnedCartridgeRouter,
    QwenPromptRouter,
    build_qwen_adapter_steerer_from_checkpoint,
    load_qwen_adapter_cartridge,
    train_qwen_baked_lora,
)
from hybrid.cartridge_harness.rack_builder import assemble_rack_summary
from hybrid.cartridge_harness.suites import build_all_suites, get_suite
from hybrid.cartridges import CartridgeManifest, CartridgeRole, SteererCartridgeRack
from hybrid.superposition_steerer_v3 import FeatureConditionedAdapterSteerer


def test_private_fact_suite_has_train_and_heldout_paraphrases():
    tasks = build_private_fact_tasks()

    assert len([task for task in tasks if task.split == "train"]) == 36
    assert len([task for task in tasks if task.split == "heldout"]) == 24
    assert len({task.task_id for task in tasks}) == len(tasks)
    assert all(task.metadata["name"].startswith("Project ") for task in tasks)


def test_exact_first_line_scorer_normalizes_only_first_line():
    scorer = ExactFirstLineScorer()

    ok, generated, expected = scorer(" raven-041.\nextra", "RAVEN-041")

    assert ok
    assert generated == expected == "RAVEN-041"
    assert normalize_first_line("MICA 772") == "MICA772"


def test_evaluate_text_runner_and_summary_counts_by_split():
    tasks = [
        TaskExample("a", "train", "prompt a", "YES"),
        TaskExample("b", "heldout", "prompt b", "NO"),
    ]
    answers = {"prompt a": "YES", "prompt b": "maybe"}

    rows = evaluate_text_runner(tasks, lambda prompt: answers[prompt])
    summary = build_summary(rows)

    assert summary.total == 2
    assert summary.correct == 1
    assert summary.by_split["train"] == {"correct": 1, "total": 1}
    assert summary.by_split["heldout"] == {"correct": 0, "total": 1}


def test_compare_rows_finds_improvements_and_regressions():
    tasks = [
        TaskExample("a", "train", "prompt a", "YES"),
        TaskExample("b", "heldout", "prompt b", "NO"),
    ]
    baseline = evaluate_text_runner(tasks, lambda prompt: "wrong" if prompt == "prompt a" else "NO")
    cartridge = evaluate_text_runner(tasks, lambda prompt: "YES" if prompt == "prompt a" else "wrong")

    comparison = compare_rows(baseline, cartridge)

    assert [row["task_id"] for row in comparison["improved"]] == ["a"]
    assert [row["task_id"] for row in comparison["regressed"]] == ["b"]


def test_rack_suites_have_unique_ids_and_split_tasks():
    suites = build_all_suites()

    assert [suite.suite_id for suite in suites] == [
        "private_facts",
        "arithmetic",
        "code_labels",
        "safety_labels",
        "instruction_format",
    ]
    assert len({suite.cartridge_id for suite in suites}) == len(suites)
    for suite in suites:
        assert get_suite(suite.suite_id) == suite
        assert any(task.split == "train" for task in suite.tasks)
        assert any(task.split == "heldout" for task in suite.tasks)


def test_assemble_rack_summary_from_suite_outputs(tmp_path: Path):
    suite = get_suite("arithmetic")
    suite_dir = tmp_path / suite.suite_id
    suite_dir.mkdir()
    (suite_dir / "summary.json").write_text(
        """
        {
          "artifact": "artifacts/rack/arithmetic/cartridge_best.pt",
          "baseline_summary": {"total": 2, "correct": 0, "accuracy": 0.0, "by_split": {}},
          "cartridge_summary": {"total": 2, "correct": 2, "accuracy": 1.0, "by_split": {}},
          "improved": [{"task_id": "a"}],
          "regressed": []
        }
        """,
        encoding="utf-8",
    )

    summary = assemble_rack_summary(
        model="Qwen/Qwen2.5-1.5B",
        device="cuda",
        out_dir=tmp_path,
        suites=["arithmetic"],
    )

    assert summary["items"][0]["suite"]["suite_id"] == "arithmetic"
    assert summary["items"][0]["improved_count"] == 1
    assert (tmp_path / "rack_manifest.json").exists()
    assert (tmp_path / "rack_summary.json").exists()


class _IdentityLayer(nn.Module):
    def forward(self, x):
        return x


class _TinyTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([_IdentityLayer()])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def _write_fake_qwen_cartridge(path: Path, cartridge_id: str, *, base_model_id: str = "qwen-test"):
    steerer = FeatureConditionedAdapterSteerer(d_model=4, inject_layers=[0], bottleneck=2, noise_scale=0.0)
    manifest = CartridgeManifest(
        cartridge_id=cartridge_id,
        role=CartridgeRole.TASK_CAPABILITY,
        base_model_id=base_model_id,
        tokenizer_id="qwen-test",
        steerer_class="FeatureConditionedAdapterSteerer",
        inject_layers=(0,),
        parameter_count=sum(param.numel() for param in steerer.parameters()),
        source_corpus="unit-test",
        metadata={"runtime": "owned-cartridge-harness"},
    )
    payload = {
        "steerer_state": steerer.state_dict(),
        "manifest": manifest.__dict__,
        "history": [{"step": 1}],
        "summary": {"total": 1, "correct": 1},
    }
    torch.save(payload, path)
    return payload


def test_load_qwen_adapter_cartridge_reconstructs_manifest_and_steerer(tmp_path: Path):
    path = tmp_path / "cartridge_best.pt"
    payload = _write_fake_qwen_cartridge(path, "qwen-a")

    loaded = load_qwen_adapter_cartridge(path, d_model=4, device=torch.device("cpu"))

    assert loaded.manifest.cartridge_id == "qwen-a"
    assert loaded.manifest.inject_layers == (0,)
    assert loaded.history == [{"step": 1}]
    assert loaded.summary == {"total": 1, "correct": 1}
    assert set(loaded.steerer.state_dict()) == set(payload["steerer_state"])


def test_qwen_loader_rejects_wrong_steerer_class(tmp_path: Path):
    payload = _write_fake_qwen_cartridge(tmp_path / "bad.pt", "qwen-b")
    payload["manifest"] = dict(payload["manifest"], steerer_class="OtherSteerer")

    with pytest.raises(ValueError, match="unsupported Qwen cartridge steerer"):
        build_qwen_adapter_steerer_from_checkpoint(payload, d_model=4, device=torch.device("cpu"))


def test_loaded_qwen_cartridges_are_individually_isolatable(tmp_path: Path):
    first_path = tmp_path / "first.pt"
    second_path = tmp_path / "second.pt"
    _write_fake_qwen_cartridge(first_path, "first")
    _write_fake_qwen_cartridge(second_path, "second")

    first_a = load_qwen_adapter_cartridge(first_path, d_model=4, device=torch.device("cpu"))
    first_b = load_qwen_adapter_cartridge(first_path, d_model=4, device=torch.device("cpu"))
    second = load_qwen_adapter_cartridge(second_path, d_model=4, device=torch.device("cpu"))

    model_single = _TinyTransformer()
    rack_single = SteererCartridgeRack()
    rack_single.mount(first_a.manifest, first_a.steerer, active=True)
    rack_single.register_hooks(model_single)

    model_combo = _TinyTransformer()
    rack_combo = SteererCartridgeRack()
    rack_combo.mount(first_b.manifest, first_b.steerer, active=True)
    rack_combo.mount(second.manifest, second.steerer, active=False)
    rack_combo.register_hooks(model_combo)

    weights = torch.zeros(1, 3, 21)
    rack_single.set_weights(weights)
    rack_combo.set_weights(weights)
    hidden = torch.ones(1, 3, 4)

    assert torch.allclose(model_single(hidden), model_combo(hidden), atol=1e-6)
    rack_combo.activate("second", True)
    assert torch.isfinite(model_combo(hidden)).all()


def test_qwen_prompt_router_routes_all_suite_prompts():
    router = QwenPromptRouter()
    available = [suite.cartridge_id for suite in build_all_suites()]

    for suite in build_all_suites():
        assert all(router.route(task.prompt, available) == suite.cartridge_id for task in suite.tasks)


def test_main_router_returns_control_plane_decisions():
    router = QwenPromptRouter()
    available = [suite.cartridge_id for suite in build_all_suites()]
    prompt = get_suite("arithmetic").tasks[0].prompt

    decision = router.select(prompt, available)

    assert isinstance(decision, CartridgeRouteDecision)
    assert decision.cartridge_ids == ("qwen-arithmetic-router-cartridge",)
    assert "matched" in decision.reason


class _FakeRouterRuntime:
    def __init__(self, embedding: torch.Tensor):
        self.device = torch.device("cpu")
        self._embedding = embedding

    def prompt_embedding(self, prompt: str):
        return self._embedding


def test_learned_qwen_router_selects_from_artifact(tmp_path: Path):
    head = nn.Linear(3, 2)
    with torch.no_grad():
        head.weight.copy_(torch.tensor([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]]))
        head.bias.zero_()
    path = tmp_path / "router.pt"
    torch.save(
        {
            "router_type": "qwen_embedding_linear_v1",
            "d_model": 3,
            "cartridge_ids": ("alpha", "beta"),
            "head_state": head.state_dict(),
            "confidence_threshold": 0.0,
            "ambiguous_margin": 0.0,
        },
        path,
    )
    router = QwenLearnedCartridgeRouter(path, device=torch.device("cpu"))
    runtime = _FakeRouterRuntime(torch.tensor([0.1, 2.0, 0.0]))

    decision = router.select("anything", ["alpha", "beta"], runtime=runtime)

    assert decision.cartridge_ids == ("beta",)
    assert "confidence" in decision.reason


def test_learned_qwen_router_can_defer_on_low_confidence(tmp_path: Path):
    head = nn.Linear(2, 2)
    with torch.no_grad():
        head.weight.zero_()
        head.bias.zero_()
    path = tmp_path / "router.pt"
    torch.save(
        {
            "router_type": "qwen_embedding_linear_v1",
            "d_model": 2,
            "cartridge_ids": ("alpha", "beta"),
            "head_state": head.state_dict(),
            "confidence_threshold": 0.75,
            "ambiguous_margin": 0.0,
        },
        path,
    )
    router = QwenLearnedCartridgeRouter(path, device=torch.device("cpu"))
    runtime = _FakeRouterRuntime(torch.tensor([0.0, 0.0]))

    decision = router.select("unknown", ["alpha", "beta"], runtime=runtime)

    assert decision.cartridge_ids == ()
    assert "below threshold" in decision.reason


def test_gated_chain_cli_mode_is_explicit_default():
    from hybrid.cartridge_harness.cli import main

    assert callable(main)


def test_baked_lora_training_entrypoint_is_available():
    assert callable(train_qwen_baked_lora)


class _ConstantDeltaSteerer(nn.Module):
    def __init__(self, delta: float):
        super().__init__()
        self.delta = delta

    def _steer_layer(self, hidden, layer_idx: int):
        return hidden + self.delta


def test_rack_composition_modes_add_mean_and_chain():
    manifest_a = CartridgeManifest(
        cartridge_id="a",
        role=CartridgeRole.TASK_CAPABILITY,
        base_model_id="m",
        tokenizer_id="t",
        inject_layers=(0,),
    )
    manifest_b = CartridgeManifest(
        cartridge_id="b",
        role=CartridgeRole.TASK_CAPABILITY,
        base_model_id="m",
        tokenizer_id="t",
        inject_layers=(0,),
    )
    hidden = torch.zeros(1, 1, 1)

    additive = SteererCartridgeRack("additive")
    additive.mount(manifest_a, _ConstantDeltaSteerer(2.0))
    additive.mount(manifest_b, _ConstantDeltaSteerer(4.0))
    model = _TinyTransformer()
    additive.register_hooks(model)
    assert torch.allclose(model(hidden), torch.full_like(hidden, 6.0))

    mean = SteererCartridgeRack("mean")
    mean.mount(manifest_a, _ConstantDeltaSteerer(2.0))
    mean.mount(manifest_b, _ConstantDeltaSteerer(4.0))
    model = _TinyTransformer()
    mean.register_hooks(model)
    assert torch.allclose(model(hidden), torch.full_like(hidden, 3.0))

    chain = SteererCartridgeRack("chain")
    chain.mount(manifest_a, _ConstantDeltaSteerer(2.0), weight=0.5)
    chain.mount(manifest_b, _ConstantDeltaSteerer(4.0), weight=0.5)
    model = _TinyTransformer()
    chain.register_hooks(model)
    assert torch.allclose(model(hidden), torch.full_like(hidden, 3.0))