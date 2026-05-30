import torch

from hybrid.hf_deepseek import DeepSeekConfig, DeepSeekForCausalLM
from hybrid.superposition_steerer_v3 import SuperpositionSteererV3
from hybrid.train_4b_distributed import (
    _early_stop_metric_improved,
    _extract_steerer_state,
    _format_saved_status,
    _freeze_except,
    _control_state_dict,
    _restore_control_state,
    _resume_epoch_counters,
    _save_metric_checkpoints,
    _steerer_on_under_ppl,
    _steerer_control_parameters,
    _trainable_params,
    _trainable_surface_for_model,
    _uses_cmi_steerer,
)


def test_blind_and_steered_checkpoints_are_separate(tmp_path):
    blind_payload = {"epoch": 10, "eval_s": 20.0, "eval_b": 5.0}
    written = _save_metric_checkpoints(str(tmp_path), "b", blind_payload)

    assert str(tmp_path / "best_b.pt") in written
    assert str(tmp_path / "best.pt") not in written
    assert not (tmp_path / "best_s.pt").exists()
    assert not (tmp_path / "best.pt").exists()

    steered_payload = {"epoch": 11, "eval_s": 4.0, "eval_b": 6.0}
    written = _save_metric_checkpoints(str(tmp_path), "s", steered_payload)

    assert str(tmp_path / "best_s.pt") in written
    assert str(tmp_path / "best.pt") in written

    blind = torch.load(tmp_path / "best_b.pt", map_location="cpu", weights_only=False)
    steered = torch.load(tmp_path / "best_s.pt", map_location="cpu", weights_only=False)
    legacy = torch.load(tmp_path / "best.pt", map_location="cpu", weights_only=False)

    assert blind["checkpoint_kind"] == "blind_best"
    assert blind["epoch"] == 10
    assert steered["checkpoint_kind"] == "steered_best"
    assert steered["epoch"] == 11
    assert legacy["checkpoint_kind"] == "legacy_steered_best"
    assert legacy["epoch"] == 11


def test_combined_improvement_writes_both_metric_checkpoints(tmp_path):
    payload = {"epoch": 12, "eval_s": 3.0, "eval_b": 4.0}
    written = _save_metric_checkpoints(str(tmp_path), "bs", payload)

    assert str(tmp_path / "best_b.pt") in written
    assert str(tmp_path / "best_s.pt") in written
    assert str(tmp_path / "best.pt") in written

    assert torch.load(tmp_path / "best_b.pt", map_location="cpu", weights_only=False)["checkpoint_kind"] == "blind_best"
    assert torch.load(tmp_path / "best_s.pt", map_location="cpu", weights_only=False)["checkpoint_kind"] == "steered_best"


def test_legacy_best_tracks_steered_checkpoint_when_blind_improves_later(tmp_path):
    _save_metric_checkpoints(str(tmp_path), "s", {"epoch": 20, "eval_s": 4.0, "eval_b": 9.0})
    _save_metric_checkpoints(str(tmp_path), "b", {"epoch": 21, "eval_s": 5.0, "eval_b": 8.0})

    legacy = torch.load(tmp_path / "best.pt", map_location="cpu", weights_only=False)
    blind = torch.load(tmp_path / "best_b.pt", map_location="cpu", weights_only=False)

    assert legacy["checkpoint_kind"] == "legacy_steered_best"
    assert legacy["epoch"] == 20
    assert blind["checkpoint_kind"] == "blind_best"
    assert blind["epoch"] == 21


def test_early_stop_metric_improved_tracks_requested_metric():
    assert _early_stop_metric_improved("s", "steered")
    assert not _early_stop_metric_improved("b", "steered")
    assert _early_stop_metric_improved("b", "blind")
    assert not _early_stop_metric_improved("s", "blind")
    assert _early_stop_metric_improved("b", "either")
    assert _early_stop_metric_improved("s", "either")
    assert not _early_stop_metric_improved("", "either")
    assert not _early_stop_metric_improved("bs", "none")


def test_format_saved_status_names_checkpoint_updates_not_training():
    assert _format_saved_status("") == "-"
    assert _format_saved_status("s") == "on"
    assert _format_saved_status("b") == "off"
    assert _format_saved_status("bs") == "on,off"


def test_steerer_warmup_gate_uses_absolute_ppl_threshold():
    assert _steerer_on_under_ppl(50.0, 50.0)
    assert _steerer_on_under_ppl(49.9, 50.0)
    assert not _steerer_on_under_ppl(50.1, 50.0)
    assert not _steerer_on_under_ppl(float("inf"), 50.0)


def test_resume_epoch_counters_split_warmup_and_main_epochs():
    assert _resume_epoch_counters(None) == (0, 0, 0, True)
    assert _resume_epoch_counters({"epoch": 12}) == (12, 0, 12, True)
    assert _resume_epoch_counters({"epoch": 7, "neural_training_enabled": False}) == (0, 7, 7, False)
    assert _resume_epoch_counters({"epoch": 3, "main_epoch": 3, "warmup_epoch": 5, "total_epoch": 8, "neural_training_enabled": True}) == (3, 5, 8, True)


def test_extract_steerer_state_accepts_full_checkpoint_or_raw_state_dict():
    raw_state = {"steer_local": torch.ones(1), "gammas.0": torch.ones(1)}

    assert _extract_steerer_state({"steerer_state": raw_state}) is raw_state
    assert _extract_steerer_state(raw_state) is raw_state
    assert _extract_steerer_state({"state_dict": {"head_bias": torch.ones(1)}}) is None


def test_trainable_params_filters_frozen_parameters():
    trainable = torch.nn.Parameter(torch.ones(1), requires_grad=True)
    frozen = torch.nn.Parameter(torch.ones(1), requires_grad=False)

    assert _trainable_params([trainable, frozen]) == [trainable]


def test_steering_control_parameters_are_alpha_beta_gamma_only():
    steerer = SuperpositionSteererV3(d_model=16, inject_layers=[0, 1])
    control_params = _steerer_control_parameters(steerer)
    control_ids = {id(param) for param in control_params}
    named_controls = {name for name, param in steerer.named_parameters() if id(param) in control_ids}

    assert named_controls == {"alpha", "betas.local", "betas.mid", "betas.global", "gammas.0", "gammas.1"}

    _freeze_except(control_params, list(steerer.parameters()))
    assert all(param.requires_grad for param in control_params)
    assert not steerer.steer_local.requires_grad
    assert not steerer.local_mlp[0].weight.requires_grad


def test_control_state_restore_keeps_best_values():
    steerer = SuperpositionSteererV3(d_model=16, inject_layers=[0])
    controls = _steerer_control_parameters(steerer)
    with torch.no_grad():
        for idx, param in enumerate(controls):
            param.fill_(idx + 1)

    best_state = _control_state_dict(controls)
    with torch.no_grad():
        for param in controls:
            param.add_(100)

    _restore_control_state(controls, best_state)

    assert [float(param.item()) for param in controls] == [1.0, 2.0, 3.0, 4.0, 5.0]


def test_trainable_surface_names_separate_product_and_thesis_tracks():
    model = DeepSeekForCausalLM(DeepSeekConfig(vocab_size=32, d_model=16, n_layers=2, n_heads=4, d_ff=64, max_len=16))

    assert _trainable_surface_for_model(model, "head_bias").parameter_names == ("head_bias",)
    assert _trainable_surface_for_model(model, "cmi_steerer").parameter_names == ("head_bias",)

    full_names = _trainable_surface_for_model(model, "full_cmi_steerer").parameter_names
    assert "head_bias" in full_names
    assert "tok_emb.weight" in full_names
    assert "layers.0.ffn1.weight" in full_names
    assert "layers.0.q_proj.weight" in full_names

    top_names = _trainable_surface_for_model(model, "top1_cmi_steerer").parameter_names
    assert "head_bias" in top_names
    assert "ln_f.weight" in top_names
    assert "layers.1.ffn1.weight" in top_names
    assert "layers.1.q_proj.weight" in top_names
    assert "layers.0.ffn1.weight" not in top_names
    assert "tok_emb.weight" not in top_names

    top_zeroq_names = _trainable_surface_for_model(model, "top1_cmi_steerer", "zeroq").parameter_names
    assert "tok_emb.weight" in top_zeroq_names
    assert "pos_emb.weight" in top_zeroq_names

    assert not _uses_cmi_steerer("full")
    assert _uses_cmi_steerer("full_cmi_steerer")
    assert _uses_cmi_steerer("top2_cmi_steerer")