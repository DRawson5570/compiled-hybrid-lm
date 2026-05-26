import torch

from hybrid.train_4b_distributed import _early_stop_metric_improved, _save_metric_checkpoints


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