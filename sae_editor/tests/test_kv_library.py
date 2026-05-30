"""KVLibrary tests: search, compose, compatibility, preview."""

from __future__ import annotations

from pathlib import Path
import tempfile

import pytest
import torch

from sae_editor.kv_library import KVEntry, KVLibrary


def _prebuilt_path():
    return str(Path(__file__).parent.parent / "patches" / "qwen2.5-0.5b")


class TestKVLibrary:
    @pytest.fixture
    def datadir(self):
        with tempfile.TemporaryDirectory() as d:
            yield d

    @pytest.fixture
    def lib(self, datadir):
        return KVLibrary(datadir)

    def test_add_save_load_round_trip(self, datadir):
        keys = torch.randn(3, 64)
        values = torch.randn(3, 64)
        entry = KVEntry(
            entry_id="test-entry-v1",
            description="Test entry",
            source_model="test-model",
            source_model_d_model=64,
            layer=3,
            keys=keys,
            values=values,
            tags=["test"],
        )

        lib = KVLibrary(datadir)
        lib.add(entry, save_immediately=True)

        lib2 = KVLibrary(datadir)
        loaded = lib2.get("test-entry-v1")
        assert loaded.entry_id == "test-entry-v1"
        assert loaded.source_model_d_model == 64
        assert loaded.layer == 3
        assert torch.allclose(loaded.keys, keys, atol=1e-6)
        assert torch.allclose(loaded.values, values, atol=1e-6)

    def test_list_returns_ids(self, lib):
        lib.add(KVEntry("a", "A", "m", 64, 0, torch.randn(2, 64), torch.randn(2, 64)))
        lib.add(KVEntry("b", "B", "m", 64, 0, torch.randn(2, 64), torch.randn(2, 64)))
        assert lib.list() == ["a", "b"]

    def test_search_by_tag(self, lib):
        lib.add(KVEntry("e1", "Factual", "m", 64, 0, torch.randn(2, 64), torch.randn(2, 64),
                         tags=["factual", "geo"]))
        lib.add(KVEntry("e2", "Safety", "m", 64, 0, torch.randn(2, 64), torch.randn(2, 64),
                         tags=["safety"]))
        lib.add(KVEntry("e3", "Both", "m", 64, 0, torch.randn(2, 64), torch.randn(2, 64),
                         tags=["factual", "safety"]))

        r1 = lib.search(tags=["factual"])
        ids1 = {e.entry_id for e in r1}
        assert ids1 == {"e1", "e3"}

        r2 = lib.search(tags=["safety"])
        ids2 = {e.entry_id for e in r2}
        assert ids2 == {"e2", "e3"}

        r3 = lib.search(query="factual")
        ids3 = {e.entry_id for e in r3}
        assert "e1" in ids3

    def test_search_empty(self, lib):
        assert lib.search() == []

    def test_compose_two_entries_same_layer(self, lib):
        lib.add(KVEntry("e1", "A", "m", 64, 5, torch.randn(2, 64), torch.randn(2, 64)))
        lib.add(KVEntry("e2", "B", "m", 64, 5, torch.randn(3, 64), torch.randn(3, 64)))

        merged = lib.compose(["e1", "e2"])
        assert 5 in merged
        assert merged[5]["keys"].shape == (5, 64)
        assert merged[5]["values"].shape == (5, 64)

    def test_compose_two_entries_different_layers(self, lib):
        lib.add(KVEntry("e1", "A", "m", 64, 0, torch.randn(2, 64), torch.randn(2, 64)))
        lib.add(KVEntry("e2", "B", "m", 64, 5, torch.randn(2, 64), torch.randn(2, 64)))

        merged = lib.compose(["e1", "e2"])
        assert set(merged.keys()) == {0, 5}

    def test_is_compatible_matches(self, lib):
        lib.add(KVEntry("e1", "A", "m", 64, 0, torch.randn(2, 64), torch.randn(2, 64)))
        assert lib.is_compatible("e1", d_model=64)

    def test_is_compatible_mismatch(self, lib):
        lib.add(KVEntry("e1", "A", "m", 64, 0, torch.randn(2, 64), torch.randn(2, 64)))
        assert not lib.is_compatible("e1", d_model=128)

    def test_remove(self, lib):
        lib.add(KVEntry("e1", "A", "m", 64, 0, torch.randn(2, 64), torch.randn(2, 64)),
                save_immediately=False)
        assert "e1" in lib.list()
        lib.remove("e1")
        assert "e1" not in lib.list()

    def test_preview_from_library(self, lib, synthetic_model):
        lib.add(KVEntry("e1", "Preview test", "m", 64, 0,
                         torch.randn(2, 64), torch.randn(2, 64)))
        from sae_editor.preview import MultiLayerPreviewResult

        result = lib.preview("e1", synthetic_model, None, ["hello"])
        assert isinstance(result, MultiLayerPreviewResult)

    def test_prebuilt_library_loads(self):
        lib = KVLibrary(_prebuilt_path())
        ids = lib.list()
        assert len(ids) >= 1, f"Expected at least 1 pre-built entry, got {len(ids)}"
        for eid in ids:
            entry = lib.get(eid)
            assert entry.source_model_d_model == 896
            assert isinstance(entry.keys, torch.Tensor)
