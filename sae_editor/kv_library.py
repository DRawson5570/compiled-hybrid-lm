"""Symbolic key-value patch library — store, search, compose, and deploy NRTCS patches."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import torch


@dataclass
class KVEntry:
    """A single key-value patch entry with full provenance metadata.

    Tensors are NOT serialized in the manifest. The manifest stores
    metadata only. Tensors live in entries/{entry_id}/keys.pt and
    entries/{entry_id}/values.pt.
    """

    entry_id: str
    description: str
    source_model: str
    source_model_d_model: int
    layer: int
    keys: torch.Tensor
    values: torch.Tensor
    tags: list[str] = field(default_factory=list)
    extraction_method: str = "manual"
    verification_cosine: float | None = None
    preview_cosine_shift: float | None = None
    created_at: str = ""
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()

    def to_metadata_dict(self) -> dict:
        return {
            "entry_id": self.entry_id,
            "description": self.description,
            "source_model": self.source_model,
            "source_model_d_model": self.source_model_d_model,
            "layer": self.layer,
            "tags": self.tags,
            "extraction_method": self.extraction_method,
            "verification_cosine": self.verification_cosine,
            "preview_cosine_shift": self.preview_cosine_shift,
            "created_at": self.created_at,
            "metadata": self.metadata,
        }

    @classmethod
    def from_metadata_dict(cls, d: dict, keys: torch.Tensor, values: torch.Tensor) -> "KVEntry":
        return cls(
            entry_id=d["entry_id"],
            description=d["description"],
            source_model=d["source_model"],
            source_model_d_model=d["source_model_d_model"],
            layer=d["layer"],
            keys=keys,
            values=values,
            tags=d.get("tags", []),
            extraction_method=d.get("extraction_method", "manual"),
            verification_cosine=d.get("verification_cosine"),
            preview_cosine_shift=d.get("preview_cosine_shift"),
            created_at=d.get("created_at", ""),
            metadata=d.get("metadata", {}),
        )

    def to_edit_dict(self) -> dict[int, dict[str, torch.Tensor]]:
        return {self.layer: {"keys": self.keys, "values": self.values}}


class KVLibrary:
    """Per-model key-value patch library.

    Directory structure:
        {path}/manifest.json          ← metadata index (no tensors)
        {path}/entries/{entry_id}/
            keys.pt                   ← (N, d_in) tensor
            values.pt                 ← (N, d_out) tensor
    """

    def __init__(self, path: str):
        self.path = Path(path)
        self.manifest_path = self.path / "manifest.json"
        self.entries_dir = self.path / "entries"
        self._entries: dict[str, KVEntry] = {}
        self._model_name: str = ""
        if self.manifest_path.exists():
            self.load()

    def add(self, entry: KVEntry, save_immediately: bool = True):
        if entry.entry_id in self._entries:
            raise KeyError(f"Entry '{entry.entry_id}' already exists. Remove it first.")
        self._entries[entry.entry_id] = entry
        if save_immediately:
            self.save()

    def get(self, entry_id: str) -> KVEntry:
        if entry_id not in self._entries:
            raise KeyError(
                f"Entry '{entry_id}' not found. Available: {list(self._entries.keys())}"
            )
        return self._entries[entry_id]

    def remove(self, entry_id: str):
        if entry_id not in self._entries:
            raise KeyError(f"Entry '{entry_id}' not found.")
        del self._entries[entry_id]
        entry_dir = self.entries_dir / entry_id
        if entry_dir.exists():
            keys_path = entry_dir / "keys.pt"
            values_path = entry_dir / "values.pt"
            if keys_path.exists():
                keys_path.unlink()
            if values_path.exists():
                values_path.unlink()
            entry_dir.rmdir()
        self.save()

    def list(self) -> list[str]:
        return sorted(self._entries.keys())

    def save(self):
        self.entries_dir.mkdir(parents=True, exist_ok=True)

        manifest_entries = {}
        for entry_id, entry in self._entries.items():
            manifest_entries[entry_id] = entry.to_metadata_dict()

            entry_dir = self.entries_dir / entry_id
            entry_dir.mkdir(exist_ok=True)
            torch.save(entry.keys.detach().cpu(), entry_dir / "keys.pt")
            torch.save(entry.values.detach().cpu(), entry_dir / "values.pt")

        manifest = {
            "library_version": "1.0.0",
            "source_model": self._model_name or self._entries[list(self._entries.keys())[0]].source_model if self._entries else "",
            "source_model_d_model": next(iter(self._entries.values())).source_model_d_model if self._entries else 0,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "entries": manifest_entries,
        }

        with open(self.manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

    def load(self):
        if not self.manifest_path.exists():
            return

        with open(self.manifest_path) as f:
            manifest = json.load(f)

        self._model_name = manifest.get("source_model", "")

        self._entries = {}
        for entry_id, meta in manifest.get("entries", {}).items():
            entry_dir = self.entries_dir / entry_id
            keys_path = entry_dir / "keys.pt"
            values_path = entry_dir / "values.pt"

            if not keys_path.exists() or not values_path.exists():
                continue

            keys = torch.load(keys_path, weights_only=True)
            values = torch.load(values_path, weights_only=True)
            self._entries[entry_id] = KVEntry.from_metadata_dict(meta, keys, values)

    def search(self, query: str = "", tags: list[str] | None = None) -> list[KVEntry]:
        results = []
        for entry in self._entries.values():
            score = 0
            if tags:
                tag_set = set(entry.tags)
                query_set = set(tags)
                overlap = tag_set & query_set
                if not overlap:
                    continue
                score += len(overlap) * 10
                if tag_set == query_set:
                    score += 100

            if query:
                q = query.lower()
                if q in entry.description.lower():
                    score += 5
                if q in entry.entry_id.lower():
                    score += 3
                for tag in entry.tags:
                    if q in tag.lower():
                        score += 2

            if tags is None and query == "":
                score = 1

            if score > 0 or (tags is None and query == ""):
                results.append((score, entry))

        results.sort(key=lambda x: x[0], reverse=True)
        return [entry for _, entry in results]

    def is_compatible(self, entry_id: str, d_model: int) -> bool:
        entry = self.get(entry_id)
        return entry.source_model_d_model == d_model

    def compose(
        self,
        entry_ids: list[str],
        original_features: dict[int, torch.Tensor] | None = None,
    ) -> dict[int, dict[str, torch.Tensor]]:
        from sae_editor.recompiler import orthogonal_projection

        merged: dict[int, dict[str, list[torch.Tensor]]] = {}
        layer_dims: dict[int, int] = {}

        for eid in entry_ids:
            entry = self.get(eid)
            layer = entry.layer

            if layer in layer_dims and layer_dims[layer] != entry.source_model_d_model:
                raise ValueError(
                    f"Entry '{eid}' has d_in={entry.source_model_d_model} but layer "
                    f"{layer} already has entries with d_in={layer_dims[layer]}. "
                    f"All entries at the same layer must have matching d_model."
                )
            layer_dims[layer] = entry.source_model_d_model

            if layer not in merged:
                merged[layer] = {"keys": [], "values": []}

            new_keys = entry.keys
            if merged[layer]["keys"]:
                existing = torch.cat(merged[layer]["keys"], dim=0)
                new_keys = orthogonal_projection(new_keys.T, existing.T, eps=1e-4).T

            merged[layer]["keys"].append(new_keys)
            merged[layer]["values"].append(entry.values)

        result = {}
        for layer, tensors in merged.items():
            result[layer] = {
                "keys": torch.cat(tensors["keys"], dim=0),
                "values": torch.cat(tensors["values"], dim=0),
            }

        return result

    def preview(
        self,
        entry_ids: str | list[str],
        model,
        tokenizer,
        prompts: list[str],
        strength: float = 1.0,
        gate_threshold: float = 0.3,
        **kwargs,
    ):
        from sae_editor.pipeline import NRTCSPipeline

        if isinstance(entry_ids, str):
            entry_ids = [entry_ids]

        edits = self.compose(entry_ids)
        pipeline = NRTCSPipeline(eps=1e-3)
        return pipeline.preview(
            edits=edits, model=model, tokenizer=tokenizer,
            prompts=prompts, strength=strength, gate_threshold=gate_threshold,
            **kwargs,
        )

    def splice(
        self,
        entry_ids: str | list[str],
        safetensors_path: str,
        arch=None,
    ):
        from sae_editor.pipeline import NRTCSPipeline

        if isinstance(entry_ids, str):
            entry_ids = [entry_ids]

        edits = self.compose(entry_ids)
        pipeline = NRTCSPipeline(eps=1e-3)

        patches = pipeline.compile_from_uvm_edits(edits)
        pipeline.splice_patches(safetensors_path, patches, arch=arch)
