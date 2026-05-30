"""surfaces/registry.py — Component registry tracking installed modifications.

Part of TICKET-004. Tracks component_id, type, install timestamp, and restore
checksum for every injected component. Enables retract + compose.
"""
from __future__ import annotations

import hashlib
import time
from typing import Any


class ComponentRegistry:
    """Tracks installed components with version, timestamp, and checksum."""

    def __init__(self):
        self._components: dict[str, dict] = {}
        self._order: list[str] = []  # installation order

    def register(self, component_id: str, component_type: str,
                 metadata: dict | None = None) -> str:
        """Register a new component. Returns the component_id."""
        entry = {
            'component_id': component_id,
            'type': component_type,
            'installed_at': time.time(),
            'installed_at_iso': time.strftime('%Y-%m-%dT%H:%M:%S'),
            'version': 1,
            'metadata': metadata or {},
            'restore_checksum': None,  # set by inject on install
            'active': True,
        }
        self._components[component_id] = entry
        self._order.append(component_id)
        return component_id

    def set_checksum(self, component_id: str, checksum: str):
        if component_id in self._components:
            self._components[component_id]['restore_checksum'] = checksum

    def deactivate(self, component_id: str):
        if component_id in self._components:
            self._components[component_id]['active'] = False

    def activate(self, component_id: str):
        if component_id in self._components:
            self._components[component_id]['active'] = True

    def get(self, component_id: str) -> dict | None:
        return self._components.get(component_id)

    def list_active(self) -> list[str]:
        return [cid for cid in self._order
                if self._components.get(cid, {}).get('active', False)]

    def list_all(self) -> list[dict]:
        return [self._components[cid] for cid in self._order]

    def snapshot(self) -> dict:
        """Return JSON-serializable snapshot of current state."""
        return {
            'components': {cid: dict(v) for cid, v in self._components.items()},
            'order': list(self._order),
            'active_count': len(self.list_active()),
        }

    @staticmethod
    def compute_model_checksum(model) -> str:
        """Compute a deterministic checksum of model parameters."""
        import torch
        hasher = hashlib.sha256()
        for name, param in sorted(model.state_dict().items()):
            hasher.update(name.encode())
            hasher.update(param.cpu().numpy().tobytes())
        return hasher.hexdigest()[:16]
