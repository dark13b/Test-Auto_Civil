"""Single access layer for persisted runtime state."""

from __future__ import annotations

import copy
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


STATE_SCHEMA_VERSION = 1
STATE_KIND = "lab_state"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def build_default_lab_state() -> dict[str, Any]:
    return {
        "surface_version": 1,
        "accepted_experiments": [],
        "recent_kept_families": [],
    }


def default_runtime_state_path(project_root: Path) -> Path:
    return Path(project_root) / "outputs" / "state" / "runtime_state.json"


class JSONStateStore:
    """Persist mutable runtime state as one JSON document."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def exists(self) -> bool:
        return self.path.exists()

    def load_payload(self) -> dict[str, Any]:
        if not self.path.exists():
            raise FileNotFoundError(f"State store not found: {self.path}")
        with self.path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict):
            raise ValueError(f"State store payload must be a JSON object: {self.path}")
        return payload

    def load_state(self) -> dict[str, Any]:
        payload = self.load_payload()
        state = payload.get("state")
        if not isinstance(state, dict):
            raise ValueError(f"State store payload is missing a dictionary state body: {self.path}")
        return copy.deepcopy(state)

    def save_state(self, state: dict[str, Any], *, migrated_from: str | None = None) -> dict[str, Any]:
        existing_payload: dict[str, Any] = {}
        if self.path.exists():
            try:
                existing_payload = self.load_payload()
            except Exception:
                existing_payload = {}
        timestamp = _utc_now_iso()
        payload = {
            "schema_version": STATE_SCHEMA_VERSION,
            "state_kind": STATE_KIND,
            "created_at": existing_payload.get("created_at", timestamp),
            "updated_at": timestamp,
            "state": copy.deepcopy(state),
        }
        metadata = dict(existing_payload.get("metadata", {})) if isinstance(existing_payload.get("metadata"), dict) else {}
        if migrated_from is not None:
            metadata["migrated_from"] = str(migrated_from)
        if metadata:
            payload["metadata"] = metadata
        self._write_payload(payload)
        return copy.deepcopy(payload)

    def load_or_initialize(self, default_state: dict[str, Any] | None = None) -> dict[str, Any]:
        if self.path.exists():
            return self.load_state()
        state = copy.deepcopy(default_state if default_state is not None else build_default_lab_state())
        self.save_state(state)
        return state

    def load_or_migrate(
        self,
        *,
        legacy_path: Path | None,
        legacy_loader: Callable[[Path], dict[str, Any]] | None = None,
        default_state: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if self.path.exists():
            return self.load_state()
        if legacy_path is not None and legacy_loader is not None and legacy_path.exists():
            state = copy.deepcopy(legacy_loader(legacy_path))
            self.save_state(state, migrated_from=str(legacy_path))
            return state
        return self.load_or_initialize(default_state)

    def _write_payload(self, payload: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                "w",
                encoding="utf-8",
                dir=self.path.parent,
                prefix=f".{self.path.name}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                json.dump(payload, handle, indent=2, sort_keys=True)
                handle.flush()
                os.fsync(handle.fileno())
                temp_path = Path(handle.name)
            os.replace(temp_path, self.path)
        finally:
            if temp_path is not None and temp_path.exists():
                temp_path.unlink(missing_ok=True)
