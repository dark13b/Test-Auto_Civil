"""Coordinated atomic synchronization for CSV trial logs and best-result JSON artifacts."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import pandas as pd

PENDING_META_FILENAME = ".sync_pending_meta.json"


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("w", encoding="utf-8", delete=False, dir=path.parent, newline="") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
        temp_path = Path(handle.name)
    os.replace(temp_path, path)


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("wb", delete=False, dir=path.parent) as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
        temp_path = Path(handle.name)
    os.replace(temp_path, path)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def _coerce_trial_number(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, str) and not value.strip():
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


@dataclass(frozen=True)
class CsvTarget:
    filename: str
    columns: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {"filename": self.filename, "columns": list(self.columns)}


class AtomicArtifactWriter:
    """Persist related artifacts with a replayable pending-meta envelope."""

    def __init__(
        self,
        *,
        outputs_dir: str | Path,
        csv_targets: list[tuple[str, list[str]] | CsvTarget],
        best_result_filename: str | None = None,
        pending_meta_filename: str = PENDING_META_FILENAME,
    ) -> None:
        self.outputs_dir = Path(outputs_dir)
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        self.csv_targets = [
            target if isinstance(target, CsvTarget) else CsvTarget(filename=target[0], columns=list(target[1]))
            for target in csv_targets
        ]
        self.best_result_filename = best_result_filename
        self.pending_meta_path = self.outputs_dir / pending_meta_filename

    def append_trial(self, trial_record: dict[str, Any]) -> None:
        pending_payload = self._build_pending_payload(operation="append_trial", trial_record=trial_record)
        self._apply_operation(pending_payload)

    def record_new_best(self, *, trial_record: dict[str, Any], best_result: dict[str, Any]) -> None:
        pending_payload = self._build_pending_payload(
            operation="record_new_best",
            trial_record=trial_record,
            best_result=best_result,
        )
        self._apply_operation(pending_payload)

    def repair_on_startup(self, *, check_only: bool = False) -> dict[str, Any]:
        return repair_on_startup(
            self.outputs_dir,
            check_only=check_only,
            csv_targets=self.csv_targets,
            best_result_filename=self.best_result_filename,
            pending_meta_filename=self.pending_meta_path.name,
        )

    def _build_pending_payload(
        self,
        *,
        operation: str,
        trial_record: dict[str, Any],
        best_result: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload = {
            "operation": operation,
            "csv_targets": [target.to_dict() for target in self.csv_targets],
            "trial_record": dict(trial_record),
        }
        if self.best_result_filename is not None:
            payload["best_result_filename"] = self.best_result_filename
        if best_result is not None:
            payload["best_result"] = dict(best_result)
        return payload

    def _apply_operation(self, pending_payload: dict[str, Any]) -> None:
        _atomic_write_text(self.pending_meta_path, json.dumps(pending_payload, indent=2, sort_keys=True))
        self._replay_pending_payload(pending_payload)
        if self.pending_meta_path.exists():
            self.pending_meta_path.unlink()

    def _replay_pending_payload(self, pending_payload: dict[str, Any]) -> None:
        operation = str(pending_payload["operation"])
        trial_record = dict(pending_payload["trial_record"])
        csv_targets = [
            target if isinstance(target, CsvTarget) else CsvTarget(str(target["filename"]), list(target["columns"]))
            for target in pending_payload.get("csv_targets", [])
        ]

        for target in csv_targets:
            self._write_csv_snapshot(target, trial_record)

        if operation == "record_new_best":
            best_result_filename = str(
                pending_payload.get("best_result_filename") or self.best_result_filename or "search_state_best_result.json"
            )
            best_result = dict(pending_payload["best_result"])
            _atomic_write_text(
                self.outputs_dir / best_result_filename,
                json.dumps(best_result, indent=2, sort_keys=True),
            )
        elif operation != "append_trial":
            raise ValueError(f"Unsupported artifact sync operation: {operation}")

    def _write_csv_snapshot(self, target: CsvTarget, trial_record: dict[str, Any]) -> None:
        path = self.outputs_dir / target.filename
        frame = self._load_csv_frame(path, target.columns)
        normalized_record = {column: trial_record.get(column) for column in target.columns}
        frame = self._upsert_row(frame, normalized_record, target.columns)
        csv_payload = frame.to_csv(index=False)
        _atomic_write_text(path, csv_payload)

    @staticmethod
    def _load_csv_frame(path: Path, columns: list[str]) -> pd.DataFrame:
        if path.exists() and path.stat().st_size > 0:
            frame = pd.read_csv(path)
            for column in columns:
                if column not in frame.columns:
                    frame[column] = None
            return frame.reindex(columns=columns)
        return pd.DataFrame(columns=columns)

    @staticmethod
    def _upsert_row(frame: pd.DataFrame, record: dict[str, Any], columns: list[str]) -> pd.DataFrame:
        trial_number = record.get("trial_number")
        if trial_number is not None and "trial_number" in frame.columns:
            existing_mask = pd.to_numeric(frame["trial_number"], errors="coerce") == float(trial_number)
            if existing_mask.any():
                updated = frame.copy()
                updated.loc[existing_mask, columns] = [record.get(column) for column in columns]
                return updated
        return pd.concat([frame, pd.DataFrame([record], columns=columns)], ignore_index=True)


def repair_on_startup(
    outputs_dir: str | Path,
    *,
    check_only: bool = False,
    csv_targets: list[tuple[str, list[str]] | CsvTarget] | None = None,
    best_result_filename: str | None = None,
    pending_meta_filename: str = PENDING_META_FILENAME,
) -> dict[str, Any]:
    outputs_path = Path(outputs_dir)
    pending_meta_path = outputs_path / pending_meta_filename
    resolved_targets = [
        target if isinstance(target, CsvTarget) else CsvTarget(filename=target[0], columns=list(target[1]))
        for target in (csv_targets or [])
    ]

    if pending_meta_path.exists():
        pending_payload = _read_json(pending_meta_path)
        if check_only:
            return {
                "pending": True,
                "repaired": False,
                "operation": pending_payload.get("operation"),
                "pending_meta_path": str(pending_meta_path),
            }

        if not resolved_targets:
            resolved_targets = [
                CsvTarget(filename=str(target["filename"]), columns=list(target["columns"]))
                for target in pending_payload.get("csv_targets", [])
            ]
        writer = AtomicArtifactWriter(
            outputs_dir=outputs_path,
            csv_targets=resolved_targets,
            best_result_filename=str(pending_payload.get("best_result_filename") or best_result_filename)
            if pending_payload.get("best_result_filename") or best_result_filename
            else None,
            pending_meta_filename=pending_meta_filename,
        )
        writer._replay_pending_payload(pending_payload)
        if pending_meta_path.exists():
            pending_meta_path.unlink()
        return {
            "pending": True,
            "repaired": True,
            "operation": pending_payload.get("operation"),
            "pending_meta_path": str(pending_meta_path),
        }

    desync_report = _best_result_desync_report(
        outputs_path=outputs_path,
        csv_targets=resolved_targets,
        best_result_filename=best_result_filename,
    )
    if desync_report is None:
        return {"pending": False, "repaired": False, "operation": None}
    if check_only:
        return desync_report

    writer = AtomicArtifactWriter(
        outputs_dir=outputs_path,
        csv_targets=resolved_targets,
        best_result_filename=best_result_filename,
        pending_meta_filename=pending_meta_filename,
    )
    for target in resolved_targets:
        writer._write_csv_snapshot(target, desync_report["trial_record"])
    return {
        **desync_report,
        "repaired": True,
    }


def _best_result_desync_report(
    *,
    outputs_path: Path,
    csv_targets: list[CsvTarget],
    best_result_filename: str | None,
) -> dict[str, Any] | None:
    if not best_result_filename or not csv_targets:
        return None
    best_result_path = outputs_path / best_result_filename
    if not best_result_path.exists():
        return None

    best_result = _read_json(best_result_path)
    trial_number = _coerce_trial_number(best_result.get("trial_number", best_result.get("best_trial")))
    if trial_number is None:
        return None

    desynced_targets: list[str] = []
    for target in csv_targets:
        csv_path = outputs_path / target.filename
        if not csv_path.exists() or csv_path.stat().st_size == 0:
            desynced_targets.append(target.filename)
            continue
        frame = pd.read_csv(csv_path)
        if "trial_number" not in frame.columns:
            desynced_targets.append(target.filename)
            continue
        trial_matches = pd.to_numeric(frame["trial_number"], errors="coerce") == float(trial_number)
        if not trial_matches.any():
            desynced_targets.append(target.filename)

    if not desynced_targets:
        return None

    required_columns = sorted({column for target in csv_targets for column in target.columns})
    trial_record = {column: best_result.get(column) for column in required_columns}
    if "trial_number" in required_columns:
        trial_record["trial_number"] = trial_number
    if "best_trial" in best_result and "best_trial" in required_columns:
        trial_record["best_trial"] = best_result.get("best_trial")
    if "model_name" in required_columns and trial_record.get("model_name") is None:
        trial_record["model_name"] = best_result.get("model_name")
    if "display_name" in required_columns and trial_record.get("display_name") is None:
        trial_record["display_name"] = best_result.get("display_name") or best_result.get("model_name")
    if "selection_status" in required_columns and trial_record.get("selection_status") is None:
        trial_record["selection_status"] = "new_best"

    return {
        "pending": True,
        "repaired": False,
        "operation": "recover_best_result_desync",
        "best_result_path": str(best_result_path),
        "best_result_trial_number": trial_number,
        "desynced_targets": desynced_targets,
        "trial_record": trial_record,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Repair or inspect synchronized artifact state.")
    parser.add_argument("--outputs-dir", required=True, help="Outputs directory containing synchronized artifacts.")
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Report pending synchronization state without applying any repair.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    report = repair_on_startup(args.outputs_dir, check_only=bool(args.check_only))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 1 if args.check_only and report.get("pending") else 0


if __name__ == "__main__":
    raise SystemExit(main())
