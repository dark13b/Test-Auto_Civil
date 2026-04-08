"""Persistent hypothesis archive with prediction calibration tracking."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 2


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return round(sum(values) / len(values), 4)


def _acceptance_rate(records: list[dict[str, Any]]) -> float | None:
    resolved = [record for record in records if record.get("outcome") not in {None, "pending"}]
    if not resolved:
        return None
    accepted = [
        record
        for record in resolved
        if str(record.get("outcome")) in {"accepted", "kept"}
    ]
    return round(len(accepted) / len(resolved), 4)


class HypothesisArchive:
    """Stores proposal predictions and their realized outcomes."""

    def __init__(self, archive_path: Path):
        self.archive_path = Path(archive_path)
        self._data = self._load()

    def _empty_payload(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "records": [],
            "summary": {
                "total_hypotheses": 0,
                "resolved_hypotheses": 0,
                "mean_calibration_error": None,
                "acceptance_rate": None,
                "well_calibrated": None,
            },
        }

    def _load(self) -> dict[str, Any]:
        if not self.archive_path.exists():
            return self._empty_payload()
        with self.archive_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        records = payload.get("records")
        if not isinstance(records, list):
            payload["records"] = []
        payload["schema_version"] = max(int(payload.get("schema_version", 0) or 0), SCHEMA_VERSION)
        payload["summary"] = self._build_summary(payload["records"])
        return payload

    def _save(self) -> dict[str, Any]:
        self.archive_path.parent.mkdir(parents=True, exist_ok=True)
        self._data["summary"] = self._build_summary(self._data["records"])
        with self.archive_path.open("w", encoding="utf-8") as handle:
            json.dump(self._data, handle, indent=2, sort_keys=True)
        return self._data

    @staticmethod
    def _build_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
        calibration_errors = [
            float(record["calibration_error"])
            for record in records
            if record.get("calibration_error") is not None
        ]
        mean_calibration_error = _mean(calibration_errors)
        resolved_hypotheses = sum(1 for record in records if record.get("outcome") not in {None, "pending"})
        return {
            "total_hypotheses": len(records),
            "resolved_hypotheses": resolved_hypotheses,
            "mean_calibration_error": mean_calibration_error,
            "acceptance_rate": _acceptance_rate(records),
            "well_calibrated": None if mean_calibration_error is None else mean_calibration_error <= 0.05,
        }

    def load(self) -> dict[str, Any]:
        return json.loads(json.dumps(self._data))

    def all_hypotheses(self) -> list[dict[str, Any]]:
        return list(self._data["records"])

    def calibration_summary(self) -> dict[str, Any]:
        return dict(self._data["summary"])

    def is_well_calibrated(self) -> bool:
        value = self._data["summary"].get("well_calibrated")
        return True if value is None else bool(value)

    def as_novelty_archive(self) -> list[dict[str, Any]]:
        return [{"proposal": record.get("proposal", {})} for record in self._data["records"] if record.get("proposal")]

    def record_trial(self, record: dict[str, Any]) -> dict[str, Any]:
        if record.get("expected_delta_rmse") is None and record.get("actual_delta_rmse") is not None:
            raise ValueError("Completed archive records require expected_delta_rmse before actual_delta_rmse.")
        candidate = dict(record)
        candidate.setdefault("timestamp", _utc_now())
        self._data["records"].append(candidate)
        return self._save()

    def add(
        self,
        *,
        proposal: dict[str, Any],
        claim: str,
        mechanism: str,
        expected_delta_rmse: float | None,
        source: str = "llm",
        run_id: str = "",
        cycle: int = 0,
        prediction: str = "",
        test: str = "",
        expected_metric_effect: dict[str, Any] | None = None,
        confidence: float | None = None,
        novelty_claim: str = "",
        risk_notes: str = "",
        proposed_change: str = "",
        target_component: str = "",
        change_type: str = "other",
        expected_direction: str = "uncertain",
    ) -> str:
        hypothesis_id = f"H-{uuid.uuid4().hex[:8].upper()}"
        record = {
            "hypothesis_id": hypothesis_id,
            "run_id": run_id,
            "cycle": cycle,
            "source": source,
            "hypothesis": claim,
            "claim": claim,
            "rationale": mechanism,
            "mechanism": mechanism,
            "prediction": prediction,
            "test": test,
            "change_type": change_type,
            "target_component": target_component or proposal.get("model_name", ""),
            "proposed_change": proposed_change,
            "expected_direction": expected_direction,
            "expected_metric_effect": expected_metric_effect,
            "confidence": confidence,
            "novelty_claim": novelty_claim,
            "risk_notes": risk_notes,
            "proposal": proposal,
            "expected_delta_rmse": expected_delta_rmse,
            "actual_delta_rmse": None,
            "actual_result": None,
            "calibration_error": None,
            "calibration_outcome": None,
            "outcome": "pending",
            "trial_number": None,
            "timestamp": _utc_now(),
            "resolved_at": None,
        }
        self._data["records"].append(record)
        self._save()
        return hypothesis_id

    def resolve(
        self,
        hypothesis_id: str,
        *,
        actual_delta_rmse: float | None,
        outcome: str,
        trial_number: int | None = None,
        actual_result: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        record = next(
            (candidate for candidate in self._data["records"] if candidate.get("hypothesis_id") == hypothesis_id),
            None,
        )
        if record is None:
            raise KeyError(f"Hypothesis {hypothesis_id} not found.")
        record["actual_delta_rmse"] = actual_delta_rmse
        record["actual_result"] = actual_result or {}
        record["outcome"] = outcome
        record["trial_number"] = trial_number
        record["resolved_at"] = _utc_now()
        if record.get("expected_delta_rmse") is not None and actual_delta_rmse is not None:
            record["calibration_error"] = round(
                abs(float(record["expected_delta_rmse"]) - float(actual_delta_rmse)),
                4,
            )
            record["calibration_outcome"] = (
                "well_calibrated" if record["calibration_error"] <= 0.05 else "miscalibrated"
            )
        else:
            record["calibration_error"] = None
            record["calibration_outcome"] = None
        self._save()
        return dict(record)
