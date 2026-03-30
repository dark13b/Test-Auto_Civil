"""Typed artifact contracts, validation, and deprecated-read mappers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


LEGACY_FIELD_NAMES = frozenset({"test_metrics", "val_metrics"})
UNCERTAINTY_LINEAGE_FIELDS = ("run_id", "model_id", "model_fingerprint", "config_hash")
UNCERTAINTY_LINEAGE_COMPARISON_FIELDS = UNCERTAINTY_LINEAGE_FIELDS + ("model_artifact_id",)
CANONICAL_ARTIFACT_KINDS = frozenset(
    {
        "search_selection",
        "final_holdout_evaluation",
        "uncertainty_audit",
        "run_manifest",
        "design_tool_result",
    }
)
CANONICAL_FILENAMES = {
    "search_selection.json": "search_selection",
    "final_holdout_evaluation.json": "final_holdout_evaluation",
    "uncertainty_calibration.json": "uncertainty_audit",
    "design_uncertainty_calibration.json": "uncertainty_audit",
    "benchmark_uncertainty_calibration.json": "uncertainty_audit",
}


class ArtifactValidationError(ValueError):
    """Raised when an artifact payload violates the typed contract."""


def artifact_metadata(payload: dict[str, Any] | None) -> dict[str, Any]:
    """Return artifact metadata with root-level lineage fields merged in."""
    if not isinstance(payload, dict):
        return {}
    raw_metadata = payload.get("artifact_metadata", {})
    metadata = dict(raw_metadata) if isinstance(raw_metadata, dict) else {}
    for key in (
        "run_id",
        "artifact_id",
        "model_artifact_id",
        "model_id",
        "model_fingerprint",
        "config_hash",
        "source_mode",
        "timestamp",
    ):
        if key not in metadata and key in payload:
            metadata[key] = payload.get(key)
    return metadata


def _normalize_lineage_value(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def extract_artifact_lineage(
    payload: dict[str, Any] | None,
    *,
    fallback_model_id: str | None = None,
) -> dict[str, str | None]:
    """Return the comparable lineage tuple for one artifact payload."""
    if not isinstance(payload, dict):
        return {}
    metadata = artifact_metadata(payload)
    model_id = (
        metadata.get("model_id")
        or payload.get("model_id")
        or payload.get("model_name")
        or fallback_model_id
    )
    model_artifact_id = metadata.get("model_artifact_id") or payload.get("model_artifact_id")
    return {
        "artifact_id": _normalize_lineage_value(metadata.get("artifact_id") or payload.get("artifact_id")),
        "run_id": _normalize_lineage_value(metadata.get("run_id") or payload.get("run_id")),
        "model_artifact_id": _normalize_lineage_value(model_artifact_id),
        "model_id": _normalize_lineage_value(model_id),
        "model_fingerprint": _normalize_lineage_value(
            metadata.get("model_fingerprint")
            or payload.get("model_fingerprint")
            or model_artifact_id
            or model_id
        ),
        "config_hash": _normalize_lineage_value(metadata.get("config_hash") or payload.get("config_hash")),
    }


def coalesce_artifact_lineage(
    *sources: dict[str, Any] | None,
    fallback_model_id: str | None = None,
) -> dict[str, str | None]:
    """Merge lineage sources, keeping the first non-empty value for each field."""
    merged: dict[str, str | None] = {}
    for source in sources:
        if not isinstance(source, dict):
            continue
        if any(key in source for key in ("artifact_metadata", "artifact_kind", "model_name")):
            lineage = extract_artifact_lineage(source, fallback_model_id=fallback_model_id)
        else:
            lineage = {
                key: _normalize_lineage_value(source.get(key))
                for key in ("artifact_id", "run_id", "model_artifact_id", "model_id", "model_fingerprint", "config_hash")
            }
        for key, value in lineage.items():
            if key not in merged or merged[key] is None:
                merged[key] = value
    return merged


def merge_artifact_lineage(
    payload: dict[str, Any],
    *,
    lineage: dict[str, Any] | None,
) -> dict[str, Any]:
    """Fill missing root/artifact_metadata lineage fields from a provable source."""
    normalized = dict(payload)
    resolved_lineage = coalesce_artifact_lineage(lineage)
    if not resolved_lineage:
        return normalized
    lineage_keys = ("run_id", "model_artifact_id", "model_id", "model_fingerprint", "config_hash")
    for key in lineage_keys:
        value = resolved_lineage.get(key)
        if value is not None and normalized.get(key) in (None, ""):
            normalized[key] = value
    metadata = artifact_metadata(normalized)
    for key in lineage_keys:
        value = normalized.get(key)
        if value is not None and metadata.get(key) in (None, ""):
            metadata[key] = value
    if metadata:
        normalized["artifact_metadata"] = metadata
    return normalized


def format_uncertainty_source_label(
    lineage: dict[str, Any] | None,
    *,
    prefix: str = "Uncertainty audit",
) -> str:
    """Return a human-readable lineage label with exact source fields."""
    resolved_lineage = coalesce_artifact_lineage(lineage)
    parts = []
    if resolved_lineage.get("model_id"):
        parts.append(f"model_id={resolved_lineage['model_id']}")
    if resolved_lineage.get("model_fingerprint"):
        parts.append(f"model_fingerprint={resolved_lineage['model_fingerprint']}")
    if resolved_lineage.get("config_hash"):
        parts.append(f"config_hash={resolved_lineage['config_hash']}")
    if resolved_lineage.get("run_id"):
        parts.append(f"run_id={resolved_lineage['run_id']}")
    return prefix if not parts else f"{prefix} | " + " | ".join(parts)


def evaluate_uncertainty_lineage(
    payload: dict[str, Any] | None,
    *,
    expected_lineage: dict[str, Any] | None = None,
    fallback_model_id: str | None = None,
) -> dict[str, Any]:
    """Compare an uncertainty artifact against the expected active lineage."""
    actual = extract_artifact_lineage(payload, fallback_model_id=fallback_model_id)
    expected = coalesce_artifact_lineage(expected_lineage, fallback_model_id=fallback_model_id)
    missing_fields = [field for field in UNCERTAINTY_LINEAGE_FIELDS if not actual.get(field)]
    mismatches: list[dict[str, Any]] = []
    for field in UNCERTAINTY_LINEAGE_COMPARISON_FIELDS:
        expected_value = expected.get(field)
        actual_value = actual.get(field)
        if expected_value is None or actual_value is None:
            continue
        if expected_value != actual_value:
            mismatches.append(
                {
                    "field": field,
                    "expected": expected_value,
                    "actual": actual_value,
                }
            )
    if missing_fields:
        status = "missing"
    elif mismatches:
        status = "mismatch"
    else:
        status = "verified"
    return {
        "lineage": actual,
        "expected_lineage": expected,
        "missing_fields": missing_fields,
        "mismatches": mismatches,
        "status": status,
        "label": format_uncertainty_source_label(actual),
    }


def normalize_uncertainty_artifact(
    payload: dict[str, Any] | None,
    *,
    expected_lineage: dict[str, Any] | None = None,
    allow_provenance_fill: bool = False,
    fallback_model_id: str | None = None,
) -> dict[str, Any]:
    """Attach explicit lineage status/labels to an uncertainty artifact."""
    if not isinstance(payload, dict):
        return {}
    normalized = dict(payload)
    normalized.setdefault("artifact_kind", "uncertainty_audit")
    if allow_provenance_fill:
        normalized = merge_artifact_lineage(normalized, lineage=expected_lineage)
    lineage_report = evaluate_uncertainty_lineage(
        normalized,
        expected_lineage=expected_lineage,
        fallback_model_id=fallback_model_id,
    )
    lineage_payload = dict(lineage_report["lineage"])
    lineage_payload["status"] = lineage_report["status"]
    if lineage_report["missing_fields"]:
        lineage_payload["missing_fields"] = list(lineage_report["missing_fields"])
    if lineage_report["mismatches"]:
        lineage_payload["mismatches"] = list(lineage_report["mismatches"])
    normalized["lineage"] = lineage_payload
    normalized["lineage_status"] = lineage_report["status"]
    normalized["source_label"] = lineage_report["label"]
    if lineage_report["mismatches"]:
        normalized["lineage_mismatches"] = list(lineage_report["mismatches"])
    elif "lineage_mismatches" in normalized:
        normalized.pop("lineage_mismatches", None)
    return normalized


@dataclass(frozen=True)
class MetricAggregate:
    rmse: float
    mae: float
    r2: float
    composite_score: float

    @classmethod
    def from_payload(cls, payload: dict[str, Any], *, context: str) -> "MetricAggregate":
        _require_mapping(payload, context)
        missing = [key for key in ("rmse", "mae", "r2", "composite_score") if key not in payload]
        if missing:
            raise ArtifactValidationError(f"{context} is missing metric keys: {', '.join(missing)}")
        return cls(
            rmse=float(payload["rmse"]),
            mae=float(payload["mae"]),
            r2=float(payload["r2"]),
            composite_score=float(payload["composite_score"]),
        )


@dataclass(frozen=True)
class StageMetrics:
    stage: str
    partition: str
    aggregate: MetricAggregate
    extras: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_payload(
        cls,
        payload: dict[str, Any],
        *,
        expected_stage: str,
        expected_partition: str,
        context: str,
    ) -> "StageMetrics":
        _require_mapping(payload, context)
        stage = str(payload.get("stage", "")).strip()
        partition = str(payload.get("partition", "")).strip()
        if stage != expected_stage:
            raise ArtifactValidationError(
                f"{context}.stage must be '{expected_stage}', got '{stage or '<missing>'}'"
            )
        if partition != expected_partition:
            raise ArtifactValidationError(
                f"{context}.partition must be '{expected_partition}', got '{partition or '<missing>'}'"
            )
        aggregate = MetricAggregate.from_payload(
            _normalize_metric_aggregate_payload(payload.get("aggregate"), context=f"{context}.aggregate"),
            context=f"{context}.aggregate",
        )
        extras = {
            key: value
            for key, value in payload.items()
            if key not in {"stage", "partition", "aggregate"}
        }
        return cls(stage=stage, partition=partition, aggregate=aggregate, extras=extras)


@dataclass(frozen=True)
class SearchSelectionArtifact:
    model_name: str
    hyperparameters: dict[str, Any]
    cross_validation: StageMetrics
    selection_validation: StageMetrics
    selection_validation_report: dict[str, Any]
    selection_metric_name: str
    selection_decision_score: float

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "SearchSelectionArtifact":
        _require_mapping(payload, "search_selection")
        _forbid_legacy_fields(payload, context="search_selection")
        if payload.get("artifact_kind") != "search_selection":
            raise ArtifactValidationError("search_selection.artifact_kind must be 'search_selection'")
        if "holdout_metrics" in payload or "holdout_validation_report" in payload:
            raise ArtifactValidationError("search_selection artifacts may not contain holdout fields")
        report = payload.get("selection_validation_report")
        _require_mapping(report, "search_selection.selection_validation_report")
        if "verdict" not in report:
            raise ArtifactValidationError("search_selection.selection_validation_report.verdict is required")
        hyperparameters = payload.get("hyperparameters")
        _require_mapping(hyperparameters, "search_selection.hyperparameters")
        return cls(
            model_name=str(payload.get("model_name", "")).strip(),
            hyperparameters=dict(hyperparameters),
            cross_validation=StageMetrics.from_payload(
                payload.get("cross_validation"),
                expected_stage="cross_validation",
                expected_partition="train",
                context="search_selection.cross_validation",
            ),
            selection_validation=StageMetrics.from_payload(
                payload.get("selection_validation"),
                expected_stage="selection_validation",
                expected_partition="validation",
                context="search_selection.selection_validation",
            ),
            selection_validation_report=dict(report),
            selection_metric_name=str(payload.get("selection_metric_name", "")).strip(),
            selection_decision_score=float(payload.get("selection_decision_score", 0.0)),
        )


@dataclass(frozen=True)
class FinalHoldoutEvaluationArtifact:
    selected_model: SearchSelectionArtifact
    holdout_metrics: StageMetrics
    holdout_validation_report: dict[str, Any]

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "FinalHoldoutEvaluationArtifact":
        _require_mapping(payload, "final_holdout_evaluation")
        _forbid_legacy_fields(payload, context="final_holdout_evaluation")
        if payload.get("artifact_kind") != "final_holdout_evaluation":
            raise ArtifactValidationError(
                "final_holdout_evaluation.artifact_kind must be 'final_holdout_evaluation'"
            )
        selected_model = SearchSelectionArtifact.from_payload(payload.get("selected_model"))
        report = payload.get("holdout_validation_report")
        _require_mapping(report, "final_holdout_evaluation.holdout_validation_report")
        if "verdict" not in report:
            raise ArtifactValidationError(
                "final_holdout_evaluation.holdout_validation_report.verdict is required"
            )
        expected_uncertainty_lineage = coalesce_artifact_lineage(
            payload.get("selected_model"),
            payload,
            fallback_model_id=str(payload.get("selected_model", {}).get("model_name", "")).strip() or None,
        )
        uncertainty_audit = payload.get("uncertainty_audit")
        if uncertainty_audit is not None:
            UncertaintyAuditArtifact.from_payload(
                uncertainty_audit,
                expected_lineage=expected_uncertainty_lineage,
                context="final_holdout_evaluation.uncertainty_audit",
            )
        return cls(
            selected_model=selected_model,
            holdout_metrics=StageMetrics.from_payload(
                payload.get("holdout_metrics"),
                expected_stage="final_holdout",
                expected_partition="holdout",
                context="final_holdout_evaluation.holdout_metrics",
            ),
            holdout_validation_report=dict(report),
        )


@dataclass(frozen=True)
class UncertaintyAuditArtifact:
    audit_partition: str
    calibration_partition: str | None
    coverage_target: float
    coverage: float
    lineage_status: str

    @classmethod
    def from_payload(
        cls,
        payload: dict[str, Any],
        *,
        expected_lineage: dict[str, Any] | None = None,
        context: str = "uncertainty_audit",
    ) -> "UncertaintyAuditArtifact":
        _require_mapping(payload, context)
        if payload.get("artifact_kind") != "uncertainty_audit":
            raise ArtifactValidationError(f"{context}.artifact_kind must be 'uncertainty_audit'")
        audit_partition = str(payload.get("audit_partition", "")).strip()
        if not audit_partition:
            raise ArtifactValidationError(f"{context}.audit_partition is required")
        evaluation = evaluate_uncertainty_lineage(payload, expected_lineage=expected_lineage)
        if evaluation["missing_fields"]:
            raise ArtifactValidationError(
                f"{context} is missing lineage keys: {', '.join(evaluation['missing_fields'])}"
            )
        if evaluation["mismatches"]:
            first = evaluation["mismatches"][0]
            raise ArtifactValidationError(
                f"{context} lineage mismatch for {first['field']}: "
                f"expected '{first['expected']}', got '{first['actual']}'"
            )
        return cls(
            audit_partition=audit_partition,
            calibration_partition=_normalize_lineage_value(payload.get("calibration_partition")),
            coverage_target=float(payload.get("coverage_target", 0.0)),
            coverage=float(payload.get("coverage", 0.0)),
            lineage_status=str(payload.get("lineage_status") or evaluation["status"]),
        )


def validate_artifact_payload(filename: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Validate canonical artifacts before they are written."""
    _require_mapping(payload, filename or "artifact")
    artifact_kind = _resolve_artifact_kind(filename, payload)
    if artifact_kind == "search_selection":
        SearchSelectionArtifact.from_payload(payload)
    elif artifact_kind == "final_holdout_evaluation":
        FinalHoldoutEvaluationArtifact.from_payload(payload)
    elif artifact_kind == "uncertainty_audit":
        UncertaintyAuditArtifact.from_payload(payload)
    return payload


def map_deprecated_artifact_payload(filename: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Map deprecated artifacts into the canonical contract with explicit deprecations."""
    _require_mapping(payload, filename or "artifact")
    if filename != "final_metrics.json":
        raise ArtifactValidationError(f"No deprecated mapper is registered for {filename}")

    selected_model = _map_legacy_search_selection(payload.get("best_search_metrics", {}))
    holdout_metrics = {
        "stage": "final_holdout",
        "partition": "holdout",
        "aggregate": _normalize_metric_aggregate_payload(
            payload.get("holdout_metrics"),
            context="final_metrics.holdout_metrics",
        ),
    }
    mapped = {
        "schema_version": 1,
        "artifact_kind": "final_holdout_evaluation",
        "selected_model": selected_model,
        "selection_metric_name": str(payload.get("selection_metric_name") or "composite_score"),
        "holdout_metric_name": str(payload.get("holdout_metric_name") or "composite_score"),
        "holdout_metrics": holdout_metrics,
        "holdout_validation_report": {
            "verdict": payload.get("holdout_validation_verdict", payload.get("validation_verdict", "UNKNOWN")),
        },
        "uncertainty_audit": _map_legacy_uncertainty_summary(
            payload.get("uncertainty_summary"),
            expected_lineage=extract_artifact_lineage(
                selected_model,
                fallback_model_id=str(selected_model.get("model_name", "")).strip() or None,
            ),
        ),
        "regime_specific_modeling": payload.get("regime_specific_modeling"),
        "deprecations": [
            {
                "source": "final_metrics.json",
                "replacement": "final_holdout_evaluation.json",
                "message": "Mapped deprecated final_metrics.json into the canonical final holdout contract.",
            }
        ],
    }
    validate_artifact_payload("final_holdout_evaluation.json", mapped)
    return mapped


def get_selection_validation_aggregate(payload: dict[str, Any] | None) -> dict[str, Any]:
    """Return selection-time validation metrics from new or legacy payload shapes."""
    if not isinstance(payload, dict):
        return {}
    selection_validation = payload.get("selection_validation")
    if isinstance(selection_validation, dict):
        aggregate = selection_validation.get("aggregate")
        if isinstance(aggregate, dict):
            return dict(aggregate)
    for key in ("validation_metrics", "selection_metrics", "val_metrics", "test_metrics"):
        value = payload.get(key)
        if isinstance(value, dict):
            return dict(value)
    return {}


def _map_legacy_search_selection(payload: dict[str, Any]) -> dict[str, Any]:
    _require_mapping(payload, "best_search_metrics")
    selection_metrics = (
        payload.get("selection_metrics")
        or payload.get("validation_metrics")
        or payload.get("val_metrics")
        or payload.get("test_metrics")
        or {}
    )
    cv_metrics = _normalize_metric_aggregate_payload(
        payload.get("cv_metrics"),
        context="best_search_metrics.cv_metrics",
    )
    selection_aggregate = _normalize_metric_aggregate_payload(
        selection_metrics,
        context="best_search_metrics.selection_metrics",
    )
    mapped = {
        "schema_version": 1,
        "artifact_kind": "search_selection",
        "run_id": payload.get("run_id"),
        "artifact_id": payload.get("artifact_id"),
        "model_artifact_id": payload.get("model_artifact_id"),
        "model_name": payload.get("model_name"),
        "display_name": payload.get("display_name", payload.get("model_name")),
        "hyperparameters": dict(payload.get("hyperparameters", {})),
        "trial_number": payload.get("trial_number", payload.get("best_trial")),
        "best_trial": payload.get("best_trial", payload.get("trial_number")),
        "experiment_id": payload.get("experiment_id", payload.get("trial_number")),
        "source": payload.get("source", "search"),
        "config_hash": payload.get("config_hash"),
        "model_artifact_id": payload.get("model_artifact_id"),
        "model_id": payload.get("model_id", payload.get("model_name")),
        "model_fingerprint": payload.get("model_fingerprint", payload.get("model_artifact_id", payload.get("model_name"))),
        "composite_score": float(payload.get("composite_score", 0.0)),
        "validation_verdict": payload.get("validation_verdict", "UNKNOWN"),
        "cv_rmse": cv_metrics.get("rmse"),
        "cv_mae": cv_metrics.get("mae"),
        "cv_r2": cv_metrics.get("r2"),
        "cv_composite": cv_metrics.get("composite_score"),
        "selection_metric_name": "composite_score",
        "selection_decision_score": float(payload.get("composite_score", 0.0)),
        "cross_validation": {
            "stage": "cross_validation",
            "partition": "train",
            "aggregate": cv_metrics,
            "dispersion": _extract_legacy_cv_dispersion(payload),
        },
        "selection_validation": {
            "stage": "selection_validation",
            "partition": "validation",
            "aggregate": selection_aggregate,
        },
        "validation_metrics": selection_aggregate,
        "selection_validation_report": dict(
            payload.get("validation_report", {"verdict": payload.get("validation_verdict", "UNKNOWN")})
        ),
    }
    mapped["validation_report"] = dict(mapped["selection_validation_report"])
    validate_artifact_payload("search_selection.json", mapped)
    return mapped


def _map_legacy_uncertainty_summary(
    payload: Any,
    *,
    expected_lineage: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    if payload is None:
        return None
    _require_mapping(payload, "uncertainty_summary")
    mapped = {
        "artifact_kind": "uncertainty_audit",
        "audit_partition": payload.get("audit_partition", "holdout"),
        "calibration_partition": payload.get("calibration_partition", "validation_audit"),
        "coverage_target": float(payload.get("coverage_target", 0.0)),
        "coverage": float(payload.get("coverage", 0.0)),
        "mean_interval_width": float(payload.get("mean_interval_width", payload.get("sharpness", 0.0))),
        "coverage_audit": payload.get("coverage_audit", {}),
        "coverage_by_strength_bin": payload.get("coverage_by_strength_bin", payload.get("reliability_plot_data", [])),
    }
    return normalize_uncertainty_artifact(
        mapped,
        expected_lineage=expected_lineage,
        allow_provenance_fill=True,
    )


def _extract_legacy_cv_dispersion(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "rmse_std": _optional_float(payload.get("cv_rmse_std")),
        "mae_std": _optional_float(payload.get("cv_mae_std")),
        "r2_std": _optional_float(payload.get("cv_r2_std")),
        "composite_score_std": _optional_float(payload.get("cv_composite_std")),
    }


def _resolve_artifact_kind(filename: str, payload: dict[str, Any]) -> str | None:
    declared = payload.get("artifact_kind")
    if declared in CANONICAL_ARTIFACT_KINDS:
        return str(declared)
    return CANONICAL_FILENAMES.get(filename)


def _normalize_metric_aggregate_payload(payload: Any, *, context: str) -> dict[str, Any]:
    _require_mapping(payload, context)
    if {"stage", "partition", "aggregate"}.issubset(payload.keys()):
        payload = payload.get("aggregate")
        _require_mapping(payload, f"{context}.aggregate")
    return dict(payload)


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _require_mapping(payload: Any, context: str) -> None:
    if not isinstance(payload, dict):
        raise ArtifactValidationError(f"{context} must be a dictionary")


def _forbid_legacy_fields(payload: dict[str, Any], *, context: str) -> None:
    forbidden = sorted(LEGACY_FIELD_NAMES.intersection(payload.keys()))
    if forbidden:
        raise ArtifactValidationError(
            f"{context} may not contain deprecated fields: {', '.join(forbidden)}"
        )
