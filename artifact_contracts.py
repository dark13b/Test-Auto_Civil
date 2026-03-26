"""Typed artifact contracts, validation, and deprecated-read mappers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


LEGACY_FIELD_NAMES = frozenset({"test_metrics", "val_metrics"})
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
}


class ArtifactValidationError(ValueError):
    """Raised when an artifact payload violates the typed contract."""


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
        uncertainty_audit = payload.get("uncertainty_audit")
        if uncertainty_audit is not None:
            _require_mapping(uncertainty_audit, "final_holdout_evaluation.uncertainty_audit")
            if "audit_partition" not in uncertainty_audit:
                raise ArtifactValidationError(
                    "final_holdout_evaluation.uncertainty_audit.audit_partition is required"
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


def validate_artifact_payload(filename: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Validate canonical artifacts before they are written."""
    _require_mapping(payload, filename or "artifact")
    artifact_kind = _resolve_artifact_kind(filename, payload)
    if artifact_kind == "search_selection":
        SearchSelectionArtifact.from_payload(payload)
    elif artifact_kind == "final_holdout_evaluation":
        FinalHoldoutEvaluationArtifact.from_payload(payload)
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
        "uncertainty_audit": _map_legacy_uncertainty_summary(payload.get("uncertainty_summary")),
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


def _map_legacy_search_selection(payload: dict[str, Any]) -> dict[str, Any]:
    _require_mapping(payload, "best_search_metrics")
    selection_metrics = (
        payload.get("selection_metrics")
        or payload.get("validation_metrics")
        or payload.get("val_metrics")
        or payload.get("test_metrics")
        or {}
    )
    mapped = {
        "schema_version": 1,
        "artifact_kind": "search_selection",
        "model_name": payload.get("model_name"),
        "display_name": payload.get("display_name", payload.get("model_name")),
        "hyperparameters": dict(payload.get("hyperparameters", {})),
        "trial_number": payload.get("trial_number", payload.get("best_trial")),
        "experiment_id": payload.get("experiment_id", payload.get("trial_number")),
        "source": payload.get("source", "search"),
        "selection_metric_name": "composite_score",
        "selection_decision_score": float(payload.get("composite_score", 0.0)),
        "cross_validation": {
            "stage": "cross_validation",
            "partition": "train",
            "aggregate": _normalize_metric_aggregate_payload(
                payload.get("cv_metrics"),
                context="best_search_metrics.cv_metrics",
            ),
            "dispersion": _extract_legacy_cv_dispersion(payload),
        },
        "selection_validation": {
            "stage": "selection_validation",
            "partition": "validation",
            "aggregate": _normalize_metric_aggregate_payload(
                selection_metrics,
                context="best_search_metrics.selection_metrics",
            ),
        },
        "selection_validation_report": dict(payload.get("validation_report", {"verdict": payload.get("validation_verdict", "UNKNOWN")})),
    }
    validate_artifact_payload("search_selection.json", mapped)
    return mapped


def _map_legacy_uncertainty_summary(payload: Any) -> dict[str, Any] | None:
    if payload is None:
        return None
    _require_mapping(payload, "uncertainty_summary")
    return {
        "artifact_kind": "uncertainty_audit",
        "audit_partition": payload.get("audit_partition", "holdout"),
        "calibration_partition": payload.get("calibration_partition", "validation_audit"),
        "coverage_target": float(payload.get("coverage_target", 0.0)),
        "coverage": float(payload.get("coverage", 0.0)),
        "mean_interval_width": float(payload.get("mean_interval_width", payload.get("sharpness", 0.0))),
        "coverage_audit": payload.get("coverage_audit", {}),
        "coverage_by_strength_bin": payload.get("coverage_by_strength_bin", payload.get("reliability_plot_data", [])),
    }


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
