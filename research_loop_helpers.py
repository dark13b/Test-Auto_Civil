"""Focused helpers for the AutoCivil-Lab research loop orchestration."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pandas as pd

from hypothesis_archive import HypothesisArchive
from knowledge_base import get_knowledge_context
from deterministic_proposal_provider import DeterministicProposalProvider
from hybrid_proposal_provider import HybridProposalProvider
from llm_proposal_provider import LLMProposalProvider
from train_impl import EngineeringValidator, evaluate_candidate
from research_protocol import RESEARCH_RESULTS_COLUMNS, load_json_file


def _safe_float(value: Any, fallback: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(fallback)


def calculate_improvement_percentage(reference_score: float, candidate_score: float) -> float:
    denominator = max(abs(reference_score), 1e-8)
    return ((candidate_score - reference_score) / denominator) * 100.0


def _read_baseline_metrics(outputs_dir: Path, config: dict[str, Any]) -> dict[str, Any]:
    baseline_path = outputs_dir / "baseline_metrics.json"
    if baseline_path.exists():
        return load_json_file(baseline_path)

    import train

    original_load_config = train.load_config
    train.load_config = lambda: config
    try:
        exit_code = train.main()
    finally:
        train.load_config = original_load_config
    if exit_code != 0:
        raise RuntimeError("Baseline training failed while preparing the research loop.")
    return load_json_file(baseline_path)


def _load_current_best_result(outputs_dir: Path, baseline_metrics: dict[str, Any]) -> dict[str, Any]:
    best_result_path = outputs_dir / "best_search_result.json"
    if best_result_path.exists():
        payload = load_json_file(best_result_path)
        payload.setdefault("source", "best_search_result")
        return payload

    final_metrics_path = outputs_dir / "final_metrics.json"
    if final_metrics_path.exists():
        final_metrics = load_json_file(final_metrics_path)
        best_search_metrics = final_metrics.get("best_search_metrics")
        if isinstance(best_search_metrics, dict) and best_search_metrics:
            payload = copy.deepcopy(best_search_metrics)
            payload.setdefault("source", "final_metrics")
            return payload

    payload = copy.deepcopy(baseline_metrics)
    payload["source"] = "baseline"
    payload["trial_number"] = None
    payload["best_trial"] = None
    return payload


def _build_record(
    *,
    trial_number: int,
    experiment: dict[str, Any],
    selection_status: str,
    result: dict[str, Any] | None,
    error_message: str | None,
    trial_runtime_seconds: float | None,
    budget_status: str | None,
    scout_improvement_pct: float | None,
    confirm_improvement_pct: float | None,
    research_results_columns: list[str] | None = None,
) -> dict[str, Any]:
    columns = list(research_results_columns or RESEARCH_RESULTS_COLUMNS)
    record = {column: None for column in columns}
    record.update(
        {
            "trial_number": trial_number,
            "experiment_id": experiment.get("experiment_id"),
            "stage": experiment.get("stage"),
            "model_name": experiment.get("model_name"),
            "display_name": experiment.get("display_name", experiment.get("model_name")),
            "proposal_family": experiment.get("proposal_family"),
            "proposal_source": experiment.get("proposal_source"),
            "proposal_status": experiment.get("proposal_status"),
            "semantic_validation_result": experiment.get("semantic_validation_result"),
            "semantic_validation_error": experiment.get("semantic_validation_error"),
            "semantic_rejection_reason": experiment.get("semantic_rejection_reason"),
            "proposal_backend": experiment.get("proposal_backend"),
            "proposal_model": experiment.get("proposal_model"),
            "prompt_hash": experiment.get("prompt_hash"),
            "hypothesis": experiment.get("hypothesis"),
            "hyperparameters": json.dumps(experiment.get("params", {}), sort_keys=True),
            "confidence": experiment.get("confidence"),
            "expected_delta_rmse": experiment.get("expected_delta_rmse"),
            "actual_delta_rmse": None if result is None else result.get("actual_delta_rmse"),
            "calibration_error": None if result is None else result.get("calibration_error"),
            "selection_status": selection_status,
            "validation_verdict": None if result is None else result.get("validation_verdict"),
            "error_message": error_message,
            "trial_runtime_seconds": trial_runtime_seconds,
            "budget_status": budget_status,
            "scout_improvement_pct": scout_improvement_pct,
            "confirm_improvement_pct": confirm_improvement_pct,
        }
    )
    if result is None:
        return record

    validation_report = result.get("validation_report", {})
    val_metrics = result.get("selection_metrics", result.get("val_metrics", result.get("test_metrics", {})))
    record.update(
        {
            "rmse": result.get("rmse"),
            "mae": result.get("mae"),
            "r2": result.get("r2"),
            "composite_score": result.get("composite_score"),
            "test_rmse": val_metrics.get("rmse"),
            "test_mae": val_metrics.get("mae"),
            "test_r2": val_metrics.get("r2"),
            "test_composite_score": val_metrics.get("composite_score"),
            "validation_pass_rate": validation_report.get("pass_rate"),
            "failed_count": validation_report.get("failed_count"),
            "hard_failed_count": validation_report.get("hard_failed_count", validation_report.get("failed_count")),
            "warning_count": validation_report.get("warning_count"),
            "suspicious_count": validation_report.get("suspicious_count"),
            "durability_caution_count": validation_report.get("durability_caution_count", 0),
            "dataset_anomaly_count": validation_report.get("dataset_anomaly_count", 0),
        }
    )
    return record


def _evaluate_stage_candidate(
    *,
    experiment: dict[str, Any],
    config: dict[str, Any],
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_val: pd.DataFrame,
    y_val: pd.Series,
    validator: EngineeringValidator,
) -> tuple[Any, dict[str, Any]]:
    return evaluate_candidate(
        experiment["model_name"],
        dict(experiment["params"]),
        x_train,
        y_train,
        x_val,
        y_val,
        validator,
        config,
    )


def _build_stage_config(base_config: dict[str, Any], cv_repeats: int) -> dict[str, Any]:
    stage_config = copy.deepcopy(base_config)
    stage_config["experiment"]["cv_repeats"] = max(1, int(cv_repeats))
    return stage_config


def _evaluate_reference_if_possible(
    *,
    current_best_result: dict[str, Any],
    available_models: dict[str, dict[str, Any]],
    confirm_config: dict[str, Any],
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_val: pd.DataFrame,
    y_val: pd.Series,
    validator: EngineeringValidator,
) -> dict[str, Any]:
    model_name = str(current_best_result.get("model_name", ""))
    if model_name not in available_models:
        return current_best_result
    try:
        _, confirmed_reference = evaluate_candidate(
            model_name,
            dict(current_best_result.get("hyperparameters", {})),
            x_train,
            y_train,
            x_val,
            y_val,
            validator,
            confirm_config,
        )
    except Exception:
        return current_best_result
    confirmed_reference["source"] = "confirm_reference"
    return confirmed_reference


def _extract_feature_list(frame: Any) -> list[str]:
    columns = getattr(frame, "columns", None)
    if columns is None:
        return []
    return [str(column) for column in columns]


def _build_knowledge_context_for_loop(
    *,
    current_best: dict[str, Any],
    x_train: Any,
) -> str:
    model_name = str(current_best.get("model_name", "")).strip()
    if not model_name:
        return ""
    try:
        return get_knowledge_context(
            model_name,
            feature_list=_extract_feature_list(x_train),
            include_feature_engineering=True,
        )
    except Exception:
        return ""


def _compute_actual_delta_rmse(reference_result: dict[str, Any], candidate_result: dict[str, Any]) -> float | None:
    def _extract_rmse(payload: dict[str, Any]) -> float | None:
        for value in (
            payload.get("rmse"),
            payload.get("val_metrics", {}).get("rmse") if isinstance(payload.get("val_metrics"), dict) else None,
            payload.get("test_metrics", {}).get("rmse") if isinstance(payload.get("test_metrics"), dict) else None,
        ):
            try:
                if value is not None:
                    return float(value)
            except (TypeError, ValueError):
                continue
        return None

    reference_rmse = _extract_rmse(reference_result)
    candidate_rmse = _extract_rmse(candidate_result)
    if reference_rmse is None or candidate_rmse is None:
        return None
    return round(candidate_rmse - reference_rmse, 4)


def _archive_outcome_for_selection(selection_status: str) -> str:
    mapping = {
        "kept": "kept",
        "reverted": "rejected",
        "rejected_validation_fail": "rejected",
        "budget_exceeded": "rejected",
        "scout_no_improvement": "rejected",
        "error": "error",
    }
    return mapping.get(str(selection_status), str(selection_status))


def _register_llm_hypothesis_if_needed(
    *,
    archive: HypothesisArchive,
    experiment: dict[str, Any],
    run_id: str,
    cycle_number: int,
) -> str | None:
    if str(experiment.get("proposal_source")) != "llm":
        return None
    research_proposal = experiment.get("research_proposal")
    if not isinstance(research_proposal, dict):
        return None
    existing_id = experiment.get("hypothesis_id")
    if isinstance(existing_id, str) and existing_id.strip():
        return existing_id
    hypothesis_id = archive.add(
        proposal={
            "model_name": experiment.get("model_name"),
            "params": dict(experiment.get("params", {})),
        },
        claim=str(research_proposal.get("hypothesis", experiment.get("hypothesis", ""))),
        mechanism=str(research_proposal.get("rationale", "")),
        expected_delta_rmse=experiment.get("expected_delta_rmse"),
        source=str(experiment.get("proposal_source", "llm")),
        run_id=run_id,
        cycle=cycle_number,
        prediction=str(research_proposal.get("expected_direction", "")),
        test=str(research_proposal.get("proposed_change", "")),
        expected_metric_effect=research_proposal.get("expected_metric_effect"),
        confidence=research_proposal.get("confidence"),
        novelty_claim=str(research_proposal.get("novelty_claim", "")),
        risk_notes=str(research_proposal.get("risk_notes", "")),
        proposed_change=str(research_proposal.get("proposed_change", "")),
        target_component=str(research_proposal.get("target_component", experiment.get("model_name", ""))),
        change_type=str(research_proposal.get("change_type", "other")),
        expected_direction=str(research_proposal.get("expected_direction", "uncertain")),
    )
    experiment["hypothesis_id"] = hypothesis_id
    return hypothesis_id


def _resolve_llm_hypothesis_if_needed(
    *,
    archive: HypothesisArchive,
    experiment: dict[str, Any],
    trial_number: int,
    selection_status: str,
    actual_delta_rmse: float | None,
    actual_result: dict[str, Any] | None,
) -> None:
    hypothesis_id = experiment.get("hypothesis_id")
    if not isinstance(hypothesis_id, str) or not hypothesis_id.strip():
        return
    archive.resolve(
        hypothesis_id,
        actual_delta_rmse=actual_delta_rmse,
        outcome=_archive_outcome_for_selection(selection_status),
        trial_number=trial_number,
        actual_result=actual_result,
    )


def _propagate_confirm_metadata(
    *,
    confirm_candidates: list[dict[str, Any]],
    scout_results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_experiment_id = {
        str(candidate.get("experiment_id")): candidate
        for candidate in scout_results
    }
    enriched: list[dict[str, Any]] = []
    for candidate in confirm_candidates:
        updated = dict(candidate)
        source_experiment_id = str(updated.get("source_experiment_id", ""))
        source = by_experiment_id.get(source_experiment_id, {})
        for key in (
            "proposal_source",
            "proposal_status",
            "proposal_backend",
            "proposal_model",
            "prompt_hash",
            "confidence",
            "expected_delta_rmse",
            "research_proposal",
            "hypothesis_id",
        ):
            if key not in updated and key in source:
                updated[key] = source.get(key)
        enriched.append(updated)
    return enriched


def _build_proposal_provider(config: dict[str, Any]):
    llm_config = dict(config.get("llm", {}))
    llm_provider = LLMProposalProvider(config=config)
    det_provider = DeterministicProposalProvider(None, config)
    return HybridProposalProvider(
        llm_provider=llm_provider,
        deterministic_provider=det_provider,
        min_llm_proposals=int(llm_config.get("min_proposals", 1)),
    )


def _build_proposal_engine(config: dict[str, Any], outputs_dir: Path):
    del outputs_dir
    return _build_proposal_provider(config)
