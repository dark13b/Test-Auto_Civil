"""Autoresearch-style engineering ML loop for AutoCivil-Lab."""

from __future__ import annotations

import argparse
import copy
import logging
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from dataclasses import asdict

import pandas as pd

from artifact_sync import AtomicArtifactWriter
from failure_analyzer import FailureAnalyzer
from hypothesis_archive import HypothesisArchive
from knowledge_base import get_knowledge_context
import research_lab
from llm_admission_policy import evaluate_llm_admission_policy
from deterministic_proposal_provider import DeterministicProposalProvider
from hybrid_proposal_provider import HybridProposalProvider
from llm_proposal_provider import LLMProposalProvider
from loop_candidate_selection import select_scout_candidates
from proposal_engine import (
    ProposalBackendFailure,
    ProposalExtractionError,
    get_proposals,
    ProposalParseFailure,
    ProposalPreflightFailure,
    ProposalSemanticFailure,
    ProposalSchemaFailure,
    ProposalContext,
    ProposalProvider,
)
from research_protocol import (
    RESEARCH_RESULTS_COLUMNS,
    apply_keep_to_research_surface,
    build_acceptance_decision,
    filter_diverse_candidates,
    load_human_research_brief,
    load_json_file,
    load_or_initialize_experiment_memory,
    read_research_surface_state,
    record_experiment_memory,
    trial_budget_status,
    validate_final_artifact_consistency,
    write_json_file,
)
from search import get_available_model_configs
from train_impl import (
    EngineeringValidator,
    evaluate_candidate,
    get_outputs_dir,
    load_config,
    load_dataset,
    log_status,
    save_json_artifact,
    save_pickle_artifact,
    set_global_seed,
    split_dataset,
    write_run_scoped_json_artifact,
)


LOGGER = logging.getLogger("research_loop")
from validator import summarize_validation_report


RESEARCH_RESULTS_FILENAME = "research_results.csv"
RESEARCH_LOG_FILENAME = "research_log.txt"
PROPOSAL_DIVERSITY_FILENAME = "proposal_diversity.json"
FINAL_ARTIFACT_VALIDATION_FILENAME = "final_artifact_validation.json"
FINAL_ACCEPTANCE_FILENAME = "final_acceptance.json"
FINAL_METRICS_FILENAME = "final_metrics.json"
BEST_RESULT_FILENAME = "best_search_result.json"
BEST_MODEL_FILENAME = "best_search_model.pkl"
SEARCH_STATE_BEST_RESULT_FILENAME = "search_state_best_result.json"
SEARCH_STATE_BEST_MODEL_FILENAME = "search_state_best_model.pkl"
EXPERIMENT_MEMORY_FILENAME = "experiment_memory.json"
LLM_RUN_SUMMARY_FILENAME = "llm_run_summary.json"
LLM_SMOKE_TEST_FILENAME = "llm_smoke_test.json"
LLM_SMOKE_RELIABILITY_FILENAME = "llm_smoke_reliability.json"
RUN_MANIFEST_FILENAME = "run_manifest.json"
HYPOTHESIS_ARCHIVE_FILENAME = "hypothesis_archive.json"


def _safe_float(value: Any, fallback: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(fallback)


def calculate_improvement_percentage(reference_score: float, candidate_score: float) -> float:
    denominator = max(abs(reference_score), 1e-8)
    return ((candidate_score - reference_score) / denominator) * 100.0


def _initialize_results_csv(path: Path) -> None:
    if path.exists() and path.stat().st_size > 0:
        return
    pd.DataFrame(columns=RESEARCH_RESULTS_COLUMNS).to_csv(path, index=False)


def _append_results_row(path: Path, record: dict[str, Any]) -> None:
    pd.DataFrame([record], columns=RESEARCH_RESULTS_COLUMNS).to_csv(
        path,
        mode="a",
        header=not path.exists() or path.stat().st_size == 0,
        index=False,
    )


def _build_results_sync_writer(outputs_dir: Path) -> AtomicArtifactWriter:
    return AtomicArtifactWriter(
        outputs_dir=outputs_dir,
        csv_targets=[
            (RESEARCH_RESULTS_FILENAME, RESEARCH_RESULTS_COLUMNS),
            ("optuna_results.csv", RESEARCH_RESULTS_COLUMNS),
        ],
    )


def _append_synchronized_results(sync_writer: AtomicArtifactWriter, record: dict[str, Any]) -> None:
    sync_writer.append_trial(record)


def _initialize_research_log(path: Path, baseline_metrics: dict[str, Any]) -> None:
    timestamp = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")
    baseline_line = (
        f"[{timestamp}] Baseline | Model: {baseline_metrics['model_name']} | "
        f"Composite: {baseline_metrics['composite_score']:.4f} | "
        f"Validation: {baseline_metrics['validation_verdict']}"
    )
    if path.exists() and path.stat().st_size > 0:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(f"\n=== New Research Run {timestamp} ===\n")
            handle.write(baseline_line + "\n")
        return
    with path.open("w", encoding="utf-8") as handle:
        handle.write("AutoCivil-Lab Engineering Research Log\n")
        handle.write(baseline_line + "\n")


def _append_research_log(path: Path, line: str) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def _increment_count(mapping: dict[str, int], key: Any, *, amount: int = 1) -> None:
    normalized_key = str(key)
    mapping[normalized_key] = mapping.get(normalized_key, 0) + int(amount)


def _build_run_manifest_payload(
    *,
    run_id: str,
    run_started_at: str,
    run_finished_at: str,
    config: dict[str, Any],
    brief: dict[str, Any],
    llm_config: dict[str, Any],
    allow_deterministic_fallback: bool,
    preflight_result: dict[str, Any] | None,
    proposal_metadata: dict[str, Any] | None,
    smoke_summary: dict[str, Any] | None,
    proposal_status_counts: dict[str, int],
    semantic_rejection_counts: dict[str, int],
    llm_failure_counts: dict[str, int],
    archive_update_summary: dict[str, int],
    number_of_candidates_evaluated: int,
    holdout_touched_during_search: bool,
    final_abort_reason: str | None,
    final_run_status: str,
    final_metrics: dict[str, Any] | None,
    acceptance: dict[str, Any] | None,
    validation_report: dict[str, Any] | None,
) -> dict[str, Any]:
    backend = None
    model = None
    smoke_test_status = None
    preflight_status = None
    if isinstance(preflight_result, dict):
        backend = preflight_result.get("backend")
        model = preflight_result.get("model")
        smoke_test_status = preflight_result.get("status")
        preflight_status = "passed" if bool(preflight_result.get("ok", False)) else "failed"
    elif isinstance(proposal_metadata, dict):
        backend = proposal_metadata.get("proposal_backend")
        model = proposal_metadata.get("proposal_model")

    selection_metric_name = str(brief.get("acceptance_metric") or "composite_score")
    manifest = {
        "run_id": run_id,
        "timestamp": {
            "started_at": run_started_at,
            "finished_at": run_finished_at,
        },
        "backend": backend,
        "model": model,
        "fallback_allowed": bool(allow_deterministic_fallback),
        "preflight_status": preflight_status,
        "smoke_test_status": smoke_test_status,
        "proposal_status_counts": dict(sorted(proposal_status_counts.items())),
        "semantic_rejection_counts": dict(sorted(semantic_rejection_counts.items())),
        "final_abort_reason": final_abort_reason,
        "holdout_touched_during_search": bool(holdout_touched_during_search),
        "selection_metric_name": selection_metric_name,
        "selection_partition": "validation",
        "final_holdout_metric_name": None,
        "number_of_candidates_evaluated": int(number_of_candidates_evaluated),
        "llm_failure_counts": dict(sorted(llm_failure_counts.items())),
        "archive_update_summary": dict(sorted(archive_update_summary.items())) or None,
        "final_run_status": final_run_status,
        "runtime_context": {
            "backend_mode": llm_config.get("backend_mode"),
            "llm_enabled": bool(llm_config.get("enabled", False)),
            "selection_metric_name": selection_metric_name,
            "config_random_seed": config.get("experiment", {}).get("random_seed"),
        },
    }
    if isinstance(smoke_summary, dict):
        manifest["smoke_test_summary"] = smoke_summary
        failure_counts = smoke_summary.get("failure_class_counts", {})
        manifest["llm_failure_counts"] = dict(sorted(failure_counts.items())) if isinstance(failure_counts, dict) else dict(sorted(llm_failure_counts.items()))
        admission_policy = smoke_summary.get("admission_policy", {})
        manifest["smoke_test_status"] = smoke_summary.get("compatibility", {}).get("interpretation", smoke_test_status)
        manifest["smoke_test_verdict"] = admission_policy.get("verdict")
        manifest["smoke_test_admission_policy"] = admission_policy
    if isinstance(final_metrics, dict):
        holdout_metrics = final_metrics.get("holdout_metrics")
        if isinstance(holdout_metrics, dict) and holdout_metrics:
            manifest["final_holdout_metric_name"] = next(iter(holdout_metrics.keys()))
    if isinstance(acceptance, dict):
        manifest["acceptance_decision"] = {
            "accepted": acceptance.get("accepted"),
            "decision_reason": acceptance.get("decision_reason"),
            "source_mode": acceptance.get("source_mode"),
        }
    if isinstance(validation_report, dict):
        manifest["final_artifact_validation"] = {
            "consistent": validation_report.get("consistent"),
            "mismatches": validation_report.get("mismatches", []),
        }
    return manifest


def _write_run_manifest(outputs_dir: Path, payload: dict[str, Any]) -> None:
    run_id = str(payload.get("run_id", "unknown"))
    write_run_scoped_json_artifact(
        outputs_dir=outputs_dir,
        filename=RUN_MANIFEST_FILENAME,
        payload=payload,
        run_id=run_id,
        source_mode="manifest",
    )


def _load_run_manifest_counts(
    *,
    memory_path: Path,
    archive_path: Path,
    run_id: str,
) -> tuple[dict[str, int], dict[str, int], dict[str, int], dict[str, int]]:
    proposal_status_counts: dict[str, int] = {}
    semantic_rejection_counts: dict[str, int] = {}
    llm_failure_counts: dict[str, int] = {}
    archive_update_summary: dict[str, int] = {"added": 0, "resolved": 0}

    if memory_path.exists():
        memory_payload = load_json_file(memory_path)
        for run in memory_payload.get("runs", []):
            if str(run.get("run_id")) != str(run_id):
                continue
            for trial in run.get("trials", []):
                _increment_count(proposal_status_counts, trial.get("selection_status", "unknown"))
                if str(trial.get("proposal_source")) == "llm" and trial.get("proposal_status") == "error":
                    _increment_count(llm_failure_counts, "error")
                semantic_result = str(trial.get("semantic_validation_result", "")).lower()
                if semantic_result == "failed":
                    rejection = trial.get("semantic_rejection_reason")
                    if isinstance(rejection, dict):
                        _increment_count(semantic_rejection_counts, rejection.get("code", "unknown"))
                    else:
                        _increment_count(semantic_rejection_counts, trial.get("proposal_status", "unknown"))

    if archive_path.exists():
        archive_payload = load_json_file(archive_path)
        for record in archive_payload.get("records", []):
            if str(record.get("run_id")) != str(run_id):
                continue
            if record.get("timestamp") is not None:
                archive_update_summary["added"] += int(record.get("outcome") == "pending")
                archive_update_summary["resolved"] += int(record.get("outcome") not in {None, "pending"})
            if record.get("calibration_error") is not None and record.get("outcome") is not None:
                _increment_count(llm_failure_counts, str(record.get("outcome")))
    return proposal_status_counts, semantic_rejection_counts, llm_failure_counts, archive_update_summary


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
    best_result_path = outputs_dir / BEST_RESULT_FILENAME
    if best_result_path.exists():
        payload = load_json_file(best_result_path)
        payload.setdefault("source", "best_search_result")
        return payload

    final_metrics_path = outputs_dir / FINAL_METRICS_FILENAME
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


def _build_validation_summary(validation_report: dict[str, Any]) -> dict[str, Any]:
    return summarize_validation_report(validation_report)


def _build_final_metrics_payload(
    baseline_metrics: dict[str, Any],
    best_result: dict[str, Any],
) -> dict[str, Any]:
    improvement_pct = calculate_improvement_percentage(
        _safe_float(baseline_metrics.get("composite_score")),
        _safe_float(best_result.get("composite_score")),
    )
    validation_report = dict(best_result.get("validation_report", {}))
    return {
        "baseline_metrics": copy.deepcopy(baseline_metrics),
        "best_search_metrics": copy.deepcopy(best_result),
        "improvement_percentage": improvement_pct,
        "composite_improvement_pct": improvement_pct,
        "validation_verdict": best_result["validation_verdict"],
        "best_model_name": best_result["model_name"],
        "best_model_hyperparameters": copy.deepcopy(best_result["hyperparameters"]),
        "validation_summary": _build_validation_summary(validation_report),
        "validation_metrics": copy.deepcopy(
            best_result.get("selection_metrics", best_result.get("val_metrics", best_result.get("test_metrics", {})))
        ),
    }


def _sync_final_artifacts_from_source_of_truth(
    *,
    outputs_dir: Path,
    baseline_metrics: dict[str, Any],
    best_result: dict[str, Any],
    best_model_source_path: Path,
) -> dict[str, Any]:
    final_metrics_payload = _build_final_metrics_payload(baseline_metrics, best_result)
    save_json_artifact(outputs_dir / FINAL_METRICS_FILENAME, final_metrics_payload)

    best_from_truth = copy.deepcopy(final_metrics_payload["best_search_metrics"])
    save_json_artifact(outputs_dir / BEST_RESULT_FILENAME, best_from_truth)
    save_json_artifact(outputs_dir / SEARCH_STATE_BEST_RESULT_FILENAME, best_from_truth)

    if not best_model_source_path.exists():
        raise FileNotFoundError(f"Best model artifact is missing: {best_model_source_path}")
    best_model_bytes = best_model_source_path.read_bytes()
    (outputs_dir / BEST_MODEL_FILENAME).write_bytes(best_model_bytes)
    (outputs_dir / SEARCH_STATE_BEST_MODEL_FILENAME).write_bytes(best_model_bytes)

    validation_report = validate_final_artifact_consistency(outputs_dir)
    write_json_file(outputs_dir / FINAL_ARTIFACT_VALIDATION_FILENAME, validation_report)
    if not validation_report["consistent"]:
        raise RuntimeError(f"Final artifact mismatch: {validation_report['mismatches']}")
    return final_metrics_payload


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
) -> dict[str, Any]:
    record = {column: None for column in RESEARCH_RESULTS_COLUMNS}
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


def _write_diversity_summary(
    *,
    outputs_dir: Path,
    run_id: str,
    current_run_signatures: set[tuple[str, str]],
    memory_payload: dict[str, Any],
) -> None:
    family_counts: dict[str, int] = {}
    for run in memory_payload.get("runs", []):
        for trial in run.get("trials", []):
            proposal_family = str(trial.get("proposal_family", "unknown"))
            family_counts[proposal_family] = family_counts.get(proposal_family, 0) + 1
    payload = {
        "run_id": run_id,
        "unique_signatures_this_run": len(current_run_signatures),
        "historic_family_counts": family_counts,
    }
    write_json_file(outputs_dir / PROPOSAL_DIVERSITY_FILENAME, payload)


def _build_diversity_state(memory_payload: dict[str, Any]) -> dict[str, Any]:
    family_counts: dict[str, int] = {}
    for run in memory_payload.get("runs", []):
        for trial in run.get("trials", []):
            proposal_family = str(trial.get("proposal_family", "unknown"))
            family_counts[proposal_family] = family_counts.get(proposal_family, 0) + 1
    return {"historic_family_counts": family_counts}


def _normalize_llm_candidates(
    candidates: list[dict[str, Any]],
    available_models: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates, start=1):
        model_name = str(candidate.get("model_name", ""))
        display_name = str(
            candidate.get(
                "display_name",
                available_models.get(model_name, {}).get("display_name", model_name),
            )
        )
        normalized.append(
            {
                "experiment_id": str(candidate.get("experiment_id", f"llm-scout-{index:03d}")),
                "stage": str(candidate.get("stage", "scout")),
                "model_name": model_name,
                "display_name": display_name,
                "proposal_family": str(candidate.get("proposal_family", f"{model_name}-llm")),
                "proposal_source": str(candidate.get("proposal_source", "llm")),
                "proposal_status": str(candidate.get("proposal_status", "llm_success")),
                "proposal_backend": candidate.get("proposal_backend"),
                "proposal_model": candidate.get("proposal_model"),
                "prompt_hash": candidate.get("prompt_hash"),
                "hypothesis": str(candidate.get("hypothesis", "LLM-generated suggestion")),
                "confidence": candidate.get("confidence"),
                "expected_delta_rmse": candidate.get("expected_delta_rmse"),
                "research_proposal": candidate.get("research_proposal"),
                "params": dict(candidate.get("params", {})),
            }
        )
    return normalized


def _mark_fallback_candidates(candidates: list[dict[str, Any]], *, status: str) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates, start=1):
        updated = dict(candidate)
        updated.setdefault("experiment_id", f"fallback-scout-{index:03d}")
        updated.setdefault("stage", "scout")
        updated["proposal_source"] = "deterministic_fallback"
        updated["proposal_status"] = status
        updated["proposal_backend"] = "fallback"
        updated["proposal_model"] = None
        updated["prompt_hash"] = None
        updated["confidence"] = 0.0
        updated["expected_delta_rmse"] = None
        updated["research_proposal"] = {
            "hypothesis": str(updated.get("hypothesis", "Deterministic fallback proposal")),
            "rationale": "Generated by the deterministic search fallback because the LLM proposal path was unavailable.",
            "change_type": "other",
            "target_component": str(updated.get("model_name", "")),
            "proposed_change": json.dumps(updated.get("params", {}), sort_keys=True),
            "expected_direction": "uncertain",
            "expected_metric_effect": {
                "metric": "rmse",
                "direction": "down",
                "magnitude_estimate": "unknown",
            },
            "confidence": 0.0,
            "novelty_claim": "Fallback-generated proposal; no LLM novelty claim available.",
            "risk_notes": "This proposal was not generated by an LLM.",
            "candidate_config": {
                "model_name": str(updated.get("model_name", "")),
                "params": dict(updated.get("params", {})),
            },
        }
        normalized.append(updated)
    return normalized


def _build_proposal_provider(config: dict[str, Any]) -> ProposalProvider:
    llm_config = dict(config.get("llm", {}))
    llm_provider = LLMProposalProvider(config=config)
    det_provider = DeterministicProposalProvider(None, config)
    return HybridProposalProvider(
        llm_provider=llm_provider,
        deterministic_provider=det_provider,
        min_llm_proposals=int(llm_config.get("min_proposals", 1)),
    )


def _build_proposal_engine(config: dict[str, Any], outputs_dir: Path) -> ProposalProvider:
    del outputs_dir
    return _build_proposal_provider(config)


def _apply_llm_smoke_overrides(
    config: dict[str, Any],
    *,
    backend_mode: str | None = None,
) -> dict[str, Any]:
    if not backend_mode:
        return config
    updated = copy.deepcopy(config)
    llm_config = dict(updated.get("llm", {}))
    llm_config["enabled"] = True
    llm_config["backend_mode"] = backend_mode
    updated["llm"] = llm_config
    return updated


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


def _run_llm_preflight(
    *,
    proposal_engine: ProposalEngine | None,
    outputs_dir: Path,
    available_models: dict[str, dict[str, Any]],
    brief: dict[str, Any],
    current_best: dict[str, Any],
    memory_payload: dict[str, Any],
    knowledge_context: str,
    failure_patterns: dict[str, Any],
    archive_records: list[dict[str, Any]],
    llm_enabled: bool,
    allow_deterministic_fallback: bool,
    model_hint: str | None = None,
) -> dict[str, Any] | None:
    if proposal_engine is None or not hasattr(proposal_engine, "run_proposal_smoke_test"):
        if not llm_enabled:
            result = {
                "ok": False,
                "status": "aborted_due_to_preflight_failure",
                "backend": "disabled",
                "model": None,
                "prompt_hash": None,
                "proposal_preview": None,
                "error": "LLM proposal engine is disabled by config.",
            }
            write_json_file(outputs_dir / LLM_SMOKE_TEST_FILENAME, result)
            if not allow_deterministic_fallback:
                raise ProposalPreflightFailure(
                    result["error"],
                    failure_kind="aborted_due_to_preflight_failure",
                    interaction=result,
                )
            return result
        return None

    smoke_result = proposal_engine.run_proposal_smoke_test(
        available_models=available_models,
        research_brief=brief,
        current_best=current_best,
        experiment_memory=memory_payload,
        diversity_state=_build_diversity_state(memory_payload),
        trial_history=[
            trial
            for run in memory_payload.get("runs", [])
            for trial in run.get("trials", [])
        ],
        search_progress={"phase": "preflight"},
        failure_patterns=failure_patterns,
        knowledge_context=knowledge_context,
        archive_records=archive_records,
        model_hint=model_hint,
    )
    write_json_file(outputs_dir / LLM_SMOKE_TEST_FILENAME, smoke_result)
    if not bool(smoke_result.get("ok", False)) and not allow_deterministic_fallback:
        raise ProposalPreflightFailure(
            str(smoke_result.get("error") or "LLM smoke test failed."),
            failure_kind="aborted_due_to_preflight_failure",
            interaction=smoke_result,
        )
    return smoke_result


def _percentile(sorted_values: list[float], percentile: float) -> float | None:
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return sorted_values[0]
    bounded = min(max(percentile, 0.0), 1.0)
    index = max(0, min(len(sorted_values) - 1, int((len(sorted_values) * bounded) + 0.999999) - 1))
    return sorted_values[index]


def summarize_smoke_test_trials(
    trials: list[dict[str, Any]],
    *,
    requested_backend: str | None = None,
    requested_model: str | None = None,
    admission_policy_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    total_trials = len(trials)
    if total_trials == 0:
        summary = {
            "total_trials": 0,
            "success_count": 0,
            "success_rate": 0.0,
            "backend_failure_count": 0,
            "backend_failure_rate": 0.0,
            "parse_failure_count": 0,
            "parse_failure_rate": 0.0,
            "schema_failure_count": 0,
            "schema_failure_rate": 0.0,
            "semantic_failure_count": 0,
            "semantic_failure_rate": 0.0,
            "hidden_channel_count": 0,
            "hidden_channel_incidence": 0.0,
            "empty_visible_count": 0,
            "empty_visible_response_incidence": 0.0,
            "repair_usage_rate": 0.0,
            "median_latency_seconds": None,
            "p95_latency_seconds": None,
            "status_counts": {},
            "backend_counts": {},
            "model_counts": {},
            "backend_model_counts": {},
            "failure_class_counts": {},
            "unexpected_backend_model_pairs": {},
        }
        admission_policy = evaluate_llm_admission_policy(summary, admission_policy_config=admission_policy_config)
        summary["admission_policy"] = admission_policy
        summary["compatibility"] = {
            "issue_codes": admission_policy["issue_codes"],
            "issues": [check for check in admission_policy["checks"] if check["status"] != "passed"],
            "interpretation": admission_policy["interpretation"],
            "verdict": admission_policy["verdict"],
            "eligible_for_control_tasks": admission_policy["eligible_for_control_tasks"],
        }
        return summary

    status_counts: dict[str, int] = {}
    backend_counts: dict[str, int] = {}
    model_counts: dict[str, int] = {}
    backend_model_counts: dict[str, int] = {}
    failure_class_counts: dict[str, int] = {}
    unexpected_pairs: dict[str, int] = {}
    success_count = 0
    backend_failure_count = 0
    parse_failure_count = 0
    schema_failure_count = 0
    semantic_failure_count = 0
    hidden_channel_count = 0
    empty_visible_count = 0
    repair_count = 0
    latencies: list[float] = []

    for trial in trials:
        status = str(trial.get("status", "unknown"))
        backend = str(trial.get("backend", "unknown"))
        model = str(trial.get("model", "unknown"))
        channel = str(trial.get("extracted_from_channel", "unknown"))
        pair = f"{backend}::{model}"

        status_counts[status] = status_counts.get(status, 0) + 1
        backend_counts[backend] = backend_counts.get(backend, 0) + 1
        model_counts[model] = model_counts.get(model, 0) + 1
        backend_model_counts[pair] = backend_model_counts.get(pair, 0) + 1

        failure_class = trial.get("failure_class")
        if failure_class:
            code = str(failure_class)
            failure_class_counts[code] = failure_class_counts.get(code, 0) + 1

        if bool(trial.get("ok", False)):
            success_count += 1
        if status == "llm_backend_failure":
            backend_failure_count += 1
        if status == "llm_parse_failure":
            parse_failure_count += 1
        if status == "llm_schema_failure":
            schema_failure_count += 1
        semantic_result = str(trial.get("semantic_validation_result", "")).lower()
        if semantic_result == "failed" or status in {
            "duplicate_proposal",
            "near_duplicate_proposal",
            "semantically_invalid_candidate_config",
            "unsupported_novelty_claim",
            "inconsistent_expected_effect",
            "non_executable_semantic_config",
        }:
            semantic_failure_count += 1
        if channel in {"thinking", "repaired_thinking"}:
            hidden_channel_count += 1
        if bool(trial.get("visible_response_empty", False)):
            empty_visible_count += 1
        if bool(trial.get("repair_used", False)):
            repair_count += 1

        latency = trial.get("latency_seconds")
        if isinstance(latency, (int, float)):
            latencies.append(float(latency))

        backend_unexpected = requested_backend not in {None, "", "hybrid"} and backend != requested_backend
        model_unexpected = bool(requested_model) and model != requested_model
        if backend_unexpected or model_unexpected:
            unexpected_pairs[pair] = unexpected_pairs.get(pair, 0) + 1

    latencies.sort()
    success_rate = success_count / total_trials
    backend_failure_rate = backend_failure_count / total_trials
    parse_failure_rate = parse_failure_count / total_trials
    schema_failure_rate = schema_failure_count / total_trials
    semantic_failure_rate = semantic_failure_count / total_trials
    hidden_channel_incidence = hidden_channel_count / total_trials
    empty_visible_response_incidence = empty_visible_count / total_trials
    repair_usage_rate = repair_count / total_trials

    summary = {
        "total_trials": total_trials,
        "success_count": success_count,
        "success_rate": success_rate,
        "backend_failure_count": backend_failure_count,
        "backend_failure_rate": backend_failure_rate,
        "parse_failure_count": parse_failure_count,
        "parse_failure_rate": parse_failure_rate,
        "schema_failure_count": schema_failure_count,
        "schema_failure_rate": schema_failure_rate,
        "semantic_failure_count": semantic_failure_count,
        "semantic_failure_rate": semantic_failure_rate,
        "hidden_channel_count": hidden_channel_count,
        "hidden_channel_incidence": hidden_channel_incidence,
        "empty_visible_count": empty_visible_count,
        "empty_visible_response_incidence": empty_visible_response_incidence,
        "repair_usage_rate": repair_usage_rate,
        "median_latency_seconds": _percentile(latencies, 0.5),
        "p95_latency_seconds": _percentile(latencies, 0.95),
        "status_counts": status_counts,
        "backend_counts": backend_counts,
        "model_counts": model_counts,
        "backend_model_counts": backend_model_counts,
        "failure_class_counts": failure_class_counts,
        "unexpected_backend_model_pairs": unexpected_pairs,
    }
    admission_policy = evaluate_llm_admission_policy(summary, admission_policy_config=admission_policy_config)
    summary["admission_policy"] = admission_policy
    summary["compatibility"] = {
        "issue_codes": admission_policy["issue_codes"],
        "issues": [check for check in admission_policy["checks"] if check["status"] != "passed"],
        "interpretation": admission_policy["interpretation"],
        "verdict": admission_policy["verdict"],
        "eligible_for_control_tasks": admission_policy["eligible_for_control_tasks"],
    }
    return summary


def run_repeated_llm_smoke_test(
    *,
    proposal_engine: ProposalEngine | None,
    available_models: dict[str, dict[str, Any]],
    brief: dict[str, Any],
    current_best: dict[str, Any],
    memory_payload: dict[str, Any],
    knowledge_context: str,
    failure_patterns: dict[str, Any],
    archive_records: list[dict[str, Any]],
    trials: int,
    summary_path: Path | None = None,
    requested_backend: str | None = None,
    model_hint: str | None = None,
    admission_policy_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    requested_trials = max(1, int(trials))
    timestamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    trial_history = [
        trial
        for run in memory_payload.get("runs", [])
        for trial in run.get("trials", [])
    ]
    normalized_trials: list[dict[str, Any]] = []

    for trial_index in range(1, requested_trials + 1):
        if proposal_engine is None or not hasattr(proposal_engine, "run_proposal_smoke_test"):
            trial_result = {
                "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                "ok": False,
                "status": "aborted_due_to_preflight_failure",
                "backend": requested_backend or "disabled",
                "model": model_hint,
                "prompt_hash": None,
                "proposal_preview": None,
                "error": "LLM proposal engine is disabled by config.",
                "extracted_from_channel": "response",
                "visible_response_empty": True,
                "parse_success": False,
                "schema_success": False,
                "repair_used": False,
                "latency_seconds": None,
                "failure_class": "llm_backend_failure",
            }
        else:
            trial_result = proposal_engine.run_proposal_smoke_test(
                available_models=available_models,
                research_brief=brief,
                current_best=current_best,
                experiment_memory=memory_payload,
                diversity_state=_build_diversity_state(memory_payload),
                trial_history=trial_history,
                search_progress={"phase": "preflight", "trial_index": trial_index, "trial_count": requested_trials},
                failure_patterns=failure_patterns,
                knowledge_context=knowledge_context,
                archive_records=archive_records,
                model_hint=model_hint,
            )
        normalized = dict(trial_result)
        normalized["trial_index"] = trial_index
        normalized.setdefault("timestamp", datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"))
        normalized_trials.append(normalized)

    summary = summarize_smoke_test_trials(
        normalized_trials,
        requested_backend=requested_backend or (getattr(proposal_engine, "backend_name", None) if proposal_engine is not None else None),
        requested_model=model_hint,
        admission_policy_config=admission_policy_config,
    )
    report = {
        "generated_at": timestamp,
        "requested_trials": requested_trials,
        "requested_backend": requested_backend or (getattr(proposal_engine, "backend_name", None) if proposal_engine is not None else None),
        "requested_model": model_hint,
        "summary_path": str(summary_path) if summary_path is not None else None,
        "trials": normalized_trials,
        "summary": summary,
    }
    if summary_path is not None:
        write_json_file(summary_path, report)
    return report


def _select_scout_candidates(
    *,
    brief: dict[str, Any],
    lab_state: dict[str, Any],
    available_models: dict[str, dict[str, Any]],
    memory_payload: dict[str, Any],
    current_best: dict[str, Any],
    scout_limit: int,
    family_limit: int,
    current_run_signatures: set[tuple[str, str]],
    proposal_engine: ProposalProvider | None,
    allow_deterministic_fallback: bool = False,
    trial_history: list[dict[str, Any]] | None = None,
    search_progress: dict[str, Any] | None = None,
    failure_patterns: dict[str, Any] | None = None,
    knowledge_context: str | None = None,
    archive_records: list[dict[str, Any]] | None = None,
    preflight_result: dict[str, Any] | None = None,
    exploit_delta_ratio: float = 0.15,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    llm_config = dict(getattr(proposal_engine, "llm_config", {}) or {})
    context = {
        "brief": brief,
        "lab_state": lab_state,
        "available_models": available_models,
        "memory_payload": memory_payload,
        "current_best": current_best,
        "diversity_state": _build_diversity_state(memory_payload),
        "scout_limit": scout_limit,
        "trial_history": trial_history or [],
        "search_progress": search_progress or {},
        "failure_patterns": failure_patterns or {},
        "knowledge_context": knowledge_context or "",
        "archive_records": archive_records or [],
        "exploit_delta_ratio": exploit_delta_ratio,
    }
    if proposal_engine is None:
        return [], {
            "proposal_mode": "fallback_used",
            "proposal_status": "fallback_used",
            "proposal_backend": "fallback",
            "proposal_count": 0,
            "proposal_model": None,
            "prompt_variant": None,
            "proposal_error": None,
        }
    proposal_context = ProposalContext(
        brief=brief,
        recent_results=list(trial_history or []),
        best_metrics=current_best,
        config=context,
        trial_budget_remaining=int(scout_limit),
    )
    if hasattr(proposal_engine, "get_proposals"):
        llm_candidates = proposal_engine.get_proposals(proposal_context, n=scout_limit)
    elif hasattr(proposal_engine, "generate_research_proposals"):
        try:
            llm_result = proposal_engine.generate_research_proposals(
            available_models=available_models,
            research_brief=brief,
            current_best=current_best,
            experiment_memory=memory_payload,
            diversity_state=_build_diversity_state(memory_payload),
            proposal_count=scout_limit,
            trial_history=trial_history or [],
            search_progress=search_progress or {},
            failure_patterns=failure_patterns or {},
            knowledge_context=knowledge_context or "",
            model_hint=None,
            )
        except Exception as exc:
            if allow_deterministic_fallback and exc.__class__.__name__ in {"ProposalExtractionError", "ProposalParseFailure"}:
                llm_candidates = []
            else:
                raise
        else:
            llm_candidates = list(llm_result.get("proposals", [])) if isinstance(llm_result, dict) else list(llm_result or [])
    else:
        llm_candidates = get_proposals(  # type: ignore[name-defined]
            context,
            {"llm": llm_config},
            logger=LOGGER,
            n=scout_limit,
        )
    llm_candidates = [
        asdict(candidate) if hasattr(candidate, "__dataclass_fields__") else candidate
        for candidate in llm_candidates
    ]
    llm_candidates = _normalize_llm_candidates(list(llm_candidates), available_models)
    llm_candidates = filter_diverse_candidates(
        candidates=llm_candidates,
        memory_payload=memory_payload,
        current_run_signatures=current_run_signatures,
        family_limit=family_limit,
        scout_limit=scout_limit,
    )
    if llm_candidates:
        return llm_candidates, {
            "proposal_mode": "llm",
            "proposal_status": "llm_success",
            "proposal_backend": "proposal_provider",
            "proposal_count": len(llm_candidates),
            "proposal_model": None,
            "prompt_variant": None,
            "proposal_error": None,
        }
    if allow_deterministic_fallback:
        deterministic_candidates = research_lab.scout_experiments(
            brief=brief,
            lab_state=lab_state,
            available_models=available_models,
            experiment_memory=memory_payload,
            current_best=current_best,
            scout_limit=max(1, scout_limit),
            exploit_delta_ratio=exploit_delta_ratio,
        )
        deterministic_candidates = filter_diverse_candidates(
            candidates=deterministic_candidates,
            memory_payload=memory_payload,
            current_run_signatures=current_run_signatures,
            family_limit=family_limit,
            scout_limit=scout_limit,
        )
        return deterministic_candidates, {
            "proposal_mode": "fallback_used",
            "proposal_status": "fallback_used",
            "proposal_backend": "fallback",
            "proposal_count": len(deterministic_candidates),
            "proposal_model": None,
            "prompt_variant": None,
            "proposal_error": None,
        }
    return [], {
        "proposal_mode": "fallback_used",
        "proposal_status": "fallback_used",
        "proposal_backend": "fallback",
        "proposal_count": 0,
        "proposal_model": None,
        "prompt_variant": None,
        "proposal_error": None,
    }


def analyze_interaction_log(outputs_dir: Path, *, filename: str = "llm_interactions.jsonl") -> dict[str, Any]:
    interaction_path = outputs_dir / filename
    if not interaction_path.exists():
        return {
            "exists": False,
            "interaction_count": 0,
            "channel_counts": {},
            "rejection_counts": {},
            "semantic_rejection_counts": {},
        }

    channel_counts: dict[str, int] = {}
    rejection_counts: dict[str, int] = {}
    semantic_rejection_counts: dict[str, int] = {}
    repair_count = 0
    parse_success_count = 0
    with interaction_path.open("r", encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    for record in records:
        channel = str(record.get("extracted_from_channel", "unknown"))
        channel_counts[channel] = channel_counts.get(channel, 0) + 1
        if bool(record.get("repair_used", False)):
            repair_count += 1
        if bool(record.get("parse_success", False)):
            parse_success_count += 1
        rejection = record.get("rejection_reason")
        if isinstance(rejection, dict):
            code = str(rejection.get("code", "unknown"))
            rejection_counts[code] = rejection_counts.get(code, 0) + 1
        semantic_rejection = record.get("semantic_rejection_reason")
        if isinstance(semantic_rejection, dict):
            code = str(semantic_rejection.get("code", "unknown"))
            semantic_rejection_counts[code] = semantic_rejection_counts.get(code, 0) + 1
    return {
        "exists": True,
        "interaction_count": len(records),
        "channel_counts": channel_counts,
        "rejection_counts": rejection_counts,
        "semantic_rejection_counts": semantic_rejection_counts,
        "repair_count": repair_count,
        "parse_success_count": parse_success_count,
    }


def run_engineering_research_loop(
    *,
    cycles_override: int | None = None,
    with_report: bool = False,
) -> dict[str, Any]:
    """Run the governed scout-confirm-keep research loop."""
    config = load_config()
    set_global_seed(int(config["experiment"]["random_seed"]))
    outputs_dir = get_outputs_dir(config)
    baseline_metrics = _read_baseline_metrics(outputs_dir, config)
    current_best_result = _load_current_best_result(outputs_dir, baseline_metrics)

    research_config = dict(config.get("research", {}))
    project_root = Path(__file__).resolve().parent
    brief_path = project_root / str(research_config.get("brief_path", "research_brief.md"))
    research_lab_path = project_root / str(research_config.get("editable_surface_path", "research_lab.py"))
    brief = load_human_research_brief(brief_path)
    log_status(
        "INFO research_brief_loaded | "
        f"path={brief_path} | accept_metric={brief['acceptance_metric']} | "
        f"min_improvement_pct={brief['min_improvement_pct']:.4f}"
    )

    available_models = get_available_model_configs(config)
    data = load_dataset(config)
    x_train, x_val, _, y_train, y_val, _ = split_dataset(data, config)
    validator = EngineeringValidator.from_config(config)

    results_path = outputs_dir / RESEARCH_RESULTS_FILENAME
    compatibility_results_path = outputs_dir / "optuna_results.csv"
    log_path = outputs_dir / RESEARCH_LOG_FILENAME
    memory_path = outputs_dir / str(research_config.get("memory_filename", EXPERIMENT_MEMORY_FILENAME))
    memory_payload = load_or_initialize_experiment_memory(memory_path)
    _initialize_results_csv(results_path)
    _initialize_results_csv(compatibility_results_path)
    results_sync_writer = _build_results_sync_writer(outputs_dir)
    _initialize_research_log(log_path, baseline_metrics)

    run_id = pd.Timestamp.now().strftime("%Y%m%dT%H%M%S")
    run_started_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    current_run_signatures: set[tuple[str, str]] = set()
    proposal_engine = _build_proposal_engine(config, outputs_dir)
    llm_config = dict(config.get("llm", {}))
    allow_deterministic_fallback = bool(llm_config.get("allow_deterministic_fallback", False))
    max_cycles = max(1, int(cycles_override or research_config.get("max_cycles", 6)))
    family_limit = max(1, int(research_config.get("max_family_repeats_per_cycle", 1)))
    scout_repeats = max(1, int(research_config.get("scout_cv_repeats", 1)))
    confirm_repeats = max(1, int(research_config.get("confirm_cv_repeats", 3)))
    max_trial_seconds = float(research_config.get("max_trial_seconds", 0.0))
    max_runtime_minutes = float(research_config.get("max_runtime_minutes", 0.0))
    max_runtime_seconds = max_runtime_minutes * 60.0
    rebuild_reports_on_keep = bool(research_config.get("rebuild_reports_on_keep", True))
    research_start = time.perf_counter()

    trial_number = 0
    scout_config = _build_stage_config(config, scout_repeats)
    confirm_config = _build_stage_config(config, confirm_repeats)
    archive_path = outputs_dir / str(research_config.get("hypothesis_archive_filename", HYPOTHESIS_ARCHIVE_FILENAME))
    hypothesis_archive = HypothesisArchive(archive_path)
    knowledge_context = _build_knowledge_context_for_loop(current_best=current_best_result, x_train=x_train)
    failure_patterns = FailureAnalyzer(memory_path).extract_failure_patterns()
    archive_records = hypothesis_archive.load().get("records", [])
    preflight_result: dict[str, Any] | None = None
    final_metrics: dict[str, Any] | None = None
    acceptance: dict[str, Any] | None = None
    validation_report: dict[str, Any] | None = None
    final_abort_reason: str | None = None
    final_run_status = "running"
    try:
        preflight_result = _run_llm_preflight(
            proposal_engine=proposal_engine,
            outputs_dir=outputs_dir,
            available_models=available_models,
            brief=brief,
            current_best=current_best_result,
            memory_payload=memory_payload,
            knowledge_context=knowledge_context,
            failure_patterns=failure_patterns,
            archive_records=archive_records,
            llm_enabled=bool(llm_config.get("enabled", False)),
            allow_deterministic_fallback=allow_deterministic_fallback,
        )
    except ProposalPreflightFailure as exc:
        preflight_result = dict(getattr(exc, "interaction", {}) or {})
        final_abort_reason = str(exc)
        final_run_status = "aborted"
        proposal_status_counts, semantic_rejection_counts, llm_failure_counts, archive_update_summary = _load_run_manifest_counts(
            memory_path=memory_path,
            archive_path=archive_path,
            run_id=run_id,
        )
        manifest = _build_run_manifest_payload(
            run_id=run_id,
            run_started_at=run_started_at,
            run_finished_at=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            config=config,
            brief=brief,
            llm_config=llm_config,
            allow_deterministic_fallback=allow_deterministic_fallback,
            preflight_result=preflight_result,
            proposal_metadata=None,
            smoke_summary=None,
            proposal_status_counts=proposal_status_counts,
            semantic_rejection_counts=semantic_rejection_counts,
            llm_failure_counts=llm_failure_counts,
            archive_update_summary=archive_update_summary,
            number_of_candidates_evaluated=0,
            holdout_touched_during_search=False,
            final_abort_reason=final_abort_reason,
            final_run_status=final_run_status,
            final_metrics=None,
            acceptance=None,
            validation_report=None,
        )
        _write_run_manifest(outputs_dir, manifest)
        raise

    for cycle_number in range(1, max_cycles + 1):
        elapsed_runtime = time.perf_counter() - research_start
        if max_runtime_seconds > 0 and elapsed_runtime >= max_runtime_seconds:
            log_status(f"INFO runtime_budget_reached | cycle={cycle_number} | elapsed_seconds={elapsed_runtime:.2f}")
            break

        lab_state = read_research_surface_state(research_lab_path)
        memory_payload = load_or_initialize_experiment_memory(memory_path)
        failure_patterns = FailureAnalyzer(memory_path).extract_failure_patterns()
        archive_records = hypothesis_archive.load().get("records", [])
        knowledge_context = _build_knowledge_context_for_loop(current_best=current_best_result, x_train=x_train)
        scout_limit = max(
            1,
            int(brief.get("scout_candidates_per_cycle", research_config.get("scout_candidates_per_cycle", 8))),
        )
        confirm_top_k = max(1, int(brief.get("confirm_top_k", research_config.get("confirm_top_k", 2))))
        scout_candidates, proposal_metadata = _select_scout_candidates(
            brief=brief,
            lab_state=lab_state,
            available_models=available_models,
            memory_payload=memory_payload,
            current_best=current_best_result,
            scout_limit=scout_limit,
            family_limit=family_limit,
            current_run_signatures=current_run_signatures,
            proposal_engine=proposal_engine,
            allow_deterministic_fallback=allow_deterministic_fallback,
            trial_history=[
                trial
                for run in memory_payload.get("runs", [])
                for trial in run.get("trials", [])
            ],
            search_progress={
                "phase": "scout",
                "cycle": cycle_number,
                "trial_number": trial_number,
                "current_best_model": current_best_result.get("model_name"),
            },
            failure_patterns=failure_patterns,
            knowledge_context=knowledge_context,
            archive_records=archive_records,
            preflight_result=preflight_result,
            exploit_delta_ratio=float(research_config.get("exploit_delta_ratio", 0.15)),
        )
        log_status(
            "INFO scout_candidate_source | "
            f"cycle={cycle_number} | mode={proposal_metadata['proposal_mode']} | "
            f"backend={proposal_metadata['proposal_backend']} | "
            f"model={proposal_metadata.get('proposal_model')} | "
            f"prompt_variant={proposal_metadata.get('prompt_variant')} | "
            f"count={proposal_metadata['proposal_count']}"
        )
        if not scout_candidates:
            log_status(f"INFO no_scout_candidates_remaining | cycle={cycle_number}")
            break

        scout_results: list[dict[str, Any]] = []
        for experiment in scout_candidates:
            current_run_signatures.add(
                (
                    str(experiment["model_name"]),
                    json.dumps(dict(experiment["params"]), sort_keys=True, separators=(",", ":")),
                )
            )
            trial_number += 1
            started_at = time.perf_counter()
            _register_llm_hypothesis_if_needed(
                archive=hypothesis_archive,
                experiment=experiment,
                run_id=run_id,
                cycle_number=cycle_number,
            )
            try:
                _, result = _evaluate_stage_candidate(
                    experiment=experiment,
                    config=scout_config,
                    x_train=x_train,
                    y_train=y_train,
                    x_val=x_val,
                    y_val=y_val,
                    validator=validator,
                )
                runtime_seconds = time.perf_counter() - started_at
                budget_status = trial_budget_status(
                    elapsed_seconds=runtime_seconds,
                    max_trial_seconds=max_trial_seconds,
                )
                scout_improvement_pct = calculate_improvement_percentage(
                    _safe_float(current_best_result.get("composite_score")),
                    _safe_float(result.get("composite_score")),
                )
                actual_delta_rmse = _compute_actual_delta_rmse(current_best_result, result)
                result["actual_delta_rmse"] = actual_delta_rmse
                if experiment.get("expected_delta_rmse") is not None and actual_delta_rmse is not None:
                    result["calibration_error"] = round(
                        abs(float(experiment["expected_delta_rmse"]) - float(actual_delta_rmse)),
                        4,
                    )
                else:
                    result["calibration_error"] = None
                if budget_status == "budget_exceeded":
                    selection_status = "budget_exceeded"
                    error_message = (
                        f"Scout runtime {runtime_seconds:.2f}s exceeded max_trial_seconds={max_trial_seconds:.2f}s"
                    )
                elif result["validation_verdict"] == "FAIL":
                    selection_status = "rejected_validation_fail"
                    error_message = None
                elif scout_improvement_pct > 0.0:
                    selection_status = "scout_promising"
                    error_message = None
                    scout_result = copy.deepcopy(experiment)
                    scout_result["scout_improvement_pct"] = scout_improvement_pct
                    scout_results.append(scout_result)
                else:
                    selection_status = "scout_no_improvement"
                    error_message = None

                if selection_status != "scout_promising":
                    _resolve_llm_hypothesis_if_needed(
                        archive=hypothesis_archive,
                        experiment=experiment,
                        trial_number=trial_number,
                        selection_status=selection_status,
                        actual_delta_rmse=actual_delta_rmse,
                        actual_result=result,
                    )

                record = _build_record(
                    trial_number=trial_number,
                    experiment=experiment,
                    selection_status=selection_status,
                    result=result,
                    error_message=error_message,
                    trial_runtime_seconds=runtime_seconds,
                    budget_status=budget_status,
                    scout_improvement_pct=scout_improvement_pct,
                    confirm_improvement_pct=None,
                )
                _append_synchronized_results(results_sync_writer, record)
                record_experiment_memory(memory_path, run_id, record)
                timestamp = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")
                _append_research_log(
                    log_path,
                    (
                        f"[{timestamp}] Trial {trial_number:03d} | Stage: scout | Model: {experiment['display_name']} | "
                        f"Composite: {result['composite_score']:.4f} | "
                        f"ScoutImprovementPct: {scout_improvement_pct:.4f} | Status: {selection_status}"
                    ),
                )
            except Exception as exc:
                runtime_seconds = time.perf_counter() - started_at
                budget_status = trial_budget_status(
                    elapsed_seconds=runtime_seconds,
                    max_trial_seconds=max_trial_seconds,
                )
                _resolve_llm_hypothesis_if_needed(
                    archive=hypothesis_archive,
                    experiment=experiment,
                    trial_number=trial_number,
                    selection_status="error",
                    actual_delta_rmse=None,
                    actual_result={"error": str(exc)},
                )
                record = _build_record(
                    trial_number=trial_number,
                    experiment=experiment,
                    selection_status="error",
                    result=None,
                    error_message=str(exc),
                    trial_runtime_seconds=runtime_seconds,
                    budget_status=budget_status,
                    scout_improvement_pct=None,
                    confirm_improvement_pct=None,
                )
                _append_synchronized_results(results_sync_writer, record)
                record_experiment_memory(memory_path, run_id, record)

        confirm_candidates = research_lab.confirm_experiments(
            scout_results=scout_results,
            current_best=current_best_result,
            confirm_top_k=confirm_top_k,
        )
        confirm_candidates = _propagate_confirm_metadata(
            confirm_candidates=confirm_candidates,
            scout_results=scout_results,
        )
        confirmed_reference = (
            _evaluate_reference_if_possible(
                current_best_result=current_best_result,
                available_models=available_models,
                confirm_config=confirm_config,
                x_train=x_train,
                y_train=y_train,
                x_val=x_val,
                y_val=y_val,
                validator=validator,
            )
            if confirm_candidates
            else current_best_result
        )

        for experiment in confirm_candidates:
            current_run_signatures.add(
                (
                    str(experiment["model_name"]),
                    json.dumps(dict(experiment["params"]), sort_keys=True, separators=(",", ":")),
                )
            )
            trial_number += 1
            started_at = time.perf_counter()
            try:
                _register_llm_hypothesis_if_needed(
                    archive=hypothesis_archive,
                    experiment=experiment,
                    run_id=run_id,
                    cycle_number=cycle_number,
                )
                model, result = _evaluate_stage_candidate(
                    experiment=experiment,
                    config=confirm_config,
                    x_train=x_train,
                    y_train=y_train,
                    x_val=x_val,
                    y_val=y_val,
                    validator=validator,
                )
                runtime_seconds = time.perf_counter() - started_at
                budget_status = trial_budget_status(
                    elapsed_seconds=runtime_seconds,
                    max_trial_seconds=max_trial_seconds,
                )
                confirm_improvement_pct = calculate_improvement_percentage(
                    _safe_float(confirmed_reference.get("composite_score")),
                    _safe_float(result.get("composite_score")),
                )
                actual_delta_rmse = _compute_actual_delta_rmse(confirmed_reference, result)
                result["actual_delta_rmse"] = actual_delta_rmse
                if experiment.get("expected_delta_rmse") is not None and actual_delta_rmse is not None:
                    result["calibration_error"] = round(
                        abs(float(experiment["expected_delta_rmse"]) - float(actual_delta_rmse)),
                        4,
                    )
                else:
                    result["calibration_error"] = None
                if budget_status == "budget_exceeded":
                    selection_status = "budget_exceeded"
                    error_message = (
                        f"Confirm runtime {runtime_seconds:.2f}s exceeded max_trial_seconds={max_trial_seconds:.2f}s"
                    )
                elif result["validation_verdict"] == "FAIL":
                    selection_status = "rejected_validation_fail"
                    error_message = None
                elif confirm_improvement_pct > 0.0:
                    selection_status = "kept"
                    error_message = None
                    result["trial_number"] = trial_number
                    result["best_trial"] = trial_number
                    result["source"] = "research_loop"
                    save_pickle_artifact(
                        outputs_dir / SEARCH_STATE_BEST_MODEL_FILENAME,
                        model,
                        config=confirm_config,
                        model_id=experiment["model_name"],
                    )
                    current_best_result = copy.deepcopy(result)
                    _sync_final_artifacts_from_source_of_truth(
                        outputs_dir=outputs_dir,
                        baseline_metrics=baseline_metrics,
                        best_result=current_best_result,
                        best_model_source_path=outputs_dir / SEARCH_STATE_BEST_MODEL_FILENAME,
                    )
                    apply_keep_to_research_surface(
                        research_lab_path=research_lab_path,
                        accepted_entry={
                            "experiment_id": experiment["experiment_id"],
                            "proposal_family": experiment["proposal_family"],
                            "model_name": experiment["model_name"],
                            "composite_score": result["composite_score"],
                            "confirm_improvement_pct": confirm_improvement_pct,
                        },
                    )
                    from uncertainty import recalibrate_uncertainty_artifacts

                    recalibrate_uncertainty_artifacts(
                        model_path=outputs_dir / BEST_MODEL_FILENAME,
                        method=str(config["engineering"]["uncertainty_method"]),
                        outputs_dir=outputs_dir,
                        audit_partition="validation_audit",
                    )
                    confirmed_reference = current_best_result
                else:
                    selection_status = "reverted"
                    error_message = None

                _resolve_llm_hypothesis_if_needed(
                    archive=hypothesis_archive,
                    experiment=experiment,
                    trial_number=trial_number,
                    selection_status=selection_status,
                    actual_delta_rmse=actual_delta_rmse,
                    actual_result=result,
                )

                record = _build_record(
                    trial_number=trial_number,
                    experiment=experiment,
                    selection_status=selection_status,
                    result=result,
                    error_message=error_message,
                    trial_runtime_seconds=runtime_seconds,
                    budget_status=budget_status,
                    scout_improvement_pct=None,
                    confirm_improvement_pct=confirm_improvement_pct,
                )
                _append_synchronized_results(results_sync_writer, record)
                record_experiment_memory(memory_path, run_id, record)
                timestamp = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")
                _append_research_log(
                    log_path,
                    (
                        f"[{timestamp}] Trial {trial_number:03d} | Stage: confirm | Model: {experiment['display_name']} | "
                        f"Composite: {result['composite_score']:.4f} | "
                        f"ConfirmImprovementPct: {confirm_improvement_pct:.4f} | Status: {selection_status}"
                    ),
                )
            except Exception as exc:
                runtime_seconds = time.perf_counter() - started_at
                budget_status = trial_budget_status(
                    elapsed_seconds=runtime_seconds,
                    max_trial_seconds=max_trial_seconds,
                )
                _resolve_llm_hypothesis_if_needed(
                    archive=hypothesis_archive,
                    experiment=experiment,
                    trial_number=trial_number,
                    selection_status="error",
                    actual_delta_rmse=None,
                    actual_result={"error": str(exc)},
                )
                record = _build_record(
                    trial_number=trial_number,
                    experiment=experiment,
                    selection_status="error",
                    result=None,
                    error_message=str(exc),
                    trial_runtime_seconds=runtime_seconds,
                    budget_status=budget_status,
                    scout_improvement_pct=None,
                    confirm_improvement_pct=None,
                )
                _append_synchronized_results(results_sync_writer, record)
                record_experiment_memory(memory_path, run_id, record)

        memory_payload = load_or_initialize_experiment_memory(memory_path)
        _write_diversity_summary(
            outputs_dir=outputs_dir,
            run_id=run_id,
            current_run_signatures=current_run_signatures,
            memory_payload=memory_payload,
        )

    if not (outputs_dir / BEST_MODEL_FILENAME).exists():
        baseline_model_path = outputs_dir / "baseline_model.pkl"
        if not baseline_model_path.exists():
            raise FileNotFoundError("Baseline model artifact is missing after research loop.")
        current_best_result.setdefault("source", "baseline")
        _sync_final_artifacts_from_source_of_truth(
            outputs_dir=outputs_dir,
            baseline_metrics=baseline_metrics,
            best_result=current_best_result,
            best_model_source_path=baseline_model_path,
        )

    final_metrics_path = outputs_dir / FINAL_METRICS_FILENAME
    if final_metrics_path.exists():
        final_metrics = load_json_file(final_metrics_path)
    else:
        final_metrics = _build_final_metrics_payload(baseline_metrics, current_best_result)
        write_json_file(final_metrics_path, final_metrics)
    acceptance = build_acceptance_decision(final_metrics=final_metrics, brief=brief)
    write_json_file(outputs_dir / FINAL_ACCEPTANCE_FILENAME, acceptance)
    if (
        proposal_engine is not None
        and hasattr(proposal_engine, "summarize_run")
        and bool(llm_config.get("tasks", {}).get("summary_enabled", True))
    ):
        llm_summary = proposal_engine.summarize_run(
            research_brief=brief,
            final_metrics=final_metrics,
            acceptance=acceptance,
        )
        write_json_file(outputs_dir / LLM_RUN_SUMMARY_FILENAME, llm_summary)
    validation_report = validate_final_artifact_consistency(outputs_dir)
    write_json_file(outputs_dir / FINAL_ARTIFACT_VALIDATION_FILENAME, validation_report)
    if not validation_report["consistent"]:
        raise RuntimeError(f"Final artifact mismatch: {validation_report['mismatches']}")
    if with_report:
        from uncertainty import recalibrate_uncertainty_artifacts
        import report

        recalibrate_uncertainty_artifacts(
            model_path=outputs_dir / BEST_MODEL_FILENAME,
            method=str(config["engineering"]["uncertainty_method"]),
            outputs_dir=outputs_dir,
            audit_partition="validation_audit",
        )
        if report.main() != 0:
            raise RuntimeError("Report generation failed after search finalization.")
        validation_report = validate_final_artifact_consistency(outputs_dir)
        write_json_file(outputs_dir / FINAL_ARTIFACT_VALIDATION_FILENAME, validation_report)
        if not validation_report["consistent"]:
            raise RuntimeError(f"Post-report artifact mismatch: {validation_report['mismatches']}")
    final_run_status = "success"
    proposal_status_counts, semantic_rejection_counts, llm_failure_counts, archive_update_summary = _load_run_manifest_counts(
        memory_path=memory_path,
        archive_path=archive_path,
        run_id=run_id,
    )
    manifest = _build_run_manifest_payload(
        run_id=run_id,
        run_started_at=run_started_at,
        run_finished_at=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        config=config,
        brief=brief,
        llm_config=llm_config,
        allow_deterministic_fallback=allow_deterministic_fallback,
        preflight_result=preflight_result,
        proposal_metadata=proposal_metadata,
        smoke_summary=None,
        proposal_status_counts=proposal_status_counts,
        semantic_rejection_counts=semantic_rejection_counts,
        llm_failure_counts=llm_failure_counts,
        archive_update_summary=archive_update_summary,
        number_of_candidates_evaluated=trial_number,
        holdout_touched_during_search=bool(with_report),
        final_abort_reason=final_abort_reason,
        final_run_status=final_run_status,
        final_metrics=final_metrics,
        acceptance=acceptance,
        validation_report=validation_report,
    )
    _write_run_manifest(outputs_dir, manifest)
    return final_metrics["best_search_metrics"]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the governed AutoCivil-Lab research loop.")
    parser.add_argument("--cycles", type=int, default=None, help="Override research.max_cycles for this run.")
    parser.add_argument(
        "--with-report",
        action="store_true",
        help="Run uncertainty recalibration and report generation at the end of the loop.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate final artifact consistency and exit.",
    )
    parser.add_argument(
        "--analyze-interactions",
        action="store_true",
        help="Summarize llm_interactions.jsonl and exit.",
    )
    parser.add_argument(
        "--llm-smoke-test",
        action="store_true",
        help="Run the strict proposal-engine preflight check and exit.",
    )
    parser.add_argument(
        "--llm-smoke-repeat",
        type=int,
        default=None,
        help="Run the strict proposal-engine smoke test repeatedly and export a reliability summary.",
    )
    parser.add_argument(
        "--llm-backend-mode",
        choices=["ollama", "openai", "hybrid"],
        default=None,
        help="Override llm.backend_mode for smoke-test commands.",
    )
    parser.add_argument(
        "--llm-model",
        default=None,
        help="Override the proposal model hint for smoke-test commands.",
    )
    parser.add_argument(
        "--llm-smoke-summary",
        default=None,
        help="Optional path for the repeated smoke-test JSON summary artifact.",
    )
    args = parser.parse_args()

    try:
        outputs_dir = get_outputs_dir(load_config())
        if args.validate_only:
            validation_report = validate_final_artifact_consistency(outputs_dir)
            write_json_file(outputs_dir / FINAL_ARTIFACT_VALIDATION_FILENAME, validation_report)
            log_status(f"Final artifact validation complete. consistent={validation_report['consistent']}")
            return 0 if validation_report["consistent"] else 1

        if args.analyze_interactions:
            interaction_summary = analyze_interaction_log(outputs_dir)
            log_status(
                "Interaction analysis | "
                f"exists={interaction_summary['exists']} | "
                f"count={interaction_summary['interaction_count']} | "
                f"channels={json.dumps(interaction_summary['channel_counts'], sort_keys=True)} | "
                f"rejections={json.dumps(interaction_summary['rejection_counts'], sort_keys=True)} | "
                f"semantic_rejections={json.dumps(interaction_summary.get('semantic_rejection_counts', {}), sort_keys=True)} | "
                f"repairs={interaction_summary.get('repair_count', 0)} | "
                f"parse_success={interaction_summary.get('parse_success_count', 0)}"
            )
            return 0

        if args.llm_smoke_test or args.llm_smoke_repeat is not None:
            config = _apply_llm_smoke_overrides(
                load_config(),
                backend_mode=args.llm_backend_mode,
            )
            outputs_dir = get_outputs_dir(config)
            baseline_metrics = _read_baseline_metrics(outputs_dir, config)
            current_best_result = _load_current_best_result(outputs_dir, baseline_metrics)
            research_config = dict(config.get("research", {}))
            project_root = Path(__file__).resolve().parent
            brief_path = project_root / str(research_config.get("brief_path", "research_brief.md"))
            brief = load_human_research_brief(brief_path)
            available_models = get_available_model_configs(config)
            data = load_dataset(config)
            x_train, _, _, _, _, _ = split_dataset(data, config)
            memory_path = outputs_dir / str(research_config.get("memory_filename", EXPERIMENT_MEMORY_FILENAME))
            memory_payload = load_or_initialize_experiment_memory(memory_path)
            proposal_engine = _build_proposal_engine(config, outputs_dir)
            llm_config = dict(config.get("llm", {}))
            archive_path = outputs_dir / str(
                research_config.get("hypothesis_archive_filename", HYPOTHESIS_ARCHIVE_FILENAME)
            )
            hypothesis_archive = HypothesisArchive(archive_path)
            knowledge_context = _build_knowledge_context_for_loop(
                current_best=current_best_result,
                x_train=x_train,
            )
            failure_patterns = FailureAnalyzer(memory_path).extract_failure_patterns()
            archive_records = hypothesis_archive.load().get("records", [])

            if args.llm_smoke_repeat is not None:
                summary_path = (
                    Path(args.llm_smoke_summary)
                    if args.llm_smoke_summary
                    else outputs_dir / LLM_SMOKE_RELIABILITY_FILENAME
                )
                report = run_repeated_llm_smoke_test(
                    proposal_engine=proposal_engine,
                    available_models=available_models,
                    brief=brief,
                    current_best=current_best_result,
                    memory_payload=memory_payload,
                    knowledge_context=knowledge_context,
                    failure_patterns=failure_patterns,
                    archive_records=archive_records,
                    trials=args.llm_smoke_repeat,
                    summary_path=summary_path,
                    requested_backend=args.llm_backend_mode,
                    model_hint=args.llm_model,
                    admission_policy_config=llm_config.get("admission_policy"),
                )
                log_status(
                    "LLM smoke reliability | "
                    f"trials={report['requested_trials']} | "
                    f"success_rate={report['summary']['success_rate']:.3f} | "
                    f"semantic_failure_rate={report['summary'].get('semantic_failure_rate', 0.0):.3f} | "
                    f"verdict={report['summary']['admission_policy']['verdict']} | "
                    f"interpretation={report['summary']['compatibility']['interpretation']} | "
                    f"summary={summary_path}"
                )
                return 0 if report["summary"]["success_count"] == report["summary"]["total_trials"] else 1

            smoke_result = _run_llm_preflight(
                proposal_engine=proposal_engine,
                outputs_dir=outputs_dir,
                available_models=available_models,
                brief=brief,
                current_best=current_best_result,
                memory_payload=memory_payload,
                knowledge_context=knowledge_context,
                failure_patterns=failure_patterns,
                archive_records=archive_records,
                llm_enabled=bool(llm_config.get("enabled", False)),
                allow_deterministic_fallback=bool(llm_config.get("allow_deterministic_fallback", False)),
                model_hint=args.llm_model,
            )
            log_status(
                "LLM smoke test | "
                f"ok={None if smoke_result is None else smoke_result.get('ok')} | "
                f"status={None if smoke_result is None else smoke_result.get('status')} | "
                f"backend={None if smoke_result is None else smoke_result.get('backend')} | "
                f"model={None if smoke_result is None else smoke_result.get('model')}"
            )
            return 0 if smoke_result is not None and bool(smoke_result.get("ok", False)) else 1

        best_result = run_engineering_research_loop(
            cycles_override=args.cycles,
            with_report=args.with_report,
        )
        log_status(
            f"Research loop complete. Best model: {best_result['model_name']} | "
            f"Composite={best_result['composite_score']:.4f} | "
            f"Validation={best_result['validation_verdict']}"
        )
        return 0
    except Exception as exc:
        log_status(f"Research loop failed: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
