"""Artifact write helpers for the AutoCivil-Lab research loop."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pandas as pd

from artifact_sync import AtomicArtifactWriter
from research_protocol import (
    RESEARCH_RESULTS_COLUMNS,
    load_json_file,
    validate_final_artifact_consistency,
    write_json_file,
)
from train_impl import save_json_artifact, write_run_scoped_json_artifact
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
RUN_MANIFEST_FILENAME = "run_manifest.json"


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
    if path.exists() and path.stat().st_size > 0:
        return
    with path.open("w", encoding="utf-8") as handle:
        handle.write(
            "research loop initialized | "
            f"baseline_composite_score={baseline_metrics.get('composite_score')}\n"
        )


def _append_research_log(path: Path, line: str) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def _increment_count(mapping: dict[str, int], key: Any, *, amount: int = 1) -> None:
    normalized_key = str(key)
    mapping[normalized_key] = mapping.get(normalized_key, 0) + int(amount)


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
        "timestamp": {"started_at": run_started_at, "finished_at": run_finished_at},
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
    if archive_path.exists():
        archive_payload = load_json_file(archive_path)
        for record in archive_payload.get("records", []):
            if str(record.get("run_id")) != str(run_id):
                continue
            if record.get("timestamp") is not None:
                archive_update_summary["added"] += int(record.get("outcome") == "pending")
                archive_update_summary["resolved"] += int(record.get("outcome") not in {None, "pending"})
    return proposal_status_counts, semantic_rejection_counts, llm_failure_counts, archive_update_summary


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
        display_name = str(candidate.get("display_name", available_models.get(model_name, {}).get("display_name", model_name)))
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
            "expected_metric_effect": {"metric": "rmse", "direction": "down", "magnitude_estimate": "unknown"},
            "confidence": 0.0,
            "novelty_claim": "Fallback-generated proposal; no LLM novelty claim available.",
            "risk_notes": "This proposal was not generated by an LLM.",
            "candidate_config": {"model_name": str(updated.get("model_name", "")), "params": dict(updated.get("params", {}))},
        }
        normalized.append(updated)
    return normalized
