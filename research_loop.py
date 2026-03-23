"""Autoresearch-style engineering ML loop for AutoCivil-Lab."""

from __future__ import annotations

import argparse
import copy
import json
import logging
import sys
import research_lab
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from dataclasses import asdict

import pandas as pd

from artifact_sync import AtomicArtifactWriter
from deterministic_proposal_provider import DeterministicProposalProvider
from failure_analyzer import FailureAnalyzer
from hypothesis_archive import HypothesisArchive
from llm_admission_policy import evaluate_llm_admission_policy
from hybrid_proposal_provider import HybridProposalProvider
from llm_proposal_provider import LLMProposalProvider
from loop_candidate_selection import select_scout_candidates
from proposal_engine import (
    ProposalBackendFailure,
    ProposalContext,
    ProposalExtractionError,
    ProposalParseFailure,
    ProposalPreflightFailure,
    ProposalProvider,
    ProposalSemanticFailure,
    ProposalSchemaFailure,
    get_proposals,
)
from research_loop_helpers import (
    _archive_outcome_for_selection,
    _build_knowledge_context_for_loop,
    _build_record,
    _build_proposal_engine,
    _build_stage_config,
    _compute_actual_delta_rmse,
    _evaluate_reference_if_possible,
    _evaluate_stage_candidate,
    _load_current_best_result,
    _read_baseline_metrics,
    _propagate_confirm_metadata,
    _register_llm_hypothesis_if_needed,
    _resolve_llm_hypothesis_if_needed,
    calculate_improvement_percentage,
    _safe_float,
)
from research_loop_smoke import (
    _apply_llm_smoke_overrides,
    _run_llm_preflight,
    analyze_interaction_log,
    run_repeated_llm_smoke_test,
    summarize_smoke_test_trials,
)
from research_protocol import (
    RESEARCH_RESULTS_COLUMNS,
    apply_keep_to_research_surface,
    build_acceptance_decision,
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
    get_outputs_dir,
    load_config,
    load_dataset,
    log_status,
    save_pickle_artifact,
    set_global_seed,
    split_dataset,
)
from loop_artifacts import (
    BEST_MODEL_FILENAME,
    BEST_RESULT_FILENAME,
    FINAL_ACCEPTANCE_FILENAME,
    FINAL_ARTIFACT_VALIDATION_FILENAME,
    FINAL_METRICS_FILENAME,
    PROPOSAL_DIVERSITY_FILENAME,
    RESEARCH_LOG_FILENAME,
    RESEARCH_RESULTS_FILENAME,
    RUN_MANIFEST_FILENAME,
    SEARCH_STATE_BEST_MODEL_FILENAME,
    SEARCH_STATE_BEST_RESULT_FILENAME,
    _append_research_log,
    _append_results_row,
    _append_synchronized_results,
    _build_final_metrics_payload,
    _build_results_sync_writer,
    _build_run_manifest_payload,
    _increment_count,
    _initialize_research_log,
    _initialize_results_csv,
    _load_run_manifest_counts,
    _sync_final_artifacts_from_source_of_truth,
    _write_diversity_summary,
    _write_run_manifest,
)

LOGGER = logging.getLogger("research_loop")
from validator import summarize_validation_report

EXPERIMENT_MEMORY_FILENAME = "experiment_memory.json"
HYPOTHESIS_ARCHIVE_FILENAME = "hypothesis_archive.json"
LLM_RUN_SUMMARY_FILENAME = "llm_run_summary.json"
LLM_SMOKE_TEST_FILENAME = "llm_smoke_test.json"
LLM_SMOKE_RELIABILITY_FILENAME = "llm_smoke_reliability.json"

_select_scout_candidates = select_scout_candidates

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
        scout_candidates, proposal_metadata = select_scout_candidates(
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
