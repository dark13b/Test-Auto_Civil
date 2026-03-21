"""Autoresearch-style engineering ML loop for AutoCivil-Lab."""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd

from artifact_sync import AtomicArtifactWriter
import research_lab
from llm_backend import get_llm_config, resolve_backend
from proposal_engine import ProposalEngine, ProposalExtractionError
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
from train import (
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
)
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
    final_metrics_path = outputs_dir / FINAL_METRICS_FILENAME
    if final_metrics_path.exists():
        final_metrics = load_json_file(final_metrics_path)
        best_search_metrics = final_metrics.get("best_search_metrics")
        if isinstance(best_search_metrics, dict) and best_search_metrics:
            payload = copy.deepcopy(best_search_metrics)
            payload.setdefault("source", "final_metrics")
            return payload

    best_result_path = outputs_dir / BEST_RESULT_FILENAME
    if best_result_path.exists():
        payload = load_json_file(best_result_path)
        payload.setdefault("source", "best_search_result")
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
        "validation_metrics": copy.deepcopy(best_result.get("val_metrics", best_result.get("test_metrics", {}))),
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
            "hypothesis": experiment.get("hypothesis"),
            "hyperparameters": json.dumps(experiment.get("params", {}), sort_keys=True),
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
    val_metrics = result.get("val_metrics", result.get("test_metrics", {}))
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
                "hypothesis": str(candidate.get("hypothesis", "LLM-generated suggestion")),
                "params": dict(candidate.get("params", {})),
            }
        )
    return normalized


def _build_proposal_engine(config: dict[str, Any], outputs_dir: Path) -> ProposalEngine | None:
    llm_config = get_llm_config(config)
    if not llm_config.get("enabled", False):
        return None
    interaction_log_path = outputs_dir / str(llm_config.get("interaction_log_filename", "llm_interactions.jsonl"))
    return ProposalEngine(
        backend=resolve_backend(config),
        interaction_log_path=interaction_log_path,
        log_interactions=bool(llm_config.get("log_interactions", True)),
        include_no_think_directive=bool(llm_config.get("include_no_think_directive", False)),
        llm_config=llm_config,
    )


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
    proposal_engine: ProposalEngine | None,
    allow_deterministic_fallback: bool = True,
    trial_history: list[dict[str, Any]] | None = None,
    search_progress: dict[str, Any] | None = None,
    exploit_delta_ratio: float = 0.15,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    llm_candidates: list[dict[str, Any]] = []
    if proposal_engine is not None and proposal_engine.is_available():
        try:
            llm_candidates = proposal_engine.generate_experiment_proposals(
                available_models=available_models,
                research_brief=brief,
                current_best=current_best,
                experiment_memory=memory_payload,
                diversity_state=_build_diversity_state(memory_payload),
                proposal_count=scout_limit,
                trial_history=trial_history or [],
                search_progress=search_progress or {},
            )
        except ProposalExtractionError:
            llm_candidates = []
        llm_candidates = _normalize_llm_candidates(llm_candidates, available_models)
        llm_candidates = filter_diverse_candidates(
            candidates=llm_candidates,
            memory_payload=memory_payload,
            current_run_signatures=current_run_signatures,
            family_limit=family_limit,
            scout_limit=scout_limit,
        )
        if llm_candidates:
            interaction_summary = getattr(proposal_engine, "last_interaction_summary", {})
            return llm_candidates, {
                "proposal_mode": "llm",
                "proposal_backend": proposal_engine.backend_name,
                "proposal_count": len(llm_candidates),
                "proposal_model": interaction_summary.get("model"),
                "prompt_variant": interaction_summary.get("prompt_variant"),
            }
        if not allow_deterministic_fallback:
            return [], {
                "proposal_mode": "llm_unavailable",
                "proposal_backend": proposal_engine.backend_name,
                "proposal_count": 0,
                "proposal_model": None,
                "prompt_variant": None,
            }

    deterministic_candidates = research_lab.scout_experiments(
        brief=brief,
        lab_state=lab_state,
        available_models=available_models,
        experiment_memory=memory_payload,
        current_best=current_best,
        scout_limit=scout_limit * 2,
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
        "proposal_mode": "deterministic_fallback",
        "proposal_backend": "fallback",
        "proposal_count": len(deterministic_candidates),
        "proposal_model": None,
        "prompt_variant": None,
    }


def analyze_interaction_log(outputs_dir: Path, *, filename: str = "llm_interactions.jsonl") -> dict[str, Any]:
    interaction_path = outputs_dir / filename
    if not interaction_path.exists():
        return {
            "exists": False,
            "interaction_count": 0,
            "channel_counts": {},
            "rejection_counts": {},
        }

    channel_counts: dict[str, int] = {}
    rejection_counts: dict[str, int] = {}
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
    return {
        "exists": True,
        "interaction_count": len(records),
        "channel_counts": channel_counts,
        "rejection_counts": rejection_counts,
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
    current_run_signatures: set[tuple[str, str]] = set()
    proposal_engine = _build_proposal_engine(config, outputs_dir)
    llm_config = get_llm_config(config)
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

    for cycle_number in range(1, max_cycles + 1):
        elapsed_runtime = time.perf_counter() - research_start
        if max_runtime_seconds > 0 and elapsed_runtime >= max_runtime_seconds:
            log_status(f"INFO runtime_budget_reached | cycle={cycle_number} | elapsed_seconds={elapsed_runtime:.2f}")
            break

        lab_state = read_research_surface_state(research_lab_path)
        memory_payload = load_or_initialize_experiment_memory(memory_path)
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
            allow_deterministic_fallback=bool(llm_config.get("allow_deterministic_fallback", True)),
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
                    if rebuild_reports_on_keep:
                        import report

                        if report.main() != 0:
                            raise RuntimeError("Report generation failed after a kept confirm experiment.")
                        validation_report = validate_final_artifact_consistency(outputs_dir)
                        write_json_file(outputs_dir / FINAL_ARTIFACT_VALIDATION_FILENAME, validation_report)
                        if not validation_report["consistent"]:
                            raise RuntimeError(f"Post-report artifact mismatch: {validation_report['mismatches']}")
                    confirmed_reference = current_best_result
                else:
                    selection_status = "reverted"
                    error_message = None

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

    if with_report and not rebuild_reports_on_keep:
        from uncertainty import recalibrate_uncertainty_artifacts
        import report

        recalibrate_uncertainty_artifacts(
            model_path=outputs_dir / BEST_MODEL_FILENAME,
            method=str(config["engineering"]["uncertainty_method"]),
            outputs_dir=outputs_dir,
            audit_partition="validation_audit",
        )
        if report.main() != 0:
            raise RuntimeError("Report generation failed at end of research loop.")

    final_metrics = load_json_file(outputs_dir / FINAL_METRICS_FILENAME)
    acceptance = build_acceptance_decision(final_metrics=final_metrics, brief=brief)
    write_json_file(outputs_dir / FINAL_ACCEPTANCE_FILENAME, acceptance)
    if (
        proposal_engine is not None
        and proposal_engine.is_available()
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
                f"repairs={interaction_summary.get('repair_count', 0)} | "
                f"parse_success={interaction_summary.get('parse_success_count', 0)}"
            )
            return 0

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
