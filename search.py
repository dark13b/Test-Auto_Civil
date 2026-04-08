"""Legacy compatibility wrappers and artifact helpers for AutoCivil-Lab research."""

from __future__ import annotations

import argparse
import copy
import json
import re
import shutil
import sys
import time
import warnings
from collections import Counter
from pathlib import Path
from typing import Any

import optuna
import pandas as pd

from artifact_sync import AtomicArtifactWriter, repair_on_startup as repair_synced_artifacts_on_startup
from llm_backend import get_llm_config
from research_protocol import (
    build_acceptance_decision,
    build_family_state_summary,
    build_config_signature,
    gate_proposal,
    load_human_research_brief,
    load_or_initialize_experiment_memory,
    record_experiment_memory,
    should_skip_duplicate_proposal,
    trial_budget_status,
    validate_final_artifact_consistency,
)
from train_impl import (
    EngineeringValidator,
    artifact_id,
    artifact_run_id,
    build_stacking_ensemble,
    create_run_id,
    evaluate_candidate,
    get_outputs_dir,
    load_config,
    load_dataset,
    load_pickle_artifact,
    log_status,
    mark_json_artifact_stale,
    save_json_artifact,
    save_pickle_artifact,
    set_global_seed,
    split_dataset,
    to_serializable,
    write_run_scoped_json_artifact,
)
from validator import summarize_validation_report


OPTUNA_RESULTS_COLUMNS = [
    "trial_number",
    "model_name",
    "display_name",
    "hyperparameters",
    "selection_status",
    "validation_verdict",
    "error_message",
    "rmse",
    "mae",
    "r2",
    "composite_score",
    "test_rmse",
    "test_mae",
    "test_r2",
    "test_composite_score",
    "validation_pass_rate",
    "failed_count",
    "hard_failed_count",
    "warning_count",
    "suspicious_count",
    "durability_caution_count",
    "dataset_anomaly_count",
    "trial_runtime_seconds",
    "budget_status",
]

SEARCH_STATE_BEST_RESULT_FILENAME = "search_state_best_result.json"
SEARCH_STATE_BEST_MODEL_FILENAME = "search_state_best_model.pkl"
FINAL_BEST_RESULT_FILENAME = "best_search_result.json"
FINAL_BEST_MODEL_FILENAME = "best_search_model.pkl"
FINAL_METRICS_FILENAME = "final_metrics.json"
FINAL_ACCEPTANCE_FILENAME = "final_acceptance.json"
EXPERIMENT_MEMORY_FILENAME = "experiment_memory.json"
LEGACY_RUNTIME_MESSAGE = (
    "search.py is deprecated as a research runtime. "
    "Use `python research_loop.py` for canonical research execution."
)


class _CanonicalResearchLoopProxy:
    """Lazy proxy so compatibility wrappers can call research_loop without import cycles."""

    @staticmethod
    def run_engineering_research_loop(**kwargs: Any) -> dict[str, Any]:
        import research_loop as _research_loop

        return _research_loop.run_engineering_research_loop(**kwargs)


research_loop = _CanonicalResearchLoopProxy()


def load_json_artifact(path: Path) -> dict[str, Any]:
    """Load a JSON artifact from disk."""
    if not path.exists():
        raise FileNotFoundError(f"Required artifact not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _artifact_is_stale(payload: dict[str, Any] | None) -> bool:
    """Return whether an artifact payload has been marked stale."""
    if not isinstance(payload, dict):
        return False
    metadata = payload.get("artifact_metadata", {})
    if isinstance(metadata, dict) and metadata.get("stale"):
        return True
    return bool(payload.get("stale"))


def _strip_stale_fields(payload: dict[str, Any]) -> dict[str, Any]:
    """Remove stale markers from a recovered artifact payload before rewriting it canonically."""
    cleaned = copy.deepcopy(payload)
    for key in ("stale", "stale_reason", "active_run_id"):
        cleaned.pop(key, None)
    metadata = cleaned.get("artifact_metadata")
    if isinstance(metadata, dict):
        for key in ("stale", "stale_reason", "stale_checked_against_run_id", "canonical_latest"):
            metadata.pop(key, None)
    return cleaned


def sample_search_parameter(trial: optuna.trial.Trial, name: str, spec: dict[str, Any]) -> Any:
    """Sample a parameter from a config-defined search space specification."""
    parameter_type = spec["type"]
    if parameter_type == "int":
        return trial.suggest_int(name, int(spec["low"]), int(spec["high"]), step=int(spec.get("step", 1)))
    if parameter_type == "float":
        return trial.suggest_float(
            name,
            float(spec["low"]),
            float(spec["high"]),
            log=bool(spec.get("log", False)),
        )
    if parameter_type == "categorical":
        return trial.suggest_categorical(name, list(spec["choices"]))
    raise ValueError(f"Unsupported search parameter type: {parameter_type}")


def get_available_model_configs(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Return model configurations that are both enabled and importable."""
    available_models: dict[str, dict[str, Any]] = {}
    for family, family_config in config["search"]["models"].items():
        if not family_config.get("enabled", False):
            continue
        if family == "XGBRegressor":
            try:
                from xgboost import XGBRegressor as _  # noqa: F401
            except ImportError:
                log_status("Skipping XGBRegressor because xgboost is not installed.")
                continue
        if family == "LGBMRegressor":
            try:
                from lightgbm import LGBMRegressor as _  # noqa: F401
            except ImportError:
                log_status("Skipping LGBMRegressor because lightgbm is not installed.")
                continue
        if family == "CatBoostRegressor":
            try:
                from catboost import CatBoostRegressor as _  # noqa: F401
            except ImportError:
                log_status("Skipping CatBoostRegressor because catboost is not installed.")
                continue
        available_models[family] = family_config
    if not available_models:
        raise RuntimeError("No enabled model families are available for the search loop.")
    return available_models


def sample_model_configuration(
    trial: optuna.trial.Trial,
    available_models: dict[str, dict[str, Any]],
) -> tuple[str, str, dict[str, Any]]:
    """Sample a model family and its hyperparameters for a single trial."""
    model_name = trial.suggest_categorical("model_family", list(available_models.keys()))
    model_config = available_models[model_name]
    display_name = str(model_config.get("display_name", model_name))
    sampled_params: dict[str, Any] = {}
    for parameter_name, parameter_spec in model_config["search_space"].items():
        sampled_params[parameter_name] = sample_search_parameter(
            trial,
            f"{model_name}__{parameter_name}",
            parameter_spec,
        )
    return model_name, display_name, sampled_params


def format_hyperparameters(params: dict[str, Any]) -> str:
    """Render trial hyperparameters as a compact log string."""
    parts = []
    for key, value in params.items():
        if isinstance(value, float):
            parts.append(f"{key}={value:.4g}")
        else:
            parts.append(f"{key}={value}")
    return " ".join(parts)


def build_config_signature(model_name: str, params: dict[str, Any]) -> tuple[str, str]:
    """Build a stable signature for one model configuration."""
    return model_name, json.dumps(to_serializable(params), sort_keys=True, separators=(",", ":"))


def llm_suggestion_is_duplicate(
    suggestion: dict[str, Any],
    trial_history: list[dict[str, Any]],
    *,
    lookback: int = 200,
) -> bool:
    """Return whether an LLM suggestion duplicates a recently evaluated configuration."""
    suggestion_signature = build_config_signature(
        str(suggestion["model_name"]),
        dict(suggestion["params"]),
    )
    for trial_record in trial_history[-lookback:]:
        trial_model_name = str(trial_record.get("model_name", ""))
        raw_params = trial_record.get("hyperparameters", {})
        if isinstance(raw_params, str):
            try:
                trial_params = json.loads(raw_params)
            except Exception:
                trial_params = {}
        elif isinstance(raw_params, dict):
            trial_params = raw_params
        else:
            trial_params = {}
        if suggestion_signature == build_config_signature(trial_model_name, trial_params):
            return True
    return False


def build_llm_progress_context(
    *,
    elapsed_seconds: float,
    runtime_target_seconds: float,
    interaction_index: int,
    trial_records: list[dict[str, Any]],
    available_models: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Build run-progress context for timed LLM interactions."""
    remaining_seconds = max(0.0, runtime_target_seconds - elapsed_seconds) if runtime_target_seconds > 0 else 0.0
    progress_ratio = 0.0 if runtime_target_seconds <= 0 else min(elapsed_seconds / runtime_target_seconds, 1.0)
    if progress_ratio < 1.0 / 3.0:
        stage = "early"
    elif progress_ratio < 2.0 / 3.0:
        stage = "mid"
    else:
        stage = "late"

    model_counts = Counter(str(record.get("model_name", "unknown")) for record in trial_records)
    family_counts = {model_name: int(model_counts.get(model_name, 0)) for model_name in available_models}
    minimum_count = min(family_counts.values(), default=0)
    underexplored_families = [
        model_name for model_name, count in family_counts.items() if count == minimum_count
    ]
    return {
        "stage": stage,
        "interaction_index": interaction_index,
        "elapsed_minutes": elapsed_seconds / 60.0,
        "remaining_minutes": remaining_seconds / 60.0 if runtime_target_seconds > 0 else 0.0,
        "progress_percent": progress_ratio * 100.0,
        "family_counts": family_counts,
        "underexplored_families": underexplored_families,
    }


def build_search_family_state(
    *,
    available_models: dict[str, dict[str, Any]],
    best_result: dict[str, Any],
    trial_records: list[dict[str, Any]],
    historical_memory: dict[str, Any],
    llm_config: dict[str, Any],
) -> dict[str, Any]:
    return build_family_state_summary(
        available_models=available_models,
        current_best=best_result,
        trial_history=trial_records,
        memory_payload=historical_memory,
        diversity_settings=llm_config.get("diversity", {}),
    )


def gate_search_candidate(
    *,
    model_name: str,
    params: dict[str, Any],
    available_models: dict[str, dict[str, Any]],
    current_run_signatures: set[tuple[str, str]],
    historical_memory: dict[str, Any],
    trial_records: list[dict[str, Any]],
    family_state: dict[str, Any],
    llm_config: dict[str, Any],
) -> dict[str, Any]:
    return gate_proposal(
        proposal={"model_name": model_name, "params": params},
        available_models=available_models,
        current_run_signatures=current_run_signatures,
        memory_payload=historical_memory,
        trial_history=trial_records,
        family_state=family_state,
        duplicate_settings=llm_config.get("duplicate_similarity_thresholds", {}),
    )


def initialize_research_log(log_path: Path, baseline_metrics: dict[str, Any]) -> None:
    """Create a fresh research log with the baseline threshold recorded."""
    timestamp = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")
    baseline_line = (
        f"[{timestamp}] Baseline | Model: {baseline_metrics['model_name']} | "
        f"RMSE: {baseline_metrics['rmse']:.2f} | R2: {baseline_metrics['r2']:.2f} | "
        f"Composite: {baseline_metrics['composite_score']:.3f} | "
        f"Validation: {baseline_metrics['validation_verdict']} | Threshold established"
    )
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write("AutoCivil-Lab Research Log\n")
        handle.write(baseline_line + "\n")


def append_research_log(log_path: Path, line: str) -> None:
    """Append a single formatted line to the research log."""
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def _build_baseline_best_result(
    baseline_metrics: dict[str, Any],
) -> dict[str, Any]:
    """Normalize the baseline into the same result shape used by the search loop."""
    baseline_result = copy.deepcopy(baseline_metrics)
    baseline_result["source"] = "baseline"
    baseline_result["beats_baseline"] = False
    baseline_result["trial_number"] = None
    baseline_result["best_trial"] = None
    return baseline_result


def initialize_search_state(
    outputs_dir: Path,
    baseline_metrics: dict[str, Any],
    *,
    config: dict[str, Any],
    run_id: str,
) -> dict[str, Any]:
    """Seed the live search-state artifacts from the saved baseline."""
    baseline_model_path = outputs_dir / "baseline_model.pkl"
    state_model_path = outputs_dir / SEARCH_STATE_BEST_MODEL_FILENAME
    final_json_artifacts = (
        outputs_dir / FINAL_BEST_RESULT_FILENAME,
        outputs_dir / FINAL_METRICS_FILENAME,
        outputs_dir / "ensemble_metrics.json",
    )
    for artifact_path in final_json_artifacts:
        artifact_path.unlink(missing_ok=True)
    (outputs_dir / FINAL_BEST_MODEL_FILENAME).unlink(missing_ok=True)
    shutil.copy2(baseline_model_path, state_model_path)
    baseline_result = _build_baseline_best_result(baseline_metrics)
    baseline_result["run_id"] = run_id
    write_run_scoped_json_artifact(
        outputs_dir=outputs_dir,
        filename=SEARCH_STATE_BEST_RESULT_FILENAME,
        payload=baseline_result,
        run_id=run_id,
        source_mode="baseline",
        config=config,
        model_id=str(baseline_result.get("model_name", "baseline")),
    )
    return baseline_result


def build_trial_record(
    trial_number: int,
    model_name: str,
    display_name: str,
    params: dict[str, Any],
    result: dict[str, Any] | None,
    selection_status: str,
    error_message: str | None = None,
    trial_runtime_seconds: float | None = None,
    budget_status: str | None = None,
) -> dict[str, Any]:
    """Build a flat record suitable for CSV export."""
    base_record = {
        "trial_number": trial_number,
        "model_name": model_name,
        "display_name": display_name,
        "hyperparameters": json.dumps(to_serializable(params)),
        "selection_status": selection_status,
        "validation_verdict": None if result is None else result["validation_verdict"],
        "error_message": error_message,
        "trial_runtime_seconds": trial_runtime_seconds,
        "budget_status": budget_status,
    }
    if result is None:
        base_record.update(
            {
                "rmse": None,
                "mae": None,
                "r2": None,
                "composite_score": None,
                "test_rmse": None,
                "test_mae": None,
                "test_r2": None,
                "test_composite_score": None,
                "validation_pass_rate": None,
                "failed_count": None,
                "hard_failed_count": None,
                "warning_count": None,
                "suspicious_count": None,
                "durability_caution_count": None,
                "dataset_anomaly_count": None,
            }
        )
        return base_record

    validation_report = result["validation_report"]
    validation_metrics = result.get(
        "selection_metrics",
        result.get("val_metrics", result.get("test_metrics", {})),
    )
    base_record.update(
        {
            "rmse": result["rmse"],
            "mae": result["mae"],
            "r2": result["r2"],
            "composite_score": result["composite_score"],
            "test_rmse": validation_metrics.get("rmse"),
            "test_mae": validation_metrics.get("mae"),
            "test_r2": validation_metrics.get("r2"),
            "test_composite_score": validation_metrics.get("composite_score"),
            "validation_pass_rate": validation_report["pass_rate"],
            "failed_count": validation_report["failed_count"],
            "hard_failed_count": validation_report.get("hard_failed_count", validation_report["failed_count"]),
            "warning_count": validation_report["warning_count"],
            "suspicious_count": validation_report["suspicious_count"],
            "durability_caution_count": validation_report.get("durability_caution_count", 0),
            "dataset_anomaly_count": validation_report.get("dataset_anomaly_count", 0),
        }
    )
    return base_record


def write_final_acceptance_artifact(
    outputs_dir: Path,
    brief: dict[str, Any],
    *,
    config: dict[str, Any],
    run_id: str,
) -> dict[str, Any]:
    """Write final acceptance status derived from final_metrics.json as source of truth."""
    final_metrics = load_json_artifact(outputs_dir / FINAL_METRICS_FILENAME)
    decision = build_acceptance_decision(final_metrics=final_metrics, brief=brief)
    consistency_report = validate_final_artifact_consistency(outputs_dir)
    if not consistency_report["consistent"]:
        decision["accepted"] = False
        decision["decision_reason"] = "rejected: stale or mixed-run artifacts detected"
        decision["stale_artifact_inputs"] = consistency_report["mismatches"]
    _, _, enriched_decision = write_run_scoped_json_artifact(
        outputs_dir=outputs_dir,
        filename=FINAL_ACCEPTANCE_FILENAME,
        payload=decision,
        run_id=run_id,
        source_mode="acceptance",
        config=config,
        model_artifact_id=final_metrics.get("best_search_metrics", {}).get("model_artifact_id"),
        model_id=str(final_metrics.get("best_model_name", "unknown")),
        parent_artifact_ids=[artifact_id(final_metrics)] if artifact_id(final_metrics) else [],
    )
    return enriched_decision


def initialize_optuna_results_csv(csv_path: Path) -> None:
    """Create a fresh results CSV with the stable project schema."""
    pd.DataFrame(columns=OPTUNA_RESULTS_COLUMNS).to_csv(csv_path, index=False)


def append_optuna_trial_record(csv_path: Path, record: dict[str, Any]) -> None:
    """Append a single trial record to the persistent Optuna results CSV."""
    pd.DataFrame([record], columns=OPTUNA_RESULTS_COLUMNS).to_csv(
        csv_path,
        mode="a",
        header=not csv_path.exists() or csv_path.stat().st_size == 0,
        index=False,
    )


def build_search_sync_writer(outputs_dir: Path) -> AtomicArtifactWriter:
    """Create the synchronized writer used by the search loop."""
    return AtomicArtifactWriter(
        outputs_dir=outputs_dir,
        csv_targets=[("optuna_results.csv", OPTUNA_RESULTS_COLUMNS)],
        best_result_filename=SEARCH_STATE_BEST_RESULT_FILENAME,
    )


def _safe_int(value: Any) -> int | None:
    """Normalize a possibly-missing integer value."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, str) and not value.strip():
        return None
    return int(value)


def _csv_best_trial(csv_path: Path) -> dict[str, Any] | None:
    """Return the current best-scoring trial from the Optuna CSV."""
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return None

    frame = pd.read_csv(csv_path)
    if frame.empty or "composite_score" not in frame.columns:
        return None

    new_best_rows = frame.loc[frame["selection_status"] == "new_best"].copy()
    if not new_best_rows.empty:
        chosen_row = new_best_rows.sort_values("trial_number").iloc[-1]
        return {
            "trial_number": _safe_int(chosen_row["trial_number"]),
            "model_name": str(chosen_row["model_name"]),
            "composite_score": float(chosen_row["composite_score"]),
        }

    valid_rows = frame.dropna(subset=["composite_score"]).copy()
    if valid_rows.empty:
        return None

    best_index = valid_rows["composite_score"].astype(float).idxmax()
    best_row = valid_rows.loc[best_index]
    return {
        "trial_number": _safe_int(best_row["trial_number"]),
        "model_name": str(best_row["model_name"]),
        "composite_score": float(best_row["composite_score"]),
    }


def sync_check(outputs_dir: Path, *, emit_warning: bool = True) -> bool:
    """Compare the live search-state winner against the Optuna CSV."""
    state_path = outputs_dir / SEARCH_STATE_BEST_RESULT_FILENAME
    csv_path = outputs_dir / "optuna_results.csv"
    if not state_path.exists():
        return True

    state_best = load_json_artifact(state_path)
    state_trial_number = _safe_int(state_best.get("trial_number", state_best.get("best_trial")))
    csv_best = _csv_best_trial(csv_path)

    if state_trial_number is None and csv_best is None:
        return True

    if state_trial_number is None:
        if csv_best is None:
            return True
        state_score = float(state_best["composite_score"])
        in_sync = state_score >= float(csv_best["composite_score"]) - 1e-12
        if not in_sync and emit_warning:
            log_status(
                "WARNING search_artifact_desync "
                f"| state_best_trial=None "
                f"| state_best_model={state_best['model_name']} "
                f"| csv_best_trial={csv_best['trial_number']} "
                f"| csv_best_model={csv_best['model_name']}"
            )
        return in_sync

    if csv_best is None:
        if emit_warning:
            log_status(
                "WARNING search_artifact_desync "
                f"| state_best_trial={state_trial_number} "
                "| csv_best_trial=None"
            )
        return False

    score_matches = abs(float(state_best["composite_score"]) - float(csv_best["composite_score"])) < 1e-12
    in_sync = (
        state_trial_number == csv_best["trial_number"]
        and str(state_best["model_name"]) == str(csv_best["model_name"])
        and score_matches
    )
    if not in_sync and emit_warning:
        log_status(
            "WARNING search_artifact_desync "
            f"| state_best_trial={state_trial_number} "
            f"| state_best_model={state_best['model_name']} "
            f"| csv_best_trial={csv_best['trial_number']} "
            f"| csv_best_model={csv_best['model_name']}"
        )
    return in_sync


def calculate_improvement_percentage(baseline_score: float, best_score: float) -> float:
    """Calculate percentage improvement in composite score relative to baseline."""
    denominator = max(abs(baseline_score), 1e-8)
    return ((best_score - baseline_score) / denominator) * 100.0


def build_validation_summary(validation_report: dict[str, Any]) -> dict[str, Any]:
    """Flatten the validator report into the stable final-metrics summary shape."""
    return summarize_validation_report(validation_report)


def build_final_metrics_payload(
    baseline_metrics: dict[str, Any],
    final_best_result: dict[str, Any],
) -> dict[str, Any]:
    """Build the canonical final-metrics artifact from the finalized winning result."""
    improvement_percentage = calculate_improvement_percentage(
        float(baseline_metrics["composite_score"]),
        float(final_best_result["composite_score"]),
    )
    validation_report = dict(final_best_result.get("validation_report", {}))
    return {
        "baseline_metrics": copy.deepcopy(baseline_metrics),
        "best_search_metrics": copy.deepcopy(final_best_result),
        "improvement_percentage": improvement_percentage,
        "composite_improvement_pct": improvement_percentage,
        "validation_verdict": final_best_result["validation_verdict"],
        "best_model_name": final_best_result["model_name"],
        "best_model_hyperparameters": copy.deepcopy(final_best_result["hyperparameters"]),
        "validation_summary": build_validation_summary(validation_report),
        "validation_metrics": copy.deepcopy(
            final_best_result.get(
                "selection_metrics",
                final_best_result.get("val_metrics", final_best_result.get("test_metrics", {})),
            )
        ),
    }


def _best_trial_id(payload: dict[str, Any]) -> int | None:
    """Extract the canonical trial identifier from a result payload."""
    return _safe_int(payload.get("trial_number", payload.get("best_trial")))


def _load_best_result_for_repair(outputs_dir: Path) -> dict[str, Any] | None:
    """Load the most relevant best-result payload for CSV repair or backfill."""
    search_state_path = outputs_dir / SEARCH_STATE_BEST_RESULT_FILENAME
    if search_state_path.exists():
        return load_json_artifact(search_state_path)

    final_metrics_path = outputs_dir / FINAL_METRICS_FILENAME
    if final_metrics_path.exists():
        final_metrics = load_json_artifact(final_metrics_path)
        best_search_metrics = final_metrics.get("best_search_metrics")
        if isinstance(best_search_metrics, dict):
            return best_search_metrics

    final_best_path = outputs_dir / FINAL_BEST_RESULT_FILENAME
    if final_best_path.exists():
        return load_json_artifact(final_best_path)

    return None


def resolve_final_best_result(outputs_dir: Path, baseline_metrics: dict[str, Any]) -> dict[str, Any]:
    """Recompute the winning result from the final study outputs and validate artifact alignment."""
    baseline_result = _build_baseline_best_result(baseline_metrics)
    csv_best = _csv_best_trial(outputs_dir / "optuna_results.csv")
    candidate_results: list[dict[str, Any]] = []

    final_metrics_path = outputs_dir / FINAL_METRICS_FILENAME
    if final_metrics_path.exists():
        final_metrics = load_json_artifact(final_metrics_path)
        best_search_metrics = final_metrics.get("best_search_metrics")
        if isinstance(best_search_metrics, dict):
            candidate_results.append(best_search_metrics)

    final_best_path = outputs_dir / FINAL_BEST_RESULT_FILENAME
    if final_best_path.exists():
        candidate_results.append(load_json_artifact(final_best_path))

    search_state_path = outputs_dir / SEARCH_STATE_BEST_RESULT_FILENAME
    if search_state_path.exists():
        candidate_results.append(load_json_artifact(search_state_path))

    search_state_result = next(
        (
            candidate
            for candidate in candidate_results
            if isinstance(candidate, dict) and not _artifact_is_stale(candidate)
        ),
        baseline_result,
    )

    if csv_best is None or float(csv_best["composite_score"]) <= float(baseline_result["composite_score"]) + 1e-12:
        state_trial_id = _best_trial_id(search_state_result)
        state_score = float(search_state_result["composite_score"])
        baseline_score = float(baseline_result["composite_score"])
        if state_trial_id is not None and state_score > baseline_score + 1e-12:
            raise RuntimeError(
                "Final artifact desync | "
                f"state_trial={state_trial_id} | baseline_composite={baseline_score:.12f} | "
                f"state_composite={state_score:.12f}"
            )
        return baseline_result

    csv_trial_id = csv_best["trial_number"]
    matching_candidate = next(
        (
            candidate
            for candidate in candidate_results
            if isinstance(candidate, dict)
            and not _artifact_is_stale(candidate)
            and _best_trial_id(candidate) == csv_trial_id
            and str(candidate.get("model_name")) == str(csv_best["model_name"])
            and abs(float(candidate.get("composite_score", float("-inf"))) - float(csv_best["composite_score"]))
            < 1e-12
        ),
        None,
    )
    if matching_candidate is not None:
        return matching_candidate

    recoverable_candidate = next(
        (
            candidate
            for candidate in candidate_results
            if isinstance(candidate, dict)
            and _best_trial_id(candidate) == csv_trial_id
            and str(candidate.get("model_name")) == str(csv_best["model_name"])
            and abs(float(candidate.get("composite_score", float("-inf"))) - float(csv_best["composite_score"]))
            < 1e-12
        ),
        None,
    )
    if recoverable_candidate is not None:
        return recoverable_candidate

    state_trial_id = _best_trial_id(search_state_result)
    if state_trial_id != csv_trial_id:
        raise RuntimeError(
            "Final artifact desync | "
            f"state_trial={state_trial_id} | csv_best_trial={csv_trial_id}"
        )
    if str(search_state_result["model_name"]) != str(csv_best["model_name"]):
        raise RuntimeError(
            "Final artifact desync | "
            f"state_model={search_state_result['model_name']} | csv_best_model={csv_best['model_name']}"
        )
    if abs(float(search_state_result["composite_score"]) - float(csv_best["composite_score"])) >= 1e-12:
        raise RuntimeError(
            "Final artifact desync | "
            f"state_composite={float(search_state_result['composite_score']):.12f} | "
            f"csv_composite={float(csv_best['composite_score']):.12f}"
        )
    return search_state_result


def finalize_search_artifacts(
    outputs_dir: Path,
    baseline_metrics: dict[str, Any],
    *,
    config: dict[str, Any],
    run_id: str,
) -> dict[str, Any]:
    """Write the final JSON artifacts from one canonical finalized winner object."""
    final_best_result = _strip_stale_fields(resolve_final_best_result(outputs_dir, baseline_metrics))
    final_trial_id = _best_trial_id(final_best_result)
    state_model_path = outputs_dir / SEARCH_STATE_BEST_MODEL_FILENAME
    source_model_path = outputs_dir / "baseline_model.pkl" if final_trial_id is None else state_model_path
    if not source_model_path.exists():
        raise FileNotFoundError(f"Missing search-state model artifact: {source_model_path}")
    source_model = load_pickle_artifact(source_model_path)
    final_model_metadata = save_pickle_artifact(
        outputs_dir / FINAL_BEST_MODEL_FILENAME,
        source_model,
        config=config,
        model_id=str(final_best_result["model_name"]),
        run_id=run_id,
        source_mode="search",
        parent_artifact_ids=[artifact_id(final_best_result)] if artifact_id(final_best_result) else [],
    )
    final_best_result["run_id"] = run_id
    final_best_result["model_artifact_id"] = final_model_metadata["artifact_id"]

    final_metrics = build_final_metrics_payload(baseline_metrics, final_best_result)
    write_run_scoped_json_artifact(
        outputs_dir=outputs_dir,
        filename=FINAL_METRICS_FILENAME,
        payload=final_metrics,
        run_id=run_id,
        source_mode="search",
        config=config,
        model_artifact_id=final_model_metadata["artifact_id"],
        model_id=str(final_best_result["model_name"]),
        parent_artifact_ids=[artifact_id(final_best_result)] if artifact_id(final_best_result) else [],
    )
    write_run_scoped_json_artifact(
        outputs_dir=outputs_dir,
        filename=FINAL_BEST_RESULT_FILENAME,
        payload=final_best_result,
        run_id=run_id,
        source_mode="search",
        config=config,
        model_artifact_id=final_model_metadata["artifact_id"],
        model_id=str(final_best_result["model_name"]),
    )

    ensemble_metrics_path = outputs_dir / "ensemble_metrics.json"
    if str(final_best_result.get("source")) == "post_search_ensemble":
        write_run_scoped_json_artifact(
            outputs_dir=outputs_dir,
            filename="ensemble_metrics.json",
            payload=dict(final_best_result),
            run_id=run_id,
            source_mode="ensemble",
            config=config,
            model_artifact_id=final_model_metadata["artifact_id"],
            model_id=str(final_best_result["model_name"]),
            parent_artifact_ids=[artifact_id(final_best_result)] if artifact_id(final_best_result) else [],
        )
    else:
        mark_json_artifact_stale(
            ensemble_metrics_path,
            active_run_id=run_id,
            reason="finalized winner is not the current ensemble artifact",
        )

    written_final_metrics = load_json_artifact(outputs_dir / FINAL_METRICS_FILENAME)
    written_best_result = load_json_artifact(outputs_dir / FINAL_BEST_RESULT_FILENAME)
    metrics_best_result = written_final_metrics.get("best_search_metrics", {})
    metrics_trial_id = None if not isinstance(metrics_best_result, dict) else _best_trial_id(metrics_best_result)
    best_result_trial_id = _best_trial_id(written_best_result)
    if metrics_trial_id != best_result_trial_id:
        raise RuntimeError(
            "Final artifact desync | "
            f"final_metrics_trial={metrics_trial_id} | best_search_result_trial={best_result_trial_id}"
        )

    if ensemble_metrics_path.exists():
        ensemble_payload = load_json_artifact(ensemble_metrics_path)
        if str(ensemble_payload.get("status")) == "new_best":
            ensemble_trial_id = _best_trial_id(ensemble_payload)
            if ensemble_trial_id != best_result_trial_id:
                raise RuntimeError(
                    "Final artifact desync | "
                    f"ensemble_trial={ensemble_trial_id} | best_search_result_trial={best_result_trial_id}"
                )

    return final_best_result


def _parse_param_value(raw_value: str) -> Any:
    """Parse a scalar parameter token from the research log."""
    token = raw_value.strip()
    if token == "None":
        return None
    if token == "True":
        return True
    if token == "False":
        return False
    if re.fullmatch(r"[-+]?\d+", token):
        return int(token)
    if re.fullmatch(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:e[-+]?\d+)?", token, flags=re.IGNORECASE):
        return float(token)
    return token


def _parse_hyperparameters_from_log(raw_text: str) -> dict[str, Any]:
    """Parse compact `key=value` hyperparameters from a research-log line."""
    params: dict[str, Any] = {}
    for match in re.finditer(r"([A-Za-z0-9_]+)=([^\s|]+)", raw_text):
        params[match.group(1)] = _parse_param_value(match.group(2))
    return params


def parse_research_log_trials(
    log_path: Path,
    available_models: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Parse trial metadata from the append-only research log."""
    if not log_path.exists():
        raise FileNotFoundError(f"Research log not found: {log_path}")

    display_to_model = {
        str(model_config.get("display_name", model_name)): model_name
        for model_name, model_config in available_models.items()
    }
    display_to_model.update({model_name: model_name for model_name in available_models})
    float_pattern = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:e[-+]?\d+)?"
    trial_pattern = re.compile(
        rf"^\[(?P<timestamp>[^\]]+)\]\s+Trial\s+(?P<trial_number>\d+)\s+\|\s+Model:\s+"
        rf"(?P<display_name>[^|]+?)\s+\|\s+(?P<rest>.+)$"
    )

    parsed_trials: dict[int, dict[str, Any]] = {}
    with log_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            match = trial_pattern.match(line)
            if not match:
                continue

            trial_number = int(match.group("trial_number"))
            display_name = match.group("display_name").strip()
            model_name = display_to_model.get(display_name, display_name)
            rest = match.group("rest")
            params_text = rest.split("| RMSE:", 1)[0].split("| Validation:", 1)[0].strip()
            selection_status = "no_improvement"
            error_message = None
            if "Validation: ERROR" in rest:
                selection_status = "error"
                error_message = rest.split("| Validation: ERROR |", 1)[-1].strip(" |")
            elif "Rejected" in rest:
                selection_status = "rejected_validation_fail"
            elif "New best" in rest:
                selection_status = "new_best"

            def extract_float(label: str) -> float | None:
                metric_match = re.search(rf"{label}:\s*({float_pattern})", rest, flags=re.IGNORECASE)
                return None if metric_match is None else float(metric_match.group(1))

            validation_match = re.search(r"Validation:\s*([A-Z]+)", rest)
            parsed_trials[trial_number] = {
                "trial_number": trial_number,
                "model_name": model_name,
                "display_name": display_name,
                "hyperparameters": _parse_hyperparameters_from_log(params_text),
                "selection_status": selection_status,
                "validation_verdict": None if validation_match is None else validation_match.group(1),
                "rmse": extract_float("RMSE"),
                "r2": extract_float("R2"),
                "composite_score": extract_float("Composite"),
                "error_message": error_message,
            }

    return [parsed_trials[trial_number] for trial_number in sorted(parsed_trials)]


def _build_repaired_trial_record(
    parsed_trial: dict[str, Any],
    existing_rows_by_trial: dict[int, dict[str, Any]],
    best_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Construct a CSV row from the research log, backfilling known metrics from the stale CSV."""
    trial_number = int(parsed_trial["trial_number"])
    base_row = {column: None for column in OPTUNA_RESULTS_COLUMNS}
    base_row.update(existing_rows_by_trial.get(trial_number, {}))
    base_row.update(
        {
            "trial_number": trial_number,
            "model_name": parsed_trial["model_name"],
            "display_name": parsed_trial["display_name"],
            "hyperparameters": json.dumps(to_serializable(parsed_trial["hyperparameters"])),
            "selection_status": parsed_trial["selection_status"],
            "validation_verdict": parsed_trial["validation_verdict"],
            "error_message": parsed_trial["error_message"],
            "rmse": parsed_trial["rmse"],
            "r2": parsed_trial["r2"],
            "composite_score": parsed_trial["composite_score"],
        }
    )
    best_trial_number = None if best_result is None else _safe_int(best_result.get("trial_number", best_result.get("best_trial")))
    if best_result is not None and best_trial_number == trial_number:
        validation_report = best_result.get("validation_report", {})
        test_metrics = best_result.get(
            "selection_metrics",
            best_result.get("val_metrics", best_result.get("test_metrics", {})),
        )
        base_row.update(
            {
                "model_name": best_result.get("model_name", base_row["model_name"]),
                "display_name": base_row["display_name"],
                "hyperparameters": json.dumps(to_serializable(best_result.get("hyperparameters", parsed_trial["hyperparameters"]))),
                "validation_verdict": best_result.get("validation_verdict", base_row["validation_verdict"]),
                "rmse": best_result.get("rmse", base_row["rmse"]),
                "mae": best_result.get("mae", base_row["mae"]),
                "r2": best_result.get("r2", base_row["r2"]),
                "composite_score": best_result.get("composite_score", base_row["composite_score"]),
                "test_rmse": test_metrics.get("rmse", base_row["test_rmse"]),
                "test_mae": test_metrics.get("mae", base_row["test_mae"]),
                "test_r2": test_metrics.get("r2", base_row["test_r2"]),
                "test_composite_score": test_metrics.get("composite_score", base_row["test_composite_score"]),
                "validation_pass_rate": validation_report.get("pass_rate", base_row["validation_pass_rate"]),
                "failed_count": validation_report.get("failed_count", base_row["failed_count"]),
                "hard_failed_count": validation_report.get("hard_failed_count", base_row["hard_failed_count"]),
                "warning_count": validation_report.get("warning_count", base_row["warning_count"]),
                "suspicious_count": validation_report.get("suspicious_count", base_row["suspicious_count"]),
                "durability_caution_count": validation_report.get(
                    "durability_caution_count",
                    base_row["durability_caution_count"],
                ),
                "dataset_anomaly_count": validation_report.get(
                    "dataset_anomaly_count",
                    base_row["dataset_anomaly_count"],
                ),
            }
        )
    return {column: base_row.get(column) for column in OPTUNA_RESULTS_COLUMNS}


def csv_is_stale(outputs_dir: Path, available_models: dict[str, dict[str, Any]]) -> bool:
    """Return whether the Optuna CSV is stale relative to the research log or best-result JSON."""
    csv_path = outputs_dir / "optuna_results.csv"
    log_path = outputs_dir / "research_log.txt"
    if not csv_path.exists() or not log_path.exists():
        return True

    parsed_trials = parse_research_log_trials(log_path, available_models)
    if not parsed_trials:
        return False

    try:
        csv_frame = pd.read_csv(csv_path)
    except Exception:
        return True

    if len(csv_frame) != len(parsed_trials):
        return True
    return not sync_check(outputs_dir, emit_warning=False)


def repair_optuna_results_csv(outputs_dir: Path, config: dict[str, Any]) -> Path:
    """Rebuild the Optuna results CSV from the append-only research log."""
    available_models = get_available_model_configs(config)
    csv_path = outputs_dir / "optuna_results.csv"
    log_path = outputs_dir / "research_log.txt"
    best_result = _load_best_result_for_repair(outputs_dir)
    parsed_trials = parse_research_log_trials(log_path, available_models)
    if not parsed_trials:
        raise RuntimeError(f"No trial entries were found in {log_path}.")

    existing_rows_by_trial: dict[int, dict[str, Any]] = {}
    if csv_path.exists() and csv_path.stat().st_size > 0:
        existing_frame = pd.read_csv(csv_path)
        for row in existing_frame.to_dict(orient="records"):
            trial_number = _safe_int(row.get("trial_number"))
            if trial_number is None:
                continue
            existing_rows_by_trial[trial_number] = row

    repaired_records = [
        _build_repaired_trial_record(parsed_trial, existing_rows_by_trial, best_result=best_result)
        for parsed_trial in parsed_trials
    ]
    pd.DataFrame(repaired_records, columns=OPTUNA_RESULTS_COLUMNS).to_csv(csv_path, index=False)
    sync_check(outputs_dir, emit_warning=True)
    log_status(f"Repaired stale Optuna CSV from research log at {csv_path}")
    return csv_path


def build_post_search_ensemble(
    outputs_dir: Path,
    config: dict[str, Any],
    run_id: str,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_val: pd.DataFrame,
    y_val: pd.Series,
    validator: EngineeringValidator,
    sync_writer: AtomicArtifactWriter | None = None,
) -> dict[str, Any]:
    """Build and evaluate a stacking ensemble from the best unique model families in the Optuna CSV."""
    ensemble_metrics_path = outputs_dir / "ensemble_metrics.json"
    optuna_results_path = outputs_dir / "optuna_results.csv"
    search_state_best_result_path = outputs_dir / SEARCH_STATE_BEST_RESULT_FILENAME
    current_best_result = load_json_artifact(search_state_best_result_path)

    if not optuna_results_path.exists() or optuna_results_path.stat().st_size == 0:
        ensemble_summary = {
            "status": "skipped",
            "reason": "optuna_results_missing",
            "selected_base_models": [],
        }
        write_run_scoped_json_artifact(
            outputs_dir=outputs_dir,
            filename="ensemble_metrics.json",
            payload=ensemble_summary,
            run_id=run_id,
            source_mode="ensemble",
            config=config,
        )
        return current_best_result

    optuna_frame = pd.read_csv(optuna_results_path)
    if optuna_frame.empty:
        ensemble_summary = {
            "status": "skipped",
            "reason": "optuna_results_empty",
            "selected_base_models": [],
        }
        write_run_scoped_json_artifact(
            outputs_dir=outputs_dir,
            filename="ensemble_metrics.json",
            payload=ensemble_summary,
            run_id=run_id,
            source_mode="ensemble",
            config=config,
        )
        return current_best_result

    filtered = optuna_frame.loc[
        optuna_frame["composite_score"].notna()
        & optuna_frame["model_name"].notna()
        & (optuna_frame["validation_verdict"].fillna("").astype(str) != "FAIL")
        & (optuna_frame["model_name"].astype(str) != "StackingRegressor")
    ].copy()
    if filtered.empty:
        ensemble_summary = {
            "status": "skipped",
            "reason": "no_valid_trials",
            "selected_base_models": [],
        }
        write_run_scoped_json_artifact(
            outputs_dir=outputs_dir,
            filename="ensemble_metrics.json",
            payload=ensemble_summary,
            run_id=run_id,
            source_mode="ensemble",
            config=config,
        )
        return current_best_result

    filtered["composite_score"] = filtered["composite_score"].astype(float)
    ranked = filtered.sort_values(["composite_score", "trial_number"], ascending=[False, True])
    selected_base_models: list[dict[str, Any]] = []
    ensemble_configs: list[tuple[str, dict[str, Any]]] = []
    # LLM PROPOSAL
    llm_top_trials: list[dict[str, Any]] = []
    for row in ranked.head(12).to_dict(orient="records"):
        raw_hyperparameters = row.get("hyperparameters", "{}")
        parsed_hyperparameters = (
            json.loads(raw_hyperparameters)
            if isinstance(raw_hyperparameters, str) and raw_hyperparameters.strip()
            else {}
        )
        llm_top_trials.append(
            {
                "trial_number": _safe_int(row.get("trial_number")),
                "model_name": str(row["model_name"]),
                "rmse": row.get("rmse"),
                "r2": row.get("r2"),
                "composite_score": float(row["composite_score"]),
                "validation_verdict": row.get("validation_verdict"),
                "params": parsed_hyperparameters,
            }
        )

    llm_ensemble_selection = None
    llm_config = config["search"].get("llm_proposals", {})
    if llm_config.get("enabled", False) and llm_top_trials:
        from llm_proposer import LLMProposer

        llm_proposer = LLMProposer(config)
        if llm_proposer.is_available():
            llm_ensemble_selection = llm_proposer.propose_ensemble(
                top_trials=llm_top_trials,
                available_models=get_available_model_configs(config),
            )
            if llm_ensemble_selection:
                log_status(
                    f"INFO llm_ensemble_selection | model_used={llm_proposer.smart_model} | "
                    f"selected={len(llm_ensemble_selection)}"
                )

    if llm_ensemble_selection:
        llm_top_trial_lookup = {
            (
                trial["model_name"],
                json.dumps(trial["params"], sort_keys=True, separators=(",", ":")),
            ): trial
            for trial in llm_top_trials
        }
        for suggestion in llm_ensemble_selection:
            ensemble_configs.append((suggestion["model_name"], suggestion["params"]))
            lookup_key = (
                suggestion["model_name"],
                json.dumps(suggestion["params"], sort_keys=True, separators=(",", ":")),
            )
            matched_trial = llm_top_trial_lookup.get(lookup_key)
            selected_base_models.append(
                {
                    "trial_number": None if matched_trial is None else matched_trial["trial_number"],
                    "model_name": suggestion["model_name"],
                    "composite_score": None if matched_trial is None else matched_trial["composite_score"],
                    "hyperparameters": suggestion["params"],
                }
            )
    else:
        top_family_rows = ranked.drop_duplicates(subset=["model_name"], keep="first").head(3)
        for row in top_family_rows.to_dict(orient="records"):
            raw_hyperparameters = row.get("hyperparameters", "{}")
            parsed_hyperparameters = (
                json.loads(raw_hyperparameters)
                if isinstance(raw_hyperparameters, str) and raw_hyperparameters.strip()
                else {}
            )
            model_name = str(row["model_name"])
            ensemble_configs.append((model_name, parsed_hyperparameters))
            selected_base_models.append(
                {
                    "trial_number": _safe_int(row.get("trial_number")),
                    "model_name": model_name,
                    "composite_score": float(row["composite_score"]),
                    "hyperparameters": parsed_hyperparameters,
                }
            )

    ensemble_model, ensemble_result = build_stacking_ensemble(
        ensemble_configs,
        x_train,
        y_train,
        x_val,
        y_val,
        validator,
        config,
    )
    ensemble_result = copy.deepcopy(ensemble_result)
    ensemble_result["source"] = "post_search_ensemble"
    ensemble_result["selected_base_models"] = selected_base_models
    ensemble_result["ensemble_size"] = len(selected_base_models)
    ensemble_result["status"] = "built"
    ensemble_result["run_id"] = run_id
    write_run_scoped_json_artifact(
        outputs_dir=outputs_dir,
        filename="ensemble_metrics.json",
        payload=ensemble_result,
        run_id=run_id,
        source_mode="ensemble",
        config=config,
        model_id="StackingRegressor",
        parent_artifact_ids=[artifact_id(current_best_result)] if artifact_id(current_best_result) else [],
    )

    previous_best_score = float(current_best_result.get("cv_r2", current_best_result.get("r2", -1e9)))
    ensemble_score = float(ensemble_result["cv_r2"])
    if ensemble_score > previous_best_score:
        numeric_trial_numbers = pd.to_numeric(optuna_frame["trial_number"], errors="coerce")
        max_trial_number = numeric_trial_numbers.max()
        next_trial_number = 1 if pd.isna(max_trial_number) else int(max_trial_number) + 1
        ensemble_result["trial_number"] = next_trial_number
        ensemble_result["best_trial"] = next_trial_number
        ensemble_result["beats_baseline"] = True
        ensemble_result["status"] = "new_best"

        ensemble_model_metadata = save_pickle_artifact(
            outputs_dir / SEARCH_STATE_BEST_MODEL_FILENAME,
            ensemble_model,
            config=config,
            model_id="StackingRegressor",
            run_id=run_id,
            source_mode="ensemble",
            parent_artifact_ids=[artifact_id(current_best_result)] if artifact_id(current_best_result) else [],
        )
        ensemble_result["model_artifact_id"] = ensemble_model_metadata["artifact_id"]
        write_run_scoped_json_artifact(
            outputs_dir=outputs_dir,
            filename="ensemble_metrics.json",
            payload=ensemble_result,
            run_id=run_id,
            source_mode="ensemble",
            config=config,
            model_artifact_id=ensemble_model_metadata["artifact_id"],
            model_id="StackingRegressor",
            parent_artifact_ids=[artifact_id(current_best_result)] if artifact_id(current_best_result) else [],
        )
        ensemble_record = build_trial_record(
            next_trial_number,
            "StackingRegressor",
            "StackingEnsemble",
            ensemble_result["hyperparameters"],
            ensemble_result,
            "new_best",
        )
        active_sync_writer = sync_writer or build_search_sync_writer(outputs_dir)
        active_sync_writer.record_new_best(trial_record=ensemble_record, best_result=ensemble_result)

        research_log_path = outputs_dir / "research_log.txt"
        timestamp = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")
        append_research_log(
            research_log_path,
            (
                f"[{timestamp}] Trial {next_trial_number:03d} | Model: StackingEnsemble | "
                f"ensemble_size={len(selected_base_models)} meta_learner=Ridge passthrough=True | "
                f"RMSE: {ensemble_result['rmse']:.2f} | R2: {ensemble_result['r2']:.2f} | "
                f"Composite: {ensemble_result['composite_score']:.3f} | "
                f"Validation: {ensemble_result['validation_verdict']} | New best ensemble"
            ),
        )
        log_status(
            f"INFO ensemble_beats_best_cv_r2 | ensemble={ensemble_score:.4f} | previous_best={previous_best_score:.4f}"
        )
        mark_json_artifact_stale(
            outputs_dir / FINAL_METRICS_FILENAME,
            active_run_id=run_id,
            reason="new best ensemble saved before final report refresh",
        )
        try:
            from uncertainty import recalibrate_uncertainty_artifacts

            recalibrate_uncertainty_artifacts(
                model_path=outputs_dir / SEARCH_STATE_BEST_MODEL_FILENAME,
                method=str(config["engineering"]["uncertainty_method"]),
                outputs_dir=outputs_dir,
                audit_partition="validation_audit",
            )
        except Exception as recalibration_exc:
            log_status(f"WARNING uncertainty_recalibration_failed | trial={next_trial_number} | {recalibration_exc}")
        return ensemble_result

    ensemble_result["status"] = "no_improvement"
    write_run_scoped_json_artifact(
        outputs_dir=outputs_dir,
        filename="ensemble_metrics.json",
        payload=ensemble_result,
        run_id=run_id,
        source_mode="ensemble",
        config=config,
        model_id=str(ensemble_result.get("model_name", "StackingRegressor")),
        parent_artifact_ids=[artifact_id(current_best_result)] if artifact_id(current_best_result) else [],
    )
    return current_best_result


def _warn_legacy_runtime(entrypoint: str) -> None:
    warnings.warn(LEGACY_RUNTIME_MESSAGE, DeprecationWarning, stacklevel=3)
    log_status(
        "WARNING legacy_runtime_redirect | "
        f"entrypoint={entrypoint} | canonical=research_loop.py"
    )


def _build_canonical_runtime_config(
    config: dict[str, Any],
    *,
    n_trials: int | None = None,
    min_runtime_minutes: float | None = None,
) -> dict[str, Any]:
    """Translate legacy search runtime settings onto the canonical research_loop config surface."""
    normalized = copy.deepcopy(config)
    experiment_config = normalized.setdefault("experiment", {})
    search_config = normalized.setdefault("search", {})
    research_config = normalized.setdefault("research", {})

    resolved_cycles = n_trials if n_trials is not None else experiment_config.get("optuna_trials")
    if resolved_cycles is not None:
        research_config["max_cycles"] = max(1, int(resolved_cycles))

    resolved_runtime_minutes = (
        min_runtime_minutes if min_runtime_minutes is not None else search_config.get("min_runtime_minutes")
    )
    if resolved_runtime_minutes is not None:
        research_config["max_runtime_minutes"] = max(0.0, float(resolved_runtime_minutes))

    if "max_trial_seconds" not in research_config and "max_trial_seconds" in search_config:
        research_config["max_trial_seconds"] = max(0.0, float(search_config["max_trial_seconds"]))

    if not research_config.get("brief_path") and search_config.get("research_brief_path"):
        research_config["brief_path"] = str(search_config["research_brief_path"])

    return normalized


def run_autocivil_loop(n_trials: int, config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Compatibility wrapper that redirects legacy callers into research_loop."""
    if config is None:
        config = load_config()
    _warn_legacy_runtime("search.run_autocivil_loop")
    normalized_config = _build_canonical_runtime_config(config, n_trials=n_trials)
    cycles_override = int(normalized_config["research"]["max_cycles"])
    return research_loop.run_engineering_research_loop(
        cycles_override=cycles_override,
        with_report=False,
        config_override=normalized_config,
    )

def main() -> int:
    """Run the legacy compatibility wrapper for the canonical research loop."""
    parser = argparse.ArgumentParser(
        description="Deprecated compatibility wrapper. Use research_loop.py for canonical research execution."
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=None,
        help="Override experiment.optuna_trials for this run only.",
    )
    parser.add_argument(
        "--min-runtime-minutes",
        type=float,
        default=None,
        help="Override search.min_runtime_minutes for this run only.",
    )
    parser.add_argument(
        "--repair",
        action="store_true",
        help="Repair a stale optuna_results.csv from outputs/research_log.txt and exit.",
    )
    args = parser.parse_args()

    try:
        config = load_config()
        if args.trials is not None:
            config["experiment"]["optuna_trials"] = int(args.trials)
        if args.min_runtime_minutes is not None:
            config["search"]["min_runtime_minutes"] = float(args.min_runtime_minutes)
        if args.repair:
            outputs_dir = get_outputs_dir(config)
            available_models = get_available_model_configs(config)
            if csv_is_stale(outputs_dir, available_models):
                repair_optuna_results_csv(outputs_dir, config)
            else:
                log_status("Optuna CSV is already in sync; no repair was needed.")
            return 0

        n_trials = int(config["experiment"]["optuna_trials"])
        normalized_config = _build_canonical_runtime_config(
            config,
            n_trials=n_trials,
            min_runtime_minutes=args.min_runtime_minutes,
        )
        _warn_legacy_runtime("search.main")
        best_result = research_loop.run_engineering_research_loop(
            cycles_override=int(normalized_config["research"]["max_cycles"]),
            with_report=False,
            config_override=normalized_config,
        )
        log_status(
            f"Legacy search wrapper complete. Best model: {best_result['model_name']} | "
            f"Composite={best_result['composite_score']:.4f} | "
            f"Validation={best_result['validation_verdict']}"
        )
        return 0
    except Exception as exc:
        log_status(f"Search failed: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
