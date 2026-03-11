"""Optuna-driven research loop for AutoCivil-Lab."""

from __future__ import annotations

import argparse
import copy
import json
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import optuna
import pandas as pd

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
    to_serializable,
)


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
]


def load_json_artifact(path: Path) -> dict[str, Any]:
    """Load a JSON artifact from disk."""
    if not path.exists():
        raise FileNotFoundError(f"Required artifact not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


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


def seed_best_result_from_baseline(
    outputs_dir: Path,
    baseline_metrics: dict[str, Any],
) -> dict[str, Any]:
    """Seed the current best search artifact from the saved baseline."""
    baseline_model_path = outputs_dir / "baseline_model.pkl"
    best_model_path = outputs_dir / "best_search_model.pkl"
    shutil.copy2(baseline_model_path, best_model_path)
    baseline_result = copy.deepcopy(baseline_metrics)
    baseline_result["source"] = "baseline"
    baseline_result["beats_baseline"] = False
    baseline_result["trial_number"] = None
    baseline_result["best_trial"] = None
    save_json_artifact(outputs_dir / "best_search_result.json", baseline_result)
    return baseline_result


def build_trial_record(
    trial_number: int,
    model_name: str,
    display_name: str,
    params: dict[str, Any],
    result: dict[str, Any] | None,
    selection_status: str,
    error_message: str | None = None,
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
    base_record.update(
        {
            "rmse": result["rmse"],
            "mae": result["mae"],
            "r2": result["r2"],
            "composite_score": result["composite_score"],
            "test_rmse": result["test_metrics"]["rmse"],
            "test_mae": result["test_metrics"]["mae"],
            "test_r2": result["test_metrics"]["r2"],
            "test_composite_score": result["test_metrics"]["composite_score"],
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
    """Compare the best trial recorded in JSON and CSV artifacts."""
    json_path = outputs_dir / "best_search_result.json"
    csv_path = outputs_dir / "optuna_results.csv"
    if not json_path.exists():
        return True

    json_best = load_json_artifact(json_path)
    json_trial_number = _safe_int(json_best.get("trial_number", json_best.get("best_trial")))
    csv_best = _csv_best_trial(csv_path)

    if json_trial_number is None and csv_best is None:
        return True
    if json_trial_number is None or csv_best is None:
        if emit_warning:
            log_status(
                "WARNING search_artifact_desync "
                f"| json_best_trial={json_trial_number} "
                f"| csv_best_trial={None if csv_best is None else csv_best['trial_number']}"
            )
        return False

    score_matches = abs(float(json_best["composite_score"]) - float(csv_best["composite_score"])) < 1e-12
    in_sync = (
        json_trial_number == csv_best["trial_number"]
        and str(json_best["model_name"]) == str(csv_best["model_name"])
        and score_matches
    )
    if not in_sync and emit_warning:
        log_status(
            "WARNING search_artifact_desync "
            f"| json_best_trial={json_trial_number} "
            f"| json_best_model={json_best['model_name']} "
            f"| csv_best_trial={csv_best['trial_number']} "
            f"| csv_best_model={csv_best['model_name']}"
        )
    return in_sync


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
        test_metrics = best_result.get("test_metrics", {})
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
    best_result_path = outputs_dir / "best_search_result.json"
    best_result = load_json_artifact(best_result_path) if best_result_path.exists() else None
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


def run_autocivil_loop(n_trials: int) -> dict[str, Any]:
    """Execute the visible propose-run-compare-keep research loop."""
    config = load_config()
    set_global_seed(int(config["experiment"]["random_seed"]))
    outputs_dir = get_outputs_dir(config)
    min_runtime_minutes = float(config["search"].get("min_runtime_minutes", 0.0))
    if min_runtime_minutes < 0.0:
        raise ValueError("search.min_runtime_minutes must be non-negative.")
    min_runtime_seconds = min_runtime_minutes * 60.0
    search_start_time = time.perf_counter()
    baseline_metrics_path = outputs_dir / "baseline_metrics.json"
    baseline_metrics = load_json_artifact(baseline_metrics_path)

    data = load_dataset(config)
    x_train, x_test, y_train, y_test = split_dataset(data, config)
    validator = EngineeringValidator.from_config(config)
    available_models = get_available_model_configs(config)

    best_result = seed_best_result_from_baseline(outputs_dir, baseline_metrics)
    current_best_composite = float(best_result["composite_score"])
    current_best_name = str(best_result["model_name"])

    research_log_path = outputs_dir / "research_log.txt"
    initialize_research_log(research_log_path, baseline_metrics)
    optuna_results_path = outputs_dir / "optuna_results.csv"
    initialize_optuna_results_csv(optuna_results_path)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", study_name="autocivil_search")
    trial_records: list[dict[str, Any]] = []

    trial_number = 0
    while True:
        trial_number += 1
        trial = study.ask()
        model_name, display_name, params = sample_model_configuration(trial, available_models)
        try:
            model, result = evaluate_candidate(
                model_name,
                params,
                x_train,
                y_train,
                x_test,
                y_test,
                validator,
                config,
            )
            verdict = result["validation_verdict"]
            if verdict == "FAIL":
                selection_status = "rejected_validation_fail"
                study.tell(trial, -1e9)
                trial_record = build_trial_record(
                    trial_number,
                    model_name,
                    display_name,
                    params,
                    result,
                    selection_status,
                )
                trial_records.append(trial_record)
                append_optuna_trial_record(optuna_results_path, trial_record)
                timestamp = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")
                append_research_log(
                    research_log_path,
                    (
                        f"[{timestamp}] Trial {trial_number:03d} | Model: {display_name} | "
                        f"{format_hyperparameters(params)} | RMSE: {result['rmse']:.2f} | "
                        f"R2: {result['r2']:.2f} | Composite: {result['composite_score']:.3f} | "
                        f"Validation: {verdict} | Rejected"
                    ),
                )
            else:
                composite_score = float(result["composite_score"])
                study.tell(trial, composite_score)
                improved = composite_score > current_best_composite
                selection_status = "new_best" if improved else "no_improvement"
                if improved:
                    current_best_composite = composite_score
                    current_best_name = display_name
                    best_result = copy.deepcopy(result)
                    best_result["source"] = "search"
                    best_result["beats_baseline"] = True
                    best_result["trial_number"] = trial_number
                    best_result["best_trial"] = trial_number
                    save_pickle_artifact(
                        outputs_dir / "best_search_model.pkl",
                        model,
                        config=config,
                        model_id=model_name,
                    )
                    save_json_artifact(outputs_dir / "best_search_result.json", best_result)
                    try:
                        from uncertainty import recalibrate_uncertainty_artifacts

                        recalibrate_uncertainty_artifacts(
                            model_path=outputs_dir / "best_search_model.pkl",
                            method=str(config["engineering"]["uncertainty_method"]),
                        )
                    except Exception as recalibration_exc:
                        log_status(
                            f"WARNING uncertainty_recalibration_failed | trial={trial_number} | "
                            f"{recalibration_exc}"
                        )

                trial_record = build_trial_record(
                    trial_number,
                    model_name,
                    display_name,
                    params,
                    result,
                    selection_status,
                )
                trial_records.append(trial_record)
                append_optuna_trial_record(optuna_results_path, trial_record)
                timestamp = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")
                status_label = "New best" if improved else "No improvement"
                append_research_log(
                    research_log_path,
                    (
                        f"[{timestamp}] Trial {trial_number:03d} | Model: {display_name} | "
                        f"{format_hyperparameters(params)} | RMSE: {result['rmse']:.2f} | "
                        f"R2: {result['r2']:.2f} | Composite: {result['composite_score']:.3f} | "
                        f"Validation: {verdict} | {status_label}"
                    ),
                )
            sync_check(outputs_dir)
        except Exception as exc:
            error_text = str(exc)
            study.tell(trial, -1e9)
            trial_record = build_trial_record(
                trial_number,
                model_name,
                display_name,
                params,
                None,
                "error",
                error_message=error_text,
            )
            trial_records.append(trial_record)
            append_optuna_trial_record(optuna_results_path, trial_record)
            timestamp = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")
            append_research_log(
                research_log_path,
                (
                    f"[{timestamp}] Trial {trial_number:03d} | Model: {display_name} | "
                    f"{format_hyperparameters(params)} | Validation: ERROR | {error_text}"
                ),
            )
            sync_check(outputs_dir)

        elapsed_seconds = time.perf_counter() - search_start_time
        trials_floor_reached = trial_number >= n_trials
        runtime_floor_reached = elapsed_seconds >= min_runtime_seconds
        progress_interval = int(config["search"]["progress_interval"])
        if trial_number % progress_interval == 0:
            valid_trials = [
                row for row in trial_records if row["selection_status"] not in {"error", "rejected_validation_fail"}
            ]
            remaining_seconds = max(0.0, min_runtime_seconds - elapsed_seconds)
            log_status(
                f"Progress trial={trial_number} | target_trials={n_trials} | "
                f"elapsed={elapsed_seconds / 60.0:.1f}m | "
                f"min_runtime_remaining={remaining_seconds / 60.0:.1f}m | "
                f"valid_trials={len(valid_trials)} | "
                f"current_best_model={current_best_name} | "
                f"current_best_composite={current_best_composite:.4f}"
            )
        if trials_floor_reached and runtime_floor_reached:
            break

    save_json_artifact(outputs_dir / "best_search_result.json", best_result)
    sync_check(outputs_dir)
    return best_result


def main() -> int:
    """Run the full model search loop."""
    parser = argparse.ArgumentParser(description="Optuna-driven research loop for AutoCivil-Lab.")
    parser.add_argument(
        "--repair",
        action="store_true",
        help="Repair a stale optuna_results.csv from outputs/research_log.txt and exit.",
    )
    args = parser.parse_args()

    try:
        config = load_config()
        outputs_dir = get_outputs_dir(config)
        available_models = get_available_model_configs(config)
        if args.repair:
            if csv_is_stale(outputs_dir, available_models):
                repair_optuna_results_csv(outputs_dir, config)
            else:
                log_status("Optuna CSV is already in sync; no repair was needed.")
            return 0

        n_trials = int(config["experiment"]["optuna_trials"])
        best_result = run_autocivil_loop(n_trials=n_trials)
        log_status(
            f"Search complete. Best model: {best_result['model_name']} | "
            f"Composite={best_result['composite_score']:.4f} | "
            f"Validation={best_result['validation_verdict']}"
        )
        return 0
    except Exception as exc:
        log_status(f"Search failed: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
