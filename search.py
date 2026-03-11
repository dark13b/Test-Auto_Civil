"""Optuna-driven research loop for AutoCivil-Lab."""

from __future__ import annotations

import copy
import json
import shutil
import sys
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


def run_autocivil_loop(n_trials: int) -> dict[str, Any]:
    """Execute the visible propose-run-compare-keep research loop."""
    config = load_config()
    set_global_seed(int(config["experiment"]["random_seed"]))
    outputs_dir = get_outputs_dir(config)
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

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", study_name="autocivil_search")
    trial_records: list[dict[str, Any]] = []

    for trial_number in range(1, n_trials + 1):
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
            # FAIL is reserved for impossible outputs; engineering warnings are still
            # logged but remain eligible for selection.
            if verdict == "FAIL":
                selection_status = "rejected_validation_fail"
                study.tell(trial, -1e9)
                trial_records.append(
                    build_trial_record(trial_number, model_name, display_name, params, result, selection_status)
                )
                timestamp = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")
                log_line = (
                    f"[{timestamp}] Trial {trial_number:03d} | Model: {display_name} | "
                    f"{format_hyperparameters(params)} | RMSE: {result['rmse']:.2f} | "
                    f"R2: {result['r2']:.2f} | Composite: {result['composite_score']:.3f} | "
                    f"Validation: {verdict} | ❌ Rejected"
                )
                append_research_log(research_log_path, log_line)
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
                    save_pickle_artifact(outputs_dir / "best_search_model.pkl", model)
                    save_json_artifact(outputs_dir / "best_search_result.json", best_result)

                trial_records.append(
                    build_trial_record(trial_number, model_name, display_name, params, result, selection_status)
                )
                timestamp = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")
                status_label = "✅ New best" if improved else "❌ No improvement"
                log_line = (
                    f"[{timestamp}] Trial {trial_number:03d} | Model: {display_name} | "
                    f"{format_hyperparameters(params)} | RMSE: {result['rmse']:.2f} | "
                    f"R2: {result['r2']:.2f} | Composite: {result['composite_score']:.3f} | "
                    f"Validation: {verdict} | {status_label}"
                )
                append_research_log(research_log_path, log_line)
        except Exception as exc:
            error_text = str(exc)
            study.tell(trial, -1e9)
            trial_records.append(
                build_trial_record(
                    trial_number,
                    model_name,
                    display_name,
                    params,
                    None,
                    "error",
                    error_message=error_text,
                )
            )
            timestamp = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")
            log_line = (
                f"[{timestamp}] Trial {trial_number:03d} | Model: {display_name} | "
                f"{format_hyperparameters(params)} | Validation: ERROR | ❌ {error_text}"
            )
            append_research_log(research_log_path, log_line)

        progress_interval = int(config["search"]["progress_interval"])
        if trial_number % progress_interval == 0 or trial_number == n_trials:
            valid_trials = [
                row for row in trial_records if row["selection_status"] not in {"error", "rejected_validation_fail"}
            ]
            log_status(
                f"Progress {trial_number}/{n_trials} trials | valid_trials={len(valid_trials)} | "
                f"current_best_model={current_best_name} | current_best_composite={current_best_composite:.4f}"
            )

    pd.DataFrame(trial_records).to_csv(outputs_dir / "optuna_results.csv", index=False)
    save_json_artifact(outputs_dir / "best_search_result.json", best_result)
    return best_result


def main() -> int:
    """Run the full model search loop."""
    try:
        config = load_config()
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
