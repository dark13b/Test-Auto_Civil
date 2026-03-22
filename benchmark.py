"""Academic benchmark runner for AutoCivil-Lab."""

from __future__ import annotations

import copy
import json
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd

from search import csv_is_stale, repair_optuna_results_csv, sync_check
from train import (
    EngineeringValidator,
    build_cv_splitter,
    compute_regression_metrics,
    cross_validate_model,
    get_outputs_dir,
    instantiate_model,
    load_config,
    load_dataset,
    log_status,
    save_json_artifact,
    save_pickle_artifact,
    set_global_seed,
    split_dataset,
    to_serializable,
)
from uncertainty import UncertaintyEstimator
from validator import summarize_validation_report as summarize_validation_report_payload


def load_json_artifact(path: Path) -> dict[str, Any]:
    """Load a JSON artifact from disk."""
    if not path.exists():
        raise FileNotFoundError(f"Required artifact not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def sample_search_parameter(trial: optuna.trial.Trial, name: str, spec: dict[str, Any]) -> Any:
    """Sample one hyperparameter from the configured search space."""
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


def save_figure(path: Path) -> None:
    """Save the active matplotlib figure."""
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()


def format_float(value: Any, digits: int = 4) -> str:
    """Render floats consistently for markdown tables."""
    if value is None:
        return "NA"
    if isinstance(value, str):
        return value
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        if np.isnan(float(value)):
            return "NA"
        return f"{float(value):.{digits}f}"
    return str(value)


def markdown_table(frame: pd.DataFrame, *, digits: int = 4) -> str:
    """Convert a DataFrame into a simple markdown table."""
    headers = [str(column) for column in frame.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in frame.to_dict(orient="records"):
        rendered = [str(format_float(row[column], digits=digits)).replace("|", "\\|") for column in headers]
        lines.append("| " + " | ".join(rendered) + " |")
    return "\n".join(lines)


def summarize_validation_report(validation_report: dict[str, Any]) -> dict[str, Any]:
    """Keep only compact validator fields needed for the benchmark artifacts."""
    summary = summarize_validation_report_payload(validation_report)
    summary["warn_reasons"] = list(validation_report.get("warn_reasons", []))
    return summary


def compute_range_thresholds(y_train: pd.Series) -> dict[str, float]:
    """Compute training-derived thresholds for low, mid, and high strength ranges."""
    return {
        "low_max": float(y_train.quantile(0.2)),
        "high_min": float(y_train.quantile(0.8)),
    }


def compute_rmse_by_range(
    y_true: pd.Series,
    y_pred: np.ndarray,
    thresholds: dict[str, float],
) -> dict[str, float]:
    """Compute RMSE for low, mid, and high strength groups using fixed thresholds."""
    low_max = float(thresholds["low_max"])
    high_min = float(thresholds["high_min"])
    masks = {
        "low": y_true <= low_max,
        "mid": (y_true > low_max) & (y_true < high_min),
        "high": y_true >= high_min,
    }
    rmse_by_range: dict[str, float] = {}
    for label, mask in masks.items():
        group_true = y_true.loc[mask]
        group_pred = np.asarray(y_pred)[mask.to_numpy()]
        rmse_by_range[label] = float(np.sqrt(np.mean((np.asarray(group_true) - group_pred) ** 2)))
    return rmse_by_range


def get_benchmark_model_configs(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Return benchmark model configurations that are enabled and importable."""
    benchmark_models = config.get("benchmark", {}).get("models", {})
    available_models: dict[str, dict[str, Any]] = {}
    for model_name, model_config in benchmark_models.items():
        if not model_config.get("enabled", False):
            continue
        if model_name == "XGBRegressor":
            try:
                from xgboost import XGBRegressor as _  # noqa: F401
            except ImportError:
                log_status("Skipping XGBRegressor because xgboost is not installed in the active venv.")
                continue
        if model_name == "LGBMRegressor":
            try:
                from lightgbm import LGBMRegressor as _  # noqa: F401
            except ImportError:
                log_status("Skipping LGBMRegressor because lightgbm is not installed in the active venv.")
                continue
        if model_name == "CatBoostRegressor":
            try:
                from catboost import CatBoostRegressor as _  # noqa: F401
            except ImportError:
                log_status("Skipping CatBoostRegressor because catboost is not installed in the active venv.")
                continue
        available_models[model_name] = model_config
    if not available_models:
        raise RuntimeError("No benchmark model families are available.")
    return available_models


def ensure_reference_artifacts(outputs_dir: Path, config: dict[str, Any]) -> dict[str, Any]:
    """Repair stale search artifacts before comparing against the reference run."""
    repaired_csv = False
    if csv_is_stale(outputs_dir, config):
        repair_optuna_results_csv(outputs_dir, config)
        repaired_csv = True
    sync_ok = bool(sync_check(outputs_dir, emit_warning=True))
    return {
        "repaired_csv": repaired_csv,
        "search_artifacts_in_sync": sync_ok,
    }


def load_reference_payload(outputs_dir: Path, config: dict[str, Any]) -> dict[str, Any]:
    """Load the current LightGBM reference run and its uncertainty artifact."""
    benchmark_config = config["benchmark"]
    reference_result_path = outputs_dir / Path(str(benchmark_config["reference_result_path"])).name
    reference_uncertainty_path = outputs_dir / Path(str(benchmark_config["reference_uncertainty_path"])).name
    reference_result = load_json_artifact(reference_result_path) if reference_result_path.exists() else {}
    baseline_result = load_json_artifact(outputs_dir / "baseline_metrics.json")
    final_metrics = load_json_artifact(outputs_dir / "final_metrics.json") if (outputs_dir / "final_metrics.json").exists() else {}
    finalized_best = final_metrics.get("best_search_metrics", {}) if isinstance(final_metrics, dict) else {}
    finalized_holdout = final_metrics.get("holdout_metrics", {}) if isinstance(final_metrics, dict) else {}
    reference_uncertainty = load_json_artifact(reference_uncertainty_path) if reference_uncertainty_path.exists() else {}

    reference_payload = {
        "baseline": {
            "model_name": baseline_result["model_name"],
            "holdout_rmse": baseline_result.get("holdout_rmse"),
            "holdout_mae": baseline_result.get("holdout_mae"),
            "holdout_r2": baseline_result.get("holdout_r2"),
            "holdout_composite": baseline_result.get("holdout_composite"),
        },
        "reference_run": {
            "model_name": reference_result.get("model_name", finalized_best.get("model_name")),
            "trial_number": reference_result.get(
                "trial_number",
                reference_result.get("best_trial", finalized_best.get("trial_number", finalized_best.get("best_trial"))),
            ),
            "holdout_rmse": reference_result.get("holdout_rmse", finalized_holdout.get("rmse")),
            "holdout_mae": reference_result.get("holdout_mae", finalized_holdout.get("mae")),
            "holdout_r2": reference_result.get("holdout_r2", finalized_holdout.get("r2")),
            "holdout_composite": reference_result.get("holdout_composite", finalized_holdout.get("composite_score")),
            "cv_rmse": reference_result.get("cv_rmse", finalized_best.get("cv_rmse")),
            "cv_mae": reference_result.get("cv_mae", finalized_best.get("cv_mae")),
            "cv_r2": reference_result.get("cv_r2", finalized_best.get("cv_r2")),
            "cv_composite": reference_result.get("cv_composite", finalized_best.get("cv_composite")),
            "validation_verdict": str(reference_result.get("validation_verdict", finalized_best.get("validation_verdict"))),
            "validation_pass_rate": reference_result.get("validation_pass_rate", finalized_best.get("validation_pass_rate")),
        },
        "reference_uncertainty": {
            "method": reference_uncertainty.get("method"),
            "coverage": reference_uncertainty.get("coverage"),
            "mean_interval_width": reference_uncertainty.get("mean_interval_width"),
            "global_status": reference_uncertainty.get("coverage_audit", {}).get("global_status"),
        },
    }
    return reference_payload


def run_tuning_study(
    model_name: str,
    model_config: dict[str, Any],
    x_train: pd.DataFrame,
    y_train: pd.Series,
    config: dict[str, Any],
    cv_splitter: Any,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Tune one model family and return the best parameters and trial log."""
    search_space = dict(model_config.get("search_space", {}))
    if not model_config.get("tune", True) or not search_space:
        return {}, []

    seed = int(config["experiment"]["random_seed"])
    trials_per_model = int(config["benchmark"]["trials_per_model"])
    trial_records: list[dict[str, Any]] = []

    def objective(trial: optuna.trial.Trial) -> float:
        params = {
            parameter_name: sample_search_parameter(trial, parameter_name, parameter_spec)
            for parameter_name, parameter_spec in search_space.items()
        }
        start_time = time.perf_counter()
        try:
            model = instantiate_model(model_name, params, config)
            cv_metrics = cross_validate_model(
                model,
                x_train,
                y_train,
                config,
                cv=cv_splitter,
                return_fold_metrics=False,
            )
            objective_value = float(cv_metrics["rmse"])
            error_message = None
        except Exception as exc:
            cv_metrics = {"rmse": float("inf"), "mae": float("inf"), "r2": float("-inf"), "composite_score": float("-inf")}
            objective_value = float("inf")
            error_message = str(exc)

        trial_records.append(
            {
                "model_name": model_name,
                "trial_number": int(trial.number),
                "status": "error" if error_message else "ok",
                "objective_rmse": objective_value,
                "cv_mae": float(cv_metrics["mae"]),
                "cv_r2": float(cv_metrics["r2"]),
                "cv_composite": float(cv_metrics["composite_score"]),
                "duration_seconds": float(time.perf_counter() - start_time),
                "hyperparameters": json.dumps(to_serializable(params), sort_keys=True),
                "error_message": error_message,
            }
        )
        return objective_value

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=seed),
    )
    study.optimize(objective, n_trials=trials_per_model, show_progress_bar=False)
    return dict(study.best_trial.params), trial_records


def evaluate_benchmark_model(
    model_name: str,
    display_name: str,
    params: dict[str, Any],
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_val: pd.DataFrame,
    y_val: pd.Series,
    config: dict[str, Any],
    validator: EngineeringValidator,
    cv_splitter: Any,
    range_thresholds: dict[str, float],
) -> tuple[Any, dict[str, Any]]:
    """Fit, validate, and summarize one benchmark model on the selection partition only."""
    cv_metrics = cross_validate_model(
        instantiate_model(model_name, params, config),
        x_train,
        y_train,
        config,
        cv=cv_splitter,
        return_fold_metrics=True,
    )

    model = instantiate_model(model_name, params, config)
    fit_start = time.perf_counter()
    model.fit(x_train, y_train)
    training_time_seconds = float(time.perf_counter() - fit_start)

    prediction_start = time.perf_counter()
    y_pred = np.asarray(model.predict(x_val), dtype=float)
    prediction_time_seconds = float(time.perf_counter() - prediction_start)

    repeated_prediction_start = time.perf_counter()
    repeat_count = 20
    for _ in range(repeat_count):
        model.predict(x_val)
    repeated_prediction_seconds = float(time.perf_counter() - repeated_prediction_start) / repeat_count

    validation_metrics = compute_regression_metrics(y_val, y_pred, config, float(y_train.mean()))
    validation_report = validator.validate_predictions(y_pred, x_val, y_val)
    validation_summary = summarize_validation_report(validation_report)
    range_metrics = compute_rmse_by_range(y_val, y_pred, range_thresholds)

    result = {
        "model_name": model_name,
        "display_name": display_name,
        "hyperparameters": copy.deepcopy(params),
        "cv_rmse": float(cv_metrics["rmse"]),
        "cv_rmse_std": float(cv_metrics.get("rmse_std", 0.0)),
        "cv_mae": float(cv_metrics["mae"]),
        "cv_mae_std": float(cv_metrics.get("mae_std", 0.0)),
        "cv_r2": float(cv_metrics["r2"]),
        "cv_r2_std": float(cv_metrics.get("r2_std", 0.0)),
        "cv_composite": float(cv_metrics["composite_score"]),
        "cv_composite_std": float(cv_metrics.get("composite_std", 0.0)),
        "fold_metrics": copy.deepcopy(cv_metrics.get("fold_metrics", {})),
        "selection_partition": "validation",
        "validation_rmse": float(validation_metrics["rmse"]),
        "validation_mae": float(validation_metrics["mae"]),
        "validation_r2": float(validation_metrics["r2"]),
        "validation_composite": float(validation_metrics["composite_score"]),
        "training_time_seconds": training_time_seconds,
        "prediction_time_seconds": prediction_time_seconds,
        "inference_time_ms_per_sample": float((repeated_prediction_seconds * 1000.0) / max(len(x_val), 1)),
        "validation_summary": validation_summary,
        "validation_verdict": validation_summary["verdict"],
        "validation_pass_rate": validation_summary["pass_rate"],
        "hard_failed_count": validation_summary["hard_failed_count"],
        "warning_count": validation_summary["warning_count"],
        "suspicious_count": validation_summary["suspicious_count"],
        "durability_caution_count": validation_summary["durability_caution_count"],
        "dataset_anomaly_count": validation_summary["dataset_anomaly_count"],
        "range_metrics": range_metrics,
        "range_low_rmse": float(range_metrics["low"]),
        "range_mid_rmse": float(range_metrics["mid"]),
        "range_high_rmse": float(range_metrics["high"]),
    }
    return model, result


def compare_against_reference(result: dict[str, Any], reference: dict[str, Any]) -> dict[str, Any]:
    """Attach deltas versus the current LightGBM reference run."""
    reference_run = reference["reference_run"]
    enriched = dict(result)
    enriched.update(
        {
            "delta_vs_reference_rmse": float(result["holdout_rmse"] - reference_run["holdout_rmse"]),
            "delta_vs_reference_mae": float(result["holdout_mae"] - reference_run["holdout_mae"]),
            "delta_vs_reference_r2": float(result["holdout_r2"] - reference_run["holdout_r2"]),
            "delta_vs_reference_composite": float(
                result["holdout_composite"] - reference_run["holdout_composite"]
            ),
            "beats_reference_rmse": bool(result["holdout_rmse"] < reference_run["holdout_rmse"]),
            "beats_reference_all_primary_metrics": bool(
                result["holdout_rmse"] < reference_run["holdout_rmse"]
                and result["holdout_mae"] < reference_run["holdout_mae"]
                and result["holdout_r2"] > reference_run["holdout_r2"]
            ),
        }
    )
    return enriched


def evaluate_final_holdout_winner(
    model_name: str,
    display_name: str,
    params: dict[str, Any],
    x_train_full: pd.DataFrame,
    y_train_full: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    config: dict[str, Any],
    validator: EngineeringValidator,
    range_thresholds: dict[str, float],
) -> tuple[Any, dict[str, Any]]:
    """Refit the selected winner on all non-holdout data and score the locked holdout once."""
    model = instantiate_model(model_name, params, config)
    fit_start = time.perf_counter()
    model.fit(x_train_full, y_train_full)
    training_time_seconds = float(time.perf_counter() - fit_start)

    prediction_start = time.perf_counter()
    y_pred = np.asarray(model.predict(x_test), dtype=float)
    prediction_time_seconds = float(time.perf_counter() - prediction_start)

    repeated_prediction_start = time.perf_counter()
    repeat_count = 20
    for _ in range(repeat_count):
        model.predict(x_test)
    repeated_prediction_seconds = float(time.perf_counter() - repeated_prediction_start) / repeat_count

    holdout_metrics = compute_regression_metrics(y_test, y_pred, config, float(y_train_full.mean()))
    holdout_validation_report = validator.validate_predictions(y_pred, x_test, y_test)
    holdout_validation_summary = summarize_validation_report(holdout_validation_report)
    range_metrics = compute_rmse_by_range(y_test, y_pred, range_thresholds)
    return model, {
        "model_name": model_name,
        "display_name": display_name,
        "hyperparameters": copy.deepcopy(params),
        "holdout_rmse": float(holdout_metrics["rmse"]),
        "holdout_mae": float(holdout_metrics["mae"]),
        "holdout_r2": float(holdout_metrics["r2"]),
        "holdout_composite": float(holdout_metrics["composite_score"]),
        "holdout_validation_verdict": holdout_validation_summary["verdict"],
        "holdout_validation_summary": holdout_validation_summary,
        "training_time_seconds": training_time_seconds,
        "prediction_time_seconds": prediction_time_seconds,
        "inference_time_ms_per_sample": float((repeated_prediction_seconds * 1000.0) / max(len(x_test), 1)),
        "range_metrics": range_metrics,
        "range_low_rmse": float(range_metrics["low"]),
        "range_mid_rmse": float(range_metrics["mid"]),
        "range_high_rmse": float(range_metrics["high"]),
    }


def rank_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Rank models using CV+validation selection metrics, never the locked holdout."""
    ranked = sorted(
        results,
        key=lambda item: (
            1 if str(item.get("validation_verdict", "")).upper() == "FAIL" else 0,
            -float(item["cv_composite"]),
            -float(item["validation_composite"]),
            float(item["cv_rmse_std"]),
            float(item["training_time_seconds"]),
        ),
    )
    for rank, result in enumerate(ranked, start=1):
        result["final_rank"] = rank
    return ranked


def create_selection_plot(results_frame: pd.DataFrame, output_path: Path) -> None:
    """Create a ranked validation-composite plot for model selection."""
    plot_frame = results_frame.sort_values("final_rank", ascending=True).copy()
    colors = ["#2a9d8f" if rank == 1 else "#457b9d" if name == "LGBMRegressor" else "#9aa5b1" for rank, name in zip(plot_frame["final_rank"], plot_frame["model_name"], strict=False)]
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.figure(figsize=(10, 6))
    plt.barh(plot_frame["display_name"], plot_frame["validation_composite"], color=colors)
    plt.xlabel("Validation Composite Score")
    plt.ylabel("Model")
    plt.title("Academic Benchmark: Selection Ranking by Validation Composite")
    plt.gca().invert_yaxis()
    save_figure(output_path)


def create_reference_delta_plot(results_frame: pd.DataFrame, output_path: Path) -> None:
    """Create delta-to-reference plots for the primary metrics."""
    plot_frame = results_frame.sort_values("delta_vs_reference_rmse", ascending=True).copy()
    labels = plot_frame["display_name"].tolist()
    plt.style.use("seaborn-v0_8-whitegrid")
    figure, axes = plt.subplots(2, 2, figsize=(12, 9))
    metric_specs = [
        ("delta_vs_reference_rmse", "Delta RMSE (MPa)", "#457b9d"),
        ("delta_vs_reference_mae", "Delta MAE (MPa)", "#2a9d8f"),
        ("delta_vs_reference_r2", "Delta R2", "#f4a261"),
        ("delta_vs_reference_composite", "Delta Composite", "#8d99ae"),
    ]
    for axis, (column_name, title, color) in zip(axes.flatten(), metric_specs, strict=False):
        axis.barh(labels, plot_frame[column_name], color=color)
        axis.axvline(0.0, linestyle="--", color="#e63946", linewidth=1.0)
        axis.set_title(title)
    save_figure(output_path)


def create_range_heatmap(range_frame: pd.DataFrame, output_path: Path) -> None:
    """Create a heatmap of strength-range RMSE by model."""
    heatmap_values = range_frame[["low_rmse", "mid_rmse", "high_rmse"]].to_numpy(dtype=float)
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.figure(figsize=(8, 6))
    image = plt.imshow(heatmap_values, aspect="auto", cmap="YlOrRd")
    plt.colorbar(image, label="RMSE (MPa)")
    plt.xticks([0, 1, 2], ["Low", "Mid", "High"])
    plt.yticks(range(len(range_frame)), range_frame["display_name"])
    plt.title("Academic Benchmark: RMSE by Strength Range")
    save_figure(output_path)


def build_summary_frame(results: list[dict[str, Any]]) -> pd.DataFrame:
    """Build the main model-selection comparison table."""
    rows = []
    for result in results:
        rows.append(
            {
                "rank": int(result["final_rank"]),
                "final_rank": int(result["final_rank"]),
                "model": result["display_name"],
                "display_name": result["display_name"],
                "family_id": result["model_name"],
                "model_name": result["model_name"],
                "cv_rmse": float(result["cv_rmse"]),
                "cv_rmse_std": float(result["cv_rmse_std"]),
                "cv_mae": float(result["cv_mae"]),
                "cv_mae_std": float(result["cv_mae_std"]),
                "cv_r2": float(result["cv_r2"]),
                "cv_r2_std": float(result["cv_r2_std"]),
                "cv_composite": float(result["cv_composite"]),
                "validation_rmse": float(result["validation_rmse"]),
                "validation_mae": float(result["validation_mae"]),
                "validation_r2": float(result["validation_r2"]),
                "validation_composite": float(result["validation_composite"]),
                "validation_verdict": result["validation_verdict"],
                "validation_pass_rate": float(result["validation_pass_rate"]),
                "hard_failed_count": int(result["hard_failed_count"]),
                "warning_count": int(result["warning_count"]),
                "training_time_seconds": float(result["training_time_seconds"]),
                "inference_time_ms_per_sample": float(result["inference_time_ms_per_sample"]),
                "hyperparameters": json.dumps(to_serializable(result["hyperparameters"]), sort_keys=True),
            }
        )
    return pd.DataFrame(rows)


def build_range_frame(results: list[dict[str, Any]]) -> pd.DataFrame:
    """Build the strength-range RMSE table."""
    rows = []
    for result in results:
        range_metrics = result["range_metrics"]
        hardest_range = max(range_metrics.items(), key=lambda item: item[1])[0]
        rows.append(
            {
                "rank": int(result["final_rank"]),
                "display_name": result["display_name"],
                "low_rmse": float(range_metrics["low"]),
                "mid_rmse": float(range_metrics["mid"]),
                "high_rmse": float(range_metrics["high"]),
                "hardest_range": hardest_range,
            }
        )
    return pd.DataFrame(rows)


def build_trial_frame(trial_records: list[dict[str, Any]]) -> pd.DataFrame:
    """Build the Optuna trial table for all benchmark models."""
    if not trial_records:
        return pd.DataFrame(
            columns=[
                "model_name",
                "trial_number",
                "status",
                "objective_rmse",
                "cv_mae",
                "cv_r2",
                "cv_composite",
                "duration_seconds",
                "hyperparameters",
                "error_message",
            ]
        )
    return pd.DataFrame(trial_records)


def build_uncertainty_summary(
    outputs_dir: Path,
    config: dict[str, Any],
    best_model_path: Path,
    best_result: dict[str, Any],
    reference_payload: dict[str, Any],
) -> dict[str, Any]:
    """Run conformal uncertainty for the benchmark winner on the final holdout audit."""
    estimator = UncertaintyEstimator(
        model_path=best_model_path,
        method=str(config["benchmark"]["uncertainty_method"]),
        outputs_dir=outputs_dir,
        report_filename="benchmark_uncertainty_calibration.json",
        audit_partition="holdout",
    )
    winner_uncertainty = estimator.calibration_report()
    reference_uncertainty = reference_payload.get("reference_uncertainty", {})
    return {
        "winner_model_name": best_result["model_name"],
        "winner_display_name": best_result["display_name"],
        "winner_uncertainty": {
            "method": winner_uncertainty.get("method"),
            "coverage": float(winner_uncertainty.get("coverage", 0.0)),
            "mean_interval_width": float(winner_uncertainty.get("mean_interval_width", 0.0)),
            "global_status": winner_uncertainty.get("coverage_audit", {}).get("global_status"),
        },
        "reference_uncertainty": copy.deepcopy(reference_uncertainty),
        "delta_coverage_vs_reference": float(
            float(winner_uncertainty.get("coverage", 0.0)) - float(reference_uncertainty.get("coverage", 0.0) or 0.0)
        ),
        "delta_width_vs_reference": float(
            float(winner_uncertainty.get("mean_interval_width", 0.0))
            - float(reference_uncertainty.get("mean_interval_width", 0.0) or 0.0)
        ),
    }


def write_markdown_report(
    output_path: Path,
    reference_payload: dict[str, Any],
    artifact_health: dict[str, Any],
    summary_frame: pd.DataFrame,
    range_frame: pd.DataFrame,
    uncertainty_summary: dict[str, Any],
    best_result: dict[str, Any],
) -> None:
    """Write the paper-ready benchmark comparison markdown."""
    top_row = summary_frame.sort_values("rank", ascending=True).iloc[0]
    lightgbm_is_best = str(top_row["family_id"]) == "LGBMRegressor"
    reference_run = reference_payload["reference_run"]
    baseline_reference = reference_payload["baseline"]
    uncertainty_reference = reference_payload["reference_uncertainty"]

    executive_table = pd.DataFrame(
        [
            {
                "Artifact": "Baseline RF reference",
                "RMSE": baseline_reference["holdout_rmse"],
                "MAE": baseline_reference["holdout_mae"],
                "R2": baseline_reference["holdout_r2"],
                "Composite": baseline_reference["holdout_composite"],
            },
            {
                "Artifact": "Current LightGBM reference",
                "RMSE": reference_run["holdout_rmse"],
                "MAE": reference_run["holdout_mae"],
                "R2": reference_run["holdout_r2"],
                "Composite": reference_run["holdout_composite"],
            },
            {
                "Artifact": f"Benchmark winner ({best_result['display_name']})",
                "RMSE": best_result["holdout_rmse"],
                "MAE": best_result["holdout_mae"],
                "R2": best_result["holdout_r2"],
                "Composite": best_result["holdout_composite"],
            },
        ]
    )

    cv_table = summary_frame[
        ["rank", "model", "cv_rmse", "cv_rmse_std", "cv_mae", "cv_mae_std", "cv_r2", "cv_r2_std", "cv_composite"]
    ].copy()
    selection_table = summary_frame[
        [
            "rank",
            "model",
            "cv_composite",
            "validation_composite",
            "validation_rmse",
            "validation_mae",
            "validation_r2",
            "validation_verdict",
        ]
    ].copy()
    uncertainty_table = pd.DataFrame(
        [
            {
                "Model": "Current LightGBM reference",
                "Coverage": uncertainty_reference.get("coverage"),
                "Mean interval width": uncertainty_reference.get("mean_interval_width"),
                "Global status": uncertainty_reference.get("global_status"),
            },
            {
                "Model": f"Benchmark winner ({best_result['display_name']})",
                "Coverage": uncertainty_summary["winner_uncertainty"]["coverage"],
                "Mean interval width": uncertainty_summary["winner_uncertainty"]["mean_interval_width"],
                "Global status": uncertainty_summary["winner_uncertainty"]["global_status"],
            },
        ]
    )

    lightgbm_note = "LightGBM remains the best-performing family in the updated academic benchmark." if lightgbm_is_best else (
        f"LightGBM no longer ranks first; the updated benchmark winner is {best_result['display_name']}."
    )

    report_lines = [
        "# Academic Benchmark Comparison Against the Current LightGBM Reference",
        "",
        "## Summary",
        "",
        f"The benchmark ranked candidate model families using cross-validation composite score with validation-composite tie-breaking, then evaluated only the selected winner on the locked holdout. The current finalized LightGBM reference run remains the external holdout comparator (trial `{reference_run['trial_number']}`).",
        "",
        f"- Current baseline RF reference: RMSE `{format_float(baseline_reference['holdout_rmse'])}` MPa, MAE `{format_float(baseline_reference['holdout_mae'])}` MPa, R2 `{format_float(baseline_reference['holdout_r2'])}`, composite `{format_float(baseline_reference['holdout_composite'])}`.",
        f"- Current LightGBM reference: RMSE `{format_float(reference_run['holdout_rmse'])}` MPa, MAE `{format_float(reference_run['holdout_mae'])}` MPa, R2 `{format_float(reference_run['holdout_r2'])}`, composite `{format_float(reference_run['holdout_composite'])}`.",
        f"- Benchmark winner: `{best_result['display_name']}` with RMSE `{format_float(best_result['holdout_rmse'])}` MPa, MAE `{format_float(best_result['holdout_mae'])}` MPa, R2 `{format_float(best_result['holdout_r2'])}`, composite `{format_float(best_result['holdout_composite'])}`.",
        f"- Artifact consistency check: search artifacts in sync = `{artifact_health['search_artifacts_in_sync']}`, repair applied = `{artifact_health['repaired_csv']}`.",
        "",
        f"**Conclusion:** {lightgbm_note}",
        "",
        "## Executive Comparison",
        "",
        markdown_table(executive_table),
        "",
        "## Selection Ranking",
        "",
        markdown_table(selection_table),
        "",
        "## Cross-Validation Stability",
        "",
        markdown_table(cv_table),
        "",
        "## Strength-Range RMSE",
        "",
        markdown_table(range_frame.rename(columns={"display_name": "model"})),
        "",
        "## Uncertainty Comparison",
        "",
        markdown_table(uncertainty_table),
        "",
        "## Recommendation",
        "",
        f"The benchmark winner is `{best_result['display_name']}`. "
        f"This model {'improves upon' if bool(best_result['beats_reference_rmse']) else 'does not improve upon'} "
        f"the current LightGBM reference in holdout RMSE. "
        f"The benchmark should therefore {'replace' if bool(best_result['beats_reference_rmse']) else 'retain'} "
        f"the current LightGBM reference as the primary model recommendation.",
        "",
    ]
    output_path.write_text("\n".join(line for line in report_lines if line is not None), encoding="utf-8")


def main() -> int:
    """Run the academic benchmark and save comparison artifacts."""
    try:
        config = load_config()
        set_global_seed(int(config["experiment"]["random_seed"]))
        outputs_dir = get_outputs_dir(config)

        artifact_health = ensure_reference_artifacts(outputs_dir, config)
        reference_payload = load_reference_payload(outputs_dir, config)

        dataset = load_dataset(config)
        # x_test is the locked holdout. It is reserved for one-time final
        # reporting after winner selection is complete.
        x_train, x_val, x_test, y_train, y_val, y_test = split_dataset(dataset, config)
        # Integrity check: confirm holdout is separate from the training partition.
        assert len(set(x_test.index) & set(x_train.index)) == 0, (
            "Holdout leakage: x_test and x_train share indices."
        )
        assert len(set(x_test.index) & set(x_val.index)) == 0, (
            "Holdout leakage: x_test and x_val share indices."
        )
        validator = EngineeringValidator.from_config(config)
        benchmark_config = config["benchmark"]
        cv_splitter = build_cv_splitter(
            config,
            n_splits=int(benchmark_config["cv_folds"]),
            repeats=int(benchmark_config.get("cv_repeats", 1)),
            random_state=int(config["experiment"]["random_seed"]),
        )
        range_thresholds = compute_range_thresholds(y_train)
        benchmark_models = get_benchmark_model_configs(config)

        all_results: list[dict[str, Any]] = []
        all_trial_records: list[dict[str, Any]] = []

        for model_name, model_config in benchmark_models.items():
            display_name = str(model_config.get("display_name", model_name))
            log_status(f"Benchmarking {display_name}...")
            best_params, trial_records = run_tuning_study(
                model_name,
                model_config,
                x_train,
                y_train,
                config,
                cv_splitter,
            )
            all_trial_records.extend(trial_records)
            model, result = evaluate_benchmark_model(
                model_name,
                display_name,
                best_params,
                x_train,
                y_train,
                x_val,
                y_val,
                config,
                validator,
                cv_splitter,
                range_thresholds,
            )
            all_results.append(result)
            log_status(
                f"Completed {display_name} | cv_composite={result['cv_composite']:.4f} | "
                f"validation_composite={result['validation_composite']:.4f}"
            )

        ranked_results = rank_results(all_results)
        selection_winner = ranked_results[0]
        x_train_full = pd.concat([x_train, x_val], axis=0)
        y_train_full = pd.concat([y_train, y_val], axis=0)
        best_model, holdout_result = evaluate_final_holdout_winner(
            selection_winner["model_name"],
            selection_winner["display_name"],
            dict(selection_winner["hyperparameters"]),
            x_train_full,
            y_train_full,
            x_test,
            y_test,
            config,
            validator,
            range_thresholds,
        )
        best_result = compare_against_reference(
            {
                **selection_winner,
                **holdout_result,
                "selected_by": "cv_composite_then_validation_composite",
            },
            reference_payload,
        )

        best_model_path = outputs_dir / "benchmark_best_model.pkl"
        best_result_path = outputs_dir / "benchmark_best_result.json"
        save_pickle_artifact(
            best_model_path,
            best_model,
            config=config,
            model_id=best_result["model_name"],
        )
        save_json_artifact(best_result_path, best_result)

        summary_frame = build_summary_frame(ranked_results)
        range_frame = build_range_frame(ranked_results)
        trial_frame = build_trial_frame(all_trial_records)

        summary_frame.to_csv(outputs_dir / "benchmark_results.csv", index=False)
        range_frame.to_csv(outputs_dir / "benchmark_range_metrics.csv", index=False)
        trial_frame.to_csv(outputs_dir / "benchmark_trials.csv", index=False)

        uncertainty_summary = build_uncertainty_summary(
            outputs_dir,
            config,
            best_model_path,
            best_result,
            reference_payload,
        )
        save_json_artifact(outputs_dir / "benchmark_uncertainty.json", uncertainty_summary)

        create_selection_plot(
            summary_frame,
            outputs_dir / "benchmark_selection_composite.png",
        )
        create_range_heatmap(range_frame, outputs_dir / "benchmark_strength_range_heatmap.png")

        benchmark_payload = {
            "artifact_health": artifact_health,
            "reference_payload": reference_payload,
            "range_thresholds": range_thresholds,
            "selection_winner": selection_winner,
            "best_result": best_result,
            "uncertainty_summary": uncertainty_summary,
            "results": ranked_results,
        }
        save_json_artifact(outputs_dir / "benchmark_results.json", benchmark_payload)

        write_markdown_report(
            outputs_dir / "benchmark_comparison.md",
            reference_payload,
            artifact_health,
            summary_frame,
            range_frame,
            uncertainty_summary,
            best_result,
        )

        log_status(
            f"Benchmark complete. Winner={best_result['display_name']} | "
            f"Holdout RMSE={best_result['holdout_rmse']:.4f} | "
            f"Reference RMSE={format_float(reference_payload['reference_run']['holdout_rmse'])}"
        )
        return 0
    except Exception as exc:
        log_status(f"Benchmark failed: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
