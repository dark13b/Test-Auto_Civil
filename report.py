"""Reporting utilities for AutoCivil-Lab."""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from train import (
    compute_regression_metrics,
    get_input_columns,
    get_outputs_dir,
    load_config,
    load_dataset,
    log_status,
    save_json_artifact,
    set_global_seed,
    split_dataset,
)
from uncertainty import UncertaintyEstimator


def load_json_artifact(path: Path) -> dict[str, Any]:
    """Load a JSON artifact from disk."""
    if not path.exists():
        raise FileNotFoundError(f"Required artifact not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_pickle_artifact(path: Path) -> Any:
    """Load a pickle artifact from disk."""
    if not path.exists():
        raise FileNotFoundError(f"Required artifact not found: {path}")
    with path.open("rb") as handle:
        return pickle.load(handle)


def save_figure(path: Path) -> None:
    """Save the current matplotlib figure and close it."""
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()


def create_search_progress_plot(
    trials_frame: pd.DataFrame,
    baseline_score: float,
    outputs_dir: Path,
) -> None:
    """Create the search progress plot across all trials."""
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.figure(figsize=(10, 5))
    valid_scores = trials_frame.dropna(subset=["composite_score"]).copy()
    if not valid_scores.empty:
        plt.plot(
            valid_scores["trial_number"],
            valid_scores["composite_score"],
            marker="o",
            linewidth=1.5,
            label="Trial composite score",
        )
        best_row = valid_scores.loc[valid_scores["composite_score"].idxmax()]
        plt.scatter(
            [best_row["trial_number"]],
            [best_row["composite_score"]],
            color="red",
            s=80,
            label="Best trial",
            zorder=5,
        )
    plt.axhline(baseline_score, linestyle="--", color="black", label="Baseline composite")
    plt.xlabel("Trial Number")
    plt.ylabel("Composite Score")
    plt.title("Search Progress")
    plt.legend()
    save_figure(outputs_dir / "search_progress.png")


def create_actual_vs_predicted_plot(
    y_true: pd.Series,
    y_pred: np.ndarray,
    outputs_dir: Path,
) -> None:
    """Create the actual vs predicted scatter plot."""
    errors = np.abs(np.asarray(y_true) - y_pred)
    min_axis = min(float(np.min(y_true)), float(np.min(y_pred)))
    max_axis = max(float(np.max(y_true)), float(np.max(y_pred)))

    plt.style.use("seaborn-v0_8-whitegrid")
    plt.figure(figsize=(7, 6))
    scatter = plt.scatter(y_true, y_pred, c=errors, cmap="viridis", alpha=0.8)
    plt.plot([min_axis, max_axis], [min_axis, max_axis], linestyle="--", color="red", linewidth=1.3)
    plt.xlabel("Actual Strength (MPa)")
    plt.ylabel("Predicted Strength (MPa)")
    plt.title("Actual vs Predicted")
    colorbar = plt.colorbar(scatter)
    colorbar.set_label("Absolute Error (MPa)")
    save_figure(outputs_dir / "actual_vs_predicted.png")


def create_residuals_plot(
    y_true: pd.Series,
    y_pred: np.ndarray,
    outputs_dir: Path,
) -> None:
    """Create the residuals vs predicted plot."""
    residuals = np.asarray(y_true) - y_pred
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.figure(figsize=(7, 5))
    plt.scatter(y_pred, residuals, alpha=0.75, color="#2a6f97")
    plt.axhline(0.0, linestyle="--", color="red", linewidth=1.2)
    plt.xlabel("Predicted Strength (MPa)")
    plt.ylabel("Residual (Actual - Predicted)")
    plt.title("Residuals vs Predicted")
    save_figure(outputs_dir / "residuals_plot.png")


def compute_feature_importance(
    model: Any,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    config: dict[str, Any],
    best_search_result: dict[str, Any],
) -> pd.Series:
    """Compute feature importance, using permutation importance when required."""
    feature_names = get_input_columns(config)
    model_name = str(best_search_result["model_name"])

    if model_name in {"SVR", "Ridge"}:
        importance = permutation_importance(
            model,
            x_test,
            y_test,
            n_repeats=15,
            random_state=int(config["experiment"]["random_seed"]),
            scoring="neg_root_mean_squared_error",
        )
        return pd.Series(importance.importances_mean, index=feature_names).sort_values()

    if hasattr(model, "feature_importances_"):
        return pd.Series(model.feature_importances_, index=feature_names).sort_values()

    if hasattr(model, "named_steps") and "model" in model.named_steps:
        final_step = model.named_steps["model"]
        if hasattr(final_step, "feature_importances_"):
            return pd.Series(final_step.feature_importances_, index=feature_names).sort_values()

    importance = permutation_importance(
        model,
        x_test,
        y_test,
        n_repeats=15,
        random_state=int(config["experiment"]["random_seed"]),
        scoring="neg_root_mean_squared_error",
    )
    return pd.Series(importance.importances_mean, index=feature_names).sort_values()


def create_feature_importance_plot(feature_importance: pd.Series, outputs_dir: Path) -> None:
    """Create the horizontal feature importance plot."""
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.figure(figsize=(8, 5))
    plt.barh(feature_importance.index, feature_importance.values, color="#6c8f5d")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.title("Feature Importance")
    save_figure(outputs_dir / "feature_importance.png")


def compute_rmse_by_range(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    """Compute RMSE across low, mid, and high target ranges without plotting."""
    lower_bound = float(y_true.quantile(0.2))
    upper_bound = float(y_true.quantile(0.8))
    ranges = {
        "low": y_true <= lower_bound,
        "mid": (y_true > lower_bound) & (y_true < upper_bound),
        "high": y_true >= upper_bound,
    }
    rmse_by_range: dict[str, float] = {}
    for label, mask in ranges.items():
        group_true = y_true[mask]
        group_pred = y_pred[mask]
        rmse_by_range[label] = float(np.sqrt(np.mean((np.asarray(group_true) - np.asarray(group_pred)) ** 2)))
    return rmse_by_range


def create_performance_by_range_plot(
    y_true: pd.Series,
    y_pred: np.ndarray,
    outputs_dir: Path,
) -> dict[str, float]:
    """Create a plot showing RMSE across low, mid, and high target ranges."""
    rmse_by_range = compute_rmse_by_range(y_true, y_pred)
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.figure(figsize=(7, 5))
    plt.bar(rmse_by_range.keys(), rmse_by_range.values(), color=["#9ec1a3", "#f4b860", "#d96c75"])
    plt.xlabel("Target Range")
    plt.ylabel("RMSE (MPa)")
    plt.title("Performance by Strength Range")
    save_figure(outputs_dir / "performance_by_range.png")
    return rmse_by_range


def create_uncertainty_plot(interval_frame: pd.DataFrame, outputs_dir: Path) -> dict[str, Any]:
    """Create the interval-width plot across the predicted strength range."""
    color_map = {
        "TIGHT": "#2a9d8f",
        "MODERATE": "#e9c46a",
        "WIDE": "#e76f51",
    }
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.figure(figsize=(8, 5))
    for label, color in color_map.items():
        mask = interval_frame["confidence_label"] == label
        plt.scatter(
            interval_frame.loc[mask, "predicted"],
            interval_frame.loc[mask, "interval_width"],
            label=label,
            alpha=0.8,
            color=color,
        )
    plt.xlabel("Predicted Strength (MPa)")
    plt.ylabel("Interval Width (MPa)")
    plt.title("Prediction Uncertainty by Strength Range")
    plt.legend()
    save_figure(outputs_dir / "uncertainty_plot.png")
    return {
        "mean_interval_width": float(interval_frame["interval_width"].mean()),
        "label_counts": interval_frame["confidence_label"].value_counts().to_dict(),
    }


def calculate_improvement_percentage(baseline_score: float, best_score: float) -> float:
    """Calculate percentage improvement in composite score relative to baseline."""
    denominator = max(abs(baseline_score), 1e-8)
    return ((best_score - baseline_score) / denominator) * 100.0


def main() -> int:
    """Generate the final plots and consolidated metrics report."""
    try:
        config = load_config()
        set_global_seed(int(config["experiment"]["random_seed"]))
        outputs_dir = get_outputs_dir(config)

        baseline_metrics = load_json_artifact(outputs_dir / "baseline_metrics.json")
        baseline_model = load_pickle_artifact(outputs_dir / "baseline_model.pkl")
        best_search_result = load_json_artifact(outputs_dir / "best_search_result.json")
        best_model = load_pickle_artifact(outputs_dir / "best_search_model.pkl")
        optuna_results = pd.read_csv(outputs_dir / "optuna_results.csv")

        data = load_dataset(config)
        x_train, x_test, y_train, y_test = split_dataset(data, config)
        baseline_pred = np.asarray(baseline_model.predict(x_test), dtype=float)
        y_pred = np.asarray(best_model.predict(x_test), dtype=float)
        holdout_metrics = compute_regression_metrics(y_test, y_pred, config, float(y_train.mean()))
        baseline_rmse_by_range = compute_rmse_by_range(y_test, baseline_pred)

        create_search_progress_plot(optuna_results, float(baseline_metrics["composite_score"]), outputs_dir)
        create_actual_vs_predicted_plot(y_test, y_pred, outputs_dir)
        create_residuals_plot(y_test, y_pred, outputs_dir)
        feature_importance = compute_feature_importance(best_model, x_test, y_test, config, best_search_result)
        create_feature_importance_plot(feature_importance, outputs_dir)
        rmse_by_range = create_performance_by_range_plot(y_test, y_pred, outputs_dir)
        uncertainty_estimator = UncertaintyEstimator(method=str(config["engineering"]["uncertainty_method"]))
        interval_frame = uncertainty_estimator.predict_with_interval(x_test)
        uncertainty_summary = create_uncertainty_plot(interval_frame, outputs_dir)

        improvement_percentage = calculate_improvement_percentage(
            float(baseline_metrics["composite_score"]),
            float(best_search_result["composite_score"]),
        )
        final_metrics = {
            "baseline_metrics": baseline_metrics,
            "best_search_metrics": best_search_result,
            "improvement_percentage": improvement_percentage,
            "composite_improvement_pct": improvement_percentage,
            "validation_verdict": best_search_result["validation_verdict"],
            "best_model_name": best_search_result["model_name"],
            "best_model_hyperparameters": best_search_result["hyperparameters"],
            "holdout_metrics": holdout_metrics,
            "rmse_by_range": rmse_by_range,
            "baseline_rmse_by_range": baseline_rmse_by_range,
            "uncertainty_summary": uncertainty_summary,
        }
        save_json_artifact(outputs_dir / "final_metrics.json", final_metrics)

        log_status(
            f"Final report ready. Best model={best_search_result['model_name']} | "
            f"Composite={best_search_result['composite_score']:.4f} | "
            f"Improvement={improvement_percentage:.2f}% | "
            f"Validation={best_search_result['validation_verdict']}"
        )
        return 0
    except Exception as exc:
        log_status(f"Report generation failed: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
