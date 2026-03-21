"""Reporting utilities for AutoCivil-Lab."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from train import (
    artifact_id,
    artifact_run_id,
    compute_regression_metrics,
    EngineeringValidator,
    get_outputs_dir,
    infer_active_run_id,
    load_pickle_artifact,
    load_config,
    load_dataset,
    log_status,
    mark_json_artifact_stale,
    resolve_model_feature_columns,
    set_global_seed,
    split_dataset,
    write_run_scoped_json_artifact,
)
from uncertainty import UncertaintyEstimator
from validator import summarize_validation_report


REPORT_CONTEXT = SimpleNamespace()


def load_json_artifact(path: Path) -> dict[str, Any]:
    """Load a JSON artifact from disk."""
    if not path.exists():
        raise FileNotFoundError(f"Required artifact not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


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
    feature_names = resolve_model_feature_columns(model, config)
    model_name = str(best_search_result["model_name"])

    if model_name in {"SVR", "Ridge", "LinearRegression", "ElasticNet", "KNeighborsRegressor"}:
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
        "HIGH": "#2a9d8f",
        "MODERATE": "#e9c46a",
        "LOW": "#e76f51",
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


def confidence_labels_from_coverage(bin_coverages: np.ndarray) -> np.ndarray:
    """Map observed bin coverage rates to human-readable confidence labels."""
    return np.where(
        bin_coverages > 0.92,
        "HIGH",
        np.where(bin_coverages >= 0.85, "MODERATE", "LOW"),
    )


def main() -> int:
    """Generate the final plots and consolidated metrics report."""
    try:
        config = load_config()
        set_global_seed(int(config["experiment"]["random_seed"]))
        outputs_dir = get_outputs_dir(config)

        baseline_metrics = load_json_artifact(outputs_dir / "baseline_metrics.json")
        baseline_model = load_pickle_artifact(outputs_dir / "baseline_model.pkl")
        existing_final_metrics = (
            load_json_artifact(outputs_dir / "final_metrics.json")
            if (outputs_dir / "final_metrics.json").exists()
            else {}
        )
        best_search_result = load_json_artifact(outputs_dir / "best_search_result.json")
        active_run_id = infer_active_run_id(outputs_dir, best_search_result, existing_final_metrics)
        if existing_final_metrics and artifact_run_id(existing_final_metrics) not in {None, active_run_id}:
            log_status(
                "WARNING stale_final_metrics_ignored | "
                f"final_metrics_run_id={artifact_run_id(existing_final_metrics)} | active_run_id={active_run_id}"
            )
            mark_json_artifact_stale(
                outputs_dir / "final_metrics.json",
                active_run_id=active_run_id,
                reason="stale final metrics ignored during report generation",
            )
            existing_final_metrics = {}
        mark_json_artifact_stale(
            outputs_dir / "search_state_best_result.json",
            active_run_id=active_run_id,
            reason="stale intermediate search-state artifact from a different run",
        )
        mark_json_artifact_stale(
            outputs_dir / "ensemble_metrics.json",
            active_run_id=active_run_id,
            reason="stale ensemble artifact from a different run",
        )
        best_model = load_pickle_artifact(outputs_dir / "best_search_model.pkl")
        optuna_results = pd.read_csv(outputs_dir / "optuna_results.csv")

        data = load_dataset(config)
        x_train, _, x_test, y_train, _, y_test = split_dataset(data, config)
        ctx = REPORT_CONTEXT
        assert getattr(ctx, "_test_set_used_in_report", False) is False, (
            "x_test has already been used. Holdout integrity violated."
        )
        ctx._test_set_used_in_report = True
        validator = EngineeringValidator.from_config(config)
        baseline_feature_columns = resolve_model_feature_columns(baseline_model, config)
        best_feature_columns = resolve_model_feature_columns(best_model, config)
        baseline_features = (
            x_test if list(x_test.columns) == baseline_feature_columns else x_test[baseline_feature_columns]
        )
        best_features = x_test if list(x_test.columns) == best_feature_columns else x_test[best_feature_columns]
        baseline_pred = np.asarray(baseline_model.predict(baseline_features), dtype=float)
        y_pred = np.asarray(best_model.predict(best_features), dtype=float)
        holdout_metrics = compute_regression_metrics(y_test, y_pred, config, float(y_train.mean()))
        validation_report = validator.validate_model(best_model, best_features, y_test)
        validation_result = dict(best_search_result)
        validation_result["validation_verdict"] = validation_report["verdict"]
        validation_result["validation_report"] = validation_report
        baseline_rmse_by_range = compute_rmse_by_range(y_test, baseline_pred)

        create_search_progress_plot(optuna_results, float(baseline_metrics["composite_score"]), outputs_dir)
        create_actual_vs_predicted_plot(y_test, y_pred, outputs_dir)
        create_residuals_plot(y_test, y_pred, outputs_dir)
        feature_importance = compute_feature_importance(
            best_model,
            best_features,
            y_test,
            config,
            best_search_result,
        )
        create_feature_importance_plot(feature_importance, outputs_dir)
        rmse_by_range = create_performance_by_range_plot(y_test, y_pred, outputs_dir)
        uncertainty_estimator = UncertaintyEstimator(
            model=best_model,
            report_model=best_model,
            method=str(config["engineering"]["uncertainty_method"]),
            outputs_dir=outputs_dir,
        )
        uncertainty_audit = uncertainty_estimator.calibration_report()
        interval_frame = uncertainty_estimator.predict_with_interval(best_features)
        audit_coverages = {
            int(row["bin_id"]): float(row["observed_coverage"])
            for row in uncertainty_audit.get("reliability_plot_data", [])
        }
        if audit_coverages:
            interval_frame["bin_coverage"] = interval_frame["strength_bin"].map(audit_coverages).astype(float)
            interval_frame["confidence_label"] = confidence_labels_from_coverage(
                interval_frame["bin_coverage"].to_numpy(dtype=float)
            )
        uncertainty_summary = create_uncertainty_plot(interval_frame, outputs_dir)
        uncertainty_summary["coverage"] = float(uncertainty_audit["coverage"])
        uncertainty_summary["coverage_target"] = float(uncertainty_audit["coverage_target"])
        uncertainty_summary["coverage_audit"] = uncertainty_audit.get("coverage_audit", {})

        improvement_percentage = calculate_improvement_percentage(
            float(baseline_metrics["composite_score"]),
            float(best_search_result["composite_score"]),
        )
        validation_summary = summarize_validation_report(validation_report)
        final_metrics = dict(existing_final_metrics) if isinstance(existing_final_metrics, dict) else {}
        final_metrics.update(
            {
                "baseline_metrics": baseline_metrics,
                "best_search_metrics": validation_result,
                "improvement_percentage": improvement_percentage,
                "composite_improvement_pct": improvement_percentage,
                "validation_verdict": validation_result["validation_verdict"],
                "best_model_name": validation_result["model_name"],
                "best_model_hyperparameters": validation_result["hyperparameters"],
                "validation_summary": validation_summary,
                "holdout_metrics": holdout_metrics,
                "rmse_by_range": rmse_by_range,
                "baseline_rmse_by_range": baseline_rmse_by_range,
                "uncertainty_summary": uncertainty_summary,
                "uncertainty_audit": uncertainty_audit.get("coverage_audit", {}),
            }
        )
        write_run_scoped_json_artifact(
            outputs_dir=outputs_dir,
            filename="final_metrics.json",
            payload=final_metrics,
            run_id=active_run_id,
            source_mode="report",
            config=config,
            model_artifact_id=best_search_result.get("model_artifact_id"),
            model_id=str(best_search_result["model_name"]),
            parent_artifact_ids=[
                parent_id
                for parent_id in (artifact_id(best_search_result), artifact_id(uncertainty_audit))
                if parent_id
            ],
        )

        log_status(
            f"Final report ready. Best model={validation_result['model_name']} | "
            f"Composite={validation_result['composite_score']:.4f} | "
            f"Improvement={improvement_percentage:.2f}% | "
            f"Validation={validation_result['validation_verdict']} | "
            f"HardConstraints={validation_summary['hard_constraint_count']} | "
            f"EngineeringCautions={validation_summary['engineering_caution_count']} | "
            f"DataReviewFlags={validation_summary['data_review_flag_count']}"
        )
        return 0
    except Exception as exc:
        log_status(f"Report generation failed: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
