"""Baseline training utilities and entry point for AutoCivil-Lab."""

from __future__ import annotations

import json
import os
import pickle
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from feature_engineering import ENGINEERED_FEATURE_COLUMNS, build_engineering_features
from validator import EngineeringValidator

try:
    from lightgbm import LGBMRegressor
except ImportError:
    LGBMRegressor = None

try:
    from xgboost import XGBRegressor
except ImportError:
    XGBRegressor = None


def log_status(message: str) -> None:
    """Print a timestamped status message."""
    timestamp = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")


def get_project_root() -> Path:
    """Return the project root directory."""
    return Path(__file__).resolve().parent


def load_config() -> dict[str, Any]:
    """Load the YAML project configuration."""
    config_path = get_project_root() / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("Configuration file did not parse into a dictionary.")
    return config


def set_global_seed(seed: int) -> None:
    """Set Python, NumPy, and hash seeds."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def get_data_path(config: dict[str, Any]) -> Path:
    """Return the configured dataset path."""
    return get_project_root() / config["paths"]["data_dir"] / config["paths"]["dataset_filename"]


def get_outputs_dir(config: dict[str, Any]) -> Path:
    """Return the configured outputs directory, creating it if needed."""
    outputs_dir = get_project_root() / config["paths"]["outputs_dir"]
    outputs_dir.mkdir(parents=True, exist_ok=True)
    return outputs_dir


def get_input_columns(config: dict[str, Any]) -> list[str]:
    """Return the configured feature columns."""
    feature_columns = get_base_input_columns(config)
    if feature_engineering_enabled(config):
        feature_columns.extend(ENGINEERED_FEATURE_COLUMNS)
    return feature_columns


def get_base_input_columns(config: dict[str, Any]) -> list[str]:
    """Return the configured base mix-design input columns."""
    return list(config["task"]["input_columns"])


def feature_engineering_enabled(config: dict[str, Any]) -> bool:
    """Return whether engineering features should be applied."""
    return bool(config.get("engineering", {}).get("feature_engineering", False))


def get_target_column(config: dict[str, Any]) -> str:
    """Return the configured target column."""
    return str(config["task"]["target_column"])


def validate_loaded_dataset(frame: pd.DataFrame, config: dict[str, Any]) -> None:
    """Validate that the dataset contains the expected schema and values."""
    expected_columns = get_input_columns(config) + [get_target_column(config)]
    missing_columns = [column for column in expected_columns if column not in frame.columns]
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {missing_columns}")
    if frame[expected_columns].isnull().any().any():
        null_columns = frame[expected_columns].columns[frame[expected_columns].isnull().any()].tolist()
        raise ValueError(f"Dataset contains missing values in columns: {null_columns}")
    numeric_frame = frame[expected_columns].apply(pd.to_numeric, errors="coerce")
    if numeric_frame.isnull().any().any():
        invalid_columns = numeric_frame.columns[numeric_frame.isnull().any()].tolist()
        raise ValueError(f"Dataset contains non-numeric values in columns: {invalid_columns}")
    if not np.isfinite(numeric_frame.to_numpy()).all():
        raise ValueError("Dataset contains non-finite numeric values.")


def ensure_engineered_dataset(frame: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """Ensure configured engineered features are present and refreshed."""
    if not feature_engineering_enabled(config):
        return frame.copy()
    return build_engineering_features(frame)


def load_dataset(config: dict[str, Any]) -> pd.DataFrame:
    """Load the prepared dataset from disk."""
    data_path = get_data_path(config)
    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {data_path}. Run generate_data.py before training."
        )
    frame = pd.read_csv(data_path)
    frame = ensure_engineered_dataset(frame, config)
    validate_loaded_dataset(frame, config)
    return frame


def build_stratification_bins(target: pd.Series, n_bins: int) -> pd.Series:
    """Create deterministic bins for stratified train-test splitting."""
    if n_bins < 2:
        raise ValueError("Stratification bin count must be at least 2.")
    ranked_target = target.rank(method="first")
    return pd.qcut(ranked_target, q=n_bins, labels=False, duplicates="drop")


def split_dataset(
    frame: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split the dataset into train and test partitions with target stratification."""
    feature_columns = get_input_columns(config)
    target_column = get_target_column(config)
    x = frame[feature_columns].copy()
    y = frame[target_column].copy()
    stratify_labels = build_stratification_bins(y, int(config["data"]["stratify_bins"]))
    return train_test_split(
        x,
        y,
        test_size=float(config["data"]["test_size"]),
        random_state=int(config["experiment"]["random_seed"]),
        stratify=stratify_labels,
    )


def calculate_composite_score(
    rmse: float,
    mae: float,
    r2: float,
    target_mean: float,
    weights: dict[str, float],
) -> float:
    """Calculate the weighted composite score used for model selection."""
    safe_mean = max(abs(target_mean), 1e-8)
    norm_rmse = rmse / safe_mean
    norm_mae = mae / safe_mean
    return (
        float(weights["rmse"]) * (1.0 - norm_rmse)
        + float(weights["r2"]) * r2
        + float(weights["mae"]) * (1.0 - norm_mae)
    )


def compute_regression_metrics(
    y_true: pd.Series | np.ndarray,
    y_pred: np.ndarray,
    config: dict[str, Any],
    target_mean: float,
) -> dict[str, float]:
    """Compute regression metrics and the project composite score."""
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    composite = float(
        calculate_composite_score(rmse, mae, r2, target_mean, config["metrics"]["composite_weights"])
    )
    return {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "composite_score": composite,
    }


def build_cv_splitter(config: dict[str, Any]) -> KFold:
    """Create the cross-validation splitter defined in config."""
    return KFold(
        n_splits=int(config["experiment"]["cv_folds"]),
        shuffle=True,
        random_state=int(config["experiment"]["random_seed"]),
    )


def cross_validate_model(
    model: Any,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    config: dict[str, Any],
) -> dict[str, float]:
    """Run configured cross-validation and return averaged metrics."""
    scoring = {
        "rmse": "neg_root_mean_squared_error",
        "mae": "neg_mean_absolute_error",
        "r2": "r2",
    }
    scores = cross_validate(
        model,
        x_train,
        y_train,
        cv=build_cv_splitter(config),
        scoring=scoring,
        n_jobs=None,
        return_train_score=False,
        error_score="raise",
    )
    rmse = float(-np.mean(scores["test_rmse"]))
    mae = float(-np.mean(scores["test_mae"]))
    r2 = float(np.mean(scores["test_r2"]))
    composite = float(
        calculate_composite_score(
            rmse,
            mae,
            r2,
            float(y_train.mean()),
            config["metrics"]["composite_weights"],
        )
    )
    return {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "composite_score": composite,
    }


def instantiate_model(model_name: str, params: dict[str, Any], config: dict[str, Any]) -> Any:
    """Create a model instance for the requested family and parameters."""
    seed = int(config["experiment"]["random_seed"])
    clean_params = dict(params)

    if model_name == "RandomForestRegressor":
        default_params = {"random_state": seed}
        default_params.update(clean_params)
        return RandomForestRegressor(**default_params)

    if model_name == "GradientBoostingRegressor":
        default_params = {"random_state": seed}
        default_params.update(clean_params)
        return GradientBoostingRegressor(**default_params)

    if model_name == "XGBRegressor":
        if XGBRegressor is None:
            raise ImportError("xgboost is not installed. Install requirements.txt before running search.")
        default_params = {
            "objective": "reg:squarederror",
            "random_state": seed,
            "n_jobs": -1,
            "verbosity": 0,
        }
        default_params.update(clean_params)
        return XGBRegressor(**default_params)

    if model_name == "LGBMRegressor":
        if LGBMRegressor is None:
            raise ImportError("lightgbm is not installed. Install requirements.txt before running search.")
        default_params = {
            "random_state": seed,
            "n_jobs": -1,
            "verbosity": -1,
        }
        default_params.update(clean_params)
        return LGBMRegressor(**default_params)

    if model_name == "SVR":
        return Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("model", SVR(**clean_params)),
            ]
        )

    if model_name == "Ridge":
        ridge_params = {"random_state": seed}
        ridge_params.update(clean_params)
        return Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("model", Ridge(**ridge_params)),
            ]
        )

    raise ValueError(f"Unsupported model family: {model_name}")


def evaluate_candidate(
    model_name: str,
    params: dict[str, Any],
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    validator: EngineeringValidator,
    config: dict[str, Any],
) -> tuple[Any, dict[str, Any]]:
    """Train, cross-validate, test, and validate a candidate model."""
    candidate_model = instantiate_model(model_name, params, config)
    cv_metrics = cross_validate_model(candidate_model, x_train, y_train, config)
    candidate_model.fit(x_train, y_train)
    test_predictions = np.asarray(candidate_model.predict(x_test), dtype=float)
    test_metrics = compute_regression_metrics(y_test, test_predictions, config, float(y_train.mean()))
    validation_report = validator.validate_predictions(test_predictions, x_test, y_test)
    result = {
        "model_name": model_name,
        "hyperparameters": params,
        "rmse": cv_metrics["rmse"],
        "mae": cv_metrics["mae"],
        "r2": cv_metrics["r2"],
        "composite_score": cv_metrics["composite_score"],
        "cv_metrics": cv_metrics,
        "test_metrics": test_metrics,
        "validation_verdict": validation_report["verdict"],
        "validation_report": validation_report,
    }
    # Keep nested metrics as the source of truth, but expose flat aliases for
    # downstream artifacts and the dashboard.
    result.update(
        {
            "cv_rmse": cv_metrics["rmse"],
            "cv_mae": cv_metrics["mae"],
            "cv_r2": cv_metrics["r2"],
            "cv_composite": cv_metrics["composite_score"],
            "holdout_rmse": test_metrics["rmse"],
            "holdout_mae": test_metrics["mae"],
            "holdout_r2": test_metrics["r2"],
            "holdout_composite": test_metrics["composite_score"],
            "validation_pass_rate": validation_report["pass_rate"],
            "failed_count": validation_report["failed_count"],
            "hard_failed_count": validation_report.get("hard_failed_count", validation_report["failed_count"]),
            "warning_count": validation_report["warning_count"],
            "suspicious_count": validation_report["suspicious_count"],
            "durability_caution_count": validation_report.get("durability_caution_count", 0),
            "dataset_anomaly_count": validation_report.get("dataset_anomaly_count", 0),
            "warn_reasons": validation_report["warn_reasons"],
            "hard_fail_reasons": validation_report.get("hard_fail_reasons", []),
            "durability_caution_reasons": validation_report.get("durability_caution_reasons", []),
            "dataset_anomaly_reasons": validation_report.get("dataset_anomaly_reasons", []),
        }
    )
    return candidate_model, result


def save_pickle_artifact(path: Path, obj: Any) -> None:
    """Serialize an object as a pickle file."""
    with path.open("wb") as handle:
        pickle.dump(obj, handle)


def to_serializable(value: Any) -> Any:
    """Convert NumPy and pandas values into JSON-serializable Python types."""
    if isinstance(value, dict):
        return {str(key): to_serializable(inner_value) for key, inner_value in value.items()}
    if isinstance(value, list):
        return [to_serializable(item) for item in value]
    if isinstance(value, tuple):
        return [to_serializable(item) for item in value]
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def save_json_artifact(path: Path, payload: dict[str, Any]) -> None:
    """Serialize a dictionary to JSON with stable formatting."""
    with path.open("w", encoding="utf-8") as handle:
        json.dump(to_serializable(payload), handle, indent=2)


def format_metrics_summary(metrics: dict[str, Any]) -> str:
    """Format a concise metric summary for console logging."""
    return (
        f"RMSE={metrics['rmse']:.4f} | MAE={metrics['mae']:.4f} | "
        f"R2={metrics['r2']:.4f} | Composite={metrics['composite_score']:.4f}"
    )


def main() -> int:
    """Train and save the baseline model and metrics."""
    try:
        config = load_config()
        set_global_seed(int(config["experiment"]["random_seed"]))
        outputs_dir = get_outputs_dir(config)
        data = load_dataset(config)
        x_train, x_test, y_train, y_test = split_dataset(data, config)
        validator = EngineeringValidator.from_config(config)

        baseline_name = str(config["baseline_model"]["model_name"])
        baseline_params = dict(config["baseline_model"]["params"])
        log_status(f"Training baseline model: {baseline_name}")
        baseline_model, baseline_result = evaluate_candidate(
            baseline_name,
            baseline_params,
            x_train,
            y_train,
            x_test,
            y_test,
            validator,
            config,
        )

        baseline_model_path = outputs_dir / "baseline_model.pkl"
        baseline_metrics_path = outputs_dir / "baseline_metrics.json"
        save_pickle_artifact(baseline_model_path, baseline_model)
        save_json_artifact(baseline_metrics_path, baseline_result)

        log_status(f"Saved baseline model to {baseline_model_path}")
        log_status(f"Saved baseline metrics to {baseline_metrics_path}")
        log_status(
            "Baseline CV metrics: "
            + format_metrics_summary(baseline_result["cv_metrics"])
            + f" | Validation={baseline_result['validation_verdict']}"
        )
        log_status("Baseline test metrics: " + format_metrics_summary(baseline_result["test_metrics"]))
        return 0
    except Exception as exc:
        log_status(f"Baseline training failed: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
