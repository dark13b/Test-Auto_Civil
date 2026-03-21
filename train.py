"""Baseline training utilities and entry point for AutoCivil-Lab."""

from __future__ import annotations

import json
import os
import pickle
import random
import re
import subprocess
import sys
import tempfile
import warnings
from hashlib import sha256
from importlib import import_module
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.base import clone
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
    StackingRegressor,
)
from sklearn.linear_model import ElasticNet, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, RepeatedKFold, cross_val_score, cross_validate, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

from feature_engineering import ENGINEERED_FEATURE_COLUMNS, build_engineering_features
from integrity_checks import assert_no_cv_label_fraud
from validator import EngineeringValidator, summarize_validation_report

try:
    from lightgbm import LGBMRegressor
except ImportError:
    LGBMRegressor = None

try:
    from xgboost import XGBRegressor
except ImportError:
    XGBRegressor = None

try:
    from catboost import CatBoostRegressor
except ImportError:
    CatBoostRegressor = None


ARTIFACT_VERSION_MODULES = {
    "scikit-learn": "sklearn",
    "lightgbm": "lightgbm",
    "xgboost": "xgboost",
    "catboost": "catboost",
    "optuna": "optuna",
    "numpy": "numpy",
    "pandas": "pandas",
    "joblib": "joblib",
}

RUN_ARTIFACTS_DIRNAME = "runs"


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


def create_run_id() -> str:
    """Create a timestamp-based run identifier."""
    return pd.Timestamp.now().strftime("%Y%m%dT%H%M%S")


def get_input_columns(config: dict[str, Any]) -> list[str]:
    """Return the configured feature columns."""
    feature_columns = get_base_input_columns(config)
    if feature_engineering_enabled(config):
        feature_columns.extend(ENGINEERED_FEATURE_COLUMNS)
    return feature_columns


def get_base_input_columns(config: dict[str, Any]) -> list[str]:
    """Return the configured base mix-design input columns."""
    return list(config["task"]["input_columns"])


def resolve_model_feature_columns(model: Any, config: dict[str, Any]) -> list[str]:
    """Return the feature schema expected by a fitted model artifact."""
    configured_columns = get_input_columns(config)
    feature_names = getattr(model, "feature_names_in_", None)
    if feature_names is not None:
        return [str(name) for name in feature_names]

    feature_count = getattr(model, "n_features_in_", None)
    if feature_count is None:
        return configured_columns
    if int(feature_count) == len(configured_columns):
        return configured_columns

    base_columns = get_base_input_columns(config)
    if len(base_columns) <= int(feature_count) <= len(configured_columns):
        return configured_columns[: int(feature_count)]
    raise ValueError(
        "Model artifact feature schema is incompatible with the configured dataset columns."
    )


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
    return build_engineering_features(frame, config=config)


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
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Split the dataset into train, validation, and locked-test partitions."""
    feature_columns = get_input_columns(config)
    target_column = get_target_column(config)
    x = frame[feature_columns].copy()
    y = frame[target_column].copy()
    stratify_labels = build_stratification_bins(y, int(config["data"]["stratify_bins"]))
    data_split = dict(config.get("data_split", {}))
    train_size = float(data_split.get("train_size", 0.70))
    val_size = float(data_split.get("val_size", 0.15))
    test_size = float(data_split.get("test_size", config.get("data", {}).get("test_size", 0.15)))
    total = train_size + val_size + test_size
    if not np.isclose(total, 1.0):
        raise ValueError("data_split train/val/test sizes must sum to 1.0")

    x_train_val, x_test, y_train_val, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=42,
        stratify=stratify_labels,
    )
    train_val_stratify = build_stratification_bins(y_train_val, int(config["data"]["stratify_bins"]))
    val_relative_size = val_size / max(train_size + val_size, 1e-8)
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_val,
        y_train_val,
        test_size=val_relative_size,
        random_state=42,
        stratify=train_val_stratify,
    )
    return x_train, x_val, x_test, y_train, y_val, y_test


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


def build_cv_splitter(
    config: dict[str, Any],
    *,
    n_splits: int | None = None,
    repeats: int | None = None,
    random_state: int | None = None,
) -> KFold | RepeatedKFold:
    """Create the cross-validation splitter defined in config."""
    split_count = int(n_splits or config["experiment"]["cv_folds"])
    repeat_count = int(repeats or config.get("experiment", {}).get("cv_repeats", 1))
    seed = int(random_state or config["experiment"]["random_seed"])
    if repeat_count > 1:
        return RepeatedKFold(
            n_splits=split_count,
            n_repeats=repeat_count,
            random_state=seed,
        )
    return KFold(
        n_splits=split_count,
        shuffle=True,
        random_state=seed,
    )


def cross_validate_model(
    model: Any,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    config: dict[str, Any],
    *,
    cv: Any | None = None,
    return_fold_metrics: bool = False,
) -> dict[str, Any]:
    """Run configured cross-validation and return averaged metrics."""
    scoring = {
        "rmse": "neg_root_mean_squared_error",
        "mae": "neg_mean_absolute_error",
        "r2": "r2",
    }
    splitter = cv or build_cv_splitter(config)
    def _slice_frame(frame: Any, indices: np.ndarray) -> Any:
        return frame.iloc[indices] if hasattr(frame, "iloc") else frame[indices]

    def _manual_cross_validate_scores() -> dict[str, np.ndarray]:
        rmse_scores: list[float] = []
        mae_scores: list[float] = []
        r2_scores: list[float] = []
        for train_indices, test_indices in splitter.split(x_train, y_train):
            x_fold_train = _slice_frame(x_train, train_indices)
            x_fold_test = _slice_frame(x_train, test_indices)
            y_fold_train = _slice_frame(y_train, train_indices)
            y_fold_test = _slice_frame(y_train, test_indices)
            try:
                fold_model = clone(model)
            except Exception:
                if hasattr(model, "get_params"):
                    fold_model = type(model)(**model.get_params())
                else:
                    raise
            fold_model.fit(x_fold_train, y_fold_train)
            fold_predictions = np.asarray(fold_model.predict(x_fold_test), dtype=float)
            fold_metrics = compute_regression_metrics(
                y_fold_test,
                fold_predictions,
                config,
                float(np.asarray(y_fold_train, dtype=float).mean()),
            )
            rmse_scores.append(-float(fold_metrics["rmse"]))
            mae_scores.append(-float(fold_metrics["mae"]))
            r2_scores.append(float(fold_metrics["r2"]))
        return {
            "test_rmse": np.asarray(rmse_scores, dtype=float),
            "test_mae": np.asarray(mae_scores, dtype=float),
            "test_r2": np.asarray(r2_scores, dtype=float),
        }

    try:
        scores = cross_validate(
            model,
            x_train,
            y_train,
            cv=splitter,
            scoring=scoring,
            n_jobs=None,
            return_train_score=False,
            error_score="raise",
        )
    except Exception as exc:
        if "__sklearn_tags__" not in str(exc):
            raise
        scores = _manual_cross_validate_scores()
    rmse_scores = -scores["test_rmse"]
    mae_scores = -scores["test_mae"]
    r2_scores = scores["test_r2"]
    rmse = float(np.mean(rmse_scores))
    mae = float(np.mean(mae_scores))
    r2 = float(np.mean(r2_scores))
    composite = float(
        calculate_composite_score(
            rmse,
            mae,
            r2,
            float(y_train.mean()),
            config["metrics"]["composite_weights"],
        )
    )
    result: dict[str, Any] = {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "composite_score": composite,
    }
    if return_fold_metrics:
        safe_mean = float(y_train.mean())
        composite_scores = [
            calculate_composite_score(
                float(fold_rmse),
                float(fold_mae),
                float(fold_r2),
                safe_mean,
                config["metrics"]["composite_weights"],
            )
            for fold_rmse, fold_mae, fold_r2 in zip(rmse_scores, mae_scores, r2_scores, strict=False)
        ]
        std_kwargs = {"ddof": 1} if len(rmse_scores) > 1 else {}
        result.update(
            {
                "rmse_std": float(np.std(rmse_scores, **std_kwargs)),
                "mae_std": float(np.std(mae_scores, **std_kwargs)),
                "r2_std": float(np.std(r2_scores, **std_kwargs)),
                "composite_std": float(np.std(composite_scores, **std_kwargs)),
                "fold_metrics": {
                    "rmse": [float(value) for value in rmse_scores],
                    "mae": [float(value) for value in mae_scores],
                    "r2": [float(value) for value in r2_scores],
                    "composite_score": [float(value) for value in composite_scores],
                },
            }
        )
    return result


def instantiate_model(model_name: str, params: dict[str, Any], config: dict[str, Any]) -> Any:
    """Create a model instance for the requested family and parameters."""
    seed = int(config["experiment"]["random_seed"])
    clean_params = dict(params)

    if model_name == "LinearRegression":
        return Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("model", LinearRegression(**clean_params)),
            ]
        )

    if model_name == "RandomForestRegressor":
        default_params = {"random_state": seed, "n_jobs": -1}
        default_params.update(clean_params)
        return RandomForestRegressor(**default_params)

    if model_name == "ExtraTreesRegressor":
        default_params = {"random_state": seed, "n_jobs": -1}
        default_params.update(clean_params)
        return ExtraTreesRegressor(**default_params)

    if model_name == "DecisionTreeRegressor":
        default_params = {"random_state": seed}
        default_params.update(clean_params)
        return DecisionTreeRegressor(**default_params)

    if model_name == "GradientBoostingRegressor":
        default_params = {"random_state": seed}
        default_params.update(clean_params)
        return GradientBoostingRegressor(**default_params)

    if model_name == "HistGradientBoostingRegressor":
        default_params = {"random_state": seed}
        default_params.update(clean_params)
        return HistGradientBoostingRegressor(**default_params)

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

    if model_name == "CatBoostRegressor":
        if CatBoostRegressor is None:
            raise ImportError("catboost is not installed. Install requirements.txt before running benchmark.")
        default_params = {
            "loss_function": "RMSE",
            "random_state": seed,
            "verbose": False,
            "allow_writing_files": False,
        }
        default_params.update(clean_params)
        return CatBoostRegressor(**default_params)

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

    if model_name == "ElasticNet":
        elastic_net_params = {
            "random_state": seed,
            "max_iter": 20000,
        }
        elastic_net_params.update(clean_params)
        return Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("model", ElasticNet(**elastic_net_params)),
            ]
        )

    if model_name == "KNeighborsRegressor":
        return Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("model", KNeighborsRegressor(**clean_params)),
            ]
        )

    raise ValueError(f"Unsupported model family: {model_name}")


def evaluate_candidate(
    model_name: str,
    params: dict[str, Any],
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_val: pd.DataFrame,
    y_val: pd.Series,
    validator: EngineeringValidator,
    config: dict[str, Any],
) -> tuple[Any, dict[str, Any]]:
    """Train, cross-validate, validate, and validate engineering constraints."""
    candidate_model = instantiate_model(model_name, params, config)
    cv_metrics = cross_validate_model(candidate_model, x_train, y_train, config)
    candidate_model.fit(x_train, y_train)
    val_predictions = np.asarray(candidate_model.predict(x_val), dtype=float)
    val_metrics = compute_regression_metrics(y_val, val_predictions, config, float(y_train.mean()))
    validation_report = validator.validate_predictions(val_predictions, x_val, y_val)
    validation_summary = summarize_validation_report(validation_report)
    result = {
        "model_name": model_name,
        "hyperparameters": params,
        "rmse": cv_metrics["rmse"],
        "mae": cv_metrics["mae"],
        "r2": cv_metrics["r2"],
        "composite_score": cv_metrics["composite_score"],
        "cv_metrics": cv_metrics,
        "val_metrics": val_metrics,
        "test_metrics": val_metrics,
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
            "val_rmse": val_metrics["rmse"],
            "val_mae": val_metrics["mae"],
            "val_r2": val_metrics["r2"],
            "val_composite": val_metrics["composite_score"],
            "validator_context_type": validation_report.get("context_type", "general"),
            "validation_pass_rate": validation_report["pass_rate"],
            "failed_count": validation_summary["failed_count"],
            "hard_constraint_count": validation_summary["hard_constraint_count"],
            "hard_failed_count": validation_summary["hard_failed_count"],
            "warning_count": validation_summary["warning_count"],
            "engineering_caution_count": validation_summary["engineering_caution_count"],
            "suspicious_count": validation_summary["suspicious_count"],
            "statistical_errors": validation_summary["statistical_errors"],
            "statistical_error_count": validation_summary["statistical_error_count"],
            "durability_warnings": validation_summary["durability_warnings"],
            "durability_warning_count": validation_summary["durability_warning_count"],
            "durability_caution_count": validation_summary["durability_caution_count"],
            "data_review_flag_count": validation_summary["data_review_flag_count"],
            "dataset_anomalies": validation_summary["dataset_anomalies"],
            "dataset_anomaly_count": validation_summary["dataset_anomaly_count"],
            "warn_reasons": validation_report["warn_reasons"],
            "hard_constraint_reasons": validation_summary["hard_constraint_reasons"],
            "statistical_error_reasons": validation_summary["statistical_error_reasons"],
            "hard_fail_reasons": validation_summary["hard_fail_reasons"],
            "engineering_caution_reasons": validation_summary["engineering_caution_reasons"],
            "durability_warning_reasons": validation_summary["durability_warning_reasons"],
            "durability_caution_reasons": validation_summary["durability_caution_reasons"],
            "data_review_flag_reasons": validation_summary["data_review_flag_reasons"],
            "dataset_anomaly_reasons": validation_summary["dataset_anomaly_reasons"],
            "contextual_summary": validation_summary["contextual_summary"],
            "confidence_of_warning_assessment": validation_summary["confidence_of_warning_assessment"],
        }
    )
    assert_no_cv_label_fraud(result)
    return candidate_model, result


def build_stacking_ensemble(
    configs: list[tuple[str, dict[str, Any]]],
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_val: pd.DataFrame,
    y_val: pd.Series,
    validator: EngineeringValidator,
    config: dict[str, Any],
) -> tuple[Any, dict[str, Any]]:
    """Train a stacking ensemble from a list of base-model configurations."""
    if not configs:
        raise ValueError("At least one base-model configuration is required to build a stacking ensemble.")

    estimators: list[tuple[str, Any]] = []
    base_model_specs: list[dict[str, Any]] = []
    for index, (model_name, hyperparams) in enumerate(configs, start=1):
        estimator_name = f"{model_name.lower()}_{index}"
        estimators.append((estimator_name, instantiate_model(model_name, hyperparams, config)))
        base_model_specs.append(
            {
                "model_name": model_name,
                "hyperparameters": dict(hyperparams),
            }
        )

    seed = int(config["experiment"]["random_seed"])
    stacking_model = StackingRegressor(
        estimators=estimators,
        final_estimator=Ridge(random_state=seed),
        passthrough=True,
        cv=build_cv_splitter(config),
        n_jobs=-1,
    )

    cv_splitter = build_cv_splitter(config, n_splits=5, repeats=1, random_state=42)
    cv_r2 = float(np.mean(cross_val_score(stacking_model, x_train, y_train, cv=cv_splitter, scoring="r2")))
    cv_rmse = float(
        np.mean(
            -cross_val_score(
                stacking_model,
                x_train,
                y_train,
                cv=cv_splitter,
                scoring="neg_root_mean_squared_error",
            )
        )
    )
    cv_mae = float(
        np.mean(
            -cross_val_score(
                stacking_model,
                x_train,
                y_train,
                cv=cv_splitter,
                scoring="neg_mean_absolute_error",
            )
        )
    )
    cv_metrics = {
        "rmse": cv_rmse,
        "mae": cv_mae,
        "r2": cv_r2,
        "composite_score": float(
            calculate_composite_score(
                cv_rmse,
                cv_mae,
                cv_r2,
                float(y_train.mean()),
                config["metrics"]["composite_weights"],
            )
        ),
    }
    stacking_model.fit(x_train, y_train)

    val_predictions = np.asarray(stacking_model.predict(x_val), dtype=float)
    val_metrics = compute_regression_metrics(y_val, val_predictions, config, float(y_train.mean()))
    validation_report = validator.validate_predictions(val_predictions, x_val, y_val)
    validation_summary = summarize_validation_report(validation_report)
    result = {
        "model_name": "StackingRegressor",
        "hyperparameters": {
            "base_models": base_model_specs,
            "meta_learner": "Ridge",
            "passthrough": True,
        },
        "rmse": cv_metrics["rmse"],
        "mae": cv_metrics["mae"],
        "r2": cv_metrics["r2"],
        "composite_score": cv_metrics["composite_score"],
        "cv_metrics": cv_metrics,
        "val_metrics": val_metrics,
        "test_metrics": val_metrics,
        "validation_verdict": validation_report["verdict"],
        "validation_report": validation_report,
    }
    result.update(
        {
            "cv_rmse": cv_metrics["rmse"],
            "cv_mae": cv_metrics["mae"],
            "cv_r2": cv_metrics["r2"],
            "cv_composite": cv_metrics["composite_score"],
            "val_rmse": val_metrics["rmse"],
            "val_mae": val_metrics["mae"],
            "val_r2": val_metrics["r2"],
            "val_composite": val_metrics["composite_score"],
            "validator_context_type": validation_report.get("context_type", "general"),
            "validation_pass_rate": validation_report["pass_rate"],
            "failed_count": validation_summary["failed_count"],
            "hard_constraint_count": validation_summary["hard_constraint_count"],
            "hard_failed_count": validation_summary["hard_failed_count"],
            "warning_count": validation_summary["warning_count"],
            "engineering_caution_count": validation_summary["engineering_caution_count"],
            "suspicious_count": validation_summary["suspicious_count"],
            "statistical_errors": validation_summary["statistical_errors"],
            "statistical_error_count": validation_summary["statistical_error_count"],
            "durability_warnings": validation_summary["durability_warnings"],
            "durability_warning_count": validation_summary["durability_warning_count"],
            "durability_caution_count": validation_summary["durability_caution_count"],
            "data_review_flag_count": validation_summary["data_review_flag_count"],
            "dataset_anomalies": validation_summary["dataset_anomalies"],
            "dataset_anomaly_count": validation_summary["dataset_anomaly_count"],
            "warn_reasons": validation_report["warn_reasons"],
            "hard_constraint_reasons": validation_summary["hard_constraint_reasons"],
            "statistical_error_reasons": validation_summary["statistical_error_reasons"],
            "hard_fail_reasons": validation_summary["hard_fail_reasons"],
            "engineering_caution_reasons": validation_summary["engineering_caution_reasons"],
            "durability_warning_reasons": validation_summary["durability_warning_reasons"],
            "durability_caution_reasons": validation_summary["durability_caution_reasons"],
            "data_review_flag_reasons": validation_summary["data_review_flag_reasons"],
            "dataset_anomaly_reasons": validation_summary["dataset_anomaly_reasons"],
            "contextual_summary": validation_summary["contextual_summary"],
            "confidence_of_warning_assessment": validation_summary["confidence_of_warning_assessment"],
        }
    )
    assert_no_cv_label_fraud(result)
    return stacking_model, result

def get_runtime_library_versions() -> dict[str, str]:
    """Return the currently installed versions for tracked runtime libraries."""
    versions: dict[str, str] = {}
    for package_name, module_name in ARTIFACT_VERSION_MODULES.items():
        try:
            versions[package_name] = str(import_module(module_name).__version__)
        except Exception:
            versions[package_name] = "unavailable"
    return versions


def compute_config_hash(config: dict[str, Any] | None) -> str | None:
    """Return a stable hash of the active configuration."""
    if config is None:
        return None
    normalized = json.dumps(to_serializable(config), sort_keys=True, separators=(",", ":"))
    return sha256(normalized.encode("utf-8")).hexdigest()


def get_git_commit_hash(project_root: Path | None = None) -> str | None:
    """Return the current git commit hash when available."""
    resolved_root = get_project_root() if project_root is None else project_root
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=resolved_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    commit = completed.stdout.strip()
    return commit or None


def compute_file_hash(path: Path) -> str | None:
    """Return a SHA-256 hash for a file when it exists."""
    if not path.exists():
        return None
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extract_artifact_metadata(payload: dict[str, Any] | None) -> dict[str, Any]:
    """Return normalized artifact metadata from a JSON payload."""
    if not isinstance(payload, dict):
        return {}
    raw_metadata = payload.get("artifact_metadata", {})
    metadata = dict(raw_metadata) if isinstance(raw_metadata, dict) else {}
    for key in ("artifact_id", "run_id", "source_mode", "timestamp", "model_artifact_id"):
        if key not in metadata and key in payload:
            metadata[key] = payload[key]
    return metadata


def artifact_run_id(payload: dict[str, Any] | None) -> str | None:
    """Return the run id associated with a JSON payload."""
    metadata = extract_artifact_metadata(payload)
    run_id = metadata.get("run_id")
    return None if run_id is None else str(run_id)


def artifact_id(payload: dict[str, Any] | None) -> str | None:
    """Return the artifact id associated with a JSON payload."""
    metadata = extract_artifact_metadata(payload)
    current_artifact_id = metadata.get("artifact_id")
    return None if current_artifact_id is None else str(current_artifact_id)


def infer_active_run_id(
    outputs_dir: Path,
    *payloads: dict[str, Any] | None,
    fallback: str | None = None,
) -> str:
    """Resolve the active run id from in-memory payloads or the latest-run manifest."""
    for payload in payloads:
        run_id = artifact_run_id(payload)
        if run_id:
            return run_id

    latest_manifest_path = outputs_dir / "latest_run_manifest.json"
    if latest_manifest_path.exists():
        try:
            with latest_manifest_path.open("r", encoding="utf-8") as handle:
                manifest = json.load(handle)
            manifest_run_id = manifest.get("run_id")
            if manifest_run_id:
                return str(manifest_run_id)
        except Exception:
            pass

    return fallback or create_run_id()


def mark_json_artifact_stale(
    path: str | Path,
    *,
    active_run_id: str | None,
    reason: str | None = None,
) -> bool:
    """Mark a canonical JSON artifact as stale when it does not belong to the active run."""
    artifact_path = Path(path)
    if not artifact_path.exists():
        return False

    try:
        with artifact_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception:
        return False

    current_run_id = artifact_run_id(payload)
    metadata = dict(extract_artifact_metadata(payload))
    already_stale = bool(metadata.get("stale") or payload.get("stale"))
    if current_run_id == active_run_id and not already_stale:
        return False

    stale_reason = reason or (
        "artifact does not belong to the active run"
        if active_run_id
        else "artifact lineage could not be resolved"
    )
    metadata["stale"] = True
    metadata["canonical_latest"] = False
    metadata["stale_reason"] = stale_reason
    metadata["stale_checked_against_run_id"] = active_run_id
    if current_run_id is not None:
        metadata.setdefault("run_id", current_run_id)
    payload["artifact_metadata"] = metadata
    payload["stale"] = True
    payload["stale_reason"] = stale_reason
    if active_run_id is not None:
        payload["active_run_id"] = active_run_id

    save_json_artifact(artifact_path, payload)
    return True


def build_json_artifact_metadata(
    *,
    outputs_dir: Path,
    filename: str,
    run_id: str,
    source_mode: str,
    config: dict[str, Any] | None = None,
    model_artifact_id: str | None = None,
    model_id: str | None = None,
    parent_artifact_ids: list[str] | None = None,
    parent_run_ids: list[str] | None = None,
    extra_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the shared metadata envelope for a JSON artifact."""
    timestamp = pd.Timestamp.now().isoformat()
    canonical_path = outputs_dir / filename
    run_path = outputs_dir / RUN_ARTIFACTS_DIRNAME / run_id / filename
    metadata = {
        "artifact_schema_version": 1,
        "artifact_name": filename,
        "artifact_type": "json",
        "run_id": run_id,
        "timestamp": timestamp,
        "source_mode": source_mode,
        "config_hash": compute_config_hash(config),
        "code_fingerprint": {
            "git_commit": get_git_commit_hash(outputs_dir.parent),
        },
        "model_artifact_id": model_artifact_id,
        "model_id": model_id,
        "parent_artifact_ids": list(parent_artifact_ids or []),
        "parent_run_ids": list(parent_run_ids or []),
        "canonical_path": str(canonical_path),
        "run_scoped_path": str(run_path),
        "canonical_latest": True,
        "stale": False,
    }
    artifact_key = json.dumps(
        {
            "filename": filename,
            "run_id": run_id,
            "source_mode": source_mode,
            "timestamp": timestamp,
            "model_artifact_id": model_artifact_id,
            "parent_artifact_ids": metadata["parent_artifact_ids"],
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    metadata["artifact_id"] = sha256(artifact_key.encode("utf-8")).hexdigest()
    if isinstance(extra_metadata, dict):
        metadata.update(to_serializable(extra_metadata))
    return metadata


def _update_run_manifest(outputs_dir: Path, metadata: dict[str, Any]) -> None:
    """Record the latest artifact set for a run and refresh the latest-run pointer."""
    run_id = str(metadata["run_id"])
    run_dir = outputs_dir / RUN_ARTIFACTS_DIRNAME / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / "run_manifest.json"
    if manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
    else:
        manifest = {
            "run_id": run_id,
            "created_at": metadata["timestamp"],
            "artifacts": {},
        }
    manifest.setdefault("artifacts", {})
    manifest["updated_at"] = metadata["timestamp"]
    manifest["artifacts"][str(metadata["artifact_name"])] = {
        "artifact_id": metadata["artifact_id"],
        "source_mode": metadata["source_mode"],
        "timestamp": metadata["timestamp"],
        "canonical_path": metadata["canonical_path"],
        "run_scoped_path": metadata["run_scoped_path"],
        "model_artifact_id": metadata.get("model_artifact_id"),
        "parent_artifact_ids": list(metadata.get("parent_artifact_ids", [])),
    }
    save_json_artifact(manifest_path, manifest)
    latest_manifest = {
        "run_id": run_id,
        "updated_at": metadata["timestamp"],
        "manifest_path": str(manifest_path),
        "artifacts": manifest["artifacts"],
    }
    save_json_artifact(outputs_dir / "latest_run_manifest.json", latest_manifest)


def write_run_scoped_json_artifact(
    *,
    outputs_dir: Path,
    filename: str,
    payload: dict[str, Any],
    run_id: str,
    source_mode: str,
    config: dict[str, Any] | None = None,
    model_artifact_id: str | None = None,
    model_id: str | None = None,
    parent_artifact_ids: list[str] | None = None,
    parent_run_ids: list[str] | None = None,
    extra_metadata: dict[str, Any] | None = None,
) -> tuple[Path, Path, dict[str, Any]]:
    """Write one JSON artifact to both run-scoped and canonical latest locations."""
    metadata = build_json_artifact_metadata(
        outputs_dir=outputs_dir,
        filename=filename,
        run_id=run_id,
        source_mode=source_mode,
        config=config,
        model_artifact_id=model_artifact_id,
        model_id=model_id,
        parent_artifact_ids=parent_artifact_ids,
        parent_run_ids=parent_run_ids,
        extra_metadata=extra_metadata,
    )
    enriched_payload = dict(payload)
    enriched_payload["artifact_id"] = metadata["artifact_id"]
    enriched_payload["run_id"] = metadata["run_id"]
    enriched_payload["timestamp"] = metadata["timestamp"]
    enriched_payload["source_mode"] = metadata["source_mode"]
    enriched_payload["parent_artifact_ids"] = list(metadata["parent_artifact_ids"])
    if metadata.get("model_artifact_id") is not None:
        enriched_payload["model_artifact_id"] = metadata["model_artifact_id"]
    enriched_payload["artifact_metadata"] = metadata

    canonical_path = outputs_dir / filename
    run_path = outputs_dir / RUN_ARTIFACTS_DIRNAME / run_id / filename
    run_path.parent.mkdir(parents=True, exist_ok=True)
    save_json_artifact(run_path, enriched_payload)
    save_json_artifact(canonical_path, enriched_payload)
    _update_run_manifest(outputs_dir, metadata)
    return canonical_path, run_path, enriched_payload


def write_run_scoped_dataframe(
    *,
    outputs_dir: Path,
    filename: str,
    frame: pd.DataFrame,
    run_id: str,
    source_mode: str,
    config: dict[str, Any] | None = None,
    model_artifact_id: str | None = None,
    parent_artifact_ids: list[str] | None = None,
    extra_metadata: dict[str, Any] | None = None,
) -> tuple[Path, Path, pd.DataFrame, dict[str, Any]]:
    """Write one dataframe to both run-scoped and canonical latest CSV paths."""
    metadata = build_json_artifact_metadata(
        outputs_dir=outputs_dir,
        filename=filename,
        run_id=run_id,
        source_mode=source_mode,
        config=config,
        model_artifact_id=model_artifact_id,
        parent_artifact_ids=parent_artifact_ids,
        extra_metadata=extra_metadata,
    )
    enriched_frame = frame.copy()
    enriched_frame["artifact_id"] = metadata["artifact_id"]
    enriched_frame["run_id"] = metadata["run_id"]
    enriched_frame["timestamp"] = metadata["timestamp"]
    enriched_frame["source_mode"] = metadata["source_mode"]
    enriched_frame["model_artifact_id"] = metadata.get("model_artifact_id")
    enriched_frame["parent_artifact_ids"] = json.dumps(list(metadata["parent_artifact_ids"]))

    canonical_path = outputs_dir / filename
    run_path = outputs_dir / RUN_ARTIFACTS_DIRNAME / run_id / filename
    run_path.parent.mkdir(parents=True, exist_ok=True)
    enriched_frame.to_csv(run_path, index=False)
    enriched_frame.to_csv(canonical_path, index=False)
    _update_run_manifest(outputs_dir, metadata)
    return canonical_path, run_path, enriched_frame, metadata


def read_pickle_artifact_metadata(path: Path) -> dict[str, Any] | None:
    """Read only the metadata envelope for a pickled artifact when present."""
    if not path.exists():
        return None
    with path.open("rb") as handle:
        raw_artifact = pickle.load(handle)
    if (
        isinstance(raw_artifact, dict)
        and "payload" in raw_artifact
        and isinstance(raw_artifact.get("artifact_metadata"), dict)
    ):
        return dict(raw_artifact["artifact_metadata"])
    return None


def build_pickle_artifact_metadata(
    obj: Any,
    config: dict[str, Any] | None = None,
    model_id: str | None = None,
    run_id: str | None = None,
    source_mode: str | None = None,
    parent_artifact_ids: list[str] | None = None,
) -> dict[str, Any]:
    """Build metadata stored alongside pickled model artifacts."""
    saved_at = pd.Timestamp.now().isoformat()
    artifact_key = json.dumps(
        {
            "saved_at": saved_at,
            "model_id": model_id or type(obj).__name__,
            "run_id": run_id,
            "source_mode": source_mode,
            "parent_artifact_ids": list(parent_artifact_ids or []),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return {
        "artifact_schema_version": 2,
        "library_versions": get_runtime_library_versions(),
        "saved_at": saved_at,
        "model_id": model_id or type(obj).__name__,
        "config_hash": compute_config_hash(config),
        "run_id": run_id,
        "source_mode": source_mode,
        "parent_artifact_ids": list(parent_artifact_ids or []),
        "artifact_id": sha256(artifact_key.encode("utf-8")).hexdigest(),
        "code_fingerprint": {
            "git_commit": get_git_commit_hash(),
        },
    }


def _emit_artifact_version_warning(path: Path, metadata: dict[str, Any]) -> None:
    """Log a structured warning when a saved artifact differs from the runtime environment."""
    saved_versions = metadata.get("library_versions")
    if not isinstance(saved_versions, dict):
        return

    current_versions = get_runtime_library_versions()
    mismatches = []
    for package_name, saved_version in saved_versions.items():
        current_version = current_versions.get(package_name)
        if current_version is None or str(saved_version) == str(current_version):
            continue
        mismatches.append(
            {
                "package": package_name,
                "saved": str(saved_version),
                "current": str(current_version),
            }
        )

    if not mismatches:
        return

    warning_payload = {
        "path": str(path),
        "model_id": metadata.get("model_id"),
        "saved_at": metadata.get("saved_at"),
        "config_hash": metadata.get("config_hash"),
        "mismatches": mismatches,
    }
    log_status(
        "WARNING artifact_version_mismatch "
        + json.dumps(to_serializable(warning_payload), sort_keys=True)
    )


def save_pickle_artifact(
    path: Path,
    obj: Any,
    *,
    config: dict[str, Any] | None = None,
    model_id: str | None = None,
    run_id: str | None = None,
    source_mode: str | None = None,
    parent_artifact_ids: list[str] | None = None,
) -> dict[str, Any]:
    """Serialize an object as a pickle file with backward-compatible metadata."""
    metadata = build_pickle_artifact_metadata(
        obj,
        config=config,
        model_id=model_id,
        run_id=run_id,
        source_mode=source_mode,
        parent_artifact_ids=parent_artifact_ids,
    )
    artifact_bundle = {
        "artifact_metadata": metadata,
        "payload": obj,
    }
    with path.open("wb") as handle:
        pickle.dump(artifact_bundle, handle)
    return metadata


def load_pickle_artifact(path: Path, *, return_metadata: bool = False) -> Any:
    """Load a pickle artifact, preserving compatibility with older raw-model pickles."""
    if not path.exists():
        raise FileNotFoundError(f"Required artifact not found: {path}")

    with path.open("rb") as handle:
        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always")
            raw_artifact = pickle.load(handle)

    current_versions = get_runtime_library_versions()
    emitted_warnings: set[tuple[str, str | None, str | None]] = set()
    for caught_warning in caught_warnings:
        category_name = caught_warning.category.__name__
        warning_text = str(caught_warning.message)
        if category_name != "InconsistentVersionWarning":
            continue
        mismatch_match = re.search(
            r"from version\s+(?P<saved>[^\s]+)\s+when using version\s+(?P<current>[^\s]+)",
            warning_text,
        )
        saved_version = None if mismatch_match is None else mismatch_match.group("saved")
        current_version = current_versions.get("scikit-learn")
        warning_signature = ("scikit-learn", saved_version, current_version)
        if warning_signature in emitted_warnings:
            continue
        emitted_warnings.add(warning_signature)
        warning_payload = {
            "path": str(path),
            "model_id": None,
            "saved_at": None,
            "config_hash": None,
            "mismatches": [
                {
                    "package": "scikit-learn",
                    "saved": saved_version,
                    "current": current_version,
                }
            ],
        }
        log_status(
            "WARNING artifact_version_mismatch "
            + json.dumps(to_serializable(warning_payload), sort_keys=True)
        )

    if (
        isinstance(raw_artifact, dict)
        and "payload" in raw_artifact
        and isinstance(raw_artifact.get("artifact_metadata"), dict)
    ):
        metadata = dict(raw_artifact["artifact_metadata"])
        _emit_artifact_version_warning(path, metadata)
        payload = raw_artifact["payload"]
    else:
        metadata = None
        payload = raw_artifact

    if return_metadata:
        return payload, metadata
    return payload


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
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized_payload = to_serializable(payload)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            json.dump(serialized_payload, handle, indent=2, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
            temp_path = Path(handle.name)
        os.replace(temp_path, path)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink(missing_ok=True)


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
        x_train, x_val, _, y_train, y_val, _ = split_dataset(data, config)
        validator = EngineeringValidator.from_config(config)

        baseline_name = str(config["baseline_model"]["model_name"])
        baseline_params = dict(config["baseline_model"]["params"])
        log_status(f"Training baseline model: {baseline_name}")
        baseline_model, baseline_result = evaluate_candidate(
            baseline_name,
            baseline_params,
            x_train,
            y_train,
            x_val,
            y_val,
            validator,
            config,
        )

        baseline_model_path = outputs_dir / "baseline_model.pkl"
        baseline_metrics_path = outputs_dir / "baseline_metrics.json"
        save_pickle_artifact(
            baseline_model_path,
            baseline_model,
            config=config,
            model_id=baseline_name,
        )
        save_json_artifact(baseline_metrics_path, baseline_result)

        log_status(f"Saved baseline model to {baseline_model_path}")
        log_status(f"Saved baseline metrics to {baseline_metrics_path}")
        log_status(
            "Baseline CV metrics: "
            + format_metrics_summary(baseline_result["cv_metrics"])
            + f" | Validation={baseline_result['validation_verdict']}"
        )
        log_status("Baseline validation metrics: " + format_metrics_summary(baseline_result["val_metrics"]))
        return 0
    except Exception as exc:
        log_status(f"Baseline training failed: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
