"""Baseline training utilities and entry point for AutoCivil-Lab."""

from __future__ import annotations

import json
import os
import pickle
import random
import re
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


def build_pickle_artifact_metadata(
    obj: Any,
    config: dict[str, Any] | None = None,
    model_id: str | None = None,
) -> dict[str, Any]:
    """Build metadata stored alongside pickled model artifacts."""
    return {
        "artifact_schema_version": 1,
        "library_versions": get_runtime_library_versions(),
        "saved_at": pd.Timestamp.now().isoformat(),
        "model_id": model_id or type(obj).__name__,
        "config_hash": compute_config_hash(config),
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
) -> None:
    """Serialize an object as a pickle file with backward-compatible metadata."""
    artifact_bundle = {
        "artifact_metadata": build_pickle_artifact_metadata(obj, config=config, model_id=model_id),
        "payload": obj,
    }
    with path.open("wb") as handle:
        pickle.dump(artifact_bundle, handle)


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
