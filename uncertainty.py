"""Uncertainty estimation utilities for AutoCivil-Lab."""

from __future__ import annotations

import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import train_test_split

from feature_engineering import build_engineering_features
from train import (
    build_stratification_bins,
    get_base_input_columns,
    get_input_columns,
    get_outputs_dir,
    get_project_root,
    load_config,
    load_dataset,
    log_status,
    save_json_artifact,
    set_global_seed,
    split_dataset,
)

try:
    from lightgbm import LGBMRegressor
except ImportError:
    LGBMRegressor = None


class UncertaintyEstimator:
    """Estimate prediction intervals with quantile and conformal methods."""

    def __init__(
        self,
        model_path: str | Path | None = None,
        config_path: str | Path | None = None,
        method: str | None = None,
    ) -> None:
        """Load configuration, data, and interval estimators."""
        self.project_root = get_project_root()
        self.config = self._load_config(config_path)
        self.outputs_dir = get_outputs_dir(self.config)
        self.model_path = self._resolve_model_path(model_path)
        self.method = str(
            method or self.config.get("engineering", {}).get("uncertainty_method", "conformal")
        ).strip().lower()
        self.seed = int(self.config["experiment"]["random_seed"])
        self.coverage_level = float(self.config["uncertainty"]["coverage_level"])
        self.feature_columns = get_input_columns(self.config)
        self.base_columns = get_base_input_columns(self.config)
        self.tight_threshold = float(self.config["uncertainty"]["tight_threshold_mpa"])
        self.wide_threshold = float(self.config["uncertainty"]["wide_threshold_mpa"])
        self.quantile_models: dict[str, Any] = {}
        self.conformal_model: Any | None = None
        self.conformal_quantile: float | None = None

        if self.method not in {"conformal", "quantile", "both"}:
            raise ValueError(
                "engineering.uncertainty_method must be one of: conformal, quantile, both."
            )
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Best search model not found at {self.model_path}. Run search.py before uncertainty.py."
            )

        set_global_seed(self.seed)
        self.loaded_model = self._load_pickle(self.model_path)
        dataset = load_dataset(self.config)
        x_train_full, x_test, y_train_full, y_test = split_dataset(dataset, self.config)
        calibration_fraction = float(self.config["uncertainty"]["calibration_fraction"])
        calibration_bins = build_stratification_bins(
            y_train_full,
            max(2, int(self.config["data"]["stratify_bins"])),
        )
        self.x_train, self.x_calibration, self.y_train, self.y_calibration = train_test_split(
            x_train_full,
            y_train_full,
            test_size=calibration_fraction,
            random_state=self.seed,
            stratify=calibration_bins,
        )
        self.x_test = x_test
        self.y_test = y_test
        self._fit_estimators()

    def _load_config(self, config_path: str | Path | None) -> dict[str, Any]:
        """Load the project configuration from disk."""
        if config_path is None:
            return load_config()
        resolved_path = Path(config_path)
        if not resolved_path.is_absolute():
            resolved_path = self.project_root / resolved_path
        if not resolved_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {resolved_path}")
        return load_config() if resolved_path == self.project_root / "config.yaml" else self._read_yaml(resolved_path)

    @staticmethod
    def _read_yaml(path: Path) -> dict[str, Any]:
        """Read a YAML configuration file."""
        import yaml

        with path.open("r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
        if not isinstance(config, dict):
            raise ValueError("Configuration file did not parse into a dictionary.")
        return config

    def _resolve_model_path(self, model_path: str | Path | None) -> Path:
        """Resolve the model path relative to the project root."""
        if model_path is None:
            return self.project_root / self.config["paths"]["outputs_dir"] / "best_search_model.pkl"
        resolved_path = Path(model_path)
        if not resolved_path.is_absolute():
            resolved_path = self.project_root / resolved_path
        return resolved_path

    @staticmethod
    def _load_pickle(path: Path) -> Any:
        """Load a pickle artifact from disk."""
        with path.open("rb") as handle:
            return pickle.load(handle)

    def _fit_estimators(self) -> None:
        """Fit the interval estimators required by the selected method."""
        if self.method in {"conformal", "both"}:
            self.conformal_model = clone(self.loaded_model)
            self.conformal_model.fit(self.x_train, self.y_train)
            calibration_predictions = np.asarray(
                self.conformal_model.predict(self.x_calibration),
                dtype=float,
            )
            nonconformity_scores = np.abs(np.asarray(self.y_calibration, dtype=float) - calibration_predictions)
            alpha = 1.0 - self.coverage_level
            quantile_level = min(
                1.0,
                np.ceil((len(nonconformity_scores) + 1) * (1.0 - alpha)) / len(nonconformity_scores),
            )
            self.conformal_quantile = self._quantile(nonconformity_scores, quantile_level)

        if self.method in {"quantile", "both"}:
            if LGBMRegressor is None:
                raise ImportError(
                    "lightgbm is required for quantile uncertainty estimation. Install requirements.txt first."
                )
            quantile_config = dict(self.config["uncertainty"]["quantile_model"])
            quantile_config.update(
                {
                    "random_state": self.seed,
                    "verbosity": -1,
                    "n_jobs": -1,
                }
            )
            lower_alpha = float(self.config["uncertainty"]["lower_alpha"])
            median_alpha = float(self.config["uncertainty"]["median_alpha"])
            upper_alpha = float(self.config["uncertainty"]["upper_alpha"])
            for key, alpha in {
                "lower": lower_alpha,
                "median": median_alpha,
                "upper": upper_alpha,
            }.items():
                model = LGBMRegressor(objective="quantile", alpha=alpha, **quantile_config)
                model.fit(self.x_train, self.y_train)
                self.quantile_models[key] = model

    @staticmethod
    def _quantile(values: np.ndarray, quantile_level: float) -> float:
        """Return a stable empirical quantile across NumPy versions."""
        try:
            return float(np.quantile(values, quantile_level, method="higher"))
        except TypeError:
            return float(np.quantile(values, quantile_level, interpolation="higher"))

    def _prepare_input_frame(self, x: pd.DataFrame | np.ndarray) -> pd.DataFrame:
        """Normalize inference input into the trained feature schema."""
        if isinstance(x, pd.DataFrame):
            frame = x.copy()
        else:
            array = np.asarray(x, dtype=float)
            if array.ndim == 1:
                array = array.reshape(1, -1)
            if array.shape[1] == len(self.feature_columns):
                frame = pd.DataFrame(array, columns=self.feature_columns)
            elif array.shape[1] == len(self.base_columns):
                frame = pd.DataFrame(array, columns=self.base_columns)
            else:
                raise ValueError(
                    "Input array width does not match base or engineered feature column counts."
                )

        missing_feature_columns = [column for column in self.feature_columns if column not in frame.columns]
        if missing_feature_columns:
            if all(column in frame.columns for column in self.base_columns):
                frame = build_engineering_features(frame)
            else:
                raise ValueError(
                    f"Input data is missing required columns for interval prediction: {missing_feature_columns}"
                )
        return frame[self.feature_columns].copy()

    def _confidence_labels(self, interval_widths: np.ndarray) -> np.ndarray:
        """Map interval widths to human-readable confidence labels."""
        return np.where(
            interval_widths < self.tight_threshold,
            "TIGHT",
            np.where(interval_widths <= self.wide_threshold, "MODERATE", "WIDE"),
        )

    def _predict_conformal(self, x: pd.DataFrame) -> pd.DataFrame:
        """Predict conformal intervals."""
        if self.conformal_model is None or self.conformal_quantile is None:
            raise RuntimeError("Conformal estimator is not initialized.")
        predicted = np.asarray(self.conformal_model.predict(x), dtype=float)
        lower = predicted - self.conformal_quantile
        upper = predicted + self.conformal_quantile
        return pd.DataFrame(
            {
                "predicted": predicted,
                "lower_90": lower,
                "upper_90": upper,
            },
            index=x.index,
        )

    def _predict_quantile(self, x: pd.DataFrame) -> pd.DataFrame:
        """Predict quantile-regression intervals."""
        if not self.quantile_models:
            raise RuntimeError("Quantile estimator is not initialized.")
        lower = np.asarray(self.quantile_models["lower"].predict(x), dtype=float)
        predicted = np.asarray(self.quantile_models["median"].predict(x), dtype=float)
        upper = np.asarray(self.quantile_models["upper"].predict(x), dtype=float)
        return pd.DataFrame(
            {
                "predicted": predicted,
                "lower_90": np.minimum(lower, upper),
                "upper_90": np.maximum(lower, upper),
            },
            index=x.index,
        )

    def predict_with_interval(self, x: pd.DataFrame | np.ndarray) -> pd.DataFrame:
        """Return interval predictions and confidence labels for the provided inputs."""
        feature_frame = self._prepare_input_frame(x)

        if self.method == "conformal":
            result = self._predict_conformal(feature_frame)
        elif self.method == "quantile":
            result = self._predict_quantile(feature_frame)
        else:
            conformal_result = self._predict_conformal(feature_frame)
            quantile_result = self._predict_quantile(feature_frame)
            result = pd.DataFrame(
                {
                    "predicted": quantile_result["predicted"].to_numpy(dtype=float),
                    "lower_90": np.minimum(
                        conformal_result["lower_90"].to_numpy(dtype=float),
                        quantile_result["lower_90"].to_numpy(dtype=float),
                    ),
                    "upper_90": np.maximum(
                        conformal_result["upper_90"].to_numpy(dtype=float),
                        quantile_result["upper_90"].to_numpy(dtype=float),
                    ),
                },
                index=feature_frame.index,
            )

        interval_width = result["upper_90"].to_numpy(dtype=float) - result["lower_90"].to_numpy(dtype=float)
        result["interval_width"] = interval_width
        result["confidence_label"] = self._confidence_labels(interval_width)
        return result

    def calibration_report(self) -> dict[str, Any]:
        """Calculate and save calibration statistics for the configured interval method."""
        interval_frame = self.predict_with_interval(self.x_test)
        covered = (
            (self.y_test.to_numpy(dtype=float) >= interval_frame["lower_90"].to_numpy(dtype=float))
            & (self.y_test.to_numpy(dtype=float) <= interval_frame["upper_90"].to_numpy(dtype=float))
        )
        coverage = float(np.mean(covered))
        mean_interval_width = float(interval_frame["interval_width"].mean())
        reliability_bins = min(int(self.config["uncertainty"]["reliability_bins"]), len(interval_frame))
        prediction_bins = pd.qcut(
            interval_frame["predicted"].rank(method="first"),
            q=max(1, reliability_bins),
            labels=False,
            duplicates="drop",
        )
        reliability_frame = pd.DataFrame(
            {
                "predicted": interval_frame["predicted"],
                "interval_width": interval_frame["interval_width"],
                "covered": covered.astype(float),
                "bin_id": prediction_bins,
            }
        )
        grouped = (
            reliability_frame.groupby("bin_id", dropna=False)
            .agg(
                predicted_min=("predicted", "min"),
                predicted_max=("predicted", "max"),
                observed_coverage=("covered", "mean"),
                mean_interval_width=("interval_width", "mean"),
                sample_count=("covered", "size"),
            )
            .reset_index(drop=True)
        )
        report = {
            "method": self.method,
            "coverage_target": self.coverage_level,
            "coverage": coverage,
            "mean_interval_width": mean_interval_width,
            "sharpness": mean_interval_width,
            "calibration_sample_count": int(len(self.y_calibration)),
            "test_sample_count": int(len(self.y_test)),
            "reliability_plot_data": grouped.to_dict(orient="records"),
        }
        report_path = self.outputs_dir / "uncertainty_calibration.json"
        save_json_artifact(report_path, report)
        log_status(
            f"Saved uncertainty calibration report to {report_path} | "
            f"Coverage={coverage:.3f} | Mean interval width={mean_interval_width:.3f}"
        )
        return report


def main() -> int:
    """Generate the configured uncertainty calibration report."""
    try:
        estimator = UncertaintyEstimator()
        estimator.calibration_report()
        return 0
    except Exception as exc:
        log_status(f"Uncertainty estimation failed: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
