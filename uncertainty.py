"""Uncertainty estimation utilities for AutoCivil-Lab."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from feature_engineering import build_engineering_features
from train import (
    artifact_id,
    compute_file_hash,
    create_run_id,
    get_base_input_columns,
    get_input_columns,
    get_outputs_dir,
    get_project_root,
    infer_active_run_id,
    load_pickle_artifact,
    load_config,
    load_dataset,
    log_status,
    read_pickle_artifact_metadata,
    resolve_model_feature_columns,
    save_json_artifact,
    set_global_seed,
    split_dataset,
    write_run_scoped_json_artifact,
)

try:
    from lightgbm import LGBMRegressor
except ImportError:
    LGBMRegressor = None


class UncertaintyEstimator:
    """Estimate intervals without refitting the report model.

    This estimator does not refit the model. It calibrates intervals on the
    validation partition using the exact model object passed at construction.
    The conformal path is strength-aware: it learns a residual scale curve
    across predicted strength bins, then conformalizes standardized residuals.
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        model: Any | None = None,
        report_model: Any | None = None,
        config_path: str | Path | None = None,
        method: str | None = None,
        outputs_dir: str | Path | None = None,
        report_filename: str = "uncertainty_calibration.json",
        refit_model: bool = False,
    ) -> None:
        """Load configuration, data, and interval estimators."""
        self.project_root = get_project_root()
        self.config = self._load_config(config_path)
        self.outputs_dir = self._resolve_outputs_dir(outputs_dir)
        self.model_path = self._resolve_model_path(model_path)
        self.current_best_payload = self._load_json_payload(self.outputs_dir / "best_search_result.json")
        self.method = str(
            method or self.config.get("engineering", {}).get("uncertainty_method", "conformal")
        ).strip().lower()
        self.report_filename = str(report_filename)
        self.seed = int(self.config["experiment"]["random_seed"])
        self.coverage_level = max(0.92, float(self.config["uncertainty"]["coverage_level"]))
        self.base_columns = get_base_input_columns(self.config)
        self.tight_threshold = float(self.config["uncertainty"]["tight_threshold_mpa"])
        self.wide_threshold = float(self.config["uncertainty"]["wide_threshold_mpa"])
        self.quantile_models: dict[str, Any] = {}
        self.conformal_model: Any | None = None
        self.conformal_global_quantile: float | None = None
        self.conformal_global_scale: float | None = None
        self.conformal_bin_edges: np.ndarray | None = None
        self.conformal_bin_scales: dict[int, float] = {}
        self.conformal_bin_relative_scales: dict[int, float] = {}
        self.conformal_bin_centers: dict[int, float] = {}
        self.conformal_bin_quantiles: dict[int, float] = {}
        self.conformal_bin_calibration_coverage: dict[int, float] = {}
        self.uses_explicit_strength_bins = False

        if self.method not in {"conformal", "quantile", "both"}:
            raise ValueError(
                "engineering.uncertainty_method must be one of: conformal, quantile, both."
            )
        if refit_model:
            raise ValueError("UncertaintyEstimator does not support refitting the model.")
        if model is None and not self.model_path.exists():
            raise FileNotFoundError(
                f"Best search model not found at {self.model_path}. Run search.py before uncertainty.py."
            )

        set_global_seed(self.seed)
        self.model = model if model is not None else load_pickle_artifact(self.model_path)
        self.model_metadata = read_pickle_artifact_metadata(self.model_path) or {}
        self.model_artifact_id = str(
            self.model_metadata.get("artifact_id") or compute_file_hash(self.model_path) or "unknown-model-artifact"
        )
        self.active_run_id = infer_active_run_id(
            self.outputs_dir,
            self.current_best_payload,
            fallback=str(self.model_metadata.get("run_id") or create_run_id()),
        )
        self.report_model = report_model if report_model is not None else self.model
        self.feature_columns = resolve_model_feature_columns(self.model, self.config)
        assert id(self.model) == id(self.report_model), (
            "Uncertainty model and report model must be the same fitted object."
        )
        dataset = load_dataset(self.config)
        self.x_train, self.x_calibration, _, self.y_train, self.y_calibration, _ = split_dataset(
            dataset,
            self.config,
        )
        self.x_train = self.x_train[self.feature_columns].copy()
        self.x_calibration = self.x_calibration[self.feature_columns].copy()
        self.x_validation = self.x_calibration
        self.y_validation = self.y_calibration
        self._fit_estimators()

    @staticmethod
    def _load_json_payload(path: Path) -> dict[str, Any]:
        """Load a JSON artifact when present, otherwise return an empty dictionary."""
        if not path.exists():
            return {}
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return payload if isinstance(payload, dict) else {}

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

    def _resolve_outputs_dir(self, outputs_dir: str | Path | None) -> Path:
        """Resolve the output directory relative to the project root."""
        if outputs_dir is None:
            return get_outputs_dir(self.config)
        resolved_path = Path(outputs_dir)
        if not resolved_path.is_absolute():
            resolved_path = self.project_root / resolved_path
        resolved_path.mkdir(parents=True, exist_ok=True)
        return resolved_path

    def _fit_estimators(self) -> None:
        """Fit the interval estimators required by the selected method."""
        if self.method in {"conformal", "both"}:
            self.conformal_model = self.model
            calibration_predictions = np.asarray(
                self.model.predict(self.x_calibration),
                dtype=float,
            )
            calibration_targets = np.asarray(self.y_calibration, dtype=float)
            nonconformity_scores = np.abs(calibration_targets - calibration_predictions)
            alpha = 1.0 - self.coverage_level
            quantile_level = min(
                1.0,
                np.ceil((len(nonconformity_scores) + 1) * (1.0 - alpha)) / len(nonconformity_scores),
            )
            self.conformal_global_scale = float(max(self._quantile(nonconformity_scores, 0.5), 1e-6))
            calibration_bins, bin_edges = self._build_strength_bins(calibration_predictions)
            self.conformal_bin_edges = bin_edges
            unique_bins = sorted(np.unique(calibration_bins).tolist())
            global_scale = float(self.conformal_global_scale)
            shrink_count = max(4.0, float(np.sqrt(max(len(nonconformity_scores), 1))))
            for bin_id in unique_bins:
                bin_mask = calibration_bins == bin_id
                bin_scores = nonconformity_scores[bin_mask]
                bin_predictions = calibration_predictions[bin_mask]
                if len(bin_scores) == 0:
                    fallback_scale = global_scale
                    self.conformal_bin_scales[int(bin_id)] = fallback_scale
                    self.conformal_bin_relative_scales[int(bin_id)] = fallback_scale / max(global_scale, 1e-6)
                    self.conformal_bin_centers[int(bin_id)] = global_scale
                    continue
                local_scale = float(max(self._quantile(bin_scores, 0.5), 1e-6))
                smoothed_scale = float(
                    (len(bin_scores) * local_scale + shrink_count * global_scale) / (len(bin_scores) + shrink_count)
                )
                self.conformal_bin_scales[int(bin_id)] = smoothed_scale
                bin_center = float(np.mean(bin_predictions)) if len(bin_predictions) else global_scale
                self.conformal_bin_centers[int(bin_id)] = bin_center
                self.conformal_bin_relative_scales[int(bin_id)] = smoothed_scale / max(bin_center, 1.0)

            standardized_scores = np.asarray(
                [
                    float(score) / max(self.conformal_bin_scales.get(int(bin_id), global_scale), 1e-6)
                    for score, bin_id in zip(nonconformity_scores, calibration_bins, strict=False)
                ],
                dtype=float,
            )
            self.conformal_global_quantile = self._quantile(standardized_scores, quantile_level)
            for bin_id in unique_bins:
                bin_mask = calibration_bins == bin_id
                bin_scores = nonconformity_scores[bin_mask]
                bin_scale = float(self.conformal_bin_scales.get(int(bin_id), global_scale))
                bin_width = float(self.conformal_global_quantile * bin_scale)
                observed_coverage = float(np.mean(bin_scores <= bin_width)) if len(bin_scores) else float(self.coverage_level)
                self.conformal_bin_quantiles[int(bin_id)] = bin_width
                self.conformal_bin_calibration_coverage[int(bin_id)] = observed_coverage

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

    def _build_strength_bins(self, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Construct quantile-based strength bins and reusable cut edges."""
        array = np.asarray(values, dtype=float)
        if array.size == 0:
            return np.asarray([], dtype=int), np.asarray([0.0, 1.0], dtype=float)

        configured_edges = self.config["uncertainty"].get("strength_bins_mpa", [])
        if isinstance(configured_edges, list) and len(configured_edges) >= 2:
            clean_edges = np.asarray(sorted({float(edge) for edge in configured_edges}), dtype=float)
            if clean_edges.size >= 2:
                self.uses_explicit_strength_bins = True
                return np.digitize(array, clean_edges[1:], right=True).astype(int), clean_edges

        requested_bins = max(1, min(int(self.config["uncertainty"]["reliability_bins"]), int(array.size)))
        self.uses_explicit_strength_bins = False
        try:
            bin_codes, bin_edges = pd.qcut(
                pd.Series(array),
                q=requested_bins,
                labels=False,
                retbins=True,
                duplicates="drop",
            )
        except ValueError:
            bin_codes = pd.Series(np.zeros(len(array), dtype=int))
            center = float(array[0])
            bin_edges = np.asarray([center - 1e-6, center + 1e-6], dtype=float)
        clean_edges = np.asarray(bin_edges, dtype=float)
        if clean_edges.size < 2:
            center = float(array[0])
            clean_edges = np.asarray([center - 1e-6, center + 1e-6], dtype=float)
        return np.asarray(pd.Series(bin_codes).fillna(0), dtype=int), clean_edges

    def _assign_strength_bins(self, values: np.ndarray) -> np.ndarray:
        """Assign predicted strengths to the fitted calibration bins."""
        array = np.asarray(values, dtype=float)
        if self.conformal_bin_edges is None or self.conformal_bin_edges.size < 2:
            return np.zeros(len(array), dtype=int)
        cut_edges = self.conformal_bin_edges[1:] if self.uses_explicit_strength_bins else self.conformal_bin_edges[1:-1]
        return np.digitize(array, cut_edges, right=True).astype(int)

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
                frame = build_engineering_features(frame, config=self.config)
            else:
                raise ValueError(
                    f"Input data is missing required columns for interval prediction: {missing_feature_columns}"
                )
        return frame[self.feature_columns].copy()

    def _confidence_labels(self, bin_coverages: np.ndarray) -> np.ndarray:
        """Map per-bin coverage rates to human-readable confidence labels."""
        return np.where(
            bin_coverages > 0.92,
            "HIGH",
            np.where(bin_coverages >= 0.85, "MODERATE", "LOW"),
        )

    def _predict_conformal(self, x: pd.DataFrame) -> pd.DataFrame:
        """Predict conformal intervals."""
        if self.conformal_global_quantile is None:
            raise RuntimeError("Conformal estimator is not initialized.")
        predicted = np.asarray(self.model.predict(x), dtype=float)
        strength_bins = self._assign_strength_bins(predicted)
        bin_scales = np.asarray(
            [
                float(self.conformal_bin_scales.get(int(bin_id), self.conformal_global_scale or 1.0))
                for bin_id in strength_bins
            ],
            dtype=float,
        )
        quantiles = np.asarray(
            [
                self.conformal_bin_quantiles.get(
                    int(bin_id),
                    float((self.conformal_global_quantile or 1.0) * (self.conformal_global_scale or 1.0)),
                )
                for bin_id in strength_bins
            ],
            dtype=float,
        )
        lower = predicted - quantiles
        upper = predicted + quantiles
        estimated_cv = bin_scales / np.maximum(np.abs(predicted), 1.0)
        return pd.DataFrame(
            {
                "predicted": predicted,
                "lower_90": lower,
                "upper_90": upper,
                "strength_bin": strength_bins,
                "bin_scale": bin_scales,
                "bin_quantile": quantiles,
                "estimated_cv": estimated_cv,
            },
            index=x.index,
        )

    def _predict_quantile(self, x: pd.DataFrame) -> pd.DataFrame:
        """Predict quantile-regression intervals."""
        if not self.quantile_models:
            raise RuntimeError("Quantile estimator is not initialized.")
        lower = np.asarray(self.quantile_models["lower"].predict(x), dtype=float)
        predicted = np.asarray(self.model.predict(x), dtype=float)
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
            result["strength_bin"] = self._assign_strength_bins(result["predicted"].to_numpy(dtype=float))
            result["bin_scale"] = np.nan
            result["bin_quantile"] = np.nan
            result["estimated_cv"] = np.nan
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
                    "strength_bin": conformal_result["strength_bin"].to_numpy(dtype=int),
                    "bin_scale": conformal_result["bin_scale"].to_numpy(dtype=float),
                    "bin_quantile": conformal_result["bin_quantile"].to_numpy(dtype=float),
                    "estimated_cv": conformal_result["estimated_cv"].to_numpy(dtype=float),
                },
                index=feature_frame.index,
            )

        interval_width = result["upper_90"].to_numpy(dtype=float) - result["lower_90"].to_numpy(dtype=float)
        result["interval_width"] = interval_width
        default_coverages = np.asarray(
            [
                self.conformal_bin_calibration_coverage.get(int(bin_id), float(self.coverage_level))
                for bin_id in result["strength_bin"].to_numpy(dtype=int)
            ],
            dtype=float,
        )
        result["bin_coverage"] = default_coverages
        result["confidence_label"] = self._confidence_labels(default_coverages)
        return result

    def calibration_report(self) -> dict[str, Any]:
        """Calculate and save calibration statistics for the configured interval method."""
        interval_frame = self.predict_with_interval(self.x_validation)
        covered = (
            (self.y_validation.to_numpy(dtype=float) >= interval_frame["lower_90"].to_numpy(dtype=float))
            & (self.y_validation.to_numpy(dtype=float) <= interval_frame["upper_90"].to_numpy(dtype=float))
        )
        coverage = float(np.mean(covered))
        mean_interval_width = float(interval_frame["interval_width"].mean())
        observed_bins, observed_bin_edges = self._build_strength_bins(np.asarray(self.y_validation, dtype=float))
        reliability_frame = pd.DataFrame(
            {
                "actual_strength": np.asarray(self.y_validation, dtype=float),
                "predicted": interval_frame["predicted"],
                "interval_width": interval_frame["interval_width"],
                "covered": covered.astype(float),
                "bin_id": observed_bins,
                "bin_scale": interval_frame["bin_scale"].to_numpy(dtype=float),
                "estimated_cv": interval_frame["estimated_cv"].to_numpy(dtype=float),
            }
        )
        grouped = (
            reliability_frame.groupby("bin_id", dropna=False)
            .agg(
                actual_strength_min=("actual_strength", "min"),
                actual_strength_max=("actual_strength", "max"),
                actual_strength_mean=("actual_strength", "mean"),
                predicted_min=("predicted", "min"),
                predicted_max=("predicted", "max"),
                observed_coverage=("covered", "mean"),
                mean_interval_width=("interval_width", "mean"),
                mean_bin_scale=("bin_scale", "mean"),
                mean_estimated_cv=("estimated_cv", "mean"),
                sample_count=("covered", "size"),
            )
            .reset_index()
        )
        grouped["coverage_flag"] = np.where(grouped["observed_coverage"] < 0.80, "LOW_COVERAGE", "OK")
        grouped["confidence_label"] = self._confidence_labels(
            grouped["observed_coverage"].to_numpy(dtype=float)
        )
        bin_coverages = {
            int(bin_id): float(observed_coverage)
            for bin_id, observed_coverage in zip(grouped["bin_id"], grouped["observed_coverage"])
        }
        interval_frame["covered"] = covered.astype(bool)
        interval_frame["observed_strength_bin"] = observed_bins
        interval_frame["bin_coverage"] = interval_frame["observed_strength_bin"].map(bin_coverages).astype(float)
        interval_frame["confidence_label"] = self._confidence_labels(
            interval_frame["bin_coverage"].to_numpy(dtype=float)
        )
        low_strength_bins = grouped.nsmallest(max(1, min(2, len(grouped))), columns="actual_strength_mean")
        high_strength_bins = grouped.nlargest(max(1, min(2, len(grouped))), columns="actual_strength_mean")
        audit_status = "FAIL" if coverage < 0.88 else "PASS"
        report = {
            "method": self.method,
            "coverage_target": self.coverage_level,
            "coverage": coverage,
            "mean_interval_width": mean_interval_width,
            "sharpness": mean_interval_width,
            "model_artifact_id": self.model_artifact_id,
            "calibration_sample_count": int(len(self.y_calibration)),
            "validation_sample_count": int(len(self.y_validation)),
            "reliability_plot_data": grouped.to_dict(orient="records"),
            "coverage_by_strength_bin": grouped.to_dict(orient="records"),
            "strength_bin_audit": {
                "bin_edges": observed_bin_edges.tolist(),
                "low_strength_bins": low_strength_bins.to_dict(orient="records"),
                "high_strength_bins": high_strength_bins.to_dict(orient="records"),
                "coverage_gap_low_minus_global": float(
                    low_strength_bins["observed_coverage"].mean() - coverage
                ),
            },
            "coverage_audit": {
                "global_status": audit_status,
                "global_coverage_pass_threshold": 0.88,
                "bin_coverage_pass_threshold": 0.80,
                "bins_below_threshold": grouped.loc[
                    grouped["observed_coverage"] < 0.80,
                    ["actual_strength_min", "actual_strength_max", "observed_coverage", "sample_count"],
                ].to_dict(orient="records"),
                "bin_details": grouped.to_dict(orient="records"),
            },
        }
        report_path = self.outputs_dir / self.report_filename
        _, _, enriched_report = write_run_scoped_json_artifact(
            outputs_dir=self.outputs_dir,
            filename=self.report_filename,
            payload=report,
            run_id=self.active_run_id,
            source_mode="uncertainty",
            config=self.config,
            model_artifact_id=self.model_artifact_id,
            model_id=str(type(self.model).__name__),
            parent_artifact_ids=[artifact_id(self.current_best_payload)] if artifact_id(self.current_best_payload) else [],
        )
        for row in grouped.to_dict(orient="records"):
            flag = row["coverage_flag"]
            log_status(
                "Coverage audit bin "
                f"{int(row['bin_id'])}: {row['predicted_min']:.2f}-{row['predicted_max']:.2f} MPa | "
                f"coverage={row['observed_coverage']:.3f} | n={int(row['sample_count'])} | {flag}"
            )
        if audit_status == "FAIL":
            log_status(
                f"FAIL uncertainty coverage audit | coverage={coverage:.3f} | required_min=0.880"
            )
        log_status(
            f"Saved uncertainty calibration report to {report_path} | "
            f"Coverage={coverage:.3f} | Mean interval width={mean_interval_width:.3f}"
        )
        return enriched_report

    def validate_coverage_target(self) -> bool:
        """Run the calibration audit and validate it against the configured target."""
        report = self.calibration_report()
        actual_coverage = float(report["coverage"])
        coverage_target = float(report["coverage_target"])
        meets_target = actual_coverage >= coverage_target
        if not meets_target:
            log_status(
                f"WARNING coverage_below_target | actual={actual_coverage:.4f} | target={coverage_target:.4f}"
            )
        return meets_target


def recalibrate_uncertainty_artifacts(
    model_path: str | Path | None = None,
    config_path: str | Path | None = None,
    method: str | None = None,
    outputs_dir: str | Path | None = None,
    report_filename: str = "uncertainty_calibration.json",
) -> dict[str, Any]:
    """Recompute and save uncertainty calibration artifacts for the current best model."""
    estimator = UncertaintyEstimator(
        model_path=model_path,
        config_path=config_path,
        method=method,
        outputs_dir=outputs_dir,
        report_filename=report_filename,
    )
    return estimator.calibration_report()


def main() -> int:
    """Generate the configured uncertainty calibration report."""
    try:
        estimator = UncertaintyEstimator()
        estimator.validate_coverage_target()
        return 0
    except Exception as exc:
        log_status(f"Uncertainty estimation failed: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
