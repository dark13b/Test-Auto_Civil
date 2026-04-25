"""Prediction layer for the concrete mix design subsystem."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import pandas as pd

from feature_engineering import build_engineering_features
from mix_design.contracts import DesignContext, PredictionResult, UncertaintyInterval


@dataclass
class MixPerformancePredictor:
    """Run model inference and attach uncertainty for one candidate mix."""

    model: Any
    config: dict[str, Any]
    base_columns: list[str]
    feature_columns: list[str]
    model_artifact_id: str
    uncertainty_estimator: Any | None = None

    def _normalize_context(self, context: DesignContext | None) -> dict[str, str]:
        if context is None:
            return {}
        normalized: dict[str, str] = {}
        if context.exposure_class:
            normalized["exposure_class"] = str(context.exposure_class).strip().lower().replace(" ", "_")
        if context.structural_application:
            normalized["structural_application"] = str(context.structural_application).strip().lower().replace(" ", "_")
        return normalized

    def build_candidate_frame(
        self,
        mix_design: Mapping[str, float],
        context: DesignContext | None = None,
    ) -> pd.DataFrame:
        """Convert one candidate mix into the predictor input frame."""

        normalized_mix = {column: float(mix_design[column]) for column in self.base_columns}
        for column in ("slag", "fly_ash"):
            if abs(normalized_mix[column]) < 1.0:
                normalized_mix[column] = 0.0
        normalized_mix.update(self._normalize_context(context))
        return pd.DataFrame([normalized_mix])

    def _frame_from_mix(
        self,
        mix_design: Mapping[str, float],
        context: DesignContext | None = None,
    ) -> pd.DataFrame:
        """Backward-compatible alias for callers migrated from the legacy tool."""

        return self.build_candidate_frame(mix_design, context=context)

    def predict_strengths(
        self,
        mix_designs: Sequence[Mapping[str, float]],
        *,
        context: DesignContext | None = None,
    ) -> list[float]:
        """Predict strengths for several candidate mixes without packaging intervals."""

        if not mix_designs:
            return []
        frame = pd.concat(
            [self.build_candidate_frame(mix_design, context=context) for mix_design in mix_designs],
            ignore_index=True,
        )
        engineered = build_engineering_features(frame, config=self.config)
        predictions = self.model.predict(engineered[self.feature_columns])
        return [float(value) for value in predictions]

    def _build_uncertainty_interval(
        self,
        candidate_frame: pd.DataFrame,
        predicted_strength: float,
        target_strength_mpa: float,
        tolerance_mpa: float,
    ) -> UncertaintyInterval:
        warning_reasons: list[str] = []
        is_calibrated = True
        if self.uncertainty_estimator is None:
            is_calibrated = False
            configured_wide_threshold = (
                self.config.get("uncertainty", {}).get("wide_threshold_mpa")
                if isinstance(self.config.get("uncertainty", {}), dict)
                else None
            )
            fallback_width = max(
                float(tolerance_mpa) * 2.0,
                float(configured_wide_threshold or 0.0),
            )
            if fallback_width <= 0.0:
                fallback_width = 1.0
            half_width = fallback_width / 2.0
            lower = predicted_strength - half_width
            upper = predicted_strength + half_width
            width = upper - lower
            confidence_label = "UNCALIBRATED"
            warning_reasons.append(
                "Calibrated uncertainty estimator is unavailable; interval is a conservative fallback, not a validated 90% interval."
            )
        else:
            interval_frame = self.uncertainty_estimator.predict_with_interval(candidate_frame)
            required_columns = {"predicted", "lower_90", "upper_90", "interval_width"}
            missing_columns = sorted(required_columns.difference(interval_frame.columns))
            if missing_columns:
                raise RuntimeError(
                    "Uncertainty estimator returned an incomplete interval payload: "
                    + ", ".join(missing_columns)
                )
            lower = float(interval_frame.iloc[0]["lower_90"])
            upper = float(interval_frame.iloc[0]["upper_90"])
            predicted_strength = float(interval_frame.iloc[0]["predicted"])
            width = float(interval_frame.iloc[0]["interval_width"])
            confidence_label = str(interval_frame.iloc[0].get("confidence_label", "UNKNOWN"))

        target_lower = float(target_strength_mpa) - float(tolerance_mpa)
        target_upper = float(target_strength_mpa) + float(tolerance_mpa)
        overlap = max(0.0, min(upper, target_upper) - max(lower, target_lower))
        window_width = max(target_upper - target_lower, 1e-6)
        uncertainty_config = self.config.get("uncertainty", {})
        wide_threshold = None
        if isinstance(uncertainty_config, dict) and uncertainty_config.get("wide_threshold_mpa") is not None:
            wide_threshold = float(uncertainty_config["wide_threshold_mpa"])
        if wide_threshold is not None and float(width) >= wide_threshold:
            warning_reasons.append(
                f"Uncertainty interval width is {float(width):.2f} MPa, meeting or exceeding the configured wide threshold of {wide_threshold:.2f} MPa."
            )
        normalized_label = str(confidence_label).strip().upper()
        if normalized_label in {"LOW", "UNKNOWN", "UNCALIBRATED"}:
            warning_reasons.append(f"Uncertainty confidence is {normalized_label}.")
        return UncertaintyInterval(
            predicted=float(predicted_strength),
            lower_90=float(lower),
            upper_90=float(upper),
            interval_width=float(width),
            confidence_label=confidence_label,
            target_window_overlap=float(overlap / window_width),
            is_calibrated=is_calibrated,
            warning_reasons=tuple(warning_reasons),
        )

    def predict(
        self,
        mix_design: Mapping[str, float],
        *,
        target_strength_mpa: float,
        tolerance_mpa: float,
        context: DesignContext | None = None,
    ) -> PredictionResult:
        """Predict performance for one mix candidate."""

        candidate_frame = self.build_candidate_frame(mix_design, context=context)
        engineered = build_engineering_features(candidate_frame, config=self.config)
        predicted_strength = float(self.model.predict(engineered[self.feature_columns])[0])
        interval = self._build_uncertainty_interval(
            candidate_frame=candidate_frame,
            predicted_strength=predicted_strength,
            target_strength_mpa=target_strength_mpa,
            tolerance_mpa=tolerance_mpa,
        )
        engineered_row = engineered.iloc[0]
        engineered_features = {
            column: float(engineered_row[column])
            for column in engineered.columns
            if column not in self.base_columns and hasattr(engineered_row[column], "__float__")
        }
        return PredictionResult(
            predicted_strength_mpa=float(interval.predicted),
            engineered_features=engineered_features,
            uncertainty_interval=interval,
            model_artifact_id=str(self.model_artifact_id),
        )
