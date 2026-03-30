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
        if self.uncertainty_estimator is None:
            lower = predicted_strength - float(tolerance_mpa)
            upper = predicted_strength + float(tolerance_mpa)
            width = upper - lower
            confidence_label = "UNKNOWN"
        else:
            interval_frame = self.uncertainty_estimator.predict_with_interval(candidate_frame)
            lower = float(interval_frame.iloc[0]["lower_90"])
            upper = float(interval_frame.iloc[0]["upper_90"])
            predicted_strength = float(interval_frame.iloc[0]["predicted"])
            width = float(interval_frame.iloc[0]["interval_width"])
            confidence_label = str(interval_frame.iloc[0].get("confidence_label", "UNKNOWN"))

        target_lower = float(target_strength_mpa) - float(tolerance_mpa)
        target_upper = float(target_strength_mpa) + float(tolerance_mpa)
        overlap = max(0.0, min(upper, target_upper) - max(lower, target_lower))
        window_width = max(target_upper - target_lower, 1e-6)
        return UncertaintyInterval(
            predicted=float(predicted_strength),
            lower_90=float(lower),
            upper_90=float(upper),
            interval_width=float(width),
            confidence_label=confidence_label,
            target_window_overlap=float(overlap / window_width),
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
