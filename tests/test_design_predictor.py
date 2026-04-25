import unittest

import numpy as np
import pandas as pd

from feature_engineering import ENGINEERED_FEATURE_COLUMNS, build_engineering_features
from mix_design.contracts import DesignContext
from mix_design.predictor import MixPerformancePredictor


class StubModel:
    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        effective_binder = frame["effective_binder"].to_numpy(dtype=float)
        water_effective_binder_ratio = frame["water_effective_binder_ratio"].to_numpy(dtype=float)
        return (0.11 * effective_binder) - (22.0 * water_effective_binder_ratio) - 3.0


class StubUncertaintyEstimator:
    method = "conformal"
    coverage_level = 0.92

    def __init__(self, model: StubModel, config: dict) -> None:
        self._model = model
        self._config = config

    def predict_with_interval(self, frame: pd.DataFrame) -> pd.DataFrame:
        engineered = build_engineering_features(frame, config=self._config)
        predicted = self._model.predict(engineered)
        return pd.DataFrame(
            {
                "predicted": predicted,
                "lower_90": predicted - 2.0,
                "upper_90": predicted + 2.0,
                "interval_width": np.full(len(predicted), 4.0),
                "confidence_label": ["HIGH"] * len(predicted),
            }
        )


def make_config() -> dict:
    return {
        "engineering": {
            "binder_efficiency": {"fly_ash_k": 0.35, "slag_k": 0.80},
        },
        "task": {
            "input_columns": [
                "cement",
                "slag",
                "fly_ash",
                "water",
                "superplasticizer",
                "coarse_aggregate",
                "fine_aggregate",
                "age",
            ]
        },
    }


class DesignPredictorTests(unittest.TestCase):
    def test_predictor_returns_engineered_features_and_uncertainty(self) -> None:
        config = make_config()
        predictor = MixPerformancePredictor(
            model=StubModel(),
            config=config,
            base_columns=config["task"]["input_columns"],
            feature_columns=config["task"]["input_columns"] + list(ENGINEERED_FEATURE_COLUMNS),
            model_artifact_id="model-artifact-1",
            uncertainty_estimator=StubUncertaintyEstimator(StubModel(), config),
        )

        result = predictor.predict(
            mix_design={
                "cement": 170.0,
                "slag": 110.0,
                "fly_ash": 35.0,
                "water": 168.0,
                "superplasticizer": 7.0,
                "coarse_aggregate": 1015.0,
                "fine_aggregate": 770.0,
                "age": 28.0,
            },
            target_strength_mpa=13.0,
            tolerance_mpa=2.0,
            context=DesignContext(exposure_class="marine", structural_application="column"),
        )

        self.assertIn("water_cement_ratio", result.engineered_features)
        self.assertEqual(result.uncertainty_interval.confidence_label, "HIGH")
        self.assertEqual(result.model_artifact_id, "model-artifact-1")
        self.assertGreater(result.uncertainty_interval.target_window_overlap, 0.0)

    def test_predictor_marks_missing_estimator_as_uncalibrated(self) -> None:
        config = make_config()
        predictor = MixPerformancePredictor(
            model=StubModel(),
            config=config,
            base_columns=config["task"]["input_columns"],
            feature_columns=config["task"]["input_columns"] + list(ENGINEERED_FEATURE_COLUMNS),
            model_artifact_id="model-artifact-2",
            uncertainty_estimator=None,
        )

        result = predictor.predict(
            mix_design={
                "cement": 170.0,
                "slag": 110.0,
                "fly_ash": 35.0,
                "water": 168.0,
                "superplasticizer": 7.0,
                "coarse_aggregate": 1015.0,
                "fine_aggregate": 770.0,
                "age": 28.0,
            },
            target_strength_mpa=25.0,
            tolerance_mpa=2.0,
        )

        self.assertEqual(result.uncertainty_interval.interval_width, 4.0)
        self.assertEqual(result.uncertainty_interval.confidence_label, "UNCALIBRATED")
        self.assertFalse(result.uncertainty_interval.is_calibrated)
        self.assertTrue(result.uncertainty_interval.warning_reasons)

    def test_predictor_warns_when_interval_is_wide(self) -> None:
        config = make_config()
        config["uncertainty"] = {"wide_threshold_mpa": 3.0}
        predictor = MixPerformancePredictor(
            model=StubModel(),
            config=config,
            base_columns=config["task"]["input_columns"],
            feature_columns=config["task"]["input_columns"] + list(ENGINEERED_FEATURE_COLUMNS),
            model_artifact_id="model-artifact-3",
            uncertainty_estimator=StubUncertaintyEstimator(StubModel(), config),
        )

        result = predictor.predict(
            mix_design={
                "cement": 170.0,
                "slag": 110.0,
                "fly_ash": 35.0,
                "water": 168.0,
                "superplasticizer": 7.0,
                "coarse_aggregate": 1015.0,
                "fine_aggregate": 770.0,
                "age": 28.0,
            },
            target_strength_mpa=13.0,
            tolerance_mpa=2.0,
        )

        self.assertTrue(any("wide threshold" in reason for reason in result.uncertainty_interval.warning_reasons))


if __name__ == "__main__":
    unittest.main()
