import unittest

import numpy as np
import pandas as pd

from train import (
    RegimeEnsembleRegressor,
    build_regime_blend_weights,
    fit_regime_models,
    identify_strength_regime,
)


class _MeanRegressor:
    def __init__(self) -> None:
        self.mean_: float | None = None
        self.fit_rows: int = 0

    def fit(self, x_frame: pd.DataFrame, y_values: pd.Series) -> "_MeanRegressor":
        self.fit_rows = len(x_frame)
        self.mean_ = float(np.asarray(y_values, dtype=float).mean())
        return self

    def predict(self, x_frame: pd.DataFrame) -> np.ndarray:
        if self.mean_ is None:
            raise AssertionError("Model must be fitted before prediction.")
        return np.full(len(x_frame), self.mean_, dtype=float)


class RegimeModelingTests(unittest.TestCase):
    def test_identify_strength_regime_uses_requested_thresholds(self) -> None:
        self.assertEqual(identify_strength_regime(25.0), "low")
        self.assertEqual(identify_strength_regime(25.01), "mid")
        self.assertEqual(identify_strength_regime(60.0), "mid")
        self.assertEqual(identify_strength_regime(60.01), "high")

    def test_build_regime_blend_weights_softens_predictions_near_boundaries(self) -> None:
        weights = build_regime_blend_weights(
            np.asarray([24.0, 25.0, 26.0, 59.0, 60.0, 61.0], dtype=float),
            transition_width_mpa=2.0,
        )

        self.assertAlmostEqual(weights[0]["low"], 0.75, places=6)
        self.assertAlmostEqual(weights[0]["mid"], 0.25, places=6)
        self.assertAlmostEqual(weights[1]["low"], 0.5, places=6)
        self.assertAlmostEqual(weights[1]["mid"], 0.5, places=6)
        self.assertAlmostEqual(weights[2]["low"], 0.25, places=6)
        self.assertAlmostEqual(weights[2]["mid"], 0.75, places=6)
        self.assertAlmostEqual(weights[3]["mid"], 0.75, places=6)
        self.assertAlmostEqual(weights[3]["high"], 0.25, places=6)
        self.assertAlmostEqual(weights[4]["mid"], 0.5, places=6)
        self.assertAlmostEqual(weights[4]["high"], 0.5, places=6)
        self.assertAlmostEqual(weights[5]["mid"], 0.25, places=6)
        self.assertAlmostEqual(weights[5]["high"], 0.75, places=6)

    def test_fit_regime_models_trains_one_model_per_regime_partition(self) -> None:
        x_train = pd.DataFrame({"cement": [100, 110, 200, 210, 300, 310]})
        y_train = pd.Series([20.0, 24.0, 35.0, 55.0, 65.0, 80.0])

        regime_models = fit_regime_models(
            x_train,
            y_train,
            model_factory=_MeanRegressor,
        )

        self.assertEqual(set(regime_models), {"low", "mid", "high"})
        self.assertEqual(regime_models["low"].fit_rows, 2)
        self.assertEqual(regime_models["mid"].fit_rows, 2)
        self.assertEqual(regime_models["high"].fit_rows, 2)

    def test_regime_ensemble_regressor_blends_boundary_predictions(self) -> None:
        ensemble = RegimeEnsembleRegressor(
            regime_models={
                "low": _MeanRegressor(),
                "mid": _MeanRegressor(),
                "high": _MeanRegressor(),
            },
            transition_width_mpa=2.0,
        )
        ensemble.regime_models["low"].mean_ = 20.0
        ensemble.regime_models["mid"].mean_ = 40.0
        ensemble.regime_models["high"].mean_ = 80.0
        ensemble.feature_names_in_ = np.asarray(["cement"])
        ensemble.n_features_in_ = 1

        x_frame = pd.DataFrame({"cement": [1.0, 2.0, 3.0, 4.0]})
        blended = ensemble.predict(x_frame, base_predictions=np.asarray([24.0, 26.0, 59.0, 61.0]))

        self.assertAlmostEqual(blended[0], 25.0, places=6)
        self.assertAlmostEqual(blended[1], 35.0, places=6)
        self.assertAlmostEqual(blended[2], 50.0, places=6)
        self.assertAlmostEqual(blended[3], 70.0, places=6)


if __name__ == "__main__":
    unittest.main()
