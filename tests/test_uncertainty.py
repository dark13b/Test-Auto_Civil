import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from uncertainty import UncertaintyEstimator


def make_config() -> dict:
    return {
        "paths": {"outputs_dir": "outputs"},
        "experiment": {"random_seed": 42},
        "engineering": {"uncertainty_method": "conformal", "feature_engineering": False},
        "task": {"input_columns": ["cement", "water"], "target_column": "strength"},
        "uncertainty": {
            "coverage_level": 0.90,
            "tight_threshold_mpa": 5.0,
            "wide_threshold_mpa": 10.0,
            "reliability_bins": 4,
            "lower_alpha": 0.05,
            "median_alpha": 0.5,
            "upper_alpha": 0.95,
            "quantile_model": {},
        },
    }


class UncertaintyTests(unittest.TestCase):
    def _build_estimator(self) -> tuple[UncertaintyEstimator, LinearRegression, pd.DataFrame, pd.Series]:
        config = make_config()
        x_train = pd.DataFrame({"cement": np.linspace(100, 180, 24), "water": np.linspace(160, 210, 24)})
        y_train = pd.Series(12 + (0.11 * x_train["cement"]) - (0.03 * x_train["water"]))
        x_val = pd.DataFrame({"cement": np.linspace(105, 175, 12), "water": np.linspace(162, 205, 12)})
        y_val = pd.Series(12 + (0.11 * x_val["cement"]) - (0.03 * x_val["water"]))
        x_test = pd.DataFrame({"cement": np.linspace(110, 170, 6), "water": np.linspace(164, 202, 6)})
        y_test = pd.Series(12 + (0.11 * x_test["cement"]) - (0.03 * x_test["water"]))
        model = LinearRegression().fit(x_train, y_train)

        with TemporaryDirectory() as tmpdir, patch("uncertainty.load_config", return_value=config), patch(
            "uncertainty.load_dataset",
            return_value=pd.DataFrame(),
        ), patch(
            "uncertainty.split_dataset",
            return_value=(x_train, x_val, x_test, y_train, y_val, y_test),
        ), patch("uncertainty.set_global_seed"), patch(
            "uncertainty.get_project_root",
            return_value=Path(tmpdir),
        ):
            estimator = UncertaintyEstimator(model=model, report_model=model, outputs_dir=Path(tmpdir) / "outputs")

        return estimator, model, x_test, y_test

    def test_interval_centers_match_report_model_predictions(self) -> None:
        estimator, model, x_test, _ = self._build_estimator()

        interval_frame = estimator.predict_with_interval(x_test)
        point_predictions = model.predict(x_test)

        np.testing.assert_allclose(interval_frame["predicted"].to_numpy(dtype=float), point_predictions)

    def test_validation_coverage_meets_configured_target(self) -> None:
        estimator, _, _, _ = self._build_estimator()

        report = estimator.calibration_report()

        self.assertGreaterEqual(report["coverage"], 0.90)

    def test_refit_request_is_rejected(self) -> None:
        config = make_config()
        model = LinearRegression()
        with TemporaryDirectory() as tmpdir, patch("uncertainty.load_config", return_value=config), patch(
            "uncertainty.load_dataset",
            return_value=pd.DataFrame(),
        ), patch(
            "uncertainty.split_dataset",
            return_value=(
                pd.DataFrame({"cement": [1.0], "water": [1.0]}),
                pd.DataFrame({"cement": [1.0], "water": [1.0]}),
                pd.DataFrame({"cement": [1.0], "water": [1.0]}),
                pd.Series([1.0]),
                pd.Series([1.0]),
                pd.Series([1.0]),
            ),
        ), patch("uncertainty.set_global_seed"), patch(
            "uncertainty.get_project_root",
            return_value=Path(tmpdir),
        ):
            with self.assertRaises(ValueError):
                UncertaintyEstimator(
                    model=model,
                    report_model=model,
                    outputs_dir=Path(tmpdir) / "outputs",
                    refit_model=True,
                )


if __name__ == "__main__":
    unittest.main()
