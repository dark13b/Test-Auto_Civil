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
            "strength_bins_mpa": [0.0, 20.0, 35.0, 50.0],
            "lower_alpha": 0.05,
            "median_alpha": 0.5,
            "upper_alpha": 0.95,
            "quantile_model": {},
        },
    }


class UncertaintyTests(unittest.TestCase):
    def _make_heteroscedastic_data(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
        x_train = pd.DataFrame(
            {
                "cement": np.linspace(120, 320, 36),
                "water": np.linspace(230, 150, 36),
            }
        )
        x_val = pd.DataFrame(
            {
                "cement": np.linspace(130, 310, 18),
                "water": np.linspace(225, 155, 18),
            }
        )
        x_test = pd.DataFrame(
            {
                "cement": np.linspace(140, 300, 8),
                "water": np.linspace(220, 160, 8),
            }
        )

        def build_target(frame: pd.DataFrame) -> pd.Series:
            base = 0.12 * frame["cement"] - 0.04 * frame["water"] + 4.0
            heteroscedastic_noise = np.linspace(0.25, 3.5, len(frame))
            return pd.Series(base + heteroscedastic_noise, index=frame.index)

        return (
            x_train,
            x_val,
            x_test,
            build_target(x_train),
            build_target(x_val),
            build_target(x_test),
        )

    def _build_estimator(
        self,
        *,
        audit_partition: str = "validation_audit",
    ) -> tuple[UncertaintyEstimator, LinearRegression, pd.DataFrame, pd.Series]:
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
            estimator = UncertaintyEstimator(
                model=model,
                report_model=model,
                outputs_dir=Path(tmpdir) / "outputs",
                audit_partition=audit_partition,
            )

        return estimator, model, x_test, y_test

    def _build_heteroscedastic_estimator(self) -> tuple[UncertaintyEstimator, LinearRegression, pd.DataFrame]:
        config = make_config()
        x_train, x_val, x_test, y_train, y_val, y_test = self._make_heteroscedastic_data()
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

        return estimator, model, x_test

    def test_interval_centers_match_report_model_predictions(self) -> None:
        estimator, model, x_test, _ = self._build_estimator()

        interval_frame = estimator.predict_with_interval(x_test)
        point_predictions = model.predict(x_test)

        np.testing.assert_allclose(interval_frame["predicted"].to_numpy(dtype=float), point_predictions)

    def test_validation_coverage_meets_configured_target(self) -> None:
        estimator, _, _, _ = self._build_estimator()

        report = estimator.calibration_report()

        self.assertGreaterEqual(report["coverage"], 0.90)

    def test_calibration_and_validation_partitions_are_disjoint(self) -> None:
        estimator, _, _, _ = self._build_estimator()
        cal_indices = set(estimator.x_calibration.index.tolist())
        val_indices = set(estimator.x_validation.index.tolist())
        self.assertEqual(
            len(cal_indices & val_indices),
            0,
            "Calibration and coverage-audit partitions must be fully disjoint."
        )
        self.assertGreater(len(cal_indices), 0, "Calibration partition must be non-empty.")
        self.assertGreater(len(val_indices), 0, "Coverage audit partition must be non-empty.")

    def test_coverage_is_measured_on_audit_partition_not_calibration(self) -> None:
        estimator, _, _, _ = self._build_estimator()
        # The validation set used for coverage reporting must not overlap
        # with the calibration set used to fit the conformal quantile.
        cal_idx = set(estimator.x_calibration.index.tolist())
        val_idx = set(estimator.x_validation.index.tolist())
        self.assertTrue(
            val_idx.isdisjoint(cal_idx),
            "Coverage audit must use a partition disjoint from conformal calibration data."
        )

    def test_holdout_audit_partition_is_disjoint_from_calibration_at_runtime(self) -> None:
        estimator, _, _, y_test = self._build_estimator(audit_partition="holdout")

        report = estimator.calibration_report()

        self.assertEqual(report["audit_partition"], "holdout")
        self.assertEqual(report["audit_sample_count"], len(y_test))
        self.assertTrue(report["coverage_audit"]["partitions_disjoint"])
        self.assertEqual(report["coverage_audit"]["expected_partition"], "holdout")

    def test_holdout_audit_coverage_does_not_reuse_validation_rows(self) -> None:
        estimator, _, _, _ = self._build_estimator(audit_partition="holdout")

        cal_idx = set(estimator.x_calibration.index.tolist())
        audit_idx = set(estimator.x_audit.index.tolist())

        self.assertTrue(audit_idx.isdisjoint(cal_idx))
        self.assertEqual(estimator.audit_partition, "holdout")

    def test_strength_dependent_scale_is_reported_for_intervals(self) -> None:
        estimator, _, x_test = self._build_heteroscedastic_estimator()

        interval_frame = estimator.predict_with_interval(x_test)

        self.assertIn("bin_scale", interval_frame.columns)
        self.assertIn("estimated_cv", interval_frame.columns)
        low_width = float(interval_frame.iloc[0]["interval_width"])
        high_width = float(interval_frame.iloc[-1]["interval_width"])
        self.assertGreater(high_width, low_width)
        self.assertGreater(float(interval_frame.iloc[-1]["bin_scale"]), float(interval_frame.iloc[0]["bin_scale"]))

    def test_calibration_report_includes_strength_bin_coverage_audit(self) -> None:
        estimator, _, _ = self._build_heteroscedastic_estimator()

        report = estimator.calibration_report()

        self.assertIn("coverage_by_strength_bin", report)
        self.assertIn("strength_bin_audit", report)
        low_strength_bins = report["strength_bin_audit"].get("low_strength_bins", [])
        self.assertTrue(low_strength_bins)
        self.assertIn("coverage_gap_low_minus_global", report["strength_bin_audit"])
        self.assertTrue(
            all("actual_strength_min" in bin_row and "predicted_min" in bin_row for bin_row in low_strength_bins)
        )
        self.assertEqual(len(report["coverage_by_strength_bin"]), len(report["reliability_plot_data"]))

    def test_explicit_strength_bins_are_used_when_configured(self) -> None:
        estimator, _, _ = self._build_heteroscedastic_estimator()

        report = estimator.calibration_report()

        self.assertEqual(report["strength_bin_audit"]["bin_edges"], [0.0, 20.0, 35.0, 50.0])

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
