import unittest

import numpy as np
import pandas as pd

from train import build_stacking_ensemble


class DummyValidator:
    def validate_predictions(
        self,
        predictions: np.ndarray,
        x_frame: pd.DataFrame,
        y_true: pd.Series,
    ) -> dict:
        return {
            "verdict": "PASS",
            "pass_rate": 1.0,
            "failed_count": 0,
            "hard_failed_count": 0,
            "warning_count": 0,
            "suspicious_count": 0,
            "dataset_anomaly_count": 0,
            "warn_reasons": [],
        }


class IntegrityCheckTests(unittest.TestCase):
    def test_assert_no_cv_label_fraud_raises_when_validation_score_is_mislabeled(self) -> None:
        from integrity_checks import assert_no_cv_label_fraud

        with self.assertRaises(ValueError):
            assert_no_cv_label_fraud(
                {
                    "model_name": "StackingRegressor",
                    "val_r2": 0.81,
                    "val_rmse": 3.2,
                    "val_mae": 2.1,
                }
            )

    def test_build_stacking_ensemble_reports_separate_cv_and_validation_metrics(self) -> None:
        from integrity_checks import assert_no_cv_label_fraud

        config = {
            "experiment": {
                "random_seed": 42,
                "cv_folds": 5,
                "cv_repeats": 1,
            },
            "metrics": {
                "composite_weights": {
                    "rmse": 0.4,
                    "r2": 0.4,
                    "mae": 0.2,
                }
            },
        }
        x_train = pd.DataFrame(
            {
                "cement": np.linspace(100.0, 250.0, 24),
                "water": np.linspace(160.0, 200.0, 24),
            }
        )
        y_train = pd.Series(15.0 + (0.12 * x_train["cement"]) - (0.04 * x_train["water"]))
        x_val = pd.DataFrame(
            {
                "cement": np.linspace(105.0, 245.0, 8),
                "water": np.linspace(158.0, 198.0, 8),
            }
        )
        y_val = pd.Series(15.0 + (0.12 * x_val["cement"]) - (0.04 * x_val["water"]))

        _, result = build_stacking_ensemble(
            [("LinearRegression", {}), ("Ridge", {"alpha": 1.0})],
            x_train,
            y_train,
            x_val,
            y_val,
            DummyValidator(),
            config,
        )

        self.assertIn("cv_metrics", result)
        self.assertIn("val_metrics", result)
        self.assertIn("cv_r2", result)
        self.assertIn("cv_rmse", result)
        self.assertIn("cv_mae", result)
        self.assertIn("val_r2", result)
        self.assertIn("val_rmse", result)
        self.assertIn("val_mae", result)
        assert_no_cv_label_fraud(result)
        self.assertEqual(result["cv_r2"], result["cv_metrics"]["r2"])
        self.assertEqual(result["val_r2"], result["val_metrics"]["r2"])


if __name__ == "__main__":
    unittest.main()
