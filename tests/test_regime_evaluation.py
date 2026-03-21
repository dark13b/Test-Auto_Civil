import unittest

import numpy as np
import pandas as pd

from report import compute_regime_rmse_by_range, evaluate_regime_transition_stability


class RegimeEvaluationTests(unittest.TestCase):
    def test_compute_regime_rmse_by_range_uses_fixed_thresholds(self) -> None:
        y_true = pd.Series([20.0, 25.0, 26.0, 60.0, 61.0])
        y_pred = np.asarray([19.0, 27.0, 28.0, 59.0, 64.0], dtype=float)

        rmse = compute_regime_rmse_by_range(y_true, y_pred)

        self.assertAlmostEqual(rmse["low"], np.sqrt((1.0**2 + 2.0**2) / 2.0))
        self.assertAlmostEqual(rmse["mid"], np.sqrt((2.0**2 + 1.0**2) / 2.0))
        self.assertAlmostEqual(rmse["high"], 3.0)

    def test_evaluate_regime_transition_stability_reports_boundary_deltas(self) -> None:
        y_true = pd.Series([24.5, 25.5, 59.5, 60.5, 40.0])
        hard_pred = np.asarray([22.0, 30.0, 54.0, 66.0, 40.5], dtype=float)
        soft_pred = np.asarray([24.0, 27.0, 57.5, 63.0, 40.2], dtype=float)

        summary = evaluate_regime_transition_stability(
            y_true=y_true,
            hard_predictions=hard_pred,
            soft_predictions=soft_pred,
            transition_window_mpa=2.0,
        )

        self.assertGreater(summary["boundary_sample_count"], 0)
        self.assertGreater(summary["hard_boundary_rmse"], summary["soft_boundary_rmse"])
        self.assertTrue(summary["soft_blending_improves_transitions"])


if __name__ == "__main__":
    unittest.main()
