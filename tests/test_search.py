import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from search import build_post_search_ensemble


class SearchTests(unittest.TestCase):
    def test_build_post_search_ensemble_promotes_on_cv_r2_not_holdout_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            outputs_dir = Path(tmpdir)
            pd.DataFrame(
                [
                    {
                        "trial_number": 1,
                        "model_name": "Ridge",
                        "composite_score": 0.95,
                        "validation_verdict": "PASS",
                        "hyperparameters": json.dumps({"alpha": 1.0}),
                    }
                ]
            ).to_csv(outputs_dir / "optuna_results.csv", index=False)
            (outputs_dir / "search_state_best_result.json").write_text(
                json.dumps(
                    {
                        "model_name": "Ridge",
                        "composite_score": 0.95,
                        "cv_r2": 0.61,
                        "val_r2": 0.88,
                    }
                ),
                encoding="utf-8",
            )

            config = {
                "search": {"llm_proposals": {"enabled": False}},
                "engineering": {"uncertainty_method": "conformal"},
            }

            with patch(
                "search.build_stacking_ensemble",
                return_value=(
                    object(),
                    {
                        "model_name": "StackingRegressor",
                        "hyperparameters": {"base_models": []},
                        "rmse": 1.1,
                        "mae": 0.8,
                        "r2": 0.72,
                        "cv_r2": 0.72,
                        "cv_rmse": 1.1,
                        "cv_mae": 0.8,
                        "cv_metrics": {"r2": 0.72, "rmse": 1.1, "mae": 0.8, "composite_score": 0.50},
                        "val_r2": 0.40,
                        "val_rmse": 2.5,
                        "val_mae": 1.7,
                        "val_metrics": {"r2": 0.40, "rmse": 2.5, "mae": 1.7, "composite_score": 0.20},
                        "composite_score": 0.50,
                        "validation_verdict": "PASS",
                        "validation_report": {
                            "pass_rate": 1.0,
                            "failed_count": 0,
                            "hard_failed_count": 0,
                            "warning_count": 0,
                            "suspicious_count": 0,
                            "durability_caution_count": 0,
                            "dataset_anomaly_count": 0,
                        },
                    },
                ),
            ), patch("search.save_pickle_artifact") as save_pickle_mock, patch(
                "search.append_optuna_trial_record"
            ), patch("search.append_research_log"), patch("search.log_status"):
                result = build_post_search_ensemble(
                    outputs_dir=outputs_dir,
                    config=config,
                    x_train=pd.DataFrame({"cement": [1.0, 2.0]}),
                    y_train=pd.Series([2.0, 4.0]),
                    x_val=pd.DataFrame({"cement": [3.0, 4.0]}),
                    y_val=pd.Series([6.0, 8.0]),
                    validator=object(),
                )

        self.assertEqual(result["status"], "new_best")
        save_pickle_mock.assert_called_once()


if __name__ == "__main__":
    unittest.main()
