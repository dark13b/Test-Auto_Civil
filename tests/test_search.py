import json
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd

import search
from search import build_post_search_ensemble, finalize_search_artifacts, resolve_final_best_result


class SearchTests(unittest.TestCase):
    def test_run_autocivil_loop_delegates_to_research_loop_with_deprecation_warning(self) -> None:
        config = {
            "experiment": {"optuna_trials": 30},
            "research": {"max_cycles": 4, "max_runtime_minutes": 0},
            "search": {
                "min_runtime_minutes": 25,
                "max_trial_seconds": 90,
                "research_brief_path": "program.md",
            },
        }

        with warnings.catch_warnings(record=True) as caught, patch(
            "search.research_loop.run_engineering_research_loop",
            return_value={"model_name": "LGBMRegressor", "composite_score": 0.91, "validation_verdict": "PASS"},
        ) as run_mock, patch("search.log_status"):
            warnings.simplefilter("always")
            result = search.run_autocivil_loop(n_trials=12, config=config)

        self.assertEqual(result["model_name"], "LGBMRegressor")
        run_mock.assert_called_once()
        kwargs = run_mock.call_args.kwargs
        self.assertEqual(kwargs["cycles_override"], 12)
        self.assertFalse(kwargs["with_report"])
        delegated_config = kwargs["config_override"]
        self.assertEqual(delegated_config["research"]["max_cycles"], 12)
        self.assertEqual(delegated_config["research"]["max_runtime_minutes"], 25)
        self.assertEqual(delegated_config["research"]["max_trial_seconds"], 90)
        self.assertEqual(delegated_config["research"]["brief_path"], "program.md")
        self.assertTrue(any("deprecated" in str(item.message).lower() for item in caught))

    def test_search_main_routes_runtime_execution_through_research_loop(self) -> None:
        config = {
            "experiment": {"optuna_trials": 7},
            "research": {"max_cycles": 3, "max_runtime_minutes": 0},
            "search": {"min_runtime_minutes": 11},
        }

        with warnings.catch_warnings(record=True) as caught, patch(
            "search.load_config",
            return_value=config,
        ), patch(
            "search.research_loop.run_engineering_research_loop",
            return_value={"model_name": "RandomForestRegressor", "composite_score": 0.88, "validation_verdict": "PASS"},
        ) as run_mock, patch(
            "search.log_status"
        ), patch(
            "sys.argv",
            ["search.py"],
        ):
            warnings.simplefilter("always")
            exit_code = search.main()

        self.assertEqual(exit_code, 0)
        run_mock.assert_called_once()
        kwargs = run_mock.call_args.kwargs
        self.assertEqual(kwargs["cycles_override"], 7)
        self.assertFalse(kwargs["with_report"])
        self.assertTrue(any("deprecated" in str(item.message).lower() for item in caught))

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
                "search.append_research_log"
            ), patch("search.log_status"), patch(
                "uncertainty.recalibrate_uncertainty_artifacts"
            ) as recalibrate_mock:
                save_pickle_mock.return_value = {"artifact_id": "ensemble-model-artifact"}
                sync_writer = Mock()
                result = build_post_search_ensemble(
                    outputs_dir=outputs_dir,
                    config=config,
                    run_id="run-20260320T172804",
                    x_train=pd.DataFrame({"cement": [1.0, 2.0]}),
                    y_train=pd.Series([2.0, 4.0]),
                    x_val=pd.DataFrame({"cement": [3.0, 4.0]}),
                    y_val=pd.Series([6.0, 8.0]),
                    validator=object(),
                    sync_writer=sync_writer,
                )

        self.assertEqual(result["status"], "new_best")
        save_pickle_mock.assert_called_once()
        sync_writer.record_new_best.assert_called_once()
        recalibrate_mock.assert_called_once()

    def test_finalize_search_artifacts_rewrites_current_ensemble_metrics_for_ensemble_winner(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            outputs_dir = Path(tmpdir)
            current_best = {
                "trial_number": 31,
                "best_trial": 31,
                "model_name": "StackingRegressor",
                "source": "post_search_ensemble",
                "status": "new_best",
                "cv_r2": 0.91,
                "composite_score": 0.91,
                "validation_verdict": "PASS",
                "hyperparameters": {"base_models": []},
                "validation_report": {"pass_rate": 1.0},
                "selected_base_models": [{"model_name": "XGBRegressor", "trial_number": 21}],
                "artifact_id": "ensemble-parent",
            }
            baseline_metrics = {"composite_score": 0.75}
            config = {"experiment": {"random_seed": 42}}

            (outputs_dir / "search_state_best_model.pkl").write_bytes(b"placeholder")

            with patch("search.resolve_final_best_result", return_value=dict(current_best)), patch(
                "search.load_pickle_artifact",
                return_value=object(),
            ), patch(
                "search.save_pickle_artifact",
                return_value={"artifact_id": "final-model-artifact"},
            ), patch(
                "search.write_run_scoped_json_artifact"
            ) as write_json_mock, patch(
                "search.load_json_artifact",
                side_effect=[
                    {"best_search_metrics": {"trial_number": 31}},
                    {"trial_number": 31, "status": "new_best"},
                ],
            ):
                finalize_search_artifacts(
                    outputs_dir,
                    baseline_metrics,
                    config=config,
                    run_id="run-20260321T004724",
                )

        written_filenames = [call.kwargs["filename"] for call in write_json_mock.call_args_list]
        self.assertIn("ensemble_metrics.json", written_filenames)
        ensemble_call = next(
            call for call in write_json_mock.call_args_list if call.kwargs["filename"] == "ensemble_metrics.json"
        )
        self.assertEqual(ensemble_call.kwargs["source_mode"], "ensemble")
        self.assertEqual(ensemble_call.kwargs["run_id"], "run-20260321T004724")
        self.assertEqual(ensemble_call.kwargs["payload"]["source"], "post_search_ensemble")

    def test_resolve_final_best_result_ignores_stale_search_state_when_final_metrics_match_csv(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            outputs_dir = Path(tmpdir)
            pd.DataFrame(
                [
                    {
                        "trial_number": 31,
                        "model_name": "StackingRegressor",
                        "composite_score": 0.91,
                        "selection_status": "new_best",
                        "validation_verdict": "PASS",
                    }
                ]
            ).to_csv(outputs_dir / "optuna_results.csv", index=False)
            (outputs_dir / "search_state_best_result.json").write_text(
                json.dumps(
                    {
                        "trial_number": 31,
                        "best_trial": 31,
                        "model_name": "StackingRegressor",
                        "composite_score": 0.91,
                        "source": "post_search_ensemble",
                        "stale": True,
                    }
                ),
                encoding="utf-8",
            )
            final_best = {
                "trial_number": 31,
                "best_trial": 31,
                "model_name": "StackingRegressor",
                "composite_score": 0.91,
                "source": "post_search_ensemble",
                "validation_verdict": "PASS",
            }
            (outputs_dir / "final_metrics.json").write_text(
                json.dumps({"best_search_metrics": final_best}),
                encoding="utf-8",
            )

            resolved = resolve_final_best_result(outputs_dir, {"composite_score": 0.75})

        self.assertEqual(resolved["model_name"], "StackingRegressor")
        self.assertEqual(resolved["source"], "post_search_ensemble")
        self.assertNotIn("stale", resolved)


if __name__ == "__main__":
    unittest.main()
