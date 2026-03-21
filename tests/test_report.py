import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

import report


class _RecordingModel:
    def __init__(self, expected_frame: pd.DataFrame, output: list[float]) -> None:
        self.expected_frame = expected_frame
        self.output = np.asarray(output, dtype=float)
        self.seen_frames: list[pd.DataFrame] = []

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        self.seen_frames.append(frame)
        if frame is not self.expected_frame:
            raise AssertionError("report.py used a partition other than x_test for predictions")
        return self.output


class _FakeUncertaintyEstimator:
    last_kwargs: dict | None = None

    def __init__(self, *_args, **_kwargs) -> None:
        type(self).last_kwargs = dict(_kwargs)

    def calibration_report(self) -> dict:
        return {
            "coverage": 0.93,
            "coverage_target": 0.90,
            "coverage_audit": {"expected_partition": "holdout", "partitions_disjoint": True},
            "audit_partition": "holdout",
            "audit_sample_count": 3,
            "reliability_plot_data": [
                {"bin_id": 0, "observed_coverage": 0.95},
                {"bin_id": 1, "observed_coverage": 0.90},
            ],
        }

    def predict_with_interval(self, x_test: pd.DataFrame) -> pd.DataFrame:
        strength_bins = [0 if index % 2 == 0 else 1 for index in range(len(x_test))]
        labels = ["HIGH" if index % 2 == 0 else "MODERATE" for index in range(len(x_test))]
        return pd.DataFrame(
            {
                "predicted": np.linspace(30.0, 32.0, len(x_test)),
                "interval_width": np.full(len(x_test), 2.0),
                "strength_bin": strength_bins,
                "confidence_label": labels,
            }
        )


class ReportTests(unittest.TestCase):
    def tearDown(self) -> None:
        report.REPORT_CONTEXT = SimpleNamespace()

    def test_report_main_uses_holdout_partition_and_preserves_cv_metrics(self) -> None:
        config = {
            "experiment": {"random_seed": 42},
            "engineering": {"uncertainty_method": "conformal"},
            "task": {"input_columns": ["cement", "water"], "target_column": "strength"},
        }
        x_train = pd.DataFrame({"cement": [1.0, 2.0, 3.0], "water": [3.0, 4.0, 5.0]})
        x_val = pd.DataFrame({"cement": [10.0, 20.0, 30.0], "water": [30.0, 40.0, 50.0]})
        x_test = pd.DataFrame(
            {
                "cement": [100.0, 200.0, 300.0, 400.0, 500.0],
                "water": [300.0, 400.0, 500.0, 600.0, 700.0],
            }
        )
        y_train = pd.Series([10.0, 12.0, 14.0])
        y_val = pd.Series([20.0, 22.0, 24.0])
        y_test = pd.Series([30.0, 32.0, 34.0, 36.0, 38.0])

        baseline_model = _RecordingModel(x_test, [29.0, 31.0, 33.0, 35.0, 37.0])
        best_model = _RecordingModel(x_test, [30.0, 32.0, 34.0, 36.0, 38.0])
        saved_payloads: list[tuple[Path, dict]] = []

        best_search_metrics = {
            "run_id": "run-current",
            "artifact_id": "best-search-artifact",
            "model_name": "Ridge",
            "hyperparameters": {"alpha": 1.0},
            "composite_score": 0.82,
            "cv_r2": 0.61,
            "cv_rmse": 5.0,
            "cv_mae": 4.0,
            "validation_verdict": "PASS",
        }

        def _write_run_scoped_json_artifact(*, outputs_dir: Path, filename: str, payload: dict, **_kwargs):
            path = outputs_dir / filename
            saved_payloads.append((path, payload))
            return path, path, payload

        with tempfile.TemporaryDirectory() as tmpdir:
            outputs_dir = Path(tmpdir)
            (outputs_dir / "final_metrics.json").write_text("{}", encoding="utf-8")

            with patch("report.load_config", return_value=config), patch(
            "report.get_outputs_dir",
            return_value=outputs_dir,
        ), patch(
            "report.load_json_artifact",
            side_effect=[
                {"composite_score": 0.75},
                {"best_search_metrics": best_search_metrics},
                best_search_metrics,
            ],
        ), patch(
            "report.load_pickle_artifact",
            side_effect=[baseline_model, best_model],
        ), patch(
            "report.pd.read_csv",
            return_value=pd.DataFrame([{"trial_number": 1, "composite_score": 0.82}]),
        ), patch(
            "report.load_dataset",
            return_value=pd.DataFrame(),
        ), patch(
            "report.split_dataset",
            return_value=(x_train, x_val, x_test, y_train, y_val, y_test),
        ), patch(
            "report.EngineeringValidator.from_config",
            return_value=SimpleNamespace(
                validate_model=lambda _model, features, target: (
                    {"verdict": "PASS", "features_id": id(features), "target_id": id(target)}
                )
            ),
        ), patch(
            "report.compute_regression_metrics",
            return_value={"rmse": 1.0, "mae": 0.5, "r2": 0.8, "composite_score": 0.84},
        ), patch(
            "report.create_search_progress_plot"
        ), patch(
            "report.create_actual_vs_predicted_plot"
        ), patch(
            "report.create_residuals_plot"
        ), patch(
            "report.compute_feature_importance",
            return_value=pd.Series([0.5, 0.5], index=["cement", "water"]),
        ), patch(
            "report.create_feature_importance_plot"
        ), patch(
            "report.create_performance_by_range_plot",
            return_value={"low": 1.0, "mid": 1.1, "high": 1.2},
        ), patch(
            "report.create_uncertainty_plot",
            return_value={"mean_interval_width": 2.0, "label_counts": {"HIGH": 1, "MODERATE": 1}},
        ), patch(
            "report.UncertaintyEstimator",
            _FakeUncertaintyEstimator,
        ), patch(
            "report.summarize_validation_report",
            return_value={
                "hard_constraint_count": 0,
                "engineering_caution_count": 0,
                "data_review_flag_count": 0,
            },
        ), patch(
            "report.write_run_scoped_json_artifact",
            side_effect=_write_run_scoped_json_artifact,
        ), patch(
            "report.log_status"
        ):
                report.REPORT_CONTEXT = SimpleNamespace()
                exit_code = report.main()

        self.assertEqual(exit_code, 0)
        self.assertEqual(baseline_model.seen_frames, [x_test])
        self.assertEqual(best_model.seen_frames, [x_test])
        self.assertTrue(saved_payloads)
        final_metrics = saved_payloads[-1][1]
        self.assertEqual(final_metrics["best_search_metrics"]["cv_r2"], 0.61)
        self.assertEqual(final_metrics["validation_verdict"], "PASS")
        self.assertEqual(final_metrics["holdout_metrics"]["rmse"], 1.0)
        self.assertEqual(final_metrics["holdout_metrics"]["mae"], 0.5)
        self.assertEqual(final_metrics["holdout_metrics"]["r2"], 0.8)
        self.assertEqual(final_metrics["holdout_metrics"]["composite_score"], 0.84)
        self.assertEqual(final_metrics["best_model_name"], "Ridge")
        self.assertEqual(_FakeUncertaintyEstimator.last_kwargs["audit_partition"], "holdout")
        self.assertEqual(final_metrics["uncertainty_summary"]["coverage_audit"]["expected_partition"], "holdout")
        self.assertIn("regime_specific_modeling", final_metrics)
        self.assertIn("structured_metrics_summary", final_metrics["regime_specific_modeling"])
        self.assertIn("pass_fail", final_metrics["regime_specific_modeling"])
        review_payloads = [payload for path, payload in saved_payloads if path.name == "scientific_report_review.json"]
        self.assertEqual(len(review_payloads), 1)
        self.assertIn("scientific_accuracy", review_payloads[0]["scores"])

    def test_report_context_blocks_second_holdout_use(self) -> None:
        report.REPORT_CONTEXT = SimpleNamespace(_test_set_used_in_report=True)
        with patch("report.load_config", return_value={"experiment": {"random_seed": 42}}), patch(
            "report.log_status"
        ):
            self.assertEqual(report.main(), 1)

    def test_report_ignores_stale_final_metrics_from_different_run(self) -> None:
        config = {
            "experiment": {"random_seed": 42},
            "engineering": {"uncertainty_method": "conformal"},
            "task": {"input_columns": ["cement", "water"], "target_column": "strength"},
        }
        x_train = pd.DataFrame({"cement": [1.0, 2.0, 3.0], "water": [3.0, 4.0, 5.0]})
        x_val = pd.DataFrame({"cement": [10.0, 20.0, 30.0], "water": [30.0, 40.0, 50.0]})
        x_test = pd.DataFrame({"cement": [100.0, 200.0], "water": [300.0, 400.0]})
        y_train = pd.Series([10.0, 12.0, 14.0])
        y_val = pd.Series([20.0, 22.0, 24.0])
        y_test = pd.Series([30.0, 32.0])

        baseline_model = _RecordingModel(x_test, [29.0, 31.0])
        best_model = _RecordingModel(x_test, [30.0, 32.0])
        saved_payloads: list[tuple[Path, dict]] = []

        stale_final_metrics = {
            "run_id": "old-run",
            "best_search_metrics": {
                "run_id": "old-run",
                "artifact_id": "old-best",
                "model_name": "OldModel",
                "hyperparameters": {"alpha": 9.0},
                "composite_score": 0.40,
                "cv_r2": 0.11,
                "validation_verdict": "WARN",
            },
        }
        current_best_search_result = {
            "run_id": "new-run",
            "artifact_id": "new-best",
            "model_name": "Ridge",
            "hyperparameters": {"alpha": 1.0},
            "composite_score": 0.82,
            "cv_r2": 0.61,
            "cv_rmse": 5.0,
            "cv_mae": 4.0,
            "validation_verdict": "PASS",
        }

        def _write_run_scoped_json_artifact(*, outputs_dir: Path, filename: str, payload: dict, **_kwargs):
            path = outputs_dir / filename
            saved_payloads.append((path, payload))
            return path, path, payload

        with tempfile.TemporaryDirectory() as tmpdir:
            outputs_dir = Path(tmpdir)
            (outputs_dir / "final_metrics.json").write_text("{}", encoding="utf-8")

            with patch("report.load_config", return_value=config), patch(
                "report.get_outputs_dir",
                return_value=outputs_dir,
            ), patch(
                "report.load_json_artifact",
                side_effect=[
                    {"composite_score": 0.75},
                    stale_final_metrics,
                    current_best_search_result,
                ],
            ), patch(
                "report.load_pickle_artifact",
                side_effect=[baseline_model, best_model],
            ), patch(
                "report.pd.read_csv",
                return_value=pd.DataFrame([{"trial_number": 31, "composite_score": 0.82}]),
            ), patch(
                "report.load_dataset",
                return_value=pd.DataFrame(),
            ), patch(
                "report.split_dataset",
                return_value=(x_train, x_val, x_test, y_train, y_val, y_test),
            ), patch(
                "report.EngineeringValidator.from_config",
                return_value=SimpleNamespace(validate_model=lambda _model, _features, _target: {"verdict": "PASS"}),
            ), patch(
                "report.compute_regression_metrics",
                return_value={"rmse": 1.0, "mae": 0.5, "r2": 0.8, "composite_score": 0.84},
            ), patch("report.create_search_progress_plot"), patch("report.create_actual_vs_predicted_plot"), patch(
                "report.create_residuals_plot"
            ), patch(
                "report.compute_feature_importance",
                return_value=pd.Series([0.5, 0.5], index=["cement", "water"]),
            ), patch("report.create_feature_importance_plot"), patch(
                "report.create_performance_by_range_plot",
                return_value={"low": 1.0, "mid": 1.1, "high": 1.2},
            ), patch(
                "report.create_uncertainty_plot",
                return_value={"mean_interval_width": 2.0, "label_counts": {"HIGH": 1}},
            ), patch(
                "report.UncertaintyEstimator",
                _FakeUncertaintyEstimator,
            ), patch(
                "report.summarize_validation_report",
                return_value={
                    "hard_constraint_count": 0,
                    "engineering_caution_count": 0,
                    "data_review_flag_count": 0,
                },
            ), patch(
            "report.write_run_scoped_json_artifact",
            side_effect=_write_run_scoped_json_artifact,
        ), patch("report.log_status"):
                report.REPORT_CONTEXT = SimpleNamespace()
                exit_code = report.main()

        self.assertEqual(exit_code, 0)
        final_metrics = saved_payloads[-1][1]
        self.assertEqual(final_metrics["best_search_metrics"]["run_id"], "new-run")
        self.assertEqual(final_metrics["best_search_metrics"]["model_name"], "Ridge")

    def test_report_marks_stale_search_state_artifact(self) -> None:
        config = {
            "experiment": {"random_seed": 42},
            "engineering": {"uncertainty_method": "conformal"},
            "task": {"input_columns": ["cement", "water"], "target_column": "strength"},
        }
        x_train = pd.DataFrame({"cement": [1.0, 2.0, 3.0], "water": [3.0, 4.0, 5.0]})
        x_val = pd.DataFrame({"cement": [10.0, 20.0, 30.0], "water": [30.0, 40.0, 50.0]})
        x_test = pd.DataFrame({"cement": [100.0, 200.0], "water": [300.0, 400.0]})
        y_train = pd.Series([10.0, 12.0, 14.0])
        y_val = pd.Series([20.0, 22.0, 24.0])
        y_test = pd.Series([30.0, 32.0])

        baseline_model = _RecordingModel(x_test, [29.0, 31.0])
        best_model = _RecordingModel(x_test, [30.0, 32.0])
        current_best_search_result = {
            "run_id": "current-run",
            "artifact_id": "new-best",
            "model_name": "Ridge",
            "hyperparameters": {"alpha": 1.0},
            "composite_score": 0.82,
            "cv_r2": 0.61,
            "cv_rmse": 5.0,
            "cv_mae": 4.0,
            "validation_verdict": "PASS",
        }

        def _write_run_scoped_json_artifact(*, outputs_dir: Path, filename: str, payload: dict, **_kwargs):
            path = outputs_dir / filename
            return path, path, payload

        with tempfile.TemporaryDirectory() as tmpdir:
            outputs_dir = Path(tmpdir)
            (outputs_dir / "final_metrics.json").write_text("{}", encoding="utf-8")
            (outputs_dir / "search_state_best_result.json").write_text(
                '{"model_name": "OldState"}',
                encoding="utf-8",
            )

            with patch("report.load_config", return_value=config), patch(
                "report.get_outputs_dir",
                return_value=outputs_dir,
            ), patch(
                "report.load_json_artifact",
                side_effect=[
                    {"composite_score": 0.75},
                    {},
                    current_best_search_result,
                ],
            ), patch(
                "report.load_pickle_artifact",
                side_effect=[baseline_model, best_model],
            ), patch(
                "report.pd.read_csv",
                return_value=pd.DataFrame([{"trial_number": 31, "composite_score": 0.82}]),
            ), patch(
                "report.load_dataset",
                return_value=pd.DataFrame(),
            ), patch(
                "report.split_dataset",
                return_value=(x_train, x_val, x_test, y_train, y_val, y_test),
            ), patch(
                "report.EngineeringValidator.from_config",
                return_value=SimpleNamespace(
                    validate_model=lambda _model, features, target: (
                        {"verdict": "PASS", "features_id": id(features), "target_id": id(target)}
                    )
                ),
            ), patch(
                "report.compute_regression_metrics",
                return_value={"rmse": 1.0, "mae": 0.5, "r2": 0.8, "composite_score": 0.84},
            ), patch(
                "report.create_search_progress_plot"
            ), patch(
                "report.create_actual_vs_predicted_plot"
            ), patch(
                "report.create_residuals_plot"
            ), patch(
                "report.compute_feature_importance",
                return_value=pd.Series([0.5, 0.5], index=["cement", "water"]),
            ), patch(
                "report.create_feature_importance_plot"
            ), patch(
                "report.create_performance_by_range_plot",
                return_value={"low": 1.0, "mid": 1.1, "high": 1.2},
            ), patch(
                "report.create_uncertainty_plot",
                return_value={"mean_interval_width": 2.0, "label_counts": {"HIGH": 1, "MODERATE": 1}},
            ), patch(
                "report.UncertaintyEstimator",
                _FakeUncertaintyEstimator,
            ), patch(
                "report.summarize_validation_report",
                return_value={
                    "hard_constraint_count": 0,
                    "engineering_caution_count": 0,
                    "data_review_flag_count": 0,
                },
            ), patch(
                "report.write_run_scoped_json_artifact",
                side_effect=_write_run_scoped_json_artifact,
            ), patch("report.log_status"):
                report.REPORT_CONTEXT = SimpleNamespace()
                exit_code = report.main()

            stale_state = json.loads((outputs_dir / "search_state_best_result.json").read_text(encoding="utf-8"))

        self.assertEqual(exit_code, 0)
        self.assertTrue(stale_state["stale"])
        self.assertEqual(stale_state["active_run_id"], "current-run")
        self.assertTrue(stale_state["artifact_metadata"]["stale"])


if __name__ == "__main__":
    unittest.main()
