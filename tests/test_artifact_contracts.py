import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from train import write_run_scoped_json_artifact

from artifact_contracts import (
    ArtifactValidationError,
    map_deprecated_artifact_payload,
    validate_artifact_payload,
)


def _metric_aggregate() -> dict:
    return {
        "rmse": 3.1,
        "mae": 2.4,
        "r2": 0.82,
        "composite_score": 0.91,
    }


def _validation_report() -> dict:
    return {
        "verdict": "PASS",
        "context_type": "general",
        "pass_rate": 1.0,
        "failed_count": 0,
        "hard_failed_count": 0,
        "warning_count": 0,
        "engineering_caution_count": 0,
        "dataset_anomaly_count": 0,
        "hard_fail_reasons": [],
        "engineering_caution_reasons": [],
        "dataset_anomaly_reasons": [],
    }


def _search_selection_payload() -> dict:
    return {
        "schema_version": 1,
        "artifact_kind": "search_selection",
        "model_name": "Ridge",
        "display_name": "Ridge",
        "hyperparameters": {"alpha": 1.0},
        "trial_number": 12,
        "experiment_id": "confirm-012",
        "source": "search",
        "selection_metric_name": "composite_score",
        "selection_decision_score": 0.91,
        "cross_validation": {
            "stage": "cross_validation",
            "partition": "train",
            "folds": 5,
            "repeats": 3,
            "splitter": "RepeatedKFold",
            "random_seed": 42,
            "aggregate": _metric_aggregate(),
            "dispersion": {
                "rmse_std": 0.1,
                "mae_std": 0.1,
                "r2_std": 0.02,
                "composite_score_std": 0.01,
            },
        },
        "selection_validation": {
            "stage": "selection_validation",
            "partition": "validation",
            "aggregate": _metric_aggregate(),
        },
        "selection_validation_report": _validation_report(),
    }


def _final_holdout_payload() -> dict:
    return {
        "schema_version": 1,
        "artifact_kind": "final_holdout_evaluation",
        "selected_model": _search_selection_payload(),
        "selection_metric_name": "composite_score",
        "holdout_metric_name": "composite_score",
        "holdout_metrics": {
            "stage": "final_holdout",
            "partition": "holdout",
            "aggregate": _metric_aggregate(),
            "rmse_by_strength_range": {"low": 1.0, "mid": 1.2, "high": 1.5},
        },
        "holdout_validation_report": _validation_report(),
        "uncertainty_audit": {
            "artifact_kind": "uncertainty_audit",
            "audit_partition": "holdout",
            "calibration_partition": "validation_audit",
            "coverage_target": 0.9,
            "coverage": 0.92,
            "mean_interval_width": 2.5,
            "coverage_audit": {"expected_partition": "holdout"},
            "coverage_by_strength_bin": [],
        },
        "regime_specific_modeling": {
            "status": "evaluated",
            "structured_metrics_summary": {},
            "pass_fail": "PASS",
        },
    }


class ArtifactContractTests(unittest.TestCase):
    def test_validate_search_selection_rejects_holdout_metrics(self) -> None:
        payload = _search_selection_payload()
        payload["holdout_metrics"] = {
            "stage": "final_holdout",
            "partition": "holdout",
            "aggregate": _metric_aggregate(),
        }

        with self.assertRaises(ArtifactValidationError):
            validate_artifact_payload("search_selection.json", payload)

    def test_validate_final_holdout_rejects_legacy_test_metrics(self) -> None:
        payload = _final_holdout_payload()
        payload["test_metrics"] = {"rmse": 1.0}

        with self.assertRaises(ArtifactValidationError):
            validate_artifact_payload("final_holdout_evaluation.json", payload)

    def test_map_legacy_final_metrics_payload_emits_deprecations(self) -> None:
        legacy_payload = {
            "baseline_metrics": {"model_name": "Baseline", "composite_score": 0.8},
            "best_search_metrics": {
                "model_name": "Ridge",
                "display_name": "Ridge",
                "hyperparameters": {"alpha": 1.0},
                "trial_number": 12,
                "source": "search",
                "cv_metrics": _metric_aggregate(),
                "selection_metrics": _metric_aggregate(),
                "validation_report": _validation_report(),
                "validation_verdict": "PASS",
            },
            "validation_metrics": _metric_aggregate(),
            "holdout_metrics": _metric_aggregate(),
            "holdout_validation_verdict": "PASS",
            "uncertainty_summary": {
                "coverage_target": 0.9,
                "coverage": 0.92,
                "coverage_audit": {"expected_partition": "holdout"},
            },
            "regime_specific_modeling": {
                "status": "evaluated",
                "structured_metrics_summary": {},
                "pass_fail": "PASS",
            },
        }

        mapped = map_deprecated_artifact_payload("final_metrics.json", legacy_payload)

        self.assertEqual(mapped["artifact_kind"], "final_holdout_evaluation")
        self.assertEqual(mapped["holdout_metrics"]["stage"], "final_holdout")
        self.assertEqual(mapped["holdout_metrics"]["partition"], "holdout")
        self.assertEqual(mapped["selected_model"]["artifact_kind"], "search_selection")
        self.assertEqual(mapped["selected_model"]["composite_score"], 0.0)
        self.assertEqual(mapped["selected_model"]["validation_verdict"], "PASS")
        self.assertEqual(mapped["selected_model"]["validation_metrics"]["rmse"], 3.1)
        self.assertTrue(mapped["deprecations"])
        self.assertIn("final_metrics.json", mapped["deprecations"][0]["source"])

    def test_write_run_scoped_json_artifact_rejects_invalid_canonical_holdout_artifact(self) -> None:
        config = {
            "paths": {"outputs_dir": "outputs"},
            "data": {"mode": "local_file", "local_file": {"path": "dataset.csv"}},
            "task": {"input_columns": ["cement", "water", "age"]},
            "engineering": {"feature_engineering": True, "binder_efficiency": {"fly_ash_k": 0.35}},
            "experiment": {"random_seed": 42},
        }
        payload = _final_holdout_payload()
        payload["test_metrics"] = {"rmse": 1.0}

        with TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            (tmp_path / "dataset.csv").write_text(
                "cement,water,age,compressive_strength\n1,2,3,4\n",
                encoding="utf-8",
            )
            outputs_dir = tmp_path / "outputs"
            outputs_dir.mkdir(parents=True, exist_ok=True)

            with self.assertRaises(ArtifactValidationError):
                write_run_scoped_json_artifact(
                    outputs_dir=outputs_dir,
                    filename="final_holdout_evaluation.json",
                    payload=payload,
                    run_id="run-20260327T100000",
                    source_mode="report",
                    config=config,
                    model_artifact_id="model-artifact-1",
                    model_id="Ridge",
                )


if __name__ == "__main__":
    unittest.main()
