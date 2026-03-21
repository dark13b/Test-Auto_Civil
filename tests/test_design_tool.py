import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from design_tool import MixDesignOptimizer
from feature_engineering import ENGINEERED_FEATURE_COLUMNS, build_engineering_features
from validator import EngineeringValidator


class StubModel:
    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        effective_binder = frame["effective_binder"].to_numpy(dtype=float)
        water_effective_binder_ratio = frame["water_effective_binder_ratio"].to_numpy(dtype=float)
        age = frame["age"].to_numpy(dtype=float)
        strength = (0.11 * effective_binder) - (22.0 * water_effective_binder_ratio) + (0.04 * age) - 3.0
        return strength.astype(float)


class StubUncertaintyEstimator:
    def __init__(self, model: StubModel) -> None:
        self.model = model
        self.method = "conformal"
        self.coverage_level = 0.92

    def predict_with_interval(self, frame: pd.DataFrame) -> pd.DataFrame:
        engineered = build_engineering_features(frame)
        predicted = self.model.predict(engineered)
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
        "paths": {"outputs_dir": "outputs"},
        "experiment": {"random_seed": 42},
        "engineering_bounds": {"min": 0.0, "max": 120.0},
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
            ],
            "target_column": "compressive_strength",
        },
        "engineering": {
            "feature_engineering": True,
            "binder_efficiency": {"fly_ash_k": 0.35, "slag_k": 0.80},
            "design_tool": {
                "default_age_days": 28,
                "target_tolerance_mpa": 2.0,
                "cost_proxy": "cement_content",
                "top_ranked_candidates": 5,
                "reference_mix": {"water_cement_ratio": 0.5},
                "search": {
                    "initial_samples": 64,
                    "reference_grid_points": 24,
                    "optuna_trials": 12,
                },
                "constraints": {
                    "cement": {"min": 100.0, "max": 450.0},
                    "slag": {"min": 0.0, "max": 220.0},
                    "fly_ash": {"min": 0.0, "max": 180.0},
                    "water": {"min": 130.0, "max": 210.0},
                    "superplasticizer": {"min": 0.0, "max": 16.0},
                    "coarse_aggregate": {"min": 900.0, "max": 1100.0},
                    "fine_aggregate": {"min": 650.0, "max": 850.0},
                    "age": {"fixed": 28.0},
                    "water_cement_ratio": {"max": 0.70},
                    "fly_ash_replacement_ratio": {"max": 0.45},
                    "slag_replacement_ratio": {"max": 0.70},
                },
            },
        },
        "validator": {
            "context_type": "general",
            "suspicious_water_cement_ratio": 0.7,
            "suspicious_strength_mpa": 30.0,
            "durability_water_cement_warn": 0.6,
            "low_water_binder_warn": 0.25,
            "total_binder_low_warn": 250.0,
            "total_binder_high_warn": 550.0,
            "fly_ash_replacement_warn": 0.4,
            "slag_replacement_warn": 0.7,
            "early_age_days_warn": 3.0,
            "early_age_strength_warn": 30.0,
            "scm_meaningful_replacement_threshold": 0.15,
            "high_volume_scm_replacement_threshold": 0.45,
        },
    }


def make_reference_dataset() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "cement": 150.0,
                "slag": 120.0,
                "fly_ash": 40.0,
                "water": 170.0,
                "superplasticizer": 6.0,
                "coarse_aggregate": 1030.0,
                "fine_aggregate": 760.0,
                "age": 28.0,
                "compressive_strength": 24.5,
            },
            {
                "cement": 165.0,
                "slag": 110.0,
                "fly_ash": 35.0,
                "water": 168.0,
                "superplasticizer": 7.0,
                "coarse_aggregate": 1015.0,
                "fine_aggregate": 770.0,
                "age": 28.0,
                "compressive_strength": 25.8,
            },
            {
                "cement": 185.0,
                "slag": 95.0,
                "fly_ash": 25.0,
                "water": 166.0,
                "superplasticizer": 7.5,
                "coarse_aggregate": 1000.0,
                "fine_aggregate": 780.0,
                "age": 28.0,
                "compressive_strength": 29.8,
            },
            {
                "cement": 215.0,
                "slag": 80.0,
                "fly_ash": 20.0,
                "water": 162.0,
                "superplasticizer": 8.5,
                "coarse_aggregate": 990.0,
                "fine_aggregate": 785.0,
                "age": 28.0,
                "compressive_strength": 34.2,
            },
        ]
    )


def make_optimizer() -> MixDesignOptimizer:
    optimizer = MixDesignOptimizer.__new__(MixDesignOptimizer)
    optimizer.project_root = Path.cwd()
    optimizer.config = make_config()
    optimizer.outputs_dir = Path(tempfile.mkdtemp())
    optimizer.seed = 42
    optimizer.base_columns = list(optimizer.config["task"]["input_columns"])
    optimizer.feature_columns = optimizer.base_columns + list(ENGINEERED_FEATURE_COLUMNS)
    optimizer.target_column = "compressive_strength"
    optimizer.design_config = optimizer.config["engineering"]["design_tool"]
    optimizer.reference_dataset = make_reference_dataset()
    optimizer.model = StubModel()
    optimizer.model_artifact_id = "model-artifact-1"
    optimizer.current_best_payload = {"artifact_id": "best-search-artifact", "run_id": "run-current"}
    optimizer.current_uncertainty_payload = {"artifact_id": "uncertainty-artifact", "run_id": "run-current"}
    optimizer.active_run_id = "run-current"
    optimizer.uncertainty_estimator = StubUncertaintyEstimator(optimizer.model)
    optimizer.validator = EngineeringValidator.from_config(optimizer.config)
    return optimizer


class DesignToolTests(unittest.TestCase):
    def test_low_strength_engineering_priors_are_dataset_conditioned(self) -> None:
        optimizer = make_optimizer()
        constraints = optimizer._normalize_constraints(None)
        constraints = optimizer._apply_target_dependent_bounds(25.0, constraints)

        priors = optimizer._engineering_prior_mixes(25.0, constraints)

        self.assertTrue(priors)
        self.assertLessEqual(min(prior["cement"] for prior in priors), 190.0)
        self.assertTrue(any((prior["slag"] + prior["fly_ash"]) > 0.0 for prior in priors))

    def test_candidate_evaluation_reports_uncertainty_and_plausibility(self) -> None:
        optimizer = make_optimizer()
        constraints = optimizer._normalize_constraints(None)
        constraints = optimizer._apply_target_dependent_bounds(25.0, constraints)
        candidate = optimizer._evaluate_mix(
            {
                "cement": 170.0,
                "slag": 110.0,
                "fly_ash": 35.0,
                "water": 168.0,
                "superplasticizer": 7.0,
                "coarse_aggregate": 1015.0,
                "fine_aggregate": 770.0,
                "age": 28.0,
            },
            target_strength=25.0,
            tolerance=2.0,
            constraints=constraints,
        )

        self.assertIn("uncertainty_interval", candidate)
        self.assertIn("plausibility_penalty", candidate)
        self.assertIn("ranking_breakdown", candidate)
        self.assertIn("target_window_overlap", candidate["uncertainty_interval"])

    def test_optimize_returns_ranked_candidates_and_legacy_winner_fields(self) -> None:
        optimizer = make_optimizer()

        result = optimizer.optimize(25.0)

        self.assertIn("ranked_candidates", result)
        self.assertTrue(result["ranked_candidates"])
        self.assertIn("mix_design", result)
        self.assertIn("predicted_strength", result)
        self.assertIn("uncertainty_interval", result)
        self.assertIn("ranking_breakdown", result)

    def test_uncertainty_summary_matches_official_estimator_output(self) -> None:
        optimizer = make_optimizer()
        candidate_frame = pd.DataFrame(
            [
                {
                    "cement": 170.0,
                    "slag": 110.0,
                    "fly_ash": 35.0,
                    "water": 168.0,
                    "superplasticizer": 7.0,
                    "coarse_aggregate": 1015.0,
                    "fine_aggregate": 770.0,
                    "age": 28.0,
                }
            ]
        )

        summary = optimizer._uncertainty_interval_summary(candidate_frame, target_strength=25.0, tolerance=2.0)
        official = optimizer.uncertainty_estimator.predict_with_interval(candidate_frame).iloc[0]

        self.assertAlmostEqual(summary["predicted"], float(official["predicted"]))
        self.assertAlmostEqual(summary["lower_90"], float(official["lower_90"]))
        self.assertAlmostEqual(summary["upper_90"], float(official["upper_90"]))
        self.assertAlmostEqual(summary["interval_width"], float(official["interval_width"]))

    def test_batch_export_writes_matching_csv_and_per_target_jsons_from_same_results(self) -> None:
        optimizer = make_optimizer()
        run_id = "run-20260320T173301"
        stale_path = optimizer.outputs_dir / "design_99MPa.json"
        stale_path.write_text(json.dumps({"run_id": "old-run"}), encoding="utf-8")

        batch_results = [
            {
                "success": True,
                "target_strength": 25.0,
                "tolerance_mpa": 2.0,
                "predicted_strength": 24.9,
                "mix_design": {
                    "cement": 150.0,
                    "slag": 50.0,
                    "fly_ash": 40.0,
                    "water": 160.0,
                    "superplasticizer": 8.0,
                    "coarse_aggregate": 1000.0,
                    "fine_aggregate": 700.0,
                    "age": 28.0,
                },
                "engineered_ratios": {
                    "water_cement_ratio": 1.0666666667,
                    "water_binder_ratio": 0.6666666667,
                    "total_binder": 240.0,
                },
                "validation_verdict": "WARN",
                "hard_constraints": [],
                "engineering_cautions": [],
                "data_review_flags": [],
                "contextual_summary": "warning",
                "confidence_of_warning_assessment": "moderate",
                "validation_warning_reasons": [],
                "validation_failure_reasons": [],
                "design_constraint_violations": [],
                "uncertainty_interval": {
                    "predicted": 24.9,
                    "lower_90": 23.4,
                    "upper_90": 26.4,
                    "interval_width": 3.0,
                    "confidence_label": "MODERATE",
                    "target_window_overlap": 0.75,
                },
                "ranking_breakdown": {},
                "plausibility_penalty": 0.5,
                "deviation_mpa": 0.1,
                "estimated_cement_saving_vs_reference": {
                    "cement_saving_kg_per_m3": 15.0,
                    "cement_saving_percent": 9.0,
                    "reference_cement_content": 165.0,
                    "reference_predicted_strength": 25.2,
                },
                "ranked_candidates": [],
            },
            {
                "success": True,
                "target_strength": 30.0,
                "tolerance_mpa": 2.0,
                "predicted_strength": 29.7,
                "mix_design": {
                    "cement": 180.0,
                    "slag": 60.0,
                    "fly_ash": 20.0,
                    "water": 150.0,
                    "superplasticizer": 6.0,
                    "coarse_aggregate": 990.0,
                    "fine_aggregate": 720.0,
                    "age": 28.0,
                },
                "engineered_ratios": {
                    "water_cement_ratio": 0.8333333333,
                    "water_binder_ratio": 0.5769230769,
                    "total_binder": 260.0,
                },
                "validation_verdict": "PASS",
                "hard_constraints": [],
                "engineering_cautions": [],
                "data_review_flags": [],
                "contextual_summary": "pass",
                "confidence_of_warning_assessment": "high",
                "validation_warning_reasons": [],
                "validation_failure_reasons": [],
                "design_constraint_violations": [],
                "uncertainty_interval": {
                    "predicted": 29.7,
                    "lower_90": 28.1,
                    "upper_90": 31.3,
                    "interval_width": 3.2,
                    "confidence_label": "HIGH",
                    "target_window_overlap": 0.8,
                },
                "ranking_breakdown": {},
                "plausibility_penalty": 0.3,
                "deviation_mpa": 0.3,
                "estimated_cement_saving_vs_reference": {
                    "cement_saving_kg_per_m3": 10.0,
                    "cement_saving_percent": 5.0,
                    "reference_cement_content": 190.0,
                    "reference_predicted_strength": 30.1,
                },
                "ranked_candidates": [],
            },
        ]

        with patch.object(optimizer, "optimize", side_effect=batch_results):
            batch_frame, exported_paths = optimizer.export_batch_artifacts([25.0, 30.0], run_id=run_id)

        csv_frame = pd.read_csv(optimizer.outputs_dir / "batch_design_results.csv")
        design_25 = json.loads((optimizer.outputs_dir / "design_25MPa.json").read_text(encoding="utf-8"))
        design_30 = json.loads((optimizer.outputs_dir / "design_30MPa.json").read_text(encoding="utf-8"))

        self.assertEqual(len(exported_paths), 2)
        self.assertFalse(stale_path.exists())
        self.assertEqual(list(batch_frame["target_strength"]), [25.0, 30.0])
        self.assertEqual(csv_frame["run_id"].unique().tolist(), [run_id])
        self.assertEqual(design_25["run_id"], run_id)
        self.assertEqual(design_30["run_id"], run_id)
        self.assertEqual(float(csv_frame.loc[csv_frame["target_strength"] == 25.0, "cement"].iloc[0]), design_25["mix_design"]["cement"])
        self.assertEqual(float(csv_frame.loc[csv_frame["target_strength"] == 30.0, "cement"].iloc[0]), design_30["mix_design"]["cement"])
        self.assertEqual(str(csv_frame.loc[csv_frame["target_strength"] == 25.0, "validation_verdict"].iloc[0]), design_25["validation_verdict"])
        self.assertEqual(bool(csv_frame.loc[csv_frame["target_strength"] == 30.0, "success"].iloc[0]), design_30["success"])

    def test_batch_and_single_exports_share_same_winner_contract(self) -> None:
        optimizer = make_optimizer()
        single_result = optimizer.optimize(25.0)

        with patch.object(optimizer, "optimize", return_value=single_result):
            csv_frame, _ = optimizer.export_batch_artifacts([25.0], run_id="run-20260320T173500")

        row = csv_frame.iloc[0]
        self.assertAlmostEqual(float(row["predicted_strength"]), float(single_result["predicted_strength"]))
        self.assertAlmostEqual(float(row["cement"]), float(single_result["mix_design"]["cement"]))
        self.assertEqual(str(row["validation_verdict"]), str(single_result["validation_verdict"]))
        self.assertEqual(bool(row["success"]), bool(single_result["success"]))


if __name__ == "__main__":
    unittest.main()
