import tempfile
import unittest
from pathlib import Path

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


if __name__ == "__main__":
    unittest.main()
