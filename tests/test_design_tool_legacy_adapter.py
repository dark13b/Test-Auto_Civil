import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from design_tool import MixDesignOptimizer
from feature_engineering import ENGINEERED_FEATURE_COLUMNS, build_engineering_features
from mix_design import DesignConstraints, DesignContext, MixDesignRequest, ObjectiveSpec
from mix_design.facade import ConcreteMixDesignService
from mix_design.objective_engine import MixObjectiveEngine
from mix_design.optimizer import MixDesignSpaceOptimizer
from mix_design.predictor import MixPerformancePredictor
from validator import EngineeringValidator


class StubModel:
    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        effective_binder = frame["effective_binder"].to_numpy(dtype=float)
        water_effective_binder_ratio = frame["water_effective_binder_ratio"].to_numpy(dtype=float)
        age = frame["age"].to_numpy(dtype=float)
        return ((0.11 * effective_binder) - (22.0 * water_effective_binder_ratio) + (0.04 * age) - 3.0).astype(float)


class StubUncertaintyEstimator:
    method = "conformal"
    coverage_level = 0.92

    def __init__(self, model: StubModel, config: dict) -> None:
        self._model = model
        self._config = config

    def predict_with_interval(self, frame: pd.DataFrame) -> pd.DataFrame:
        engineered = build_engineering_features(frame, config=self._config)
        predicted = self._model.predict(engineered)
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
                    "optuna_trials": 8,
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
        ]
    )


def make_service() -> ConcreteMixDesignService:
    config = make_config()
    base_columns = list(config["task"]["input_columns"])
    model = StubModel()
    predictor = MixPerformancePredictor(
        model=model,
        config=config,
        base_columns=base_columns,
        feature_columns=base_columns + list(ENGINEERED_FEATURE_COLUMNS),
        model_artifact_id="model-artifact-1",
        uncertainty_estimator=StubUncertaintyEstimator(model, config),
    )
    reference_dataset = make_reference_dataset()
    return ConcreteMixDesignService(
        config=config,
        predictor=predictor,
        optimizer=MixDesignSpaceOptimizer(
            reference_dataset=reference_dataset,
            base_columns=base_columns,
            target_column="compressive_strength",
            design_config=config["engineering"]["design_tool"],
            seed=42,
        ),
        objective_engine=MixObjectiveEngine(cost_proxy="cement_content"),
        validator=EngineeringValidator.from_config(config),
        reference_dataset=reference_dataset,
        base_columns=base_columns,
        target_column="compressive_strength",
        design_config=config["engineering"]["design_tool"],
        project_root=Path.cwd(),
        outputs_dir=Path(tempfile.mkdtemp()),
        model_artifact_id="model-artifact-1",
    )


def make_optimizer() -> MixDesignOptimizer:
    service = make_service()
    optimizer = MixDesignOptimizer.__new__(MixDesignOptimizer)
    optimizer.service = service
    optimizer.project_root = Path.cwd()
    optimizer.config = service.config
    optimizer.outputs_dir = service.outputs_dir
    optimizer.seed = 42
    optimizer.base_columns = service.base_columns
    optimizer.feature_columns = service.predictor.feature_columns
    optimizer.target_column = service.target_column
    optimizer.design_config = service.design_config
    optimizer.reference_dataset = service.reference_dataset
    optimizer.model = service.predictor.model
    optimizer.model_artifact_id = service.model_artifact_id
    optimizer.current_best_payload = {"artifact_id": "best-search-artifact", "run_id": "run-current"}
    optimizer.current_uncertainty_payload = {"artifact_id": "uncertainty-artifact", "run_id": "run-current"}
    optimizer.active_run_id = "run-current"
    optimizer.uncertainty_estimator = service.predictor.uncertainty_estimator
    optimizer.validator = service.validator
    return optimizer


class LegacyAdapterTests(unittest.TestCase):
    def test_optimize_emits_deprecation_warning_and_translates_comparison_result(self) -> None:
        optimizer = make_optimizer()
        comparison = optimizer.service.compare_scenarios(
            MixDesignRequest(
                target_strength_mpa=25.0,
                constraints=DesignConstraints(),
                context=DesignContext(exposure_class="marine", structural_application="column"),
                objectives=(
                    ObjectiveSpec(name="target_fit", weight=1.0),
                    ObjectiveSpec(name="cost_proxy", weight=0.5),
                    ObjectiveSpec(name="co2_proxy", weight=0.25),
                    ObjectiveSpec(name="validator_risk", weight=1.0),
                ),
                candidate_limit=3,
            )
        )

        with unittest.mock.patch.object(optimizer.service, "compare_scenarios", return_value=comparison):
            with self.assertWarns(DeprecationWarning):
                result = optimizer.optimize(
                    25.0,
                    constraints={"water": {"max": 180.0}, "tolerance_mpa": 1.5},
                    context={"exposure_class": "marine", "structural_application": "column"},
                )

        self.assertIn("ranked_candidates", result)
        self.assertEqual(result["mix_design"], result["ranked_candidates"][0]["mix_design"])
        self.assertIn("estimated_cement_saving_vs_reference", result)
        self.assertIn("cost_proxy", result["ranking_breakdown"])
        self.assertEqual(result["design_context"]["exposure_class"], "marine")


if __name__ == "__main__":
    unittest.main()
