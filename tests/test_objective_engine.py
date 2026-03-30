import unittest

from mix_design.contracts import (
    ConstraintEvaluation,
    DesignConstraints,
    ObjectiveSpec,
    PredictionResult,
    UncertaintyInterval,
    ValidatorOutcome,
)
from mix_design.objective_engine import MixObjectiveEngine


def make_prediction(predicted_strength: float) -> PredictionResult:
    return PredictionResult(
        predicted_strength_mpa=predicted_strength,
        engineered_features={
            "water_cement_ratio": 0.55,
            "fly_ash_replacement_ratio": 0.20,
            "slag_replacement_ratio": 0.30,
        },
        uncertainty_interval=UncertaintyInterval(
            predicted=predicted_strength,
            lower_90=predicted_strength - 2.0,
            upper_90=predicted_strength + 2.0,
            interval_width=4.0,
            confidence_label="HIGH",
            target_window_overlap=0.75,
        ),
        model_artifact_id="model-artifact-1",
    )


def make_validator(verdict: str, warnings: tuple[str, ...] = (), failures: tuple[str, ...] = ()) -> ValidatorOutcome:
    return ValidatorOutcome(
        overall_verdict=verdict,
        warning_reasons=warnings,
        failure_reasons=failures,
        hard_constraints=failures,
        engineering_cautions=warnings,
        data_review_flags=(),
        contextual_summary="validator summary",
        confidence_of_warning_assessment="high",
        raw_report={"overall_verdict": verdict},
    )


class ObjectiveEngineTests(unittest.TestCase):
    def test_score_candidate_returns_explicit_component_breakdown(self) -> None:
        engine = MixObjectiveEngine()

        scorecard = engine.score_candidate(
            mix_design={"cement": 180.0, "slag": 80.0, "fly_ash": 40.0},
            prediction=make_prediction(31.5),
            constraints=ConstraintEvaluation(passed=True, checks=(), hard_failures=()),
            validator=make_validator("WARN", warnings=("durability caution",)),
            target_strength_mpa=32.0,
            design_constraints=DesignConstraints(tolerance_mpa=2.0),
            objectives=(
                ObjectiveSpec(name="target_fit", weight=1.0),
                ObjectiveSpec(name="cement_penalty", weight=0.5),
                ObjectiveSpec(name="co2_proxy", weight=0.25),
                ObjectiveSpec(name="validator_risk", weight=2.0),
            ),
        )

        self.assertEqual(
            [component.name for component in scorecard.components],
            ["target_fit", "cement_penalty", "co2_proxy", "validator_risk"],
        )
        self.assertTrue(all(component.explanation for component in scorecard.components))
        self.assertAlmostEqual(
            scorecard.total_score,
            sum(component.weighted_score for component in scorecard.components),
        )

    def test_validator_risk_scores_fail_higher_than_warn(self) -> None:
        engine = MixObjectiveEngine()
        objectives = (ObjectiveSpec(name="validator_risk", weight=1.0),)

        warn_score = engine.score_candidate(
            mix_design={"cement": 180.0, "slag": 80.0, "fly_ash": 40.0},
            prediction=make_prediction(31.5),
            constraints=ConstraintEvaluation(passed=True, checks=(), hard_failures=()),
            validator=make_validator("WARN", warnings=("warning",)),
            target_strength_mpa=32.0,
            design_constraints=DesignConstraints(tolerance_mpa=2.0),
            objectives=objectives,
        )
        fail_score = engine.score_candidate(
            mix_design={"cement": 180.0, "slag": 80.0, "fly_ash": 40.0},
            prediction=make_prediction(31.5),
            constraints=ConstraintEvaluation(passed=False, checks=(), hard_failures=("water_cement_ratio.max",)),
            validator=make_validator("FAIL", failures=("hard fail",)),
            target_strength_mpa=32.0,
            design_constraints=DesignConstraints(tolerance_mpa=2.0),
            objectives=objectives,
        )

        self.assertGreater(fail_score.total_score, warn_score.total_score)

    def test_cost_proxy_objective_uses_configured_proxy_and_strength_fit_alias(self) -> None:
        engine = MixObjectiveEngine(cost_proxy="water")

        scorecard = engine.score_candidate(
            mix_design={"cement": 180.0, "slag": 80.0, "fly_ash": 40.0, "water": 150.0},
            prediction=make_prediction(32.2),
            constraints=ConstraintEvaluation(passed=True, checks=(), hard_failures=()),
            validator=make_validator("PASS"),
            target_strength_mpa=32.0,
            design_constraints=DesignConstraints(tolerance_mpa=2.0),
            objectives=(
                ObjectiveSpec(name="strength_fit", weight=1.0),
                ObjectiveSpec(name="cost_proxy", weight=0.5),
            ),
        )

        self.assertEqual(
            [component.name for component in scorecard.components],
            ["strength_fit", "cost_proxy"],
        )
        self.assertEqual(scorecard.components[1].raw_value, 150.0)
        self.assertIn("water", scorecard.components[1].explanation.lower())


if __name__ == "__main__":
    unittest.main()
