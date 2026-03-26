import unittest

from mix_design.contracts import (
    CandidateScenario,
    ConstraintCheck,
    ConstraintEvaluation,
    DesignConstraints,
    DesignContext,
    MixDesignRequest,
    ObjectiveComponentScore,
    ObjectiveScorecard,
    ObjectiveSpec,
    PredictionResult,
    ScenarioComparisonResult,
    UncertaintyInterval,
    ValidatorOutcome,
    default_objectives,
)


class DesignContractTests(unittest.TestCase):
    def test_default_objectives_enable_expected_phase_a_components(self) -> None:
        objectives = default_objectives()

        self.assertEqual(
            [objective.name for objective in objectives],
            ["target_fit", "cement_penalty", "co2_proxy", "validator_risk"],
        )
        self.assertTrue(all(objective.enabled for objective in objectives))

    def test_mix_design_request_uses_explicit_typed_fields(self) -> None:
        request = MixDesignRequest(
            target_strength_mpa=32.0,
            constraints=DesignConstraints(tolerance_mpa=2.5),
            context=DesignContext(exposure_class="marine", structural_application="column"),
            objectives=(ObjectiveSpec(name="target_fit", weight=1.0),),
            candidate_limit=4,
        )

        self.assertEqual(request.target_strength_mpa, 32.0)
        self.assertEqual(request.constraints.tolerance_mpa, 2.5)
        self.assertEqual(request.context.exposure_class, "marine")
        self.assertEqual(request.candidate_limit, 4)

    def test_scenario_comparison_result_keeps_multiple_ranked_candidates(self) -> None:
        scenario = CandidateScenario(
            scenario_id="candidate-1",
            mix_design={"cement": 180.0, "water": 160.0},
            prediction=PredictionResult(
                predicted_strength_mpa=31.8,
                engineered_features={"water_cement_ratio": 0.89},
                uncertainty_interval=UncertaintyInterval(
                    predicted=31.8,
                    lower_90=29.8,
                    upper_90=33.8,
                    interval_width=4.0,
                    confidence_label="HIGH",
                    target_window_overlap=1.0,
                ),
                model_artifact_id="model-artifact-1",
            ),
            constraints=ConstraintEvaluation(
                passed=True,
                checks=(
                    ConstraintCheck(
                        name="cement.min",
                        passed=True,
                        actual=180.0,
                        limit=100.0,
                        message="cement meets minimum",
                    ),
                ),
                hard_failures=(),
            ),
            validator=ValidatorOutcome(
                overall_verdict="PASS",
                warning_reasons=(),
                failure_reasons=(),
                hard_constraints=(),
                engineering_cautions=(),
                data_review_flags=(),
                contextual_summary="clean",
                confidence_of_warning_assessment="high",
                raw_report={"overall_verdict": "PASS"},
            ),
            objective_scorecard=ObjectiveScorecard(
                total_score=12.5,
                components=(
                    ObjectiveComponentScore(
                        name="target_fit",
                        weight=1.0,
                        raw_value=0.2,
                        weighted_score=0.2,
                        explanation="Candidate is close to the requested target.",
                    ),
                ),
                rank_explanation=("Lowest weighted total score.",),
            ),
            success=True,
            source="unit-test",
        )
        result = ScenarioComparisonResult(
            request=MixDesignRequest(target_strength_mpa=32.0, constraints=DesignConstraints()),
            best_scenario_id="candidate-1",
            scenarios=(scenario,),
            comparison_summary=("candidate-1 ranks first on weighted score.",),
            reference_comparison={"cement_saving_kg_per_m3": 12.0},
        )

        self.assertEqual(result.best_scenario_id, "candidate-1")
        self.assertEqual(len(result.scenarios), 1)
        self.assertEqual(result.scenarios[0].objective_scorecard.components[0].name, "target_fit")


if __name__ == "__main__":
    unittest.main()
