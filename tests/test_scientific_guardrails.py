"""Tests for the scientific guardrails scoring layer in MixObjectiveEngine."""

from __future__ import annotations

import unittest

from mix_design.contracts import (
    ConstraintEvaluation,
    DesignConstraints,
    ObjectiveSpec,
    PredictionResult,
    RangeConstraint,
    UncertaintyInterval,
    ValidatorOutcome,
)
from mix_design.objective_engine import MixObjectiveEngine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_interval(
    *,
    width: float = 4.0,
    overlap: float = 1.0,
    calibrated: bool = True,
    label: str = "HIGH",
) -> UncertaintyInterval:
    return UncertaintyInterval(
        predicted=30.0,
        lower_90=28.0,
        upper_90=32.0,
        interval_width=width,
        confidence_label=label,
        target_window_overlap=overlap,
        is_calibrated=calibrated,
        warning_reasons=(),
    )


def _make_prediction(
    *,
    wc_ratio: float = 0.50,
    strength: float = 30.0,
    interval_width: float = 4.0,
    interval_overlap: float = 1.0,
) -> PredictionResult:
    return PredictionResult(
        predicted_strength_mpa=strength,
        engineered_features={
            "water_cement_ratio": wc_ratio,
            "effective_binder": 300.0,
        },
        uncertainty_interval=_make_interval(width=interval_width, overlap=interval_overlap),
        model_artifact_id="test-model",
    )


def _make_pass_validator() -> ValidatorOutcome:
    return ValidatorOutcome(
        overall_verdict="PASS",
        warning_reasons=(),
        failure_reasons=(),
        hard_constraints=(),
        engineering_cautions=(),
        data_review_flags=(),
        contextual_summary="",
        confidence_of_warning_assessment="",
        raw_report={},
    )


def _make_fail_validator(reasons: tuple[str, ...] = ("W/C too high",)) -> ValidatorOutcome:
    return ValidatorOutcome(
        overall_verdict="FAIL",
        warning_reasons=(),
        failure_reasons=reasons,
        hard_constraints=reasons,
        engineering_cautions=(),
        data_review_flags=(),
        contextual_summary="",
        confidence_of_warning_assessment="",
        raw_report={},
    )


def _make_constraints(
    *,
    wc_max: float = 0.70,
    cement_min: float = 100.0,
    cement_max: float = 450.0,
    water_min: float = 130.0,
    water_max: float = 210.0,
    tolerance: float = 2.0,
) -> DesignConstraints:
    return DesignConstraints(
        cement=RangeConstraint(min=cement_min, max=cement_max),
        water=RangeConstraint(min=water_min, max=water_max),
        water_cement_ratio=RangeConstraint(max=wc_max),
        tolerance_mpa=tolerance,
    )


def _make_clean_constraints_eval() -> ConstraintEvaluation:
    return ConstraintEvaluation(passed=True, checks=(), hard_failures=())


def _make_objectives() -> tuple[ObjectiveSpec, ...]:
    return (
        ObjectiveSpec(name="target_fit", weight=1.0),
        ObjectiveSpec(name="cement_penalty", weight=1.0),
    )


# ---------------------------------------------------------------------------
# Guardrail 1 — W/C ratio
# ---------------------------------------------------------------------------


class GuardrailWCRatioTests(unittest.TestCase):
    """Guardrail 1: penalize very high water/cement ratio."""

    def test_wc_above_ceiling_triggers_positive_guardrail_penalty(self) -> None:
        engine = MixObjectiveEngine()
        penalty, _ = engine._score_scientific_guardrails(
            mix_design={"cement": 300.0, "water": 180.0},
            prediction=_make_prediction(wc_ratio=0.80),  # above 0.70 ceiling
            validator=_make_pass_validator(),
            design_constraints=_make_constraints(wc_max=0.70),
        )
        self.assertGreater(penalty, 0.0)

    def test_wc_above_ceiling_produces_warning_mentioning_ratio(self) -> None:
        engine = MixObjectiveEngine()
        _, warnings = engine._score_scientific_guardrails(
            mix_design={"cement": 300.0, "water": 180.0},
            prediction=_make_prediction(wc_ratio=0.85),
            validator=_make_pass_validator(),
            design_constraints=_make_constraints(wc_max=0.70),
        )
        self.assertTrue(
            any("W/C" in w or "water_cement" in w or "water/cement" in w.lower() for w in warnings),
            f"Expected W/C warning, got: {warnings}",
        )

    def test_higher_wc_ratio_gets_higher_guardrail_penalty(self) -> None:
        engine = MixObjectiveEngine()
        constraints = _make_constraints(wc_max=0.70)
        low_penalty, _ = engine._score_scientific_guardrails(
            mix_design={"cement": 300.0, "water": 180.0},
            prediction=_make_prediction(wc_ratio=0.72),
            validator=_make_pass_validator(),
            design_constraints=constraints,
        )
        high_penalty, _ = engine._score_scientific_guardrails(
            mix_design={"cement": 300.0, "water": 180.0},
            prediction=_make_prediction(wc_ratio=0.90),
            validator=_make_pass_validator(),
            design_constraints=constraints,
        )
        self.assertGreater(high_penalty, low_penalty)

    def test_wc_within_ceiling_produces_no_wc_warning(self) -> None:
        engine = MixObjectiveEngine()
        _, warnings = engine._score_scientific_guardrails(
            mix_design={"cement": 300.0, "water": 150.0},
            prediction=_make_prediction(wc_ratio=0.50),
            validator=_make_pass_validator(),
            design_constraints=_make_constraints(wc_max=0.70),
        )
        self.assertFalse(
            any("W/C" in w or "water_cement" in w for w in warnings),
            f"Unexpected W/C warning on clean candidate: {warnings}",
        )


# ---------------------------------------------------------------------------
# Guardrail 2 — cement bounds
# ---------------------------------------------------------------------------


class GuardrailCementBoundsTests(unittest.TestCase):
    """Guardrail 2: penalize cement content outside realistic configured bounds."""

    def test_cement_above_max_triggers_penalty_and_warning(self) -> None:
        engine = MixObjectiveEngine()
        penalty, warnings = engine._score_scientific_guardrails(
            mix_design={"cement": 500.0, "water": 160.0},  # 500 > 450 max
            prediction=_make_prediction(),
            validator=_make_pass_validator(),
            design_constraints=_make_constraints(cement_max=450.0),
        )
        self.assertGreater(penalty, 0.0)
        self.assertTrue(
            any("cement" in w.lower() for w in warnings),
            f"Expected cement warning, got: {warnings}",
        )

    def test_cement_below_min_triggers_penalty_and_warning(self) -> None:
        engine = MixObjectiveEngine()
        penalty, warnings = engine._score_scientific_guardrails(
            mix_design={"cement": 50.0, "water": 160.0},  # 50 < 100 min
            prediction=_make_prediction(),
            validator=_make_pass_validator(),
            design_constraints=_make_constraints(cement_min=100.0),
        )
        self.assertGreater(penalty, 0.0)
        self.assertTrue(
            any("cement" in w.lower() for w in warnings),
            f"Expected cement warning, got: {warnings}",
        )

    def test_cement_within_bounds_produces_no_cement_warning(self) -> None:
        engine = MixObjectiveEngine()
        _, warnings = engine._score_scientific_guardrails(
            mix_design={"cement": 300.0, "water": 160.0},
            prediction=_make_prediction(),
            validator=_make_pass_validator(),
            design_constraints=_make_constraints(cement_min=100.0, cement_max=450.0),
        )
        self.assertFalse(
            any("cement" in w.lower() for w in warnings),
            f"Unexpected cement warning: {warnings}",
        )


# ---------------------------------------------------------------------------
# Guardrail 3 — water bounds
# ---------------------------------------------------------------------------


class GuardrailWaterBoundsTests(unittest.TestCase):
    """Guardrail 3: penalize water content outside realistic configured bounds."""

    def test_water_above_max_triggers_penalty_and_warning(self) -> None:
        engine = MixObjectiveEngine()
        penalty, warnings = engine._score_scientific_guardrails(
            mix_design={"cement": 300.0, "water": 230.0},  # 230 > 210 max
            prediction=_make_prediction(),
            validator=_make_pass_validator(),
            design_constraints=_make_constraints(water_max=210.0),
        )
        self.assertGreater(penalty, 0.0)
        self.assertTrue(
            any("water" in w.lower() for w in warnings),
            f"Expected water warning, got: {warnings}",
        )

    def test_water_below_min_triggers_penalty_and_warning(self) -> None:
        engine = MixObjectiveEngine()
        penalty, warnings = engine._score_scientific_guardrails(
            mix_design={"cement": 300.0, "water": 100.0},  # 100 < 130 min
            prediction=_make_prediction(),
            validator=_make_pass_validator(),
            design_constraints=_make_constraints(water_min=130.0),
        )
        self.assertGreater(penalty, 0.0)
        self.assertTrue(
            any("water" in w.lower() for w in warnings),
            f"Expected water warning, got: {warnings}",
        )

    def test_water_within_bounds_produces_no_water_warning(self) -> None:
        engine = MixObjectiveEngine()
        _, warnings = engine._score_scientific_guardrails(
            mix_design={"cement": 300.0, "water": 170.0},
            prediction=_make_prediction(),
            validator=_make_pass_validator(),
            design_constraints=_make_constraints(water_min=130.0, water_max=210.0),
        )
        self.assertFalse(
            any("water" in w.lower() for w in warnings),
            f"Unexpected water warning: {warnings}",
        )


# ---------------------------------------------------------------------------
# Guardrail 4 — FAIL verdict
# ---------------------------------------------------------------------------


class GuardrailFailVerdictTests(unittest.TestCase):
    """Guardrail 4: penalize candidates with FAIL validator verdict."""

    def test_fail_verdict_produces_higher_penalty_than_pass(self) -> None:
        engine = MixObjectiveEngine()
        constraints = _make_constraints()
        mix = {"cement": 300.0, "water": 170.0}
        pred = _make_prediction()
        pass_penalty, _ = engine._score_scientific_guardrails(
            mix_design=mix, prediction=pred,
            validator=_make_pass_validator(), design_constraints=constraints,
        )
        fail_penalty, _ = engine._score_scientific_guardrails(
            mix_design=mix, prediction=pred,
            validator=_make_fail_validator(), design_constraints=constraints,
        )
        self.assertGreater(fail_penalty, pass_penalty)

    def test_fail_verdict_produces_warning_mentioning_fail(self) -> None:
        engine = MixObjectiveEngine()
        _, warnings = engine._score_scientific_guardrails(
            mix_design={"cement": 300.0, "water": 170.0},
            prediction=_make_prediction(),
            validator=_make_fail_validator(("durability constraint violated",)),
            design_constraints=_make_constraints(),
        )
        self.assertTrue(
            any("FAIL" in w or "fail" in w.lower() for w in warnings),
            f"Expected FAIL warning, got: {warnings}",
        )

    def test_pass_verdict_produces_no_verdict_warning(self) -> None:
        engine = MixObjectiveEngine()
        _, warnings = engine._score_scientific_guardrails(
            mix_design={"cement": 300.0, "water": 170.0},
            prediction=_make_prediction(),
            validator=_make_pass_validator(),
            design_constraints=_make_constraints(),
        )
        self.assertFalse(
            any("FAIL" in w for w in warnings),
            f"Unexpected FAIL warning on PASS candidate: {warnings}",
        )


# ---------------------------------------------------------------------------
# Guardrail 5 — uncertainty interval width
# ---------------------------------------------------------------------------


class GuardrailUncertaintyTests(unittest.TestCase):
    """Guardrail 5: penalize candidates with very wide uncertainty interval."""

    def test_very_wide_interval_gets_higher_penalty_than_narrow(self) -> None:
        engine = MixObjectiveEngine()
        constraints = _make_constraints(tolerance=2.0)
        mix = {"cement": 300.0, "water": 170.0}
        narrow_penalty, _ = engine._score_scientific_guardrails(
            mix_design=mix,
            prediction=_make_prediction(interval_width=4.0),   # 2× tolerance
            validator=_make_pass_validator(),
            design_constraints=constraints,
        )
        wide_penalty, _ = engine._score_scientific_guardrails(
            mix_design=mix,
            prediction=_make_prediction(interval_width=10.0),  # 5× tolerance → very wide
            validator=_make_pass_validator(),
            design_constraints=constraints,
        )
        self.assertGreater(wide_penalty, narrow_penalty)

    def test_very_wide_interval_produces_uncertainty_warning(self) -> None:
        engine = MixObjectiveEngine()
        _, warnings = engine._score_scientific_guardrails(
            mix_design={"cement": 300.0, "water": 170.0},
            prediction=_make_prediction(interval_width=10.0),  # 5× tolerance
            validator=_make_pass_validator(),
            design_constraints=_make_constraints(tolerance=2.0),
        )
        self.assertTrue(
            any("uncertainty" in w.lower() or "interval" in w.lower() for w in warnings),
            f"Expected uncertainty warning, got: {warnings}",
        )

    def test_moderate_interval_produces_no_uncertainty_warning(self) -> None:
        engine = MixObjectiveEngine()
        _, warnings = engine._score_scientific_guardrails(
            mix_design={"cement": 300.0, "water": 170.0},
            prediction=_make_prediction(interval_width=4.0),  # 2× tolerance — not very wide
            validator=_make_pass_validator(),
            design_constraints=_make_constraints(tolerance=2.0),
        )
        self.assertFalse(
            any("uncertainty" in w.lower() or "interval" in w.lower() for w in warnings),
            f"Unexpected uncertainty warning: {warnings}",
        )


# ---------------------------------------------------------------------------
# Guardrail 6 — warnings surfaced in scorecard
# ---------------------------------------------------------------------------


class GuardrailWarningsInScorecardTests(unittest.TestCase):
    """Guardrail 6: surface clear warning reasons in the scorecard result."""

    def test_scorecard_exposes_guardrail_warnings_field(self) -> None:
        engine = MixObjectiveEngine()
        scorecard = engine.score_candidate(
            mix_design={"cement": 300.0, "water": 170.0},
            prediction=_make_prediction(),
            constraints=_make_clean_constraints_eval(),
            validator=_make_pass_validator(),
            target_strength_mpa=30.0,
            design_constraints=_make_constraints(),
            objectives=_make_objectives(),
        )
        self.assertTrue(hasattr(scorecard, "guardrail_warnings"))
        self.assertIsInstance(scorecard.guardrail_warnings, tuple)

    def test_fail_verdict_warning_surfaces_in_scorecard(self) -> None:
        engine = MixObjectiveEngine()
        scorecard = engine.score_candidate(
            mix_design={"cement": 300.0, "water": 170.0},
            prediction=_make_prediction(),
            constraints=_make_clean_constraints_eval(),
            validator=_make_fail_validator(("binder too low",)),
            target_strength_mpa=30.0,
            design_constraints=_make_constraints(),
            objectives=_make_objectives(),
        )
        self.assertTrue(
            any("FAIL" in w or "fail" in w.lower() for w in scorecard.guardrail_warnings),
            f"Expected FAIL in guardrail_warnings, got: {scorecard.guardrail_warnings}",
        )

    def test_high_wc_ratio_warning_surfaces_in_scorecard(self) -> None:
        engine = MixObjectiveEngine()
        scorecard = engine.score_candidate(
            mix_design={"cement": 300.0, "water": 170.0},
            prediction=_make_prediction(wc_ratio=0.85),
            constraints=_make_clean_constraints_eval(),
            validator=_make_pass_validator(),
            target_strength_mpa=30.0,
            design_constraints=_make_constraints(wc_max=0.70),
            objectives=_make_objectives(),
        )
        self.assertTrue(
            any("W/C" in w or "water_cement" in w or "water/cement" in w.lower()
                for w in scorecard.guardrail_warnings),
            f"Expected W/C in guardrail_warnings, got: {scorecard.guardrail_warnings}",
        )

    def test_clean_candidate_has_empty_guardrail_warnings(self) -> None:
        engine = MixObjectiveEngine()
        scorecard = engine.score_candidate(
            mix_design={"cement": 300.0, "water": 170.0},
            prediction=_make_prediction(wc_ratio=0.55, interval_width=4.0),
            constraints=_make_clean_constraints_eval(),
            validator=_make_pass_validator(),
            target_strength_mpa=30.0,
            design_constraints=_make_constraints(
                wc_max=0.70, cement_min=100.0, cement_max=450.0,
                water_min=130.0, water_max=210.0, tolerance=2.0,
            ),
            objectives=_make_objectives(),
        )
        self.assertEqual(scorecard.guardrail_warnings, ())

    def test_scorecard_components_include_scientific_guardrails(self) -> None:
        engine = MixObjectiveEngine()
        scorecard = engine.score_candidate(
            mix_design={"cement": 300.0, "water": 170.0},
            prediction=_make_prediction(wc_ratio=0.85),
            constraints=_make_clean_constraints_eval(),
            validator=_make_pass_validator(),
            target_strength_mpa=30.0,
            design_constraints=_make_constraints(wc_max=0.70),
            objectives=_make_objectives(),
        )
        component_names = {c.name for c in scorecard.components}
        self.assertIn("scientific_guardrails", component_names)


if __name__ == "__main__":
    unittest.main()
