"""Objective scoring layer for the concrete mix design subsystem."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from mix_design.contracts import (
    ConstraintEvaluation,
    DesignConstraints,
    ObjectiveComponentScore,
    ObjectiveScorecard,
    ObjectiveSpec,
    PredictionResult,
    ValidatorOutcome,
)


@dataclass
class MixObjectiveEngine:
    """Score candidate scenarios against explicit objectives."""

    cement_co2_factor: float = 1.0
    slag_co2_factor: float = 0.15
    fly_ash_co2_factor: float = 0.10

    def _score_target_fit(
        self,
        prediction: PredictionResult,
        target_strength_mpa: float,
        design_constraints: DesignConstraints,
    ) -> tuple[float, str]:
        tolerance = max(float(design_constraints.tolerance_mpa), 1e-6)
        deviation = abs(float(prediction.predicted_strength_mpa) - float(target_strength_mpa))
        raw_value = deviation / tolerance
        explanation = (
            f"Predicted strength is {deviation:.2f} MPa away from the {target_strength_mpa:.2f} MPa target "
            f"using a {tolerance:.2f} MPa tolerance window."
        )
        return raw_value, explanation

    def _score_cement_penalty(self, mix_design: dict[str, float]) -> tuple[float, str]:
        cement = float(mix_design.get("cement", 0.0))
        return cement, f"Cement penalty uses {cement:.2f} kg/m^3 cement as the direct cost proxy."

    def _score_co2_proxy(self, mix_design: dict[str, float]) -> tuple[float, str]:
        cement = float(mix_design.get("cement", 0.0))
        slag = float(mix_design.get("slag", 0.0))
        fly_ash = float(mix_design.get("fly_ash", 0.0))
        raw_value = (
            cement * self.cement_co2_factor
            + slag * self.slag_co2_factor
            + fly_ash * self.fly_ash_co2_factor
        )
        explanation = (
            "CO2 proxy weights cement most heavily and applies reduced factors to slag and fly ash."
        )
        return raw_value, explanation

    def _score_validator_risk(
        self,
        constraints: ConstraintEvaluation,
        validator: ValidatorOutcome,
    ) -> tuple[float, str]:
        verdict_penalty = {
            "PASS": 0.0,
            "WARN": 50.0,
            "FAIL": 500.0,
        }.get(str(validator.overall_verdict), 250.0)
        warning_penalty = float(len(validator.warning_reasons) * 10.0)
        failure_penalty = float(len(validator.failure_reasons) * 100.0)
        constraint_penalty = float(len(constraints.hard_failures) * 100.0)
        raw_value = verdict_penalty + warning_penalty + failure_penalty + constraint_penalty
        explanation = (
            f"Validator risk reflects verdict={validator.overall_verdict}, "
            f"{len(validator.warning_reasons)} warning(s), {len(validator.failure_reasons)} failure(s), "
            f"and {len(constraints.hard_failures)} explicit constraint failure(s)."
        )
        return raw_value, explanation

    def score_candidate(
        self,
        *,
        mix_design: dict[str, float],
        prediction: PredictionResult,
        constraints: ConstraintEvaluation,
        validator: ValidatorOutcome,
        target_strength_mpa: float,
        design_constraints: DesignConstraints,
        objectives: tuple[ObjectiveSpec, ...],
    ) -> ObjectiveScorecard:
        """Score one candidate against the configured objective list."""

        scorers: dict[str, Callable[[], tuple[float, str]]] = {
            "target_fit": lambda: self._score_target_fit(prediction, target_strength_mpa, design_constraints),
            "cement_penalty": lambda: self._score_cement_penalty(mix_design),
            "co2_proxy": lambda: self._score_co2_proxy(mix_design),
            "validator_risk": lambda: self._score_validator_risk(constraints, validator),
        }

        components: list[ObjectiveComponentScore] = []
        explanations: list[str] = []
        for objective in objectives:
            if not objective.enabled:
                continue
            if objective.name not in scorers:
                raise ValueError(f"Unsupported objective: {objective.name}")
            raw_value, explanation = scorers[objective.name]()
            weighted_score = float(raw_value) * float(objective.weight)
            components.append(
                ObjectiveComponentScore(
                    name=objective.name,
                    weight=float(objective.weight),
                    raw_value=float(raw_value),
                    weighted_score=float(weighted_score),
                    explanation=explanation,
                )
            )
            explanations.append(f"{objective.name} contributed {weighted_score:.2f} to the total score.")

        total_score = float(sum(component.weighted_score for component in components))
        rank_explanation = tuple(explanations or ["No enabled objectives produced a score."])
        return ObjectiveScorecard(
            total_score=total_score,
            components=tuple(components),
            rank_explanation=rank_explanation,
        )
