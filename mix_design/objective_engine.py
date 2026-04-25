"""Objective scoring layer for the concrete mix design subsystem."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

from mix_design.contracts import (
    CandidateScenario,
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

    cost_proxy: str = "cement_content"
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
        if deviation > tolerance:
            outside_window = (deviation - tolerance) / tolerance
            raw_value += outside_window * outside_window * 100.0
        explanation = (
            f"Predicted strength is {deviation:.2f} MPa away from the {target_strength_mpa:.2f} MPa target "
            f"using a {tolerance:.2f} MPa tolerance window."
        )
        if deviation > tolerance:
            explanation += " A large explicit penalty applies once the candidate falls outside the tolerance window."
        return raw_value, explanation

    def _resolve_cost_proxy_value(
        self,
        mix_design: dict[str, float],
        prediction: PredictionResult,
    ) -> tuple[float, str]:
        proxy_name = str(self.cost_proxy).strip().lower()
        if proxy_name in {"cement", "cement_content"}:
            value = float(mix_design.get("cement", 0.0))
            return value, "cement"
        if proxy_name in mix_design:
            return float(mix_design[proxy_name]), proxy_name
        if proxy_name in prediction.engineered_features:
            return float(prediction.engineered_features[proxy_name]), proxy_name
        raise ValueError(f"Unsupported objective cost proxy: {self.cost_proxy}")

    def _score_cost_proxy(
        self,
        mix_design: dict[str, float],
        prediction: PredictionResult,
    ) -> tuple[float, str]:
        raw_value, proxy_name = self._resolve_cost_proxy_value(mix_design, prediction)
        return raw_value, f"Cost proxy uses '{proxy_name}' with a candidate value of {raw_value:.2f}."

    def _score_cement_penalty(
        self,
        mix_design: dict[str, float],
        prediction: PredictionResult,
    ) -> tuple[float, str]:
        raw_value, explanation = self._score_cost_proxy(mix_design, prediction)
        return raw_value, f"Cement penalty alias: {explanation}"

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

    def _score_engineering_quality(
        self,
        mix_design: dict[str, float],
        prediction: PredictionResult,
        constraints: ConstraintEvaluation,
        validator: ValidatorOutcome,
        target_strength_mpa: float,
        design_constraints: DesignConstraints,
    ) -> tuple[float, str]:
        tolerance = max(float(design_constraints.tolerance_mpa or 0.0), 1e-6)
        target_gap = abs(float(prediction.predicted_strength_mpa) - float(target_strength_mpa))
        target_penalty = (target_gap / tolerance) * 25.0

        failed_checks = tuple(check for check in constraints.checks if not check.passed)
        material_failures = tuple(
            check for check in failed_checks if str(check.name).split(".", 1)[0] in {
                "cement",
                "slag",
                "fly_ash",
                "water",
                "superplasticizer",
                "coarse_aggregate",
                "fine_aggregate",
                "age",
            }
        )
        constraint_penalty = float(len(constraints.hard_failures) * 250.0)
        material_penalty = float(len(material_failures) * 150.0)

        water_cement_ratio = prediction.engineered_features.get("water_cement_ratio")
        water_cement_penalty = 0.0
        water_cement_note = "no configured W/C ceiling was available"
        if water_cement_ratio is not None and design_constraints.water_cement_ratio.max is not None:
            ratio = float(water_cement_ratio)
            ceiling = max(float(design_constraints.water_cement_ratio.max), 1e-6)
            utilization = ratio / ceiling
            water_cement_penalty = max(utilization, 0.0) * 20.0
            if ratio > ceiling:
                water_cement_penalty += ((ratio - ceiling) / ceiling) * 500.0
            water_cement_note = f"W/C={ratio:.3f} against max={ceiling:.3f}"

        uncertainty_width = max(float(prediction.uncertainty_interval.interval_width), 0.0)
        uncertainty_penalty = (uncertainty_width / tolerance) * 10.0
        if float(prediction.uncertainty_interval.target_window_overlap) <= 0.0:
            uncertainty_penalty += 50.0
        confidence_label = str(prediction.uncertainty_interval.confidence_label).strip().upper()
        if not bool(prediction.uncertainty_interval.is_calibrated):
            uncertainty_penalty += 100.0
        if confidence_label in {"LOW", "UNKNOWN", "UNCALIBRATED"}:
            uncertainty_penalty += 75.0
        uncertainty_penalty += float(len(prediction.uncertainty_interval.warning_reasons) * 25.0)

        verdict_penalty = {
            "PASS": 0.0,
            "WARN": 75.0,
            "FAIL": 1000.0,
        }.get(str(validator.overall_verdict), 300.0)
        constructability_penalty = (
            float(len(validator.engineering_cautions) * 35.0)
            + float(len(validator.data_review_flags) * 20.0)
            + float(len(validator.warning_reasons) * 10.0)
            + float(len(validator.failure_reasons) * 200.0)
        )

        raw_value = (
            target_penalty
            + constraint_penalty
            + material_penalty
            + water_cement_penalty
            + uncertainty_penalty
            + verdict_penalty
            + constructability_penalty
        )
        explanation = (
            "Engineering quality combines target distance, explicit constraint failures, material range failures, "
            f"water/cement safety ({water_cement_note}), uncertainty interval width={uncertainty_width:.2f} MPa, "
            f"uncertainty confidence={confidence_label}, calibrated={prediction.uncertainty_interval.is_calibrated}, "
            f"validator verdict={validator.overall_verdict}, and practical constructability cautions from the validator. "
            f"Penalty terms: target={target_penalty:.2f}, constraints={constraint_penalty:.2f}, "
            f"materials={material_penalty:.2f}, water_cement={water_cement_penalty:.2f}, "
            f"uncertainty={uncertainty_penalty:.2f}, verdict={verdict_penalty:.2f}, "
            f"constructability={constructability_penalty:.2f}."
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
            "strength_fit": lambda: self._score_target_fit(prediction, target_strength_mpa, design_constraints),
            "cost_proxy": lambda: self._score_cost_proxy(mix_design, prediction),
            "cement_penalty": lambda: self._score_cement_penalty(mix_design, prediction),
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
            explanations.append(
                f"{objective.name} contributed {weighted_score:.2f} (raw={raw_value:.2f}, weight={objective.weight:.2f})."
            )

        raw_value, explanation = self._score_engineering_quality(
            mix_design,
            prediction,
            constraints,
            validator,
            target_strength_mpa,
            design_constraints,
        )
        components.append(
            ObjectiveComponentScore(
                name="engineering_quality",
                weight=1.0,
                raw_value=float(raw_value),
                weighted_score=float(raw_value),
                explanation=explanation,
            )
        )
        explanations.append(f"engineering_quality contributed {raw_value:.2f} as a mandatory safety ranking term.")

        total_score = float(sum(component.weighted_score for component in components))
        if explanations:
            explanations.append(f"Total weighted score is {total_score:.2f}. Lower scores rank better.")
        rank_explanation = tuple(explanations or ["No enabled objectives produced a score."])
        return ObjectiveScorecard(
            total_score=total_score,
            components=tuple(components),
            rank_explanation=rank_explanation,
        )

    @staticmethod
    def _component_raw(scorecard: ObjectiveScorecard, *names: str) -> float:
        for component in scorecard.components:
            if component.name in names:
                return float(component.raw_value)
        return 0.0

    def rank_scenarios(self, scenarios: Sequence[CandidateScenario]) -> tuple[CandidateScenario, ...]:
        """Apply deterministic ranking after the facade has packaged scenarios."""

        verdict_rank = {
            "PASS": 0,
            "WARN": 1,
            "FAIL": 2,
        }
        return tuple(
            sorted(
                scenarios,
                key=lambda scenario: (
                    0 if scenario.success else 1,
                    verdict_rank.get(str(scenario.validator.overall_verdict), 3),
                    float(scenario.objective_scorecard.total_score),
                    self._component_raw(scenario.objective_scorecard, "engineering_quality"),
                    self._component_raw(scenario.objective_scorecard, "target_fit", "strength_fit"),
                    self._component_raw(scenario.objective_scorecard, "cost_proxy", "cement_penalty"),
                    float(scenario.mix_design.get("cement", 0.0)),
                    str(scenario.scenario_id),
                ),
            )
        )

    def build_comparison_summary(
        self,
        ranked_scenarios: Sequence[CandidateScenario],
    ) -> tuple[str, ...]:
        """Explain why the top scenario ranked ahead of the next best alternatives."""

        if not ranked_scenarios:
            return ("No candidate scenarios were available for comparison.",)

        best = ranked_scenarios[0]
        summary = [
            (
                f"{best.scenario_id} ranked first because it "
                f"{'satisfied' if best.success else 'did not satisfy'} the explicit feasibility gates "
                f"and achieved a weighted objective score of {best.objective_scorecard.total_score:.2f}."
            )
        ]
        summary.extend(best.objective_scorecard.rank_explanation)

        if len(ranked_scenarios) > 1:
            runner_up = ranked_scenarios[1]
            summary.append(
                f"Runner-up {runner_up.scenario_id} scored {runner_up.objective_scorecard.total_score:.2f}, "
                f"a gap of {runner_up.objective_scorecard.total_score - best.objective_scorecard.total_score:.2f}."
            )
        return tuple(summary)
