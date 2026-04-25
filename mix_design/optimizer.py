"""Candidate generation and search for the concrete mix design subsystem."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import optuna
import pandas as pd

from mix_design.constraints import BASE_CONSTRAINT_FIELDS
from mix_design.contracts import CandidateProposal, DesignConstraints, OptimizationResult, RangeConstraint


@dataclass
class MixDesignSpaceOptimizer:
    """Generate candidate mixes under explicit design constraints."""

    reference_dataset: pd.DataFrame
    base_columns: list[str]
    target_column: str
    design_config: dict[str, object]
    seed: int

    def _target_regime(self, target_strength_mpa: float) -> str:
        target_strength = float(target_strength_mpa)
        if target_strength <= 35.0:
            return "low_strength"
        if target_strength <= 50.0:
            return "medium_strength"
        return "high_strength"

    @staticmethod
    def _constraint_bounds(constraint: RangeConstraint, *, name: str) -> tuple[float, float]:
        minimum = constraint.min
        maximum = constraint.max
        if minimum is None or maximum is None:
            raise ValueError(f"Constraint range is incomplete for '{name}'.")
        if float(minimum) > float(maximum):
            raise ValueError(f"Constraint range is invalid for '{name}': min={minimum}, max={maximum}.")
        return float(minimum), float(maximum)

    def _mix_signature(self, mix_design: dict[str, float]) -> tuple[float, ...]:
        return tuple(round(float(mix_design[column]), 4) for column in self.base_columns)

    def _water_cement_max(self, constraints: DesignConstraints) -> float:
        if constraints.water_cement_ratio.max is None:
            return float(np.inf)
        return float(constraints.water_cement_ratio.max)

    def _water_minimum(self, constraints: DesignConstraints) -> float:
        if constraints.water.fixed is not None:
            return float(constraints.water.fixed)
        if constraints.water.min is not None:
            return float(constraints.water.min)
        return 0.0

    def _apply_constraint_to_value(
        self,
        value: float,
        constraint: RangeConstraint,
    ) -> float:
        if constraint.fixed is not None:
            return float(constraint.fixed)
        clipped = float(value)
        if constraint.min is not None:
            clipped = max(clipped, float(constraint.min))
        if constraint.max is not None:
            clipped = min(clipped, float(constraint.max))
        return clipped

    def _enforce_water_cement_link(
        self,
        mix_design: dict[str, float],
        constraints: DesignConstraints,
    ) -> dict[str, float]:
        water_cement_max = self._water_cement_max(constraints)
        if not np.isfinite(water_cement_max) or water_cement_max <= 0.0:
            return mix_design

        water_minimum = self._water_minimum(constraints)
        cement = max(float(mix_design["cement"]), water_minimum / water_cement_max)
        if constraints.cement.max is not None:
            cement = min(cement, float(constraints.cement.max))
        if constraints.cement.min is not None:
            cement = max(cement, float(constraints.cement.min))

        water = min(float(mix_design["water"]), cement * water_cement_max)
        water = max(water, water_minimum)
        if constraints.water.max is not None:
            water = min(water, float(constraints.water.max))

        enforced = dict(mix_design)
        enforced["cement"] = float(cement)
        enforced["water"] = float(water)
        return enforced

    def _target_conditioned_reference_rows(self, target_strength_mpa: float) -> pd.DataFrame:
        candidate_rows = self.reference_dataset.copy()
        windows = [2.0, 4.0, 6.0, 8.0]
        filtered = candidate_rows.iloc[0:0].copy()
        for window in windows:
            filtered = candidate_rows[
                candidate_rows[self.target_column].between(
                    float(target_strength_mpa) - window,
                    float(target_strength_mpa) + window,
                )
            ].copy()
            if len(filtered) >= 8:
                break
        if filtered.empty:
            target_regime = self._target_regime(target_strength_mpa)
            if target_regime == "low_strength":
                filtered = candidate_rows[candidate_rows[self.target_column] <= 35.0].copy()
            elif target_regime == "medium_strength":
                filtered = candidate_rows[candidate_rows[self.target_column].between(30.0, 50.0)].copy()
            else:
                filtered = candidate_rows[candidate_rows[self.target_column] >= 45.0].copy()
        return filtered if not filtered.empty else candidate_rows

    def _engineering_prior_mixes(
        self,
        target_strength_mpa: float,
        constraints: DesignConstraints,
    ) -> list[dict[str, float]]:
        candidate_rows = self._target_conditioned_reference_rows(target_strength_mpa).copy()
        if candidate_rows.empty:
            return []

        target_regime = self._target_regime(target_strength_mpa)
        sort_columns = ["cement", self.target_column] if target_regime == "low_strength" else [self.target_column, "cement"]
        candidate_rows["target_gap"] = (candidate_rows[self.target_column] - float(target_strength_mpa)).abs()
        candidate_rows = candidate_rows.sort_values(["target_gap", *sort_columns]).head(24)

        priors: list[dict[str, float]] = []
        seen_signatures: set[tuple[float, ...]] = set()
        for _, row in candidate_rows.iterrows():
            prior = {
                column: self._apply_constraint_to_value(float(row[column]), getattr(constraints, column))
                for column in self.base_columns
            }
            prior = self._enforce_water_cement_link(prior, constraints)
            signature = self._mix_signature(prior)
            if signature in seen_signatures:
                continue
            seen_signatures.add(signature)
            priors.append(prior)
            if len(priors) >= 6:
                break
        return priors

    def _warm_start_mixes(
        self,
        target_strength_mpa: float,
        constraints: DesignConstraints,
    ) -> list[dict[str, float]]:
        if float(target_strength_mpa) > 30.0:
            return []

        candidate_rows = self.reference_dataset[self.reference_dataset[self.target_column] <= 30.0].copy()
        if candidate_rows.empty:
            return []

        candidate_rows["target_gap"] = (candidate_rows[self.target_column] - float(target_strength_mpa)).abs()
        candidate_rows = candidate_rows.sort_values(["target_gap", "cement", self.target_column]).head(30)

        warm_starts: list[dict[str, float]] = []
        seen_signatures: set[tuple[float, ...]] = set()
        for _, row in candidate_rows.iterrows():
            mix_design = {
                column: self._apply_constraint_to_value(float(row[column]), getattr(constraints, column))
                for column in self.base_columns
            }
            mix_design = self._enforce_water_cement_link(mix_design, constraints)
            signature = self._mix_signature(mix_design)
            if signature in seen_signatures:
                continue
            seen_signatures.add(signature)
            warm_starts.append(mix_design)
            if len(warm_starts) >= 3:
                break
        return warm_starts

    def _sample_trial_mix(
        self,
        trial: optuna.trial.Trial,
        constraints: DesignConstraints,
    ) -> dict[str, float]:
        mix_design: dict[str, float] = {}
        water_cement_max = self._water_cement_max(constraints)
        water_minimum = self._water_minimum(constraints)

        for column in self.base_columns:
            constraint = getattr(constraints, column)
            if constraint.fixed is not None:
                mix_design[column] = float(constraint.fixed)
                continue

            minimum, maximum = self._constraint_bounds(constraint, name=column)
            if column == "cement" and np.isfinite(water_cement_max) and water_cement_max > 0.0:
                minimum = max(minimum, water_minimum / water_cement_max)
            if column == "water" and np.isfinite(water_cement_max) and water_cement_max > 0.0:
                maximum = min(maximum, float(mix_design["cement"]) * water_cement_max)
            if minimum > maximum:
                raise ValueError(f"Constraint range is invalid for '{column}': min={minimum}, max={maximum}.")
            mix_design[column] = float(trial.suggest_float(column, minimum, maximum))

        return mix_design

    def optimize(
        self,
        *,
        target_strength_mpa: float,
        constraints: DesignConstraints,
        candidate_limit: int,
        evaluate_candidate: Callable[[CandidateProposal], float],
    ) -> OptimizationResult:
        """Run search and return typed candidate proposals ranked by the evaluation callback."""

        initial_samples = int(self.design_config["search"]["initial_samples"])
        default_budget = max(300, initial_samples // 8)
        if float(target_strength_mpa) < 30.0:
            default_budget = max(default_budget, 420)
        trial_budget = int(self.design_config["search"].get("optuna_trials", default_budget))

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        sampler = optuna.samplers.TPESampler(seed=int(self.seed), warn_independent_sampling=False)
        study = optuna.create_study(direction="minimize", sampler=sampler)

        proposal_counter = 0
        best_by_signature: dict[tuple[float, ...], tuple[CandidateProposal, float]] = {}

        def record_proposal(mix_design: dict[str, float], source: str) -> float:
            nonlocal proposal_counter
            normalized_mix = {column: float(mix_design[column]) for column in self.base_columns}
            proposal_counter += 1
            proposal = CandidateProposal(
                proposal_id=f"{source}-{proposal_counter:04d}",
                mix_design=normalized_mix,
                source=source,
            )
            score = float(evaluate_candidate(proposal))
            signature = self._mix_signature(normalized_mix)
            current = best_by_signature.get(signature)
            if current is None or score < current[1]:
                best_by_signature[signature] = (proposal, score)
            return score

        engineering_priors = self._engineering_prior_mixes(target_strength_mpa, constraints)
        for prior_mix in engineering_priors:
            record_proposal(prior_mix, "engineering_prior")

        for warm_start in self._warm_start_mixes(target_strength_mpa, constraints):
            study.enqueue_trial(
                {
                    column: value
                    for column, value in warm_start.items()
                    if getattr(constraints, column).fixed is None
                }
            )

        for prior_mix in engineering_priors:
            study.enqueue_trial(
                {
                    column: value
                    for column, value in prior_mix.items()
                    if getattr(constraints, column).fixed is None
                }
            )

        def objective(trial: optuna.trial.Trial) -> float:
            return record_proposal(self._sample_trial_mix(trial, constraints), "optuna_trial")

        study.optimize(objective, n_trials=trial_budget, show_progress_bar=False)
        if not best_by_signature:
            raise RuntimeError("No candidate mixes were evaluated during optimization.")

        proposal_pool_limit = max(int(candidate_limit) * 8, int(self.design_config.get("top_ranked_candidates", 5)), 16)
        ordered_proposals = tuple(
            proposal
            for proposal, _ in sorted(
                best_by_signature.values(),
                key=lambda item: (item[1], item[0].proposal_id),
            )[:proposal_pool_limit]
        )
        return OptimizationResult(
            proposals=ordered_proposals,
            metadata={
                "trial_budget": trial_budget,
                "evaluated_candidates": len(best_by_signature),
            },
        )
