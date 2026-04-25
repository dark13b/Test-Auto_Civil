"""Typed contracts for the concrete mix design subsystem."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


ObjectiveName = Literal[
    "target_fit",
    "strength_fit",
    "cost_proxy",
    "cement_penalty",
    "co2_proxy",
    "validator_risk",
]
ValidatorVerdict = Literal["PASS", "WARN", "FAIL"]


@dataclass(frozen=True)
class DesignContext:
    """Optional domain context used during validation and reporting."""

    exposure_class: str | None = None
    structural_application: str | None = None


@dataclass(frozen=True)
class RangeConstraint:
    """Explicit numeric constraint for one design dimension."""

    min: float | None = None
    max: float | None = None
    fixed: float | None = None


@dataclass(frozen=True)
class DesignConstraints:
    """Explicit material and ratio constraints used during mix design."""

    cement: RangeConstraint = field(default_factory=RangeConstraint)
    slag: RangeConstraint = field(default_factory=RangeConstraint)
    fly_ash: RangeConstraint = field(default_factory=RangeConstraint)
    water: RangeConstraint = field(default_factory=RangeConstraint)
    superplasticizer: RangeConstraint = field(default_factory=RangeConstraint)
    coarse_aggregate: RangeConstraint = field(default_factory=RangeConstraint)
    fine_aggregate: RangeConstraint = field(default_factory=RangeConstraint)
    age: RangeConstraint = field(default_factory=RangeConstraint)
    water_cement_ratio: RangeConstraint = field(default_factory=RangeConstraint)
    fly_ash_replacement_ratio: RangeConstraint = field(default_factory=RangeConstraint)
    slag_replacement_ratio: RangeConstraint = field(default_factory=RangeConstraint)
    tolerance_mpa: float | None = None


@dataclass(frozen=True)
class ObjectiveSpec:
    """One explicit objective used by the objective engine."""

    name: ObjectiveName
    weight: float
    enabled: bool = True


def default_objectives() -> tuple[ObjectiveSpec, ...]:
    """Return the default scoring objective set for scenario comparison."""

    return (
        ObjectiveSpec(name="target_fit", weight=1.0),
        ObjectiveSpec(name="cement_penalty", weight=1.0),
        ObjectiveSpec(name="co2_proxy", weight=0.35),
        ObjectiveSpec(name="validator_risk", weight=0.8),
    )


@dataclass(frozen=True)
class MixDesignRequest:
    """Canonical request for scenario comparison."""

    target_strength_mpa: float
    constraints: DesignConstraints = field(default_factory=DesignConstraints)
    context: DesignContext = field(default_factory=DesignContext)
    objectives: tuple[ObjectiveSpec, ...] = field(default_factory=default_objectives)
    candidate_limit: int = 5


@dataclass(frozen=True)
class UncertaintyInterval:
    """Structured interval summary attached to a prediction."""

    predicted: float
    lower_90: float
    upper_90: float
    interval_width: float
    confidence_label: str
    target_window_overlap: float
    is_calibrated: bool = True
    warning_reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class PredictionResult:
    """Model inference result for one candidate scenario."""

    predicted_strength_mpa: float
    engineered_features: dict[str, float]
    uncertainty_interval: UncertaintyInterval
    model_artifact_id: str


@dataclass(frozen=True)
class ConstraintCheck:
    """One visible constraint check entry."""

    name: str
    passed: bool
    actual: float | str | None
    limit: float | str | None
    message: str


@dataclass(frozen=True)
class ConstraintEvaluation:
    """Constraint summary for one candidate scenario."""

    passed: bool
    checks: tuple[ConstraintCheck, ...]
    hard_failures: tuple[str, ...]


@dataclass(frozen=True)
class ValidatorOutcome:
    """Validator summary attached to a scenario."""

    overall_verdict: ValidatorVerdict
    warning_reasons: tuple[str, ...]
    failure_reasons: tuple[str, ...]
    hard_constraints: tuple[str, ...]
    engineering_cautions: tuple[str, ...]
    data_review_flags: tuple[str, ...]
    contextual_summary: str
    confidence_of_warning_assessment: str
    raw_report: dict[str, Any]


@dataclass(frozen=True)
class ObjectiveComponentScore:
    """One objective contribution to the final ranking."""

    name: str
    weight: float
    raw_value: float
    weighted_score: float
    explanation: str


@dataclass(frozen=True)
class ObjectiveScorecard:
    """Scoring breakdown used to compare candidate scenarios."""

    total_score: float
    components: tuple[ObjectiveComponentScore, ...]
    rank_explanation: tuple[str, ...]


@dataclass(frozen=True)
class CandidateProposal:
    """One optimizer-generated candidate before scenario packaging."""

    proposal_id: str
    mix_design: dict[str, float]
    source: str


@dataclass(frozen=True)
class OptimizationResult:
    """Typed optimizer output used by the facade."""

    proposals: tuple[CandidateProposal, ...]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CandidateScenario:
    """Canonical scenario comparison unit."""

    scenario_id: str
    mix_design: dict[str, float]
    prediction: PredictionResult
    constraints: ConstraintEvaluation
    validator: ValidatorOutcome
    objective_scorecard: ObjectiveScorecard
    success: bool
    source: str


@dataclass(frozen=True)
class ScenarioComparisonResult:
    """Canonical comparison output for one design request."""

    request: MixDesignRequest
    best_scenario_id: str
    scenarios: tuple[CandidateScenario, ...]
    comparison_summary: tuple[str, ...]
    reference_comparison: dict[str, float | None]
