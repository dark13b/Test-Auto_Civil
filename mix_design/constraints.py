"""Explicit constraint loading and evaluation for mix design."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping

from mix_design.contracts import ConstraintCheck, ConstraintEvaluation, DesignConstraints, RangeConstraint


_BASE_COLUMNS = (
    "cement",
    "slag",
    "fly_ash",
    "water",
    "superplasticizer",
    "coarse_aggregate",
    "fine_aggregate",
    "age",
)
_RATIO_COLUMNS = (
    "water_cement_ratio",
    "fly_ash_replacement_ratio",
    "slag_replacement_ratio",
)


def _constraint_from_raw(raw_value: Any) -> RangeConstraint:
    if raw_value is None:
        return RangeConstraint()
    if isinstance(raw_value, Mapping):
        return RangeConstraint(
            min=None if raw_value.get("min") is None else float(raw_value["min"]),
            max=None if raw_value.get("max") is None else float(raw_value["max"]),
            fixed=None if raw_value.get("fixed") is None else float(raw_value["fixed"]),
        )
    return RangeConstraint(fixed=float(raw_value))


def load_design_constraints(config: Mapping[str, Any], overrides: Mapping[str, Any] | None = None) -> DesignConstraints:
    """Load config-defined design constraints into a typed contract."""

    design_config = config["engineering"]["design_tool"]
    raw_constraints = dict(design_config.get("constraints", {}))
    if overrides:
        for key, value in overrides.items():
            if key == "tolerance_mpa":
                continue
            raw_constraints[key] = value

    tolerance = float(overrides["tolerance_mpa"]) if overrides and "tolerance_mpa" in overrides else float(
        design_config.get("target_tolerance_mpa", 2.0)
    )
    return DesignConstraints(
        cement=_constraint_from_raw(raw_constraints.get("cement")),
        slag=_constraint_from_raw(raw_constraints.get("slag")),
        fly_ash=_constraint_from_raw(raw_constraints.get("fly_ash")),
        water=_constraint_from_raw(raw_constraints.get("water")),
        superplasticizer=_constraint_from_raw(raw_constraints.get("superplasticizer")),
        coarse_aggregate=_constraint_from_raw(raw_constraints.get("coarse_aggregate")),
        fine_aggregate=_constraint_from_raw(raw_constraints.get("fine_aggregate")),
        age=_constraint_from_raw(raw_constraints.get("age")),
        water_cement_ratio=_constraint_from_raw(raw_constraints.get("water_cement_ratio")),
        fly_ash_replacement_ratio=_constraint_from_raw(raw_constraints.get("fly_ash_replacement_ratio")),
        slag_replacement_ratio=_constraint_from_raw(raw_constraints.get("slag_replacement_ratio")),
        tolerance_mpa=tolerance,
    )


def apply_target_strength_bounds(constraints: DesignConstraints, target_strength_mpa: float) -> DesignConstraints:
    """Apply target-regime bounds without hiding the resulting limits."""

    target_strength = float(target_strength_mpa)
    if target_strength < 30.0:
        cement_bounds = (100.0, 260.0)
        water_bounds = (150.0, 210.0)
        water_cement_max = 1.45
    elif target_strength <= 35.0:
        cement_bounds = (110.0, 280.0)
        water_bounds = (145.0, 210.0)
        water_cement_max = 1.25
    elif target_strength <= 45.0:
        cement_bounds = (140.0, 300.0)
        water_bounds = (140.0, 210.0)
        water_cement_max = 0.55
    else:
        cement_bounds = (280.0, 500.0)
        water_bounds = (150.0, 200.0)
        water_cement_max = 0.45

    cement = RangeConstraint(
        min=max(constraints.cement.min or cement_bounds[0], cement_bounds[0]),
        max=min(constraints.cement.max or cement_bounds[1], cement_bounds[1]),
        fixed=constraints.cement.fixed,
    )
    water = RangeConstraint(
        min=max(constraints.water.min or water_bounds[0], water_bounds[0]),
        max=min(constraints.water.max or water_bounds[1], water_bounds[1]),
        fixed=constraints.water.fixed,
    )
    existing_max = constraints.water_cement_ratio.max
    bounded_max = (
        max(existing_max if existing_max is not None else water_cement_max, water_cement_max)
        if target_strength <= 35.0
        else min(existing_max if existing_max is not None else water_cement_max, water_cement_max)
    )
    water_cement_ratio = replace(constraints.water_cement_ratio, max=bounded_max)

    if cement.min is not None and cement.max is not None and cement.min > cement.max:
        raise ValueError("Target-dependent cement bounds conflict with configured constraints.")
    if water.min is not None and water.max is not None and water.min > water.max:
        raise ValueError("Target-dependent water bounds conflict with configured constraints.")

    return replace(
        constraints,
        cement=cement,
        water=water,
        water_cement_ratio=water_cement_ratio,
    )


def _evaluate_named_constraint(
    checks: list[ConstraintCheck],
    hard_failures: list[str],
    name: str,
    value: float | None,
    constraint: RangeConstraint,
) -> None:
    if value is None:
        return
    if constraint.fixed is not None:
        passed = abs(float(value) - float(constraint.fixed)) <= 1e-9
        checks.append(
            ConstraintCheck(
                name=f"{name}.fixed",
                passed=passed,
                actual=float(value),
                limit=float(constraint.fixed),
                message=f"{name} must match fixed value {constraint.fixed}.",
            )
        )
        if not passed:
            hard_failures.append(f"{name}.fixed")
    if constraint.min is not None:
        passed = float(value) >= float(constraint.min)
        checks.append(
            ConstraintCheck(
                name=f"{name}.min",
                passed=passed,
                actual=float(value),
                limit=float(constraint.min),
                message=f"{name} must be at least {constraint.min}.",
            )
        )
        if not passed:
            hard_failures.append(f"{name}.min")
    if constraint.max is not None:
        passed = float(value) <= float(constraint.max)
        checks.append(
            ConstraintCheck(
                name=f"{name}.max",
                passed=passed,
                actual=float(value),
                limit=float(constraint.max),
                message=f"{name} must be at most {constraint.max}.",
            )
        )
        if not passed:
            hard_failures.append(f"{name}.max")


def evaluate_constraints(
    mix_design: Mapping[str, float],
    engineered_features: Mapping[str, float],
    constraints: DesignConstraints,
) -> ConstraintEvaluation:
    """Evaluate explicit material and engineered-ratio constraints."""

    checks: list[ConstraintCheck] = []
    hard_failures: list[str] = []

    for name in _BASE_COLUMNS:
        _evaluate_named_constraint(
            checks=checks,
            hard_failures=hard_failures,
            name=name,
            value=None if mix_design.get(name) is None else float(mix_design[name]),
            constraint=getattr(constraints, name),
        )

    for name in _RATIO_COLUMNS:
        _evaluate_named_constraint(
            checks=checks,
            hard_failures=hard_failures,
            name=name,
            value=None if engineered_features.get(name) is None else float(engineered_features[name]),
            constraint=getattr(constraints, name),
        )

    return ConstraintEvaluation(
        passed=not hard_failures,
        checks=tuple(checks),
        hard_failures=tuple(hard_failures),
    )
