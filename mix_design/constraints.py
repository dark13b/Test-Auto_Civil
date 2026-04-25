"""Explicit constraint loading and evaluation for mix design."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping

from mix_design.contracts import ConstraintCheck, ConstraintEvaluation, DesignConstraints, RangeConstraint


BASE_CONSTRAINT_FIELDS = (
    "cement",
    "slag",
    "fly_ash",
    "water",
    "superplasticizer",
    "coarse_aggregate",
    "fine_aggregate",
    "age",
)
RATIO_CONSTRAINT_FIELDS = (
    "water_cement_ratio",
    "fly_ash_replacement_ratio",
    "slag_replacement_ratio",
)
ALL_CONSTRAINT_FIELDS = BASE_CONSTRAINT_FIELDS + RATIO_CONSTRAINT_FIELDS


def constraint_from_raw(raw_value: Any) -> RangeConstraint:
    """Normalize one raw config or override value into a typed range constraint."""

    if raw_value is None:
        return RangeConstraint()
    if isinstance(raw_value, Mapping):
        return RangeConstraint(
            min=None if raw_value.get("min") is None else float(raw_value["min"]),
            max=None if raw_value.get("max") is None else float(raw_value["max"]),
            fixed=None if raw_value.get("fixed") is None else float(raw_value["fixed"]),
        )
    return RangeConstraint(fixed=float(raw_value))


def design_constraints_from_legacy(
    overrides: Mapping[str, Any] | None = None,
    *,
    tolerance_mpa: float | None = None,
) -> DesignConstraints:
    """Convert legacy mapping-style overrides into the typed constraint contract."""

    raw = overrides or {}
    resolved_tolerance = tolerance_mpa
    if resolved_tolerance is None and "tolerance_mpa" in raw:
        resolved_tolerance = float(raw["tolerance_mpa"])

    return DesignConstraints(
        cement=constraint_from_raw(raw.get("cement")),
        slag=constraint_from_raw(raw.get("slag")),
        fly_ash=constraint_from_raw(raw.get("fly_ash")),
        water=constraint_from_raw(raw.get("water")),
        superplasticizer=constraint_from_raw(raw.get("superplasticizer")),
        coarse_aggregate=constraint_from_raw(raw.get("coarse_aggregate")),
        fine_aggregate=constraint_from_raw(raw.get("fine_aggregate")),
        age=constraint_from_raw(raw.get("age")),
        water_cement_ratio=constraint_from_raw(raw.get("water_cement_ratio")),
        fly_ash_replacement_ratio=constraint_from_raw(raw.get("fly_ash_replacement_ratio")),
        slag_replacement_ratio=constraint_from_raw(raw.get("slag_replacement_ratio")),
        tolerance_mpa=None if resolved_tolerance is None else float(resolved_tolerance),
    )


def merge_design_constraints(defaults: DesignConstraints, overrides: DesignConstraints | None) -> DesignConstraints:
    """Merge request overrides into config-derived defaults without hiding the result."""

    if overrides is None:
        return defaults

    merged_fields: dict[str, RangeConstraint] = {}
    for field_name in ALL_CONSTRAINT_FIELDS:
        base_constraint = getattr(defaults, field_name)
        override_constraint = getattr(overrides, field_name)
        merged_fields[field_name] = RangeConstraint(
            min=override_constraint.min if override_constraint.min is not None else base_constraint.min,
            max=override_constraint.max if override_constraint.max is not None else base_constraint.max,
            fixed=override_constraint.fixed if override_constraint.fixed is not None else base_constraint.fixed,
        )

    return DesignConstraints(
        **merged_fields,
        tolerance_mpa=(
            float(overrides.tolerance_mpa)
            if overrides.tolerance_mpa is not None
            else defaults.tolerance_mpa
        ),
    )


def load_design_constraints(config: Mapping[str, Any], overrides: Mapping[str, Any] | None = None) -> DesignConstraints:
    """Load config-defined design constraints into a typed contract."""

    design_config = config["engineering"]["design_tool"]
    default_constraints = design_constraints_from_legacy(
        design_config.get("constraints", {}),
        tolerance_mpa=float(design_config.get("target_tolerance_mpa", 2.0)),
    )
    if not overrides:
        return default_constraints
    return merge_design_constraints(default_constraints, design_constraints_from_legacy(overrides))


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
    bounded_max = min(
        existing_max if existing_max is not None else water_cement_max,
        water_cement_max,
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


def ensure_default_age_constraint(constraints: DesignConstraints, default_age_days: float) -> DesignConstraints:
    """Make the design age explicit when the request did not override it."""

    if constraints.age.fixed is not None:
        return constraints
    return replace(
        constraints,
        age=RangeConstraint(
            min=constraints.age.min,
            max=constraints.age.max,
            fixed=float(default_age_days),
        ),
    )


def prepare_design_constraints(
    config: Mapping[str, Any],
    *,
    target_strength_mpa: float,
    overrides: DesignConstraints | Mapping[str, Any] | None = None,
) -> DesignConstraints:
    """Merge config defaults, request overrides, regime bounds, and default age."""

    default_constraints = load_design_constraints(config)
    typed_overrides: DesignConstraints | None
    if overrides is None:
        typed_overrides = None
    elif isinstance(overrides, DesignConstraints):
        typed_overrides = overrides
    else:
        typed_overrides = design_constraints_from_legacy(overrides)

    merged = merge_design_constraints(default_constraints, typed_overrides)
    bounded = apply_target_strength_bounds(merged, target_strength_mpa=float(target_strength_mpa))
    return ensure_default_age_constraint(
        bounded,
        float(config["engineering"]["design_tool"]["default_age_days"]),
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

    for name in BASE_CONSTRAINT_FIELDS:
        _evaluate_named_constraint(
            checks=checks,
            hard_failures=hard_failures,
            name=name,
            value=None if mix_design.get(name) is None else float(mix_design[name]),
            constraint=getattr(constraints, name),
        )

    for name in RATIO_CONSTRAINT_FIELDS:
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
