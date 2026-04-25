"""Context-aware engineering validation rules for concrete strength predictions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from feature_engineering import build_engineering_features

WARNING_CATEGORY_HARD_CONSTRAINT = "Hard Constraint"
WARNING_CATEGORY_ENGINEERING_CAUTION = "Engineering Caution"
WARNING_CATEGORY_DATA_REVIEW_FLAG = "Data Review Flag"

SEVERITY_HIGH = "high"
SEVERITY_MEDIUM = "medium"
SEVERITY_LOW = "low"

CONFIDENCE_HIGH = "high"
CONFIDENCE_MODERATE = "moderate"
CONFIDENCE_LIMITED = "limited"

DEFAULT_DURABILITY_WATER_CEMENT_LIMITS = {
    "general": 0.60,
    "structural": 0.50,
    "exposed": 0.45,
    "severe": 0.45,
    "marine": 0.40,
    "freeze_thaw": 0.45,
    "sulfate": 0.45,
}

DEFAULT_DURABILITY_EFFECTIVE_BINDER_LIMITS = {
    "general": 0.60,
    "structural": 0.50,
    "exposed": 0.45,
    "severe": 0.45,
    "marine": 0.40,
    "freeze_thaw": 0.45,
    "sulfate": 0.45,
}

EXPOSURE_CLASS_ALIASES = {
    "general": "general",
    "mild": "general",
    "normal": "general",
    "structural": "structural",
    "exposed": "exposed",
    "exterior": "exposed",
    "severe": "severe",
    "marine": "marine",
    "chloride": "marine",
    "freeze_thaw": "freeze_thaw",
    "freeze-thaw": "freeze_thaw",
    "frost": "freeze_thaw",
    "sulfate": "sulfate",
}

CONFIDENCE_ORDER = {
    CONFIDENCE_LIMITED: 0,
    CONFIDENCE_MODERATE: 1,
    CONFIDENCE_HIGH: 2,
}


def _safe_float(value: Any) -> float | None:
    """Return a finite float when possible."""
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(result):
        return None
    return result


def _normalize_token(value: Any) -> str:
    """Normalize free-text metadata to a stable token."""
    return str(value).strip().lower().replace("-", "_").replace(" ", "_")


def _unique_strings(values: list[str]) -> list[str]:
    """Preserve input order while removing duplicates."""
    seen: set[str] = set()
    unique: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        unique.append(value)
    return unique


def _confidence_minimum(*levels: str) -> str:
    """Return the lowest confidence level in a collection."""
    if not levels:
        return CONFIDENCE_HIGH
    return min(levels, key=lambda level: CONFIDENCE_ORDER.get(level, -1))


def _format_factor_value(value: Any) -> str:
    """Render a factor value for evidence summaries."""
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.3f}"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return str(value)


@dataclass(frozen=True)
class WarningRecord:
    """Structured warning payload exposed to users and reports."""

    warning_code: str
    warning_category: str
    severity: str
    message: str
    triggering_factors: dict[str, Any]
    evidence_summary: str
    academic_note: str
    recommended_review_action: str
    assessment_confidence: str
    downgraded_from: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "warning_code": self.warning_code,
            "warning_category": self.warning_category,
            "severity": self.severity,
            "message": self.message,
            "triggering_factors": dict(self.triggering_factors),
            "evidence_summary": self.evidence_summary,
            "academic_note": self.academic_note,
            "recommended_review_action": self.recommended_review_action,
            "assessment_confidence": self.assessment_confidence,
        }
        if self.downgraded_from is not None:
            payload["downgraded_from"] = self.downgraded_from
        return payload


@dataclass
class EngineeringValidator:
    """Validate predictions against context-aware engineering rules."""

    min_strength_mpa: float
    max_strength_mpa: float
    suspicious_water_cement_ratio: float
    suspicious_strength_mpa: float
    durability_water_cement_warn: float
    low_water_binder_warn: float
    total_binder_low_warn: float
    total_binder_high_warn: float
    fly_ash_replacement_warn: float
    slag_replacement_warn: float
    early_age_days_warn: float
    early_age_strength_warn: float
    high_water_cement_strength_hard_fail_max_age_days: float
    high_water_cement_strength_low_scm_threshold: float
    high_water_cement_strength_unfavorable_water_binder_ratio: float
    context_type: str = "general"
    water_column: str = "water"
    cement_column: str = "cement"
    durability_water_cement_limits: dict[str, float] = field(
        default_factory=lambda: dict(DEFAULT_DURABILITY_WATER_CEMENT_LIMITS)
    )
    durability_effective_binder_limits: dict[str, float] = field(
        default_factory=lambda: dict(DEFAULT_DURABILITY_EFFECTIVE_BINDER_LIMITS)
    )
    scm_meaningful_replacement_threshold: float = 0.15
    high_volume_scm_replacement_threshold: float = 0.45
    age_regime_early_max_days: float = 7.0
    age_regime_later_min_days: float = 56.0
    workability_support_superplasticizer_ratio: float = 0.01
    superplasticizer_assumed_unit: str = "kg_per_m3"
    superplasticizer_unit_confidence: str = CONFIDENCE_MODERATE
    superplasticizer_dosage_warn_kg_per_m3: float = 18.0
    superplasticizer_binder_ratio_warn: float = 0.05
    paste_rich_aggregate_paste_ratio_threshold: float = 2.85
    shrinkage_low_water_binder_threshold: float = 0.38
    shrinkage_strength_threshold: float = 50.0
    scm_durability_preferred_water_binder_ratio: float = 0.45
    low_target_strength_upper_bound_mpa: float = 35.0
    low_target_cement_factor_warn: float = 9.0
    low_target_cement_minimum_warn: float = 320.0
    feature_engineering_config: dict[str, Any] | None = None

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "EngineeringValidator":
        """Construct a validator from the project configuration."""
        bounds = config["engineering_bounds"]
        rules = config["validator"]
        durability_limits = dict(DEFAULT_DURABILITY_WATER_CEMENT_LIMITS)
        configured_limits = rules.get("durability_water_cement_limits", {})
        if isinstance(configured_limits, dict):
            for exposure_name, limit in configured_limits.items():
                normalized_name = _normalize_token(exposure_name)
                coerced_limit = _safe_float(limit)
                if coerced_limit is not None:
                    durability_limits[normalized_name] = coerced_limit
        effective_binder_limits = dict(DEFAULT_DURABILITY_EFFECTIVE_BINDER_LIMITS)
        configured_effective_limits = rules.get("durability_effective_binder_limits", {})
        if isinstance(configured_effective_limits, dict):
            for exposure_name, limit in configured_effective_limits.items():
                normalized_name = _normalize_token(exposure_name)
                coerced_limit = _safe_float(limit)
                if coerced_limit is not None:
                    effective_binder_limits[normalized_name] = coerced_limit

        return cls(
            min_strength_mpa=float(bounds["min"]),
            max_strength_mpa=float(bounds["max"]),
            suspicious_water_cement_ratio=float(rules["suspicious_water_cement_ratio"]),
            suspicious_strength_mpa=float(rules["suspicious_strength_mpa"]),
            durability_water_cement_warn=float(rules["durability_water_cement_warn"]),
            context_type=str(rules.get("context_type", "general")).strip().lower(),
            low_water_binder_warn=float(rules["low_water_binder_warn"]),
            total_binder_low_warn=float(rules["total_binder_low_warn"]),
            total_binder_high_warn=float(rules["total_binder_high_warn"]),
            fly_ash_replacement_warn=float(rules["fly_ash_replacement_warn"]),
            slag_replacement_warn=float(rules["slag_replacement_warn"]),
            early_age_days_warn=float(rules["early_age_days_warn"]),
            early_age_strength_warn=float(rules["early_age_strength_warn"]),
            high_water_cement_strength_hard_fail_max_age_days=float(
                rules.get("high_water_cement_strength_hard_fail_max_age_days", 28.0)
            ),
            high_water_cement_strength_low_scm_threshold=float(
                rules.get("high_water_cement_strength_low_scm_threshold", 0.20)
            ),
            high_water_cement_strength_unfavorable_water_binder_ratio=float(
                rules.get("high_water_cement_strength_unfavorable_water_binder_ratio", 0.50)
            ),
            durability_water_cement_limits=durability_limits,
            durability_effective_binder_limits=effective_binder_limits,
            scm_meaningful_replacement_threshold=float(
                rules.get("scm_meaningful_replacement_threshold", 0.15)
            ),
            high_volume_scm_replacement_threshold=float(
                rules.get("high_volume_scm_replacement_threshold", 0.45)
            ),
            age_regime_early_max_days=float(rules.get("age_regime_early_max_days", 7.0)),
            age_regime_later_min_days=float(rules.get("age_regime_later_min_days", 56.0)),
            workability_support_superplasticizer_ratio=float(
                rules.get("workability_support_superplasticizer_ratio", 0.01)
            ),
            superplasticizer_assumed_unit=str(
                rules.get("superplasticizer_assumed_unit", "kg_per_m3")
            ),
            superplasticizer_unit_confidence=str(
                rules.get("superplasticizer_unit_confidence", CONFIDENCE_MODERATE)
            ),
            superplasticizer_dosage_warn_kg_per_m3=float(
                rules.get("superplasticizer_dosage_warn_kg_per_m3", 18.0)
            ),
            superplasticizer_binder_ratio_warn=float(
                rules.get("superplasticizer_binder_ratio_warn", 0.05)
            ),
            paste_rich_aggregate_paste_ratio_threshold=float(
                rules.get("paste_rich_aggregate_paste_ratio_threshold", 2.85)
            ),
            shrinkage_low_water_binder_threshold=float(
                rules.get("shrinkage_low_water_binder_threshold", 0.38)
            ),
            shrinkage_strength_threshold=float(rules.get("shrinkage_strength_threshold", 50.0)),
            scm_durability_preferred_water_binder_ratio=float(
                rules.get("scm_durability_preferred_water_binder_ratio", 0.45)
            ),
            low_target_strength_upper_bound_mpa=float(
                rules.get("low_target_strength_upper_bound_mpa", 35.0)
            ),
            low_target_cement_factor_warn=float(rules.get("low_target_cement_factor_warn", 9.0)),
            low_target_cement_minimum_warn=float(
                rules.get("low_target_cement_minimum_warn", 320.0)
            ),
            feature_engineering_config=config,
        )

    def __post_init__(self) -> None:
        """Validate configuration after dataclass initialization."""
        if self.context_type not in self.durability_water_cement_limits:
            valid_contexts = ", ".join(sorted(self.durability_water_cement_limits))
            raise ValueError(f"validator.context_type must be one of: {valid_contexts}")

    def _prepare_frame(self, x_frame: pd.DataFrame) -> pd.DataFrame:
        """Ensure engineered columns required by the validator are present."""
        if self.water_column not in x_frame.columns or self.cement_column not in x_frame.columns:
            raise ValueError(
                f"Required columns '{self.water_column}' and '{self.cement_column}' are missing."
            )
        return build_engineering_features(x_frame, config=self.feature_engineering_config)

    def _build_evidence_summary(self, factors: dict[str, Any]) -> str:
        """Convert triggering factors into a compact evidence summary."""
        rendered = [
            f"{name}={_format_factor_value(value)}"
            for name, value in factors.items()
            if value is not None and value != ""
        ]
        return ", ".join(rendered)

    def _make_warning_record(
        self,
        *,
        warning_code: str,
        warning_category: str,
        severity: str,
        message: str,
        triggering_factors: dict[str, Any],
        academic_note: str,
        recommended_review_action: str,
        assessment_confidence: str,
        downgraded_from: str | None = None,
    ) -> dict[str, Any]:
        """Build one structured warning record."""
        return WarningRecord(
            warning_code=warning_code,
            warning_category=warning_category,
            severity=severity,
            message=message,
            triggering_factors=triggering_factors,
            evidence_summary=self._build_evidence_summary(triggering_factors),
            academic_note=academic_note,
            recommended_review_action=recommended_review_action,
            assessment_confidence=assessment_confidence,
            downgraded_from=downgraded_from,
        ).to_dict()

    def _age_regime(self, age_days: float | None) -> tuple[str, str]:
        """Return the age regime code and label."""
        if age_days is None:
            return "unknown_age", "unknown age regime"
        if age_days < self.age_regime_early_max_days:
            return "early_age", "early-age regime"
        if age_days > self.age_regime_later_min_days:
            return "later_age", "later-age / extended curing regime"
        return "standard_28_day", "standard 28-day curing regime"

    def _resolve_exposure_context(self, sample: pd.Series) -> dict[str, Any]:
        """Resolve exposure metadata into a durability screening context."""
        for column_name in ("exposure_class", "exposure_severity", "durability_context", "exposure"):
            if column_name not in sample.index:
                continue
            normalized = EXPOSURE_CLASS_ALIASES.get(_normalize_token(sample[column_name]))
            if normalized is None:
                continue
            return {
                "label": normalized,
                "source": column_name,
                "water_cement_limit": float(self.durability_water_cement_limits[normalized]),
                "effective_binder_limit": float(self.durability_effective_binder_limits[normalized]),
                "confidence": CONFIDENCE_HIGH,
                "metadata_available": True,
            }

        return {
            "label": self.context_type,
            "source": "config_default",
            "water_cement_limit": float(
                self.durability_water_cement_limits.get(self.context_type, self.durability_water_cement_warn)
            ),
            "effective_binder_limit": float(
                self.durability_effective_binder_limits.get(
                    self.context_type,
                    self.durability_water_cement_warn,
                )
            ),
            "confidence": CONFIDENCE_MODERATE,
            "metadata_available": False,
        }

    def _resolve_workability_support(self, sample: pd.Series) -> dict[str, Any]:
        """Resolve explicit or inferred workability-support metadata."""
        for column_name in (
            "workability_support",
            "admixture_workability_support",
            "workability_support_metadata",
        ):
            if column_name not in sample.index:
                continue
            value = sample[column_name]
            if isinstance(value, (bool, np.bool_)):
                return {
                    "status": "supported" if bool(value) else "unsupported",
                    "source": column_name,
                    "confidence": CONFIDENCE_HIGH,
                }
            normalized = _normalize_token(value)
            if normalized in {"supported", "support", "true", "yes", "available"}:
                return {
                    "status": "supported",
                    "source": column_name,
                    "confidence": CONFIDENCE_HIGH,
                }
            if normalized in {"unsupported", "false", "no", "absent"}:
                return {
                    "status": "unsupported",
                    "source": column_name,
                    "confidence": CONFIDENCE_HIGH,
                }

        superplasticizer_binder_ratio = _safe_float(sample.get("superplasticizer_binder_ratio"))
        superplasticizer = _safe_float(sample.get("superplasticizer"))
        if superplasticizer_binder_ratio is not None and (
            superplasticizer_binder_ratio >= self.workability_support_superplasticizer_ratio
        ):
            return {
                "status": "supported",
                "source": "superplasticizer_binder_ratio",
                "confidence": CONFIDENCE_MODERATE,
            }
        if superplasticizer is not None and superplasticizer > 0.0:
            return {
                "status": "limited",
                "source": "superplasticizer",
                "confidence": CONFIDENCE_MODERATE,
            }
        if superplasticizer is not None:
            return {
                "status": "unsupported",
                "source": "superplasticizer",
                "confidence": CONFIDENCE_MODERATE,
            }
        return {
            "status": "unknown",
            "source": "missing",
            "confidence": CONFIDENCE_LIMITED,
        }

    def _scm_regime(
        self,
        supplementary_ratio: float | None,
        fly_ash_ratio: float | None,
        slag_ratio: float | None,
    ) -> tuple[str, str]:
        """Classify the SCM replacement regime."""
        supplementary_ratio = supplementary_ratio or 0.0
        fly_ash_ratio = fly_ash_ratio or 0.0
        slag_ratio = slag_ratio or 0.0
        high_volume_slag_threshold = min(
            self.slag_replacement_warn,
            max(0.55, self.high_volume_scm_replacement_threshold + 0.10),
        )
        if supplementary_ratio < self.scm_meaningful_replacement_threshold:
            return "plain_cement", "plain-cement regime"
        if (
            supplementary_ratio >= self.high_volume_scm_replacement_threshold
            or fly_ash_ratio >= self.fly_ash_replacement_warn
            or slag_ratio >= high_volume_slag_threshold
        ):
            return "high_volume_scm", "high-volume SCM regime"
        return "scm_bearing", "SCM-bearing regime"

    def _build_context(self, sample: pd.Series, prediction: float) -> dict[str, Any]:
        """Build the context bundle used by the rule engine."""
        water_cement_ratio = _safe_float(sample.get("water_cement_ratio"))
        water_binder_ratio = _safe_float(sample.get("water_binder_ratio"))
        water_effective_binder_ratio = _safe_float(sample.get("water_effective_binder_ratio"))
        total_binder = _safe_float(sample.get("total_binder"))
        effective_binder = _safe_float(sample.get("effective_binder"))
        supplementary_replacement_ratio = _safe_float(sample.get("supplementary_replacement_ratio"))
        fly_ash_replacement_ratio = _safe_float(sample.get("fly_ash_replacement_ratio"))
        slag_replacement_ratio = _safe_float(sample.get("slag_replacement_ratio"))
        aggregate_paste_ratio = _safe_float(sample.get("aggregate_paste_ratio"))
        superplasticizer_binder_ratio = _safe_float(sample.get("superplasticizer_binder_ratio"))
        cement_content = _safe_float(sample.get(self.cement_column))
        water_content = _safe_float(sample.get(self.water_column))
        slag_content = _safe_float(sample.get("slag"))
        fly_ash_content = _safe_float(sample.get("fly_ash"))
        age_days = _safe_float(sample.get("age"))
        superplasticizer = _safe_float(sample.get("superplasticizer"))
        target_strength = _safe_float(sample.get("target_strength"))
        structural_application = sample.get("structural_application")
        if structural_application in ("", None):
            structural_application = None
        elif isinstance(structural_application, str):
            structural_application = structural_application.strip().lower().replace(" ", "_")

        age_regime_code, age_regime_label = self._age_regime(age_days)
        scm_regime_code, scm_regime_label = self._scm_regime(
            supplementary_replacement_ratio,
            fly_ash_replacement_ratio,
            slag_replacement_ratio,
        )
        exposure_context = self._resolve_exposure_context(sample)
        workability_support = self._resolve_workability_support(sample)

        return {
            "prediction_mpa": float(prediction),
            "cement_content": cement_content,
            "water_content": water_content,
            "slag_content": slag_content,
            "fly_ash_content": fly_ash_content,
            "scm_content": None
            if slag_content is None and fly_ash_content is None
            else float((slag_content or 0.0) + (fly_ash_content or 0.0)),
            "water_cement_ratio": water_cement_ratio,
            "water_binder_ratio": water_binder_ratio,
            "water_effective_binder_ratio": water_effective_binder_ratio,
            "total_binder": total_binder,
            "effective_binder": effective_binder,
            "supplementary_replacement_ratio": supplementary_replacement_ratio,
            "fly_ash_replacement_ratio": fly_ash_replacement_ratio,
            "slag_replacement_ratio": slag_replacement_ratio,
            "aggregate_paste_ratio": aggregate_paste_ratio,
            "superplasticizer": superplasticizer,
            "superplasticizer_binder_ratio": superplasticizer_binder_ratio,
            "age_days": age_days,
            "target_strength": target_strength,
            "structural_application": structural_application,
            "age_regime_code": age_regime_code,
            "age_regime_label": age_regime_label,
            "scm_regime_code": scm_regime_code,
            "scm_regime_label": scm_regime_label,
            "exposure_context": exposure_context,
            "workability_support": workability_support,
            "assessment_confidence": CONFIDENCE_HIGH,
        }

    def _component_hard_constraints(self, context: dict[str, Any]) -> list[tuple[str, str]]:
        """Return component-level hard-constraint violations."""
        violations: list[tuple[str, str]] = []
        numeric_components = {
            "cement_content": context["cement_content"],
            "water_content": context["water_content"],
            "slag_content": context["slag_content"],
            "fly_ash_content": context["fly_ash_content"],
            "age_days": context["age_days"],
            "total_binder": context["total_binder"],
        }
        for name, value in numeric_components.items():
            if value is None:
                continue
            if value < 0.0:
                violations.append(
                    (
                        f"invalid_{name}_state",
                        f"{name.replace('_', ' ').capitalize()} is negative and the mix definition is invalid.",
                    )
                )
        if context["total_binder"] is not None and context["total_binder"] <= 0.0:
            violations.append(
                (
                    "invalid_total_binder_state",
                    "Total binder content is non-positive and the mix cannot be evaluated reliably.",
                )
            )
        if context["age_days"] is not None and context["age_days"] <= 0.0:
            violations.append(
                (
                    "invalid_age_state",
                    "Age must be positive for a physically interpretable curing regime.",
                )
            )
        return violations

    _TOTAL_MASS_MIN_KG_M3: float = 1800.0
    _TOTAL_MASS_MAX_KG_M3: float = 2700.0

    def _volumetric_hard_constraints(self, sample: "pd.Series") -> list[tuple[str, str]]:
        """Return a hard-constraint violation if total mix mass is outside physical bounds."""
        coarse = _safe_float(sample.get("coarse_aggregate"))
        fine = _safe_float(sample.get("fine_aggregate"))
        cement = _safe_float(sample.get(self.cement_column))
        slag = _safe_float(sample.get("slag"))
        fly_ash = _safe_float(sample.get("fly_ash"))
        water = _safe_float(sample.get(self.water_column))
        sp = _safe_float(sample.get("superplasticizer"))
        components = [coarse, fine, cement, slag, fly_ash, water, sp]
        if any(v is None for v in components):
            return []
        total_mass = sum(float(v) for v in components)  # type: ignore[arg-type]
        if total_mass < self._TOTAL_MASS_MIN_KG_M3:
            return [
                (
                    "invalid_mix_total_mass_too_low",
                    f"Total mix mass {total_mass:.0f} kg/m\u00b3 is below the physical minimum "
                    f"({self._TOTAL_MASS_MIN_KG_M3:.0f} kg/m\u00b3). Mix proportions are unrealistic.",
                )
            ]
        if total_mass > self._TOTAL_MASS_MAX_KG_M3:
            return [
                (
                    "invalid_mix_total_mass_too_high",
                    f"Total mix mass {total_mass:.0f} kg/m\u00b3 exceeds the physical maximum "
                    f"({self._TOTAL_MASS_MAX_KG_M3:.0f} kg/m\u00b3). Mix proportions are unrealistic.",
                )
            ]
        return []

    def _build_contextual_summary(
        self,
        *,
        context: dict[str, Any],
        hard_constraints: list[dict[str, Any]],
        engineering_cautions: list[dict[str, Any]],
        data_review_flags: list[dict[str, Any]],
        downgraded_warnings: list[dict[str, Any]],
    ) -> str:
        """Build a short human-readable summary for one evaluated mix."""
        summary_parts = [
            context["scm_regime_label"],
            context["age_regime_label"],
            f"durability context={context['exposure_context']['label']}",
        ]
        if context.get("structural_application"):
            summary_parts.append(f"application={context['structural_application']}")
        if hard_constraints:
            summary_parts.append(f"{len(hard_constraints)} hard constraint(s)")
        if engineering_cautions:
            summary_parts.append(f"{len(engineering_cautions)} engineering caution(s)")
        if data_review_flags:
            summary_parts.append(f"{len(data_review_flags)} data review flag(s)")
        if downgraded_warnings:
            summary_parts.append("water/cement-only screening was downgraded where SCM-age context was available")
        if len(summary_parts) == 3:
            summary_parts.append("no contextual warnings fired")
        return "; ".join(summary_parts)

    def _evaluate_single_sample(
        self,
        row_index: Any,
        sample: pd.Series,
        prediction: float,
        actual_value: float | None = None,
    ) -> dict[str, Any]:
        """Evaluate a single prediction against context-aware engineering rules."""
        context = self._build_context(sample, prediction)
        hard_constraints_by_code: dict[str, dict[str, Any]] = {}
        engineering_cautions_by_code: dict[str, dict[str, Any]] = {}
        data_review_flags_by_code: dict[str, dict[str, Any]] = {}
        downgraded_warnings: list[dict[str, Any]] = []
        rule_evaluation_trace: list[dict[str, Any]] = []

        def add_warning(
            store: dict[str, dict[str, Any]],
            *,
            warning_code: str,
            warning_category: str,
            severity: str,
            message: str,
            triggering_factors: dict[str, Any],
            academic_note: str,
            recommended_review_action: str,
            assessment_confidence: str,
            downgraded_from: str | None = None,
        ) -> None:
            if warning_code in store:
                return
            warning_record = self._make_warning_record(
                warning_code=warning_code,
                warning_category=warning_category,
                severity=severity,
                message=message,
                triggering_factors=triggering_factors,
                academic_note=academic_note,
                recommended_review_action=recommended_review_action,
                assessment_confidence=assessment_confidence,
                downgraded_from=downgraded_from,
            )
            store[warning_code] = warning_record
            rule_evaluation_trace.append(
                {
                    "warning_code": warning_code,
                    "outcome": "fired",
                    "warning_category": warning_category,
                    "severity": severity,
                    "considered_factors": dict(triggering_factors),
                }
            )

        def add_downgrade(
            *,
            warning_code: str,
            downgraded_from: str,
            downgrade_reason: str,
            contextual_basis: dict[str, Any],
        ) -> None:
            if warning_code in {item["warning_code"] for item in downgraded_warnings}:
                return
            downgraded_warnings.append(
                {
                    "warning_code": warning_code,
                    "downgraded_from": downgraded_from,
                    "downgraded_to": "suppressed",
                    "downgrade_reason": downgrade_reason,
                    "contextual_basis": dict(contextual_basis),
                }
            )
            rule_evaluation_trace.append(
                {
                    "warning_code": warning_code,
                    "outcome": "downgraded",
                    "from_category": downgraded_from,
                    "to_category": "suppressed",
                    "reason": downgrade_reason,
                    "considered_factors": dict(contextual_basis),
                }
            )

        prediction_value = context["prediction_mpa"]
        if not np.isfinite(prediction_value):
            add_warning(
                hard_constraints_by_code,
                warning_code="prediction_non_finite",
                warning_category=WARNING_CATEGORY_HARD_CONSTRAINT,
                severity=SEVERITY_HIGH,
                message="Prediction is non-finite and cannot be interpreted.",
                triggering_factors={"prediction_mpa": prediction_value},
                academic_note=(
                    "A non-finite prediction does not represent a physically interpretable compressive "
                    "strength and should be treated as an invalid model output."
                ),
                recommended_review_action="Inspect the model output path before using this prediction.",
                assessment_confidence=CONFIDENCE_HIGH,
            )
        else:
            if prediction_value < 0.0:
                add_warning(
                    hard_constraints_by_code,
                    warning_code="prediction_negative_strength",
                    warning_category=WARNING_CATEGORY_HARD_CONSTRAINT,
                    severity=SEVERITY_HIGH,
                    message="Predicted strength is negative.",
                    triggering_factors={"prediction_mpa": prediction_value},
                    academic_note=(
                        "Negative compressive strength is physically impossible and indicates a model or "
                        "data integrity failure."
                    ),
                    recommended_review_action="Reject the prediction and inspect the model or feature inputs.",
                    assessment_confidence=CONFIDENCE_HIGH,
                )
            if prediction_value < self.min_strength_mpa:
                add_warning(
                    hard_constraints_by_code,
                    warning_code="prediction_below_min_bound",
                    warning_category=WARNING_CATEGORY_HARD_CONSTRAINT,
                    severity=SEVERITY_HIGH,
                    message="Predicted strength falls below the configured lower engineering bound.",
                    triggering_factors={
                        "prediction_mpa": prediction_value,
                        "configured_min_strength_mpa": self.min_strength_mpa,
                    },
                    academic_note=(
                        "Configured engineering bounds define the accepted operating range for governed model "
                        "outputs and should be enforced before downstream interpretation."
                    ),
                    recommended_review_action="Reject the prediction or revise the governing configuration.",
                    assessment_confidence=CONFIDENCE_HIGH,
                )
            if prediction_value > self.max_strength_mpa:
                add_warning(
                    hard_constraints_by_code,
                    warning_code="prediction_above_max_bound",
                    warning_category=WARNING_CATEGORY_HARD_CONSTRAINT,
                    severity=SEVERITY_HIGH,
                    message="Predicted strength exceeds the configured upper engineering bound.",
                    triggering_factors={
                        "prediction_mpa": prediction_value,
                        "configured_max_strength_mpa": self.max_strength_mpa,
                    },
                    academic_note=(
                        "Configured engineering bounds define the accepted operating range for governed model "
                        "outputs and prevent unreviewed extrapolation."
                    ),
                    recommended_review_action="Reject the prediction or revise the governing configuration.",
                    assessment_confidence=CONFIDENCE_HIGH,
                )

        for warning_code, message in self._component_hard_constraints(context):
            add_warning(
                hard_constraints_by_code,
                warning_code=warning_code,
                warning_category=WARNING_CATEGORY_HARD_CONSTRAINT,
                severity=SEVERITY_HIGH,
                message=message,
                triggering_factors={
                    "cement_content": context["cement_content"],
                    "water_content": context["water_content"],
                    "slag_content": context["slag_content"],
                    "fly_ash_content": context["fly_ash_content"],
                    "age_days": context["age_days"],
                    "total_binder": context["total_binder"],
                },
                academic_note=(
                    "Negative constituent values, non-positive binder content, or non-positive age indicate an "
                    "invalid mixture state rather than a debatable engineering warning."
                ),
                recommended_review_action="Verify the raw mix metadata before accepting the record.",
                assessment_confidence=CONFIDENCE_HIGH,
            )

        for warning_code, message in self._volumetric_hard_constraints(sample):
            add_warning(
                hard_constraints_by_code,
                warning_code=warning_code,
                warning_category=WARNING_CATEGORY_HARD_CONSTRAINT,
                severity=SEVERITY_HIGH,
                message=message,
                triggering_factors={
                    "coarse_aggregate": _safe_float(sample.get("coarse_aggregate")),
                    "fine_aggregate": _safe_float(sample.get("fine_aggregate")),
                    "cement_content": context["cement_content"],
                    "water_content": context["water_content"],
                    "total_binder": context["total_binder"],
                    "superplasticizer": context["superplasticizer"],
                    "total_mass_min_kg_m3": self._TOTAL_MASS_MIN_KG_M3,
                    "total_mass_max_kg_m3": self._TOTAL_MASS_MAX_KG_M3,
                },
                academic_note=(
                    "Normal-weight concrete has a fresh density of 2300–2500 kg/m\u00b3. "
                    "Mixes outside [1800, 2700] kg/m\u00b3 indicate a proportioning error, "
                    "not a legitimate lightweight or heavyweight concrete design."
                ),
                recommended_review_action=(
                    "Verify aggregate, binder, and water quantities are in kg/m\u00b3 "
                    "and that no constituent was duplicated or omitted."
                ),
                assessment_confidence=CONFIDENCE_HIGH,
            )

        if hard_constraints_by_code:
            hard_constraints = list(hard_constraints_by_code.values())
            engineering_cautions: list[dict[str, Any]] = []
            data_review_flags: list[dict[str, Any]] = []
            contextual_summary = self._build_contextual_summary(
                context=context,
                hard_constraints=hard_constraints,
                engineering_cautions=engineering_cautions,
                data_review_flags=data_review_flags,
                downgraded_warnings=downgraded_warnings,
            )
            return self._build_sample_report(
                row_index=row_index,
                prediction=prediction_value,
                actual_value=actual_value,
                context=context,
                hard_constraints=hard_constraints,
                engineering_cautions=engineering_cautions,
                data_review_flags=data_review_flags,
                downgraded_warnings=downgraded_warnings,
                rule_evaluation_trace=rule_evaluation_trace,
                contextual_summary=contextual_summary,
            )

        water_cement_ratio = context["water_cement_ratio"]
        water_binder_ratio = context["water_binder_ratio"]
        water_effective_binder_ratio = context["water_effective_binder_ratio"]
        total_binder = context["total_binder"]
        supplementary_replacement_ratio = context["supplementary_replacement_ratio"]
        age_days = context["age_days"]
        exposure_context = context["exposure_context"]
        workability_support = context["workability_support"]
        target_strength = context["target_strength"]

        high_strength = prediction_value > self.suspicious_strength_mpa
        high_water_cement = (water_cement_ratio or 0.0) > self.suspicious_water_cement_ratio
        high_water_binder = (
            (water_binder_ratio or 0.0) >= self.high_water_cement_strength_unfavorable_water_binder_ratio
        )
        low_binder = total_binder is not None and total_binder < self.total_binder_low_warn
        meaningful_scm = (
            (supplementary_replacement_ratio or 0.0) >= self.scm_meaningful_replacement_threshold
        )
        high_volume_scm = context["scm_regime_code"] == "high_volume_scm"
        early_age = context["age_regime_code"] == "early_age"
        later_age = context["age_regime_code"] == "later_age"

        if high_volume_scm:
            review_factors = {
                "age_regime": context["age_regime_label"],
                "water_binder_ratio": water_binder_ratio,
                "total_binder": total_binder,
                "supplementary_replacement_ratio": supplementary_replacement_ratio,
                "fly_ash_replacement_ratio": context["fly_ash_replacement_ratio"],
                "slag_replacement_ratio": context["slag_replacement_ratio"],
                "prediction_mpa": prediction_value,
            }
            add_warning(
                data_review_flags_by_code,
                warning_code="high_volume_scm_regime_review",
                warning_category=WARNING_CATEGORY_DATA_REVIEW_FLAG,
                severity=SEVERITY_MEDIUM,
                message="High-volume SCM regime requires age-aware review.",
                triggering_factors=review_factors,
                academic_note=(
                    "High replacement SCM systems can follow different strength-development trajectories than "
                    "plain-cement mixtures, so they should be reviewed with water/binder ratio, total binder, "
                    "and curing age in view before judging plausibility."
                ),
                recommended_review_action=(
                    "Review curing age, binder replacement strategy, and strength-test metadata before treating "
                    "the mix as anomalous."
                ),
                assessment_confidence=CONFIDENCE_HIGH,
            )

        if high_strength and high_water_cement:
            plausibility_factors = {
                "prediction_mpa": prediction_value,
                "water_cement_ratio": water_cement_ratio,
                "water_binder_ratio": water_binder_ratio,
                "total_binder": total_binder,
                "supplementary_replacement_ratio": supplementary_replacement_ratio,
                "age_days": age_days,
                "age_regime": context["age_regime_label"],
                "scm_regime": context["scm_regime_label"],
            }
            contradictory_indicators: list[str] = []
            if high_water_binder:
                contradictory_indicators.append("water_binder_ratio")
            if low_binder:
                contradictory_indicators.append("total_binder")
            if early_age:
                contradictory_indicators.append("age_days")
            if not meaningful_scm:
                contradictory_indicators.append("supplementary_replacement_ratio")

            if meaningful_scm and later_age and not high_water_binder and not low_binder:
                add_downgrade(
                    warning_code="high_water_cement_high_strength_context_review",
                    downgraded_from=WARNING_CATEGORY_DATA_REVIEW_FLAG,
                    downgrade_reason=(
                        "SCM replacement, water/binder ratio, and later-age curing provide a more relevant "
                        "context than water/cement ratio alone for this mix."
                    ),
                    contextual_basis=plausibility_factors,
                )
            elif not meaningful_scm and early_age and high_water_binder:
                add_warning(
                    hard_constraints_by_code,
                    warning_code="high_water_cement_high_strength_context_review",
                    warning_category=WARNING_CATEGORY_HARD_CONSTRAINT,
                    severity=SEVERITY_HIGH,
                    message="Early-age high strength is incompatible with the binder-water context.",
                    triggering_factors=plausibility_factors,
                    academic_note=(
                        "High early-age strength at very high water/cement ratio remains implausible when SCM "
                        "replacement is limited and the binder-based water ratio is still unfavorable."
                    ),
                    recommended_review_action="Reject unless the mix metadata or test age is corrected.",
                    assessment_confidence=CONFIDENCE_HIGH,
                )
            elif meaningful_scm and len(contradictory_indicators) >= 2:
                add_warning(
                    data_review_flags_by_code,
                    warning_code="high_water_cement_high_strength_context_review",
                    warning_category=WARNING_CATEGORY_DATA_REVIEW_FLAG,
                    severity=SEVERITY_MEDIUM,
                    message="Strength remains unusual after SCM-age screening.",
                    triggering_factors=plausibility_factors,
                    academic_note=(
                        "SCM-bearing mixes should be screened with water/binder ratio, total binder, and curing "
                        "age before they are called anomalous; once several of those indicators still disagree, "
                        "manual review is warranted."
                    ),
                    recommended_review_action="Review curing age, binder chemistry, and test metadata together.",
                    assessment_confidence=CONFIDENCE_HIGH,
                )
            else:
                add_warning(
                    data_review_flags_by_code,
                    warning_code="high_water_cement_high_strength_context_review",
                    warning_category=WARNING_CATEGORY_DATA_REVIEW_FLAG,
                    severity=SEVERITY_MEDIUM,
                    message="Strength is high relative to the water-ratio context.",
                    triggering_factors=plausibility_factors,
                    academic_note=(
                        "High strength alongside a very high water/cement ratio is not automatically impossible, "
                        "but it becomes more credible only when binder ratio, age, and SCM regime support it."
                    ),
                    recommended_review_action="Review age, binder ratio, and supplementary binder context together.",
                    assessment_confidence=CONFIDENCE_HIGH,
                )

        if age_days is not None and age_days <= self.early_age_days_warn and prediction_value > self.early_age_strength_warn:
            early_strength_factors = {
                "prediction_mpa": prediction_value,
                "age_days": age_days,
                "water_binder_ratio": water_binder_ratio,
                "total_binder": total_binder,
                "scm_regime": context["scm_regime_label"],
            }
            if meaningful_scm and not high_water_binder and not low_binder:
                add_warning(
                    engineering_cautions_by_code,
                    warning_code="early_age_strength_review",
                    warning_category=WARNING_CATEGORY_ENGINEERING_CAUTION,
                    severity=SEVERITY_LOW,
                    message="Very early strength should be checked against curing details.",
                    triggering_factors=early_strength_factors,
                    academic_note=(
                        "Very early-age strengths can be legitimate in well-controlled systems, but they remain "
                        "sensitive to curing regime, temperature history, and binder chemistry."
                    ),
                    recommended_review_action="Confirm the curing and test-age metadata before final acceptance.",
                    assessment_confidence=CONFIDENCE_HIGH,
                )
            else:
                add_warning(
                    data_review_flags_by_code,
                    warning_code="early_age_strength_review",
                    warning_category=WARNING_CATEGORY_DATA_REVIEW_FLAG,
                    severity=SEVERITY_MEDIUM,
                    message="Very early strength looks unusually high for the recorded age.",
                    triggering_factors=early_strength_factors,
                    academic_note=(
                        "Very high strength at a very early age is sensitive to curing regime and binder "
                        "chemistry, so the observation should be reviewed before it is treated as ordinary."
                    ),
                    recommended_review_action="Verify the test age, curing regime, and specimen history.",
                    assessment_confidence=CONFIDENCE_HIGH,
                )

        durability_factors = {
            "water_cement_ratio": water_cement_ratio,
            "water_binder_ratio": water_binder_ratio,
            "water_effective_binder_ratio": water_effective_binder_ratio,
            "durability_limit": exposure_context["water_cement_limit"],
            "effective_binder_limit": exposure_context["effective_binder_limit"],
            "exposure_class": exposure_context["label"],
            "scm_regime": context["scm_regime_label"],
            "age_regime": context["age_regime_label"],
        }
        durability_ratio_value = water_cement_ratio
        durability_limit = exposure_context["water_cement_limit"]
        durability_ratio_type = "water_cement_ratio"
        if meaningful_scm and water_effective_binder_ratio is not None:
            durability_ratio_value = water_effective_binder_ratio
            durability_limit = exposure_context["effective_binder_limit"]
            durability_ratio_type = "water_effective_binder_ratio"
        durability_factors["screening_ratio_type"] = durability_ratio_type
        durability_factors["screening_ratio_value"] = durability_ratio_value
        durability_factors["screening_ratio_limit"] = durability_limit
        if durability_ratio_value is not None and durability_ratio_value > durability_limit:
            severity = SEVERITY_MEDIUM
            message = "Water ratio exceeds exposure-based durability guidance."
            confidence = exposure_context["confidence"]
            if not exposure_context["metadata_available"]:
                message = "Water ratio exceeds fallback general guidance for durability."
            if durability_ratio_type == "water_effective_binder_ratio":
                message = "Effective water/binder ratio exceeds durability guidance for the SCM-bearing mix."
            emit_durability_caution = True
            if (
                meaningful_scm
                and (water_effective_binder_ratio or 0.0) <= self.scm_durability_preferred_water_binder_ratio
                and not early_age
            ):
                severity = SEVERITY_LOW
                durability_factors["binder_context_softens_screen"] = True
                add_downgrade(
                    warning_code=(
                        "durability_exposure_water_ratio_caution"
                        if exposure_context["metadata_available"]
                        else "durability_general_water_ratio_caution"
                    ),
                    downgraded_from=WARNING_CATEGORY_ENGINEERING_CAUTION,
                    downgrade_reason=(
                        "SCM-bearing mixes are screened primarily with water/binder ratio and curing context, "
                        "so the cement-only water ratio remains secondary guidance here."
                    ),
                    contextual_basis=durability_factors,
                )
                if not exposure_context["metadata_available"]:
                    emit_durability_caution = False

            if emit_durability_caution:
                add_warning(
                    engineering_cautions_by_code,
                    warning_code=(
                        "durability_exposure_water_ratio_caution"
                        if exposure_context["metadata_available"]
                        else "durability_general_water_ratio_caution"
                    ),
                    warning_category=WARNING_CATEGORY_ENGINEERING_CAUTION,
                    severity=severity,
                    message=message,
                    triggering_factors=durability_factors,
                    academic_note=(
                        "Durability limits are exposure-dependent. When exposure metadata is known, the mix should be "
                        "screened against that context; when it is absent, only general guidance can be applied."
                    ),
                    recommended_review_action=(
                        "Confirm the exposure class and review curing, permeability, and binder selection before "
                        "final durability sign-off."
                        if exposure_context["metadata_available"]
                        else "Add exposure metadata before treating this screen as a definitive durability decision."
                    ),
                    assessment_confidence=confidence,
                )

        if (
            target_strength is not None
            and target_strength <= self.low_target_strength_upper_bound_mpa
            and context["cement_content"] is not None
        ):
            cement_warn_threshold = max(
                self.low_target_cement_minimum_warn,
                target_strength * self.low_target_cement_factor_warn,
            )
            if context["cement_content"] > cement_warn_threshold and prediction_value <= target_strength + 5.0:
                add_warning(
                    engineering_cautions_by_code,
                    warning_code="low_target_overcemented_mix_warning",
                    warning_category=WARNING_CATEGORY_ENGINEERING_CAUTION,
                    severity=SEVERITY_MEDIUM,
                    message="Low-strength target is being met with unusually high cement demand.",
                    triggering_factors={
                        "target_strength": target_strength,
                        "prediction_mpa": prediction_value,
                        "cement_content": context["cement_content"],
                        "cement_warn_threshold": cement_warn_threshold,
                        "water_effective_binder_ratio": water_effective_binder_ratio,
                    },
                    academic_note=(
                        "Low-strength mixes should not routinely require high cement contents when binder "
                        "replacement, water ratio, and aggregate balance are reasonable. This is a "
                        "configurable economy-and-plausibility screen, not a universal law."
                    ),
                    recommended_review_action=(
                        "Review whether a lower-cement or SCM-bearing mix can meet the same target with "
                        "better engineering economy."
                    ),
                    assessment_confidence=CONFIDENCE_MODERATE,
                )

        if total_binder is not None and total_binder < self.total_binder_low_warn:
            low_binder_factors = {
                "total_binder": total_binder,
                "water_binder_ratio": water_binder_ratio,
                "water_cement_ratio": water_cement_ratio,
                "exposure_class": exposure_context["label"],
                "prediction_mpa": prediction_value,
            }
            severity = SEVERITY_LOW
            if exposure_context["metadata_available"] or (water_binder_ratio or 0.0) > 0.50:
                severity = SEVERITY_MEDIUM
            add_warning(
                engineering_cautions_by_code,
                warning_code="low_binder_durability_caution",
                warning_category=WARNING_CATEGORY_ENGINEERING_CAUTION,
                severity=severity,
                message="Binder content may be lean for the durability context.",
                triggering_factors=low_binder_factors,
                academic_note=(
                    "Low total binder content can make durability more sensitive to curing quality and exposure "
                    "severity, particularly when the water ratio is not correspondingly low."
                ),
                recommended_review_action="Review exposure severity, curing assumptions, and paste sufficiency.",
                assessment_confidence=(
                    exposure_context["confidence"]
                    if exposure_context["metadata_available"]
                    else CONFIDENCE_MODERATE
                ),
            )

        paste_rich = (context["aggregate_paste_ratio"] or np.inf) <= self.paste_rich_aggregate_paste_ratio_threshold
        dense_matrix = (water_binder_ratio or np.inf) <= self.shrinkage_low_water_binder_threshold
        high_strength_demand = prediction_value >= self.shrinkage_strength_threshold
        shrinkage_indicators = sum([paste_rich, dense_matrix, high_strength_demand])
        if total_binder is not None and total_binder > self.total_binder_high_warn and shrinkage_indicators >= 2:
            add_warning(
                engineering_cautions_by_code,
                warning_code="high_binder_shrinkage_caution",
                warning_category=WARNING_CATEGORY_ENGINEERING_CAUTION,
                severity=SEVERITY_MEDIUM,
                message="Paste-rich high-binder mix may have elevated shrinkage risk.",
                triggering_factors={
                    "total_binder": total_binder,
                    "aggregate_paste_ratio": context["aggregate_paste_ratio"],
                    "water_binder_ratio": water_binder_ratio,
                    "prediction_mpa": prediction_value,
                },
                academic_note=(
                    "High binder content becomes more shrinkage-sensitive when it is paired with a paste-rich "
                    "aggregate balance, low water/binder ratio, or high strength demand."
                ),
                recommended_review_action="Review shrinkage mitigation, curing duration, and paste volume control.",
                assessment_confidence=CONFIDENCE_HIGH,
            )

        if water_binder_ratio is not None and water_binder_ratio < self.low_water_binder_warn:
            severity = SEVERITY_MEDIUM
            if workability_support["status"] == "supported":
                severity = SEVERITY_LOW
            elif workability_support["status"] == "unsupported":
                severity = SEVERITY_HIGH
            add_warning(
                engineering_cautions_by_code,
                warning_code="low_water_binder_workability_caution",
                warning_category=WARNING_CATEGORY_ENGINEERING_CAUTION,
                severity=severity,
                message=(
                    "Very low water/binder ratio relies on declared workability support."
                    if workability_support["status"] == "supported"
                    else "Very low water/binder ratio may compromise workability."
                ),
                triggering_factors={
                    "water_binder_ratio": water_binder_ratio,
                    "workability_support": workability_support["status"],
                    "support_source": workability_support["source"],
                    "superplasticizer": context["superplasticizer"],
                },
                academic_note=(
                    "Very low water/binder mixtures often need verified admixture support or placement controls "
                    "to remain workable and consolidatable."
                ),
                recommended_review_action="Check slump/workability evidence and admixture support before acceptance.",
                assessment_confidence=workability_support["confidence"],
            )

        superplasticizer = context["superplasticizer"]
        superplasticizer_binder_ratio = context.get("superplasticizer_binder_ratio")
        if superplasticizer is not None and (
            superplasticizer >= self.superplasticizer_dosage_warn_kg_per_m3
            or (superplasticizer_binder_ratio or 0.0) >= self.superplasticizer_binder_ratio_warn
        ):
            add_warning(
                data_review_flags_by_code,
                warning_code="superplasticizer_dosage_review",
                warning_category=WARNING_CATEGORY_DATA_REVIEW_FLAG,
                severity=SEVERITY_MEDIUM,
                message=(
                    f"Superplasticizer dosage is high under the assumed {self.superplasticizer_assumed_unit} convention."
                ),
                triggering_factors={
                    "superplasticizer": superplasticizer,
                    "assumed_unit": self.superplasticizer_assumed_unit,
                    "unit_confidence": self.superplasticizer_unit_confidence,
                    "superplasticizer_binder_ratio": superplasticizer_binder_ratio,
                    "dosage_warn_threshold": self.superplasticizer_dosage_warn_kg_per_m3,
                    "binder_ratio_warn_threshold": self.superplasticizer_binder_ratio_warn,
                },
                academic_note=(
                    "High admixture dosage can reflect a true high-range water reducer demand, a unit mismatch, "
                    "or a missing conversion between absolute dosage and binder-relative reporting."
                ),
                recommended_review_action=(
                    "Confirm the superplasticizer units and dosage basis before treating the mix as plausible."
                ),
                assessment_confidence=self.superplasticizer_unit_confidence,
            )

        hard_constraints = list(hard_constraints_by_code.values())
        engineering_cautions = list(engineering_cautions_by_code.values())
        data_review_flags = list(data_review_flags_by_code.values())
        contextual_summary = self._build_contextual_summary(
            context=context,
            hard_constraints=hard_constraints,
            engineering_cautions=engineering_cautions,
            data_review_flags=data_review_flags,
            downgraded_warnings=downgraded_warnings,
        )
        return self._build_sample_report(
            row_index=row_index,
            prediction=prediction_value,
            actual_value=actual_value,
            context=context,
            hard_constraints=hard_constraints,
            engineering_cautions=engineering_cautions,
            data_review_flags=data_review_flags,
            downgraded_warnings=downgraded_warnings,
            rule_evaluation_trace=rule_evaluation_trace,
            contextual_summary=contextual_summary,
        )

    def _build_sample_report(
        self,
        *,
        row_index: Any,
        prediction: float,
        actual_value: float | None,
        context: dict[str, Any],
        hard_constraints: list[dict[str, Any]],
        engineering_cautions: list[dict[str, Any]],
        data_review_flags: list[dict[str, Any]],
        downgraded_warnings: list[dict[str, Any]],
        rule_evaluation_trace: list[dict[str, Any]],
        contextual_summary: str,
    ) -> dict[str, Any]:
        """Build the public sample report with modern and compatibility fields."""
        warning_reasons = [item["message"] for item in engineering_cautions + data_review_flags]
        failure_reasons = [item["message"] for item in hard_constraints]
        engineering_caution_reasons = [item["message"] for item in engineering_cautions]
        data_review_flag_reasons = [item["message"] for item in data_review_flags]
        overall_verdict = "PASS"
        if hard_constraints:
            overall_verdict = "FAIL"
        elif warning_reasons:
            overall_verdict = "WARN"

        return {
            "index": int(row_index) if isinstance(row_index, (int, np.integer)) else str(row_index),
            "prediction_mpa": float(prediction),
            "actual_mpa": actual_value,
            "context_type": self.context_type,
            "water_cement_ratio": context["water_cement_ratio"],
            "water_binder_ratio": context["water_binder_ratio"],
            "water_effective_binder_ratio": context["water_effective_binder_ratio"],
            "total_binder": context["total_binder"],
            "effective_binder": context["effective_binder"],
            "cement_content": context["cement_content"],
            "scm_content": context["scm_content"],
            "supplementary_replacement_ratio": context["supplementary_replacement_ratio"],
            "fly_ash_replacement_ratio": context["fly_ash_replacement_ratio"],
            "slag_replacement_ratio": context["slag_replacement_ratio"],
            "age_days": context["age_days"],
            "target_strength": context["target_strength"],
            "exposure_class": context["exposure_context"]["label"],
            "structural_application": context.get("structural_application"),
            "workability_support": context["workability_support"]["status"],
            "hard_fails": hard_constraints,
            "engineering_warnings": engineering_cautions,
            "dataset_anomalies": data_review_flags,
            "hard_constraints": hard_constraints,
            "engineering_cautions": engineering_cautions,
            "data_review_flags": data_review_flags,
            "warning_reasons": warning_reasons,
            "failure_reasons": failure_reasons,
            "hard_failure_reasons": list(failure_reasons),
            "engineering_caution_reasons": engineering_caution_reasons,
            "durability_caution_reasons": list(engineering_caution_reasons),
            "durability_warning_reasons": list(engineering_caution_reasons),
            "data_review_flag_reasons": data_review_flag_reasons,
            "dataset_anomaly_reasons": list(data_review_flag_reasons),
            "statistical_error_reasons": list(data_review_flag_reasons),
            "triggered_rules": [
                item["warning_code"]
                for item in hard_constraints + engineering_cautions + data_review_flags
            ],
            "hard_constraint_reasons": list(failure_reasons),
            "downgraded_warnings": downgraded_warnings,
            "downgraded_rule_codes": [item["warning_code"] for item in downgraded_warnings],
            "rule_evaluation_trace": rule_evaluation_trace,
            "contextual_summary": contextual_summary,
            "confidence_of_warning_assessment": context["assessment_confidence"],
            "hard_constraint_count": len(hard_constraints),
            "hard_fail_count": len(hard_constraints),
            "engineering_caution_count": len(engineering_cautions),
            "engineering_warning_count": len(engineering_cautions),
            "data_review_flag_count": len(data_review_flags),
            "warning_count": len(engineering_cautions) + len(data_review_flags),
            "failure_count": len(hard_constraints),
            "hard_failure_count": len(hard_constraints),
            "durability_caution_count": len(engineering_cautions),
            "durability_warning_count": len(engineering_cautions),
            "dataset_anomaly_count": len(data_review_flags),
            "statistical_error_count": len(data_review_flags),
            "overall_verdict": overall_verdict,
        }

    def evaluate_samples(
        self,
        predictions: np.ndarray,
        x_frame: pd.DataFrame,
        y_true: pd.Series | None = None,
    ) -> list[dict[str, Any]]:
        """Return per-sample validation details for a set of predictions."""
        if len(predictions) != len(x_frame):
            raise ValueError("Prediction length does not match the provided feature frame.")

        prepared_frame = self._prepare_frame(x_frame)
        sample_reports: list[dict[str, Any]] = []
        for position, (row_index, prediction) in enumerate(zip(prepared_frame.index.tolist(), predictions)):
            actual_value = None if y_true is None else float(y_true.iloc[position])
            sample_reports.append(
                self._evaluate_single_sample(
                    row_index=row_index,
                    sample=prepared_frame.iloc[position],
                    prediction=float(prediction),
                    actual_value=actual_value,
                )
            )
        return sample_reports

    def validate_model(
        self,
        model: Any,
        x_frame: pd.DataFrame,
        y_true: pd.Series | None = None,
    ) -> dict[str, Any]:
        """Run a trained model on a feature frame and evaluate engineering rules."""
        predictions = np.asarray(model.predict(x_frame), dtype=float)
        return self.validate_predictions(predictions, x_frame, y_true)

    def validate_predictions(
        self,
        predictions: np.ndarray,
        x_frame: pd.DataFrame,
        y_true: pd.Series | None = None,
    ) -> dict[str, Any]:
        """Validate raw predictions and return a structured report."""
        sample_reports = self.evaluate_samples(predictions, x_frame, y_true)
        hard_constraint_samples = [sample for sample in sample_reports if sample["hard_constraints"]]
        engineering_caution_samples = [sample for sample in sample_reports if sample["engineering_cautions"]]
        data_review_flag_samples = [sample for sample in sample_reports if sample["data_review_flags"]]
        warning_samples = [
            sample
            for sample in sample_reports
            if sample["engineering_cautions"] or sample["data_review_flags"]
        ]
        warning_only_samples = [sample for sample in sample_reports if sample["overall_verdict"] == "WARN"]
        total_samples = len(sample_reports)
        passed_samples = total_samples - len(hard_constraint_samples) - len(warning_only_samples)
        pass_rate = passed_samples / total_samples if total_samples else 0.0

        overall_verdict = "PASS"
        if hard_constraint_samples:
            overall_verdict = "FAIL"
        elif warning_only_samples:
            overall_verdict = "WARN"

        hard_constraint_reasons = _unique_strings(
            [reason for sample in sample_reports for reason in sample["hard_constraint_reasons"]]
        )
        engineering_caution_reasons = _unique_strings(
            [reason for sample in sample_reports for reason in sample["engineering_caution_reasons"]]
        )
        data_review_flag_reasons = _unique_strings(
            [reason for sample in sample_reports for reason in sample["data_review_flag_reasons"]]
        )
        rule_violations_by_sample: dict[str, int] = {}
        downgraded_warning_details: list[dict[str, Any]] = []
        high_priority_warnings: list[dict[str, Any]] = []
        for sample in sample_reports:
            for rule_name in sample["triggered_rules"]:
                rule_violations_by_sample[rule_name] = rule_violations_by_sample.get(rule_name, 0) + 1
            for downgraded_warning in sample["downgraded_warnings"]:
                downgraded_warning_details.append(
                    {
                        "sample_index": sample["index"],
                        **dict(downgraded_warning),
                    }
                )
            for warning in sample["hard_constraints"] + sample["engineering_cautions"] + sample["data_review_flags"]:
                if (
                    warning["warning_category"] == WARNING_CATEGORY_HARD_CONSTRAINT
                    or warning["severity"] == SEVERITY_HIGH
                ):
                    high_priority_warnings.append(
                        {
                            "sample_index": sample["index"],
                            **dict(warning),
                        }
                    )

        report_confidence = _confidence_minimum(
            *[sample["confidence_of_warning_assessment"] for sample in sample_reports]
        )
        contextual_summary = (
            sample_reports[0]["contextual_summary"]
            if len(sample_reports) == 1
            else (
                f"{len(hard_constraint_samples)} sample(s) with hard constraints, "
                f"{len(engineering_caution_samples)} with engineering cautions, "
                f"{len(data_review_flag_samples)} with data review flags, "
                f"{len(downgraded_warning_details)} downgraded rule decision(s)."
            )
        )

        return {
            "context_type": self.context_type,
            "pass_rate": pass_rate,
            "hard_fail_samples": hard_constraint_samples,
            "failed_samples": hard_constraint_samples,
            "hard_constraint_samples": hard_constraint_samples,
            "warning_samples": warning_samples,
            "suspicious_samples": data_review_flag_samples,
            "statistical_error_samples": data_review_flag_samples,
            "engineering_warning_samples": engineering_caution_samples,
            "engineering_caution_samples": engineering_caution_samples,
            "durability_caution_samples": engineering_caution_samples,
            "durability_warning_samples": engineering_caution_samples,
            "dataset_anomaly_samples_v2": data_review_flag_samples,
            "data_review_flag_samples": data_review_flag_samples,
            "dataset_anomaly_samples": data_review_flag_samples,
            "sample_reports": sample_reports,
            "hard_constraint_count": len(hard_constraint_samples),
            "hard_fail_count": len(hard_constraint_samples),
            "failed_count": len(hard_constraint_samples),
            "hard_failed_count": len(hard_constraint_samples),
            "engineering_caution_count": len(engineering_caution_samples),
            "engineering_warning_count": len(engineering_caution_samples),
            "warning_count": len(warning_samples),
            "data_review_flag_count": len(data_review_flag_samples),
            "suspicious_count": len(data_review_flag_samples),
            "statistical_errors": len(data_review_flag_samples),
            "statistical_error_count": len(data_review_flag_samples),
            "durability_warnings": len(engineering_caution_samples),
            "durability_caution_count": len(engineering_caution_samples),
            "durability_warning_count": len(engineering_caution_samples),
            "dataset_anomalies": len(data_review_flag_samples),
            "dataset_anomaly_count": len(data_review_flag_samples),
            "warn_reasons": _unique_strings(engineering_caution_reasons + data_review_flag_reasons),
            "hard_constraint_reasons": hard_constraint_reasons,
            "hard_fail_reasons": list(hard_constraint_reasons),
            "engineering_caution_reasons": engineering_caution_reasons,
            "durability_caution_reasons": list(engineering_caution_reasons),
            "durability_warning_reasons": list(engineering_caution_reasons),
            "data_review_flag_reasons": data_review_flag_reasons,
            "dataset_anomaly_reasons": list(data_review_flag_reasons),
            "statistical_error_reasons": list(data_review_flag_reasons),
            "rule_violations_by_sample": rule_violations_by_sample,
            "downgraded_warning_details": downgraded_warning_details,
            "downgraded_warning_count": len(downgraded_warning_details),
            "high_priority_warnings": high_priority_warnings,
            "contextual_summary": contextual_summary,
            "confidence_of_warning_assessment": report_confidence,
            "overall_verdict": overall_verdict,
            "verdict": overall_verdict,
        }


def summarize_validation_report(validation_report: dict[str, Any]) -> dict[str, Any]:
    """Return a compact validation summary with new and legacy aliases."""
    hard_constraint_count = int(
        validation_report.get(
            "hard_constraint_count",
            validation_report.get("hard_failed_count", validation_report.get("failed_count", 0)),
        )
    )
    engineering_caution_count = int(
        validation_report.get(
            "engineering_caution_count",
            validation_report.get("durability_caution_count", validation_report.get("warning_count", 0)),
        )
    )
    data_review_flag_count = int(
        validation_report.get(
            "data_review_flag_count",
            validation_report.get(
                "dataset_anomaly_count",
                validation_report.get("suspicious_count", validation_report.get("statistical_error_count", 0)),
            ),
        )
    )
    hard_constraint_reasons = list(
        validation_report.get(
            "hard_constraint_reasons",
            validation_report.get("hard_fail_reasons", []),
        )
    )
    engineering_caution_reasons = list(
        validation_report.get(
            "engineering_caution_reasons",
            validation_report.get("durability_caution_reasons", []),
        )
    )
    data_review_flag_reasons = list(
        validation_report.get(
            "data_review_flag_reasons",
            validation_report.get(
                "dataset_anomaly_reasons",
                validation_report.get("statistical_error_reasons", []),
            ),
        )
    )
    return {
        "context_type": validation_report.get("context_type", "general"),
        "pass_rate": float(validation_report.get("pass_rate", 0.0)),
        "verdict": validation_report.get("verdict", validation_report.get("overall_verdict", "UNKNOWN")),
        "hard_constraint_count": hard_constraint_count,
        "hard_failed_count": hard_constraint_count,
        "failed_count": hard_constraint_count,
        "engineering_caution_count": engineering_caution_count,
        "warning_count": int(
            validation_report.get("warning_count", engineering_caution_count + data_review_flag_count)
        ),
        "durability_caution_count": engineering_caution_count,
        "durability_warning_count": engineering_caution_count,
        "durability_warnings": engineering_caution_count,
        "data_review_flag_count": data_review_flag_count,
        "dataset_anomaly_count": data_review_flag_count,
        "dataset_anomalies": data_review_flag_count,
        "suspicious_count": data_review_flag_count,
        "statistical_errors": data_review_flag_count,
        "statistical_error_count": data_review_flag_count,
        "hard_constraint_reasons": hard_constraint_reasons,
        "hard_fail_reasons": list(hard_constraint_reasons),
        "engineering_caution_reasons": engineering_caution_reasons,
        "durability_caution_reasons": list(engineering_caution_reasons),
        "durability_warning_reasons": list(engineering_caution_reasons),
        "data_review_flag_reasons": data_review_flag_reasons,
        "dataset_anomaly_reasons": list(data_review_flag_reasons),
        "statistical_error_reasons": list(data_review_flag_reasons),
        "contextual_summary": validation_report.get("contextual_summary", ""),
        "confidence_of_warning_assessment": validation_report.get(
            "confidence_of_warning_assessment",
            CONFIDENCE_HIGH,
        ),
    }
