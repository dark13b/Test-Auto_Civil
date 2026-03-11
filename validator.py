"""Engineering validation rules for concrete strength predictions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from feature_engineering import build_engineering_features

CONTEXT_DURABILITY_WATER_CEMENT_LIMITS = {
    "exposed": 0.45,
    "structural": 0.50,
    "general": 0.60,
}


@dataclass
class EngineeringValidator:
    """Validate predictions against domain-specific engineering rules."""

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

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "EngineeringValidator":
        """Construct a validator from the project configuration."""
        bounds = config["engineering_bounds"]
        rules = config["validator"]
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
        )

    def __post_init__(self) -> None:
        """Validate validator configuration after dataclass initialization."""
        if self.context_type not in CONTEXT_DURABILITY_WATER_CEMENT_LIMITS:
            valid_contexts = ", ".join(sorted(CONTEXT_DURABILITY_WATER_CEMENT_LIMITS))
            raise ValueError(f"validator.context_type must be one of: {valid_contexts}")

    def _durability_water_cement_limit(self) -> float:
        """Return the water/cement durability threshold for the active context."""
        return float(
            CONTEXT_DURABILITY_WATER_CEMENT_LIMITS.get(
                self.context_type,
                self.durability_water_cement_warn,
            )
        )

    def _prepare_frame(self, x_test: pd.DataFrame) -> pd.DataFrame:
        """Ensure engineered columns required by the validator are present."""
        if self.water_column not in x_test.columns or self.cement_column not in x_test.columns:
            raise ValueError(
                f"Required columns '{self.water_column}' and '{self.cement_column}' are missing."
            )
        return build_engineering_features(x_test)

    def _evaluate_single_sample(
        self,
        row_index: Any,
        sample: pd.Series,
        prediction: float,
        actual_value: float | None = None,
    ) -> dict[str, Any]:
        """Evaluate a single prediction against engineering rules."""
        warning_reasons: list[str] = []
        failure_reasons: list[str] = []
        statistical_error_reasons: list[str] = []
        durability_caution_reasons: list[str] = []
        dataset_anomaly_reasons: list[str] = []
        triggered_rules: list[str] = []

        def add_warning(
            reason: str,
            rule_name: str,
            *,
            statistical: bool = False,
            durability: bool = False,
            dataset_anomaly: bool = False,
        ) -> None:
            """Append a warning once and optionally classify it."""
            if reason not in warning_reasons:
                warning_reasons.append(reason)
            if statistical and reason not in statistical_error_reasons:
                statistical_error_reasons.append(reason)
            if durability and reason not in durability_caution_reasons:
                durability_caution_reasons.append(reason)
            if dataset_anomaly and reason not in dataset_anomaly_reasons:
                dataset_anomaly_reasons.append(reason)
            if rule_name not in triggered_rules:
                triggered_rules.append(rule_name)

        def add_failure(reason: str, rule_name: str) -> None:
            """Append a hard failure once."""
            if reason not in failure_reasons:
                failure_reasons.append(reason)
            if reason not in statistical_error_reasons:
                statistical_error_reasons.append(reason)
            if rule_name not in triggered_rules:
                triggered_rules.append(rule_name)

        water_cement_ratio = float(sample["water_cement_ratio"])
        water_binder_ratio = float(sample["water_binder_ratio"])
        total_binder = float(sample["total_binder"])
        supplementary_replacement_ratio = float(sample["supplementary_replacement_ratio"])
        fly_ash_replacement_ratio = float(sample["fly_ash_replacement_ratio"])
        slag_replacement_ratio = float(sample["slag_replacement_ratio"])
        age = float(sample["age"])
        scm_present = supplementary_replacement_ratio > 0.0

        if not np.isfinite(prediction):
            add_failure(
                "Prediction is non-finite and cannot be interpreted as a concrete strength.",
                "prediction_non_finite",
            )
        else:
            if prediction < 0.0:
                add_failure("Negative predicted strength is physically impossible.", "prediction_negative_strength")
            if prediction < self.min_strength_mpa:
                add_failure(
                    "Prediction falls below configured minimum strength bound.",
                    "prediction_below_min_bound",
                )
            if prediction > self.max_strength_mpa:
                add_failure(
                    "Prediction exceeds configured maximum strength bound.",
                    "prediction_above_max_bound",
                )

            high_water_cement_high_strength = (
                water_cement_ratio > self.suspicious_water_cement_ratio
                and prediction > self.suspicious_strength_mpa
            )
            if high_water_cement_high_strength:
                early_age = age <= self.high_water_cement_strength_hard_fail_max_age_days
                low_scm_replacement = (
                    supplementary_replacement_ratio
                    < self.high_water_cement_strength_low_scm_threshold
                )
                unfavorable_water_binder_ratio = (
                    water_binder_ratio
                    >= self.high_water_cement_strength_unfavorable_water_binder_ratio
                )

                # Plain w/c becomes too blunt once SCM replacement and later-age strength gain
                # enter the mix. Keep this combination as a hard reject only when the binder
                # context also looks implausible for early-age concrete.
                if early_age and low_scm_replacement and unfavorable_water_binder_ratio:
                    add_failure(
                        "Very high water/cement ratio paired with high early-age strength, low SCM replacement, and an unfavorable water/binder ratio remains implausible.",
                        "high_water_cement_ratio_with_high_strength_hard_fail",
                    )
                else:
                    add_warning(
                        "High strength at very high water/cement ratio is unusual; review curing age and SCM binder effects before accepting it.",
                        "high_water_cement_ratio_with_high_strength_warn",
                        statistical=True,
                        dataset_anomaly=True,
                    )
                    if scm_present:
                        add_warning(
                            "SCM-bearing mixes should be screened with water/binder ratio and curing age, not water/cement ratio alone.",
                            "scm_mix_water_binder_context_used",
                            statistical=True,
                        )
                    if early_age and unfavorable_water_binder_ratio:
                        add_warning(
                            "Binder-based water ratio is still high for the predicted early-age strength; verify curing and testing context.",
                            "early_age_high_strength_with_unfavorable_water_binder_ratio_warn",
                            statistical=True,
                        )

        if water_cement_ratio > self._durability_water_cement_limit():
            add_warning(
                f"Water/cement ratio exceeds the typical durability limit for {self.context_type} concrete.",
                "water_cement_ratio_warn_exceeds_durability_limit",
                durability=True,
            )
        if water_binder_ratio < self.low_water_binder_warn:
            add_warning(
                "Water/binder ratio is very low; workability may be compromised.",
                "water_binder_ratio_warn_too_low",
                durability=True,
            )
        if total_binder < self.total_binder_low_warn:
            add_warning(
                "Total binder content is low; durability may be at risk.",
                "total_binder_warn_too_low",
                durability=True,
            )
        if total_binder > self.total_binder_high_warn:
            add_warning(
                "Total binder content is high; shrinkage risk increases.",
                "total_binder_warn_too_high",
                durability=True,
            )
        if fly_ash_replacement_ratio > self.fly_ash_replacement_warn:
            add_warning(
                "Fly ash replacement ratio exceeds the typical ACI substitution guidance.",
                "fly_ash_replacement_ratio_warn_too_high",
                dataset_anomaly=True,
            )
        if slag_replacement_ratio > self.slag_replacement_warn:
            add_warning(
                "Slag replacement ratio exceeds the typical BS 8500 guidance.",
                "slag_replacement_ratio_warn_too_high",
                dataset_anomaly=True,
            )
        if age < self.early_age_days_warn and prediction > self.early_age_strength_warn:
            add_warning(
                "Early-age strength prediction is suspiciously high for the curing age.",
                "early_age_strength_warn_suspicious",
                statistical=True,
            )
        if (
            scm_present
            and water_binder_ratio >= self.high_water_cement_strength_unfavorable_water_binder_ratio
            and age <= self.high_water_cement_strength_hard_fail_max_age_days
            and np.isfinite(prediction)
            and prediction > self.suspicious_strength_mpa
        ):
            add_warning(
                "SCM-rich mix still shows a relatively high water/binder ratio for the predicted early-age strength; review curing and binder chemistry.",
                "scm_mix_high_water_binder_ratio_warn",
                statistical=True,
            )

        overall_verdict = "PASS"
        if failure_reasons:
            overall_verdict = "FAIL"
        elif warning_reasons:
            overall_verdict = "WARN"

        return {
            "index": int(row_index) if isinstance(row_index, (int, np.integer)) else str(row_index),
            "prediction_mpa": float(prediction),
            "actual_mpa": actual_value,
            "context_type": self.context_type,
            "water_cement_ratio": water_cement_ratio,
            "water_binder_ratio": water_binder_ratio,
            "total_binder": total_binder,
            "supplementary_replacement_ratio": supplementary_replacement_ratio,
            "fly_ash_replacement_ratio": fly_ash_replacement_ratio,
            "slag_replacement_ratio": slag_replacement_ratio,
            "age_days": age,
            "statistical_error_reasons": statistical_error_reasons,
            "warning_reasons": warning_reasons,
            "failure_reasons": failure_reasons,
            "hard_failure_reasons": list(failure_reasons),
            "durability_caution_reasons": durability_caution_reasons,
            "durability_warning_reasons": list(durability_caution_reasons),
            "dataset_anomaly_reasons": dataset_anomaly_reasons,
            "triggered_rules": triggered_rules,
            "statistical_error_count": len(statistical_error_reasons),
            "warning_count": len(warning_reasons),
            "failure_count": len(failure_reasons),
            "hard_failure_count": len(failure_reasons),
            "durability_caution_count": len(durability_caution_reasons),
            "durability_warning_count": len(durability_caution_reasons),
            "dataset_anomaly_count": len(dataset_anomaly_reasons),
            "overall_verdict": overall_verdict,
        }

    def evaluate_samples(
        self,
        predictions: np.ndarray,
        x_test: pd.DataFrame,
        y_test: pd.Series | None = None,
    ) -> list[dict[str, Any]]:
        """Return per-sample validation details for a set of predictions."""
        if len(predictions) != len(x_test):
            raise ValueError("Prediction length does not match the provided feature frame.")

        prepared_frame = self._prepare_frame(x_test)
        sample_reports: list[dict[str, Any]] = []
        for position, (row_index, prediction) in enumerate(zip(prepared_frame.index.tolist(), predictions)):
            actual_value = None if y_test is None else float(y_test.iloc[position])
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
        x_test: pd.DataFrame,
        y_test: pd.Series | None = None,
    ) -> dict[str, Any]:
        """Run a trained model on a test set and evaluate engineering rules."""
        predictions = np.asarray(model.predict(x_test), dtype=float)
        return self.validate_predictions(predictions, x_test, y_test)

    def validate_predictions(
        self,
        predictions: np.ndarray,
        x_test: pd.DataFrame,
        y_test: pd.Series | None = None,
    ) -> dict[str, Any]:
        """Validate raw predictions and return a structured report."""
        sample_reports = self.evaluate_samples(predictions, x_test, y_test)
        failed_samples = [sample for sample in sample_reports if sample["overall_verdict"] == "FAIL"]
        warning_samples = [sample for sample in sample_reports if sample["warning_reasons"]]
        warning_only_samples = [sample for sample in sample_reports if sample["overall_verdict"] == "WARN"]
        statistical_error_samples = [
            sample for sample in sample_reports if sample["statistical_error_reasons"]
        ]
        durability_caution_samples = [
            sample for sample in sample_reports if sample["durability_caution_reasons"]
        ]
        dataset_anomaly_samples = [sample for sample in sample_reports if sample["dataset_anomaly_reasons"]]
        total_samples = len(sample_reports)
        passed_samples = total_samples - len(failed_samples) - len(warning_only_samples)
        pass_rate = passed_samples / total_samples if total_samples else 0.0
        overall_verdict = "PASS"
        if failed_samples:
            overall_verdict = "FAIL"
        elif warning_only_samples:
            overall_verdict = "WARN"

        warn_reasons = sorted(
            {
                reason
                for sample in sample_reports
                for reason in sample["warning_reasons"]
            }
        )
        statistical_error_reasons = sorted(
            {
                reason
                for sample in sample_reports
                for reason in sample["statistical_error_reasons"]
            }
        )
        hard_fail_reasons = sorted(
            {
                reason
                for sample in sample_reports
                for reason in sample["failure_reasons"]
            }
        )
        durability_caution_reasons = sorted(
            {
                reason
                for sample in sample_reports
                for reason in sample["durability_caution_reasons"]
            }
        )
        dataset_anomaly_reasons = sorted(
            {
                reason
                for sample in sample_reports
                for reason in sample["dataset_anomaly_reasons"]
            }
        )
        rule_violations_by_sample: dict[str, int] = {}
        for sample in sample_reports:
            for rule_name in sample["triggered_rules"]:
                rule_violations_by_sample[rule_name] = rule_violations_by_sample.get(rule_name, 0) + 1

        return {
            "context_type": self.context_type,
            "pass_rate": pass_rate,
            "failed_samples": failed_samples,
            "warning_samples": warning_samples,
            "suspicious_samples": statistical_error_samples,
            "statistical_error_samples": statistical_error_samples,
            "durability_caution_samples": durability_caution_samples,
            "durability_warning_samples": durability_caution_samples,
            "dataset_anomaly_samples": dataset_anomaly_samples,
            "failed_count": len(failed_samples),
            "hard_failed_count": len(failed_samples),
            "warning_count": len(warning_samples),
            "suspicious_count": len(statistical_error_samples),
            "statistical_errors": len(statistical_error_samples),
            "statistical_error_count": len(statistical_error_samples),
            "durability_warnings": len(durability_caution_samples),
            "durability_caution_count": len(durability_caution_samples),
            "durability_warning_count": len(durability_caution_samples),
            "dataset_anomalies": len(dataset_anomaly_samples),
            "dataset_anomaly_count": len(dataset_anomaly_samples),
            "warn_reasons": warn_reasons,
            "statistical_error_reasons": statistical_error_reasons,
            "hard_fail_reasons": hard_fail_reasons,
            "durability_caution_reasons": durability_caution_reasons,
            "durability_warning_reasons": durability_caution_reasons,
            "dataset_anomaly_reasons": dataset_anomaly_reasons,
            "rule_violations_by_sample": rule_violations_by_sample,
            "sample_reports": sample_reports,
            "overall_verdict": overall_verdict,
            "verdict": overall_verdict,
        }
