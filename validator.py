"""Engineering validation rules for concrete strength predictions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from feature_engineering import build_engineering_features


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
            low_water_binder_warn=float(rules["low_water_binder_warn"]),
            total_binder_low_warn=float(rules["total_binder_low_warn"]),
            total_binder_high_warn=float(rules["total_binder_high_warn"]),
            fly_ash_replacement_warn=float(rules["fly_ash_replacement_warn"]),
            slag_replacement_warn=float(rules["slag_replacement_warn"]),
            early_age_days_warn=float(rules["early_age_days_warn"]),
            early_age_strength_warn=float(rules["early_age_strength_warn"]),
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
        triggered_rules: list[str] = []

        water_cement_ratio = float(sample["water_cement_ratio"])
        water_binder_ratio = float(sample["water_binder_ratio"])
        total_binder = float(sample["total_binder"])
        fly_ash_replacement_ratio = float(sample["fly_ash_replacement_ratio"])
        slag_replacement_ratio = float(sample["slag_replacement_ratio"])
        age = float(sample["age"])

        if prediction < self.min_strength_mpa:
            failure_reasons.append("Prediction falls below configured minimum strength bound.")
            triggered_rules.append("prediction_below_min_bound")
        if prediction > self.max_strength_mpa:
            failure_reasons.append("Prediction exceeds configured maximum strength bound.")
            triggered_rules.append("prediction_above_max_bound")
        if (
            water_cement_ratio > self.suspicious_water_cement_ratio
            and prediction > self.suspicious_strength_mpa
        ):
            failure_reasons.append(
                "Very high water/cement ratio paired with high predicted strength exceeds the hard screening limit."
            )
            triggered_rules.append("water_cement_ratio_above_hard_limit")

        if water_cement_ratio > self.durability_water_cement_warn:
            warning_reasons.append(
                "Water/cement ratio exceeds the typical durability limit for moderate exposure."
            )
            triggered_rules.append("water_cement_ratio_warn_exceeds_durability_limit")
        if water_binder_ratio < self.low_water_binder_warn:
            warning_reasons.append("Water/binder ratio is very low; workability may be compromised.")
            triggered_rules.append("water_binder_ratio_warn_too_low")
        if total_binder < self.total_binder_low_warn:
            warning_reasons.append("Total binder content is low; durability may be at risk.")
            triggered_rules.append("total_binder_warn_too_low")
        if total_binder > self.total_binder_high_warn:
            warning_reasons.append("Total binder content is high; shrinkage risk increases.")
            triggered_rules.append("total_binder_warn_too_high")
        if fly_ash_replacement_ratio > self.fly_ash_replacement_warn:
            warning_reasons.append(
                "Fly ash replacement ratio exceeds the typical ACI substitution guidance."
            )
            triggered_rules.append("fly_ash_replacement_ratio_warn_too_high")
        if slag_replacement_ratio > self.slag_replacement_warn:
            warning_reasons.append("Slag replacement ratio exceeds the typical BS 8500 guidance.")
            triggered_rules.append("slag_replacement_ratio_warn_too_high")
        if age < self.early_age_days_warn and prediction > self.early_age_strength_warn:
            warning_reasons.append("Early-age strength prediction is suspiciously high for the curing age.")
            triggered_rules.append("early_age_strength_warn_suspicious")
        if (
            water_cement_ratio > self.suspicious_water_cement_ratio
            and prediction > self.suspicious_strength_mpa
        ):
            warning_reasons.append(
                "High strength at very high water/cement ratio is suspicious and should be checked."
            )
            triggered_rules.append("high_water_cement_ratio_with_high_strength")

        overall_verdict = "PASS"
        if failure_reasons:
            overall_verdict = "FAIL"
        elif warning_reasons:
            overall_verdict = "WARN"

        return {
            "index": int(row_index) if isinstance(row_index, (int, np.integer)) else str(row_index),
            "prediction_mpa": float(prediction),
            "actual_mpa": actual_value,
            "water_cement_ratio": water_cement_ratio,
            "water_binder_ratio": water_binder_ratio,
            "total_binder": total_binder,
            "fly_ash_replacement_ratio": fly_ash_replacement_ratio,
            "slag_replacement_ratio": slag_replacement_ratio,
            "age_days": age,
            "warning_reasons": warning_reasons,
            "failure_reasons": failure_reasons,
            "triggered_rules": triggered_rules,
            "warning_count": len(warning_reasons),
            "failure_count": len(failure_reasons),
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
        rule_violations_by_sample: dict[str, int] = {}
        for sample in sample_reports:
            for rule_name in sample["triggered_rules"]:
                rule_violations_by_sample[rule_name] = rule_violations_by_sample.get(rule_name, 0) + 1

        return {
            "pass_rate": pass_rate,
            "failed_samples": failed_samples,
            "warning_samples": warning_samples,
            "suspicious_samples": warning_samples,
            "failed_count": len(failed_samples),
            "warning_count": len(warning_samples),
            "suspicious_count": len(warning_samples),
            "warn_reasons": warn_reasons,
            "rule_violations_by_sample": rule_violations_by_sample,
            "sample_reports": sample_reports,
            "overall_verdict": overall_verdict,
            "verdict": overall_verdict,
        }
