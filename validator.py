"""Engineering validation rules for concrete strength predictions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class EngineeringValidator:
    """Validate predictions against domain-specific engineering rules."""

    min_strength_mpa: float
    max_strength_mpa: float
    suspicious_water_cement_ratio: float
    suspicious_strength_mpa: float
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
        )

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
        if len(predictions) != len(x_test):
            raise ValueError("Prediction length does not match the provided feature frame.")
        if self.water_column not in x_test.columns or self.cement_column not in x_test.columns:
            raise ValueError(
                f"Required columns '{self.water_column}' and '{self.cement_column}' are missing."
            )

        failed_samples: list[dict[str, Any]] = []
        suspicious_samples: list[dict[str, Any]] = []
        cement_values = np.asarray(x_test[self.cement_column], dtype=float)
        safe_cement = np.where(cement_values <= 1e-8, np.nan, cement_values)
        water_cement_ratio = np.asarray(x_test[self.water_column], dtype=float) / safe_cement

        for position, (row_index, prediction) in enumerate(zip(x_test.index.tolist(), predictions)):
            actual_value = None if y_test is None else float(y_test.iloc[position])
            ratio = float(water_cement_ratio[position]) if not np.isnan(water_cement_ratio[position]) else float("inf")
            base_record = {
                "index": int(row_index) if isinstance(row_index, (int, np.integer)) else str(row_index),
                "prediction_mpa": float(prediction),
                "actual_mpa": actual_value,
                "water_cement_ratio": ratio,
            }
            if prediction < self.min_strength_mpa:
                failed_samples.append({**base_record, "reason": "prediction_below_min_bound"})
                continue
            if prediction > self.max_strength_mpa:
                failed_samples.append({**base_record, "reason": "prediction_above_max_bound"})
                continue
            if ratio > self.suspicious_water_cement_ratio and prediction > self.suspicious_strength_mpa:
                suspicious_samples.append(
                    {**base_record, "reason": "high_water_cement_ratio_with_high_strength"}
                )

        total_samples = len(predictions)
        passed_samples = total_samples - len(failed_samples) - len(suspicious_samples)
        pass_rate = passed_samples / total_samples if total_samples else 0.0
        verdict = "PASS"
        if failed_samples:
            verdict = "FAIL"
        elif suspicious_samples:
            verdict = "WARN"

        return {
            "pass_rate": pass_rate,
            "failed_samples": failed_samples,
            "suspicious_samples": suspicious_samples,
            "failed_count": len(failed_samples),
            "suspicious_count": len(suspicious_samples),
            "verdict": verdict,
        }
