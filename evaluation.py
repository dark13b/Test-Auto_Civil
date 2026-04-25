"""Single responsibility: metric computation, validation loop, and scoring."""

from __future__ import annotations

from typing import Any

from train_impl import compute_regression_metrics


def evaluate_model(model: Any, data: Any, config: dict[str, Any]) -> dict[str, Any]:
    predictions = model.predict(data.features)
    return compute_regression_metrics(data.target, predictions, config, float(data.target.mean()))
