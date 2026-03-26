"""Integrity checks for scientifically credible result payloads."""

from __future__ import annotations

from typing import Any


def assert_no_cv_label_fraud(result_dict: dict[str, Any]) -> None:
    """Reject result payloads that expose held-out metrics without genuine CV metrics."""
    has_cv_metrics = any(result_dict.get(key) is not None for key in ("cv_r2", "cv_rmse", "cv_mae"))
    has_non_cv_metrics = any(
        result_dict.get(key) is not None
        for key in ("validation_r2", "validation_rmse", "validation_mae", "holdout_r2", "holdout_rmse", "holdout_mae")
    )
    if has_non_cv_metrics and not has_cv_metrics:
        raise ValueError("Validation or holdout metrics are present without genuine cross-validation metrics.")
