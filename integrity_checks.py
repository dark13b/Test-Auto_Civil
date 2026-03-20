"""Integrity checks for scientifically credible result payloads."""

from __future__ import annotations

from typing import Any


def assert_no_cv_label_fraud(result_dict: dict[str, Any]) -> None:
    """Reject result payloads that expose held-out metrics without genuine CV metrics."""
    has_cv_metrics = any(result_dict.get(key) is not None for key in ("cv_r2", "cv_rmse", "cv_mae"))
    has_non_cv_metrics = any(
        result_dict.get(key) is not None
        for key in ("val_r2", "val_rmse", "val_mae", "test_r2", "test_rmse", "test_mae")
    )
    if has_non_cv_metrics and not has_cv_metrics:
        raise ValueError("Validation or holdout metrics are present without genuine cross-validation metrics.")
