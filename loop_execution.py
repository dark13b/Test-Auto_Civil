"""Trial execution helpers for the AutoCivil-Lab research loop."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from train import EngineeringValidator, evaluate_candidate


def validate_before_keep(run_dir: Path, logger) -> bool:
    """Return True only if all required artifacts are present and non-empty."""
    required = ["final_holdout_evaluation.json", "best_search_result.json", "final_acceptance.json"]
    for fname in required:
        p = run_dir / fname
        if not p.exists() or p.stat().st_size == 0:
            logger.error("validate_before_keep: missing or empty %s — aborting keep", fname)
            return False
    return True


def _safe_float(value: Any, fallback: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(fallback)


def calculate_improvement_percentage(reference_score: float, candidate_score: float) -> float:
    denominator = max(abs(reference_score), 1e-8)
    return ((candidate_score - reference_score) / denominator) * 100.0


def _evaluate_stage_candidate(
    *,
    experiment: dict[str, Any],
    config: dict[str, Any],
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_val: pd.DataFrame,
    y_val: pd.Series,
    validator: EngineeringValidator,
) -> tuple[Any, dict[str, Any]]:
    return evaluate_candidate(experiment["model_name"], dict(experiment["params"]), x_train, y_train, x_val, y_val, validator, config)


def _build_stage_config(base_config: dict[str, Any], cv_repeats: int) -> dict[str, Any]:
    import copy
    stage_config = copy.deepcopy(base_config)
    stage_config["experiment"]["cv_repeats"] = max(1, int(cv_repeats))
    return stage_config
