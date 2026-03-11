"""Engineering feature generation utilities for concrete mix datasets."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


ENGINEERED_FEATURE_COLUMNS = [
    "water_cement_ratio",
    "water_binder_ratio",
    "slag_replacement_ratio",
    "fly_ash_replacement_ratio",
    "aggregate_paste_ratio",
    "fine_to_coarse_ratio",
    "total_binder",
    "supplementary_replacement_ratio",
    "superplasticizer_binder_ratio",
    "paste_volume_proxy",
    "log_age",
    "cement_age_interaction",
    "binder_age_interaction",
    "water_binder_age",
]

_REQUIRED_BASE_COLUMNS = [
    "cement",
    "slag",
    "fly_ash",
    "water",
    "superplasticizer",
    "coarse_aggregate",
    "fine_aggregate",
    "age",
]

_EPSILON = 1e-12


def log_status(message: str) -> None:
    """Print a timestamped status message."""
    timestamp = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")


def _safe_divide(
    numerator: float | np.ndarray | pd.Series,
    denominator: float | np.ndarray | pd.Series,
    default: float = 0.0,
) -> np.ndarray:
    """Safely divide arrays and return a finite NumPy array."""
    numerator_array, denominator_array = np.broadcast_arrays(
        np.asarray(numerator, dtype=float),
        np.asarray(denominator, dtype=float),
    )
    result = np.full(numerator_array.shape, default, dtype=float)
    np.divide(
        numerator_array,
        denominator_array,
        out=result,
        where=np.abs(denominator_array) > _EPSILON,
    )
    return result


def _missing_columns(columns: Iterable[str], frame: pd.DataFrame) -> list[str]:
    """Return missing columns from a dataframe."""
    return [column for column in columns if column not in frame.columns]


def build_engineering_features(df: pd.DataFrame) -> pd.DataFrame:
    """Append civil-engineering-derived features to a normalized concrete dataframe."""
    missing_columns = _missing_columns(_REQUIRED_BASE_COLUMNS, df)
    if missing_columns:
        raise ValueError(f"Dataframe is missing base engineering columns: {missing_columns}")

    frame = df.copy()
    numeric_frame = frame[_REQUIRED_BASE_COLUMNS].apply(pd.to_numeric, errors="coerce")
    if numeric_frame.isnull().any().any():
        invalid_columns = numeric_frame.columns[numeric_frame.isnull().any()].tolist()
        raise ValueError(f"Engineering feature inputs contain non-numeric values: {invalid_columns}")

    cement = numeric_frame["cement"].to_numpy(dtype=float)
    slag = numeric_frame["slag"].to_numpy(dtype=float)
    fly_ash = numeric_frame["fly_ash"].to_numpy(dtype=float)
    water = numeric_frame["water"].to_numpy(dtype=float)
    superplasticizer = numeric_frame["superplasticizer"].to_numpy(dtype=float)
    coarse_aggregate = numeric_frame["coarse_aggregate"].to_numpy(dtype=float)
    fine_aggregate = numeric_frame["fine_aggregate"].to_numpy(dtype=float)
    age = numeric_frame["age"].to_numpy(dtype=float)

    total_binder = cement + slag + fly_ash
    binder_plus_water = total_binder + water
    log_age = np.log(np.maximum(age, 0.0) + 1.0)

    frame["water_cement_ratio"] = _safe_divide(water, cement)
    frame["water_binder_ratio"] = _safe_divide(water, total_binder)
    frame["slag_replacement_ratio"] = _safe_divide(slag, total_binder)
    frame["fly_ash_replacement_ratio"] = _safe_divide(fly_ash, total_binder)
    frame["aggregate_paste_ratio"] = _safe_divide(
        coarse_aggregate + fine_aggregate,
        binder_plus_water,
    )
    frame["fine_to_coarse_ratio"] = _safe_divide(fine_aggregate, coarse_aggregate)
    frame["total_binder"] = total_binder
    frame["supplementary_replacement_ratio"] = _safe_divide(slag + fly_ash, total_binder)
    frame["superplasticizer_binder_ratio"] = _safe_divide(superplasticizer, total_binder)
    frame["paste_volume_proxy"] = total_binder + water + superplasticizer
    frame["log_age"] = log_age
    frame["cement_age_interaction"] = cement * log_age
    frame["binder_age_interaction"] = total_binder * log_age
    frame["water_binder_age"] = _safe_divide(frame["water_binder_ratio"], log_age)
    return frame


def validate_features(df: pd.DataFrame) -> bool:
    """Validate engineered features and print a compact summary."""
    missing_columns = _missing_columns(ENGINEERED_FEATURE_COLUMNS, df)
    if missing_columns:
        log_status(f"Feature validation failed. Missing engineered columns: {missing_columns}")
        return False

    feature_frame = df[ENGINEERED_FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce")
    invalid_mask = ~np.isfinite(feature_frame.to_numpy(dtype=float))
    if invalid_mask.any():
        invalid_columns = feature_frame.columns[np.any(invalid_mask, axis=0)].tolist()
        log_status(f"Feature validation failed. Non-finite values found in: {invalid_columns}")
        return False

    summary = feature_frame.agg(["min", "max", "mean"]).transpose()
    log_status("Engineered feature summary:")
    for column_name, row in summary.iterrows():
        log_status(
            f"{column_name}: min={row['min']:.6f}, max={row['max']:.6f}, mean={row['mean']:.6f}"
        )
    return True
