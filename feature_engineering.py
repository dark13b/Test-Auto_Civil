"""Engineering feature generation utilities for concrete mix datasets."""

from __future__ import annotations

import importlib
from typing import Any, Iterable

import numpy as np
import pandas as pd


ENGINEERED_FEATURE_COLUMNS = [
    "water_cement_ratio",
    "water_binder_ratio",
    "water_effective_binder_ratio",
    "binder_to_water_ratio",
    "effective_binder_to_water_ratio",
    "slag_replacement_ratio",
    "fly_ash_replacement_ratio",
    "total_binder",
    "effective_binder",
    "supplementary_replacement_ratio",
    "effective_scm_replacement_ratio",
    "aggregate_paste_ratio",
    "paste_to_aggregate_ratio",
    "fine_to_coarse_ratio",
    "superplasticizer_binder_ratio",
    "superplasticizer_effective_binder_ratio",
    "paste_volume_proxy",
    "log_age",
    "cement_age_interaction",
    "binder_age_interaction",
    "effective_binder_age_interaction",
    "water_binder_age",
    "water_effective_binder_age",
]

ENGINEERING_DIAGNOSTIC_COLUMNS = [
    "effective_scm_content",
    "binder_efficiency_gap",
    "cement_fraction_of_binder",
    "cement_fraction_of_effective_binder",
    "scm_fraction_of_effective_binder",
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
_DEFAULT_BINDER_EFFICIENCY = {
    "fly_ash_k": 0.35,
    "slag_k": 0.80,
}


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


def resolve_feature_engineering_options(config: dict[str, Any] | None = None) -> dict[str, float]:
    """Resolve semi-empirical SCM efficiency coefficients from config."""
    options = dict(_DEFAULT_BINDER_EFFICIENCY)
    if not config:
        return options

    engineering = config.get("engineering", {})
    binder_efficiency = engineering.get("binder_efficiency", {})
    if not isinstance(binder_efficiency, dict):
        return options

    for key in options:
        raw_value = binder_efficiency.get(key)
        if raw_value is None:
            continue
        options[key] = float(raw_value)
    return options


def _load_promoted_feature_entries(config: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    if not config:
        return []
    engineering = config.get("engineering", {})
    module_names = list(engineering.get("experimental_feature_modules", []))
    if not module_names:
        module_names = ["experimental_features"]
    loaded_entries: list[dict[str, Any]] = []
    for module_name in module_names:
        try:
            module = importlib.import_module(str(module_name))
        except Exception:
            continue
        entries = getattr(module, "PROMOTED_EXPERIMENTAL_FEATURES", [])
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            function = entry.get("function")
            name = str(entry.get("name", "")).strip()
            if not name or not callable(function):
                continue
            loaded_entries.append({"name": name, "function": function, "source": entry.get("source", "unknown")})
    return loaded_entries


def build_engineering_features(
    df: pd.DataFrame,
    config: dict[str, Any] | None = None,
) -> pd.DataFrame:
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
    options = resolve_feature_engineering_options(config)
    slag_k = float(options["slag_k"])
    fly_ash_k = float(options["fly_ash_k"])

    total_binder = cement + slag + fly_ash
    effective_scm_content = (slag * slag_k) + (fly_ash * fly_ash_k)
    effective_binder = cement + effective_scm_content
    binder_plus_water = total_binder + water
    total_aggregate = coarse_aggregate + fine_aggregate
    log_age = np.log(np.maximum(age, 0.0) + 1.0)

    frame["water_cement_ratio"] = _safe_divide(water, cement)
    frame["water_binder_ratio"] = _safe_divide(water, total_binder)
    frame["water_effective_binder_ratio"] = _safe_divide(water, effective_binder)
    frame["binder_to_water_ratio"] = _safe_divide(total_binder, water)
    frame["effective_binder_to_water_ratio"] = _safe_divide(effective_binder, water)
    frame["slag_replacement_ratio"] = _safe_divide(slag, total_binder)
    frame["fly_ash_replacement_ratio"] = _safe_divide(fly_ash, total_binder)
    frame["aggregate_paste_ratio"] = _safe_divide(total_aggregate, binder_plus_water)
    frame["paste_to_aggregate_ratio"] = _safe_divide(binder_plus_water, total_aggregate)
    frame["fine_to_coarse_ratio"] = _safe_divide(fine_aggregate, coarse_aggregate)
    frame["total_binder"] = total_binder
    frame["effective_binder"] = effective_binder
    frame["supplementary_replacement_ratio"] = _safe_divide(slag + fly_ash, total_binder)
    frame["effective_scm_replacement_ratio"] = _safe_divide(effective_scm_content, effective_binder)
    frame["superplasticizer_binder_ratio"] = _safe_divide(superplasticizer, total_binder)
    frame["superplasticizer_effective_binder_ratio"] = _safe_divide(
        superplasticizer,
        effective_binder,
    )
    frame["paste_volume_proxy"] = total_binder + water + superplasticizer
    frame["log_age"] = log_age
    frame["cement_age_interaction"] = cement * log_age
    frame["binder_age_interaction"] = total_binder * log_age
    frame["effective_binder_age_interaction"] = effective_binder * log_age
    frame["water_binder_age"] = _safe_divide(frame["water_binder_ratio"], log_age)
    frame["water_effective_binder_age"] = _safe_divide(frame["water_effective_binder_ratio"], log_age)
    frame["effective_scm_content"] = effective_scm_content
    frame["binder_efficiency_gap"] = total_binder - effective_binder
    frame["cement_fraction_of_binder"] = _safe_divide(cement, total_binder)
    frame["cement_fraction_of_effective_binder"] = _safe_divide(cement, effective_binder)
    frame["scm_fraction_of_effective_binder"] = _safe_divide(effective_scm_content, effective_binder)
    for entry in _load_promoted_feature_entries(config):
        try:
            series = entry["function"](frame.copy())
            if not isinstance(series, pd.Series):
                series = pd.Series(series, index=frame.index)
            frame[entry["name"]] = pd.to_numeric(series, errors="coerce").fillna(0.0)
        except Exception:
            continue
    return frame


def validate_features(df: pd.DataFrame) -> bool:
    """Validate engineered features and print a compact summary."""
    expected_columns = ENGINEERED_FEATURE_COLUMNS + ENGINEERING_DIAGNOSTIC_COLUMNS
    missing_columns = _missing_columns(expected_columns, df)
    if missing_columns:
        log_status(f"Feature validation failed. Missing engineered columns: {missing_columns}")
        return False

    feature_frame = df[expected_columns].apply(pd.to_numeric, errors="coerce")
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
