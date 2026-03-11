"""Download or synthesize the concrete compressive strength dataset."""

from __future__ import annotations

import io
import random
import sys
import warnings
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
import yaml
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

from feature_engineering import build_engineering_features, validate_features


def log_status(message: str) -> None:
    """Print a timestamped status message."""
    timestamp = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")


def get_project_root() -> Path:
    """Return the project root directory."""
    return Path(__file__).resolve().parent


def load_config() -> dict[str, Any]:
    """Load the YAML configuration for the project."""
    config_path = get_project_root() / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def set_global_seed(seed: int) -> None:
    """Seed Python and NumPy RNGs for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)


def get_dataset_path(config: dict[str, Any]) -> Path:
    """Resolve the dataset output path from config."""
    project_root = get_project_root()
    data_dir = project_root / config["paths"]["data_dir"]
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir / config["paths"]["dataset_filename"]


def normalize_concrete_columns(frame: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """Standardize downloaded dataset columns to the configured schema."""
    frame = frame.dropna(axis=0, how="all").dropna(axis=1, how="all")
    expected_columns = config["task"]["input_columns"] + [config["task"]["target_column"]]
    if frame.shape[1] != len(expected_columns):
        raise ValueError(
            f"Expected {len(expected_columns)} columns after loading, but found {frame.shape[1]}."
        )
    frame = frame.copy()
    frame.columns = expected_columns
    return frame


def resolve_local_data_path(config: dict[str, Any]) -> Path:
    """Resolve the configured local dataset path."""
    configured_path = Path(str(config["data"]["local_file"]["path"]))
    if configured_path.is_absolute():
        return configured_path
    return get_project_root() / configured_path


def load_and_harmonize_lab_data(config: dict[str, Any]) -> pd.DataFrame:
    """Load local lab data, harmonize headers and units, and preserve source-sheet metadata."""
    from difflib import get_close_matches

    local_settings = config["data"]["local_file"]
    local_path = resolve_local_data_path(config)
    if not local_path.exists():
        raise FileNotFoundError(f"Configured local dataset was not found: {local_path}")

    expected_inputs = [str(column) for column in config["task"]["input_columns"]]
    target_column = str(config["task"]["target_column"])
    expected_columns = expected_inputs + [target_column]
    optional_columns = {"slag", "fly_ash", "superplasticizer"}
    required_columns = [column for column in expected_columns if column not in optional_columns]

    column_mapping = {
        str(source): str(destination)
        for source, destination in local_settings.get("column_mapping", {}).items()
    }
    unit_conversions = {
        str(source): spec for source, spec in local_settings.get("unit_conversions", {}).items()
    }
    drop_columns = [str(column) for column in local_settings.get("drop_columns", [])]

    def normalize_header(value: Any) -> str:
        text = str(value).strip().lower()
        return "".join(character for character in text if character.isalnum())

    def find_matching_column(alias: str, candidates: list[str]) -> tuple[str | None, bool]:
        alias_normalized = normalize_header(alias)
        if not alias_normalized:
            return None, False

        normalized_to_columns: dict[str, list[str]] = {}
        for candidate in candidates:
            normalized_candidate = normalize_header(candidate)
            normalized_to_columns.setdefault(normalized_candidate, []).append(candidate)

        if alias_normalized in normalized_to_columns:
            return normalized_to_columns[alias_normalized][0], False

        close_matches = get_close_matches(
            alias_normalized,
            [value for value in normalized_to_columns if value],
            n=1,
            cutoff=0.72,
        )
        if not close_matches:
            return None, False
        return normalized_to_columns[close_matches[0]][0], True

    alias_specs: list[tuple[str, str, float]] = []
    processed_aliases: set[str] = set()

    for source_column, destination_column in column_mapping.items():
        conversion_spec = unit_conversions.get(source_column, {})
        multiplier = float(conversion_spec.get("multiplier", 1.0)) if isinstance(conversion_spec, dict) else 1.0
        alias_specs.append((source_column, destination_column, multiplier))
        processed_aliases.add(source_column)

    for source_column, conversion_spec in unit_conversions.items():
        if isinstance(conversion_spec, dict):
            destination_column = str(conversion_spec.get("rename", column_mapping.get(source_column, source_column)))
            multiplier = float(conversion_spec.get("multiplier", 1.0))
        else:
            destination_column = str(column_mapping.get(source_column, source_column))
            multiplier = float(conversion_spec)
        if source_column in processed_aliases:
            continue
        alias_specs.append((source_column, destination_column, multiplier))
        processed_aliases.add(source_column)

    for canonical_column in expected_columns:
        if canonical_column in processed_aliases:
            continue
        alias_specs.append((canonical_column, canonical_column, 1.0))
        processed_aliases.add(canonical_column)

    suffix = local_path.suffix.lower()
    sheet_frames: dict[str, pd.DataFrame]
    if suffix == ".csv":
        sheet_frames = {local_path.stem: pd.read_csv(local_path)}
    elif suffix in {".xlsx", ".xls"}:
        engine = "openpyxl" if suffix == ".xlsx" else "xlrd"
        workbook = pd.ExcelFile(local_path, engine=engine)
        configured_sheet = local_settings.get("sheet_name")
        if configured_sheet in {None, "", "*", "all", "ALL"}:
            sheet_names = workbook.sheet_names
        elif isinstance(configured_sheet, list):
            sheet_names = [str(sheet_name) for sheet_name in configured_sheet]
        elif isinstance(configured_sheet, int):
            sheet_names = [workbook.sheet_names[configured_sheet]]
        else:
            sheet_names = [str(configured_sheet)]
        sheet_frames = {sheet_name: workbook.parse(sheet_name) for sheet_name in sheet_names}
    else:
        raise ValueError(f"Unsupported local dataset format: {local_path.suffix}")

    harmonized_frames: list[pd.DataFrame] = []
    for sheet_name, raw_frame in sheet_frames.items():
        frame = raw_frame.dropna(axis=0, how="all").dropna(axis=1, how="all").copy()
        if frame.empty:
            log_status(f"WARNING local_sheet_empty | sheet={sheet_name} | skipping")
            continue

        matched_drop_columns: list[str] = []
        available_columns = [str(column) for column in frame.columns]
        for drop_column in drop_columns:
            matched_column, _ = find_matching_column(drop_column, available_columns)
            if matched_column is not None and matched_column not in matched_drop_columns:
                matched_drop_columns.append(matched_column)
        if matched_drop_columns:
            frame = frame.drop(columns=matched_drop_columns)

        harmonized_frame = pd.DataFrame(index=frame.index)
        harmonized_frame["source_sheet"] = str(sheet_name)
        matched_source_columns: set[str] = set()

        for source_alias, canonical_column, multiplier in alias_specs:
            candidate_columns = [
                str(column)
                for column in frame.columns
                if str(column) not in matched_source_columns
            ]
            matched_column, used_fuzzy_match = find_matching_column(source_alias, candidate_columns)
            if matched_column is None:
                continue

            numeric_series = pd.to_numeric(frame[matched_column], errors="coerce") * multiplier
            if canonical_column in harmonized_frame.columns:
                harmonized_frame[canonical_column] = harmonized_frame[canonical_column].combine_first(
                    numeric_series
                )
            else:
                harmonized_frame[canonical_column] = numeric_series

            matched_source_columns.add(matched_column)
            if used_fuzzy_match:
                log_status(
                    f"Fuzzy-mapped local column '{matched_column}' to '{canonical_column}' using alias '{source_alias}'."
                )

        harmonized_frames.append(harmonized_frame)

    if not harmonized_frames:
        raise ValueError(f"No usable rows were found in local dataset: {local_path}")

    harmonized = pd.concat(harmonized_frames, ignore_index=True, sort=False)

    for column_name in optional_columns:
        if column_name not in harmonized.columns:
            harmonized[column_name] = 0.0
            log_status(
                f"WARNING optional_column_imputed | column={column_name} | rows={len(harmonized)} | fill_value=0.0"
            )
            continue
        missing_count = int(harmonized[column_name].isna().sum())
        if missing_count > 0:
            harmonized[column_name] = harmonized[column_name].fillna(0.0)
            log_status(
                f"WARNING optional_column_imputed | column={column_name} | rows={missing_count} | fill_value=0.0"
            )

    missing_required_columns = [column for column in required_columns if column not in harmonized.columns]
    if missing_required_columns:
        raise ValueError(
            "Local lab dataset is missing required columns after harmonization: "
            f"{missing_required_columns}"
        )

    selected_columns = ["source_sheet"] + expected_columns
    harmonized = harmonized[selected_columns].copy()
    numeric_values = harmonized[expected_columns].apply(pd.to_numeric, errors="coerce")
    invalid_row_mask = numeric_values.isnull().any(axis=1)
    if invalid_row_mask.any():
        dropped_rows = int(invalid_row_mask.sum())
        harmonized = harmonized.loc[~invalid_row_mask].copy()
        numeric_values = numeric_values.loc[~invalid_row_mask].copy()
        log_status(
            f"WARNING dropped_invalid_local_rows | rows={dropped_rows} | reason=missing_or_non_numeric_values"
        )
    harmonized[expected_columns] = numeric_values.astype(float)

    initial_row_count = len(harmonized)
    harmonized = harmonized.drop_duplicates(subset=expected_inputs, keep="first").reset_index(drop=True)
    duplicate_count = initial_row_count - len(harmonized)
    if duplicate_count > 0:
        log_status(f"Removed {duplicate_count} duplicate local lab rows based on configured input columns.")

    log_status(
        f"Harmonized local lab dataset from {local_path} with {len(harmonized)} rows across "
        f"{harmonized['source_sheet'].nunique()} sheet(s)."
    )
    return harmonized


def load_local_dataset(config: dict[str, Any]) -> pd.DataFrame:
    """Load a local CSV or Excel dataset and normalize it to the project schema."""
    harmonized = load_and_harmonize_lab_data(config)
    expected_columns = config["task"]["input_columns"] + [config["task"]["target_column"]]
    return harmonized[expected_columns].copy()


def load_from_download_bytes(raw_bytes: bytes, config: dict[str, Any], source_url: str) -> pd.DataFrame:
    """Load a concrete dataset from downloaded bytes."""
    byte_stream = io.BytesIO(raw_bytes)
    if zipfile.is_zipfile(byte_stream):
        with zipfile.ZipFile(byte_stream) as archive:
            members = archive.namelist()
            candidate_name = next(
                (name for name in members if name.lower().endswith((".xls", ".xlsx", ".csv"))),
                None,
            )
            if candidate_name is None:
                raise ValueError(f"No tabular dataset file found inside archive from {source_url}.")
            extracted_bytes = archive.read(candidate_name)
            return load_from_download_bytes(extracted_bytes, config, candidate_name)

    lower_source = source_url.lower()
    if lower_source.endswith(".csv"):
        frame = pd.read_csv(io.BytesIO(raw_bytes))
    elif lower_source.endswith(".xlsx"):
        frame = pd.read_excel(io.BytesIO(raw_bytes))
    else:
        frame = pd.read_excel(io.BytesIO(raw_bytes), engine="xlrd")
    return normalize_concrete_columns(frame, config)


def download_real_dataset(config: dict[str, Any]) -> pd.DataFrame:
    """Download the real UCI dataset using the configured source URLs."""
    errors: list[str] = []
    for source_url in config["data"]["real_data_sources"]:
        try:
            log_status(f"Downloading real dataset from {source_url}")
            response = requests.get(source_url, timeout=30)
            response.raise_for_status()
            frame = load_from_download_bytes(response.content, config, source_url)
            log_status(f"Downloaded and parsed dataset from {source_url}")
            return frame
        except Exception as exc:
            error_message = f"{source_url} -> {exc}"
            errors.append(error_message)
            log_status(f"Download source failed: {error_message}")
    error_text = "\n".join(errors)
    raise RuntimeError(f"Unable to download the real dataset.\n{error_text}")


def generate_synthetic_dataset(config: dict[str, Any]) -> pd.DataFrame:
    """Generate a synthetic concrete dataset with realistic engineering relationships."""
    seed = int(config["experiment"]["random_seed"])
    rng = np.random.default_rng(seed)
    rows = int(config["data"]["synthetic_rows"])

    cement = rng.uniform(120.0, 540.0, rows)
    slag = rng.uniform(0.0, 280.0, rows)
    fly_ash = rng.uniform(0.0, 220.0, rows)
    water = rng.uniform(120.0, 250.0, rows)
    superplasticizer = rng.uniform(0.0, 32.0, rows)
    coarse_aggregate = rng.uniform(800.0, 1145.0, rows)
    fine_aggregate = rng.uniform(500.0, 990.0, rows)
    age = rng.choice(
        [1, 3, 7, 14, 28, 56, 90, 180, 365],
        size=rows,
        p=[0.06, 0.08, 0.12, 0.12, 0.28, 0.14, 0.1, 0.05, 0.05],
    )

    effective_binder = cement + 0.62 * slag + 0.38 * fly_ash
    water_binder_ratio = water / np.maximum(cement + 0.45 * slag + 0.3 * fly_ash, 80.0)
    age_factor = np.log1p(age) / np.log(366.0)
    aggregate_penalty = np.abs((coarse_aggregate + fine_aggregate) - 1750.0)

    strength = (
        7.0
        + 0.072 * effective_binder
        - 31.0 * water_binder_ratio
        + 19.0 * age_factor
        + 0.42 * superplasticizer
        + 0.018 * cement * age_factor
        - 0.006 * aggregate_penalty
        - 0.035 * np.maximum(water - 205.0, 0.0)
    )
    noise = rng.normal(0.0, 4.5, rows)
    compressive_strength = np.clip(strength + noise, 2.0, 110.0)

    frame = pd.DataFrame(
        {
            "cement": cement,
            "slag": slag,
            "fly_ash": fly_ash,
            "water": water,
            "superplasticizer": superplasticizer,
            "coarse_aggregate": coarse_aggregate,
            "fine_aggregate": fine_aggregate,
            "age": age.astype(float),
            "compressive_strength": compressive_strength,
        }
    )
    return frame.round(4)


# OUTLIER DETECTION
def get_outlier_detection_settings(config: dict[str, Any]) -> tuple[bool, float]:
    """Return validated outlier-detection settings with safe defaults."""
    outlier_config = config.get("data", {}).get("outlier_detection", {})
    enabled = bool(outlier_config.get("enabled", True))
    contamination = float(outlier_config.get("contamination", 0.03))
    if not 0.0 < contamination <= 0.5:
        raise ValueError("data.outlier_detection.contamination must be in the range (0.0, 0.5].")
    return enabled, contamination


# OUTLIER DETECTION
def detect_consensus_outliers(frame: pd.DataFrame, config: dict[str, Any]) -> pd.Series:
    """Identify conservative outliers where IsolationForest and LOF agree."""
    _, contamination = get_outlier_detection_settings(config)
    if len(frame) < 3:
        log_status("Skipping outlier detection because the dataset has fewer than 3 rows.")
        return pd.Series(False, index=frame.index, dtype=bool)

    numeric_frame = frame.apply(pd.to_numeric, errors="coerce")
    if numeric_frame.isnull().any().any():
        invalid_columns = numeric_frame.columns[numeric_frame.isnull().any()].tolist()
        raise ValueError(
            "Outlier detection requires a fully numeric dataset after feature engineering. "
            f"Invalid columns: {invalid_columns}"
        )

    scaled_values = StandardScaler().fit_transform(numeric_frame.to_numpy(dtype=float))
    random_seed = int(config["experiment"]["random_seed"])
    primary_detector = IsolationForest(
        contamination=contamination,
        random_state=random_seed,
    )
    secondary_detector = LocalOutlierFactor(
        contamination=contamination,
        n_neighbors=max(2, min(20, len(frame) - 1)),
    )

    primary_flags = primary_detector.fit_predict(scaled_values) == -1
    secondary_flags = secondary_detector.fit_predict(scaled_values) == -1
    consensus_flags = primary_flags & secondary_flags

    log_status(
        "Outlier detection summary: "
        f"IsolationForest={int(primary_flags.sum())}, "
        f"LOF={int(secondary_flags.sum())}, "
        f"consensus={int(consensus_flags.sum())}"
    )
    return pd.Series(consensus_flags, index=frame.index, dtype=bool)


# OUTLIER DETECTION
def apply_outlier_detection(frame: pd.DataFrame, config: dict[str, Any], mode: str) -> pd.DataFrame:
    """Remove consensus outliers unless the active mode is local_file."""
    enabled, _ = get_outlier_detection_settings(config)
    if not enabled:
        log_status("Outlier detection disabled in config; saving dataset without removal.")
        return frame

    consensus_flags = detect_consensus_outliers(frame, config)
    flagged_count = int(consensus_flags.sum())
    if flagged_count == 0:
        log_status("Outlier detection found no consensus outliers.")
        return frame

    target_column = str(config["task"]["target_column"])
    strength_values = frame.loc[consensus_flags, target_column].round(4).tolist()

    if mode == "local_file":
        warning_message = (
            f"Outlier detection flagged {flagged_count} rows in local_file mode but kept them. "
            f"{target_column} values: {strength_values}"
        )
        warnings.warn(warning_message, UserWarning, stacklevel=2)
        log_status(warning_message)
        return frame

    log_status(
        f"Removing {flagged_count} consensus outliers before saving. "
        f"{target_column} values: {strength_values}"
    )
    return frame.loc[~consensus_flags].reset_index(drop=True)


def save_dataset(frame: pd.DataFrame, config: dict[str, Any]) -> Path:
    """Persist the prepared dataset to disk."""
    dataset_path = get_dataset_path(config)
    frame.to_csv(dataset_path, index=False)
    return dataset_path


def main() -> int:
    """Run the dataset preparation workflow."""
    try:
        import argparse

        parser = argparse.ArgumentParser(description="Prepare the concrete dataset for AutoCivil-Lab.")
        parser.add_argument(
            "--validate-only",
            action="store_true",
            help="Load and harmonize local lab data, print a validation summary, and exit without saving.",
        )
        args = parser.parse_args()

        config = load_config()
        set_global_seed(int(config["experiment"]["random_seed"]))
        mode = str(config["data"]["mode"]).strip().lower()
        if mode not in {"real", "synthetic", "local_file"}:
            raise ValueError("config.yaml data.mode must be 'real', 'synthetic', or 'local_file'.")

        if args.validate_only:
            if mode != "local_file":
                raise ValueError("--validate-only is only supported when config.yaml data.mode is 'local_file'.")
            harmonized = load_and_harmonize_lab_data(config)
            target_column = str(config["task"]["target_column"])
            input_columns = [str(column) for column in config["task"]["input_columns"]]
            log_status(
                f"Validation summary | rows={len(harmonized)} | sheets={harmonized['source_sheet'].nunique()} | "
                f"target_mean={harmonized[target_column].mean():.3f}"
            )
            log_status(f"Validated schema: {', '.join(['source_sheet'] + input_columns + [target_column])}")
            sheet_counts = harmonized["source_sheet"].value_counts().sort_index()
            for sheet_name, row_count in sheet_counts.items():
                log_status(f"Sheet {sheet_name}: {int(row_count)} rows")
            return 0

        if mode == "local_file":
            dataset = load_local_dataset(config)
            data_origin = "local_file"
        elif mode == "real":
            try:
                dataset = download_real_dataset(config)
                data_origin = "real"
            except Exception as exc:
                if config["data"].get("fallback_to_synthetic_on_error", False):
                    log_status(f"Real-data download failed. Falling back to synthetic mode. Reason: {exc}")
                    dataset = generate_synthetic_dataset(config)
                    data_origin = "synthetic_fallback"
                else:
                    raise
        else:
            dataset = generate_synthetic_dataset(config)
            data_origin = "synthetic"

        if bool(config.get("engineering", {}).get("feature_engineering", False)):
            log_status("Applying engineering feature generation to normalized dataset.")
            dataset = build_engineering_features(dataset)
            if not validate_features(dataset):
                raise ValueError("Engineered feature validation failed during data generation.")

        # OUTLIER DETECTION
        dataset = apply_outlier_detection(dataset, config, mode)

        dataset_path = save_dataset(dataset, config)
        log_status(f"Saved {data_origin} dataset to {dataset_path} with {len(dataset)} rows.")
        log_status(f"Columns: {', '.join(dataset.columns.tolist())}")
        return 0
    except Exception as exc:
        log_status(f"Dataset generation failed: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
