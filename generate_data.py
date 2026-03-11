"""Download or synthesize the concrete compressive strength dataset."""

from __future__ import annotations

import io
import random
import sys
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
import yaml

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


def load_local_dataset(config: dict[str, Any]) -> pd.DataFrame:
    """Load a local CSV or Excel dataset and normalize it to the project schema."""
    local_settings = config["data"]["local_file"]
    local_path = resolve_local_data_path(config)
    if not local_path.exists():
        raise FileNotFoundError(f"Configured local dataset was not found: {local_path}")

    suffix = local_path.suffix.lower()
    if suffix == ".csv":
        frame = pd.read_csv(local_path)
    elif suffix in {".xlsx", ".xls"}:
        sheet_name = local_settings.get("sheet_name", 0)
        engine = "openpyxl" if suffix == ".xlsx" else "xlrd"
        frame = pd.read_excel(local_path, sheet_name=sheet_name, engine=engine)
    else:
        raise ValueError(f"Unsupported local dataset format: {local_path.suffix}")

    frame = frame.copy()
    drop_columns = [str(column) for column in local_settings.get("drop_columns", [])]
    existing_drop_columns = [column for column in drop_columns if column in frame.columns]
    if existing_drop_columns:
        frame = frame.drop(columns=existing_drop_columns)

    column_mapping = {str(key): str(value) for key, value in local_settings.get("column_mapping", {}).items()}
    if column_mapping:
        frame = frame.rename(columns=column_mapping)

    expected_columns = config["task"]["input_columns"] + [config["task"]["target_column"]]
    missing_columns = [column for column in expected_columns if column not in frame.columns]
    if missing_columns:
        raise ValueError(f"Local dataset is missing required columns after mapping: {missing_columns}")

    normalized = frame[expected_columns].copy()
    normalized = normalized.dropna(axis=0, how="any")
    return normalized


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


def save_dataset(frame: pd.DataFrame, config: dict[str, Any]) -> Path:
    """Persist the prepared dataset to disk."""
    dataset_path = get_dataset_path(config)
    frame.to_csv(dataset_path, index=False)
    return dataset_path


def main() -> int:
    """Run the dataset preparation workflow."""
    try:
        config = load_config()
        set_global_seed(int(config["experiment"]["random_seed"]))
        mode = str(config["data"]["mode"]).strip().lower()
        if mode not in {"real", "synthetic", "local_file"}:
            raise ValueError("config.yaml data.mode must be 'real', 'synthetic', or 'local_file'.")

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

        dataset_path = save_dataset(dataset, config)
        log_status(f"Saved {data_origin} dataset to {dataset_path} with {len(dataset)} rows.")
        log_status(f"Columns: {', '.join(dataset.columns.tolist())}")
        return 0
    except Exception as exc:
        log_status(f"Dataset generation failed: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
