"""Track field validation results for generated mix-design artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4


BASE_DIR = Path(__file__).resolve().parent
OUTPUTS_DIR = BASE_DIR / "outputs"
FIELD_LOG_PATH = OUTPUTS_DIR / "field_validation_log.json"
TOLERANCE_MPA = 2.0


def log_status(message: str) -> None:
    """Print a timestamped status message."""
    timestamp = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")


def resolve_path(raw_path: str | Path) -> Path:
    """Resolve a user-supplied path relative to the project root."""
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return BASE_DIR / path


def relative_to_project(path: Path) -> str:
    """Render a path relative to the project root when possible."""
    try:
        return str(path.resolve().relative_to(BASE_DIR.resolve()))
    except ValueError:
        return str(path.resolve())


def load_json_file(path: Path) -> Any:
    """Load JSON from disk."""
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json_file(path: Path, payload: Any) -> None:
    """Persist JSON with stable formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def load_field_validation_records() -> list[dict[str, Any]]:
    """Load the append-only field validation log."""
    if not FIELD_LOG_PATH.exists():
        return []
    payload = load_json_file(FIELD_LOG_PATH)
    if not isinstance(payload, list):
        raise ValueError(f"Field validation log must contain a list of records: {FIELD_LOG_PATH}")
    normalized_records: list[dict[str, Any]] = []
    for index, item in enumerate(payload):
        if not isinstance(item, dict):
            raise ValueError(f"Field validation record at index {index} is not a JSON object.")
        normalized_records.append(dict(item))
    return normalized_records


def load_design_artifact(design_path: str | Path) -> tuple[Path, dict[str, Any]]:
    """Load a saved design artifact from disk."""
    resolved_path = resolve_path(design_path)
    if not resolved_path.exists():
        raise FileNotFoundError(f"Design artifact not found: {resolved_path}")
    payload = load_json_file(resolved_path)
    if not isinstance(payload, dict):
        raise ValueError(f"Design artifact must contain a JSON object: {resolved_path}")
    required_keys = ["target_strength", "predicted_strength", "mix_design", "validation_verdict"]
    missing_keys = [key for key in required_keys if key not in payload]
    if missing_keys:
        raise ValueError(f"Design artifact is missing required keys {missing_keys}: {resolved_path}")
    if not isinstance(payload.get("mix_design"), dict):
        raise ValueError(f"Design artifact field 'mix_design' must be an object: {resolved_path}")
    return resolved_path, payload


def build_field_record(design_path: str | Path, actual_strength: float, notes: str) -> dict[str, Any]:
    """Build one field validation record from a saved design artifact."""
    resolved_design_path, design_payload = load_design_artifact(design_path)
    predicted_strength = float(design_payload["predicted_strength"])
    target_strength = float(design_payload["target_strength"])
    actual_strength = float(actual_strength)
    prediction_error = actual_strength - predicted_strength
    return {
        "record_id": str(uuid4()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "design_file": relative_to_project(resolved_design_path),
        "target_strength": target_strength,
        "predicted_strength": predicted_strength,
        "actual_strength": actual_strength,
        "prediction_error_mpa": prediction_error,
        "within_tolerance": abs(prediction_error) <= TOLERANCE_MPA,
        "mix_design": dict(design_payload["mix_design"]),
        "validation_verdict": str(design_payload["validation_verdict"]),
        "notes": str(notes),
    }


def append_field_record(record: dict[str, Any]) -> None:
    """Append one record to the append-only field validation log."""
    records = load_field_validation_records()
    records.append(record)
    save_json_file(FIELD_LOG_PATH, records)


def compute_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute aggregate error statistics for recorded field tests."""
    if not records:
        return {
            "total_tests": 0,
            "mean_prediction_error_mpa": None,
            "rmse_mpa": None,
            "within_tolerance_pct": None,
            "best_record": None,
            "worst_record": None,
        }

    errors = [float(record["prediction_error_mpa"]) for record in records]
    squared_errors = [error ** 2 for error in errors]
    within_tolerance_count = sum(1 for record in records if bool(record["within_tolerance"]))
    sorted_by_absolute_error = sorted(records, key=lambda item: abs(float(item["prediction_error_mpa"])))
    return {
        "total_tests": len(records),
        "mean_prediction_error_mpa": sum(errors) / len(errors),
        "rmse_mpa": math.sqrt(sum(squared_errors) / len(squared_errors)),
        "within_tolerance_pct": (within_tolerance_count / len(records)) * 100.0,
        "best_record": sorted_by_absolute_error[0],
        "worst_record": sorted_by_absolute_error[-1],
    }


def print_summary(records: list[dict[str, Any]]) -> None:
    """Print a human-readable field validation summary."""
    summary = compute_summary(records)
    print(f"Total tests recorded: {summary['total_tests']}")
    if summary["total_tests"] == 0:
        print("No field validation records found.")
        return

    print(f"Mean prediction error (bias): {summary['mean_prediction_error_mpa']:.3f} MPa")
    print(f"RMSE between predicted and actual: {summary['rmse_mpa']:.3f} MPa")
    print(f"Within +/- {TOLERANCE_MPA:.1f} MPa tolerance: {summary['within_tolerance_pct']:.1f}%")

    best_record = summary["best_record"]
    worst_record = summary["worst_record"]
    print(
        "Best performing mix design: "
        f"{best_record['design_file']} | error={best_record['prediction_error_mpa']:.3f} MPa | "
        f"actual={best_record['actual_strength']:.3f} MPa"
    )
    print(
        "Worst performing mix design: "
        f"{worst_record['design_file']} | error={worst_record['prediction_error_mpa']:.3f} MPa | "
        f"actual={worst_record['actual_strength']:.3f} MPa"
    )


def export_records(records: list[dict[str, Any]], output_path: str | Path) -> Path:
    """Export the append-only field validation log to a CSV report."""
    resolved_output_path = resolve_path(output_path)
    resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "record_id",
        "timestamp",
        "design_file",
        "target_strength",
        "predicted_strength",
        "actual_strength",
        "prediction_error_mpa",
        "within_tolerance",
        "mix_design",
        "validation_verdict",
        "notes",
    ]
    with resolved_output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            export_row = dict(record)
            export_row["mix_design"] = json.dumps(record.get("mix_design", {}), ensure_ascii=True, sort_keys=True)
            writer.writerow(export_row)
    return resolved_output_path


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line parser."""
    parser = argparse.ArgumentParser(description="Track field validation against generated mix designs.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    record_parser = subparsers.add_parser("record", help="Append one lab-test result to the field validation log.")
    record_parser.add_argument("--design", required=True, help="Path to the saved design JSON artifact.")
    record_parser.add_argument("--actual-strength", required=True, type=float, help="Measured compressive strength in MPa.")
    record_parser.add_argument("--notes", default="", help="Free-form field or lab notes for this record.")

    subparsers.add_parser("summary", help="Print aggregate validation statistics.")

    export_parser = subparsers.add_parser("export", help="Export the field validation log to CSV.")
    export_parser.add_argument("--output", required=True, help="Destination CSV path.")
    return parser


def main() -> int:
    """Run the field validation tracker CLI."""
    parser = build_parser()
    args = parser.parse_args()

    try:
        if args.command == "record":
            record = build_field_record(args.design, args.actual_strength, args.notes)
            append_field_record(record)
            log_status(
                "Recorded field validation "
                f"| record_id={record['record_id']} "
                f"| design={record['design_file']} "
                f"| error={record['prediction_error_mpa']:.3f} MPa "
                f"| within_tolerance={record['within_tolerance']}"
            )
            return 0

        records = load_field_validation_records()
        if args.command == "summary":
            print_summary(records)
            return 0

        if args.command == "export":
            export_path = export_records(records, args.output)
            log_status(f"Exported {len(records)} field validation records to {export_path}")
            return 0

        parser.error(f"Unsupported command: {args.command}")
        return 2
    except Exception as exc:
        log_status(f"Field validation tracker failed: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
