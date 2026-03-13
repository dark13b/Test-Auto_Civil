"""Governance helpers for the autoresearch-style AutoCivil-Lab loop."""

from __future__ import annotations

import ast
import copy
import json
import pprint
import re
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


RESEARCH_SURFACE_STATE_START = "# RESEARCH_SURFACE_STATE_START"
RESEARCH_SURFACE_STATE_END = "# RESEARCH_SURFACE_STATE_END"

DEFAULT_BRIEF = {
    "goal": "Maximize composite_score under validator-safe constraints.",
    "acceptance_metric": "composite_score",
    "min_improvement_pct": 0.0,
    "required_model_families": [],
    "focus_areas": [],
    "scout_candidates_per_cycle": 8,
    "confirm_top_k": 2,
}

RESEARCH_RESULTS_COLUMNS = [
    "trial_number",
    "experiment_id",
    "stage",
    "model_name",
    "display_name",
    "proposal_family",
    "hypothesis",
    "hyperparameters",
    "selection_status",
    "validation_verdict",
    "error_message",
    "rmse",
    "mae",
    "r2",
    "composite_score",
    "test_rmse",
    "test_mae",
    "test_r2",
    "test_composite_score",
    "validation_pass_rate",
    "failed_count",
    "hard_failed_count",
    "warning_count",
    "suspicious_count",
    "durability_caution_count",
    "dataset_anomaly_count",
    "trial_runtime_seconds",
    "budget_status",
    "scout_improvement_pct",
    "confirm_improvement_pct",
]


def _to_float(value: Any, fallback: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(fallback)


def _to_int(value: Any, fallback: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(fallback)


def _to_serializable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _to_serializable(inner) for key, inner in value.items()}
    if isinstance(value, list):
        return [_to_serializable(item) for item in value]
    if isinstance(value, tuple):
        return [_to_serializable(item) for item in value]
    return value


def load_json_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Required JSON artifact not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON artifact must contain an object: {path}")
    return payload


def write_json_file(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(_to_serializable(payload), handle, indent=2, sort_keys=True)


def build_config_signature(model_name: str, params: dict[str, Any]) -> tuple[str, str]:
    """Build a stable signature for one model configuration."""
    canonical = json.dumps(_to_serializable(params), sort_keys=True, separators=(",", ":"))
    return model_name, canonical


def _extract_front_matter(markdown_text: str) -> dict[str, Any]:
    stripped = markdown_text.strip()
    if not stripped.startswith("---"):
        return {}

    lines = stripped.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}

    closing_index = None
    for index in range(1, len(lines)):
        if lines[index].strip() == "---":
            closing_index = index
            break
    if closing_index is None:
        return {}

    yaml_text = "\n".join(lines[1:closing_index]).strip()
    if not yaml_text:
        return {}
    try:
        parsed = yaml.safe_load(yaml_text)
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def normalize_brief(payload: dict[str, Any] | None) -> dict[str, Any]:
    """Normalize a parsed brief payload to the stable internal schema."""
    normalized = copy.deepcopy(DEFAULT_BRIEF)
    if isinstance(payload, dict):
        normalized.update(payload)

    def _normalize_string_list(raw_value: Any) -> list[str]:
        if isinstance(raw_value, str):
            return [raw_value] if raw_value.strip() else []
        if isinstance(raw_value, list):
            return [str(item) for item in raw_value if str(item).strip()]
        return []

    return {
        "goal": str(normalized.get("goal", DEFAULT_BRIEF["goal"])),
        "acceptance_metric": str(normalized.get("acceptance_metric", DEFAULT_BRIEF["acceptance_metric"])),
        "min_improvement_pct": _to_float(
            normalized.get("min_improvement_pct"),
            DEFAULT_BRIEF["min_improvement_pct"],
        ),
        "required_model_families": _normalize_string_list(normalized.get("required_model_families")),
        "focus_areas": _normalize_string_list(normalized.get("focus_areas")),
        "scout_candidates_per_cycle": _to_int(
            normalized.get("scout_candidates_per_cycle"),
            DEFAULT_BRIEF["scout_candidates_per_cycle"],
        ),
        "confirm_top_k": _to_int(
            normalized.get("confirm_top_k"),
            DEFAULT_BRIEF["confirm_top_k"],
        ),
    }


def load_human_research_brief(path: Path) -> dict[str, Any]:
    """Load a human research brief from Markdown front matter."""
    if not path.exists():
        return copy.deepcopy(DEFAULT_BRIEF)
    return normalize_brief(_extract_front_matter(path.read_text(encoding="utf-8")))


def _empty_memory() -> dict[str, Any]:
    return {
        "schema_version": 2,
        "accepted_experiments": [],
        "runs": [],
    }


def load_or_initialize_experiment_memory(memory_path: Path) -> dict[str, Any]:
    """Load the structured experiment-memory payload, creating it when absent."""
    if not memory_path.exists():
        payload = _empty_memory()
        write_json_file(memory_path, payload)
        return payload

    try:
        payload = load_json_file(memory_path)
    except Exception:
        payload = _empty_memory()

    if "runs" not in payload or not isinstance(payload.get("runs"), list):
        payload["runs"] = []
    if "accepted_experiments" not in payload or not isinstance(payload.get("accepted_experiments"), list):
        payload["accepted_experiments"] = []
    if "schema_version" not in payload:
        payload["schema_version"] = 2
    return payload


def _extract_trial_params(trial_record: dict[str, Any]) -> dict[str, Any]:
    raw_params = trial_record.get("params", trial_record.get("hyperparameters", {}))
    if isinstance(raw_params, dict):
        return raw_params
    if isinstance(raw_params, str):
        stripped = raw_params.strip()
        if not stripped:
            return {}
        try:
            parsed = json.loads(stripped)
        except Exception:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def record_experiment_memory(memory_path: Path, run_id: str, trial_record: dict[str, Any]) -> None:
    """Append one research trial record into persistent experiment memory."""
    payload = load_or_initialize_experiment_memory(memory_path)
    run_payload = None
    for candidate in payload["runs"]:
        if str(candidate.get("run_id")) == run_id:
            run_payload = candidate
            break
    if run_payload is None:
        run_payload = {
            "run_id": run_id,
            "started_at": pd.Timestamp.now().isoformat(),
            "trials": [],
        }
        payload["runs"].append(run_payload)

    params = _extract_trial_params(trial_record)
    model_name = str(trial_record.get("model_name", ""))
    signature = build_config_signature(model_name, params)
    run_payload["trials"].append(
        {
            "trial_number": trial_record.get("trial_number"),
            "experiment_id": trial_record.get("experiment_id"),
            "stage": trial_record.get("stage"),
            "proposal_family": trial_record.get("proposal_family"),
            "model_name": model_name,
            "params": params,
            "selection_status": trial_record.get("selection_status"),
            "validation_verdict": trial_record.get("validation_verdict"),
            "composite_score": trial_record.get("composite_score"),
            "scout_improvement_pct": trial_record.get("scout_improvement_pct"),
            "confirm_improvement_pct": trial_record.get("confirm_improvement_pct"),
            "signature": [signature[0], signature[1]],
        }
    )
    if str(trial_record.get("selection_status")) == "kept":
        payload["accepted_experiments"].append(
            {
                "experiment_id": trial_record.get("experiment_id"),
                "proposal_family": trial_record.get("proposal_family"),
                "model_name": model_name,
                "composite_score": trial_record.get("composite_score"),
            }
        )
    write_json_file(memory_path, payload)


def should_skip_duplicate_proposal(
    *,
    model_name: str,
    params: dict[str, Any],
    current_run_signatures: set[tuple[str, str]],
    memory_payload: dict[str, Any],
) -> bool:
    """Return whether a proposal duplicates any current-run or historic signature."""
    signature = build_config_signature(model_name, params)
    if signature in current_run_signatures:
        return True

    for run in memory_payload.get("runs", []):
        for trial in run.get("trials", []):
            raw_signature = trial.get("signature")
            if (
                isinstance(raw_signature, list)
                and len(raw_signature) == 2
                and str(raw_signature[0]) == signature[0]
                and str(raw_signature[1]) == signature[1]
            ):
                return True
    return False


def filter_diverse_candidates(
    *,
    candidates: list[dict[str, Any]],
    memory_payload: dict[str, Any],
    current_run_signatures: set[tuple[str, str]],
    family_limit: int,
    scout_limit: int,
) -> list[dict[str, Any]]:
    """Filter scout candidates to preserve signature uniqueness and family diversity."""
    selected: list[dict[str, Any]] = []
    family_counts: dict[str, int] = {}
    effective_family_limit = max(1, int(family_limit))
    effective_scout_limit = max(1, int(scout_limit))

    for candidate in candidates:
        model_name = str(candidate.get("model_name", ""))
        params = dict(candidate.get("params", {}))
        if should_skip_duplicate_proposal(
            model_name=model_name,
            params=params,
            current_run_signatures=current_run_signatures,
            memory_payload=memory_payload,
        ):
            continue
        proposal_family = str(candidate.get("proposal_family", "unknown"))
        if family_counts.get(proposal_family, 0) >= effective_family_limit:
            continue
        signature = build_config_signature(model_name, params)
        current_run_signatures.add(signature)
        family_counts[proposal_family] = family_counts.get(proposal_family, 0) + 1
        selected.append(candidate)
        if len(selected) >= effective_scout_limit:
            break
    return selected


def trial_budget_status(*, elapsed_seconds: float, max_trial_seconds: float) -> str:
    """Return the budget status for a trial runtime."""
    max_seconds = float(max_trial_seconds)
    if max_seconds <= 0.0:
        return "disabled"
    if float(elapsed_seconds) > max_seconds:
        return "budget_exceeded"
    return "within_budget"


def read_research_surface_state(research_lab_path: Path) -> dict[str, Any]:
    """Read the mutable LAB_STATE block from research_lab.py."""
    raw_text = research_lab_path.read_text(encoding="utf-8")
    pattern = re.compile(
        rf"{re.escape(RESEARCH_SURFACE_STATE_START)}\n(?P<body>.*?)\n{re.escape(RESEARCH_SURFACE_STATE_END)}",
        flags=re.DOTALL,
    )
    match = pattern.search(raw_text)
    if match is None:
        raise ValueError(f"Research surface state block not found in {research_lab_path}")

    body = match.group("body").strip()
    if not body.startswith("LAB_STATE"):
        raise ValueError(f"LAB_STATE assignment missing in {research_lab_path}")
    _, value_text = body.split("=", 1)
    parsed = ast.literal_eval(value_text.strip())
    if not isinstance(parsed, dict):
        raise ValueError("LAB_STATE must evaluate to a dictionary.")
    return parsed


def write_research_surface_state(research_lab_path: Path, state: dict[str, Any]) -> None:
    """Rewrite the mutable LAB_STATE block inside research_lab.py."""
    raw_text = research_lab_path.read_text(encoding="utf-8")
    state_literal = pprint.pformat(_to_serializable(state), sort_dicts=True, width=100)
    replacement = (
        f"{RESEARCH_SURFACE_STATE_START}\n"
        f"LAB_STATE = {state_literal}\n"
        f"{RESEARCH_SURFACE_STATE_END}"
    )
    pattern = re.compile(
        rf"{re.escape(RESEARCH_SURFACE_STATE_START)}\n.*?\n{re.escape(RESEARCH_SURFACE_STATE_END)}",
        flags=re.DOTALL,
    )
    updated_text, replacements = pattern.subn(replacement, raw_text, count=1)
    if replacements != 1:
        raise ValueError(f"Failed to rewrite LAB_STATE block in {research_lab_path}")
    research_lab_path.write_text(updated_text, encoding="utf-8")


def apply_keep_to_research_surface(research_lab_path: Path, accepted_entry: dict[str, Any]) -> dict[str, Any]:
    """Persist an accepted experiment into the controlled research surface."""
    state = read_research_surface_state(research_lab_path)
    accepted = list(state.get("accepted_experiments", []))
    accepted.append(copy.deepcopy(accepted_entry))
    state["accepted_experiments"] = accepted

    recent_families = list(state.get("recent_kept_families", []))
    proposal_family = accepted_entry.get("proposal_family")
    if proposal_family:
        recent_families.append(str(proposal_family))
    state["recent_kept_families"] = recent_families[-25:]
    write_research_surface_state(research_lab_path, state)
    return state


def build_acceptance_decision(final_metrics: dict[str, Any], brief: dict[str, Any]) -> dict[str, Any]:
    """Build a final acceptance decision from final_metrics.json and brief thresholds."""
    minimum_improvement_pct = _to_float(brief.get("min_improvement_pct"), 0.0)
    improvement_pct = _to_float(final_metrics.get("composite_improvement_pct"), 0.0)
    best_metrics = final_metrics.get("best_search_metrics", {})
    best_model_name = str(best_metrics.get("model_name", "unknown")) if isinstance(best_metrics, dict) else "unknown"

    accepted = improvement_pct >= minimum_improvement_pct
    decision_reason = (
        "accepted: measured improvement meets threshold"
        if accepted
        else "rejected: measured improvement below threshold"
    )
    return {
        "source_of_truth": "final_metrics.json",
        "acceptance_metric": str(brief.get("acceptance_metric", "composite_score")),
        "measured_improvement_pct": improvement_pct,
        "required_min_improvement_pct": minimum_improvement_pct,
        "best_model_name": best_model_name,
        "accepted": accepted,
        "decision_reason": decision_reason,
    }


def validate_final_artifact_consistency(outputs_dir: Path) -> dict[str, Any]:
    """Validate that final artifacts agree with final_metrics.json as source of truth."""
    final_metrics = load_json_file(outputs_dir / "final_metrics.json")
    best_search_metrics = final_metrics.get("best_search_metrics", {})
    if not isinstance(best_search_metrics, dict) or not best_search_metrics:
        raise ValueError("final_metrics.json must contain best_search_metrics.")

    mismatches: list[dict[str, Any]] = []
    best_result_path = outputs_dir / "best_search_result.json"
    if best_result_path.exists():
        best_result = load_json_file(best_result_path)
        for key in ("model_name", "hyperparameters", "composite_score", "validation_verdict"):
            if _to_serializable(best_result.get(key)) != _to_serializable(best_search_metrics.get(key)):
                mismatches.append(
                    {
                        "artifact": "best_search_result.json",
                        "field": key,
                        "expected": best_search_metrics.get(key),
                        "actual": best_result.get(key),
                    }
                )
    else:
        mismatches.append(
            {
                "artifact": "best_search_result.json",
                "field": "exists",
                "expected": True,
                "actual": False,
            }
        )

    if final_metrics.get("best_model_name") != best_search_metrics.get("model_name"):
        mismatches.append(
            {
                "artifact": "final_metrics.json",
                "field": "best_model_name",
                "expected": best_search_metrics.get("model_name"),
                "actual": final_metrics.get("best_model_name"),
            }
        )
    if _to_serializable(final_metrics.get("best_model_hyperparameters")) != _to_serializable(
        best_search_metrics.get("hyperparameters")
    ):
        mismatches.append(
            {
                "artifact": "final_metrics.json",
                "field": "best_model_hyperparameters",
                "expected": best_search_metrics.get("hyperparameters"),
                "actual": final_metrics.get("best_model_hyperparameters"),
            }
        )
    if final_metrics.get("validation_verdict") != best_search_metrics.get("validation_verdict"):
        mismatches.append(
            {
                "artifact": "final_metrics.json",
                "field": "validation_verdict",
                "expected": best_search_metrics.get("validation_verdict"),
                "actual": final_metrics.get("validation_verdict"),
            }
        )

    model_path = outputs_dir / "best_search_model.pkl"
    if not model_path.exists():
        mismatches.append(
            {
                "artifact": "best_search_model.pkl",
                "field": "exists",
                "expected": True,
                "actual": False,
            }
        )

    return {
        "source_of_truth": "final_metrics.json",
        "consistent": not mismatches,
        "mismatches": mismatches,
    }
