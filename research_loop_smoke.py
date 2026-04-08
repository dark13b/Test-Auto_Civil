"""LLM smoke-test and interaction-analysis helpers for the research loop."""

from __future__ import annotations

import copy
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from llm_admission_policy import evaluate_llm_admission_policy
from proposal_engine import ProposalPreflightFailure
from research_protocol import load_json_file, write_json_file


def _apply_llm_smoke_overrides(
    config: dict[str, Any],
    *,
    backend_mode: str | None = None,
) -> dict[str, Any]:
    if not backend_mode:
        return config
    updated = copy.deepcopy(config)
    llm_config = dict(updated.get("llm", {}))
    llm_config["enabled"] = True
    llm_config["backend_mode"] = backend_mode
    updated["llm"] = llm_config
    return updated


def _build_diversity_state(memory_payload: dict[str, Any]) -> dict[str, Any]:
    family_counts: dict[str, int] = {}
    for run in memory_payload.get("runs", []):
        for trial in run.get("trials", []):
            proposal_family = str(trial.get("proposal_family", "unknown"))
            family_counts[proposal_family] = family_counts.get(proposal_family, 0) + 1
    return {"historic_family_counts": family_counts}


def _run_llm_preflight(
    *,
    proposal_engine: Any | None,
    outputs_dir: Path,
    available_models: dict[str, dict[str, Any]],
    brief: dict[str, Any],
    current_best: dict[str, Any],
    memory_payload: dict[str, Any],
    knowledge_context: str,
    failure_patterns: dict[str, Any],
    archive_records: list[dict[str, Any]],
    llm_enabled: bool,
    allow_deterministic_fallback: bool,
    model_hint: str | None = None,
) -> dict[str, Any] | None:
    if proposal_engine is None or not hasattr(proposal_engine, "run_proposal_smoke_test"):
        if not llm_enabled:
            result = {
                "ok": False,
                "status": "aborted_due_to_preflight_failure",
                "backend": "disabled",
                "model": None,
                "prompt_hash": None,
                "proposal_preview": None,
                "error": "LLM proposal engine is disabled by config.",
            }
            write_json_file(outputs_dir / "llm_smoke_test.json", result)
            if not allow_deterministic_fallback:
                raise ProposalPreflightFailure(
                    result["error"],
                    failure_kind="aborted_due_to_preflight_failure",
                    interaction=result,
                )
            return result
        return None

    smoke_result = proposal_engine.run_proposal_smoke_test(
        available_models=available_models,
        research_brief=brief,
        current_best=current_best,
        experiment_memory=memory_payload,
        diversity_state=_build_diversity_state(memory_payload),
        trial_history=[
            trial
            for run in memory_payload.get("runs", [])
            for trial in run.get("trials", [])
        ],
        search_progress={"phase": "preflight"},
        failure_patterns=failure_patterns,
        knowledge_context=knowledge_context,
        archive_records=archive_records,
        model_hint=model_hint,
    )
    write_json_file(outputs_dir / "llm_smoke_test.json", smoke_result)
    if not bool(smoke_result.get("ok", False)) and not allow_deterministic_fallback:
        raise ProposalPreflightFailure(
            str(smoke_result.get("error") or "LLM smoke test failed."),
            failure_kind="aborted_due_to_preflight_failure",
            interaction=smoke_result,
        )
    return smoke_result


def _percentile(sorted_values: list[float], percentile: float) -> float | None:
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return sorted_values[0]
    bounded = min(max(percentile, 0.0), 1.0)
    index = max(0, min(len(sorted_values) - 1, int((len(sorted_values) * bounded) + 0.999999) - 1))
    return sorted_values[index]


def summarize_smoke_test_trials(
    trials: list[dict[str, Any]],
    *,
    requested_backend: str | None = None,
    requested_model: str | None = None,
    admission_policy_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    total_trials = len(trials)
    if total_trials == 0:
        summary = {
            "total_trials": 0,
            "success_count": 0,
            "success_rate": 0.0,
            "backend_failure_count": 0,
            "backend_failure_rate": 0.0,
            "parse_failure_count": 0,
            "parse_failure_rate": 0.0,
            "schema_failure_count": 0,
            "schema_failure_rate": 0.0,
            "semantic_failure_count": 0,
            "semantic_failure_rate": 0.0,
            "hidden_channel_count": 0,
            "hidden_channel_incidence": 0.0,
            "empty_visible_count": 0,
            "empty_visible_response_incidence": 0.0,
            "repair_usage_rate": 0.0,
            "median_latency_seconds": None,
            "p95_latency_seconds": None,
            "status_counts": {},
            "backend_counts": {},
            "model_counts": {},
            "backend_model_counts": {},
            "failure_class_counts": {},
            "unexpected_backend_model_pairs": {},
        }
        admission_policy = evaluate_llm_admission_policy(summary, admission_policy_config=admission_policy_config)
        summary["admission_policy"] = admission_policy
        summary["compatibility"] = {
            "issue_codes": admission_policy["issue_codes"],
            "issues": [check for check in admission_policy["checks"] if check["status"] != "passed"],
            "interpretation": admission_policy["interpretation"],
            "verdict": admission_policy["verdict"],
            "eligible_for_control_tasks": admission_policy["eligible_for_control_tasks"],
        }
        return summary

    status_counts: dict[str, int] = {}
    backend_counts: dict[str, int] = {}
    model_counts: dict[str, int] = {}
    backend_model_counts: dict[str, int] = {}
    failure_class_counts: dict[str, int] = {}
    unexpected_pairs: dict[str, int] = {}
    success_count = 0
    backend_failure_count = 0
    parse_failure_count = 0
    schema_failure_count = 0
    semantic_failure_count = 0
    hidden_channel_count = 0
    empty_visible_count = 0
    repair_count = 0
    latencies: list[float] = []

    for trial in trials:
        status = str(trial.get("status", "unknown"))
        backend = str(trial.get("backend", "unknown"))
        model = str(trial.get("model", "unknown"))
        channel = str(trial.get("extracted_from_channel", "unknown"))
        pair = f"{backend}::{model}"

        status_counts[status] = status_counts.get(status, 0) + 1
        backend_counts[backend] = backend_counts.get(backend, 0) + 1
        model_counts[model] = model_counts.get(model, 0) + 1
        backend_model_counts[pair] = backend_model_counts.get(pair, 0) + 1

        failure_class = trial.get("failure_class")
        if failure_class:
            code = str(failure_class)
            failure_class_counts[code] = failure_class_counts.get(code, 0) + 1

        if bool(trial.get("ok", False)):
            success_count += 1
        if status == "llm_backend_failure":
            backend_failure_count += 1
        if status == "llm_parse_failure":
            parse_failure_count += 1
        if status == "llm_schema_failure":
            schema_failure_count += 1
        semantic_result = str(trial.get("semantic_validation_result", "")).lower()
        if semantic_result == "failed" or status in {
            "duplicate_proposal",
            "near_duplicate_proposal",
            "semantically_invalid_candidate_config",
            "unsupported_novelty_claim",
            "inconsistent_expected_effect",
            "non_executable_semantic_config",
        }:
            semantic_failure_count += 1
        if channel in {"thinking", "repaired_thinking"}:
            hidden_channel_count += 1
        if bool(trial.get("visible_response_empty", False)):
            empty_visible_count += 1
        if bool(trial.get("repair_used", False)):
            repair_count += 1

        latency = trial.get("latency_seconds")
        if isinstance(latency, (int, float)):
            latencies.append(float(latency))

        backend_unexpected = requested_backend not in {None, "", "hybrid"} and backend != requested_backend
        model_unexpected = bool(requested_model) and model != requested_model
        if backend_unexpected or model_unexpected:
            unexpected_pairs[pair] = unexpected_pairs.get(pair, 0) + 1

    latencies.sort()
    summary = {
        "total_trials": total_trials,
        "success_count": success_count,
        "success_rate": success_count / total_trials,
        "backend_failure_count": backend_failure_count,
        "backend_failure_rate": backend_failure_count / total_trials,
        "parse_failure_count": parse_failure_count,
        "parse_failure_rate": parse_failure_count / total_trials,
        "schema_failure_count": schema_failure_count,
        "schema_failure_rate": schema_failure_count / total_trials,
        "semantic_failure_count": semantic_failure_count,
        "semantic_failure_rate": semantic_failure_count / total_trials,
        "hidden_channel_count": hidden_channel_count,
        "hidden_channel_incidence": hidden_channel_count / total_trials,
        "empty_visible_count": empty_visible_count,
        "empty_visible_response_incidence": empty_visible_count / total_trials,
        "repair_usage_rate": repair_count / total_trials,
        "median_latency_seconds": _percentile(latencies, 0.5),
        "p95_latency_seconds": _percentile(latencies, 0.95),
        "status_counts": status_counts,
        "backend_counts": backend_counts,
        "model_counts": model_counts,
        "backend_model_counts": backend_model_counts,
        "failure_class_counts": failure_class_counts,
        "unexpected_backend_model_pairs": unexpected_pairs,
    }
    admission_policy = evaluate_llm_admission_policy(summary, admission_policy_config=admission_policy_config)
    summary["admission_policy"] = admission_policy
    summary["compatibility"] = {
        "issue_codes": admission_policy["issue_codes"],
        "issues": [check for check in admission_policy["checks"] if check["status"] != "passed"],
        "interpretation": admission_policy["interpretation"],
        "verdict": admission_policy["verdict"],
        "eligible_for_control_tasks": admission_policy["eligible_for_control_tasks"],
    }
    return summary


def run_repeated_llm_smoke_test(
    *,
    proposal_engine: Any | None,
    available_models: dict[str, dict[str, Any]],
    brief: dict[str, Any],
    current_best: dict[str, Any],
    memory_payload: dict[str, Any],
    knowledge_context: str,
    failure_patterns: dict[str, Any],
    archive_records: list[dict[str, Any]],
    trials: int,
    summary_path: Path | None = None,
    requested_backend: str | None = None,
    model_hint: str | None = None,
    admission_policy_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    requested_trials = max(1, int(trials))
    timestamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    trial_history = [
        trial
        for run in memory_payload.get("runs", [])
        for trial in run.get("trials", [])
    ]
    normalized_trials: list[dict[str, Any]] = []

    for trial_index in range(1, requested_trials + 1):
        if proposal_engine is None or not hasattr(proposal_engine, "run_proposal_smoke_test"):
            trial_result = {
                "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                "ok": False,
                "status": "aborted_due_to_preflight_failure",
                "backend": requested_backend or "disabled",
                "model": model_hint,
                "prompt_hash": None,
                "proposal_preview": None,
                "error": "LLM proposal engine is disabled by config.",
                "extracted_from_channel": "response",
                "visible_response_empty": True,
                "parse_success": False,
                "schema_success": False,
                "repair_used": False,
                "latency_seconds": None,
                "failure_class": "llm_backend_failure",
            }
        else:
            trial_result = proposal_engine.run_proposal_smoke_test(
                available_models=available_models,
                research_brief=brief,
                current_best=current_best,
                experiment_memory=memory_payload,
                diversity_state=_build_diversity_state(memory_payload),
                trial_history=trial_history,
                search_progress={"phase": "preflight", "trial_index": trial_index, "trial_count": requested_trials},
                failure_patterns=failure_patterns,
                knowledge_context=knowledge_context,
                archive_records=archive_records,
                model_hint=model_hint,
            )
        normalized = dict(trial_result)
        normalized["trial_index"] = trial_index
        normalized.setdefault("timestamp", datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"))
        normalized_trials.append(normalized)

    summary = summarize_smoke_test_trials(
        normalized_trials,
        requested_backend=requested_backend or (getattr(proposal_engine, "backend_name", None) if proposal_engine is not None else None),
        requested_model=model_hint,
        admission_policy_config=admission_policy_config,
    )
    report = {
        "generated_at": timestamp,
        "requested_trials": requested_trials,
        "requested_backend": requested_backend or (getattr(proposal_engine, "backend_name", None) if proposal_engine is not None else None),
        "requested_model": model_hint,
        "summary_path": str(summary_path) if summary_path is not None else None,
        "trials": normalized_trials,
        "summary": summary,
    }
    if summary_path is not None:
        write_json_file(summary_path, report)
    return report


def analyze_interaction_log(outputs_dir: Path, *, filename: str = "llm_interactions.jsonl") -> dict[str, Any]:
    interaction_path = outputs_dir / filename
    if not interaction_path.exists():
        return {
            "exists": False,
            "interaction_count": 0,
            "channel_counts": {},
            "rejection_counts": {},
            "semantic_rejection_counts": {},
        }

    channel_counts: dict[str, int] = {}
    rejection_counts: dict[str, int] = {}
    semantic_rejection_counts: dict[str, int] = {}
    repair_count = 0
    parse_success_count = 0
    with interaction_path.open("r", encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    for record in records:
        channel = str(record.get("extracted_from_channel", "unknown"))
        channel_counts[channel] = channel_counts.get(channel, 0) + 1
        if bool(record.get("repair_used", False)):
            repair_count += 1
        if bool(record.get("parse_success", False)):
            parse_success_count += 1
        rejection = record.get("rejection_reason")
        if isinstance(rejection, dict):
            code = str(rejection.get("code", "unknown"))
            rejection_counts[code] = rejection_counts.get(code, 0) + 1
        semantic_rejection = record.get("semantic_rejection_reason")
        if isinstance(semantic_rejection, dict):
            code = str(semantic_rejection.get("code", "unknown"))
            semantic_rejection_counts[code] = semantic_rejection_counts.get(code, 0) + 1
    return {
        "exists": True,
        "interaction_count": len(records),
        "channel_counts": channel_counts,
        "rejection_counts": rejection_counts,
        "semantic_rejection_counts": semantic_rejection_counts,
        "repair_count": repair_count,
        "parse_success_count": parse_success_count,
    }
