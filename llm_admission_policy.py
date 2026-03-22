from __future__ import annotations

from copy import deepcopy
from typing import Any


DEFAULT_LLM_ADMISSION_POLICY = {
    "policy_name": "backend_admission",
    "version": 1,
    "warning_band_fraction": 0.8,
    "thresholds": {
        "min_success_rate": 0.9,
        "max_backend_failure_rate": 0.1,
        "max_parse_failure_rate": 0.1,
        "max_schema_failure_rate": 0.1,
        "max_hidden_channel_incidence": 0.05,
        "max_empty_visible_response_incidence": 0.1,
        "max_p95_latency_seconds": 5.0,
    },
}


VERDICT_INTERPRETATIONS = {
    "usable": "usable",
    "usable_with_caution": "usable with caution",
    "unstable": "unstable",
    "blocked_for_control_tasks": "blocked for control tasks",
}


_HARD_BLOCK_CODES = {
    "no_trials",
    "success_rate_below_minimum",
    "backend_failure_rate_exceeded",
    "p95_latency_unavailable",
}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _to_float(value: Any, fallback: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(fallback)


def _normalize_thresholds(raw_thresholds: dict[str, Any] | None) -> dict[str, float]:
    defaults = DEFAULT_LLM_ADMISSION_POLICY["thresholds"]
    raw_thresholds = raw_thresholds if isinstance(raw_thresholds, dict) else {}
    return {
        "min_success_rate": _to_float(raw_thresholds.get("min_success_rate"), defaults["min_success_rate"]),
        "max_backend_failure_rate": _to_float(
            raw_thresholds.get("max_backend_failure_rate"), defaults["max_backend_failure_rate"]
        ),
        "max_parse_failure_rate": _to_float(
            raw_thresholds.get("max_parse_failure_rate"), defaults["max_parse_failure_rate"]
        ),
        "max_schema_failure_rate": _to_float(
            raw_thresholds.get("max_schema_failure_rate"), defaults["max_schema_failure_rate"]
        ),
        "max_hidden_channel_incidence": _to_float(
            raw_thresholds.get("max_hidden_channel_incidence"), defaults["max_hidden_channel_incidence"]
        ),
        "max_empty_visible_response_incidence": _to_float(
            raw_thresholds.get("max_empty_visible_response_incidence"),
            defaults["max_empty_visible_response_incidence"],
        ),
        "max_p95_latency_seconds": _to_float(
            raw_thresholds.get("max_p95_latency_seconds"), defaults["max_p95_latency_seconds"]
        ),
    }


def normalize_llm_admission_policy_config(raw_policy: dict[str, Any] | None) -> dict[str, Any]:
    """Return a normalized admission-policy config with defaults filled in."""
    if not isinstance(raw_policy, dict) or not raw_policy:
        normalized = deepcopy(DEFAULT_LLM_ADMISSION_POLICY)
    else:
        normalized = _deep_merge(DEFAULT_LLM_ADMISSION_POLICY, raw_policy)

    normalized["policy_name"] = str(normalized.get("policy_name") or DEFAULT_LLM_ADMISSION_POLICY["policy_name"])
    normalized["version"] = int(normalized.get("version", DEFAULT_LLM_ADMISSION_POLICY["version"]))
    warning_band_fraction = _to_float(
        normalized.get("warning_band_fraction"), DEFAULT_LLM_ADMISSION_POLICY["warning_band_fraction"]
    )
    normalized["warning_band_fraction"] = min(max(warning_band_fraction, 0.0), 1.0)
    normalized["thresholds"] = _normalize_thresholds(normalized.get("thresholds"))
    return normalized


def _ratio(numerator: Any, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    try:
        return float(numerator) / float(denominator)
    except (TypeError, ValueError, ZeroDivisionError):
        return 0.0


def _evaluate_maximum(
    *,
    metric_name: str,
    measured: Any,
    threshold: float,
    warning_band_fraction: float,
) -> dict[str, Any]:
    measured_value = None if measured is None else _to_float(measured, 0.0)
    warning_threshold = threshold * warning_band_fraction
    if measured_value is None:
        return {
            "metric": metric_name,
            "comparison": "<=",
            "measured": None,
            "threshold": threshold,
            "warning_threshold": warning_threshold,
            "status": "failed",
            "issue_code": f"{metric_name}_unavailable",
        }
    if measured_value > threshold:
        return {
            "metric": metric_name,
            "comparison": "<=",
            "measured": measured_value,
            "threshold": threshold,
            "warning_threshold": warning_threshold,
            "status": "failed",
            "issue_code": f"{metric_name}_exceeded",
        }
    if measured_value > warning_threshold:
        return {
            "metric": metric_name,
            "comparison": "<=",
            "measured": measured_value,
            "threshold": threshold,
            "warning_threshold": warning_threshold,
            "status": "warning",
            "issue_code": f"{metric_name}_near_limit",
        }
    return {
        "metric": metric_name,
        "comparison": "<=",
        "measured": measured_value,
        "threshold": threshold,
        "warning_threshold": warning_threshold,
        "status": "passed",
        "issue_code": None,
    }


def _evaluate_minimum(
    *,
    metric_name: str,
    measured: Any,
    threshold: float,
    warning_band_fraction: float,
) -> dict[str, Any]:
    measured_value = None if measured is None else _to_float(measured, 0.0)
    warning_floor = threshold + ((1.0 - threshold) * (1.0 - warning_band_fraction))
    if measured_value is None:
        return {
            "metric": metric_name,
            "comparison": ">=",
            "measured": None,
            "threshold": threshold,
            "warning_threshold": warning_floor,
            "status": "failed",
            "issue_code": f"{metric_name}_unavailable",
        }
    if measured_value < threshold:
        return {
            "metric": metric_name,
            "comparison": ">=",
            "measured": measured_value,
            "threshold": threshold,
            "warning_threshold": warning_floor,
            "status": "failed",
            "issue_code": f"{metric_name}_below_minimum",
        }
    if measured_value < warning_floor:
        return {
            "metric": metric_name,
            "comparison": ">=",
            "measured": measured_value,
            "threshold": threshold,
            "warning_threshold": warning_floor,
            "status": "warning",
            "issue_code": f"{metric_name}_near_minimum",
        }
    return {
        "metric": metric_name,
        "comparison": ">=",
        "measured": measured_value,
        "threshold": threshold,
        "warning_threshold": warning_floor,
        "status": "passed",
        "issue_code": None,
    }


def evaluate_llm_admission_policy(
    summary: dict[str, Any],
    *,
    admission_policy_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Classify a smoke-test summary into a policy verdict."""
    policy = normalize_llm_admission_policy_config(admission_policy_config)
    thresholds = policy["thresholds"]
    warning_band_fraction = float(policy["warning_band_fraction"])

    total_trials = int(summary.get("total_trials", 0) or 0)
    status_counts = summary.get("status_counts", {})
    status_counts = status_counts if isinstance(status_counts, dict) else {}

    backend_failure_count = int(
        summary.get("backend_failure_count", status_counts.get("llm_backend_failure", 0)) or 0
    )
    parse_failure_count = int(summary.get("parse_failure_count", 0) or 0)
    schema_failure_count = int(summary.get("schema_failure_count", 0) or 0)
    hidden_channel_count = int(summary.get("hidden_channel_count", 0) or 0)
    empty_visible_count = int(summary.get("empty_visible_count", 0) or 0)

    metrics = {
        "total_trials": total_trials,
        "success_rate": _to_float(summary.get("success_rate"), 0.0),
        "backend_failure_rate": _ratio(backend_failure_count, total_trials),
        "parse_failure_rate": _to_float(summary.get("parse_failure_rate"), 0.0),
        "schema_failure_rate": _to_float(summary.get("schema_failure_rate"), 0.0),
        "hidden_channel_incidence": _to_float(summary.get("hidden_channel_incidence"), 0.0),
        "empty_visible_response_incidence": _to_float(summary.get("empty_visible_response_incidence"), 0.0),
        "p95_latency_seconds": summary.get("p95_latency_seconds"),
    }

    checks = [
        _evaluate_minimum(
            metric_name="success_rate",
            measured=metrics["success_rate"],
            threshold=thresholds["min_success_rate"],
            warning_band_fraction=warning_band_fraction,
        ),
        _evaluate_maximum(
            metric_name="backend_failure_rate",
            measured=metrics["backend_failure_rate"],
            threshold=thresholds["max_backend_failure_rate"],
            warning_band_fraction=warning_band_fraction,
        ),
        _evaluate_maximum(
            metric_name="parse_failure_rate",
            measured=metrics["parse_failure_rate"],
            threshold=thresholds["max_parse_failure_rate"],
            warning_band_fraction=warning_band_fraction,
        ),
        _evaluate_maximum(
            metric_name="schema_failure_rate",
            measured=metrics["schema_failure_rate"],
            threshold=thresholds["max_schema_failure_rate"],
            warning_band_fraction=warning_band_fraction,
        ),
        _evaluate_maximum(
            metric_name="hidden_channel_incidence",
            measured=metrics["hidden_channel_incidence"],
            threshold=thresholds["max_hidden_channel_incidence"],
            warning_band_fraction=warning_band_fraction,
        ),
        _evaluate_maximum(
            metric_name="empty_visible_response_incidence",
            measured=metrics["empty_visible_response_incidence"],
            threshold=thresholds["max_empty_visible_response_incidence"],
            warning_band_fraction=warning_band_fraction,
        ),
        _evaluate_maximum(
            metric_name="p95_latency_seconds",
            measured=metrics["p95_latency_seconds"],
            threshold=thresholds["max_p95_latency_seconds"],
            warning_band_fraction=warning_band_fraction,
        ),
    ]

    issue_codes = [str(check["issue_code"]) for check in checks if check.get("issue_code")]
    warning_checks = [check for check in checks if check["status"] == "warning"]
    failed_checks = [check for check in checks if check["status"] == "failed"]
    hard_block_detected = any(code in _HARD_BLOCK_CODES for code in issue_codes)

    if total_trials <= 0:
        verdict = "blocked_for_control_tasks"
    elif hard_block_detected:
        verdict = "blocked_for_control_tasks"
    elif len(failed_checks) >= 2:
        verdict = "blocked_for_control_tasks"
    elif len(failed_checks) == 1:
        verdict = "unstable"
    elif warning_checks:
        verdict = "usable_with_caution"
    else:
        verdict = "usable"

    interpretation = VERDICT_INTERPRETATIONS[verdict]
    return {
        "policy_name": policy["policy_name"],
        "policy_version": policy["version"],
        "verdict": verdict,
        "interpretation": interpretation,
        "eligible_for_control_tasks": verdict in {"usable", "usable_with_caution"},
        "thresholds": thresholds,
        "metrics": metrics,
        "checks": checks,
        "issue_codes": issue_codes,
        "warning_issue_codes": [str(check["issue_code"]) for check in warning_checks if check.get("issue_code")],
        "failure_issue_codes": [str(check["issue_code"]) for check in failed_checks if check.get("issue_code")],
    }
