"""Single responsibility: extract and repair structured proposal data from raw LLM text output."""

from __future__ import annotations

import json
import re
from typing import Any

JSON_FENCE_PATTERN = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)
INVALID_PARAM = object()


class ProposalExtractionError(RuntimeError):
    """Raised when the visible model response does not contain valid structured output."""


def extract_structured_output(
    *,
    channels: dict[str, str],
    expect_array: bool,
) -> tuple[str, str, Any, bool]:
    for channel_name, channel_text in (
        ("response", channels.get("response_text", "")),
        ("thinking", channels.get("thinking_text", "")),
    ):
        parsed = _extract_json(channel_text, expect_array=expect_array)
        if parsed is not None:
            return channel_text.strip(), channel_name, parsed, False

        repaired = _repair_json_text(channel_text, expect_array=expect_array)
        if repaired is not None:
            try:
                parsed = json.loads(repaired)
            except Exception:
                parsed = None
            if parsed is not None and (
                (expect_array and isinstance(parsed, list)) or (not expect_array and isinstance(parsed, dict))
            ):
                extracted_from = "repaired_json" if channel_name == "response" else "repaired_thinking"
                return repaired, extracted_from, parsed, True

    raise ProposalExtractionError("Visible response was empty or did not contain valid JSON.")


def extract_proposals(raw_text: str, *, expect_array: bool = False) -> Any:
    channels = {"response_text": raw_text, "thinking_text": ""}
    _, _, parsed, _ = extract_structured_output(channels=channels, expect_array=expect_array)
    return parsed


def validate_proposal(
    payload: Any,
    available_models: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    if not isinstance(payload, dict):
        return None
    model_name = payload.get("model_name")
    params = payload.get("params")
    if not isinstance(model_name, str) or model_name not in available_models:
        return None
    if not isinstance(params, dict):
        return None

    search_space = available_models[model_name].get("search_space", {})
    normalized_params: dict[str, Any] = {}
    for param_name, param_value in params.items():
        if not isinstance(param_name, str) or param_name not in search_space:
            return None
        normalized_value = _validate_param_value(param_value, search_space[param_name])
        if normalized_value is INVALID_PARAM:
            return None
        normalized_params[param_name] = normalized_value

    return {
        "model_name": model_name,
        "display_name": str(available_models[model_name].get("display_name", model_name)),
        "params": normalized_params,
        "proposal_family": str(payload.get("proposal_family", "llm-generated")),
        "hypothesis": str(payload.get("hypothesis", "LLM-generated suggestion")),
        "expected_delta": float(payload["expected_delta"]) if payload.get("expected_delta") is not None else 0.0,
    }


def _extract_json(raw_text: str, *, expect_array: bool) -> Any:
    cleaned = _clean_json_candidate(raw_text)
    if not cleaned:
        return None
    try:
        parsed = json.loads(cleaned)
    except Exception:
        parsed = None
    if parsed is not None and ((expect_array and isinstance(parsed, list)) or (not expect_array and isinstance(parsed, dict))):
        return parsed

    start_char = "[" if expect_array else "{"
    end_char = "]" if expect_array else "}"
    start = cleaned.find(start_char)
    end = cleaned.rfind(end_char)
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        parsed = json.loads(cleaned[start:end + 1])
    except Exception:
        return None
    return parsed if ((expect_array and isinstance(parsed, list)) or (not expect_array and isinstance(parsed, dict))) else None


def _repair_json_text(raw_text: str, *, expect_array: bool) -> str | None:
    cleaned = _clean_json_candidate(raw_text)
    if not cleaned:
        return None
    if re.search(r"}\s*{", cleaned) or re.search(r"]\s*\[", cleaned):
        return None

    start_char = "[" if expect_array else "{"
    start = cleaned.find(start_char)
    if start == -1:
        return None
    candidate = re.sub(r",(\s*[}\]])", r"\1", cleaned[start:])

    if _contains_multiple_top_level_objects(candidate, expect_array=expect_array):
        return None

    brace = bracket = 0
    in_string = escape = False
    for ch in candidate:
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            brace += 1
        elif ch == "}":
            brace -= 1
            if brace < 0:
                return None
        elif ch == "[":
            bracket += 1
        elif ch == "]":
            bracket -= 1
            if bracket < 0:
                return None

    repaired = candidate + ("}" * brace) + ("]" * bracket)
    try:
        parsed = json.loads(repaired)
    except Exception:
        return None
    return repaired if ((expect_array and isinstance(parsed, list)) or (not expect_array and isinstance(parsed, dict))) else None


def _clean_json_candidate(raw_text: str) -> str:
    cleaned = str(raw_text or "").strip()
    return JSON_FENCE_PATTERN.sub("", cleaned).strip() if cleaned else ""


def _contains_multiple_top_level_objects(candidate: str, *, expect_array: bool) -> bool:
    if expect_array:
        return False
    depth = object_count = 0
    in_string = escape = False
    for ch in candidate:
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
            if depth == 1:
                object_count += 1
                if object_count > 1:
                    return True
        elif ch == "}":
            depth = max(0, depth - 1)
    return False


def _validate_param_value(value: Any, spec: dict[str, Any]) -> Any:
    t = spec.get("type")
    if t == "int":
        coerced = _coerce_number(value)
        if coerced is None or coerced != int(coerced):
            return INVALID_PARAM
        normalized = int(coerced)
        low, high, step = int(spec["low"]), int(spec["high"]), int(spec.get("step", 1))
        if normalized < low or normalized > high:
            return INVALID_PARAM
        if step > 0 and (normalized - low) % step != 0:
            return INVALID_PARAM
        return normalized
    if t == "float":
        coerced = _coerce_number(value)
        if coerced is None:
            return INVALID_PARAM
        normalized = float(coerced)
        if normalized < (float(spec["low"]) - 1e-12) or normalized > (float(spec["high"]) + 1e-12):
            return INVALID_PARAM
        return normalized
    if t == "categorical":
        choices = list(spec.get("choices", []))
        if value == "null" and None in choices:
            return None
        return value if value in choices else INVALID_PARAM
    return INVALID_PARAM


def _coerce_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        stripped = value.strip()
        try:
            return float(stripped) if stripped else None
        except ValueError:
            return None
    return None
