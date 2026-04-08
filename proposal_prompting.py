"""Single responsibility: build prompt strings and format proposal context for LLM calls."""

from __future__ import annotations

import json
import re
from typing import Any


def build_experiment_prompt(
    *,
    available_models: dict[str, dict[str, Any]],
    research_brief: dict[str, Any],
    current_best: dict[str, Any],
    experiment_memory: dict[str, Any],
    diversity_state: dict[str, Any],
    proposal_count: int,
    trial_history: list[dict[str, Any]],
    search_progress: dict[str, Any],
    failure_patterns: dict[str, Any],
    knowledge_context: str,
    underexplored_families: list[str],
    family_state: dict[str, Any],
    prompt_variant: str,
) -> str:
    max_recent = 5 if prompt_variant == "compact" else 10
    recent_trials = trial_history[-min(max_recent, len(trial_history)):] if trial_history else []
    single_mode = max(1, int(proposal_count)) == 1

    lines = [
        "Generate experiment proposals for concrete compressive strength regression.",
        "Think internally using these hidden steps only: STEP 1: diagnose, STEP 2: hypothesize, STEP 3: predict, STEP 4: propose JSON.",
        "Do not reveal the hidden steps. Output JSON only.",
        f"Goal: {research_brief.get('goal', 'Improve composite_score')}",
        f"Acceptance metric: {research_brief.get('acceptance_metric', 'composite_score')}",
        f"Current best: {current_best.get('model_name', 'unknown')} | composite={_format_metric(current_best.get('composite_score'))}",
        f"Strongest family: {family_state.get('strongest_active_family') or 'unknown'}",
        "Allowed families: " + ", ".join(sorted(available_models.keys())),
        "Saturated families: " + _format_family_list(family_state.get("saturated_families", [])),
        "Underexplored but weak families: " + _format_family_list(family_state.get("underexplored_weak_families", [])),
        "Underexplored and still worth probing: " + _format_family_list(family_state.get("underexplored_promising_families", [])),
        "Temporarily blocked families: " + _format_family_list(family_state.get("temporarily_blocked_families", [])),
        "Avoid saturated or temporarily blocked families unless the proposal is materially different from recent runs.",
        "Do not repeat exact recent configs.",
        'Schema example: {"model_name":"MODEL","params":{}}',
    ]
    if prompt_variant == "compact":
        lines.append("Output ONLY the final JSON object.")
    if single_mode:
        lines.extend(["Return exactly one JSON object.", "No markdown.", "No explanation."])
    else:
        lines.extend([
            f"Return exactly one JSON array with up to {max(1, int(proposal_count))} proposal objects.",
            "No markdown.", "No explanation.",
        ])
    if search_progress and prompt_variant == "rich":
        lines.append("Search progress: " + json.dumps(search_progress, sort_keys=True))
    if failure_patterns:
        lines.append("Failure patterns: " + json.dumps(failure_patterns, sort_keys=True))
    if knowledge_context:
        lines.append("Knowledge context:\n" + knowledge_context)
    if underexplored_families:
        lines.append("Underexplored families: " + ", ".join(sorted(str(f) for f in underexplored_families)))
    if diversity_state.get("historic_family_counts") and prompt_variant == "rich":
        lines.append("Historic family counts: " + json.dumps(diversity_state.get("historic_family_counts", {}), sort_keys=True))
    if prompt_variant == "rich":
        lines.append(f"Accepted experiments in memory: {len(experiment_memory.get('accepted_experiments', []))}")
        lines.append("Family state: " + json.dumps(_compact_family_state(family_state), sort_keys=True))
    if recent_trials:
        lines.append(f"Recent trials (last {len(recent_trials)}):")
        for trial in recent_trials:
            trial_id = trial.get("experiment_id") or f"trial-{trial.get('trial_number', 'unknown')}"
            lines.append(
                f"{trial_id} | model={trial.get('model_name', 'unknown')} | "
                f"composite={_format_metric(trial.get('composite_score'))} | "
                f"verdict={trial.get('validation_verdict', 'UNKNOWN')} | "
                f"params={json.dumps(_extract_trial_params(trial), sort_keys=True)}"
            )
    lines.append("Available models and parameter ranges:")
    for model_name, model_config in available_models.items():
        lines.append(f"{model_name}: {format_search_space(model_config.get('search_space', {}))}")
    return "\n".join(lines)


def build_prompt(context: Any, config: dict[str, Any]) -> str:
    if hasattr(context, "brief"):
        context_dict = {
            "brief": dict(getattr(context, "brief", {})),
            "recent_results": list(getattr(context, "recent_results", [])),
            "best_metrics": dict(getattr(context, "best_metrics", {})),
            "config": dict(getattr(context, "config", {})),
        }
    else:
        context_dict = dict(context or {})
    llm_config = dict(config or {})
    return build_experiment_prompt(
        available_models=dict(llm_config.get("available_models", {})),
        research_brief=dict(context_dict.get("brief", {})),
        current_best=dict(context_dict.get("best_metrics", {})),
        experiment_memory=dict(llm_config.get("experiment_memory", {})),
        diversity_state=dict(llm_config.get("diversity_state", {})),
        proposal_count=int(llm_config.get("proposal_count", 1)),
        trial_history=list(context_dict.get("recent_results", [])),
        search_progress=dict(llm_config.get("search_progress", {})),
        failure_patterns=dict(llm_config.get("failure_patterns", {})),
        knowledge_context=str(llm_config.get("knowledge_context", "") or ""),
        underexplored_families=list(llm_config.get("underexplored_families", [])),
        family_state=dict(llm_config.get("family_state", {})),
        prompt_variant=str(llm_config.get("prompt_variant", "rich")),
    )


def build_research_proposal_prompt(
    *,
    available_models: dict[str, dict[str, Any]],
    research_brief: dict[str, Any],
    current_best: dict[str, Any],
    experiment_memory: dict[str, Any],
    diversity_state: dict[str, Any],
    proposal_count: int,
    trial_history: list[dict[str, Any]],
    search_progress: dict[str, Any],
    failure_patterns: dict[str, Any],
    knowledge_context: str,
    archive_records: list[dict[str, Any]],
    family_state: dict[str, Any],
    prompt_variant: str,
    selected_candidates: list[dict[str, Any]],
    smoke_test: bool,
) -> str:
    recent_trials = trial_history[-5:] if prompt_variant == "compact" else trial_history[-8:]
    archive_summary = _build_archive_history_summary(archive_records)
    novelty_summary = _build_novelty_summary(archive_records, diversity_state)
    selected_summary = [
        {"model_name": c.get("model_name"), "proposal_family": c.get("proposal_family"), "hypothesis": c.get("hypothesis")}
        for c in selected_candidates
    ]
    lines = [
        "You are generating one research hypothesis for an AutoResearch loop.",
        "Your proposal must be falsifiable, concrete, non-redundant, and directly executable.",
        f"Goal: {research_brief.get('goal', 'Improve composite_score')}",
        f"Acceptance metric: {research_brief.get('acceptance_metric', 'composite_score')}",
        f"Current best model: {current_best.get('model_name', 'unknown')}",
        f"Current best composite_score: {_format_metric(current_best.get('composite_score'))}",
        f"Requested proposals this step: {max(1, int(proposal_count))}",
        f"Mode: {'smoke_test' if smoke_test else 'research'}",
        "Recent failures:",
        json.dumps(failure_patterns, sort_keys=True) if failure_patterns else "(none)",
        "Accepted and rejected hypothesis history:",
        archive_summary,
        "Novelty pressure summary:",
        novelty_summary,
        "Current experiment context:",
        json.dumps(search_progress, sort_keys=True) if search_progress else "(none)",
        "Family state summary:",
        json.dumps(_compact_family_state(family_state), sort_keys=True),
        "Already selected this step:",
        json.dumps(selected_summary, sort_keys=True) if selected_summary else "(none)",
        "Relevant domain knowledge:",
        knowledge_context or "(none provided)",
        "Allowed executable model families and parameter ranges:",
    ]
    for model_name, model_config in available_models.items():
        lines.append(f"{model_name}: {format_search_space(model_config.get('search_space', {}))}")
    lines.extend([
        "Return exactly one JSON object matching the required schema.",
        "Do not output markdown, comments, prose, or multiple objects.",
        "The candidate_config.model_name must be one of the allowed executable families.",
        "The candidate_config.params must stay within the declared ranges.",
        "Schema:",
        json.dumps(build_research_proposal_schema(), sort_keys=True),
    ])
    return "\n".join(lines)


def build_research_proposal_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "hypothesis": {"type": "string"},
            "rationale": {"type": "string"},
            "change_type": {"type": "string", "enum": ["hyperparameter", "feature", "preprocessing", "model_family", "objective", "sampling", "other"]},
            "target_component": {"type": "string"},
            "proposed_change": {"type": "string"},
            "expected_direction": {"type": "string", "enum": ["improve", "worsen_risk", "uncertain"]},
            "expected_metric_effect": {
                "type": "object",
                "properties": {
                    "metric": {"type": "string"},
                    "direction": {"type": "string", "enum": ["up", "down"]},
                    "magnitude_estimate": {"type": "string"},
                },
                "required": ["metric", "direction", "magnitude_estimate"],
                "additionalProperties": False,
            },
            "confidence": {"type": "number"},
            "novelty_claim": {"type": "string"},
            "risk_notes": {"type": "string"},
            "candidate_config": {
                "type": "object",
                "properties": {"model_name": {"type": "string"}, "params": {"type": "object"}},
                "required": ["model_name", "params"],
                "additionalProperties": False,
            },
        },
        "required": ["hypothesis", "rationale", "change_type", "target_component", "proposed_change",
                     "expected_direction", "expected_metric_effect", "confidence", "novelty_claim",
                     "risk_notes", "candidate_config"],
        "additionalProperties": False,
    }


def build_proposals_schema(available_models: dict[str, dict[str, Any]], proposal_count: int) -> dict[str, Any]:
    return {"type": "array", "items": build_proposal_variant_schema(available_models), "minItems": 1, "maxItems": max(1, int(proposal_count))}


def build_proposal_variant_schema(available_models: dict[str, dict[str, Any]]) -> dict[str, Any]:
    variants = []
    for model_name, model_config in available_models.items():
        param_schema = {
            name: _build_param_schema(spec)
            for name, spec in model_config.get("search_space", {}).items()
        }
        variants.append({
            "type": "object",
            "properties": {
                "model_name": {"type": "string", "const": model_name},
                "params": {"type": "object", "properties": param_schema, "additionalProperties": False},
                "proposal_family": {"type": "string"},
                "hypothesis": {"type": "string"},
                "expected_delta": {"type": "number"},
            },
            "required": ["model_name", "params", "proposal_family", "hypothesis", "expected_delta"],
            "additionalProperties": False,
        })
    return variants[0] if len(variants) == 1 else {"oneOf": variants}


def format_search_space(search_space: dict[str, dict[str, Any]]) -> str:
    parts: list[str] = []
    for name, spec in search_space.items():
        t = spec.get("type")
        if t == "int":
            parts.append(f"{name}=int[{spec['low']},{spec['high']},step={int(spec.get('step', 1))}]")
        elif t == "float":
            suffix = ",log" if spec.get("log") else ""
            parts.append(f"{name}=float[{spec['low']},{spec['high']}{suffix}]")
        elif t == "categorical":
            choices = ",".join(_format_choice(c) for c in spec.get("choices", []))
            parts.append(f"{name}=categorical[{choices}]")
        else:
            parts.append(f"{name}=unknown")
    return "; ".join(parts)


def estimate_numeric_delta(expected_metric_effect: dict[str, Any]) -> float | None:
    metric = str(expected_metric_effect.get("metric", "")).strip().lower()
    if metric != "rmse":
        return None
    direction = str(expected_metric_effect.get("direction", "")).strip().lower()
    match = re.search(r"(-?\d+(?:\.\d+)?)", str(expected_metric_effect.get("magnitude_estimate", "")))
    if match is None:
        return None
    magnitude = abs(float(match.group(1)))
    return -magnitude if direction == "down" else magnitude


def _build_param_schema(spec: dict[str, Any]) -> dict[str, Any]:
    t = spec.get("type")
    if t == "int":
        schema: dict[str, Any] = {"type": "integer", "minimum": int(spec["low"]), "maximum": int(spec["high"])}
        step = int(spec.get("step", 1))
        if step > 0:
            schema["multipleOf"] = step
        return schema
    if t == "float":
        return {"type": "number", "minimum": float(spec["low"]), "maximum": float(spec["high"])}
    if t == "categorical":
        return {"enum": list(spec.get("choices", []))}
    return {}


def _build_archive_history_summary(archive_records: list[dict[str, Any]]) -> str:
    if not archive_records:
        return "(no prior hypothesis archive entries)"
    lines = []
    for record in archive_records[-5:]:
        hypothesis = str(record.get("hypothesis") or record.get("claim") or "").strip() or "(missing hypothesis)"
        lines.append(f"- {hypothesis} | outcome={record.get('outcome', 'pending')} | actual_delta_rmse={record.get('actual_delta_rmse')}")
    return "\n".join(lines)


def _build_novelty_summary(archive_records: list[dict[str, Any]], diversity_state: dict[str, Any]) -> str:
    return json.dumps({"archive_size": len(archive_records), "historic_family_counts": diversity_state.get("historic_family_counts", {})}, sort_keys=True)


def _compact_family_state(family_state: dict[str, Any]) -> dict[str, Any]:
    return {
        "strongest_active_family": family_state.get("strongest_active_family"),
        "saturated_families": family_state.get("saturated_families", []),
        "underexplored_promising_families": family_state.get("underexplored_promising_families", []),
        "underexplored_weak_families": family_state.get("underexplored_weak_families", []),
        "temporarily_blocked_families": family_state.get("temporarily_blocked_families", []),
    }


def _format_metric(value: Any) -> str:
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return "n/a"


def _format_family_list(families: list[str]) -> str:
    return ", ".join(str(f) for f in families) if families else "none"


def _format_choice(value: Any) -> str:
    if value is None:
        return "null"
    return value if isinstance(value, str) else json.dumps(value)


def _extract_trial_params(trial: dict[str, Any]) -> dict[str, Any]:
    raw = trial.get("params", trial.get("hyperparameters", {}))
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}
    return {}
