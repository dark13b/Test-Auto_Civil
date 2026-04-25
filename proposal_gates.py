"""Single responsibility: proposal gating, semantic validity, novelty scoring, and accept/reject decisions."""

from __future__ import annotations

import re
from typing import Any

from proposal_parsing import validate_proposal
from proposal_prompting import estimate_numeric_delta
from policy_layer import PolicyLayer


class ProposalGenerationError(RuntimeError):
    def __init__(self, message: str, *, failure_kind: str, interaction: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.failure_kind = failure_kind
        self.interaction = dict(interaction or {})


class ProposalBackendFailure(ProposalGenerationError):
    pass


class ProposalParseFailure(ProposalGenerationError):
    pass


class ProposalSchemaFailure(ProposalGenerationError):
    pass


class ProposalSemanticFailure(ProposalGenerationError):
    pass


class ProposalPreflightFailure(ProposalGenerationError):
    pass


_CHANGE_TYPES = {"hyperparameter", "feature", "preprocessing", "model_family", "objective", "sampling", "other"}
_EXPECTED_DIRECTIONS = {"improve", "worsen_risk", "uncertain"}
_METRIC_DIRECTIONS = {"up", "down"}
_REQUIRED_TEXT_FIELDS = ["hypothesis", "rationale", "change_type", "target_component", "proposed_change", "expected_direction", "novelty_claim", "risk_notes"]
_ALLOWED_KEYS = set(_REQUIRED_TEXT_FIELDS) | {"expected_metric_effect", "confidence", "candidate_config"}


def validate_research_proposal(
    payload: Any,
    *,
    available_models: dict[str, dict[str, Any]],
    backend_name: str = "unknown",
) -> dict[str, Any]:
    """Validate a parsed research proposal dict against the strict schema."""
    if not isinstance(payload, dict):
        raise ProposalSchemaFailure("Proposal payload must be a JSON object.", failure_kind="llm_schema_failure", interaction={"backend": backend_name})

    unexpected = sorted(set(payload.keys()) - _ALLOWED_KEYS)
    if unexpected:
        raise ProposalSchemaFailure(f"Proposal contains unexpected fields: {', '.join(unexpected)}.", failure_kind="llm_schema_failure", interaction={"backend": backend_name})

    normalized: dict[str, Any] = {}
    for field_name in _REQUIRED_TEXT_FIELDS:
        value = payload.get(field_name)
        if not isinstance(value, str) or not value.strip():
            raise ProposalSchemaFailure(f"Proposal field '{field_name}' must be a non-empty string.", failure_kind="llm_schema_failure", interaction={"backend": backend_name})
        normalized[field_name] = value.strip()

    if normalized["change_type"] not in _CHANGE_TYPES:
        raise ProposalSchemaFailure("change_type is outside the allowed enum.", failure_kind="llm_schema_failure", interaction={"backend": backend_name})
    if normalized["expected_direction"] not in _EXPECTED_DIRECTIONS:
        raise ProposalSchemaFailure("expected_direction is outside the allowed enum.", failure_kind="llm_schema_failure", interaction={"backend": backend_name})

    try:
        confidence = float(payload.get("confidence"))
    except (TypeError, ValueError) as exc:
        raise ProposalSchemaFailure("Proposal confidence must be numeric.", failure_kind="llm_schema_failure", interaction={"backend": backend_name}) from exc
    if not (0.0 <= confidence <= 1.0):
        raise ProposalSchemaFailure("Proposal confidence must be bounded to [0,1].", failure_kind="llm_schema_failure", interaction={"backend": backend_name})
    normalized["confidence"] = confidence

    eme = payload.get("expected_metric_effect")
    if not isinstance(eme, dict):
        raise ProposalSchemaFailure("Proposal expected_metric_effect must be an object.", failure_kind="llm_schema_failure", interaction={"backend": backend_name})
    unexpected_eme = sorted(set(eme.keys()) - {"metric", "direction", "magnitude_estimate"})
    if unexpected_eme:
        raise ProposalSchemaFailure(f"expected_metric_effect contains unexpected fields: {', '.join(unexpected_eme)}.", failure_kind="llm_schema_failure", interaction={"backend": backend_name})
    for key in ("metric", "direction", "magnitude_estimate"):
        if not isinstance(eme.get(key), str) or not eme[key].strip():
            raise ProposalSchemaFailure(f"expected_metric_effect.{key} must be a non-empty string.", failure_kind="llm_schema_failure", interaction={"backend": backend_name})
    if eme["direction"] not in _METRIC_DIRECTIONS:
        raise ProposalSchemaFailure("expected_metric_effect.direction must be 'up' or 'down'.", failure_kind="llm_schema_failure", interaction={"backend": backend_name})
    normalized["expected_metric_effect"] = {k: eme[k].strip() for k in ("metric", "direction", "magnitude_estimate")}

    candidate_config = payload.get("candidate_config")
    if not isinstance(candidate_config, dict):
        raise ProposalSchemaFailure("candidate_config must be an object.", failure_kind="llm_schema_failure", interaction={"backend": backend_name})
    unexpected_cc = sorted(set(candidate_config.keys()) - {"model_name", "params"})
    if unexpected_cc:
        raise ProposalSchemaFailure(f"candidate_config contains unexpected fields: {', '.join(unexpected_cc)}.", failure_kind="llm_schema_failure", interaction={"backend": backend_name})

    executable = validate_proposal(
        {
            "model_name": candidate_config.get("model_name"),
            "params": candidate_config.get("params"),
            "proposal_family": payload.get("change_type", "llm-generated"),
            "hypothesis": payload.get("hypothesis"),
            "expected_delta": estimate_numeric_delta(normalized["expected_metric_effect"]),
        },
        available_models,
    )
    if executable is None:
        raise ProposalSchemaFailure("candidate_config does not map to a valid executable model configuration.", failure_kind="llm_schema_failure", interaction={"backend": backend_name})
    normalized["candidate_config"] = {"model_name": executable["model_name"], "params": executable["params"]}
    return normalized


def normalize_research_proposal_to_candidate(
    *,
    proposal: dict[str, Any],
    available_models: dict[str, dict[str, Any]],
    candidate_index: int,
    backend: str,
    model: str,
    prompt_hash: str,
) -> dict[str, Any]:
    candidate_config = dict(proposal["candidate_config"])
    model_name = str(candidate_config["model_name"])
    return {
        "experiment_id": f"llm-scout-{candidate_index:03d}",
        "stage": "scout",
        "model_name": model_name,
        "display_name": str(available_models[model_name].get("display_name", model_name)),
        "proposal_family": f"llm-{proposal['change_type']}",
        "hypothesis": proposal["hypothesis"],
        "params": dict(candidate_config["params"]),
        "proposal_source": "llm",
        "proposal_status": "llm_success",
        "proposal_backend": backend,
        "proposal_model": model,
        "prompt_hash": prompt_hash,
        "confidence": float(proposal["confidence"]),
        "expected_delta_rmse": estimate_numeric_delta(proposal["expected_metric_effect"]),
        "research_proposal": proposal,
    }


def filter_proposals(parsed: Any, context: Any, policy: PolicyLayer) -> list[dict[str, Any]]:
    proposals = parsed if isinstance(parsed, list) else [parsed]
    accepted: list[dict[str, Any]] = []
    for proposal in proposals:
        decision = policy.evaluate(proposal, context)
        if decision.accepted and isinstance(proposal, dict):
            accepted.append(proposal)
    return accepted
