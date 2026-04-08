"""
policy_layer.py
Single responsibility: ALL proposal acceptance decisions live here.
No other file may independently accept or reject a proposal.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from research_protocol import gate_proposal, gate_research_proposal


@dataclass
class PolicyDecision:
    accepted: bool
    reason: str
    scores: dict[str, Any] = field(default_factory=dict)


class PolicyLayer:
    def __init__(self, config: dict):
        self._cfg = dict(config or {})

    def evaluate(self, proposal: Any, context: Any) -> PolicyDecision:
        ctx = context if isinstance(context, dict) else getattr(context, "__dict__", {})
        sem = self._check_semantic_validity(proposal, ctx)
        if not sem.accepted:
            return sem
        nov = self._check_novelty(proposal, ctx)
        if not nov.accepted:
            return nov
        return self._check_diversity(proposal, ctx)

    def _check_semantic_validity(self, proposal: Any, context: dict[str, Any]) -> PolicyDecision:
        if not isinstance(proposal, dict):
            return PolicyDecision(False, "proposal must be a dict", {"semantic_validity": 0.0})
        if "candidate_config" in proposal:
            result = gate_research_proposal(
                proposal=proposal,
                available_models=dict(context.get("available_models", {})),
                current_run_signatures=set(context.get("current_run_signatures", set())),
                memory_payload=dict(context.get("memory_payload", {})),
                trial_history=list(context.get("trial_history", [])),
                family_state=dict(context.get("family_state", {})),
                duplicate_settings=dict(context.get("duplicate_settings", {})),
                current_best=context.get("current_best"),
                archive_records=list(context.get("archive_records", [])),
            )
            return PolicyDecision(bool(result.get("accepted")), _reason_text(result), dict(result))
        result = gate_proposal(
            proposal=proposal,
            available_models=dict(context.get("available_models", {})),
            current_run_signatures=set(context.get("current_run_signatures", set())),
            memory_payload=dict(context.get("memory_payload", {})),
            trial_history=list(context.get("trial_history", [])),
            family_state=dict(context.get("family_state", {})),
            duplicate_settings=dict(context.get("duplicate_settings", {})),
        )
        return PolicyDecision(bool(result.get("accepted")), _reason_text(result), dict(result))

    def _check_novelty(self, proposal: Any, context: dict[str, Any]) -> PolicyDecision:
        return self._check_semantic_validity(proposal, context)

    def _check_diversity(self, proposal: Any, context: dict[str, Any]) -> PolicyDecision:
        return self._check_semantic_validity(proposal, context)


def _reason_text(result: dict[str, Any]) -> str:
    reason = result.get("semantic_rejection_reason") or result.get("reason") or {}
    if isinstance(reason, dict):
        return str(reason.get("message", "accepted" if result.get("accepted") else "rejected"))
    return str(reason or ("accepted" if result.get("accepted") else "rejected"))
