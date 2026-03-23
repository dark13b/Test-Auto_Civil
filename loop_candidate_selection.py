"""Candidate selection helpers for the AutoCivil-Lab research loop."""

from __future__ import annotations

import logging
from typing import Any

from proposal_engine import ProposalContext, ProposalProvider
from research_protocol import filter_diverse_candidates

from loop_artifacts import _build_diversity_state, _normalize_llm_candidates


LOGGER = logging.getLogger("loop_candidate_selection")


def select_scout_candidates(
    *,
    brief: dict[str, Any],
    lab_state: dict[str, Any],
    available_models: dict[str, dict[str, Any]],
    memory_payload: dict[str, Any],
    current_best: dict[str, Any],
    scout_limit: int,
    family_limit: int,
    current_run_signatures: set[tuple[str, str]],
    proposal_engine: ProposalProvider | None,
    trial_history: list[dict[str, Any]] | None = None,
    search_progress: dict[str, Any] | None = None,
    failure_patterns: dict[str, Any] | None = None,
    knowledge_context: str | None = None,
    archive_records: list[dict[str, Any]] | None = None,
    preflight_result: dict[str, Any] | None = None,
    exploit_delta_ratio: float = 0.15,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if proposal_engine is None:
        return [], {
            "proposal_mode": "fallback_used",
            "proposal_status": "fallback_used",
            "proposal_backend": "fallback",
            "proposal_count": 0,
            "proposal_model": None,
            "prompt_variant": None,
            "proposal_error": None,
        }
    context = {
        "brief": brief,
        "lab_state": lab_state,
        "available_models": available_models,
        "memory_payload": memory_payload,
        "current_best": current_best,
        "diversity_state": _build_diversity_state(memory_payload),
        "scout_limit": scout_limit,
        "trial_history": trial_history or [],
        "search_progress": search_progress or {},
        "failure_patterns": failure_patterns or {},
        "knowledge_context": knowledge_context or "",
        "archive_records": archive_records or [],
        "exploit_delta_ratio": exploit_delta_ratio,
    }
    proposals = proposal_engine.get_proposals(
        ProposalContext(
            brief=brief,
            recent_results=list(trial_history or []),
            best_metrics=current_best,
            config=context,
            trial_budget_remaining=int(scout_limit),
        ),
        n=scout_limit,
    )
    if proposals:
        llm_candidates = _normalize_llm_candidates(list(proposals), available_models)
        llm_candidates = filter_diverse_candidates(
            candidates=llm_candidates,
            memory_payload=memory_payload,
            current_run_signatures=current_run_signatures,
            family_limit=family_limit,
            scout_limit=scout_limit,
        )
        if llm_candidates:
            return llm_candidates, {
                "proposal_mode": "llm",
                "proposal_status": "llm_success",
                "proposal_backend": "proposal_provider",
                "proposal_count": len(llm_candidates),
                "proposal_model": None,
                "prompt_variant": None,
                "proposal_error": None,
            }
    return [], {
        "proposal_mode": "fallback_used",
        "proposal_status": "fallback_used",
        "proposal_backend": "fallback",
        "proposal_count": 0,
        "proposal_model": None,
        "prompt_variant": None,
        "proposal_error": None,
    }
