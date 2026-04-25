"""Candidate selection helpers for the AutoCivil-Lab research loop."""

from __future__ import annotations

import logging
from typing import Any

import research_lab

from proposal_contract import ProposalRequest
from proposal_engine import ProposalContext, ProposalProvider, get_proposals
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
    allow_deterministic_fallback: bool = False,
    exploit_delta_ratio: float = 0.15,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    del preflight_result
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
    proposal_context = ProposalContext(
        brief=brief,
        recent_results=list(trial_history or []),
        best_metrics=current_best,
        config=context,
        trial_budget_remaining=int(scout_limit),
    )
    if hasattr(proposal_engine, "generate_candidates"):
        batch = proposal_engine.generate_candidates(
            ProposalRequest(
                brief=brief,
                lab_state=lab_state,
                available_models=available_models,
                memory_payload=memory_payload,
                current_best=current_best,
                scout_limit=scout_limit,
                trial_history=list(trial_history or []),
                search_progress=dict(search_progress or {}),
                failure_patterns=dict(failure_patterns or {}),
                knowledge_context=knowledge_context or "",
                archive_records=list(archive_records or []),
                exploit_delta_ratio=exploit_delta_ratio,
            )
        )
        llm_candidates = list(batch.candidates)
        metadata = dict(batch.metadata)
    elif hasattr(proposal_engine, "get_proposals"):
        llm_candidates = proposal_engine.get_proposals(proposal_context, n=scout_limit)
        metadata = {
            "proposal_mode": "llm",
            "proposal_status": "llm_success",
            "proposal_backend": "proposal_provider",
            "proposal_count": len(llm_candidates),
            "proposal_model": None,
            "prompt_variant": None,
            "proposal_error": None,
        }
    elif hasattr(proposal_engine, "generate_research_proposals"):
        try:
            llm_result = proposal_engine.generate_research_proposals(
                available_models=available_models,
                research_brief=brief,
                current_best=current_best,
                experiment_memory=memory_payload,
                diversity_state=_build_diversity_state(memory_payload),
                proposal_count=scout_limit,
                trial_history=trial_history or [],
                search_progress=search_progress or {},
                failure_patterns=failure_patterns or {},
                knowledge_context=knowledge_context or "",
                archive_records=archive_records or [],
                model_hint=None,
            )
        except Exception as exc:
            if allow_deterministic_fallback and exc.__class__.__name__ in {"ProposalExtractionError", "ProposalParseFailure"}:
                llm_candidates = []
            else:
                raise
        else:
            llm_candidates = list(llm_result.get("proposals", [])) if isinstance(llm_result, dict) else list(llm_result or [])
        metadata = {
            "proposal_mode": "llm",
            "proposal_status": str(llm_result.get("status", "llm_success")) if "llm_result" in locals() and isinstance(llm_result, dict) else "llm_success",
            "proposal_backend": str(llm_result.get("backend")) if "llm_result" in locals() and isinstance(llm_result, dict) else "proposal_provider",
            "proposal_count": len(llm_candidates),
            "proposal_model": llm_result.get("model") if "llm_result" in locals() and isinstance(llm_result, dict) else None,
            "prompt_variant": llm_result.get("prompt_variant") if "llm_result" in locals() and isinstance(llm_result, dict) else None,
            "proposal_error": llm_result.get("error") if "llm_result" in locals() and isinstance(llm_result, dict) else None,
        }
    else:
        llm_candidates = get_proposals(
            context,
            {"llm": dict(getattr(proposal_engine, "llm_config", {}) or {})},
            logger=LOGGER,
            n=scout_limit,
        )
        metadata = {
            "proposal_mode": "llm",
            "proposal_status": "llm_success",
            "proposal_backend": "proposal_provider",
            "proposal_count": len(llm_candidates),
            "proposal_model": None,
            "prompt_variant": None,
            "proposal_error": None,
        }
    llm_candidates = [
        candidate.__dict__ if hasattr(candidate, "__dict__") and not isinstance(candidate, dict) else candidate
        for candidate in llm_candidates
    ]
    llm_candidates = _normalize_llm_candidates(list(llm_candidates), available_models)
    llm_candidates = filter_diverse_candidates(
        candidates=llm_candidates,
        memory_payload=memory_payload,
        current_run_signatures=current_run_signatures,
        family_limit=family_limit,
        scout_limit=scout_limit,
    )
    if llm_candidates:
        metadata["proposal_count"] = len(llm_candidates)
        return llm_candidates, metadata
    if allow_deterministic_fallback:
        deterministic_candidates = research_lab.scout_experiments(
            brief=brief,
            lab_state=lab_state,
            available_models=available_models,
            experiment_memory=memory_payload,
            current_best=current_best,
            scout_limit=max(1, scout_limit),
            exploit_delta_ratio=exploit_delta_ratio,
        )
        deterministic_candidates = filter_diverse_candidates(
            candidates=deterministic_candidates,
            memory_payload=memory_payload,
            current_run_signatures=current_run_signatures,
            family_limit=family_limit,
            scout_limit=scout_limit,
        )
        return deterministic_candidates, {
            "proposal_mode": "fallback_used",
            "proposal_status": "fallback_used",
            "proposal_backend": "fallback",
            "proposal_count": len(deterministic_candidates),
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
