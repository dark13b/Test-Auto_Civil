"""
llm_proposal_provider.py
Single responsibility: call the configured LLM backend and return Proposal
objects. All retry, timeout, fallback, and schema-repair logic lives here.
The research loop never imports from llm_backend directly.
"""

from __future__ import annotations

import logging

from proposal_engine import Proposal, ProposalContext, ProposalProvider, ProposalEngine

logger = logging.getLogger(__name__)


class LLMProposalProvider:
    """Implements ProposalProvider using the configured LLM backend."""

    def __init__(self, backend=None, config: dict | None = None):
        if backend is None:
            from llm_backend import resolve_backend

            backend = resolve_backend(config or {})
        self._engine = ProposalEngine(
            backend=backend,
            interaction_log_path=None,
            log_interactions=bool((config or {}).get("llm", {}).get("log_interactions", True)),
            include_no_think_directive=bool((config or {}).get("llm", {}).get("include_no_think_directive", False)),
            llm_config=dict((config or {}).get("llm", {})),
        )
        self._config = config
        self.backend_name = getattr(backend, "backend_name", "unknown")
        self.llm_config = dict((config or {}).get("llm", {}))

    def is_available(self) -> bool:
        return self._engine.is_available()

    def run_proposal_smoke_test(self, **kwargs):
        return self._engine.run_proposal_smoke_test(**kwargs)

    def summarize_run(self, **kwargs):
        return self._engine.summarize_run(**kwargs)

    def get_proposals(self, context: ProposalContext, n: int = 3) -> list[Proposal]:
        try:
            raw_result = self._engine.generate_experiment_proposals(
                available_models=dict(context.config.get("available_models", {})),
                research_brief=dict(context.brief),
                current_best=dict(context.best_metrics),
                experiment_memory=dict(context.config.get("memory_payload", {})),
                diversity_state=dict(context.config.get("diversity_state", {})),
                proposal_count=n,
                trial_history=list(context.recent_results),
                search_progress=dict(context.config.get("search_progress", {})),
                failure_patterns=dict(context.config.get("failure_patterns", {})),
                knowledge_context=str(context.config.get("knowledge_context", "") or ""),
            )
            raw_proposals = list(raw_result or [])
            return [self._to_proposal(r) for r in raw_proposals]
        except Exception as exc:
            logger.warning("LLMProposalProvider failed (%s), returning empty list", exc)
            return []

    def _to_proposal(self, raw: dict) -> Proposal:
        return Proposal(
            hypothesis=raw.get("hypothesis", ""),
            hyperparameters=raw.get("hyperparameters", {}),
            rationale=raw.get("rationale", ""),
            source="llm",
            confidence=float(raw.get("confidence", 0.5)),
            raw=raw,
        )
