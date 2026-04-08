"""
hybrid_proposal_provider.py
Single responsibility: try LLM first, fall back to deterministic if LLM
returns fewer than min_proposals results. Fallback logic lives HERE only.
The research loop never re-implements fallback logic inline.
"""

from __future__ import annotations

import logging

from proposal_engine import Proposal, ProposalContext

logger = logging.getLogger(__name__)


class HybridProposalProvider:
    def __init__(self, llm_provider, deterministic_provider, min_llm_proposals: int = 1):
        self._llm = llm_provider
        self._det = deterministic_provider
        self._min = min_llm_proposals
        self.backend_name = getattr(llm_provider, "backend_name", "hybrid")
        self.llm_config = dict(getattr(llm_provider, "llm_config", {}) or {})

    def is_available(self) -> bool:
        return bool(getattr(self._llm, "is_available", lambda: True)())

    def run_proposal_smoke_test(self, **kwargs):
        if hasattr(self._llm, "run_proposal_smoke_test"):
            return self._llm.run_proposal_smoke_test(**kwargs)
        raise AttributeError("LLM smoke test is not available")

    def summarize_run(self, **kwargs):
        if hasattr(self._llm, "summarize_run"):
            return self._llm.summarize_run(**kwargs)
        raise AttributeError("LLM summary is not available")

    def get_proposals(self, context: ProposalContext, n: int = 3) -> list[Proposal]:
        llm_results = self._llm.get_proposals(context, n)

        if len(llm_results) >= self._min:
            logger.info("HybridProvider: using %d LLM proposals", len(llm_results))
            return llm_results

        logger.warning(
            "HybridProvider: LLM returned %d proposals (min=%d), using deterministic fallback",
            len(llm_results),
            self._min,
        )
        return self._det.get_proposals(context, n)
