"""
proposal_api.py
Single responsibility: public interface for proposal generation.
Orchestrates prompting -> LLM call -> parsing -> gating.
No implementation logic lives here.
"""

from __future__ import annotations

from proposal_engine_impl import (
    Proposal,
    ProposalBackendFailure,
    ProposalContext,
    ProposalEngine as _ProposalEngineImpl,
    ProposalExtractionError,
    ProposalGenerationError,
    ProposalParseFailure,
    ProposalPreflightFailure,
    ProposalProvider,
    ProposalSchemaFailure,
    ProposalSemanticFailure,
    get_proposals,
)
from proposal_gates import filter_proposals
from proposal_parsing import extract_proposals
from proposal_prompting import build_prompt
from policy_layer import PolicyLayer
from llm_backend import extract_text_channels
from proposal_service import ProposalService


class ProposalEngine(_ProposalEngineImpl):
    """Compatibility wrapper that exposes a generate() entry point."""

    def generate(self, context: ProposalContext, n: int = 3) -> list[Proposal]:
        if not isinstance(context, ProposalContext):
            return []
        config = dict(getattr(context, "config", {}) or {})
        available_models = dict(config.get("available_models", {}))
        if not available_models:
            return []
        prompt = build_prompt(context, {**self.llm_config, **config})
        raw_payload = self.backend.generate_text(
            prompt,
            system_prompt="You are a proposal engine for AutoCivil-Lab.",
            model=self._resolve_model_hint(config.get("model_hint")),
        )
        channels = extract_text_channels(raw_payload if isinstance(raw_payload, dict) else {"text": str(raw_payload)})
        raw_text = channels.get("response_text") or channels.get("backend_raw_text") or ""
        parsed = extract_proposals(raw_text)
        return filter_proposals(parsed, context, PolicyLayer({**self.llm_config, **config}))


__all__ = [
    "Proposal",
    "ProposalBackendFailure",
    "ProposalContext",
    "ProposalEngine",
    "ProposalService",
    "ProposalExtractionError",
    "ProposalGenerationError",
    "ProposalParseFailure",
    "ProposalPreflightFailure",
    "ProposalProvider",
    "ProposalSchemaFailure",
    "ProposalSemanticFailure",
    "get_proposals",
]
