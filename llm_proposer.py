"""Backend-agnostic proposal adapter for the legacy search loop."""

from __future__ import annotations

# DEPRECATED - use HybridProposalProvider instead.
# This file is retained for reference only and will be removed in Stage 2.
import warnings

warnings.warn("llm_proposer is deprecated", DeprecationWarning, stacklevel=2)

from pathlib import Path
from typing import Any

from llm_backend import get_llm_config, resolve_backend
from model_routing import resolve_model_for_backend
from proposal_engine import ProposalEngine


class LLMProposer:
    """Compatibility adapter that preserves the legacy search.py interface."""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.llm_config = get_llm_config(config)
        outputs_dir = Path(__file__).resolve().parent / str(config.get("paths", {}).get("outputs_dir", "outputs"))
        outputs_dir.mkdir(parents=True, exist_ok=True)
        self.interaction_log_path = outputs_dir / str(
            self.llm_config.get("interaction_log_filename", "llm_interactions.jsonl")
        )
        self.backend = resolve_backend(config)
        self.engine = ProposalEngine(
            backend=self.backend,
            interaction_log_path=self.interaction_log_path,
            log_interactions=bool(self.llm_config.get("log_interactions", True)),
            include_no_think_directive=bool(self.llm_config.get("include_no_think_directive", False)),
            llm_config=self.llm_config,
        )
        self.fast_model, self.smart_model, self.compact_model = self._resolve_model_hints()
        self.consecutive_invalid_responses = 0

    def is_available(self) -> bool:
        return self.engine.is_available()

    def propose(
        self,
        trial_history: list[dict[str, Any]],
        available_models: dict[str, dict[str, Any]],
        current_best: dict[str, Any] | None,
        use_smart_model: bool = False,
        search_progress: dict[str, Any] | None = None,
        research_brief: dict[str, Any] | None = None,
        experiment_memory: dict[str, Any] | None = None,
        diversity_state: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        proposals = self.engine.generate_experiment_proposals(
            available_models=available_models,
            research_brief=research_brief or {},
            current_best=current_best or {},
            experiment_memory=experiment_memory or {"runs": [], "accepted_experiments": []},
            diversity_state=diversity_state or {"historic_family_counts": {}},
            proposal_count=1,
            trial_history=trial_history,
            search_progress=search_progress or {},
            model_hint=self.smart_model if use_smart_model else self.fast_model,
        )
        if not proposals:
            self.consecutive_invalid_responses += 1
            return None
        self.consecutive_invalid_responses = 0
        return proposals[0]

    def propose_ensemble(
        self,
        top_trials: list[dict[str, Any]],
        available_models: dict[str, dict[str, Any]],
    ) -> list[dict[str, Any]] | None:
        proposals = self.engine.generate_ensemble_selection(
            top_trials=top_trials,
            available_models=available_models,
            max_items=3,
            model_hint=self.smart_model,
        )
        if not proposals:
            self.consecutive_invalid_responses += 1
            return None
        self.consecutive_invalid_responses = 0
        return proposals

    def generate_hypotheses(
        self,
        *,
        research_brief: dict[str, Any],
        current_best: dict[str, Any],
        experiment_memory: dict[str, Any],
        limit: int = 5,
    ) -> list[str]:
        return self.engine.generate_hypotheses(
            research_brief=research_brief,
            current_best=current_best,
            experiment_memory=experiment_memory,
            limit=limit,
            model_hint=self.smart_model,
        )

    def generate_feature_ideas(
        self,
        *,
        research_brief: dict[str, Any],
        current_best: dict[str, Any],
        limit: int = 5,
    ) -> list[str]:
        return self.engine.generate_feature_ideas(
            research_brief=research_brief,
            current_best=current_best,
            limit=limit,
            model_hint=self.smart_model,
        )

    def generate_search_space_suggestions(
        self,
        *,
        available_models: dict[str, dict[str, Any]],
        research_brief: dict[str, Any],
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        return self.engine.generate_search_space_suggestions(
            available_models=available_models,
            research_brief=research_brief,
            limit=limit,
            model_hint=self.smart_model,
        )

    def summarize_run(
        self,
        *,
        research_brief: dict[str, Any],
        final_metrics: dict[str, Any],
        acceptance: dict[str, Any],
    ) -> dict[str, Any]:
        return self.engine.summarize_run(
            research_brief=research_brief,
            final_metrics=final_metrics,
            acceptance=acceptance,
            model_hint=self.smart_model,
        )

    def _resolve_model_hints(self) -> tuple[str | None, str | None, str | None]:
        backend_mode = str(self.llm_config.get("backend_mode", "ollama")).lower()
        resolved_default = resolve_model_for_backend(None, backend_mode, self.config)
        if backend_mode == "openai":
            return resolved_default, resolved_default, resolved_default
        ollama_config = self.llm_config.get("ollama", {})
        compact_models = [str(item) for item in self.llm_config.get("compact_prompt_models", []) if str(item).strip()]
        compact_model = compact_models[0] if compact_models else str(ollama_config.get("fast_model", resolved_default))
        smart_model = str(ollama_config.get("smart_model", resolved_default))
        return resolved_default, smart_model, compact_model
