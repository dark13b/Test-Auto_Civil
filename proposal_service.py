"""Canonical proposal generation service with explicit provider modes."""

from __future__ import annotations

from typing import Any

import research_lab

from proposal_backend_policy import resolve_backend_policy
from proposal_contract import ProposalBatch, ProposalRequest, ProviderExecution
from proposal_validation import normalize_deterministic_candidate, normalize_llm_candidate


def proposal_request_from_context(context: Any, *, limit: int) -> ProposalRequest:
    config = dict(getattr(context, "config", {}) or {})
    return ProposalRequest(
        brief=dict(getattr(context, "brief", {}) or config.get("brief", {})),
        lab_state=dict(config.get("lab_state", {})),
        available_models=dict(config.get("available_models", {})),
        memory_payload=dict(config.get("memory_payload", {})),
        current_best=dict(getattr(context, "best_metrics", {}) or config.get("current_best", {})),
        scout_limit=max(1, int(limit)),
        trial_history=list(getattr(context, "recent_results", []) or config.get("trial_history", [])),
        search_progress=dict(config.get("search_progress", {})),
        failure_patterns=dict(config.get("failure_patterns", {})),
        knowledge_context=str(config.get("knowledge_context", "") or ""),
        archive_records=list(config.get("archive_records", [])),
        exploit_delta_ratio=float(config.get("exploit_delta_ratio", 0.15)),
    )


class DeterministicProposalProvider:
    mode = "deterministic"

    def __init__(self, search_module=None, config: dict[str, Any] | None = None) -> None:
        del search_module
        self._config = dict(config or {})
        self.backend_name = "deterministic"

    def is_available(self) -> bool:
        return True

    def generate(self, request: ProposalRequest, *, limit: int) -> ProviderExecution:
        candidates = research_lab.scout_experiments(
            brief=dict(request.brief),
            lab_state=dict(request.lab_state),
            available_models=dict(request.available_models),
            experiment_memory=dict(request.memory_payload),
            current_best=dict(request.current_best),
            scout_limit=max(1, int(limit)),
            exploit_delta_ratio=float(request.exploit_delta_ratio),
        )
        return ProviderExecution(
            proposals=list(candidates),
            status="deterministic_success",
            backend="deterministic",
            model=None,
            prompt_variant=None,
            error=None,
        )

    def get_proposals(self, context: Any, n: int = 3) -> list[dict[str, Any]]:
        request = proposal_request_from_context(context, limit=n)
        return list(self.generate(request, limit=n).proposals)


class LLMProposalProvider:
    mode = "llm"

    def __init__(self, backend=None, config: dict[str, Any] | None = None, engine=None) -> None:
        self._config = dict(config or {})
        self.llm_config = dict(self._config.get("llm", {}))
        if engine is None:
            from llm_backend import resolve_backend
            from proposal_engine_impl import ProposalEngine

            resolved_backend = backend or resolve_backend(self._config)
            engine = ProposalEngine(
                backend=resolved_backend,
                interaction_log_path=None,
                log_interactions=bool(self.llm_config.get("log_interactions", True)),
                include_no_think_directive=bool(self.llm_config.get("include_no_think_directive", False)),
                llm_config=self.llm_config,
            )
        self._engine = engine
        self.backend_name = getattr(engine, "backend_name", getattr(backend, "backend_name", "unknown"))

    def is_available(self) -> bool:
        return bool(getattr(self._engine, "is_available", lambda: False)())

    def generate(self, request: ProposalRequest, *, limit: int) -> ProviderExecution:
        if not self.is_available():
            return ProviderExecution(
                proposals=[],
                status="llm_backend_failure",
                backend=self.backend_name,
                model=None,
                prompt_variant=None,
                error="Configured LLM backend is unavailable.",
            )
        try:
            result = self._engine.generate_research_proposals(
                available_models=dict(request.available_models),
                research_brief=dict(request.brief),
                current_best=dict(request.current_best),
                experiment_memory=dict(request.memory_payload),
                diversity_state={},
                proposal_count=max(1, int(limit)),
                trial_history=list(request.trial_history),
                search_progress=dict(request.search_progress),
                failure_patterns=dict(request.failure_patterns),
                knowledge_context=str(request.knowledge_context or ""),
                archive_records=list(request.archive_records),
                model_hint=None,
            )
        except Exception as exc:
            interaction = dict(getattr(exc, "interaction", {}) or {})
            return ProviderExecution(
                proposals=[],
                status=str(getattr(exc, "failure_kind", "llm_backend_failure")),
                backend=str(interaction.get("backend", self.backend_name)),
                model=interaction.get("model"),
                prompt_variant=interaction.get("prompt_variant"),
                error=str(exc),
            )
        return ProviderExecution(
            proposals=list(result.get("proposals", [])),
            status=str(result.get("status", "llm_success")),
            backend=str(result.get("backend", self.backend_name)),
            model=result.get("model"),
            prompt_variant=result.get("prompt_variant"),
            error=None,
        )

    def get_proposals(self, context: Any, n: int = 3) -> list[dict[str, Any]]:
        request = proposal_request_from_context(context, limit=n)
        return list(self.generate(request, limit=n).proposals)

    def run_proposal_smoke_test(self, **kwargs: Any) -> dict[str, Any]:
        return self._engine.run_proposal_smoke_test(**kwargs)

    def summarize_run(self, **kwargs: Any) -> dict[str, Any]:
        return self._engine.summarize_run(**kwargs)


class HybridProposalProvider:
    mode = "hybrid"

    def __init__(self, llm_provider: LLMProposalProvider, deterministic_provider: DeterministicProposalProvider, min_llm_proposals: int = 1) -> None:
        self._llm = llm_provider
        self._deterministic = deterministic_provider
        self._minimum = max(1, int(min_llm_proposals))
        self.backend_name = getattr(llm_provider, "backend_name", "hybrid")
        self.llm_config = dict(getattr(llm_provider, "llm_config", {}) or {})

    def is_available(self) -> bool:
        return self._llm.is_available() or self._deterministic.is_available()

    def generate(self, request: ProposalRequest, *, limit: int) -> ProviderExecution:
        llm_execution = self._llm.generate(request, limit=limit)
        if len(llm_execution.proposals) >= self._minimum:
            return llm_execution
        deterministic_execution = self._deterministic.generate(request, limit=limit)
        return ProviderExecution(
            proposals=list(deterministic_execution.proposals),
            status="fallback_used",
            backend="fallback",
            model=None,
            prompt_variant=llm_execution.prompt_variant,
            error=llm_execution.error,
        )

    def get_proposals(self, context: Any, n: int = 3) -> list[dict[str, Any]]:
        request = proposal_request_from_context(context, limit=n)
        return list(self.generate(request, limit=n).proposals)

    def run_proposal_smoke_test(self, **kwargs: Any) -> dict[str, Any]:
        return self._llm.run_proposal_smoke_test(**kwargs)

    def summarize_run(self, **kwargs: Any) -> dict[str, Any]:
        return self._llm.summarize_run(**kwargs)


class ProposalService:
    """Single entry point for deterministic, llm, and hybrid proposal generation."""

    def __init__(
        self,
        *,
        config: dict[str, Any] | None = None,
        deterministic_provider=None,
        llm_provider=None,
    ) -> None:
        self._config = dict(config or {})
        self.llm_config = dict(self._config.get("llm", {}))
        self._deterministic_provider = deterministic_provider or DeterministicProposalProvider(config=self._config)
        self._llm_provider = llm_provider or LLMProposalProvider(config=self._config)
        self._hybrid_provider = HybridProposalProvider(
            llm_provider=self._llm_provider,
            deterministic_provider=self._deterministic_provider,
            min_llm_proposals=int(self.llm_config.get("min_proposals", 1)),
        )
        resolution = resolve_backend_policy(self._config)
        self.backend_name = resolution.backend_mode or "deterministic"

    def is_available(self) -> bool:
        resolution = resolve_backend_policy(self._config)
        if resolution.proposal_mode == "deterministic":
            return True
        if resolution.proposal_mode == "hybrid":
            return self._hybrid_provider.is_available()
        return self._llm_provider.is_available()

    def generate_candidates(self, request: ProposalRequest) -> ProposalBatch:
        resolution = resolve_backend_policy(self._config)
        limit = max(1, int(request.scout_limit))
        if resolution.proposal_mode == "deterministic":
            execution = self._deterministic_provider.generate(request, limit=limit)
            candidates = [
                normalize_deterministic_candidate(
                    candidate,
                    available_models=request.available_models,
                    candidate_index=index,
                    proposal_source="deterministic",
                    proposal_status="deterministic_success",
                    backend="deterministic",
                )
                for index, candidate in enumerate(execution.proposals, start=1)
            ]
            return ProposalBatch(
                candidates=candidates,
                metadata={
                    "proposal_mode": "deterministic",
                    "proposal_status": execution.status,
                    "proposal_backend": "deterministic",
                    "proposal_model": None,
                    "prompt_variant": execution.prompt_variant,
                    "proposal_error": execution.error,
                    "provider_path": "deterministic",
                },
            )

        if resolution.proposal_mode == "llm":
            execution = self._llm_provider.generate(request, limit=limit)
            candidates = [
                normalize_llm_candidate(
                    candidate,
                    available_models=request.available_models,
                    candidate_index=index,
                    backend=str(execution.backend or resolution.backend_mode or "unknown"),
                    model=execution.model or resolution.model_name,
                )
                for index, candidate in enumerate(execution.proposals, start=1)
            ]
            return ProposalBatch(
                candidates=candidates,
                metadata={
                    "proposal_mode": "llm",
                    "proposal_status": execution.status,
                    "proposal_backend": execution.backend or resolution.backend_mode,
                    "proposal_model": execution.model or resolution.model_name,
                    "prompt_variant": execution.prompt_variant,
                    "proposal_error": execution.error,
                    "provider_path": "llm",
                },
            )

        llm_execution = self._llm_provider.generate(request, limit=limit)
        if llm_execution.proposals:
            candidates = [
                normalize_llm_candidate(
                    candidate,
                    available_models=request.available_models,
                    candidate_index=index,
                    backend=str(llm_execution.backend or resolution.backend_mode or "unknown"),
                    model=llm_execution.model or resolution.model_name,
                )
                for index, candidate in enumerate(llm_execution.proposals, start=1)
            ]
            return ProposalBatch(
                candidates=candidates,
                metadata={
                    "proposal_mode": "hybrid",
                    "proposal_status": llm_execution.status,
                    "proposal_backend": llm_execution.backend or resolution.backend_mode,
                    "proposal_model": llm_execution.model or resolution.model_name,
                    "prompt_variant": llm_execution.prompt_variant,
                    "proposal_error": llm_execution.error,
                    "provider_path": "llm",
                },
            )

        deterministic_execution = self._deterministic_provider.generate(request, limit=limit)
        candidates = [
            normalize_deterministic_candidate(
                candidate,
                available_models=request.available_models,
                candidate_index=index,
                proposal_source="deterministic_fallback",
                proposal_status="fallback_used",
                backend="fallback",
            )
            for index, candidate in enumerate(deterministic_execution.proposals, start=1)
        ]
        return ProposalBatch(
            candidates=candidates,
            metadata={
                "proposal_mode": "hybrid",
                "proposal_status": "fallback_used",
                "proposal_backend": "fallback",
                "proposal_model": None,
                "prompt_variant": llm_execution.prompt_variant,
                "proposal_error": llm_execution.error,
                "provider_path": "deterministic",
                "fallback_from_backend": llm_execution.backend or resolution.backend_mode,
                "fallback_from_model": llm_execution.model or resolution.model_name,
                "fallback_from_status": llm_execution.status,
            },
        )

    def get_proposals(self, context: Any, n: int = 3) -> list[dict[str, Any]]:
        request = proposal_request_from_context(context, limit=n)
        return list(self.generate_candidates(request).candidates)

    def generate_research_proposals(
        self,
        *,
        available_models: dict[str, dict[str, Any]],
        research_brief: dict[str, Any],
        current_best: dict[str, Any],
        experiment_memory: dict[str, Any],
        diversity_state: dict[str, Any],
        proposal_count: int,
        trial_history: list[dict[str, Any]] | None = None,
        search_progress: dict[str, Any] | None = None,
        failure_patterns: dict[str, Any] | None = None,
        knowledge_context: str | None = None,
        archive_records: list[dict[str, Any]] | None = None,
        model_hint: str | None = None,
    ) -> dict[str, Any]:
        del diversity_state, model_hint
        request = ProposalRequest(
            brief=dict(research_brief),
            lab_state={},
            available_models=dict(available_models),
            memory_payload=dict(experiment_memory),
            current_best=dict(current_best),
            scout_limit=max(1, int(proposal_count)),
            trial_history=list(trial_history or []),
            search_progress=dict(search_progress or {}),
            failure_patterns=dict(failure_patterns or {}),
            knowledge_context=str(knowledge_context or ""),
            archive_records=list(archive_records or []),
        )
        batch = self.generate_candidates(request)
        return {
            "status": batch.metadata["proposal_status"],
            "backend": batch.metadata["proposal_backend"],
            "model": batch.metadata["proposal_model"],
            "prompt_variant": batch.metadata["prompt_variant"],
            "proposals": list(batch.candidates),
            "mode": batch.metadata["proposal_mode"],
            "provider_path": batch.metadata["provider_path"],
            "error": batch.metadata.get("proposal_error"),
        }

    def run_proposal_smoke_test(self, **kwargs: Any) -> dict[str, Any]:
        resolution = resolve_backend_policy(self._config)
        if resolution.proposal_mode == "deterministic" or not self.llm_config.get("enabled", False):
            return {
                "ok": False,
                "status": "aborted_due_to_preflight_failure",
                "backend": "disabled",
                "model": None,
                "prompt_hash": None,
                "proposal_preview": None,
                "error": "LLM proposal engine is disabled by config.",
                "extracted_from_channel": "response",
                "visible_response_empty": True,
                "parse_success": False,
                "schema_success": False,
                "repair_used": False,
                "latency_seconds": None,
                "failure_class": "llm_backend_failure",
            }
        return self._llm_provider.run_proposal_smoke_test(**kwargs)

    def summarize_run(self, **kwargs: Any) -> dict[str, Any]:
        resolution = resolve_backend_policy(self._config)
        if resolution.proposal_mode == "deterministic" or not self.llm_config.get("tasks", {}).get("summary_enabled", True):
            return {"backend": "disabled", "available": False, "summary": ""}
        return self._llm_provider.summarize_run(**kwargs)

    @property
    def config(self) -> dict[str, Any]:
        return dict(self._config)
