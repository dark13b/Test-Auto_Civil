"""Repository-aware proposal generation on top of provider backends."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import logging
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

import research_lab

from llm_backend import BackendUnavailableError, LLMBackend, extract_text_channels, resolve_backend, resolve_prompt_variant
from model_routing import resolve_model_for_backend
from research_protocol import build_family_state_summary, gate_proposal, gate_research_proposal


LOGGER = logging.getLogger("proposal_engine")
JSON_FENCE_PATTERN = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)
INVALID_PARAM = object()


@dataclass
class ProposalContext:
    """Everything the proposer needs to generate candidates."""

    brief: dict
    recent_results: list[dict]
    best_metrics: dict
    config: dict
    trial_budget_remaining: int


@dataclass
class Proposal:
    """One candidate experiment suggested by the proposer."""

    hypothesis: str
    hyperparameters: dict
    rationale: str
    source: str = "unknown"
    confidence: float = 0.5
    raw: dict = field(default_factory=dict)


class ProposalProvider(Protocol):
    """Any object with this method is a valid proposal source."""

    def get_proposals(self, context: ProposalContext, n: int = 3) -> list[Proposal]:
        ...


def get_proposals(
    context: dict,
    config: dict,
    logger: logging.Logger,
    *,
    n: int = 5,
) -> list[dict]:
    """Return up to `n` candidate proposals for the next scout round."""
    llm_config = dict(config.get("llm", {})) if isinstance(config, dict) else {}
    normalized_n = max(1, int(n))
    if not llm_config.get("enabled", False):
        return _deterministic_proposals_from_context(context, normalized_n)

    outputs_dir_value = context.get("outputs_dir")
    outputs_dir = Path(outputs_dir_value) if outputs_dir_value is not None else None
    interaction_log_path = None
    if outputs_dir is not None:
        interaction_log_name = str(llm_config.get("interaction_log_filename", "llm_interactions.jsonl"))
        interaction_log_path = outputs_dir / interaction_log_name

    engine = ProposalEngine(
        backend=resolve_backend(config),
        interaction_log_path=interaction_log_path,
        log_interactions=bool(llm_config.get("log_interactions", True)),
        include_no_think_directive=bool(llm_config.get("include_no_think_directive", False)),
        llm_config=llm_config,
    )

    if not engine.is_available():
        logger.info("Proposal backend unavailable; using deterministic fallback.")
        return _deterministic_proposals_from_context(context, normalized_n)

    try:
        llm_result = engine.generate_experiment_proposals(
            available_models=dict(context.get("available_models", {})),
            research_brief=dict(context.get("brief", {})),
            current_best=dict(context.get("current_best", {})),
            experiment_memory=dict(context.get("memory_payload", {})),
            diversity_state=dict(context.get("diversity_state", {})),
            proposal_count=normalized_n,
            trial_history=list(context.get("trial_history", [])),
            search_progress=dict(context.get("search_progress", {})),
            failure_patterns=dict(context.get("failure_patterns", {})),
            knowledge_context=str(context.get("knowledge_context", "") or ""),
            archive_records=list(context.get("archive_records", [])),
            model_hint=context.get("model_hint"),
        )
        proposals = list(llm_result.get("proposals", []))
        if proposals:
            return proposals[:normalized_n]
        logger.info("LLM proposal engine returned no proposals; using deterministic fallback.")
    except (BackendUnavailableError, ProposalBackendFailure, ProposalParseFailure, ProposalSchemaFailure, ProposalSemanticFailure) as exc:
        logger.warning("LLM proposal generation failed: %s", exc)
    except Exception as exc:
        logger.exception("Unexpected proposal generation failure: %s", exc)

    return _deterministic_proposals_from_context(context, normalized_n)


def _deterministic_proposals_from_context(context: dict, n: int) -> list[dict]:
    brief = dict(context.get("brief", {}))
    lab_state = dict(context.get("lab_state", {}))
    available_models = dict(context.get("available_models", {}))
    memory_payload = dict(context.get("memory_payload", {}))
    current_best = dict(context.get("current_best", {}))
    exploit_delta_ratio = float(context.get("exploit_delta_ratio", 0.15))
    scout_limit = max(1, int(context.get("scout_limit", n)))
    candidates = research_lab.scout_experiments(
        brief=brief,
        lab_state=lab_state,
        available_models=available_models,
        experiment_memory=memory_payload,
        current_best=current_best,
        scout_limit=max(scout_limit, n),
        exploit_delta_ratio=exploit_delta_ratio,
    )
    return list(candidates)[:n]


class ProposalExtractionError(RuntimeError):
    """Raised when the visible model response does not contain valid structured output."""


class ProposalGenerationError(RuntimeError):
    """Raised when strict proposal generation fails and the caller must handle it explicitly."""

    def __init__(
        self,
        message: str,
        *,
        failure_kind: str,
        interaction: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.failure_kind = failure_kind
        self.interaction = dict(interaction or {})


class ProposalBackendFailure(ProposalGenerationError):
    """Raised when the configured backend cannot produce a response."""


class ProposalParseFailure(ProposalGenerationError):
    """Raised when the model response is empty or cannot be parsed as structured JSON."""


class ProposalSchemaFailure(ProposalGenerationError):
    """Raised when the parsed JSON fails strict proposal-schema validation."""


class ProposalSemanticFailure(ProposalGenerationError):
    """Raised when a schema-valid proposal fails semantic governance checks."""


class ProposalPreflightFailure(ProposalGenerationError):
    """Raised when the proposal smoke test fails before the research loop starts."""


import proposal_engine_helpers as _proposal_engine_helpers
from proposal_engine_helpers import ProposalEngineSupportMixin

_proposal_engine_helpers.ProposalGenerationError = ProposalGenerationError
_proposal_engine_helpers.ProposalBackendFailure = ProposalBackendFailure
_proposal_engine_helpers.ProposalParseFailure = ProposalParseFailure
_proposal_engine_helpers.ProposalSchemaFailure = ProposalSchemaFailure
_proposal_engine_helpers.ProposalSemanticFailure = ProposalSemanticFailure
_proposal_engine_helpers.ProposalPreflightFailure = ProposalPreflightFailure
_proposal_engine_helpers.ProposalExtractionError = ProposalExtractionError


class ProposalEngine(ProposalEngineSupportMixin):
    """Generate structured research proposals while keeping execution in repository code."""

    def __init__(
        self,
        *,
        backend: LLMBackend,
        interaction_log_path: Path | None = None,
        log_interactions: bool = True,
        include_no_think_directive: bool = False,
        llm_config: dict[str, Any] | None = None,
    ) -> None:
        self.backend = backend
        self.backend_name = backend.backend_name
        self.interaction_log_path = interaction_log_path
        self.log_interactions = bool(log_interactions)
        self.include_no_think_directive = bool(include_no_think_directive)
        self.llm_config = dict(llm_config or {})
        self.last_interaction_summary: dict[str, Any] = {}

    def is_available(self) -> bool:
        return self.backend.is_available()

    def _resolve_model_hint(self, model_hint: str | None) -> str | None:
        backend_mode = str(self.llm_config.get("backend_mode", self.backend_name)).lower()
        return resolve_model_for_backend(model_hint, backend_mode, {"llm": self.llm_config})

    def generate_hypotheses(
        self,
        *,
        research_brief: dict[str, Any],
        current_best: dict[str, Any],
        experiment_memory: dict[str, Any],
        limit: int = 5,
        model_hint: str | None = None,
    ) -> list[str]:
        resolved_model_hint = self._resolve_model_hint(model_hint)
        prompt = "\n".join(
            [
                "Generate concise ML research hypotheses as a JSON array of strings.",
                f"Goal: {research_brief.get('goal', 'Improve model quality.')}",
                f"Current best model: {current_best.get('model_name', 'unknown')}",
                f"Accepted experiments: {len(experiment_memory.get('accepted_experiments', []))}",
                f"Limit: {max(1, int(limit))}",
            ]
        )
        interaction = self._perform_interaction(
            task="hypotheses",
            prompt=prompt,
            response_format={
                "type": "array",
                "items": {"type": "string"},
                "minItems": 1,
                "maxItems": max(1, int(limit)),
            },
            model_hint=resolved_model_hint,
            prompt_variant=resolve_prompt_variant(self.llm_config, resolved_model_hint),
            expect_array=True,
        )
        parsed = interaction.get("parsed_json")
        self._write_interaction_log(interaction)
        if not isinstance(parsed, list):
            return []
        return [str(item) for item in parsed if str(item).strip()][: max(1, int(limit))]

    def generate_feature_ideas(
        self,
        *,
        research_brief: dict[str, Any],
        current_best: dict[str, Any],
        limit: int = 5,
        model_hint: str | None = None,
    ) -> list[str]:
        resolved_model_hint = self._resolve_model_hint(model_hint)
        prompt = "\n".join(
            [
                "Suggest feature-engineering ideas as a JSON array of strings.",
                "These are ideas only. They must not modify fixed validator or uncertainty code.",
                f"Goal: {research_brief.get('goal', 'Improve composite_score')}",
                f"Current best model: {current_best.get('model_name', 'unknown')}",
                f"Limit: {max(1, int(limit))}",
            ]
        )
        interaction = self._perform_interaction(
            task="feature_ideas",
            prompt=prompt,
            response_format={
                "type": "array",
                "items": {"type": "string"},
                "minItems": 1,
                "maxItems": max(1, int(limit)),
            },
            model_hint=resolved_model_hint,
            prompt_variant=resolve_prompt_variant(self.llm_config, resolved_model_hint),
            expect_array=True,
        )
        parsed = interaction.get("parsed_json")
        self._write_interaction_log(interaction)
        if not isinstance(parsed, list):
            return []
        return [str(item) for item in parsed if str(item).strip()][: max(1, int(limit))]

    def generate_search_space_suggestions(
        self,
        *,
        available_models: dict[str, dict[str, Any]],
        research_brief: dict[str, Any],
        limit: int = 5,
        model_hint: str | None = None,
    ) -> list[dict[str, Any]]:
        resolved_model_hint = self._resolve_model_hint(model_hint)
        prompt_lines = [
            "Suggest search-space refinements as a JSON array of objects.",
            "Each object must include model_name, parameter, and rationale.",
            f"Goal: {research_brief.get('goal', 'Improve composite_score')}",
            "Available models:",
        ]
        for model_name, model_config in available_models.items():
            prompt_lines.append(f"{model_name}: {self._format_search_space(model_config.get('search_space', {}))}")
        interaction = self._perform_interaction(
            task="search_space_suggestions",
            prompt="\n".join(prompt_lines),
            response_format={
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "model_name": {"type": "string"},
                        "parameter": {"type": "string"},
                        "rationale": {"type": "string"},
                    },
                    "required": ["model_name", "parameter", "rationale"],
                    "additionalProperties": False,
                },
                "minItems": 1,
                "maxItems": max(1, int(limit)),
            },
            model_hint=resolved_model_hint,
            prompt_variant=resolve_prompt_variant(self.llm_config, resolved_model_hint),
            expect_array=True,
        )
        parsed = interaction.get("parsed_json")
        self._write_interaction_log(interaction)
        if not isinstance(parsed, list):
            return []
        return [item for item in parsed if isinstance(item, dict)][: max(1, int(limit))]

    def generate_experiment_proposals(
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
        underexplored_families: list[str] | None = None,
        model_hint: str | None = None,
    ) -> list[dict[str, Any]]:
        if not self.is_available():
            return []

        trial_history = trial_history or []
        resolved_model_hint = self._resolve_model_hint(model_hint)
        prompt_variant = resolve_prompt_variant(self.llm_config, resolved_model_hint)
        family_state = build_family_state_summary(
            available_models=available_models,
            current_best=current_best,
            trial_history=trial_history,
            memory_payload=experiment_memory,
            diversity_settings=self.llm_config.get("diversity", {}),
        )
        max_attempts = 1
        if self.llm_config.get("enable_regeneration_on_reject", True):
            max_attempts += max(0, int(self.llm_config.get("max_regeneration_attempts", 1)))

        seen_signatures: set[tuple[str, str]] = set()
        request_counts = [max(1, int(proposal_count))]
        if request_counts[0] > 1:
            request_counts.append(1)

        for request_index, requested_count in enumerate(request_counts, start=1):
            expected_array = requested_count > 1
            for attempt_index in range(1, max_attempts + 1):
                prompt = self._build_experiment_prompt(
                    available_models=available_models,
                    research_brief=research_brief,
                    current_best=current_best,
                    experiment_memory=experiment_memory,
                    diversity_state=diversity_state,
                    proposal_count=requested_count,
                    trial_history=trial_history,
                    search_progress=search_progress or {},
                    failure_patterns=failure_patterns or {},
                    knowledge_context=str(knowledge_context or ""),
                    underexplored_families=underexplored_families or [],
                    family_state=family_state,
                    prompt_variant=prompt_variant,
                )
                schema = (
                    self._build_proposals_schema(available_models, proposal_count=requested_count)
                    if expected_array
                    else self._build_proposal_variant_schema(available_models)
                )
                try:
                    interaction = self._perform_interaction(
                        task="experiment_proposals",
                        prompt=prompt,
                        response_format=schema,
                        model_hint=resolved_model_hint,
                        prompt_variant=prompt_variant,
                        expect_array=expected_array,
                    )
                except ProposalExtractionError:
                    if expected_array and request_index < len(request_counts):
                        break
                    raise
                parsed = interaction.get("parsed_json")
                candidates = (
                    parsed if expected_array and isinstance(parsed, list) else [parsed] if isinstance(parsed, dict) else []
                )

                validated: list[dict[str, Any]] = []
                rejection_reason = interaction.get("rejection_reason")
                duplicate_rejected = False
                for item in candidates:
                    suggestion = self._validate_proposal(item, available_models)
                    if suggestion is None:
                        rejection_reason = {
                            "code": "malformed_or_incomplete",
                            "message": "Proposal payload failed schema-aware validation.",
                        }
                        continue
                    gate_result = gate_proposal(
                        proposal=suggestion,
                        available_models=available_models,
                        current_run_signatures=seen_signatures,
                        memory_payload=experiment_memory,
                        trial_history=trial_history,
                        family_state=family_state,
                        duplicate_settings=self.llm_config.get("duplicate_similarity_thresholds", {}),
                    )
                    if not gate_result["accepted"]:
                        rejection_reason = gate_result["reason"]
                        duplicate_rejected = bool(gate_result.get("duplicate_rejected", False))
                        continue
                    seen_signatures.add(gate_result["signature"])
                    validated.append(suggestion)
                    if len(validated) >= requested_count:
                        break

                interaction["duplicate_rejected"] = duplicate_rejected
                interaction["rejection_reason"] = rejection_reason
                interaction["semantic_validation_result"] = "not_run"
                interaction["semantic_validation_error"] = None
                interaction["semantic_rejection_reason"] = None
                interaction["final_parsed_candidate"] = (
                    validated[0] if len(validated) == 1 else validated or interaction.get("final_parsed_candidate")
                )
                interaction["regeneration_attempted"] = attempt_index < max_attempts and not validated
                interaction["fallback_used"] = (
                    request_index < len(request_counts) or (attempt_index == max_attempts and not validated)
                )
                self._write_interaction_log(interaction)

                if validated:
                    return validated

        return []

    def run_proposal_smoke_test(
        self,
        *,
        available_models: dict[str, dict[str, Any]],
        research_brief: dict[str, Any],
        current_best: dict[str, Any],
        experiment_memory: dict[str, Any],
        diversity_state: dict[str, Any],
        trial_history: list[dict[str, Any]] | None = None,
        search_progress: dict[str, Any] | None = None,
        failure_patterns: dict[str, Any] | None = None,
        knowledge_context: str | None = None,
        archive_records: list[dict[str, Any]] | None = None,
        model_hint: str | None = None,
    ) -> dict[str, Any]:
        resolved_model_hint = self._resolve_model_hint(model_hint)
        prompt_variant = resolve_prompt_variant(self.llm_config, resolved_model_hint)
        family_state = build_family_state_summary(
            available_models=available_models,
            current_best=current_best,
            trial_history=trial_history or [],
            memory_payload=experiment_memory,
            diversity_settings=self.llm_config.get("diversity", {}),
        )
        prompt = self._build_research_proposal_prompt(
            available_models=available_models,
            research_brief=research_brief,
            current_best=current_best,
            experiment_memory=experiment_memory,
            diversity_state=diversity_state,
            proposal_count=1,
            trial_history=trial_history or [],
            search_progress=search_progress or {},
            failure_patterns=failure_patterns or {},
            knowledge_context=str(knowledge_context or ""),
            archive_records=archive_records or [],
            family_state=family_state,
            prompt_variant=prompt_variant,
            selected_candidates=[],
            smoke_test=True,
        )
        try:
            interaction = self._perform_strict_interaction(
                task="proposal_smoke_test",
                prompt=prompt,
                response_format=self._build_research_proposal_schema(),
                model_hint=resolved_model_hint,
                prompt_variant=prompt_variant,
                expect_array=False,
            )
            try:
                proposal = self._validate_research_proposal(
                    interaction.get("parsed_json"),
                    available_models=available_models,
                )
            except ProposalSchemaFailure as exc:
                interaction["schema_validation_result"] = "failed"
                interaction["schema_validation_error"] = str(exc)
                interaction["proposal_status"] = exc.failure_kind
                interaction["proposal_source"] = "llm"
                interaction["error"] = str(exc)
                raise ProposalSchemaFailure(
                    str(exc),
                    failure_kind=exc.failure_kind,
                    interaction=interaction,
                ) from exc
            interaction["schema_validation_result"] = "passed"
            semantic_result = gate_research_proposal(
                proposal=proposal,
                available_models=available_models,
                current_run_signatures=set(),
                memory_payload=experiment_memory,
                trial_history=trial_history or [],
                family_state=family_state,
                duplicate_settings=self.llm_config.get("duplicate_similarity_thresholds", {}),
                current_best=current_best,
                archive_records=archive_records or [],
            )
            interaction["semantic_validation_result"] = semantic_result.get("semantic_validation_result")
            interaction["semantic_rejection_reason"] = semantic_result.get("semantic_rejection_reason")
            interaction["semantic_validation_error"] = (
                None
                if semantic_result.get("accepted", False)
                else str((semantic_result.get("semantic_rejection_reason") or {}).get("message", ""))
            )
            interaction["duplicate_rejected"] = bool(semantic_result.get("duplicate_rejected", False))
            interaction["rejection_reason"] = semantic_result.get("semantic_rejection_reason")
            interaction["final_parsed_candidate"] = proposal
            interaction["proposal_status"] = (
                "llm_success"
                if semantic_result.get("accepted", False)
                else str((semantic_result.get("semantic_rejection_reason") or {}).get("code", "semantic_rejection"))
            )
            interaction["proposal_source"] = "llm"
            self._write_interaction_log(interaction)
            if not semantic_result.get("accepted", False):
                raise ProposalSemanticFailure(
                    str((semantic_result.get("semantic_rejection_reason") or {}).get("message", "Semantic validation failed.")),
                    failure_kind=str((semantic_result.get("semantic_rejection_reason") or {}).get("code", "semantic_rejection")),
                    interaction=interaction,
                )
            return self._build_smoke_test_result(
                interaction=interaction,
                ok=True,
                status="llm_success",
                proposal_preview=proposal,
                error=None,
                failure_class=None,
            )
        except ProposalGenerationError as exc:
            interaction = dict(getattr(exc, "interaction", {}) or {})
            if interaction:
                self._write_interaction_log(interaction)
            failure_class = exc.failure_kind
            extracted_from = str(interaction.get("extracted_from_channel", "response"))
            if exc.failure_kind == "llm_parse_failure" and extracted_from in {"thinking", "repaired_thinking"}:
                failure_class = "hidden_channel_only"
            return self._build_smoke_test_result(
                interaction=interaction,
                ok=False,
                status=exc.failure_kind,
                proposal_preview=None,
                error=str(exc),
                failure_class=failure_class,
            )

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
        if not self.is_available():
            raise ProposalBackendFailure(
                "Configured LLM backend is unavailable.",
                failure_kind="llm_backend_failure",
                interaction={"backend": self.backend_name, "model": model_hint},
            )

        trial_history = trial_history or []
        resolved_model_hint = self._resolve_model_hint(model_hint)
        if not resolved_model_hint:
            raise ProposalBackendFailure(
                "No proposal model is configured for the selected backend.",
                failure_kind="llm_backend_failure",
                interaction={"backend": self.backend_name, "model": resolved_model_hint},
            )

        prompt_variant = resolve_prompt_variant(self.llm_config, resolved_model_hint)
        family_state = build_family_state_summary(
            available_models=available_models,
            current_best=current_best,
            trial_history=trial_history,
            memory_payload=experiment_memory,
            diversity_settings=self.llm_config.get("diversity", {}),
        )
        max_attempts = max(1, int(proposal_count)) + max(
            0,
            int(self.llm_config.get("max_regeneration_attempts", 1))
            if self.llm_config.get("enable_regeneration_on_reject", True)
            else 0,
        )
        selected_candidates: list[dict[str, Any]] = []
        seen_signatures: set[tuple[str, str]] = set()
        last_error: ProposalGenerationError | None = None
        last_rejection_reason: dict[str, Any] | None = None

        for attempt_index in range(1, max_attempts + 1):
            if len(selected_candidates) >= max(1, int(proposal_count)):
                break
            prompt = self._build_research_proposal_prompt(
                available_models=available_models,
                research_brief=research_brief,
                current_best=current_best,
                experiment_memory=experiment_memory,
                diversity_state=diversity_state,
                proposal_count=proposal_count,
                trial_history=trial_history,
                search_progress=search_progress or {},
                failure_patterns=failure_patterns or {},
                knowledge_context=str(knowledge_context or ""),
                archive_records=archive_records or [],
                family_state=family_state,
                prompt_variant=prompt_variant,
                selected_candidates=selected_candidates,
                smoke_test=False,
            )
            try:
                interaction = self._perform_strict_interaction(
                    task="research_proposal",
                    prompt=prompt,
                    response_format=self._build_research_proposal_schema(),
                    model_hint=resolved_model_hint,
                    prompt_variant=prompt_variant,
                    expect_array=False,
                )
                try:
                    proposal = self._validate_research_proposal(
                        interaction.get("parsed_json"),
                        available_models=available_models,
                    )
                except ProposalSchemaFailure as exc:
                    interaction["schema_validation_result"] = "failed"
                    interaction["schema_validation_error"] = str(exc)
                    interaction["proposal_status"] = exc.failure_kind
                    interaction["proposal_source"] = "llm"
                    interaction["error"] = str(exc)
                    raise ProposalSchemaFailure(
                        str(exc),
                        failure_kind=exc.failure_kind,
                        interaction=interaction,
                    ) from exc
                candidate = self._normalize_research_proposal_to_candidate(
                    proposal=proposal,
                    available_models=available_models,
                    candidate_index=len(selected_candidates) + 1,
                    backend=str(interaction.get("backend", self.backend_name)),
                    model=str(interaction.get("model", resolved_model_hint)),
                    prompt_hash=str(interaction.get("prompt_hash", "")),
                )
                gate_result = gate_research_proposal(
                    proposal=proposal,
                    available_models=available_models,
                    current_run_signatures=seen_signatures,
                    memory_payload=experiment_memory,
                    trial_history=trial_history,
                    family_state=family_state,
                    duplicate_settings=self.llm_config.get("duplicate_similarity_thresholds", {}),
                    current_best=current_best,
                    archive_records=archive_records or [],
                )
                interaction["schema_validation_result"] = "passed"
                interaction["schema_validation_error"] = None
                interaction["semantic_validation_result"] = gate_result.get("semantic_validation_result")
                interaction["semantic_validation_error"] = (
                    None
                    if gate_result.get("accepted", False)
                    else str((gate_result.get("semantic_rejection_reason") or {}).get("message", ""))
                )
                interaction["semantic_rejection_reason"] = gate_result.get("semantic_rejection_reason")
                interaction["rejection_reason"] = gate_result.get("semantic_rejection_reason")
                interaction["final_parsed_candidate"] = candidate
                interaction["proposal_source"] = "llm"
                interaction["proposal_status"] = (
                    "llm_success"
                    if gate_result.get("accepted", False)
                    else str((gate_result.get("semantic_rejection_reason") or {}).get("code", "semantic_rejection"))
                )
                if not gate_result.get("accepted", False):
                    interaction["duplicate_rejected"] = bool(gate_result.get("duplicate_rejected", False))
                    interaction["regeneration_attempted"] = attempt_index < max_attempts
                    last_rejection_reason = dict(gate_result.get("semantic_rejection_reason") or {})
                    self._write_interaction_log(interaction)
                    continue

                seen_signatures.add(gate_result["signature"])
                candidate["novelty_score"] = gate_result.get("novelty_score")
                candidate["max_similarity"] = gate_result.get("max_similarity")
                selected_candidates.append(candidate)
                self._write_interaction_log(interaction)
            except ProposalGenerationError as exc:
                last_error = exc
                interaction = dict(getattr(exc, "interaction", {}) or {})
                if interaction:
                    interaction["regeneration_attempted"] = attempt_index < max_attempts
                    self._write_interaction_log(interaction)
                if isinstance(exc, (ProposalBackendFailure, ProposalParseFailure, ProposalSchemaFailure)):
                    raise

        if not selected_candidates:
            if last_error is not None:
                raise last_error
            if last_rejection_reason is not None:
                raise ProposalSemanticFailure(
                    str(last_rejection_reason.get("message", "Semantic validation failed.")),
                    failure_kind=str(last_rejection_reason.get("code", "semantic_rejection")),
                    interaction={
                        "backend": self.backend_name,
                        "model": resolved_model_hint,
                        "prompt_variant": prompt_variant,
                        "schema_validation_result": "passed",
                        "semantic_validation_result": "failed",
                        "semantic_rejection_reason": last_rejection_reason,
                        "proposal_source": "llm",
                        "proposal_status": str(last_rejection_reason.get("code", "semantic_rejection")),
                    },
                )
            self.last_interaction_summary = {
                "backend": self.backend_name,
                "model": resolved_model_hint,
                "prompt_variant": prompt_variant,
                "parse_success": True,
            }
            return {
                "status": str(last_rejection_reason.get("code", "llm_success")) if last_rejection_reason else "llm_success",
                "backend": self.backend_name,
                "model": resolved_model_hint,
                "prompt_variant": prompt_variant,
                "rejection_reason": last_rejection_reason,
                "proposals": [],
            }

        self.last_interaction_summary = {
            "backend": self.backend_name,
            "model": resolved_model_hint,
            "prompt_variant": prompt_variant,
            "parse_success": True,
        }
        return {
            "status": "llm_success",
            "backend": self.backend_name,
            "model": resolved_model_hint,
            "prompt_variant": prompt_variant,
            "proposals": selected_candidates,
        }

    def generate_ensemble_selection(
        self,
        *,
        top_trials: list[dict[str, Any]],
        available_models: dict[str, dict[str, Any]],
        max_items: int = 3,
        model_hint: str | None = None,
    ) -> list[dict[str, Any]]:
        if not self.is_available():
            return []
        resolved_model_hint = self._resolve_model_hint(model_hint)
        prompt_lines = [
            "Choose the best base models for a stacking ensemble.",
            "Return only a JSON array of validated model_name and params objects.",
            f"Max items: {max(1, int(max_items))}",
            "Top trials:",
        ]
        if top_trials:
            for trial in top_trials:
                prompt_lines.append(
                    f"{trial.get('model_name', 'unknown')} | composite={self._format_metric(trial.get('composite_score'))} | "
                    f"params={json.dumps(self._extract_trial_params(trial), sort_keys=True)}"
                )
        else:
            prompt_lines.append("No top trials available.")
        prompt_lines.append("Available models and search spaces:")
        for model_name, model_config in available_models.items():
            prompt_lines.append(f"{model_name}: {self._format_search_space(model_config.get('search_space', {}))}")
        interaction = self._perform_interaction(
            task="ensemble_selection",
            prompt="\n".join(prompt_lines),
            response_format={
                "type": "array",
                "items": self._build_proposal_variant_schema(available_models),
                "minItems": 1,
                "maxItems": max(1, int(max_items)),
            },
            model_hint=resolved_model_hint,
            prompt_variant=resolve_prompt_variant(self.llm_config, resolved_model_hint),
            expect_array=True,
        )
        parsed = interaction.get("parsed_json")
        self._write_interaction_log(interaction)
        if not isinstance(parsed, list):
            return []
        validated: list[dict[str, Any]] = []
        for item in parsed:
            suggestion = self._validate_proposal(item, available_models)
            if suggestion is not None:
                validated.append(suggestion)
        return validated[: max(1, int(max_items))]

    def summarize_run(
        self,
        *,
        research_brief: dict[str, Any],
        final_metrics: dict[str, Any],
        acceptance: dict[str, Any],
        model_hint: str | None = None,
    ) -> dict[str, Any]:
        if not self.is_available():
            return {
                "backend": self.backend_name,
                "available": False,
                "summary": "",
            }
        resolved_model_hint = self._resolve_model_hint(model_hint)
        prompt = "\n".join(
            [
                "Summarize this ML research run in concise JSON.",
                f"Goal: {research_brief.get('goal', 'Improve composite_score')}",
                f"Accepted: {acceptance.get('accepted')}",
                f"Best model: {acceptance.get('best_model_name', 'unknown')}",
                f"Improvement pct: {acceptance.get('measured_improvement_pct', 'n/a')}",
                f"Validation verdict: {final_metrics.get('validation_verdict', 'n/a')}",
            ]
        )
        interaction = self._perform_interaction(
            task="run_summary",
            prompt=prompt,
            response_format={
                "type": "object",
                "properties": {"summary": {"type": "string"}},
                "required": ["summary"],
                "additionalProperties": False,
            },
            model_hint=resolved_model_hint,
            prompt_variant=resolve_prompt_variant(self.llm_config, resolved_model_hint),
            expect_array=False,
        )
        parsed = interaction.get("parsed_json")
        self._write_interaction_log(interaction)
        if isinstance(parsed, dict) and isinstance(parsed.get("summary"), str):
            return {
                "backend": interaction.get("backend", self.backend_name),
                "model": interaction.get("model"),
                "available": True,
                "summary": parsed["summary"],
            }
        return {
            "backend": interaction.get("backend", self.backend_name),
            "model": interaction.get("model"),
            "available": True,
            "summary": interaction.get("final_extracted_text", ""),
        }
