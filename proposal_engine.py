"""Repository-aware proposal generation on top of provider backends."""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import research_lab

from llm_backend import BackendUnavailableError, LLMBackend, extract_text_channels, resolve_backend, resolve_prompt_variant
from model_routing import resolve_model_for_backend
from research_protocol import build_family_state_summary, gate_proposal, gate_research_proposal


LOGGER = logging.getLogger("proposal_engine")
JSON_FENCE_PATTERN = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)
INVALID_PARAM = object()


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


class ProposalEngine:
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

    def _perform_strict_interaction(
        self,
        *,
        task: str,
        prompt: str,
        response_format: dict[str, Any] | None,
        model_hint: str | None,
        prompt_variant: str,
        expect_array: bool,
    ) -> dict[str, Any]:
        system_prompt = (
            "You are the proposal engine inside an AutoResearch loop. "
            "Return exactly one JSON object and no narrative."
        )
        interaction: dict[str, Any] = {
            "task": task,
            "backend": self.backend_name,
            "model": model_hint,
            "prompt": prompt,
            "prompt_variant": prompt_variant,
            "prompt_hash": hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16],
            "prompt_summary": self._truncate_text(prompt, limit=320),
            "raw_response_text": "",
            "raw_thinking_text": "",
            "final_extracted_text": "",
            "extracted_from_channel": "response",
            "parse_success": False,
            "repair_used": False,
            "visible_response_empty": True,
            "duplicate_rejected": False,
            "rejection_reason": None,
            "schema_validation_result": "not_run",
            "schema_validation_error": None,
            "proposal_status": None,
            "proposal_source": None,
            "regeneration_attempted": False,
            "fallback_used": False,
            "final_parsed_candidate": None,
            "error": None,
            "parsed_json": None,
        }
        started_at = time.perf_counter()
        try:
            payload = self.backend.generate_text(
                prompt,
                system_prompt=system_prompt,
                response_format=response_format,
                model=model_hint,
            )
        except Exception as exc:
            interaction["error"] = str(exc)
            interaction["latency_seconds"] = round(time.perf_counter() - started_at, 6)
            self.last_interaction_summary = {
                "backend": self.backend_name,
                "model": model_hint,
                "prompt_variant": prompt_variant,
                "parse_success": False,
            }
            raise ProposalBackendFailure(
                str(exc),
                failure_kind="llm_backend_failure",
                interaction=interaction,
            ) from exc

        interaction["backend"] = payload.get("backend", self.backend_name)
        interaction["model"] = payload.get("model", model_hint)
        channels = extract_text_channels(payload)
        interaction["latency_seconds"] = round(time.perf_counter() - started_at, 6)
        interaction["raw_response_text"] = self._truncate_text(channels["response_text"], limit=4000)
        interaction["raw_thinking_text"] = self._truncate_text(channels["thinking_text"], limit=4000)
        interaction["visible_response_empty"] = not bool(str(channels["response_text"]).strip())
        try:
            final_text, extracted_from, parsed_json, repair_used = self._extract_structured_output(
                channels=channels,
                expect_array=expect_array,
            )
        except ProposalExtractionError as exc:
            interaction["error"] = str(exc)
            self.last_interaction_summary = {
                "backend": interaction["backend"],
                "model": interaction["model"],
                "prompt_variant": prompt_variant,
                "parse_success": False,
            }
            raise ProposalParseFailure(
                str(exc),
                failure_kind="llm_parse_failure",
                interaction=interaction,
            ) from exc

        interaction["final_extracted_text"] = self._truncate_text(final_text, limit=4000)
        interaction["extracted_from_channel"] = extracted_from
        interaction["parsed_json"] = parsed_json
        interaction["parse_success"] = parsed_json is not None
        interaction["repair_used"] = repair_used
        if extracted_from in {"thinking", "repaired_thinking"}:
            interaction["error"] = "Model returned JSON only in the hidden thinking channel."
            self.last_interaction_summary = {
                "backend": interaction["backend"],
                "model": interaction["model"],
                "prompt_variant": prompt_variant,
                "parse_success": False,
            }
            raise ProposalParseFailure(
                "Model returned JSON only in the hidden thinking channel.",
                failure_kind="llm_parse_failure",
                interaction=interaction,
            )
        self.last_interaction_summary = {
            "backend": interaction["backend"],
            "model": interaction["model"],
            "prompt_variant": prompt_variant,
            "parse_success": interaction["parse_success"],
        }
        return interaction

    def _build_research_proposal_prompt(
        self,
        *,
        available_models: dict[str, dict[str, Any]],
        research_brief: dict[str, Any],
        current_best: dict[str, Any],
        experiment_memory: dict[str, Any],
        diversity_state: dict[str, Any],
        proposal_count: int,
        trial_history: list[dict[str, Any]],
        search_progress: dict[str, Any],
        failure_patterns: dict[str, Any],
        knowledge_context: str,
        archive_records: list[dict[str, Any]],
        family_state: dict[str, Any],
        prompt_variant: str,
        selected_candidates: list[dict[str, Any]],
        smoke_test: bool,
    ) -> str:
        recent_trials = trial_history[-5:] if prompt_variant == "compact" else trial_history[-8:]
        archive_summary = self._build_archive_history_summary(archive_records)
        novelty_summary = self._build_novelty_summary(archive_records, diversity_state)
        selected_summary = [
            {
                "model_name": candidate.get("model_name"),
                "proposal_family": candidate.get("proposal_family"),
                "hypothesis": candidate.get("hypothesis"),
            }
            for candidate in selected_candidates
        ]

        lines = [
            "You are generating one research hypothesis for an AutoResearch loop.",
            "Your proposal must be falsifiable, concrete, non-redundant, and directly executable.",
            f"Goal: {research_brief.get('goal', 'Improve composite_score')}",
            f"Acceptance metric: {research_brief.get('acceptance_metric', 'composite_score')}",
            f"Current best model: {current_best.get('model_name', 'unknown')}",
            f"Current best composite_score: {self._format_metric(current_best.get('composite_score'))}",
            f"Requested proposals this step: {max(1, int(proposal_count))}",
            f"Mode: {'smoke_test' if smoke_test else 'research'}",
            "Recent failures:",
            json.dumps(failure_patterns, sort_keys=True) if failure_patterns else "(none)",
            "Accepted and rejected hypothesis history:",
            archive_summary,
            "Novelty pressure summary:",
            novelty_summary,
            "Current experiment context:",
            json.dumps(search_progress, sort_keys=True) if search_progress else "(none)",
            "Family state summary:",
            json.dumps(self._compact_family_state_payload(family_state), sort_keys=True),
            "Already selected this step:",
            json.dumps(selected_summary, sort_keys=True) if selected_summary else "(none)",
            "Relevant domain knowledge:",
            knowledge_context or "(none provided)",
            "Allowed executable model families and parameter ranges:",
        ]
        for model_name, model_config in available_models.items():
            lines.append(f"{model_name}: {self._format_search_space(model_config.get('search_space', {}))}")

        lines.extend(
            [
                "Return exactly one JSON object matching the required schema.",
                "Do not output markdown, comments, prose, or multiple objects.",
                "The candidate_config.model_name must be one of the allowed executable families.",
                "The candidate_config.params must stay within the declared ranges.",
                "Use change_type to describe the intervention class.",
                "Use target_component for the model family or subsystem being changed.",
                "Use proposed_change to describe the exact intervention in plain language.",
                "The proposal must explain why it is new relative to recent attempts.",
                "The proposal must include risk notes and a bounded confidence score in [0,1].",
                "The expected_metric_effect must predict direction and approximate magnitude.",
                "Schema:",
                json.dumps(self._build_research_proposal_schema(), sort_keys=True),
            ]
        )
        return "\n".join(lines)

    @staticmethod
    def _build_research_proposal_schema() -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "hypothesis": {"type": "string"},
                "rationale": {"type": "string"},
                "change_type": {
                    "type": "string",
                    "enum": [
                        "hyperparameter",
                        "feature",
                        "preprocessing",
                        "model_family",
                        "objective",
                        "sampling",
                        "other",
                    ],
                },
                "target_component": {"type": "string"},
                "proposed_change": {"type": "string"},
                "expected_direction": {
                    "type": "string",
                    "enum": ["improve", "worsen_risk", "uncertain"],
                },
                "expected_metric_effect": {
                    "type": "object",
                    "properties": {
                        "metric": {"type": "string"},
                        "direction": {"type": "string", "enum": ["up", "down"]},
                        "magnitude_estimate": {"type": "string"},
                    },
                    "required": ["metric", "direction", "magnitude_estimate"],
                    "additionalProperties": False,
                },
                "confidence": {"type": "number"},
                "novelty_claim": {"type": "string"},
                "risk_notes": {"type": "string"},
                "candidate_config": {
                    "type": "object",
                    "properties": {
                        "model_name": {"type": "string"},
                        "params": {"type": "object"},
                    },
                    "required": ["model_name", "params"],
                    "additionalProperties": False,
                },
            },
            "required": [
                "hypothesis",
                "rationale",
                "change_type",
                "target_component",
                "proposed_change",
                "expected_direction",
                "expected_metric_effect",
                "confidence",
                "novelty_claim",
                "risk_notes",
                "candidate_config",
            ],
            "additionalProperties": False,
        }

    def _validate_research_proposal(
        self,
        payload: Any,
        *,
        available_models: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        if not isinstance(payload, dict):
            raise ProposalSchemaFailure(
                "Proposal payload must be a JSON object.",
                failure_kind="llm_schema_failure",
                interaction={"backend": self.backend_name},
            )
        allowed_keys = {
            "hypothesis",
            "rationale",
            "change_type",
            "target_component",
            "proposed_change",
            "expected_direction",
            "expected_metric_effect",
            "confidence",
            "novelty_claim",
            "risk_notes",
            "candidate_config",
        }
        unexpected_keys = sorted(set(payload.keys()) - allowed_keys)
        if unexpected_keys:
            raise ProposalSchemaFailure(
                f"Proposal contains unexpected fields: {', '.join(unexpected_keys)}.",
                failure_kind="llm_schema_failure",
                interaction={"backend": self.backend_name},
            )
        required_text_fields = [
            "hypothesis",
            "rationale",
            "change_type",
            "target_component",
            "proposed_change",
            "expected_direction",
            "novelty_claim",
            "risk_notes",
        ]
        normalized: dict[str, Any] = {}
        for field_name in required_text_fields:
            value = payload.get(field_name)
            if not isinstance(value, str) or not value.strip():
                raise ProposalSchemaFailure(
                    f"Proposal field '{field_name}' must be a non-empty string.",
                    failure_kind="llm_schema_failure",
                    interaction={"backend": self.backend_name},
                )
            normalized[field_name] = value.strip()
        if normalized["change_type"] not in {
            "hyperparameter",
            "feature",
            "preprocessing",
            "model_family",
            "objective",
            "sampling",
            "other",
        }:
            raise ProposalSchemaFailure(
                "change_type is outside the allowed enum.",
                failure_kind="llm_schema_failure",
                interaction={"backend": self.backend_name},
            )
        if normalized["expected_direction"] not in {"improve", "worsen_risk", "uncertain"}:
            raise ProposalSchemaFailure(
                "expected_direction is outside the allowed enum.",
                failure_kind="llm_schema_failure",
                interaction={"backend": self.backend_name},
            )

        try:
            confidence = float(payload.get("confidence"))
        except (TypeError, ValueError) as exc:
            raise ProposalSchemaFailure(
                "Proposal confidence must be numeric.",
                failure_kind="llm_schema_failure",
                interaction={"backend": self.backend_name},
            ) from exc
        if confidence < 0.0 or confidence > 1.0:
            raise ProposalSchemaFailure(
                "Proposal confidence must be bounded to [0,1].",
                failure_kind="llm_schema_failure",
                interaction={"backend": self.backend_name},
            )
        normalized["confidence"] = confidence

        expected_metric_effect = payload.get("expected_metric_effect")
        if not isinstance(expected_metric_effect, dict):
            raise ProposalSchemaFailure(
                "Proposal expected_metric_effect must be an object.",
                failure_kind="llm_schema_failure",
                interaction={"backend": self.backend_name},
            )
        unexpected_metric_keys = sorted(
            set(expected_metric_effect.keys()) - {"metric", "direction", "magnitude_estimate"}
        )
        if unexpected_metric_keys:
            raise ProposalSchemaFailure(
                f"expected_metric_effect contains unexpected fields: {', '.join(unexpected_metric_keys)}.",
                failure_kind="llm_schema_failure",
                interaction={"backend": self.backend_name},
            )
        for key in ("metric", "direction", "magnitude_estimate"):
            value = expected_metric_effect.get(key)
            if not isinstance(value, str) or not value.strip():
                raise ProposalSchemaFailure(
                    f"expected_metric_effect.{key} must be a non-empty string.",
                    failure_kind="llm_schema_failure",
                    interaction={"backend": self.backend_name},
                )
        if str(expected_metric_effect.get("direction")) not in {"up", "down"}:
            raise ProposalSchemaFailure(
                "expected_metric_effect.direction must be 'up' or 'down'.",
                failure_kind="llm_schema_failure",
                interaction={"backend": self.backend_name},
            )
        normalized["expected_metric_effect"] = {
            "metric": str(expected_metric_effect["metric"]).strip(),
            "direction": str(expected_metric_effect["direction"]).strip(),
            "magnitude_estimate": str(expected_metric_effect["magnitude_estimate"]).strip(),
        }

        candidate_config = payload.get("candidate_config")
        if not isinstance(candidate_config, dict):
            raise ProposalSchemaFailure(
                "candidate_config must be an object.",
                failure_kind="llm_schema_failure",
                interaction={"backend": self.backend_name},
            )
        unexpected_candidate_keys = sorted(set(candidate_config.keys()) - {"model_name", "params"})
        if unexpected_candidate_keys:
            raise ProposalSchemaFailure(
                f"candidate_config contains unexpected fields: {', '.join(unexpected_candidate_keys)}.",
                failure_kind="llm_schema_failure",
                interaction={"backend": self.backend_name},
            )
        model_name = candidate_config.get("model_name")
        params = candidate_config.get("params")
        executable = self._validate_proposal(
            {
                "model_name": model_name,
                "params": params,
                "proposal_family": payload.get("change_type", "llm-generated"),
                "hypothesis": payload.get("hypothesis"),
                "expected_delta": self._estimate_numeric_delta(normalized["expected_metric_effect"]),
            },
            available_models,
        )
        if executable is None:
            raise ProposalSchemaFailure(
                "candidate_config does not map to a valid executable model configuration.",
                failure_kind="llm_schema_failure",
                interaction={"backend": self.backend_name},
            )
        normalized["candidate_config"] = {
            "model_name": executable["model_name"],
            "params": executable["params"],
        }
        return normalized

    def _normalize_research_proposal_to_candidate(
        self,
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
            "expected_delta_rmse": self._estimate_numeric_delta(proposal["expected_metric_effect"]),
            "research_proposal": proposal,
        }

    def _build_archive_history_summary(self, archive_records: list[dict[str, Any]]) -> str:
        if not archive_records:
            return "(no prior hypothesis archive entries)"
        lines: list[str] = []
        for record in archive_records[-5:]:
            hypothesis = str(record.get("hypothesis") or record.get("claim") or "").strip() or "(missing hypothesis)"
            outcome = str(record.get("outcome", "pending"))
            actual = record.get("actual_delta_rmse")
            lines.append(f"- {hypothesis} | outcome={outcome} | actual_delta_rmse={actual}")
        return "\n".join(lines)

    def _build_novelty_summary(
        self,
        archive_records: list[dict[str, Any]],
        diversity_state: dict[str, Any],
    ) -> str:
        family_counts = diversity_state.get("historic_family_counts", {})
        summary = {
            "archive_size": len(archive_records),
            "historic_family_counts": family_counts,
        }
        return json.dumps(summary, sort_keys=True)

    @staticmethod
    def _estimate_numeric_delta(expected_metric_effect: dict[str, Any]) -> float | None:
        metric = str(expected_metric_effect.get("metric", "")).strip().lower()
        if metric != "rmse":
            return None
        direction = str(expected_metric_effect.get("direction", "")).strip().lower()
        magnitude_text = str(expected_metric_effect.get("magnitude_estimate", ""))
        match = re.search(r"(-?\d+(?:\.\d+)?)", magnitude_text)
        if match is None:
            return None
        magnitude = abs(float(match.group(1)))
        return -magnitude if direction == "down" else magnitude

    @staticmethod
    def _truncate_text(text: str, *, limit: int) -> str:
        normalized = str(text or "")
        if len(normalized) <= limit:
            return normalized
        return normalized[: limit - 3] + "..."

    def _perform_interaction(
        self,
        *,
        task: str,
        prompt: str,
        response_format: dict[str, Any] | None,
        model_hint: str | None,
        prompt_variant: str,
        expect_array: bool,
    ) -> dict[str, Any]:
        system_prompt = (
            "You are a proposal engine for AutoCivil-Lab. "
            "Return compact JSON only. Do not include markdown or explanation."
        )
        if self.include_no_think_directive:
            prompt = f"/no_think\n{prompt}"

        interaction: dict[str, Any] = {
            "task": task,
            "backend": self.backend_name,
            "model": model_hint,
            "prompt": prompt,
            "prompt_variant": prompt_variant,
            "raw_response_text": "",
            "raw_thinking_text": "",
            "final_extracted_text": "",
            "extracted_from_channel": "response",
            "parse_success": False,
            "repair_used": False,
            "visible_response_empty": True,
            "duplicate_rejected": False,
            "rejection_reason": None,
            "regeneration_attempted": False,
            "fallback_used": False,
            "final_parsed_candidate": None,
            "error": None,
            "parsed_json": None,
        }
        started_at = time.perf_counter()
        try:
            payload = self.backend.generate_text(
                prompt,
                system_prompt=system_prompt,
                response_format=response_format,
                model=model_hint,
            )
        except Exception as exc:
            interaction["error"] = str(exc)
            interaction["latency_seconds"] = round(time.perf_counter() - started_at, 6)
            self.last_interaction_summary = {
                "backend": self.backend_name,
                "model": model_hint,
                "prompt_variant": prompt_variant,
                "parse_success": False,
            }
            return interaction

        interaction["backend"] = payload.get("backend", self.backend_name)
        interaction["model"] = payload.get("model", model_hint)
        channels = extract_text_channels(payload)
        interaction["latency_seconds"] = round(time.perf_counter() - started_at, 6)
        interaction["raw_response_text"] = channels["response_text"]
        interaction["raw_thinking_text"] = channels["thinking_text"]
        interaction["visible_response_empty"] = not bool(str(channels["response_text"]).strip())
        try:
            final_text, extracted_from, parsed_json, repair_used = self._extract_structured_output(
                channels=channels,
                expect_array=expect_array,
            )
        except ProposalExtractionError as exc:
            LOGGER.warning("visible response did not contain valid structured JSON: %s", exc)
            interaction["error"] = str(exc)
            self.last_interaction_summary = {
                "backend": interaction["backend"],
                "model": interaction["model"],
                "prompt_variant": prompt_variant,
                "parse_success": False,
            }
            raise

        interaction["final_extracted_text"] = final_text
        interaction["extracted_from_channel"] = extracted_from
        interaction["parsed_json"] = parsed_json
        interaction["parse_success"] = parsed_json is not None
        interaction["repair_used"] = repair_used
        interaction["final_parsed_candidate"] = parsed_json if parsed_json is not None else None
        self.last_interaction_summary = {
            "backend": interaction["backend"],
            "model": interaction["model"],
            "prompt_variant": prompt_variant,
            "parse_success": interaction["parse_success"],
        }
        return interaction

    def _write_interaction_log(self, interaction: dict[str, Any]) -> None:
        if not self.log_interactions or self.interaction_log_path is None:
            return
        self.interaction_log_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "task": interaction.get("task"),
            "backend": interaction.get("backend", self.backend_name),
            "model": interaction.get("model"),
            "prompt_variant": interaction.get("prompt_variant"),
            "prompt_hash": interaction.get("prompt_hash"),
            "prompt_summary": interaction.get("prompt_summary"),
            "prompt": interaction.get("prompt", ""),
            "raw_response_text": interaction.get("raw_response_text", ""),
            "raw_thinking_text": interaction.get("raw_thinking_text", ""),
            "final_extracted_text": interaction.get("final_extracted_text", ""),
            "extracted_from_channel": interaction.get("extracted_from_channel", "response"),
            "visible_response_empty": bool(interaction.get("visible_response_empty", False)),
            "parse_success": bool(interaction.get("parse_success", False)),
            "repair_used": bool(interaction.get("repair_used", False)),
            "latency_seconds": interaction.get("latency_seconds"),
            "duplicate_rejected": bool(interaction.get("duplicate_rejected", False)),
            "rejection_reason": interaction.get("rejection_reason"),
            "schema_validation_result": interaction.get("schema_validation_result"),
            "schema_validation_error": interaction.get("schema_validation_error"),
            "semantic_validation_result": interaction.get("semantic_validation_result"),
            "semantic_validation_error": interaction.get("semantic_validation_error"),
            "semantic_rejection_reason": interaction.get("semantic_rejection_reason"),
            "proposal_status": interaction.get("proposal_status"),
            "proposal_source": interaction.get("proposal_source"),
            "regeneration_attempted": bool(interaction.get("regeneration_attempted", False)),
            "fallback_used": bool(interaction.get("fallback_used", False)),
            "final_parsed_candidate": interaction.get("final_parsed_candidate"),
            "error": interaction.get("error"),
        }
        with self.interaction_log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=True) + "\n")

    def _build_smoke_test_result(
        self,
        *,
        interaction: dict[str, Any],
        ok: bool,
        status: str,
        proposal_preview: dict[str, Any] | None,
        error: str | None,
        failure_class: str | None,
    ) -> dict[str, Any]:
        schema_success = interaction.get("schema_validation_result") == "passed"
        semantic_success = interaction.get("semantic_validation_result") == "passed"
        return {
            "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "ok": bool(ok),
            "status": status,
            "backend": interaction.get("backend", self.backend_name),
            "model": interaction.get("model"),
            "prompt_hash": interaction.get("prompt_hash"),
            "prompt_variant": interaction.get("prompt_variant"),
            "proposal_preview": proposal_preview,
            "error": error,
            "extracted_from_channel": interaction.get("extracted_from_channel", "response"),
            "visible_response_empty": bool(interaction.get("visible_response_empty", False)),
            "parse_success": bool(interaction.get("parse_success", False)),
            "schema_success": bool(schema_success),
            "semantic_success": bool(semantic_success),
            "semantic_validation_result": interaction.get("semantic_validation_result"),
            "semantic_rejection_reason": interaction.get("semantic_rejection_reason"),
            "repair_used": bool(interaction.get("repair_used", False)),
            "latency_seconds": interaction.get("latency_seconds"),
            "failure_class": failure_class,
        }

    def _build_experiment_prompt(
        self,
        *,
        available_models: dict[str, dict[str, Any]],
        research_brief: dict[str, Any],
        current_best: dict[str, Any],
        experiment_memory: dict[str, Any],
        diversity_state: dict[str, Any],
        proposal_count: int,
        trial_history: list[dict[str, Any]],
        search_progress: dict[str, Any],
        failure_patterns: dict[str, Any],
        knowledge_context: str,
        underexplored_families: list[str],
        family_state: dict[str, Any],
        prompt_variant: str,
    ) -> str:
        max_recent_trials = 5 if prompt_variant == "compact" else 10
        recent_trials = trial_history[-min(max_recent_trials, len(trial_history)) :] if trial_history else []
        single_proposal_mode = max(1, int(proposal_count)) == 1

        lines = [
            "Generate experiment proposals for concrete compressive strength regression.",
            "Think internally using these hidden steps only: STEP 1: diagnose, STEP 2: hypothesize, STEP 3: predict, STEP 4: propose JSON.",
            "Do not reveal the hidden steps. Output JSON only.",
            f"Goal: {research_brief.get('goal', 'Improve composite_score')}",
            f"Acceptance metric: {research_brief.get('acceptance_metric', 'composite_score')}",
            f"Current best: {current_best.get('model_name', 'unknown')} | composite={self._format_metric(current_best.get('composite_score'))}",
            f"Strongest family: {family_state.get('strongest_active_family') or 'unknown'}",
            "Allowed families: " + ", ".join(sorted(available_models.keys())),
            "Saturated families: " + self._format_family_list(family_state.get("saturated_families", [])),
            "Underexplored but weak families: "
            + self._format_family_list(family_state.get("underexplored_weak_families", [])),
            "Underexplored and still worth probing: "
            + self._format_family_list(family_state.get("underexplored_promising_families", [])),
            "Temporarily blocked families: "
            + self._format_family_list(family_state.get("temporarily_blocked_families", [])),
            "STEP 1: diagnose current best metrics, recent failures, and underexplored families.",
            "STEP 2: hypothesize one mechanism for improvement.",
            "STEP 3: predict an expected_delta in RMSE improvement units.",
            "STEP 4: propose JSON only with model_name, params, proposal_family, hypothesis, expected_delta.",
            "Avoid saturated or temporarily blocked families unless the proposal is materially different from recent runs.",
            "Do not repeat exact recent configs.",
            "Schema example: {\"model_name\":\"MODEL\",\"params\":{}}",
        ]
        if prompt_variant == "compact":
            lines.append("Output ONLY the final JSON object.")
        if single_proposal_mode:
            lines.extend(
                [
                    "Return exactly one JSON object.",
                    "No markdown.",
                    "No explanation.",
                ]
            )
        else:
            lines.extend(
                [
                    f"Return exactly one JSON array with up to {max(1, int(proposal_count))} proposal objects.",
                    "No markdown.",
                    "No explanation.",
                ]
            )

        if search_progress and prompt_variant == "rich":
            lines.append("Search progress: " + json.dumps(search_progress, sort_keys=True))
        if failure_patterns:
            lines.append("Failure patterns: " + json.dumps(failure_patterns, sort_keys=True))
        if knowledge_context:
            lines.append("Knowledge context:\n" + knowledge_context)
        if underexplored_families:
            lines.append("Underexplored families: " + ", ".join(sorted(str(item) for item in underexplored_families)))
        if diversity_state.get("historic_family_counts") and prompt_variant == "rich":
            lines.append(
                "Historic family counts: " + json.dumps(diversity_state.get("historic_family_counts", {}), sort_keys=True)
            )
        if prompt_variant == "rich":
            lines.append(
                f"Accepted experiments in memory: {len(experiment_memory.get('accepted_experiments', []))}"
            )
            lines.append("Family state: " + json.dumps(self._compact_family_state_payload(family_state), sort_keys=True))

        if recent_trials:
            lines.append(f"Recent trials (last {len(recent_trials)}):")
            for trial in recent_trials:
                trial_id = trial.get("experiment_id")
                if not trial_id:
                    trial_id = f"trial-{trial.get('trial_number', 'unknown')}"
                lines.append(
                    f"{trial_id} | "
                    f"model={trial.get('model_name', 'unknown')} | "
                    f"composite={self._format_metric(trial.get('composite_score'))} | "
                    f"verdict={trial.get('validation_verdict', 'UNKNOWN')} | "
                    f"params={json.dumps(self._extract_trial_params(trial), sort_keys=True)}"
                )

        lines.append("Available models and parameter ranges:")
        for model_name, model_config in available_models.items():
            lines.append(f"{model_name}: {self._format_search_space(model_config.get('search_space', {}))}")
        return "\n".join(lines)

    def _build_proposals_schema(self, available_models: dict[str, dict[str, Any]], proposal_count: int) -> dict[str, Any]:
        return {
            "type": "array",
            "items": self._build_proposal_variant_schema(available_models),
            "minItems": 1,
            "maxItems": max(1, int(proposal_count)),
        }

    def _build_proposal_variant_schema(self, available_models: dict[str, dict[str, Any]]) -> dict[str, Any]:
        variants = []
        for model_name, model_config in available_models.items():
            param_schema = {
                param_name: self._build_param_schema(param_spec)
                for param_name, param_spec in model_config.get("search_space", {}).items()
            }
            variants.append(
                {
                    "type": "object",
                    "properties": {
                        "model_name": {"type": "string", "const": model_name},
                        "params": {
                            "type": "object",
                            "properties": param_schema,
                            "additionalProperties": False,
                        },
                        "proposal_family": {"type": "string"},
                        "hypothesis": {"type": "string"},
                        "expected_delta": {"type": "number"},
                    },
                    "required": ["model_name", "params", "proposal_family", "hypothesis", "expected_delta"],
                    "additionalProperties": False,
                }
            )
        if len(variants) == 1:
            return variants[0]
        return {"oneOf": variants}

    def _build_param_schema(self, spec: dict[str, Any]) -> dict[str, Any]:
        parameter_type = spec.get("type")
        if parameter_type == "int":
            schema = {
                "type": "integer",
                "minimum": int(spec["low"]),
                "maximum": int(spec["high"]),
            }
            step = int(spec.get("step", 1))
            if step > 0:
                schema["multipleOf"] = step
            return schema
        if parameter_type == "float":
            return {
                "type": "number",
                "minimum": float(spec["low"]),
                "maximum": float(spec["high"]),
            }
        if parameter_type == "categorical":
            return {"enum": list(spec.get("choices", []))}
        return {}

    def _validate_proposal(
        self,
        payload: Any,
        available_models: dict[str, dict[str, Any]],
    ) -> dict[str, Any] | None:
        if not isinstance(payload, dict):
            return None
        model_name = payload.get("model_name")
        params = payload.get("params")
        if not isinstance(model_name, str) or model_name not in available_models:
            return None
        if not isinstance(params, dict):
            return None

        search_space = available_models[model_name].get("search_space", {})
        normalized_params: dict[str, Any] = {}
        for param_name, param_value in params.items():
            if not isinstance(param_name, str) or param_name not in search_space:
                return None
            normalized_value = self._validate_param_value(param_value, search_space[param_name])
            if normalized_value is INVALID_PARAM:
                return None
            normalized_params[param_name] = normalized_value

        return {
            "model_name": model_name,
            "display_name": str(available_models[model_name].get("display_name", model_name)),
            "params": normalized_params,
            "proposal_family": str(payload.get("proposal_family", "llm-generated")),
            "hypothesis": str(payload.get("hypothesis", "LLM-generated suggestion")),
            "expected_delta": (
                float(payload.get("expected_delta"))
                if payload.get("expected_delta") is not None
                else 0.0
            ),
        }

    def generate_feature_proposal(
        self,
        *,
        research_brief: dict[str, Any],
        current_best: dict[str, Any],
        knowledge_context: str = "",
        failure_patterns: dict[str, Any] | None = None,
        model_hint: str | None = None,
    ) -> dict[str, Any] | None:
        """Request one feature-engineering proposal as JSON only."""
        if not self.is_available():
            return None
        resolved_model_hint = self._resolve_model_hint(model_hint)
        prompt = "\n".join(
            [
                "Propose exactly one new engineered feature for concrete strength regression.",
                "Return JSON only with hypothesis, mechanism, expected_effect, and code.",
                "The code must define: def new_feature(df: pd.DataFrame) -> pd.Series:",
                f"Goal: {research_brief.get('goal', 'Improve RMSE')}",
                f"Current best model: {current_best.get('model_name', 'unknown')}",
                "Knowledge context:",
                knowledge_context or "None",
                "Failure patterns: " + json.dumps(failure_patterns or {}, sort_keys=True),
            ]
        )
        interaction = self._perform_interaction(
            task="feature_proposal",
            prompt=prompt,
            response_format={
                "type": "object",
                "properties": {
                    "hypothesis": {"type": "string"},
                    "mechanism": {"type": "string"},
                    "expected_effect": {"type": "string"},
                    "code": {"type": "string"},
                },
                "required": ["hypothesis", "mechanism", "expected_effect", "code"],
                "additionalProperties": False,
            },
            model_hint=resolved_model_hint,
            prompt_variant=resolve_prompt_variant(self.llm_config, resolved_model_hint),
            expect_array=False,
        )
        self._write_interaction_log(interaction)
        parsed = interaction.get("parsed_json")
        return parsed if isinstance(parsed, dict) else None

    def _validate_param_value(self, value: Any, spec: dict[str, Any]) -> Any:
        parameter_type = spec.get("type")
        if parameter_type == "int":
            coerced = self._coerce_number(value)
            if coerced is None or coerced != int(coerced):
                return INVALID_PARAM
            normalized = int(coerced)
            low = int(spec["low"])
            high = int(spec["high"])
            step = int(spec.get("step", 1))
            if normalized < low or normalized > high:
                return INVALID_PARAM
            if step > 0 and (normalized - low) % step != 0:
                return INVALID_PARAM
            return normalized

        if parameter_type == "float":
            coerced = self._coerce_number(value)
            if coerced is None:
                return INVALID_PARAM
            normalized = float(coerced)
            low = float(spec["low"])
            high = float(spec["high"])
            if normalized < (low - 1e-12) or normalized > (high + 1e-12):
                return INVALID_PARAM
            return normalized

        if parameter_type == "categorical":
            choices = list(spec.get("choices", []))
            if value == "null" and None in choices:
                return None
            return value if value in choices else INVALID_PARAM

        return INVALID_PARAM

    @staticmethod
    def _coerce_number(value: Any) -> float | None:
        if isinstance(value, bool):
            return None
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                return None
            try:
                return float(stripped)
            except ValueError:
                return None
        return None

    @staticmethod
    def _extract_trial_params(trial: dict[str, Any]) -> dict[str, Any]:
        raw_params = trial.get("params", trial.get("hyperparameters", {}))
        if isinstance(raw_params, dict):
            return raw_params
        if isinstance(raw_params, str):
            try:
                parsed = json.loads(raw_params)
                return parsed if isinstance(parsed, dict) else {}
            except Exception:
                return {}
        return {}

    @staticmethod
    def _format_metric(value: Any) -> str:
        if value is None:
            return "n/a"
        try:
            return f"{float(value):.4f}"
        except (TypeError, ValueError):
            return "n/a"

    @staticmethod
    def _format_family_list(families: list[str]) -> str:
        if not families:
            return "none"
        return ", ".join(str(item) for item in families)

    @staticmethod
    def _compact_family_state_payload(family_state: dict[str, Any]) -> dict[str, Any]:
        return {
            "strongest_active_family": family_state.get("strongest_active_family"),
            "saturated_families": family_state.get("saturated_families", []),
            "underexplored_promising_families": family_state.get("underexplored_promising_families", []),
            "underexplored_weak_families": family_state.get("underexplored_weak_families", []),
            "temporarily_blocked_families": family_state.get("temporarily_blocked_families", []),
        }

    def _format_search_space(self, search_space: dict[str, dict[str, Any]]) -> str:
        parts: list[str] = []
        for name, spec in search_space.items():
            parameter_type = spec.get("type")
            if parameter_type == "int":
                step = int(spec.get("step", 1))
                parts.append(f"{name}=int[{spec['low']},{spec['high']},step={step}]")
            elif parameter_type == "float":
                suffix = ",log" if spec.get("log", False) else ""
                parts.append(f"{name}=float[{spec['low']},{spec['high']}{suffix}]")
            elif parameter_type == "categorical":
                choices = ",".join(self._format_choice(choice) for choice in spec.get("choices", []))
                parts.append(f"{name}=categorical[{choices}]")
            else:
                parts.append(f"{name}=unknown")
        return "; ".join(parts)

    @staticmethod
    def _format_choice(value: Any) -> str:
        if value is None:
            return "null"
        if isinstance(value, str):
            return value
        return json.dumps(value)

    def _extract_structured_output(
        self,
        *,
        channels: dict[str, str],
        expect_array: bool,
    ) -> tuple[str, str, Any, bool]:
        for channel_name, channel_text in (
            ("response", channels.get("response_text", "")),
            ("thinking", channels.get("thinking_text", "")),
        ):
            parsed = self._extract_json(channel_text, expect_array=expect_array)
            if parsed is not None:
                return channel_text.strip(), channel_name, parsed, False

            repaired = self._repair_json_text(channel_text, expect_array=expect_array)
            if repaired is not None:
                try:
                    parsed = json.loads(repaired)
                except Exception:
                    parsed = None
                if parsed is not None and (
                    (expect_array and isinstance(parsed, list)) or (not expect_array and isinstance(parsed, dict))
                ):
                    extracted_from = "repaired_json" if channel_name == "response" else "repaired_thinking"
                    return repaired, extracted_from, parsed, True

        raise ProposalExtractionError("Visible response was empty or did not contain valid JSON.")

    def _extract_json(self, raw_text: str, *, expect_array: bool) -> Any:
        cleaned = self._clean_json_candidate(raw_text)
        if not cleaned:
            return None
        try:
            parsed = json.loads(cleaned)
        except Exception:
            parsed = None
        if parsed is not None and ((expect_array and isinstance(parsed, list)) or (not expect_array and isinstance(parsed, dict))):
            return parsed

        start_char = "[" if expect_array else "{"
        end_char = "]" if expect_array else "}"
        start_index = cleaned.find(start_char)
        end_index = cleaned.rfind(end_char)
        if start_index == -1 or end_index == -1 or end_index <= start_index:
            return None
        candidate = cleaned[start_index : end_index + 1]
        try:
            parsed = json.loads(candidate)
        except Exception:
            return None
        if (expect_array and isinstance(parsed, list)) or (not expect_array and isinstance(parsed, dict)):
            return parsed
        return None

    def _repair_json_text(self, raw_text: str, *, expect_array: bool) -> str | None:
        cleaned = self._clean_json_candidate(raw_text)
        if not cleaned:
            return None
        if re.search(r"}\s*{", cleaned) or re.search(r"]\s*\[", cleaned):
            return None

        start_char = "[" if expect_array else "{"
        start_index = cleaned.find(start_char)
        if start_index == -1:
            return None
        candidate = cleaned[start_index:]
        candidate = re.sub(r",(\s*[}\]])", r"\1", candidate)

        if self._contains_multiple_top_level_objects(candidate, expect_array=expect_array):
            return None

        brace_balance = 0
        bracket_balance = 0
        in_string = False
        escape = False
        for character in candidate:
            if escape:
                escape = False
                continue
            if character == "\\":
                escape = True
                continue
            if character == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if character == "{":
                brace_balance += 1
            elif character == "}":
                brace_balance -= 1
                if brace_balance < 0:
                    return None
            elif character == "[":
                bracket_balance += 1
            elif character == "]":
                bracket_balance -= 1
                if bracket_balance < 0:
                    return None

        repaired = candidate + ("}" * brace_balance) + ("]" * bracket_balance)
        try:
            parsed = json.loads(repaired)
        except Exception:
            return None
        if (expect_array and isinstance(parsed, list)) or (not expect_array and isinstance(parsed, dict)):
            return repaired
        return None

    @staticmethod
    def _clean_json_candidate(raw_text: str) -> str:
        cleaned = str(raw_text or "").strip()
        if not cleaned:
            return ""
        cleaned = JSON_FENCE_PATTERN.sub("", cleaned).strip()
        return cleaned

    @staticmethod
    def _contains_multiple_top_level_objects(candidate: str, *, expect_array: bool) -> bool:
        if expect_array:
            return False
        depth = 0
        in_string = False
        escape = False
        object_count = 0
        for character in candidate:
            if escape:
                escape = False
                continue
            if character == "\\":
                escape = True
                continue
            if character == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if character == "{":
                depth += 1
                if depth == 1:
                    object_count += 1
                    if object_count > 1:
                        return True
            elif character == "}":
                depth = max(0, depth - 1)
        return False
