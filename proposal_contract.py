"""Shared contracts for proposal generation providers and the service layer."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol


ProposalMode = Literal["deterministic", "llm", "hybrid"]


@dataclass(frozen=True)
class ProposalRequest:
    brief: dict[str, Any]
    lab_state: dict[str, Any]
    available_models: dict[str, dict[str, Any]]
    memory_payload: dict[str, Any]
    current_best: dict[str, Any]
    scout_limit: int
    trial_history: list[dict[str, Any]] = field(default_factory=list)
    search_progress: dict[str, Any] = field(default_factory=dict)
    failure_patterns: dict[str, Any] = field(default_factory=dict)
    knowledge_context: str = ""
    archive_records: list[dict[str, Any]] = field(default_factory=list)
    exploit_delta_ratio: float = 0.15


@dataclass(frozen=True)
class ProviderExecution:
    proposals: list[dict[str, Any]]
    status: str
    backend: str | None
    model: str | None
    prompt_variant: str | None
    error: str | None


@dataclass(frozen=True)
class ProposalBatch:
    candidates: list[dict[str, Any]]
    metadata: dict[str, Any]


class ProposalProviderContract(Protocol):
    mode: ProposalMode

    def is_available(self) -> bool:
        ...

    def generate(self, request: ProposalRequest, *, limit: int) -> ProviderExecution:
        ...
