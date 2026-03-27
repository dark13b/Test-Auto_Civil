"""Centralized proposal mode, backend, and model resolution policy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from proposal_contract import ProposalMode


DEFAULT_LOCAL_PROPOSAL_MODEL = "qwen3-coder:480b-cloud"
DEFAULT_OPENAI_PROPOSAL_MODEL = "gpt-4o"
DEFAULT_BACKEND_MODE = "ollama"
VALID_BACKEND_MODES = {"ollama", "openai", "hybrid"}
VALID_PROPOSAL_MODES = {"deterministic", "llm", "hybrid"}


@dataclass(frozen=True)
class BackendPolicyResolution:
    proposal_mode: ProposalMode
    llm_enabled: bool
    allow_deterministic_fallback: bool
    backend_mode: str | None
    primary_backend: str | None
    fallback_backend: str | None
    model_name: str | None


def resolve_backend_policy(
    config: dict[str, Any],
    *,
    proposal_mode: str | None = None,
    model_hint: str | None = None,
) -> BackendPolicyResolution:
    llm_config = _llm_section(config)
    llm_enabled = bool(llm_config.get("enabled", False))
    allow_fallback = bool(llm_config.get("allow_deterministic_fallback", False))
    normalized_mode = _normalize_proposal_mode(
        proposal_mode or llm_config.get("proposal_mode"),
        llm_enabled=llm_enabled,
        allow_deterministic_fallback=allow_fallback,
    )
    if normalized_mode == "deterministic":
        return BackendPolicyResolution(
            proposal_mode="deterministic",
            llm_enabled=False,
            allow_deterministic_fallback=False,
            backend_mode=None,
            primary_backend=None,
            fallback_backend=None,
            model_name=None,
        )

    backend_mode = str(llm_config.get("backend_mode", DEFAULT_BACKEND_MODE) or DEFAULT_BACKEND_MODE).lower()
    if backend_mode not in VALID_BACKEND_MODES:
        backend_mode = DEFAULT_BACKEND_MODE

    primary_backend = backend_mode
    fallback_backend = None
    if backend_mode == "hybrid":
        hybrid_config = llm_config.get("hybrid", {})
        primary_backend = _normalize_backend_name(hybrid_config.get("primary"), default="ollama")
        fallback_backend = _normalize_backend_name(hybrid_config.get("fallback"), default="openai")

    resolved_model = str(model_hint) if model_hint else _default_model_for_backend(primary_backend, llm_config)
    return BackendPolicyResolution(
        proposal_mode=normalized_mode,
        llm_enabled=llm_enabled,
        allow_deterministic_fallback=allow_fallback,
        backend_mode=backend_mode,
        primary_backend=primary_backend,
        fallback_backend=fallback_backend,
        model_name=resolved_model,
    )


def resolve_model_for_backend(
    model_hint: str | None,
    backend_mode: str,
    config: dict[str, Any],
) -> str | None:
    llm_config = _llm_section(config)
    return resolve_backend_policy(
        {"llm": {**llm_config, "enabled": True, "backend_mode": backend_mode}},
        proposal_mode="llm",
        model_hint=model_hint,
    ).model_name


def _normalize_proposal_mode(
    requested_mode: str | None,
    *,
    llm_enabled: bool,
    allow_deterministic_fallback: bool,
) -> ProposalMode:
    normalized = str(requested_mode or "").strip().lower()
    if normalized in VALID_PROPOSAL_MODES:
        return normalized  # type: ignore[return-value]
    if not llm_enabled:
        return "deterministic"
    if allow_deterministic_fallback:
        return "hybrid"
    return "llm"


def _default_model_for_backend(primary_backend: str | None, llm_config: dict[str, Any]) -> str:
    if primary_backend == "openai":
        return str(
            llm_config.get("default_openai_model")
            or llm_config.get("openai", {}).get("model")
            or DEFAULT_OPENAI_PROPOSAL_MODEL
        )
    return str(
        llm_config.get("default_local_proposal_model")
        or llm_config.get("ollama", {}).get("model")
        or DEFAULT_LOCAL_PROPOSAL_MODEL
    )


def _llm_section(config: dict[str, Any]) -> dict[str, Any]:
    llm_config = config.get("llm")
    return dict(llm_config) if isinstance(llm_config, dict) else dict(config or {})


def _normalize_backend_name(name: Any, *, default: str) -> str:
    candidate = str(name or default).lower()
    return candidate if candidate in {"ollama", "openai"} else default
