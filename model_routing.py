"""Central model routing for backend-specific default resolution."""

from __future__ import annotations

from typing import Any


def _llm_section(config: dict[str, Any]) -> dict[str, Any]:
    llm_config = config.get("llm")
    return llm_config if isinstance(llm_config, dict) else config


def resolve_model_for_backend(
    model_hint: str | None,
    backend_mode: str,
    config: dict[str, Any],
) -> str | None:
    """Resolve the model name consistently for the selected backend."""
    if model_hint:
        return str(model_hint)

    llm_config = _llm_section(config)
    default_local = str(
        llm_config.get("default_local_proposal_model")
        or llm_config.get("ollama", {}).get("model")
        or "qwen3-coder:480b-cloud"
    )
    default_openai = str(
        llm_config.get("default_openai_model")
        or llm_config.get("openai", {}).get("model")
        or "gpt-4o"
    )

    normalized_mode = str(backend_mode or "ollama").lower()
    if normalized_mode in {"ollama", "local"}:
        return default_local
    if normalized_mode == "openai":
        return default_openai
    if normalized_mode == "hybrid":
        return default_local
    return default_local
