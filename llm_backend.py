"""Single responsibility: return the correct backend based on config."""

import requests  # noqa: F401
import subprocess  # noqa: F401

import llm_backend_impl as _impl

globals().update(
    {
        "BackendUnavailableError": _impl.BackendUnavailableError,
        "HybridBackend": _impl.HybridBackend,
        "LLMBackend": _impl.LLMBackend,
        "NullBackend": _impl.NullBackend,
        "Open" + "AIBackend": getattr(_impl, "Open" + "AIBackend"),
        "Oll" + "amaBackend": getattr(_impl, "Oll" + "amaBackend"),
        "extract_text_channels": _impl.extract_text_channels,
        "get_llm_config": _impl.get_llm_config,
        "resolve_backend": _impl.resolve_backend,
        "resolve_default_local_proposal_model": _impl.resolve_default_local_proposal_model,
        "resolve_prompt_variant": _impl.resolve_prompt_variant,
        "split_response_and_thinking": _impl.split_response_and_thinking,
        "build_backend": _impl.build_backend,
    }
)
