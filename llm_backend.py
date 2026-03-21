"""Backend abstraction for local and remote proposal-oriented LLM calls."""

from __future__ import annotations

import copy
import json
import logging
import os
import shutil
import subprocess
from typing import Any

# Declared in requirements.txt for reproducible backend installs.
import requests

from model_routing import resolve_model_for_backend


DEFAULT_LLM_CONFIG: dict[str, Any] = {
    "enabled": False,
    "backend_mode": "ollama",
    "allow_deterministic_fallback": False,
    "default_local_proposal_model": "qwen3:8b",
    "default_openai_model": "gpt-4o",
    "qwen_thinking_mode": False,
    "compact_prompt_models": ["qwen3:4b"],
    "enable_regeneration_on_reject": True,
    "max_regeneration_attempts": 1,
    "duplicate_similarity_thresholds": {
        "numeric_tolerance": 0.05,
        "float_round_digits": 4,
    },
    "temporarily_block_saturated_families": True,
    "diversity": {
        "max_family_share": 0.35,
    },
    "interval_trials": 15,
    "interaction_interval_minutes": 5.0,
    "smart_model_after_progress": 0.67,
    "include_no_think_directive": False,
    "log_interactions": True,
    "interaction_log_filename": "llm_interactions.jsonl",
    "ollama": {
        "base_url": "http://localhost:11434",
        "model": "qwen3:8b",
        "fast_model": "qwen3:4b",
        "smart_model": "qwen3:8b",
        "options": {},
        "timeout_seconds": 30,
        "use_cli_fallback": True,
    },
    "openai": {
        "base_url": "https://api.openai.com/v1/responses",
        "model": "gpt-5.1-mini",
        "api_key_env": "OPENAI_API_KEY",
        "timeout_seconds": 30,
    },
    "hybrid": {
        "primary": "ollama",
        "fallback": "openai",
    },
    "tasks": {
        "proposal_count": 8,
        "summary_enabled": True,
    },
}

LOGGER = logging.getLogger("llm_backend")


class BackendUnavailableError(RuntimeError):
    """Raised when a backend cannot serve a generation request."""


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def get_llm_config(config: dict[str, Any]) -> dict[str, Any]:
    """Return a normalized LLM configuration with legacy compatibility."""
    llm_config = config.get("llm")
    if isinstance(llm_config, dict) and llm_config:
        normalized = _deep_merge(DEFAULT_LLM_CONFIG, llm_config)
        normalized["default_local_proposal_model"] = str(
            normalized.get("default_local_proposal_model") or normalized.get("ollama", {}).get("model") or "qwen3:8b"
        )
        normalized["default_openai_model"] = str(
            normalized.get("default_openai_model") or normalized.get("openai", {}).get("model") or "gpt-4o"
        )
        normalized["ollama"]["model"] = str(
            normalized.get("ollama", {}).get("model") or normalized["default_local_proposal_model"]
        )
        normalized["openai"]["model"] = str(
            normalized.get("openai", {}).get("model") or normalized["default_openai_model"]
        )
        return normalized

    legacy = dict(config.get("search", {}).get("llm_proposals", {}))
    if not legacy:
        return copy.deepcopy(DEFAULT_LLM_CONFIG)

    normalized = copy.deepcopy(DEFAULT_LLM_CONFIG)
    normalized.update(
        {
            "enabled": bool(legacy.get("enabled", False)),
            "default_local_proposal_model": str(
                legacy.get("smart_model", normalized["default_local_proposal_model"])
            ),
            "interval_trials": int(legacy.get("interval_trials", normalized["interval_trials"])),
            "interaction_interval_minutes": float(
                legacy.get("interaction_interval_minutes", normalized["interaction_interval_minutes"])
            ),
            "smart_model_after_progress": float(
                legacy.get("smart_model_after_progress", normalized["smart_model_after_progress"])
            ),
            "include_no_think_directive": bool(
                legacy.get("include_no_think_directive", normalized["include_no_think_directive"])
            ),
            "log_interactions": bool(legacy.get("log_interactions", normalized["log_interactions"])),
            "interaction_log_filename": str(
                legacy.get("interaction_log_filename", normalized["interaction_log_filename"])
            ),
        }
    )
    normalized["ollama"] = _deep_merge(
        normalized["ollama"],
        {
            "base_url": str(legacy.get("ollama_base_url", normalized["ollama"]["base_url"])),
            "model": str(legacy.get("smart_model", normalized["default_local_proposal_model"])),
            "fast_model": str(legacy.get("fast_model", normalized["ollama"]["fast_model"])),
            "smart_model": str(legacy.get("smart_model", normalized["ollama"]["smart_model"])),
            "timeout_seconds": int(legacy.get("timeout_seconds", normalized["ollama"]["timeout_seconds"])),
        },
    )
    normalized["default_openai_model"] = str(normalized.get("openai", {}).get("model", "gpt-4o"))
    return normalized


def _normalize_model_name(model_name: str | None) -> str:
    return str(model_name or "").strip().lower()


def resolve_prompt_variant(llm_config: dict[str, Any], model_name: str | None) -> str:
    compact_models = {
        _normalize_model_name(item)
        for item in llm_config.get("compact_prompt_models", [])
        if str(item).strip()
    }
    return "compact" if _normalize_model_name(model_name) in compact_models else "rich"


def resolve_default_local_proposal_model(llm_config: dict[str, Any]) -> str:
    default_local = str(
        llm_config.get("default_local_proposal_model")
        or llm_config.get("ollama", {}).get("model")
        or DEFAULT_LLM_CONFIG["default_local_proposal_model"]
    )
    return default_local


def split_response_and_thinking(raw_text: str) -> tuple[str, str]:
    text = str(raw_text or "").strip()
    if not text:
        return "", ""

    lowered = text.lower()
    start_token = "<think>"
    end_token = "</think>"
    if start_token in lowered and end_token in lowered:
        start_index = lowered.find(start_token)
        end_index = lowered.rfind(end_token)
        thinking = text[start_index + len(start_token) : end_index].strip()
        response = text[end_index + len(end_token) :].strip()
        return response, thinking
    return text, ""


def extract_text_channels(payload: dict[str, Any]) -> dict[str, str]:
    raw_payload = payload.get("raw", {})
    raw_payload = raw_payload if isinstance(raw_payload, dict) else {}

    response_text = str(payload.get("response_text", "") or "")
    thinking_text = str(payload.get("thinking_text", "") or "")
    backend_raw_text = str(payload.get("backend_raw_text", "") or "")

    direct_text = str(payload.get("text", "") or "")
    if direct_text and not response_text:
        extracted_response, extracted_thinking = split_response_and_thinking(direct_text)
        response_text = extracted_response or response_text
        thinking_text = extracted_thinking or thinking_text
        backend_raw_text = backend_raw_text or direct_text

    for candidate_key in ("response", "output_text", "text"):
        candidate_value = raw_payload.get(candidate_key)
        if isinstance(candidate_value, str) and candidate_value.strip():
            extracted_response, extracted_thinking = split_response_and_thinking(candidate_value)
            if not response_text:
                response_text = extracted_response
            if not thinking_text:
                thinking_text = extracted_thinking
            if not backend_raw_text:
                backend_raw_text = candidate_value
            break

    # LOGGING ONLY - never parse as control output.
    for thinking_key in ("thinking", "thought", "reasoning"):
        candidate_value = raw_payload.get(thinking_key)
        if isinstance(candidate_value, str) and candidate_value.strip() and not thinking_text:
            thinking_text = candidate_value.strip()
            break

    if not backend_raw_text:
        for backend_key in ("response", "output_text", "text"):
            candidate_value = raw_payload.get(backend_key)
            if isinstance(candidate_value, str) and candidate_value.strip():
                backend_raw_text = candidate_value.strip()
                break

    return {
        "response_text": response_text.strip(),
        "thinking_text": thinking_text.strip(),
        "backend_raw_text": backend_raw_text.strip(),
    }


class LLMBackend:
    """Simple transport interface shared across provider implementations."""

    backend_name = "base"

    def is_available(self) -> bool:
        raise NotImplementedError

    def generate_text(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        response_format: dict[str, Any] | None = None,
        model: str | None = None,
        max_output_tokens: int = 512,
        temperature: float = 0.2,
    ) -> dict[str, Any]:
        raise NotImplementedError


class NullBackend(LLMBackend):
    """Disabled or unavailable backend."""

    backend_name = "disabled"

    def is_available(self) -> bool:
        return False

    def generate_text(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        response_format: dict[str, Any] | None = None,
        model: str | None = None,
        max_output_tokens: int = 512,
        temperature: float = 0.2,
    ) -> dict[str, Any]:
        raise BackendUnavailableError("No LLM backend is configured.")


class OllamaBackend(LLMBackend):
    """Ollama-backed local backend using HTTP first and CLI as fallback."""

    backend_name = "ollama"

    def __init__(self, config: dict[str, Any]):
        llm_config = get_llm_config(config)
        ollama_config = llm_config["ollama"]
        self.base_url = str(ollama_config.get("base_url", "http://localhost:11434")).rstrip("/")
        self.default_model = str(resolve_model_for_backend(None, "ollama", {"llm": llm_config}))
        self.fast_model = str(ollama_config.get("fast_model", self.default_model))
        self.smart_model = str(ollama_config.get("smart_model", self.default_model))
        self.request_options = dict(ollama_config.get("options", {}))
        self.timeout_seconds = max(1, int(ollama_config.get("timeout_seconds", 30)))
        self.use_cli_fallback = bool(ollama_config.get("use_cli_fallback", True))
        self.qwen_thinking_mode = bool(llm_config.get("qwen_thinking_mode", False))

    @staticmethod
    def _is_qwen_model(model: str) -> bool:
        return "qwen" in str(model).lower()

    def _resolved_request_options(
        self,
        *,
        model: str,
        max_output_tokens: int,
        temperature: float,
    ) -> dict[str, Any]:
        options = dict(self.request_options)
        options.update(
            {
                "temperature": temperature,
                "num_predict": max_output_tokens,
            }
        )
        if self._is_qwen_model(model) and not self.qwen_thinking_mode:
            options["think"] = False
        return options

    def _compose_prompt(self, prompt: str, system_prompt: str | None, model: str) -> str:
        prompt_parts = []
        if system_prompt:
            prompt_parts.append(system_prompt)
        prompt_parts.append(prompt)
        combined = "\n\n".join(prompt_parts)
        if self._is_qwen_model(model) and not self.qwen_thinking_mode:
            combined = combined.rstrip() + "\n/no_think"
        return combined

    def _http_available(self) -> bool:
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=self.timeout_seconds)
            response.raise_for_status()
            return True
        except Exception:
            return False

    def _cli_available(self) -> bool:
        if not self.use_cli_fallback or shutil.which("ollama") is None:
            return False
        try:
            result = subprocess.run(
                ["ollama", "list"],
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                check=False,
            )
            return result.returncode == 0
        except Exception:
            return False

    def is_available(self) -> bool:
        return self._http_available() or self._cli_available()

    def generate_text(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        response_format: dict[str, Any] | None = None,
        model: str | None = None,
        max_output_tokens: int = 512,
        temperature: float = 0.2,
    ) -> dict[str, Any]:
        model_name = str(model or self.default_model)
        LOGGER.debug("[LLM] Resolved model: %s via ollama", model_name)
        if self._http_available():
            try:
                return self._generate_http(
                    prompt,
                    system_prompt=system_prompt,
                    response_format=response_format,
                    model=model_name,
                    max_output_tokens=max_output_tokens,
                    temperature=temperature,
                )
            except Exception:
                if not self._cli_available():
                    raise
        if self._cli_available():
            return self._generate_cli(
                prompt,
                system_prompt=system_prompt,
                model=model_name,
                max_output_tokens=max_output_tokens,
                temperature=temperature,
            )
        raise BackendUnavailableError("Ollama is not reachable via HTTP or CLI.")

    def _generate_http(
        self,
        prompt: str,
        *,
        system_prompt: str | None,
        response_format: dict[str, Any] | None,
        model: str,
        max_output_tokens: int,
        temperature: float,
    ) -> dict[str, Any]:
        composed_prompt = self._compose_prompt(prompt, system_prompt, model)
        options = self._resolved_request_options(
            model=model,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
        )
        payload: dict[str, Any] = {
            "model": model,
            "prompt": composed_prompt,
            "stream": False,
            "options": options,
        }
        if response_format is not None:
            payload["format"] = response_format
        response = requests.post(
            f"{self.base_url}/api/generate",
            json=payload,
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        raw_payload = response.json()
        text = raw_payload.get("response", "")
        response_text, thinking_text = split_response_and_thinking(text if isinstance(text, str) else str(text))
        return {
            "backend": self.backend_name,
            "transport": "http",
            "model": model,
            "text": text if isinstance(text, str) else str(text),
            "response_text": response_text,
            "thinking_text": thinking_text,
            "backend_raw_text": text if isinstance(text, str) else str(text),
            "raw": raw_payload,
        }

    def _generate_cli(
        self,
        prompt: str,
        *,
        system_prompt: str | None,
        model: str,
        max_output_tokens: int,
        temperature: float,
    ) -> dict[str, Any]:
        cli_prompt = self._compose_prompt(prompt, system_prompt, model)
        options = self._resolved_request_options(
            model=model,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
        )
        result = subprocess.run(
            ["ollama", "run", model],
            input=cli_prompt,
            capture_output=True,
            text=True,
            timeout=self.timeout_seconds + max(1, int(max_output_tokens / 32)),
            check=False,
        )
        if result.returncode != 0:
            raise BackendUnavailableError(result.stderr.strip() or "ollama CLI request failed")
        response_text, thinking_text = split_response_and_thinking(result.stdout.strip())
        return {
            "backend": self.backend_name,
            "transport": "cli",
            "model": model,
            "text": result.stdout.strip(),
            "response_text": response_text,
            "thinking_text": thinking_text,
            "backend_raw_text": result.stdout.strip(),
            "raw": {
                "returncode": result.returncode,
                "stderr": result.stderr,
                "options": options,
            },
        }


class OpenAIBackend(LLMBackend):
    """OpenAI-backed backend using the Responses API over plain HTTP."""

    backend_name = "openai"

    def __init__(self, config: dict[str, Any]):
        llm_config = get_llm_config(config)
        openai_config = llm_config["openai"]
        self.base_url = str(openai_config.get("base_url", "https://api.openai.com/v1/responses"))
        self.default_model = str(resolve_model_for_backend(None, "openai", {"llm": llm_config}))
        self.api_key_env = str(openai_config.get("api_key_env", "OPENAI_API_KEY"))
        self.timeout_seconds = max(1, int(openai_config.get("timeout_seconds", 30)))

    def is_available(self) -> bool:
        return bool(os.environ.get(self.api_key_env))

    def generate_text(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        response_format: dict[str, Any] | None = None,
        model: str | None = None,
        max_output_tokens: int = 512,
        temperature: float = 0.2,
    ) -> dict[str, Any]:
        api_key = os.environ.get(self.api_key_env)
        if not api_key:
            raise BackendUnavailableError(f"Missing API key in environment variable {self.api_key_env}.")

        model_name = str(model or self.default_model)
        combined_prompt = prompt if not system_prompt else f"{system_prompt}\n\n{prompt}"
        payload: dict[str, Any] = {
            "model": model_name,
            "input": combined_prompt,
            "temperature": temperature,
            "max_output_tokens": max_output_tokens,
        }
        if response_format is not None:
            payload["text"] = {
                "format": {
                    "type": "json_schema",
                    "name": "proposal_payload",
                    "schema": response_format,
                }
            }

        response = requests.post(
            self.base_url,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        raw_payload = response.json()
        return {
            "backend": self.backend_name,
            "transport": "http",
            "model": model_name,
            "text": self._extract_output_text(raw_payload),
            "response_text": self._extract_output_text(raw_payload),
            "thinking_text": "",
            "backend_raw_text": self._extract_output_text(raw_payload),
            "raw": raw_payload,
        }

    @staticmethod
    def _extract_output_text(payload: dict[str, Any]) -> str:
        direct_text = payload.get("output_text")
        if isinstance(direct_text, str) and direct_text.strip():
            return direct_text

        output = payload.get("output", [])
        if not isinstance(output, list):
            return ""
        chunks: list[str] = []
        for item in output:
            if not isinstance(item, dict):
                continue
            for content_item in item.get("content", []):
                if not isinstance(content_item, dict):
                    continue
                if isinstance(content_item.get("text"), str):
                    chunks.append(content_item["text"])
                elif isinstance(content_item.get("output_text"), str):
                    chunks.append(content_item["output_text"])
        return "\n".join(part for part in chunks if part)


class HybridBackend(LLMBackend):
    """Fallback composition over two concrete backends."""

    backend_name = "hybrid"

    def __init__(self, primary: LLMBackend, fallback: LLMBackend):
        self.primary = primary
        self.fallback = fallback

    def is_available(self) -> bool:
        return self.primary.is_available() or self.fallback.is_available()

    def generate_text(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        response_format: dict[str, Any] | None = None,
        model: str | None = None,
        max_output_tokens: int = 512,
        temperature: float = 0.2,
    ) -> dict[str, Any]:
        backends = [self.primary, self.fallback]
        last_error: Exception | None = None
        for index, backend in enumerate(backends):
            if not backend.is_available():
                continue
            try:
                return backend.generate_text(
                    prompt,
                    system_prompt=system_prompt,
                    response_format=response_format,
                    model=model if index == 0 else None,
                    max_output_tokens=max_output_tokens,
                    temperature=temperature,
                )
            except Exception as exc:
                last_error = exc
                continue
        raise BackendUnavailableError(str(last_error or "No hybrid backend is available."))


def resolve_backend(config: dict[str, Any]) -> LLMBackend:
    """Build the configured backend implementation."""
    llm_config = get_llm_config(config)
    if not llm_config.get("enabled", False):
        return NullBackend()

    mode = str(llm_config.get("backend_mode", "ollama")).lower()
    if mode == "ollama":
        return OllamaBackend(config)
    if mode == "openai":
        return OpenAIBackend(config)
    if mode == "hybrid":
        primary_name = str(llm_config.get("hybrid", {}).get("primary", "ollama")).lower()
        fallback_name = str(llm_config.get("hybrid", {}).get("fallback", "openai")).lower()
        available = {
            "ollama": OllamaBackend(config),
            "openai": OpenAIBackend(config),
        }
        return HybridBackend(primary=available.get(primary_name, available["ollama"]), fallback=available.get(fallback_name, available["openai"]))
    raise ValueError(f"Unsupported llm.backend_mode: {mode}")
