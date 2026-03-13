"""Backend abstraction for local and remote proposal-oriented LLM calls."""

from __future__ import annotations

import copy
import json
import os
import shutil
import subprocess
from typing import Any

import requests


DEFAULT_LLM_CONFIG: dict[str, Any] = {
    "enabled": False,
    "backend_mode": "ollama",
    "allow_deterministic_fallback": True,
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
        return _deep_merge(DEFAULT_LLM_CONFIG, llm_config)

    legacy = dict(config.get("search", {}).get("llm_proposals", {}))
    if not legacy:
        return copy.deepcopy(DEFAULT_LLM_CONFIG)

    normalized = copy.deepcopy(DEFAULT_LLM_CONFIG)
    normalized.update(
        {
            "enabled": bool(legacy.get("enabled", False)),
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
            "model": str(legacy.get("smart_model", normalized["ollama"]["model"])),
            "fast_model": str(legacy.get("fast_model", normalized["ollama"]["fast_model"])),
            "smart_model": str(legacy.get("smart_model", normalized["ollama"]["smart_model"])),
            "timeout_seconds": int(legacy.get("timeout_seconds", normalized["ollama"]["timeout_seconds"])),
        },
    )
    return normalized


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
        ollama_config = get_llm_config(config)["ollama"]
        self.base_url = str(ollama_config.get("base_url", "http://localhost:11434")).rstrip("/")
        self.default_model = str(ollama_config.get("model", "qwen3:8b"))
        self.fast_model = str(ollama_config.get("fast_model", self.default_model))
        self.smart_model = str(ollama_config.get("smart_model", self.default_model))
        self.timeout_seconds = max(1, int(ollama_config.get("timeout_seconds", 30)))
        self.use_cli_fallback = bool(ollama_config.get("use_cli_fallback", True))

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
        if self._http_available():
            return self._generate_http(
                prompt,
                system_prompt=system_prompt,
                response_format=response_format,
                model=model_name,
                max_output_tokens=max_output_tokens,
                temperature=temperature,
            )
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
        composed_prompt = prompt if not system_prompt else f"{system_prompt}\n\n{prompt}"
        payload: dict[str, Any] = {
            "model": model,
            "prompt": composed_prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_output_tokens,
            },
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
        return {
            "backend": self.backend_name,
            "transport": "http",
            "model": model,
            "text": text if isinstance(text, str) else str(text),
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
        prompt_parts = []
        if system_prompt:
            prompt_parts.append(system_prompt)
        prompt_parts.append(prompt)
        cli_prompt = "\n\n".join(prompt_parts)
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
        return {
            "backend": self.backend_name,
            "transport": "cli",
            "model": model,
            "text": result.stdout.strip(),
            "raw": {
                "returncode": result.returncode,
                "stderr": result.stderr,
                "temperature": temperature,
            },
        }


class OpenAIBackend(LLMBackend):
    """OpenAI-backed backend using the Responses API over plain HTTP."""

    backend_name = "openai"

    def __init__(self, config: dict[str, Any]):
        openai_config = get_llm_config(config)["openai"]
        self.base_url = str(openai_config.get("base_url", "https://api.openai.com/v1/responses"))
        self.default_model = str(openai_config.get("model", "gpt-5.1-mini"))
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
