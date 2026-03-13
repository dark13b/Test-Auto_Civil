import os
import unittest
from unittest.mock import patch

from llm_backend import (
    HybridBackend,
    NullBackend,
    OpenAIBackend,
    OllamaBackend,
    get_llm_config,
    resolve_backend,
    resolve_prompt_variant,
)


class StubBackend:
    def __init__(self, backend_name: str, *, available: bool, response: dict | None = None) -> None:
        self.backend_name = backend_name
        self._available = available
        self._response = response or {"text": "", "backend": backend_name}

    def is_available(self) -> bool:
        return self._available

    def generate_text(self, prompt: str, **_: object) -> dict:
        if not self._available:
            raise RuntimeError(f"{self.backend_name} unavailable")
        payload = dict(self._response)
        payload.setdefault("backend", self.backend_name)
        payload.setdefault("prompt", prompt)
        return payload


class LLMBackendTests(unittest.TestCase):
    def test_get_llm_config_includes_safe_prompt_and_gate_defaults(self) -> None:
        llm_config = get_llm_config({})

        self.assertEqual(llm_config["default_local_proposal_model"], "qwen3:8b")
        self.assertEqual(llm_config["compact_prompt_models"], ["qwen3:4b"])
        self.assertTrue(llm_config["enable_regeneration_on_reject"])
        self.assertEqual(llm_config["max_regeneration_attempts"], 1)
        self.assertEqual(llm_config["duplicate_similarity_thresholds"]["numeric_tolerance"], 0.05)
        self.assertEqual(llm_config["duplicate_similarity_thresholds"]["float_round_digits"], 4)

    def test_resolve_prompt_variant_uses_configured_compact_models(self) -> None:
        llm_config = get_llm_config({"llm": {"compact_prompt_models": ["tiny-local"]}})

        self.assertEqual(resolve_prompt_variant(llm_config, "tiny-local"), "compact")
        self.assertEqual(resolve_prompt_variant(llm_config, "larger-local"), "rich")

    def test_resolve_backend_builds_ollama_backend_for_ollama_mode(self) -> None:
        config = {
            "llm": {
                "enabled": True,
                "backend_mode": "ollama",
                "ollama": {"model": "qwen3:8b", "base_url": "http://localhost:11434"},
            }
        }

        backend = resolve_backend(config)

        self.assertIsInstance(backend, OllamaBackend)
        self.assertEqual(backend.backend_name, "ollama")

    def test_resolve_backend_returns_null_backend_when_disabled(self) -> None:
        backend = resolve_backend({"llm": {"enabled": False}})
        self.assertIsInstance(backend, NullBackend)
        self.assertFalse(backend.is_available())

    def test_hybrid_backend_uses_fallback_when_primary_unavailable(self) -> None:
        primary = StubBackend("ollama", available=False)
        fallback = StubBackend("openai", available=True, response={"text": "ok"})
        backend = HybridBackend(primary=primary, fallback=fallback)

        response = backend.generate_text("hello")

        self.assertEqual(response["backend"], "openai")
        self.assertEqual(response["text"], "ok")

    def test_openai_backend_unavailable_without_api_key(self) -> None:
        config = {
            "llm": {
                "enabled": True,
                "backend_mode": "openai",
                "openai": {"model": "gpt-5.1-mini", "api_key_env": "OPENAI_API_KEY_MISSING"},
            }
        }
        with patch.dict(os.environ, {}, clear=True):
            backend = resolve_backend(config)

        self.assertIsInstance(backend, OpenAIBackend)
        self.assertFalse(backend.is_available())


if __name__ == "__main__":
    unittest.main()
