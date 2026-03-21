import os
import unittest
from unittest.mock import MagicMock, patch

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

    def test_get_llm_config_preserves_ollama_runtime_options(self) -> None:
        llm_config = get_llm_config(
            {
                "llm": {
                    "ollama": {
                        "options": {
                            "num_ctx": 2048,
                            "num_gpu": 1,
                        }
                    }
                }
            }
        )

        self.assertEqual(llm_config["ollama"]["options"]["num_ctx"], 2048)
        self.assertEqual(llm_config["ollama"]["options"]["num_gpu"], 1)

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

    def test_ollama_backend_forwards_configured_runtime_options(self) -> None:
        config = {
            "llm": {
                "enabled": True,
                "backend_mode": "ollama",
                "ollama": {
                    "model": "qwen3:8b",
                    "base_url": "http://localhost:11434",
                    "options": {
                        "num_ctx": 2048,
                        "num_gpu": 1,
                    },
                },
            }
        }
        backend = OllamaBackend(config)
        response = MagicMock()
        response.json.return_value = {"response": "{\"ok\":true}"}
        response.raise_for_status.return_value = None

        with patch.object(backend, "_http_available", return_value=True), patch(
            "llm_backend.requests.post",
            return_value=response,
        ) as mock_post:
            backend.generate_text("hello", max_output_tokens=128, temperature=0.3)

        payload = mock_post.call_args.kwargs["json"]
        self.assertEqual(payload["options"]["num_ctx"], 2048)
        self.assertEqual(payload["options"]["num_gpu"], 1)
        self.assertEqual(payload["options"]["num_predict"], 128)

    def test_ollama_backend_appends_no_think_for_qwen_structured_calls(self) -> None:
        config = {
            "llm": {
                "enabled": True,
                "backend_mode": "ollama",
                "default_local_proposal_model": "qwen3:8b",
                "qwen_thinking_mode": False,
                "ollama": {
                    "model": "qwen3:8b",
                    "base_url": "http://localhost:11434",
                    "options": {},
                },
            }
        }
        backend = OllamaBackend(config)
        response = MagicMock()
        response.json.return_value = {"response": "{\"ok\":true}"}
        response.raise_for_status.return_value = None

        with patch.object(backend, "_http_available", return_value=True), patch(
            "llm_backend.requests.post",
            return_value=response,
        ) as mock_post:
            backend.generate_text("return json", system_prompt="system", response_format={"type": "object"})

        payload = mock_post.call_args.kwargs["json"]
        self.assertIn("/no_think", payload["prompt"])

    def test_ollama_backend_falls_back_to_cli_when_http_generation_fails(self) -> None:
        config = {
            "llm": {
                "enabled": True,
                "backend_mode": "ollama",
                "default_local_proposal_model": "qwen3:4b",
                "ollama": {
                    "model": "qwen3:4b",
                    "base_url": "http://localhost:11434",
                    "use_cli_fallback": True,
                },
            }
        }
        backend = OllamaBackend(config)

        with patch.object(backend, "_http_available", return_value=True), patch.object(
            backend,
            "_generate_http",
            side_effect=RuntimeError("http failed"),
        ), patch.object(
            backend,
            "_cli_available",
            return_value=True,
        ), patch.object(
            backend,
            "_generate_cli",
            return_value={
                "backend": "ollama",
                "transport": "cli",
                "model": "qwen3:4b",
                "text": '{"ok": true}',
                "response_text": '{"ok": true}',
                "thinking_text": "",
                "backend_raw_text": '{"ok": true}',
                "raw": {},
            },
        ) as mock_cli:
            payload = backend.generate_text("hello")

        self.assertEqual(payload["transport"], "cli")
        self.assertEqual(payload["model"], "qwen3:4b")
        mock_cli.assert_called_once()


if __name__ == "__main__":
    unittest.main()
