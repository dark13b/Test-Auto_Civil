import unittest

from model_routing import resolve_model_for_backend


class ModelRoutingTests(unittest.TestCase):
    def test_resolve_model_for_backend_returns_openai_default_for_openai_mode(self) -> None:
        config = {
            "llm": {
                "default_local_proposal_model": "qwen3:8b",
                "default_openai_model": "gpt-4o",
            }
        }

        self.assertEqual(resolve_model_for_backend(None, "openai", config), "gpt-4o")

    def test_resolve_model_for_backend_returns_local_default_for_ollama_and_hybrid(self) -> None:
        config = {
            "llm": {
                "default_local_proposal_model": "qwen3:8b",
                "default_openai_model": "gpt-4o",
            }
        }

        self.assertEqual(resolve_model_for_backend(None, "ollama", config), "qwen3:8b")
        self.assertEqual(resolve_model_for_backend(None, "hybrid", config), "qwen3:8b")


if __name__ == "__main__":
    unittest.main()
