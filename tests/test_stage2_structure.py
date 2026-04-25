import unittest
from pathlib import Path


class Stage2StructureTests(unittest.TestCase):
    def test_proposal_api_module_is_available(self) -> None:
        from proposal_api import ProposalEngine  # noqa: F401

    def test_policy_layer_module_is_available(self) -> None:
        from policy_layer import PolicyLayer  # noqa: F401

    def test_llm_backend_provider_modules_are_available(self) -> None:
        from llm_backend_ollama import OllamaBackend  # noqa: F401
        from llm_backend_openai import OpenAIBackend  # noqa: F401

    def test_train_entry_point_is_short_orchestrator(self) -> None:
        train_path = Path(__file__).resolve().parent.parent / "train.py"
        self.assertLessEqual(len(train_path.read_text(encoding="utf-8").splitlines()), 80)

    def test_llm_backend_factory_only_module_is_short(self) -> None:
        backend_path = Path(__file__).resolve().parent.parent / "llm_backend.py"
        self.assertLessEqual(len(backend_path.read_text(encoding="utf-8").splitlines()), 30)


if __name__ == "__main__":
    unittest.main()
