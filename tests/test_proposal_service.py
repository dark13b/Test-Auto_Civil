import unittest

from proposal_backend_policy import resolve_backend_policy
from proposal_contract import ProposalRequest, ProviderExecution
from proposal_service import ProposalService


class StubProvider:
    def __init__(self, mode: str, execution: ProviderExecution, *, available: bool = True) -> None:
        self.mode = mode
        self.execution = execution
        self._available = available
        self.calls: list[tuple[ProposalRequest, int]] = []
        self.backend_name = execution.backend or mode

    def is_available(self) -> bool:
        return self._available

    def generate(self, request: ProposalRequest, *, limit: int) -> ProviderExecution:
        self.calls.append((request, limit))
        return self.execution


class ProposalServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.available_models = {
            "ModelFamilyA": {
                "display_name": "Model A",
                "search_space": {
                    "depth": {"type": "int", "low": 1, "high": 8},
                    "learning_rate": {"type": "float", "low": 0.01, "high": 0.3},
                },
            }
        }
        self.request = ProposalRequest(
            brief={"goal": "Improve composite_score"},
            lab_state={"accepted_experiments": []},
            available_models=self.available_models,
            memory_payload={"runs": [], "accepted_experiments": []},
            current_best={"model_name": "ModelFamilyA", "composite_score": 0.90},
            scout_limit=2,
        )

    def test_deterministic_mode_returns_structured_validated_candidates(self) -> None:
        provider = StubProvider(
            "deterministic",
            ProviderExecution(
                proposals=[
                    {
                        "model_name": "ModelFamilyA",
                        "params": {"depth": 3, "learning_rate": 0.1},
                        "description": "Reduce depth to improve generalization.",
                        "proposal_family": "deterministic-search",
                    }
                ],
                status="deterministic_success",
                backend="deterministic",
                model=None,
                prompt_variant=None,
                error=None,
            ),
        )
        service = ProposalService(
            config={"llm": {"enabled": False, "proposal_mode": "deterministic"}},
            deterministic_provider=provider,
            llm_provider=StubProvider("llm", ProviderExecution([], "unused", None, None, None, None)),
        )

        result = service.generate_candidates(self.request)

        self.assertEqual(result.metadata["proposal_mode"], "deterministic")
        self.assertEqual(result.metadata["proposal_status"], "deterministic_success")
        self.assertEqual(result.metadata["provider_path"], "deterministic")
        self.assertEqual(result.candidates[0]["proposal_source"], "deterministic")
        self.assertEqual(
            result.candidates[0]["research_proposal"]["candidate_config"]["params"],
            {"depth": 3, "learning_rate": 0.1},
        )

    def test_llm_mode_returns_validated_candidates_without_fallback(self) -> None:
        llm_provider = StubProvider(
            "llm",
            ProviderExecution(
                proposals=[
                    {
                        "experiment_id": "llm-scout-001",
                        "model_name": "ModelFamilyA",
                        "params": {"depth": 4, "learning_rate": 0.12},
                        "proposal_family": "llm-hyperparameter",
                        "proposal_source": "llm",
                        "proposal_status": "llm_success",
                        "proposal_backend": "openai",
                        "proposal_model": "gpt-5.1-mini",
                        "prompt_hash": "abc123",
                        "confidence": 0.62,
                        "expected_delta_rmse": -0.05,
                        "research_proposal": {
                            "hypothesis": "A slightly deeper tree may recover underfit regimes.",
                            "rationale": "Recent results suggest the current depth is conservative.",
                            "change_type": "hyperparameter",
                            "target_component": "ModelFamilyA",
                            "proposed_change": "Increase depth to 4 and learning_rate to 0.12.",
                            "expected_direction": "improve",
                            "expected_metric_effect": {
                                "metric": "rmse",
                                "direction": "down",
                                "magnitude_estimate": "0.02-0.06 MPa",
                            },
                            "confidence": 0.62,
                            "novelty_claim": "This exact configuration is not in recent runs.",
                            "risk_notes": "Could overfit if the depth increase is too aggressive.",
                            "candidate_config": {
                                "model_name": "ModelFamilyA",
                                "params": {"depth": 4, "learning_rate": 0.12},
                            },
                        },
                    }
                ],
                status="llm_success",
                backend="openai",
                model="gpt-5.1-mini",
                prompt_variant="rich",
                error=None,
            ),
        )
        deterministic_provider = StubProvider(
            "deterministic",
            ProviderExecution([], "deterministic_success", "deterministic", None, None, None),
        )
        service = ProposalService(
            config={
                "llm": {
                    "enabled": True,
                    "proposal_mode": "llm",
                    "backend_mode": "openai",
                    "openai": {"model": "gpt-5.1-mini"},
                }
            },
            deterministic_provider=deterministic_provider,
            llm_provider=llm_provider,
        )

        result = service.generate_candidates(self.request)

        self.assertEqual(result.metadata["proposal_mode"], "llm")
        self.assertEqual(result.metadata["proposal_status"], "llm_success")
        self.assertEqual(result.metadata["proposal_backend"], "openai")
        self.assertEqual(result.metadata["provider_path"], "llm")
        self.assertEqual(result.candidates[0]["proposal_source"], "llm")
        self.assertEqual(len(deterministic_provider.calls), 0)

    def test_hybrid_mode_falls_back_to_safe_deterministic_candidates(self) -> None:
        llm_provider = StubProvider(
            "llm",
            ProviderExecution(
                proposals=[],
                status="llm_parse_failure",
                backend="ollama",
                model="qwen3:8b",
                prompt_variant="compact",
                error="Model returned no visible JSON payload.",
            ),
        )
        deterministic_provider = StubProvider(
            "deterministic",
            ProviderExecution(
                proposals=[
                    {
                        "model_name": "ModelFamilyA",
                        "params": {"depth": 2, "learning_rate": 0.08},
                        "description": "Fallback deterministic candidate.",
                    }
                ],
                status="deterministic_success",
                backend="deterministic",
                model=None,
                prompt_variant=None,
                error=None,
            ),
        )
        service = ProposalService(
            config={
                "llm": {
                    "enabled": True,
                    "proposal_mode": "hybrid",
                    "allow_deterministic_fallback": True,
                    "backend_mode": "ollama",
                    "ollama": {"model": "qwen3:8b"},
                }
            },
            deterministic_provider=deterministic_provider,
            llm_provider=llm_provider,
        )

        result = service.generate_candidates(self.request)

        self.assertEqual(result.metadata["proposal_mode"], "hybrid")
        self.assertEqual(result.metadata["proposal_status"], "fallback_used")
        self.assertEqual(result.metadata["provider_path"], "deterministic")
        self.assertEqual(result.metadata["proposal_error"], "Model returned no visible JSON payload.")
        self.assertEqual(result.candidates[0]["proposal_source"], "deterministic_fallback")
        self.assertEqual(result.candidates[0]["proposal_backend"], "fallback")


class ProposalBackendPolicyTests(unittest.TestCase):
    def test_policy_defaults_to_deterministic_when_llm_is_disabled(self) -> None:
        resolution = resolve_backend_policy({"llm": {"enabled": False}})

        self.assertEqual(resolution.proposal_mode, "deterministic")
        self.assertIsNone(resolution.backend_mode)
        self.assertIsNone(resolution.model_name)

    def test_policy_uses_openai_defaults_for_llm_mode(self) -> None:
        resolution = resolve_backend_policy(
            {
                "llm": {
                    "enabled": True,
                    "proposal_mode": "llm",
                    "backend_mode": "openai",
                    "default_openai_model": "gpt-5.1-mini",
                }
            }
        )

        self.assertEqual(resolution.proposal_mode, "llm")
        self.assertEqual(resolution.backend_mode, "openai")
        self.assertEqual(resolution.model_name, "gpt-5.1-mini")

    def test_policy_resolves_hybrid_backend_model_from_primary_backend(self) -> None:
        resolution = resolve_backend_policy(
            {
                "llm": {
                    "enabled": True,
                    "proposal_mode": "hybrid",
                    "allow_deterministic_fallback": True,
                    "backend_mode": "hybrid",
                    "default_local_proposal_model": "qwen3:8b",
                    "default_openai_model": "gpt-5.1-mini",
                    "hybrid": {"primary": "openai", "fallback": "ollama"},
                }
            }
        )

        self.assertEqual(resolution.proposal_mode, "hybrid")
        self.assertEqual(resolution.backend_mode, "hybrid")
        self.assertEqual(resolution.primary_backend, "openai")
        self.assertEqual(resolution.model_name, "gpt-5.1-mini")


if __name__ == "__main__":
    unittest.main()
