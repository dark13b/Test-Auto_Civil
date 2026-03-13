import unittest

from proposal_engine import ProposalEngine


class StubBackend:
    def __init__(self, response_text: str, *, available: bool = True, backend_name: str = "ollama") -> None:
        self._response_text = response_text
        self._available = available
        self.backend_name = backend_name

    def is_available(self) -> bool:
        return self._available

    def generate_text(self, prompt: str, **_: object) -> dict:
        return {
            "backend": self.backend_name,
            "model": "stub-model",
            "text": self._response_text,
            "prompt": prompt,
        }


class ProposalEngineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.available_models = {
            "RandomForestRegressor": {
                "display_name": "RandomForest",
                "search_space": {
                    "n_estimators": {"type": "int", "low": 100, "high": 500, "step": 100},
                    "max_depth": {"type": "categorical", "choices": [None, 4, 8]},
                },
            }
        }
        self.context = {
            "research_brief": {"goal": "Improve composite_score"},
            "current_best": {"model_name": "RandomForestRegressor", "composite_score": 0.90},
            "experiment_memory": {"runs": []},
            "diversity_state": {"historic_family_counts": {}},
            "proposal_count": 2,
        }

    def test_generate_experiment_proposals_returns_validated_proposals(self) -> None:
        backend = StubBackend(
            '[{"model_name":"RandomForestRegressor","params":{"n_estimators":200,"max_depth":4},"proposal_family":"tree-search","hypothesis":"tune depth"}]'
        )
        engine = ProposalEngine(backend=backend)

        proposals = engine.generate_experiment_proposals(
            available_models=self.available_models,
            **self.context,
        )

        self.assertEqual(len(proposals), 1)
        self.assertEqual(proposals[0]["model_name"], "RandomForestRegressor")
        self.assertEqual(proposals[0]["params"]["n_estimators"], 200)

    def test_generate_experiment_proposals_returns_empty_list_for_invalid_json(self) -> None:
        backend = StubBackend("not-json")
        engine = ProposalEngine(backend=backend)

        proposals = engine.generate_experiment_proposals(
            available_models=self.available_models,
            **self.context,
        )

        self.assertEqual(proposals, [])


if __name__ == "__main__":
    unittest.main()
