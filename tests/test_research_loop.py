import tempfile
import unittest
from pathlib import Path

import pandas as pd

from research_loop import _initialize_research_log, _initialize_results_csv, _select_scout_candidates


class StubProposalEngine:
    def __init__(self, proposals: list[dict]) -> None:
        self._proposals = proposals
        self.backend_name = "stub"

    def is_available(self) -> bool:
        return True

    def generate_experiment_proposals(self, **_: object) -> list[dict]:
        return list(self._proposals)


class ResearchLoopPersistenceTests(unittest.TestCase):
    def test_initialize_results_csv_preserves_existing_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "research_results.csv"
            pd.DataFrame(
                [
                    {
                        "trial_number": 1,
                        "experiment_id": "existing-1",
                        "selection_status": "kept",
                    }
                ]
            ).to_csv(csv_path, index=False)

            _initialize_results_csv(csv_path)
            frame = pd.read_csv(csv_path)

        self.assertEqual(len(frame), 1)
        self.assertEqual(str(frame.loc[0, "experiment_id"]), "existing-1")

    def test_initialize_research_log_appends_instead_of_overwriting(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "research_log.txt"
            log_path.write_text("existing line\n", encoding="utf-8")

            _initialize_research_log(
                log_path,
                {
                    "model_name": "RandomForestRegressor",
                    "composite_score": 0.9,
                    "validation_verdict": "WARN",
                },
            )
            contents = log_path.read_text(encoding="utf-8")

        self.assertIn("existing line", contents)
        self.assertIn("Baseline | Model: RandomForestRegressor", contents)

    def test_select_scout_candidates_falls_back_to_deterministic_when_llm_returns_none(self) -> None:
        available_models = {
            "RandomForestRegressor": {
                "display_name": "RandomForest",
                "search_space": {
                    "n_estimators": {"type": "int", "low": 100, "high": 300, "step": 100},
                },
            }
        }
        candidates, metadata = _select_scout_candidates(
            brief={"scout_candidates_per_cycle": 2},
            lab_state={"accepted_experiments": []},
            available_models=available_models,
            memory_payload={"runs": []},
            current_best={"model_name": "RandomForestRegressor", "hyperparameters": {}},
            scout_limit=2,
            family_limit=1,
            current_run_signatures=set(),
            proposal_engine=StubProposalEngine([]),
        )

        self.assertTrue(candidates)
        self.assertEqual(metadata["proposal_mode"], "deterministic_fallback")
        self.assertEqual(metadata["proposal_backend"], "fallback")

    def test_select_scout_candidates_normalizes_llm_proposals_for_confirm_stage(self) -> None:
        available_models = {
            "LGBMRegressor": {
                "display_name": "LightGBM",
                "search_space": {
                    "n_estimators": {"type": "int", "low": 100, "high": 500, "step": 100},
                    "learning_rate": {"type": "float", "low": 0.01, "high": 0.3},
                },
            }
        }
        candidates, metadata = _select_scout_candidates(
            brief={"scout_candidates_per_cycle": 1},
            lab_state={"accepted_experiments": []},
            available_models=available_models,
            memory_payload={"runs": [], "accepted_experiments": []},
            current_best={"model_name": "LGBMRegressor", "hyperparameters": {}},
            scout_limit=1,
            family_limit=1,
            current_run_signatures=set(),
            proposal_engine=StubProposalEngine(
                [
                    {
                        "model_name": "LGBMRegressor",
                        "display_name": "LightGBM",
                        "proposal_family": "llm-boosting",
                        "hypothesis": "Try stronger regularization.",
                        "params": {"n_estimators": 500, "learning_rate": 0.1},
                    }
                ]
            ),
        )

        self.assertEqual(metadata["proposal_mode"], "llm")
        self.assertEqual(candidates[0]["stage"], "scout")
        self.assertEqual(candidates[0]["experiment_id"], "llm-scout-001")
        self.assertEqual(candidates[0]["proposal_family"], "llm-boosting")


if __name__ == "__main__":
    unittest.main()
