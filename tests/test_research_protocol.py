import json
import tempfile
import unittest
from pathlib import Path

from research_protocol import (
    apply_keep_to_research_surface,
    build_acceptance_decision,
    build_config_signature,
    filter_diverse_candidates,
    load_human_research_brief,
    load_or_initialize_experiment_memory,
    read_research_surface_state,
    record_experiment_memory,
    should_skip_duplicate_proposal,
    trial_budget_status,
    validate_final_artifact_consistency,
)


class ResearchProtocolTests(unittest.TestCase):
    def test_load_human_research_brief_front_matter(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            brief_path = Path(tmpdir) / "research_brief.md"
            brief_path.write_text(
                "---\n"
                "goal: Minimize composite\n"
                "min_improvement_pct: 1.5\n"
                "acceptance_metric: composite_score\n"
                "required_model_families:\n"
                "  - LGBMRegressor\n"
                "  - XGBRegressor\n"
                "---\n\n"
                "# Research Brief\n",
                encoding="utf-8",
            )
            brief = load_human_research_brief(brief_path)

        self.assertEqual(brief["goal"], "Minimize composite")
        self.assertEqual(brief["acceptance_metric"], "composite_score")
        self.assertAlmostEqual(float(brief["min_improvement_pct"]), 1.5)
        self.assertEqual(brief["required_model_families"], ["LGBMRegressor", "XGBRegressor"])

    def test_memory_round_trip_and_duplicate_detection(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            memory_path = Path(tmpdir) / "experiment_memory.json"
            memory = load_or_initialize_experiment_memory(memory_path)
            self.assertEqual(memory["runs"], [])

            trial_record = {
                "trial_number": 7,
                "model_name": "LGBMRegressor",
                "hyperparameters": json.dumps({"n_estimators": 300, "learning_rate": 0.05}),
                "selection_status": "new_best",
                "composite_score": 0.91,
                "validation_verdict": "WARN",
                "proposal_family": "boosting-balanced",
            }
            record_experiment_memory(
                memory_path=memory_path,
                run_id="run-1",
                trial_record=trial_record,
            )

            parsed_again = load_or_initialize_experiment_memory(memory_path)
            self.assertEqual(len(parsed_again["runs"]), 1)
            self.assertEqual(len(parsed_again["runs"][0]["trials"]), 1)

            signature = build_config_signature("LGBMRegressor", {"n_estimators": 300, "learning_rate": 0.05})
            self.assertTrue(
                should_skip_duplicate_proposal(
                    model_name="LGBMRegressor",
                    params={"n_estimators": 300, "learning_rate": 0.05},
                    current_run_signatures={signature},
                    memory_payload=parsed_again,
                )
            )

    def test_filter_diverse_candidates_limits_repeated_proposal_families(self) -> None:
        candidates = [
            {
                "experiment_id": "exp-1",
                "proposal_family": "boosting-balanced",
                "model_name": "LGBMRegressor",
                "params": {"n_estimators": 300},
            },
            {
                "experiment_id": "exp-2",
                "proposal_family": "boosting-balanced",
                "model_name": "XGBRegressor",
                "params": {"n_estimators": 300},
            },
            {
                "experiment_id": "exp-3",
                "proposal_family": "tree-diversify",
                "model_name": "RandomForestRegressor",
                "params": {"n_estimators": 300},
            },
        ]
        filtered = filter_diverse_candidates(
            candidates=candidates,
            memory_payload={"schema_version": 1, "runs": []},
            current_run_signatures=set(),
            family_limit=1,
            scout_limit=3,
        )
        self.assertEqual([item["experiment_id"] for item in filtered], ["exp-1", "exp-3"])

    def test_research_surface_state_round_trip_and_keep_ratchet(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            lab_path = Path(tmpdir) / "research_lab.py"
            lab_path.write_text(
                '"""editable surface"""\n'
                "# RESEARCH_SURFACE_STATE_START\n"
                "LAB_STATE = {\n"
                "    'surface_version': 1,\n"
                "    'accepted_experiments': [],\n"
                "}\n"
                "# RESEARCH_SURFACE_STATE_END\n",
                encoding="utf-8",
            )
            state = read_research_surface_state(lab_path)
            self.assertEqual(state["accepted_experiments"], [])

            apply_keep_to_research_surface(
                research_lab_path=lab_path,
                accepted_entry={
                    "experiment_id": "confirm-001",
                    "proposal_family": "boosting-balanced",
                    "model_name": "LGBMRegressor",
                    "composite_score": 0.91,
                },
            )
            updated_state = read_research_surface_state(lab_path)

        self.assertEqual(len(updated_state["accepted_experiments"]), 1)
        self.assertEqual(updated_state["accepted_experiments"][0]["experiment_id"], "confirm-001")

    def test_trial_budget_status(self) -> None:
        self.assertEqual(trial_budget_status(elapsed_seconds=31.0, max_trial_seconds=30.0), "budget_exceeded")
        self.assertEqual(trial_budget_status(elapsed_seconds=12.0, max_trial_seconds=30.0), "within_budget")
        self.assertEqual(trial_budget_status(elapsed_seconds=12.0, max_trial_seconds=0.0), "disabled")

    def test_acceptance_decision_uses_final_metrics(self) -> None:
        final_metrics = {
            "baseline_metrics": {"composite_score": 0.80},
            "best_search_metrics": {"composite_score": 0.83, "model_name": "LGBMRegressor"},
            "composite_improvement_pct": 3.75,
        }
        brief = {
            "goal": "Improve score",
            "acceptance_metric": "composite_score",
            "min_improvement_pct": 4.0,
            "required_model_families": [],
        }

        decision = build_acceptance_decision(final_metrics=final_metrics, brief=brief)
        self.assertFalse(decision["accepted"])
        self.assertEqual(decision["source_of_truth"], "final_metrics.json")
        self.assertEqual(decision["best_model_name"], "LGBMRegressor")

    def test_validate_final_artifact_consistency_uses_final_metrics_as_truth(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            outputs_dir = Path(tmpdir)
            (outputs_dir / "best_search_model.pkl").write_bytes(b"placeholder")
            (outputs_dir / "final_metrics.json").write_text(
                json.dumps(
                    {
                        "baseline_metrics": {"composite_score": 0.8},
                        "best_search_metrics": {
                            "model_name": "LGBMRegressor",
                            "hyperparameters": {"n_estimators": 300},
                            "composite_score": 0.83,
                            "validation_verdict": "WARN",
                        },
                        "best_model_name": "LGBMRegressor",
                        "best_model_hyperparameters": {"n_estimators": 300},
                        "validation_verdict": "WARN",
                    }
                ),
                encoding="utf-8",
            )
            (outputs_dir / "best_search_result.json").write_text(
                json.dumps(
                    {
                        "model_name": "XGBRegressor",
                        "hyperparameters": {"n_estimators": 300},
                        "composite_score": 0.83,
                        "validation_verdict": "WARN",
                    }
                ),
                encoding="utf-8",
            )

            report = validate_final_artifact_consistency(outputs_dir)

        self.assertFalse(report["consistent"])
        self.assertEqual(report["source_of_truth"], "final_metrics.json")
        self.assertEqual(report["mismatches"][0]["artifact"], "best_search_result.json")


if __name__ == "__main__":
    unittest.main()
