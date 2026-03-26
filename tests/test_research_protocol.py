import json
import tempfile
import unittest
from pathlib import Path

import run_qwen_only

from loop_candidate_selection import select_scout_candidates
from research_protocol import (
    DuplicateExperimentError,
    apply_keep_to_research_surface,
    build_acceptance_decision,
    build_config_signature,
    build_family_state_summary,
    filter_diverse_candidates,
    gate_proposal,
    gate_research_proposal,
    load_human_research_brief,
    load_or_initialize_experiment_memory,
    read_research_surface_state,
    record_experiment_memory,
    should_skip_duplicate_proposal,
    trial_budget_status,
    validate_lab_state_integrity,
    validate_final_artifact_consistency,
)


class ResearchProtocolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.available_models = {
            "ModelFamilyA": {
                "display_name": "Model A",
                "search_space": {
                    "depth": {"type": "int", "low": 1, "high": 8},
                    "learning_rate": {"type": "float", "low": 0.01, "high": 0.3},
                },
            },
            "ModelFamilyB": {
                "display_name": "Model B",
                "search_space": {
                    "alpha": {"type": "float", "low": 0.01, "high": 1.0},
                },
            },
        }
        self.duplicate_settings = {
            "numeric_tolerance": 0.05,
            "float_round_digits": 4,
        }

    def _research_proposal(
        self,
        *,
        model_name: str = "ModelFamilyA",
        params: dict[str, object] | None = None,
        hypothesis: str = "Reduce depth slightly to improve generalization.",
        rationale: str = "The current family is close to a plateau.",
        change_type: str = "hyperparameter",
        target_component: str = "ModelFamilyA",
        proposed_change: str = "Decrease depth to 3 while keeping learning_rate near 0.1.",
        expected_direction: str = "improve",
        expected_metric_effect: dict[str, str] | None = None,
        confidence: float = 0.64,
        novelty_claim: str = "This proposal makes a bounded change to the search surface.",
        risk_notes: str = "May underfit if the change is too aggressive.",
        candidate_model_name: str | None = None,
        candidate_params: dict[str, object] | None = None,
    ) -> dict[str, object]:
        raw_params = params if candidate_params is None else candidate_params
        if raw_params is None:
            raw_params = {"depth": 3, "learning_rate": 0.1}
        return {
            "hypothesis": hypothesis,
            "rationale": rationale,
            "change_type": change_type,
            "target_component": target_component,
            "proposed_change": proposed_change,
            "expected_direction": expected_direction,
            "expected_metric_effect": expected_metric_effect
            or {"metric": "rmse", "direction": "down", "magnitude_estimate": "0.03-0.08 MPa"},
            "confidence": confidence,
            "novelty_claim": novelty_claim,
            "risk_notes": risk_notes,
            "candidate_config": {
                "model_name": candidate_model_name or model_name,
                "params": dict(raw_params),
            },
        }

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
                "rmse": 4.8,
                "test_rmse": 4.7,
                "validation_verdict": "WARN",
                "proposal_family": "boosting-balanced",
                "expected_delta_rmse": 0.2,
                "actual_delta_rmse": 0.1,
                "calibration_error": 0.1,
                "novelty_score": 0.44,
            }
            record_experiment_memory(
                memory_path=memory_path,
                run_id="run-1",
                trial_record=trial_record,
            )

            parsed_again = load_or_initialize_experiment_memory(memory_path)
            self.assertEqual(parsed_again["schema_version"], 3)
            self.assertEqual(len(parsed_again["runs"]), 1)
            self.assertEqual(len(parsed_again["runs"][0]["trials"]), 1)
            self.assertEqual(parsed_again["runs"][0]["trials"][0]["novelty_score"], 0.44)

            signature = build_config_signature("LGBMRegressor", {"n_estimators": 300, "learning_rate": 0.05})
            self.assertTrue(
                should_skip_duplicate_proposal(
                    model_name="LGBMRegressor",
                    params={"n_estimators": 300, "learning_rate": 0.05},
                    current_run_signatures={signature},
                    memory_payload=parsed_again,
                )
            )

    def test_gate_proposal_rejects_exact_duplicate(self) -> None:
        memory_payload = {
            "runs": [
                {
                    "run_id": "old-run",
                    "trials": [
                        {
                            "model_name": "ModelFamilyA",
                            "params": {"depth": 3, "learning_rate": 0.1},
                            "proposal_family": "family-a",
                            "composite_score": 0.88,
                            "validation_verdict": "WARN",
                            "signature": ["ModelFamilyA", '{"depth":3,"learning_rate":0.1}'],
                        }
                    ],
                }
            ],
            "accepted_experiments": [],
        }
        family_state = build_family_state_summary(
            available_models=self.available_models,
            current_best={"model_name": "ModelFamilyA", "composite_score": 0.90},
            trial_history=[],
            memory_payload=memory_payload,
            diversity_settings={"max_family_share": 0.35},
        )

        result = gate_proposal(
            proposal={"model_name": "ModelFamilyA", "params": {"depth": 3, "learning_rate": 0.1}},
            available_models=self.available_models,
            current_run_signatures=set(),
            memory_payload=memory_payload,
            trial_history=[],
            family_state=family_state,
            duplicate_settings=self.duplicate_settings,
        )

        self.assertFalse(result["accepted"])
        self.assertEqual(result["reason"]["code"], "exact_duplicate")
        self.assertIn("novelty_score", result)

    def test_gate_research_proposal_rejects_exact_duplicate(self) -> None:
        proposal = self._research_proposal(
            proposed_change="Keep depth at 3 and learning_rate at 0.1.",
            candidate_params={"depth": 3, "learning_rate": 0.1},
        )
        trial_history = [
            {
                "model_name": "ModelFamilyA",
                "hyperparameters": json.dumps({"depth": 3, "learning_rate": 0.1}),
                "proposal_family": "family-a",
                "composite_score": 0.91,
                "validation_verdict": "WARN",
            }
        ]
        memory_payload = {"runs": [], "accepted_experiments": []}

        result = gate_research_proposal(
            proposal=proposal,
            available_models=self.available_models,
            current_run_signatures=set(),
            memory_payload=memory_payload,
            trial_history=trial_history,
            family_state=build_family_state_summary(
                available_models=self.available_models,
                current_best={"model_name": "ModelFamilyA", "composite_score": 0.90},
                trial_history=trial_history,
                memory_payload=memory_payload,
                diversity_settings={"max_family_share": 0.35},
            ),
            duplicate_settings=self.duplicate_settings,
            archive_records=[],
        )

        self.assertFalse(result["accepted"])
        self.assertEqual(result["semantic_rejection_reason"]["code"], "duplicate_proposal")

    def test_gate_research_proposal_rejects_near_duplicate_with_float_tolerance(self) -> None:
        proposal = self._research_proposal(
            proposed_change="Decrease learning_rate to 0.104 while keeping depth at 3.",
            candidate_params={"depth": 3, "learning_rate": 0.104},
        )
        trial_history = [
            {
                "model_name": "ModelFamilyA",
                "hyperparameters": json.dumps({"depth": 3, "learning_rate": 0.1000}),
                "proposal_family": "family-a",
                "composite_score": 0.89,
                "validation_verdict": "WARN",
            }
        ]
        memory_payload = {"runs": [], "accepted_experiments": []}
        family_state = build_family_state_summary(
            available_models=self.available_models,
            current_best={"model_name": "ModelFamilyA", "composite_score": 0.90},
            trial_history=trial_history,
            memory_payload=memory_payload,
            diversity_settings={"max_family_share": 0.35},
        )

        result = gate_research_proposal(
            proposal=proposal,
            available_models=self.available_models,
            current_run_signatures=set(),
            memory_payload=memory_payload,
            trial_history=trial_history,
            family_state=family_state,
            duplicate_settings=self.duplicate_settings,
            archive_records=[],
        )

        self.assertFalse(result["accepted"])
        self.assertEqual(result["semantic_rejection_reason"]["code"], "near_duplicate_proposal")

    def test_gate_research_proposal_rejects_schema_valid_but_meaningless_candidate_config(self) -> None:
        proposal = self._research_proposal(candidate_params={})
        memory_payload = {"runs": [], "accepted_experiments": []}

        result = gate_research_proposal(
            proposal=proposal,
            available_models=self.available_models,
            current_run_signatures=set(),
            memory_payload=memory_payload,
            trial_history=[],
            family_state=build_family_state_summary(
                available_models=self.available_models,
                current_best={"model_name": "ModelFamilyA", "composite_score": 0.90},
                trial_history=[],
                memory_payload=memory_payload,
                diversity_settings={"max_family_share": 0.35},
            ),
            duplicate_settings=self.duplicate_settings,
            archive_records=[],
        )

        self.assertFalse(result["accepted"])
        self.assertEqual(result["semantic_rejection_reason"]["code"], "semantically_invalid_candidate_config")

    def test_gate_research_proposal_rejects_inconsistent_hypothesis_vs_config(self) -> None:
        proposal = self._research_proposal(
            proposed_change="Decrease depth to 5 and keep learning_rate at 0.05.",
            candidate_params={"depth": 3, "learning_rate": 0.1},
        )
        memory_payload = {"runs": [], "accepted_experiments": []}

        result = gate_research_proposal(
            proposal=proposal,
            available_models=self.available_models,
            current_run_signatures=set(),
            memory_payload=memory_payload,
            trial_history=[],
            family_state=build_family_state_summary(
                available_models=self.available_models,
                current_best={"model_name": "ModelFamilyA", "composite_score": 0.90},
                trial_history=[],
                memory_payload=memory_payload,
                diversity_settings={"max_family_share": 0.35},
            ),
            duplicate_settings=self.duplicate_settings,
            archive_records=[],
        )

        self.assertFalse(result["accepted"])
        self.assertEqual(result["semantic_rejection_reason"]["code"], "inconsistent_expected_effect")

    def test_gate_research_proposal_rejects_novelty_claim_contradicted_by_archive(self) -> None:
        proposal = self._research_proposal(
            novelty_claim="No accepted run used this setting.",
        )
        archive_records = [
            {
                "outcome": "accepted",
                "proposal": {
                    "model_name": "ModelFamilyA",
                    "params": {"depth": 3, "learning_rate": 0.1},
                }
            }
        ]
        memory_payload = {"runs": [], "accepted_experiments": []}

        result = gate_research_proposal(
            proposal=proposal,
            available_models=self.available_models,
            current_run_signatures=set(),
            memory_payload=memory_payload,
            trial_history=[],
            family_state=build_family_state_summary(
                available_models=self.available_models,
                current_best={"model_name": "ModelFamilyA", "composite_score": 0.90},
                trial_history=[],
                memory_payload=memory_payload,
                diversity_settings={"max_family_share": 0.35},
            ),
            duplicate_settings=self.duplicate_settings,
            archive_records=archive_records,
        )

        self.assertFalse(result["accepted"])
        self.assertEqual(result["semantic_rejection_reason"]["code"], "unsupported_novelty_claim")

    def test_gate_research_proposal_ignores_pending_archive_records_for_novelty_checks(self) -> None:
        proposal = self._research_proposal(
            novelty_claim="This is a new configuration.",
        )
        archive_records = [
            {
                "outcome": "pending",
                "proposal": {
                    "model_name": "ModelFamilyA",
                    "params": {"depth": 3, "learning_rate": 0.1},
                },
            }
        ]
        memory_payload = {"runs": [], "accepted_experiments": []}

        result = gate_research_proposal(
            proposal=proposal,
            available_models=self.available_models,
            current_run_signatures=set(),
            memory_payload=memory_payload,
            trial_history=[],
            family_state=build_family_state_summary(
                available_models=self.available_models,
                current_best={"model_name": "ModelFamilyA", "composite_score": 0.90},
                trial_history=[],
                memory_payload=memory_payload,
                diversity_settings={"max_family_share": 0.35},
            ),
            duplicate_settings=self.duplicate_settings,
            archive_records=archive_records,
        )

        self.assertTrue(result["accepted"])
        self.assertEqual(result["semantic_validation_result"], "passed")

    def test_gate_research_proposal_rejects_invalid_candidate_config_mapping(self) -> None:
        proposal = self._research_proposal(candidate_model_name="MissingFamily")
        memory_payload = {"runs": [], "accepted_experiments": []}

        result = gate_research_proposal(
            proposal=proposal,
            available_models=self.available_models,
            current_run_signatures=set(),
            memory_payload=memory_payload,
            trial_history=[],
            family_state=build_family_state_summary(
                available_models=self.available_models,
                current_best={"model_name": "ModelFamilyA", "composite_score": 0.90},
                trial_history=[],
                memory_payload=memory_payload,
                diversity_settings={"max_family_share": 0.35},
            ),
            duplicate_settings=self.duplicate_settings,
            archive_records=[],
        )

        self.assertFalse(result["accepted"])
        self.assertEqual(result["semantic_rejection_reason"]["code"], "non_executable_semantic_config")

    def test_gate_proposal_rejects_near_duplicate_with_float_tolerance(self) -> None:
        trial_history = [
            {
                "model_name": "ModelFamilyA",
                "hyperparameters": json.dumps({"depth": 4, "learning_rate": 0.1000}),
                "proposal_family": "family-a",
                "composite_score": 0.89,
                "validation_verdict": "WARN",
            }
        ]
        family_state = build_family_state_summary(
            available_models=self.available_models,
            current_best={"model_name": "ModelFamilyA", "composite_score": 0.90},
            trial_history=trial_history,
            memory_payload={"runs": [], "accepted_experiments": []},
            diversity_settings={"max_family_share": 0.35},
        )

        result = gate_proposal(
            proposal={"model_name": "ModelFamilyA", "params": {"depth": 4, "learning_rate": 0.104}},
            available_models=self.available_models,
            current_run_signatures=set(),
            memory_payload={"runs": [], "accepted_experiments": []},
            trial_history=trial_history,
            family_state=family_state,
            duplicate_settings=self.duplicate_settings,
        )

        self.assertFalse(result["accepted"])
        self.assertEqual(result["reason"]["code"], "near_duplicate")
        self.assertLess(result["novelty_score"], 0.30)

    def test_gate_proposal_rejects_saturated_family_when_recent_run_is_too_similar(self) -> None:
        trial_history = [
            {
                "model_name": "ModelFamilyA",
                "hyperparameters": json.dumps({"depth": 3, "learning_rate": 0.09}),
                "proposal_family": "family-a",
                "composite_score": 0.91,
                "validation_verdict": "WARN",
            },
            {
                "model_name": "ModelFamilyA",
                "hyperparameters": json.dumps({"depth": 4, "learning_rate": 0.10}),
                "proposal_family": "family-a",
                "composite_score": 0.905,
                "validation_verdict": "WARN",
            },
            {
                "model_name": "ModelFamilyA",
                "hyperparameters": json.dumps({"depth": 4, "learning_rate": 0.11}),
                "proposal_family": "family-a",
                "composite_score": 0.904,
                "validation_verdict": "WARN",
            },
            {
                "model_name": "ModelFamilyB",
                "hyperparameters": json.dumps({"alpha": 0.6}),
                "proposal_family": "family-b",
                "composite_score": 0.86,
                "validation_verdict": "WARN",
            },
        ]
        family_state = build_family_state_summary(
            available_models=self.available_models,
            current_best={"model_name": "ModelFamilyA", "composite_score": 0.91},
            trial_history=trial_history,
            memory_payload={"runs": [], "accepted_experiments": []},
            diversity_settings={"max_family_share": 0.35},
        )

        result = gate_proposal(
            proposal={"model_name": "ModelFamilyA", "params": {"depth": 4, "learning_rate": 0.109}},
            available_models=self.available_models,
            current_run_signatures=set(),
            memory_payload={"runs": [], "accepted_experiments": []},
            trial_history=trial_history,
            family_state=family_state,
            duplicate_settings=self.duplicate_settings,
        )

        self.assertIn("ModelFamilyA", family_state["saturated_families"])
        self.assertFalse(result["accepted"])
        self.assertEqual(result["reason"]["code"], "saturated_family_similarity")

    def test_build_family_state_marks_underexplored_weak_family_from_recent_outcomes(self) -> None:
        trial_history = [
            {
                "model_name": "ModelFamilyA",
                "hyperparameters": json.dumps({"depth": 3, "learning_rate": 0.09}),
                "proposal_family": "family-a",
                "composite_score": 0.92,
                "validation_verdict": "WARN",
            },
            {
                "model_name": "ModelFamilyA",
                "hyperparameters": json.dumps({"depth": 4, "learning_rate": 0.10}),
                "proposal_family": "family-a",
                "composite_score": 0.91,
                "validation_verdict": "WARN",
            },
            {
                "model_name": "ModelFamilyB",
                "hyperparameters": json.dumps({"alpha": 0.8}),
                "proposal_family": "family-b",
                "composite_score": 0.75,
                "validation_verdict": "WARN",
            },
            {
                "model_name": "ModelFamilyB",
                "hyperparameters": json.dumps({"alpha": 0.7}),
                "proposal_family": "family-b",
                "composite_score": 0.74,
                "validation_verdict": "WARN",
            },
        ]

        family_state = build_family_state_summary(
            available_models=self.available_models,
            current_best={"model_name": "ModelFamilyA", "composite_score": 0.92},
            trial_history=trial_history,
            memory_payload={"runs": [], "accepted_experiments": []},
            diversity_settings={"max_family_share": 0.35},
        )

        self.assertEqual(family_state["strongest_active_family"], "ModelFamilyA")
        self.assertIn("ModelFamilyB", family_state["underexplored_weak_families"])
        self.assertIn("ModelFamilyB", family_state["temporarily_blocked_families"])

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

    def test_apply_keep_to_research_surface_rejects_duplicate_experiment_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            lab_path = Path(tmpdir) / "research_lab.py"
            lab_path.write_text(
                '"""editable surface"""\n'
                "# RESEARCH_SURFACE_STATE_START\n"
                "LAB_STATE = {\n"
                "    'surface_version': 1,\n"
                "    'accepted_experiments': [\n"
                "        {'experiment_id': 'confirm-001', 'proposal_family': 'boosting', 'model_name': 'LGBMRegressor', 'composite_score': 0.91},\n"
                "    ],\n"
                "    'recent_kept_families': ['boosting'],\n"
                "}\n"
                "# RESEARCH_SURFACE_STATE_END\n",
                encoding="utf-8",
            )

            with self.assertRaises(DuplicateExperimentError):
                apply_keep_to_research_surface(
                    research_lab_path=lab_path,
                    accepted_entry={
                        "experiment_id": "confirm-001",
                        "proposal_family": "boosting",
                        "model_name": "LGBMRegressor",
                        "composite_score": 0.92,
                    },
                )

    def test_validate_lab_state_integrity_rejects_conflicting_duplicate_scores(self) -> None:
        with self.assertRaises(DuplicateExperimentError):
            validate_lab_state_integrity(
                {
                    "surface_version": 1,
                    "accepted_experiments": [
                        {
                            "experiment_id": "confirm-001",
                            "proposal_family": "boosting",
                            "model_name": "LGBMRegressor",
                            "composite_score": 0.91,
                        },
                        {
                            "experiment_id": "confirm-001",
                            "proposal_family": "boosting",
                            "model_name": "LGBMRegressor",
                            "composite_score": 0.95,
                        },
                    ],
                }
            )

    def test_trial_budget_status(self) -> None:
        self.assertEqual(trial_budget_status(elapsed_seconds=31.0, max_trial_seconds=30.0), "budget_exceeded")
        self.assertEqual(trial_budget_status(elapsed_seconds=12.0, max_trial_seconds=30.0), "within_budget")
        self.assertEqual(trial_budget_status(elapsed_seconds=12.0, max_trial_seconds=0.0), "disabled")

    def test_acceptance_decision_uses_search_selection_artifact(self) -> None:
        final_metrics = {
            "artifact_kind": "search_selection",
            "model_name": "LGBMRegressor",
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
        self.assertEqual(decision["source_of_truth"], "best_search_result.json")
        self.assertEqual(decision["best_model_name"], "LGBMRegressor")

    def test_validate_final_artifact_consistency_uses_best_search_result_as_truth(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            outputs_dir = Path(tmpdir)
            (outputs_dir / "best_search_model.pkl").write_bytes(b"placeholder")
            (outputs_dir / "best_search_result.json").write_text(
                json.dumps(
                    {
                        "artifact_kind": "search_selection",
                        "model_name": "LGBMRegressor",
                        "hyperparameters": {"n_estimators": 300},
                        "composite_score": 0.83,
                        "validation_verdict": "WARN",
                    }
                ),
                encoding="utf-8",
            )
            (outputs_dir / "final_holdout_evaluation.json").write_text(
                json.dumps(
                    {
                        "artifact_kind": "final_holdout_evaluation",
                        "selected_model": {
                            "artifact_kind": "search_selection",
                            "model_name": "XGBRegressor",
                            "hyperparameters": {"n_estimators": 300},
                            "composite_score": 0.83,
                            "validation_verdict": "WARN",
                        },
                        "holdout_metrics": {
                            "stage": "final_holdout",
                            "partition": "holdout",
                            "aggregate": {"rmse": 4.1, "mae": 3.2, "r2": 0.8, "composite_score": 0.84},
                        },
                        "holdout_validation_report": {"verdict": "PASS"},
                    }
                ),
                encoding="utf-8",
            )

            report = validate_final_artifact_consistency(outputs_dir)

        self.assertFalse(report["consistent"])
        self.assertEqual(report["source_of_truth"], "best_search_result.json")
        self.assertEqual(report["mismatches"][0]["artifact"], "final_holdout_evaluation.json")

    def test_validate_final_artifact_consistency_detects_run_scope_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            outputs_dir = Path(tmpdir)
            (outputs_dir / "best_search_model.pkl").write_bytes(b"placeholder")
            (outputs_dir / "best_search_result.json").write_text(
                json.dumps(
                    {
                        "artifact_kind": "search_selection",
                        "run_id": "run-current",
                        "artifact_id": "best-current",
                        "model_name": "LGBMRegressor",
                        "hyperparameters": {"n_estimators": 300},
                        "composite_score": 0.83,
                        "validation_verdict": "WARN",
                    }
                ),
                encoding="utf-8",
            )
            (outputs_dir / "final_holdout_evaluation.json").write_text(
                json.dumps(
                    {
                        "artifact_kind": "final_holdout_evaluation",
                        "run_id": "run-stale",
                        "selected_model": {
                            "artifact_kind": "search_selection",
                            "run_id": "run-current",
                            "artifact_id": "best-current",
                            "model_name": "LGBMRegressor",
                            "hyperparameters": {"n_estimators": 300},
                            "composite_score": 0.83,
                            "validation_verdict": "WARN",
                        },
                        "holdout_metrics": {
                            "stage": "final_holdout",
                            "partition": "holdout",
                            "aggregate": {"rmse": 4.1, "mae": 3.2, "r2": 0.8, "composite_score": 0.84},
                        },
                        "holdout_validation_report": {"verdict": "PASS"},
                    }
                ),
                encoding="utf-8",
            )

            report = validate_final_artifact_consistency(outputs_dir)

        self.assertFalse(report["consistent"])
        self.assertTrue(any(item["field"] == "run_id" for item in report["mismatches"]))

    def test_validate_final_artifact_consistency_accepts_report_run_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            outputs_dir = Path(tmpdir)
            (outputs_dir / "best_search_model.pkl").write_bytes(b"placeholder")
            (outputs_dir / "best_search_result.json").write_text(
                json.dumps(
                    {
                        "artifact_kind": "search_selection",
                        "artifact_id": "best-artifact",
                        "artifact_metadata": {
                            "artifact_id": "best-artifact",
                            "run_id": "run-123",
                            "model_artifact_id": "model-artifact",
                        },
                        "model_name": "LGBMRegressor",
                        "hyperparameters": {"n_estimators": 300},
                        "composite_score": 0.83,
                        "validation_verdict": "WARN",
                        "model_artifact_id": "model-artifact",
                    }
                ),
                encoding="utf-8",
            )
            (outputs_dir / "final_holdout_evaluation.json").write_text(
                json.dumps(
                    {
                        "artifact_kind": "final_holdout_evaluation",
                        "artifact_id": "final-artifact",
                        "artifact_metadata": {
                            "artifact_id": "final-artifact",
                            "run_id": "run-123",
                            "model_artifact_id": "model-artifact",
                        },
                        "selected_model": {
                            "artifact_kind": "search_selection",
                            "artifact_id": "best-artifact",
                            "model_name": "LGBMRegressor",
                            "hyperparameters": {"n_estimators": 300},
                            "composite_score": 0.83,
                            "validation_verdict": "WARN",
                            "model_artifact_id": "model-artifact",
                        },
                        "holdout_metrics": {
                            "stage": "final_holdout",
                            "partition": "holdout",
                            "aggregate": {"rmse": 4.1, "mae": 3.2, "r2": 0.8, "composite_score": 0.84},
                        },
                        "holdout_validation_report": {"verdict": "WARN"},
                    }
                ),
                encoding="utf-8",
            )

            report = validate_final_artifact_consistency(outputs_dir)

        self.assertTrue(report["consistent"], report["mismatches"])

    def test_qwen_only_config_enables_deterministic_fallback(self) -> None:
        config = run_qwen_only._build_qwen_only_config()
        self.assertTrue(config["llm"]["allow_deterministic_fallback"])
        self.assertEqual(config["llm"]["backend_mode"], "ollama")

    def test_select_scout_candidates_falls_back_when_llm_returns_no_candidates(self) -> None:
        class EmptyProvider:
            llm_config = {"enabled": True}

            def get_proposals(self, context, n=3):
                return []

        with tempfile.TemporaryDirectory() as tmpdir:
            outputs_dir = Path(tmpdir)
            candidates, metadata = select_scout_candidates(
                brief={"goal": "Improve composite_score"},
                lab_state={},
                available_models=self.available_models,
                memory_payload={"runs": []},
                current_best={"model_name": "ModelFamilyA", "composite_score": 0.9},
                scout_limit=2,
                family_limit=1,
                current_run_signatures=set(),
                proposal_engine=EmptyProvider(),
                trial_history=[],
                search_progress={},
                failure_patterns={},
                knowledge_context="",
                archive_records=[],
                preflight_result=None,
                allow_deterministic_fallback=True,
                exploit_delta_ratio=0.15,
            )

        self.assertGreaterEqual(len(candidates), 1)
        self.assertEqual(metadata["proposal_mode"], "fallback_used")


if __name__ == "__main__":
    unittest.main()
