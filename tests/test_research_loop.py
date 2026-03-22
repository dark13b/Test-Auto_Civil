import json
import sys
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd

import research_loop
from novelty_scorer import NoveltyScorer
from proposal_engine import ProposalEngine, ProposalParseFailure, ProposalPreflightFailure
from research_loop import (
    _append_synchronized_results,
    _initialize_research_log,
    _initialize_results_csv,
    _build_run_manifest_payload,
    _select_scout_candidates,
)


class StubProposalEngine:
    def __init__(
        self,
        proposals: list[dict],
        *,
        status: str = "llm_success",
        smoke_test_ok: bool = True,
        smoke_test_status: str = "llm_success",
        smoke_test_error: str | None = None,
        smoke_runs: list[dict[str, object]] | None = None,
    ) -> None:
        self._proposals = proposals
        self._status = status
        self._smoke_test_ok = smoke_test_ok
        self._smoke_test_status = smoke_test_status
        self._smoke_test_error = smoke_test_error
        self._smoke_runs = [dict(item) for item in smoke_runs] if smoke_runs is not None else None
        self.backend_name = "stub"
        self.last_interaction_summary = {
            "model": "qwen3:8b",
            "prompt_variant": "rich",
        }

    def is_available(self) -> bool:
        return True

    def generate_experiment_proposals(self, **_: object) -> list[dict]:
        return list(self._proposals)

    def generate_research_proposals(self, **_: object) -> dict[str, object]:
        normalized = []
        for proposal in self._proposals:
            normalized.append(
                {
                    **proposal,
                    "proposal_source": proposal.get("proposal_source", "llm"),
                    "research_proposal": proposal.get(
                        "research_proposal",
                        {
                            "hypothesis": proposal.get("hypothesis", "stub hypothesis"),
                            "rationale": "stub rationale",
                            "change_type": "hyperparameter",
                            "target_component": proposal.get("model_name", "unknown"),
                            "proposed_change": "stub change",
                            "expected_direction": "improve",
                            "expected_metric_effect": {
                                "metric": "rmse",
                                "direction": "down",
                                "magnitude_estimate": "small",
                            },
                            "confidence": 0.5,
                            "novelty_claim": "stub novelty",
                            "risk_notes": "stub risk",
                            "candidate_config": {
                                "model_name": proposal.get("model_name", "unknown"),
                                "params": dict(proposal.get("params", {})),
                            },
                        },
                    ),
                }
            )
        return {
            "status": self._status,
            "backend": self.backend_name,
            "model": "qwen3:8b",
            "prompt_variant": "rich",
            "proposals": normalized,
        }

    def run_proposal_smoke_test(self, **_: object) -> dict[str, object]:
        if self._smoke_runs is not None:
            if not self._smoke_runs:
                raise AssertionError("No stub smoke-test result remaining")
            return dict(self._smoke_runs.pop(0))
        return {
            "ok": self._smoke_test_ok,
            "status": self._smoke_test_status,
            "backend": self.backend_name,
            "model": "qwen3:8b",
            "error": self._smoke_test_error,
        }

    def summarize_run(self, **_: object) -> dict[str, object]:
        return {"backend": "stub", "available": True, "summary": "stub-summary"}


class AmbiguousBackend:
    backend_name = "ollama"

    def is_available(self) -> bool:
        return True

    def generate_text(self, prompt: str, **_: object) -> dict[str, object]:
        return {
            "backend": "ollama",
            "model": "qwen3:4b",
            "response_text": '{"model_name":"RandomForestRegressor","params":{"n_estimators":200}} {"model_name":"ExtraTreesRegressor","params":{"n_estimators":300}}',
            "text": "",
        }


class ResearchLoopPersistenceTests(unittest.TestCase):
    def test_load_current_best_result_prefers_best_search_result_over_holdout_augmented_final_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            outputs_dir = Path(tmpdir)
            (outputs_dir / research_loop.FINAL_METRICS_FILENAME).write_text(
                json.dumps(
                    {
                        "best_search_metrics": {
                            "model_name": "HoldoutDecoratedModel",
                            "composite_score": 0.95,
                            "validation_verdict": "FAIL",
                            "holdout_metrics": {"rmse": 0.1},
                        }
                    }
                ),
                encoding="utf-8",
            )
            (outputs_dir / research_loop.BEST_RESULT_FILENAME).write_text(
                json.dumps(
                    {
                        "model_name": "SearchWinner",
                        "composite_score": 0.83,
                        "validation_verdict": "PASS",
                    }
                ),
                encoding="utf-8",
            )

            current_best = research_loop._load_current_best_result(
                outputs_dir,
                baseline_metrics={"model_name": "Baseline", "composite_score": 0.80, "validation_verdict": "WARN"},
            )

        self.assertEqual(current_best["model_name"], "SearchWinner")
        self.assertEqual(current_best["validation_verdict"], "PASS")

    def test_append_synchronized_results_writes_one_aligned_row_to_both_csvs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            outputs_dir = Path(tmpdir)
            writer = research_loop._build_results_sync_writer(outputs_dir)
            record = {
                "trial_number": 5,
                "experiment_id": "confirm-005",
                "selection_status": "kept",
                "proposal_status": "llm_success",
                "semantic_validation_result": "passed",
                "semantic_validation_error": None,
                "semantic_rejection_reason": None,
            }

            _append_synchronized_results(writer, record)
            _append_synchronized_results(writer, record)

            results_rows = pd.read_csv(outputs_dir / research_loop.RESEARCH_RESULTS_FILENAME)
            compat_rows = pd.read_csv(outputs_dir / "optuna_results.csv")

        self.assertEqual(len(results_rows), 1)
        self.assertTrue(results_rows.equals(compat_rows))
        self.assertEqual(str(results_rows.loc[0, "experiment_id"]), "confirm-005")
        self.assertEqual(str(results_rows.loc[0, "semantic_validation_result"]), "passed")

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
            allow_deterministic_fallback=True,
        )

        self.assertTrue(candidates)
        self.assertEqual(metadata["proposal_mode"], "fallback_used")
        self.assertEqual(metadata["proposal_backend"], "fallback")
        self.assertEqual(metadata["proposal_status"], "fallback_used")

    def test_select_scout_candidates_falls_back_when_json_repair_is_ambiguous(self) -> None:
        available_models = {
            "RandomForestRegressor": {
                "display_name": "RandomForest",
                "search_space": {
                    "n_estimators": {"type": "int", "low": 100, "high": 300, "step": 100},
                },
            }
        }
        proposal_engine = ProposalEngine(
            backend=AmbiguousBackend(),
            log_interactions=False,
            llm_config={
                "compact_prompt_models": ["qwen3:4b"],
                "enable_regeneration_on_reject": False,
                "max_regeneration_attempts": 0,
                "duplicate_similarity_thresholds": {
                    "numeric_tolerance": 0.05,
                    "float_round_digits": 4,
                },
                "temporarily_block_saturated_families": True,
                "diversity": {"max_family_share": 0.35},
            },
        )

        candidates, metadata = _select_scout_candidates(
            brief={"scout_candidates_per_cycle": 1},
            lab_state={"accepted_experiments": []},
            available_models=available_models,
            memory_payload={"runs": [], "accepted_experiments": []},
            current_best={"model_name": "RandomForestRegressor", "hyperparameters": {}},
            scout_limit=1,
            family_limit=1,
            current_run_signatures=set(),
            proposal_engine=proposal_engine,
            allow_deterministic_fallback=True,
        )

        self.assertTrue(candidates)
        self.assertEqual(metadata["proposal_mode"], "fallback_used")

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
                        "proposal_source": "llm",
                        "research_proposal": {
                            "hypothesis": "Try stronger regularization.",
                            "rationale": "The current family is close but unstable.",
                            "change_type": "hyperparameter",
                            "target_component": "LGBMRegressor",
                            "proposed_change": "Increase learning_rate to 0.1 at 500 trees.",
                            "expected_direction": "improve",
                            "expected_metric_effect": {
                                "metric": "rmse",
                                "direction": "down",
                                "magnitude_estimate": "0.03-0.08 MPa",
                            },
                            "confidence": 0.61,
                            "novelty_claim": "No accepted run used this setting.",
                            "risk_notes": "May overfit if num_leaves remains large.",
                            "candidate_config": {
                                "model_name": "LGBMRegressor",
                                "params": {"n_estimators": 500, "learning_rate": 0.1},
                            },
                        },
                        "params": {"n_estimators": 500, "learning_rate": 0.1},
                    }
                ]
            ),
        )

        self.assertEqual(metadata["proposal_mode"], "llm")
        self.assertEqual(metadata["proposal_status"], "llm_success")
        self.assertEqual(candidates[0]["stage"], "scout")
        self.assertEqual(candidates[0]["experiment_id"], "llm-scout-001")
        self.assertEqual(candidates[0]["proposal_family"], "llm-boosting")
        self.assertEqual(candidates[0]["proposal_source"], "llm")

    def test_select_scout_candidates_raises_when_fallback_disabled_and_llm_parse_fails(self) -> None:
        available_models = {
            "RandomForestRegressor": {
                "display_name": "RandomForest",
                "search_space": {
                    "n_estimators": {"type": "int", "low": 100, "high": 300, "step": 100},
                },
            }
        }

        class ParseFailProposalEngine(StubProposalEngine):
            def generate_research_proposals(self, **_: object) -> dict[str, object]:
                raise ProposalParseFailure("invalid json", failure_kind="llm_parse_failure")

        with self.assertRaises(ProposalParseFailure):
            _select_scout_candidates(
                brief={"scout_candidates_per_cycle": 1},
                lab_state={"accepted_experiments": []},
                available_models=available_models,
                memory_payload={"runs": [], "accepted_experiments": []},
                current_best={"model_name": "RandomForestRegressor", "hyperparameters": {}},
                scout_limit=1,
                family_limit=1,
                current_run_signatures=set(),
                proposal_engine=ParseFailProposalEngine([]),
                allow_deterministic_fallback=False,
            )

    def test_run_engineering_research_loop_keeps_confirmed_improvement(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            outputs_dir = Path(tmpdir)
            (outputs_dir / "baseline_model.pkl").write_bytes(b"baseline")

            available_models = {
                "ModelFamilyA": {
                    "display_name": "Model A",
                    "search_space": {
                        "depth": {"type": "int", "low": 1, "high": 8},
                    },
                }
            }
            baseline_metrics = {
                "model_name": "ModelFamilyA",
                "display_name": "Model A",
                "hyperparameters": {"depth": 2},
                "composite_score": 0.80,
                "validation_verdict": "WARN",
                "validation_report": {},
                "test_metrics": {},
            }
            scout_result = {
                "model_name": "ModelFamilyA",
                "display_name": "Model A",
                "hyperparameters": {"depth": 3},
                "composite_score": 0.84,
                "validation_verdict": "WARN",
                "validation_report": {},
                "test_metrics": {},
            }
            confirm_result = {
                "model_name": "ModelFamilyA",
                "display_name": "Model A",
                "hyperparameters": {"depth": 3},
                "composite_score": 0.86,
                "validation_verdict": "WARN",
                "validation_report": {},
                "test_metrics": {},
            }

            def fake_save_pickle_artifact(path: Path, model: object, **_: object) -> None:
                Path(path).write_bytes(b"model")

            def fake_sync_final_artifacts_from_source_of_truth(
                *,
                outputs_dir: Path,
                baseline_metrics: dict,
                best_result: dict,
                best_model_source_path: Path,
            ) -> dict:
                payload = {
                    "baseline_metrics": baseline_metrics,
                    "best_search_metrics": best_result,
                    "composite_improvement_pct": 7.5,
                    "best_model_name": best_result["model_name"],
                    "best_model_hyperparameters": best_result["hyperparameters"],
                    "validation_verdict": best_result["validation_verdict"],
                }
                (outputs_dir / research_loop.FINAL_METRICS_FILENAME).write_text(
                    json.dumps(payload),
                    encoding="utf-8",
                )
                (outputs_dir / research_loop.BEST_MODEL_FILENAME).write_bytes(best_model_source_path.read_bytes())
                return payload

            config = {
                "experiment": {"random_seed": 42},
                "engineering": {"uncertainty_method": "conformal"},
                "research": {
                    "max_cycles": 1,
                    "max_family_repeats_per_cycle": 1,
                    "scout_cv_repeats": 1,
                    "confirm_cv_repeats": 1,
                    "max_trial_seconds": 0.0,
                    "max_runtime_minutes": 0.0,
                    "rebuild_reports_on_keep": False,
                    "brief_path": "research_brief.md",
                    "editable_surface_path": "research_lab.py",
                },
                "llm": {
                    "enabled": True,
                    "tasks": {"summary_enabled": True},
                    "compact_prompt_models": ["qwen3:4b"],
                    "default_local_proposal_model": "qwen3:8b",
                },
            }

            with ExitStack() as stack:
                stack.enter_context(patch.object(research_loop, "load_config", return_value=config))
                stack.enter_context(patch.object(research_loop, "get_outputs_dir", return_value=outputs_dir))
                stack.enter_context(patch.object(research_loop, "set_global_seed"))
                stack.enter_context(patch.object(research_loop, "_read_baseline_metrics", return_value=baseline_metrics))
                stack.enter_context(patch.object(research_loop, "load_human_research_brief", return_value={
                    "goal": "Improve composite_score",
                    "acceptance_metric": "composite_score",
                    "min_improvement_pct": 0.0,
                    "required_model_families": [],
                    "scout_candidates_per_cycle": 1,
                    "confirm_top_k": 1,
                }))
                stack.enter_context(patch.object(research_loop, "get_available_model_configs", return_value=available_models))
                stack.enter_context(patch.object(research_loop, "load_dataset", return_value="dataset"))
                stack.enter_context(
                    patch.object(
                        research_loop,
                        "split_dataset",
                        return_value=("x_train", "x_val", "x_test", "y_train", "y_val", "y_test"),
                    )
                )
                stack.enter_context(patch.object(research_loop.EngineeringValidator, "from_config", return_value=object()))
                stack.enter_context(
                    patch.object(research_loop, "read_research_surface_state", return_value={"accepted_experiments": []})
                )
                stack.enter_context(
                    patch.object(
                        research_loop,
                        "_build_proposal_engine",
                        return_value=StubProposalEngine([
                            {
                                "model_name": "ModelFamilyA",
                                "display_name": "Model A",
                                "proposal_family": "llm-family-a",
                                "hypothesis": "Try a slightly deeper model.",
                                "params": {"depth": 3},
                            }
                        ]),
                    )
                )
                stack.enter_context(
                    patch.object(
                        research_loop,
                        "_evaluate_stage_candidate",
                        side_effect=[(object(), scout_result), (object(), confirm_result)],
                    )
                )
                stack.enter_context(
                    patch.object(
                        research_loop.research_lab,
                        "confirm_experiments",
                        return_value=[
                            {
                                "experiment_id": "confirm-001",
                                "stage": "confirm",
                                "model_name": "ModelFamilyA",
                                "display_name": "Model A",
                                "proposal_family": "llm-family-a",
                                "hypothesis": "Confirm the deeper model.",
                                "params": {"depth": 3},
                            }
                        ],
                    )
                )
                stack.enter_context(patch.object(research_loop, "_evaluate_reference_if_possible", return_value=baseline_metrics))
                stack.enter_context(patch.object(research_loop, "save_pickle_artifact", side_effect=fake_save_pickle_artifact))
                stack.enter_context(
                    patch.object(
                        research_loop,
                        "_sync_final_artifacts_from_source_of_truth",
                        side_effect=fake_sync_final_artifacts_from_source_of_truth,
                    )
                )
                stack.enter_context(patch.object(research_loop, "apply_keep_to_research_surface"))
                stack.enter_context(
                    patch.object(
                        research_loop,
                        "validate_final_artifact_consistency",
                        return_value={"consistent": True, "mismatches": []},
                    )
                )
                stack.enter_context(patch.object(research_loop, "_build_results_sync_writer", return_value=Mock()))
                stack.enter_context(patch.object(research_loop, "log_status"))
                recalibrate_mock = stack.enter_context(patch("uncertainty.recalibrate_uncertainty_artifacts"))
                best_result = research_loop.run_engineering_research_loop(cycles_override=1, with_report=False)

            final_metrics = json.loads((outputs_dir / research_loop.FINAL_METRICS_FILENAME).read_text(encoding="utf-8"))
            manifest = json.loads((outputs_dir / research_loop.RUN_MANIFEST_FILENAME).read_text(encoding="utf-8"))
            run_scoped_manifest = json.loads(
                (outputs_dir / "runs" / manifest["run_id"] / research_loop.RUN_MANIFEST_FILENAME).read_text(encoding="utf-8")
            )
            self.assertEqual(best_result["composite_score"], 0.86)
            self.assertEqual(final_metrics["best_search_metrics"]["composite_score"], 0.86)
            self.assertEqual(run_scoped_manifest["run_id"], manifest["run_id"])
            self.assertEqual(manifest["final_run_status"], "success")
            self.assertEqual(manifest["preflight_status"], "passed")
            self.assertEqual(manifest["smoke_test_status"], "llm_success")
            self.assertEqual(manifest["selection_metric_name"], "composite_score")
            self.assertEqual(manifest["selection_partition"], "validation")
            self.assertEqual(manifest["fallback_allowed"], False)
            self.assertEqual(manifest["number_of_candidates_evaluated"], 2)
            self.assertEqual(manifest["holdout_touched_during_search"], False)
            self.assertIn("proposal_status_counts", manifest)
            recalibrate_mock.assert_called_once()

    def test_run_engineering_research_loop_defers_holdout_report_until_acceptance_written(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            outputs_dir = Path(tmpdir)
            (outputs_dir / "baseline_model.pkl").write_bytes(b"baseline")

            available_models = {
                "ModelFamilyA": {
                    "display_name": "Model A",
                    "search_space": {
                        "depth": {"type": "int", "low": 1, "high": 8},
                    },
                }
            }
            baseline_metrics = {
                "model_name": "ModelFamilyA",
                "display_name": "Model A",
                "hyperparameters": {"depth": 2},
                "composite_score": 0.80,
                "validation_verdict": "WARN",
                "validation_report": {},
                "test_metrics": {},
            }
            scout_result = {
                "model_name": "ModelFamilyA",
                "display_name": "Model A",
                "hyperparameters": {"depth": 3},
                "composite_score": 0.84,
                "validation_verdict": "WARN",
                "validation_report": {},
                "test_metrics": {},
            }
            confirm_result = {
                "model_name": "ModelFamilyA",
                "display_name": "Model A",
                "hyperparameters": {"depth": 3},
                "composite_score": 0.86,
                "validation_verdict": "WARN",
                "validation_report": {},
                "test_metrics": {},
            }

            def fake_save_pickle_artifact(path: Path, model: object, **_: object) -> None:
                Path(path).write_bytes(b"model")

            def fake_sync_final_artifacts_from_source_of_truth(
                *,
                outputs_dir: Path,
                baseline_metrics: dict,
                best_result: dict,
                best_model_source_path: Path,
            ) -> dict:
                payload = {
                    "baseline_metrics": baseline_metrics,
                    "best_search_metrics": best_result,
                    "composite_improvement_pct": 7.5,
                    "best_model_name": best_result["model_name"],
                    "best_model_hyperparameters": best_result["hyperparameters"],
                    "validation_verdict": best_result["validation_verdict"],
                }
                (outputs_dir / research_loop.FINAL_METRICS_FILENAME).write_text(
                    json.dumps(payload),
                    encoding="utf-8",
                )
                (outputs_dir / research_loop.BEST_MODEL_FILENAME).write_bytes(best_model_source_path.read_bytes())
                return payload

            def fake_report_main() -> int:
                acceptance_path = outputs_dir / research_loop.FINAL_ACCEPTANCE_FILENAME
                self.assertTrue(
                    acceptance_path.exists(),
                    "Holdout report must run only after final acceptance has been written.",
                )
                return 0

            config = {
                "experiment": {"random_seed": 42},
                "engineering": {"uncertainty_method": "conformal"},
                "research": {
                    "max_cycles": 1,
                    "max_family_repeats_per_cycle": 1,
                    "scout_cv_repeats": 1,
                    "confirm_cv_repeats": 1,
                    "max_trial_seconds": 0.0,
                    "max_runtime_minutes": 0.0,
                    "rebuild_reports_on_keep": True,
                    "brief_path": "research_brief.md",
                    "editable_surface_path": "research_lab.py",
                },
                "llm": {
                    "enabled": True,
                    "tasks": {"summary_enabled": False},
                    "compact_prompt_models": ["qwen3:4b"],
                    "default_local_proposal_model": "qwen3:8b",
                },
            }

            with ExitStack() as stack:
                stack.enter_context(patch.object(research_loop, "load_config", return_value=config))
                stack.enter_context(patch.object(research_loop, "get_outputs_dir", return_value=outputs_dir))
                stack.enter_context(patch.object(research_loop, "set_global_seed"))
                stack.enter_context(patch.object(research_loop, "_read_baseline_metrics", return_value=baseline_metrics))
                stack.enter_context(
                    patch.object(
                        research_loop,
                        "load_human_research_brief",
                        return_value={
                            "goal": "Improve composite_score",
                            "acceptance_metric": "composite_score",
                            "min_improvement_pct": 0.0,
                            "required_model_families": [],
                            "scout_candidates_per_cycle": 1,
                            "confirm_top_k": 1,
                        },
                    )
                )
                stack.enter_context(patch.object(research_loop, "get_available_model_configs", return_value=available_models))
                stack.enter_context(patch.object(research_loop, "load_dataset", return_value="dataset"))
                stack.enter_context(
                    patch.object(
                        research_loop,
                        "split_dataset",
                        return_value=("x_train", "x_val", "x_test", "y_train", "y_val", "y_test"),
                    )
                )
                stack.enter_context(patch.object(research_loop.EngineeringValidator, "from_config", return_value=object()))
                stack.enter_context(
                    patch.object(research_loop, "read_research_surface_state", return_value={"accepted_experiments": []})
                )
                stack.enter_context(
                    patch.object(
                        research_loop,
                        "_build_proposal_engine",
                        return_value=StubProposalEngine(
                            [
                                {
                                    "model_name": "ModelFamilyA",
                                    "display_name": "Model A",
                                    "proposal_family": "llm-family-a",
                                    "hypothesis": "Try a slightly deeper model.",
                                    "params": {"depth": 3},
                                }
                            ]
                        ),
                    )
                )
                stack.enter_context(
                    patch.object(
                        research_loop,
                        "_evaluate_stage_candidate",
                        side_effect=[(object(), scout_result), (object(), confirm_result)],
                    )
                )
                stack.enter_context(
                    patch.object(
                        research_loop.research_lab,
                        "confirm_experiments",
                        return_value=[
                            {
                                "experiment_id": "confirm-001",
                                "stage": "confirm",
                                "model_name": "ModelFamilyA",
                                "display_name": "Model A",
                                "proposal_family": "llm-family-a",
                                "hypothesis": "Confirm the deeper model.",
                                "params": {"depth": 3},
                            }
                        ],
                    )
                )
                stack.enter_context(patch.object(research_loop, "_evaluate_reference_if_possible", return_value=baseline_metrics))
                stack.enter_context(patch.object(research_loop, "save_pickle_artifact", side_effect=fake_save_pickle_artifact))
                stack.enter_context(
                    patch.object(
                        research_loop,
                        "_sync_final_artifacts_from_source_of_truth",
                        side_effect=fake_sync_final_artifacts_from_source_of_truth,
                    )
                )
                stack.enter_context(patch.object(research_loop, "apply_keep_to_research_surface"))
                stack.enter_context(
                    patch.object(
                        research_loop,
                        "validate_final_artifact_consistency",
                        return_value={"consistent": True, "mismatches": []},
                    )
                )
                stack.enter_context(patch.object(research_loop, "_build_results_sync_writer", return_value=Mock()))
                stack.enter_context(patch.object(research_loop, "log_status"))
                stack.enter_context(patch("uncertainty.recalibrate_uncertainty_artifacts"))
                report_mock = stack.enter_context(patch("report.main", side_effect=fake_report_main))
                best_result = research_loop.run_engineering_research_loop(cycles_override=1, with_report=True)

            self.assertEqual(best_result["composite_score"], 0.86)
            report_mock.assert_called_once()

    def test_validate_only_path_still_works(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            outputs_dir = Path(tmpdir)
            with (
                patch.object(research_loop, "load_config", return_value={"paths": {"outputs_dir": str(outputs_dir)}}),
                patch.object(research_loop, "get_outputs_dir", return_value=outputs_dir),
                patch.object(
                    research_loop,
                    "validate_final_artifact_consistency",
                    return_value={"consistent": True, "mismatches": []},
                ),
                patch.object(research_loop, "write_json_file"),
                patch.object(research_loop, "log_status"),
                patch.object(sys, "argv", ["research_loop.py", "--validate-only"]),
            ):
                exit_code = research_loop.main()

        self.assertEqual(exit_code, 0)

    def test_run_engineering_research_loop_rejects_low_novelty_before_evaluation(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            outputs_dir = Path(tmpdir)
            (outputs_dir / "baseline_model.pkl").write_bytes(b"baseline")

            available_models = {
                "ModelFamilyA": {
                    "display_name": "Model A",
                    "search_space": {
                        "depth": {"type": "int", "low": 1, "high": 8},
                    },
                }
            }
            baseline_metrics = {
                "model_name": "ModelFamilyA",
                "display_name": "Model A",
                "hyperparameters": {"depth": 2},
                "composite_score": 0.80,
                "validation_verdict": "WARN",
                "validation_report": {},
                "test_metrics": {"rmse": 4.5},
            }
            memory_payload = {
                "schema_version": 3,
                "accepted_experiments": [],
                "runs": [
                    {
                        "run_id": "old-run",
                        "trials": [
                            {
                                "model_name": "ModelFamilyA",
                                "proposal_family": "family-a",
                                "params": {"depth": 3},
                                "selection_status": "scout_no_improvement",
                                "validation_verdict": "WARN",
                                "signature": ["ModelFamilyA", '{"depth":3}'],
                            }
                        ],
                    }
                ],
            }

            config = {
                "experiment": {"random_seed": 42},
                "engineering": {"uncertainty_method": "conformal"},
                "research": {
                    "max_cycles": 1,
                    "max_family_repeats_per_cycle": 1,
                    "scout_cv_repeats": 1,
                    "confirm_cv_repeats": 1,
                    "max_trial_seconds": 0.0,
                    "max_runtime_minutes": 0.0,
                    "rebuild_reports_on_keep": False,
                    "brief_path": "research_brief.md",
                    "editable_surface_path": "research_lab.py",
                    "novelty_gate_threshold": 0.20,
                    "hypothesis_archive_filename": "hypothesis_archive.json",
                },
                "llm": {
                    "enabled": True,
                    "tasks": {"summary_enabled": False},
                    "compact_prompt_models": ["qwen3:4b"],
                },
            }

            with ExitStack() as stack:
                stack.enter_context(patch.object(research_loop, "load_config", return_value=config))
                stack.enter_context(patch.object(research_loop, "get_outputs_dir", return_value=outputs_dir))
                stack.enter_context(patch.object(research_loop, "set_global_seed"))
                stack.enter_context(patch.object(research_loop, "_read_baseline_metrics", return_value=baseline_metrics))
                stack.enter_context(
                    patch.object(
                        research_loop,
                        "load_human_research_brief",
                        return_value={
                            "goal": "Improve composite_score",
                            "acceptance_metric": "composite_score",
                            "min_improvement_pct": 0.0,
                            "required_model_families": [],
                            "scout_candidates_per_cycle": 1,
                            "confirm_top_k": 1,
                        },
                    )
                )
                stack.enter_context(patch.object(research_loop, "get_available_model_configs", return_value=available_models))
                stack.enter_context(patch.object(research_loop, "load_dataset", return_value="dataset"))
                stack.enter_context(
                    patch.object(
                        research_loop,
                        "split_dataset",
                        return_value=("x_train", "x_val", "x_test", "y_train", "y_val", "y_test"),
                    )
                )
                stack.enter_context(patch.object(research_loop.EngineeringValidator, "from_config", return_value=object()))
                stack.enter_context(
                    patch.object(research_loop, "read_research_surface_state", return_value={"accepted_experiments": []})
                )
                stack.enter_context(
                    patch.object(
                        research_loop,
                        "_build_proposal_engine",
                        return_value=StubProposalEngine(
                            [
                                {
                                    "model_name": "ModelFamilyA",
                                    "display_name": "Model A",
                                    "proposal_family": "family-a",
                                    "hypothesis": "Duplicate depth",
                                    "expected_delta": 0.1,
                                    "params": {"depth": 3},
                                }
                            ]
                        ),
                    )
                )
                evaluate_mock = stack.enter_context(patch.object(research_loop, "_evaluate_stage_candidate"))
                stack.enter_context(
                    patch.object(
                        research_loop,
                        "load_or_initialize_experiment_memory",
                        side_effect=lambda _path: memory_payload,
                    )
                )
                stack.enter_context(
                    patch.object(
                        research_loop,
                        "validate_final_artifact_consistency",
                        return_value={"consistent": True, "mismatches": []},
                    )
                )
                stack.enter_context(
                    patch.object(
                        research_loop,
                        "_sync_final_artifacts_from_source_of_truth",
                        return_value={"best_search_metrics": baseline_metrics},
                    )
                )
                stack.enter_context(patch.object(research_loop, "_build_results_sync_writer", return_value=Mock()))
                stack.enter_context(patch.object(research_loop, "log_status"))
                best_result = research_loop.run_engineering_research_loop(cycles_override=1, with_report=False)

            self.assertEqual(best_result["model_name"], "ModelFamilyA")
            evaluate_mock.assert_not_called()

    def test_run_engineering_research_loop_aborts_when_preflight_fails_and_fallback_disabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            outputs_dir = Path(tmpdir)
            (outputs_dir / "baseline_model.pkl").write_bytes(b"baseline")

            config = {
                "experiment": {"random_seed": 42},
                "engineering": {"uncertainty_method": "conformal"},
                "research": {
                    "max_cycles": 1,
                    "max_family_repeats_per_cycle": 1,
                    "scout_cv_repeats": 1,
                    "confirm_cv_repeats": 1,
                    "max_trial_seconds": 0.0,
                    "max_runtime_minutes": 0.0,
                    "rebuild_reports_on_keep": False,
                    "brief_path": "research_brief.md",
                    "editable_surface_path": "research_lab.py",
                },
                "llm": {
                    "enabled": True,
                    "allow_deterministic_fallback": False,
                    "tasks": {"summary_enabled": False},
                },
            }

            with ExitStack() as stack:
                stack.enter_context(patch.object(research_loop, "load_config", return_value=config))
                stack.enter_context(patch.object(research_loop, "get_outputs_dir", return_value=outputs_dir))
                stack.enter_context(patch.object(research_loop, "set_global_seed"))
                stack.enter_context(
                    patch.object(
                        research_loop,
                        "_read_baseline_metrics",
                        return_value={
                            "model_name": "ModelFamilyA",
                            "display_name": "Model A",
                            "hyperparameters": {"depth": 2},
                            "composite_score": 0.80,
                            "validation_verdict": "WARN",
                            "validation_report": {},
                            "test_metrics": {},
                        },
                    )
                )
                stack.enter_context(
                    patch.object(
                        research_loop,
                        "load_human_research_brief",
                        return_value={
                            "goal": "Improve composite_score",
                            "acceptance_metric": "composite_score",
                            "min_improvement_pct": 0.0,
                            "required_model_families": [],
                            "scout_candidates_per_cycle": 1,
                            "confirm_top_k": 1,
                        },
                    )
                )
                stack.enter_context(patch.object(research_loop, "get_available_model_configs", return_value={}))
                stack.enter_context(patch.object(research_loop, "load_dataset", return_value="dataset"))
                stack.enter_context(
                    patch.object(
                        research_loop,
                        "split_dataset",
                        return_value=("x_train", "x_val", "x_test", "y_train", "y_val", "y_test"),
                    )
                )
                stack.enter_context(patch.object(research_loop.EngineeringValidator, "from_config", return_value=object()))
                stack.enter_context(
                    patch.object(
                        research_loop,
                        "_build_proposal_engine",
                        return_value=StubProposalEngine(
                            [],
                            smoke_test_ok=False,
                            smoke_test_status="llm_backend_failure",
                            smoke_test_error="backend unreachable",
                        ),
                    )
                )
                stack.enter_context(patch.object(research_loop, "_build_results_sync_writer", return_value=Mock()))
                stack.enter_context(patch.object(research_loop, "log_status"))

                with self.assertRaises(ProposalPreflightFailure):
                    research_loop.run_engineering_research_loop(cycles_override=1, with_report=False)

            manifest = json.loads((outputs_dir / research_loop.RUN_MANIFEST_FILENAME).read_text(encoding="utf-8"))
            run_scoped_manifest = json.loads(
                (outputs_dir / "runs" / manifest["run_id"] / research_loop.RUN_MANIFEST_FILENAME).read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["final_run_status"], "aborted")
            self.assertEqual(manifest["preflight_status"], "failed")
            self.assertIsNotNone(manifest["final_abort_reason"])
            self.assertEqual(manifest["number_of_candidates_evaluated"], 0)
            self.assertFalse(manifest["holdout_touched_during_search"])
            self.assertEqual(run_scoped_manifest["final_run_status"], "aborted")

    def test_analyze_interactions_path_still_works(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            outputs_dir = Path(tmpdir)
            (outputs_dir / "llm_interactions.jsonl").write_text(
                json.dumps(
                    {
                        "extracted_from_channel": "thinking",
                        "parse_success": True,
                        "repair_used": False,
                        "rejection_reason": {"code": "near_duplicate"},
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            with (
                patch.object(research_loop, "load_config", return_value={"paths": {"outputs_dir": str(outputs_dir)}}),
                patch.object(research_loop, "get_outputs_dir", return_value=outputs_dir),
                patch.object(research_loop, "log_status"),
                patch.object(sys, "argv", ["research_loop.py", "--analyze-interactions"]),
            ):
                exit_code = research_loop.main()

        self.assertEqual(exit_code, 0)

    def test_summarize_smoke_test_trials_reports_rates_latency_and_compatibility(self) -> None:
        summary = research_loop.summarize_smoke_test_trials(
            [
                {
                    "ok": True,
                    "status": "llm_success",
                    "backend": "ollama",
                    "model": "qwen3:8b",
                    "extracted_from_channel": "response",
                    "visible_response_empty": False,
                    "parse_success": True,
                    "schema_success": True,
                    "repair_used": False,
                    "latency_seconds": 1.0,
                    "failure_class": None,
                    "semantic_validation_result": "passed",
                },
                {
                    "ok": False,
                    "status": "llm_parse_failure",
                    "backend": "ollama",
                    "model": "qwen3:8b",
                    "extracted_from_channel": "thinking",
                    "visible_response_empty": True,
                    "parse_success": True,
                    "schema_success": False,
                    "repair_used": False,
                    "latency_seconds": 2.0,
                    "failure_class": "hidden_channel_only",
                },
                {
                    "ok": False,
                    "status": "llm_schema_failure",
                    "backend": "ollama",
                    "model": "qwen3:8b",
                    "extracted_from_channel": "response",
                    "visible_response_empty": False,
                    "parse_success": True,
                    "schema_success": False,
                    "repair_used": True,
                    "latency_seconds": 3.0,
                    "failure_class": "llm_schema_failure",
                },
                {
                    "ok": False,
                    "status": "unsupported_novelty_claim",
                    "backend": "ollama",
                    "model": "qwen3:8b",
                    "extracted_from_channel": "response",
                    "visible_response_empty": False,
                    "parse_success": True,
                    "schema_success": True,
                    "repair_used": False,
                    "latency_seconds": 1.5,
                    "failure_class": "unsupported_novelty_claim",
                    "semantic_validation_result": "failed",
                    "semantic_rejection_reason": {"code": "unsupported_novelty_claim"},
                },
                {
                    "ok": False,
                    "status": "llm_backend_failure",
                    "backend": "ollama",
                    "model": "qwen3:8b",
                    "extracted_from_channel": "response",
                    "visible_response_empty": True,
                    "parse_success": False,
                    "schema_success": False,
                    "repair_used": False,
                    "latency_seconds": None,
                    "failure_class": "llm_backend_failure",
                },
            ]
        )

        self.assertEqual(summary["total_trials"], 5)
        self.assertAlmostEqual(summary["success_rate"], 0.2)
        self.assertAlmostEqual(summary["parse_failure_rate"], 0.2)
        self.assertAlmostEqual(summary["schema_failure_rate"], 0.2)
        self.assertAlmostEqual(summary["semantic_failure_rate"], 0.2)
        self.assertAlmostEqual(summary["hidden_channel_incidence"], 0.2)
        self.assertAlmostEqual(summary["empty_visible_response_incidence"], 0.4)
        self.assertAlmostEqual(summary["repair_usage_rate"], 0.2)
        self.assertEqual(summary["median_latency_seconds"], 1.5)
        self.assertEqual(summary["p95_latency_seconds"], 3.0)
        self.assertEqual(summary["compatibility"]["interpretation"], "incompatible for control tasks")
        self.assertIn("backend_or_model_failure", summary["compatibility"]["issue_codes"])
        self.assertIn("semantic_rejection", summary["compatibility"]["issue_codes"])

    def test_summarize_smoke_test_trials_reports_usable_when_all_trials_are_clean(self) -> None:
        summary = research_loop.summarize_smoke_test_trials(
            [
                {
                    "ok": True,
                    "status": "llm_success",
                    "backend": "ollama",
                    "model": "qwen3:8b",
                    "extracted_from_channel": "response",
                    "visible_response_empty": False,
                    "parse_success": True,
                    "schema_success": True,
                    "repair_used": False,
                    "latency_seconds": 0.7,
                    "failure_class": None,
                },
                {
                    "ok": True,
                    "status": "llm_success",
                    "backend": "ollama",
                    "model": "qwen3:8b",
                    "extracted_from_channel": "response",
                    "visible_response_empty": False,
                    "parse_success": True,
                    "schema_success": True,
                    "repair_used": False,
                    "latency_seconds": 0.9,
                    "failure_class": None,
                },
            ],
            requested_backend="ollama",
            requested_model="qwen3:8b",
        )

        self.assertEqual(summary["total_trials"], 2)
        self.assertEqual(summary["success_rate"], 1.0)
        self.assertEqual(summary["parse_failure_rate"], 0.0)
        self.assertEqual(summary["schema_failure_rate"], 0.0)
        self.assertEqual(summary["hidden_channel_incidence"], 0.0)
        self.assertEqual(summary["empty_visible_response_incidence"], 0.0)
        self.assertEqual(summary["repair_usage_rate"], 0.0)
        self.assertEqual(summary["compatibility"]["issue_codes"], [])
        self.assertEqual(summary["compatibility"]["interpretation"], "usable")

    def test_summarize_smoke_test_trials_surfaces_unexpected_backend_model_pair(self) -> None:
        summary = research_loop.summarize_smoke_test_trials(
            [
                {
                    "ok": True,
                    "status": "llm_success",
                    "backend": "openai",
                    "model": "gpt-5.1-mini",
                    "extracted_from_channel": "response",
                    "visible_response_empty": False,
                    "parse_success": True,
                    "schema_success": True,
                    "repair_used": False,
                    "latency_seconds": 1.1,
                    "failure_class": None,
                }
            ],
            requested_backend="ollama",
            requested_model="qwen3:8b",
        )

        self.assertEqual(summary["unexpected_backend_model_pairs"], {"openai::gpt-5.1-mini": 1})
        self.assertIn("unexpected_backend_model_pair", summary["compatibility"]["issue_codes"])
        self.assertEqual(summary["compatibility"]["interpretation"], "unstable")

    def test_run_repeated_llm_smoke_test_writes_summary_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            summary_path = Path(tmpdir) / "smoke-summary.json"
            engine = StubProposalEngine(
                [],
                smoke_runs=[
                    {
                        "ok": True,
                        "status": "llm_success",
                        "backend": "ollama",
                        "model": "qwen3:8b",
                        "timestamp": "2026-03-22T10:00:00Z",
                        "extracted_from_channel": "response",
                        "visible_response_empty": False,
                        "parse_success": True,
                        "schema_success": True,
                        "repair_used": False,
                        "latency_seconds": 0.8,
                        "failure_class": None,
                    },
                    {
                        "ok": False,
                        "status": "duplicate_proposal",
                        "backend": "ollama",
                        "model": "qwen3:8b",
                        "timestamp": "2026-03-22T10:00:00Z",
                        "extracted_from_channel": "response",
                        "visible_response_empty": False,
                        "parse_success": True,
                        "schema_success": True,
                        "semantic_validation_result": "failed",
                        "semantic_rejection_reason": {"code": "duplicate_proposal"},
                        "repair_used": False,
                        "latency_seconds": 1.1,
                        "failure_class": "duplicate_proposal",
                    },
                    {
                        "ok": False,
                        "status": "llm_parse_failure",
                        "backend": "ollama",
                        "model": "qwen3:8b",
                        "timestamp": "2026-03-22T10:00:01Z",
                        "extracted_from_channel": "thinking",
                        "visible_response_empty": True,
                        "parse_success": True,
                        "schema_success": False,
                        "repair_used": False,
                        "latency_seconds": 1.4,
                        "failure_class": "hidden_channel_only",
                    },
                ],
            )

            report = research_loop.run_repeated_llm_smoke_test(
                proposal_engine=engine,
                available_models={},
                brief={"goal": "Improve composite_score"},
                current_best={"model_name": "ModelFamilyA"},
                memory_payload={"runs": []},
                knowledge_context="",
                failure_patterns={},
                archive_records=[],
                trials=3,
                summary_path=summary_path,
                model_hint="qwen3:8b",
            )

            persisted = json.loads(summary_path.read_text(encoding="utf-8"))

        self.assertEqual(report["requested_trials"], 3)
        self.assertEqual(len(report["trials"]), 3)
        self.assertEqual(report["summary"]["total_trials"], 3)
        self.assertEqual(report["summary"]["compatibility"]["interpretation"], "incompatible for control tasks")
        self.assertAlmostEqual(persisted["summary"]["hidden_channel_incidence"], 1 / 3)
        self.assertAlmostEqual(persisted["summary"]["semantic_failure_rate"], 1 / 3)

    def test_run_repeated_llm_smoke_test_reports_disabled_engine_as_incompatible(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            summary_path = Path(tmpdir) / "smoke-summary.json"

            report = research_loop.run_repeated_llm_smoke_test(
                proposal_engine=None,
                available_models={},
                brief={"goal": "Improve composite_score"},
                current_best={"model_name": "ModelFamilyA"},
                memory_payload={"runs": []},
                knowledge_context="",
                failure_patterns={},
                archive_records=[],
                trials=2,
                summary_path=summary_path,
                requested_backend="ollama",
                model_hint="qwen3:8b",
            )

            persisted = json.loads(summary_path.read_text(encoding="utf-8"))

        self.assertEqual(report["requested_trials"], 2)
        self.assertEqual(len(report["trials"]), 2)
        self.assertEqual(report["trials"][0]["status"], "aborted_due_to_preflight_failure")
        self.assertEqual(report["trials"][0]["failure_class"], "llm_backend_failure")
        self.assertFalse(report["trials"][0]["parse_success"])
        self.assertFalse(report["trials"][0]["schema_success"])
        self.assertEqual(report["summary"]["compatibility"]["interpretation"], "incompatible for control tasks")
        self.assertEqual(persisted["summary"]["status_counts"]["aborted_due_to_preflight_failure"], 2)

    def test_run_repeated_llm_smoke_test_persists_unicode_backend_errors(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            summary_path = Path(tmpdir) / "smoke-summary.json"
            unicode_error = "Unicode transport failure \u26a0 \u0627"
            engine = StubProposalEngine(
                [],
                smoke_runs=[
                    {
                        "ok": False,
                        "status": "llm_backend_failure",
                        "backend": "ollama",
                        "model": "qwen3:8b",
                        "timestamp": "2026-03-22T10:00:00Z",
                        "extracted_from_channel": "response",
                        "visible_response_empty": True,
                        "parse_success": False,
                        "schema_success": False,
                        "repair_used": False,
                        "latency_seconds": 1.1,
                        "failure_class": "llm_backend_failure",
                        "error": unicode_error,
                    }
                ],
            )

            report = research_loop.run_repeated_llm_smoke_test(
                proposal_engine=engine,
                available_models={},
                brief={"goal": "Improve composite_score"},
                current_best={"model_name": "ModelFamilyA"},
                memory_payload={"runs": []},
                knowledge_context="",
                failure_patterns={},
                archive_records=[],
                trials=1,
                summary_path=summary_path,
                requested_backend="ollama",
                model_hint="qwen3:8b",
            )

            persisted = json.loads(summary_path.read_text(encoding="utf-8"))

        self.assertEqual(report["trials"][0]["error"], unicode_error)
        self.assertEqual(persisted["trials"][0]["error"], unicode_error)
        self.assertEqual(persisted["summary"]["status_counts"]["llm_backend_failure"], 1)

    def test_run_manifest_payload_leaves_unknowns_explicit(self) -> None:
        payload = _build_run_manifest_payload(
            run_id="run-1",
            run_started_at="2026-03-22T10:00:00Z",
            run_finished_at="2026-03-22T10:05:00Z",
            config={"experiment": {"random_seed": 7}},
            brief={"acceptance_metric": "composite_score"},
            llm_config={"enabled": False, "backend_mode": "hybrid"},
            allow_deterministic_fallback=False,
            preflight_result=None,
            proposal_metadata=None,
            smoke_summary=None,
            proposal_status_counts={},
            semantic_rejection_counts={},
            llm_failure_counts={},
            archive_update_summary={},
            number_of_candidates_evaluated=0,
            holdout_touched_during_search=False,
            final_abort_reason=None,
            final_run_status="success",
            final_metrics=None,
            acceptance=None,
            validation_report=None,
        )

        self.assertIsNone(payload["backend"])
        self.assertIsNone(payload["model"])
        self.assertIsNone(payload["preflight_status"])
        self.assertIsNone(payload["smoke_test_status"])
        self.assertIsNone(payload["archive_update_summary"])
        self.assertEqual(payload["final_run_status"], "success")


if __name__ == "__main__":
    unittest.main()
