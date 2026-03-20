import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

import research_loop
from proposal_engine import ProposalEngine
from research_loop import _initialize_research_log, _initialize_results_csv, _select_scout_candidates


class StubProposalEngine:
    def __init__(self, proposals: list[dict]) -> None:
        self._proposals = proposals
        self.backend_name = "stub"
        self.last_interaction_summary = {
            "model": "qwen3:8b",
            "prompt_variant": "rich",
        }

    def is_available(self) -> bool:
        return True

    def generate_experiment_proposals(self, **_: object) -> list[dict]:
        return list(self._proposals)

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
        )

        self.assertTrue(candidates)
        self.assertEqual(metadata["proposal_mode"], "deterministic_fallback")

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

            with (
                patch.object(research_loop, "load_config", return_value=config),
                patch.object(research_loop, "get_outputs_dir", return_value=outputs_dir),
                patch.object(research_loop, "set_global_seed"),
                patch.object(research_loop, "_read_baseline_metrics", return_value=baseline_metrics),
                patch.object(research_loop, "load_human_research_brief", return_value={
                    "goal": "Improve composite_score",
                    "acceptance_metric": "composite_score",
                    "min_improvement_pct": 0.0,
                    "required_model_families": [],
                    "scout_candidates_per_cycle": 1,
                    "confirm_top_k": 1,
                }),
                patch.object(research_loop, "get_available_model_configs", return_value=available_models),
                patch.object(research_loop, "load_dataset", return_value="dataset"),
                patch.object(
                    research_loop,
                    "split_dataset",
                    return_value=("x_train", "x_val", "x_test", "y_train", "y_val", "y_test"),
                ),
                patch.object(research_loop.EngineeringValidator, "from_config", return_value=object()),
                patch.object(research_loop, "read_research_surface_state", return_value={"accepted_experiments": []}),
                patch.object(research_loop, "_build_proposal_engine", return_value=StubProposalEngine([
                    {
                        "model_name": "ModelFamilyA",
                        "display_name": "Model A",
                        "proposal_family": "llm-family-a",
                        "hypothesis": "Try a slightly deeper model.",
                        "params": {"depth": 3},
                    }
                ])),
                patch.object(research_loop, "_evaluate_stage_candidate", side_effect=[(object(), scout_result), (object(), confirm_result)]),
                patch.object(research_loop.research_lab, "confirm_experiments", return_value=[
                    {
                        "experiment_id": "confirm-001",
                        "stage": "confirm",
                        "model_name": "ModelFamilyA",
                        "display_name": "Model A",
                        "proposal_family": "llm-family-a",
                        "hypothesis": "Confirm the deeper model.",
                        "params": {"depth": 3},
                    }
                ]),
                patch.object(research_loop, "_evaluate_reference_if_possible", return_value=baseline_metrics),
                patch.object(research_loop, "save_pickle_artifact", side_effect=fake_save_pickle_artifact),
                patch.object(research_loop, "_sync_final_artifacts_from_source_of_truth", side_effect=fake_sync_final_artifacts_from_source_of_truth),
                patch.object(research_loop, "apply_keep_to_research_surface"),
                patch.object(research_loop, "validate_final_artifact_consistency", return_value={"consistent": True, "mismatches": []}),
                patch.object(research_loop, "log_status"),
            ):
                best_result = research_loop.run_engineering_research_loop(cycles_override=1, with_report=False)

            final_metrics = json.loads((outputs_dir / research_loop.FINAL_METRICS_FILENAME).read_text(encoding="utf-8"))
            self.assertEqual(best_result["composite_score"], 0.86)
            self.assertEqual(final_metrics["best_search_metrics"]["composite_score"], 0.86)

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


if __name__ == "__main__":
    unittest.main()
