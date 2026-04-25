import json
import tempfile
import unittest
from pathlib import Path

from proposal_engine import (
    ProposalBackendFailure,
    ProposalEngine,
    ProposalExtractionError,
    ProposalParseFailure,
    ProposalSemanticFailure,
    ProposalSchemaFailure,
)


class RecordingBackend:
    def __init__(self, responses: list[dict], *, available: bool = True, backend_name: str = "ollama") -> None:
        self._responses = list(responses)
        self._available = available
        self.backend_name = backend_name
        self.prompts: list[str] = []
        self.models: list[str | None] = []

    def is_available(self) -> bool:
        return self._available

    def generate_text(self, prompt: str, **kwargs: object) -> dict:
        self.prompts.append(prompt)
        self.models.append(kwargs.get("model"))
        if not self._responses:
            raise AssertionError("No stub response remaining for generate_text()")
        payload = dict(self._responses.pop(0))
        payload.setdefault("backend", self.backend_name)
        payload.setdefault("model", kwargs.get("model") or "stub-model")
        payload.setdefault("text", "")
        return payload


class ProposalEngineTests(unittest.TestCase):
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
        self.research_brief = {
            "goal": "Improve composite_score",
            "acceptance_metric": "composite_score",
            "min_improvement_pct": 0.5,
        }
        self.current_best = {"model_name": "ModelFamilyA", "composite_score": 0.90}
        self.experiment_memory = {"runs": [], "accepted_experiments": []}
        self.search_progress = {"phase": "scout"}

    def _make_engine(
        self,
        backend: RecordingBackend,
        *,
        llm_config: dict | None = None,
        log_path: Path | None = None,
    ) -> ProposalEngine:
        return ProposalEngine(
            backend=backend,
            interaction_log_path=log_path,
            log_interactions=log_path is not None,
            llm_config=llm_config or {
                "compact_prompt_models": ["qwen3:4b"],
                "enable_regeneration_on_reject": True,
                "max_regeneration_attempts": 1,
                "duplicate_similarity_thresholds": {
                    "numeric_tolerance": 0.05,
                    "float_round_digits": 4,
                },
                "temporarily_block_saturated_families": True,
                "diversity": {"max_family_share": 0.35},
            },
        )

    @staticmethod
    def _read_jsonl(path: Path) -> list[dict]:
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    @staticmethod
    def _build_trial_history(count: int) -> list[dict]:
        history: list[dict] = []
        for index in range(1, count + 1):
            history.append(
                {
                    "trial_number": index,
                    "experiment_id": f"trial-{index}",
                    "model_name": "ModelFamilyA" if index % 2 else "ModelFamilyB",
                    "proposal_family": "family-a" if index % 2 else "family-b",
                    "hyperparameters": json.dumps(
                        {
                            "depth": min(index, 8),
                            "learning_rate": round(0.02 * index, 4),
                        }
                    ),
                    "composite_score": 0.80 + (index / 100.0),
                    "validation_verdict": "WARN",
                }
            )
        return history

    def test_prompt_builder_uses_schema_only_example_and_no_anchored_real_model_example(self) -> None:
        backend = RecordingBackend(
            [
                {
                    "response_text": '{"model_name":"ModelFamilyA","params":{"depth":3,"learning_rate":0.1},"proposal_family":"family-a","hypothesis":"prompt probe","expected_delta":0.15}',
                }
            ]
        )
        engine = self._make_engine(backend)

        engine.generate_experiment_proposals(
            available_models=self.available_models,
            research_brief=self.research_brief,
            current_best=self.current_best,
            experiment_memory=self.experiment_memory,
            diversity_state={},
            proposal_count=1,
            trial_history=self._build_trial_history(2),
            search_progress=self.search_progress,
            model_hint="qwen3:4b",
        )

        prompt = backend.prompts[0]
        self.assertIn('{"model_name":"MODEL","params":{}}', prompt)
        self.assertNotIn('{"model_name":"XGBRegressor"', prompt)
        self.assertNotIn('{"model_name":"LGBMRegressor"', prompt)

    def test_qwen4b_uses_compact_prompt_mode_and_only_last_five_trials(self) -> None:
        backend = RecordingBackend(
            [
                {
                    "response_text": '{"model_name":"ModelFamilyA","params":{"depth":3,"learning_rate":0.1},"proposal_family":"family-a","hypothesis":"compact probe","expected_delta":0.12}',
                }
            ]
        )
        engine = self._make_engine(backend)

        engine.generate_experiment_proposals(
            available_models=self.available_models,
            research_brief=self.research_brief,
            current_best=self.current_best,
            experiment_memory=self.experiment_memory,
            diversity_state={},
            proposal_count=1,
            trial_history=self._build_trial_history(7),
            search_progress=self.search_progress,
            model_hint="qwen3:4b",
        )

        prompt = backend.prompts[0]
        self.assertEqual(engine.last_interaction_summary["prompt_variant"], "compact")
        self.assertIn("trial-7", prompt)
        self.assertIn("trial-3", prompt)
        self.assertNotIn("trial-2", prompt)
        self.assertNotIn("trial-1", prompt)
        self.assertNotIn("Think step by step internally", prompt)

    def test_qwen8b_uses_rich_prompt_mode(self) -> None:
        backend = RecordingBackend(
            [
                {
                    "response_text": '{"model_name":"ModelFamilyA","params":{"depth":3,"learning_rate":0.1},"proposal_family":"family-a","hypothesis":"rich probe","expected_delta":0.11}',
                }
            ]
        )
        engine = self._make_engine(backend)

        engine.generate_experiment_proposals(
            available_models=self.available_models,
            research_brief=self.research_brief,
            current_best=self.current_best,
            experiment_memory=self.experiment_memory,
            diversity_state={},
            proposal_count=1,
            trial_history=self._build_trial_history(7),
            search_progress=self.search_progress,
            model_hint="qwen3:8b",
        )

        prompt = backend.prompts[0]
        self.assertEqual(engine.last_interaction_summary["prompt_variant"], "rich")
        self.assertIn("trial-1", prompt)
        self.assertIn("Family state", prompt)

    def test_default_local_qwen8b_uses_compact_prompt_when_no_model_hint_is_passed(self) -> None:
        backend = RecordingBackend(
            [
                {
                    "response_text": '{"model_name":"ModelFamilyA","params":{"depth":3,"learning_rate":0.1},"proposal_family":"family-a","hypothesis":"default model probe","expected_delta":0.18}',
                }
            ]
        )
        engine = self._make_engine(
            backend,
            llm_config={
                "default_local_proposal_model": "qwen3:8b",
                "compact_prompt_models": ["qwen3:8b"],
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

        engine.generate_experiment_proposals(
            available_models=self.available_models,
            research_brief=self.research_brief,
            current_best=self.current_best,
            experiment_memory=self.experiment_memory,
            diversity_state={},
            proposal_count=1,
            trial_history=self._build_trial_history(7),
            search_progress=self.search_progress,
            model_hint=None,
        )

        prompt = backend.prompts[0]
        self.assertEqual(engine.last_interaction_summary["prompt_variant"], "compact")
        self.assertEqual(backend.models[0], "qwen3:8b")
        self.assertIn("trial-7", prompt)
        self.assertNotIn("trial-1", prompt)

    def test_response_extraction_prefers_response_text_when_valid(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "llm_interactions.jsonl"
            backend = RecordingBackend(
                [
                    {
                        "response_text": '{"model_name":"ModelFamilyA","params":{"depth":3,"learning_rate":0.1},"proposal_family":"family-a","hypothesis":"try depth","expected_delta":0.20}',
                        "thinking_text": '{"model_name":"ModelFamilyB","params":{"alpha":0.2},"proposal_family":"family-b","hypothesis":"ignored","expected_delta":0.10}',
                    }
                ]
            )
            engine = self._make_engine(backend, log_path=log_path)

            proposals = engine.generate_experiment_proposals(
                available_models=self.available_models,
                research_brief=self.research_brief,
                current_best=self.current_best,
                experiment_memory=self.experiment_memory,
                diversity_state={},
                proposal_count=1,
                model_hint="qwen3:8b",
            )
            record = self._read_jsonl(log_path)[0]

        self.assertEqual(len(proposals), 1)
        self.assertEqual(proposals[0]["model_name"], "ModelFamilyA")
        self.assertEqual(proposals[0]["expected_delta"], 0.20)

    def test_prompt_includes_failure_patterns_knowledge_and_expected_delta_instruction(self) -> None:
        backend = RecordingBackend(
            [
                {
                    "response_text": '{"model_name":"ModelFamilyA","params":{"depth":3,"learning_rate":0.1},"proposal_family":"family-a","hypothesis":"guided probe","expected_delta":0.14}',
                }
            ]
        )
        engine = self._make_engine(backend)

        proposals = engine.generate_experiment_proposals(
            available_models=self.available_models,
            research_brief=self.research_brief,
            current_best=self.current_best,
            experiment_memory=self.experiment_memory,
            diversity_state={},
            proposal_count=1,
            trial_history=self._build_trial_history(3),
            search_progress=self.search_progress,
            failure_patterns={"validator_failure_reasons": [{"reason": "high water ratio", "count": 2}]},
            knowledge_context="Empirical Best Ranges\n- Keep depth moderate [source: repo]",
            underexplored_families=["ModelFamilyB"],
            model_hint="qwen3:8b",
        )

        prompt = backend.prompts[0]
        self.assertIn("STEP 1: diagnose", prompt)
        self.assertIn("Failure patterns", prompt)
        self.assertIn("Knowledge context", prompt)
        self.assertIn("expected_delta", prompt)
        self.assertEqual(proposals[0]["expected_delta"], 0.14)

    def test_response_extraction_falls_back_to_thinking_text_when_visible_response_is_empty(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "llm_interactions.jsonl"
            backend = RecordingBackend(
                [
                    {
                        "model": "qwen3:4b",
                        "response_text": "",
                        "thinking_text": '{"model_name":"ModelFamilyA","params":{"depth":4,"learning_rate":0.12},"proposal_family":"family-a","hypothesis":"use thinking"}',
                    }
                ]
            )
            engine = self._make_engine(backend, log_path=log_path)

            proposals = engine.generate_experiment_proposals(
                available_models=self.available_models,
                research_brief=self.research_brief,
                current_best=self.current_best,
                experiment_memory=self.experiment_memory,
                diversity_state={},
                proposal_count=1,
                model_hint="qwen3:4b",
            )
            record = self._read_jsonl(log_path)[0]

        self.assertEqual(len(proposals), 1)
        self.assertEqual(proposals[0]["model_name"], "ModelFamilyA")
        self.assertEqual(record["extracted_from_channel"], "thinking")

    def test_json_repair_success_marks_repair_used(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "llm_interactions.jsonl"
            backend = RecordingBackend(
                [
                    {
                        "response_text": '{"model_name":"ModelFamilyA","params":{"depth":5,"learning_rate":0.14},"proposal_family":"family-a","hypothesis":"repair me"',
                    }
                ]
            )
            engine = self._make_engine(backend, log_path=log_path)

            proposals = engine.generate_experiment_proposals(
                available_models=self.available_models,
                research_brief=self.research_brief,
                current_best=self.current_best,
                experiment_memory=self.experiment_memory,
                diversity_state={},
                proposal_count=1,
                model_hint="qwen3:8b",
            )
            record = self._read_jsonl(log_path)[0]

        self.assertEqual(len(proposals), 1)
        self.assertTrue(record["repair_used"])
        self.assertEqual(record["extracted_from_channel"], "repaired_json")

    def test_duplicate_rejection_logs_reason_and_triggers_single_regeneration(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "llm_interactions.jsonl"
            backend = RecordingBackend(
                [
                    {
                        "response_text": '{"model_name":"ModelFamilyA","params":{"depth":3,"learning_rate":0.1},"proposal_family":"family-a","hypothesis":"duplicate"}',
                    },
                    {
                        "response_text": '{"model_name":"ModelFamilyB","params":{"alpha":0.4},"proposal_family":"family-b","hypothesis":"novel"}',
                    },
                ]
            )
            engine = self._make_engine(backend, log_path=log_path)

            proposals = engine.generate_experiment_proposals(
                available_models=self.available_models,
                research_brief=self.research_brief,
                current_best=self.current_best,
                experiment_memory={
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
                                    "signature": [
                                        "ModelFamilyA",
                                        '{"depth":3,"learning_rate":0.1}',
                                    ],
                                }
                            ],
                        }
                    ],
                    "accepted_experiments": [],
                },
                diversity_state={},
                proposal_count=1,
                model_hint="qwen3:8b",
            )
            records = self._read_jsonl(log_path)

        self.assertEqual(len(proposals), 1)
        self.assertEqual(proposals[0]["model_name"], "ModelFamilyB")
        self.assertEqual(len(records), 2)
        self.assertTrue(records[0]["duplicate_rejected"])
        self.assertEqual(records[0]["rejection_reason"]["code"], "exact_duplicate")
        self.assertTrue(records[0]["regeneration_attempted"])

    def test_multi_proposal_request_degrades_to_single_proposal_mode_when_array_response_is_unusable(self) -> None:
        backend = RecordingBackend(
            [
                {
                    "response_text": "",
                    "thinking_text": "",
                },
                {
                    "response_text": '{"model_name":"ModelFamilyB","params":{"alpha":0.4},"proposal_family":"family-b","hypothesis":"single fallback"}',
                },
            ]
        )
        engine = self._make_engine(
            backend,
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

        proposals = engine.generate_experiment_proposals(
            available_models=self.available_models,
            research_brief=self.research_brief,
            current_best=self.current_best,
            experiment_memory=self.experiment_memory,
            diversity_state={},
            proposal_count=4,
            model_hint="qwen3:4b",
        )

        self.assertEqual(len(proposals), 1)
        self.assertEqual(proposals[0]["model_name"], "ModelFamilyB")
        self.assertEqual(len(backend.prompts), 2)
        self.assertIn("Return exactly one JSON object.", backend.prompts[1])

    def test_ambiguous_json_is_not_silently_accepted(self) -> None:
        backend = RecordingBackend(
            [
                {
                    "response_text": '{"model_name":"ModelFamilyA","params":{"depth":3}} {"model_name":"ModelFamilyB","params":{"alpha":0.2}}',
                }
            ]
        )
        engine = self._make_engine(
            backend,
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

        with self.assertRaises(ProposalExtractionError):
            engine.generate_experiment_proposals(
                available_models=self.available_models,
                research_brief=self.research_brief,
                current_best=self.current_best,
                experiment_memory=self.experiment_memory,
                diversity_state={},
                proposal_count=1,
                model_hint="qwen3:8b",
            )

    def test_generate_research_proposals_accepts_valid_strict_hypothesis_json(self) -> None:
        backend = RecordingBackend(
            [
                {
                    "response_text": json.dumps(
                        {
                            "hypothesis": "Reducing LightGBM num_leaves should reduce variance on this dataset.",
                            "rationale": "Recent boosted-tree runs plateaued while showing instability in high-strength mixes.",
                            "change_type": "hyperparameter",
                            "target_component": "ModelFamilyA",
                            "proposed_change": "Set depth=3 and learning_rate=0.1 while keeping the tree family fixed.",
                            "expected_direction": "improve",
                            "expected_metric_effect": {
                                "metric": "rmse",
                                "direction": "down",
                                "magnitude_estimate": "0.05-0.12 MPa",
                            },
                            "confidence": 0.64,
                            "novelty_claim": "Recent accepted trials used different depth values, so this is a new setting.",
                            "risk_notes": "May underfit low-strength regime if regularization is too strong.",
                            "candidate_config": {
                                "model_name": "ModelFamilyA",
                                "params": {"depth": 3, "learning_rate": 0.1},
                            },
                        }
                    )
                }
            ]
        )
        engine = self._make_engine(backend)

        result = engine.generate_research_proposals(
            available_models=self.available_models,
            research_brief=self.research_brief,
            current_best=self.current_best,
            experiment_memory=self.experiment_memory,
            diversity_state={},
            proposal_count=1,
            model_hint="qwen3:8b",
        )

        self.assertEqual(result["status"], "llm_success")
        self.assertEqual(len(result["proposals"]), 1)
        proposal = result["proposals"][0]
        self.assertEqual(proposal["proposal_source"], "llm")
        self.assertEqual(proposal["research_proposal"]["expected_metric_effect"]["metric"], "rmse")
        self.assertEqual(proposal["params"]["depth"], 3)

    def test_generate_research_proposals_rejects_empty_response(self) -> None:
        backend = RecordingBackend([{"response_text": "", "thinking_text": ""}])
        engine = self._make_engine(backend)

        with self.assertRaises(ProposalParseFailure):
            engine.generate_research_proposals(
                available_models=self.available_models,
                research_brief=self.research_brief,
                current_best=self.current_best,
                experiment_memory=self.experiment_memory,
                diversity_state={},
                proposal_count=1,
                model_hint="qwen3:8b",
            )

    def test_generate_research_proposals_rejects_schema_mismatch(self) -> None:
        backend = RecordingBackend(
            [
                {
                    "response_text": json.dumps(
                        {
                            "hypothesis": "",
                            "rationale": "blank hypothesis should be rejected",
                            "change_type": "hyperparameter",
                            "target_component": "ModelFamilyA",
                            "proposed_change": "depth=3",
                            "expected_direction": "improve",
                            "expected_metric_effect": {
                                "metric": "rmse",
                                "direction": "down",
                                "magnitude_estimate": "small",
                            },
                            "confidence": 1.2,
                            "novelty_claim": "none",
                            "risk_notes": "none",
                            "candidate_config": {
                                "model_name": "ModelFamilyA",
                                "params": {"depth": 3, "learning_rate": 0.1},
                            },
                        }
                    )
                }
            ]
        )
        engine = self._make_engine(backend)

        with self.assertRaises(ProposalSchemaFailure):
            engine.generate_research_proposals(
                available_models=self.available_models,
                research_brief=self.research_brief,
                current_best=self.current_best,
                experiment_memory=self.experiment_memory,
                diversity_state={},
                proposal_count=1,
                model_hint="qwen3:8b",
            )

    def test_run_proposal_smoke_test_reports_backend_failure(self) -> None:
        class FailingBackend(RecordingBackend):
            def generate_text(self, prompt: str, **kwargs: object) -> dict:
                raise ProposalBackendFailure("backend down", failure_kind="llm_backend_failure")

        backend = FailingBackend([])
        engine = self._make_engine(backend)

        smoke_result = engine.run_proposal_smoke_test(
            available_models=self.available_models,
            research_brief=self.research_brief,
            current_best=self.current_best,
            experiment_memory=self.experiment_memory,
            diversity_state={},
            model_hint="qwen3:8b",
        )

        self.assertFalse(smoke_result["ok"])
        self.assertEqual(smoke_result["status"], "llm_backend_failure")

    def test_run_proposal_smoke_test_returns_reliability_metadata_on_success(self) -> None:
        backend = RecordingBackend(
            [
                {
                    "response_text": json.dumps(
                        {
                            "hypothesis": "Reducing depth should improve generalization.",
                            "rationale": "Recent runs show mild overfitting.",
                            "change_type": "hyperparameter",
                            "target_component": "ModelFamilyA",
                            "proposed_change": "depth=3",
                            "expected_direction": "improve",
                            "expected_metric_effect": {
                                "metric": "rmse",
                                "direction": "down",
                                "magnitude_estimate": "0.05-0.10 MPa",
                            },
                            "confidence": 0.61,
                            "novelty_claim": "Depth 3 has not been tried in the latest runs.",
                            "risk_notes": "Could underfit if the current best is already shallow.",
                            "candidate_config": {
                                "model_name": "ModelFamilyA",
                                "params": {"depth": 3, "learning_rate": 0.1},
                            },
                        }
                    )
                }
            ]
        )
        engine = self._make_engine(backend)

        smoke_result = engine.run_proposal_smoke_test(
            available_models=self.available_models,
            research_brief=self.research_brief,
            current_best=self.current_best,
            experiment_memory=self.experiment_memory,
            diversity_state={},
            model_hint="qwen3:8b",
        )

        self.assertTrue(smoke_result["ok"])
        self.assertEqual(smoke_result["status"], "llm_success")
        self.assertEqual(smoke_result["extracted_from_channel"], "response")
        self.assertFalse(smoke_result["visible_response_empty"])
        self.assertTrue(smoke_result["parse_success"])
        self.assertTrue(smoke_result["schema_success"])
        self.assertFalse(smoke_result["repair_used"])
        self.assertIsNone(smoke_result["failure_class"])
        self.assertIsInstance(smoke_result["timestamp"], str)
        self.assertGreaterEqual(smoke_result["latency_seconds"], 0.0)

    def test_run_proposal_smoke_test_surfaces_semantic_rejection_metadata(self) -> None:
        backend = RecordingBackend(
            [
                {
                    "response_text": json.dumps(
                        {
                            "hypothesis": "Try to improve performance somehow.",
                            "rationale": "The model should do better.",
                            "change_type": "hyperparameter",
                            "target_component": "ModelFamilyA",
                            "proposed_change": "Use a better configuration.",
                            "expected_direction": "improve",
                            "expected_metric_effect": {
                                "metric": "rmse",
                                "direction": "down",
                                "magnitude_estimate": "0.05-0.10 MPa",
                            },
                            "confidence": 0.61,
                            "novelty_claim": "This is new.",
                            "risk_notes": "This might be risky.",
                            "candidate_config": {
                                "model_name": "ModelFamilyA",
                                "params": {},
                            },
                        }
                    )
                }
            ]
        )
        engine = self._make_engine(backend)

        smoke_result = engine.run_proposal_smoke_test(
            available_models=self.available_models,
            research_brief=self.research_brief,
            current_best=self.current_best,
            experiment_memory=self.experiment_memory,
            diversity_state={},
            model_hint="qwen3:8b",
        )

        self.assertFalse(smoke_result["ok"])
        self.assertEqual(smoke_result["status"], "semantically_invalid_candidate_config")
        self.assertFalse(smoke_result["semantic_success"])
        self.assertEqual(smoke_result["semantic_validation_result"], "failed")
        self.assertEqual(smoke_result["failure_class"], "semantically_invalid_candidate_config")

    def test_generate_research_proposals_raises_semantic_failure_for_meaningless_candidate(self) -> None:
        backend = RecordingBackend(
            [
                {
                    "response_text": json.dumps(
                        {
                            "hypothesis": "Try to improve performance somehow.",
                            "rationale": "The model should do better.",
                            "change_type": "hyperparameter",
                            "target_component": "ModelFamilyA",
                            "proposed_change": "Use a better configuration.",
                            "expected_direction": "improve",
                            "expected_metric_effect": {
                                "metric": "rmse",
                                "direction": "down",
                                "magnitude_estimate": "0.05-0.10 MPa",
                            },
                            "confidence": 0.61,
                            "novelty_claim": "This is new.",
                            "risk_notes": "This might be risky.",
                            "candidate_config": {
                                "model_name": "ModelFamilyA",
                                "params": {},
                            },
                        }
                    )
                }
            ]
        )
        engine = self._make_engine(
            backend,
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

        with self.assertRaises(ProposalSemanticFailure) as ctx:
            engine.generate_research_proposals(
                available_models=self.available_models,
                research_brief=self.research_brief,
                current_best=self.current_best,
                experiment_memory=self.experiment_memory,
                diversity_state={},
                proposal_count=1,
                model_hint="qwen3:8b",
            )

        self.assertEqual(ctx.exception.failure_kind, "semantically_invalid_candidate_config")

    def test_run_proposal_smoke_test_surfaces_hidden_channel_failures(self) -> None:
        backend = RecordingBackend(
            [
                {
                    "response_text": "",
                    "thinking_text": json.dumps(
                        {
                            "hypothesis": "Reducing depth should improve generalization.",
                            "rationale": "Recent runs show mild overfitting.",
                            "change_type": "hyperparameter",
                            "target_component": "ModelFamilyA",
                            "proposed_change": "depth=3",
                            "expected_direction": "improve",
                            "expected_metric_effect": {
                                "metric": "rmse",
                                "direction": "down",
                                "magnitude_estimate": "0.05-0.10 MPa",
                            },
                            "confidence": 0.61,
                            "novelty_claim": "Depth 3 has not been tried in the latest runs.",
                            "risk_notes": "Could underfit if the current best is already shallow.",
                            "candidate_config": {
                                "model_name": "ModelFamilyA",
                                "params": {"depth": 3, "learning_rate": 0.1},
                            },
                        }
                    ),
                }
            ]
        )
        engine = self._make_engine(backend)

        smoke_result = engine.run_proposal_smoke_test(
            available_models=self.available_models,
            research_brief=self.research_brief,
            current_best=self.current_best,
            experiment_memory=self.experiment_memory,
            diversity_state={},
            model_hint="qwen3:8b",
        )

        self.assertFalse(smoke_result["ok"])
        self.assertEqual(smoke_result["status"], "llm_parse_failure")
        self.assertEqual(smoke_result["extracted_from_channel"], "thinking")
        self.assertTrue(smoke_result["visible_response_empty"])
        self.assertTrue(smoke_result["parse_success"])
        self.assertFalse(smoke_result["schema_success"])
        self.assertEqual(smoke_result["failure_class"], "hidden_channel_only")


if __name__ == "__main__":
    unittest.main()
