import unittest

from llm_admission_policy import evaluate_llm_admission_policy, normalize_llm_admission_policy_config


class LLMAdmissionPolicyTests(unittest.TestCase):
    def _summary(self, **overrides: object) -> dict[str, object]:
        summary: dict[str, object] = {
            "total_trials": 10,
            "success_rate": 1.0,
            "backend_failure_count": 0,
            "backend_failure_rate": 0.0,
            "parse_failure_count": 0,
            "parse_failure_rate": 0.0,
            "schema_failure_count": 0,
            "schema_failure_rate": 0.0,
            "semantic_failure_count": 0,
            "semantic_failure_rate": 0.0,
            "hidden_channel_count": 0,
            "hidden_channel_incidence": 0.0,
            "empty_visible_count": 0,
            "empty_visible_response_incidence": 0.0,
            "repair_usage_rate": 0.0,
            "median_latency_seconds": 1.0,
            "p95_latency_seconds": 1.0,
            "status_counts": {"llm_success": 10},
            "backend_counts": {"ollama": 10},
            "model_counts": {"qwen3:8b": 10},
            "backend_model_counts": {"ollama::qwen3:8b": 10},
            "failure_class_counts": {},
            "unexpected_backend_model_pairs": {},
        }
        summary.update(overrides)
        return summary

    def test_clean_pass_is_usable(self) -> None:
        evaluation = evaluate_llm_admission_policy(self._summary())

        self.assertEqual(evaluation["verdict"], "usable")
        self.assertEqual(evaluation["interpretation"], "usable")
        self.assertTrue(evaluation["eligible_for_control_tasks"])
        self.assertEqual(evaluation["issue_codes"], [])
        self.assertTrue(all(check["status"] == "passed" for check in evaluation["checks"]))

    def test_caution_case_remains_eligible_but_flags_near_limit_metrics(self) -> None:
        evaluation = evaluate_llm_admission_policy(
            self._summary(
                success_rate=0.91,
                parse_failure_rate=0.09,
                p95_latency_seconds=4.5,
            )
        )

        self.assertEqual(evaluation["verdict"], "usable_with_caution")
        self.assertTrue(evaluation["eligible_for_control_tasks"])
        self.assertIn("success_rate_near_minimum", evaluation["issue_codes"])
        self.assertIn("parse_failure_rate_near_limit", evaluation["issue_codes"])
        self.assertIn("p95_latency_seconds_near_limit", evaluation["issue_codes"])

    def test_single_threshold_failure_is_unstable(self) -> None:
        evaluation = evaluate_llm_admission_policy(
            self._summary(
                parse_failure_rate=0.11,
                parse_failure_count=1,
            )
        )

        self.assertEqual(evaluation["verdict"], "unstable")
        self.assertFalse(evaluation["eligible_for_control_tasks"])
        self.assertIn("parse_failure_rate_exceeded", evaluation["issue_codes"])

    def test_hard_failure_is_blocked_for_control_tasks(self) -> None:
        evaluation = evaluate_llm_admission_policy(
            self._summary(
                success_rate=0.85,
                backend_failure_count=2,
                backend_failure_rate=0.2,
                status_counts={"llm_backend_failure": 2, "llm_success": 8},
            )
        )

        self.assertEqual(evaluation["verdict"], "blocked_for_control_tasks")
        self.assertFalse(evaluation["eligible_for_control_tasks"])
        self.assertIn("success_rate_below_minimum", evaluation["issue_codes"])

    def test_exact_threshold_values_pass_cleanly(self) -> None:
        evaluation = evaluate_llm_admission_policy(
            self._summary(
                success_rate=0.9,
                backend_failure_count=1,
                backend_failure_rate=0.1,
                parse_failure_rate=0.1,
                schema_failure_rate=0.1,
                semantic_failure_rate=0.1,
                hidden_channel_incidence=0.05,
                empty_visible_response_incidence=0.1,
                p95_latency_seconds=5.0,
                status_counts={"llm_backend_failure": 1, "llm_success": 9},
            )
        )

        self.assertEqual(evaluation["verdict"], "usable")
        self.assertEqual(evaluation["issue_codes"], [])
        self.assertTrue(all(check["status"] == "passed" for check in evaluation["checks"]))

    def test_custom_policy_thresholds_override_defaults(self) -> None:
        custom_policy = normalize_llm_admission_policy_config(
            {
                "thresholds": {
                    "min_success_rate": 0.95,
                }
            }
        )
        evaluation = evaluate_llm_admission_policy(
            self._summary(success_rate=0.94),
            admission_policy_config=custom_policy,
        )

        self.assertEqual(custom_policy["thresholds"]["min_success_rate"], 0.95)
        self.assertEqual(evaluation["verdict"], "blocked_for_control_tasks")
        self.assertIn("success_rate_below_minimum", evaluation["issue_codes"])


if __name__ == "__main__":
    unittest.main()
