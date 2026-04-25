import unittest

from failure_analyzer import FailureAnalyzer


class FailureAnalyzerTests(unittest.TestCase):
    def test_extract_failure_patterns_handles_partial_memory(self) -> None:
        analyzer = FailureAnalyzer(
            {
                "runs": [
                    {
                        "run_id": "run-1",
                        "trials": [
                            {
                                "model_name": "ModelFamilyA",
                                "proposal_family": "family-a",
                                "selection_status": "rejected_validation_fail",
                                "validation_verdict": "FAIL",
                                "params": {"depth": 8},
                                "hard_fail_reasons": ["high water ratio"],
                                "composite_score": 0.5,
                            },
                            {
                                "model_name": "ModelFamilyA",
                                "proposal_family": "family-a",
                                "selection_status": "scout_no_improvement",
                                "validation_verdict": "WARN",
                                "params": {"depth": 7},
                                "engineering_caution_reasons": ["durability caution"],
                                "composite_score": 0.6,
                            },
                            {
                                "model_name": "ModelFamilyB",
                                "proposal_family": "family-b",
                                "selection_status": "kept",
                                "validation_verdict": "PASS",
                                "params": {"alpha": 0.3},
                                "composite_score": 0.9,
                            },
                        ],
                    }
                ]
            }
        )

        patterns = analyzer.extract_failure_patterns()

        self.assertIn("families_with_high_fail_rate", patterns)
        self.assertIn("parameter_ranges_that_always_fail", patterns)
        self.assertIn("validator_failure_reasons", patterns)
        self.assertIn("low_score_hyperparameter_profiles", patterns)
        self.assertEqual(patterns["families_with_high_fail_rate"][0]["family"], "family-a")
        self.assertEqual(patterns["validator_failure_reasons"][0]["reason"], "high water ratio")


if __name__ == "__main__":
    unittest.main()
