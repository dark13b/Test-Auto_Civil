import unittest

from novelty_scorer import NoveltyScorer


class NoveltyScorerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.available_models = {
            "ModelFamilyA": {
                "search_space": {
                    "depth": {"type": "int", "low": 1, "high": 9},
                    "learning_rate": {"type": "float", "low": 0.01, "high": 0.31},
                    "booster": {"type": "categorical", "choices": ["gbdt", "dart"]},
                }
            },
            "ModelFamilyB": {
                "search_space": {
                    "alpha": {"type": "float", "low": 0.0, "high": 1.0},
                }
            },
        }
        self.scorer = NoveltyScorer(threshold=0.20)

    def test_exact_duplicate_is_rejected(self) -> None:
        history = [
            {
                "model_name": "ModelFamilyA",
                "params": {"depth": 4, "learning_rate": 0.11, "booster": "gbdt"},
            }
        ]

        result = self.scorer.score_proposal(
            proposal={"model_name": "ModelFamilyA", "params": {"depth": 4, "learning_rate": 0.11, "booster": "gbdt"}},
            history=history,
            available_models=self.available_models,
        )

        self.assertFalse(result["accepted"])
        self.assertAlmostEqual(result["novelty_score"], 0.0)
        self.assertEqual(result["closest_match"]["model_name"], "ModelFamilyA")

    def test_different_family_stays_novel(self) -> None:
        history = [
            {
                "model_name": "ModelFamilyA",
                "params": {"depth": 4, "learning_rate": 0.11, "booster": "gbdt"},
            }
        ]

        result = self.scorer.score_proposal(
            proposal={"model_name": "ModelFamilyB", "params": {"alpha": 0.8}},
            history=history,
            available_models=self.available_models,
        )

        self.assertTrue(result["accepted"])
        self.assertGreaterEqual(result["novelty_score"], 0.5)

    def test_nearby_numeric_config_has_lower_novelty_than_far_config(self) -> None:
        history = [
            {
                "model_name": "ModelFamilyA",
                "params": {"depth": 4, "learning_rate": 0.10, "booster": "gbdt"},
            }
        ]

        near_result = self.scorer.score_proposal(
            proposal={"model_name": "ModelFamilyA", "params": {"depth": 4, "learning_rate": 0.11, "booster": "gbdt"}},
            history=history,
            available_models=self.available_models,
        )
        far_result = self.scorer.score_proposal(
            proposal={"model_name": "ModelFamilyA", "params": {"depth": 8, "learning_rate": 0.28, "booster": "dart"}},
            history=history,
            available_models=self.available_models,
        )

        self.assertLess(near_result["novelty_score"], far_result["novelty_score"])


if __name__ == "__main__":
    unittest.main()
