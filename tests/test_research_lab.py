import unittest

from research_lab import confirm_experiments, scout_experiments


class ResearchLabTests(unittest.TestCase):
    def test_scout_experiments_respect_required_model_families(self) -> None:
        brief = {"required_model_families": ["LGBMRegressor"]}
        lab_state = {"accepted_experiments": []}
        available_models = {
            "LGBMRegressor": {
                "display_name": "LightGBM",
                "search_space": {
                    "n_estimators": {"type": "int", "low": 100, "high": 500, "step": 100},
                    "learning_rate": {"type": "float", "low": 0.01, "high": 0.2},
                },
            },
            "XGBRegressor": {
                "display_name": "XGBoost",
                "search_space": {
                    "n_estimators": {"type": "int", "low": 100, "high": 500, "step": 100},
                    "learning_rate": {"type": "float", "low": 0.01, "high": 0.2},
                },
            },
        }
        experiments = scout_experiments(
            brief=brief,
            lab_state=lab_state,
            available_models=available_models,
            experiment_memory={"runs": []},
            current_best={"model_name": "RandomForestRegressor", "hyperparameters": {}},
            scout_limit=6,
        )

        self.assertTrue(experiments)
        self.assertEqual({item["model_name"] for item in experiments}, {"LGBMRegressor"})

    def test_confirm_experiments_only_promote_positive_scouts(self) -> None:
        scout_results = [
            {
                "experiment_id": "scout-001",
                "proposal_family": "boosting-balanced",
                "model_name": "LGBMRegressor",
                "params": {"n_estimators": 300},
                "scout_improvement_pct": 2.0,
            },
            {
                "experiment_id": "scout-002",
                "proposal_family": "tree-diversify",
                "model_name": "RandomForestRegressor",
                "params": {"n_estimators": 300},
                "scout_improvement_pct": -0.5,
            },
        ]

        confirms = confirm_experiments(
            scout_results=scout_results,
            current_best={"model_name": "RandomForestRegressor", "hyperparameters": {}},
            confirm_top_k=2,
        )

        self.assertEqual(len(confirms), 1)
        self.assertEqual(confirms[0]["stage"], "confirm")
        self.assertEqual(confirms[0]["experiment_id"], "confirm-scout-001")


if __name__ == "__main__":
    unittest.main()
