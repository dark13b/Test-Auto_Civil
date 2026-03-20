import unittest

from research_lab import confirm_experiments, scout_experiments


class ResearchLabTests(unittest.TestCase):
    def test_log_scaled_exploit_variants_use_multiplicative_perturbation(self) -> None:
        brief = {"required_model_families": ["Ridge"]}
        lab_state = {"accepted_experiments": [], "recent_kept_families": []}
        available_models = {
            "Ridge": {
                "display_name": "Ridge",
                "search_space": {
                    "alpha": {"type": "float", "low": 0.01, "high": 10.0, "log": True},
                    "fit_intercept": {"type": "categorical", "choices": [True, False]},
                },
            }
        }

        experiments = scout_experiments(
            brief=brief,
            lab_state=lab_state,
            available_models=available_models,
            experiment_memory={"runs": []},
            current_best={
                "model_name": "Ridge",
                "hyperparameters": {
                    "alpha": 0.10,
                    "fit_intercept": True,
                },
            },
            scout_limit=10,
            exploit_delta_ratio=0.20,
        )

        exploit_alphas = sorted(
            item["params"]["alpha"]
            for item in experiments
            if item["proposal_family"] == "Ridge-exploit" and "alpha" in item["params"]
        )

        self.assertIn(0.08, exploit_alphas)
        self.assertIn(0.12, exploit_alphas)
        self.assertNotIn(0.07, exploit_alphas)
        self.assertNotIn(0.13, exploit_alphas)

    def test_scout_experiments_generate_multiple_exploit_variants_around_current_best(self) -> None:
        brief = {"required_model_families": ["LGBMRegressor"]}
        lab_state = {"accepted_experiments": [], "recent_kept_families": []}
        available_models = {
            "LGBMRegressor": {
                "display_name": "LightGBM",
                "search_space": {
                    "n_estimators": {"type": "int", "low": 100, "high": 500, "step": 100},
                    "learning_rate": {"type": "float", "low": 0.01, "high": 0.2},
                    "num_leaves": {"type": "int", "low": 15, "high": 63, "step": 8},
                    "subsample": {"type": "categorical", "choices": [0.7, 0.8, 0.9, 1.0]},
                },
            }
        }
        experiments = scout_experiments(
            brief=brief,
            lab_state=lab_state,
            available_models=available_models,
            experiment_memory={"runs": []},
            current_best={
                "model_name": "LGBMRegressor",
                "hyperparameters": {
                    "n_estimators": 300,
                    "learning_rate": 0.05,
                    "num_leaves": 31,
                    "subsample": 0.8,
                },
            },
            scout_limit=20,
        )

        exploit_experiments = [item for item in experiments if item["proposal_family"] == "LGBMRegressor-exploit"]
        exploit_signatures = {tuple(sorted(item["params"].items())) for item in exploit_experiments}

        self.assertGreaterEqual(len(exploit_experiments), 3)
        self.assertEqual(len(exploit_signatures), len(exploit_experiments))

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
