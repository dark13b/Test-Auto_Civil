import unittest

from mix_design.constraints import (
    apply_target_strength_bounds,
    evaluate_constraints,
    load_design_constraints,
)


def make_config() -> dict:
    return {
        "engineering": {
            "design_tool": {
                "target_tolerance_mpa": 2.0,
                "constraints": {
                    "cement": {"min": 100.0, "max": 450.0},
                    "slag": {"min": 0.0, "max": 220.0},
                    "fly_ash": {"min": 0.0, "max": 180.0},
                    "water": {"min": 130.0, "max": 210.0},
                    "superplasticizer": {"min": 0.0, "max": 16.0},
                    "coarse_aggregate": {"min": 900.0, "max": 1100.0},
                    "fine_aggregate": {"min": 650.0, "max": 850.0},
                    "age": {"fixed": 28.0},
                    "water_cement_ratio": {"max": 0.70},
                    "fly_ash_replacement_ratio": {"max": 0.45},
                    "slag_replacement_ratio": {"max": 0.70},
                },
            }
        }
    }


class DesignConstraintTests(unittest.TestCase):
    def test_load_design_constraints_maps_config_to_explicit_contract(self) -> None:
        constraints = load_design_constraints(make_config())

        self.assertEqual(constraints.cement.min, 100.0)
        self.assertEqual(constraints.cement.max, 450.0)
        self.assertEqual(constraints.age.fixed, 28.0)
        self.assertEqual(constraints.water_cement_ratio.max, 0.70)
        self.assertEqual(constraints.tolerance_mpa, 2.0)

    def test_apply_target_strength_bounds_makes_regime_rules_visible(self) -> None:
        constraints = load_design_constraints(make_config())

        adjusted = apply_target_strength_bounds(constraints, target_strength_mpa=25.0)

        self.assertEqual(adjusted.cement.min, 100.0)
        self.assertEqual(adjusted.cement.max, 260.0)
        self.assertEqual(adjusted.water.min, 150.0)
        self.assertEqual(adjusted.water.max, 210.0)
        self.assertEqual(adjusted.water_cement_ratio.max, 1.45)

    def test_evaluate_constraints_returns_visible_check_records(self) -> None:
        constraints = apply_target_strength_bounds(load_design_constraints(make_config()), 25.0)

        evaluation = evaluate_constraints(
            mix_design={
                "cement": 120.0,
                "slag": 100.0,
                "fly_ash": 50.0,
                "water": 190.0,
                "superplasticizer": 7.0,
                "coarse_aggregate": 1000.0,
                "fine_aggregate": 750.0,
                "age": 28.0,
            },
            engineered_features={
                "water_cement_ratio": 1.58,
                "fly_ash_replacement_ratio": 0.50,
                "slag_replacement_ratio": 0.40,
            },
            constraints=constraints,
        )

        self.assertFalse(evaluation.passed)
        self.assertGreaterEqual(len(evaluation.checks), 3)
        self.assertIn("water_cement_ratio.max", evaluation.hard_failures)
        self.assertIn("fly_ash_replacement_ratio.max", evaluation.hard_failures)


if __name__ == "__main__":
    unittest.main()
