import unittest

import numpy as np
import pandas as pd

from feature_engineering import (
    ENGINEERED_FEATURE_COLUMNS,
    ENGINEERING_DIAGNOSTIC_COLUMNS,
    build_engineering_features,
)


def make_config(fly_ash_k: float = 0.35, slag_k: float = 0.8) -> dict:
    return {
        "engineering": {
            "binder_efficiency": {
                "fly_ash_k": fly_ash_k,
                "slag_k": slag_k,
            }
        }
    }


def make_frame(**overrides: float) -> pd.DataFrame:
    row = {
        "cement": 200.0,
        "slag": 120.0,
        "fly_ash": 80.0,
        "water": 160.0,
        "superplasticizer": 8.0,
        "coarse_aggregate": 1050.0,
        "fine_aggregate": 700.0,
        "age": 28.0,
    }
    row.update(overrides)
    return pd.DataFrame([row])


class FeatureEngineeringTests(unittest.TestCase):
    def test_effective_binder_uses_configurable_scm_efficiency(self) -> None:
        frame = make_frame()

        engineered = build_engineering_features(frame, config=make_config(fly_ash_k=0.30, slag_k=0.75))

        expected_effective_binder = 200.0 + (120.0 * 0.75) + (80.0 * 0.30)
        self.assertAlmostEqual(float(engineered.loc[0, "effective_binder"]), expected_effective_binder)
        self.assertAlmostEqual(
            float(engineered.loc[0, "water_effective_binder_ratio"]),
            160.0 / expected_effective_binder,
        )
        self.assertAlmostEqual(
            float(engineered.loc[0, "effective_scm_replacement_ratio"]),
            ((120.0 * 0.75) + (80.0 * 0.30)) / expected_effective_binder,
        )

    def test_inverse_ratio_features_match_forward_ratios(self) -> None:
        frame = make_frame(cement=250.0, slag=50.0, fly_ash=50.0, water=140.0)

        engineered = build_engineering_features(frame, config=make_config())

        self.assertAlmostEqual(
            float(engineered.loc[0, "binder_to_water_ratio"]),
            1.0 / float(engineered.loc[0, "water_binder_ratio"]),
        )
        self.assertAlmostEqual(
            float(engineered.loc[0, "effective_binder_to_water_ratio"]),
            1.0 / float(engineered.loc[0, "water_effective_binder_ratio"]),
        )
        self.assertAlmostEqual(
            float(engineered.loc[0, "paste_to_aggregate_ratio"]),
            1.0 / float(engineered.loc[0, "aggregate_paste_ratio"]),
        )

    def test_age_coupled_effective_binder_features_are_monotonic(self) -> None:
        young = build_engineering_features(make_frame(age=7.0), config=make_config())
        mature = build_engineering_features(make_frame(age=56.0), config=make_config())

        self.assertGreater(
            float(mature.loc[0, "effective_binder_age_interaction"]),
            float(young.loc[0, "effective_binder_age_interaction"]),
        )
        self.assertLess(
            float(mature.loc[0, "water_effective_binder_age"]),
            float(young.loc[0, "water_effective_binder_age"]),
        )

    def test_zero_scm_mix_keeps_effective_and_total_binder_aligned(self) -> None:
        engineered = build_engineering_features(
            make_frame(slag=0.0, fly_ash=0.0),
            config=make_config(),
        )

        self.assertAlmostEqual(
            float(engineered.loc[0, "effective_binder"]),
            float(engineered.loc[0, "total_binder"]),
        )
        self.assertAlmostEqual(
            float(engineered.loc[0, "water_effective_binder_ratio"]),
            float(engineered.loc[0, "water_binder_ratio"]),
        )
        self.assertAlmostEqual(float(engineered.loc[0, "binder_efficiency_gap"]), 0.0)

    def test_training_feature_list_excludes_diagnostics(self) -> None:
        self.assertIn("effective_binder", ENGINEERED_FEATURE_COLUMNS)
        self.assertIn("water_effective_binder_ratio", ENGINEERED_FEATURE_COLUMNS)
        self.assertIn("effective_scm_replacement_ratio", ENGINEERED_FEATURE_COLUMNS)
        self.assertIn("binder_efficiency_gap", ENGINEERING_DIAGNOSTIC_COLUMNS)
        self.assertNotIn("binder_efficiency_gap", ENGINEERED_FEATURE_COLUMNS)

    def test_generated_features_are_finite(self) -> None:
        engineered = build_engineering_features(
            make_frame(cement=0.0, slag=0.0, fly_ash=250.0, water=180.0),
            config=make_config(),
        )

        values = engineered[ENGINEERED_FEATURE_COLUMNS + ENGINEERING_DIAGNOSTIC_COLUMNS].to_numpy(dtype=float)
        self.assertTrue(np.isfinite(values).all())

    def test_optional_promoted_features_are_loaded_when_configured(self) -> None:
        config = make_config()
        config["engineering"]["experimental_feature_modules"] = ["tests.support_test_promoted_features"]

        engineered = build_engineering_features(make_frame(), config=config)

        self.assertIn("cement_plus_water_feature", engineered.columns)
        self.assertAlmostEqual(
            float(engineered.loc[0, "cement_plus_water_feature"]),
            360.0,
        )


if __name__ == "__main__":
    unittest.main()
