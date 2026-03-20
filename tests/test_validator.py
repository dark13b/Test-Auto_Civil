import copy
import unittest

import numpy as np
import pandas as pd

from validator import EngineeringValidator


BASE_VALIDATOR_CONFIG = {
    "engineering_bounds": {
        "min": 0.0,
        "max": 120.0,
    },
    "validator": {
        "context_type": "general",
        "suspicious_water_cement_ratio": 0.7,
        "suspicious_strength_mpa": 30.0,
        "durability_water_cement_warn": 0.6,
        "high_water_cement_strength_hard_fail_max_age_days": 28.0,
        "high_water_cement_strength_low_scm_threshold": 0.20,
        "high_water_cement_strength_unfavorable_water_binder_ratio": 0.50,
        "low_water_binder_warn": 0.25,
        "total_binder_low_warn": 250.0,
        "total_binder_high_warn": 550.0,
        "fly_ash_replacement_warn": 0.4,
        "slag_replacement_warn": 0.7,
        "early_age_days_warn": 3.0,
        "early_age_strength_warn": 30.0,
    },
}


def make_validator(**validator_overrides: float | str) -> EngineeringValidator:
    config = copy.deepcopy(BASE_VALIDATOR_CONFIG)
    config["validator"].update(validator_overrides)
    return EngineeringValidator.from_config(config)


def make_mix(**overrides: float | str | bool) -> pd.DataFrame:
    mix = {
        "cement": 320.0,
        "slag": 0.0,
        "fly_ash": 0.0,
        "water": 180.0,
        "superplasticizer": 6.0,
        "coarse_aggregate": 1050.0,
        "fine_aggregate": 720.0,
        "age": 28.0,
    }
    mix.update(overrides)
    return pd.DataFrame([mix])


def warning_codes(report: dict, key: str) -> set[str]:
    return {item["warning_code"] for item in report[key]}


def warning_by_code(report: dict, key: str, code: str) -> dict:
    for item in report[key]:
        if item["warning_code"] == code:
            return item
    raise AssertionError(f"Warning code not found in {key}: {code}")


class EngineeringValidatorContextTests(unittest.TestCase):
    def test_scm_bearing_mix_uses_water_binder_and_age_context_before_water_cement_logic(self) -> None:
        validator = make_validator()
        x_test = make_mix(
            cement=200.0,
            slag=100.0,
            fly_ash=50.0,
            water=155.0,
            superplasticizer=8.0,
            age=90.0,
        )

        sample_report = validator.validate_predictions(np.asarray([48.0]), x_test)["sample_reports"][0]

        self.assertFalse(sample_report["hard_constraints"])
        self.assertFalse(sample_report["data_review_flags"])
        self.assertTrue(sample_report["downgraded_warnings"])
        self.assertIn("scm", sample_report["contextual_summary"].lower())
        self.assertEqual(sample_report["confidence_of_warning_assessment"], "high")
        self.assertIn("high_water_cement_high_strength_context_review", sample_report["downgraded_rule_codes"])

    def test_high_fly_ash_replacement_becomes_special_regime_review_not_automatic_anomaly(self) -> None:
        validator = make_validator()
        x_test = make_mix(
            cement=180.0,
            fly_ash=200.0,
            water=160.0,
            superplasticizer=10.0,
            age=56.0,
        )

        sample_report = validator.validate_predictions(np.asarray([36.0]), x_test)["sample_reports"][0]

        self.assertFalse(sample_report["hard_constraints"])
        self.assertIn("high_volume_scm_regime_review", warning_codes(sample_report, "data_review_flags"))
        review_flag = warning_by_code(sample_report, "data_review_flags", "high_volume_scm_regime_review")
        self.assertEqual(review_flag["warning_category"], "Data Review Flag")
        self.assertEqual(review_flag["severity"], "medium")
        self.assertIn("replacement", review_flag["academic_note"].lower())

    def test_high_strength_high_water_cement_is_not_over_flagged_when_scm_and_age_support_it(self) -> None:
        validator = make_validator()
        x_test = make_mix(
            cement=250.0,
            slag=100.0,
            fly_ash=80.0,
            water=175.0,
            superplasticizer=10.0,
            age=90.0,
        )

        sample_report = validator.validate_predictions(np.asarray([56.0]), x_test)["sample_reports"][0]

        self.assertFalse(sample_report["hard_constraints"])
        self.assertFalse(sample_report["data_review_flags"])
        self.assertNotIn(
            "high_water_cement_high_strength_context_review",
            warning_codes(sample_report, "engineering_cautions") | warning_codes(sample_report, "data_review_flags"),
        )
        self.assertIn("downgraded", sample_report["contextual_summary"].lower())

    def test_durability_warning_uses_exposure_aware_logic_when_metadata_exists(self) -> None:
        validator = make_validator()
        x_test = make_mix(
            cement=400.0,
            water=176.0,
            exposure_class="marine",
        )

        sample_report = validator.validate_predictions(np.asarray([34.0]), x_test)["sample_reports"][0]

        caution = warning_by_code(sample_report, "engineering_cautions", "durability_exposure_water_ratio_caution")
        self.assertEqual(caution["warning_category"], "Engineering Caution")
        self.assertEqual(caution["severity"], "medium")
        self.assertIn("marine", caution["evidence_summary"].lower())
        self.assertEqual(caution["assessment_confidence"], "high")

    def test_fallback_durability_caution_works_without_exposure_metadata(self) -> None:
        validator = make_validator()
        x_test = make_mix(
            cement=280.0,
            water=180.0,
        )

        sample_report = validator.validate_predictions(np.asarray([28.0]), x_test)["sample_reports"][0]

        caution = warning_by_code(sample_report, "engineering_cautions", "durability_general_water_ratio_caution")
        self.assertIn("general guidance", caution["message"].lower())
        self.assertEqual(caution["assessment_confidence"], "moderate")
        self.assertIn("exposure", caution["recommended_review_action"].lower())

    def test_high_binder_shrinkage_logic_is_conditional(self) -> None:
        validator = make_validator()
        paste_rich = make_mix(
            cement=420.0,
            slag=120.0,
            fly_ash=40.0,
            water=170.0,
            coarse_aggregate=900.0,
            fine_aggregate=700.0,
        )
        balanced = make_mix(
            cement=420.0,
            slag=120.0,
            fly_ash=40.0,
            water=250.0,
            coarse_aggregate=1350.0,
            fine_aggregate=950.0,
        )

        paste_rich_report = validator.validate_predictions(np.asarray([62.0]), paste_rich)["sample_reports"][0]
        balanced_report = validator.validate_predictions(np.asarray([40.0]), balanced)["sample_reports"][0]

        self.assertIn("high_binder_shrinkage_caution", warning_codes(paste_rich_report, "engineering_cautions"))
        self.assertNotIn("high_binder_shrinkage_caution", warning_codes(balanced_report, "engineering_cautions"))

    def test_low_water_binder_workability_warning_changes_severity_with_support_metadata(self) -> None:
        validator = make_validator()
        unsupported = make_mix(
            cement=430.0,
            water=105.0,
            superplasticizer=0.0,
            workability_support=False,
        )
        supported = make_mix(
            cement=430.0,
            water=105.0,
            superplasticizer=14.0,
            workability_support=True,
        )

        unsupported_report = validator.validate_predictions(np.asarray([58.0]), unsupported)["sample_reports"][0]
        supported_report = validator.validate_predictions(np.asarray([58.0]), supported)["sample_reports"][0]

        unsupported_warning = warning_by_code(
            unsupported_report,
            "engineering_cautions",
            "low_water_binder_workability_caution",
        )
        supported_warning = warning_by_code(
            supported_report,
            "engineering_cautions",
            "low_water_binder_workability_caution",
        )
        self.assertEqual(unsupported_warning["severity"], "high")
        self.assertEqual(supported_warning["severity"], "low")
        self.assertIn("support", supported_warning["evidence_summary"].lower())

    def test_structured_outputs_are_classified_into_explicit_categories(self) -> None:
        validator = make_validator()
        x_test = pd.concat(
            [
                make_mix(cement=0.0, water=180.0),
                make_mix(cement=260.0, water=180.0, exposure_class="marine"),
                make_mix(cement=180.0, fly_ash=200.0, water=160.0, age=56.0),
            ],
            ignore_index=True,
        )
        predictions = np.asarray([-3.0, 32.0, 34.0])

        report = validator.validate_predictions(predictions, x_test)

        self.assertIn("hard_constraint_count", report)
        self.assertIn("engineering_caution_count", report)
        self.assertIn("data_review_flag_count", report)
        self.assertEqual(report["hard_constraint_count"], 1)
        self.assertEqual(report["engineering_caution_count"], 1)
        self.assertEqual(report["data_review_flag_count"], 1)

    def test_academic_note_and_evidence_summary_are_generated(self) -> None:
        validator = make_validator()
        x_test = make_mix(
            cement=400.0,
            water=176.0,
            exposure_class="marine",
        )

        sample_report = validator.validate_predictions(np.asarray([34.0]), x_test)["sample_reports"][0]

        caution = warning_by_code(sample_report, "engineering_cautions", "durability_exposure_water_ratio_caution")
        self.assertTrue(caution["academic_note"])
        self.assertTrue(caution["evidence_summary"])
        self.assertIsInstance(caution["triggering_factors"], dict)
        self.assertIn("water_cement_ratio", caution["triggering_factors"])


if __name__ == "__main__":
    unittest.main()
