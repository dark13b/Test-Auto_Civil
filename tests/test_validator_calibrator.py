import copy
import unittest

import pandas as pd

from validator_calibrator import ValidatorCalibrator


class ValidatorCalibratorTests(unittest.TestCase):
    def test_suggest_thresholds_uses_dataset_and_preserves_original_config(self) -> None:
        config = {
            "validator": {
                "total_binder_low_warn": 250.0,
                "suspicious_water_cement_ratio": 0.7,
                "fly_ash_replacement_warn": 0.4,
            }
        }
        original = copy.deepcopy(config)
        frame = pd.DataFrame(
            {
                "cement": [250.0, 300.0, 350.0],
                "slag": [20.0, 40.0, 60.0],
                "fly_ash": [0.0, 30.0, 80.0],
                "water": [160.0, 170.0, 180.0],
            }
        )

        calibrator = ValidatorCalibrator()
        suggestions = calibrator.suggest_thresholds(frame, config)

        self.assertIn("total_binder_low_warn", suggestions)
        self.assertIn("suspicious_water_cement_ratio", suggestions)
        self.assertIn("fly_ash_replacement_warn", suggestions)
        self.assertEqual(config, original)


if __name__ == "__main__":
    unittest.main()
