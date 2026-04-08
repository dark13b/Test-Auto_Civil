import tempfile
import unittest
from pathlib import Path

import pandas as pd

from feature_lab import FeatureLab


class FeatureLabTests(unittest.TestCase):
    def setUp(self) -> None:
        self.frame = pd.DataFrame(
            {
                "cement": [200.0, 300.0],
                "water": [150.0, 180.0],
                "slag": [20.0, 30.0],
                "fly_ash": [10.0, 20.0],
                "superplasticizer": [5.0, 6.0],
                "coarse_aggregate": [1000.0, 1020.0],
                "fine_aggregate": [700.0, 710.0],
                "age": [7.0, 28.0],
            }
        )

    def test_rejects_import_statements(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            lab = FeatureLab(Path(tmpdir), timeout_seconds=30, min_delta_rmse=0.05)

            result = lab.validate_feature_code(
                "import os\n\ndef new_feature(df: pd.DataFrame) -> pd.Series:\n    return df['cement']\n"
            )

        self.assertFalse(result["accepted"])
        self.assertIn("import", result["reason"].lower())

    def test_rejects_unsafe_names(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            lab = FeatureLab(Path(tmpdir), timeout_seconds=30, min_delta_rmse=0.05)

            result = lab.validate_feature_code(
                "def new_feature(df: pd.DataFrame) -> pd.Series:\n    return os.listdir('.')\n"
            )

        self.assertFalse(result["accepted"])
        self.assertIn("unsafe", result["reason"].lower())

    def test_executes_safe_feature_code(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            lab = FeatureLab(Path(tmpdir), timeout_seconds=30, min_delta_rmse=0.05)

            result = lab.execute_feature_code(
                code="def new_feature(df: pd.DataFrame) -> pd.Series:\n    return df['cement'] / (df['water'] + 1.0)\n",
                frame=self.frame,
            )

        self.assertTrue(result["accepted"])
        self.assertEqual(result["series_name"], "new_feature")
        self.assertEqual(len(result["values"]), len(self.frame))


if __name__ == "__main__":
    unittest.main()
