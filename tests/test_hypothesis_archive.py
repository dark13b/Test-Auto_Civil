import tempfile
import unittest
from pathlib import Path

from hypothesis_archive import HypothesisArchive


class HypothesisArchiveTests(unittest.TestCase):
    def test_archive_initializes_when_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            archive = HypothesisArchive(Path(tmpdir) / "hypothesis_archive.json")
            payload = archive.load()

        self.assertEqual(payload["records"], [])
        self.assertEqual(payload["summary"]["total_hypotheses"], 0)

    def test_completed_record_requires_expected_and_actual_delta(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            archive = HypothesisArchive(Path(tmpdir) / "hypothesis_archive.json")

            with self.assertRaises(ValueError):
                archive.record_trial(
                    {
                        "hypothesis_id": "h-1",
                        "run_id": "run-1",
                        "cycle": 1,
                        "source": "llm",
                        "claim": "claim",
                        "mechanism": "mechanism",
                        "expected_delta_rmse": None,
                        "actual_delta_rmse": 0.2,
                        "calibration_error": 0.2,
                        "proposal": {"model_name": "ModelA"},
                        "outcome": "kept",
                        "trial_number": 1,
                        "timestamp": "2026-03-21T00:00:00",
                    }
                )

    def test_summary_updates_from_recorded_trials(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            archive = HypothesisArchive(Path(tmpdir) / "hypothesis_archive.json")
            archive.record_trial(
                {
                    "hypothesis_id": "h-1",
                    "run_id": "run-1",
                    "cycle": 1,
                    "source": "llm",
                    "claim": "claim",
                    "mechanism": "mechanism",
                    "expected_delta_rmse": 0.3,
                    "actual_delta_rmse": 0.2,
                    "calibration_error": 0.1,
                    "proposal": {"model_name": "ModelA"},
                    "outcome": "kept",
                    "trial_number": 1,
                    "timestamp": "2026-03-21T00:00:00",
                }
            )
            payload = archive.record_trial(
                {
                    "hypothesis_id": "h-2",
                    "run_id": "run-1",
                    "cycle": 1,
                    "source": "llm",
                    "claim": "claim",
                    "mechanism": "mechanism",
                    "expected_delta_rmse": 0.2,
                    "actual_delta_rmse": -0.1,
                    "calibration_error": 0.3,
                    "proposal": {"model_name": "ModelB"},
                    "outcome": "reverted",
                    "trial_number": 2,
                    "timestamp": "2026-03-21T00:05:00",
                }
            )

        self.assertEqual(payload["summary"]["total_hypotheses"], 2)
        self.assertAlmostEqual(payload["summary"]["mean_calibration_error"], 0.2)
        self.assertAlmostEqual(payload["summary"]["acceptance_rate"], 0.5)


if __name__ == "__main__":
    unittest.main()
