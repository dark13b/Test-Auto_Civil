import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from artifact_sync import AtomicArtifactWriter, repair_on_startup


class ArtifactSyncTests(unittest.TestCase):
    def test_append_trial_writes_all_csv_targets_and_clears_pending_meta(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            outputs_dir = Path(tmpdir)
            writer = AtomicArtifactWriter(
                outputs_dir=outputs_dir,
                csv_targets=[
                    ("research_results.csv", ["trial_number", "model_name", "selection_status"]),
                    ("optuna_results.csv", ["trial_number", "model_name", "selection_status"]),
                ],
            )

            writer.append_trial(
                {
                    "trial_number": 1,
                    "model_name": "Ridge",
                    "selection_status": "no_improvement",
                }
            )

            self.assertFalse((outputs_dir / ".sync_pending_meta.json").exists())
            first = pd.read_csv(outputs_dir / "research_results.csv")
            second = pd.read_csv(outputs_dir / "optuna_results.csv")
            self.assertEqual(first.to_dict(orient="records"), second.to_dict(orient="records"))
            self.assertEqual(int(first.loc[0, "trial_number"]), 1)

    def test_check_only_reports_desync_when_best_json_exists_but_csv_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            outputs_dir = Path(tmpdir)
            (outputs_dir / "search_state_best_result.json").write_text(
                json.dumps(
                    {
                        "trial_number": 11,
                        "best_trial": 11,
                        "model_name": "CatBoostRegressor",
                        "selection_status": "new_best",
                        "composite_score": 0.88,
                    }
                ),
                encoding="utf-8",
            )

            report = repair_on_startup(
                outputs_dir,
                check_only=True,
                csv_targets=[("optuna_results.csv", ["trial_number", "model_name", "selection_status"])],
                best_result_filename="search_state_best_result.json",
            )

            self.assertTrue(report["pending"])
            self.assertFalse(report["repaired"])
            self.assertEqual(report["operation"], "recover_best_result_desync")
            self.assertEqual(report["best_result_trial_number"], 11)
            self.assertIn("optuna_results.csv", report["desynced_targets"])

    def test_repair_on_startup_restores_missing_csv_from_best_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            outputs_dir = Path(tmpdir)
            (outputs_dir / "search_state_best_result.json").write_text(
                json.dumps(
                    {
                        "trial_number": 12,
                        "best_trial": 12,
                        "model_name": "LGBMRegressor",
                        "selection_status": "new_best",
                        "composite_score": 0.91,
                    }
                ),
                encoding="utf-8",
            )

            report = repair_on_startup(
                outputs_dir,
                csv_targets=[("optuna_results.csv", ["trial_number", "model_name", "selection_status"])],
                best_result_filename="search_state_best_result.json",
            )

            rows = pd.read_csv(outputs_dir / "optuna_results.csv")
            self.assertTrue(report["pending"])
            self.assertTrue(report["repaired"])
            self.assertEqual(report["operation"], "recover_best_result_desync")
            self.assertEqual(int(rows.loc[0, "trial_number"]), 12)
            self.assertEqual(str(rows.loc[0, "model_name"]), "LGBMRegressor")
            self.assertEqual(str(rows.loc[0, "selection_status"]), "new_best")

    def test_record_new_best_appends_trial_and_updates_best_payload(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            outputs_dir = Path(tmpdir)
            writer = AtomicArtifactWriter(
                outputs_dir=outputs_dir,
                csv_targets=[("optuna_results.csv", ["trial_number", "model_name", "selection_status"])],
                best_result_filename="search_state_best_result.json",
            )

            writer.record_new_best(
                trial_record={
                    "trial_number": 3,
                    "model_name": "ExtraTreesRegressor",
                    "selection_status": "new_best",
                },
                best_result={
                    "trial_number": 3,
                    "best_trial": 3,
                    "model_name": "ExtraTreesRegressor",
                    "composite_score": 0.91,
                },
            )

            best_payload = json.loads((outputs_dir / "search_state_best_result.json").read_text(encoding="utf-8"))
            rows = pd.read_csv(outputs_dir / "optuna_results.csv")
            self.assertEqual(best_payload["best_trial"], 3)
            self.assertEqual(best_payload["model_name"], "ExtraTreesRegressor")
            self.assertEqual(int(rows.loc[0, "trial_number"]), 3)
            self.assertEqual(str(rows.loc[0, "selection_status"]), "new_best")

    def test_repair_on_startup_replays_pending_record_new_best(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            outputs_dir = Path(tmpdir)
            pending_path = outputs_dir / ".sync_pending_meta.json"
            pending_payload = {
                "operation": "record_new_best",
                "csv_targets": [
                    {
                        "filename": "optuna_results.csv",
                        "columns": ["trial_number", "model_name", "selection_status"],
                    }
                ],
                "best_result_filename": "search_state_best_result.json",
                "trial_record": {
                    "trial_number": 4,
                    "model_name": "RandomForestRegressor",
                    "selection_status": "new_best",
                },
                "best_result": {
                    "trial_number": 4,
                    "best_trial": 4,
                    "model_name": "RandomForestRegressor",
                    "composite_score": 0.93,
                },
            }
            pending_path.write_text(json.dumps(pending_payload), encoding="utf-8")

            report = repair_on_startup(outputs_dir)

            self.assertTrue(report["repaired"])
            self.assertFalse((outputs_dir / ".sync_pending_meta.json").exists())
            rows = pd.read_csv(outputs_dir / "optuna_results.csv")
            best_payload = json.loads((outputs_dir / "search_state_best_result.json").read_text(encoding="utf-8"))
            self.assertEqual(len(rows), 1)
            self.assertEqual(int(rows.loc[0, "trial_number"]), 4)
            self.assertEqual(best_payload["best_trial"], 4)

    def test_repair_on_startup_check_only_reports_pending_without_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            outputs_dir = Path(tmpdir)
            pending_path = outputs_dir / ".sync_pending_meta.json"
            pending_path.write_text(
                json.dumps(
                    {
                        "operation": "append_trial",
                        "csv_targets": [
                            {
                                "filename": "optuna_results.csv",
                                "columns": ["trial_number", "model_name", "selection_status"],
                            }
                        ],
                        "trial_record": {
                            "trial_number": 7,
                            "model_name": "LGBMRegressor",
                            "selection_status": "error",
                        },
                    }
                ),
                encoding="utf-8",
            )

            report = repair_on_startup(outputs_dir, check_only=True)

            self.assertFalse(report["repaired"])
            self.assertTrue(report["pending"])
            self.assertEqual(report["operation"], "append_trial")
            self.assertTrue(pending_path.exists())
            self.assertFalse((outputs_dir / "optuna_results.csv").exists())


if __name__ == "__main__":
    unittest.main()
