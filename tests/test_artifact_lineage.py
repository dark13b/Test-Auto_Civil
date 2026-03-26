import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from train import write_run_scoped_json_artifact


class ArtifactLineageTests(unittest.TestCase):
    def test_run_scoped_json_artifact_contains_lineage_and_writes_versioned_copy(self) -> None:
        config = {
            "paths": {"outputs_dir": "outputs"},
            "data": {"mode": "local_file", "local_file": {"path": "dataset.csv"}},
            "task": {"input_columns": ["cement", "water", "age"]},
            "engineering": {"feature_engineering": True, "binder_efficiency": {"fly_ash_k": 0.35}},
            "experiment": {"random_seed": 42},
        }
        with TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "dataset.csv").write_text("cement,water,age,compressive_strength\n1,2,3,4\n", encoding="utf-8")
            outputs_dir = Path(tmpdir) / "outputs"
            outputs_dir.mkdir(parents=True, exist_ok=True)

            canonical_path, run_path, payload = write_run_scoped_json_artifact(
                outputs_dir=outputs_dir,
                filename="best_search_result.json",
                payload={"model_name": "StackingRegressor", "composite_score": 0.91},
                run_id="run-20260320T172800",
                source_mode="search",
                config=config,
                model_artifact_id="model-artifact-123",
                model_id="StackingRegressor",
                parent_artifact_ids=["baseline-artifact"],
                parent_run_ids=["run-20260319T010203"],
            )

            written_payload = json.loads(canonical_path.read_text(encoding="utf-8"))
            manifest_payload = json.loads((outputs_dir / "runs" / "run-20260320T172800" / "run_manifest.json").read_text(encoding="utf-8"))
            self.assertTrue(run_path.exists())

        self.assertEqual(canonical_path.name, "best_search_result.json")
        self.assertEqual(written_payload["run_id"], "run-20260320T172800")
        self.assertEqual(written_payload["source_mode"], "search")
        self.assertIn("timestamp", written_payload)
        self.assertIn("artifact_id", written_payload)
        self.assertEqual(written_payload["artifact_metadata"]["parent_artifact_ids"], ["baseline-artifact"])
        self.assertEqual(written_payload["artifact_metadata"]["parent_run_id"], "run-20260319T010203")
        self.assertEqual(written_payload["parent_run_id"], "run-20260319T010203")
        self.assertIsNotNone(written_payload["artifact_metadata"]["config_hash"])
        self.assertIsNotNone(written_payload["artifact_metadata"]["dataset_hash"])
        self.assertIsNotNone(written_payload["artifact_metadata"]["feature_hash"])
        self.assertEqual(written_payload["artifact_metadata"]["model_fingerprint"], "model-artifact-123")
        self.assertEqual(manifest_payload["artifacts"]["best_search_result.json"]["dataset_hash"], written_payload["artifact_metadata"]["dataset_hash"])
        self.assertEqual(payload["artifact_metadata"]["run_scoped_path"], str(run_path))


if __name__ == "__main__":
    unittest.main()
