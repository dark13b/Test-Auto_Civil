import json
import tempfile
import unittest
from pathlib import Path

from research_protocol import apply_keep_to_research_surface, read_research_surface_state
from state_store import JSONStateStore, default_runtime_state_path


class StateStoreTests(unittest.TestCase):
    def test_json_state_store_round_trip_returns_copies(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = JSONStateStore(Path(tmpdir) / "runtime_state.json")
            saved_state = {
                "surface_version": 1,
                "accepted_experiments": [{"experiment_id": "confirm-001", "composite_score": 0.91}],
                "recent_kept_families": ["boosting"],
            }

            metadata = store.save_state(saved_state, migrated_from="legacy")
            loaded_state = store.load_state()
            loaded_state["accepted_experiments"].append({"experiment_id": "confirm-002"})

            reloaded_state = store.load_state()

        self.assertEqual(metadata["schema_version"], 1)
        self.assertEqual(metadata["state_kind"], "lab_state")
        self.assertEqual(len(reloaded_state["accepted_experiments"]), 1)
        self.assertEqual(reloaded_state["accepted_experiments"][0]["experiment_id"], "confirm-001")

    def test_research_protocol_migrates_legacy_lab_state_to_runtime_store(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir)
            lab_path = project_root / "research_lab.py"
            legacy_text = (
                '"""editable surface"""\n'
                "# RESEARCH_SURFACE_STATE_START\n"
                "LAB_STATE = {\n"
                "    'surface_version': 1,\n"
                "    'accepted_experiments': [],\n"
                "    'recent_kept_families': [],\n"
                "}\n"
                "# RESEARCH_SURFACE_STATE_END\n"
            )
            lab_path.write_text(legacy_text, encoding="utf-8")

            migrated_state = read_research_surface_state(lab_path)
            runtime_state_path = default_runtime_state_path(project_root)
            apply_keep_to_research_surface(
                research_lab_path=lab_path,
                accepted_entry={
                    "experiment_id": "confirm-001",
                    "proposal_family": "boosting-balanced",
                    "model_name": "LGBMRegressor",
                    "composite_score": 0.91,
                },
            )
            persisted_payload = json.loads(runtime_state_path.read_text(encoding="utf-8"))
            source_after_keep = lab_path.read_text(encoding="utf-8")
            store_exists = runtime_state_path.exists()

        self.assertEqual(migrated_state["accepted_experiments"], [])
        self.assertTrue(store_exists)
        self.assertEqual(len(persisted_payload["state"]["accepted_experiments"]), 1)
        self.assertIn("LAB_STATE =", source_after_keep)
        self.assertNotIn("confirm-001", source_after_keep)


if __name__ == "__main__":
    unittest.main()
