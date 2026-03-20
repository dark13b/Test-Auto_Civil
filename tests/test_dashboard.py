import importlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


class DashboardTests(unittest.TestCase):
    def test_dashboard_imports_flask_module_using_lowercase_name(self) -> None:
        source = Path("dashboard.py").read_text(encoding="utf-8")

        self.assertIn("from flask import", source)

    def test_dashboard_module_imports_and_exposes_app(self) -> None:
        dashboard = importlib.import_module("dashboard")

        self.assertEqual(dashboard.app.import_name, "dashboard")

    def test_normalize_result_payload_prefers_new_warning_category_counts(self) -> None:
        dashboard = importlib.import_module("dashboard")

        payload = {
            "validation_report": {
                "hard_constraint_count": 2,
                "engineering_caution_count": 3,
                "data_review_flag_count": 4,
                "hard_constraint_reasons": ["hard constraint"],
                "engineering_caution_reasons": ["engineering caution"],
                "data_review_flag_reasons": ["data review flag"],
            }
        }

        normalized = dashboard.normalize_result_payload(payload)

        self.assertEqual(normalized["hard_failed_count"], 2)
        self.assertEqual(normalized["engineering_caution_count"], 3)
        self.assertEqual(normalized["dataset_anomaly_count"], 4)
        self.assertEqual(normalized["hard_fail_reasons"], ["hard constraint"])
        self.assertEqual(normalized["engineering_caution_reasons"], ["engineering caution"])
        self.assertEqual(normalized["dataset_anomaly_reasons"], ["data review flag"])

    def test_normalize_result_payload_preserves_suspicious_count_and_dataset_anomaly_count(self) -> None:
        dashboard = importlib.import_module("dashboard")

        payload = {
            "suspicious_count": 2,
            "dataset_anomaly_count": 5,
            "validation_report": {
                "warning_count": 1,
            },
        }

        normalized = dashboard.normalize_result_payload(payload)

        self.assertEqual(normalized["suspicious_count"], 2)
        self.assertEqual(normalized["dataset_anomaly_count"], 5)

    def test_dashboard_requires_auth_even_without_env_password(self) -> None:
        with patch.dict(os.environ, {}, clear=False):
            dashboard = importlib.reload(importlib.import_module("dashboard"))

        client = dashboard.app.test_client()
        response = client.get("/")

        self.assertEqual(response.status_code, 401)
        self.assertTrue(dashboard.DASHBOARD_PASSWORD)

    def test_field_validation_api_escapes_notes(self) -> None:
        dashboard = importlib.import_module("dashboard")
        with tempfile.TemporaryDirectory() as tmpdir:
            outputs_dir = Path(tmpdir)
            (outputs_dir / "field_validation_log.json").write_text(
                json.dumps(
                    [
                        {
                            "timestamp": "2026-03-15T12:00:00",
                            "notes": "<script>alert(1)</script>",
                        }
                    ]
                ),
                encoding="utf-8",
            )
            with patch.object(dashboard, "OUTPUTS_DIR", outputs_dir), patch.object(
                dashboard,
                "DASHBOARD_PASSWORD",
                "dummy",
            ):
                client = dashboard.app.test_client()
                response = client.get(
                    "/api/field_validation",
                    headers={"Authorization": "Basic YXV0b2NpdmlsOmR1bW15"},
                )

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload[0]["notes"], "&lt;script&gt;alert(1)&lt;/script&gt;")

    def test_share_latest_creates_public_link_and_tracks_event(self) -> None:
        dashboard = importlib.import_module("dashboard")
        with tempfile.TemporaryDirectory() as tmpdir:
            outputs_dir = Path(tmpdir) / "outputs"
            data_dir = Path(tmpdir) / "data"
            outputs_dir.mkdir(parents=True, exist_ok=True)
            data_dir.mkdir(parents=True, exist_ok=True)

            (outputs_dir / "baseline_metrics.json").write_text(
                json.dumps({"model_name": "RandomForestRegressor", "composite_score": 0.9}),
                encoding="utf-8",
            )
            (outputs_dir / "final_metrics.json").write_text(
                json.dumps(
                    {
                        "composite_improvement_pct": 1.2,
                        "best_search_metrics": {"model_name": "LGBMRegressor", "composite_score": 0.92},
                    }
                ),
                encoding="utf-8",
            )
            (data_dir / "concrete_data.csv").write_text(
                "cement,water,compressive_strength\n300,180,35\n",
                encoding="utf-8",
            )

            with patch.object(dashboard, "OUTPUTS_DIR", outputs_dir), patch.object(
                dashboard, "DATA_DIR", data_dir
            ), patch.object(
                dashboard,
                "DASHBOARD_PASSWORD",
                "dummy",
            ):
                client = dashboard.app.test_client()
                response = client.post(
                    "/api/share_latest",
                    headers={"Authorization": "Basic YXV0b2NpdmlsOmR1bW15"},
                )

            self.assertEqual(response.status_code, 201)
            payload = response.get_json()
            self.assertIn("/shared/", payload["share_url"])
            self.assertIn("/shared/", payload["share_path"])
            token = payload["share_url"].rsplit("/", 1)[-1]

            snapshot_store = json.loads((outputs_dir / "shared_snapshots.json").read_text(encoding="utf-8"))
            self.assertIn(token, snapshot_store["links"])

            event_lines = (outputs_dir / "share_events.jsonl").read_text(encoding="utf-8").strip().splitlines()
            events = [json.loads(line) for line in event_lines]
            self.assertEqual(events[-1]["event"], "share_link_created")

    def test_shared_link_route_is_public_and_tracks_opens(self) -> None:
        dashboard = importlib.import_module("dashboard")
        with tempfile.TemporaryDirectory() as tmpdir:
            outputs_dir = Path(tmpdir) / "outputs"
            outputs_dir.mkdir(parents=True, exist_ok=True)
            token = "abc123"
            (outputs_dir / "shared_snapshots.json").write_text(
                json.dumps(
                    {
                        "links": {
                            token: {
                                "created_at": "2026-03-15T08:00:00Z",
                                "expires_at": "2099-03-15T08:00:00Z",
                                "snapshot": {
                                    "best_model": "LGBMRegressor",
                                    "composite_improvement_pct": 1.2,
                                },
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )

            with patch.object(dashboard, "OUTPUTS_DIR", outputs_dir), patch.object(
                dashboard,
                "DASHBOARD_PASSWORD",
                "dummy",
            ):
                client = dashboard.app.test_client()
                response = client.get(f"/shared/{token}")

            self.assertEqual(response.status_code, 200)
            self.assertIn(b"LGBMRegressor", response.data)
            self.assertIn(b"Composite Improvement", response.data)

            event_lines = (outputs_dir / "share_events.jsonl").read_text(encoding="utf-8").strip().splitlines()
            events = [json.loads(line) for line in event_lines]
            self.assertEqual(events[-1]["event"], "share_link_opened")

    def test_dashboard_template_defines_best_metric_source(self) -> None:
        dashboard = importlib.import_module("dashboard")
        with patch.object(dashboard, "DASHBOARD_PASSWORD", "dummy"):
            client = dashboard.app.test_client()
            response = client.get(
                "/",
                headers={"Authorization": "Basic YXV0b2NpdmlsOmR1bW15"},
            )

        self.assertEqual(response.status_code, 200)
        page = response.get_data(as_text=True)
        self.assertIn("const bestMetricSource =", page)


if __name__ == "__main__":
    unittest.main()
