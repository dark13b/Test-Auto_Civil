import importlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
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

    def test_normalize_result_payload_reads_canonical_stage_metrics(self) -> None:
        dashboard = importlib.import_module("dashboard")

        payload = {
            "cross_validation": {
                "stage": "cross_validation",
                "partition": "train",
                "aggregate": {"rmse": 5.0, "mae": 4.0, "r2": 0.61, "composite_score": 0.82},
            },
            "selection_validation": {
                "stage": "selection_validation",
                "partition": "validation",
                "aggregate": {"rmse": 2.0, "mae": 1.0, "r2": 0.7, "composite_score": 0.8},
            },
            "holdout_metrics": {
                "stage": "final_holdout",
                "partition": "holdout",
                "aggregate": {"rmse": 1.0, "mae": 0.5, "r2": 0.8, "composite_score": 0.84},
            },
        }

        normalized = dashboard.normalize_result_payload(payload)

        self.assertEqual(normalized["cv_rmse"], 5.0)
        self.assertEqual(normalized["validation_composite"], 0.8)
        self.assertEqual(normalized["holdout_composite"], 0.84)

    def test_validation_details_exposes_distinct_metric_sources(self) -> None:
        dashboard = importlib.import_module("dashboard")
        with tempfile.TemporaryDirectory() as tmpdir:
            outputs_dir = Path(tmpdir)
            (outputs_dir / "best_search_result.json").write_text(
                json.dumps(
                    {
                        "artifact_kind": "search_selection",
                        "model_name": "LGBMRegressor",
                        "cross_validation": {
                            "stage": "cross_validation",
                            "partition": "train",
                            "aggregate": {
                                "rmse": 5.0,
                                "mae": 4.0,
                                "r2": 0.61,
                                "composite_score": 0.82,
                            },
                        },
                        "selection_validation": {
                            "stage": "selection_validation",
                            "partition": "validation",
                            "aggregate": {
                                "rmse": 2.0,
                                "mae": 1.0,
                                "r2": 0.7,
                                "composite_score": 0.80,
                            },
                        },
                        "selection_validation_report": {
                            "verdict": "PASS",
                            "pass_rate": 1.0,
                            "hard_constraint_count": 0,
                            "engineering_caution_count": 0,
                            "data_review_flag_count": 0,
                        },
                    }
                ),
                encoding="utf-8",
            )
            (outputs_dir / "final_holdout_evaluation.json").write_text(
                json.dumps(
                    {
                        "artifact_kind": "final_holdout_evaluation",
                        "selected_model": {"model_name": "LGBMRegressor"},
                        "holdout_metrics": {
                            "stage": "final_holdout",
                            "partition": "holdout",
                            "aggregate": {
                                "rmse": 1.0,
                                "mae": 0.5,
                                "r2": 0.8,
                                "composite_score": 0.84,
                            },
                        },
                        "holdout_validation_report": {
                            "verdict": "WARN",
                            "pass_rate": 0.92,
                            "hard_constraint_count": 1,
                            "engineering_caution_count": 2,
                            "data_review_flag_count": 3,
                        },
                        "uncertainty_audit": {
                            "artifact_kind": "uncertainty_audit",
                            "audit_partition": "holdout",
                            "calibration_partition": "validation_audit",
                            "coverage_target": 0.9,
                            "coverage": 0.92,
                            "coverage_audit": {"expected_partition": "holdout"},
                        },
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
                response = client.get(
                    "/api/validation_details",
                    headers={"Authorization": "Basic YXV0b2NpdmlsOmR1bW15"},
                )

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["cross_validation"]["source_type"], "cross_validation")
        self.assertEqual(payload["validation_metrics"]["source_type"], "selection_validation")
        self.assertEqual(payload["holdout_metrics"]["source_type"], "holdout_metrics")
        self.assertEqual(payload["cross_validation"]["aggregate"]["composite_score"], 0.82)
        self.assertEqual(payload["validation_metrics"]["aggregate"]["composite_score"], 0.80)
        self.assertEqual(payload["holdout_metrics"]["aggregate"]["composite_score"], 0.84)
        self.assertEqual(payload["holdout_metrics"]["source_label"], "Final holdout metrics")
        self.assertEqual(payload["uncertainty_audit"]["source_type"], "uncertainty_audit")

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
            (outputs_dir / "best_search_result.json").write_text(
                json.dumps(
                    {
                        "artifact_kind": "search_selection",
                        "model_name": "LGBMRegressor",
                        "composite_score": 0.92,
                        "composite_improvement_pct": 1.2,
                    }
                ),
                encoding="utf-8",
            )
            (outputs_dir / "final_holdout_evaluation.json").write_text(
                json.dumps(
                    {
                        "artifact_kind": "final_holdout_evaluation",
                        "selected_model": {"model_name": "LGBMRegressor", "composite_score": 0.92},
                        "holdout_metrics": {
                            "stage": "final_holdout",
                            "partition": "holdout",
                            "aggregate": {"rmse": 4.0, "mae": 3.0, "r2": 0.8, "composite_score": 0.9},
                        },
                        "composite_improvement_pct": 1.2,
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
                "/?mode=experimental",
                headers={"Authorization": "Basic YXV0b2NpdmlsOmR1bW15"},
            )

        self.assertEqual(response.status_code, 200)
        page = response.get_data(as_text=True)
        self.assertIn("Cross-validation metrics", page)
        self.assertIn("Final holdout metrics", page)
        self.assertNotIn("const bestMetricSource =", page)
        self.assertNotIn("best.holdout_composite||best.val_composite", page)

    def test_dashboard_template_renders_source_labeled_panels(self) -> None:
        dashboard = importlib.import_module("dashboard")
        with patch.object(dashboard, "DASHBOARD_PASSWORD", "dummy"):
            client = dashboard.app.test_client()
            response = client.get(
                "/?mode=experimental",
                headers={"Authorization": "Basic YXV0b2NpdmlsOmR1bW15"},
            )

        self.assertEqual(response.status_code, 200)
        page = response.get_data(as_text=True)
        self.assertIn("Cross-validation metrics", page)
        self.assertIn("Validation metrics", page)
        self.assertIn("Final holdout metrics", page)
        self.assertIn("Uncertainty audit", page)
        self.assertIn("Design scenarios / trade-offs", page)
        self.assertNotIn("holdout_composite||best.val_composite", page)
        self.assertNotIn("best.holdout_r2||best.r2", page)

    def test_dashboard_defaults_to_normal_mode(self) -> None:
        dashboard = importlib.import_module("dashboard")
        with patch.object(dashboard, "DASHBOARD_PASSWORD", "dummy"):
            client = dashboard.app.test_client()
            response = client.get(
                "/",
                headers={"Authorization": "Basic YXV0b2NpdmlsOmR1bW15"},
            )

        self.assertEqual(response.status_code, 200)
        page = response.get_data(as_text=True)
        self.assertIn("Normal Mode", page)
        self.assertIn("Experimental Mode", page)
        self.assertIn('data-default-mode="normal"', page)

    def test_normal_mode_renders_project_health_shell(self) -> None:
        dashboard = importlib.import_module("dashboard")
        with patch.object(dashboard, "DASHBOARD_PASSWORD", "dummy"):
            client = dashboard.app.test_client()
            response = client.get(
                "/",
                headers={"Authorization": "Basic YXV0b2NpdmlsOmR1bW15"},
            )

        self.assertEqual(response.status_code, 200)
        page = response.get_data(as_text=True)
        self.assertIn("Project Health", page)
        self.assertIn("Evidence status", page)
        self.assertIn("Next action", page)
        self.assertIn("Open Experimental View", page)

    def test_normal_mode_sidebar_targets_project_health_only(self) -> None:
        dashboard = importlib.import_module("dashboard")
        with patch.object(dashboard, "DASHBOARD_PASSWORD", "dummy"):
            client = dashboard.app.test_client()
            response = client.get(
                "/",
                headers={"Authorization": "Basic YXV0b2NpdmlsOmR1bW15"},
            )

        self.assertEqual(response.status_code, 200)
        page = response.get_data(as_text=True)
        self.assertIn('href="#project-health"', page)
        self.assertNotIn('href="#overview"', page)
        self.assertNotIn('href="#validation"', page)
        self.assertNotIn('href="#design"', page)
        self.assertNotIn('href="#gallery"', page)
        self.assertNotIn('href="#field-results"', page)

    def test_normal_mode_does_not_render_experimental_sections_by_default(self) -> None:
        dashboard = importlib.import_module("dashboard")
        with patch.object(dashboard, "DASHBOARD_PASSWORD", "dummy"):
            client = dashboard.app.test_client()
            response = client.get(
                "/",
                headers={"Authorization": "Basic YXV0b2NpdmlsOmR1bW15"},
            )

        self.assertEqual(response.status_code, 200)
        page = response.get_data(as_text=True)
        self.assertNotIn("Experimental research workspace", page)
        self.assertNotIn("Trial-by-trial search log", page)
        self.assertNotIn("Research Log", page)
        self.assertNotIn("Model Comparison", page)
        self.assertNotIn("Run History", page)
        self.assertNotIn('id="validation"', page)
        self.assertNotIn('id="design"', page)
        self.assertNotIn('id="field-results"', page)
        self.assertNotIn('id="gallery"', page)

    def test_dashboard_renders_experimental_mode_when_requested(self) -> None:
        dashboard = importlib.import_module("dashboard")
        with patch.object(dashboard, "DASHBOARD_PASSWORD", "dummy"):
            client = dashboard.app.test_client()
            response = client.get(
                "/?mode=experimental",
                headers={"Authorization": "Basic YXV0b2NpdmlsOmR1bW15"},
            )

        self.assertEqual(response.status_code, 200)
        page = response.get_data(as_text=True)
        self.assertIn('data-default-mode="experimental"', page)
        self.assertIn("Experimental research workspace", page)
        self.assertIn('id="validation"', page)
        self.assertIn('id="design"', page)
        self.assertIn('id="field-results"', page)
        self.assertIn('id="gallery"', page)

    def test_dashboard_uses_mode_aware_loading(self) -> None:
        page = Path("dashboard.py").read_text(encoding="utf-8")
        self.assertIn("async function loadNormalMode()", page)
        self.assertIn("async function loadExperimentalMode()", page)
        self.assertIn("if (isExperimentalMode()) {", page)
        self.assertIn("Overview artifacts not available.", page)

    def test_api_overview_ignores_stale_final_metrics(self) -> None:
        dashboard = importlib.import_module("dashboard")
        with tempfile.TemporaryDirectory() as tmpdir:
            outputs_dir = Path(tmpdir)
            (outputs_dir / "baseline_metrics.json").write_text(
                json.dumps({"model_name": "RandomForestRegressor", "composite_score": 0.9}),
                encoding="utf-8",
            )
            (outputs_dir / "final_holdout_evaluation.json").write_text(
                json.dumps(
                    {
                        "stale": True,
                        "selected_model": {"model_name": "OldModel", "composite_score": 0.7},
                        "composite_improvement_pct": -3.0,
                    }
                ),
                encoding="utf-8",
            )
            (outputs_dir / "best_search_result.json").write_text(
                json.dumps({"model_name": "FreshModel", "composite_score": 0.95}),
                encoding="utf-8",
            )

            with patch.object(dashboard, "OUTPUTS_DIR", outputs_dir), patch.object(
                dashboard,
                "DASHBOARD_PASSWORD",
                "dummy",
            ):
                client = dashboard.app.test_client()
                response = client.get(
                    "/api/overview",
                    headers={"Authorization": "Basic YXV0b2NpdmlsOmR1bW15"},
                )

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["best"]["model_name"], "FreshModel")
        self.assertIsNone(payload["final"].get("composite_improvement_pct"))

    def test_api_overview_reports_missing_holdout_and_uncertainty_warnings(self) -> None:
        dashboard = importlib.import_module("dashboard")
        with tempfile.TemporaryDirectory() as tmpdir:
            outputs_dir = Path(tmpdir)
            (outputs_dir / "baseline_metrics.json").write_text(
                json.dumps({"model_name": "RandomForestRegressor", "composite_score": 0.9}),
                encoding="utf-8",
            )
            (outputs_dir / "best_search_result.json").write_text(
                json.dumps(
                    {
                        "artifact_kind": "search_selection",
                        "model_name": "FreshModel",
                        "selection_validation": {
                            "stage": "selection_validation",
                            "partition": "validation",
                            "aggregate": {
                                "rmse": 2.0,
                                "mae": 1.0,
                                "r2": 0.7,
                                "composite_score": 0.8,
                            },
                        },
                        "selection_validation_report": {
                            "verdict": "PASS",
                            "pass_rate": 1.0,
                            "hard_constraint_count": 0,
                            "engineering_caution_count": 0,
                            "data_review_flag_count": 0,
                        },
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
                response = client.get(
                    "/api/overview",
                    headers={"Authorization": "Basic YXV0b2NpdmlsOmR1bW15"},
                )

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        warnings = payload.get("warnings") or []
        self.assertIn("Missing artifact", {warning.get("title") for warning in warnings})
        warning_messages = {warning.get("message") for warning in warnings}
        self.assertIn("Final holdout metrics not available.", warning_messages)
        self.assertIn("Uncertainty audit not available.", warning_messages)
        self.assertEqual(payload["decision_metric"]["source_type"], "selection_validation")
        self.assertEqual(payload["decision_metric"]["source_label"], "Validation metrics")
        self.assertIsNone(payload["final"].get("holdout_composite"))

    def test_api_overview_treats_empty_holdout_artifact_as_missing_evidence(self) -> None:
        dashboard = importlib.import_module("dashboard")
        with tempfile.TemporaryDirectory() as tmpdir:
            outputs_dir = Path(tmpdir)
            (outputs_dir / "baseline_metrics.json").write_text(
                json.dumps({"model_name": "RandomForestRegressor", "composite_score": 0.9}),
                encoding="utf-8",
            )
            (outputs_dir / "best_search_result.json").write_text(
                json.dumps(
                    {
                        "artifact_kind": "search_selection",
                        "model_name": "FreshModel",
                    }
                ),
                encoding="utf-8",
            )
            (outputs_dir / "final_holdout_evaluation.json").write_text(
                json.dumps(
                    {
                        "holdout_metrics": {
                            "stage": "holdout_metrics",
                            "partition": "holdout",
                            "aggregate": {},
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
                response = client.get(
                    "/api/overview",
                    headers={"Authorization": "Basic YXV0b2NpdmlsOmR1bW15"},
                )

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        warning_messages = {warning.get("message") for warning in payload.get("warnings") or []}
        self.assertIn("Final holdout metrics not available.", warning_messages)
        self.assertEqual(payload["decision_metric"]["source_type"], None)
        self.assertEqual(payload["evidence_status"]["label"], "Action required")

    def test_experimental_design_view_uses_real_source_labels(self) -> None:
        page = Path("dashboard.py").read_text(encoding="utf-8")
        self.assertIn("Source-separated batch comparison", page)
        self.assertIn("Artifact source", page)
        self.assertIn("Source: ${escapeHtml(s.source_mode || 'design_single')}", page)

    def test_design_generate_flows_exposure_and_structural_context_into_backend(self) -> None:
        dashboard = importlib.import_module("dashboard")
        captured: dict = {}

        class _StubOptimizer:
            def optimize(self, target_strength, constraints=None, context=None):
                captured["target_strength"] = target_strength
                captured["constraints"] = constraints
                captured["context"] = context
                return {
                    "success": True,
                    "target_strength": target_strength,
                    "predicted_strength": 34.8,
                    "validation_verdict": "WARN",
                    "design_context": dict(context or {}),
                    "sample_validation": {
                        "exposure_class": context.get("exposure_class"),
                        "structural_application": context.get("structural_application"),
                        "overall_verdict": "WARN",
                    },
                }

        with patch.object(dashboard, "DASHBOARD_PASSWORD", "dummy"), patch.object(
            dashboard,
            "MixDesignOptimizer",
            return_value=_StubOptimizer(),
            create=True,
        ):
            client = dashboard.app.test_client()
            response = client.post(
                "/api/design_generate",
                headers={"Authorization": "Basic YXV0b2NpdmlsOmR1bW15"},
                json={
                    "target_strength": 35,
                    "exposure_class": "marine",
                    "structural_application": "column",
                },
            )

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(captured["target_strength"], 35.0)
        self.assertEqual(captured["context"]["exposure_class"], "marine")
        self.assertEqual(captured["context"]["structural_application"], "column")
        self.assertEqual(payload["design_context"]["exposure_class"], "marine")
        self.assertEqual(payload["sample_validation"]["structural_application"], "column")


if __name__ == "__main__":
    unittest.main()
