import copy
import difflib
import json
import tempfile
import unittest
from pathlib import Path
from typing import SupportsFloat

import pandas as pd
import pytest

from research_loop import run_engineering_research_loop
from train_impl import load_config


pytestmark = pytest.mark.golden


def _fixture_dir() -> Path:
    return Path(__file__).resolve().parent / "fixtures" / "golden_run"


def _expected_summary_path() -> Path:
    return _fixture_dir() / "expected_summary.json"


def _rounded_metrics(metrics: dict[str, SupportsFloat]) -> dict[str, float]:
    return {key: round(float(value), 6) for key, value in metrics.items()}


def _build_golden_config(outputs_dir: Path) -> dict:
    config = copy.deepcopy(load_config())
    fixture_dir = _fixture_dir()
    outputs_dir.parent.mkdir(parents=True, exist_ok=True)
    research_lab_path = outputs_dir.parent / "golden_research_lab.py"
    research_lab_path.write_text((fixture_dir / "research_lab.py").read_text(encoding="utf-8"), encoding="utf-8")

    config["paths"]["data_dir"] = str(fixture_dir)
    config["paths"]["dataset_filename"] = "concrete_fixture.csv"
    config["paths"]["outputs_dir"] = str(outputs_dir)
    config["data"]["stratify_bins"] = 3
    config["data_split"] = {"train_size": 0.6, "val_size": 0.2, "test_size": 0.2}
    config["experiment"]["cv_folds"] = 3
    config["experiment"]["cv_repeats"] = 1
    config["experiment"]["random_seed"] = 7
    config["engineering"]["feature_engineering"] = True
    config["engineering"]["regime_modeling"]["enabled"] = False
    config["engineering"]["uncertainty_method"] = "conformal"
    config["uncertainty"]["reliability_bins"] = 3
    config["baseline_model"] = {
        "model_name": "Ridge",
        "params": {"alpha": 10.0, "fit_intercept": True, "solver": "auto"},
    }
    for family in list(config["search"]["models"].keys()):
        config["search"]["models"][family]["enabled"] = False
    config["search"]["models"]["RandomForestRegressor"] = {
        "enabled": True,
        "display_name": "RandomForest",
        "search_space": {
            "n_estimators": {"type": "int", "low": 40, "high": 80, "step": 20},
            "max_depth": {"type": "categorical", "choices": [2, 4, 6]},
            "min_samples_leaf": {"type": "int", "low": 1, "high": 2, "step": 1},
            "max_features": {"type": "categorical", "choices": ["sqrt", 1.0]},
            "n_jobs": {"type": "categorical", "choices": [1]},
        },
    }
    config["research"]["max_cycles"] = 1
    config["research"]["scout_candidates_per_cycle"] = 3
    config["research"]["confirm_top_k"] = 1
    config["research"]["scout_cv_repeats"] = 1
    config["research"]["confirm_cv_repeats"] = 1
    config["research"]["rebuild_reports_on_keep"] = False
    config["research"]["brief_path"] = str((fixture_dir / "research_brief.md").relative_to(Path(__file__).resolve().parent.parent))
    config["research"]["editable_surface_path"] = str(research_lab_path)
    config["llm"]["enabled"] = False
    config["llm"]["allow_deterministic_fallback"] = True
    config["llm"]["tasks"]["summary_enabled"] = False
    return config


def _normalize_golden_outputs(outputs_dir: Path) -> dict[str, object]:
    baseline = json.loads((outputs_dir / "baseline_metrics.json").read_text(encoding="utf-8"))
    best = json.loads((outputs_dir / "best_search_result.json").read_text(encoding="utf-8"))
    acceptance = json.loads((outputs_dir / "final_acceptance.json").read_text(encoding="utf-8"))
    validation = json.loads((outputs_dir / "final_artifact_validation.json").read_text(encoding="utf-8"))
    manifest = json.loads((outputs_dir / "run_manifest.json").read_text(encoding="utf-8"))
    uncertainty = json.loads((outputs_dir / "uncertainty_calibration.json").read_text(encoding="utf-8"))
    research_results = pd.read_csv(outputs_dir / "research_results.csv")
    return {
        "baseline_metrics": {
            "model_name": baseline["model_name"],
            "composite_score": round(float(baseline["composite_score"]), 6),
            "validation_verdict": baseline["validation_verdict"],
            "cross_validation": _rounded_metrics(baseline["cross_validation"]["aggregate"]),
            "selection_validation": _rounded_metrics(baseline["selection_validation"]["aggregate"]),
        },
        "best_search_result": {
            "model_name": best["model_name"],
            "composite_score": round(float(best["composite_score"]), 6),
            "validation_verdict": best["validation_verdict"],
            "cross_validation": _rounded_metrics(best["cross_validation"]["aggregate"]),
            "selection_validation": _rounded_metrics(best["selection_validation"]["aggregate"]),
        },
        "final_acceptance": {
            "accepted": acceptance["accepted"],
            "acceptance_metric": acceptance["acceptance_metric"],
            "best_model_name": acceptance["best_model_name"],
            "decision_reason": acceptance["decision_reason"],
            "source_of_truth": acceptance["source_of_truth"],
        },
        "final_artifact_validation": {
            "consistent": validation["consistent"],
            "source_of_truth": validation["source_of_truth"],
            "mismatch_count": len(validation["mismatches"]),
        },
        "research_results": research_results[
            ["experiment_id", "stage", "model_name", "selection_status", "proposal_family"]
        ].to_dict(orient="records"),
        "run_manifest": {
            "backend": manifest["backend"],
            "model": manifest["model"],
            "final_run_status": manifest["final_run_status"],
            "number_of_candidates_evaluated": manifest["number_of_candidates_evaluated"],
            "preflight_status": manifest["preflight_status"],
            "smoke_test_status": manifest["smoke_test_status"],
        },
        "uncertainty_calibration": {
            "audit_partition": uncertainty["audit_partition"],
            "coverage": round(float(uncertainty["coverage"]), 6),
            "global_status": uncertainty["coverage_audit"]["global_status"],
            "mean_interval_width": round(float(uncertainty["mean_interval_width"]), 6),
        },
    }


def _summary_diff(expected: dict[str, object], actual: dict[str, object]) -> str:
    expected_text = json.dumps(expected, indent=2, sort_keys=True).splitlines()
    actual_text = json.dumps(actual, indent=2, sort_keys=True).splitlines()
    return "\n".join(
        difflib.unified_diff(
            expected_text,
            actual_text,
            fromfile="expected_summary.json",
            tofile="actual_summary.json",
            lineterm="",
        )
    )


class GoldenRunTests(unittest.TestCase):
    def test_golden_run_writes_artifacts_to_its_temp_output_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            outputs_dir = Path(tmpdir) / "golden_outputs"
            config = _build_golden_config(outputs_dir)

            result = run_engineering_research_loop(
                cycles_override=1,
                with_report=False,
                config_override=config,
            )

            self.assertEqual(result["artifact_kind"], "search_selection")
            self.assertTrue((outputs_dir / "baseline_metrics.json").exists())
            self.assertTrue((outputs_dir / "best_search_result.json").exists())
            self.assertTrue((outputs_dir / "final_acceptance.json").exists())
            self.assertTrue((outputs_dir / "final_artifact_validation.json").exists())
            self.assertTrue((outputs_dir / "uncertainty_calibration.json").exists())
            actual_summary = _normalize_golden_outputs(outputs_dir)
            expected_summary = json.loads(_expected_summary_path().read_text(encoding="utf-8"))
            if actual_summary != expected_summary:
                self.fail("Golden run summary drifted:\n" + _summary_diff(expected_summary, actual_summary))


if __name__ == "__main__":
    unittest.main()
