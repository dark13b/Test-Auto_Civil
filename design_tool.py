"""Legacy adapter for the canonical concrete mix design subsystem."""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Any

import pandas as pd
from artifact_contracts import coalesce_artifact_lineage, evaluate_uncertainty_lineage

from mix_design.constraints import design_constraints_from_legacy
from mix_design.contracts import CandidateScenario, DesignContext, MixDesignRequest, ObjectiveSpec, ScenarioComparisonResult
from mix_design.facade import ConcreteMixDesignService
from train import (
    artifact_id,
    artifact_run_id,
    compute_config_hash,
    create_run_id,
    log_status,
    save_json_artifact,
    write_run_scoped_dataframe,
    write_run_scoped_json_artifact,
)


class MixDesignOptimizer:
    """Deprecated compatibility facade over ``ConcreteMixDesignService``."""

    def __init__(self, model_path: str | Path | None = None, config_path: str | Path | None = None) -> None:
        self.service = ConcreteMixDesignService.from_default_artifacts(
            model_path=model_path,
            config_path=config_path,
        )
        self.project_root = self.service.project_root
        self.config = self.service.config
        self.outputs_dir = self.service.outputs_dir
        self.seed = int(self.config["experiment"]["random_seed"])
        self.base_columns = list(self.service.base_columns)
        self.feature_columns = list(self.service.predictor.feature_columns)
        self.target_column = self.service.target_column
        self.design_config = dict(self.service.design_config)
        self.reference_dataset = self.service.reference_dataset.copy()
        self.model = self.service.predictor.model
        self.model_artifact_id = self.service.model_artifact_id
        self.current_best_payload = dict(self.service.current_best_payload)
        self.current_uncertainty_payload = dict(self.service.current_uncertainty_payload)
        self.active_run_id = str(self.service.active_run_id or create_run_id())
        self.expected_uncertainty_lineage = coalesce_artifact_lineage(
            self.current_best_payload,
            {
                "run_id": self.active_run_id,
                "model_artifact_id": self.model_artifact_id,
                "model_id": str(type(self.model).__name__),
                "model_fingerprint": self.model_artifact_id,
                "config_hash": compute_config_hash(self.config),
            },
            fallback_model_id=str(type(self.model).__name__),
        )
        self.current_uncertainty_lineage = evaluate_uncertainty_lineage(
            self.current_uncertainty_payload,
            expected_lineage=self.expected_uncertainty_lineage,
            fallback_model_id=str(type(self.model).__name__),
        )
        if self.current_uncertainty_payload and self.current_uncertainty_lineage["status"] != "verified":
            log_status(
                "Ignoring canonical uncertainty artifact due to lineage mismatch | "
                f"status={self.current_uncertainty_lineage['status']} | "
                f"run_id={artifact_run_id(self.current_uncertainty_payload) or 'missing'}"
            )
        self.uncertainty_estimator = self.service.predictor.uncertainty_estimator
        self.validator = self.service.validator

    def _legacy_objectives(self) -> tuple[ObjectiveSpec, ...]:
        return (
            ObjectiveSpec(name="target_fit", weight=1.0),
            ObjectiveSpec(name="cost_proxy", weight=1.0),
            ObjectiveSpec(name="co2_proxy", weight=0.35),
            ObjectiveSpec(name="validator_risk", weight=0.8),
        )

    @staticmethod
    def _design_context_from_legacy(context: dict[str, Any] | None) -> DesignContext:
        context = context or {}
        return DesignContext(
            exposure_class=None if context.get("exposure_class") in (None, "") else str(context["exposure_class"]),
            structural_application=(
                None if context.get("structural_application") in (None, "") else str(context["structural_application"])
            ),
        )

    def _build_request(
        self,
        *,
        target_strength_mpa: float,
        constraints: dict[str, Any] | None = None,
        context: dict[str, Any] | None = None,
    ) -> MixDesignRequest:
        return MixDesignRequest(
            target_strength_mpa=float(target_strength_mpa),
            constraints=design_constraints_from_legacy(constraints),
            context=self._design_context_from_legacy(context),
            objectives=self._legacy_objectives(),
            candidate_limit=int(self.design_config.get("top_ranked_candidates", 5)),
        )

    def _superplasticizer_unit_metadata(self) -> dict[str, str]:
        validator_config = self.config.get("validator", {})
        return {
            "assumed_unit": str(validator_config.get("superplasticizer_assumed_unit", "kg_per_m3")),
            "confidence": str(validator_config.get("superplasticizer_unit_confidence", "moderate")),
        }

    def _uncertainty_metadata(self) -> dict[str, Any]:
        official_run_id = artifact_run_id(self.current_uncertainty_payload)
        official_artifact_id = artifact_id(self.current_uncertainty_payload)
        official_model_artifact_id = self.current_uncertainty_payload.get("model_artifact_id")
        official_lineage = getattr(self, "current_uncertainty_lineage", None)
        official_lineage_status = (
            str(official_lineage.get("status"))
            if isinstance(official_lineage, dict) and official_lineage.get("status")
            else "verified"
        )
        official_matches_current_run = (
            official_run_id == self.active_run_id
            and str(official_model_artifact_id or self.model_artifact_id) == str(self.model_artifact_id)
            and official_lineage_status == "verified"
        )
        uncertainty_estimator = self.uncertainty_estimator
        return {
            "method": str(getattr(uncertainty_estimator, "method", "unknown")),
            "coverage_target": float(getattr(uncertainty_estimator, "coverage_level", 0.0)),
            "derived_from_official_estimator": bool(official_matches_current_run),
            "official_calibration_artifact_id": official_artifact_id if official_matches_current_run else None,
            "official_calibration_run_id": official_run_id if official_matches_current_run else None,
            "official_calibration_matches_current_run": bool(official_matches_current_run),
            "official_calibration_lineage_status": official_lineage_status,
            "official_calibration_lineage_mismatches": (
                list(official_lineage.get("mismatches", []))
                if isinstance(official_lineage, dict)
                else []
            ),
            "model_artifact_id": self.model_artifact_id,
            "model_id": str(type(self.model).__name__),
            "model_fingerprint": self.model_artifact_id,
            "config_hash": compute_config_hash(self.config),
            "run_id": self.active_run_id,
        }

    def _design_parent_artifact_ids(self) -> list[str]:
        parent_ids = [artifact_id(self.current_best_payload)]
        uncertainty_metadata = self._uncertainty_metadata()
        if uncertainty_metadata["derived_from_official_estimator"]:
            parent_ids.append(artifact_id(self.current_uncertainty_payload))
        return [str(parent_id) for parent_id in parent_ids if parent_id]

    def _ranking_breakdown(self, scenario: CandidateScenario) -> dict[str, Any]:
        breakdown = {
            component.name: {
                "raw_value": float(component.raw_value),
                "weight": float(component.weight),
                "weighted_score": float(component.weighted_score),
                "explanation": component.explanation,
            }
            for component in scenario.objective_scorecard.components
        }
        breakdown["total_score"] = float(scenario.objective_scorecard.total_score)
        breakdown["rank_explanation"] = list(scenario.objective_scorecard.rank_explanation)
        return breakdown

    def _translate_scenario(
        self,
        scenario: CandidateScenario,
        request: MixDesignRequest,
    ) -> dict[str, Any]:
        design_context = {
            key: value
            for key, value in {
                "exposure_class": request.context.exposure_class,
                "structural_application": request.context.structural_application,
            }.items()
            if value is not None
        }
        uncertainty_interval = {
            "predicted": float(scenario.prediction.uncertainty_interval.predicted),
            "lower_90": float(scenario.prediction.uncertainty_interval.lower_90),
            "upper_90": float(scenario.prediction.uncertainty_interval.upper_90),
            "interval_width": float(scenario.prediction.uncertainty_interval.interval_width),
            "confidence_label": str(scenario.prediction.uncertainty_interval.confidence_label),
            "target_window_overlap": float(scenario.prediction.uncertainty_interval.target_window_overlap),
            "is_calibrated": bool(scenario.prediction.uncertainty_interval.is_calibrated),
            "warning_reasons": list(scenario.prediction.uncertainty_interval.warning_reasons),
        }
        sample_validation = dict(scenario.validator.raw_report)
        deviation = abs(float(scenario.prediction.predicted_strength_mpa) - float(request.target_strength_mpa))
        uncertainty_warnings = list(scenario.prediction.uncertainty_interval.warning_reasons)
        return {
            "objective": float(scenario.objective_scorecard.total_score),
            "success": bool(scenario.success),
            "target_strength": float(request.target_strength_mpa),
            "tolerance_mpa": float(request.constraints.tolerance_mpa or 0.0),
            "predicted_strength": float(scenario.prediction.predicted_strength_mpa),
            "mix_design": {column: float(scenario.mix_design[column]) for column in self.base_columns},
            "engineered_ratios": {
                key: float(value)
                for key, value in scenario.prediction.engineered_features.items()
            },
            "design_context": design_context,
            "sample_validation": sample_validation,
            "validation_verdict": str(scenario.validator.overall_verdict),
            "hard_constraints": list(sample_validation.get("hard_constraints", scenario.validator.hard_constraints)),
            "engineering_cautions": list(sample_validation.get("engineering_cautions", scenario.validator.engineering_cautions)),
            "data_review_flags": list(sample_validation.get("data_review_flags", scenario.validator.data_review_flags)),
            "contextual_summary": str(scenario.validator.contextual_summary),
            "confidence_of_warning_assessment": str(scenario.validator.confidence_of_warning_assessment),
            "validation_warning_reasons": list(scenario.validator.warning_reasons),
            "validation_failure_reasons": list(scenario.validator.failure_reasons),
            "design_constraint_violations": list(scenario.constraints.hard_failures),
            "uncertainty_interval": uncertainty_interval,
            "uncertainty_warnings": uncertainty_warnings,
            "ranking_breakdown": self._ranking_breakdown(scenario),
            "plausibility_penalty": 0.0,
            "deviation_mpa": float(deviation),
            "source": str(scenario.source),
        }

    def _translate_comparison_result(self, result: ScenarioComparisonResult) -> dict[str, Any]:
        ranked_candidates = [self._translate_scenario(scenario, result.request) for scenario in result.scenarios]
        translated = dict(ranked_candidates[0])
        translated["ranked_candidates"] = ranked_candidates
        translated["estimated_cement_saving_vs_reference"] = dict(result.reference_comparison)
        return translated

    def optimize(
        self,
        target_strength_mpa: float,
        constraints: dict[str, Any] | None = None,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Translate legacy optimize calls into the canonical scenario-comparison service."""

        warnings.warn(
            "MixDesignOptimizer.optimize() is deprecated; use ConcreteMixDesignService.compare_scenarios() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        request = self._build_request(
            target_strength_mpa=float(target_strength_mpa),
            constraints=constraints,
            context=context,
        )
        comparison = self.service.compare_scenarios(request)
        return self._translate_comparison_result(comparison)

    def _result_to_batch_record(self, result: dict[str, Any]) -> dict[str, Any]:
        reference = result["estimated_cement_saving_vs_reference"]
        return {
            "target_strength": result["target_strength"],
            "success": result["success"],
            "predicted_strength": result["predicted_strength"],
            "validation_verdict": result["validation_verdict"],
            "cement": result["mix_design"]["cement"],
            "slag": result["mix_design"]["slag"],
            "fly_ash": result["mix_design"]["fly_ash"],
            "water": result["mix_design"]["water"],
            "superplasticizer": result["mix_design"]["superplasticizer"],
            "coarse_aggregate": result["mix_design"]["coarse_aggregate"],
            "fine_aggregate": result["mix_design"]["fine_aggregate"],
            "age": result["mix_design"]["age"],
            "water_cement_ratio": result["engineered_ratios"]["water_cement_ratio"],
            "water_binder_ratio": result["engineered_ratios"]["water_binder_ratio"],
            "total_binder": result["engineered_ratios"]["total_binder"],
            "uncertainty_interval_width": result["uncertainty_interval"]["interval_width"],
            "uncertainty_confidence": result["uncertainty_interval"]["confidence_label"],
            "uncertainty_is_calibrated": result["uncertainty_interval"].get("is_calibrated", False),
            "uncertainty_warning_count": len(result.get("uncertainty_warnings", [])),
            "cement_saving_kg_per_m3": reference["cement_saving_kg_per_m3"],
            "cement_saving_percent": reference["cement_saving_percent"],
        }

    def _build_design_artifact_payload(
        self,
        result: dict[str, Any],
        *,
        source_mode: str,
        batch_summary_row: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload = dict(result)
        payload["source_mode"] = source_mode
        payload["prediction_precision_note"] = (
            "Predicted strength is a model estimate. Use the uncertainty interval, calibration status, and warnings "
            "when comparing candidates; lab validation is required before production use."
        )
        payload["material_units"] = {
            "superplasticizer": self._superplasticizer_unit_metadata(),
        }
        payload["uncertainty_metadata"] = self._uncertainty_metadata()
        if batch_summary_row is not None:
            payload["batch_summary_row"] = batch_summary_row
        return payload

    def _clear_canonical_design_jsons(self, keep_filenames: set[str]) -> None:
        for existing_path in self.outputs_dir.glob("design_*MPa.json"):
            if existing_path.name in keep_filenames:
                continue
            existing_path.unlink(missing_ok=True)

    def batch_optimize(self, target_strengths: list[float]) -> pd.DataFrame:
        return pd.DataFrame([self._result_to_batch_record(self.optimize(target_strength)) for target_strength in target_strengths])

    def export_single_target_artifact(
        self,
        target_strength: float,
        *,
        run_id: str | None = None,
    ) -> tuple[dict[str, Any], Path]:
        current_run_id = str(run_id or self.active_run_id)
        result = self.optimize(target_strength)
        payload = self._build_design_artifact_payload(result, source_mode="design_single")
        filename = f"design_{_target_label(target_strength)}MPa.json"
        canonical_path, _, _ = write_run_scoped_json_artifact(
            outputs_dir=self.outputs_dir,
            filename=filename,
            payload=payload,
            run_id=current_run_id,
            source_mode="design_single",
            config=self.config,
            model_artifact_id=self.model_artifact_id,
            model_id=str(type(self.model).__name__),
            parent_artifact_ids=self._design_parent_artifact_ids(),
        )
        log_status(f"Saved design report to {canonical_path}")
        return payload, canonical_path

    def export_batch_artifacts(
        self,
        target_strengths: list[float],
        *,
        run_id: str | None = None,
    ) -> tuple[pd.DataFrame, list[Path]]:
        current_run_id = str(run_id or self.active_run_id)
        results = [self.optimize(target_strength) for target_strength in target_strengths]
        batch_frame = pd.DataFrame([self._result_to_batch_record(result) for result in results])
        _, _, enriched_frame, batch_metadata = write_run_scoped_dataframe(
            outputs_dir=self.outputs_dir,
            filename="batch_design_results.csv",
            frame=batch_frame,
            run_id=current_run_id,
            source_mode="design_batch",
            config=self.config,
            model_artifact_id=self.model_artifact_id,
            parent_artifact_ids=self._design_parent_artifact_ids(),
        )

        keep_filenames = {f"design_{_target_label(target)}MPa.json" for target in target_strengths}
        self._clear_canonical_design_jsons(keep_filenames)

        exported_paths: list[Path] = []
        for result in results:
            target_strength = float(result["target_strength"])
            row = enriched_frame.loc[enriched_frame["target_strength"] == target_strength].iloc[0].to_dict()
            payload = self._build_design_artifact_payload(
                result,
                source_mode="design_batch",
                batch_summary_row=row,
            )
            filename = f"design_{_target_label(target_strength)}MPa.json"
            canonical_path, _, _ = write_run_scoped_json_artifact(
                outputs_dir=self.outputs_dir,
                filename=filename,
                payload=payload,
                run_id=current_run_id,
                source_mode="design_batch",
                config=self.config,
                model_artifact_id=self.model_artifact_id,
                model_id=str(type(self.model).__name__),
                parent_artifact_ids=[batch_metadata["artifact_id"], *self._design_parent_artifact_ids()],
            )
            exported_paths.append(canonical_path)
        log_status(f"Saved batch design results to {self.outputs_dir / 'batch_design_results.csv'}")
        return enriched_frame, exported_paths

    def save_design_report(self, result: dict[str, Any], output_path: str | Path) -> Path:
        resolved_path = Path(output_path)
        if not resolved_path.is_absolute():
            resolved_path = self.project_root / resolved_path
        resolved_path.parent.mkdir(parents=True, exist_ok=True)
        save_json_artifact(resolved_path, self._build_design_artifact_payload(result, source_mode="design_single"))
        log_status(f"Saved design report to {resolved_path}")
        return resolved_path


def _target_label(target_strength: float) -> str:
    if float(target_strength).is_integer():
        return str(int(target_strength))
    return str(target_strength).replace(".", "_")


def main() -> int:
    """Run the inverse mix-design tool from the command line."""

    parser = argparse.ArgumentParser(description="Inverse concrete mix design tool for AutoCivil-Lab.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--target", type=float, help="Single target compressive strength in MPa.")
    group.add_argument(
        "--batch",
        type=str,
        help="Comma-separated list of target strengths in MPa.",
    )
    args = parser.parse_args()

    try:
        optimizer = MixDesignOptimizer()
        if args.target is not None:
            result, _ = optimizer.export_single_target_artifact(args.target)
            log_status(
                f"Target={result['target_strength']:.2f} MPa | "
                f"Predicted={result['predicted_strength']:.2f} MPa | "
                f"Cement={result['mix_design']['cement']:.2f} kg/m^3 | "
                f"Validation={result['validation_verdict']}"
            )
        else:
            target_strengths = [float(item.strip()) for item in str(args.batch).split(",") if item.strip()]
            optimizer.export_batch_artifacts(target_strengths)
        return 0
    except Exception as exc:
        log_status(f"Design optimization failed: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
