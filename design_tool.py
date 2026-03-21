"""Inverse concrete mix design utilities for AutoCivil-Lab."""

from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import pandas as pd

from feature_engineering import build_engineering_features
from train import (
    artifact_id,
    artifact_run_id,
    compute_file_hash,
    create_run_id,
    get_base_input_columns,
    get_outputs_dir,
    get_project_root,
    get_target_column,
    infer_active_run_id,
    load_config,
    load_dataset,
    load_pickle_artifact,
    log_status,
    read_pickle_artifact_metadata,
    resolve_model_feature_columns,
    save_json_artifact,
    set_global_seed,
    write_run_scoped_dataframe,
    write_run_scoped_json_artifact,
)
from uncertainty import UncertaintyEstimator
from validator import EngineeringValidator


class MixDesignOptimizer:
    """Search for low-cement mix designs that meet a target compressive strength."""

    def __init__(self, model_path: str | Path | None = None, config_path: str | Path | None = None) -> None:
        """Load the trained model, configuration, validator, and reference dataset."""
        self.project_root = get_project_root()
        self.config = self._load_config(config_path)
        self.outputs_dir = get_outputs_dir(self.config)
        self.model_path = self._resolve_model_path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Best search model not found at {self.model_path}. Run search.py before design_tool.py."
            )

        self.seed = int(self.config["experiment"]["random_seed"])
        set_global_seed(self.seed)
        self.base_columns = get_base_input_columns(self.config)
        self.target_column = get_target_column(self.config)
        self.design_config = self.config["engineering"]["design_tool"]
        self.validator = EngineeringValidator.from_config(self.config)
        self.model = load_pickle_artifact(self.model_path)
        self.model_metadata = read_pickle_artifact_metadata(self.model_path) or {}
        self.model_artifact_id = str(
            self.model_metadata.get("artifact_id") or compute_file_hash(self.model_path) or "unknown-model-artifact"
        )
        self.feature_columns = resolve_model_feature_columns(self.model, self.config)
        dataset = load_dataset(self.config)
        self.reference_dataset = dataset[self.base_columns + [self.target_column]].copy()
        self.current_best_payload = self._load_json_payload(self.outputs_dir / "best_search_result.json")
        self.current_uncertainty_payload = self._load_json_payload(self.outputs_dir / "uncertainty_calibration.json")
        self.active_run_id = infer_active_run_id(
            self.outputs_dir,
            self.current_best_payload,
            self.current_uncertainty_payload,
            fallback=create_run_id(),
        )
        self.uncertainty_estimator = UncertaintyEstimator(
            model=self.model,
            report_model=self.model,
            config_path=config_path,
            outputs_dir=self.outputs_dir,
            report_filename="design_uncertainty_calibration.json",
        )

    @staticmethod
    def _load_json_payload(path: Path) -> dict[str, Any]:
        """Load a JSON payload when the file exists, otherwise return an empty dictionary."""
        if not path.exists():
            return {}
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return payload if isinstance(payload, dict) else {}

    def _load_config(self, config_path: str | Path | None) -> dict[str, Any]:
        """Load the project configuration from disk."""
        if config_path is None:
            return load_config()

        resolved_path = Path(config_path)
        if not resolved_path.is_absolute():
            resolved_path = self.project_root / resolved_path
        if not resolved_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {resolved_path}")
        if resolved_path == self.project_root / "config.yaml":
            return load_config()

        import yaml

        with resolved_path.open("r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
        if not isinstance(config, dict):
            raise ValueError("Configuration file did not parse into a dictionary.")
        return config

    def _resolve_model_path(self, model_path: str | Path | None) -> Path:
        """Resolve model path relative to the project root."""
        if model_path is None:
            return self.project_root / self.config["paths"]["outputs_dir"] / "best_search_model.pkl"
        resolved_path = Path(model_path)
        if not resolved_path.is_absolute():
            resolved_path = self.project_root / resolved_path
        return resolved_path

    def _normalize_constraints(self, constraints: dict[str, Any] | None) -> dict[str, Any]:
        """Merge user constraints into the config-defined defaults."""
        merged = deepcopy(self.design_config["constraints"])
        if not constraints:
            return merged

        for key, value in constraints.items():
            if key == "tolerance_mpa":
                merged[key] = float(value)
                continue
            if isinstance(value, dict):
                existing = merged.get(key, {})
                existing.update(value)
                merged[key] = existing
            else:
                merged[key] = {"fixed": float(value)}
        return merged

    def _apply_target_dependent_bounds(
        self,
        target_strength: float,
        constraints: dict[str, Any],
    ) -> dict[str, Any]:
        """Overlay target-dependent cement, water, and water/cement bounds."""
        bounded = deepcopy(constraints)
        if target_strength < 30.0:
            cement_bounds = (100.0, 260.0)
            water_bounds = (150.0, 210.0)
            water_cement_max = 1.45
        elif target_strength <= 35.0:
            cement_bounds = (110.0, 280.0)
            water_bounds = (145.0, 210.0)
            water_cement_max = 1.25
        elif target_strength <= 45.0:
            cement_bounds = (140.0, 300.0)
            water_bounds = (140.0, 210.0)
            water_cement_max = 0.55
        else:
            cement_bounds = (280.0, 500.0)
            water_bounds = (150.0, 200.0)
            water_cement_max = 0.45

        bounded.setdefault("cement", {})
        bounded.setdefault("water", {})
        bounded.setdefault("water_cement_ratio", {})
        bounded["cement"]["min"] = max(float(bounded["cement"].get("min", cement_bounds[0])), cement_bounds[0])
        bounded["cement"]["max"] = min(float(bounded["cement"].get("max", cement_bounds[1])), cement_bounds[1])
        bounded["water"]["min"] = max(float(bounded["water"].get("min", water_bounds[0])), water_bounds[0])
        bounded["water"]["max"] = min(float(bounded["water"].get("max", water_bounds[1])), water_bounds[1])
        existing_water_cement_max = float(
            bounded["water_cement_ratio"].get("max", water_cement_max)
        )
        bounded["water_cement_ratio"]["max"] = (
            max(existing_water_cement_max, water_cement_max)
            if target_strength <= 35.0
            else min(existing_water_cement_max, water_cement_max)
        )
        if float(bounded["cement"]["min"]) > float(bounded["cement"]["max"]):
            raise ValueError("Target-dependent cement bounds conflict with the configured constraints.")
        if float(bounded["water"]["min"]) > float(bounded["water"]["max"]):
            raise ValueError("Target-dependent water bounds conflict with the configured constraints.")
        return bounded

    def _target_tolerance(self, constraints: dict[str, Any]) -> float:
        """Resolve the design tolerance."""
        if "tolerance_mpa" in constraints:
            return float(constraints["tolerance_mpa"])
        return float(self.design_config["target_tolerance_mpa"])

    def _target_regime(self, target_strength: float) -> str:
        """Classify the target strength into a design regime."""
        if target_strength <= 35.0:
            return "low_strength"
        if target_strength <= 50.0:
            return "medium_strength"
        return "high_strength"

    def _cost_value(self, mix_design: dict[str, float]) -> float:
        """Return the configured cost proxy for a candidate mix."""
        cost_proxy = str(self.design_config["cost_proxy"]).strip().lower()
        if cost_proxy == "cement_content":
            return float(mix_design["cement"])
        if cost_proxy in mix_design:
            return float(mix_design[cost_proxy])
        raise ValueError(f"Unsupported design-tool cost proxy: {self.design_config['cost_proxy']}")

    def _check_design_constraints(
        self,
        feature_row: pd.Series,
        constraints: dict[str, Any],
    ) -> list[str]:
        """Evaluate design-tool-specific ratio constraints."""
        violations: list[str] = []
        for column in [
            "water_cement_ratio",
            "fly_ash_replacement_ratio",
            "slag_replacement_ratio",
        ]:
            column_constraints = constraints.get(column, {})
            if "max" in column_constraints and float(feature_row[column]) > float(column_constraints["max"]):
                violations.append(f"{column} exceeds configured maximum.")
            if "min" in column_constraints and float(feature_row[column]) < float(column_constraints["min"]):
                violations.append(f"{column} falls below configured minimum.")
        return violations

    def _normalize_design_context(self, context: dict[str, Any] | None) -> dict[str, Any]:
        """Normalize optional UI/CLI design context."""
        if not isinstance(context, dict):
            return {}
        normalized: dict[str, Any] = {}
        for key in ("exposure_class", "structural_application"):
            value = context.get(key)
            if value in (None, ""):
                continue
            normalized[key] = str(value).strip().lower().replace(" ", "_")
        return normalized

    def _frame_from_mix(
        self,
        mix_design: dict[str, float],
        *,
        context: dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        """Convert a mix-design dictionary into a one-row dataframe."""
        normalized_mix = {column: float(mix_design[column]) for column in self.base_columns}
        for column in ("slag", "fly_ash"):
            if abs(normalized_mix[column]) < 1.0:
                normalized_mix[column] = 0.0
        normalized_mix.update(self._normalize_design_context(context))
        return pd.DataFrame([normalized_mix])

    def _target_conditioned_reference_rows(self, target_strength: float) -> pd.DataFrame:
        """Return a dataset subset centered on the requested target strength."""
        candidate_rows = self.reference_dataset.copy()
        windows = [2.0, 4.0, 6.0, 8.0]
        filtered = candidate_rows.iloc[0:0].copy()
        for window in windows:
            filtered = candidate_rows[
                candidate_rows[self.target_column].between(target_strength - window, target_strength + window)
            ].copy()
            if len(filtered) >= 8:
                break
        if filtered.empty:
            target_regime = self._target_regime(target_strength)
            if target_regime == "low_strength":
                filtered = candidate_rows[candidate_rows[self.target_column] <= 35.0].copy()
            elif target_regime == "medium_strength":
                filtered = candidate_rows[
                    candidate_rows[self.target_column].between(30.0, 50.0)
                ].copy()
            else:
                filtered = candidate_rows[candidate_rows[self.target_column] >= 45.0].copy()
        return filtered if not filtered.empty else candidate_rows

    def _engineering_prior_mixes(
        self,
        target_strength: float,
        constraints: dict[str, Any],
    ) -> list[dict[str, float]]:
        """Build dataset-conditioned engineering priors for the target regime."""
        candidate_rows = self._target_conditioned_reference_rows(target_strength).copy()
        if candidate_rows.empty:
            return []

        target_regime = self._target_regime(target_strength)
        sort_columns = ["cement", self.target_column]
        if target_regime == "low_strength":
            sort_columns = ["cement", self.target_column]
        elif target_regime == "medium_strength":
            sort_columns = [self.target_column, "cement"]
        else:
            sort_columns = [self.target_column, "cement"]
        candidate_rows["target_gap"] = (candidate_rows[self.target_column] - target_strength).abs()
        candidate_rows = candidate_rows.sort_values(["target_gap", *sort_columns]).head(24)

        priors: list[dict[str, float]] = []
        seen_signatures: set[tuple[float, ...]] = set()
        water_cement_max = float(constraints.get("water_cement_ratio", {}).get("max", np.inf))
        water_constraints = constraints.get("water", {})
        water_minimum = float(water_constraints.get("fixed", water_constraints.get("min", 0.0)))
        for _, row in candidate_rows.iterrows():
            prior = {column: float(row[column]) for column in self.base_columns}
            for column in self.base_columns:
                column_constraints = constraints.get(column, {})
                if "fixed" in column_constraints:
                    prior[column] = float(column_constraints["fixed"])
                else:
                    prior[column] = float(
                        np.clip(prior[column], float(column_constraints["min"]), float(column_constraints["max"]))
                    )
            if np.isfinite(water_cement_max) and water_cement_max > 0.0:
                prior["cement"] = max(prior["cement"], water_minimum / water_cement_max)
                cement_constraints = constraints.get("cement", {})
                if "max" in cement_constraints:
                    prior["cement"] = min(prior["cement"], float(cement_constraints["max"]))
                prior["water"] = min(prior["water"], prior["cement"] * water_cement_max)
                prior["water"] = max(prior["water"], water_minimum)
            signature = tuple(round(prior[column], 4) for column in self.base_columns)
            if signature in seen_signatures:
                continue
            seen_signatures.add(signature)
            priors.append(prior)
            if len(priors) >= 6:
                break
        return priors

    def _uncertainty_interval_summary(
        self,
        candidate_frame: pd.DataFrame,
        target_strength: float,
        tolerance: float,
    ) -> dict[str, Any]:
        """Predict a candidate interval and summarize target-window overlap."""
        if self.uncertainty_estimator is None:
            predicted = float(self.model.predict(build_engineering_features(candidate_frame, config=self.config)[self.feature_columns])[0])
            lower = predicted - tolerance
            upper = predicted + tolerance
            width = upper - lower
            confidence_label = "UNKNOWN"
        else:
            interval_frame = self.uncertainty_estimator.predict_with_interval(candidate_frame)
            lower = float(interval_frame.iloc[0]["lower_90"])
            upper = float(interval_frame.iloc[0]["upper_90"])
            predicted = float(interval_frame.iloc[0]["predicted"])
            width = float(interval_frame.iloc[0]["interval_width"])
            confidence_label = str(interval_frame.iloc[0].get("confidence_label", "UNKNOWN"))

        target_lower = target_strength - tolerance
        target_upper = target_strength + tolerance
        overlap = max(0.0, min(upper, target_upper) - max(lower, target_lower))
        target_window_width = max(target_upper - target_lower, 1e-6)
        return {
            "predicted": predicted,
            "lower_90": lower,
            "upper_90": upper,
            "interval_width": width,
            "confidence_label": confidence_label,
            "target_window_overlap": overlap / target_window_width,
        }

    def _superplasticizer_unit_metadata(self) -> dict[str, str]:
        """Return the configured superplasticizer unit assumption used in artifacts."""
        validator_config = self.config.get("validator", {})
        return {
            "assumed_unit": str(validator_config.get("superplasticizer_assumed_unit", "kg_per_m3")),
            "confidence": str(validator_config.get("superplasticizer_unit_confidence", "moderate")),
        }

    def _uncertainty_metadata(self) -> dict[str, Any]:
        """Describe the uncertainty lineage used for design ranking and export."""
        official_run_id = artifact_run_id(self.current_uncertainty_payload)
        official_artifact_id = artifact_id(self.current_uncertainty_payload)
        official_model_artifact_id = self.current_uncertainty_payload.get("model_artifact_id")
        official_matches_current_run = (
            official_run_id == self.active_run_id
            and str(official_model_artifact_id or self.model_artifact_id) == str(self.model_artifact_id)
        )
        return {
            "method": str(self.uncertainty_estimator.method),
            "coverage_target": float(self.uncertainty_estimator.coverage_level),
            "derived_from_official_estimator": True,
            "official_calibration_artifact_id": official_artifact_id,
            "official_calibration_run_id": official_run_id,
            "official_calibration_matches_current_run": bool(official_matches_current_run),
            "model_artifact_id": self.model_artifact_id,
        }

    def _design_parent_artifact_ids(self) -> list[str]:
        """Return lineage parents for design artifacts."""
        parent_ids = [artifact_id(self.current_best_payload), artifact_id(self.current_uncertainty_payload)]
        return [str(parent_id) for parent_id in parent_ids if parent_id]

    def _result_to_batch_record(self, result: dict[str, Any]) -> dict[str, Any]:
        """Flatten one optimized result into the canonical batch CSV schema."""
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
        """Attach shared design lineage, uncertainty, and units metadata to one result."""
        payload = dict(result)
        payload["source_mode"] = source_mode
        payload["material_units"] = {
            "superplasticizer": self._superplasticizer_unit_metadata(),
        }
        payload["uncertainty_metadata"] = self._uncertainty_metadata()
        if batch_summary_row is not None:
            payload["batch_summary_row"] = batch_summary_row
        return payload

    def _clear_canonical_design_jsons(self, keep_filenames: set[str]) -> None:
        """Delete stale canonical design JSONs before writing the new batch export set."""
        for existing_path in self.outputs_dir.glob("design_*MPa.json"):
            if existing_path.name in keep_filenames:
                continue
            existing_path.unlink(missing_ok=True)

    def _plausibility_penalty(
        self,
        engineered_row: pd.Series,
        target_strength: float,
        mix_design: dict[str, float],
    ) -> float:
        """Penalize candidates that drift far from dataset-conditioned engineering ranges."""
        reference_rows = self._target_conditioned_reference_rows(target_strength)
        reference_engineered = build_engineering_features(reference_rows[self.base_columns], config=self.config)
        comparison_columns = [
            "cement",
            "total_binder",
            "effective_binder",
            "water_effective_binder_ratio",
            "supplementary_replacement_ratio",
        ]
        penalties: list[float] = []
        for column in comparison_columns:
            if column in mix_design:
                candidate_value = float(mix_design[column])
            else:
                candidate_value = float(engineered_row[column])
            series = (
                reference_rows[column].astype(float)
                if column in reference_rows.columns
                else reference_engineered[column].astype(float)
            )
            mean = float(series.mean())
            std = float(series.std(ddof=0))
            if std <= 1e-6:
                penalties.append(0.0)
                continue
            penalties.append(abs(candidate_value - mean) / std)
        return float(np.mean(penalties) * 15.0)

    def _sample_trial_mix(
        self,
        trial: optuna.trial.Trial,
        constraints: dict[str, Any],
    ) -> dict[str, float]:
        """Sample a mix design for one Optuna trial."""
        mix_design: dict[str, float] = {}
        water_cement_max = float(constraints.get("water_cement_ratio", {}).get("max", np.inf))
        water_constraints = constraints.get("water", {})
        water_minimum = float(water_constraints.get("fixed", water_constraints.get("min", 0.0)))
        for column in self.base_columns:
            column_constraints = constraints.get(column, {})
            if "fixed" in column_constraints:
                mix_design[column] = float(column_constraints["fixed"])
                continue
            minimum = float(column_constraints["min"])
            maximum = float(column_constraints["max"])
            if column == "cement" and np.isfinite(water_cement_max) and water_cement_max > 0.0:
                minimum = max(minimum, water_minimum / water_cement_max)
            if column == "water" and np.isfinite(water_cement_max) and water_cement_max > 0.0:
                maximum = min(maximum, float(mix_design["cement"]) * water_cement_max)
            if minimum > maximum:
                raise ValueError(f"Constraint range is invalid for '{column}': min={minimum}, max={maximum}.")
            mix_design[column] = float(trial.suggest_float(column, minimum, maximum))
        return mix_design

    def _evaluate_mix(
        self,
        mix_design: dict[str, float],
        target_strength: float,
        tolerance: float,
        constraints: dict[str, Any],
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Predict and score a candidate mix design."""
        design_context = self._normalize_design_context(context)
        candidate_frame = self._frame_from_mix(mix_design, context=design_context)
        engineered = build_engineering_features(candidate_frame, config=self.config)
        predicted_strength = float(self.model.predict(engineered[self.feature_columns])[0])
        validation_frame = candidate_frame.copy()
        validation_frame["target_strength"] = float(target_strength)
        sample_report = self.validator.evaluate_samples(
            np.asarray([predicted_strength], dtype=float),
            validation_frame,
        )[0]
        engineered_row = engineered.iloc[0]
        design_constraint_violations = self._check_design_constraints(engineered_row, constraints)
        uncertainty_interval = self._uncertainty_interval_summary(candidate_frame, target_strength, tolerance)
        plausibility_penalty = self._plausibility_penalty(engineered_row, target_strength, mix_design)
        target_regime = self._target_regime(target_strength)
        if target_regime == "low_strength":
            progressive_penalty = max(0.0, mix_design["cement"] - target_strength * 7.0) ** 2 * 0.05
        else:
            progressive_penalty = max(0.0, mix_design["cement"] - target_strength * 8.0) ** 2 * 0.01
        over_strength_penalty = 0.0
        if predicted_strength > target_strength * 1.10:
            over_strength_penalty = (predicted_strength - target_strength) * 0.5
        warning_penalty = float(sample_report["warning_count"]) * 4000.0
        interval_penalty = 0.0
        if float(uncertainty_interval["target_window_overlap"]) <= 0.0:
            interval_penalty += float(uncertainty_interval["interval_width"]) * 150.0

        deviation = abs(predicted_strength - target_strength)
        objective = (
            self._cost_value(mix_design)
            + deviation * 12.0
            + progressive_penalty
            + over_strength_penalty
            + warning_penalty
            + plausibility_penalty
            + interval_penalty
        )
        if predicted_strength < target_strength - tolerance:
            objective += ((target_strength - tolerance) - predicted_strength) ** 2 * 1500.0
        if predicted_strength > target_strength + tolerance:
            objective += (predicted_strength - (target_strength + tolerance)) ** 2 * 1200.0
        if design_constraint_violations:
            objective += 100000.0 * len(design_constraint_violations)
        if sample_report["overall_verdict"] == "FAIL":
            objective += 1000000.0

        feasible = (
            sample_report["overall_verdict"] != "FAIL"
            and not design_constraint_violations
            and deviation <= tolerance
            and float(uncertainty_interval["target_window_overlap"]) > 0.0
        )
        return {
            "objective": float(objective),
            "success": feasible,
            "target_strength": float(target_strength),
            "tolerance_mpa": float(tolerance),
            "predicted_strength": predicted_strength,
            "mix_design": {column: float(mix_design[column]) for column in self.base_columns},
            "engineered_ratios": {
                column: float(engineered_row[column])
                for column in engineered.columns
                if column not in self.base_columns and np.issubdtype(type(engineered_row[column]), np.number)
            },
            "design_context": design_context,
            "sample_validation": dict(sample_report),
            "validation_verdict": sample_report["overall_verdict"],
            "hard_constraints": list(sample_report["hard_constraints"]),
            "engineering_cautions": list(sample_report["engineering_cautions"]),
            "data_review_flags": list(sample_report["data_review_flags"]),
            "contextual_summary": str(sample_report["contextual_summary"]),
            "confidence_of_warning_assessment": str(sample_report["confidence_of_warning_assessment"]),
            "validation_warning_reasons": list(sample_report["warning_reasons"]),
            "validation_failure_reasons": list(sample_report["failure_reasons"]),
            "design_constraint_violations": design_constraint_violations,
            "uncertainty_interval": uncertainty_interval,
            "plausibility_penalty": float(plausibility_penalty),
            "ranking_breakdown": {
                "cost_value": float(self._cost_value(mix_design)),
                "deviation_penalty": float(deviation * 12.0),
                "progressive_cement_penalty": float(progressive_penalty),
                "over_strength_penalty": float(over_strength_penalty),
                "warning_penalty": float(warning_penalty),
                "interval_penalty": float(interval_penalty),
                "plausibility_penalty": float(plausibility_penalty),
            },
            "progressive_cement_penalty": float(progressive_penalty),
            "over_strength_penalty": float(over_strength_penalty),
            "warning_penalty": float(warning_penalty),
            "deviation_mpa": float(deviation),
        }

    def _ranking_key(self, candidate: dict[str, Any]) -> tuple[Any, ...]:
        """Return a deterministic ranking key for candidate selection."""
        verdict_rank = {
            "PASS": 0,
            "WARN": 1,
            "FAIL": 2,
        }.get(str(candidate["validation_verdict"]), 3)
        return (
            0 if candidate["success"] else 1,
            verdict_rank,
            float(candidate["objective"]),
            float(candidate["deviation_mpa"]),
            self._cost_value(candidate["mix_design"]),
        )

    def _warm_start_mixes(self, target_strength: float, constraints: dict[str, Any]) -> list[dict[str, float]]:
        """Select known low-strength reference mixes for Optuna warm starts."""
        if target_strength > 30.0:
            return []

        candidate_rows = self.reference_dataset[self.reference_dataset[self.target_column] <= 30.0].copy()
        if candidate_rows.empty:
            return []

        candidate_rows["target_gap"] = (candidate_rows[self.target_column] - target_strength).abs()
        candidate_rows = candidate_rows.sort_values(["target_gap", "cement", self.target_column]).head(30)

        warm_starts: list[dict[str, float]] = []
        seen_signatures: set[tuple[float, ...]] = set()
        water_cement_max = float(constraints.get("water_cement_ratio", {}).get("max", np.inf))
        water_constraints = constraints.get("water", {})
        water_minimum = float(water_constraints.get("fixed", water_constraints.get("min", 0.0)))
        for _, row in candidate_rows.iterrows():
            mix_design = {column: float(row[column]) for column in self.base_columns}
            for column in self.base_columns:
                column_constraints = constraints.get(column, {})
                if "fixed" in column_constraints:
                    mix_design[column] = float(column_constraints["fixed"])
                else:
                    minimum = float(column_constraints["min"])
                    maximum = float(column_constraints["max"])
                    mix_design[column] = float(np.clip(mix_design[column], minimum, maximum))
            if np.isfinite(water_cement_max) and water_cement_max > 0.0:
                mix_design["cement"] = max(mix_design["cement"], water_minimum / water_cement_max)
                cement_constraints = constraints.get("cement", {})
                if "max" in cement_constraints:
                    mix_design["cement"] = min(mix_design["cement"], float(cement_constraints["max"]))
                if "water" in mix_design:
                    mix_design["water"] = min(mix_design["water"], mix_design["cement"] * water_cement_max)
                    mix_design["water"] = max(mix_design["water"], water_minimum)
            signature = tuple(round(mix_design[column], 4) for column in self.base_columns)
            if signature in seen_signatures:
                continue
            seen_signatures.add(signature)
            warm_starts.append(mix_design)
            if len(warm_starts) == 3:
                break
        return warm_starts

    def _estimate_reference_mix(
        self,
        result: dict[str, Any],
        constraints: dict[str, Any],
    ) -> dict[str, Any]:
        """Estimate cement demand for a pure-cement reference mix."""
        reference_ratio = float(self.design_config["reference_mix"]["water_cement_ratio"])
        grid_points = int(self.design_config["search"]["reference_grid_points"])
        cement_constraints = constraints["cement"]
        water_constraints = constraints["water"]
        cement_values = np.linspace(
            float(cement_constraints["min"]),
            float(cement_constraints["max"]),
            grid_points,
        )

        reference_rows: list[dict[str, float]] = []
        optimized_mix = result["mix_design"]
        for cement in cement_values:
            water = cement * reference_ratio
            if water < float(water_constraints["min"]) or water > float(water_constraints["max"]):
                continue
            reference_rows.append(
                {
                    "cement": float(cement),
                    "slag": 0.0,
                    "fly_ash": 0.0,
                    "water": float(water),
                    "superplasticizer": float(optimized_mix["superplasticizer"]),
                    "coarse_aggregate": float(optimized_mix["coarse_aggregate"]),
                    "fine_aggregate": float(optimized_mix["fine_aggregate"]),
                    "age": float(optimized_mix["age"]),
                }
            )

        if not reference_rows:
            return {
                "reference_cement_content": None,
                "reference_predicted_strength": None,
                "cement_saving_kg_per_m3": None,
                "cement_saving_percent": None,
            }

        reference_frame = build_engineering_features(pd.DataFrame(reference_rows), config=self.config)
        reference_predictions = np.asarray(
            self.model.predict(reference_frame[self.feature_columns]),
            dtype=float,
        )
        tolerance = float(result["tolerance_mpa"])
        target_strength = float(result["target_strength"])
        selected_index = None
        for index, prediction in enumerate(reference_predictions):
            if prediction >= target_strength - tolerance:
                selected_index = index
                break
        if selected_index is None:
            selected_index = int(np.argmin(np.abs(reference_predictions - target_strength)))

        reference_mix = reference_rows[selected_index]
        reference_cement = float(reference_mix["cement"])
        cement_saving = reference_cement - float(optimized_mix["cement"])
        cement_saving_percent = None
        if abs(reference_cement) > 1e-8:
            cement_saving_percent = (cement_saving / reference_cement) * 100.0
        return {
            "reference_cement_content": reference_cement,
            "reference_predicted_strength": float(reference_predictions[selected_index]),
            "cement_saving_kg_per_m3": float(cement_saving),
            "cement_saving_percent": None if cement_saving_percent is None else float(cement_saving_percent),
        }

    def optimize(
        self,
        target_strength_mpa: float,
        constraints: dict[str, Any] | None = None,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Search for a mix design that meets the target strength with minimum cement."""
        target_strength = float(target_strength_mpa)
        design_context = self._normalize_design_context(context)
        merged_constraints = self._normalize_constraints(constraints)
        merged_constraints = self._apply_target_dependent_bounds(target_strength, merged_constraints)
        tolerance = self._target_tolerance(merged_constraints)
        if "age" not in merged_constraints:
            merged_constraints["age"] = {"fixed": float(self.design_config["default_age_days"])}
        elif "fixed" not in merged_constraints["age"]:
            merged_constraints["age"]["fixed"] = float(self.design_config["default_age_days"])

        initial_samples = int(self.design_config["search"]["initial_samples"])
        default_budget = max(300, initial_samples // 8)
        if target_strength < 30.0:
            default_budget = max(default_budget, 420)
        trial_budget = int(self.design_config["search"].get("optuna_trials", default_budget))
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        sampler = optuna.samplers.TPESampler(seed=self.seed, warn_independent_sampling=False)
        study = optuna.create_study(direction="minimize", sampler=sampler)
        evaluated_candidates: list[dict[str, Any]] = []

        for prior_mix in self._engineering_prior_mixes(target_strength, merged_constraints):
            evaluated_candidates.append(
                self._evaluate_mix(
                    prior_mix,
                    target_strength,
                    tolerance,
                    merged_constraints,
                    context=design_context,
                )
            )
        for warm_start_mix in self._warm_start_mixes(target_strength, merged_constraints):
            enqueued_params = {
                column: value
                for column, value in warm_start_mix.items()
                if "fixed" not in merged_constraints.get(column, {})
            }
            study.enqueue_trial(enqueued_params)
        for prior_mix in self._engineering_prior_mixes(target_strength, merged_constraints):
            enqueued_params = {
                column: value
                for column, value in prior_mix.items()
                if "fixed" not in merged_constraints.get(column, {})
            }
            study.enqueue_trial(enqueued_params)

        def objective(trial: optuna.trial.Trial) -> float:
            mix_design = self._sample_trial_mix(trial, merged_constraints)
            candidate = self._evaluate_mix(
                mix_design,
                target_strength,
                tolerance,
                merged_constraints,
                context=design_context,
            )
            evaluated_candidates.append(candidate)
            return float(candidate["objective"])

        study.optimize(objective, n_trials=trial_budget, show_progress_bar=False)
        if not evaluated_candidates:
            raise RuntimeError("No candidate mixes were evaluated during Optuna optimization.")

        ranked_candidates = sorted(evaluated_candidates, key=self._ranking_key)
        best_candidate = ranked_candidates[0]
        top_ranked_candidates = int(self.design_config.get("top_ranked_candidates", 5))
        result = {
            "success": bool(best_candidate["success"]),
            "target_strength": target_strength,
            "tolerance_mpa": tolerance,
            "predicted_strength": float(best_candidate["predicted_strength"]),
            "mix_design": best_candidate["mix_design"],
            "engineered_ratios": best_candidate["engineered_ratios"],
            "validation_verdict": best_candidate["validation_verdict"],
            "hard_constraints": best_candidate["hard_constraints"],
            "engineering_cautions": best_candidate["engineering_cautions"],
            "data_review_flags": best_candidate["data_review_flags"],
            "contextual_summary": best_candidate["contextual_summary"],
            "confidence_of_warning_assessment": best_candidate["confidence_of_warning_assessment"],
            "validation_warning_reasons": best_candidate["validation_warning_reasons"],
            "validation_failure_reasons": best_candidate["validation_failure_reasons"],
            "design_constraint_violations": best_candidate["design_constraint_violations"],
            "uncertainty_interval": best_candidate["uncertainty_interval"],
            "ranking_breakdown": best_candidate["ranking_breakdown"],
            "plausibility_penalty": best_candidate["plausibility_penalty"],
            "design_context": design_context,
            "sample_validation": best_candidate["sample_validation"],
            "ranked_candidates": ranked_candidates[:top_ranked_candidates],
        }
        result["estimated_cement_saving_vs_reference"] = self._estimate_reference_mix(result, merged_constraints)
        return result

    def batch_optimize(self, target_strengths: list[float]) -> pd.DataFrame:
        """Optimize a list of target strengths and return a flat results dataframe."""
        return pd.DataFrame([self._result_to_batch_record(self.optimize(target_strength)) for target_strength in target_strengths])

    def export_single_target_artifact(
        self,
        target_strength: float,
        *,
        run_id: str | None = None,
    ) -> tuple[dict[str, Any], Path]:
        """Optimize and export one design JSON artifact with shared lineage metadata."""
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
        """Export a batch CSV and matching per-target JSONs from one shared result set."""
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
        """Save a JSON report for an optimized concrete mix design."""
        resolved_path = Path(output_path)
        if not resolved_path.is_absolute():
            resolved_path = self.project_root / resolved_path
        resolved_path.parent.mkdir(parents=True, exist_ok=True)
        save_json_artifact(resolved_path, self._build_design_artifact_payload(result, source_mode="design_single"))
        log_status(f"Saved design report to {resolved_path}")
        return resolved_path


def _target_label(target_strength: float) -> str:
    """Convert a target strength into a file-safe suffix."""
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
