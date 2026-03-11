"""Inverse concrete mix design utilities for AutoCivil-Lab."""

from __future__ import annotations

import argparse
import pickle
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import qmc

from feature_engineering import build_engineering_features
from train import (
    get_base_input_columns,
    get_input_columns,
    get_outputs_dir,
    get_project_root,
    load_config,
    log_status,
    save_json_artifact,
    set_global_seed,
)
from validator import EngineeringValidator


class MixDesignOptimizer:
    """Search for low-cement mix designs that meet a target compressive strength."""

    def __init__(self, model_path: str | Path | None = None, config_path: str | Path | None = None) -> None:
        """Load the trained model, configuration, and engineering validator."""
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
        self.feature_columns = get_input_columns(self.config)
        self.design_config = self.config["engineering"]["design_tool"]
        self.validator = EngineeringValidator.from_config(self.config)
        self.model = self._load_pickle(self.model_path)

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

    @staticmethod
    def _load_pickle(path: Path) -> Any:
        """Load a pickle artifact."""
        with path.open("rb") as handle:
            return pickle.load(handle)

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

    def _target_tolerance(self, constraints: dict[str, Any]) -> float:
        """Resolve the design tolerance."""
        if "tolerance_mpa" in constraints:
            return float(constraints["tolerance_mpa"])
        return float(self.design_config["target_tolerance_mpa"])

    def _cost_value(self, mix_design: dict[str, float]) -> float:
        """Return the configured cost proxy for a candidate mix."""
        cost_proxy = str(self.design_config["cost_proxy"]).strip().lower()
        if cost_proxy == "cement_content":
            return float(mix_design["cement"])
        if cost_proxy in mix_design:
            return float(mix_design[cost_proxy])
        raise ValueError(f"Unsupported design-tool cost proxy: {self.design_config['cost_proxy']}")

    def _variable_bounds(self, constraints: dict[str, Any]) -> tuple[list[str], list[tuple[float, float]]]:
        """Return free variables and their min/max bounds."""
        variable_columns: list[str] = []
        bounds: list[tuple[float, float]] = []
        for column in self.base_columns:
            column_constraints = constraints.get(column, {})
            if "fixed" in column_constraints:
                continue
            if "min" not in column_constraints or "max" not in column_constraints:
                raise ValueError(f"Constraint for '{column}' must define min/max or fixed.")
            variable_columns.append(column)
            bounds.append((float(column_constraints["min"]), float(column_constraints["max"])))
        return variable_columns, bounds

    def _frame_from_matrix(
        self,
        matrix: np.ndarray | None,
        variable_columns: list[str],
        constraints: dict[str, Any],
    ) -> pd.DataFrame:
        """Build a candidate dataframe from a sample matrix and fixed constraints."""
        row_count = 1 if matrix is None else int(matrix.shape[0])
        frame = pd.DataFrame(index=np.arange(row_count))
        if matrix is not None:
            for position, column in enumerate(variable_columns):
                frame[column] = matrix[:, position]

        for column in self.base_columns:
            column_constraints = constraints.get(column, {})
            if column not in frame.columns:
                if "fixed" in column_constraints:
                    frame[column] = float(column_constraints["fixed"])
                else:
                    minimum = float(column_constraints["min"])
                    maximum = float(column_constraints["max"])
                    frame[column] = (minimum + maximum) / 2.0
        return frame[self.base_columns].astype(float)

    def _sample_global_candidates(
        self,
        variable_columns: list[str],
        bounds: list[tuple[float, float]],
        constraints: dict[str, Any],
    ) -> pd.DataFrame:
        """Generate a global Latin-hypercube candidate pool."""
        initial_samples = int(self.design_config["search"]["initial_samples"])
        if not variable_columns:
            return self._frame_from_matrix(None, variable_columns, constraints)

        sampler = qmc.LatinHypercube(d=len(variable_columns), seed=self.seed)
        unit_samples = sampler.random(n=initial_samples)
        lower_bounds = np.asarray([bound[0] for bound in bounds], dtype=float)
        upper_bounds = np.asarray([bound[1] for bound in bounds], dtype=float)
        scaled_samples = qmc.scale(unit_samples, lower_bounds, upper_bounds)
        return self._frame_from_matrix(scaled_samples, variable_columns, constraints)

    def _sample_local_candidates(
        self,
        center_mix: dict[str, float],
        variable_columns: list[str],
        bounds: list[tuple[float, float]],
        constraints: dict[str, Any],
        round_index: int,
    ) -> pd.DataFrame:
        """Generate local perturbations around a promising mix design."""
        local_samples = int(self.design_config["search"]["local_samples_per_candidate"])
        local_scale = float(self.design_config["search"]["local_scale"])
        rng = np.random.default_rng(self.seed + round_index)
        if not variable_columns:
            return self._frame_from_matrix(None, variable_columns, constraints)

        local_matrix = np.zeros((local_samples, len(variable_columns)), dtype=float)
        for position, column in enumerate(variable_columns):
            minimum, maximum = bounds[position]
            spread = (maximum - minimum) * local_scale
            local_matrix[:, position] = rng.normal(
                loc=float(center_mix[column]),
                scale=max(spread, 1e-6),
                size=local_samples,
            )
            local_matrix[:, position] = np.clip(local_matrix[:, position], minimum, maximum)
        return self._frame_from_matrix(local_matrix, variable_columns, constraints)

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

    def _ranking_key(self, candidate: dict[str, Any], tolerance: float) -> tuple[Any, ...]:
        """Return a deterministic ranking key for candidate selection."""
        deviation = abs(float(candidate["predicted_strength"]) - float(candidate["target_strength"]))
        feasible = (
            candidate["validation_verdict"] != "FAIL"
            and not candidate["design_constraint_violations"]
            and deviation <= tolerance
        )
        verdict_rank = {"PASS": 0, "WARN": 1, "FAIL": 2}.get(candidate["validation_verdict"], 3)
        return (
            0 if feasible else 1,
            verdict_rank,
            0 if not candidate["design_constraint_violations"] else 1,
            max(0.0, deviation - tolerance),
            self._cost_value(candidate["mix_design"]),
            deviation,
        )

    def _evaluate_candidates(
        self,
        candidate_frame: pd.DataFrame,
        target_strength_mpa: float,
        tolerance: float,
        constraints: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Predict, validate, and summarize a pool of candidate mixes."""
        enriched = build_engineering_features(candidate_frame)
        predictions = np.asarray(self.model.predict(enriched[self.feature_columns]), dtype=float)
        sample_reports = self.validator.evaluate_samples(predictions, enriched[self.feature_columns])

        evaluated_candidates: list[dict[str, Any]] = []
        for position in range(len(enriched)):
            base_row = enriched.iloc[position]
            sample_report = sample_reports[position]
            engineered_snapshot = {
                column: float(base_row[column])
                for column in enriched.columns
                if column not in self.base_columns
            }
            mix_design = {column: float(base_row[column]) for column in self.base_columns}
            candidate = {
                "target_strength": float(target_strength_mpa),
                "predicted_strength": float(predictions[position]),
                "mix_design": mix_design,
                "engineered_ratios": engineered_snapshot,
                "validation_verdict": sample_report["overall_verdict"],
                "validation_warning_reasons": list(sample_report["warning_reasons"]),
                "validation_failure_reasons": list(sample_report["failure_reasons"]),
                "design_constraint_violations": self._check_design_constraints(base_row, constraints),
            }
            evaluated_candidates.append(candidate)
        return evaluated_candidates

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

        reference_frame = build_engineering_features(pd.DataFrame(reference_rows))
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
    ) -> dict[str, Any]:
        """Search for a mix design that meets the target strength with minimum cement."""
        target_strength = float(target_strength_mpa)
        merged_constraints = self._normalize_constraints(constraints)
        tolerance = self._target_tolerance(merged_constraints)
        if "age" not in merged_constraints:
            merged_constraints["age"] = {"fixed": float(self.design_config["default_age_days"])}
        elif "fixed" not in merged_constraints["age"]:
            merged_constraints["age"]["fixed"] = float(self.design_config["default_age_days"])

        variable_columns, bounds = self._variable_bounds(merged_constraints)
        evaluated_candidates = self._evaluate_candidates(
            self._sample_global_candidates(variable_columns, bounds, merged_constraints),
            target_strength,
            tolerance,
            merged_constraints,
        )

        refinement_rounds = int(self.design_config["search"]["refinement_rounds"])
        top_candidates = int(self.design_config["search"]["top_candidates"])
        for round_index in range(1, refinement_rounds + 1):
            current_best = sorted(evaluated_candidates, key=lambda item: self._ranking_key(item, tolerance))
            local_candidates: list[pd.DataFrame] = []
            for candidate in current_best[:top_candidates]:
                local_candidates.append(
                    self._sample_local_candidates(
                        candidate["mix_design"],
                        variable_columns,
                        bounds,
                        merged_constraints,
                        round_index,
                    )
                )
            if not local_candidates:
                continue
            local_frame = pd.concat(local_candidates, ignore_index=True)
            evaluated_candidates.extend(
                self._evaluate_candidates(local_frame, target_strength, tolerance, merged_constraints)
            )

        best_candidate = min(evaluated_candidates, key=lambda item: self._ranking_key(item, tolerance))
        feasible = self._ranking_key(best_candidate, tolerance)[0] == 0
        result = {
            "success": feasible,
            "target_strength": target_strength,
            "tolerance_mpa": tolerance,
            "predicted_strength": float(best_candidate["predicted_strength"]),
            "mix_design": best_candidate["mix_design"],
            "engineered_ratios": best_candidate["engineered_ratios"],
            "validation_verdict": best_candidate["validation_verdict"],
            "validation_warning_reasons": best_candidate["validation_warning_reasons"],
            "validation_failure_reasons": best_candidate["validation_failure_reasons"],
            "design_constraint_violations": best_candidate["design_constraint_violations"],
        }
        result["estimated_cement_saving_vs_reference"] = self._estimate_reference_mix(result, merged_constraints)
        return result

    def batch_optimize(self, target_strengths: list[float]) -> pd.DataFrame:
        """Optimize a list of target strengths and return a flat results dataframe."""
        records: list[dict[str, Any]] = []
        for target_strength in target_strengths:
            result = self.optimize(target_strength)
            reference = result["estimated_cement_saving_vs_reference"]
            record = {
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
            records.append(record)
        return pd.DataFrame(records)

    def save_design_report(self, result: dict[str, Any], output_path: str | Path) -> Path:
        """Save a JSON report for an optimized concrete mix design."""
        resolved_path = Path(output_path)
        if not resolved_path.is_absolute():
            resolved_path = self.project_root / resolved_path
        resolved_path.parent.mkdir(parents=True, exist_ok=True)
        save_json_artifact(resolved_path, result)
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
            result = optimizer.optimize(args.target)
            output_path = optimizer.outputs_dir / f"design_{_target_label(args.target)}MPa.json"
            optimizer.save_design_report(result, output_path)
            log_status(
                f"Target={result['target_strength']:.2f} MPa | "
                f"Predicted={result['predicted_strength']:.2f} MPa | "
                f"Cement={result['mix_design']['cement']:.2f} kg/m^3 | "
                f"Validation={result['validation_verdict']}"
            )
        else:
            target_strengths = [float(item.strip()) for item in str(args.batch).split(",") if item.strip()]
            batch_frame = optimizer.batch_optimize(target_strengths)
            output_path = optimizer.outputs_dir / "batch_design_results.csv"
            batch_frame.to_csv(output_path, index=False)
            log_status(f"Saved batch design results to {output_path}")
        return 0
    except Exception as exc:
        log_status(f"Design optimization failed: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
