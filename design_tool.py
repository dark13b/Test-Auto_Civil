"""Inverse concrete mix design utilities for AutoCivil-Lab."""

from __future__ import annotations

import argparse
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import pandas as pd

from feature_engineering import build_engineering_features
from train import (
    get_base_input_columns,
    get_input_columns,
    get_outputs_dir,
    get_project_root,
    get_target_column,
    load_config,
    load_dataset,
    load_pickle_artifact,
    log_status,
    save_json_artifact,
    set_global_seed,
)
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
        self.feature_columns = get_input_columns(self.config)
        self.target_column = get_target_column(self.config)
        self.design_config = self.config["engineering"]["design_tool"]
        self.validator = EngineeringValidator.from_config(self.config)
        self.model = load_pickle_artifact(self.model_path)
        dataset = load_dataset(self.config)
        self.reference_dataset = dataset[self.base_columns + [self.target_column]].copy()

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
            cement_bounds = (120.0, 280.0)
            water_bounds = (140.0, 200.0)
            water_cement_max = 0.65
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
        bounded["water_cement_ratio"]["max"] = min(
            float(bounded["water_cement_ratio"].get("max", water_cement_max)),
            water_cement_max,
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

    def _frame_from_mix(self, mix_design: dict[str, float]) -> pd.DataFrame:
        """Convert a mix-design dictionary into a one-row dataframe."""
        normalized_mix = {column: float(mix_design[column]) for column in self.base_columns}
        for column in ("slag", "fly_ash"):
            if abs(normalized_mix[column]) < 1.0:
                normalized_mix[column] = 0.0
        return pd.DataFrame([normalized_mix])

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
    ) -> dict[str, Any]:
        """Predict and score a candidate mix design."""
        candidate_frame = self._frame_from_mix(mix_design)
        engineered = build_engineering_features(candidate_frame)
        predicted_strength = float(self.model.predict(engineered[self.feature_columns])[0])
        sample_report = self.validator.evaluate_samples(
            np.asarray([predicted_strength], dtype=float),
            engineered[self.feature_columns],
        )[0]
        engineered_row = engineered.iloc[0]
        design_constraint_violations = self._check_design_constraints(engineered_row, constraints)

        progressive_penalty = max(0.0, mix_design["cement"] - target_strength * 8.0) ** 2 * 0.01
        over_strength_penalty = 0.0
        if predicted_strength > target_strength * 1.10:
            over_strength_penalty = (predicted_strength - target_strength) * 0.5
        warning_penalty = float(sample_report["warning_count"]) * 5000.0

        deviation = abs(predicted_strength - target_strength)
        objective = (
            self._cost_value(mix_design)
            + deviation * 12.0
            + progressive_penalty
            + over_strength_penalty
            + warning_penalty
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
                if column not in self.base_columns
            },
            "validation_verdict": sample_report["overall_verdict"],
            "validation_warning_reasons": list(sample_report["warning_reasons"]),
            "validation_failure_reasons": list(sample_report["failure_reasons"]),
            "design_constraint_violations": design_constraint_violations,
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

        for warm_start_mix in self._warm_start_mixes(target_strength, merged_constraints):
            enqueued_params = {
                column: value
                for column, value in warm_start_mix.items()
                if "fixed" not in merged_constraints.get(column, {})
            }
            study.enqueue_trial(enqueued_params)

        def objective(trial: optuna.trial.Trial) -> float:
            mix_design = self._sample_trial_mix(trial, merged_constraints)
            candidate = self._evaluate_mix(mix_design, target_strength, tolerance, merged_constraints)
            evaluated_candidates.append(candidate)
            return float(candidate["objective"])

        study.optimize(objective, n_trials=trial_budget, show_progress_bar=False)
        if not evaluated_candidates:
            raise RuntimeError("No candidate mixes were evaluated during Optuna optimization.")

        best_candidate = min(evaluated_candidates, key=self._ranking_key)
        result = {
            "success": bool(best_candidate["success"]),
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
