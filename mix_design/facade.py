"""Facade that composes predictor, optimizer, constraints, and scoring."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from mix_design.constraints import evaluate_constraints, prepare_design_constraints
from mix_design.contracts import (
    CandidateProposal,
    CandidateScenario,
    DesignConstraints,
    DesignContext,
    MixDesignRequest,
    ObjectiveSpec,
    ScenarioComparisonResult,
    ValidatorOutcome,
)
from mix_design.objective_engine import MixObjectiveEngine
from mix_design.optimizer import MixDesignSpaceOptimizer
from mix_design.predictor import MixPerformancePredictor
from train import (
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
    read_pickle_artifact_metadata,
    resolve_model_feature_columns,
    set_global_seed,
)
from uncertainty import UncertaintyEstimator
from validator import EngineeringValidator


class ConcreteMixDesignService:
    """Canonical entrypoint for concrete mix design scenario comparison."""

    def __init__(
        self,
        *,
        config: dict[str, Any],
        predictor: MixPerformancePredictor,
        optimizer: MixDesignSpaceOptimizer,
        objective_engine: MixObjectiveEngine,
        validator: EngineeringValidator,
        reference_dataset: pd.DataFrame,
        base_columns: list[str],
        target_column: str,
        design_config: dict[str, Any],
        project_root: Path,
        outputs_dir: Path,
        model_artifact_id: str,
        current_best_payload: dict[str, Any] | None = None,
        current_uncertainty_payload: dict[str, Any] | None = None,
        active_run_id: str | None = None,
    ) -> None:
        self.config = config
        self.predictor = predictor
        self.optimizer = optimizer
        self.objective_engine = objective_engine
        self.validator = validator
        self.reference_dataset = reference_dataset
        self.base_columns = base_columns
        self.target_column = target_column
        self.design_config = design_config
        self.project_root = project_root
        self.outputs_dir = outputs_dir
        self.model_artifact_id = model_artifact_id
        self.current_best_payload = current_best_payload or {}
        self.current_uncertainty_payload = current_uncertainty_payload or {}
        self.active_run_id = str(active_run_id or create_run_id())

    @classmethod
    def from_default_artifacts(
        cls,
        model_path: str | Path | None = None,
        config_path: str | Path | None = None,
    ) -> "ConcreteMixDesignService":
        project_root = get_project_root()
        config = cls._load_config(project_root=project_root, config_path=config_path)
        outputs_dir = get_outputs_dir(config)
        resolved_model_path = cls._resolve_model_path(project_root=project_root, config=config, model_path=model_path)
        if not resolved_model_path.exists():
            raise FileNotFoundError(
                f"Best search model not found at {resolved_model_path}. Run research_loop.py before design_tool.py."
            )

        seed = int(config["experiment"]["random_seed"])
        set_global_seed(seed)
        base_columns = get_base_input_columns(config)
        target_column = get_target_column(config)
        design_config = config["engineering"]["design_tool"]
        model = load_pickle_artifact(resolved_model_path)
        model_metadata = read_pickle_artifact_metadata(resolved_model_path) or {}
        model_artifact_id = str(
            model_metadata.get("artifact_id") or compute_file_hash(resolved_model_path) or "unknown-model-artifact"
        )
        feature_columns = resolve_model_feature_columns(model, config)
        dataset = load_dataset(config)
        reference_dataset = dataset[base_columns + [target_column]].copy()
        current_best_payload = cls._load_json_payload(outputs_dir / "best_search_result.json")
        current_uncertainty_payload = cls._load_json_payload(outputs_dir / "uncertainty_calibration.json")
        active_run_id = infer_active_run_id(
            outputs_dir,
            current_best_payload,
            current_uncertainty_payload,
            fallback=create_run_id(),
        )
        predictor = MixPerformancePredictor(
            model=model,
            config=config,
            base_columns=base_columns,
            feature_columns=feature_columns,
            model_artifact_id=model_artifact_id,
            uncertainty_estimator=UncertaintyEstimator(
                model=model,
                report_model=model,
                config_path=config_path,
                outputs_dir=outputs_dir,
                report_filename="design_uncertainty_calibration.json",
            ),
        )
        return cls(
            config=config,
            predictor=predictor,
            optimizer=MixDesignSpaceOptimizer(
                reference_dataset=reference_dataset,
                base_columns=base_columns,
                target_column=target_column,
                design_config=design_config,
                seed=seed,
            ),
            objective_engine=MixObjectiveEngine(
                cost_proxy=str(design_config.get("cost_proxy", "cement_content")),
            ),
            validator=EngineeringValidator.from_config(config),
            reference_dataset=reference_dataset,
            base_columns=base_columns,
            target_column=target_column,
            design_config=design_config,
            project_root=project_root,
            outputs_dir=outputs_dir,
            model_artifact_id=model_artifact_id,
            current_best_payload=current_best_payload,
            current_uncertainty_payload=current_uncertainty_payload,
            active_run_id=active_run_id,
        )

    @staticmethod
    def _load_json_payload(path: Path) -> dict[str, Any]:
        if not path.exists():
            return {}
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return payload if isinstance(payload, dict) else {}

    @staticmethod
    def _load_config(*, project_root: Path, config_path: str | Path | None) -> dict[str, Any]:
        if config_path is None:
            return load_config()

        resolved_path = Path(config_path)
        if not resolved_path.is_absolute():
            resolved_path = project_root / resolved_path
        if not resolved_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {resolved_path}")
        if resolved_path == project_root / "config.yaml":
            return load_config()

        import yaml

        with resolved_path.open("r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
        if not isinstance(config, dict):
            raise ValueError("Configuration file did not parse into a dictionary.")
        return config

    @staticmethod
    def _resolve_model_path(
        *,
        project_root: Path,
        config: dict[str, Any],
        model_path: str | Path | None,
    ) -> Path:
        if model_path is None:
            return project_root / config["paths"]["outputs_dir"] / "best_search_model.pkl"
        resolved_path = Path(model_path)
        if not resolved_path.is_absolute():
            resolved_path = project_root / resolved_path
        return resolved_path

    def _normalize_context(self, context: DesignContext) -> DesignContext:
        def normalize(value: str | None) -> str | None:
            if value in (None, ""):
                return None
            return str(value).strip().lower().replace(" ", "_")

        return DesignContext(
            exposure_class=normalize(context.exposure_class),
            structural_application=normalize(context.structural_application),
        )

    def _normalize_request(self, request: MixDesignRequest) -> MixDesignRequest:
        objectives = tuple(objective for objective in request.objectives if objective.enabled)
        if not objectives:
            raise ValueError("At least one enabled objective is required for scenario comparison.")

        constraints = prepare_design_constraints(
            self.config,
            target_strength_mpa=float(request.target_strength_mpa),
            overrides=request.constraints,
        )
        return MixDesignRequest(
            target_strength_mpa=float(request.target_strength_mpa),
            constraints=constraints,
            context=self._normalize_context(request.context),
            objectives=objectives,
            candidate_limit=max(int(request.candidate_limit), 1),
        )

    def _build_validator_outcome(self, sample_report: dict[str, Any]) -> ValidatorOutcome:
        return ValidatorOutcome(
            overall_verdict=str(sample_report["overall_verdict"]),
            warning_reasons=tuple(str(reason) for reason in sample_report.get("warning_reasons", [])),
            failure_reasons=tuple(str(reason) for reason in sample_report.get("failure_reasons", [])),
            hard_constraints=tuple(sample_report.get("hard_constraints", [])),
            engineering_cautions=tuple(sample_report.get("engineering_cautions", [])),
            data_review_flags=tuple(sample_report.get("data_review_flags", [])),
            contextual_summary=str(sample_report.get("contextual_summary", "")),
            confidence_of_warning_assessment=str(sample_report.get("confidence_of_warning_assessment", "")),
            raw_report=dict(sample_report),
        )

    @staticmethod
    def _has_durability_warning(scenario: CandidateScenario) -> bool:
        """True when any fired rule is a durability-class engineering caution."""
        triggered = scenario.validator.raw_report.get("triggered_rules", [])
        return any(
            str(code).startswith("durability_") or str(code) == "low_binder_durability_caution"
            for code in triggered
        )

    def _scenario_success(self, scenario: CandidateScenario, request: MixDesignRequest) -> bool:
        return (
            scenario.constraints.passed
            and scenario.validator.overall_verdict != "FAIL"
            and not self._has_durability_warning(scenario)
            and float(scenario.prediction.uncertainty_interval.target_window_overlap) > 0.0
        )

    def _search_penalty(self, scenario: CandidateScenario, request: MixDesignRequest) -> float:
        predicted_strength = float(scenario.prediction.predicted_strength_mpa)
        target_strength = float(request.target_strength_mpa)
        tolerance = max(float(request.constraints.tolerance_mpa or 0.0), 1e-6)
        penalty = 0.0
        if predicted_strength < target_strength - tolerance:
            penalty += ((target_strength - tolerance) - predicted_strength) ** 2 * 1500.0
        if predicted_strength > target_strength + tolerance:
            penalty += (predicted_strength - (target_strength + tolerance)) ** 2 * 1200.0
        if scenario.constraints.hard_failures:
            penalty += 100000.0 * len(scenario.constraints.hard_failures)
        if scenario.validator.overall_verdict == "FAIL":
            penalty += 1000000.0
        elif scenario.validator.overall_verdict == "WARN":
            penalty += 2500.0
        penalty += float(len(scenario.validator.engineering_cautions) * 750.0)
        penalty += float(len(scenario.validator.data_review_flags) * 500.0)
        water_cement_ratio = scenario.prediction.engineered_features.get("water_cement_ratio")
        water_cement_max = request.constraints.water_cement_ratio.max
        if water_cement_ratio is not None and water_cement_max is not None:
            ratio = float(water_cement_ratio)
            ceiling = max(float(water_cement_max), 1e-6)
            if ratio > ceiling:
                penalty += ((ratio - ceiling) / ceiling) * 100000.0
        if float(scenario.prediction.uncertainty_interval.target_window_overlap) <= 0.0:
            penalty += float(scenario.prediction.uncertainty_interval.interval_width) * 150.0
        penalty += float(scenario.prediction.uncertainty_interval.interval_width) * 25.0
        confidence_label = str(scenario.prediction.uncertainty_interval.confidence_label).strip().upper()
        if not bool(scenario.prediction.uncertainty_interval.is_calibrated):
            penalty += 10000.0
        if confidence_label in {"LOW", "UNKNOWN", "UNCALIBRATED"}:
            penalty += 5000.0
        penalty += float(len(scenario.prediction.uncertainty_interval.warning_reasons) * 1000.0)
        return penalty

    def _evaluate_proposal(
        self,
        proposal: CandidateProposal,
        request: MixDesignRequest,
    ) -> CandidateScenario:
        candidate_frame = self.predictor.build_candidate_frame(proposal.mix_design, context=request.context)
        prediction = self.predictor.predict(
            proposal.mix_design,
            target_strength_mpa=float(request.target_strength_mpa),
            tolerance_mpa=float(request.constraints.tolerance_mpa or 0.0),
            context=request.context,
        )
        validation_frame = candidate_frame.copy()
        validation_frame["target_strength"] = float(request.target_strength_mpa)
        sample_report = self.validator.evaluate_samples(
            np.asarray([prediction.predicted_strength_mpa], dtype=float),
            validation_frame,
        )[0]
        validator_outcome = self._build_validator_outcome(sample_report)
        constraint_evaluation = evaluate_constraints(
            proposal.mix_design,
            prediction.engineered_features,
            request.constraints,
        )
        objective_scorecard = self.objective_engine.score_candidate(
            mix_design=proposal.mix_design,
            prediction=prediction,
            constraints=constraint_evaluation,
            validator=validator_outcome,
            target_strength_mpa=float(request.target_strength_mpa),
            design_constraints=request.constraints,
            objectives=request.objectives,
        )
        scenario = CandidateScenario(
            scenario_id=proposal.proposal_id,
            mix_design={column: float(proposal.mix_design[column]) for column in self.base_columns},
            prediction=prediction,
            constraints=constraint_evaluation,
            validator=validator_outcome,
            objective_scorecard=objective_scorecard,
            success=False,
            source=proposal.source,
        )
        return CandidateScenario(
            scenario_id=scenario.scenario_id,
            mix_design=scenario.mix_design,
            prediction=scenario.prediction,
            constraints=scenario.constraints,
            validator=scenario.validator,
            objective_scorecard=scenario.objective_scorecard,
            success=self._scenario_success(scenario, request),
            source=scenario.source,
        )

    def _estimate_reference_comparison(
        self,
        best_scenario: CandidateScenario,
        request: MixDesignRequest,
        constraints: DesignConstraints,
        context: DesignContext,
    ) -> dict[str, float | None]:
        reference_ratio = float(self.design_config["reference_mix"]["water_cement_ratio"])
        grid_points = int(self.design_config["search"]["reference_grid_points"])
        cement_min = constraints.cement.min
        cement_max = constraints.cement.max
        water_min = constraints.water.min
        water_max = constraints.water.max
        if cement_min is None or cement_max is None or water_min is None or water_max is None:
            return {
                "reference_cement_content": None,
                "reference_predicted_strength": None,
                "cement_saving_kg_per_m3": None,
                "cement_saving_percent": None,
            }

        cement_values = np.linspace(float(cement_min), float(cement_max), grid_points)
        reference_rows: list[dict[str, float]] = []
        optimized_mix = best_scenario.mix_design
        for cement in cement_values:
            water = float(cement) * reference_ratio
            if water < float(water_min) or water > float(water_max):
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

        reference_predictions = self.predictor.predict_strengths(reference_rows, context=context)
        tolerance = float(request.constraints.tolerance_mpa or 0.0)
        target_strength = float(request.target_strength_mpa)
        selected_index = None
        for index, prediction in enumerate(reference_predictions):
            if float(prediction) >= target_strength - tolerance:
                selected_index = index
                break
        if selected_index is None:
            selected_index = int(np.argmin(np.abs(np.asarray(reference_predictions, dtype=float) - target_strength)))

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

    def compare_scenarios(self, request: MixDesignRequest) -> ScenarioComparisonResult:
        """Generate, evaluate, and rank multiple candidate scenarios."""

        normalized_request = self._normalize_request(request)
        scenario_cache: dict[str, CandidateScenario] = {}

        def evaluate_candidate(proposal: CandidateProposal) -> float:
            scenario = self._evaluate_proposal(proposal, normalized_request)
            scenario_cache[proposal.proposal_id] = scenario
            return float(scenario.objective_scorecard.total_score + self._search_penalty(scenario, normalized_request))

        optimization_result = self.optimizer.optimize(
            target_strength_mpa=normalized_request.target_strength_mpa,
            constraints=normalized_request.constraints,
            candidate_limit=normalized_request.candidate_limit,
            evaluate_candidate=evaluate_candidate,
        )
        scenarios = tuple(
            scenario_cache[proposal.proposal_id]
            for proposal in optimization_result.proposals
            if proposal.proposal_id in scenario_cache
        )
        if not scenarios:
            raise RuntimeError("No candidate scenarios were available for comparison.")

        ranked_scenarios = self.objective_engine.rank_scenarios(scenarios)
        visible_scenarios = ranked_scenarios[: normalized_request.candidate_limit]
        best_scenario = visible_scenarios[0]
        return ScenarioComparisonResult(
            request=normalized_request,
            best_scenario_id=best_scenario.scenario_id,
            scenarios=visible_scenarios,
            comparison_summary=self.objective_engine.build_comparison_summary(visible_scenarios),
            reference_comparison=self._estimate_reference_comparison(
                best_scenario,
                normalized_request,
                normalized_request.constraints,
                normalized_request.context,
            ),
        )

    def export_comparison(
        self,
        result: ScenarioComparisonResult,
        output_path: str | Path,
    ) -> Path:
        """Serialize a canonical comparison result as JSON."""

        resolved_path = Path(output_path)
        if not resolved_path.is_absolute():
            resolved_path = self.project_root / resolved_path
        resolved_path.parent.mkdir(parents=True, exist_ok=True)

        from mix_design.exporters import comparison_result_to_dict

        with resolved_path.open("w", encoding="utf-8") as handle:
            json.dump(comparison_result_to_dict(result), handle, indent=2, sort_keys=True)
        return resolved_path
