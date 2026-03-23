"""Controlled research-editable surface for AutoCivil-Lab.

The infrastructure loop should treat this file as the only mutable research
surface. Future research changes belong here, not in the orchestration code.
"""

from __future__ import annotations

import logging
from typing import Any

from research_protocol import DuplicateExperimentError, validate_lab_state_integrity


LOGGER = logging.getLogger(__name__)


# RESEARCH_SURFACE_STATE_START
LAB_STATE = {'accepted_experiments': [{'composite_score': 0.9043986057974769,
                           'confirm_improvement_pct': 0.0007343236027245474,
                           'experiment_id': 'confirm-scout-004-randomforestregressor-exploit',
                           'model_name': 'RandomForestRegressor',
                           'proposal_family': 'RandomForestRegressor-exploit'},
                          {'composite_score': 0.9116529719083861,
                           'confirm_improvement_pct': 0.8022905290182418,
                           'experiment_id': 'confirm-scout-009-lgbmregressor-expansive',
                           'model_name': 'LGBMRegressor',
                           'proposal_family': 'LGBMRegressor-expansive'},
                          {'composite_score': 0.9192112496389842,
                           'confirm_improvement_pct': 0.8290739967398005,
                           'experiment_id': 'confirm-llm-scout-002',
                           'model_name': 'LGBMRegressor',
                           'proposal_family': 'LGBMRegressor-expansive'},
                          {'composite_score': 0.9193021590292377,
                           'confirm_improvement_pct': 0.009889934472547395,
                           'experiment_id': 'confirm-scout-010-lgbmregressor-exploit',
                           'model_name': 'LGBMRegressor',
                           'proposal_family': 'LGBMRegressor-exploit'},
                          {'composite_score': 0.9201538422832174,
                           'confirm_improvement_pct': 0.0842258027725169,
                           'experiment_id': 'confirm-scout-011-lgbmregressor-exploit-learning_rate-down',
                           'model_name': 'LGBMRegressor',
                           'proposal_family': 'LGBMRegressor-exploit'},
                          {'composite_score': 0.920218743794953,
                           'confirm_improvement_pct': 0.007053332687783711,
                           'experiment_id': 'confirm-scout-019-lgbmregressor-exploit-colsample_bytree-down',
                           'model_name': 'LGBMRegressor',
                           'proposal_family': 'LGBMRegressor-exploit'},
                          {'composite_score': 0.882910773048627,
                           'confirm_improvement_pct': 0.03583735827544764,
                           'experiment_id': 'confirm-scout-010-randomforestregressor-exploit-n_estimators-down',
                           'model_name': 'RandomForestRegressor',
                           'proposal_family': 'RandomForestRegressor-exploit'}],
 'recent_kept_families': ['RandomForestRegressor-exploit',
                          'LGBMRegressor-expansive',
                          'LGBMRegressor-expansive',
                          'LGBMRegressor-exploit',
                          'LGBMRegressor-exploit',
                          'LGBMRegressor-exploit',
                          'RandomForestRegressor-exploit'],
 'surface_version': 1}
# RESEARCH_SURFACE_STATE_END

try:
    validate_lab_state_integrity(LAB_STATE)
except DuplicateExperimentError as exc:
    LOGGER.critical("Invalid checked-in LAB_STATE: %s", exc)
    raise


def _numeric_anchor(low: float, high: float, anchor: str, *, step: float | None = None, log: bool = False) -> float:
    if anchor == "low":
        value = low
    elif anchor == "high":
        value = high
    else:
        value = (low * high) ** 0.5 if log and low > 0 and high > 0 else (low + high) / 2.0
    if step and step > 0:
        value = low + round((value - low) / step) * step
    return value


def _anchor_value(parameter_name: str, spec: dict[str, Any], profile: str) -> Any:
    parameter_type = spec.get("type")
    name = parameter_name.lower()

    if parameter_type == "int":
        if any(token in name for token in ("n_estimators", "iterations")):
            anchor = "high" if profile in {"balanced", "expansive"} else "mid"
        elif any(token in name for token in ("depth", "leaves", "leaf_nodes")):
            anchor = "high" if profile == "expansive" else ("low" if profile == "conservative" else "mid")
        elif any(token in name for token in ("min_child", "min_samples")):
            anchor = "high" if profile == "conservative" else "mid"
        else:
            anchor = "mid"
        value = _numeric_anchor(
            float(spec["low"]),
            float(spec["high"]),
            anchor,
            step=float(spec.get("step", 1)),
        )
        return int(round(value))

    if parameter_type == "float":
        if "learning_rate" in name:
            anchor = "low" if profile in {"balanced", "conservative"} else "high"
        elif any(token in name for token in ("alpha", "lambda", "regularization", "temperature", "strength")):
            anchor = "high" if profile == "conservative" else "mid"
        elif any(token in name for token in ("subsample", "colsample", "bagging")):
            anchor = "mid"
        else:
            anchor = "mid"
        return float(
            _numeric_anchor(
                float(spec["low"]),
                float(spec["high"]),
                anchor,
                log=bool(spec.get("log", False)),
            )
        )

    if parameter_type == "categorical":
        choices = list(spec.get("choices", []))
        if not choices:
            return None
        if profile == "conservative":
            return choices[0]
        if profile == "expansive":
            return choices[-1]
        return choices[len(choices) // 2]

    raise ValueError(f"Unsupported search parameter type: {parameter_type}")


def _build_profile_candidate(
    *,
    model_name: str,
    display_name: str,
    search_space: dict[str, dict[str, Any]],
    profile: str,
    sequence_id: int,
) -> dict[str, Any]:
    params = {
        parameter_name: _anchor_value(parameter_name, parameter_spec, profile)
        for parameter_name, parameter_spec in search_space.items()
    }
    return {
        "experiment_id": f"scout-{sequence_id:03d}-{model_name.lower()}-{profile}",
        "stage": "scout",
        "model_name": model_name,
        "display_name": display_name,
        "proposal_family": f"{model_name}-{profile}",
        "hypothesis": f"{display_name} {profile} anchor profile",
        "params": params,
    }


def _mutate_current_best(
    *,
    current_best: dict[str, Any],
    available_models: dict[str, dict[str, Any]],
    sequence_id: int,
    exploit_delta_ratio: float = 0.15,
) -> list[dict[str, Any]]:
    model_name = str(current_best.get("model_name", ""))
    if model_name not in available_models:
        return []

    search_space = dict(available_models[model_name].get("search_space", {}))
    base_params = dict(current_best.get("hyperparameters", {}))
    if not search_space or not base_params:
        return []

    display_name = str(available_models[model_name].get("display_name", model_name))
    candidates: list[dict[str, Any]] = []
    seen_signatures: set[tuple[tuple[str, Any], ...]] = set()

    def _register_candidate(mutated_params: dict[str, Any], mutation_label: str) -> None:
        signature = tuple(sorted(mutated_params.items()))
        if signature in seen_signatures:
            return
        seen_signatures.add(signature)
        candidates.append(
            {
                "experiment_id": f"scout-{len(candidates) + sequence_id:03d}-{model_name.lower()}-exploit-{mutation_label}",
                "stage": "scout",
                "model_name": model_name,
                "display_name": display_name,
                "proposal_family": f"{model_name}-exploit",
                "hypothesis": f"{display_name} exploit around current best via {mutation_label}",
                "params": mutated_params,
            }
        )

    for parameter_name, spec in search_space.items():
        if parameter_name not in base_params:
            continue
        parameter_type = spec.get("type")
        base_value = base_params[parameter_name]
        if parameter_type == "int":
            step = max(1, int(spec.get("step", 1)))
            low = int(spec["low"])
            high = int(spec["high"])
            for direction, delta in (("down", -step), ("up", step)):
                mutated_value = int(base_value) + delta
                mutated_value = min(max(mutated_value, low), high)
                if mutated_value == int(base_value):
                    continue
                mutated = dict(base_params)
                mutated[parameter_name] = mutated_value
                _register_candidate(mutated, f"{parameter_name}-{direction}")
        elif parameter_type == "float":
            low = float(spec["low"])
            high = float(spec["high"])
            base_numeric = float(base_value)
            if bool(spec.get("log", False)):
                delta_ratio = max(float(exploit_delta_ratio), 1e-6)
                for direction, factor in (("down", 1.0 - delta_ratio), ("up", 1.0 + delta_ratio)):
                    if factor <= 0.0:
                        continue
                    mutated_value = min(max(base_numeric * factor, low), high)
                    mutated_value = round(float(mutated_value), 12)
                    if abs(mutated_value - base_numeric) <= 1e-12:
                        continue
                    mutated = dict(base_params)
                    mutated[parameter_name] = mutated_value
                    _register_candidate(mutated, f"{parameter_name}-{direction}")
                continue

            span = high - low
            delta = max(span * 0.15, abs(base_numeric) * 0.15, 1e-6)
            for direction, sign in (("down", -1.0), ("up", 1.0)):
                mutated_value = min(max(base_numeric + sign * delta, low), high)
                mutated_value = round(float(mutated_value), 12)
                if abs(mutated_value - base_numeric) <= 1e-12:
                    continue
                mutated = dict(base_params)
                mutated[parameter_name] = float(mutated_value)
                _register_candidate(mutated, f"{parameter_name}-{direction}")
        elif parameter_type == "categorical":
            choices = list(spec.get("choices", []))
            if not choices:
                continue
            if base_value in choices:
                index = choices.index(base_value)
                neighbor_indices = [index - 1, index + 1]
            else:
                neighbor_indices = [0, len(choices) - 1]
            for neighbor_index in neighbor_indices:
                if neighbor_index < 0 or neighbor_index >= len(choices):
                    continue
                mutated_value = choices[neighbor_index]
                if mutated_value == base_value:
                    continue
                mutated = dict(base_params)
                mutated[parameter_name] = mutated_value
                direction = "alt" if mutated_value not in {choices[0], choices[-1]} else (
                    "down" if neighbor_index < choices.index(base_value) else "up"
                ) if base_value in choices else "alt"
                _register_candidate(mutated, f"{parameter_name}-{direction}")

    return candidates


def scout_experiments(
    *,
    brief: dict[str, Any],
    lab_state: dict[str, Any],
    available_models: dict[str, dict[str, Any]],
    experiment_memory: dict[str, Any],
    current_best: dict[str, Any],
    scout_limit: int,
    exploit_delta_ratio: float = 0.15,
) -> list[dict[str, Any]]:
    """Build deterministic scout experiments from the controlled research surface."""
    del experiment_memory

    required_families = set(str(item) for item in brief.get("required_model_families", []))
    kept_families = set(str(item) for item in lab_state.get("recent_kept_families", []))

    ordered_models = []
    deferred_models = []
    for model_name, model_config in available_models.items():
        if required_families and model_name not in required_families:
            continue
        if model_name in kept_families:
            deferred_models.append((model_name, model_config))
        else:
            ordered_models.append((model_name, model_config))
    ordered_models.extend(deferred_models)

    candidates: list[dict[str, Any]] = []
    sequence_id = 1
    for model_name, model_config in ordered_models:
        display_name = str(model_config.get("display_name", model_name))
        search_space = dict(model_config.get("search_space", {}))
        if not search_space:
            continue
        for profile in ("balanced", "conservative", "expansive"):
            candidates.append(
                _build_profile_candidate(
                    model_name=model_name,
                    display_name=display_name,
                    search_space=search_space,
                    profile=profile,
                    sequence_id=sequence_id,
                )
            )
            sequence_id += 1

    exploit_candidates = _mutate_current_best(
        current_best=current_best,
        available_models=available_models,
        sequence_id=sequence_id,
        exploit_delta_ratio=exploit_delta_ratio,
    )
    if exploit_candidates:
        candidates = exploit_candidates + candidates

    return candidates[: max(1, int(scout_limit))]


def confirm_experiments(
    *,
    scout_results: list[dict[str, Any]],
    current_best: dict[str, Any],
    confirm_top_k: int,
) -> list[dict[str, Any]]:
    """Promote only promising scout results into confirm experiments."""
    del current_best

    positive_scouts = [
        result for result in scout_results if float(result.get("scout_improvement_pct", 0.0)) > 0.0
    ]
    positive_scouts.sort(key=lambda item: float(item.get("scout_improvement_pct", 0.0)), reverse=True)

    confirms: list[dict[str, Any]] = []
    for scout_result in positive_scouts[: max(1, int(confirm_top_k))]:
        confirms.append(
            {
                "experiment_id": f"confirm-{scout_result['experiment_id']}",
                "stage": "confirm",
                "source_experiment_id": scout_result["experiment_id"],
                "model_name": scout_result["model_name"],
                "display_name": scout_result.get("display_name", scout_result["model_name"]),
                "proposal_family": scout_result["proposal_family"],
                "hypothesis": scout_result.get("hypothesis", ""),
                "params": dict(scout_result["params"]),
            }
        )
    return confirms
