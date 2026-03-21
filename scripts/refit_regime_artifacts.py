"""Refit baseline and current best-search artifacts with regime-specific modeling."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from search import load_json_artifact
from train import (
    EngineeringValidator,
    build_stacking_ensemble,
    create_run_id,
    evaluate_candidate,
    get_outputs_dir,
    load_config,
    load_dataset,
    log_status,
    save_json_artifact,
    save_pickle_artifact,
    set_global_seed,
    split_dataset,
    write_run_scoped_json_artifact,
)


def main() -> int:
    config = load_config()
    set_global_seed(int(config["experiment"]["random_seed"]))
    outputs_dir = get_outputs_dir(config)
    current_best = load_json_artifact(outputs_dir / "best_search_result.json")
    run_id = create_run_id()
    dataset = load_dataset(config)
    x_train, x_val, _, y_train, y_val, _ = split_dataset(dataset, config)
    validator = EngineeringValidator.from_config(config)
    baseline_model, baseline_result = evaluate_candidate(
        str(config["baseline_model"]["model_name"]),
        dict(config["baseline_model"]["params"]),
        x_train,
        y_train,
        x_val,
        y_val,
        validator,
        config,
    )
    save_pickle_artifact(
        outputs_dir / "baseline_model.pkl",
        baseline_model,
        config=config,
        model_id=str(config["baseline_model"]["model_name"]),
        run_id=run_id,
        source_mode="baseline_refit",
    )
    save_json_artifact(outputs_dir / "baseline_metrics.json", baseline_result)

    model_name = str(current_best["model_name"])
    if model_name == "StackingRegressor":
        base_models = current_best["hyperparameters"]["base_models"]
        configs = [(item["model_name"], dict(item["hyperparameters"])) for item in base_models]
        model, result = build_stacking_ensemble(configs, x_train, y_train, x_val, y_val, validator, config)
    else:
        model, result = evaluate_candidate(
            model_name,
            dict(current_best["hyperparameters"]),
            x_train,
            y_train,
            x_val,
            y_val,
            validator,
            config,
        )

    merged = dict(current_best)
    merged.update(result)
    merged["run_id"] = run_id
    merged["status"] = current_best.get("status", "refit")
    merged["source"] = current_best.get("source", merged.get("source", "search_refit"))
    merged["best_trial"] = current_best.get("best_trial", current_best.get("trial_number"))
    merged["trial_number"] = current_best.get("trial_number", merged.get("best_trial"))
    merged["selection_status"] = current_best.get("selection_status", "new_best")
    merged["beats_baseline"] = bool(merged.get("composite_score", 0.0) >= 0.0)
    if "selected_base_models" in current_best:
        merged["selected_base_models"] = current_best["selected_base_models"]
    if "ensemble_size" in current_best:
        merged["ensemble_size"] = current_best["ensemble_size"]

    best_model_metadata = save_pickle_artifact(
        outputs_dir / "best_search_model.pkl",
        model,
        config=config,
        model_id=model_name,
        run_id=run_id,
        source_mode="search_refit",
    )
    search_state_metadata = save_pickle_artifact(
        outputs_dir / "search_state_best_model.pkl",
        model,
        config=config,
        model_id=model_name,
        run_id=run_id,
        source_mode="search_refit",
    )
    merged["model_artifact_id"] = best_model_metadata["artifact_id"]

    write_run_scoped_json_artifact(
        outputs_dir=outputs_dir,
        filename="best_search_result.json",
        payload=merged,
        run_id=run_id,
        source_mode="search_refit",
        config=config,
        model_artifact_id=best_model_metadata["artifact_id"],
        model_id=model_name,
    )
    write_run_scoped_json_artifact(
        outputs_dir=outputs_dir,
        filename="search_state_best_result.json",
        payload=merged,
        run_id=run_id,
        source_mode="search_refit",
        config=config,
        model_artifact_id=search_state_metadata["artifact_id"],
        model_id=model_name,
    )

    log_status(
        "Regime refit complete: "
        + json.dumps(
            {
                "run_id": run_id,
                "model_name": model_name,
                "cv_rmse": merged.get("cv_rmse"),
                "val_rmse": merged.get("val_rmse"),
                "validation_verdict": merged.get("validation_verdict"),
                "model_artifact_id": merged.get("model_artifact_id"),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
