"""
train.py
Single responsibility: orchestrate one training run.
All implementation lives in submodules.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from artifact_io import save_training_artifacts
from data_loading import build_dataloaders
from evaluation import evaluate_model
from feature_pipeline import apply_feature_pipeline
from model_registry import build_model
from train_impl import *  # noqa: F401,F403
from train_impl import main


def run_training(config: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    data = build_dataloaders(config)
    data = apply_feature_pipeline(data, config)
    model = build_model(config)
    model.fit(data.train.features, data.train.target)
    metrics = evaluate_model(model, data.val, config)
    save_training_artifacts(model, metrics, output_dir)
    return metrics


def evaluate_candidate(*args, **kwargs):
    from train_impl import evaluate_candidate as _evaluate_candidate

    return _evaluate_candidate(*args, **kwargs)


def build_stacking_ensemble(*args, **kwargs):
    from train_impl import build_stacking_ensemble as _build_stacking_ensemble

    return _build_stacking_ensemble(*args, **kwargs)


def main() -> int:
    from train_impl import main as _main

    return _main()


if __name__ == "__main__":
    raise SystemExit(main())
