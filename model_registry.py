"""Single responsibility: model class definitions and model construction from config."""

from __future__ import annotations

from typing import Any

from train_impl import instantiate_model


def build_model(config: dict[str, Any]) -> Any:
    model_name = str(
        config.get("model_name")
        or config.get("training", {}).get("model_name")
        or config.get("model", {}).get("name")
        or "LinearRegression"
    )
    params = dict(config.get("model_params", config.get("training", {}).get("params", {})))
    return instantiate_model(model_name, params, config)
