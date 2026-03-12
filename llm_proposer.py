"""Ollama-backed LLM trial proposals for AutoCivil-Lab."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

import requests


JSON_OBJECT_PATTERN = re.compile(r"\{.*\}", re.DOTALL)
JSON_ARRAY_PATTERN = re.compile(r"\[\s*\{.*\}\s*\]", re.DOTALL)
JSON_ONLY_INSTRUCTION = (
    'Return ONLY a JSON object on a single line. No explanation. No markdown.\n'
    'Format: {"model_name": "XGBRegressor", "params": {"n_estimators": 300,\n'
    '"learning_rate": 0.05, "max_depth": 5, "subsample": 0.8,\n'
    '"colsample_bytree": 0.8, "reg_alpha": 0.01}}'
)
INVALID_PARAM = object()


class LLMProposer:
    """Request trial suggestions from a local Ollama-hosted Qwen model."""

    def __init__(self, config: dict[str, Any]):
        llm_config = config.get("search", {}).get("llm_proposals", {})
        self.base_url = str(llm_config.get("ollama_base_url", "http://localhost:11434")).rstrip("/")
        self.fast_model = str(llm_config.get("fast_model", "qwen3:4b"))
        self.smart_model = str(llm_config.get("smart_model", "qwen3:8b"))
        self.timeout_seconds = 30
        self.consecutive_invalid_responses = 0

    def is_available(self) -> bool:
        """Return True when Ollama is reachable and responding."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=self.timeout_seconds)
            response.raise_for_status()
            return True
        except Exception:
            return False

    def _build_prompt(
        self,
        trial_history: list[dict[str, Any]],
        available_models: dict[str, dict[str, Any]],
        current_best: dict[str, Any] | None,
    ) -> str:
        """Build a compact JSON-only prompt for trial suggestions."""
        recent_trials = trial_history[-min(15, len(trial_history)) :] if trial_history else []
        best_name = "unknown" if current_best is None else str(current_best.get("model_name", "unknown"))
        best_composite = self._format_metric(None if current_best is None else current_best.get("composite_score"))
        best_rmse = self._format_metric(None if current_best is None else current_best.get("rmse"))

        lines = [
            "/no_think",
            "You are guiding Optuna trial proposals for concrete compressive strength regression.",
            f"Current best: {best_name} | composite={best_composite} | RMSE={best_rmse}",
            "Recent trials:",
        ]
        if recent_trials:
            for index, trial in enumerate(recent_trials, start=1):
                trial_number = trial.get("trial_number", index)
                model_name = str(trial.get("model_name", "unknown"))
                rmse = self._format_metric(trial.get("rmse"))
                r2 = self._format_metric(trial.get("r2"))
                composite = self._format_metric(trial.get("composite_score"))
                verdict = str(trial.get("validation_verdict", "UNKNOWN"))
                lines.append(
                    f"Trial {trial_number}: {model_name} | RMSE={rmse} | "
                    f"R2={r2} | composite={composite} | {verdict}"
                )
        else:
            lines.append("No completed trials yet.")

        lines.append("Available models and parameter ranges:")
        for model_name, model_config in available_models.items():
            search_space = model_config.get("search_space", {})
            lines.append(f"{model_name}: {self._format_search_space(search_space)}")

        lines.append(JSON_ONLY_INSTRUCTION)
        return "\n".join(lines)

    def _parse_response(self, raw_text: str, available_models: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
        """Extract and validate the first JSON object returned by the LLM."""
        try:
            cleaned = raw_text.strip()
            match = JSON_OBJECT_PATTERN.search(cleaned)
            if match is None:
                return None
            payload = json.loads(match.group(0))
        except Exception:
            return None
        return self._validate_suggestion(payload, available_models)

    def propose(
        self,
        trial_history: list[dict[str, Any]],
        available_models: dict[str, dict[str, Any]],
        current_best: dict[str, Any] | None,
        use_smart_model: bool = False,
    ) -> dict[str, Any] | None:
        """Request a single trial proposal from Ollama."""
        model_to_use = self.smart_model if use_smart_model else self.fast_model
        prompt = self._build_prompt(trial_history, available_models, current_best)
        response_format = self._build_proposal_schema(available_models)

        raw_text = self._generate(model_to_use, prompt, response_format=response_format, num_predict=256)
        if raw_text is None:
            self._emit_log(
                f"INFO llm_proposal | model_used={model_to_use} | suggested=None | params={{}} | valid=False"
            )
            return None

        parsed = self._parse_response(raw_text, available_models)
        if parsed is None:
            self.consecutive_invalid_responses += 1
        else:
            self.consecutive_invalid_responses = 0

        suggested_model = None if parsed is None else parsed["model_name"]
        suggested_params = {} if parsed is None else parsed["params"]
        self._emit_log(
            f"INFO llm_proposal | model_used={model_to_use} | suggested={suggested_model} | "
            f"params={json.dumps(suggested_params, sort_keys=True)} | valid={parsed is not None}"
        )
        return parsed

    def propose_ensemble(
        self,
        top_trials: list[dict[str, Any]],
        available_models: dict[str, dict[str, Any]],
    ) -> list[dict[str, Any]] | None:
        """Request up to three stacking candidates from the smart model."""
        prompt_lines = [
            "/no_think",
            "Pick the best 3 model configs for a stacking ensemble.",
            "Choose only from the provided top trials.",
            "Top trials:",
        ]
        if top_trials:
            for trial in top_trials:
                prompt_lines.append(
                    f"Trial {trial.get('trial_number', 'n/a')}: {trial.get('model_name', 'unknown')} | "
                    f"RMSE={self._format_metric(trial.get('rmse'))} | "
                    f"R2={self._format_metric(trial.get('r2'))} | "
                    f"composite={self._format_metric(trial.get('composite_score'))} | "
                    f"{trial.get('validation_verdict', 'UNKNOWN')} | "
                    f"params={json.dumps(trial.get('params', {}), sort_keys=True)}"
                )
        else:
            prompt_lines.append("No top trials available.")

        prompt_lines.append("Available models and parameter ranges:")
        for model_name, model_config in available_models.items():
            prompt_lines.append(f"{model_name}: {self._format_search_space(model_config.get('search_space', {}))}")
        prompt_lines.append(
            'Return ONLY a compact JSON array on a single line with no extra spaces. No explanation. No markdown. '
            'Format: [{"model_name": "XGBRegressor", "params": {"n_estimators": 300}}]'
        )

        raw_text = self._generate(
            self.smart_model,
            "\n".join(prompt_lines),
            response_format=self._build_ensemble_schema(available_models),
            num_predict=512,
        )
        if raw_text is None:
            return None

        try:
            match = JSON_ARRAY_PATTERN.search(raw_text.strip())
            if match is None:
                self.consecutive_invalid_responses += 1
                return None
            payload = json.loads(match.group(0))
        except Exception:
            self.consecutive_invalid_responses += 1
            return None

        if not isinstance(payload, list):
            self.consecutive_invalid_responses += 1
            return None

        validated: list[dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()
        for item in payload:
            suggestion = self._validate_suggestion(item, available_models)
            if suggestion is None:
                continue
            signature = (
                suggestion["model_name"],
                json.dumps(suggestion["params"], sort_keys=True, separators=(",", ":")),
            )
            if signature in seen:
                continue
            seen.add(signature)
            validated.append(suggestion)
            if len(validated) == 3:
                break

        if not validated:
            self.consecutive_invalid_responses += 1
            return None

        self.consecutive_invalid_responses = 0
        return validated

    def _generate(
        self,
        model_name: str,
        prompt: str,
        response_format: Any | None = None,
        num_predict: int = 256,
    ) -> str | None:
        """Call the Ollama generate endpoint and return raw text."""
        try:
            payload = {
                "model": model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.2,
                    "num_predict": num_predict,
                },
            }
            if response_format is not None:
                payload["format"] = response_format
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
            payload = response.json()
            raw_text = payload.get("response") or payload.get("thinking", "")
            return raw_text if isinstance(raw_text, str) else str(raw_text)
        except Exception:
            return None

    def _validate_suggestion(
        self,
        payload: Any,
        available_models: dict[str, dict[str, Any]],
    ) -> dict[str, Any] | None:
        """Validate a single {model_name, params} object against the search space."""
        if not isinstance(payload, dict):
            return None

        model_name = payload.get("model_name")
        params = payload.get("params")
        if not isinstance(model_name, str) or model_name not in available_models:
            return None
        if not isinstance(params, dict):
            return None

        search_space = available_models[model_name].get("search_space", {})
        normalized_params: dict[str, Any] = {}
        for param_name, param_value in params.items():
            if not isinstance(param_name, str) or param_name not in search_space:
                return None
            normalized_value = self._validate_param_value(param_value, search_space[param_name])
            if normalized_value is INVALID_PARAM:
                return None
            normalized_params[param_name] = normalized_value

        return {
            "model_name": model_name,
            "params": normalized_params,
        }

    def _validate_param_value(self, value: Any, spec: dict[str, Any]) -> Any:
        """Validate and normalize one parameter value for the configured search space."""
        parameter_type = spec.get("type")
        if parameter_type == "int":
            coerced = self._coerce_number(value)
            if coerced is None or coerced != int(coerced):
                return INVALID_PARAM
            normalized = int(coerced)
            low = int(spec["low"])
            high = int(spec["high"])
            step = int(spec.get("step", 1))
            if normalized < low or normalized > high:
                return INVALID_PARAM
            if step > 0 and (normalized - low) % step != 0:
                return INVALID_PARAM
            return normalized

        if parameter_type == "float":
            coerced = self._coerce_number(value)
            if coerced is None:
                return INVALID_PARAM
            normalized = float(coerced)
            low = float(spec["low"])
            high = float(spec["high"])
            if normalized < (low - 1e-12) or normalized > (high + 1e-12):
                return INVALID_PARAM
            return normalized

        if parameter_type == "categorical":
            choices = list(spec.get("choices", []))
            return value if value in choices else INVALID_PARAM

        return INVALID_PARAM

    @staticmethod
    def _coerce_number(value: Any) -> float | None:
        """Convert numeric-like values into Python numbers."""
        if isinstance(value, bool):
            return None
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                return None
            try:
                return float(stripped)
            except ValueError:
                return None
        return None

    @staticmethod
    def _format_metric(value: Any) -> str:
        """Render metric values consistently for prompts."""
        if value is None:
            return "n/a"
        try:
            return f"{float(value):.4f}"
        except (TypeError, ValueError):
            return "n/a"

    def _format_search_space(self, search_space: dict[str, dict[str, Any]]) -> str:
        """Render a model search space in a compact single-line format."""
        parts: list[str] = []
        for name, spec in search_space.items():
            parameter_type = spec.get("type")
            if parameter_type == "int":
                step = int(spec.get("step", 1))
                parts.append(f"{name}=int[{spec['low']},{spec['high']},step={step}]")
            elif parameter_type == "float":
                suffix = ",log" if spec.get("log", False) else ""
                parts.append(f"{name}=float[{spec['low']},{spec['high']}{suffix}]")
            elif parameter_type == "categorical":
                choices = ",".join(self._format_choice(choice) for choice in spec.get("choices", []))
                parts.append(f"{name}=categorical[{choices}]")
            else:
                parts.append(f"{name}=unknown")
        return "; ".join(parts)

    @staticmethod
    def _format_choice(value: Any) -> str:
        """Render categorical values without Python-specific formatting."""
        if value is None:
            return "null"
        if isinstance(value, str):
            return value
        return json.dumps(value)

    def _build_proposal_schema(self, available_models: dict[str, dict[str, Any]]) -> dict[str, Any]:
        """Build a JSON schema constraining single-model proposals."""
        variants = [self._build_model_variant_schema(model_name, model_config) for model_name, model_config in available_models.items()]
        if len(variants) == 1:
            return variants[0]
        return {"oneOf": variants}

    def _build_ensemble_schema(self, available_models: dict[str, dict[str, Any]]) -> dict[str, Any]:
        """Build a JSON schema constraining ensemble proposal arrays."""
        return {
            "type": "array",
            "minItems": 1,
            "maxItems": 3,
            "items": self._build_proposal_schema(available_models),
        }

    def _build_model_variant_schema(self, model_name: str, model_config: dict[str, Any]) -> dict[str, Any]:
        """Build a schema variant for a single model family."""
        search_space = model_config.get("search_space", {})
        property_schema = {
            param_name: self._build_param_schema(param_spec)
            for param_name, param_spec in search_space.items()
        }
        return {
            "type": "object",
            "properties": {
                "model_name": {
                    "type": "string",
                    "const": model_name,
                },
                "params": {
                    "type": "object",
                    "properties": property_schema,
                    "additionalProperties": False,
                },
            },
            "required": ["model_name", "params"],
            "additionalProperties": False,
        }

    def _build_param_schema(self, spec: dict[str, Any]) -> dict[str, Any]:
        """Build a JSON schema fragment for one search-space parameter."""
        parameter_type = spec.get("type")
        if parameter_type == "int":
            schema = {
                "type": "integer",
                "minimum": int(spec["low"]),
                "maximum": int(spec["high"]),
            }
            step = int(spec.get("step", 1))
            if step > 0:
                schema["multipleOf"] = step
            return schema

        if parameter_type == "float":
            return {
                "type": "number",
                "minimum": float(spec["low"]),
                "maximum": float(spec["high"]),
            }

        if parameter_type == "categorical":
            enum_values = []
            for choice in spec.get("choices", []):
                enum_values.append(choice)
            return {"enum": enum_values}

        return {}

    @staticmethod
    def _emit_log(message: str) -> None:
        """Log visibly even when the stdlib logging system is not configured."""
        root_logger = logging.getLogger()
        if root_logger.handlers:
            root_logger.info(message)
            return
        print(message)
