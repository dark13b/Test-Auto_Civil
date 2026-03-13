"""Repository-aware proposal generation on top of provider backends."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from llm_backend import LLMBackend


JSON_OBJECT_PATTERN = re.compile(r"\{.*\}", re.DOTALL)
JSON_ARRAY_PATTERN = re.compile(r"\[\s*.*\s*\]", re.DOTALL)
INVALID_PARAM = object()


class ProposalEngine:
    """Generate structured research proposals while keeping execution in repository code."""

    def __init__(
        self,
        *,
        backend: LLMBackend,
        interaction_log_path: Path | None = None,
        log_interactions: bool = True,
        include_no_think_directive: bool = False,
    ) -> None:
        self.backend = backend
        self.backend_name = backend.backend_name
        self.interaction_log_path = interaction_log_path
        self.log_interactions = bool(log_interactions)
        self.include_no_think_directive = bool(include_no_think_directive)

    def is_available(self) -> bool:
        return self.backend.is_available()

    def generate_hypotheses(
        self,
        *,
        research_brief: dict[str, Any],
        current_best: dict[str, Any],
        experiment_memory: dict[str, Any],
        limit: int = 5,
        model_hint: str | None = None,
    ) -> list[str]:
        prompt = "\n".join(
            [
                "Generate concise ML research hypotheses as a JSON array of strings.",
                f"Goal: {research_brief.get('goal', 'Improve model quality.')}",
                f"Current best model: {current_best.get('model_name', 'unknown')}",
                f"Accepted experiments: {len(experiment_memory.get('accepted_experiments', []))}",
                f"Limit: {max(1, int(limit))}",
            ]
        )
        payload = self._run_interaction(
            task="hypotheses",
            prompt=prompt,
            response_format={
                "type": "array",
                "items": {"type": "string"},
                "minItems": 1,
                "maxItems": max(1, int(limit)),
            },
            model_hint=model_hint,
        )
        parsed = self._extract_json(payload.get("text", ""), expect_array=True)
        if not isinstance(parsed, list):
            return []
        return [str(item) for item in parsed if str(item).strip()][: max(1, int(limit))]

    def generate_feature_ideas(
        self,
        *,
        research_brief: dict[str, Any],
        current_best: dict[str, Any],
        limit: int = 5,
        model_hint: str | None = None,
    ) -> list[str]:
        prompt = "\n".join(
            [
                "Suggest feature-engineering ideas as a JSON array of strings.",
                "These are ideas only. They must not modify fixed validator or uncertainty code.",
                f"Goal: {research_brief.get('goal', 'Improve composite_score')}",
                f"Current best model: {current_best.get('model_name', 'unknown')}",
                f"Limit: {max(1, int(limit))}",
            ]
        )
        payload = self._run_interaction(
            task="feature_ideas",
            prompt=prompt,
            response_format={
                "type": "array",
                "items": {"type": "string"},
                "minItems": 1,
                "maxItems": max(1, int(limit)),
            },
            model_hint=model_hint,
        )
        parsed = self._extract_json(payload.get("text", ""), expect_array=True)
        if not isinstance(parsed, list):
            return []
        return [str(item) for item in parsed if str(item).strip()][: max(1, int(limit))]

    def generate_search_space_suggestions(
        self,
        *,
        available_models: dict[str, dict[str, Any]],
        research_brief: dict[str, Any],
        limit: int = 5,
        model_hint: str | None = None,
    ) -> list[dict[str, Any]]:
        prompt_lines = [
            "Suggest search-space refinements as a JSON array of objects.",
            "Each object must include model_name, parameter, and rationale.",
            f"Goal: {research_brief.get('goal', 'Improve composite_score')}",
            "Available models:",
        ]
        for model_name, model_config in available_models.items():
            prompt_lines.append(f"{model_name}: {self._format_search_space(model_config.get('search_space', {}))}")
        payload = self._run_interaction(
            task="search_space_suggestions",
            prompt="\n".join(prompt_lines),
            response_format={
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "model_name": {"type": "string"},
                        "parameter": {"type": "string"},
                        "rationale": {"type": "string"},
                    },
                    "required": ["model_name", "parameter", "rationale"],
                    "additionalProperties": False,
                },
                "minItems": 1,
                "maxItems": max(1, int(limit)),
            },
            model_hint=model_hint,
        )
        parsed = self._extract_json(payload.get("text", ""), expect_array=True)
        if not isinstance(parsed, list):
            return []
        return [item for item in parsed if isinstance(item, dict)][: max(1, int(limit))]

    def generate_experiment_proposals(
        self,
        *,
        available_models: dict[str, dict[str, Any]],
        research_brief: dict[str, Any],
        current_best: dict[str, Any],
        experiment_memory: dict[str, Any],
        diversity_state: dict[str, Any],
        proposal_count: int,
        trial_history: list[dict[str, Any]] | None = None,
        search_progress: dict[str, Any] | None = None,
        model_hint: str | None = None,
    ) -> list[dict[str, Any]]:
        if not self.is_available():
            return []

        prompt = self._build_experiment_prompt(
            available_models=available_models,
            research_brief=research_brief,
            current_best=current_best,
            experiment_memory=experiment_memory,
            diversity_state=diversity_state,
            proposal_count=proposal_count,
            trial_history=trial_history or [],
            search_progress=search_progress or {},
        )
        schema = self._build_proposals_schema(available_models, proposal_count=max(1, int(proposal_count)))
        payload = self._run_interaction(
            task="experiment_proposals",
            prompt=prompt,
            response_format=schema,
            model_hint=model_hint,
        )
        parsed = self._extract_json(payload.get("text", ""), expect_array=True)
        if not isinstance(parsed, list):
            return []

        validated: list[dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()
        for item in parsed:
            suggestion = self._validate_proposal(item, available_models)
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
            if len(validated) >= max(1, int(proposal_count)):
                break
        return validated

    def generate_ensemble_selection(
        self,
        *,
        top_trials: list[dict[str, Any]],
        available_models: dict[str, dict[str, Any]],
        max_items: int = 3,
        model_hint: str | None = None,
    ) -> list[dict[str, Any]]:
        if not self.is_available():
            return []
        prompt_lines = [
            "Choose the best base models for a stacking ensemble.",
            "Return only a JSON array of validated model_name and params objects.",
            f"Max items: {max(1, int(max_items))}",
            "Top trials:",
        ]
        if top_trials:
            for trial in top_trials:
                prompt_lines.append(
                    f"{trial.get('model_name', 'unknown')} | composite={self._format_metric(trial.get('composite_score'))} | "
                    f"params={json.dumps(self._extract_trial_params(trial), sort_keys=True)}"
                )
        else:
            prompt_lines.append("No top trials available.")
        prompt_lines.append("Available models and search spaces:")
        for model_name, model_config in available_models.items():
            prompt_lines.append(f"{model_name}: {self._format_search_space(model_config.get('search_space', {}))}")
        payload = self._run_interaction(
            task="ensemble_selection",
            prompt="\n".join(prompt_lines),
            response_format={
                "type": "array",
                "items": self._build_proposal_variant_schema(available_models),
                "minItems": 1,
                "maxItems": max(1, int(max_items)),
            },
            model_hint=model_hint,
        )
        parsed = self._extract_json(payload.get("text", ""), expect_array=True)
        if not isinstance(parsed, list):
            return []
        validated: list[dict[str, Any]] = []
        for item in parsed:
            suggestion = self._validate_proposal(item, available_models)
            if suggestion is not None:
                validated.append(suggestion)
        return validated[: max(1, int(max_items))]

    def summarize_run(
        self,
        *,
        research_brief: dict[str, Any],
        final_metrics: dict[str, Any],
        acceptance: dict[str, Any],
        model_hint: str | None = None,
    ) -> dict[str, Any]:
        if not self.is_available():
            return {
                "backend": self.backend_name,
                "available": False,
                "summary": "",
            }
        prompt = "\n".join(
            [
                "Summarize this ML research run in concise JSON.",
                f"Goal: {research_brief.get('goal', 'Improve composite_score')}",
                f"Accepted: {acceptance.get('accepted')}",
                f"Best model: {acceptance.get('best_model_name', 'unknown')}",
                f"Improvement pct: {acceptance.get('measured_improvement_pct', 'n/a')}",
                f"Validation verdict: {final_metrics.get('validation_verdict', 'n/a')}",
            ]
        )
        payload = self._run_interaction(
            task="run_summary",
            prompt=prompt,
            response_format={
                "type": "object",
                "properties": {
                    "summary": {"type": "string"},
                },
                "required": ["summary"],
                "additionalProperties": False,
            },
            model_hint=model_hint,
        )
        parsed = self._extract_json(payload.get("text", ""), expect_array=False)
        if isinstance(parsed, dict) and isinstance(parsed.get("summary"), str):
            return {
                "backend": payload.get("backend", self.backend_name),
                "model": payload.get("model"),
                "available": True,
                "summary": parsed["summary"],
            }
        return {
            "backend": payload.get("backend", self.backend_name),
            "model": payload.get("model"),
            "available": True,
            "summary": payload.get("text", ""),
        }

    def _run_interaction(
        self,
        *,
        task: str,
        prompt: str,
        response_format: dict[str, Any] | None,
        model_hint: str | None,
    ) -> dict[str, Any]:
        system_prompt = (
            "You are a proposal engine for AutoCivil-Lab. "
            "Return compact JSON only. Do not include markdown or explanation."
        )
        if self.include_no_think_directive:
            prompt = f"/no_think\n{prompt}"
        try:
            payload = self.backend.generate_text(
                prompt,
                system_prompt=system_prompt,
                response_format=response_format,
                model=model_hint,
            )
            self._write_interaction_log(task=task, prompt=prompt, payload=payload, valid=True)
            return payload
        except Exception as exc:
            payload = {
                "backend": self.backend_name,
                "model": model_hint,
                "text": "",
                "error": str(exc),
            }
            self._write_interaction_log(task=task, prompt=prompt, payload=payload, valid=False)
            return payload

    def _write_interaction_log(
        self,
        *,
        task: str,
        prompt: str,
        payload: dict[str, Any],
        valid: bool,
    ) -> None:
        if not self.log_interactions or self.interaction_log_path is None:
            return
        self.interaction_log_path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "task": task,
            "valid": valid,
            "backend": payload.get("backend", self.backend_name),
            "model": payload.get("model"),
            "prompt": prompt,
            "text": payload.get("text", ""),
            "error": payload.get("error"),
        }
        with self.interaction_log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")

    def _build_experiment_prompt(
        self,
        *,
        available_models: dict[str, dict[str, Any]],
        research_brief: dict[str, Any],
        current_best: dict[str, Any],
        experiment_memory: dict[str, Any],
        diversity_state: dict[str, Any],
        proposal_count: int,
        trial_history: list[dict[str, Any]],
        search_progress: dict[str, Any],
    ) -> str:
        recent_trials = trial_history[-min(10, len(trial_history)) :] if trial_history else []
        lines = [
            "Generate ML experiment proposals for concrete compressive strength regression.",
            f"Goal: {research_brief.get('goal', 'Improve composite_score')}",
            f"Acceptance metric: {research_brief.get('acceptance_metric', 'composite_score')}",
            f"Minimum improvement pct: {self._format_metric(research_brief.get('min_improvement_pct'))}",
            f"Current best model: {current_best.get('model_name', 'unknown')}",
            f"Current best composite_score: {self._format_metric(current_best.get('composite_score'))}",
            f"Accepted experiments in memory: {len(experiment_memory.get('accepted_experiments', []))}",
            f"Requested proposals: {max(1, int(proposal_count))}",
        ]
        required_families = research_brief.get("required_model_families", [])
        if isinstance(required_families, list) and required_families:
            lines.append("Required model families: " + ", ".join(str(item) for item in required_families))
        historic_counts = diversity_state.get("historic_family_counts", {})
        if isinstance(historic_counts, dict) and historic_counts:
            lines.append("Historic proposal family counts: " + json.dumps(historic_counts, sort_keys=True))
        if search_progress:
            lines.append("Search progress: " + json.dumps(search_progress, sort_keys=True))
        if recent_trials:
            lines.append("Recent trials:")
            for trial in recent_trials:
                lines.append(
                    f"{trial.get('model_name', 'unknown')} | composite={self._format_metric(trial.get('composite_score'))} | "
                    f"verdict={trial.get('validation_verdict', 'UNKNOWN')} | params={json.dumps(self._extract_trial_params(trial), sort_keys=True)}"
                )
        lines.append("Available models and parameter ranges:")
        for model_name, model_config in available_models.items():
            lines.append(f"{model_name}: {self._format_search_space(model_config.get('search_space', {}))}")
        lines.append(
            "Each proposal must include model_name, params, proposal_family, and hypothesis. "
            "Do not repeat exact model-plus-parameter combinations."
        )
        return "\n".join(lines)

    def _build_proposals_schema(self, available_models: dict[str, dict[str, Any]], proposal_count: int) -> dict[str, Any]:
        return {
            "type": "array",
            "items": self._build_proposal_variant_schema(available_models),
            "minItems": 1,
            "maxItems": max(1, int(proposal_count)),
        }

    def _build_proposal_variant_schema(self, available_models: dict[str, dict[str, Any]]) -> dict[str, Any]:
        variants = []
        for model_name, model_config in available_models.items():
            param_schema = {
                param_name: self._build_param_schema(param_spec)
                for param_name, param_spec in model_config.get("search_space", {}).items()
            }
            variants.append(
                {
                    "type": "object",
                    "properties": {
                        "model_name": {"type": "string", "const": model_name},
                        "params": {
                            "type": "object",
                            "properties": param_schema,
                            "additionalProperties": False,
                        },
                        "proposal_family": {"type": "string"},
                        "hypothesis": {"type": "string"},
                    },
                    "required": ["model_name", "params", "proposal_family", "hypothesis"],
                    "additionalProperties": False,
                }
            )
        if len(variants) == 1:
            return variants[0]
        return {"oneOf": variants}

    def _build_param_schema(self, spec: dict[str, Any]) -> dict[str, Any]:
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
            return {"enum": list(spec.get("choices", []))}
        return {}

    def _validate_proposal(
        self,
        payload: Any,
        available_models: dict[str, dict[str, Any]],
    ) -> dict[str, Any] | None:
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
            "display_name": str(available_models[model_name].get("display_name", model_name)),
            "params": normalized_params,
            "proposal_family": str(payload.get("proposal_family", "llm-generated")),
            "hypothesis": str(payload.get("hypothesis", "LLM-generated suggestion")),
        }

    def _validate_param_value(self, value: Any, spec: dict[str, Any]) -> Any:
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
            if value == "null" and None in choices:
                return None
            return value if value in choices else INVALID_PARAM

        return INVALID_PARAM

    @staticmethod
    def _coerce_number(value: Any) -> float | None:
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
    def _extract_trial_params(trial: dict[str, Any]) -> dict[str, Any]:
        raw_params = trial.get("params", trial.get("hyperparameters", {}))
        if isinstance(raw_params, dict):
            return raw_params
        if isinstance(raw_params, str):
            try:
                parsed = json.loads(raw_params)
                return parsed if isinstance(parsed, dict) else {}
            except Exception:
                return {}
        return {}

    @staticmethod
    def _format_metric(value: Any) -> str:
        if value is None:
            return "n/a"
        try:
            return f"{float(value):.4f}"
        except (TypeError, ValueError):
            return "n/a"

    def _format_search_space(self, search_space: dict[str, dict[str, Any]]) -> str:
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
        if value is None:
            return "null"
        if isinstance(value, str):
            return value
        return json.dumps(value)

    def _extract_json(self, raw_text: str, *, expect_array: bool) -> Any:
        cleaned = raw_text.strip()
        if not cleaned:
            return None
        matcher = JSON_ARRAY_PATTERN if expect_array else JSON_OBJECT_PATTERN
        match = matcher.search(cleaned)
        if match is None:
            return None
        try:
            return json.loads(match.group(0))
        except Exception:
            return None
