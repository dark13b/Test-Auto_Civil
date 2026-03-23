"""Run AutoCivil-Lab with Ollama restricted to Qwen3 4B/8B models."""

from __future__ import annotations

import copy
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import train


PROJECT_ROOT = Path(__file__).resolve().parent
FAST_MODEL = "qwen3:4b"
SMART_MODEL = "qwen3:8b"


def _reexec_into_project_venv() -> None:
    """Prefer the project's virtualenv interpreter when it exists."""
    venv_python = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
    if not venv_python.exists():
        return

    current_python = Path(sys.executable).resolve()
    if current_python == venv_python.resolve():
        return

    completed = subprocess.run(
        [str(venv_python), str(PROJECT_ROOT / "run_qwen_only.py"), *sys.argv[1:]],
        check=False,
    )
    raise SystemExit(int(completed.returncode))


def _build_qwen_only_config() -> dict[str, Any]:
    """Overlay the project config with a Qwen-only Ollama backend."""
    config = copy.deepcopy(train.load_config())
    llm_config = config.setdefault("llm", {})
    ollama_config = llm_config.setdefault("ollama", {})

    llm_config["enabled"] = True
    llm_config["backend_mode"] = "ollama"
    llm_config["allow_deterministic_fallback"] = True
    llm_config["default_local_proposal_model"] = FAST_MODEL
    llm_config["compact_prompt_models"] = [FAST_MODEL]
    llm_config["include_no_think_directive"] = False
    llm_config["qwen_thinking_mode"] = True

    ollama_config["model"] = FAST_MODEL
    ollama_config["fast_model"] = FAST_MODEL
    ollama_config["smart_model"] = SMART_MODEL

    return config


def _install_qwen_only_config(config: dict[str, Any]) -> None:
    """Patch module-level config loaders so the run stays Qwen-only."""

    def _patched_load_config() -> dict[str, Any]:
        return copy.deepcopy(config)

    train.load_config = _patched_load_config

    import benchmark
    import report
    import research_loop
    import uncertainty

    benchmark.load_config = _patched_load_config
    report.load_config = _patched_load_config
    research_loop.load_config = _patched_load_config
    uncertainty.load_config = _patched_load_config


def main() -> int:
    """Run the standard research loop with a Qwen-only local backend."""
    _reexec_into_project_venv()
    qwen_config = _build_qwen_only_config()
    _install_qwen_only_config(qwen_config)

    os.environ["AUTOCIVIL_LLM_BACKEND"] = "ollama"
    os.environ["AUTOCIVIL_LLM_FAST_MODEL"] = FAST_MODEL
    os.environ["AUTOCIVIL_LLM_SMART_MODEL"] = SMART_MODEL

    import research_loop

    print(f"Running AutoCivil-Lab with Ollama models: fast={FAST_MODEL}, smart={SMART_MODEL}")
    return int(research_loop.main())


if __name__ == "__main__":
    sys.exit(main())
