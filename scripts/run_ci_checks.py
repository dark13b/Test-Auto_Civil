from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Keep lint and type checks scoped to the regression harness until the wider
# repository has a stable repo-wide baseline.
LINT_TARGETS = [
    "research_loop_helpers.py",
    "scripts/run_ci_checks.py",
    "tests/test_artifact_contracts.py",
    "tests/test_design_contracts.py",
    "tests/test_golden_run.py",
]

TYPE_TARGETS = [
    "research_loop_helpers.py",
    "scripts/run_ci_checks.py",
    "tests/test_golden_run.py",
]

CHECK_COMMANDS: dict[str, list[str]] = {
    "lint": [sys.executable, "-m", "ruff", "check", *LINT_TARGETS],
    "types": [
        sys.executable,
        "-m",
        "mypy",
        *TYPE_TARGETS,
        "--ignore-missing-imports",
        "--follow-imports=skip",
    ],
    "unit": [sys.executable, "-m", "pytest", "-m", "not contract and not golden", "-q"],
    "contract": [sys.executable, "-m", "pytest", "-m", "contract", "-q"],
    "golden": [sys.executable, "-m", "pytest", "-m", "golden", "-q"],
}


def _display_command(command: Sequence[str]) -> str:
    return shlex.join(command)


def _run_check(name: str, command: Sequence[str]) -> int:
    print(f"==> {name}", flush=True)
    print(f"$ {_display_command(command)}", flush=True)
    result = subprocess.run(command, cwd=REPO_ROOT)
    return result.returncode


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the same CI checks locally.")
    parser.add_argument(
        "--only",
        choices=list(CHECK_COMMANDS.keys()),
        help="Run only one named check.",
    )
    args = parser.parse_args()

    selected_checks = [args.only] if args.only else list(CHECK_COMMANDS.keys())
    for check_name in selected_checks:
        exit_code = _run_check(check_name, CHECK_COMMANDS[check_name])
        if exit_code != 0:
            print(f"{check_name} failed with exit code {exit_code}.", flush=True)
            return exit_code

    print("All requested checks passed.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
