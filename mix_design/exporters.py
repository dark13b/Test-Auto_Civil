"""Serialization helpers for canonical mix design comparison results."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from mix_design.contracts import ScenarioComparisonResult


def comparison_result_to_dict(result: ScenarioComparisonResult) -> dict[str, Any]:
    """Convert the typed comparison contract into a JSON-ready dictionary."""

    return asdict(result)
