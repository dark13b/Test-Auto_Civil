"""Single responsibility: saving model weights, training logs, and exported results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def save_training_artifacts(model: Any, metrics: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "training_metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True),
        encoding="utf-8",
    )
