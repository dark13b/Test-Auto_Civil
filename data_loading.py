"""Single responsibility: dataset loading, tokenizer setup, dataloaders, and splits."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import pandas as pd

from train_impl import load_dataset, split_dataset


@dataclass
class DataSplit:
    features: pd.DataFrame
    target: pd.Series


def build_dataloaders(config: dict[str, Any]) -> SimpleNamespace:
    frame = load_dataset(config)
    x_train, x_val, x_test, y_train, y_val, y_test = split_dataset(frame, config)
    return SimpleNamespace(
        train=DataSplit(x_train, y_train),
        val=DataSplit(x_val, y_val),
        test=DataSplit(x_test, y_test),
        frame=frame,
    )
