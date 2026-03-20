import re
import unittest
from pathlib import Path

import pandas as pd

from train import split_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _read(path: str) -> str:
    return (PROJECT_ROOT / path).read_text(encoding="utf-8")


def _extract_function_block(source: str, function_name: str) -> str:
    match = re.search(
        rf"^def {function_name}\(.*?(?=^def |\Z)",
        source,
        flags=re.MULTILINE | re.DOTALL,
    )
    if match is None:
        raise AssertionError(f"Unable to locate function {function_name}")
    return match.group(0)


class HoldoutIntegrityTests(unittest.TestCase):
    def test_split_dataset_creates_train_validation_and_test_partitions(self) -> None:
        frame = pd.DataFrame(
            {
                "cement": range(40),
                "water": range(100, 140),
                "strength": [20 + (index % 10) for index in range(40)],
            }
        )
        config = {
            "task": {
                "input_columns": ["cement", "water"],
                "target_column": "strength",
            },
            "data": {
                "stratify_bins": 4,
                "test_size": 0.15,
            },
            "data_split": {
                "train_size": 0.70,
                "val_size": 0.15,
                "test_size": 0.15,
            },
            "experiment": {
                "random_seed": 42,
            },
        }

        partitions = split_dataset(frame, config)

        self.assertEqual(len(partitions), 6)
        if len(partitions) != 6:
            return

        x_train, x_val, x_test, y_train, y_val, y_test = partitions
        self.assertEqual(len(x_train) + len(x_val) + len(x_test), len(frame))
        self.assertEqual(len(y_train) + len(y_val) + len(y_test), len(frame))
        self.assertFalse(set(x_train.index) & set(x_val.index))
        self.assertFalse(set(x_train.index) & set(x_test.index))
        self.assertFalse(set(x_val.index) & set(x_test.index))

    def test_search_time_modules_do_not_reference_locked_test_partition(self) -> None:
        for path in ("research_loop.py", "uncertainty.py", "search.py"):
            contents = _read(path)
            self.assertNotRegex(contents, r"\bx_test\b", msg=path)
            self.assertNotRegex(contents, r"\by_test\b", msg=path)

        train_source = _read("train.py")
        for function_name in ("evaluate_candidate", "build_stacking_ensemble", "main"):
            block = _extract_function_block(train_source, function_name)
            self.assertNotRegex(block, r"\bx_test\b", msg=function_name)
            self.assertNotRegex(block, r"\by_test\b", msg=function_name)


if __name__ == "__main__":
    unittest.main()
