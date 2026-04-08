import re
import unittest
from pathlib import Path

import pandas as pd

from benchmark import rank_results
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
    def test_benchmark_ranking_uses_selection_metrics_not_holdout_metrics(self) -> None:
        ranked = rank_results(
            [
                {
                    "model_name": "ScientificallyValidWinner",
                    "display_name": "Scientifically Valid Winner",
                    "cv_composite": 0.92,
                    "validation_composite": 0.88,
                    "validation_rmse": 3.0,
                    "validation_mae": 2.0,
                    "validation_r2": 0.82,
                    "validation_verdict": "PASS",
                    "cv_rmse_std": 0.05,
                    "training_time_seconds": 1.0,
                    "holdout_rmse": 9.0,
                    "holdout_mae": 8.0,
                    "holdout_r2": 0.10,
                },
                {
                    "model_name": "HoldoutOnlyWinner",
                    "display_name": "Holdout Only Winner",
                    "cv_composite": 0.40,
                    "validation_composite": 0.35,
                    "validation_rmse": 8.0,
                    "validation_mae": 7.0,
                    "validation_r2": 0.20,
                    "validation_verdict": "PASS",
                    "cv_rmse_std": 0.20,
                    "training_time_seconds": 1.0,
                    "holdout_rmse": 0.1,
                    "holdout_mae": 0.1,
                    "holdout_r2": 0.99,
                },
            ]
        )

        self.assertEqual(ranked[0]["model_name"], "ScientificallyValidWinner")

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

    def test_benchmark_module_uses_validation_for_selection_and_holdout_only_for_final_winner(self) -> None:
        source = _read("benchmark.py")
        main_block = _extract_function_block(source, "main")
        self.assertIn("evaluate_benchmark_model(", main_block)
        self.assertIn("x_val,", main_block, "benchmark.main() must rank candidate models on x_val, not x_test")
        self.assertIn(
            "evaluate_final_holdout_winner(",
            main_block,
            "benchmark.main() must reserve x_test for one-time final winner reporting",
        )
        self.assertIn("x_test,", main_block, "benchmark.main() must pass x_test only to the final holdout helper")


if __name__ == "__main__":
    unittest.main()
