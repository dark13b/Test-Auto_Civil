# Academic Benchmark Comparison Section

Source: [ACADEMIC_BENCHMARK_COMPARISON_SECTION.md](<E:\Random IDEA\AutoResearch\auto-civil-lab\ACADEMIC_BENCHMARK_COMPARISON_SECTION.md>)

## Source Summary

This document is a manuscript-ready benchmark design and comparison template. It is valuable as a methodological proposal, but it is not itself an empirical results paper.

## Source Facts

- It frames the study as concrete compressive strength prediction from mix composition and curing age.
- It proposes a broad model family comparison including linear, kernel, bagging, boosting, and uncertainty-oriented models.
- It assumes 22 predictive inputs after engineering.
- It proposes a repeated `5 x 3` CV design and a single 80/20 holdout split.
- It leaves the result tables as placeholders rather than filled measurements.

## Research Objective

Define a fair, leakage-safe benchmark protocol that can compare a wide set of regression models on a concrete materials dataset.

## Materials and Mix Details

- The document uses the same base concrete variables as the repository pipeline.
- It explicitly names the engineered ratio and interaction features.

## Methodology

- Fixed train/validation/test protocol.
- Repeated cross-validation.
- Optuna-based hyperparameter tuning.
- Scaling only inside pipelines that need it.
- Holdout used once after model selection.
- Engineering validation treated as a secondary diagnostic.

## Key Results

- No empirical benchmark results are filled in.
- The tables are templates with placeholder cells such as `<rmse>` and `<best_params>`.

## Limitations

- It is a protocol document, not a finalized benchmark report.
- Its repeated `5 x 3` CV design does not match the current live config, which uses `cv_repeats = 1`.
- It should not be cited as evidence of observed performance.

## Relevance To The Current Research Direction

- Strong relevance as a benchmarking blueprint.
- Useful for understanding the intended methodological standards around fairness, leakage prevention, and model comparison.

## Related Pages

- [Data lineage and feature engineering](<E:\Random IDEA\AutoResearch\auto-civil-lab\wiki\methods\data-lineage-and-feature-engineering.md>)
- [Current artifacts](<E:\Random IDEA\AutoResearch\auto-civil-lab\wiki\results\current-artifacts.md>)
- [Feature catalog](<E:\Random IDEA\AutoResearch\auto-civil-lab\wiki\variables\feature-catalog.md>)

