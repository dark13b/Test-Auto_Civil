# Final Project Report

Source: [FINAL_PROJECT_REPORT.md](<E:\Random IDEA\AutoResearch\auto-civil-lab\FINAL_PROJECT_REPORT.md>)

## Source Summary

This report summarizes an earlier project state from March 11, 2026. It describes the AutoCivil-Lab pipeline as operational end to end and anchors the project around a `LGBMRegressor` best model.

## Source Facts

- Date on the report: March 11, 2026.
- Dataset source in the report: `concrete_combined.xlsx`, `Combined` sheet.
- Reported dataset size: 2060 rows, 8 base variables, 22 total features after engineering.
- Reported current best model: `LGBMRegressor`.
- Reported holdout R2: `0.9823`.
- Reported holdout RMSE: `2.1374`.
- Reported validator verdict: `WARN`.

## Research Objective

Improve compressive-strength prediction while keeping engineering validation active and preserving inverse-design usefulness.

## Materials and Mix Details

- Concrete mix variables are the classic cement, slag, fly ash, water, superplasticizer, coarse aggregate, fine aggregate, and age inputs.
- The report emphasizes binder ratios, age interactions, and paste proxies as derived features.

## Methodology

- Prepare the dataset.
- Add engineering features.
- Train a baseline model.
- Run search.
- Validate predictions with engineering rules.
- Produce uncertainty and inverse-design outputs.

## Key Results

The report claims the following best-model comparisons:

| Metric | Baseline RandomForest | Current Best LightGBM |
| --- | ---: | ---: |
| CV RMSE | 2.8975 | 2.3584 |
| CV MAE | 1.7660 | 1.0808 |
| CV R2 | 0.9694 | 0.9792 |
| Holdout RMSE | 2.4396 | 2.1374 |
| Holdout MAE | 1.4175 | 0.9043 |
| Holdout R2 | 0.9769 | 0.9823 |

## Limitations

- The report is now stale relative to the live `outputs/` artifacts.
- It does not match the current canonical `RandomForestRegressor` state.
- It should be preserved as historical evidence, not treated as current ground truth.

## Relevance To The Current Research Direction

- Useful as a historical benchmark snapshot.
- Useful for tracing how the narrative changed from a LightGBM-centered state to the current artifact state.
- Useful for comparing the older project summary style against the live artifact summaries.

## Related Pages

- [Report vs artifacts](<E:\Random IDEA\AutoResearch\auto-civil-lab\wiki\comparisons\report-vs-artifacts.md>)
- [Current artifacts](<E:\Random IDEA\AutoResearch\auto-civil-lab\wiki\results\current-artifacts.md>)
- [Concrete dataset lineage](<E:\Random IDEA\AutoResearch\auto-civil-lab\wiki\materials\concrete-dataset.md>)

