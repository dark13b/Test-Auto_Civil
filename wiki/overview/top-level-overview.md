# Top-Level Overview

## What This Repository Currently Is

This repository is an AutoCivil-Lab workspace centered on concrete compressive strength prediction, engineering-aware validation, uncertainty estimation, and inverse mix design.

The durable knowledge base should treat the following as the current source hierarchy:

1. Raw source files in the workspace
2. Derived dataset lineage in `data/`
3. Written project reports
4. Live JSON/CSV/PDF output artifacts
5. Wiki synthesis pages under `wiki/`

## Current Canonical State

| Topic | Current fact |
| --- | --- |
| Canonical live best-model artifact | `outputs/best_search_result.json` |
| Canonical live model | `RandomForestRegressor` |
| Canonical trial number | `24` |
| Canonical CV composite score | `0.8837009413159578` |
| Canonical CV RMSE | `4.873969592786195` |
| Canonical validation verdict | `WARN` |
| Hard fails | `0` |
| Warning count | `129` |
| Durability warnings | `90` |
| Dataset anomalies | `91` |
| Canonical acceptance | `false` in `outputs/final_acceptance.json` |
| Uncertainty coverage | `0.9796954314720813` in `outputs/uncertainty_calibration.json` |

## How To Read The Sources

- `wiki/materials/concrete-dataset.md` describes the physical dataset lineage and the row-count drift across source files.
- `wiki/methods/data-lineage-and-feature-engineering.md` describes how the dataset is harmonized and engineered.
- `wiki/literature/*.md` captures the written project reports and legacy design docs as sources, not as ground truth.
- `wiki/results/*.md` tracks the live output artifacts and the current metric state.
- `wiki/comparisons/report-vs-artifacts.md` records the contradictions that matter.

## Derived Interpretation

- The workspace has at least two incompatible knowledge layers: older written reports and newer live artifacts.
- The dataset lineage is not a simple one-file story. The workbook, the legacy CSV, and the engineered CSV each represent a different stage of processing.
- The strongest current evidence is the live JSON artifact set, not the older markdown reports.

## Open Questions

- What explains the discrepancy between the older LightGBM-based report and the current RandomForest-based artifact state?
- Is `data/concrete_data.csv` the canonical training dataset for the live pipeline, or just one intermediate snapshot?
- Why does the current dataset lineage collapse to 1309 unique rows after harmonization?
- Where is the missing `outputs/final_holdout_evaluation.json` in the current workspace?

