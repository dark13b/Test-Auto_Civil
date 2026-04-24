# Log

Append-only record of wiki ingest and maintenance actions.

## 2026-04-24

- Polished the normal-mode mix design assistant candidate cards in `dashboard.py` and indexed the frontend-only change in `wiki/results/normal-mode-candidate-cards.md`.
- Added a concise `AutoCivil MVP` section to `README.md` covering the inverse mix design assistant scope, target users, workflow, dashboard launch path, normal-mode URL, internal experimental mode, lab validation disclaimer, and the expected trained model artifact dependency.
- Added `MVP_CLEANUP_NOTES.md` and indexed it from `wiki/index.md` as a short review list for stale or misleading files and claims.

## 2026-04-05

- Scanned the repository structure and identified the active research materials: `concrete_combined.xlsx`, `concrete_data.csv`, `data/concrete_data.csv`, the project reports, and the live `outputs/` artifacts.
- Created the required folder scaffold under `raw/` and `wiki/` without modifying any files in `raw/`.
- Extracted the current dataset lineage facts: the workbook has 2060 rows split across two source sheets, the legacy CSV has 1030 rows, and the engineered CSV in `data/` has 1309 rows and 23 columns.
- Captured the current artifact state: the live canonical model artifact is `RandomForestRegressor` in `outputs/best_search_result.json`, while the older markdown reports still describe a `LGBMRegressor` state.
- Built the first wiki pass with index, overview, materials, methods, variables, literature, results, comparisons, and outputs pages.
- Expanded the literature layer with source summaries for `README.md`, `PROJECT_INDEX.md`, and `research_brief.md`.
- Added the validator policy page to preserve the rule thresholds that drive the current `WARN` state in the live artifacts.
