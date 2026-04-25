# Knowledge Gaps

## Highest Priority

### 1. Missing final holdout summary

- `outputs/final_holdout_evaluation.json` is referenced in the project docs but is not present in the active `outputs/` directory.
- This blocks a clean current-state statement for the terminal holdout evaluation.

### 2. Dataset lineage needs row-level reconciliation

- `concrete_combined.xlsx` has 2060 rows.
- `concrete_data.csv` has 1030 rows.
- `data/concrete_data.csv` has 1309 rows after harmonization, duplicate removal, and feature engineering.
- The wiki still needs a row-level explanation for which records were duplicated across sheets and which records were removed or collapsed.

### 3. Canonical model state is split across sources

- The older reports describe a `LGBMRegressor` best model.
- The live artifacts describe a `RandomForestRegressor` best model.
- A durable knowledge base needs to preserve both facts, but also mark which one is the current live state.

### 4. Written benchmark section is not aligned with the live configuration

- `ACADEMIC_BENCHMARK_COMPARISON_SECTION.md` is a template with placeholders.
- It describes a repeated `5 x 3` CV protocol and many model families, but the current live config uses a narrower and different operating setup.

### 5. No source papers have been ingested yet

- `raw/papers/` is empty.
- The repository currently has project reports and artifacts, but not external journal or conference sources.
- This limits literature synthesis and contradiction tracking.

## Medium Priority

### 6. Validator assumptions need a literature cross-check

- The validator encodes durability and plausibility heuristics.
- The wiki should eventually trace those heuristics to the civil engineering standards or review literature they approximate.

### 7. Artifact provenance needs cleanup

- Multiple output files exist with overlapping roles, including selection-time, report-time, and archived states.
- The current wiki records the important ones, but a future pass should define a stricter canonical artifact policy.

### 8. The benchmark history needs a timeline page

- The current wiki records the reports and the live artifacts separately.
- A next pass should add a dated timeline of how the reported best model changed over time.

## Derived Priority Order

1. Recover or explain the missing holdout report.
2. Reconcile the dataset lineage and deduplication path.
3. Separate current canonical artifacts from stale report text.
4. Ingest external civil engineering literature into `raw/papers/`.

