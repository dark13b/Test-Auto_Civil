# AutoCivil-Lab Final Project Report

Date: March 11, 2026

## Scope

This report summarizes the full AutoCivil-Lab project state from the codebase and output artifacts present in the repository on March 11, 2026.

The report uses the current saved model and refreshed downstream artifacts from:

- `report.py`
- `uncertainty.py`
- `design_tool.py --target 35`
- `design_tool.py --batch 25,30,35,40,45`

Important note:

- `best_search_result.json` and `best_search_model.pkl` are newer than `optuna_results.csv` and parts of `research_log.txt`.
- The best-model conclusions in this report therefore treat `best_search_result.json` and refreshed downstream outputs as authoritative for current model performance.
- The search tracking CSV is still included, but it is called out separately as inconsistent.

## Executive Summary

AutoCivil-Lab is a local, non-LLM, autonomous experimentation pipeline for concrete compressive strength prediction. It combines dataset preparation, domain-specific feature engineering, baseline training, Optuna-driven search, engineering validation, uncertainty estimation, inverse mix design, plot generation, and a Flask dashboard.

Current overall status:

- The project is operational end to end.
- The current best saved model is `LGBMRegressor`.
- The baseline `RandomForestRegressor` was improved on both cross-validation and holdout performance.
- Engineering validation no longer hard-fails the saved best model, but the overall verdict remains `WARN` because many holdout samples still violate durability-oriented and anomaly-oriented rules.
- Uncertainty estimation is functioning, but observed conformal coverage is below the target 90 percent after the latest refresh.
- Inverse design is functioning and produces `PASS` designs, but performance is stronger for 30-45 MPa than for the 25 MPa target.
- Search bookkeeping is not fully reproducible right now because `optuna_results.csv` is stale relative to the latest saved best model.

## Project Objective

The project goal is to adapt the "propose, run, evaluate, keep only improvements, repeat" autoresearch loop to civil engineering regression without using an LLM. In this implementation:

- the task is concrete compressive strength prediction
- candidate experiments are proposed by Optuna
- candidate models are trained and evaluated automatically
- engineering rules screen results for plausibility and durability concerns
- only improved models are retained

## System Overview

Primary components:

- `generate_data.py`: loads local, real, or synthetic data and writes the normalized training dataset
- `feature_engineering.py`: builds civil-engineering-derived features such as binder ratios, age interactions, and paste proxies
- `train.py`: trains the baseline model, computes metrics, and applies engineering validation
- `search.py`: runs the Optuna search loop and updates the saved best model
- `validator.py`: applies engineering plausibility, durability, and anomaly rules
- `report.py`: creates plots and consolidated final metrics
- `uncertainty.py`: builds conformal and quantile interval estimates
- `design_tool.py`: performs inverse mix design search for target strengths
- `dashboard.py`: serves the project dashboard and output browser

## Data and Configuration

Dataset source:

- Mode: `local_file`
- Input workbook: `concrete_combined.xlsx`
- Sheet: `Combined`
- Rows in prepared dataset: `2060`
- Predictive target: `compressive_strength`
- Base input variables: `8`
- Total model features after engineering: `22`

Prepared dataset statistics:

| Metric | Value |
| --- | ---: |
| Minimum strength | 2.33 MPa |
| Mean strength | 35.818 MPa |
| Standard deviation | 16.702 MPa |
| Maximum strength | 82.60 MPa |

Key configuration choices:

- Train/test split: 80/20 with target stratification
- Cross-validation: 5-fold
- Random seed: 42
- Primary metric direction: minimize RMSE
- Composite score weights:
  - RMSE: 0.5
  - R2: 0.3
  - MAE: 0.2
- Search runtime policy:
  - minimum trials: 30
  - minimum runtime floor: 50 minutes
- Uncertainty method: conformal
- Design tool objective: satisfy target strength while minimizing cement content

## Feature Engineering

The project expands the original mix variables with engineering-derived features that encode binder chemistry, workability balance, and age-strength interaction.

Engineered features include:

- `water_cement_ratio`
- `water_binder_ratio`
- `slag_replacement_ratio`
- `fly_ash_replacement_ratio`
- `aggregate_paste_ratio`
- `fine_to_coarse_ratio`
- `total_binder`
- `supplementary_replacement_ratio`
- `superplasticizer_binder_ratio`
- `paste_volume_proxy`
- `log_age`
- `cement_age_interaction`
- `binder_age_interaction`
- `water_binder_age`

Relative top split-importance features from the current saved LightGBM model:

| Rank | Feature |
| --- | --- |
| 1 | `cement_age_interaction` |
| 2 | `water_binder_age` |
| 3 | `binder_age_interaction` |
| 4 | `coarse_aggregate` |
| 5 | `fine_aggregate` |
| 6 | `superplasticizer` |
| 7 | `water` |
| 8 | `water_cement_ratio` |
| 9 | `superplasticizer_binder_ratio` |
| 10 | `fine_to_coarse_ratio` |

Interpretation:

- age-interaction terms dominate the saved model
- binder-aware ratios contribute meaningfully
- raw aggregate and admixture quantities still matter after feature engineering

## Modeling Workflow

The modeling workflow is:

1. Prepare and validate the dataset.
2. Add engineered features.
3. Train a baseline model.
4. Split the dataset into train and holdout sets.
5. Evaluate each candidate model with cross-validation and holdout testing.
6. Apply engineering validation to holdout predictions.
7. Keep only improved candidates.
8. Save the best model and produce downstream reports.

Enabled model families in configuration:

- RandomForestRegressor
- GradientBoostingRegressor
- XGBRegressor
- LGBMRegressor
- Ridge
- SVR

## Performance Results

### Baseline vs Current Best

| Metric | Baseline RandomForest | Current Best LightGBM |
| --- | ---: | ---: |
| CV RMSE | 2.8975 | 2.3584 |
| CV MAE | 1.7660 | 1.0808 |
| CV R2 | 0.9694 | 0.9792 |
| CV Composite | 0.9405 | 0.9548 |
| Holdout RMSE | 2.4396 | 2.1374 |
| Holdout MAE | 1.4175 | 0.9043 |
| Holdout R2 | 0.9769 | 0.9823 |
| Holdout Composite | 0.9511 | 0.9598 |
| Validation Verdict | WARN | WARN |

Best saved model metadata:

- Model: `LGBMRegressor`
- Saved best trial number: `62`
- Hyperparameters:
  - `n_estimators=450`
  - `learning_rate=0.16661100531078857`
  - `num_leaves=18`
  - `min_child_samples=5`
  - `subsample=0.7744376792558894`
  - `colsample_bytree=0.6465075081212768`

### Improvement Summary

Measured against the baseline holdout model:

- RMSE improved by `12.39%`
- MAE improved by `36.20%`
- R2 improved by `0.5364` percentage points
- Composite score improved by `1.5185%`

Interpretation:

- The search achieved a real but controlled lift rather than a dramatic jump.
- The strongest gain is in absolute error reduction, especially MAE.
- The best saved model is better calibrated to the holdout set than the baseline.

### Performance by Strength Range

| Strength Range | Baseline RMSE | Best RMSE | Improvement |
| --- | ---: | ---: | ---: |
| Low | 1.4909 | 0.9518 | 36.16% |
| Mid | 2.7038 | 2.4489 | 9.43% |
| High | 2.3778 | 1.9991 | 15.93% |

Interpretation:

- The largest gain appears in the low-strength regime.
- High-strength prediction also improved materially.
- Mid-range performance improved, but less dramatically than the low and high ranges.

## Search Process Summary

Search tracking artifacts currently report:

- `165` rows in `optuna_results.csv`
- maximum CSV trial number: `165`
- maximum trial number mentioned in `research_log.txt`: `165`

CSV selection status counts:

| Status | Count |
| --- | ---: |
| `no_improvement` | 137 |
| `rejected_validation_fail` | 18 |
| `new_best` | 10 |

CSV validation verdict counts:

| Verdict | Count |
| --- | ---: |
| `WARN` | 147 |
| `FAIL` | 18 |

CSV model-family counts:

| Model Family | Count |
| --- | ---: |
| GradientBoosting | 132 |
| RandomForest | 11 |
| SVR | 11 |
| Ridge | 11 |

Best trial according to the stale CSV:

- Trial `47`
- Model family `GradientBoosting`
- Composite score `0.9533`

## Engineering Validation Results

Current validation summary for the saved best model:

- Verdict: `WARN`
- Holdout pass rate: `33.5%`
- Hard-failed samples: `0`
- Warning samples: `274`
- Durability cautions: `244`
- Dataset anomalies: high but non-failing, embedded in warning counts

Most frequent triggered warning rules for the saved best model:

| Rule | Count |
| --- | ---: |
| `water_cement_ratio_warn_exceeds_durability_limit` | 244 |
| `high_water_cement_ratio_with_high_strength_warn` | 81 |
| `scm_mix_water_binder_context_used` | 81 |
| `fly_ash_replacement_ratio_warn_too_high` | 34 |
| `total_binder_warn_too_high` | 30 |
| `early_age_high_strength_with_unfavorable_water_binder_ratio_warn` | 12 |
| `scm_mix_high_water_binder_ratio_warn` | 12 |
| `total_binder_warn_too_low` | 5 |

Interpretation:

- The validator is no longer blocking the best saved model with hard failures.
- The dominant issue is still durability-related high water/cement ratio behavior in the dataset.
- The validation subsystem is functioning as a screening and annotation layer rather than a hard rejection layer for the saved best model.

## Uncertainty Estimation

Method:

- Conformal prediction

Current calibration summary:

| Metric | Value |
| --- | ---: |
| Target coverage | 0.9000 |
| Observed coverage | 0.8689 |
| Mean interval width | 5.0046 MPa |
| Calibration sample count | 330 |
| Test sample count | 412 |
| Best bin coverage | 0.9762 |
| Worst bin coverage | 0.6829 |

Confidence label summary in refreshed final metrics:

- All `412` holdout predictions fell into `MODERATE`

Interpretation:

- The interval estimator is working mechanically.
- Coverage is below the configured 90 percent target after the latest refresh.
- Reliability is uneven across the prediction range, with at least one bin materially under-covered.

## Inverse Design Results

### Single 35 MPa Design

| Metric | Value |
| --- | ---: |
| Target strength | 35.00 MPa |
| Predicted strength | 34.7553 MPa |
| Validation verdict | PASS |
| Cement content | 204.448 kg/m3 |
| Water/cement ratio | 0.5882 |
| Estimated cement saving vs reference | 15.82% |

Interpretation:

- The 35 MPa inverse design is feasible.
- The returned design passes engineering validation.
- Cement saving is positive, but lower than the earlier stale artifact set.

### Batch Design Summary

| Target MPa | Predicted MPa | Verdict | Cement kg/m3 | Cement Saving % |
| --- | ---: | --- | ---: | ---: |
| 25 | 26.2201 | PASS | 372.463 | -53.37 |
| 30 | 29.1741 | PASS | 201.308 | 17.11 |
| 35 | 34.7553 | PASS | 204.448 | 15.82 |
| 40 | 38.9341 | PASS | 205.978 | 26.61 |
| 45 | 43.0701 | PASS | 208.543 | 30.87 |

Interpretation:

- The design tool performs reasonably from 30 MPa upward.
- The 25 MPa solution is technically valid but economically poor under the current search behavior because it increases cement relative to the reference baseline.

## Generated Deliverables

Model artifacts:

- `outputs/baseline_model.pkl`
- `outputs/best_search_model.pkl`

Metric artifacts:

- `outputs/baseline_metrics.json`
- `outputs/best_search_result.json`
- `outputs/final_metrics.json`
- `outputs/uncertainty_calibration.json`

Search and reporting artifacts:

- `outputs/optuna_results.csv`
- `outputs/research_log.txt`
- `outputs/search_progress.png`
- `outputs/actual_vs_predicted.png`
- `outputs/residuals_plot.png`
- `outputs/feature_importance.png`
- `outputs/performance_by_range.png`
- `outputs/uncertainty_plot.png`

Design artifacts:

- `outputs/design_35MPa.json`
- `outputs/batch_design_results.csv`

Product artifact:

- `dashboard.py` serving the local dashboard

## Risks and Open Issues

### 1. Search artifact inconsistency

This is the main reporting risk in the project right now.

Observed inconsistency:

- `best_search_result.json` says the best current model is `LGBMRegressor`, trial `62`
- `final_metrics.json` now matches that after refresh
- `optuna_results.csv` still has no `LGBMRegressor` rows and says the best CSV trial is GradientBoosting trial `47`

Most likely explanation:

- `search.py` updates `best_search_result.json` during the run
- `optuna_results.csv` is written only at the end
- if a run is interrupted, partially rerun, or artifacts are mixed across runs, the JSON and CSV diverge

Impact:

- the search leaderboard cannot currently be trusted as a single source of truth
- the dashboard table and some reporting views may disagree with the saved best model

### 2. Dependency version drift

While refreshing downstream outputs, the environment raised scikit-learn version warnings when loading saved pickles:

- pickled estimators were created under scikit-learn `1.7.2`
- the current environment loaded them under scikit-learn `1.8.0`

Impact:

- model deserialization currently works, but reproducibility is weaker than it should be
- future library changes may break artifact compatibility

### 3. Validation remains warning-heavy

The best model has no hard failures, but only about one-third of holdout samples receive `PASS`.

Impact:

- the model is statistically strong, but the dataset still contains many mixes that trigger engineering concerns
- downstream adoption should treat predictions as engineering-assistance output, not automatic acceptance decisions

### 4. Uncertainty under-covers the target

Observed conformal coverage is `0.8689`, below the `0.9000` target.

Impact:

- interval widths may be too narrow for some regions of the strength range
- the uncertainty layer is useful, but not fully calibrated yet

### 5. Low-target design optimization is not yet robust

The 25 MPa design is valid but uses substantially more cement than the reference.

Impact:

- the current cost proxy and candidate search do not consistently favor economical low-strength solutions

## Recommendations

### Immediate

1. Fix search artifact synchronization so `optuna_results.csv`, `research_log.txt`, and `best_search_result.json` are always written atomically or checkpointed together.
2. Pin dependency versions in `requirements.txt` or add a lockfile to eliminate pickle compatibility drift.
3. Regenerate the search CSV from a full clean search run, or rerun `search.py` to completion once the export logic is tightened.

### Short term

1. Add artifact metadata to every output file:
   - generation timestamp
   - git commit hash if available
   - model id
   - config hash
2. Add a post-search consistency check that fails if:
   - the best trial in JSON is missing from CSV
   - the saved best model family is absent from the CSV export
3. Recalibrate conformal uncertainty after any new best model is saved and flag coverage shortfalls automatically.

### Medium term

1. Improve low-target inverse design by adding:
   - a softer penalty for over-strength
   - target-dependent economic regularization
   - stronger priors for low-strength mixes
2. Add dataset profiling reports for rule violations before training so validator behavior is visible before the search loop starts.
3. Expand the report layer to distinguish:
   - statistical error
   - engineering durability warnings
   - dataset anomaly warnings

## Final Assessment

AutoCivil-Lab is a successful v1 research automation project for civil-engineering regression.

What is working well:

- automated dataset preparation
- meaningful engineering feature engineering
- solid baseline and improved best-model performance
- domain-aware validation
- usable uncertainty estimation
- functioning inverse design
- a usable local dashboard and artifact set

What still needs hardening:

- artifact consistency and checkpointing
- dependency reproducibility
- uncertainty calibration quality
- low-strength design optimization quality

Bottom line:

- The project is technically credible and useful in its current form.
- The best saved model is materially better than the baseline.
- The system is ready for the next round of engineering hardening rather than a conceptual rewrite.
