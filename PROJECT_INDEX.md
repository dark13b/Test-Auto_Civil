# AutoCivil-Lab Project Index

Generated: 2026-03-11
Workspace: `E:\Random IDEA\AutoResearch\auto-civil-lab`

## Current State

- Task: concrete compressive strength regression with governed research, uncertainty estimation, validation, and inverse design.
- Active data mode: `local_file` from `concrete_combined.xlsx`, sheet `Combined`.
- Current saved best model: `LGBMRegressor`, trial `62`, holdout `R2 = 0.9822873303967959`.
- Current uncertainty audit: global coverage `0.9004854368932039`, audit `PASS`.
- Current validator verdict for saved best model: `WARN`, hard fails `0`.
- Current low-target design check: `25 MPa -> 238.9098322267276 kg/m3 cement`.

## Runtime Flow

1. `generate_data.py`
   Normalizes the workbook or downloaded dataset and writes `data/concrete_data.csv`.
2. `train.py`
   Builds the baseline model, saves baseline metrics, and provides shared training, split, scoring, and artifact I/O helpers.
3. `research_loop.py`
   Runs the governed scout-confirm-keep loop, appends research rows to `outputs/research_results.csv`, updates `outputs/best_search_result.json`, ratchets accepted state into `research_lab.py`, and validates final artifacts.
4. `uncertainty.py`
   Builds conformal and optional quantile intervals, then writes `outputs/uncertainty_calibration.json`.
5. `report.py`
   Regenerates plots and writes `outputs/final_holdout_evaluation.json` from the saved artifacts.
6. `design_tool.py`
   Uses the saved best model to search for feasible inverse mix designs.
7. `dashboard.py`
   Serves the generated artifacts via Flask.

## Module Index

| File | Size | Role | Key symbols | Primary outputs |
| --- | ---: | --- | --- | --- |
| `generate_data.py` | 259 lines | Dataset ingest and normalization | `load_local_dataset`, `download_real_dataset`, `generate_synthetic_dataset`, `save_dataset`, `main` | `data/concrete_data.csv` |
| `feature_engineering.py` | 138 lines | Derived mix and binder features | `build_engineering_features`, `validate_features` | in-memory engineered columns |
| `train.py` | 641 lines | Core ML utilities, baseline training, persistence | `split_dataset`, `cross_validate_model`, `instantiate_model`, `evaluate_candidate`, `save_pickle_artifact`, `load_pickle_artifact`, `main` | `outputs/baseline_model.pkl`, `outputs/baseline_metrics.json` |
| `validator.py` | 434 lines | Engineering rule engine over predictions and mixes | `EngineeringValidator.from_config`, `_evaluate_single_sample`, `validate_predictions` | embedded validation reports in JSON artifacts |
| `research_lab.py` | controlled surface | Research-editable experiment generation and kept-state ratchet | `scout_experiments`, `confirm_experiments`, `LAB_STATE` | accepted surface state in file |
| `research_protocol.py` | governance helpers | Brief parsing, memory, surface ratcheting, artifact validation | `load_human_research_brief`, `record_experiment_memory`, `validate_final_artifact_consistency` | governance JSON artifacts |
| `research_loop.py` | governed research loop | Scout/confirm execution and keep/revert orchestration | `run_engineering_research_loop`, `main` | `outputs/research_results.csv`, `outputs/experiment_memory.json`, `outputs/final_artifact_validation.json` |
| `uncertainty.py` | 447 lines | Conformal and quantile uncertainty estimation | `UncertaintyEstimator._fit_estimators`, `predict_with_interval`, `calibration_report`, `recalibrate_uncertainty_artifacts`, `main` | `outputs/uncertainty_calibration.json` |
| `report.py` | 358 lines | Plot and summary artifact generation | `create_search_progress_plot`, `create_uncertainty_plot`, `compute_feature_importance`, `main` | `outputs/final_holdout_evaluation.json`, plot PNGs |
| `design_tool.py` | 533 lines | Inverse mix design optimization | `MixDesignOptimizer.optimize`, `_sample_trial_mix`, `_evaluate_mix`, `_warm_start_mixes`, `batch_optimize`, `main` | `outputs/design_*MPa.json`, `outputs/batch_design_results.csv` |
| `dashboard.py` | 1309 lines | Flask dashboard and artifact API | `api_overview`, `api_research_log`, `api_optuna_results`, `api_status`, `index` | HTTP endpoints over `outputs/` |

## Dependency Map

- `generate_data.py` -> `feature_engineering.py`
- `train.py` -> `feature_engineering.py`, `validator.py`
- `research_lab.py` -> controlled by infrastructure, no domain-side dependencies
- `research_protocol.py` -> governance helpers only
- `research_loop.py` -> `research_lab.py`, `research_protocol.py`, `train.py`, `validator.py`, `uncertainty.py`, `report.py`
- `uncertainty.py` -> `train.py`, `feature_engineering.py`
- `report.py` -> `train.py`, `uncertainty.py`
- `design_tool.py` -> `train.py`, `validator.py`, `feature_engineering.py`, `optuna`
- `dashboard.py` -> reads JSON, CSV, PNG, and log artifacts only

## Model Families

- Baseline: `RandomForestRegressor`
- Search candidates: `RandomForestRegressor`, `GradientBoostingRegressor`, `XGBRegressor`, `LGBMRegressor`, `Ridge`, `SVR`
- Uncertainty methods: conformal by default, optional LightGBM quantile regression
- Inverse design objective: target matching plus cement-minimizing penalties and rule screening

## Artifact Inventory

### Data

- `data/concrete_data.csv`
- `concrete_combined.xlsx`
- `concrete_data.csv`
- `Concrete_DataUCI.xls`

### Model and research artifacts

- `outputs/baseline_model.pkl`
- `outputs/baseline_metrics.json`
- `outputs/best_search_model.pkl`
- `outputs/best_search_result.json`
- `outputs/research_results.csv`
- `outputs/optuna_results.csv`
- `outputs/research_log.txt`
- `outputs/experiment_memory.json`
- `outputs/proposal_diversity.json`
- `outputs/final_artifact_validation.json`
- `outputs/final_acceptance.json`

### Reporting and uncertainty

- `outputs/final_holdout_evaluation.json`
- `outputs/uncertainty_calibration.json`
- `outputs/search_progress.png`
- `outputs/actual_vs_predicted.png`
- `outputs/residuals_plot.png`
- `outputs/feature_importance.png`
- `outputs/performance_by_range.png`
- `outputs/uncertainty_plot.png`

### Inverse design

- `outputs/design_35MPa.json`
- `outputs/batch_design_results.csv`

## Entry Points

- `python generate_data.py`
- `python train.py`
- `python research_loop.py`
- `python research_loop.py --with-report`
- `python research_loop.py --validate-only`
- `python uncertainty.py`
- `python report.py`
- `python design_tool.py --target 35`
- `python design_tool.py --batch 25,30,35`
- `python dashboard.py`

## Operational Notes

- The governed loop writes scout and confirm rows incrementally and mirrors them to `optuna_results.csv` for backward-compatible reporting.
- `research_lab.py` is now the intended editable research surface; the core pipeline remains infrastructure-owned.
- Pickled model artifacts now embed metadata with package versions, save timestamp, model id, and config hash.
- Validator output is now categorized into `statistical_errors`, `durability_warnings`, and `dataset_anomalies`.
- Uncertainty calibration now uses per-bin conformal quantiles and writes an explicit coverage audit.
- Inverse design uses Optuna warm starts and target-dependent bounds for low, medium, and high strength regimes.

## Risks Still Present

- The repo tracks generated artifacts in `outputs/` and Python bytecode in `__pycache__/`, which creates noisy diffs and makes the Git history look like a trash fire.
- There is still no full end-to-end ML regression suite; verification remains helper-level plus import/compile checks unless the full loop is run.
- Model persistence still uses Python pickle, so artifacts must be treated as trusted-only and environment-coupled.
