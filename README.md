# AutoCivil-Lab v1

AutoCivil-Lab now runs as an autoresearch-style engineering ML system for concrete compressive-strength regression. The repo uses a governed scout -> confirm -> keep/revert loop, with `research_lab.py` as the controlled research surface, `research_brief.md` as the human strategy input, and an internal LLM proposal stack that can run through Ollama/Qwen, OpenAI, or a hybrid fallback path.

## Official runtime

The official runtime for core research is `research_loop.py`.

```bash
python research_loop.py --with-report
```

- Use `search.py` only as a deprecated compatibility wrapper.
- Use `run_qwen_only.py` only when you specifically want the Qwen-only local backend wrapper.
- Use `benchmark.py` only for post-run evaluation, not for core research execution.

## What this system actually does

This system preserves the existing concrete-specific ML pipeline, feature engineering, validator, uncertainty estimation, and reporting stack. The infrastructure owns orchestration and artifact governance; the research surface owns what gets explored.

## How the loop maps to autoresearch

| autoresearch concept | AutoCivil-Lab equivalent |
| --- | --- |
| Baseline experiment | `train.py` output |
| Propose a change | `proposal_engine.py` via `llm_backend.py`, or `research_lab.py` fallback |
| Run experiment | CV training + evaluation |
| Measure result | composite score (`RMSE + R2 + MAE`) |
| Confirm promising changes | stronger repeat-based confirmation run |
| Keep only if better | ratchet accepted state into `research_lab.py` |
| Repeat | `research_loop.py` cycles |

Additional autoresearch-style controls now included:
- Human brief loaded from `research_brief.md` (front matter)
- Controlled editable research surface in `research_lab.py`
- Provider abstraction in `llm_backend.py`
- Structured proposal engine in `proposal_engine.py`
- Cross-run experiment memory in `outputs/experiment_memory.json`
- Proposal diversity tracking in `outputs/proposal_diversity.json`
- Optional per-trial runtime budget via `research.max_trial_seconds`
- Final artifact consistency validation in `outputs/final_artifact_validation.json`
- Final acceptance decision in `outputs/final_acceptance.json` based on `outputs/best_search_result.json`

## Project structure

```text
auto-civil-lab/
|-- data/
|-- outputs/
|-- config.yaml
|-- design_tool.py
|-- feature_engineering.py
|-- generate_data.py
|-- llm_backend.py
|-- proposal_engine.py
|-- research_brief.md
|-- research_lab.py
|-- research_loop.py
|-- validator.py
|-- train.py
|-- search.py
|-- benchmark.py
|-- report.py
|-- uncertainty.py
|-- requirements.txt
`-- README.md
```

## Installation

Use Python 3.10+ and install the dependencies:

```bash
pip install -r requirements.txt
```

For contributor checks and CI parity, install the dev tools as well:

```bash
pip install -r requirements.txt -r requirements-dev.txt
```

## CI checks

Run the same slices locally that CI runs on every push and pull request:

```bash
python scripts/run_ci_checks.py
```

If you only need one slice, run it directly:

```bash
python scripts/run_ci_checks.py --only lint
python scripts/run_ci_checks.py --only types
python scripts/run_ci_checks.py --only unit
python scripts/run_ci_checks.py --only contract
python scripts/run_ci_checks.py --only golden
```

The golden fixture lives under `tests/fixtures/golden_run/` and executes one deterministic `research_loop.py` cycle with:

- a tiny concrete dataset fixture
- LLM proposals disabled
- a fixed random seed
- reports disabled for speed

Lint and type checks are intentionally scoped to the regression harness files for now, because the broader repository does not yet have a clean repo-wide baseline.

## Contributor path

```bash
python generate_data.py
python train.py
python research_loop.py --with-report
python benchmark.py
python design_tool.py --target 35
```

If you are looking for the old `search.py` path, treat it as legacy compatibility only. New runtime work belongs in `research_loop.py`.

## Engineering Upgrades

### 1. Engineering feature engineering

`feature_engineering.py` adds mix-design ratios, binder chemistry proxies, and age-interaction terms such as:

- `water_cement_ratio`
- `water_binder_ratio`
- `slag_replacement_ratio`
- `fly_ash_replacement_ratio`
- `total_binder`
- `supplementary_replacement_ratio`
- `paste_volume_proxy`
- `cement_age_interaction`
- `binder_age_interaction`

`generate_data.py` now enriches the normalized dataset before saving `data/concrete_data.csv`, and the training/search/report stack consumes the enriched feature set automatically.

Run:

```bash
python generate_data.py
```

### 2. Extended engineering validator

`validator.py` now keeps the original strength-bound checks and adds ACI 318 / BS 8500 inspired warning rules for:

- durability-oriented `water_cement_ratio` limits
- unusually low `water_binder_ratio`
- low or high `total_binder`
- aggressive fly ash or slag replacement levels
- suspicious early-age high-strength claims

Validation reports now include `warn_reasons`, `rule_violations_by_sample`, and an `overall_verdict` of `PASS`, `WARN`, or `FAIL`.

### 3. Inverse design tool

`design_tool.py` turns the trained model into a mix-design search utility. It searches for mixes that satisfy a target compressive strength while minimizing cement content and screening candidates through the engineering validator.

Run:

```bash
python design_tool.py --target 35
python design_tool.py --batch 25,30,35,40,45
```

Outputs:

- `outputs/design_35MPa.json`
- `outputs/batch_design_results.csv`

### 4. Uncertainty quantification

`uncertainty.py` supports two interval methods:

- quantile LightGBM regression (`5th`, `50th`, `95th` percentiles)
- conformal prediction around the saved best model

`report.py` now adds `outputs/uncertainty_plot.png`, and `uncertainty.py` saves `outputs/uncertainty_calibration.json`.

Run:

```bash
python report.py
python uncertainty.py
```

## Output files

- `outputs/baseline_model.pkl`: saved baseline `RandomForestRegressor`.
- `outputs/baseline_metrics.json`: baseline CV metrics and engineering validation report.
- `outputs/best_search_model.pkl`: current best model after the search loop.
- `outputs/best_search_result.json`: canonical search-selection artifact with cross-validation and selection-validation metrics.
- `outputs/research_results.csv`: canonical per-experiment ledger for scout and confirm execution.
- `outputs/optuna_results.csv`: backward-compatible mirror for existing reports and dashboards.
- `outputs/research_log.txt`: timestamped scout/confirm research timeline.
- `outputs/final_holdout_evaluation.json`: canonical terminal holdout artifact written only by the final report step.
- `outputs/final_acceptance.json`: acceptance gate computed from `best_search_result.json` and `research_brief.md` thresholds.
- `outputs/experiment_memory.json`: cross-run memory of trial signatures, stages, and keep decisions.
- `outputs/proposal_diversity.json`: diversity summary for the governed research loop.
- `outputs/final_artifact_validation.json`: final artifact consistency report anchored to `best_search_result.json`.
- `outputs/search_progress.png`: trial composite scores versus baseline.
- `outputs/actual_vs_predicted.png`: holdout actual-vs-predicted scatter plot.
- `outputs/residuals_plot.png`: residual structure plot for the best model.
- `outputs/feature_importance.png`: feature-importance chart for the best model.
- `outputs/performance_by_range.png`: RMSE by low, mid, and high strength ranges.
- `outputs/uncertainty_plot.png`: interval width versus predicted strength with confidence bands.
- `outputs/uncertainty_calibration.json`: interval coverage, sharpness, and reliability summary.
- `outputs/design_35MPa.json`: single-target inverse mix design report.
- `outputs/batch_design_results.csv`: batch inverse-design output table.
- `outputs/benchmark_results.csv`: one row per benchmark model with CV, holdout, timing, validator, and reference-delta metrics.
- `outputs/benchmark_trials.csv`: per-model Optuna benchmark tuning records.
- `outputs/benchmark_range_metrics.csv`: low-, mid-, and high-strength RMSE by benchmark model.
- `outputs/benchmark_results.json`: canonical benchmark payload including the ranking, winner, and reference comparison.
- `outputs/benchmark_best_model.pkl`: saved best model from the academic benchmark run.
- `outputs/benchmark_best_result.json`: best benchmark result payload.
- `outputs/benchmark_uncertainty.json`: conformal uncertainty comparison between the benchmark winner and the current reference.
- `outputs/benchmark_comparison.md`: paper-ready markdown comparison against the current LightGBM reference.
- `outputs/benchmark_holdout_rmse.png`: ranked holdout RMSE plot for the academic benchmark.
- `outputs/benchmark_reference_deltas.png`: metric deltas versus the current LightGBM reference.
- `outputs/benchmark_strength_range_heatmap.png`: strength-range RMSE heatmap across benchmarked models.

## Configuration

All experiment settings are loaded from `config.yaml`, including:

- task name
- target and input columns
- primary metric and optimization direction
- composite score weights
- engineering validation bounds
- Optuna trial count
- cross-validation folds
- random seed
- data mode and download sources
- baseline model parameters
- search-space definitions for each model family
- governed research-loop settings in `research.*`
- provider selection and fallback in `llm.*`
- research brief path and acceptance thresholds (from `research_brief.md`)

## Human brief and research surface

`research_loop.py` reads YAML front matter from `research_brief.md` at startup:

```yaml
---
goal: Maximize composite_score with validator-safe predictions
acceptance_metric: composite_score
min_improvement_pct: 1.0
required_model_families:
  - LGBMRegressor
  - XGBRegressor
scout_candidates_per_cycle: 8
confirm_top_k: 2
---
```

- `required_model_families`: constrains the active proposal surface to those enabled families.
- `min_improvement_pct`: required measured gain for final acceptance.
- `research_lab.py`: the controlled research-editable file that defines scout and confirm behavior.
- final acceptance is written to `outputs/final_acceptance.json`, computed from `outputs/best_search_result.json`.

See [`docs/autoresearch_workflow.md`](docs/autoresearch_workflow.md) for the end-to-end governed loop.
See [`ARCHITECTURE_V2.md`](ARCHITECTURE_V2.md) for the canonical runtime contract and compatibility policy.

## LLM proposal modes

The repository now supports three proposal modes without changing Codex's role as the code-editing/orchestration agent:

- `ollama`: use the local Ollama endpoint first and then the `ollama` CLI if enabled
- `openai`: use the configured OpenAI model through the Responses API
- `hybrid`: use the configured primary backend and fall back to the secondary backend

Example `config.yaml` block:

```yaml
llm:
  enabled: true
  backend_mode: hybrid
  allow_deterministic_fallback: true
  default_local_proposal_model: qwen3:8b
  compact_prompt_models: [qwen3:4b]
  ollama:
    base_url: http://localhost:11434
    fast_model: qwen3:4b
    smart_model: qwen3:8b
    use_cli_fallback: true
  openai:
    model: gpt-5.1-mini
    api_key_env: OPENAI_API_KEY
  hybrid:
    primary: ollama
    fallback: openai
```

Proposal prompts now route by configured model capability instead of hardcoded prompt text. Models listed in `llm.compact_prompt_models` receive compact prompts with only the best summary, the last 5 relevant trials, current family-state blocks, allowed families, and strict JSON rules. Other models receive the richer prompt variant with a slightly wider history and family-state context.

Proposal execution now has a post-parse gate before any experiment runs. Exact duplicates, near-duplicates, saturated-family repeats, malformed payloads, and weak-family retries without meaningful novelty are rejected and logged. When the gate rejects a candidate, the proposer can regenerate once (`llm.enable_regeneration_on_reject` / `llm.max_regeneration_attempts`) and then falls back to the deterministic research surface.

What the LLM layer is allowed to do:
- generate hypotheses
- propose experiments
- suggest feature ideas
- suggest search-space adjustments
- summarize completed runs

What it is not allowed to do:
- bypass the validator
- bypass uncertainty/report generation
- override keep/revert decisions
- redefine the final artifact source of truth

`outputs/best_search_result.json` remains the selection-time source of truth even when LLM proposals are enabled, and `outputs/final_holdout_evaluation.json` remains terminal-only.

`outputs/llm_interactions.jsonl` now records the prompt variant, raw response/thinking channels, the final extracted text, `extracted_from_channel`, JSON repair usage, duplicate rejection metadata, regeneration attempts, fallback usage, and the final parsed candidate. This makes empty-response / thinking-text recoveries explicit instead of silent.

Family-state reasoning is shared between prompting and gating. Each proposal cycle summarizes:
- `strongest_active_family`
- `saturated_families`
- `underexplored_promising_families`
- `underexplored_weak_families`
- `temporarily_blocked_families`

Underexplored families are no longer encouraged on trial count alone; recent measured outcomes decide whether a family is still worth probing.

## Data modes

`generate_data.py` supports three modes through `config.yaml`:

- `local_file`: loads a local CSV or Excel workbook, applies the configured column mapping, and writes the normalized training dataset.
- `real`: downloads the UCI Concrete Compressive Strength dataset and stores it as `data/concrete_data.csv`.
- `synthetic`: generates a 1030-row engineering-inspired fallback dataset with realistic mix-design relationships.

If `real` mode fails and `fallback_to_synthetic_on_error: true`, the script automatically switches to the synthetic generator.

## Using your workbook

The current `config.yaml` is already set up to use `concrete_combined.xlsx` from the project root.

- `data.mode` is set to `local_file`
- `data.local_file.path` points to `concrete_combined.xlsx`
- `data.local_file.sheet_name` points to `Combined`
- `blast_furnace_slag` is renamed to `slag`
- `concrete_compressive_strength` is renamed to `compressive_strength`
- the extra `source` column is dropped automatically

To run the project on this workbook:

```bash
python generate_data.py
python train.py
python research_loop.py --with-report
python design_tool.py --target 35
```

## Engineering validator

`validator.py` excludes any model whose predictions fall outside the configured strength bounds or exceed the hard water/cement screening limit. Durability, workability, replacement-ratio, and early-age plausibility checks are reported as warnings and preserved in the search and design artifacts.

## Deployment

Free web hosts do not preserve local files written after deploy. For this dashboard, that means the `outputs/` artifacts shown by Flask must either be committed into the repo before deploy or regenerated during your release workflow.

### Railway

1. Push the project to GitHub with `outputs/` populated with the JSON, CSV, and PNG artifacts you want the team to see.
2. In Railway, create a new project and choose `Deploy from GitHub repo`.
3. Select this repository and let Railway detect the Python service.
4. In Railway Variables, set `PORT=5050` only if you want a local-style default; Railway will inject its own runtime `PORT` automatically.
5. Set `DASHBOARD_PASSWORD` to protect the dashboard with HTTP Basic Auth. The username is always `autocivil`.
6. Confirm the start command uses the `Procfile` entry: `gunicorn dashboard:app --bind 0.0.0.0:$PORT --workers 2`.
7. Deploy the service and wait for the first successful build.
8. Open `/health` to verify the app is live, then open `/` and log in with `autocivil` plus your `DASHBOARD_PASSWORD`.

### Render

1. Push the repository to GitHub with the dashboard artifacts already present in `outputs/`.
2. In Render, create a new Web Service and connect the repository.
3. Choose the included `render.yaml`, or manually set the environment to `Python`.
4. Keep the build command as `pip install -r requirements.txt`.
5. Keep the start command as `gunicorn dashboard:app --bind 0.0.0.0:$PORT`.
6. Add the environment variable `DASHBOARD_PASSWORD` if you want the dashboard protected. The username remains `autocivil`.
7. Deploy the service and wait for Render to finish building the container.
8. Visit `/health` first to confirm `{"status":"ok"}`, then open the main dashboard URL and verify the `outputs/` artifacts load correctly.

## Growth path

- v1 (current): local automated experimentation for one civil-engineering regression task.
- v2 (real lab data): plug in project-specific laboratory datasets and field measurements.
- v3 (expanded search space): add richer feature engineering, ensembles, calibration, and uncertainty analysis.
- v4 (LLM-assisted proposals): use a language model to propose new model families, constraints, and experiments on top of the fixed loop.
