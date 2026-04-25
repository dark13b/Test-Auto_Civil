# Git Nexus

Generated: 2026-04-22.

## Scope

This is a repository-wide structural intelligence index for AutoCivil-Lab. It is based on tracked files, import analysis, GitNexus indexing, code inspection, current artifacts under `outputs/`, and the repository's own CI check script.

GitNexus CLI indexing was run successfully:

| Metric | Value |
| --- | ---: |
| Nodes | 2,693 |
| Edges | 7,967 |
| Clusters | 150 |
| Flows | 230 |

## Project Identity

AutoCivil-Lab is a Python ML experimentation system for concrete compressive-strength regression. Its current intended runtime is an autoresearch-style loop that scouts candidate model configurations, confirms promising ones, persists governed artifacts, and optionally generates final report and dashboard artifacts.

The actual repo is not a clean single-purpose app. It contains:

- an active governed research loop
- a baseline training path
- a legacy search wrapper
- an inverse concrete mix design subsystem
- a Flask dashboard
- benchmark/report/uncertainty tooling
- old patch/reference modules
- large generated runtime artifacts and archive folders outside git
- stale or contradictory documentation

## Top-Level Folders

| Path | Actual role | Status | Notes |
| --- | --- | --- | --- |
| `.github/workflows/` | CI configuration | active | Runs scoped lint, scoped mypy, unit tests, contract tests, and one golden run through `scripts/run_ci_checks.py`. |
| `.gitnexus/` | GitNexus generated graph database | generated | Created by `npx gitnexus analyze`; ignored. Useful locally, not project source. |
| `.claude/` | Local worktree/cache residue | cleanup candidate | Untracked, 303 files, about 58.67 MB. Looks like copied worktrees and should not be committed. |
| `data/` | Generated normalized dataset location | generated/active runtime input | Ignored, but `.gitignore` explicitly allows `data/concrete_data.csv`; current tracked file list does not include it. |
| `docs/` | Plans, specs, older reviews, migration notes | mixed | Useful history, but contains duplicate review docs and stale specs that contradict canonical artifact naming. |
| `mix_design/` | Canonical inverse mix design package | active | Typed subsystem behind deprecated `design_tool.py` adapter. |
| `output/` | Old PDF output folder | cleanup candidate | Ignored. Appears historical. |
| `outputs/` | Current runtime artifacts | generated/runtime | Ignored, 45 files, about 358.45 MB. Contains important evidence but should not be treated as source. |
| `outputs_archive_*` | Historical artifact snapshots | archive/cleanup | Ignored, about 1.1 GB combined. Useful only as manual history. |
| `raw/` | Empty raw material folders | unclear | Untracked and empty. Looks like planned data/paper intake. |
| `scripts/` | Utility scripts | active | CI runner, PDF summary generator, regime refit helper. |
| `tests/` | Test suite and golden fixture | active | 219 unit tests, 9 contract tests, 1 golden test passed. |
| `tmp/` | Runtime logs and temp PDF render | cleanup candidate | Ignored. Contains old local logs and PNG. |
| `wiki/` | Generated or external wiki notes | untracked/generated | 18 markdown files. Useful as review material, but currently not part of tracked source. |

## Major Files

| File | Why it matters | Status |
| --- | --- | --- |
| `research_loop.py` | Canonical research runtime; implements scout -> confirm -> final acceptance and optional report flow. | active, central |
| `research_loop_helpers.py` | Focused helpers for baseline loading, current-best resolution, candidate evaluation, and knowledge context. | active, central |
| `research_protocol.py` | Protocol constants, artifact and acceptance helpers, brief parsing, state helpers, final artifact consistency validation. | active, central |
| `train.py` | Thin public compatibility facade for training API and CLI. | active facade |
| `train_impl.py` | Real training implementation: loading, feature engineering, CV, metrics, model instantiation, artifact IO. | active, central |
| `config.yaml` | Primary configuration for data, engineering rules, model search spaces, research loop, LLM backend, dashboard/report assumptions. | active source of truth |
| `research_brief.md` | Human strategy input for governed loop, including acceptance metric and required families. | active |
| `proposal_engine.py` | Facade that re-exports `proposal_api`. | active facade |
| `proposal_api.py` | Compatibility wrapper over proposal prompt/parse/filter pipeline. | active |
| `proposal_engine_impl.py` | Main LLM proposal engine and deterministic fallback bridge. | active but large |
| `proposal_service.py` | Newer provider abstraction selecting deterministic, LLM, or hybrid proposals. | active |
| `llm_backend_impl.py` | Ollama/OpenAI/hybrid backend implementation. | active integration |
| `search.py` | Deprecated compatibility wrapper delegating into `research_loop.py`; still large and imported by benchmark/scripts. | legacy-active |
| `report.py` | Final holdout/report generator; requires `final_acceptance.json`. | active but gated |
| `uncertainty.py` | Quantile/conformal uncertainty calibration and artifact generation. | active |
| `validator.py` | Engineering validation rules and warning/failed verdicts. | active, central |
| `dashboard.py` | Flask dashboard reading generated `outputs/` artifacts; includes auth, APIs, sharing, HTML/CSS/JS in one file. | active but monolithic |
| `design_tool.py` | Deprecated inverse-design adapter around `mix_design/`. | legacy-active |
| `mix_design/*.py` | Typed inverse mix design contracts, constraints, optimizer, predictor, facade, exporter. | active |
| `benchmark.py` | Academic benchmark runner and report writer; imports `search`, `train`, `uncertainty`, `validator`. | active auxiliary |
| `artifact_contracts.py` | Typed artifact contracts and deprecated artifact mappers. | active, central |
| `artifact_sync.py` | Atomic artifact writer and repair/replay logic. | active |
| `artifact_io.py` | Tiny save facade for training artifacts. | active but thin |
| `state_store.py` | JSON-backed state store used through `research_protocol.py`. | active |
| `run_qwen_only.py` | Qwen-only compatibility wrapper around train/report/benchmark/research loop. | legacy-active |
| `autocivil_autosearch_session.py` | Session scheduler/monitor for longer runs. | peripheral |
| `feature_lab.py`, `feature_pipeline.py` | Feature experiment surface and tiny apply facade. | partially connected |
| `proposal_engine_patch.py`, `research_loop_patch.py` | Legacy reference/patch modules. | stale/cleanup candidate |
| `research_loop_smoke.py` | Proposal smoke-test support imported by `research_loop.py`. | active support despite name |
| `test_track_b.py` | Root-level legacy test file outside `tests/`; not collected by configured pytest `testpaths = tests`. | misleading cleanup candidate |

## Entry Points

| Command/file | Actual path | Notes |
| --- | --- | --- |
| `python generate_data.py` | Data normalization/generation | Reads `config.yaml`; current config uses `concrete_combined.xlsx`. |
| `python train.py` | Baseline training | Delegates to `train_impl.main()`. |
| `python research_loop.py --with-report` | Official full research path | May contact configured Ollama backend because `llm.enabled: true` and `backend_mode: ollama`. |
| `python research_loop.py --validate-only` | Artifact consistency check | Passed, but does not require `final_holdout_evaluation.json`. |
| `python search.py` | Deprecated wrapper | Emits deprecation warning and delegates to canonical loop. |
| `python report.py` | Final holdout report | Requires `outputs/final_acceptance.json`; current acceptance says rejected. |
| `python uncertainty.py` | Uncertainty artifact generation | Active support path. |
| `python design_tool.py --target 35` | Inverse design CLI | Uses deprecated adapter over `mix_design.facade`. |
| `python benchmark.py` | Benchmark runner | Auxiliary evaluation/reporting. |
| `gunicorn dashboard:app --bind 0.0.0.0:$PORT` | Web dashboard | Used by `Procfile` and `render.yaml`. |
| `python scripts/run_ci_checks.py` | CI parity checks | Passed in this investigation. |

## Configs

| File | Role | Notes |
| --- | --- | --- |
| `config.yaml` | Main runtime config | Largest behavior source: data mode, feature engineering, validator, benchmark, research loop, LLM, search, design constraints. |
| `config_track_b_additions.yaml` | Supplemental/old track config | Looks associated with Track B experiments and `test_track_b.py`; not imported by canonical path. |
| `pytest.ini` | Test discovery and markers | Restricts test discovery to `tests/`; root `test_track_b.py` is excluded. |
| `.github/workflows/ci.yml` | CI | Installs requirements and runs `scripts/run_ci_checks.py` slices. |
| `.gitignore` | Hygiene policy | Ignores generated artifacts, venvs, archives, temp files, `.gitnexus`; still has generated ignored files present locally. |
| `Procfile`, `render.yaml` | Deployment | Dashboard-only deployment. Requires committed or regenerated `outputs/` artifacts to show real data. |
| `.env.example` | Environment sample | Minimal; does not fully document dashboard/LLM env usage. |

## Tests

The configured test suite passed:

| Slice | Command | Result |
| --- | --- | --- |
| lint | `python scripts/run_ci_checks.py --only lint` | passed |
| types | scoped mypy only | passed |
| unit | `pytest -m 'not contract and not golden'` | 219 passed |
| contract | `pytest -m contract` | 9 passed |
| golden | `pytest -m golden` | 1 passed |

Important limitation: lint and type checks are intentionally scoped. They do not prove repo-wide type or lint health.

## Important File Clusters

| Cluster | Files | Source-of-truth assessment |
| --- | --- | --- |
| Research runtime | `research_loop.py`, `research_loop_helpers.py`, `research_protocol.py`, `loop_artifacts.py`, `loop_candidate_selection.py`, `artifact_sync.py` | Real core. |
| Training/evaluation | `train.py`, `train_impl.py`, `model_registry.py`, `data_loading.py`, `evaluation.py`, `feature_pipeline.py` | `train_impl.py` is source of truth; several facades are thin. |
| Proposal/LLM | `proposal_*`, `llm_*`, `policy_layer.py`, `model_routing.py`, `llm_admission_policy.py` | Active but has overlapping generations of abstractions. |
| Engineering ML | `feature_engineering.py`, `validator.py`, `uncertainty.py`, `validator_calibrator.py`, `failure_analyzer.py`, `novelty_scorer.py`, `knowledge_base.py` | Active support. |
| Artifacts/contracts | `artifact_contracts.py`, `artifact_sync.py`, `research_protocol.py`, `report.py` | Strong contracts, but final holdout enforcement is incomplete. |
| Dashboard | `dashboard.py`, `Procfile`, `render.yaml` | Active web surface; artifact-dependent. |
| Mix design | `mix_design/*.py`, `design_tool.py` | Canonical package exists; adapter remains for compatibility. |
| Legacy/patch | `search.py`, `proposal_engine_patch.py`, `research_loop_patch.py`, `test_track_b.py`, `config_track_b_additions.yaml` | Mixed. Some are explicitly deprecated/reference-only; some still imported by legacy tests or auxiliary paths. |
| Docs/specs | `README.md`, `ARCHITECTURE_V2.md`, `PROJECT_INDEX.md`, `docs/**`, `wiki/**` | Mixed freshness; several docs contradict current artifacts or line counts. |

## Central vs Peripheral

Central files by import graph and execution role:

- `train_impl.py`
- `research_loop.py`
- `research_protocol.py`
- `artifact_contracts.py`
- `validator.py`
- `feature_engineering.py`
- `proposal_service.py`
- `proposal_engine_impl.py`
- `llm_backend_impl.py`
- `mix_design/contracts.py`

Peripheral or weakly connected:

- `field_tracker.py` has a CLI but no local incoming imports.
- `autocivil_autosearch_session.py` is a top-level session wrapper only.
- `loop_execution.py` is not imported by the current graph despite appearing to hold candidate evaluation helpers.
- `test_track_b.py` is root-level and excluded by pytest config.
- `config_track_b_additions.yaml` appears tied to excluded Track B tests.
- `raw/` is empty and untracked.

## Likely Source-of-Truth Files

| Concern | Likely source of truth |
| --- | --- |
| Official runtime | `research_loop.py` |
| Baseline and candidate evaluation | `train_impl.py` |
| Runtime config | `config.yaml` |
| Human research constraints | `research_brief.md` |
| Artifact schemas | `artifact_contracts.py` |
| Research protocol constants and final checks | `research_protocol.py` |
| Current selection artifact | `outputs/best_search_result.json` |
| Terminal holdout artifact | Intended: `outputs/final_holdout_evaluation.json`; actual current file is missing. |
| Deployment app | `dashboard.py` |
| Inverse design implementation | `mix_design/facade.py` and package modules |

## Likely Misleading Files

| File/path | Why misleading |
| --- | --- |
| `PROJECT_INDEX.md` | Claims current saved best model is `LGBMRegressor` trial 62; current `outputs/best_search_result.json` reports `RandomForestRegressor`. Also contains stale line counts. |
| `docs/superpowers/specs/2026-03-13-llm-proposal-backend-design.md` | Says `final_metrics.json` remains final source of truth; current canonical docs/code split selection and final holdout artifacts. |
| `program.md` | Only says "Legacy Note" and points to `research_brief.md`; it is not the real research program. |
| `proposal_engine_patch.py` | Header says legacy reference only but it remains tracked and has substantial implementation-looking code. |
| `research_loop_patch.py` | Same shape as an abandoned implementation branch. |
| `test_track_b.py` | Looks like a test but configured pytest ignores it because `testpaths = tests`. |
| `outputs/final_metrics.json` | Deprecated artifact still present and referenced by `final_acceptance.json` as source of truth. |
| `outputs/final_artifact_validation.json` | Says consistent while terminal holdout artifact is absent; consistency check is selection-only unless holdout exists. |
| `docs/2026-03-15-4am-senior-review*.md` | Two nearly identical review docs differ by one line. |
| `AGENTS.md`, `CLAUDE.md` | Generated by GitNexus during this pass. Useful local context, not product docs. |

