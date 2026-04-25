# Project Investigation Report

Generated: 2026-04-22.

## One-Line Summary

AutoCivil-Lab is a Python concrete-strength ML autoresearch system with a working deterministic research-loop spine, a broad artifact/report/dashboard ecosystem, and meaningful cleanup debt around stale docs, legacy wrappers, generated artifacts, and false-completeness signals.

## Probable Product Purpose

The product is intended to help a civil/materials engineering user:

1. load or generate concrete mix-design data
2. train baseline strength-prediction models
3. run governed automated model-search experiments
4. apply engineering validation constraints
5. produce uncertainty, benchmark, report, and dashboard artifacts
6. generate inverse mix-design candidates for target strength

## Actual Implementation Reality

The repository has a real implementation, not just scaffolding. The core deterministic path is testable and passed the repo's checks. However, current generated artifacts do not represent a clean complete success state:

- `outputs/best_search_result.json` exists and selects `RandomForestRegressor`.
- `outputs/final_acceptance.json` exists but says `accepted: false`.
- `outputs/final_holdout_evaluation.json` is absent.
- `outputs/final_artifact_validation.json` says `consistent: true` because the validator only compares terminal holdout if the terminal holdout file exists.
- `outputs/final_metrics.json` remains present as a deprecated artifact and is referenced by `final_acceptance.json`.

This means a research-selection happy path exists, but a current end-to-end "accepted final holdout report" happy path is not present in the current artifact set.

## Current Architecture

| Subsystem | Main files | Status |
| --- | --- | --- |
| Data ingestion | `generate_data.py`, `config.yaml`, `feature_engineering.py` | implemented |
| Training/evaluation | `train.py`, `train_impl.py`, `validator.py`, `artifact_io.py` | implemented |
| Governed research loop | `research_loop.py`, `research_loop_helpers.py`, `research_protocol.py`, `loop_artifacts.py`, `loop_candidate_selection.py` | implemented |
| Proposal/LLM | `proposal_service.py`, `proposal_engine_impl.py`, `proposal_api.py`, `llm_backend_impl.py`, `llm_admission_policy.py` | implemented but external backend-dependent |
| Artifacts/contracts | `artifact_contracts.py`, `artifact_sync.py`, `research_protocol.py` | implemented, with legacy compatibility spread |
| Report/uncertainty | `report.py`, `uncertainty.py` | implemented, but current terminal holdout artifact absent |
| Dashboard | `dashboard.py`, `Procfile`, `render.yaml` | implemented, artifact-dependent |
| Mix design | `mix_design/*`, `design_tool.py` | implemented, adapter deprecated |
| Benchmarking | `benchmark.py` | implemented auxiliary path |
| Historical/patch | `proposal_engine_patch.py`, `research_loop_patch.py`, `test_track_b.py` | stale or misleading |

## Implementation Maturity

| Area | Maturity | Evidence |
| --- | --- | --- |
| Deterministic loop | beta/internal-demo ready | Golden test passes; loop has artifact sync and acceptance. |
| Full LLM-assisted loop | partially mature | Backends and smoke paths exist; CI does not prove real Ollama/OpenAI availability. |
| Artifact contracts | medium-high | Strong typed contracts; issue remains around optional final holdout in consistency check. |
| Dashboard | medium | Broad API/UI exists; depends on generated artifacts and monolithic code. |
| Mix design | medium-high | Canonical package and tests exist; deprecated adapter still needed. |
| Docs | medium-low | Some canonical docs are good; several old docs and generated docs conflict with current state. |
| Repo hygiene | low-medium | Generated/archived artifacts are ignored but large local residue exists; stale tracked files remain. |

## Fully Implemented

- Baseline training facade and implementation.
- Feature engineering for concrete mix ratios and interactions.
- Engineering validator with warning/fail verdicts.
- Candidate evaluation with CV metrics and composite score.
- Governed research loop with scout/confirm/keep logic.
- Deterministic proposal fallback.
- Proposal normalization/gating and LLM backend abstractions.
- Artifact contract validation and legacy mapping.
- Atomic artifact writing/replay support.
- Flask dashboard with health/API/share routes.
- Mix-design canonical package behind legacy adapter.
- CI script and test suite slices.

## Partially Implemented or Incomplete

| Area | Gap |
| --- | --- |
| Final report happy path | Current `outputs/final_holdout_evaluation.json` is absent. |
| Acceptance state | Current `final_acceptance.json` rejects the run despite docs presenting the project as ready. |
| Artifact consistency | `validate_final_artifact_consistency()` does not fail when final holdout is absent. |
| LLM reliability | Real backend availability not proven by CI; default config points to Ollama with long timeout. |
| Search wrapper retirement | `search.py` is deprecated but still large and imported by benchmark/scripts. |
| Mix-design adapter retirement | `design_tool.py` is deprecated but still public CLI/API. |
| Track B work | Root `test_track_b.py` and `config_track_b_additions.yaml` look abandoned or excluded. |
| Generated wiki/raw structure | `wiki/` is untracked and `raw/` is empty. |

## Implied But Absent

- A current accepted final holdout artifact.
- A clean repo-wide lint/mypy baseline.
- A committed or reproducible dashboard artifact bundle for deployment.
- A clear policy for old root docs vs generated wiki docs.
- A final cleanup decision for patch/reference modules.
- A proof that real LLM proposal mode works on the configured machine/service.

## Documented But Contradicted

| Claim/source | Contradiction |
| --- | --- |
| `PROJECT_INDEX.md` says current saved best model is `LGBMRegressor`, trial 62. | Current `outputs/best_search_result.json` says `RandomForestRegressor`; `PROJECT_INDEX.md` appears stale. |
| `ARCHITECTURE_V2.md` says `final_holdout_evaluation.json` is canonical terminal holdout artifact. | Current `outputs/` does not contain `final_holdout_evaluation.json`. |
| `docs/superpowers/specs/2026-03-13-llm-proposal-backend-design.md` says `final_metrics.json` remains final source of truth. | Current canonical docs/code treat `final_metrics.json` as deprecated. |
| README output list implies many artifacts are produced in current runs. | Current artifact set has no `final_holdout_evaluation.json`; some artifacts have older run IDs. |
| `test_track_b.py` looks like a test. | `pytest.ini` only discovers `tests/`, so it is not run by normal CI. |

## What Tests Actually Prove

The repo's check script passed:

- 219 unit tests
- 9 contract tests
- 1 golden test
- scoped lint
- scoped mypy

This proves a meaningful deterministic slice of the system works. It does not prove:

- full repo-wide lint/type cleanliness
- real LLM backend reliability
- current final holdout artifact completeness
- dashboard deployment with fresh artifacts
- root-level `test_track_b.py` behavior
- long-running full configured research quality

## Current Repo Health

Overall: functionally promising but operationally messy.

Strengths:

- Real domain-specific ML and engineering validation code.
- Tests pass.
- Artifact lineage/contracts have been taken seriously.
- There is a coherent canonical runtime direction.
- Mix design refactor created a cleaner subsystem.

Weaknesses:

- Stale docs and generated artifacts create false confidence.
- Legacy wrappers and patch files remain in source.
- Current artifacts show rejected acceptance and missing final holdout.
- Several generated/local directories are large and should stay out of git.
- Monolithic files create maintenance drag.

## Technical Debt Shape

| Debt type | Examples |
| --- | --- |
| Legacy compatibility debt | `search.py`, `final_metrics.json` mapping, `design_tool.py` adapter |
| Artifact lifecycle debt | Missing final holdout not treated as validation failure |
| Documentation drift | `PROJECT_INDEX.md`, old specs, duplicate review docs |
| Repo hygiene debt | `.claude/`, output archives, temp logs, generated wiki, root test artifact files |
| Modularity debt | `dashboard.py`, `train_impl.py`, `research_loop.py` are very large |
| Test discoverability debt | root `test_track_b.py` excluded by pytest config |
| Integration debt | LLM backend default may block or fail outside a configured environment |

## Top Technical Risks

1. False positive artifact consistency: current final validation can pass without terminal holdout evidence.
2. Deprecated artifact fallback can mask stale or mismatched output states.
3. LLM config defaults to external service assumptions not guaranteed in CI or deployment.
4. Legacy wrappers keep duplicate conceptual paths alive.
5. Dashboard may show a polished UI over missing/stale/generated artifacts.

## Product Risks Reflected in Code

1. Users may believe the latest run is accepted when `final_acceptance.json` says rejected.
2. Users may believe final holdout evidence exists when it does not.
3. Mix-design recommendations depend on model/uncertainty lineage; stale uncertainty is explicitly possible.
4. Deployment docs require artifacts to be committed or regenerated, which is risky for large/generated files.
5. Scope is broad: ML research, dashboard, inverse design, benchmark, LLM proposals, docs/wiki. The repo needs clearer MVP boundaries.

## Naming and Identity Drift

- `AutoCivil-Lab`, `autoresearch`, `Track B`, `Qwen-only`, `search`, `research_loop`, and `mix_design` all coexist.
- `program.md` is a legacy pointer, while `research_brief.md` is the real brief.
- `search.py` sounds canonical but is deprecated.
- `final_metrics.json` sounds canonical but is deprecated.
- `design_tool.py` sounds canonical but is a deprecated adapter.
- Root `test_track_b.py` sounds active but is excluded from normal test discovery.

## Stale Docs and Dead Code Signals

| Signal | Evidence |
| --- | --- |
| Duplicate docs | `docs/2026-03-15-4am-senior-review.md` and `docs/2026-03-15-4am-senior-review-full-repo.md` differ by one line. |
| Old source of truth | LLM backend spec still names `final_metrics.json` as final source. |
| Patch modules | `proposal_engine_patch.py` header says legacy reference only; `research_loop_patch.py` similarly looks superseded. |
| Excluded tests | `test_track_b.py` outside `tests/`. |
| Empty planned folders | `raw/` has no files. |
| Generated local worktrees | `.claude/worktrees/*` contain duplicated repo trees. |

## Hidden Blockers

- `report.py` requires `final_acceptance.json`; current acceptance is false.
- `research_loop.py --with-report` can run the search loop and then fail report generation if acceptance/report prerequisites are not met.
- `llm.enabled: true` plus `backend_mode: ollama` can introduce long waits or backend failure in the full runtime.
- `outputs/` has stale mixed-run artifacts: best search run is `20260325T125804`, uncertainty calibration is `20260323T142255`.
- `final_artifact_validation.json` does not prove terminal artifact readiness.

## Real Happy Path

There is a stable deterministic internal happy path:

1. generate/load concrete data
2. train baseline
3. run deterministic/golden research loop
4. write best search result and model artifact
5. validate selection artifact consistency

There is not currently a complete accepted final product happy path in the present artifact state because:

- no `outputs/final_holdout_evaluation.json` exists
- current `final_acceptance.json` says `accepted: false`
- current consistency validation still passes without terminal holdout
- current docs overstate final artifact completeness

## Project Coherence Assessment

The architecture direction is coherent: `research_loop.py` is the runtime, `train_impl.py` owns evaluation, artifact contracts enforce schemas, and `mix_design/` is a cleaner subsystem.

The repository presentation is not coherent yet. Stale docs, old patch modules, deprecated-but-large wrappers, excluded tests, and generated artifact residue make the project look more complete and less conflicted than it is.

## Priority Fixes

1. Make final artifact validation explicitly report missing terminal holdout when final-report readiness is required.
2. Resolve current acceptance/report state: either generate a valid final holdout report from an accepted run or mark current artifacts as selection-only.
3. Update stale docs: `PROJECT_INDEX.md`, old LLM source-of-truth spec, README artifact claims.
4. Decide whether `search.py` remains a compatibility shim or should be shrunk.
5. Move, archive, or delete `test_track_b.py` and Track B config if not active.

## Priority Cleanup Opportunities

1. Remove or archive duplicate review doc.
2. Archive/delete patch reference modules after confirming no active consumers.
3. Keep `.gitnexus/` ignored; decide whether generated `AGENTS.md`/`CLAUDE.md` should be tracked.
4. Delete untracked `.claude/`, empty `raw/`, and `tmp/` logs from local workspace.
5. Move old output archives to external storage if they need preservation.

