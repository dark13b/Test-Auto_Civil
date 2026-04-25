# Current Issues

Generated: 2026-04-22.

## Issue Register

| ID | Title | Severity | Category | Blocks shipping |
| --- | --- | --- | --- | --- |
| I-001 | Final artifact validation passes without terminal holdout artifact | high | architecture/logic | yes for external claims |
| I-002 | Current acceptance rejects the selected run | high | product/logic | yes for MVP result claims |
| I-003 | Deprecated `final_metrics.json` still influences current artifact story | high | architecture/maintainability | yes for cleanup/readiness |
| I-004 | Docs claim stale best model and stale source-of-truth semantics | high | docs/naming | yes for review trust |
| I-005 | Root `test_track_b.py` is not run by configured pytest | medium | tests/cleanup | no, but misleading |
| I-006 | Legacy patch/reference modules remain tracked | medium | cleanup/maintainability | no |
| I-007 | Deprecated `search.py` remains large and imported | medium | architecture/maintainability | no |
| I-008 | LLM backend defaults require external service and long timeout | medium | integration/config | can block full runtime |
| I-009 | Generated artifact directories are large and mixed-run | medium | cleanup/GitHub hygiene | no if ignored |
| I-010 | Dashboard can look complete while artifacts are missing/stale | medium | UX/product | yes for demo claims |
| I-011 | Repo-wide lint/type health is not proven | low | tests/maintainability | no |
| I-012 | Duplicate/stale review docs inflate documentation surface | low | docs/cleanup | no |

## I-001: Final artifact validation passes without terminal holdout artifact

Severity: high  
Category: architecture / logic  
Affected files: `research_protocol.py`, `research_loop.py`, `outputs/final_artifact_validation.json`, `outputs/`

Evidence:

- `python research_loop.py --validate-only` returned success.
- `outputs/final_artifact_validation.json` contains `consistent: true`.
- `outputs/final_holdout_evaluation.json` does not exist.
- `research_protocol.validate_final_artifact_consistency()` only checks terminal holdout fields inside `if final_holdout_path.exists():`.

Why it matters:

The repo can report "consistent" even when final holdout evidence is absent. That is a false-completeness signal for ChatGPT Projects review, dashboard review, and release decisions.

Likely root cause:

The consistency check was written as a selection artifact consistency check, then expanded to terminal holdout comparison only when the terminal artifact exists.

Recommended fix:

Split validation modes:

- selection consistency: require `best_search_result.json` and `best_search_model.pkl`
- final-report readiness: also require `final_holdout_evaluation.json`, uncertainty lineage, and accepted final state

Recommended cleanup action:

Rename or annotate current `final_artifact_validation.json` semantics as selection-only unless final-report mode is explicit.

Fix priority: P0  
Blocks shipping: yes, for any claim of final model readiness.

## I-002: Current acceptance rejects the selected run

Severity: high  
Category: logic / product  
Affected files: `outputs/final_acceptance.json`, `outputs/best_search_result.json`, `README.md`, dashboard consumers

Evidence:

- `outputs/final_acceptance.json` has `accepted: false`.
- It says `decision_reason: rejected: measured improvement below threshold`.
- `measured_improvement_pct` is negative.
- Current docs present the project as if the loop can produce final artifacts, but current saved state is rejected.

Why it matters:

The current artifact set should not be presented as an accepted final result or MVP benchmark. It is a selection/research state with a rejected acceptance gate.

Likely root cause:

The system preserves artifacts from a run that did not beat the baseline sufficiently, but docs/status files were not updated to distinguish "latest selection" from "accepted result."

Recommended fix:

Add a clear run status artifact or dashboard banner: "latest run not accepted." Require accepted state before final report claims.

Recommended cleanup action:

Archive or label current `outputs/` as a rejected run if preserving it for review.

Fix priority: P0  
Blocks shipping: yes, for product/demo result claims.

## I-003: Deprecated `final_metrics.json` still influences current artifact story

Severity: high  
Category: architecture / maintainability  
Affected files: `artifact_contracts.py`, `research_loop_helpers.py`, `report.py`, `benchmark.py`, `search.py`, `outputs/final_metrics.json`, `outputs/final_acceptance.json`

Evidence:

- `outputs/final_metrics.json` still exists.
- `outputs/final_acceptance.json` says `source_of_truth: final_metrics.json`.
- Code includes deprecated mappers for `final_metrics.json`.
- Canonical docs now say `best_search_result.json` and `final_holdout_evaluation.json` are the split sources of truth.

Why it matters:

Deprecated artifacts create ambiguity over which result is authoritative. They also allow stale compatibility behavior to leak into current decisions.

Likely root cause:

Artifact migration preserved backward compatibility but did not fully retire old writes/reads/status language.

Recommended fix:

Stop writing or referencing `final_metrics.json` in new acceptance artifacts. Keep one read-only migration path with explicit deprecation metadata.

Recommended cleanup action:

Move old `final_metrics.json` examples to archive fixtures or delete generated copies after regenerating canonical final artifacts.

Fix priority: P1  
Blocks shipping: yes, if artifact provenance matters.

## I-004: Docs claim stale best model and stale source-of-truth semantics

Severity: high  
Category: docs / naming  
Affected files: `PROJECT_INDEX.md`, `docs/superpowers/specs/2026-03-13-llm-proposal-backend-design.md`, `README.md`, `ARCHITECTURE_V2.md`

Evidence:

- `PROJECT_INDEX.md` claims current saved best model is `LGBMRegressor`, trial 62.
- Current `outputs/best_search_result.json` reports `RandomForestRegressor`.
- Older LLM spec says `final_metrics.json` remains final source of truth.
- Current architecture says terminal holdout belongs to `final_holdout_evaluation.json`.

Why it matters:

ChatGPT Projects or future reviewers will ingest contradictory source docs and may plan against stale status.

Likely root cause:

Review/status docs were not versioned as historical and were not updated after artifact migration.

Recommended fix:

Mark stale docs as historical, update `PROJECT_INDEX.md`, and add a short "current truth" section pointing to canonical runtime and artifact files.

Recommended cleanup action:

Archive old docs under `docs/archive/` or `docs/history/`; keep `README.md` and `ARCHITECTURE_V2.md` current.

Fix priority: P1  
Blocks shipping: yes, for repository review trust.

## I-005: Root `test_track_b.py` is not run by configured pytest

Severity: medium  
Category: tests / cleanup  
Affected files: `test_track_b.py`, `pytest.ini`, `config_track_b_additions.yaml`

Evidence:

- `pytest.ini` sets `testpaths = tests`.
- `test_track_b.py` is in the repo root.
- CI calls `scripts/run_ci_checks.py`, which runs pytest under the configured test paths.

Why it matters:

The file looks like part of the test suite but does not protect current code in CI. It inflates perceived coverage.

Likely root cause:

Track B work was developed before test layout was centralized under `tests/`.

Recommended fix:

Either move it under `tests/` and make it pass, or archive it as historical.

Recommended cleanup action:

Verify whether Track B still matters; otherwise archive `test_track_b.py` and `config_track_b_additions.yaml`.

Fix priority: P2  
Blocks shipping: no, but misleading.

## I-006: Legacy patch/reference modules remain tracked

Severity: medium  
Category: cleanup / maintainability  
Affected files: `proposal_engine_patch.py`, `research_loop_patch.py`

Evidence:

- `proposal_engine_patch.py` header says "Legacy reference only. The active runtime is now proposal_engine.py".
- These files contain substantial implementation-like code.
- They are not central to current runtime imports.

Why it matters:

Patch files confuse source-of-truth review and invite accidental reuse.

Likely root cause:

Historical implementation branches were preserved as code files rather than archived docs.

Recommended fix:

Move to `docs/archive/legacy_patches/` or delete after confirming no active consumer.

Recommended cleanup action:

Archive first if historical rationale matters; delete if not.

Fix priority: P2  
Blocks shipping: no.

## I-007: Deprecated `search.py` remains large and imported

Severity: medium  
Category: architecture / maintainability  
Affected files: `search.py`, `benchmark.py`, `scripts/refit_regime_artifacts.py`, tests

Evidence:

- `search.py` emits deprecation text and delegates to `research_loop.py`.
- It is still about 1,430 lines and has many helpers.
- `benchmark.py` imports it.

Why it matters:

A deprecated wrapper with unique logic is not really retired. It keeps two mental models alive.

Likely root cause:

Migration prioritized compatibility over removal.

Recommended fix:

Shrink `search.py` to a minimal CLI/config translation facade and move any still-used helpers to canonical modules.

Recommended cleanup action:

Audit every non-wrapper function in `search.py`; migrate or delete.

Fix priority: P2  
Blocks shipping: no, unless users rely on old CLI.

## I-008: LLM backend defaults require external service and long timeout

Severity: medium  
Category: integration / config  
Affected files: `config.yaml`, `llm_backend_impl.py`, `proposal_engine_impl.py`, `proposal_service.py`

Evidence:

- `llm.enabled: true`.
- `backend_mode: ollama`.
- Default model is `qwen3-coder:480b-cloud`.
- Request timeout is 600 seconds.
- CI tests use mocked or deterministic paths, not a real backend.

Why it matters:

The official full runtime may hang or degrade depending on local Ollama/cloud availability.

Likely root cause:

Development config doubles as default repo config.

Recommended fix:

Add a safe default config profile with deterministic proposal mode, and document LLM enablement as opt-in.

Recommended cleanup action:

Move local Qwen/Ollama settings to an example overlay.

Fix priority: P2  
Blocks shipping: can block full runtime.

## I-009: Generated artifact directories are large and mixed-run

Severity: medium  
Category: cleanup / GitHub hygiene  
Affected files: `outputs/`, `outputs_archive_*`, `output/`, `tmp/`, `.claude/`

Evidence:

- `outputs/` is about 358.45 MB.
- `outputs_archive_*` total about 1.1 GB.
- `.claude/` is untracked and about 58.67 MB.
- `outputs/` contains artifacts from different run IDs.

Why it matters:

Large generated directories are easy to accidentally commit or upload, and mixed-run state can mislead review.

Likely root cause:

Research workflow accumulates artifacts locally without a curated artifact retention policy.

Recommended fix:

Keep generated directories ignored; publish only curated small fixtures or release artifacts.

Recommended cleanup action:

Delete local temp/tool residue and move historical archives to external storage if needed.

Fix priority: P2  
Blocks shipping: no if ignored, yes if committed.

## I-010: Dashboard can look complete while artifacts are missing/stale

Severity: medium  
Category: UX / product  
Affected files: `dashboard.py`, `outputs/`, `README.md`

Evidence:

- Dashboard has missing-artifact warnings, but the UI is fully built and deployment-ready.
- Current terminal holdout artifact is missing.
- Deployment docs say hosts do not preserve local files after deploy, so artifacts must be committed or regenerated.

Why it matters:

A polished dashboard over stale/missing artifacts is a product trust risk.

Likely root cause:

Dashboard implementation progressed faster than artifact readiness policy.

Recommended fix:

Add a top-level artifact readiness state that distinguishes selection-only, accepted, rejected, and terminal holdout-ready states.

Recommended cleanup action:

Do not commit bulky generated artifacts as the deployment strategy; add a regeneration/release script or small fixture mode.

Fix priority: P2  
Blocks shipping: yes for external demo without artifact state banner.

## I-011: Repo-wide lint/type health is not proven

Severity: low  
Category: tests / maintainability  
Affected files: `scripts/run_ci_checks.py`, many Python files

Evidence:

- README says lint and type checks are intentionally scoped.
- CI only runs ruff/mypy over selected files.

Why it matters:

Passing CI does not mean the full codebase is lint/type clean.

Likely root cause:

Legacy codebase was too large/noisy for immediate repo-wide baseline.

Recommended fix:

Gradually expand lint/type scope module by module.

Recommended cleanup action:

Track excluded modules explicitly in docs.

Fix priority: P3  
Blocks shipping: no.

## I-012: Duplicate/stale review docs inflate documentation surface

Severity: low  
Category: docs / cleanup  
Affected files: `docs/2026-03-15-4am-senior-review.md`, `docs/2026-03-15-4am-senior-review-full-repo.md`

Evidence:

- `git diff --no-index --numstat` showed one insertion and one deletion between the two docs.

Why it matters:

Duplicate docs increase review noise and reduce confidence in which docs matter.

Likely root cause:

Historical review copy was preserved under two names.

Recommended fix:

Keep one canonical copy and archive/delete the duplicate.

Recommended cleanup action:

Delete one copy or move both to `docs/archive/` with dates.

Fix priority: P3  
Blocks shipping: no.

