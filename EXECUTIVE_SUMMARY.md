# Executive Summary

Generated: 2026-04-22.

## What This Project Really Is

AutoCivil-Lab is a concrete-strength ML research and decision-support system. It combines dataset preparation, model training, automated governed search, engineering validation, uncertainty/report generation, inverse mix design, benchmarking, and a Flask dashboard.

It is not just a dashboard and not just a notebook-style experiment. There is real production-shaped Python code and a passing test suite, but the repository still carries old implementation branches, stale docs, and generated artifact state that can mislead reviewers.

## Current Scope

Current scope appears to include:

- concrete compressive-strength regression
- governed autoresearch-style model search
- LLM-assisted or deterministic proposals
- artifact lineage and contract validation
- final reporting and uncertainty
- dashboard review/sharing
- inverse mix design for target strength
- academic benchmark comparisons

The scope is broad but mostly coherent. The risk is not absence of implementation; the risk is ambiguous source of truth and overstated readiness.

## Internal Consistency

The code direction is internally coherent: `research_loop.py` is canonical, `train_impl.py` owns evaluation, `artifact_contracts.py` owns schemas, and `mix_design/` is the canonical design subsystem.

The repository presentation is not internally consistent:

- current accepted/final artifact state is not complete
- docs disagree on best model and artifact source of truth
- deprecated wrappers and patch modules still look active
- root `test_track_b.py` looks tested but is not collected
- generated artifacts and archives dominate local workspace size

## Biggest 5 Blockers

1. Current artifact state is not an accepted final result: `final_acceptance.json` says `accepted: false`.
2. `outputs/final_holdout_evaluation.json` is missing.
3. Final artifact validation can pass even when terminal holdout evidence is absent.
4. Docs/status files contain stale claims, especially `PROJECT_INDEX.md`.
5. Full LLM-assisted runtime depends on external Ollama/OpenAI availability not proven by CI.

## Biggest 5 Strengths

1. The deterministic CI/golden path passes.
2. Domain logic is real: concrete feature engineering, validation, uncertainty, and mix design exist.
3. Artifact contracts and lineage are more mature than a typical experiment repo.
4. The canonical runtime direction is documented and mostly implemented.
5. The `mix_design/` refactor created a cleaner subsystem with focused tests.

## Biggest 5 Cleanup Wins

1. Delete local generated/tool residue: `.claude/`, `tmp/`, caches, empty `raw/`.
2. Archive/delete duplicate senior review doc and stale `PROJECT_INDEX.md` content.
3. Move/archive `test_track_b.py` and `config_track_b_additions.yaml`.
4. Archive/delete `proposal_engine_patch.py` and `research_loop_patch.py` after verifying no active use.
5. Keep `outputs/` and `outputs_archive_*` out of GitHub; curate only small review fixtures.

## Readiness

| Readiness target | Assessment |
| --- | --- |
| Internal deterministic demo | Mostly ready |
| MVP result claim | Not yet |
| Beta/external demo | Not yet |
| Production/release | Not ready |

Reason: a research-selection path exists and passes tests, but the current artifact state is rejected and lacks terminal holdout output.

## Real Happy Path

Stable happy path exists for deterministic internal research selection:

`generate_data.py` -> `train.py` -> `research_loop.py` deterministic/golden path -> `best_search_result.json` -> dashboard selection view.

No stable current happy path exists for an accepted final-report product state because the current run is rejected and `final_holdout_evaluation.json` is missing.

## Immediate Next Engineering Step

Make artifact readiness explicit in code: add separate validation modes for selection-ready and final-report-ready, and require `final_holdout_evaluation.json` plus accepted final state before claiming final readiness.

## Immediate Next Cleanup Step

Clean the repository presentation: update or archive `PROJECT_INDEX.md`, old source-of-truth specs, duplicate review docs, and excluded Track B files before using these docs as ChatGPT Projects source material.

## Immediate Next Product Step

Decide the MVP boundary: selection-only research assistant, accepted final-report generator, dashboard demo, or inverse mix-design assistant. The code supports all four directions partially, but the next roadmap should choose one primary proof path.

