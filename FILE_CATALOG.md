# File Catalog

Generated: 2026-04-22.

Status vocabulary: active, unclear, stale, duplicate, dead, generated, test-only, candidate for cleanup.

## Core Runtime and ML

| Path | Type | Responsibility | Depends on | Depended on by | Status | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| `research_loop.py` | Python CLI/module | Canonical governed research loop, CLI smoke commands, acceptance, optional report. | `research_protocol`, `research_loop_helpers`, `loop_artifacts`, `loop_candidate_selection`, proposal providers, `train_impl`, `validator`, `report` | `search.py`, `run_qwen_only.py`, tests | active | Central runtime. |
| `research_loop_helpers.py` | Python module | Baseline/current-best loading, candidate evaluation helpers, knowledge context, acceptance support. | `artifact_contracts`, `hypothesis_archive`, `knowledge_base`, `proposal_service`, `train_impl`, `research_protocol` | `research_loop.py`, tests | active | Good extraction from loop. |
| `research_protocol.py` | Python module | Constants, brief parsing, artifact/acceptance helpers, state helpers, final consistency validation. | `artifact_contracts`, `novelty_scorer`, `state_store` | many runtime/tests | active | Final consistency check is too permissive for missing holdout. |
| `train.py` | Python facade/CLI | Public training API and CLI facade. | `train_impl`, thin facades | many modules/tests | active | Thin source-compatible layer. |
| `train_impl.py` | Python module/CLI | Dataset loading, CV, model construction, metrics, artifact IO, baseline training. | `feature_engineering`, `integrity_checks`, `validator`, `artifact_contracts` | central dependents | active | Very large choke point. |
| `generate_data.py` | Python CLI | Data loading/generation/normalization and feature engineering. | `feature_engineering` | user CLI | active | Uses current `config.yaml` local workbook. |
| `feature_engineering.py` | Python module | Concrete engineering feature creation and validation. | none local | `train_impl`, `validator`, `generate_data`, `mix_design/predictor` | active | Central domain logic. |
| `validator.py` | Python module/CLI | Engineering validation rules and verdict summaries. | `feature_engineering` | `train_impl`, `research_loop`, `dashboard`, `report`, `mix_design` | active | Central domain guard. |
| `uncertainty.py` | Python module/CLI | Quantile/conformal uncertainty and calibration artifact. | `artifact_contracts`, `feature_engineering`, `train_impl` | `report`, `benchmark`, `research_loop`, `mix_design` | active | Current output has older run lineage than best search. |
| `benchmark.py` | Python CLI | Academic benchmark runs and plots/reports. | `artifact_contracts`, `search`, `train`, `uncertainty`, `validator` | tests, user CLI | active auxiliary | Imports deprecated `search.py`. |
| `report.py` | Python CLI | Final report, holdout plots, final holdout artifact. | `artifact_contracts`, `train_impl`, `uncertainty`, `validator` | `research_loop`, tests | active | Current terminal output missing; gated by acceptance. |

## Proposal, LLM, and Search

| Path | Type | Responsibility | Depends on | Depended on by | Status | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| `proposal_service.py` | Python module | Provider abstraction for deterministic, LLM, hybrid proposals. | `research_lab`, `proposal_backend_policy`, `proposal_contract`, `proposal_validation`, `llm_backend`, `proposal_engine_impl` | `research_loop_helpers`, wrapper providers | active | Preferred newer provider surface. |
| `proposal_engine.py` | Python facade | Re-export/compatibility facade. | `proposal_api` | `research_loop`, `llm_proposer`, tests | active facade | Thin but name makes it look canonical. |
| `proposal_api.py` | Python module | Prompt -> LLM -> parse -> filter orchestration wrapper. | `proposal_engine_impl`, `proposal_gates`, `proposal_parsing`, `proposal_prompting`, `policy_layer`, `llm_backend`, `proposal_service` | `proposal_engine.py`, tests | active |
| `proposal_engine_impl.py` | Python module | Main LLM proposal engine, structured outputs, fallback behavior. | `research_lab`, `llm_backend`, `model_routing`, `research_protocol`, helpers | `proposal_api`, `proposal_service` | active | Large and complex. |
| `proposal_engine_helpers.py` | Python module | Prompt/log/parse helper functions. | `llm_backend`, `proposal_gates`, `proposal_parsing`, `research_protocol` | `proposal_engine_impl.py` | active |
| `proposal_contract.py` | Python module | Proposal dataclasses/contracts. | none local | proposal stack | active |
| `proposal_backend_policy.py` | Python module | Backend routing/admission policy. | `proposal_contract` | LLM/proposal stack | active |
| `proposal_validation.py` | Python module | Candidate normalization/validation. | `proposal_gates` | `proposal_service.py` | active |
| `proposal_gates.py` | Python module | Duplicate/semantic gates. | `proposal_parsing`, `proposal_prompting`, `policy_layer` | proposal stack/tests | active |
| `proposal_parsing.py` | Python module | JSON/proposal parsing. | none local | proposal stack | active |
| `proposal_prompting.py` | Python module | Prompt construction. | none local | proposal stack | active |
| `llm_backend.py` | Python facade | Re-export backend APIs. | `llm_backend_impl` | proposal stack | active facade |
| `llm_backend_impl.py` | Python module | Null/Ollama/OpenAI/Hybrid backend implementations. | `proposal_backend_policy` | LLM stack/tests | active integration |
| `llm_backend_ollama.py`, `llm_backend_openai.py` | Python facades | Backend compatibility imports. | `llm_backend_impl` | stage2 tests | active facade |
| `llm_proposer.py` | Python module | Older proposer wrapper. | `llm_backend`, `model_routing`, `proposal_engine` | `search.py` | legacy-active |
| `llm_admission_policy.py` | Python module | LLM reliability/admission evaluation. | none local | `research_loop.py`, tests | active |
| `model_routing.py`, `model_registry.py` | Python modules | Model/backend routing and registry facade. | policy/train impl | proposal/train stack | active |
| `search.py` | Python CLI/module | Deprecated compatibility wrapper and legacy helpers. | `research_loop`, `train_impl`, `artifact_sync`, `llm_proposer`, etc. | `benchmark.py`, scripts, tests | legacy-active | Too large for a deprecated wrapper. |
| `run_qwen_only.py` | Python CLI | Qwen-only wrapper. | `train`, `benchmark`, `report`, `research_loop`, `uncertainty` | tests | legacy-active |
| `deterministic_proposal_provider.py`, `hybrid_proposal_provider.py`, `llm_proposal_provider.py` | Python facades | Provider class re-exports. | `proposal_service` | `research_loop.py` | active facade |

## Artifact and State

| Path | Type | Responsibility | Depends on | Depended on by | Status | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| `artifact_contracts.py` | Python module | Typed contracts, validation, deprecated artifact mapping. | none local | report/dashboard/benchmark/loop/tests | active | Central migration boundary. |
| `artifact_sync.py` | Python module/CLI | Atomic writes, pending operations, CSV snapshots, startup repair. | none local | `research_loop`, `search`, `loop_artifacts`, tests | active |
| `artifact_io.py` | Python module | Tiny training artifact save facade. | none local | `train.py` | active but thin |
| `state_store.py` | Python module | JSON state store. | none local | `research_protocol.py`, tests | active |
| `loop_artifacts.py` | Python module | Research result rows, logs, run manifests, diversity, final sync. | `artifact_contracts`, `artifact_sync`, `research_protocol`, `train_impl`, `validator` | `research_loop`, `loop_candidate_selection` | active |
| `loop_candidate_selection.py` | Python module | Scout candidate selection. | `loop_artifacts`, proposal contracts/engine, `research_lab`, `research_protocol` | `research_loop`, tests | active |
| `loop_execution.py` | Python module | Stage candidate evaluation helper. | `train.py` | none found | candidate for cleanup | No incoming imports found. |

## Mix Design and Dashboard

| Path | Type | Responsibility | Depends on | Depended on by | Status | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| `mix_design/contracts.py` | Python module | Typed request/result/constraint contracts. | none local | mix design package/tests | active |
| `mix_design/constraints.py` | Python module | Constraint loading/merging/evaluation. | contracts | mix design package/tests | active |
| `mix_design/objective_engine.py` | Python module | Objective scoring. | contracts | facade/tests | active |
| `mix_design/optimizer.py` | Python module | Candidate generation/search. | constraints/contracts | facade/tests | active |
| `mix_design/predictor.py` | Python module | Model inference and uncertainty intervals. | `feature_engineering`, contracts | facade/tests | active |
| `mix_design/facade.py` | Python module | Composes predictor, optimizer, validator, objectives. | mix design package, `train`, `uncertainty`, `validator` | `design_tool.py`, tests | active |
| `mix_design/exporters.py` | Python module | Dict export of comparison result. | contracts | facade/tests | active |
| `mix_design/__init__.py` | Python package init | Re-export contracts. | contracts | tests/importers | active |
| `design_tool.py` | Python CLI/module | Deprecated adapter and CLI for mix design. | `artifact_contracts`, `mix_design`, `train` | `dashboard.py`, tests | legacy-active |
| `dashboard.py` | Flask app | Dashboard APIs, auth, sharing, HTML/CSS/JS. | `artifact_contracts`, `design_tool` | deployment/tests | active | Monolithic and artifact-dependent. |
| `field_tracker.py` | Python CLI | Field validation log/summary/export. | none local | no incoming imports | unclear | Keep only if workflow is used. |

## Docs, Configs, Tests, Artifacts

| Path | Type | Responsibility | Depends on | Depended on by | Status | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| `config.yaml` | YAML | Primary runtime config. | none | almost all runtime | active | Large source of truth. |
| `research_brief.md` | Markdown/YAML front matter | Human strategy and acceptance constraints. | none | `research_loop.py` | active |
| `program.md` | Markdown | Legacy pointer to `research_brief.md`. | none | none | stale/cleanup candidate | Misleading for autoresearch users. |
| `README.md` | Markdown | Main public docs. | none | humans | active but needs refresh | Overstates current final artifact state. |
| `ARCHITECTURE_V2.md` | Markdown | Canonical architecture contract. | none | humans | active | Good but current artifacts violate terminal-holdout expectation. |
| `PROJECT_INDEX.md` | Markdown | Old project index/status. | none | humans | stale | Best model and line-count claims are stale. |
| `FINAL_PROJECT_REPORT.md`, `ACADEMIC_BENCHMARK_COMPARISON_SECTION.md`, `INTEGRATION_GUIDE.md` | Markdown | Older reports/integration docs. | none | humans | unclear/stale | Keep as history or archive. |
| `docs/2026-03-15-4am-senior-review*.md` | Markdown | Senior review docs. | none | humans | duplicate | Two files differ by one line. |
| `docs/superpowers/**` | Markdown | Plans/specs | none | humans | mixed | Some plans implemented; some specs stale. |
| `tests/**` | Python tests/fixtures | Active test suite. | runtime modules | CI | active |
| `test_track_b.py` | Python test-like file | Track B legacy tests. | many old modules | none under pytest config | candidate for cleanup | Move under `tests/` or archive. |
| `.github/workflows/ci.yml` | YAML | CI pipeline. | scripts | GitHub Actions | active |
| `requirements.txt`, `requirements-dev.txt` | deps | Runtime/dev deps. | none | install/CI | active |
| `Procfile`, `render.yaml` | deployment configs | Dashboard deploy. | `dashboard.py` | hosts | active |
| `outputs/**` | generated artifacts | Runtime outputs. | runtime | dashboard/report/users | generated | Ignored; current state is selection-only/rejected. |
| `outputs_archive_*` | generated archives | Historical runs. | runtime | humans only | archive/cleanup | Large local storage. |
| `.claude/**` | local worktrees | Tool-generated local residue. | none | none | candidate for cleanup | Untracked. |
| `.gitnexus/**` | generated index | GitNexus graph. | GitNexus | local tools | generated | Ignored. |
| `AGENTS.md`, `CLAUDE.md` | generated docs | GitNexus agent context. | GitNexus | agents/humans | generated/verify | Created by GitNexus during this pass. |
| `wiki/**` | generated/untracked docs | Wiki-style repository notes. | none | humans | generated/verify | Curate before tracking. |

