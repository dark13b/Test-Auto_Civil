# AutoCivil-Lab Architecture V2

## Canonical runtime

`research_loop.py` is the single canonical runtime for core research behavior.

- Canonical CLI entry point: `python research_loop.py`
- Canonical library entry point: `research_loop.run_engineering_research_loop(...)`
- `research_brief.md` is the human strategy input.
- `research_lab.py` is the controlled research surface.
- `train.py` prepares the baseline. It does not own research orchestration.

## Runtime surface map

| Entry point | Status | Role |
| --- | --- | --- |
| `research_loop.py` | canonical | Runs scout -> confirm -> keep/revert research execution. |
| `run_qwen_only.py` | compatibility wrapper | Installs a Qwen-only config overlay, then calls `research_loop.main()`. |
| `autocivil_autosearch_session.py` | compatibility wrapper | Schedules/monitors a session, then calls `research_loop.run_engineering_research_loop(...)`. |
| `launch_auto_research.bat` | compatibility wrapper | Shell launcher for `research_loop.py`. |
| `search.py` | deprecated wrapper | Emits an explicit deprecation warning, translates legacy config keys, then delegates to `research_loop`. |
| `benchmark.py` | evaluation-only | Benchmarks models against the current research result. It is not a research runtime. |

## Artifact source of truth

The final persisted source of truth is split by lifecycle stage.

- `outputs/best_search_result.json` is the canonical search-selection artifact and the source of truth for acceptance.
- `outputs/final_holdout_evaluation.json` is the canonical terminal holdout artifact written only by `report.py`.
- `outputs/search_state_best_result.json` is the live synchronized search-state mirror of the current best selection result.
- `outputs/best_search_model.pkl` and `outputs/search_state_best_model.pkl` are synchronized copies of the same winning model artifact.
- `outputs/research_results.csv` is the canonical per-experiment ledger for scout/confirm execution.
- `outputs/optuna_results.csv` is a compatibility mirror of `outputs/research_results.csv` and must not become an independent decision source.
- `outputs/final_acceptance.json` and `outputs/final_artifact_validation.json` are derived governance artifacts written after finalization.

## Evaluation boundary rules

Core research selection happens on non-holdout data only.

- Scout and confirm experiments use the training/validation research partitions.
- Keep/revert decisions are based on validation-side research metrics, not holdout metrics.
- `research_loop.py` may refresh uncertainty artifacts on `validation_audit`; that remains inside the research boundary.
- The locked holdout belongs to explicit evaluation steps such as `report.py` and `benchmark.py`.
- `report.py` is the one-time final holdout report and requires `final_acceptance.json` before it runs.
- No legacy wrapper or helper may promote a model using holdout performance during core research execution.

## Legacy compatibility policy

Legacy or alternate entry points are allowed only if they do not own unique business logic.

- Wrappers may translate flags or config shape, but they must delegate execution to `research_loop`.
- Deprecation behavior must be explicit: warnings in Python entry points, and obvious messaging in docs/launchers.
- Deprecated compatibility reads may remain for `final_metrics.json`, but new writes must target the canonical artifacts named above.
- New runtime features belong in `research_loop.py` and its dedicated helper modules, not in wrapper entry points.

## Contributor default path

Use this sequence unless you are doing explicit evaluation or a wrapper-specific local run:

```bash
python train.py
python research_loop.py --with-report
python benchmark.py
python design_tool.py --target 35
```
