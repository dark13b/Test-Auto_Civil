# Report vs Artifacts

## Core Contradiction

The written reports and the live JSON artifacts do not describe the same current best-model state.

| Source | Reported best model | Reported metric state |
| --- | --- | --- |
| [FINAL_PROJECT_REPORT.md](<E:\Random IDEA\AutoResearch\auto-civil-lab\FINAL_PROJECT_REPORT.md>) | `LGBMRegressor` | Holdout R2 `0.9823`, holdout RMSE `2.1374` |
| [PROJECT_INDEX.md](<E:\Random IDEA\AutoResearch\auto-civil-lab\PROJECT_INDEX.md>) | `LGBMRegressor` | Same LightGBM-centered story |
| `outputs/best_search_result.json` | `RandomForestRegressor` | CV RMSE `4.873969592786195`, WARN, 0 hard fails |
| `outputs/final_metrics.json` | `RandomForestRegressor` | CV RMSE `4.814485705912134`, validation RMSE `5.166004039105162`, WARN |
| `outputs/final_acceptance.json` | N/A | `accepted = false` |

## Secondary Contradiction

The benchmark design document is a template, not a result sheet.

| Source | Status |
| --- | --- |
| [ACADEMIC_BENCHMARK_COMPARISON_SECTION.md](<E:\Random IDEA\AutoResearch\auto-civil-lab\ACADEMIC_BENCHMARK_COMPARISON_SECTION.md>) | Filled with methodology and placeholders, not actual benchmark outputs. |

## Tertiary Contradiction

The file set contains multiple summary layers with different roles:

- selection-time summary
- report-time summary
- ledger history
- stale markdown narrative

That is normal in a research workspace, but the wiki must label them carefully so they are not conflated.

## Derived Interpretation

- The live artifact set should be treated as current ground truth until a newer run supersedes it.
- The older markdown reports should be preserved as historical context and not overwritten retroactively in the narrative.

## Open Questions

- Did the project change model families after the report was written?
- Was the report generated from an earlier run that has since been superseded?
- Should the wiki add a timeline page to track this transition explicitly?

## Related Pages

- [Current artifacts](<E:\Random IDEA\AutoResearch\auto-civil-lab\wiki\results\current-artifacts.md>)
- [Final project report](<E:\Random IDEA\AutoResearch\auto-civil-lab\wiki\literature\final-project-report.md>)
- [Knowledge gaps](<E:\Random IDEA\AutoResearch\auto-civil-lab\wiki\overview\gaps.md>)

