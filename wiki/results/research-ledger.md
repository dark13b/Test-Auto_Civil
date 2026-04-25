# Research Ledger

Source: [outputs/research_results.csv](<E:\Random IDEA\AutoResearch\auto-civil-lab\outputs\research_results.csv>)

## Source Summary

This ledger records the trial-by-trial scout and confirm history for the live research loop. It is useful as execution evidence, but it is not identical to the final canonical summary artifacts.

## Source Facts

| Field | Value |
| --- | ---: |
| Rows | 33 |
| Scout rows | 19 |
| Confirm rows | 14 |
| Model families present | 1 |
| Current model family in the ledger | `RandomForestRegressor` |

### Status Distribution

| Status | Count |
| --- | ---: |
| `scout_promising` | 14 |
| `error` | 8 |
| `reverted` | 6 |
| `scout_no_improvement` | 5 |

### Validation Distribution

| Verdict | Count |
| --- | ---: |
| `WARN` | 25 |
| `NaN` / blank | 8 |

## Key Ledger Observations

- The ledger is dominated by one model family.
- Several entries are scaffolded as errors or reversions, which suggests the governed loop is actively rejecting some candidate paths.
- The top visible `test_composite_score` rows in this snapshot cluster around trials in the high 20s, but the canonical live selection artifact reports trial 24.

## Derived Interpretation

- The ledger is evidence of active experimentation, not just a static report.
- The ledger and the final canonical artifacts are related but not identical records, so the wiki should never assume one file can stand in for the others.

## Open Questions

- Why does the ledger's apparent top `test_composite_score` not line up perfectly with the canonical best artifact?
- Is there an untracked run file that bridges the ledger and the final canonical summary?
- Should a future wiki page separate scout and confirm outcomes by experiment family?

## Related Pages

- [Current artifacts](<E:\Random IDEA\AutoResearch\auto-civil-lab\wiki\results\current-artifacts.md>)
- [Report vs artifacts](<E:\Random IDEA\AutoResearch\auto-civil-lab\wiki\comparisons\report-vs-artifacts.md>)

