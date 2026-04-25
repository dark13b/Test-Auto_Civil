# Current Artifacts

## Canonical Live Summary

| Artifact | Current fact |
| --- | --- |
| `outputs/best_search_result.json` | Canonical selection-time best model artifact |
| `outputs/final_metrics.json` | Canonical report-time summary artifact |
| `outputs/final_search_summary.json` | Alternate summary artifact with similar but not identical metrics |
| `outputs/uncertainty_calibration.json` | Current uncertainty calibration artifact |
| `outputs/final_acceptance.json` | Current acceptance gate result |

## Selection-Time Artifact

### `outputs/best_search_result.json`

| Field | Value |
| --- | ---: |
| Model | `RandomForestRegressor` |
| Trial | `24` |
| CV RMSE | `4.873969592786195` |
| CV MAE | `3.2898185662183277` |
| CV R2 | `0.9041633789517465` |
| CV Composite | `0.8837009413159578` |
| Validation Verdict | `WARN` |
| Hard Failed Count | `0` |
| Warning Count | `129` |
| Durability Warning Count | `90` |
| Dataset Anomaly Count | `91` |

## Report-Time Artifact

### `outputs/final_metrics.json`

| Field | Value |
| --- | ---: |
| Model | `RandomForestRegressor` |
| CV RMSE | `4.814485705912134` |
| CV MAE | `3.2683202534874836` |
| CV R2 | `0.9062758820574771` |
| CV Composite | `0.8852976410576211` |
| Validation RMSE | `5.166004039105162` |
| Validation MAE | `3.1054685574815535` |
| Validation R2 | `0.9017321246692189` |
| Validation Composite | `0.8798840730068983` |
| Validation Pass Rate | `0.34517766497461927` |
| Hard Failed Count | `0` |
| Warning Count | `129` |
| Durability Warning Count | `90` |
| Dataset Anomaly Count | `91` |

## Alternate Summary Artifact

### `outputs/final_search_summary.json`

| Field | Value |
| --- | ---: |
| Model | `RandomForestRegressor` |
| Trial | `5` |
| CV RMSE | `4.901228615993024` |
| CV MAE | `3.3000537407904837` |
| CV R2 | `0.9030076476378037` |
| CV Composite | `0.882910773048627` |
| Validation Verdict | `WARN` |

## Uncertainty Artifact

### `outputs/uncertainty_calibration.json`

| Field | Value |
| --- | ---: |
| Coverage | `0.9796954314720813` |

## Acceptance Artifact

### `outputs/final_acceptance.json`

| Field | Value |
| --- | ---: |
| Accepted | `false` |

## Derived Interpretation

- The live artifact set is internally consistent on the model family: everything points to `RandomForestRegressor`.
- The exact metrics differ slightly across selection-time and report-time artifacts, which is normal if they were computed at different stages.
- The uncertainty calibration looks strong on coverage, but the artifact set still carries many engineering warnings.

## Open Questions

- Why does the repository keep both `best_search_result.json` and `final_metrics.json` if their metric values are slightly different?
- Which artifact should downstream documentation cite for the final numbers?
- Where is the missing `final_holdout_evaluation.json` referenced by older reports?

## Related Pages

- [Research ledger](<E:\Random IDEA\AutoResearch\auto-civil-lab\wiki\results\research-ledger.md>)
- [Report vs artifacts](<E:\Random IDEA\AutoResearch\auto-civil-lab\wiki\comparisons\report-vs-artifacts.md>)
- [Artifact catalog](<E:\Random IDEA\AutoResearch\auto-civil-lab\wiki\outputs\artifact-catalog.md>)
