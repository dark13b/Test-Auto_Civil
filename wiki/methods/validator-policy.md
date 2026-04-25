# Validator Policy

## Source Facts

The repository validator uses rule-based checks to interpret model predictions and mix proportions. The current configuration includes thresholds for:

- suspicious water/cement ratio
- durability-oriented water/cement ratio warnings
- durability-oriented water/binder ratio warnings
- early-age warnings
- total-binder warnings
- fly ash replacement warnings
- slag replacement warnings
- superplasticizer dosage warnings

## Key Thresholds From `config.yaml`

| Rule | Threshold |
| --- | ---: |
| Suspicious water/cement ratio | `0.7` |
| Durability water/cement warn | `0.6` |
| Low water/binder warn | `0.25` |
| Total binder low warn | `250.0` |
| Total binder high warn | `550.0` |
| Fly ash replacement warn | `0.4` |
| Slag replacement warn | `0.7` |
| Early age warn | `3.0` days |
| Early age strength warn | `30.0` MPa |
| Superplasticizer dosage warn | `18.0` kg/m3 |
| Superplasticizer/binder warn | `0.05` |

## Result Interpretation

The live artifacts show a `WARN` verdict even when hard fails are zero. That means the predictions are not structurally impossible, but they do trigger engineering caution or data-review heuristics.

## Derived Interpretation

- The validator is acting as a plausibility screen, not just a post-hoc alarm.
- The warnings are meaningful even when the model is statistically strong.
- The current wiki should treat the validator as a core part of the research method, not as a side utility.

## Open Questions

- Which of these thresholds are grounded in external civil engineering standards, and which are only repository heuristics?
- Are the warning thresholds too conservative for the local dataset?
- Should the wiki later split the validator into durability, data-review, and structural-plausibility subpages?

## Related Pages

- [Current artifacts](<E:\Random IDEA\AutoResearch\auto-civil-lab\wiki\results\current-artifacts.md>)
- [Knowledge gaps](<E:\Random IDEA\AutoResearch\auto-civil-lab\wiki\overview\gaps.md>)
- [Data lineage and feature engineering](<E:\Random IDEA\AutoResearch\auto-civil-lab\wiki\methods\data-lineage-and-feature-engineering.md>)

