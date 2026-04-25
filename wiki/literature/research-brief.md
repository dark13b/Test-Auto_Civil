# Research Brief

Source: [research_brief.md](<E:\Random IDEA\AutoResearch\auto-civil-lab\research_brief.md>)

## Source Summary

This brief defines the current governed research objective and the constraints that are meant to keep the loop focused.

## Source Facts

- Goal: maximize `composite_score` with validator-safe predictions and robust holdout behavior.
- Acceptance metric: `composite_score`.
- Required model families: `LGBMRegressor`, `XGBRegressor`, `RandomForestRegressor`.
- Focus areas: gradient boosting, tree ensemble regularization, robust confirmed improvements.
- Scout candidates per cycle: 8.
- Confirm top k: 2.

## Research Objective

Improve the composite score while preserving engineering-validation safety and preferring confirmed improvements over one-off scout wins.

## Materials and Mix Details

- The brief is model-oriented, not mix-oriented, but it governs the concrete prediction workflow that uses the mix data.

## Methodology

- Keep `research_lab.py` as the only research-editable Python surface.
- Start broad with scout experiments.
- Confirm promising candidates.
- Keep only confirmed improvements.

## Key Results

- No empirical results are reported.
- The brief is a policy document for how future experiments should be selected and judged.

## Limitations

- It is a governance artifact, not a performance summary.
- It does not capture later artifact drift by itself.

## Relevance To The Current Research Direction

- Strong relevance.
- This is the clearest statement of the current objective function and acceptance logic in the repository.

## Related Pages

- [Current artifacts](<E:\Random IDEA\AutoResearch\auto-civil-lab\wiki\results\current-artifacts.md>)
- [Research ledger](<E:\Random IDEA\AutoResearch\auto-civil-lab\wiki\results\research-ledger.md>)

