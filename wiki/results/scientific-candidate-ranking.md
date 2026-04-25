# Scientific Candidate Ranking

Updated: 2026-04-26

The mix design candidate ranking now adds a mandatory `engineering_quality` scorecard component to every evaluated scenario. This component is additive to the user-selected objectives and is intended to prevent predicted compressive strength from dominating the ranking by itself.

## Ranking Inputs

- Target strength distance normalized by the configured tolerance.
- Explicit constraint failures from `evaluate_constraints()`.
- Material range failures from the same constraint checks.
- Water/cement ratio utilization against the configured target-regime ceiling.
- Uncertainty interval width and whether the target window overlaps the interval.
- Validator verdict and warning/failure reasons.
- Practical constructability cautions and data review flags already emitted by `EngineeringValidator`.

## Notes

- The ranking does not add new concrete-code limits or unpublished threshold rules.
- Constructability penalties are sourced from the existing validator output, including low water/binder, high binder, admixture dosage, and related review flags when those validator rules fire.
- `ranking_breakdown` now includes an `engineering_quality` entry with the component raw value, weight, weighted score, and a plain-language explanation of the penalty terms.
