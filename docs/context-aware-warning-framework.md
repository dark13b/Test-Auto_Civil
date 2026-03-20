# Context-Aware Warning Framework

## Why SCM-bearing mixes are handled differently

Supplementary cementitious materials change the rate and pathway of strength development. A cement-only water ratio can therefore overstate anomaly risk when a mix contains meaningful SCM replacement. The validator now separates:

- plain-cement regimes
- SCM-bearing regimes
- high-volume SCM regimes

For SCM-bearing mixes, strength plausibility is screened primarily with water/binder ratio, total binder, SCM replacement ratio, and curing age. Water/cement ratio is retained as secondary guidance, not the first or only anomaly trigger.

## Why age-aware checks matter

The same predicted strength has a different meaning at 3 days, around the standard 28-day window, and after extended curing. The validator now makes that explicit:

- early-age regime: strongest skepticism for high strength when binder context is weak
- standard 28-day regime: normal reference window
- later-age / extended curing regime: less aggressive anomaly flagging when SCM and binder context support later strength gain

This reduces false positives from later-age SCM mixes that would look implausible under a blunt water/cement threshold.

## Warning categories

- Hard Constraint: invalid state or serious engineering inconsistency; intended to block acceptance unless explicitly overridden
- Engineering Caution: plausible mix that still needs engineering review for durability, shrinkage, workability, or curing concerns
- Data Review Flag: unusual observation or metadata concern that needs manual review but is not treated as automatic engineering failure

Each structured warning now includes:

- `warning_code`
- `warning_category`
- `severity`
- `triggering_factors`
- `evidence_summary`
- `academic_note`
- `recommended_review_action`

## How to read the new outputs

Per-sample validation now returns:

- `hard_constraints[]`
- `engineering_cautions[]`
- `data_review_flags[]`
- `contextual_summary`
- `confidence_of_warning_assessment`

The validator also records downgraded rule decisions when SCM or age context prevents a blunt warning from firing. That makes it easier to audit which checks fired, which were softened, and why.
