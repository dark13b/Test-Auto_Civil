# AutoCivil-Lab Search Failure Diagnosis

Date: 2026-03-11

## Executive Summary

The upgraded pipeline is running end to end, but the model search is not retaining any new model because the engineering validator is rejecting every candidate with a `FAIL`.

This is not primarily a machine-learning quality problem. The baseline model is performing well on prediction accuracy:

- Holdout RMSE: `2.44 MPa`
- Holdout MAE: `1.42 MPa`
- Holdout R2: `0.9769`

The failure is coming from the **hard screening rule in `validator.py`**, not from unstable training, broken feature engineering, or uncertainty estimation.

## What Went Wrong

### 1. The hard fail rule is too aggressive for this dataset

The current validator marks a sample as `FAIL` when:

- `water_cement_ratio > 0.70`, and
- `predicted_strength > 30 MPa`

That rule is too blunt for the current workbook because the dataset contains many legitimate historical mixes with:

- high plain `water/cement` ratio
- supplementary cementitious materials (`slag`, `fly_ash`)
- later-age testing (`28`, `56`, `90`, `100`, `180`, `365` days)

For SCM-rich mixes, plain `water/cement` can overstate the severity of the mix. In practice, `water_binder_ratio`, binder content, curing age, and replacement ratios matter.

### 2. The dataset itself contains many rows that trigger the hard rule

From the enriched dataset:

- Total rows: `2060`
- Rows where `water_cement_ratio > 0.70` and `compressive_strength > 30 MPa`: `440`
- Share of dataset: `21.36%`

Those rows are not isolated outliers. Their age profile shows they are mostly later-age results:

- Minimum age: `14 days`
- Median age: `56 days`
- Mean age: `68.49 days`
- Maximum age: `365 days`

Their binder profile is also not absurd:

- Mean total binder: `388.05 kg/m3`
- Median total binder: `380.0 kg/m3`

This means the hard fail rule conflicts with a meaningful portion of the source data distribution.

### 3. Search uses validator failure as a hard rejection gate

The search loop rejects any model whose validation verdict is `FAIL`.

Observed result:

- Baseline verdict: `FAIL`
- Baseline failed holdout samples: `86`
- Baseline warning samples: `274`
- Search trials run: `30`
- Search trials rejected with `rejected_validation_fail`: `30`

So the search loop is functioning correctly according to its current logic, but the rule set is too restrictive for the data. As a result:

- no Optuna trial can become the new best model
- `best_search_model.pkl` remains the seeded baseline
- `best_search_result.json` remains `FAIL`

## Evidence From Generated Artifacts

### Validation rule counts on the baseline holdout set

Top triggered rules:

- `water_cement_ratio_warn_exceeds_durability_limit`: `244`
- `water_cement_ratio_above_hard_limit`: `86`
- `high_water_cement_ratio_with_high_strength`: `86`
- `fly_ash_replacement_ratio_warn_too_high`: `34`
- `total_binder_warn_too_high`: `30`

This pattern is important:

- the dominant blocker is the `water_cement_ratio_above_hard_limit` rule
- the rest are mostly warnings, not system-breaking failures

### Accuracy is not the failing subsystem

The trained model is not obviously poor. The holdout metrics are strong, and the inverse design tool is able to find valid mixes such as:

- target `35 MPa`
- predicted `34.18 MPa`
- validation verdict `PASS`

That confirms the engineering upgrades are operational. The bottleneck is the validator gate used during model selection.

## Root Cause

The main root cause is a **mismatch between validation policy and dataset reality**:

1. The validator treats one engineering heuristic as a universal hard-fail rule.
2. The dataset contains many legitimate later-age SCM mixes that violate that heuristic.
3. The search loop uses validator failure as a binary keep/reject switch.
4. Therefore every candidate model is rejected, even when predictive performance is strong.

## Recommended Fixes

### Fix 1: Downgrade the hard `w/c > 0.70 + strength > 30` rule to `WARN`

This is the lowest-risk operational fix.

Recommended behavior:

- keep hard `FAIL` only for impossible predictions:
  - below minimum strength bound
  - above maximum strength bound
  - non-finite outputs
- treat high `water_cement_ratio` with high predicted strength as `WARN`

Expected outcome:

- search will stop rejecting every trial
- engineering caution is preserved in reports
- model selection can proceed again

### Fix 2: Make the hard rule age-aware and SCM-aware

If you want a stricter engineering gate, make it conditional instead of universal.

A better hard-fail condition would look like:

- `water_cement_ratio > 0.70`
- and `predicted_strength > 30 MPa`
- and `age <= 28 days`
- and `supplementary_replacement_ratio` is low
- and `water_binder_ratio` is also unfavorable

That would screen implausible early-age claims without rejecting mature SCM mixes that are common in the dataset.

### Fix 3: Use `water_binder_ratio` for SCM-rich mixes

For mixes containing slag and fly ash, `water_binder_ratio` is a better engineering control variable than plain `water_cement_ratio`.

Recommended policy:

- keep `water_cement_ratio > 0.60` as a durability `WARN`
- use `water_binder_ratio` for plausibility checks on SCM mixes
- reserve hard fail for combinations that are clearly unrealistic

### Fix 4: Change model selection from binary rejection to penalty scoring

Instead of:

- any `FAIL` in holdout -> reject model entirely

Use:

- hard reject only impossible outputs
- warning-heavy models receive a score penalty

This keeps engineering judgment in the loop without shutting down search entirely.

## Practical Resolution Order

### Immediate

1. Change the hard `water_cement_ratio_above_hard_limit` rule from `FAIL` to `WARN`, or make it conditional on early age.
2. Rerun:
   - `python train.py`
   - `python search.py`
   - `python report.py`

### Short term

1. Add a dataset profiling step before training that reports how many real samples violate each rule.
2. Separate:
   - durability warnings
   - plausibility failures
   - dataset-quality anomalies

### Better engineering policy

1. Keep bounds as hard fail.
2. Keep ACI / BS 8500 inspired thresholds as warnings.
3. Only hard-fail mixes that are physically implausible after considering:
   - age
   - binder replacement
   - total binder
   - `water_binder_ratio`

## Proposed Code Targets

Files to update for the fix:

- `validator.py`
- `config.yaml`
- optionally `search.py` if warning penalties are added

## Conclusion

The system did not fail because the ML pipeline is broken. It failed because the new validator is stricter than the statistical reality of the workbook being modeled.

The clearest fix is to relax or condition the hard `water/cement` screening rule so it reflects:

- curing age
- SCM replacement
- binder-based ratios

Once that is done, the search loop should start retaining models again while preserving meaningful engineering warnings.
