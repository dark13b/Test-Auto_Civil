# Artifact Contract Migration

## Canonical artifacts

- `best_search_result.json`: canonical search-selection artifact
- `final_holdout_evaluation.json`: canonical terminal holdout artifact
- `final_acceptance.json`: acceptance decision derived from `best_search_result.json`
- `final_artifact_validation.json`: consistency check anchored to `best_search_result.json`

## Deprecated artifact

- `final_metrics.json`: deprecated read-compatibility only

No new code may write `final_metrics.json`. Existing readers may map it through the explicit deprecated compatibility mapper in `artifact_contracts.py`.

## Field migration

- `test_metrics` -> removed from canonical writes
- `val_metrics` -> removed from canonical writes
- `selection_metrics` -> replaced by `selection_validation.aggregate`
- `best_search_metrics` -> replaced by `selected_model` inside `final_holdout_evaluation.json`, or by the top-level `best_search_result.json`

## Evaluation lifecycle

- Cross-validation metrics live in `cross_validation`
- Selection-time validation metrics live in `selection_validation`
- Final holdout metrics live only in `holdout_metrics`
- Holdout is terminal only and must not appear in search-selection artifacts

## Compatibility behavior

- If `final_holdout_evaluation.json` exists, readers must prefer it for terminal results
- If `best_search_result.json` exists, readers must prefer it for selection-time results
- If only `final_metrics.json` exists, readers may map it into the canonical contracts and should surface deprecation metadata
