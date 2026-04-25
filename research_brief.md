---
goal: Maximize composite_score with validator-safe predictions and robust holdout behavior
acceptance_metric: composite_score
min_improvement_pct: 0.0
required_model_families:
  - LGBMRegressor
  - XGBRegressor
  - RandomForestRegressor
focus_areas:
  - gradient_boosting
  - tree_ensemble_regularization
  - robust_confirmed_improvements
scout_candidates_per_cycle: 8
confirm_top_k: 2
---

# Research Brief

## Objective
- Improve `composite_score` while preserving validator hard-fail safety.
- Prefer confirmed gains over one-off scout wins.

## Constraints
- Do not change the concrete preprocessing, feature engineering, validator, or conformal uncertainty logic.
- Use `research_lab.py` as the only research-editable Python surface.
- Treat `outputs/best_search_result.json` as the selection-time source of truth and `outputs/final_holdout_evaluation.json` as terminal-only.

## Notes
- Start broad with scout experiments, then confirm only the promising candidates.
- Keep only confirmed improvements and ratchet the accepted state into `research_lab.py`.
