# Academic Benchmark Comparison Against the Current LightGBM Reference

## Summary

The benchmark was executed against the current finalized LightGBM reference run (trial `24`) using the existing AutoCivil-Lab data preparation, feature engineering, engineering validator, composite scoring rule, and conformal uncertainty workflow.

- Current baseline RF reference: RMSE `3.5710` MPa, MAE `2.5082` MPa, R2 `0.9508`, composite `0.9206`.
- Current LightGBM reference: RMSE `2.6372` MPa, MAE `1.6466` MPa, R2 `0.9732`, composite `0.9454`.
- Benchmark winner: `CatBoost` with RMSE `2.2718` MPa, MAE `1.4632` MPa, R2 `0.9801`, composite `0.9537`.
- Artifact consistency check: search artifacts in sync = `True`, repair applied = `False`.

**Conclusion:** LightGBM no longer ranks first; the updated benchmark winner is CatBoost.

The re-run LightGBM benchmark row achieved holdout RMSE `2.7175` MPa, delta `0.0803` MPa versus the current reference trial.

## Executive Comparison

| Artifact | RMSE | MAE | R2 | Composite |
| --- | --- | --- | --- | --- |
| Baseline RF reference | 3.5710 | 2.5082 | 0.9508 | 0.9206 |
| Current LightGBM reference | 2.6372 | 1.6466 | 0.9732 | 0.9454 |
| Benchmark winner (CatBoost) | 2.2718 | 1.4632 | 0.9801 | 0.9537 |

## Ranked Holdout Results

| rank | model | holdout_rmse | holdout_mae | holdout_r2 | holdout_composite | delta_rmse_vs_reference | delta_mae_vs_reference | delta_r2_vs_reference | delta_composite_vs_reference | validation_verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | CatBoost | 2.2718 | 1.4632 | 0.9801 | 0.9537 | -0.3654 | -0.1834 | 0.0069 | 0.0083 | WARN |
| 2 | XGBoost | 2.6182 | 1.7300 | 0.9736 | 0.9453 | -0.0190 | 0.0834 | 0.0004 | -0.0001 | WARN |
| 3 | LightGBM | 2.7175 | 1.8167 | 0.9715 | 0.9428 | 0.0803 | 0.1701 | -0.0017 | -0.0026 | WARN |
| 4 | HistGradientBoosting | 3.0205 | 1.9516 | 0.9648 | 0.9358 | 0.3833 | 0.3050 | -0.0084 | -0.0096 | WARN |
| 5 | GradientBoosting | 3.0260 | 2.1070 | 0.9647 | 0.9348 | 0.3888 | 0.4604 | -0.0085 | -0.0106 | WARN |
| 6 | SVR | 3.1916 | 1.9648 | 0.9607 | 0.9320 | 0.5544 | 0.3182 | -0.0125 | -0.0134 | WARN |
| 7 | ExtraTrees | 3.3570 | 2.1812 | 0.9565 | 0.9272 | 0.7197 | 0.5346 | -0.0166 | -0.0182 | WARN |
| 8 | KNN | 3.6797 | 2.0212 | 0.9478 | 0.9209 | 1.0425 | 0.3745 | -0.0254 | -0.0245 | WARN |
| 9 | RandomForest | 3.7611 | 2.6783 | 0.9454 | 0.9154 | 1.1239 | 1.0316 | -0.0277 | -0.0300 | WARN |
| 10 | LinearRegression | 5.9913 | 4.4685 | 0.8615 | 0.8486 | 3.3541 | 2.8219 | -0.1116 | -0.0968 | FAIL |
| 11 | Ridge | 6.0049 | 4.4876 | 0.8609 | 0.8481 | 3.3677 | 2.8409 | -0.1123 | -0.0973 | FAIL |
| 12 | ElasticNet | 6.0074 | 4.4900 | 0.8608 | 0.8480 | 3.3702 | 2.8434 | -0.1124 | -0.0974 | FAIL |

## Cross-Validation Stability

| rank | model | cv_rmse | cv_rmse_std | cv_mae | cv_mae_std | cv_r2 | cv_r2_std |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | CatBoost | 3.4937 | 0.4065 | 2.1022 | 0.0777 | 0.9516 | 0.0093 |
| 2 | XGBoost | 3.5884 | 0.3536 | 2.2584 | 0.0849 | 0.9489 | 0.0086 |
| 3 | LightGBM | 3.5987 | 0.3494 | 2.2235 | 0.0672 | 0.9487 | 0.0081 |
| 4 | HistGradientBoosting | 3.8501 | 0.3214 | 2.4503 | 0.0743 | 0.9413 | 0.0078 |
| 5 | GradientBoosting | 3.5403 | 0.3566 | 2.3369 | 0.1003 | 0.9502 | 0.0088 |
| 6 | SVR | 4.6156 | 0.5398 | 2.8144 | 0.2453 | 0.9147 | 0.0206 |
| 7 | ExtraTrees | 4.0005 | 0.3880 | 2.5925 | 0.1236 | 0.9366 | 0.0098 |
| 8 | KNN | 4.5124 | 0.3423 | 2.6775 | 0.1671 | 0.9187 | 0.0164 |
| 9 | RandomForest | 4.3165 | 0.3647 | 3.0275 | 0.1381 | 0.9262 | 0.0098 |
| 10 | LinearRegression | 6.5668 | 0.3347 | 4.9260 | 0.2263 | 0.8295 | 0.0132 |
| 11 | Ridge | 6.5651 | 0.3368 | 4.9219 | 0.2324 | 0.8296 | 0.0133 |
| 12 | ElasticNet | 6.5646 | 0.3363 | 4.9210 | 0.2305 | 0.8296 | 0.0133 |

## Strength-Range RMSE

| rank | model | low_rmse | mid_rmse | high_rmse | hardest_range |
| --- | --- | --- | --- | --- | --- |
| 1 | CatBoost | 2.0422 | 2.2965 | 2.4147 | high |
| 2 | XGBoost | 2.1283 | 2.7581 | 2.6371 | mid |
| 3 | LightGBM | 2.2978 | 2.7611 | 2.9680 | high |
| 4 | HistGradientBoosting | 2.1140 | 3.1983 | 3.2448 | high |
| 5 | GradientBoosting | 2.4361 | 3.0947 | 3.3422 | high |
| 6 | SVR | 3.1678 | 3.3927 | 2.5184 | mid |
| 7 | ExtraTrees | 2.5831 | 3.0977 | 4.5834 | high |
| 8 | KNN | 2.7709 | 3.4517 | 4.9421 | high |
| 9 | RandomForest | 2.8289 | 3.5177 | 5.0749 | high |
| 10 | LinearRegression | 4.4276 | 4.7501 | 9.6308 | high |
| 11 | Ridge | 4.4438 | 4.7416 | 9.6784 | high |
| 12 | ElasticNet | 4.4448 | 4.7408 | 9.6871 | high |

## Uncertainty Comparison

| Model | Coverage | Mean interval width | Global status |
| --- | --- | --- | --- |
| Current LightGBM reference | 0.9046 | 13.9241 | PASS |
| Benchmark winner (CatBoost) | 0.9008 | 13.3428 | PASS |

## Recommendation

The benchmark winner is `CatBoost`. This model improves upon the current LightGBM reference in holdout RMSE. The benchmark should therefore replace the current LightGBM reference as the primary model recommendation.
