# Academic Benchmark Comparison Section for Concrete Compressive Strength Prediction

This document provides a manuscript-ready benchmark comparison section for a civil engineering study on concrete compressive strength prediction. The benchmark is anchored to the source workbook `E:\Random IDEA\AutoResearch\auto-civil-lab\concrete_combined.xlsx`, sheet `Combined`, which contains `2060` concrete mixes. The study is framed as a structured tabular regression problem using `8` base mixture variables and `14` engineered features, for a total of `22` predictive inputs after feature construction.

## 1. Short Benchmark Objective

The objective of the benchmark is to compare a balanced set of regression models for predicting concrete compressive strength from mixture composition and curing age under a common, leakage-safe, and engineering-aware evaluation protocol. The benchmark is designed to be academically defensible rather than opportunistic. Accordingly, it includes transparent linear baselines, nonlinear kernel and distance-based models, single-tree and bagged-tree learners, modern gradient-boosting methods, and interval-capable models for uncertainty assessment. This design allows the study to determine not only which model performs best, but also which algorithmic family is most appropriate for structured concrete materials data with correlated mixture variables, nonlinear age effects, and domain-derived ratio features.

## 2. Benchmark Model Families

The benchmark should include the following model families and specific models:

| Family | Models | Benchmark role |
| --- | --- | --- |
| Linear | `LinearRegression`, `Ridge`, `ElasticNet` | Transparent low-complexity baselines |
| Kernel / distance-based | `SVR` with RBF kernel, `KNeighborsRegressor` | Nonlinear baselines based on similarity structure |
| Tree bagging | `DecisionTreeRegressor`, `RandomForestRegressor`, `ExtraTreesRegressor` | Interpretable tree reference plus variance-reduced ensembles |
| Boosting | `GradientBoostingRegressor`, `HistGradientBoostingRegressor`, `XGBoost`, `LightGBM`, `CatBoost` | Classical and modern boosting methods for structured tabular data |
| Uncertainty-oriented | Quantile-capable boosting model, with `GaussianProcessRegressor` discussed only if dataset size permits | Interval estimation and calibration analysis |

The benchmark should explicitly distinguish the core practical candidates from supporting baselines. The core candidates are `LightGBM`, `XGBoost`, `CatBoost`, `ExtraTrees`, `RandomForest`, `Ridge`, and `SVR`. The remaining models should remain in the main benchmark because they improve scientific coverage, but they should be interpreted primarily as reference models that help contextualize the performance of the core candidates.

The predictive inputs should include the base variables `cement`, `slag`, `fly_ash`, `water`, `superplasticizer`, `coarse_aggregate`, `fine_aggregate`, and `age`, together with the engineered features `water_cement_ratio`, `water_binder_ratio`, `total_binder`, `supplementary_replacement_ratio`, `slag_replacement_ratio`, `fly_ash_replacement_ratio`, `paste_volume_proxy`, `aggregate_paste_ratio`, `fine_to_coarse_ratio`, `superplasticizer_binder_ratio`, `cement_age_interaction`, `binder_age_interaction`, `water_binder_age`, and `log_age`.

## 3. Why Each Model Is Included

### 3.1 Linear Models

`LinearRegression` is included as the simplest possible global baseline. Its role is not to compete directly with modern boosting methods, but to establish the minimum performance obtainable when strength is modeled as an additive linear function of mixture components and age-derived features. If this model performs poorly, that result supports the claim that the strength response is materially nonlinear.

`Ridge` is included because multicollinearity is expected in concrete datasets that combine raw composition variables with ratio-based and interaction-based engineered features. Regularization is therefore not merely a statistical convenience, but a scientifically relevant test of whether a stable linear approximation can still retain practical predictive value under correlated predictors.

`ElasticNet` is included to test whether the predictive signal is concentrated in a sparse subset of raw and engineered variables or whether the problem is fundamentally distributed across many correlated descriptors. Its inclusion strengthens the linear family by covering both dense shrinkage and mixed sparse-dense regularization.

### 3.2 Kernel and Distance-Based Models

`SVR` with an RBF kernel is included because it can approximate smooth nonlinear relationships without imposing the axis-aligned partition structure of tree models. This is relevant for concrete strength, where strength gain may depend on continuous interactions among binder content, water demand, and curing age. SVR is also a well-established strong baseline for medium-sized tabular regression problems when features are properly scaled.

`KNeighborsRegressor` is included as a local interpolation benchmark. In mix-design data, a neighborhood-based method tests whether similar mixtures produce similar strengths in a largely local sense, independent of a learned global parametric form. Although KNN often degrades in higher-dimensional feature spaces, its inclusion is scientifically useful because it separates genuine model learning from nearest-neighbor retrieval behavior.

### 3.3 Tree and Bagging Models

`DecisionTreeRegressor` is included as an interpretable single-tree reference. It is expected to be unstable relative to ensembles, but it provides a useful lower bound on what can be achieved through recursive partitioning alone and clarifies the performance gain attributable to bagging.

`RandomForestRegressor` is included because random forests are robust, widely accepted baselines for nonlinear tabular regression, particularly when predictor interactions are important and extensive feature scaling is undesirable. In concrete materials data, random forests often provide strong performance without excessive tuning and therefore represent an essential benchmark family.

`ExtraTreesRegressor` is included because its stronger randomization can reduce variance and improve computational efficiency in noisy or moderately heterogeneous tabular datasets. In a benchmark with engineered ratio features and potentially overlapping mixture regimes, ExtraTrees is particularly relevant because it often competes closely with, or exceeds, RandomForest while remaining simpler to interpret than boosting models.

### 3.4 Boosting Models

`GradientBoostingRegressor` is included as the classical boosting reference within scikit-learn. It is useful for separating the contribution of boosting as a learning principle from the additional implementation advantages provided by modern histogram-based libraries.

`HistGradientBoostingRegressor` is included because it provides a modern, computationally efficient, histogram-based implementation within the scikit-learn ecosystem. It serves as an important reference point between classical boosting and external libraries such as XGBoost and LightGBM.

`XGBoost` is included because it remains one of the strongest and most widely accepted boosting methods for tabular regression. Its regularization controls, shrinkage behavior, and sampling options make it a standard benchmark in predictive modeling studies and an essential comparator when claiming strong performance in structured materials datasets.

`LightGBM` is included because gradient-boosted trees are already the strongest family in the current project and LightGBM has performed particularly well. Its leaf-wise growth strategy, computational efficiency, and strong empirical performance on tabular data make it a central candidate rather than an optional addition.

`CatBoost` is included because it offers a distinct boosting implementation with ordered boosting and strong regularization behavior. Even though the present dataset does not contain categorical predictors, CatBoost remains relevant because its symmetric-tree structure and robust optimization often make it competitive on fully numerical tabular datasets.

### 3.5 Uncertainty-Oriented Models

A quantile-capable boosting model should be included to produce direct prediction intervals rather than point estimates only. In this study, quantile LightGBM is a practical choice because it is already aligned with the current pipeline, scales well to the available dataset size, and permits direct estimation of lower, median, and upper conditional quantiles.

`GaussianProcessRegressor` should be mentioned only as an optional small-data extension. It offers a principled probabilistic framework and theoretically attractive uncertainty estimates, but its cubic complexity makes it difficult to justify as a primary benchmark model for a workbook of `2060` mixes. Its role should therefore be explicitly limited to sensitivity analysis or reduced-data experiments.

## 4. Proposed Experimental Protocol

### 4.1 Dataset Definition

The benchmark should be conducted on the `Combined` sheet of `concrete_combined.xlsx`. The `source` column should be excluded from modeling. The column `blast_furnace_slag` should be renamed to `slag`, and `concrete_compressive_strength` should be used as the target variable. After deterministic feature engineering, the benchmark should operate on a `22`-feature design matrix consisting of `8` original variables and `14` engineered variables.

### 4.2 Feature Engineering Policy

The engineered features should be computed once using deterministic row-wise transformations based only on the observed mixture composition and curing age. Because these features do not use target statistics or dataset-level normalization parameters, they may be generated before splitting without introducing target leakage. The engineered set should remain identical for every model so that the benchmark isolates differences in learning algorithm rather than differences in feature availability.

### 4.3 Train, Validation, and Test Design

The dataset should be split once into training and holdout test partitions using a fixed stratified `80/20` split. Stratification should be performed by discretizing the target strength into `5` quantile bins. The holdout test partition must remain untouched until final evaluation.

Within the training partition, model tuning and stability reporting should use repeated cross-validation with `5` folds and `3` repeats, implemented as a deterministic repeated splitter with `random_state = 42`. This design yields `15` validation scores per model and provides a stronger estimate of model stability than a single `5`-fold run while remaining feasible for a thesis-scale benchmark.

### 4.4 Hyperparameter Tuning Strategy

`LinearRegression` should be fitted without tuning. All other tunable models should be allocated the same search budget of `30` configurations. The search procedure should use the same optimizer for every tunable model, preferably Optuna with a Tree-structured Parzen Estimator sampler, because this is already consistent with the project workflow and provides a uniform search policy across algorithm families.

For each candidate configuration, the objective value should be the mean repeated-CV RMSE on the training partition. Hyperparameter ties should be broken first by mean repeated-CV MAE and then by mean repeated-CV R2. After the best configuration is identified, the model should be refitted on the full training partition and then evaluated exactly once on the untouched holdout test partition.

Boosting models may use early stopping only if the validation subset is drawn exclusively from the training partition within each fitting cycle. Under no circumstances should the holdout test set be used for early stopping, model selection, or calibration.

### 4.5 Fairness Rules

Every model in the benchmark should be compared under the following fairness rules:

- identical source dataset
- identical target variable
- identical base and engineered features
- identical outer train-test split
- identical repeated-CV splitter
- identical tuning budget for all tunable models
- identical random seed policy
- identical metric definitions
- identical leakage-prevention rules

These constraints are necessary because benchmark results are otherwise confounded by differences in data access, tuning effort, or preprocessing.

### 4.6 Preprocessing and Scaling

Feature scaling should be included inside the model pipeline for `LinearRegression`, `Ridge`, `ElasticNet`, `SVR`, `KNeighborsRegressor`, and optional `GaussianProcessRegressor`. The scaler must be fit only on the training fold within each CV split and only on the full training partition during final refitting. Tree-based and boosting models should be trained on the raw engineered feature values without scaling unless a particular library implementation requires otherwise.

### 4.7 Outlier Handling

The main benchmark should not remove statistically extreme observations merely because they are difficult to predict. Concrete compressive strength datasets often contain rare but physically meaningful combinations of binder chemistry, water demand, and curing age. Accordingly, observations should be excluded only if they are verified data-entry errors, impossible duplicates caused by ingestion faults, or otherwise documented corruption cases. Any optional outlier filtering should be reported as a sensitivity analysis rather than as the primary benchmark condition.

### 4.8 Leakage Prevention

The benchmark should state explicitly that the holdout set is used only once, after model selection is complete. All scaling operations, hyperparameter tuning, and optional early stopping must occur strictly within the training partition. In addition, low-, mid-, and high-strength thresholds for range-wise analysis should be computed from the training partition only and then applied unchanged to the holdout partition. Likewise, any uncertainty calibration step should be derived from a calibration subset carved out of the training data rather than from the holdout set.

### 4.9 Engineering Validation and Uncertainty

Engineering validation rules should be applied to holdout predictions as a secondary diagnostic layer. Models should not be ranked primarily by warning counts, because the primary benchmark concerns predictive accuracy; however, models that generate hard plausibility failures should be flagged and may be excluded from the final practical shortlist.

Uncertainty analysis should be performed only for interval-capable finalists. A defensible protocol is to refit the selected interval model on the training portion, reserve a stratified calibration subset within the training data for conformal calibration if needed, and then evaluate empirical coverage and interval sharpness on the untouched holdout partition.

## 5. Metrics and Statistical Comparison Plan

### 5.1 Primary Accuracy Metrics

`RMSE` should be the primary metric because it penalizes large errors more strongly than small ones and therefore reflects the practical cost of substantial strength misestimation. In a civil engineering context, large prediction errors are more consequential than minor deviations because they can distort safety margins, mixture optimization, and confidence in performance claims.

`MAE` should be reported alongside RMSE because it captures the average absolute prediction deviation in units of MPa and is less sensitive to isolated large residuals. MAE therefore complements RMSE by providing a more typical measure of error magnitude.

`R2` should be reported to quantify the fraction of strength variance explained by the model. Although R2 should not be used as the sole ranking criterion, it remains useful for interpreting overall explanatory adequacy across model families.

### 5.2 Stability Metrics

For every model, repeated-CV performance should be reported as mean plus standard deviation for RMSE, MAE, and R2 across the `15` validation folds produced by the `5 x 3` repeated-CV design. The mean reflects expected validation performance, whereas the standard deviation reflects sensitivity to data partitioning. Models with strong mean performance but high dispersion should be interpreted more cautiously than models with similar mean performance and better stability.

### 5.3 Holdout Performance

The main ranking table should report holdout RMSE, holdout MAE, and holdout R2 from the untouched test partition. These metrics should serve as the headline comparison because they represent out-of-sample generalization under the final selected hyperparameters.

### 5.4 Range-Wise Evaluation

Range-wise RMSE should be reported for low-, mid-, and high-strength concrete. The low and high thresholds should be defined as the `20th` and `80th` percentiles of the target distribution in the training partition. These same thresholds should then be applied to the holdout partition. This analysis is necessary because average metrics can conceal materially different behavior across low-strength, normal-strength, and high-strength mixtures.

### 5.5 Uncertainty Metrics

If interval predictions are included, the uncertainty comparison should report:

- empirical coverage at the nominal confidence level
- mean interval width
- calibration by bins across the predicted strength range
- optional interval-width stratification across low-, mid-, and high-strength regions

Empirical coverage evaluates whether the intervals contain the observed strengths at the intended rate. Mean interval width measures sharpness. Calibration by bins reveals whether interval quality is globally acceptable but locally uneven, which is especially important when strength regimes differ in heterogeneity.

### 5.6 Computational Metrics

Training time and inference time should be reported as secondary criteria. These metrics should be measured on the same hardware, using the same software environment, and reported for the best-selected configuration of each model. Computational metrics should not override predictive performance, but they are important for practical deployment, repeated search workflows, and inverse design loops.

### 5.7 Statistical Comparison Policy

The manuscript should avoid overstating statistical significance on the basis of a single dataset. The principal evidence should therefore consist of:

- repeated-CV mean plus standard deviation
- final holdout metrics
- optional bootstrap confidence intervals for the top `2` to `3` holdout models

If bootstrap intervals are reported, they should be described as uncertainty bounds around the observed holdout performance, not as proof of universal superiority. The paper should avoid blanket claims of dominance unless the performance separation is consistent across holdout metrics, repeated-CV stability, and practical diagnostics.

### 5.8 Final Ranking Logic

The final ranking should follow a fixed hierarchy:

1. lowest holdout RMSE
2. lower holdout MAE
3. higher holdout R2
4. better repeated-CV stability, assessed from mean plus standard deviation
5. better uncertainty quality for interval-capable finalists only
6. lower computational burden and greater interpretive simplicity

Engineering validation outcomes should be reported adjacent to the ranking, but they should operate as a diagnostic screen rather than as the primary ranking score. Only hard plausibility failures should justify exclusion from the final practical shortlist.

## 6. Paper-Ready Tables

### Table 1. Benchmark Model List and Scientific Rationale

| Family | Model | Core or supporting | Scaling inside pipeline | Scientific rationale |
| --- | --- | --- | --- | --- |
| Linear | `LinearRegression` | Supporting | Yes | Establishes the irreducible linear baseline and tests whether the target can be approximated by a purely additive response. |
| Linear | `Ridge` | Core | Yes | Addresses multicollinearity among raw, ratio, and interaction features while preserving linear interpretability. |
| Linear | `ElasticNet` | Supporting | Yes | Tests whether mixed sparse and dense regularization improves linear modeling under correlated predictors. |
| Kernel / distance | `SVR (RBF)` | Core | Yes | Captures smooth nonlinear interactions without tree partitions and is a strong medium-scale tabular baseline. |
| Kernel / distance | `KNeighborsRegressor` | Supporting | Yes | Evaluates whether local mixture similarity is sufficient for accurate interpolation. |
| Tree bagging | `DecisionTreeRegressor` | Supporting | No | Provides an interpretable high-variance single-tree reference. |
| Tree bagging | `RandomForestRegressor` | Core | No | Robust nonlinear bagging baseline for structured engineering data. |
| Tree bagging | `ExtraTreesRegressor` | Core | No | Strong randomized-tree ensemble that often rivals RandomForest at lower computational cost. |
| Boosting | `GradientBoostingRegressor` | Supporting | No | Classical boosting reference within scikit-learn. |
| Boosting | `HistGradientBoostingRegressor` | Supporting | No | Efficient histogram-based boosting benchmark within the same library ecosystem. |
| Boosting | `XGBoost` | Core | No | Widely accepted state-of-the-art gradient boosting benchmark for tabular regression. |
| Boosting | `LightGBM` | Core | No | Central candidate due to strong current project performance and high tabular efficiency. |
| Boosting | `CatBoost` | Core | No | Distinct regularized boosting implementation that remains competitive on numeric tabular data. |
| Uncertainty-oriented | `Quantile LightGBM` or equivalent | Supporting | No | Produces direct prediction intervals for uncertainty assessment. |
| Uncertainty-oriented | `GaussianProcessRegressor` | Optional only | Yes | Mentioned only for reduced-data experiments because of unfavorable scaling at the full dataset size. |

### Table 2. Hyperparameter Search Spaces and Tuning Strategy

| Model | Tuning strategy | Search budget | Proposed search space |
| --- | --- | ---: | --- |
| `LinearRegression` | None | `0` | No hyperparameter tuning |
| `Ridge` | Optuna TPE | `30` | `alpha` in `[1e-3, 1e2]` on log scale |
| `ElasticNet` | Optuna TPE | `30` | `alpha` in `[1e-4, 1e1]` on log scale; `l1_ratio` in `[0.05, 0.95]` |
| `SVR (RBF)` | Optuna TPE | `30` | `C` in `[1e-1, 1e3]` log scale; `epsilon` in `[1e-3, 1]` log scale; `gamma` in `[1e-4, 1]` log scale |
| `KNeighborsRegressor` | Optuna TPE | `30` | `n_neighbors` in `[3, 25]`; `weights` in `{uniform, distance}`; `p` in `{1, 2}` |
| `DecisionTreeRegressor` | Optuna TPE | `30` | `max_depth` in `{3, 5, 7, 10, None}`; `min_samples_split` in `[2, 20]`; `min_samples_leaf` in `[1, 10]`; `ccp_alpha` in `[1e-5, 1e-2]` log scale |
| `RandomForestRegressor` | Optuna TPE | `30` | `n_estimators` in `[200, 800]`; `max_depth` in `{None, 4, 8, 12, 16}`; `min_samples_leaf` in `[1, 5]`; `max_features` in `{sqrt, log2, 0.5, 1.0}` |
| `ExtraTreesRegressor` | Optuna TPE | `30` | `n_estimators` in `[200, 800]`; `max_depth` in `{None, 4, 8, 12, 16}`; `min_samples_leaf` in `[1, 5]`; `max_features` in `{sqrt, log2, 0.5, 1.0}` |
| `GradientBoostingRegressor` | Optuna TPE | `30` | `n_estimators` in `[100, 600]`; `learning_rate` in `[0.01, 0.2]` log scale; `max_depth` in `[2, 5]`; `min_samples_leaf` in `[1, 10]`; `subsample` in `[0.6, 1.0]` |
| `HistGradientBoostingRegressor` | Optuna TPE | `30` | `learning_rate` in `[0.01, 0.2]` log scale; `max_depth` in `{None, 3, 5, 7}`; `max_leaf_nodes` in `[15, 63]`; `min_samples_leaf` in `[5, 30]`; `l2_regularization` in `[0, 1]` |
| `XGBoost` | Optuna TPE | `30` | `n_estimators` in `[200, 800]`; `learning_rate` in `[0.01, 0.2]` log scale; `max_depth` in `[3, 10]`; `min_child_weight` in `[1, 10]`; `subsample` in `[0.6, 1.0]`; `colsample_bytree` in `[0.6, 1.0]`; `reg_alpha` in `[1e-6, 1]`; `reg_lambda` in `[1e-3, 10]` |
| `LightGBM` | Optuna TPE | `30` | `n_estimators` in `[200, 800]`; `learning_rate` in `[0.01, 0.2]` log scale; `num_leaves` in `[15, 127]`; `min_child_samples` in `[5, 40]`; `subsample` in `[0.6, 1.0]`; `colsample_bytree` in `[0.6, 1.0]`; `reg_lambda` in `[1e-3, 10]` |
| `CatBoost` | Optuna TPE | `30` | `iterations` in `[200, 800]`; `learning_rate` in `[0.01, 0.2]` log scale; `depth` in `[4, 10]`; `l2_leaf_reg` in `[1, 10]`; `bagging_temperature` in `[0, 5]`; `random_strength` in `[0, 2]` |
| `Quantile LightGBM` | Optuna TPE | `30` | Same structural space as `LightGBM`, fitted separately for lower and upper quantiles |

### Table 3. Repeated Cross-Validation Results (`5` folds x `3` repeats on training partition)

| Model | CV RMSE (mean +- SD) | CV MAE (mean +- SD) | CV R2 (mean +- SD) | Selected hyperparameters | CV rank |
| --- | --- | --- | --- | --- | ---: |
| `LinearRegression` | `<rmse_mean +- rmse_sd>` | `<mae_mean +- mae_sd>` | `<r2_mean +- r2_sd>` | `<none>` | `<rank>` |
| `Ridge` | `<...>` | `<...>` | `<...>` | `<best_params>` | `<rank>` |
| `ElasticNet` | `<...>` | `<...>` | `<...>` | `<best_params>` | `<rank>` |
| `SVR (RBF)` | `<...>` | `<...>` | `<...>` | `<best_params>` | `<rank>` |
| `KNeighborsRegressor` | `<...>` | `<...>` | `<...>` | `<best_params>` | `<rank>` |
| `DecisionTreeRegressor` | `<...>` | `<...>` | `<...>` | `<best_params>` | `<rank>` |
| `RandomForestRegressor` | `<...>` | `<...>` | `<...>` | `<best_params>` | `<rank>` |
| `ExtraTreesRegressor` | `<...>` | `<...>` | `<...>` | `<best_params>` | `<rank>` |
| `GradientBoostingRegressor` | `<...>` | `<...>` | `<...>` | `<best_params>` | `<rank>` |
| `HistGradientBoostingRegressor` | `<...>` | `<...>` | `<...>` | `<best_params>` | `<rank>` |
| `XGBoost` | `<...>` | `<...>` | `<...>` | `<best_params>` | `<rank>` |
| `LightGBM` | `<...>` | `<...>` | `<...>` | `<best_params>` | `<rank>` |
| `CatBoost` | `<...>` | `<...>` | `<...>` | `<best_params>` | `<rank>` |

### Table 4. Holdout Test Results on the Untouched `20%` Test Partition

| Model | Holdout RMSE (MPa) | Holdout MAE (MPa) | Holdout R2 | Hard validation fails | Warning rate or pass rate | Training time (s) | Inference time (ms/sample) | Final rank |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| `LinearRegression` | `<rmse>` | `<mae>` | `<r2>` | `<n>` | `<rate>` | `<time>` | `<time>` | `<rank>` |
| `Ridge` | `<...>` | `<...>` | `<...>` | `<...>` | `<...>` | `<...>` | `<...>` | `<...>` |
| `ElasticNet` | `<...>` | `<...>` | `<...>` | `<...>` | `<...>` | `<...>` | `<...>` | `<...>` |
| `SVR (RBF)` | `<...>` | `<...>` | `<...>` | `<...>` | `<...>` | `<...>` | `<...>` | `<...>` |
| `KNeighborsRegressor` | `<...>` | `<...>` | `<...>` | `<...>` | `<...>` | `<...>` | `<...>` | `<...>` |
| `DecisionTreeRegressor` | `<...>` | `<...>` | `<...>` | `<...>` | `<...>` | `<...>` | `<...>` | `<...>` |
| `RandomForestRegressor` | `<...>` | `<...>` | `<...>` | `<...>` | `<...>` | `<...>` | `<...>` | `<...>` |
| `ExtraTreesRegressor` | `<...>` | `<...>` | `<...>` | `<...>` | `<...>` | `<...>` | `<...>` | `<...>` |
| `GradientBoostingRegressor` | `<...>` | `<...>` | `<...>` | `<...>` | `<...>` | `<...>` | `<...>` | `<...>` |
| `HistGradientBoostingRegressor` | `<...>` | `<...>` | `<...>` | `<...>` | `<...>` | `<...>` | `<...>` | `<...>` |
| `XGBoost` | `<...>` | `<...>` | `<...>` | `<...>` | `<...>` | `<...>` | `<...>` | `<...>` |
| `LightGBM` | `<...>` | `<...>` | `<...>` | `<...>` | `<...>` | `<...>` | `<...>` | `<...>` |
| `CatBoost` | `<...>` | `<...>` | `<...>` | `<...>` | `<...>` | `<...>` | `<...>` | `<...>` |

### Table 5. Performance by Strength Range on the Holdout Test Partition

Range thresholds should be derived from the training partition only, using the `20th` and `80th` percentiles of compressive strength.

| Model | Low-strength RMSE (MPa) | Mid-strength RMSE (MPa) | High-strength RMSE (MPa) | Hardest range | Interpretation |
| --- | ---: | ---: | ---: | --- | --- |
| `LinearRegression` | `<rmse_low>` | `<rmse_mid>` | `<rmse_high>` | `<range>` | `<comment>` |
| `Ridge` | `<...>` | `<...>` | `<...>` | `<...>` | `<...>` |
| `ElasticNet` | `<...>` | `<...>` | `<...>` | `<...>` | `<...>` |
| `SVR (RBF)` | `<...>` | `<...>` | `<...>` | `<...>` | `<...>` |
| `KNeighborsRegressor` | `<...>` | `<...>` | `<...>` | `<...>` | `<...>` |
| `DecisionTreeRegressor` | `<...>` | `<...>` | `<...>` | `<...>` | `<...>` |
| `RandomForestRegressor` | `<...>` | `<...>` | `<...>` | `<...>` | `<...>` |
| `ExtraTreesRegressor` | `<...>` | `<...>` | `<...>` | `<...>` | `<...>` |
| `GradientBoostingRegressor` | `<...>` | `<...>` | `<...>` | `<...>` | `<...>` |
| `HistGradientBoostingRegressor` | `<...>` | `<...>` | `<...>` | `<...>` | `<...>` |
| `XGBoost` | `<...>` | `<...>` | `<...>` | `<...>` | `<...>` |
| `LightGBM` | `<...>` | `<...>` | `<...>` | `<...>` | `<...>` |
| `CatBoost` | `<...>` | `<...>` | `<...>` | `<...>` | `<...>` |

### Table 6. Uncertainty Comparison for Interval-Capable Finalists

| Model or interval method | Nominal coverage | Empirical coverage | Mean interval width (MPa) | Calibration by bins | Range-wise coverage notes | Practical interpretation |
| --- | ---: | ---: | ---: | --- | --- | --- |
| `LightGBM + conformal` | `<target>` | `<coverage>` | `<width>` | `<well calibrated / under-covered / over-covered>` | `<comment>` | `<comment>` |
| `XGBoost + conformal` | `<target>` | `<coverage>` | `<width>` | `<...>` | `<...>` | `<...>` |
| `CatBoost + conformal` | `<target>` | `<coverage>` | `<width>` | `<...>` | `<...>` | `<...>` |
| `Quantile LightGBM` | `<target>` | `<coverage>` | `<width>` | `<...>` | `<...>` | `<...>` |

### Table 7. Final Ranking and Practical Interpretation

| Final rank | Model | Family | Holdout RMSE rank | Holdout MAE rank | Holdout R2 rank | CV stability | Uncertainty note | Computational note | Practical interpretation |
| ---: | --- | --- | ---: | ---: | ---: | --- | --- | --- | --- |
| `1` | `<model>` | `<family>` | `<rank>` | `<rank>` | `<rank>` | `<stable / moderate / unstable>` | `<if applicable>` | `<fast / moderate / slow>` | `<headline interpretation>` |
| `2` | `<model>` | `<family>` | `<rank>` | `<rank>` | `<rank>` | `<...>` | `<...>` | `<...>` | `<...>` |
| `3` | `<model>` | `<family>` | `<rank>` | `<rank>` | `<rank>` | `<...>` | `<...>` | `<...>` | `<...>` |
| `4` | `<model>` | `<family>` | `<rank>` | `<rank>` | `<rank>` | `<...>` | `<...>` | `<...>` | `<...>` |
| `5` | `<model>` | `<family>` | `<rank>` | `<rank>` | `<rank>` | `<...>` | `<...>` | `<...>` | `<...>` |
| `6` | `<model>` | `<family>` | `<rank>` | `<rank>` | `<rank>` | `<...>` | `<...>` | `<...>` | `<...>` |
| `7` | `<model>` | `<family>` | `<rank>` | `<rank>` | `<rank>` | `<...>` | `<...>` | `<...>` | `<...>` |

## 7. Example Academic Writeup Paragraphs

### 7.1 Benchmark Methods Paragraph

To ensure an academically balanced benchmark, the study compared linear, kernel-based, distance-based, bagged-tree, and gradient-boosting regressors under a common evaluation protocol. The benchmark was conducted on the `Combined` sheet of `concrete_combined.xlsx` (`n = 2060`), using the standard concrete mixture variables together with engineered descriptors representing binder ratios, replacement fractions, paste-volume proxies, and age-interaction effects. The dataset was divided once into stratified training and holdout partitions in an `80/20` ratio using quantile bins of compressive strength. Model development was restricted to the training partition, where hyperparameter tuning and stability assessment were performed using repeated `5`-fold cross-validation with `3` repeats. All tunable models were assigned the same search budget of `30` configurations, whereas ordinary least squares linear regression was fitted without tuning. This design was intended to isolate algorithmic differences while avoiding confounding from unequal preprocessing, tuning effort, or data access.

### 7.2 Model Rationale Paragraph

The benchmark was designed to span the principal algorithmic families relevant to tabular materials prediction. Linear models were retained as low-complexity baselines and to assess whether a regularized linear approximation remained competitive after feature engineering. SVR with an RBF kernel and K-nearest neighbors were included to represent nonlinear similarity-based learning without tree partitioning. Tree models ranged from a single decision tree to bagged ensembles, allowing the incremental contribution of variance reduction to be examined explicitly. Modern boosting methods, including XGBoost, LightGBM, and CatBoost, were included because they represent the strongest class of learners for structured tabular regression and are particularly well suited to nonlinear interactions among mixture proportions, binder chemistry, and curing age. Finally, a quantile-capable boosting model was added for uncertainty analysis so that predictive sharpness could be evaluated jointly with interval calibration.

### 7.3 Results Paragraph Template

Across the full benchmark, the highest-ranking models were concentrated in the gradient-boosting family, with `[top_model]` achieving the lowest holdout RMSE of `[RMSE]` MPa, followed by `[second_model]` and `[third_model]`. The tree-bagging models outperformed the linear baselines, indicating that the strength response could not be adequately represented by a global linear relation even after inclusion of engineered mixture descriptors. `SVR` showed `[competitive / variable]` performance, suggesting that smooth nonlinear structure was present but that kernel sensitivity to scaling and hyperparameter selection remained non-negligible. Repeated cross-validation showed that `[model]` combined strong mean accuracy with comparatively low fold-to-fold variability, supporting its selection as the most stable high-performing model.

### 7.4 Range-Wise Performance Paragraph Template

Range-wise error analysis showed that predictive difficulty was not uniform across the strength spectrum. The `[low / mid / high]` range exhibited the largest RMSE for most models, whereas the `[range]` interval was comparatively easier to predict. This finding indicates that average error metrics alone would have masked regime-specific weaknesses. In practical terms, the result suggests that the benchmark should be interpreted not only in terms of global RMSE, but also in terms of how reliably a model generalizes across low-strength and high-strength concrete mixtures that may differ in binder chemistry, water demand, and age sensitivity.

### 7.5 Uncertainty Paragraph Template

For interval-capable finalists, global empirical coverage was `[coverage]` at a nominal level of `[target_coverage]`, with a mean interval width of `[width]` MPa. Although the global calibration profile was `[acceptable / conservative / insufficient]`, bin-wise analysis showed that interval quality was not spatially uniform across the predicted strength range. In particular, `[strength bin or region]` displayed `[under-coverage / over-coverage]`, indicating that uncertainty quality should be discussed locally rather than summarized only by a single global coverage number. This result supports the use of calibration-by-bins as a necessary complement to average interval sharpness.

### 7.6 Discussion Paragraph Template

If the linear models underperform while bagged trees and boosting models perform strongly, the most defensible interpretation is that compressive strength depends on nonlinear interactions that are only partially captured by additive feature effects. If ExtraTrees performs comparably to or better than RandomForest, that result would suggest that additional split randomization is beneficial in this dataset, possibly because the engineered features create overlapping or highly correlated predictor spaces. If LightGBM, XGBoost, and CatBoost dominate the benchmark, the conclusion should not be framed as a trivial victory of complexity, but rather as evidence that modern boosting is especially well matched to structured concrete mix data with nonlinear age-strength behavior and ratio-based descriptors. Conversely, if SVR is competitive but unstable, the paper should note that kernel methods can recover relevant nonlinear structure but may be less robust under repeated partitioning than tree ensembles. Finally, if uncertainty is globally acceptable but uneven across bins, the discussion should emphasize that calibration adequacy is regime-dependent and that interval reliability may degrade in specific strength ranges even when global coverage appears satisfactory.

## 8. Final Recommended Shortlist

If the full benchmark is to be retained in the paper while keeping the practical comparison compact, the final recommended shortlist is:

| Model | Role in final paper | Reason for retention |
| --- | --- | --- |
| `Ridge` | Linear baseline | Stronger and more stable linear comparator than ordinary least squares under correlated engineered features |
| `SVR (RBF)` | Nonlinear kernel baseline | Represents smooth nonlinear learning outside the tree family |
| `RandomForestRegressor` | Bagging baseline | Widely accepted robust ensemble baseline |
| `ExtraTreesRegressor` | Bagging comparator | Often matches or exceeds RandomForest with lower variance or faster training |
| `XGBoost` | Modern boosting core | Standard high-performance tabular benchmark |
| `LightGBM` | Modern boosting core | Current strongest family in the project and likely headline candidate |
| `CatBoost` | Modern boosting core | Distinct boosting implementation that strengthens the external validity of the comparison |

This seven-model shortlist is balanced and efficient. It retains one regularized linear model, one kernel method, two bagging ensembles, and three modern boosting methods. That structure is sufficient for a strong journal paper without wasting effort on an excessively crowded main narrative.

`LinearRegression`, `ElasticNet`, `KNeighborsRegressor`, `DecisionTreeRegressor`, `GradientBoostingRegressor`, and `HistGradientBoostingRegressor` should remain in the complete benchmark tables because they strengthen methodological completeness. However, they need not receive equal discussion space in the final Results section unless they produce an unexpected outcome. The uncertainty subsection should then add `Quantile LightGBM` or the selected conformal interval model as a companion analysis rather than as a separate point-prediction competitor.

## 9. Notes for Civil Engineering Interpretation

- Poor linear-model performance should be interpreted as evidence that concrete strength depends on nonlinear interactions among water demand, binder composition, and curing age rather than on simple additive effects alone.
- If tree-bagging models outperform the linear family but remain below the best boosting models, the most defensible interpretation is that nonlinear partitioning is necessary, but that residual bias remains unless the model is updated sequentially through boosting.
- Dominance of `LightGBM`, `XGBoost`, or `CatBoost` should be linked to their ability to model higher-order interactions among raw mixture contents and engineered binder-ratio features, not merely to their algorithmic popularity.
- If `SVR` is competitive, that result supports the presence of smooth nonlinear structure in the materials response surface. If it is unstable, the instability should be attributed to sensitivity to scaling and kernel hyperparameters rather than dismissed outright.
- If `ExtraTrees` performs as well as or better than `RandomForest`, the paper should note that stronger split randomization may be advantageous when the predictor space contains several correlated engineered ratios and interaction terms.
- If one strength range remains consistently harder than the others, the discussion should treat this as an engineering result rather than only a statistical result. The difficult range may correspond to mixtures with more variable hydration behavior, sparse representation, or more heterogeneous supplementary cementitious material content.
- Global uncertainty adequacy should never be interpreted as sufficient on its own. Calibration should also be checked across strength bins because low-strength and high-strength concretes may exhibit different residual structures and interval requirements.
- Engineering validation should be reported alongside predictive metrics because a model that is numerically accurate but systematically inconsistent with physical screening rules is less suitable for design support.
- Practical conclusions should be confined to the compositional and age ranges represented by the workbook. Extrapolation outside the observed design space should be presented cautiously, especially for inverse design or uncertainty interpretation.
