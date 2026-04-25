# Data Lineage and Feature Engineering

## Source Facts

The current generator and feature pipeline do the following:

- Read the local workbook from `config.yaml`:
  - `data.mode = local_file`
  - `data.local_file.path = concrete_combined.xlsx`
  - `data.local_file.sheet_name = Combined`
- Harmonize column names and units.
- Drop the configured `source` column.
- Map `blast_furnace_slag` to `slag`.
- Map `concrete_compressive_strength` to `compressive_strength`.
- Apply engineering features from `feature_engineering.py`.
- Remove exact duplicate rows based on the base input columns.
- Keep consensus outliers in `local_file` mode, but log the warning instead of removing them.

## Harmonization Rules

| Rule | Source fact |
| --- | --- |
| Optional inputs | `slag`, `fly_ash`, and `superplasticizer` are treated as optional columns during local harmonization and default to `0.0` if missing. |
| Duplicate handling | Duplicate rows are removed after harmonization using the base input columns only. |
| Outlier policy | IsolationForest and LocalOutlierFactor are combined into a conservative consensus detector, but `local_file` mode keeps flagged rows. |
| Output format | The generator writes the prepared dataset to `data/concrete_data.csv`. |

## Engineered Features

The current feature engineering uses both raw mix quantities and ratio/interaction terms.

| Feature | Meaning |
| --- | --- |
| `water_cement_ratio` | `water / cement` |
| `water_binder_ratio` | `water / total_binder` |
| `water_effective_binder_ratio` | `water / effective_binder` |
| `binder_to_water_ratio` | `total_binder / water` |
| `effective_binder_to_water_ratio` | `effective_binder / water` |
| `slag_replacement_ratio` | `slag / total_binder` |
| `fly_ash_replacement_ratio` | `fly_ash / total_binder` |
| `aggregate_paste_ratio` | `(coarse_aggregate + fine_aggregate) / (total_binder + water)` |
| `paste_to_aggregate_ratio` | Inverse of `aggregate_paste_ratio` |
| `fine_to_coarse_ratio` | `fine_aggregate / coarse_aggregate` |
| `total_binder` | `cement + slag + fly_ash` |
| `effective_binder` | `cement + 0.80*slag + 0.35*fly_ash` using config defaults |
| `supplementary_replacement_ratio` | `(slag + fly_ash) / total_binder` |
| `effective_scm_replacement_ratio` | `effective_scm_content / effective_binder` |
| `superplasticizer_binder_ratio` | `superplasticizer / total_binder` |
| `superplasticizer_effective_binder_ratio` | `superplasticizer / effective_binder` |
| `paste_volume_proxy` | `total_binder + water + superplasticizer` |
| `log_age` | `log(age + 1)` |
| `cement_age_interaction` | `cement * log_age` |
| `binder_age_interaction` | `total_binder * log_age` |
| `effective_binder_age_interaction` | `effective_binder * log_age` |
| `water_binder_age` | `water_binder_ratio / log_age` |
| `water_effective_binder_age` | `water_effective_binder_ratio / log_age` |

### Diagnostic Columns

| Column | Meaning |
| --- | --- |
| `effective_scm_content` | Semi-empirical SCM contribution using the configured efficiency factors. |
| `binder_efficiency_gap` | Difference between total binder and effective binder. |
| `cement_fraction_of_binder` | Cement share of total binder. |
| `cement_fraction_of_effective_binder` | Cement share of effective binder. |
| `scm_fraction_of_effective_binder` | SCM share of effective binder. |

## Derived Interpretation

- The generator is not just a loader. It encodes a civil-engineering hypothesis about binder efficiency and age-strength interaction.
- The current engineering feature set is materially richer than the original 8-variable mix description.
- The same source workbook can produce both a raw harmonized dataset and an engineered working dataset, depending on whether deduplication and feature generation are applied.

## Open Questions

- Are the configured binder efficiency factors `fly_ash_k = 0.35` and `slag_k = 0.80` appropriate for the actual local mixes?
- Should the wiki preserve the current feature set as canonical, or also track alternate engineered feature sets from prior runs?
- How many consensus outliers were identified in the local file path, and should they be surfaced in a separate evidence page?

## Related Pages

- [Concrete dataset lineage](<E:\Random IDEA\AutoResearch\auto-civil-lab\wiki\materials\concrete-dataset.md>)
- [Feature catalog](<E:\Random IDEA\AutoResearch\auto-civil-lab\wiki\variables\feature-catalog.md>)

