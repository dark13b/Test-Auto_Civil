# Feature Catalog

## Base Variables

| Variable | Role |
| --- | --- |
| `cement` | Primary binder component. |
| `slag` | Ground granulated blast-furnace slag contribution after harmonization. |
| `fly_ash` | Fly ash contribution after harmonization. |
| `water` | Mixing water content. |
| `superplasticizer` | Chemical admixture content. |
| `coarse_aggregate` | Coarse aggregate mass. |
| `fine_aggregate` | Fine aggregate mass. |
| `age` | Curing age in days. |
| `compressive_strength` | Target variable in the engineered dataset. |

## Engineered Features

| Feature | Formula or meaning |
| --- | --- |
| `water_cement_ratio` | `water / cement` |
| `water_binder_ratio` | `water / total_binder` |
| `water_effective_binder_ratio` | `water / effective_binder` |
| `binder_to_water_ratio` | `total_binder / water` |
| `effective_binder_to_water_ratio` | `effective_binder / water` |
| `slag_replacement_ratio` | `slag / total_binder` |
| `fly_ash_replacement_ratio` | `fly_ash / total_binder` |
| `total_binder` | `cement + slag + fly_ash` |
| `effective_binder` | `cement + 0.80*slag + 0.35*fly_ash` |
| `supplementary_replacement_ratio` | `(slag + fly_ash) / total_binder` |
| `effective_scm_replacement_ratio` | `effective_scm_content / effective_binder` |
| `aggregate_paste_ratio` | `(coarse_aggregate + fine_aggregate) / (total_binder + water)` |
| `paste_to_aggregate_ratio` | `(total_binder + water) / (coarse_aggregate + fine_aggregate)` |
| `fine_to_coarse_ratio` | `fine_aggregate / coarse_aggregate` |
| `superplasticizer_binder_ratio` | `superplasticizer / total_binder` |
| `superplasticizer_effective_binder_ratio` | `superplasticizer / effective_binder` |
| `paste_volume_proxy` | `total_binder + water + superplasticizer` |
| `log_age` | `log(age + 1)` |
| `cement_age_interaction` | `cement * log_age` |
| `binder_age_interaction` | `total_binder * log_age` |
| `effective_binder_age_interaction` | `effective_binder * log_age` |
| `water_binder_age` | `water_binder_ratio / log_age` |
| `water_effective_binder_age` | `water_effective_binder_ratio / log_age` |

## Diagnostic Variables

| Variable | Meaning |
| --- | --- |
| `effective_scm_content` | Effective SCM contribution under the current binder-efficiency priors. |
| `binder_efficiency_gap` | `total_binder - effective_binder` |
| `cement_fraction_of_binder` | `cement / total_binder` |
| `cement_fraction_of_effective_binder` | `cement / effective_binder` |
| `scm_fraction_of_effective_binder` | `effective_scm_content / effective_binder` |

## Derived Interpretation

- The engineered feature set emphasizes ratios, binder efficiency, and age interactions.
- The model is therefore not learning only from raw mass quantities. It is also learning from domain assumptions about hydration, replacement, and mix balance.

## Open Questions

- Which of these features are actually retained by the current model after training?
- Are the age-interaction terms stable across different model families, or do they mainly help tree-based learners?

