# Concrete Dataset Lineage

## Source Facts

| File | Rows | Columns | Notes |
| --- | ---: | ---: | --- |
| `concrete_combined.xlsx` / `Combined` | 2060 | 10 | Two source sheets with 1030 rows each, one labeled `CSV` and one labeled `XLS`. Includes a `source` column. |
| `concrete_data.csv` | 1030 | 9 | Legacy single-sheet CSV with the original concrete variables and target. |
| `data/concrete_data.csv` | 1309 | 23 | Harmonized and engineered dataset used by the current pipeline. |

### Target Distribution in `concrete_combined.xlsx`

| Statistic | Value |
| --- | ---: |
| Minimum compressive strength | 2.33 MPa |
| Mean compressive strength | 35.817898495581076 MPa |
| Standard deviation | 16.701653322437835 MPa |
| Maximum compressive strength | 82.6 MPa |

### Source Sheet Counts in `concrete_combined.xlsx`

| Sheet label | Rows |
| --- | ---: |
| `CSV` | 1030 |
| `XLS` | 1030 |

## Derived Interpretation

- The workbook is the broadest local source and appears to combine two equivalent 1030-row source sheets.
- The legacy CSV is the narrowest source and preserves the classic 1030-row schema.
- The engineered CSV in `data/` is the current working dataset because it removes duplicated local rows across the harmonized sheets and adds civil-engineering-derived features.
- The row count of `1309` indicates that the workbook contains duplicated base-input records across source sheets, not just distinct copies of the same file.

## Open Questions

- Which rows are duplicated across the CSV and XLS sheets?
- Are those duplicates intentional replicate measurements or merged ingestion artifacts?
- Should the wiki treat `concrete_combined.xlsx` or `data/concrete_data.csv` as the canonical dataset for future source summaries?

## Related Pages

- [Data lineage and feature engineering](<E:\Random IDEA\AutoResearch\auto-civil-lab\wiki\methods\data-lineage-and-feature-engineering.md>)
- [Feature catalog](<E:\Random IDEA\AutoResearch\auto-civil-lab\wiki\variables\feature-catalog.md>)
- [Current artifacts](<E:\Random IDEA\AutoResearch\auto-civil-lab\wiki\results\current-artifacts.md>)

