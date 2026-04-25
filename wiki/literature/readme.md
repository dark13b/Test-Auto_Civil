# README

Source: [README.md](<E:\Random IDEA\AutoResearch\auto-civil-lab\README.md>)

## Source Summary

This is the main repository README. It explains the AutoCivil-Lab runtime, the core orchestration path, the model pipeline, and the current output file set.

## Source Facts

- It describes the repo as an autoresearch-style engineering ML system for concrete compressive strength regression.
- It names `research_loop.py` as the official runtime.
- It treats `research_lab.py` as the controlled research surface.
- It states that the project uses validator, uncertainty, and inverse design components.
- It reports `LGBMRegressor` as the older best-model story in the narrative section.

## Research Objective

Improve concrete compressive-strength prediction through governed experimentation, with reporting and inverse design layered on top.

## Materials and Mix Details

- The README uses the same base concrete mix variables and engineering-derived features as the rest of the repo.

## Methodology

- Governed scout -> confirm -> keep/revert loop.
- Baseline training and search.
- Validation and uncertainty estimation.
- Inverse design through `design_tool.py`.

## Key Results

- The README is narrative documentation rather than a results artifact.
- It enumerates the output files and the engineering feature set, but it does not provide the current live metrics.

## Limitations

- It is a repository overview, not a canonical source of truth for the latest artifact state.
- It can lag behind live `outputs/`.

## Relevance To The Current Research Direction

- Useful as a human-facing entry point.
- Useful for understanding how the repository wants itself to be read.

## Related Pages

- [Project index](<E:\Random IDEA\AutoResearch\auto-civil-lab\PROJECT_INDEX.md>)
- [Current artifacts](<E:\Random IDEA\AutoResearch\auto-civil-lab\wiki\results\current-artifacts.md>)

