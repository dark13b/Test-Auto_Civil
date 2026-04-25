# Uncertainty Handling in Mix Design

Updated: 2026-04-26

## Current Flow

- `mix_design/predictor.py` builds one candidate frame, predicts strength, and attaches an `UncertaintyInterval` to every `PredictionResult`.
- `uncertainty.py` is the calibrated estimator source. `UncertaintyEstimator.predict_with_interval()` returns `predicted`, `lower_90`, `upper_90`, `interval_width`, and `confidence_label`.
- `mix_design/facade.py` evaluates each proposal, scores it, and ranks candidates. Interval width is part of the mandatory engineering-quality score and the optimizer search penalty.
- `design_tool.py` translates canonical scenarios into legacy JSON, single-target reports, batch JSON, and batch CSV rows.

## Gaps Found

- Missing estimator fallback produced an interval-shaped object without clearly marking that it was not calibrated.
- Candidate JSON did not expose uncertainty calibration status or uncertainty warning reasons.
- Batch CSV rows omitted uncertainty interval width, confidence, calibration status, and warning counts.
- Ranking penalized interval width, but did not explicitly penalize uncalibrated, unknown, or low-confidence uncertainty.
- Generated design artifacts lacked a clear precision note telling readers to treat predicted strength as a model estimate requiring lab validation.

## Patch

- `UncertaintyInterval` now carries `is_calibrated` and `warning_reasons`.
- Missing estimator output is marked `UNCALIBRATED` and emits an explicit warning.
- Wide intervals and low/unknown/uncalibrated confidence labels emit warning reasons.
- Ranking adds explicit penalties for uncalibrated, low-confidence, unknown, and warned uncertainty.
- Legacy JSON and batch CSV outputs expose interval width, confidence, calibration status, and warning count.
- Design artifacts include a precision note to avoid overclaiming strength estimates.
