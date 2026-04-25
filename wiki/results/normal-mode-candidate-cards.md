# Normal Mode Candidate Cards

Updated: 2026-04-24

## Change

The normal-mode mix design assistant now renders a compact result summary after successful generation and expands each candidate card for live-demo scanning.

## Visible Fields

- Rank, with a BEST badge on the first candidate.
- Predicted strength and frontend-computed difference from target when both values exist.
- Confidence label and uncertainty interval.
- Verdict badge.
- Key mix values.
- Applied constraints.
- Warning list.

## Scope

This is a frontend-only dashboard rendering update. It uses the existing `/api/design_generate` response and does not add backend fields, routes, or optimizer behavior.
