# Dashboard Redesign Design

## Goal

Redesign the AutoCivil dashboard so it serves two audiences without mixing their needs:

- `Normal` mode: fast, clear, trustworthy summary for non-technical users.
- `Experimental` mode: deeper research and diagnostics for technical users.

The redesign must use only real artifacts already produced by the repository. No mock data, synthetic placeholders, or silent substitutions are allowed.

## Recommended Direction

Use one dashboard app with one shared visual system, but split the experience into two explicit views:

- `Normal` view is the default landing screen.
- `Experimental` view is a separate research workspace reached through a clear top-level mode switch.

This is preferred over a single mixed page because it improves:

- clarity: the default view can stay legible and direct.
- performance: the default view does not need to render or prioritize heavy research panels first.
- truthfulness: advanced internals are only shown in the context where they make sense.
- maintainability: both views can reuse the same data endpoints and design language without becoming one overloaded page.

## User Experience

### Normal Mode

`Normal` mode is the default and should answer four questions immediately:

1. Did the pipeline run successfully?
2. Is the current result trustworthy?
3. What is the current best result?
4. What should the user pay attention to next?

The first screen should contain:

- a high-level health header with run status, trust status, and latest run time
- a best-result summary with the selected model and the most relevant user-facing outcome
- explicit trust cards for validation, holdout, and uncertainty
- visible warnings for missing, stale, or inconsistent artifacts
- a compact next-step card that says what to inspect next when trust is not established

`Normal` mode should not show:

- trial-by-trial logs
- raw optuna tables
- detailed model-family comparisons
- research lineage diagrams
- large JSON dumps

These should be absent, not merely collapsed.

### Experimental Mode

`Experimental` mode is a separate research workspace for technical inspection. It should contain:

- cross-validation metrics
- selection validation metrics
- final holdout metrics
- uncertainty audit
- run history and lineage
- design scenario trade-offs
- deeper comparison tables and rawer evidence

This mode should keep the current artifact separation strict and explicit. Each panel must name its source type and avoid substituting one metric source for another.

## Information Architecture

### Shared Shell

The app should keep one shared shell:

- common header
- mode switch
- consistent typography, spacing, and color system
- shared auth and existing Flask routing

### Mode Navigation

The mode switch should:

- default to `Normal`
- provide a direct switch into `Experimental`
- preserve mode in the URL so refreshes and deep links are stable

Preferred mechanism:

- route or query-driven mode, such as `/?mode=normal` and `/?mode=experimental`

This is preferable to a pure in-memory toggle because it supports refresh, sharing, and simpler initialization.

## Data Rules

The redesign must read only existing outputs and artifact-backed payloads.

Rules:

- no mock cards
- no fallback content that invents values
- no borrowing holdout values for validation or validation values for holdout
- missing data should render as explicit `Not available` or warning states
- stale artifact handling should remain visible and truthful

Normal mode should prefer already-produced summary data, but it still must label source types where source ambiguity could confuse users.

Experimental mode should continue to expose canonical source separation:

- `cross_validation`
- `selection_validation`
- `holdout_metrics`
- `holdout_validation_report`
- `uncertainty_audit`
- design batch and per-target design artifacts

## Visual Direction

The redesign should improve clarity first, but it should still look intentional and high quality.

Visual direction:

- lighter, cleaner, editorial-technical layout
- stronger hierarchy between status, evidence, and detail
- fewer simultaneous panels on the landing view
- more restrained use of color, with warnings and failures standing out sharply
- meaningful transitions only where they aid mode switching or reveal structure

Normal mode should feel calm and decisive.
Experimental mode can feel denser and more analytical, but should still be visually organized rather than dashboard-noisy.

## Component Plan

### Normal View Components

- top status strip
- trust summary cards
- best-result hero card
- evidence summary row for validation, holdout, uncertainty
- missing-artifact warning cards
- next-action card

### Experimental View Components

- source-separated metric panels
- validation vs holdout comparison table
- uncertainty audit panel
- run-history block
- design scenario comparison table
- scenario cards for individual design outputs

## Performance Strategy

Normal mode should avoid doing unnecessary work on first load.

Preferred behavior:

- initialize only the normal-mode sections by default
- load experimental-only sections when the user explicitly enters experimental mode
- keep heavy charts and dense tables out of normal-mode initialization

This means the current `loadAll()` pattern should be split so the app can selectively load by mode.

## Error Handling

The redesign should handle incomplete outputs visibly.

Cases:

- missing artifact files
- stale final artifacts
- inconsistent final artifact validation
- missing design artifacts
- missing field validation log

Behavior:

- `Normal` mode should show warning cards with plain-language explanations.
- `Experimental` mode can show more exact source details and rawer state.

## Testing

Add or update tests to cover:

- default mode is `Normal`
- the page renders both mode labels
- normal-mode content does not contain experimental-only heavy sections by default
- experimental-mode content contains source-labeled research panels
- no silent source substitution remains
- missing artifact states render explicit warnings rather than fake values

## Files Likely Affected

- `dashboard.py`
- `tests/test_dashboard.py`

No report-generation changes are required unless the dashboard reveals a genuine artifact contract gap during implementation.

## Scope Guardrails

Do not rewrite the entire backend.

Stay focused on:

- dashboard information architecture
- truthful rendering
- mode separation
- selective loading
- better design scenario readability

Do not introduce:

- new fake summary layers
- duplicate metric contracts
- broad artifact schema churn unless a real gap blocks truthful UI rendering
