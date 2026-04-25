# Dashboard Source Truth Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor the dashboard so CV, validation, holdout, uncertainty, and design scenario data are shown in separate source-labeled blocks without silent substitution.

**Architecture:** Keep the existing Flask dashboard, but add a small source-aware data shape for the dashboard endpoints and render it into distinct cards/panels in the current inline template. Preserve the artifact contracts already emitted by `report.py` and `design_tool.py`; the dashboard should only read and present them more truthfully.

**Tech Stack:** Python, Flask, embedded HTML/CSS/JS, unittest

---

### Task 1: Lock source separation with tests

**Files:**
- Modify: `tests/test_dashboard.py`

- [ ] **Step 1: Write the failing test**

```python
def test_validation_details_exposes_distinct_sources(self) -> None:
    ...
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_dashboard.py -k validation_details_exposes_distinct_sources -v`
Expected: FAIL because the dashboard does not yet expose separate CV / validation / holdout source blocks.

- [ ] **Step 3: Write minimal implementation**

No code yet. This step is reserved for the later task.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_dashboard.py -k validation_details_exposes_distinct_sources -v`
Expected: PASS.

### Task 2: Add source-aware dashboard data helpers

**Files:**
- Modify: `dashboard.py`

- [ ] **Step 1: Write the failing test**

Use the Task 1 regression to drive the new source-aware payload shape.

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_dashboard.py -k validation_details_exposes_distinct_sources -v`

- [ ] **Step 3: Write minimal implementation**

Add helpers that return explicit source-labeled metric blocks for `cross_validation`, `selection_validation`, `holdout_metrics`, and `uncertainty_audit`, and stop the dashboard from borrowing holdout values from validation fields.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_dashboard.py -k validation_details_exposes_distinct_sources -v`

### Task 3: Redraw the dashboard panels and design comparison

**Files:**
- Modify: `dashboard.py`

- [ ] **Step 1: Write the failing test**

Add a page-content assertion for the source-labeled sections and the design comparison table/card layout.

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_dashboard.py -k dashboard_renders_source_labeled_panels -v`

- [ ] **Step 3: Write minimal implementation**

Update the inline template/JS to render separate blocks for CV, validation, holdout, uncertainty, and design scenarios.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_dashboard.py -k dashboard_renders_source_labeled_panels -v`

### Task 4: Verify end-to-end regressions

**Files:**
- None

- [ ] **Step 1: Run dashboard tests**

Run: `pytest tests/test_dashboard.py -v`

- [ ] **Step 2: Run the broader contract tests**

Run: `pytest tests/test_artifact_contracts.py tests/test_holdout_integrity.py -v`

