# Dashboard Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Redesign the dashboard so `Normal` mode is the default, fast, non-technical summary while `Experimental` mode is a separate research view with strict source-separated diagnostics and no mock data.

**Architecture:** Keep the existing Flask app and inline dashboard template, but split rendering and loading behavior by mode using URL-driven state. `Normal` mode should render only trust, outcome, and next-step surfaces from real artifacts; `Experimental` mode should render the heavier research panels and charts on demand using the current artifact-backed endpoints.

**Tech Stack:** Python, Flask, inline HTML/CSS/JavaScript, `pytest`, `unittest`

---

## File Structure

- Modify: `dashboard.py`
  Responsibility: add mode-aware routing/rendering, selective data loading, normal-mode summary layout, experimental-mode research layout, and explicit warning states backed by real artifacts only.
- Modify: `tests/test_dashboard.py`
  Responsibility: lock default mode, URL-driven mode switching, absence of experimental-heavy sections in normal mode, presence of research panels in experimental mode, and warning-state rendering without invented values.

### Task 1: Lock mode behavior with failing page tests

**Files:**
- Modify: `tests/test_dashboard.py`
- Test: `tests/test_dashboard.py`

- [ ] **Step 1: Write the failing test**

Add tests for:

```python
def test_dashboard_defaults_to_normal_mode(self) -> None:
    dashboard = importlib.import_module("dashboard")
    with patch.object(dashboard, "DASHBOARD_PASSWORD", "dummy"):
        client = dashboard.app.test_client()
        response = client.get("/", headers={"Authorization": "Basic YXV0b2NpdmlsOmR1bW15"})

    self.assertEqual(response.status_code, 200)
    page = response.get_data(as_text=True)
    self.assertIn("Normal Mode", page)
    self.assertIn("Experimental Mode", page)
    self.assertIn("data-default-mode=\"normal\"", page)


def test_normal_mode_does_not_render_experimental_sections_by_default(self) -> None:
    dashboard = importlib.import_module("dashboard")
    with patch.object(dashboard, "DASHBOARD_PASSWORD", "dummy"):
        client = dashboard.app.test_client()
        response = client.get("/", headers={"Authorization": "Basic YXV0b2NpdmlsOmR1bW15"})

    page = response.get_data(as_text=True)
    self.assertNotIn("Experimental research workspace", page)
    self.assertNotIn("Trial-by-trial search log", page)
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
python -m pytest tests/test_dashboard.py -k "defaults_to_normal_mode or normal_mode_does_not_render_experimental_sections_by_default" -v
```

Expected:

- FAIL because the current page does not yet define a normal default mode shell or hide experimental sections by default.

- [ ] **Step 3: Write minimal implementation**

No production code in this task. This task exists to establish the first red state.

- [ ] **Step 4: Run test to verify it still fails for the expected reason**

Run:

```bash
python -m pytest tests/test_dashboard.py -k "defaults_to_normal_mode or normal_mode_does_not_render_experimental_sections_by_default" -v
```

Expected:

- FAIL due to missing mode-aware rendering, not due to syntax or import errors.

- [ ] **Step 5: Commit**

Do not commit yet. Wait until production code for this red-green cycle exists.

### Task 2: Add mode-aware shell and URL-driven dashboard state

**Files:**
- Modify: `dashboard.py:1446-2785`
- Test: `tests/test_dashboard.py`

- [ ] **Step 1: Write the failing test**

Add a route-level test for explicit experimental mode:

```python
def test_dashboard_renders_experimental_mode_when_requested(self) -> None:
    dashboard = importlib.import_module("dashboard")
    with patch.object(dashboard, "DASHBOARD_PASSWORD", "dummy"):
        client = dashboard.app.test_client()
        response = client.get("/?mode=experimental", headers={"Authorization": "Basic YXV0b2NpdmlsOmR1bW15"})

    self.assertEqual(response.status_code, 200)
    page = response.get_data(as_text=True)
    self.assertIn("data-default-mode=\"experimental\"", page)
    self.assertIn("Experimental research workspace", page)
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
python -m pytest tests/test_dashboard.py -k "defaults_to_normal_mode or renders_experimental_mode_when_requested or normal_mode_does_not_render_experimental_sections_by_default" -v
```

Expected:

- FAIL because the route and template are still single-mode.

- [ ] **Step 3: Write minimal implementation**

In `dashboard.py`, change the `index()` route so it reads query-string mode and injects it into the template:

```python
@app.route("/")
def index():
    requested_mode = str(request.args.get("mode") or "normal").strip().lower()
    dashboard_mode = "experimental" if requested_mode == "experimental" else "normal"
    return render_template_string(HTML, dashboard_mode=dashboard_mode)
```

Add shell markup near the top of `HTML`:

```html
<body data-default-mode="{{ dashboard_mode }}">
  <header id="mode-switch">
    <a href="/?mode=normal" class="mode-tab">Normal Mode</a>
    <a href="/?mode=experimental" class="mode-tab">Experimental Mode</a>
  </header>
```

Add the foundational mode helpers in JS:

```javascript
const dashboardMode = document.body.dataset.defaultMode || 'normal';

function isNormalMode(){
  return dashboardMode === 'normal';
}

function isExperimentalMode(){
  return dashboardMode === 'experimental';
}
```

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
python -m pytest tests/test_dashboard.py -k "defaults_to_normal_mode or renders_experimental_mode_when_requested or normal_mode_does_not_render_experimental_sections_by_default" -v
```

Expected:

- PASS.

- [ ] **Step 5: Commit**

```bash
git add dashboard.py tests/test_dashboard.py
git commit -m "feat: add mode-aware dashboard shell"
```

### Task 3: Replace eager `loadAll()` with selective mode loading

**Files:**
- Modify: `dashboard.py:1663-1675`
- Test: `tests/test_dashboard.py`

- [ ] **Step 1: Write the failing test**

Add a template-level test that locks the selective loader:

```python
def test_dashboard_uses_mode_aware_loading(self) -> None:
    page = Path("dashboard.py").read_text(encoding="utf-8")
    self.assertIn("async function loadNormalMode()", page)
    self.assertIn("async function loadExperimentalMode()", page)
    self.assertIn("if (isExperimentalMode()) {", page)
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
python -m pytest tests/test_dashboard.py -k "uses_mode_aware_loading" -v
```

Expected:

- FAIL because the current dashboard still uses the broad eager loading pattern.

- [ ] **Step 3: Write minimal implementation**

Replace the current all-at-once initializer with explicit loaders:

```javascript
async function loadNormalMode() {
  await Promise.all([
    loadStatus(),
    loadOverview(),
    loadNormalSummary(),
    loadNormalWarnings(),
  ]);
}

async function loadExperimentalMode() {
  await Promise.all([
    loadStatus(),
    loadOverview(),
    loadRunHistory(),
    loadLog(),
    loadOptuna(),
    loadValidation(),
    loadDesign(),
    loadFieldValidation(),
    loadPlots(),
  ]);
}

async function loadAll() {
  if (isExperimentalMode()) {
    await loadExperimentalMode();
    return;
  }
  await loadNormalMode();
}
```

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
python -m pytest tests/test_dashboard.py -k "uses_mode_aware_loading" -v
```

Expected:

- PASS.

- [ ] **Step 5: Commit**

```bash
git add dashboard.py tests/test_dashboard.py
git commit -m "refactor: split dashboard loading by mode"
```

### Task 4: Build the normal-mode summary view with real artifact-backed trust cards

**Files:**
- Modify: `dashboard.py:109-425`
- Modify: `dashboard.py:1446-1616`
- Modify: `dashboard.py:1780-2408`
- Test: `tests/test_dashboard.py`

- [ ] **Step 1: Write the failing test**

Add endpoint and page tests for explicit warning states:

```python
def test_normal_mode_surfaces_missing_artifacts_as_warnings(self) -> None:
    dashboard = importlib.import_module("dashboard")
    with tempfile.TemporaryDirectory() as tmpdir:
        outputs_dir = Path(tmpdir)
        (outputs_dir / "baseline_metrics.json").write_text(
            json.dumps({"model_name": "RandomForestRegressor", "composite_score": 0.9}),
            encoding="utf-8",
        )
        with patch.object(dashboard, "OUTPUTS_DIR", outputs_dir), patch.object(dashboard, "DASHBOARD_PASSWORD", "dummy"):
            client = dashboard.app.test_client()
            response = client.get("/?mode=normal", headers={"Authorization": "Basic YXV0b2NpdmlsOmR1bW15"})

    page = response.get_data(as_text=True)
    self.assertIn("Missing artifact", page)
    self.assertIn("Final holdout metrics not available", page)
    self.assertNotIn("0.8400", page)
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
python -m pytest tests/test_dashboard.py -k "surfaces_missing_artifacts_as_warnings" -v
```

Expected:

- FAIL because normal mode warning cards and explicit absence states are not yet implemented.

- [ ] **Step 3: Write minimal implementation**

Add a real summary payload helper in `dashboard.py`:

```python
def build_normal_mode_payload():
    final_raw = load_final_holdout_payload(OUTPUTS_DIR)
    best_source = load_search_selection_payload(OUTPUTS_DIR, final_raw)
    best = normalize_result_payload(best_source)
    final = normalize_final_payload(final_raw)
    warnings = []
    if final.get("holdout_source_type") is None:
        warnings.append({
            "level": "warn",
            "title": "Missing artifact",
            "message": "Final holdout metrics not available.",
        })
    return {"best": best, "final": final, "warnings": warnings}
```

Add normal-only sections in the template:

```html
<section class="section" id="normal-home">
  <div class="section-title">Project Health</div>
  <div class="section-sub">Clear summary for non-technical users</div>
  <div class="card-grid" id="normal-health-cards"></div>
  <div class="card-grid" id="normal-warning-cards"></div>
</section>
```

Add JS renderers that use only live endpoint data:

```javascript
async function loadNormalSummary() {
  const d = await fetch('/api/overview').then(r => r.json()).catch(() => ({}));
  const best = d.best || {};
  const final = d.final || {};
  document.getElementById('normal-health-cards').innerHTML = `
    <div class="card">
      <div class="card-label">Project Health</div>
      <div class="card-title">${escapeHtml(best.model_name || 'No selected model')}</div>
      <div class="metric"><div class="metric-label">Validation verdict</div><div class="metric-value sm">${escapeHtml(best.validation_verdict || best.validation || 'Not available')}</div></div>
      <div class="metric"><div class="metric-label">Holdout composite</div><div class="metric-value sm">${hasValue(final.holdout_composite) ? formatMetric(final.holdout_composite, 4) : 'Not available'}</div></div>
      <div class="metric"><div class="metric-label">Uncertainty coverage</div><div class="metric-value sm">${hasValue(final.uncertainty_coverage) ? formatPercent(final.uncertainty_coverage) : 'Not available'}</div></div>
    </div>
    <div class="card">
      <div class="card-label">Next Action</div>
      <div class="card-title">${hasValue(final.holdout_composite) ? 'Review the current winner' : 'Review missing evidence'}</div>
      <div class="metric"><div class="metric-label">Recommended next step</div><div class="metric-value sm">${hasValue(final.holdout_composite) ? 'Inspect holdout and uncertainty before sharing results.' : 'Run final report artifacts before trusting the result.'}</div></div>
    </div>
  `;
}

async function loadNormalWarnings() {
  const d = await fetch('/api/overview').then(r => r.json()).catch(() => ({}));
  const warnings = [];
  if (!hasValue(d.final?.holdout_composite)) {
    warnings.push({ title: 'Missing artifact', message: 'Final holdout metrics not available.' });
  }
  document.getElementById('normal-warning-cards').innerHTML = warnings.length
    ? warnings.map(item => `
      <div class="card">
        <div class="card-label">Warning</div>
        <div class="card-title">${escapeHtml(item.title)}</div>
        <div class="metric"><div class="metric-label">Reason</div><div class="metric-value sm">${escapeHtml(item.message)}</div></div>
      </div>
    `).join('')
    : '';
}
```

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
python -m pytest tests/test_dashboard.py -k "surfaces_missing_artifacts_as_warnings" -v
```

Expected:

- PASS.

- [ ] **Step 5: Commit**

```bash
git add dashboard.py tests/test_dashboard.py
git commit -m "feat: add normal dashboard summary and warning states"
```

### Task 5: Hide experimental-heavy sections from normal mode and mount a dedicated experimental workspace

**Files:**
- Modify: `dashboard.py:1456-1622`
- Modify: `dashboard.py:2409-2591`
- Test: `tests/test_dashboard.py`

- [ ] **Step 1: Write the failing test**

Add page assertions for mode-specific content:

```python
def test_experimental_mode_renders_research_workspace(self) -> None:
    dashboard = importlib.import_module("dashboard")
    with patch.object(dashboard, "DASHBOARD_PASSWORD", "dummy"):
        client = dashboard.app.test_client()
        response = client.get("/?mode=experimental", headers={"Authorization": "Basic YXV0b2NpdmlsOmR1bW15"})

    page = response.get_data(as_text=True)
    self.assertIn("Experimental research workspace", page)
    self.assertIn("Cross-validation metrics", page)
    self.assertIn("Design scenarios / trade-offs", page)
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
python -m pytest tests/test_dashboard.py -k "experimental_mode_renders_research_workspace" -v
```

Expected:

- FAIL because the page still mixes all sections in one default flow.

- [ ] **Step 3: Write minimal implementation**

Wrap the heavy sections in an experimental workspace block:

```html
<section class="section experimental-only" id="experimental-home">
  <div class="section-title">Experimental research workspace</div>
  <div class="section-sub">Research diagnostics and source-separated evidence</div>
</section>
```

Gate section visibility with mode-aware classes:

```javascript
function applyModeVisibility() {
  document.querySelectorAll('.experimental-only').forEach(el => {
    el.style.display = isExperimentalMode() ? '' : 'none';
  });
  document.querySelectorAll('.normal-only').forEach(el => {
    el.style.display = isNormalMode() ? '' : 'none';
  });
}
```

Call it during boot:

```javascript
window.addEventListener('DOMContentLoaded', async () => {
  applyModeVisibility();
  await loadAll();
  initScrollSpy();
});
```

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
python -m pytest tests/test_dashboard.py -k "experimental_mode_renders_research_workspace" -v
```

Expected:

- PASS.

- [ ] **Step 5: Commit**

```bash
git add dashboard.py tests/test_dashboard.py
git commit -m "feat: separate normal and experimental dashboard views"
```

### Task 6: Redesign experimental mode for clearer research navigation and design trade-off readability

**Files:**
- Modify: `dashboard.py:1587-1616`
- Modify: `dashboard.py:2409-2591`
- Test: `tests/test_dashboard.py`

- [ ] **Step 1: Write the failing test**

Add a test that locks clearer design comparison and real-source labels:

```python
def test_experimental_design_view_uses_real_source_labels(self) -> None:
    source = Path("dashboard.py").read_text(encoding="utf-8")
    self.assertIn("Source-separated batch comparison", source)
    self.assertIn("Source: ${escapeHtml(s.source_mode || 'design_single')}", source)
    self.assertIn("comparison-table", source)
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
python -m pytest tests/test_dashboard.py -k "experimental_design_view_uses_real_source_labels" -v
```

Expected:

- FAIL if the view is still using old labels or less structured comparison copy.

- [ ] **Step 3: Write minimal implementation**

Refine the experimental headings and layout copy in `dashboard.py`:

```html
<div class="section-sub">Source-separated batch comparison and per-scenario research cards</div>
```

Refine design rendering to show readable trade-offs from actual artifacts only:

```javascript
html += `<div class="table-wrap" style="margin-bottom:24px">
  <table class="comparison-table">
    <thead><tr>
      <th>Artifact source</th>
      <th>Target MPa</th>
      <th>Predicted MPa</th>
      <th>Validator verdict</th>
      <th>Interval width</th>
      <th>Cement saving</th>
    </tr></thead>
```

Keep the card details tied to:

- `source_mode`
- `_filename`
- `uncertainty_interval`
- `estimated_cement_saving_vs_reference`
- `validation_verdict`

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
python -m pytest tests/test_dashboard.py -k "experimental_design_view_uses_real_source_labels" -v
```

Expected:

- PASS.

- [ ] **Step 5: Commit**

```bash
git add dashboard.py tests/test_dashboard.py
git commit -m "refactor: clarify experimental research layout"
```

### Task 7: Full verification pass

**Files:**
- Modify: none
- Test: `tests/test_dashboard.py`

- [ ] **Step 1: Run dashboard test suite**

Run:

```bash
python -m pytest tests/test_dashboard.py -v
```

Expected:

- PASS with all dashboard mode, warning, and research rendering tests green.

- [ ] **Step 2: Run artifact integrity regressions**

Run:

```bash
python -m pytest tests/test_artifact_contracts.py tests/test_holdout_integrity.py -v
```

Expected:

- PASS, confirming no regression to source separation or holdout integrity.

- [ ] **Step 3: Inspect worktree state**

Run:

```bash
git status --short
```

Expected:

- only intended dashboard and test changes appear, with unrelated `.claude/` left untouched.

- [ ] **Step 4: Commit final verified implementation**

```bash
git add dashboard.py tests/test_dashboard.py
git commit -m "feat: redesign dashboard for normal and experimental modes"
```
