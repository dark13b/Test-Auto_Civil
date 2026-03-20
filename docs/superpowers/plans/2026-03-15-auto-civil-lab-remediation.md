# AutoCivil-Lab Remediation Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restore scientific holdout integrity, honest metric labeling, backend/output safety, and reproducibility across the AutoCivil-Lab pipeline.

**Architecture:** Keep the existing file boundaries, but route all search-time evaluation through a new validation partition, centralize integrity checks and model routing in focused helper modules, and harden the dashboard and research-state surfaces without changing the project’s main runtime flow. Verification is staged P0 to P2 so later fixes build on validated earlier boundaries.

**Tech Stack:** Python, scikit-learn, Flask, pandas, NumPy, Ollama/Qwen, unittest/pytest

---

### Task 1: Establish P0 Safety Net

**Files:**
- Create: `tests/test_integrity_checks.py`
- Create: `tests/test_holdout_integrity.py`
- Modify: `tests/test_search.py` or create it if absent

- [ ] **Step 1: Write failing tests for holdout leakage and CV-label fraud**
- [ ] **Step 2: Run the targeted tests and confirm they fail for the expected reason**
- [ ] **Step 3: Implement the minimal production changes for three-way splits and integrity checks**
- [ ] **Step 4: Re-run the targeted tests until they pass**

### Task 2: Fix P0 Runtime Paths

**Files:**
- Modify: `train.py`
- Modify: `research_loop.py`
- Modify: `uncertainty.py`
- Modify: `report.py`
- Modify: `search.py`
- Create: `integrity_checks.py`
- Modify: `config.yaml`

- [ ] **Step 1: Route search and calibration paths from `x_test`/`y_test` to `x_val`/`y_val`**
- [ ] **Step 2: Add the holdout-use assertion in the final report block**
- [ ] **Step 3: Replace ensemble fake-CV fields with real CV metrics and separate val metrics**
- [ ] **Step 4: Re-run focused P0 tests and repository grep checks**

### Task 3: Fix P1 Model Identity and LLM Control Paths

**Files:**
- Create: `tests/test_uncertainty.py`
- Create: `tests/test_model_routing.py`
- Modify: `tests/test_proposal_engine.py`
- Modify: `uncertainty.py`
- Modify: `report.py`
- Create: `model_routing.py`
- Modify: `proposal_engine.py`
- Modify: `llm_backend.py`
- Modify: `llm_proposer.py`
- Modify: `config.yaml`

- [ ] **Step 1: Write failing tests for uncertainty model identity, backend-aware model routing, and thinking-channel rejection**
- [ ] **Step 2: Run those tests and confirm they fail correctly**
- [ ] **Step 3: Implement minimal production changes to pass them**
- [ ] **Step 4: Re-run the targeted P1 tests**

### Task 4: Fix P1 Dashboard Integrity and Security

**Files:**
- Modify: `dashboard.py`
- Add or modify: dashboard tests if coverage is missing

- [ ] **Step 1: Write failing tests or reproduction checks for suspicious-count preservation, auth fallback, and untrusted text rendering**
- [ ] **Step 2: Replace unsafe DOM writes for untrusted content and enforce auth when password is absent**
- [ ] **Step 3: Add metric-source labeling so mixed-source metrics are explicit**
- [ ] **Step 4: Re-run targeted dashboard verification**

### Task 5: Fix P2 Reproducibility and State Integrity

**Files:**
- Modify: `research_lab.py`
- Modify: `research_protocol.py`
- Modify: `requirements.txt`
- Modify: `launch_auto_research.bat`
- Modify: `config.yaml`
- Modify: `tests/test_research_lab.py`
- Create: `tests/test_report.py`

- [ ] **Step 1: Write failing tests for log-space exploit behavior and duplicate experiment rejection**
- [ ] **Step 2: Implement minimal code for integrity validation, log-scale perturbation, dependency declaration, and launcher hardening**
- [ ] **Step 3: Re-run targeted P2 tests**

### Task 6: Final Verification

**Files:**
- Output: `test_results.txt`
- Output: `mini_run.txt`
- Output: `report_run.txt`

- [ ] **Step 1: Install missing test/runtime dependencies in the project virtualenv**
- [ ] **Step 2: Run the prompt-specified targeted verifications for P0 and P1**
- [ ] **Step 3: Run the full test suite**
- [ ] **Step 4: Run the mini research loop**
- [ ] **Step 5: Run report generation and capture artifacts**
