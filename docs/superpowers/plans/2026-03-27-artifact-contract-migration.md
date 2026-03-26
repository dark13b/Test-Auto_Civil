# Artifact Contract Migration Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace ambiguous metric artifacts with a typed contract that separates search-time selection from terminal holdout evaluation and validates artifacts before save.

**Architecture:** Introduce a shared artifact-contract module with typed schema objects, explicit legacy-read mappers, and central validation helpers. Refactor search to emit only selection-time artifacts, then refactor report to emit the only canonical final artifact, `final_holdout_evaluation.json`, while dashboard, reports, uncertainty, and design-tool consumers read the same normalized contract.

**Tech Stack:** Python, dataclasses, unittest/pytest-style tests, existing JSON artifact pipeline in `train_impl.py`

---

## Chunk 1: Schema Foundation

### Task 1: Add typed schema and compatibility infrastructure

**Files:**
- Create: `E:\Random IDEA\AutoResearch\auto-civil-lab\artifact_contracts.py`
- Modify: `E:\Random IDEA\AutoResearch\auto-civil-lab\train_impl.py`
- Test: `E:\Random IDEA\AutoResearch\auto-civil-lab\tests\test_artifact_contracts.py`

- [ ] **Step 1: Write failing tests for schema validation and deprecated mapping**

```python
def test_validate_search_selection_rejects_holdout_metrics(): ...
def test_validate_final_holdout_rejects_legacy_test_metrics(): ...
def test_map_legacy_final_metrics_payload_emits_deprecations(): ...
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_artifact_contracts.py -v`
Expected: FAIL because the contract module and hooks do not exist yet.

- [ ] **Step 3: Implement the contract module and save-time validation hook**

```python
class ArtifactValidationError(ValueError): ...
def validate_artifact_payload(...): ...
def map_legacy_payload(...): ...
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_artifact_contracts.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add artifact_contracts.py train_impl.py tests/test_artifact_contracts.py docs/superpowers/plans/2026-03-27-artifact-contract-migration.md
git commit -m "refactor: add artifact contract schemas and validation hooks"
```

## Chunk 2: Search-Time Artifact Split

### Task 2: Refactor selection producers to canonical search artifacts

**Files:**
- Modify: `E:\Random IDEA\AutoResearch\auto-civil-lab\train_impl.py`
- Modify: `E:\Random IDEA\AutoResearch\auto-civil-lab\search.py`
- Modify: `E:\Random IDEA\AutoResearch\auto-civil-lab\loop_artifacts.py`
- Modify: `E:\Random IDEA\AutoResearch\auto-civil-lab\research_protocol.py`
- Modify: `E:\Random IDEA\AutoResearch\auto-civil-lab\research_loop.py`
- Test: `E:\Random IDEA\AutoResearch\auto-civil-lab\tests\test_research_loop.py`
- Test: `E:\Random IDEA\AutoResearch\auto-civil-lab\tests\test_search.py`
- Test: `E:\Random IDEA\AutoResearch\auto-civil-lab\tests\test_integrity_checks.py`

- [ ] **Step 1: Write failing tests for `search_selection.json` and no legacy write aliases**
- [ ] **Step 2: Run the targeted tests and confirm the expected failures**
- [ ] **Step 3: Emit only `cross_validation` and `selection_validation` in search-time artifacts**
- [ ] **Step 4: Run the targeted tests and confirm the search contract passes**
- [ ] **Step 5: Commit**

## Chunk 3: Terminal Holdout Artifact

### Task 3: Make holdout terminal-only and canonical

**Files:**
- Modify: `E:\Random IDEA\AutoResearch\auto-civil-lab\report.py`
- Modify: `E:\Random IDEA\AutoResearch\auto-civil-lab\uncertainty.py`
- Modify: `E:\Random IDEA\AutoResearch\auto-civil-lab\benchmark.py`
- Test: `E:\Random IDEA\AutoResearch\auto-civil-lab\tests\test_report.py`
- Test: `E:\Random IDEA\AutoResearch\auto-civil-lab\tests\test_holdout_integrity.py`

- [ ] **Step 1: Write failing tests for `final_holdout_evaluation.json` as the only canonical final artifact**
- [ ] **Step 2: Run targeted tests to verify the failure mode**
- [ ] **Step 3: Refactor report-time holdout generation and uncertainty bindings to the new contract**
- [ ] **Step 4: Run targeted tests to verify terminal holdout semantics**
- [ ] **Step 5: Commit**

## Chunk 4: Shared Consumers and Design Tool

### Task 4: Migrate dashboard, reports, and design tooling to the shared contract

**Files:**
- Modify: `E:\Random IDEA\AutoResearch\auto-civil-lab\dashboard.py`
- Modify: `E:\Random IDEA\AutoResearch\auto-civil-lab\design_tool.py`
- Modify: `E:\Random IDEA\AutoResearch\auto-civil-lab\autocivil_autosearch_session.py`
- Create: `E:\Random IDEA\AutoResearch\auto-civil-lab\docs\artifact-contract-migration.md`
- Test: `E:\Random IDEA\AutoResearch\auto-civil-lab\tests\test_dashboard.py`
- Test: `E:\Random IDEA\AutoResearch\auto-civil-lab\tests\test_design_tool.py`

- [ ] **Step 1: Write failing tests for shared contract consumption and deprecated read compatibility**
- [ ] **Step 2: Run targeted tests to verify the current mismatch**
- [ ] **Step 3: Migrate dashboard and design tool to the normalized contract**
- [ ] **Step 4: Run targeted tests and the final artifact suite**
- [ ] **Step 5: Commit**
