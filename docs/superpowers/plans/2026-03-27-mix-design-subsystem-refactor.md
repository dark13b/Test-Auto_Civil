# Mix Design Subsystem Refactor Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the monolithic design tool with a package-first mix design subsystem centered on `ConcreteMixDesignService`, typed scenario-comparison contracts, and a thin deprecated `MixDesignOptimizer` adapter.

**Architecture:** Introduce a new `mix_design` package that separates contracts, explicit constraints, predictor, objective engine, optimizer, facade orchestration, and exporters. Keep the legacy `design_tool.py` public surface only as a deprecation-emitting adapter that translates between the old dict contract and the canonical comparison result.

**Tech Stack:** Python, dataclasses, Optuna, pandas, numpy, existing validator and uncertainty modules, unittest/pytest-style tests

---

## Chunk 1: Phase A Foundations

### Task 1: Add typed request/result contracts

**Files:**
- Create: `E:\Random IDEA\AutoResearch\auto-civil-lab\mix_design\contracts.py`
- Create: `E:\Random IDEA\AutoResearch\auto-civil-lab\mix_design\__init__.py`
- Test: `E:\Random IDEA\AutoResearch\auto-civil-lab\tests\test_design_contracts.py`

- [ ] **Step 1: Write failing tests for request/result dataclasses**
- [ ] **Step 2: Run `pytest tests/test_design_contracts.py -v` and confirm failure because the package does not exist**
- [ ] **Step 3: Implement the canonical typed contracts**
- [ ] **Step 4: Run `pytest tests/test_design_contracts.py -v` and confirm pass**
- [ ] **Step 5: Commit**

### Task 2: Extract explicit constraint normalization and evaluation

**Files:**
- Create: `E:\Random IDEA\AutoResearch\auto-civil-lab\mix_design\constraints.py`
- Test: `E:\Random IDEA\AutoResearch\auto-civil-lab\tests\test_design_constraints.py`

- [ ] **Step 1: Write failing tests for normalized bounds, target-dependent overlays, and explicit pass/fail checks**
- [ ] **Step 2: Run `pytest tests/test_design_constraints.py -v` and confirm failure**
- [ ] **Step 3: Implement explicit constraint parsing and evaluation helpers**
- [ ] **Step 4: Run `pytest tests/test_design_constraints.py -v` and confirm pass**
- [ ] **Step 5: Commit**

### Task 3: Extract predictor and uncertainty attachment

**Files:**
- Create: `E:\Random IDEA\AutoResearch\auto-civil-lab\mix_design\predictor.py`
- Test: `E:\Random IDEA\AutoResearch\auto-civil-lab\tests\test_design_predictor.py`

- [ ] **Step 1: Write failing tests for prediction output, engineered features, and uncertainty linkage**
- [ ] **Step 2: Run `pytest tests/test_design_predictor.py -v` and confirm failure**
- [ ] **Step 3: Implement the predictor module with no ranking or constraint logic**
- [ ] **Step 4: Run `pytest tests/test_design_predictor.py -v` and confirm pass**
- [ ] **Step 5: Commit**

### Task 4: Extract explicit objective scoring

**Files:**
- Create: `E:\Random IDEA\AutoResearch\auto-civil-lab\mix_design\objective_engine.py`
- Test: `E:\Random IDEA\AutoResearch\auto-civil-lab\tests\test_objective_engine.py`

- [ ] **Step 1: Write failing tests for target fit, cement penalty, CO2 proxy, validator risk, and explanation fields**
- [ ] **Step 2: Run `pytest tests/test_objective_engine.py -v` and confirm failure**
- [ ] **Step 3: Implement the objective engine with explicit component scorecards**
- [ ] **Step 4: Run `pytest tests/test_objective_engine.py -v` and confirm pass**
- [ ] **Step 5: Commit**

## Chunk 2: Phase B Canonical Flow

### Task 5: Add the optimizer and canonical facade

**Files:**
- Create: `E:\Random IDEA\AutoResearch\auto-civil-lab\mix_design\optimizer.py`
- Create: `E:\Random IDEA\AutoResearch\auto-civil-lab\mix_design\facade.py`
- Test: `E:\Random IDEA\AutoResearch\auto-civil-lab\tests\test_design_facade.py`

- [ ] **Step 1: Write failing tests for multi-scenario comparison and ranking**
- [ ] **Step 2: Run `pytest tests/test_design_facade.py -v` and confirm failure**
- [ ] **Step 3: Implement candidate generation and `ConcreteMixDesignService.compare_scenarios(...)`**
- [ ] **Step 4: Run `pytest tests/test_design_facade.py -v` and confirm pass**
- [ ] **Step 5: Commit**

### Task 6: Add canonical export support

**Files:**
- Create: `E:\Random IDEA\AutoResearch\auto-civil-lab\mix_design\exporters.py`
- Modify: `E:\Random IDEA\AutoResearch\auto-civil-lab\mix_design\facade.py`
- Test: `E:\Random IDEA\AutoResearch\auto-civil-lab\tests\test_design_facade.py`

- [ ] **Step 1: Write failing tests for `export_comparison(...)` from canonical results**
- [ ] **Step 2: Run the targeted facade/export tests and confirm failure**
- [ ] **Step 3: Implement export from the canonical result contract**
- [ ] **Step 4: Run the targeted facade/export tests and confirm pass**
- [ ] **Step 5: Commit**

## Chunk 3: Phase C Legacy Adapter

### Task 7: Convert `design_tool.py` into a deprecated translation adapter

**Files:**
- Modify: `E:\Random IDEA\AutoResearch\auto-civil-lab\design_tool.py`
- Test: `E:\Random IDEA\AutoResearch\auto-civil-lab\tests\test_design_tool_legacy_adapter.py`
- Test: `E:\Random IDEA\AutoResearch\auto-civil-lab\tests\test_design_tool.py`

- [ ] **Step 1: Write failing tests for deprecation warnings and canonical-result translation**
- [ ] **Step 2: Run `pytest tests/test_design_tool_legacy_adapter.py tests/test_design_tool.py -v` and confirm failure**
- [ ] **Step 3: Replace business logic in `design_tool.py` with a thin adapter over `ConcreteMixDesignService`**
- [ ] **Step 4: Run the adapter tests and confirm pass**
- [ ] **Step 5: Commit**

## Chunk 4: Phase D Optional Consumer Migration

### Task 8: Migrate downstream consumers only after package stability

**Files:**
- Modify: `E:\Random IDEA\AutoResearch\auto-civil-lab\dashboard.py`
- Test: `E:\Random IDEA\AutoResearch\auto-civil-lab\tests\test_dashboard.py`

- [ ] **Step 1: Write failing tests for canonical facade consumption in the dashboard**
- [ ] **Step 2: Run the targeted dashboard tests and confirm failure**
- [ ] **Step 3: Switch the dashboard to the canonical service**
- [ ] **Step 4: Run the targeted dashboard tests and confirm pass**
- [ ] **Step 5: Commit**

## Adapter Translation Contract

`MixDesignOptimizer.optimize(target_strength_mpa, constraints=None, context=None)` must:
- emit a `DeprecationWarning`
- construct a `MixDesignRequest`
- call `ConcreteMixDesignService.compare_scenarios(...)`
- translate the best `CandidateScenario` into the historical dict shape
- translate the returned scenario list into `ranked_candidates`
- avoid reimplementing scoring, prediction, validation, or constraint evaluation

Legacy dict compatibility requirements:
- preserve winner fields currently relied on by tests and exports
- preserve `ranked_candidates`
- source all winner values from canonical scenario objects
- never treat the legacy dict as the primary internal contract

## File-by-File Commit Sequence

1. `docs/superpowers/specs/2026-03-27-mix-design-subsystem-design.md`
2. `docs/superpowers/plans/2026-03-27-mix-design-subsystem-refactor.md`
3. `mix_design/contracts.py`, `mix_design/__init__.py`, `tests/test_design_contracts.py`
4. `mix_design/constraints.py`, `tests/test_design_constraints.py`
5. `mix_design/predictor.py`, `tests/test_design_predictor.py`
6. `mix_design/objective_engine.py`, `tests/test_objective_engine.py`
7. `mix_design/optimizer.py`, `mix_design/facade.py`, `tests/test_design_facade.py`
8. `mix_design/exporters.py`, related facade test updates
9. `design_tool.py`, `tests/test_design_tool_legacy_adapter.py`, `tests/test_design_tool.py`
10. optional later: `dashboard.py`, `tests/test_dashboard.py`

## Test Plan

- `tests/test_design_contracts.py`
  - contract construction
  - default objective specs
  - typed scenario comparison shape
- `tests/test_design_constraints.py`
  - default config mapping
  - target-dependent overlay behavior
  - visible constraint violations and pass states
- `tests/test_design_predictor.py`
  - prediction output shape
  - engineered feature attachment
  - uncertainty interval linkage
- `tests/test_objective_engine.py`
  - target fit scoring
  - cement penalty scoring
  - CO2 proxy scoring
  - validator risk scoring
  - explanation strings
- `tests/test_design_facade.py`
  - multiple scenarios returned
  - ranking explanation available
  - canonical result is comparison-first
- `tests/test_design_tool_legacy_adapter.py`
  - deprecation warning emitted
  - old dict shape translated from canonical results
- regression updates in `tests/test_design_tool.py`
  - existing external behavior still usable through the adapter

## Rollback Points

1. After docs only:
   - revert the spec and plan commits with no runtime impact.
2. After Phase A:
   - revert the `mix_design` package additions and tests while leaving `design_tool.py` untouched.
3. After Phase B:
   - revert the facade/optimizer/exporter additions and keep the old design tool path.
4. After Phase C:
   - restore the old `design_tool.py` implementation while keeping the new package dormant.
5. After Phase D:
   - point consumers back to the adapter if direct facade migration breaks downstream code.
