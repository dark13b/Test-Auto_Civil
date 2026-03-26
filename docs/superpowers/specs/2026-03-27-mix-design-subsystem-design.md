# Mix Design Subsystem Design

**Date:** 2026-03-27

**Goal**

Replace the monolithic `design_tool.py` implementation with a package-first concrete mix design subsystem that separates prediction, candidate generation, objective scoring, and scenario packaging behind a new canonical facade: `ConcreteMixDesignService`.

## Context

[`/E:/Random IDEA/AutoResearch/auto-civil-lab/design_tool.py`](/E:/Random IDEA/AutoResearch/auto-civil-lab/design_tool.py) currently owns:
- model/config loading
- constraint normalization and target-dependent bound overlays
- reference-dataset prior generation
- Optuna search and trial sampling
- feature engineering and prediction
- uncertainty estimation
- engineering validator execution
- objective math and candidate ranking
- result packaging and artifact export
- CLI wiring

That file currently mixes distinct responsibilities inside `_evaluate_mix()` and `optimize()`. The result contract is dict-based, winner-centric, and optimized for backward compatibility rather than explicit comparison. Constraints and ranking logic are partially implicit, which makes the system harder to evolve and harder to test.

The target architecture is a canonical subsystem package that makes responsibilities explicit and keeps `MixDesignOptimizer.optimize(...)` only as a thin deprecated translation layer.

## Design Summary

### 1. Canonical Package Boundary

Add a new package rooted at [`/E:/Random IDEA/AutoResearch/auto-civil-lab/mix_design/`](/E:/Random IDEA/AutoResearch/auto-civil-lab/mix_design/) with the following modules:
- `contracts.py`
- `constraints.py`
- `predictor.py`
- `objective_engine.py`
- `optimizer.py`
- `facade.py`
- `exporters.py`
- `__init__.py`

This package becomes the only canonical home for new design subsystem logic. New internal code must not import or depend on `MixDesignOptimizer`.

### 2. Canonical Public Interface

The official public entrypoint is:

```python
class ConcreteMixDesignService:
    @classmethod
    def from_default_artifacts(
        cls,
        model_path: str | Path | None = None,
        config_path: str | Path | None = None,
    ) -> "ConcreteMixDesignService": ...

    def compare_scenarios(self, request: MixDesignRequest) -> ScenarioComparisonResult: ...

    def export_comparison(
        self,
        result: ScenarioComparisonResult,
        output_path: str | Path,
    ) -> Path: ...
```

`compare_targets(...)` and `export_batch(...)` are deferred to a later phase unless an immediate consumer requires them.

### 3. Typed Contracts

The canonical result format is typed and comparison-first.

Core request objects:
- `DesignContext`
- `RangeConstraint`
- `DesignConstraints`
- `ObjectiveSpec`
- `MixDesignRequest`

Core result objects:
- `UncertaintyInterval`
- `PredictionResult`
- `ConstraintCheck`
- `ConstraintEvaluation`
- `ValidatorOutcome`
- `ObjectiveComponentScore`
- `ObjectiveScorecard`
- `CandidateScenario`
- `ScenarioComparisonResult`

Key properties of the result contract:
- each candidate carries prediction, uncertainty, constraint evaluation, validator outcome, and objective breakdown
- multiple candidate scenarios are returned together
- the top-ranked scenario is identified by id, not by collapsing the system into a winner-only contract
- uncertainty and validator outputs are attached inside canonical result objects, not as side payloads

### 4. Predictor Role

[`/E:/Random IDEA/AutoResearch/auto-civil-lab/mix_design/predictor.py`](/E:/Random IDEA/AutoResearch/auto-civil-lab/mix_design/predictor.py) owns:
- turning a candidate mix into the model input frame
- feature engineering
- model inference
- uncertainty interval generation
- model lineage attachment

The predictor returns a `PredictionResult`. It does not:
- generate candidate mixes
- rank scenarios
- compute objective penalties
- decide feasibility

### 5. Constraints Role

[`/E:/Random IDEA/AutoResearch/auto-civil-lab/mix_design/constraints.py`](/E:/Random IDEA/AutoResearch/auto-civil-lab/mix_design/constraints.py) owns:
- normalization of user constraints into `DesignConstraints`
- target-dependent constraint overlays
- explicit per-field and engineered-ratio checks
- conversion of visible checks into `ConstraintEvaluation`

Constraints become explicit and testable. No helper should silently bury hard-coded feasibility rules.

### 6. Objective Engine Role

[`/E:/Random IDEA/AutoResearch/auto-civil-lab/mix_design/objective_engine.py`](/E:/Random IDEA/AutoResearch/auto-civil-lab/mix_design/objective_engine.py) owns scoring and ranking policy through explicit objective components.

Initial supported objectives:
- `target_fit`
- `cement_penalty`
- `co2_proxy`
- `validator_risk`

The objective engine returns an `ObjectiveScorecard` with:
- per-objective raw value
- weight
- weighted contribution
- textual explanation
- total score

This layer is designed so later objectives can be added without changing optimizer or predictor contracts.

### 7. Optimizer Role

[`/E:/Random IDEA/AutoResearch/auto-civil-lab/mix_design/optimizer.py`](/E:/Random IDEA/AutoResearch/auto-civil-lab/mix_design/optimizer.py) owns:
- dataset-conditioned engineering priors
- warm starts
- Optuna trial orchestration
- candidate proposal generation under explicit constraints

The optimizer should accept a scoring/evaluation callback supplied by the facade. It does not own:
- prediction internals
- objective definitions
- legacy dict translation

### 8. Scenario Packaging Role

[`/E:/Random IDEA/AutoResearch/auto-civil-lab/mix_design/facade.py`](/E:/Random IDEA/AutoResearch/auto-civil-lab/mix_design/facade.py) assembles the subsystem flow:
1. normalize request constraints
2. generate candidate proposals through the optimizer
3. predict and attach uncertainty
4. run explicit constraint evaluation
5. run validator evaluation
6. score through the objective engine
7. package ranked `CandidateScenario` objects
8. return `ScenarioComparisonResult`

This is the only place where subsystem collaborators are composed into the final comparison contract.

### 9. Legacy Adapter

[`/E:/Random IDEA/AutoResearch/auto-civil-lab/design_tool.py`](/E:/Random IDEA/AutoResearch/auto-civil-lab/design_tool.py) remains in place for backward compatibility only.

`MixDesignOptimizer.optimize(...)` becomes:
- a thin deprecated adapter
- a translator from old args to `MixDesignRequest`
- a translator from `ScenarioComparisonResult` back to the historical dict shape

The adapter must not contain:
- prediction logic
- constraint logic
- scoring logic
- optimizer logic

Legacy compatibility is strictly translation, not duplicated behavior.

### 10. Result Translation Contract

The adapter translation contract is:
- call `ConcreteMixDesignService.compare_scenarios(request)`
- take the best ranked `CandidateScenario`
- translate it into the existing winner-oriented dict keys
- populate `ranked_candidates` by translating the top returned scenarios
- emit a `DeprecationWarning`

The canonical result remains the source of truth. The legacy dict is a lossy projection of the canonical comparison result.

### 11. Export Model

[`/E:/Random IDEA/AutoResearch/auto-civil-lab/mix_design/exporters.py`](/E:/Random IDEA/AutoResearch/auto-civil-lab/mix_design/exporters.py) will eventually own canonical artifact export from `ScenarioComparisonResult`.

Phase 1 only requires:
- `export_comparison(result, output_path)`

Batch export remains a later phase if no immediate consumer requires it.

## Error Handling

- Missing model artifact: fail during service construction, not deep inside candidate evaluation.
- Invalid constraint ranges: raise explicit validation errors from `constraints.py`.
- Empty candidate set: fail from optimizer/facade with an explicit error.
- Invalid objective names or disabled objective set: fail during request normalization.
- Validator or uncertainty availability issues: surface them through canonical result metadata or explicit exceptions, not silent fallback fields.

## Testing Strategy

Tests should be organized by subsystem responsibility:
- `tests/test_design_constraints.py`
- `tests/test_design_predictor.py`
- `tests/test_objective_engine.py`
- `tests/test_design_facade.py`
- `tests/test_design_tool_legacy_adapter.py`

Priority assertions:
- constraints are explicit, visible, and testable
- predictor outputs uncertainty and engineered features without ranking concerns
- objective scoring is independently testable
- canonical comparison results return multiple ranked scenarios
- legacy output is produced only by translating canonical results

## Migration Strategy

Stage the refactor conservatively:
1. add typed contracts and isolated subsystems
2. implement the canonical facade
3. convert `design_tool.py` into a deprecated adapter
4. only then migrate downstream consumers such as `dashboard.py`

This keeps rollback simple and prevents partially migrated consumers from depending on unstable intermediate shapes.
