# LLM Proposal Backend Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a reusable LLM backend abstraction with Ollama, OpenAI, and hybrid modes, then integrate it into the governed research loop without breaking validator, uncertainty, reporting, or `final_metrics.json` governance.

**Architecture:** Introduce a provider-agnostic transport layer in `llm_backend.py`, a repository-aware `proposal_engine.py`, and thin integration points in `research_loop.py` and `search.py`. Keep proposal generation separate from experiment execution, keep/revert, and artifact validation.

**Tech Stack:** Python 3, `requests`, optional local `ollama` CLI, existing YAML config and unittest test suite.

---

## Chunk 1: Backend Abstraction

### Task 1: Add backend resolution tests

**Files:**
- Create: `E:\Random IDEA\AutoResearch\auto-civil-lab\tests\test_llm_backend.py`
- Modify: `E:\Random IDEA\AutoResearch\auto-civil-lab\config.yaml`

- [ ] **Step 1: Write the failing test**

```python
def test_resolve_backend_builds_ollama_backend_for_ollama_mode():
    config = {"llm": {"enabled": True, "backend_mode": "ollama", "ollama": {"model": "qwen3:8b"}}}
    backend = resolve_backend(config)
    assert backend.backend_name == "ollama"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_llm_backend -v`
Expected: FAIL because `resolve_backend` does not exist yet.

- [ ] **Step 3: Write minimal implementation**

Create `llm_backend.py` with backend classes and `resolve_backend(config)`.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m unittest tests.test_llm_backend -v`
Expected: PASS for backend resolution tests.

### Task 2: Add hybrid fallback tests

**Files:**
- Modify: `E:\Random IDEA\AutoResearch\auto-civil-lab\tests\test_llm_backend.py`
- Modify: `E:\Random IDEA\AutoResearch\auto-civil-lab\llm_backend.py`

- [ ] **Step 1: Write the failing test**

```python
def test_hybrid_backend_uses_fallback_when_primary_unavailable():
    primary = StubBackend("ollama", available=False)
    fallback = StubBackend("openai", available=True, response={"text": "ok"})
    backend = HybridBackend(primary=primary, fallback=fallback)
    response = backend.generate_text("hello")
    assert response["backend"] == "openai"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_llm_backend -v`
Expected: FAIL because fallback behavior is incomplete.

- [ ] **Step 3: Write minimal implementation**

Implement availability checks, fallback ordering, and backend metadata.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m unittest tests.test_llm_backend -v`
Expected: PASS.

## Chunk 2: Proposal Engine

### Task 3: Add proposal parsing tests

**Files:**
- Create: `E:\Random IDEA\AutoResearch\auto-civil-lab\tests\test_proposal_engine.py`
- Create: `E:\Random IDEA\AutoResearch\auto-civil-lab\proposal_engine.py`

- [ ] **Step 1: Write the failing test**

```python
def test_generate_experiment_proposals_returns_validated_proposals():
    backend = StubBackend("ollama", available=True, response={"text": '[{"model_name":"RandomForestRegressor","params":{"n_estimators":200}}]'})
    proposals = generate_experiment_proposals(...)
    assert proposals[0]["model_name"] == "RandomForestRegressor"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_proposal_engine -v`
Expected: FAIL because the proposal engine does not exist yet.

- [ ] **Step 3: Write minimal implementation**

Implement prompt building, JSON extraction, validation against available model config, and summary methods.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m unittest tests.test_proposal_engine -v`
Expected: PASS.

### Task 4: Add invalid-response handling tests

**Files:**
- Modify: `E:\Random IDEA\AutoResearch\auto-civil-lab\tests\test_proposal_engine.py`
- Modify: `E:\Random IDEA\AutoResearch\auto-civil-lab\proposal_engine.py`

- [ ] **Step 1: Write the failing test**

```python
def test_generate_experiment_proposals_returns_empty_list_for_invalid_json():
    backend = StubBackend("ollama", available=True, response={"text": "not-json"})
    proposals = generate_experiment_proposals(...)
    assert proposals == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_proposal_engine -v`
Expected: FAIL because invalid-response handling is missing.

- [ ] **Step 3: Write minimal implementation**

Handle malformed responses without raising and return structured diagnostics.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m unittest tests.test_proposal_engine -v`
Expected: PASS.

## Chunk 3: Governed Loop Integration

### Task 5: Add governed-loop fallback test

**Files:**
- Modify: `E:\Random IDEA\AutoResearch\auto-civil-lab\tests\test_research_loop.py`
- Modify: `E:\Random IDEA\AutoResearch\auto-civil-lab\research_loop.py`

- [ ] **Step 1: Write the failing test**

```python
def test_loop_falls_back_to_deterministic_scouts_when_backend_unavailable(tmp_path):
    result = run_loop_with_unavailable_backend(tmp_path)
    assert result["proposal_mode"] == "deterministic_fallback"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_research_loop -v`
Expected: FAIL because proposal backend fallback is not wired in.

- [ ] **Step 3: Write minimal implementation**

Inject the proposal engine into `research_loop.py` and preserve deterministic fallback.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m unittest tests.test_research_loop -v`
Expected: PASS.

### Task 6: Add final artifact governance regression test

**Files:**
- Modify: `E:\Random IDEA\AutoResearch\auto-civil-lab\tests\test_research_loop.py`
- Modify: `E:\Random IDEA\AutoResearch\auto-civil-lab\research_loop.py`

- [ ] **Step 1: Write the failing test**

```python
def test_loop_keeps_final_metrics_as_source_of_truth_with_llm_proposals(tmp_path):
    result = run_loop_with_stub_backend(tmp_path)
    assert result["source_of_truth"] == "final_metrics.json"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_research_loop -v`
Expected: FAIL if LLM integration bypasses existing final sync logic.

- [ ] **Step 3: Write minimal implementation**

Keep artifact syncing in the existing governance path and expose proposal metadata separately.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m unittest tests.test_research_loop -v`
Expected: PASS.

## Chunk 4: Legacy Search Compatibility and Docs

### Task 7: Route legacy search proposals through the backend abstraction

**Files:**
- Modify: `E:\Random IDEA\AutoResearch\auto-civil-lab\search.py`
- Modify: `E:\Random IDEA\AutoResearch\auto-civil-lab\llm_proposer.py`
- Modify: `E:\Random IDEA\AutoResearch\auto-civil-lab\config.yaml`

- [ ] **Step 1: Write the failing test**

Add a backend-resolution test or compatibility test showing that legacy search can still obtain validated proposals through the new abstraction.

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_llm_backend tests.test_proposal_engine -v`
Expected: FAIL until legacy compatibility is wired.

- [ ] **Step 3: Write minimal implementation**

Refactor `llm_proposer.py` into a compatibility layer or adapter over the new backend/proposal engine.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m unittest tests.test_llm_backend tests.test_proposal_engine -v`
Expected: PASS.

### Task 8: Update docs and examples

**Files:**
- Modify: `E:\Random IDEA\AutoResearch\auto-civil-lab\README.md`
- Modify: `E:\Random IDEA\AutoResearch\auto-civil-lab\docs\autoresearch_workflow.md`
- Create: `E:\Random IDEA\AutoResearch\auto-civil-lab\docs\examples\llm_backend.example.json`

- [ ] **Step 1: Document the new configuration and runtime behavior**
- [ ] **Step 2: Add example outputs for backend selection and proposal summaries**
- [ ] **Step 3: Verify docs reflect fallback behavior and source-of-truth rules**

## Verification

- [ ] Run: `python -m unittest tests.test_llm_backend tests.test_proposal_engine tests.test_research_loop tests.test_research_protocol tests.test_research_lab -v`
- [ ] Run: `python -m py_compile llm_backend.py proposal_engine.py llm_proposer.py research_loop.py search.py autocivil_autosearch_session.py`
- [ ] Run: `python -c "import llm_backend, proposal_engine, research_loop, search; print('ok')"`

## Notes for Implementation

- Do not modify validator or uncertainty internals.
- Do not change the meaning of `final_metrics.json`.
- Preserve deterministic research behavior when no LLM backend is configured.
- Keep provider-specific code out of `research_loop.py` and `search.py`.
