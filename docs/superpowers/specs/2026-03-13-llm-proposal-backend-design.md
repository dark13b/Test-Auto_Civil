# LLM Proposal Backend Design

**Date:** 2026-03-13

**Goal**

Add a provider abstraction that lets AutoCivil-Lab use structured LLM proposals from Ollama/Qwen, OpenAI, or a hybrid fallback path, while keeping Codex responsible for code edits, orchestration, keep/revert, and final artifact governance.

## Context

The current repository has two divergent paths:
- [`/E:/Random IDEA/AutoResearch/auto-civil-lab/research_loop.py`](/E:/Random IDEA/AutoResearch/auto-civil-lab/research_loop.py) is the active governed loop, but it generates deterministic experiments from [`/E:/Random IDEA/AutoResearch/auto-civil-lab/research_lab.py`](/E:/Random IDEA/AutoResearch/auto-civil-lab/research_lab.py).
- [`/E:/Random IDEA/AutoResearch/auto-civil-lab/search.py`](/E:/Random IDEA/AutoResearch/auto-civil-lab/search.py) contains an Ollama-specific proposal path through [`/E:/Random IDEA/AutoResearch/auto-civil-lab/llm_proposer.py`](/E:/Random IDEA/AutoResearch/auto-civil-lab/llm_proposer.py), but that integration is provider-specific and not shared with the governed loop.

The target architecture is option 2 from the design review:
- Codex stays the editing and orchestration agent.
- LLMs provide structured research proposals and summaries only.
- Validator, conformal uncertainty, feature engineering, reporting, and final artifact governance remain fixed.
- `final_metrics.json` remains the final source of truth.

## Design Summary

### 1. Provider Abstraction

Add [`/E:/Random IDEA/AutoResearch/auto-civil-lab/llm_backend.py`](/E:/Random IDEA/AutoResearch/auto-civil-lab/llm_backend.py) with:
- `LLMBackend`: interface for availability checks and structured generation.
- `OllamaBackend`: uses the local HTTP endpoint first and optionally falls back to the `ollama` CLI.
- `OpenAIBackend`: uses configured OpenAI credentials and model name.
- `HybridBackend`: tries a primary backend and falls back to the secondary backend.
- `resolve_backend(config)`: builds the active backend from configuration.

The backend layer only knows how to generate text or JSON responses. It does not know AutoML semantics.

### 2. Proposal Engine

Add [`/E:/Random IDEA/AutoResearch/auto-civil-lab/proposal_engine.py`](/E:/Random IDEA/AutoResearch/auto-civil-lab/proposal_engine.py) with a repository-aware interface:
- `generate_hypotheses(...)`
- `generate_experiment_proposals(...)`
- `generate_feature_ideas(...)`
- `generate_search_space_suggestions(...)`
- `summarize_run(...)`

This layer owns:
- prompt building from `research_brief.md`, metrics, memory, and diversity state
- JSON schema validation and parsing
- proposal dedupe and normalization before the research loop consumes results

### 3. Governed Loop Integration

Modify [`/E:/Random IDEA/AutoResearch/auto-civil-lab/research_loop.py`](/E:/Random IDEA/AutoResearch/auto-civil-lab/research_loop.py) so scout proposals come from the proposal engine when LLM proposals are enabled.

The governed loop remains responsible for:
- running scout experiments
- selecting confirm experiments
- keep/revert decisions
- experiment memory
- proposal diversity tracking
- final artifact validation against `final_metrics.json`

If no backend is available, the loop should log the condition and either:
- fall back to deterministic `research_lab.py` proposals, or
- disable LLM proposals while keeping the research loop runnable

The fallback behavior is configuration-driven.

### 4. Legacy Search Reuse

Modify [`/E:/Random IDEA/AutoResearch/auto-civil-lab/search.py`](/E:/Random IDEA/AutoResearch/auto-civil-lab/search.py) to use the new backend abstraction through the proposal engine instead of talking directly to [`/E:/Random IDEA/AutoResearch/auto-civil-lab/llm_proposer.py`](/E:/Random IDEA/AutoResearch/auto-civil-lab/llm_proposer.py).

This avoids two separate LLM stacks in the same repository.

### 5. Config Model

Extend [`/E:/Random IDEA/AutoResearch/auto-civil-lab/config.yaml`](/E:/Random IDEA/AutoResearch/auto-civil-lab/config.yaml) with a neutral `llm` section.

Proposed shape:

```yaml
llm:
  enabled: true
  backend_mode: hybrid
  allow_deterministic_fallback: true
  ollama:
    base_url: http://localhost:11434
    model: qwen3:8b
    fast_model: qwen3:4b
    timeout_seconds: 30
    use_cli_fallback: true
  openai:
    model: gpt-5.1-mini
    api_key_env: OPENAI_API_KEY
    timeout_seconds: 30
  hybrid:
    primary: ollama
    fallback: openai
  tasks:
    proposal_count: 8
    summary_enabled: true
```

The legacy `search.llm_proposals` block can either be mapped into this new schema or preserved temporarily and read through a compatibility shim.

### 6. Safety and Boundaries

The following remain fixed and must not be editable by the backend abstraction:
- [`/E:/Random IDEA/AutoResearch/auto-civil-lab/validator.py`](/E:/Random IDEA/AutoResearch/auto-civil-lab/validator.py)
- [`/E:/Random IDEA/AutoResearch/auto-civil-lab/uncertainty.py`](/E:/Random IDEA/AutoResearch/auto-civil-lab/uncertainty.py)
- feature engineering path in [`/E:/Random IDEA/AutoResearch/auto-civil-lab/feature_engineering.py`](/E:/Random IDEA/AutoResearch/auto-civil-lab/feature_engineering.py)
- final artifact writer/validator logic

The LLM path can propose experiments, but repository code must enforce:
- allowed model families
- allowed parameter ranges
- duplicate suppression
- time budgets
- acceptance based on measured improvement

### 7. Artifact Governance

`final_metrics.json` remains canonical.

The proposal backend may generate a post-run summary, but that summary is informational only. Acceptance and consistency decisions still come from repository code comparing generated artifacts against `final_metrics.json`.

## Error Handling

- Ollama unavailable:
  - `ollama` mode: mark backend unavailable and continue without LLM proposals if deterministic fallback is allowed.
  - `hybrid` mode: attempt OpenAI fallback.
- OpenAI unavailable:
  - `openai` mode: mark backend unavailable and stop proposal generation cleanly.
  - `hybrid` mode: use Ollama if it is available.
- Invalid JSON from any backend:
  - reject the response
  - log it to the interaction artifact
  - continue without crashing the loop

## Testing Strategy

Tests should cover:
- provider resolution
- Ollama HTTP and CLI availability checks
- hybrid fallback ordering
- structured parsing and invalid-response handling
- integration with `research_loop.py` without breaking final artifact governance

## Implementation Notes

- Prefer reusing logic from `llm_proposer.py` by extracting prompt-agnostic transport behavior into `llm_backend.py`.
- Keep proposal schemas simple and JSON-only.
- Do not make the backend responsible for code diffs or model training.
- Preserve deterministic fallback behavior so the loop remains usable when no LLM backend is configured.
