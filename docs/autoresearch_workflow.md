# Governed Research Loop

## Architecture
- `research_lab.py` is the controlled research-editable surface.
- `research_brief.md` is the human strategy input.
- `llm_backend.py` selects the proposal provider: Ollama, OpenAI, or hybrid.
- `proposal_engine.py` builds structured hypotheses, proposals, and summaries from repo state.
- `research_loop.py` is infrastructure: scout, confirm, keep/revert, memory, diversity, and artifact governance.
- `train.py`, `feature_engineering.py`, `validator.py`, `uncertainty.py`, `report.py`, and `design_tool.py` remain infrastructure and concrete-domain logic.

## Execution
1. Run `python train.py` to refresh the baseline artifacts.
2. Edit `research_brief.md` to set strategy constraints and acceptance thresholds.
3. Configure `llm.*` in `config.yaml` if you want Ollama/OpenAI-backed proposal generation.
4. Edit `research_lab.py` only if you want to change the deterministic fallback research surface itself.
5. Run `python research_loop.py --with-report`.

## Loop behavior
- Scout experiments come from `proposal_engine.py` when an LLM backend is available.
- If the configured LLM backend is unavailable and deterministic fallback is allowed, scout experiments come from `research_lab.py`.
- Proposal prompts are routed by config: models in `llm.compact_prompt_models` get the compact prompt, and richer models get the rich prompt.
- Only diverse, non-duplicate candidates are evaluated.
- Parsed LLM proposals pass through a reusable gate before execution: exact duplicates, near-duplicates, saturated-family repeats, malformed payloads, and weak-family retries without novelty are rejected before evaluation.
- Confirm experiments are created only from positive scout results.
- Confirmed improvements are kept and ratcheted into `research_lab.py`.
- Non-improvements are reverted by not updating the research surface.
- Family-state summaries are shared by prompts and gates: strongest active family, saturated families, underexplored promising families, underexplored weak families, and temporarily blocked families.

## Governance artifacts
- `outputs/research_results.csv`
- `outputs/optuna_results.csv` for backward-compatible reporting
- `outputs/research_log.txt`
- `outputs/experiment_memory.json`
- `outputs/proposal_diversity.json`
- `outputs/llm_interactions.jsonl`
- `outputs/llm_run_summary.json`
- `outputs/final_artifact_validation.json`
- `outputs/final_acceptance.json`

`outputs/llm_interactions.jsonl` now records the raw response channel, raw thinking channel, final extracted text, `extracted_from_channel`, `repair_used`, duplicate rejection metadata, regeneration attempts, and deterministic fallback usage.

## Source of truth
- `outputs/final_metrics.json` is canonical.
- `best_search_result.json` and `best_search_model.pkl` are synchronized from that canonical result.
