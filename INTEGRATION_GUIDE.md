# AutoCivil Track B — Integration Guide
## Exact patches for research_loop.py · proposal_engine.py · research_protocol.py · generate_data.py

---

## 0 · File Layout After Integration

```
autocivil/
├── failure_analyzer.py          ← NEW (Track B)
├── knowledge_base.py            ← NEW (Track B)
├── novelty_scorer.py            ← NEW (Track B)
├── hypothesis_archive.py        ← NEW (Track B)
├── feature_lab.py               ← NEW (Track B)
├── validator_calibrator.py      ← NEW (Track B)
├── proposal_engine_patch.py     ← NEW (Track B)
├── research_loop_patch.py       ← NEW (Track B)
├── test_track_b.py              ← NEW (Track B tests)
├── config.yaml                  ← MODIFIED (add research: block)
├── research_loop.py             ← MODIFIED (6 injection points)
├── proposal_engine.py           ← MODIFIED (2 changes)
├── research_protocol.py         ← MODIFIED (1 change)
└── generate_data.py             ← MODIFIED (1 change)
```

Copy all `*.py` files from the `track_b/` folder into the root of the repo
(same directory as `research_loop.py`), then apply the patches below.

---

## 1 · config.yaml

Open `config.yaml` and append the following block at the end
(or merge into an existing `research:` section if one exists):

```yaml
research:
  feature_lab_enabled: false
  feature_lab_interval_cycles: 3
  feature_lab_min_delta_rmse: 0.02
  feature_lab_sandbox_timeout: 30
  novelty_gate_threshold: 0.20
  knowledge_base_enabled: true
  failure_analysis_enabled: true
  hypothesis_archive_filename: hypothesis_archive.json
  llm_chain_of_thought: true
  llm_require_expected_delta: true
  report_review_enabled: true
  report_review_min_score: 6
```

---

## 2 · research_loop.py — 6 injection points

### 2.1  Import block (top of file, after existing imports)

```python
# ── Track B imports ───────────────────────────────────────────────────────────
from research_loop_patch import AutoResearchOrchestrator
# ─────────────────────────────────────────────────────────────────────────────
```

### 2.2  Orchestrator initialisation (once per run, after config and backend load)

Find where `llm_backend` is initialised (usually near the start of
`run_research()` or equivalent), then add directly after it:

```python
    # ── TRACK B: initialise AutoResearch orchestrator ─────────────────────────
    _track_b = AutoResearchOrchestrator(
        config=config,
        outputs_dir=outputs_dir,
        llm_backend=llm_backend,   # pass your existing backend object
    )
    # ─────────────────────────────────────────────────────────────────────────
```

### 2.3  Start of each research cycle (INJECT 1 + 2)

Find the `for cycle in range(...)` loop (or equivalent cycle start).
At the **top of the loop body**, add:

```python
        # ── TRACK B INJECT 1+2: failure context + domain knowledge ────────────
        _ctx = _track_b.prepare_cycle_context(
            run_id=run_id,
            cycle=cycle,
            most_recent_family=current_best.get("model_name", "LGBMRegressor"),
            feature_list=list(x_train.columns),
        )
        failure_context   = _ctx["failure_context"]
        knowledge_context = _ctx["knowledge_context"]
        # ─────────────────────────────────────────────────────────────────────
```

Then pass both into your proposal generation call.  If the call looks like:

```python
        proposals = generate_experiment_proposals(backend, context)
```

change `context` to also carry the two new keys, or add keyword arguments
depending on how `generate_experiment_proposals` is defined (see §3 below).

### 2.4  Novelty gate (INJECT 3) — immediately after proposals are returned

```python
        # ── TRACK B INJECT 3: novelty gate ────────────────────────────────────
        proposals = _track_b.filter_proposals_by_novelty(proposals, cycle=cycle)
        # ─────────────────────────────────────────────────────────────────────
```

### 2.5  Before each trial executes (INJECT 4a)

Find where you call `execute_trial(proposal, ...)` or equivalent.
**Immediately before** that call, add:

```python
            # ── TRACK B INJECT 4a: record hypothesis before trial ─────────────
            _hyp_id = _track_b.record_hypothesis_before_trial(
                proposal=proposal,
                trial_number=trial_number,
            )
            # ─────────────────────────────────────────────────────────────────
```

### 2.6  After each trial completes (INJECT 4b)

**Immediately after** you receive the trial result and compute the delta:

```python
            # ── TRACK B INJECT 4b: resolve hypothesis ─────────────────────────
            # actual_delta = new_rmse - reference_rmse
            #                (negative = improvement, same sign convention as
            #                expected_delta in the proposal JSON)
            _actual_delta = (trial_result.get("rmse", 0)
                             - current_best.get("rmse", 0))
            _outcome = trial_result.get("selection_status", "no_improvement")
            _track_b.resolve_hypothesis_after_trial(
                trial_number=trial_number,
                actual_delta_rmse=_actual_delta,
                outcome=_outcome,
            )
            # ─────────────────────────────────────────────────────────────────
```

### 2.7  End of each cycle (INJECT 5 + 6)

At the **bottom of the cycle loop**, add:

```python
        # ── TRACK B INJECT 5: FeatureLab (every N cycles) ────────────────────
        _track_b.maybe_run_feature_lab(
            cycle=cycle,
            x_train=x_train,
            y_train=y_train,
            x_val=x_val,
            y_val=y_val,
            baseline_cv_rmse=current_best.get("rmse", float("inf")),
            final_metrics=final_metrics,
        )
        # ── TRACK B INJECT 6: scientific report review ────────────────────────
        if hasattr(report_writer, "last_report_text"):
            _track_b.review_scientific_report(report_writer.last_report_text)
        # ─────────────────────────────────────────────────────────────────────
```

### 2.8  End of run (after the cycle loop exits)

```python
    # ── TRACK B: print evaluation summary ─────────────────────────────────────
    _track_b.print_track_b_summary()
    # ─────────────────────────────────────────────────────────────────────────
```

---

## 3 · proposal_engine.py — 2 changes

### 3.1  Import the patch at the top of the file

```python
# ── Track B imports ───────────────────────────────────────────────────────────
from proposal_engine_patch import (
    build_chain_of_thought_prompt,
    parse_enhanced_proposal,
)
# ─────────────────────────────────────────────────────────────────────────────
```

### 3.2  Replace `_build_experiment_prompt()` body

Find the existing `_build_experiment_prompt()` function (or wherever the
LLM prompt string is constructed).  Replace the **return line** (or the
entire body if the function is short) with:

```python
def _build_experiment_prompt(
    context: dict,
    failure_context: dict | None = None,
    knowledge_context: str | None = None,
) -> str:
    """Build the full 4-step chain-of-thought prompt for the LLM."""
    return build_chain_of_thought_prompt(
        context=context,
        failure_context=failure_context,
        knowledge_context=knowledge_context,
    )
```

### 3.3  Update `generate_experiment_proposals()` signature and parse step

Add the two new parameters and swap the JSON parser:

```python
def generate_experiment_proposals(
    backend,
    context: dict,
    failure_context: dict | None = None,   # ← NEW
    knowledge_context: str | None = None,  # ← NEW
    **kwargs,
) -> list[dict]:
    prompt = _build_experiment_prompt(
        context,
        failure_context=failure_context,
        knowledge_context=knowledge_context,
    )
    raw = backend.generate_text(prompt, max_output_tokens=800)

    # ── TRACK B: use enhanced parser (validates expected_delta, etc.) ─────────
    try:
        proposals = parse_enhanced_proposal(raw)
    except ValueError:
        # Fall back to legacy parser if new parser fails
        proposals = _legacy_parse_proposals(raw)  # your existing parser
    # ─────────────────────────────────────────────────────────────────────────
    return proposals
```

---

## 4 · research_protocol.py — 1 change

### 4.1  Add `expected_delta` and `novelty_score` to RESEARCH_RESULTS_COLUMNS

Find the list (or tuple) that defines the columns for research_results.csv.
Add the two new fields:

```python
RESEARCH_RESULTS_COLUMNS = [
    # ... existing columns ...
    "expected_delta",    # ← NEW  from LLM proposal
    "actual_delta",      # ← NEW  measured after trial
    "novelty_score",     # ← NEW  from NoveltyScorer
]
```

### 4.2  Pass `expected_delta` and `novelty_score` when recording experiments

In `record_experiment_memory()` (or equivalent), add:

```python
    row["expected_delta"] = proposal.get("expected_delta", None)
    row["novelty_score"]  = proposal.get("_novelty_score", None)
    # _novelty_score is set by filter_proposals_by_novelty() if you add it to
    # the proposal dict, or pass it separately as a kwarg.
```

---

## 5 · generate_data.py — 1 change

### 5.1  Add validator calibration after dataset load

Find the line where the raw dataset is loaded (usually something like
`dataset = pd.read_csv(...)` or `dataset = load_dataset(config)`).
**Immediately after** that line, add:

```python
    # ── TRACK B: calibrate validator thresholds from dataset statistics ────────
    try:
        from validator_calibrator import ValidatorCalibrator
        _cal = ValidatorCalibrator()
        _calibrated = _cal.calibrate_from_dataset(dataset, config)
        log_status(f"Calibrated validator thresholds: {_calibrated}")
        for _k, _v in _calibrated.items():
            config.setdefault("validator", {})[_k] = _v
    except Exception as _exc:
        log_status(f"Validator calibration skipped: {_exc}")
    # ─────────────────────────────────────────────────────────────────────────
```

Note: calibration updates config **in-memory only**.  To persist the values
permanently, copy the printed thresholds into the `validator:` section of
`config.yaml`.

---

## 6 · Verification Checklist

After integration, run the following to confirm everything is wired up:

```bash
# 1. Run Track B tests
python test_track_b.py

# 2. Run a short research loop (5 cycles) and check outputs
python research_loop.py --cycles 5

# 3. Check that these files were created in outputs/
ls outputs/
#   hypothesis_archive.json     ← must exist and have ≥ 1 entry
#   proposal_diversity.json     ← must exist
#   feature_lab/                ← directory (if feature_lab_enabled: true)
#   scientific_report_review.json ← if report_review_enabled: true

# 4. Verify expected_delta appears in research_results.csv
head -2 outputs/research_results.csv | tr ',' '\n' | grep -n expected_delta

# 5. Print the Track B summary
python -c "
from research_loop_patch import AutoResearchOrchestrator
from pathlib import Path
import json
config = json.load(open('config.yaml'))  # or however your config loads
orch = AutoResearchOrchestrator(config, Path('outputs'))
orch.print_track_b_summary()
"
```

---

## 7 · Track B Evaluation Criteria Status

| # | Criterion | How to verify |
|---|-----------|---------------|
| 1 | LLM proposal rationale (`expected_delta` in every proposal) | `grep expected_delta outputs/hypothesis_archive.json` |
| 2 | Failure patterns injected in every cycle prompt | Check `failure_analysis_enabled: true` in logs |
| 3 | ≥ 1 feature tested per run | `cat outputs/feature_lab/feature_hypotheses.json \| python -m json.tool` |
| 4 | ≥ 1 feature promoted per 5 runs | count entries with `"accepted": true` in feature_hypotheses.json |
| 5 | Mean novelty score ≥ 0.40 | `cat outputs/proposal_diversity.json` |
| 6 | LLM calibration error ≤ 0.05 | `cat outputs/hypothesis_archive.json \| python -c "import json,sys; d=json.load(sys.stdin); print(d['calibration_summary'])"` |
| 7 | Scientific report review score ≥ 7.0 | `cat outputs/scientific_report_review.json` |
| 8 | Knowledge base ≥ 10 rules | `python -c "from knowledge_base import get_all_rule_count; print(get_all_rule_count())"` |
| 9 | Hypothesis archive ≥ 20 entries after full run | `cat outputs/hypothesis_archive.json \| python -m json.tool \| grep hypothesis_id \| wc -l` |
| 10 | Acceptance rate 12–20% per run | Check `calibration_summary.acceptance_rate` |

---

## 8 · Quick Smoke Test (no LLM needed)

```python
# smoke_test.py — run before committing Track B
import json, tempfile
from pathlib import Path

# 1. FailureAnalyzer
from failure_analyzer import FailureAnalyzer
fa = FailureAnalyzer(Path("outputs/experiment_memory.json"))
p = fa.extract_failure_patterns()
assert "_meta" in p, "FailureAnalyzer broken"
print("✓ FailureAnalyzer")

# 2. KnowledgeBase
from knowledge_base import get_knowledge_context, get_all_rule_count
assert get_all_rule_count() >= 10
ctx = get_knowledge_context("LGBMRegressor")
assert "ACI" in ctx or "empirical" in ctx.lower()
print("✓ KnowledgeBase")

# 3. NoveltyScorer
from novelty_scorer import NoveltyScorer
scorer = NoveltyScorer()
score = scorer.compute_novelty_score({"model_name": "X", "params": {}}, [])
assert score == 1.0
print("✓ NoveltyScorer")

# 4. HypothesisArchive
from hypothesis_archive import HypothesisArchive
with tempfile.TemporaryDirectory() as td:
    arch = HypothesisArchive(Path(td) / "h.json")
    hid = arch.add(proposal={"model_name": "X", "params": {}},
                   claim="c", mechanism="m", expected_delta_rmse=-0.05)
    arch.resolve(hid, actual_delta_rmse=-0.07, outcome="accepted")
    assert arch.is_well_calibrated()
print("✓ HypothesisArchive")

# 5. FeatureLab sandbox
from feature_lab import _sanitise_code, _build_sandbox_namespace, _exec_with_timeout
ns = _build_sandbox_namespace()
ok, _ = _exec_with_timeout(
    "def new_feature(df): return df['cement'] * 2", ns, timeout=5
)
assert ok and callable(ns.get("new_feature"))
print("✓ FeatureLab sandbox")

print("\nAll smoke tests passed.")
```
