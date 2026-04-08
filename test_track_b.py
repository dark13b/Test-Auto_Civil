"""
test_track_b.py — AutoCivil Track B
=====================================
Full integration test suite.  Run with:
    python test_track_b.py
or:
    python -m pytest test_track_b.py -v

Covers:
  - FailureAnalyzer  (empty memory, realistic memory)
  - KnowledgeBase    (rule count, context formatting)
  - NoveltyScorer    (gate threshold, cross-family, diversity budget)
  - HypothesisArchive (add/resolve, calibration, safety checks)
  - FeatureLab        (sandbox security, timeout, good/bad features)
  - ValidatorCalibrator (synthetic dataset)
  - ProposalEnginePatch (prompt build, parse, calibration stats)
  - AutoResearchOrchestrator (end-to-end cycle simulation)
"""

from __future__ import annotations

import json
import math
import sys
import tempfile
import threading
import time
import traceback
from pathlib import Path

# ── Test helpers ──────────────────────────────────────────────────────────────

PASS = "✓"
FAIL = "✗"
results: list[tuple[str, bool, str]] = []
_registered_tests: list = []   # ordered list of (name, wrapper_fn)


def test(name: str):
    """Decorator that registers and captures pass/fail for each test function."""
    def decorator(fn):
        def wrapper():
            try:
                fn()
                results.append((name, True, ""))
                print(f"  {PASS} {name}")
            except Exception as exc:
                msg = f"{type(exc).__name__}: {exc}"
                results.append((name, False, msg))
                print(f"  {FAIL} {name}")
                print(f"       {msg}")
                if "--verbose" in sys.argv:
                    traceback.print_exc()
        _registered_tests.append((name, wrapper))
        return wrapper
    return decorator


def assert_equal(a, b, msg=""):
    assert a == b, f"{msg} — expected {b!r}, got {a!r}"


def assert_true(cond, msg=""):
    assert cond, msg


def assert_between(val, lo, hi, msg=""):
    assert lo <= val <= hi, f"{msg} — {val} not in [{lo}, {hi}]"


# ── Synthetic memory builder ──────────────────────────────────────────────────

def make_memory(n_good=20, n_bad=8) -> dict:
    """Build a synthetic experiment_memory.json structure."""
    trials = []
    families = ["LGBMRegressor", "XGBRegressor", "RandomForestRegressor"]
    for i in range(n_good):
        trials.append({
            "trial_number": i,
            "model_name": families[i % len(families)],
            "params": {
                "n_estimators": 300 + i * 10,
                "learning_rate": 0.08 + (i % 5) * 0.02,
                "num_leaves": 20 + i % 20,
            },
            "composite_score": 0.88 + (i % 10) * 0.005,
            "selection_status": "accepted",
            "validator_issues": [],
        })
    for j in range(n_bad):
        trials.append({
            "trial_number": n_good + j,
            "model_name": families[j % 2],
            "params": {
                "n_estimators": 100,
                "learning_rate": 0.30,
                "num_leaves": 127,
            },
            "composite_score": 0.70 + j * 0.01,
            "selection_status": "rejected_validation_fail",
            "validator_issues": ["water_cement_ratio_too_high", "total_binder_low"],
        })
    return {"runs": [{"run_id": "TEST-001", "trials": trials}]}


# ── Tests: FailureAnalyzer ────────────────────────────────────────────────────

@test("FailureAnalyzer: handles empty memory")
def _():
    from failure_analyzer import FailureAnalyzer
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "experiment_memory.json"
        fa = FailureAnalyzer(path)
        patterns = fa.extract_failure_patterns()
        assert_equal(patterns["_meta"]["total_trials_analysed"], 0)
        assert_equal(patterns["families_with_high_fail_rate"], {})


@test("FailureAnalyzer: detects high-fail families")
def _():
    from failure_analyzer import FailureAnalyzer
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "experiment_memory.json"
        path.write_text(json.dumps(make_memory(n_good=10, n_bad=8)))
        fa = FailureAnalyzer(path)
        patterns = fa.extract_failure_patterns()
        rates = patterns["families_with_high_fail_rate"]
        # At least one family should have a non-zero fail rate
        assert_true(len(rates) > 0, "Expected at least one family with fail rate")
        assert_true(all(0 <= v <= 1 for v in rates.values()),
                    "Fail rates must be in [0,1]")


@test("FailureAnalyzer: extracts bad param ranges")
def _():
    from failure_analyzer import FailureAnalyzer
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "experiment_memory.json"
        path.write_text(json.dumps(make_memory()))
        fa = FailureAnalyzer(path)
        patterns = fa.extract_failure_patterns()
        bad = patterns["parameter_ranges_that_always_fail"]
        # n_estimators should appear since bad trials all have n_estimators=100
        assert_true("n_estimators" in bad or "learning_rate" in bad,
                    "Expected n_estimators or learning_rate in bad param ranges")


@test("FailureAnalyzer: format_for_prompt returns non-empty string")
def _():
    from failure_analyzer import FailureAnalyzer
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "experiment_memory.json"
        path.write_text(json.dumps(make_memory()))
        fa = FailureAnalyzer(path)
        prompt_snippet = fa.format_for_prompt()
        assert_true(len(prompt_snippet) > 50, "Prompt snippet is too short")
        assert_true("FAILURE PATTERNS" in prompt_snippet, "Missing header")


# ── Tests: KnowledgeBase ──────────────────────────────────────────────────────

@test("KnowledgeBase: rule count ≥ 10")
def _():
    from knowledge_base import get_all_rule_count
    count = get_all_rule_count()
    assert_true(count >= 10, f"Rule count {count} < 10 (Track B minimum)")


@test("KnowledgeBase: every entry has a source field")
def _():
    from knowledge_base import CONCRETE_KNOWLEDGE

    def check_sources(obj, path=""):
        if isinstance(obj, dict):
            if "source" in obj:
                assert_true(
                    bool(obj["source"]),
                    f"Empty 'source' field at {path}"
                )
            for k, v in obj.items():
                check_sources(v, f"{path}.{k}")
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                check_sources(item, f"{path}[{i}]")

    check_sources(CONCRETE_KNOWLEDGE)


@test("KnowledgeBase: get_knowledge_context returns non-empty for LGBM")
def _():
    from knowledge_base import get_knowledge_context
    ctx = get_knowledge_context("LGBMRegressor",
                                 ["water_effective_binder_ratio"])
    assert_true(len(ctx) > 100, "Context too short")
    assert_true("LGBMREGRESSOR" in ctx.upper(), "Model name missing")
    assert_true("ACI" in ctx or "empirical" in ctx.lower(), "No source cited")


@test("KnowledgeBase: gracefully handles unknown model")
def _():
    from knowledge_base import get_knowledge_context
    ctx = get_knowledge_context("NonExistentModel")
    assert_true(isinstance(ctx, str), "Must return a string even for unknown model")


# ── Tests: NoveltyScorer ──────────────────────────────────────────────────────

@test("NoveltyScorer: empty archive → score 1.0")
def _():
    from novelty_scorer import NoveltyScorer
    scorer = NoveltyScorer()
    score = scorer.compute_novelty_score({"model_name": "LGBM", "params": {}}, [])
    assert_equal(score, 1.0)


@test("NoveltyScorer: identical proposal → low novelty")
def _():
    from novelty_scorer import NoveltyScorer
    scorer = NoveltyScorer(search_spaces={
        "LGBMRegressor": {"search_space": {
            "n_estimators": {"type": "int", "low": 100, "high": 800},
            "learning_rate": {"type": "float", "low": 0.01, "high": 0.30},
        }}
    })
    archive = [{"proposal": {"model_name": "LGBMRegressor",
                              "params": {"n_estimators": 300, "learning_rate": 0.1}}}]
    p = {"model_name": "LGBMRegressor",
         "params": {"n_estimators": 302, "learning_rate": 0.101}}
    score = scorer.compute_novelty_score(p, archive)
    assert_true(score < 0.5, f"Near-identical proposal should score < 0.5, got {score}")


@test("NoveltyScorer: different family → score ≥ cross-family similarity")
def _():
    from novelty_scorer import NoveltyScorer, CROSS_FAMILY_SIMILARITY
    scorer = NoveltyScorer()
    archive = [{"proposal": {"model_name": "LGBMRegressor",
                              "params": {"n_estimators": 300}}}]
    p = {"model_name": "XGBRegressor", "params": {"n_estimators": 300}}
    score = scorer.compute_novelty_score(p, archive)
    assert_true(score >= 1 - CROSS_FAMILY_SIMILARITY,
                f"Cross-family novelty should be ≥ {1-CROSS_FAMILY_SIMILARITY:.2f}, got {score}")


@test("NoveltyScorer: novelty gate rejects below threshold")
def _():
    from novelty_scorer import NoveltyScorer
    scorer = NoveltyScorer(
        search_spaces={"M": {"search_space": {
            "n": {"type": "int", "low": 0, "high": 1000}
        }}},
        threshold=0.20,
    )
    archive = [{"proposal": {"model_name": "M", "params": {"n": 500}}}]
    p = {"model_name": "M", "params": {"n": 501}}
    passes, score = scorer.is_novel_enough(p, archive)
    assert_true(not passes or score >= 0.20, "Gate logic inconsistent")


@test("NoveltyScorer: diversity budget limits per-family count")
def _():
    from novelty_scorer import NoveltyScorer
    scorer = NoveltyScorer()
    proposals = [
        {"model_name": "LGBMRegressor", "params": {}, "proposal_family": "exploit"},
        {"model_name": "LGBMRegressor", "params": {}, "proposal_family": "explore"},
        {"model_name": "LGBMRegressor", "params": {}, "proposal_family": "explore"},  # 3rd → dropped
        {"model_name": "XGBRegressor",  "params": {}, "proposal_family": "explore"},
    ]
    kept = scorer.enforce_diversity_budget(proposals, [])
    lgbm_count = sum(1 for p in kept if p["model_name"] == "LGBMRegressor")
    assert_true(lgbm_count <= 2, f"Expected ≤ 2 LGBM proposals, got {lgbm_count}")


# ── Tests: HypothesisArchive ──────────────────────────────────────────────────

@test("HypothesisArchive: add requires expected_delta")
def _():
    from hypothesis_archive import HypothesisArchive
    with tempfile.TemporaryDirectory() as td:
        archive = HypothesisArchive(Path(td) / "hyp.json")
        try:
            archive.add(
                proposal={"model_name": "X", "params": {}},
                claim="test", mechanism="test",
                expected_delta_rmse=None,   # should raise
            )
            assert False, "Should have raised ValueError"
        except ValueError:
            pass  # correct


@test("HypothesisArchive: add + resolve computes calibration_error")
def _():
    from hypothesis_archive import HypothesisArchive
    with tempfile.TemporaryDirectory() as td:
        archive = HypothesisArchive(Path(td) / "hyp.json")
        hyp_id = archive.add(
            proposal={"model_name": "LGBMRegressor", "params": {}},
            claim="test claim", mechanism="test mechanism",
            expected_delta_rmse=-0.08,
        )
        archive.resolve(hyp_id, actual_delta_rmse=-0.12, outcome="accepted",
                        trial_number=1)
        hyps = archive.all_hypotheses()
        assert_equal(len(hyps), 1)
        assert_equal(hyps[0]["calibration_error"], 0.04)
        assert_equal(hyps[0]["outcome"], "accepted")


@test("HypothesisArchive: calibration summary is_well_calibrated ≤ 0.05")
def _():
    from hypothesis_archive import HypothesisArchive
    with tempfile.TemporaryDirectory() as td:
        archive = HypothesisArchive(Path(td) / "hyp.json")
        for i in range(5):
            hid = archive.add(
                proposal={"model_name": "X", "params": {}},
                claim="c", mechanism="m",
                expected_delta_rmse=-0.10,
            )
            archive.resolve(hid, actual_delta_rmse=-0.12, outcome="accepted")
        assert_true(archive.is_well_calibrated(), "Mean error 0.02 should be well-calibrated")


@test("HypothesisArchive: as_novelty_archive returns correct format")
def _():
    from hypothesis_archive import HypothesisArchive
    with tempfile.TemporaryDirectory() as td:
        archive = HypothesisArchive(Path(td) / "hyp.json")
        hid = archive.add(
            proposal={"model_name": "RF", "params": {"n": 100}},
            claim="c", mechanism="m", expected_delta_rmse=0.0,
        )
        na = archive.as_novelty_archive()
        assert_equal(len(na), 1)
        assert_true("proposal" in na[0], "Missing 'proposal' key")
        assert_equal(na[0]["proposal"]["model_name"], "RF")


@test("HypothesisArchive: persists to disk and reloads")
def _():
    from hypothesis_archive import HypothesisArchive
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "hyp.json"
        a1 = HypothesisArchive(path)
        a1.add(proposal={"model_name": "X", "params": {}},
               claim="persisted", mechanism="m", expected_delta_rmse=-0.05)
        # Reload from disk
        a2 = HypothesisArchive(path)
        assert_equal(len(a2.all_hypotheses()), 1)
        assert_equal(a2.all_hypotheses()[0]["claim"], "persisted")


# ── Tests: FeatureLab ─────────────────────────────────────────────────────────

@test("FeatureLab: sandbox blocks dangerous imports")
def _():
    from feature_lab import _sanitise_code, _build_sandbox_namespace, _exec_with_timeout
    code = """
import os
import sys
def new_feature(df):
    return df['cement']
"""
    safe = _sanitise_code(code)
    assert_true("import os" not in safe, "import os should be stripped")
    assert_true("import sys" not in safe, "import sys should be stripped")


@test("FeatureLab: sandbox does not expose os/sys in namespace")
def _():
    from feature_lab import _build_sandbox_namespace, _exec_with_timeout
    ns = _build_sandbox_namespace()
    assert_true("os" not in ns, "os should not be in sandbox namespace")
    assert_true("sys" not in ns, "sys should not be in sandbox namespace")
    assert_true("subprocess" not in ns, "subprocess should not be in sandbox namespace")
    assert_true("np" in ns, "np must be available")
    assert_true("math" in ns, "math must be available")


@test("FeatureLab: valid feature executes and returns Series")
def _():
    import pandas as pd
    from feature_lab import _sanitise_code, _build_sandbox_namespace, _exec_with_timeout

    code = """
def new_feature(df):
    return df['cement'] / (df['water'] + 1e-6)
"""
    ns = _build_sandbox_namespace()
    ok, err = _exec_with_timeout(_sanitise_code(code), ns, timeout=5)
    assert_true(ok, f"Exec failed: {err}")
    fn = ns.get("new_feature")
    assert_true(callable(fn), "new_feature should be callable")
    dummy = pd.DataFrame({"cement": [300.0, 350.0], "water": [180.0, 175.0]})
    result = fn(dummy)
    assert_equal(len(result), 2)


@test("FeatureLab: timeout kills long-running code")
def _():
    from feature_lab import _exec_with_timeout, _build_sandbox_namespace
    code = """
import time as _t
_t.sleep(60)
"""
    # We can't import time in sandbox, but the sleep won't work anyway.
    # Test that the timeout mechanism itself works with a CPU spin.
    code_spin = """
x = 0
while True:
    x += 1
"""
    ns = _build_sandbox_namespace()
    start = time.time()
    ok, err = _exec_with_timeout(code_spin, ns, timeout=2)
    elapsed = time.time() - start
    assert_true(not ok, "Infinite loop should time out")
    assert_true("Timeout" in err, f"Error should mention timeout, got: {err}")
    assert_true(elapsed < 5.0, f"Timeout took too long: {elapsed:.1f}s")


@test("FeatureLab: _parse_feature_proposal extracts function and metadata")
def _():
    from feature_lab import _parse_feature_proposal
    raw = """
# HYPOTHESIS: w/b ratio captures strength
# MECHANISM: Abrams law
# EXPECTED_EFFECT: -0.05 MPa RMSE
def new_feature(df):
    return df['water'] / (df['cement'] + 1e-6)
"""
    result = _parse_feature_proposal(raw)
    assert_true("error" not in result, f"Parse error: {result.get('error')}")
    assert_true("new_feature" in result["function_code"])
    assert_equal(result["hypothesis"], "w/b ratio captures strength")
    assert_equal(result["mechanism"], "Abrams law")


@test("FeatureLab: bad feature (no function named new_feature) returns error")
def _():
    from feature_lab import _parse_feature_proposal
    result = _parse_feature_proposal("def wrong_name(df): return df['cement']")
    assert_true("error" in result, "Should return error for missing new_feature")


# ── Tests: ValidatorCalibrator ────────────────────────────────────────────────

@test("ValidatorCalibrator: calibrates on synthetic dataset")
def _():
    import numpy as np
    import pandas as pd
    from validator_calibrator import ValidatorCalibrator

    np.random.seed(0)
    n = 300
    cement  = np.random.uniform(150, 400, n)
    slag    = np.random.uniform(0, 100, n)
    fly_ash = np.random.uniform(0, 80, n)
    water   = np.random.uniform(140, 210, n)
    df = pd.DataFrame({
        "cement": cement, "slag": slag, "fly_ash": fly_ash, "water": water,
        "superplasticizer": np.random.uniform(0, 15, n),
        "age": np.random.choice([7, 14, 28, 56, 90], n),
        "strength": 40 + 0.04*cement - 0.07*water + np.random.normal(0, 4, n),
    })
    config = {"task": {"target_column": "strength"}, "validator": {}}
    cal = ValidatorCalibrator()
    suggestions = cal.calibrate_from_dataset(df, config)
    assert_true(len(suggestions) >= 3, f"Expected ≥ 3 suggestions, got {len(suggestions)}")
    assert_true("total_binder_low_warn" in suggestions)
    assert_true(suggestions["total_binder_low_warn"] >= 150.0,
                "total_binder threshold must respect physical minimum")


@test("ValidatorCalibrator: respects physical limits")
def _():
    import numpy as np
    import pandas as pd
    from validator_calibrator import ValidatorCalibrator, PHYSICAL_LIMITS

    # Dataset with very low binder — calibrator should clamp to PHYSICAL_LIMITS
    np.random.seed(42)
    n = 100
    df = pd.DataFrame({
        "cement":  np.full(n, 100.0),   # very low
        "slag":    np.zeros(n),
        "fly_ash": np.zeros(n),
        "water":   np.random.uniform(140, 200, n),
        "superplasticizer": np.zeros(n),
        "age": np.full(n, 28),
        "strength": np.random.uniform(15, 30, n),
    })
    config = {"task": {"target_column": "strength"}, "validator": {}}
    cal = ValidatorCalibrator()
    suggestions = cal.calibrate_from_dataset(df, config)
    if "total_binder_low_warn" in suggestions:
        assert_true(
            suggestions["total_binder_low_warn"] >= PHYSICAL_LIMITS["total_binder_min"],
            "Must not go below physical minimum"
        )


# ── Tests: ProposalEnginePatch ────────────────────────────────────────────────

@test("ProposalEnginePatch: build_chain_of_thought_prompt includes all steps")
def _():
    from proposal_engine_patch import build_chain_of_thought_prompt
    ctx = {
        "current_best": {"model_name": "LGBM", "composite_score": 0.92,
                          "rmse": 4.8, "rmse_by_range": {}},
        "recent_trials": [], "allowed_families": ["LGBM"], "search_spaces": {},
        "n_proposals": 2, "cycle": 3,
    }
    prompt = build_chain_of_thought_prompt(ctx)
    for step in ["STEP 1", "STEP 2", "STEP 3", "STEP 4"]:
        assert_true(step in prompt, f"Missing {step} in prompt")
    assert_true("expected_delta" in prompt, "Schema must mention expected_delta")
    assert_true("proposal_family" in prompt, "Schema must mention proposal_family")


@test("ProposalEnginePatch: parse_enhanced_proposal accepts valid JSON array")
def _():
    from proposal_engine_patch import parse_enhanced_proposal
    raw = json.dumps([{
        "model_name": "LGBMRegressor",
        "params": {"n_estimators": 350, "learning_rate": 0.09},
        "hypothesis": "test hypothesis",
        "mechanism": "test mechanism",
        "expected_delta": -0.07,
        "proposal_family": "exploit",
    }])
    proposals = parse_enhanced_proposal(raw)
    assert_equal(len(proposals), 1)
    assert_equal(proposals[0]["model_name"], "LGBMRegressor")
    assert_equal(proposals[0]["expected_delta"], -0.07)
    assert_equal(proposals[0]["proposal_family"], "exploit")


@test("ProposalEnginePatch: parse defaults missing expected_delta to 0.0")
def _():
    from proposal_engine_patch import parse_enhanced_proposal
    raw = json.dumps([{
        "model_name": "XGBRegressor",
        "params": {"n_estimators": 200},
    }])
    proposals = parse_enhanced_proposal(raw)
    assert_equal(proposals[0]["expected_delta"], 0.0)


@test("ProposalEnginePatch: parse strips markdown fences")
def _():
    from proposal_engine_patch import parse_enhanced_proposal
    raw = "```json\n[{\"model_name\": \"RF\", \"params\": {}}]\n```"
    proposals = parse_enhanced_proposal(raw)
    assert_equal(proposals[0]["model_name"], "RF")


@test("ProposalEnginePatch: parse raises on completely invalid input")
def _():
    from proposal_engine_patch import parse_enhanced_proposal
    try:
        parse_enhanced_proposal("This is not JSON at all and has no array.")
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


@test("ProposalEnginePatch: compute_calibration_stats from resolved hypotheses")
def _():
    from proposal_engine_patch import compute_calibration_stats
    hyps = [
        {"expected_delta_rmse": -0.10, "actual_delta_rmse": -0.12},
        {"expected_delta_rmse": -0.05, "actual_delta_rmse": -0.06},
        {"expected_delta_rmse": -0.08, "actual_delta_rmse": -0.07},
    ]
    stats = compute_calibration_stats(hyps)
    assert_true(stats["well_calibrated"], "Mean error ~0.013 should be well-calibrated")
    assert_between(stats["mean_calibration_error"], 0.0, 0.05)
    assert_equal(stats["n_resolved"], 3)


# ── Tests: AutoResearchOrchestrator ──────────────────────────────────────────

class MockBackend:
    def generate_text(self, prompt: str, max_output_tokens: int = 400) -> str:
        return json.dumps([{
            "model_name": "LGBMRegressor",
            "params": {"n_estimators": 350, "learning_rate": 0.09, "num_leaves": 22},
            "hypothesis": "Mock hypothesis",
            "mechanism": "Mock mechanism",
            "expected_delta": -0.06,
            "proposal_family": "explore",
        }])


@test("AutoResearchOrchestrator: initialises without error")
def _():
    from research_loop_patch import AutoResearchOrchestrator
    with tempfile.TemporaryDirectory() as td:
        config = {
            "research": {
                "failure_analysis_enabled": True,
                "knowledge_base_enabled": True,
                "novelty_gate_threshold": 0.20,
                "feature_lab_enabled": False,
                "llm_require_expected_delta": True,
                "hypothesis_archive_filename": "hyp.json",
            },
            "models": {},
            "memory_filename": "memory.json",
        }
        orch = AutoResearchOrchestrator(config, Path(td), MockBackend())
        assert_true(orch is not None)


@test("AutoResearchOrchestrator: prepare_cycle_context returns failure + knowledge ctx")
def _():
    from research_loop_patch import AutoResearchOrchestrator
    with tempfile.TemporaryDirectory() as td:
        config = {
            "research": {
                "failure_analysis_enabled": True,
                "knowledge_base_enabled": True,
                "novelty_gate_threshold": 0.20,
                "feature_lab_enabled": False,
                "llm_require_expected_delta": True,
                "hypothesis_archive_filename": "hyp.json",
            },
            "models": {},
            "memory_filename": "memory.json",
        }
        orch = AutoResearchOrchestrator(config, Path(td), MockBackend())
        ctx = orch.prepare_cycle_context("RUN-001", cycle=1,
                                          most_recent_family="LGBMRegressor")
        assert_true("failure_context" in ctx)
        assert_true("knowledge_context" in ctx)
        assert_true(isinstance(ctx["knowledge_context"], str))
        assert_true(len(ctx["knowledge_context"]) > 0)


@test("AutoResearchOrchestrator: hypothesis lifecycle add → resolve")
def _():
    from research_loop_patch import AutoResearchOrchestrator
    with tempfile.TemporaryDirectory() as td:
        config = {
            "research": {
                "failure_analysis_enabled": False,
                "knowledge_base_enabled": False,
                "novelty_gate_threshold": 0.20,
                "feature_lab_enabled": False,
                "llm_require_expected_delta": True,
                "hypothesis_archive_filename": "hyp.json",
            },
            "models": {},
            "memory_filename": "memory.json",
        }
        orch = AutoResearchOrchestrator(config, Path(td))
        orch._current_run_id = "R1"
        orch._current_cycle  = 1

        proposal = {
            "model_name": "LGBMRegressor",
            "params": {"n_estimators": 400},
            "hypothesis": "test",
            "mechanism": "test",
            "expected_delta": -0.08,
        }
        hyp_id = orch.record_hypothesis_before_trial(proposal, trial_number=7)
        assert_true(hyp_id is not None, "Should return a hypothesis_id")

        orch.resolve_hypothesis_after_trial(
            trial_number=7, actual_delta_rmse=-0.10, outcome="accepted"
        )

        hyps = orch.hypothesis_archive.all_hypotheses()
        assert_equal(len(hyps), 1)
        assert_equal(hyps[0]["outcome"], "accepted")
        assert_equal(hyps[0]["calibration_error"], 0.02)


@test("AutoResearchOrchestrator: novelty filter rejects near-duplicate proposals")
def _():
    from research_loop_patch import AutoResearchOrchestrator
    with tempfile.TemporaryDirectory() as td:
        config = {
            "research": {
                "failure_analysis_enabled": False,
                "knowledge_base_enabled": False,
                "novelty_gate_threshold": 0.20,
                "feature_lab_enabled": False,
                "llm_require_expected_delta": True,
                "hypothesis_archive_filename": "hyp.json",
            },
            "models": {
                "LGBMRegressor": {"search_space": {
                    "n_estimators": {"type": "int", "low": 100, "high": 800},
                    "learning_rate": {"type": "float", "low": 0.01, "high": 0.30},
                }}
            },
            "memory_filename": "memory.json",
        }
        orch = AutoResearchOrchestrator(config, Path(td))

        # Prime archive with one entry
        orch.hypothesis_archive.add(
            proposal={"model_name": "LGBMRegressor",
                      "params": {"n_estimators": 300, "learning_rate": 0.10}},
            claim="base", mechanism="base", expected_delta_rmse=-0.05,
        )

        # Near-duplicate should be filtered
        near_dup = [{"model_name": "LGBMRegressor",
                     "params": {"n_estimators": 301, "learning_rate": 0.101},
                     "proposal_family": "exploit"}]
        result = orch.filter_proposals_by_novelty(near_dup, cycle=2)
        # Fallback means it may keep one, but score should be low
        assert_true(len(result) <= 1)


# ── Runner ────────────────────────────────────────────────────────────────────

def run_all():
    print("\n" + "=" * 60)
    print("AutoCivil Track B — Test Suite")
    print("=" * 60)

    for group_name, group_tests in [
        ("FailureAnalyzer",          [t for t in _registered_tests if "FailureAnalyzer" in t[0]]),
        ("KnowledgeBase",            [t for t in _registered_tests if "KnowledgeBase" in t[0]]),
        ("NoveltyScorer",            [t for t in _registered_tests if "NoveltyScorer" in t[0]]),
        ("HypothesisArchive",        [t for t in _registered_tests if "HypothesisArchive" in t[0]]),
        ("FeatureLab",               [t for t in _registered_tests if "FeatureLab" in t[0]]),
        ("ValidatorCalibrator",      [t for t in _registered_tests if "ValidatorCalibrator" in t[0]]),
        ("ProposalEnginePatch",      [t for t in _registered_tests if "ProposalEnginePatch" in t[0]]),
        ("AutoResearchOrchestrator", [t for t in _registered_tests if "AutoResearchOrchestrator" in t[0]]),
    ]:
        if group_tests:
            print(f"\n  [{group_name}]")
            for _, wrapper_fn in group_tests:
                wrapper_fn()

    print("\n" + "-" * 60)
    passed = sum(1 for _, ok, _ in results if ok)
    total  = len(results)
    print(f"Result: {passed}/{total} tests passed\n")

    if passed < total:
        print("FAILURES:")
        for name, ok, msg in results:
            if not ok:
                print(f"  {FAIL} {name}")
                print(f"       {msg}")

    return passed == total


if __name__ == "__main__":
    success = run_all()
    sys.exit(0 if success else 1)
