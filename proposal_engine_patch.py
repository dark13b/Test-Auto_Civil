"""
proposal_engine_patch.py — AutoCivil Track B
==============================================
This module PATCHES the existing proposal_engine.py with Track B enhancements.
It does NOT replace proposal_engine.py — it provides drop-in functions you can
call from _build_experiment_prompt() and generate_experiment_proposals().

HOW TO INTEGRATE (proposal_engine.py):
---------------------------------------
1. At top of proposal_engine.py, add:
       from proposal_engine_patch import (
           build_chain_of_thought_prompt,
           parse_enhanced_proposal,
           ENHANCED_PROPOSAL_SCHEMA,
       )

2. In generate_experiment_proposals(), replace the prompt build with:
       prompt = build_chain_of_thought_prompt(
           context=context_dict,
           failure_context=failure_context,
           knowledge_context=knowledge_context,
       )

3. Parse the JSON response with:
       proposal = parse_enhanced_proposal(raw_llm_response)
       # proposal now contains 'expected_delta' (required for hypothesis_archive)

Key additions vs. current engine:
  - 4-step chain-of-thought (diagnose → hypothesize → predict → propose)
  - expected_delta field in the output schema
  - proposal_family tag for diversity budget
  - full integration with failure_context and knowledge_context
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)


# ── Output schema (what the LLM must return) ─────────────────────────────────

ENHANCED_PROPOSAL_SCHEMA = {
    "model_name":       "string — must match an allowed family name",
    "params":           "dict — hyperparameter values within search space",
    "hypothesis":       "string — one-sentence falsifiable claim",
    "mechanism":        "string — concrete science or ML theory explanation",
    "expected_delta":   "float — expected RMSE change (negative = improvement)",
    "proposal_family":  "string — one of: exploit, explore, feature_mod, domain_challenge",
}

# Valid proposal family tags for diversity budget enforcement
PROPOSAL_FAMILIES = frozenset(
    {"exploit", "explore", "feature_mod", "domain_challenge"}
)


# ── Chain-of-thought prompt builder ──────────────────────────────────────────

def build_chain_of_thought_prompt(
    context: dict,
    failure_context: dict | None = None,
    knowledge_context: str | None = None,
) -> str:
    """
    Build the enhanced 4-step chain-of-thought prompt for the LLM.

    Args:
        context:           dict with keys:
                             current_best (dict with model_name, composite_score,
                                           rmse, rmse_by_range)
                             recent_trials (list of trial dicts)
                             allowed_families (list of model name strings)
                             search_spaces (dict model_name → search_space)
                             n_proposals (int — how many to generate)
                             run_id (str)
                             cycle (int)
        failure_context:   output of FailureAnalyzer.extract_failure_patterns()
        knowledge_context: output of get_knowledge_context()

    Returns:
        Prompt string ready to send to the LLM.
    """
    current_best    = context.get("current_best", {})
    recent_trials   = context.get("recent_trials", [])
    allowed_fams    = context.get("allowed_families", [])
    search_spaces   = context.get("search_spaces", {})
    n_proposals     = context.get("n_proposals", 3)
    cycle           = context.get("cycle", 0)

    # Summarise recent trial performance
    trial_summary = _summarise_trials(recent_trials)

    # Format failure patterns
    failure_block = ""
    if failure_context:
        fam_rates = failure_context.get("families_with_high_fail_rate", {})
        bad_params = failure_context.get("parameter_ranges_that_always_fail", {})
        if fam_rates or bad_params:
            failure_block = (
                "\n--- KNOWN FAILURE PATTERNS (AVOID THESE) ---\n"
                + (f"High-failure families: {fam_rates}\n" if fam_rates else "")
                + (f"Bad parameter ranges:  {_fmt_bad_params(bad_params)}\n"
                   if bad_params else "")
            )

    # Search space summary (top-level param names per family)
    ss_summary = {
        fam: list(ss.get("search_space", {}).keys())
        for fam, ss in search_spaces.items()
        if fam in allowed_fams
    }

    prompt_lines = [
        "You are a research scientist specialising in concrete compressive strength ML models.",
        f"Research cycle: {cycle}",
        "",
        "=== CURRENT STATE ===",
        f"Best model:       {current_best.get('model_name', 'none')}",
        f"Composite score:  {current_best.get('composite_score', 0.0):.4f}",
        f"Validation RMSE:  {current_best.get('rmse', 'unknown')} MPa",
        f"RMSE by range:    {current_best.get('rmse_by_range', {})}",
        "",
        "=== RECENT TRIAL SUMMARY ===",
        trial_summary,
        "",
    ]

    if failure_block:
        prompt_lines.append(failure_block)

    if knowledge_context:
        prompt_lines += [
            "=== DOMAIN KNOWLEDGE ===",
            knowledge_context,
            "",
        ]

    prompt_lines += [
        "=== ALLOWED MODEL FAMILIES & SEARCH SPACES ===",
        json.dumps(ss_summary, indent=2),
        "",
        "=" * 60,
        "INSTRUCTIONS — Think through the following 4 steps:",
        "=" * 60,
        "",
        "STEP 1 — DIAGNOSE",
        "What does the current best model do well and poorly?",
        "- Where is RMSE highest? (check rmse_by_range)",
        "- Which model families have been under-explored?",
        "- What failure patterns should be avoided?",
        "(This is your thinking — do not output it.)",
        "",
        "STEP 2 — HYPOTHESIZE",
        "What specific change could address the weakness? Consider:",
        "  a) Fine-tune hyperparameters of current best family (exploit)",
        "  b) Try an under-explored family (explore)",
        "  c) Challenge a domain assumption, e.g. change fly-ash k-factor (domain_challenge)",
        "  d) Propose a feature modification (feature_mod)",
        "Use domain knowledge above to ground your hypothesis.",
        "(This is your thinking — do not output it.)",
        "",
        "STEP 3 — PREDICT",
        "Be specific: 'I expect RMSE to change by X MPa because...'",
        "  - expected_delta > 0 means RMSE gets WORSE (bad proposal)",
        "  - expected_delta < 0 means RMSE IMPROVES (what we want)",
        "(This is your thinking — do not output it.)",
        "",
        "STEP 4 — PROPOSE",
        f"Output EXACTLY {n_proposals} proposal(s) as a JSON array.",
        "",
        "REQUIRED JSON schema for EACH proposal:",
        json.dumps(ENHANCED_PROPOSAL_SCHEMA, indent=2),
        "",
        "Diversity constraints:",
        "  - At most 2 proposals from the same model family",
        "  - At most 1 exploit proposal (near current best)",
        "  - At least 1 explore proposal (under-explored family or large param change)",
        f"  - proposal_family must be one of: {sorted(PROPOSAL_FAMILIES)}",
        "",
        "Output ONLY the JSON array from STEP 4. No markdown fences. No other text.",
        "",
        "Example output:",
        json.dumps(_example_proposals(), indent=2),
    ]

    return "\n".join(prompt_lines)


# ── Response parsing ──────────────────────────────────────────────────────────

def parse_enhanced_proposal(raw_response: str) -> list[dict]:
    """
    Parse the LLM's JSON array response into a list of validated proposal dicts.

    Returns:
        List of proposal dicts. Each has at minimum: model_name, params,
        hypothesis, mechanism, expected_delta, proposal_family.

    Raises:
        ValueError if parsing fails completely.
    """
    # Strip markdown fences
    cleaned = re.sub(r"```(?:json)?", "", raw_response).strip()

    # Try to extract JSON array
    try:
        proposals = json.loads(cleaned)
    except json.JSONDecodeError:
        # Try to find a JSON array inside the text
        match = re.search(r"\[.*?\]", cleaned, re.DOTALL)
        if not match:
            raise ValueError(
                f"No JSON array found in LLM response. Raw: {raw_response[:300]}"
            )
        try:
            proposals = json.loads(match.group())
        except json.JSONDecodeError as exc:
            raise ValueError(f"JSON parse failed: {exc}. Raw: {raw_response[:300]}")

    if isinstance(proposals, dict):
        proposals = [proposals]   # single proposal, wrap it

    validated = []
    for p in proposals:
        v = _validate_proposal(p)
        if v is not None:
            validated.append(v)

    if not validated:
        raise ValueError(f"No valid proposals after validation. Raw: {raw_response[:300]}")

    return validated


def _validate_proposal(p: dict) -> dict | None:
    """
    Validate and normalise a single proposal dict.
    Returns None if fatally malformed.
    """
    if not isinstance(p, dict):
        logger.warning(f"Proposal is not a dict: {p!r}")
        return None

    if "model_name" not in p:
        logger.warning(f"Proposal missing 'model_name': {p!r}")
        return None

    if "params" not in p or not isinstance(p.get("params"), dict):
        logger.warning(f"Proposal missing 'params' dict: {p!r}")
        p["params"] = {}

    # Ensure expected_delta is present and numeric
    ed = p.get("expected_delta")
    if ed is None:
        logger.warning(
            f"Proposal missing 'expected_delta' — defaulting to 0.0. "
            f"Check LLM chain-of-thought prompt."
        )
        p["expected_delta"] = 0.0
    else:
        try:
            p["expected_delta"] = float(ed)
        except (TypeError, ValueError):
            p["expected_delta"] = 0.0

    # Normalise proposal_family
    pf = p.get("proposal_family", "explore")
    if pf not in PROPOSAL_FAMILIES:
        logger.debug(f"Unknown proposal_family '{pf}' — defaulting to 'explore'")
        p["proposal_family"] = "explore"

    # Ensure hypothesis and mechanism are strings
    p.setdefault("hypothesis", "")
    p.setdefault("mechanism",  "")

    return p


# ── Helpers ───────────────────────────────────────────────────────────────────

def _summarise_trials(recent_trials: list[dict]) -> str:
    """Format recent trial history as compact text for the prompt."""
    if not recent_trials:
        return "  (no recent trials)"
    lines = []
    for t in recent_trials[-10:]:
        model   = t.get("model_name", "?")
        score   = t.get("composite_score", 0.0)
        status  = t.get("selection_status", "?")
        lines.append(f"  {model:<28} score={score:.4f}  status={status}")
    return "\n".join(lines)


def _fmt_bad_params(bad_params: dict) -> str:
    """Format bad param ranges concisely for the prompt."""
    parts = []
    for param, data in list(bad_params.items())[:5]:   # top 5
        bad = data.get("low_score_values", [])
        if bad:
            try:
                lo, hi = min(bad), max(bad)
                parts.append(f"{param}∈[{lo:.3g},{hi:.3g}]")
            except Exception:
                pass
    return ", ".join(parts) if parts else "(none)"


def _example_proposals() -> list[dict]:
    """Return illustrative example proposals for the prompt."""
    return [
        {
            "model_name": "LGBMRegressor",
            "params": {"n_estimators": 350, "num_leaves": 20,
                       "learning_rate": 0.08, "min_child_samples": 15},
            "hypothesis": "Reducing num_leaves from 31 to 20 improves generalisation on low-sample training sets",
            "mechanism": "Smaller leaf count reduces variance on n<1500 training sets (LightGBM leaf-wise growth)",
            "expected_delta": -0.08,
            "proposal_family": "exploit",
        },
        {
            "model_name": "XGBRegressor",
            "params": {"n_estimators": 300, "max_depth": 4,
                       "learning_rate": 0.10, "subsample": 0.8},
            "hypothesis": "XGBoost with depth-4 trees captures cement-age non-linearity better than current LGBM",
            "mechanism": "Level-wise growth avoids extreme splits in small samples; depth=4 matches concrete dataset complexity",
            "expected_delta": -0.05,
            "proposal_family": "explore",
        },
    ]


# ── LLM calibration utilities ────────────────────────────────────────────────

def compute_calibration_stats(hypothesis_archive: list[dict]) -> dict:
    """
    Compute LLM calibration statistics from a list of resolved hypotheses.

    Args:
        hypothesis_archive: list of dicts with expected_delta_rmse and actual_delta_rmse

    Returns:
        {mean_calibration_error, median_error, n_resolved, well_calibrated}
    """
    resolved = [
        h for h in hypothesis_archive
        if h.get("expected_delta_rmse") is not None
        and h.get("actual_delta_rmse") is not None
    ]

    if not resolved:
        return {"mean_calibration_error": None, "n_resolved": 0,
                "well_calibrated": None}

    errors = [
        abs(h["expected_delta_rmse"] - h["actual_delta_rmse"])
        for h in resolved
    ]
    errors.sort()
    mean_err = sum(errors) / len(errors)
    median_err = errors[len(errors) // 2]

    return {
        "mean_calibration_error":   round(mean_err, 4),
        "median_calibration_error": round(median_err, 4),
        "n_resolved":               len(resolved),
        "well_calibrated":          mean_err <= 0.05,  # Track B criterion
    }


# ── CLI quick-test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    sample_context = {
        "current_best": {
            "model_name": "LGBMRegressor",
            "composite_score": 0.9202,
            "rmse": 4.81,
            "rmse_by_range": {"low_0_30": 3.2, "mid_30_60": 4.8, "high_60+": 7.1},
        },
        "recent_trials": [
            {"model_name": "LGBMRegressor", "composite_score": 0.912,
             "selection_status": "no_improvement"},
            {"model_name": "XGBRegressor", "composite_score": 0.905,
             "selection_status": "rejected_validation_fail"},
        ],
        "allowed_families": ["LGBMRegressor", "XGBRegressor", "RandomForestRegressor"],
        "search_spaces": {
            "LGBMRegressor": {"search_space": {
                "n_estimators": {"type": "int", "low": 100, "high": 800},
                "num_leaves": {"type": "int", "low": 10, "high": 100},
                "learning_rate": {"type": "float", "low": 0.01, "high": 0.30, "log": True},
            }},
        },
        "n_proposals": 2,
        "cycle": 5,
    }

    prompt = build_chain_of_thought_prompt(sample_context)
    print("=== GENERATED PROMPT (first 2000 chars) ===")
    print(prompt[:2000])
    print("...")
