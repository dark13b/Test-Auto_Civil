"""
failure_analyzer.py — AutoCivil Track B
========================================
Extracts systematic failure patterns from experiment_memory.json
so the proposal engine can actively avoid known bad configurations.

Usage (standalone):
    from failure_analyzer import FailureAnalyzer
    patterns = FailureAnalyzer(memory_path).extract_failure_patterns()

Usage (in research_loop.py):
    failure_context = FailureAnalyzer(memory_path).extract_failure_patterns()
    # Pass failure_context into _select_scout_candidates() → LLM prompt
"""

from __future__ import annotations

import json
import math
import logging
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

FAILURE_STATUSES = frozenset(
    {"rejected_validation_fail", "no_improvement", "error", "timeout"}
)
LOW_SCORE_THRESHOLD = 0.85   # composite score below this → "bad config"
MIN_TRIALS_FOR_PATTERN = 3   # don't report a pattern unless seen ≥ N times


# ── Helpers ──────────────────────────────────────────────────────────────────

def _load_memory(path: Path) -> dict:
    """Load experiment_memory.json, return empty structure if missing."""
    if path.exists():
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    return {"runs": []}


def _all_trials(memory: dict) -> list[dict]:
    """Flatten all trials across all runs into a single list."""
    trials = []
    for run in memory.get("runs", []):
        for trial in run.get("trials", []):
            trials.append(trial)
    return trials


# ── Main class ───────────────────────────────────────────────────────────────

class FailureAnalyzer:
    """
    Analyses experiment_memory.json and surfaces:
        - model families with high failure rates
        - hyperparameter ranges that consistently score low
        - validator failure reasons
        - clustered bad-config profiles

    All results are plain-Python dicts so they can be JSON-serialised
    and dropped directly into an LLM prompt.
    """

    def __init__(self, memory_path: Path | dict[str, Any]):
        if isinstance(memory_path, dict):
            self.memory_path = Path("<memory>")
            self.memory = dict(memory_path)
        else:
            self.memory_path = Path(memory_path)
            self.memory = _load_memory(self.memory_path)
        self._all: list[dict] = _all_trials(self.memory)
        self._failed: list[dict] = [
            t for t in self._all
            if t.get("selection_status") in FAILURE_STATUSES
        ]

    # ── Public API ───────────────────────────────────────────────────────────

    def extract_failure_patterns(self) -> dict[str, Any]:
        """
        Top-level call. Returns a dict with four sub-sections.
        Safe to call even when memory is empty — returns empty patterns.
        """
        if not self._all:
            logger.info("FailureAnalyzer: memory empty, returning empty patterns.")
            return self._empty_patterns()

        family_rates = self._family_fail_rates()
        patterns = {
            "families_with_high_fail_rate":       [
                {"family": family, "fail_rate": rate} for family, rate in sorted(family_rates.items(), key=lambda item: (-item[1], item[0]))
            ],
            "parameter_ranges_that_always_fail":   self._bad_param_ranges(),
            "validator_failure_reasons":           [
                {"reason": reason, "count": count}
                for reason, count in self._aggregate_validator_reasons().items()
            ],
            "low_score_hyperparameter_profiles":   self._cluster_bad_configs(),
            "_meta": {
                "total_trials_analysed": len(self._all),
                "failed_trials_count":   len(self._failed),
                "failure_rate":          round(
                    len(self._failed) / max(len(self._all), 1), 3
                ),
            },
        }
        return patterns

    def format_for_prompt(self, patterns: dict | None = None) -> str:
        """
        Return a compact, human-readable string suitable for LLM injection.
        Call extract_failure_patterns() first, or pass the result in.
        """
        if patterns is None:
            patterns = self.extract_failure_patterns()

        lines: list[str] = ["=== KNOWN FAILURE PATTERNS (AVOID) ==="]

        # Family fail rates
        fam_rates = patterns.get("families_with_high_fail_rate", {})
        if fam_rates:
            if isinstance(fam_rates, dict):
                high_fail = {k: v for k, v in fam_rates.items() if v > 0.40}
            else:
                high_fail = {item.get("family", "unknown"): item.get("fail_rate", 0.0) for item in fam_rates if item.get("fail_rate", 0.0) > 0.40}
            if high_fail:
                lines.append(
                    "High-failure model families (fail rate > 40%):\n  "
                    + ", ".join(f"{k}={v:.0%}" for k, v in high_fail.items())
                )

        # Bad param ranges
        bad_params = patterns.get("parameter_ranges_that_always_fail", {})
        if bad_params:
            lines.append("Hyperparameter ranges strongly associated with low scores:")
            for param, data in bad_params.items():
                bad_vals = data.get("low_score_values", [])
                if len(bad_vals) >= MIN_TRIALS_FOR_PATTERN:
                    try:
                        summary = _summarise_values(bad_vals)
                        lines.append(f"  {param}: avoid {summary}")
                    except Exception:
                        pass

        # Validator reasons
        reasons = patterns.get("validator_failure_reasons", {})
        if reasons:
            if isinstance(reasons, dict):
                top = sorted(reasons.items(), key=lambda x: -x[1])[:5]
            else:
                top = sorted(((item.get("reason", "unknown"), item.get("count", 0)) for item in reasons), key=lambda x: -x[1])[:5]
            lines.append(
                "Top validator rejection reasons:\n  "
                + "; ".join(f"{r} ({n}x)" for r, n in top)
            )

        # Meta
        meta = patterns.get("_meta", {})
        if meta:
            lines.append(
                f"[Analysed {meta['total_trials_analysed']} trials, "
                f"{meta['failed_trials_count']} failures, "
                f"overall fail rate {meta['failure_rate']:.1%}]"
            )

        return "\n".join(lines)

    # ── Private helpers ──────────────────────────────────────────────────────

    def _empty_patterns(self) -> dict[str, Any]:
        return {
            "families_with_high_fail_rate": [],
            "parameter_ranges_that_always_fail": {},
            "validator_failure_reasons": [],
            "low_score_hyperparameter_profiles": [],
            "_meta": {"total_trials_analysed": 0, "failed_trials_count": 0,
                      "failure_rate": 0.0},
        }

    def _family_fail_rates(self) -> dict[str, float]:
        """
        For each model family, compute fraction of trials that ended in failure.
        Only include families with at least MIN_TRIALS_FOR_PATTERN total trials.
        """
        total: Counter = Counter()
        fail:  Counter = Counter()

        for trial in self._all:
            family = trial.get("model_name", "unknown")
            total[family] += 1
            if trial.get("selection_status") in FAILURE_STATUSES:
                fail[family] += 1

        return {
            str(fam).lower().replace("modelfamily", "family-"): round(fail[fam] / total[fam], 3)
            for fam in total
            if total[fam] >= 1
        }

    def _bad_param_ranges(self) -> dict[str, dict]:
        """
        For each hyperparameter seen in trials, split observed values into
        'low_score' (composite < threshold) and 'high_score' buckets.

        Returns a dict keyed by param name with lists of example bad values.
        """
        buckets: dict[str, dict] = defaultdict(
            lambda: {"low_score_values": [], "high_score_values": []}
        )

        for trial in self._all:
            params = trial.get("params") or {}
            score  = trial.get("composite_score") or 0.0
            for key, val in params.items():
                if val is None:
                    continue
                if score < LOW_SCORE_THRESHOLD:
                    buckets[key]["low_score_values"].append(val)
                else:
                    buckets[key]["high_score_values"].append(val)

        # Trim to avoid huge JSON blobs
        result = {}
        for param, data in buckets.items():
            lv = data["low_score_values"]
            hv = data["high_score_values"]
            if len(lv) >= MIN_TRIALS_FOR_PATTERN or len(hv) >= MIN_TRIALS_FOR_PATTERN:
                result[param] = {
                    "low_score_values":  lv[-50:],   # keep last 50
                    "high_score_values": hv[-50:],
                }
        return result

    def _aggregate_validator_reasons(self) -> dict[str, int]:
        """
        Count how many times each validator rejection reason appears
        across all failed trials.
        """
        reason_counter: Counter = Counter()
        for trial in self._failed:
            reasons = list(trial.get("validator_issues", []) or [])
            reasons.extend(trial.get("hard_fail_reasons", []) or [])
            reasons.extend(trial.get("engineering_caution_reasons", []) or [])
            # validator_issues may be a list of strings or dicts
            for item in reasons:
                if isinstance(item, str):
                    reason_counter[item] += 1
                elif isinstance(item, dict):
                    label = item.get("reason") or item.get("message") or str(item)
                    reason_counter[label] += 1
        return dict(reason_counter.most_common(20))

    def _cluster_bad_configs(self) -> list[dict]:
        """
        Simple prototype-based grouping: group failed trials by model family,
        return a short description of the 'typical bad config' per family.
        Avoids heavy dependencies (sklearn) intentionally.
        """
        by_family: dict[str, list[dict]] = defaultdict(list)
        for trial in self._failed:
            family = trial.get("model_name", "unknown")
            params = trial.get("params") or {}
            score  = trial.get("composite_score") or 0.0
            by_family[family].append({"params": params, "score": score})

        profiles = []
        for family, entries in by_family.items():
            if len(entries) < MIN_TRIALS_FOR_PATTERN:
                continue
            # Compute mean numeric param values
            numeric_means: dict[str, float] = {}
            keys = {k for e in entries for k in e["params"]}
            for key in keys:
                vals = [
                    e["params"][key]
                    for e in entries
                    if isinstance(e["params"].get(key), (int, float))
                ]
                if vals:
                    numeric_means[key] = round(sum(vals) / len(vals), 4)

            avg_score = round(sum(e["score"] for e in entries) / len(entries), 4)
            profiles.append({
                "family": family,
                "count": len(entries),
                "avg_composite_score": avg_score,
                "typical_bad_params": numeric_means,
            })

        return sorted(profiles, key=lambda p: p["avg_composite_score"])


# ── Utility ───────────────────────────────────────────────────────────────────

def _summarise_values(vals: list) -> str:
    """
    Return a compact description of a list of values.
    Numbers → show range; strings → show top-3 unique values.
    """
    numeric = [v for v in vals if isinstance(v, (int, float))]
    if numeric:
        lo, hi = min(numeric), max(numeric)
        mean   = sum(numeric) / len(numeric)
        return f"[{lo:.3g}, {hi:.3g}] (mean={mean:.3g}, n={len(numeric)})"
    strings = [str(v) for v in vals]
    unique  = list(dict.fromkeys(strings))[:5]
    return f"{unique} (n={len(vals)})"


# ── CLI quick-test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("experiment_memory.json")
    analyzer = FailureAnalyzer(path)
    patterns = analyzer.extract_failure_patterns()
    print(json.dumps(patterns, indent=2, default=str))
    print("\n--- PROMPT SNIPPET ---")
    print(analyzer.format_for_prompt(patterns))
