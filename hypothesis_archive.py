"""
hypothesis_archive.py — AutoCivil Track B
==========================================
Persistent store for all experiment hypotheses with calibration tracking.

Schema:
  hypothesis_archive.json → {schema_version, hypotheses: [...], calibration_summary}

Each hypothesis must eventually have BOTH:
  - expected_delta_rmse  (from LLM proposal, before trial)
  - actual_delta_rmse    (measured after trial)

This enables LLM calibration measurement:
  calibration_error = mean(|expected - actual|)

Usage:
    from hypothesis_archive import HypothesisArchive
    archive = HypothesisArchive(outputs_dir / "hypothesis_archive.json")

    # Before trial:
    hyp_id = archive.add(
        proposal=proposal_dict,
        claim="...", mechanism="...",
        expected_delta_rmse=-0.08,
        source="llm",
        run_id="...", cycle=3,
    )

    # After trial:
    archive.resolve(
        hyp_id,
        actual_delta_rmse=-0.12,
        outcome="accepted",
        trial_number=62,
    )
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1
HypothesisSource = Literal["llm", "deterministic", "feature_lab"]
HypothesisOutcome = Literal["accepted", "rejected", "no_improvement", "error",
                             "timeout", "pending"]


# ── Data structures ───────────────────────────────────────────────────────────

def _new_hypothesis(
    *,
    run_id: str,
    cycle: int,
    source: HypothesisSource,
    claim: str,
    mechanism: str,
    expected_delta_rmse: float,
    proposal: dict,
    prediction: str = "",
    test: str = "",
) -> dict:
    return {
        "hypothesis_id":       f"H-{uuid.uuid4().hex[:8].upper()}",
        "run_id":              run_id,
        "cycle":               cycle,
        "source":              source,
        "claim":               claim,
        "mechanism":           mechanism,
        "prediction":          prediction,
        "test":                test,
        "expected_delta_rmse": expected_delta_rmse,
        "actual_delta_rmse":   None,   # filled by resolve()
        "calibration_error":   None,   # filled by resolve()
        "proposal":            proposal,
        "outcome":             "pending",
        "trial_number":        None,
        "timestamp":           datetime.now(timezone.utc).isoformat(),
        "resolved_at":         None,
    }


def _calibration_summary(hypotheses: list[dict]) -> dict:
    resolved = [
        h for h in hypotheses
        if h.get("calibration_error") is not None
    ]
    if not resolved:
        return {
            "mean_calibration_error": None,
            "total_hypotheses":       len(hypotheses),
            "resolved_hypotheses":    0,
            "acceptance_rate":        None,
            "well_calibrated":        None,   # calibration_error ≤ 0.05
        }

    errors      = [h["calibration_error"] for h in resolved]
    accepted    = [h for h in resolved if h["outcome"] == "accepted"]
    mean_err    = round(sum(errors) / len(errors), 4)

    return {
        "mean_calibration_error": mean_err,
        "total_hypotheses":       len(hypotheses),
        "resolved_hypotheses":    len(resolved),
        "acceptance_rate":        round(len(accepted) / max(len(resolved), 1), 3),
        "well_calibrated":        mean_err <= 0.05,   # Track B criterion
    }


# ── Main class ────────────────────────────────────────────────────────────────

class HypothesisArchive:
    """
    Thread-unsafe, single-process store for hypotheses.
    Writes to disk after every mutation.
    """

    def __init__(self, archive_path: Path):
        self.archive_path = Path(archive_path)
        self._data: dict = self._load()

    # ── Persistence ──────────────────────────────────────────────────────────

    def _load(self) -> dict:
        if self.archive_path.exists():
            with open(self.archive_path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            # Migrate schema if needed
            if data.get("schema_version", 0) < SCHEMA_VERSION:
                data["schema_version"] = SCHEMA_VERSION
            return data
        return {"schema_version": SCHEMA_VERSION, "hypotheses": [],
                "calibration_summary": {}}

    def _save(self) -> None:
        self.archive_path.parent.mkdir(parents=True, exist_ok=True)
        self._data["calibration_summary"] = _calibration_summary(
            self._data["hypotheses"]
        )
        with open(self.archive_path, "w", encoding="utf-8") as fh:
            json.dump(self._data, fh, indent=2, default=str)

    # ── Mutations ─────────────────────────────────────────────────────────────

    def add(
        self,
        *,
        proposal: dict,
        claim: str,
        mechanism: str,
        expected_delta_rmse: float,
        source: HypothesisSource = "llm",
        run_id: str = "",
        cycle: int = 0,
        prediction: str = "",
        test: str = "",
    ) -> str:
        """
        Record a new hypothesis BEFORE the trial runs.
        Returns the hypothesis_id (use it to call resolve() later).

        NOTE: expected_delta_rmse is REQUIRED. Never store a hypothesis
        without it — the calibration_error computation depends on it.
        """
        if expected_delta_rmse is None:
            raise ValueError(
                "expected_delta_rmse is required for every hypothesis. "
                "Ensure the LLM proposal includes 'expected_delta'."
            )

        hyp = _new_hypothesis(
            run_id=run_id,
            cycle=cycle,
            source=source,
            claim=claim,
            mechanism=mechanism,
            expected_delta_rmse=float(expected_delta_rmse),
            proposal=proposal,
            prediction=prediction,
            test=test,
        )
        self._data["hypotheses"].append(hyp)
        self._save()
        logger.info(f"Hypothesis added: {hyp['hypothesis_id']} — {claim[:80]}")
        return hyp["hypothesis_id"]

    def resolve(
        self,
        hypothesis_id: str,
        *,
        actual_delta_rmse: float,
        outcome: HypothesisOutcome,
        trial_number: int | None = None,
    ) -> dict:
        """
        Fill in actual_delta_rmse and outcome after the trial has run.
        Computes calibration_error = |expected - actual|.

        Both expected AND actual delta must be stored — this is a
        non-negotiable requirement per the Track B spec.
        """
        hyp = self._find(hypothesis_id)
        if hyp is None:
            raise KeyError(f"Hypothesis {hypothesis_id} not found.")

        hyp["actual_delta_rmse"]  = float(actual_delta_rmse)
        hyp["outcome"]            = outcome
        hyp["trial_number"]       = trial_number
        hyp["resolved_at"]        = datetime.now(timezone.utc).isoformat()
        hyp["calibration_error"]  = round(
            abs(hyp["expected_delta_rmse"] - hyp["actual_delta_rmse"]), 4
        )

        self._save()
        logger.info(
            f"Hypothesis resolved: {hypothesis_id} | outcome={outcome} "
            f"| expected_delta={hyp['expected_delta_rmse']:.4f} "
            f"| actual_delta={actual_delta_rmse:.4f} "
            f"| calibration_error={hyp['calibration_error']:.4f}"
        )
        return hyp

    def add_feature_lab_entry(
        self,
        *,
        claim: str,
        mechanism: str,
        expected_delta_rmse: float,
        feature_code: str,
        run_id: str = "",
        cycle: int = 0,
    ) -> str:
        """Convenience wrapper for FeatureLab-generated hypotheses."""
        proposal = {
            "model_name":    "FeatureLab",
            "params":        {},
            "feature_code":  feature_code,
        }
        return self.add(
            proposal=proposal,
            claim=claim,
            mechanism=mechanism,
            expected_delta_rmse=expected_delta_rmse,
            source="feature_lab",
            run_id=run_id,
            cycle=cycle,
        )

    # ── Queries ──────────────────────────────────────────────────────────────

    def all_hypotheses(self) -> list[dict]:
        return self._data["hypotheses"]

    def pending(self) -> list[dict]:
        return [h for h in self._data["hypotheses"] if h["outcome"] == "pending"]

    def resolved(self) -> list[dict]:
        return [h for h in self._data["hypotheses"] if h["outcome"] != "pending"]

    def calibration_summary(self) -> dict:
        return self._data.get("calibration_summary", {})

    def is_well_calibrated(self) -> bool:
        """True if mean calibration error ≤ 0.05 (Track B criterion)."""
        summary = self.calibration_summary()
        if summary.get("well_calibrated") is None:
            return True   # not enough data yet — don't penalise
        return bool(summary["well_calibrated"])

    def as_novelty_archive(self) -> list[dict]:
        """
        Return the archive in the format expected by NoveltyScorer.compute_novelty_score().
        """
        return [
            {"proposal": h["proposal"]}
            for h in self._data["hypotheses"]
            if h.get("proposal")
        ]

    def print_summary(self) -> None:
        """Print a human-readable summary to stdout."""
        s = self.calibration_summary()
        print("=== Hypothesis Archive Summary ===")
        print(f"  Total hypotheses:      {s.get('total_hypotheses', 0)}")
        print(f"  Resolved:              {s.get('resolved_hypotheses', 0)}")
        print(f"  Acceptance rate:       {s.get('acceptance_rate', 'N/A')}")
        print(f"  Mean calibration err:  {s.get('mean_calibration_error', 'N/A')}")
        print(f"  Well-calibrated (≤0.05): {s.get('well_calibrated', 'N/A')}")
        pending_n = len(self.pending())
        if pending_n:
            print(f"  WARNING: {pending_n} hypothesis(es) still pending (resolve() not called)")

    # ── Internal ─────────────────────────────────────────────────────────────

    def _find(self, hypothesis_id: str) -> dict | None:
        for h in self._data["hypotheses"]:
            if h["hypothesis_id"] == hypothesis_id:
                return h
        return None


# ── CLI quick-test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import tempfile, os

    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "hypothesis_archive.json"
        archive = HypothesisArchive(path)

        hyp_id = archive.add(
            proposal={"model_name": "LGBMRegressor",
                      "params": {"n_estimators": 420, "num_leaves": 20}},
            claim="Reducing num_leaves from 31 to 20 improves generalisation",
            mechanism="Smaller trees reduce variance on 1400-sample training set",
            expected_delta_rmse=-0.08,
            run_id="TEST-001",
            cycle=1,
        )
        print(f"Added: {hyp_id}")

        archive.resolve(hyp_id, actual_delta_rmse=-0.12, outcome="accepted",
                        trial_number=42)
        archive.print_summary()
