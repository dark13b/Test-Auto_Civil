"""
research_loop_patch.py — AutoCivil Track B
============================================
This file documents and provides all the code snippets needed to integrate
the Track B modules into research_loop.py.

It is structured as a fully working standalone module that wraps the
existing research loop. You can either:
  A) Copy each section into the appropriate place in research_loop.py, or
  B) Import AutoResearchOrchestrator and call it from research_loop.py.

Integration map:
  ┌─────────────────────────────────────────────────────────────────┐
  │ research_loop.py                                                │
  │   start_research_cycle()                                        │
  │     │                                                           │
  │     ├─► [INJECT 1] FailureAnalyzer.extract_failure_patterns()  │
  │     ├─► [INJECT 2] get_knowledge_context()                      │
  │     │                                                           │
  │     ├─► generate_experiment_proposals()  ← patched prompt       │
  │     │     │                                                      │
  │     │     └─► [INJECT 3] NoveltyScorer.compute_novelty_score()  │
  │     │           reject if < 0.20                                 │
  │     │                                                           │
  │     ├─► execute_trial()                                         │
  │     │     │                                                      │
  │     │     └─► [INJECT 4] HypothesisArchive.resolve()           │
  │     │           record (expected_delta, actual_delta)            │
  │     │                                                           │
  │     ├─► [INJECT 5] FeatureLab.run_cycle() (every N cycles)     │
  │     └─► [INJECT 6] ScientificReportReviewer.review()           │
  └─────────────────────────────────────────────────────────────────┘
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ── Track B Orchestrator ──────────────────────────────────────────────────────

class AutoResearchOrchestrator:
    """
    Wraps all Track B modules and exposes a clean API to research_loop.py.

    Initialise once at the start of a run; call the appropriate method
    at each injection point.
    """

    def __init__(self, config: dict, outputs_dir: Path, llm_backend: Any = None):
        self.config       = config
        self.outputs_dir  = Path(outputs_dir)
        self.backend      = llm_backend
        self._rc          = config.get("research", {})

        memory_path = (
            self.outputs_dir / config.get("memory_filename", "experiment_memory.json")
        )

        # ── Module instances ──────────────────────────────────────────────────
        from failure_analyzer   import FailureAnalyzer
        from hypothesis_archive import HypothesisArchive
        from novelty_scorer     import NoveltyScorer, DiversityLogger

        self.failure_analyzer = FailureAnalyzer(memory_path)

        archive_filename = self._rc.get(
            "hypothesis_archive_filename", "hypothesis_archive.json"
        )
        self.hypothesis_archive = HypothesisArchive(
            self.outputs_dir / archive_filename
        )

        self.novelty_scorer = NoveltyScorer(
            search_spaces=config.get("models", {}),
            threshold=self._rc.get("novelty_gate_threshold", 0.20),
        )

        self.diversity_logger = DiversityLogger(
            self.outputs_dir / "proposal_diversity.json"
        )

        # FeatureLab — only if enabled
        self._feature_lab = None
        if self._rc.get("feature_lab_enabled", False) and llm_backend is not None:
            from feature_lab import FeatureLab
            self._feature_lab = FeatureLab(config, self.outputs_dir, llm_backend)

        self._current_run_id = ""
        self._pending_hypothesis_ids: dict[int, str] = {}  # trial_number → hyp_id

    # ── INJECT 1 + 2: cycle start ─────────────────────────────────────────────

    def prepare_cycle_context(
        self,
        run_id: str,
        cycle: int,
        most_recent_family: str = "",
        feature_list: list[str] | None = None,
    ) -> dict:
        """
        Call at the BEGINNING of each research cycle.

        Returns:
            dict with failure_context and knowledge_context ready
            to pass into the proposal engine.
        """
        self._current_run_id = run_id
        self._current_cycle  = cycle

        # INJECT 1 — failure patterns
        failure_context: dict = {}
        if self._rc.get("failure_analysis_enabled", True):
            try:
                # Reload memory in case it was updated in a previous cycle
                from failure_analyzer import FailureAnalyzer
                self.failure_analyzer = FailureAnalyzer(
                    self.failure_analyzer.memory_path
                )
                failure_context = self.failure_analyzer.extract_failure_patterns()
                logger.info(
                    f"Cycle {cycle}: failure_analyzer found "
                    f"{failure_context['_meta']['failed_trials_count']} failed trials."
                )
            except Exception as exc:
                logger.warning(f"FailureAnalyzer error (non-fatal): {exc}")

        # INJECT 2 — domain knowledge
        knowledge_context = ""
        if self._rc.get("knowledge_base_enabled", True):
            try:
                from knowledge_base import get_knowledge_context
                knowledge_context = get_knowledge_context(
                    most_recent_family or "LGBMRegressor",
                    feature_list or [],
                )
            except Exception as exc:
                logger.warning(f"KnowledgeBase error (non-fatal): {exc}")

        return {
            "failure_context":   failure_context,
            "knowledge_context": knowledge_context,
        }

    # ── INJECT 3: novelty gate ────────────────────────────────────────────────

    def filter_proposals_by_novelty(
        self,
        proposals: list[dict],
        cycle: int,
    ) -> list[dict]:
        """
        Apply novelty gate + diversity budget to a list of proposals.
        Log every score to proposal_diversity.json.

        Returns:
            Filtered list of proposals that pass the novelty gate.
        """
        archive = self.hypothesis_archive.as_novelty_archive()
        accepted: list[dict] = []

        for proposal in proposals:
            passes, score = self.novelty_scorer.is_novel_enough(proposal, archive)
            self.diversity_logger.record(proposal, score, cycle)

            if not passes:
                logger.info(
                    f"Novelty gate REJECTED: {proposal.get('model_name')} "
                    f"score={score:.3f} < {self.novelty_scorer.threshold}"
                )
            else:
                logger.info(
                    f"Novelty gate PASSED:   {proposal.get('model_name')} "
                    f"score={score:.3f}"
                )
                accepted.append(proposal)

        # Apply diversity budget constraints
        accepted = self.novelty_scorer.enforce_diversity_budget(accepted, archive)

        if not accepted and proposals:
            logger.warning(
                "All proposals failed novelty gate. Falling back to first proposal. "
                "Consider increasing search space exploration in config.yaml."
            )
            accepted = proposals[:1]   # safety fallback

        return accepted

    # ── INJECT 4: record hypothesis before trial ──────────────────────────────

    def record_hypothesis_before_trial(
        self,
        proposal: dict,
        trial_number: int,
    ) -> str | None:
        """
        Call BEFORE executing a trial.
        Returns hypothesis_id to pass to resolve_hypothesis_after_trial().
        """
        if not self._rc.get("llm_require_expected_delta", True):
            return None

        expected_delta = proposal.get("expected_delta")
        if expected_delta is None:
            logger.warning(
                "Proposal missing 'expected_delta' — skipping hypothesis record. "
                "Enable llm_chain_of_thought in config.yaml to fix this."
            )
            return None

        try:
            hyp_id = self.hypothesis_archive.add(
                proposal=proposal,
                claim=proposal.get("hypothesis", ""),
                mechanism=proposal.get("mechanism", ""),
                expected_delta_rmse=float(expected_delta),
                source="llm",
                run_id=self._current_run_id,
                cycle=getattr(self, "_current_cycle", 0),
            )
            self._pending_hypothesis_ids[trial_number] = hyp_id
            return hyp_id
        except Exception as exc:
            logger.error(f"Failed to record hypothesis: {exc}")
            return None

    def resolve_hypothesis_after_trial(
        self,
        trial_number: int,
        actual_delta_rmse: float,
        outcome: str,
    ) -> None:
        """
        Call AFTER a trial completes with the measured RMSE change.

        actual_delta_rmse should be:
            reference_best_rmse - new_trial_rmse
            (positive = improvement, negative = worse)
        Note: hypothesis_archive stores this as RMSE delta, sign convention:
            negative expected_delta = we expect improvement
        So pass in: actual_delta_rmse = new_rmse - reference_rmse
                                         (negative = good)
        """
        hyp_id = self._pending_hypothesis_ids.pop(trial_number, None)
        if hyp_id is None:
            return

        try:
            self.hypothesis_archive.resolve(
                hyp_id,
                actual_delta_rmse=actual_delta_rmse,
                outcome=outcome,
                trial_number=trial_number,
            )
        except Exception as exc:
            logger.error(f"Failed to resolve hypothesis {hyp_id}: {exc}")

    # ── INJECT 5: feature lab cycle ───────────────────────────────────────────

    def maybe_run_feature_lab(
        self,
        cycle: int,
        x_train,
        y_train,
        x_val,
        y_val,
        baseline_cv_rmse: float,
        final_metrics: dict,
    ) -> dict:
        """
        Run FeatureLab every N cycles if enabled.
        Call at the END of each main research cycle.

        Returns:
            dict with promoted=True/False and delta if promoted.
        """
        if self._feature_lab is None:
            return {"ran": False, "reason": "feature_lab_disabled"}

        interval = self._rc.get("feature_lab_interval_cycles", 3)
        if cycle % interval != 0:
            return {"ran": False, "reason": f"not a feature_lab cycle (every {interval})"}

        logger.info(f"Cycle {cycle}: running FeatureLab...")

        weakness = {
            "rmse_by_range": final_metrics.get("rmse_by_range", {}),
            "overall_rmse":  final_metrics.get("rmse", baseline_cv_rmse),
        }

        try:
            result = self._feature_lab.run_cycle(
                x_train=x_train,
                y_train=y_train,
                x_val=x_val,
                y_val=y_val,
                baseline_cv_rmse=baseline_cv_rmse,
                weakness=weakness,
                existing_features=list(x_train.columns),
            )

            if result.get("promoted"):
                logger.info(
                    f"FeatureLab: new feature PROMOTED "
                    f"(Δ={result.get('delta', 0.0):+.4f} MPa RMSE)"
                )
                # Record in hypothesis archive
                if result.get("delta") is not None:
                    self.hypothesis_archive.add_feature_lab_entry(
                        claim=f"Feature lab promoted feature with Δ={result['delta']:+.4f}",
                        mechanism="Auto-discovered feature via FeatureLab",
                        expected_delta_rmse=-result["delta"],   # positive delta = improvement
                        feature_code="(see feature_lab/feature_hypotheses.json)",
                        run_id=self._current_run_id,
                        cycle=cycle,
                    )
            return result

        except Exception as exc:
            logger.error(f"FeatureLab cycle failed: {exc}")
            return {"ran": True, "error": str(exc)}

    # ── INJECT 6: scientific report review ───────────────────────────────────

    def review_scientific_report(
        self,
        report_text: str,
    ) -> dict:
        """
        Use a second LLM call to review the generated scientific report.
        Flags reports with any score < 6 (advisory only — never blocks the run).

        Returns:
            {accuracy, novelty, reproducibility, issues, flagged}
        """
        if self.backend is None:
            return {"skipped": True, "reason": "no llm_backend available"}

        prompt = (
            "You are a peer reviewer for a concrete ML paper.\n"
            "Review the following report and score it on:\n"
            "1. Scientific accuracy (1-10): are all claims grounded in data?\n"
            "2. Novelty (1-10): does this contribute something new?\n"
            "3. Reproducibility (1-10): could another researcher replicate this?\n"
            "4. Identified issues: list up to 3 specific problems.\n\n"
            f"Report:\n{report_text[:3000]}\n\n"
            'Output JSON only (no markdown):\n'
            '{"accuracy": N, "novelty": N, "reproducibility": N, "issues": [...]}'
        )

        try:
            raw = self.backend.generate_text(prompt, max_output_tokens=300)
            import re
            cleaned = re.sub(r"```(?:json)?", "", raw).strip()
            review = json.loads(cleaned)
        except Exception as exc:
            logger.warning(f"Report review failed: {exc}")
            return {"skipped": True, "reason": str(exc)}

        # Flag if any score < 6
        scores = [review.get("accuracy", 10), review.get("novelty", 10),
                  review.get("reproducibility", 10)]
        review["flagged"] = any(s < 6 for s in scores if isinstance(s, (int, float)))

        if review["flagged"]:
            logger.warning(
                f"Scientific report flagged: accuracy={review.get('accuracy')} "
                f"novelty={review.get('novelty')} "
                f"reproducibility={review.get('reproducibility')}"
            )
        else:
            logger.info(
                f"Report review OK: accuracy={review.get('accuracy')} "
                f"novelty={review.get('novelty')} "
                f"reproducibility={review.get('reproducibility')}"
            )

        # Save review to disk
        review_path = self.outputs_dir / "scientific_report_review.json"
        with open(review_path, "w", encoding="utf-8") as fh:
            json.dump(review, fh, indent=2)

        return review

    # ── End-of-run summary ───────────────────────────────────────────────────

    def print_track_b_summary(self) -> None:
        """Print a Track B evaluation summary against the 50% criteria."""
        print("\n" + "=" * 60)
        print("TRACK B AutoResearch — Evaluation Summary")
        print("=" * 60)

        cs = self.hypothesis_archive.calibration_summary()
        div = self.diversity_logger.mean_novelty()

        criteria = [
            ("LLM proposal rationale (expected_delta)",
             cs.get("resolved_hypotheses", 0) > 0,
             f"{cs.get('resolved_hypotheses', 0)} resolved hypotheses"),
            ("Failure learning active",
             self._rc.get("failure_analysis_enabled", False),
             "enabled in config"),
            ("Novelty gate active",
             self.novelty_scorer.threshold > 0,
             f"threshold={self.novelty_scorer.threshold}"),
            ("Mean novelty ≥ 0.40",
             div >= 0.40,
             f"mean_novelty={div:.3f}"),
            ("LLM calibration error ≤ 0.05",
             cs.get("well_calibrated", False),
             f"mean_error={cs.get('mean_calibration_error', 'N/A')}"),
            ("Hypothesis archive ≥ 20 entries",
             len(self.hypothesis_archive.all_hypotheses()) >= 20,
             f"{len(self.hypothesis_archive.all_hypotheses())} entries"),
            ("Knowledge base enabled",
             self._rc.get("knowledge_base_enabled", False),
             "enabled in config"),
            ("Feature lab enabled",
             self._feature_lab is not None,
             "enabled" if self._feature_lab else "disabled"),
        ]

        passed = 0
        for name, ok, detail in criteria:
            mark = "✓" if ok else "✗"
            print(f"  {mark} {name}")
            print(f"      → {detail}")
            if ok:
                passed += 1

        score_pct = int(18 + (50 - 18) * passed / len(criteria))
        print(f"\n  Estimated AutoResearch score: ~{score_pct}%  "
              f"({passed}/{len(criteria)} criteria met)")
        print("=" * 60 + "\n")


# ── config.yaml additions (reference) ─────────────────────────────────────────
#
# Paste this block into your config.yaml under the top-level 'research:' key:
#
# research:
#   feature_lab_enabled: false           # set true to enable LLM feature proposals
#   feature_lab_interval_cycles: 3       # run FeatureLab every N main cycles
#   feature_lab_min_delta_rmse: 0.02     # minimum improvement (MPa) to promote
#   feature_lab_sandbox_timeout: 30      # hard timeout for sandbox execution (seconds)
#   novelty_gate_threshold: 0.20         # reject proposals with novelty < this
#   knowledge_base_enabled: true         # inject domain knowledge into LLM prompts
#   failure_analysis_enabled: true       # inject failure patterns into LLM prompts
#   hypothesis_archive_filename: hypothesis_archive.json
#   llm_chain_of_thought: true           # enable STEP 1-4 prompting
#   llm_require_expected_delta: true     # require expected_delta in proposals
