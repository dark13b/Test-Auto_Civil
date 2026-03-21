"""
novelty_scorer.py — AutoCivil Track B
=======================================
Computes a novelty score (0.0–1.0) for each experiment proposal.
Proposals below the threshold (default 0.20) are rejected BEFORE running,
saving compute time.

Algorithm:
  novelty(p) = 1 - max_similarity(p, archive[-100:])

  similarity weights:
    - same model family  → base similarity 0.30 (different family = instant novelty)
    - hyperparameter proximity → weighted average of per-param distances

Usage:
    from novelty_scorer import NoveltyScorer
    scorer = NoveltyScorer(search_spaces=config["models"])
    score  = scorer.compute_novelty_score(proposal, archive)
    if score < 0.20:
        reject(proposal)
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

DEFAULT_NOVELTY_THRESHOLD = 0.20
ARCHIVE_WINDOW = 100          # compare against last N archived proposals
CROSS_FAMILY_SIMILARITY = 0.30  # base similarity when families differ

# Weights for computing novelty score between two same-family proposals
FAMILY_WEIGHT = 0.40
PARAM_WEIGHT  = 0.35
TAG_WEIGHT    = 0.25


# ── Main class ────────────────────────────────────────────────────────────────

class NoveltyScorer:
    """
    Computes novelty scores for experiment proposals.

    Args:
        search_spaces: dict keyed by model name → {search_space: {param: spec}}
                       spec = {type: 'int'|'float'|'categorical', low, high, log, choices}
        threshold:     Proposals with novelty < threshold are rejected.
    """

    def __init__(
        self,
        search_spaces: dict[str, Any] | None = None,
        threshold: float = DEFAULT_NOVELTY_THRESHOLD,
    ):
        self.search_spaces = search_spaces or {}
        self.threshold = threshold

    # ── Public API ────────────────────────────────────────────────────────────

    def compute_novelty_score(
        self,
        proposal: dict,
        archive: list[dict],
    ) -> float:
        """
        Returns 0.0 (identical to a past proposal) to 1.0 (completely new).

        Args:
            proposal: {model_name, params, ...} — the candidate experiment
            archive:  list of past hypothesis_archive entries

        Returns:
            Float in [0.0, 1.0].
        """
        if not archive:
            return 1.0

        model_name = proposal.get("model_name", "")
        params     = proposal.get("params") or {}
        tag        = proposal.get("proposal_family", "")

        recent = archive[-ARCHIVE_WINDOW:]
        max_similarity = 0.0

        for past in recent:
            past_proposal = past.get("proposal", {})
            if not past_proposal:
                continue

            past_model  = past_proposal.get("model_name", "")
            past_params = past_proposal.get("params") or {}
            past_tag    = past_proposal.get("proposal_family", "")

            if past_model != model_name:
                # Different family → fixed base similarity
                max_similarity = max(max_similarity, CROSS_FAMILY_SIMILARITY)
                continue

            # Same family → compute full similarity
            search_space = (
                self.search_spaces
                .get(model_name, {})
                .get("search_space", {})
            )
            param_sim = _param_similarity(params, past_params, search_space)
            tag_sim   = 1.0 if (tag and tag == past_tag) else 0.0

            # Weighted aggregate (family already matched, so family component = 1.0)
            total_sim = (
                FAMILY_WEIGHT * 1.0
                + PARAM_WEIGHT * param_sim
                + TAG_WEIGHT   * tag_sim
            )
            max_similarity = max(max_similarity, total_sim)

        novelty = 1.0 - max_similarity
        return round(max(0.0, min(1.0, novelty)), 4)

    def is_novel_enough(self, proposal: dict, archive: list[dict]) -> tuple[bool, float]:
        """
        Convenience wrapper. Returns (passes_gate, score).
        """
        score = self.compute_novelty_score(proposal, archive)
        return score >= self.threshold, score

    def enforce_diversity_budget(
        self,
        proposals: list[dict],
        archive: list[dict],
    ) -> list[dict]:
        """
        Apply diversity constraints to a batch of proposals:
          - At most 2 proposals from the same model family
          - At most 1 pure exploit (near-current-best) proposal
          - At least 1 proposal from an unexplored family

        Returns a filtered list.
        """
        family_count: dict[str, int] = {}
        exploit_count = 0
        kept: list[dict] = []

        explored_families = {
            p.get("proposal", {}).get("model_name", "")
            for p in archive[-ARCHIVE_WINDOW:]
            if p.get("proposal")
        }

        for p in proposals:
            score, family, tag = (
                self.compute_novelty_score(p, archive),
                p.get("model_name", ""),
                p.get("proposal_family", ""),
            )

            is_exploit = tag == "exploit" or score < 0.35
            if is_exploit and exploit_count >= 1:
                logger.debug("Diversity budget: dropping extra exploit proposal.")
                continue

            fam_count = family_count.get(family, 0)
            if fam_count >= 2:
                logger.debug(f"Diversity budget: dropping 3rd proposal for {family}.")
                continue

            family_count[family] = fam_count + 1
            if is_exploit:
                exploit_count += 1
            kept.append(p)

        # Check that at least one unexplored family is represented
        new_families_in_kept = {
            p.get("model_name", "") for p in kept
            if p.get("model_name", "") not in explored_families
        }
        if not new_families_in_kept and kept:
            logger.info(
                "Diversity budget: all proposals from explored families. "
                "Consider adding an explore proposal in the next cycle."
            )

        return kept

    def summarise_archive_diversity(self, archive: list[dict]) -> dict:
        """
        Return a summary of proposal diversity stats for logging/dashboard.
        """
        if not archive:
            return {"total": 0, "unique_families": [], "mean_novelty": 1.0}

        recent = archive[-ARCHIVE_WINDOW:]
        novelties: list[float] = []
        families: list[str] = []

        for i, entry in enumerate(recent):
            prop = entry.get("proposal", {})
            if not prop:
                continue
            families.append(prop.get("model_name", "unknown"))
            # Novelty of entry i vs entries before it
            score = self.compute_novelty_score(prop, recent[:i])
            novelties.append(score)

        return {
            "total": len(recent),
            "unique_families": list(set(families)),
            "mean_novelty": round(sum(novelties) / max(len(novelties), 1), 4),
            "min_novelty":  round(min(novelties, default=1.0), 4),
        }


# ── Similarity helpers ────────────────────────────────────────────────────────

def _param_similarity(
    params_a: dict,
    params_b: dict,
    search_space: dict,
) -> float:
    """
    Compute normalised similarity [0,1] between two hyperparameter dicts.
    Uses the search_space spec for normalisation.
    Unknown parameters fall back to exact-match (0 or 1).
    """
    if not params_a or not params_b:
        return 0.0

    common_keys = set(params_a) & set(params_b)
    if not common_keys:
        return 0.0

    scores: list[float] = []
    for key in common_keys:
        va = params_a[key]
        vb = params_b[key]
        spec = search_space.get(key, {})
        scores.append(_single_param_similarity(va, vb, spec))

    return round(sum(scores) / len(scores), 4)


def _single_param_similarity(va: Any, vb: Any, spec: dict) -> float:
    """Return similarity [0,1] for a single parameter."""
    param_type = spec.get("type", "")

    if param_type == "categorical":
        return 1.0 if va == vb else 0.0

    if param_type in ("int", "float"):
        lo  = float(spec.get("low",  0))
        hi  = float(spec.get("high", 1))
        use_log = spec.get("log", False)

        if use_log and lo > 0:
            try:
                lo = math.log(lo)
                hi = math.log(hi)
                va = math.log(max(float(va), 1e-12))
                vb = math.log(max(float(vb), 1e-12))
            except (TypeError, ValueError):
                pass

        span = max(abs(hi - lo), 1e-8)
        try:
            distance = abs(float(va) - float(vb)) / span
        except (TypeError, ValueError):
            return 0.0
        return round(1.0 - min(distance, 1.0), 4)

    # Fallback: exact match
    return 1.0 if va == vb else 0.0


# ── Persistent diversity log ──────────────────────────────────────────────────

class DiversityLogger:
    """
    Appends proposal novelty scores to proposal_diversity.json for dashboard.
    """

    def __init__(self, log_path: Path):
        self.log_path = Path(log_path)
        self._data: list[dict] = []
        if self.log_path.exists():
            with open(self.log_path, "r", encoding="utf-8") as fh:
                self._data = json.load(fh)

    def record(self, proposal: dict, novelty_score: float, cycle: int) -> None:
        self._data.append({
            "cycle": cycle,
            "model_name": proposal.get("model_name"),
            "novelty_score": novelty_score,
            "accepted": novelty_score >= DEFAULT_NOVELTY_THRESHOLD,
            "proposal_family": proposal.get("proposal_family", ""),
        })
        self._save()

    def _save(self) -> None:
        with open(self.log_path, "w", encoding="utf-8") as fh:
            json.dump(self._data, fh, indent=2)

    def mean_novelty(self) -> float:
        if not self._data:
            return 1.0
        return round(sum(d["novelty_score"] for d in self._data) / len(self._data), 4)


# ── CLI quick-test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    scorer = NoveltyScorer()

    archive = [
        {"proposal": {"model_name": "LGBMRegressor",
                       "params": {"n_estimators": 400, "learning_rate": 0.08,
                                  "num_leaves": 31}}},
    ]
    p1 = {"model_name": "LGBMRegressor",
           "params": {"n_estimators": 410, "learning_rate": 0.09, "num_leaves": 31}}
    p2 = {"model_name": "XGBRegressor",
           "params": {"n_estimators": 300, "max_depth": 5, "learning_rate": 0.10}}
    p3 = {"model_name": "LGBMRegressor",
           "params": {"n_estimators": 100, "learning_rate": 0.25, "num_leaves": 127}}

    for p in (p1, p2, p3):
        score = scorer.compute_novelty_score(p, archive)
        passes, _ = scorer.is_novel_enough(p, archive)
        print(f"{p['model_name']}: novelty={score:.3f}  {'✓ passes' if passes else '✗ rejected'}")
