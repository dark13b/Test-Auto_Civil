"""
validator_calibrator.py — AutoCivil Track B
============================================
Computes data-driven threshold suggestions for validator.py using
the actual dataset statistics, replacing hard-coded engineering judgments.

Usage (in generate_data.py, after dataset loading):
    from validator_calibrator import ValidatorCalibrator
    cal = ValidatorCalibrator()
    calibrated = cal.calibrate_from_dataset(dataset, config)
    log_status(f"Calibrated validator thresholds: {calibrated}")
    # Update config in-memory (not written to disk — user still controls config.yaml)
    for k, v in calibrated.items():
        config["validator"][k] = v

All calibrated thresholds are ADVISORY unless manually committed to config.yaml.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# ── Constants ─────────────────────────────────────────────────────────────────

# Minimum number of samples required on each side of a threshold candidate
MIN_SAMPLES_PER_SIDE = 5

# Number of threshold candidates to evaluate in binary search
N_THRESHOLD_CANDIDATES = 50

# Hard physical limits (from domain knowledge — never relax below/above these)
PHYSICAL_LIMITS = {
    "water_cement_ratio_min":          0.20,   # Powers (1960) minimum
    "water_cement_ratio_max":          0.90,   # ACI 211.1
    "total_binder_min":                150.0,  # ACI 211.1 Table 6.3.3
    "fly_ash_replacement_ratio_max":   0.50,   # ACI 232.2R-18
    "slag_replacement_ratio_max":      0.70,   # ACI 233R-03
}


# ── Main class ────────────────────────────────────────────────────────────────

class ValidatorCalibrator:
    """
    Calibrates validator thresholds from dataset statistics.

    The calibration finds threshold values that maximally separate
    'anomalous' samples (high residual vs. a simple w/c model) from
    'normal' samples.
    """

    def calibrate_from_dataset(
        self,
        dataset,   # pd.DataFrame
        config: dict,
    ) -> dict[str, float]:
        """
        Compute calibrated validator thresholds for the given dataset.

        Returns:
            dict mapping threshold_name → recommended_float_value
            (keys match those in config["validator"])
        """
        try:
            import pandas as pd
        except ImportError:
            logger.error("ValidatorCalibrator: pandas not available.")
            return {}

        enriched = self._enrich(dataset, config)
        target   = dataset[config["task"]["target_column"]]
        residuals = self._compute_binder_residuals(enriched, target)

        suggestions: dict[str, float] = {}

        # 1. total_binder low-warn threshold
        try:
            total_binder_thresh = self._find_threshold(
                enriched["total_binder"], residuals, direction="below"
            )
            # Clamp: never below physical minimum
            suggestions["total_binder_low_warn"] = max(
                total_binder_thresh, PHYSICAL_LIMITS["total_binder_min"]
            )
        except Exception as exc:
            logger.warning(f"ValidatorCalibrator: total_binder calibration failed — {exc}")

        # 2. Suspicious w/c ratio (high end)
        try:
            wc_col = enriched.get("water_cement_ratio", None)
            if wc_col is not None:
                wc_90th = float(wc_col.quantile(0.90))
                suggestions["suspicious_water_cement_ratio"] = min(
                    wc_90th, PHYSICAL_LIMITS["water_cement_ratio_max"]
                )
        except Exception as exc:
            logger.warning(f"ValidatorCalibrator: w/c calibration failed — {exc}")

        # 3. Fly ash replacement ratio warn
        try:
            fa_col = enriched.get("fly_ash_replacement_ratio", None)
            if fa_col is not None:
                fa_90th = float(fa_col.quantile(0.90))
                suggestions["fly_ash_replacement_warn"] = min(
                    fa_90th + 0.05,
                    PHYSICAL_LIMITS["fly_ash_replacement_ratio_max"],
                )
        except Exception as exc:
            logger.warning(f"ValidatorCalibrator: fly_ash calibration failed — {exc}")

        # 4. Superplasticizer maximum dose
        try:
            sp_col = dataset.get("superplasticizer", None)
            if sp_col is not None:
                sp_99th = float(sp_col.quantile(0.99))
                suggestions["superplasticizer_max_warn"] = round(
                    max(sp_99th * 1.10, 32.0), 1   # 10% above max observed, min 32
                )
        except Exception as exc:
            logger.warning(f"ValidatorCalibrator: superplasticizer calibration failed — {exc}")

        # 5. Strength plausibility bounds
        try:
            target_series = dataset[config["task"]["target_column"]]
            suggestions["strength_min_plausible"] = max(
                round(float(target_series.quantile(0.005)), 1), 5.0
            )
            suggestions["strength_max_plausible"] = round(
                float(target_series.quantile(0.995)) * 1.20, 1
            )
        except Exception as exc:
            logger.warning(f"ValidatorCalibrator: strength bounds calibration failed — {exc}")

        # 6. Age range
        try:
            age_col = dataset.get("age", None)
            if age_col is not None:
                suggestions["age_max_warn"] = int(float(age_col.quantile(0.99)) * 1.5)
        except Exception as exc:
            logger.warning(f"ValidatorCalibrator: age calibration failed — {exc}")

        logger.info(f"ValidatorCalibrator: calibrated {len(suggestions)} thresholds.")
        return suggestions

    def format_calibration_report(
        self, suggestions: dict[str, float], original_config: dict
    ) -> str:
        """Return a human-readable diff of calibrated vs original thresholds."""
        original = original_config.get("validator", {})
        lines = ["=== Validator Threshold Calibration Report ==="]
        for key, new_val in suggestions.items():
            old_val = original.get(key, "not set")
            changed = str(old_val) != str(round(new_val, 4))
            marker = "🔄" if changed else "  "
            lines.append(f"  {marker} {key}: {old_val} → {new_val:.4g}")
        if not suggestions:
            lines.append("  No thresholds calibrated (dataset may be empty).")
        lines.append("  Note: changes are in-memory only; update config.yaml to persist.")
        return "\n".join(lines)

    # ── Private ───────────────────────────────────────────────────────────────

    def _enrich(self, dataset, config: dict) -> Any:
        """
        Attempt to call the project's build_engineering_features() if available,
        otherwise compute the minimum set needed for calibration.
        """
        try:
            from feature_engineering import build_engineering_features
            return build_engineering_features(dataset, config)
        except ImportError:
            pass

        # Fallback: compute key derived columns inline
        try:
            import pandas as pd
            df = dataset.copy()
            cement   = df.get("cement",   pd.Series(0, index=df.index))
            slag     = df.get("slag",     pd.Series(0, index=df.index))
            fly_ash  = df.get("fly_ash",  pd.Series(0, index=df.index))
            water    = df.get("water",    pd.Series(1, index=df.index))

            df["total_binder"]             = cement + slag + fly_ash
            df["water_cement_ratio"]       = water / (cement + 1e-6)
            df["fly_ash_replacement_ratio"] = fly_ash / (df["total_binder"] + 1e-6)
            df["slag_replacement_ratio"]    = slag   / (df["total_binder"] + 1e-6)
            return df
        except Exception as exc:
            logger.warning(f"ValidatorCalibrator: enrichment failed — {exc}")
            return dataset

    def _compute_binder_residuals(self, enriched, target) -> Any:
        """
        Compute residuals from a simple Ridge regression on water_cement_ratio.
        Residual = actual strength − predicted by w/c alone.
        High |residual| = the mix doesn't behave as expected from w/c.
        """
        try:
            from sklearn.linear_model import Ridge
            wc  = enriched[["water_cement_ratio"]].fillna(0.5).values
            tgt = target.values
            ridge = Ridge(alpha=1.0).fit(wc, tgt)
            return target - ridge.predict(wc)
        except Exception:
            # Fallback: residual vs. mean
            mean = float(target.mean())
            return target - mean

    def _find_threshold(
        self,
        series,
        residuals,
        direction: str = "below",
    ) -> float:
        """
        Binary search for the threshold value that maximally separates
        the mean residuals of two groups (above/below threshold).

        Args:
            series:    the feature Series (e.g. total_binder)
            residuals: residual Series from _compute_binder_residuals
            direction: 'below' → flag when below threshold;
                       'above' → flag when above threshold

        Returns:
            Best threshold float.
        """
        try:
            q05 = float(series.quantile(0.05))
            q95 = float(series.quantile(0.95))
        except Exception:
            return float(series.mean())

        candidates = np.linspace(q05, q95, N_THRESHOLD_CANDIDATES)
        best_t, best_sep = float(candidates[len(candidates) // 2]), 0.0

        for t in candidates:
            if direction == "below":
                mask = series < t
            else:
                mask = series > t

            n_in  = int(mask.sum())
            n_out = int((~mask).sum())
            if n_in < MIN_SAMPLES_PER_SIDE or n_out < MIN_SAMPLES_PER_SIDE:
                continue

            sep = abs(
                float(residuals[mask].mean()) - float(residuals[~mask].mean())
            )
            if sep > best_sep:
                best_sep = sep
                best_t   = float(t)

        return round(best_t, 2)


# ── CLI quick-test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json

    try:
        import pandas as pd
        import numpy as np

        np.random.seed(42)
        n = 500
        cement  = np.random.uniform(150, 400, n)
        slag    = np.random.uniform(0, 150, n)
        fly_ash = np.random.uniform(0, 100, n)
        water   = np.random.uniform(140, 220, n)
        dummy   = pd.DataFrame({
            "cement": cement, "slag": slag,
            "fly_ash": fly_ash, "water": water,
            "superplasticizer": np.random.uniform(0, 15, n),
            "age": np.random.choice([7, 14, 28, 56, 90], n),
            "compressive_strength": (
                40 + 0.05 * cement - 0.08 * water
                + 0.03 * slag + np.random.normal(0, 5, n)
            ),
        })
        config = {"task": {"target_column": "compressive_strength"}, "validator": {}}
        cal = ValidatorCalibrator()
        suggestions = cal.calibrate_from_dataset(dummy, config)
        print(json.dumps(suggestions, indent=2))
        print(cal.format_calibration_report(suggestions, config))
    except ImportError as e:
        print(f"Skipping CLI test: {e}")
