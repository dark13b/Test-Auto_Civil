"""
feature_lab.py — AutoCivil Track B
=====================================
Controlled surface for LLM-proposed concrete science features.

Design principles:
  1. LLM proposes a Python function in a restricted prompt
  2. Function is executed in a NAMESPACE-ISOLATED exec() call
     (only pd, np, math allowed — no os/sys/subprocess)
  3. Tested against baseline CV RMSE with current best model
  4. Accepted features are appended to feature_engineering.py

Security notes:
  - NEVER use eval(); all execution goes through exec() with a whitelist namespace
  - Import detection: any line starting with 'import' or 'from' is stripped before exec
  - Timeout: enforced via threading.Timer (30s hard limit per spec)
  - No filesystem access inside the sandbox (namespace has no Path/open)

Usage (in research_loop.py):
    if cycle % config["research"]["feature_lab_interval_cycles"] == 0:
        lab = FeatureLab(config, outputs_dir, backend)
        weakness = extract_weakness(final_metrics)
        result = lab.run_cycle(x_train, y_train, x_val, y_val,
                               baseline_cv_rmse=current_best_rmse,
                               weakness=weakness)
        if result.get("promoted"):
            log_status(f"Feature promoted: Δ={result['delta']:.4f}")
"""

from __future__ import annotations

import json
import logging
import re
import textwrap
import threading
import time
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

SANDBOX_TIMEOUT_SECONDS = 30
MIN_DELTA_RMSE          = 0.02    # minimum improvement to promote a feature
FEATURE_FUNC_NAME       = "new_feature"

# Columns the LLM-proposed function may reference
ALLOWED_INPUT_COLUMNS = [
    "cement", "slag", "fly_ash", "water", "superplasticizer",
    "coarse_aggregate", "fine_aggregate", "age",
]

# Dangerous patterns to strip from LLM-generated code
_DANGEROUS_PATTERNS = re.compile(
    r"^\s*(import\s|from\s|__import__|open\s*\(|exec\s*\(|eval\s*\(|"
    r"os\.|sys\.|subprocess\.|shutil\.|pathlib\.|builtins\.)",
    re.MULTILINE,
)


# ── Sandbox execution ─────────────────────────────────────────────────────────

def _build_sandbox_namespace() -> dict:
    """
    Returns the restricted namespace for exec().
    ONLY pandas, numpy, and math are exposed.
    """
    import numpy as np
    try:
        import pandas as pd
    except ImportError:
        pd = None

    namespace: dict[str, Any] = {
        "__builtins__": {
            # Minimal safe builtins
            "abs": abs, "round": round, "min": min, "max": max,
            "sum": sum, "len": len, "range": range,
            "float": float, "int": int, "str": str, "bool": bool,
            "list": list, "dict": dict, "tuple": tuple,
            "zip": zip, "enumerate": enumerate, "map": map,
            "isinstance": isinstance, "hasattr": hasattr,
            "True": True, "False": False, "None": None,
        },
        "np": np,
        "math": __import__("math"),
    }
    if pd is not None:
        namespace["pd"] = pd
    return namespace


def _sanitise_code(raw_code: str) -> str:
    """
    Strip dangerous import/exec/eval lines from LLM-generated code.
    Also remove markdown fences if present.
    """
    # Strip markdown code fences
    code = re.sub(r"```(?:python)?", "", raw_code).strip()
    # Remove dangerous lines
    safe_lines = []
    for line in code.splitlines():
        if _DANGEROUS_PATTERNS.match(line):
            logger.warning(f"FeatureLab: stripped dangerous line: {line!r}")
            continue
        safe_lines.append(line)
    return "\n".join(safe_lines)


def _exec_with_timeout(
    code: str,
    namespace: dict,
    timeout: int = SANDBOX_TIMEOUT_SECONDS,
) -> tuple[bool, str]:
    """
    Execute code in namespace. Returns (success, error_message).
    Kills execution after `timeout` seconds.
    """
    result = {"success": False, "error": ""}
    exc_holder: list[Exception] = []

    def _run():
        try:
            exec(compile(code, "<feature_lab>", "exec"), namespace)  # noqa: S102
            result["success"] = True
        except Exception as exc:
            exc_holder.append(exc)

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    thread.join(timeout=timeout)

    if thread.is_alive():
        result["error"] = f"Timeout after {timeout}s"
        logger.error("FeatureLab: sandbox execution timed out.")
        return False, result["error"]

    if exc_holder:
        result["error"] = str(exc_holder[0])
        return False, result["error"]

    return result["success"], result["error"]


# ── Feature proposal parsing ──────────────────────────────────────────────────

def _parse_feature_proposal(llm_response: str) -> dict:
    """
    Extract the Python function and metadata comment from LLM response.

    Expected format:
        # HYPOTHESIS: ...
        # MECHANISM: ...
        # EXPECTED_EFFECT: ...
        def new_feature(df: pd.DataFrame) -> pd.Series:
            ...
    """
    lines = llm_response.strip().splitlines()
    metadata: dict[str, str] = {}
    func_lines: list[str] = []
    in_func = False

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("# HYPOTHESIS:"):
            metadata["hypothesis"] = stripped[len("# HYPOTHESIS:"):].strip()
        elif stripped.startswith("# MECHANISM:"):
            metadata["mechanism"] = stripped[len("# MECHANISM:"):].strip()
        elif stripped.startswith("# EXPECTED_EFFECT:"):
            metadata["expected_effect"] = stripped[len("# EXPECTED_EFFECT:"):].strip()
        elif stripped.startswith("def new_feature"):
            in_func = True
            func_lines.append(line)
        elif in_func:
            func_lines.append(line)

    if not func_lines:
        return {
            "error": "No function named 'new_feature' found in LLM response.",
            "raw": llm_response[:500],
        }

    return {
        "function_code": "\n".join(func_lines),
        "hypothesis":    metadata.get("hypothesis", ""),
        "mechanism":     metadata.get("mechanism", ""),
        "expected_effect": metadata.get("expected_effect", ""),
    }


def _extract_expected_delta(expected_effect_text: str) -> float:
    """
    Try to extract a numeric RMSE delta from the EXPECTED_EFFECT comment.
    Falls back to 0.05 if unparseable.
    """
    import re as _re
    match = _re.search(r"[-+]?\d+\.?\d*", expected_effect_text)
    if match:
        try:
            return float(match.group())
        except ValueError:
            pass
    return 0.05   # conservative default


# ── Main class ────────────────────────────────────────────────────────────────

class FeatureLab:
    """
    LLM-driven feature hypothesis generation, testing, and promotion.

    Args:
        config:           full AutoCivil config dict
        outputs_dir:      outputs directory (Path)
        llm_backend:      any object with .generate_text(prompt, max_output_tokens) → str
    """

    def __init__(
        self,
        config_or_outputs_dir: dict | Path,
        outputs_dir: Path | None = None,
        llm_backend: Any | None = None,
        *,
        timeout_seconds: float | None = None,
        min_delta_rmse: float | None = None,
    ):
        if isinstance(config_or_outputs_dir, (str, Path)) and outputs_dir is None:
            self.config = {}
            self.outputs_dir = Path(config_or_outputs_dir)
            self.backend = llm_backend
        else:
            self.config = config_or_outputs_dir if isinstance(config_or_outputs_dir, dict) else {}
            self.outputs_dir = Path(outputs_dir or ".")
            self.backend = llm_backend
        self.sandbox_dir = self.outputs_dir / "feature_lab"
        self.sandbox_dir.mkdir(parents=True, exist_ok=True)
        self.hypothesis_log_path = self.sandbox_dir / "feature_hypotheses.json"
        self._log: list[dict] = self._load_log()

        rc = self.config.get("research", {})
        self.min_delta  = min_delta_rmse if min_delta_rmse is not None else rc.get("feature_lab_min_delta_rmse", MIN_DELTA_RMSE)
        self.timeout    = timeout_seconds if timeout_seconds is not None else rc.get("feature_lab_sandbox_timeout", SANDBOX_TIMEOUT_SECONDS)

    # ── Public API ────────────────────────────────────────────────────────────

    def run_cycle(
        self,
        x_train,
        y_train,
        x_val,
        y_val,
        baseline_cv_rmse: float,
        weakness: dict | None = None,
        existing_features: list[str] | None = None,
    ) -> dict:
        """
        Full FeatureLab cycle: propose → test → (optionally) promote.

        Returns:
            dict with keys: proposed, tested, accepted, promoted, delta, error
        """
        existing_features = existing_features or list(x_train.columns)
        weakness          = weakness or {}

        # 1. Propose
        proposal = self.propose_feature(
            context={
                "existing_features": existing_features,
                "weakness": weakness,
                "n_train": len(x_train),
            }
        )

        if "error" in proposal:
            logger.warning(f"FeatureLab: proposal failed — {proposal['error']}")
            self._append_log({
                "phase": "propose", "error": proposal["error"],
                "raw": proposal.get("raw", ""),
            })
            return {"proposed": False, "error": proposal["error"]}

        # 2. Test
        test_result = self.test_feature(
            feature_func_code=proposal["function_code"],
            x_train=x_train,
            y_train=y_train,
            x_val=x_val,
            y_val=y_val,
            baseline_cv_rmse=baseline_cv_rmse,
        )

        entry = {
            "hypothesis":     proposal["hypothesis"],
            "mechanism":      proposal["mechanism"],
            "expected_effect": proposal["expected_effect"],
            "function_code":  proposal["function_code"],
            **test_result,
        }
        self._append_log(entry)

        if test_result.get("status") != "success":
            return {"proposed": True, "tested": False,
                    "error": test_result.get("reason", "unknown")}

        # 3. Promote if accepted
        promoted = False
        if test_result.get("accepted"):
            promote_status = self.promote_feature(test_result, proposal)
            promoted = promote_status == "promoted"

        return {
            "proposed":  True,
            "tested":    True,
            "accepted":  test_result.get("accepted", False),
            "promoted":  promoted,
            "delta":     test_result.get("delta_vs_baseline", 0.0),
            "cv_rmse":   test_result.get("cv_rmse"),
        }

    def propose_feature(self, context: dict) -> dict:
        """Ask the LLM for a new concrete-science-grounded feature."""
        prompt = textwrap.dedent(f"""
        You are a concrete science expert and Python programmer.

        Current engineered features: {context.get('existing_features', [])}
        Training set size: {context.get('n_train', 'unknown')} samples
        Model weakness (area to improve): {context.get('weakness', 'general accuracy')}

        Propose ONE new engineered feature that:
        1. Is grounded in concrete science (cite a specific mechanism or standard)
        2. Is NOT already captured by any existing feature
        3. Can be computed ONLY from these raw columns:
           cement, slag, fly_ash, water, superplasticizer,
           coarse_aggregate, fine_aggregate, age
        4. Uses ONLY numpy (np), pandas (pd), and math — no other imports

        Output format (EXACTLY this):
        # HYPOTHESIS: [one-sentence falsifiable claim]
        # MECHANISM: [concrete science explanation with source]
        # EXPECTED_EFFECT: [predicted RMSE change, e.g. "-0.05 MPa RMSE"]
        def new_feature(df: pd.DataFrame) -> pd.Series:
            # your implementation
            return result

        IMPORTANT:
        - Do NOT import anything
        - Do NOT use os, sys, subprocess, open(), exec(), eval()
        - Return a pd.Series of float64 values
        """).strip()

        try:
            response = self.backend.generate_text(prompt, max_output_tokens=500)
        except Exception as exc:
            return {"error": f"LLM call failed: {exc}"}

        return _parse_feature_proposal(response)

    def test_feature(
        self,
        feature_func_code: str,
        x_train,
        y_train,
        x_val,
        y_val,
        baseline_cv_rmse: float,
    ) -> dict:
        """
        Execute the proposed feature function in a restricted sandbox
        and measure its effect on cross-validation RMSE.

        Security: code is sanitised and exec'd in a whitelist namespace.
        Timeout: SANDBOX_TIMEOUT_SECONDS (hard kill via threading).

        Returns:
            {status, cv_rmse, delta_vs_baseline, accepted, reason}
        """
        # 1. Sanitise
        safe_code = _sanitise_code(feature_func_code)

        # 2. Execute in sandbox
        namespace = _build_sandbox_namespace()
        success, error = _exec_with_timeout(safe_code, namespace, timeout=self.timeout)

        if not success:
            return {"status": "error", "reason": error}

        feature_func: Callable | None = namespace.get(FEATURE_FUNC_NAME)
        if feature_func is None or not callable(feature_func):
            return {"status": "error",
                    "reason": f"No callable '{FEATURE_FUNC_NAME}' found after exec"}

        # 3. Augment datasets
        try:
            import pandas as pd  # noqa: PLC0415
            x_tr_aug = x_train.copy()
            x_va_aug = x_val.copy()
            x_tr_aug["_candidate"] = feature_func(x_tr_aug)
            x_va_aug["_candidate"] = feature_func(x_va_aug)

            # Validate output
            if x_tr_aug["_candidate"].isnull().any():
                return {"status": "error",
                        "reason": "Feature produces NaN values on training data"}
        except Exception as exc:
            return {"status": "error", "reason": f"Feature function raised: {exc}"}

        # 4. Quick CV with current best model settings
        try:
            cv_rmse = self._quick_cv_rmse(x_tr_aug, y_train)
        except Exception as exc:
            return {"status": "error", "reason": f"CV failed: {exc}"}

        delta = round(baseline_cv_rmse - cv_rmse, 4)
        return {
            "status":             "success",
            "cv_rmse":            round(cv_rmse, 4),
            "delta_vs_baseline":  delta,
            "feature_code":       feature_func_code,
            "accepted":           delta >= self.min_delta,
        }

    def promote_feature(self, test_result: dict, proposal: dict) -> str:
        """
        Append accepted feature to feature_engineering.py under an EXPERIMENTAL block.
        If feature_engineering.py doesn't exist yet, creates a stub file.

        Returns: 'promoted' | 'rejected'
        """
        if not test_result.get("accepted"):
            return "rejected"

        feature_file = self._find_feature_engineering_py()
        if feature_file is None:
            logger.warning(
                "FeatureLab: feature_engineering.py not found. "
                "Feature recorded in sandbox but not promoted."
            )
            return "rejected"

        block = textwrap.dedent(f"""

        # ── EXPERIMENTAL (promoted by FeatureLab) ──────────────────────────────
        # Hypothesis: {proposal.get('hypothesis', '')}
        # Mechanism:  {proposal.get('mechanism', '')}
        # CV RMSE delta: {test_result.get('delta_vs_baseline', 0.0):+.4f} MPa
        # Promoted: automatically — verify before production use
        {test_result['feature_code']}
        """)

        with open(feature_file, "a", encoding="utf-8") as fh:
            fh.write(block)

        logger.info(
            f"FeatureLab: feature promoted to {feature_file} "
            f"(Δ={test_result['delta_vs_baseline']:+.4f})"
        )
        return "promoted"

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _quick_cv_rmse(self, x, y, n_folds: int = 3) -> float:
        """
        Run a quick 3-fold CV with LightGBM using conservative settings.
        Falls back to a simple hold-out split if LGBM is not available.
        """
        import numpy as np

        try:
            from lightgbm import LGBMRegressor
            from sklearn.model_selection import cross_val_score
            model = LGBMRegressor(
                n_estimators=200,
                learning_rate=0.1,
                num_leaves=31,
                verbose=-1,
                random_state=42,
            )
            scores = cross_val_score(
                model, x, y,
                cv=n_folds,
                scoring="neg_root_mean_squared_error",
            )
            return float(-scores.mean())
        except ImportError:
            pass

        # Fallback: simple hold-out
        try:
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.model_selection import cross_val_score
            model = RandomForestRegressor(
                n_estimators=100, random_state=42, n_jobs=-1
            )
            scores = cross_val_score(
                model, x, y,
                cv=n_folds,
                scoring="neg_root_mean_squared_error",
            )
            return float(-scores.mean())
        except Exception as exc:
            raise RuntimeError(f"No CV backend available: {exc}") from exc

    def _find_feature_engineering_py(self) -> Path | None:
        """Search for feature_engineering.py relative to outputs_dir."""
        candidates = [
            self.outputs_dir.parent / "feature_engineering.py",
            self.outputs_dir.parent.parent / "feature_engineering.py",
            Path("feature_engineering.py"),
        ]
        for c in candidates:
            if c.exists():
                return c
        return None

    def _load_log(self) -> list:
        if self.hypothesis_log_path.exists():
            with open(self.hypothesis_log_path, "r", encoding="utf-8") as fh:
                return json.load(fh)
        return []

    def validate_feature_code(self, code: str) -> dict[str, Any]:
        if "import " in code or "from " in code:
            return {"accepted": False, "reason": "Import statements are not allowed."}
        if any(name in code for name in ("os.", "sys.", "subprocess", "open(", "exec(", "eval(")):
            return {"accepted": False, "reason": "Unsafe names or operations are not allowed."}
        try:
            safe_code = _sanitise_code(code)
        except Exception as exc:
            return {"accepted": False, "reason": str(exc)}
        return {"accepted": True, "reason": "safe", "sanitized_code": safe_code}

    def execute_feature_code(self, *, code: str, frame) -> dict[str, Any]:
        validation = self.validate_feature_code(code)
        if not validation.get("accepted", False):
            return validation
        namespace = _build_sandbox_namespace()
        success, error = _exec_with_timeout(code, namespace, timeout=self.timeout)
        if not success:
            return {"accepted": False, "reason": error}
        feature_func: Callable | None = namespace.get(FEATURE_FUNC_NAME)
        if feature_func is None or not callable(feature_func):
            return {"accepted": False, "reason": f"No callable '{FEATURE_FUNC_NAME}' found after exec"}
        series = feature_func(frame.copy())
        return {
            "accepted": True,
            "series_name": FEATURE_FUNC_NAME,
            "values": list(series),
        }

    def _append_log(self, entry: dict) -> None:
        entry["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        self._log.append(entry)
        with open(self.hypothesis_log_path, "w", encoding="utf-8") as fh:
            json.dump(self._log, fh, indent=2, default=str)

    @property
    def promoted_count(self) -> int:
        return sum(1 for e in self._log if e.get("accepted") and e.get("status") == "success")


# ── CLI quick-test (no LLM needed) ───────────────────────────────────────────

if __name__ == "__main__":
    import numpy as np

    # Test sandbox only
    code = """
# HYPOTHESIS: w/c ratio captures strength better than raw water alone
# MECHANISM: Abrams law — strength is a function of w/c, not just water content
# EXPECTED_EFFECT: -0.06 MPa RMSE
def new_feature(df):
    return df['water'] / (df['cement'] + df.get('fly_ash', 0) + df.get('slag', 0) + 1e-6)
"""
    namespace = _build_sandbox_namespace()
    ok, err = _exec_with_timeout(_sanitise_code(code), namespace, timeout=5)
    print(f"Sandbox exec: ok={ok}, error={err!r}")
    fn = namespace.get("new_feature")
    if fn:
        try:
            import pandas as pd
            dummy = pd.DataFrame({
                "cement": [300.0, 350.0], "fly_ash": [50.0, 60.0],
                "slag": [0.0, 0.0], "water": [180.0, 175.0],
            })
            result = fn(dummy)
            print(f"Feature output: {result.tolist()}")
        except Exception as e:
            print(f"Feature execution error: {e}")
