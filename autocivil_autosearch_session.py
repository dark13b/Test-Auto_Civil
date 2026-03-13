"""Run an AutoCivil-Lab search session with periodic status snapshots."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent
OUTPUTS_DIR = REPO_ROOT / "outputs"
SEARCH_LOG_PATH = OUTPUTS_DIR / "search_run.log"
SESSION_LOG_PATH = OUTPUTS_DIR / "autosearch_session.log"
REPORT_LOG_PATH = OUTPUTS_DIR / "five_min_updates.log"
FINAL_SUMMARY_PATH = OUTPUTS_DIR / "final_search_summary.json"
OVERRIDES_PATH = OUTPUTS_DIR / "runtime_search_overrides.json"
LOCK_PATH = REPO_ROOT / ".autosearch_session.lock"


def _log(message: str) -> None:
    timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
    line = f"[{timestamp}] {message}"
    print(line)
    SESSION_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with SESSION_LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def _clear_outputs(outputs_dir: Path) -> None:
    outputs_dir.mkdir(parents=True, exist_ok=True)
    for child in outputs_dir.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


def _archive_existing_outputs(outputs_dir: Path) -> Path | None:
    """Archive the current outputs folder before starting a fresh session."""
    if not outputs_dir.exists():
        return None
    if not any(outputs_dir.iterdir()):
        return None

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_dir = REPO_ROOT / f"outputs_archive_{timestamp}"
    shutil.copytree(outputs_dir, archive_dir)
    return archive_dir


def _pid_is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _acquire_session_lock() -> None:
    if LOCK_PATH.exists():
        try:
            payload = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
            existing_pid = int(payload.get("pid", 0))
        except Exception:
            existing_pid = 0
        if existing_pid and existing_pid != os.getpid() and _pid_is_alive(existing_pid):
            raise RuntimeError(f"Another autosearch session is already running with PID {existing_pid}.")
        LOCK_PATH.unlink(missing_ok=True)

    LOCK_PATH.write_text(
        json.dumps(
            {
                "pid": os.getpid(),
                "started_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def _release_session_lock() -> None:
    if not LOCK_PATH.exists():
        return
    try:
        payload = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
        locked_pid = int(payload.get("pid", 0))
    except Exception:
        locked_pid = 0
    if locked_pid in {0, os.getpid()}:
        LOCK_PATH.unlink(missing_ok=True)


def _load_base_config() -> dict[str, Any]:
    import train

    return copy.deepcopy(train.load_config())


def _apply_runtime_overrides(
    base_config: dict[str, Any],
    *,
    input_path: Path,
    min_runtime_minutes: float,
    target_trials: int,
) -> dict[str, Any]:
    config = copy.deepcopy(base_config)
    config["data"]["mode"] = "local_file"
    config["data"]["local_file"]["path"] = str(input_path)
    config["research"]["max_runtime_minutes"] = max(0.0, float(min_runtime_minutes))
    config["research"]["max_cycles"] = int(target_trials)
    return config


def _run_baseline(config: dict[str, Any]) -> None:
    import train

    train.load_config = lambda: config
    exit_code = train.main()
    if exit_code != 0:
        raise RuntimeError(f"Baseline training failed with exit code {exit_code}.")


def _count_trials(csv_path: Path) -> int:
    if not csv_path.exists():
        return 0
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        return max(sum(1 for _ in csv.DictReader(handle)), 0)


def _read_best_summary(outputs_dir: Path) -> dict[str, Any]:
    final_metrics_path = outputs_dir / "final_metrics.json"
    if final_metrics_path.exists():
        with final_metrics_path.open("r", encoding="utf-8") as handle:
            final_metrics = json.load(handle)
        best_search_metrics = final_metrics.get("best_search_metrics")
        if isinstance(best_search_metrics, dict):
            data = best_search_metrics
        else:
            data = {}
    else:
        best_path = outputs_dir / "search_state_best_result.json"
        if not best_path.exists():
            return {}
        with best_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    return {
        "model_name": data.get("model_name"),
        "composite_score": data.get("composite_score"),
        "trial_number": data.get("trial_number"),
        "source": data.get("source"),
        "validation_verdict": data.get("validation_verdict"),
    }


def _read_last_research_log(outputs_dir: Path) -> str | None:
    log_path = outputs_dir / "research_log.txt"
    if not log_path.exists():
        return None
    with log_path.open("r", encoding="utf-8") as handle:
        lines = [line.strip() for line in handle.readlines() if line.strip()]
    return None if not lines else lines[-1]


def _append_report(status: str, outputs_dir: Path) -> None:
    best = _read_best_summary(outputs_dir)
    research_results_path = outputs_dir / "research_results.csv"
    trial_count_path = research_results_path if research_results_path.exists() else outputs_dir / "optuna_results.csv"
    summary = {
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        "status": status,
        "trial_count": _count_trials(trial_count_path),
        "best_model": best.get("model_name"),
        "best_composite_score": best.get("composite_score"),
        "best_trial_number": best.get("trial_number"),
        "best_source": best.get("source"),
        "validation_verdict": best.get("validation_verdict"),
        "last_research_log": _read_last_research_log(outputs_dir),
    }
    with REPORT_LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(summary, ensure_ascii=True) + "\n")


def _write_overrides(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _finalize_partial_search(overrides_path: Path, outputs_dir: Path) -> dict[str, Any] | None:
    """Finalize artifacts from the latest recorded search state when the child stops early."""
    if FINAL_SUMMARY_PATH.exists() and _final_metrics_exists(outputs_dir):
        return json.loads(FINAL_SUMMARY_PATH.read_text(encoding="utf-8"))

    if not overrides_path.exists():
        return None

    baseline_metrics_path = outputs_dir / "baseline_metrics.json"
    if not baseline_metrics_path.exists():
        return None

    final_metrics_path = outputs_dir / "final_metrics.json"
    if not final_metrics_path.exists():
        return None
    final_metrics = json.loads(final_metrics_path.read_text(encoding="utf-8"))
    best_search_metrics = final_metrics.get("best_search_metrics")
    if not isinstance(best_search_metrics, dict):
        return None
    FINAL_SUMMARY_PATH.write_text(json.dumps(best_search_metrics, indent=2, default=str), encoding="utf-8")
    return best_search_metrics


def _final_metrics_exists(outputs_dir: Path) -> bool:
    """Return whether the canonical final artifact already exists."""
    return (outputs_dir / "final_metrics.json").exists()


def _run_search_child(overrides_path: Path) -> int:
    import research_loop

    payload = json.loads(overrides_path.read_text(encoding="utf-8"))
    config = payload["config"]
    target_trials = int(payload["target_trials"])

    research_loop.load_config = lambda: config
    result = research_loop.run_engineering_research_loop(cycles_override=target_trials, with_report=True)
    FINAL_SUMMARY_PATH.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    return 0


def _run_parent(input_path: Path, end_time: datetime, target_trials: int, report_minutes: int) -> int:
    if end_time <= datetime.now().astimezone():
        raise ValueError("End time must be in the future.")
    if not input_path.exists():
        raise FileNotFoundError(f"Input dataset not found: {input_path}")

    archived_outputs = _archive_existing_outputs(OUTPUTS_DIR)
    if archived_outputs is not None:
        _log(f"Archived previous outputs to {archived_outputs}")
    _clear_outputs(OUTPUTS_DIR)
    _log(f"Cleared outputs directory at {OUTPUTS_DIR}")

    base_config = _load_base_config()
    baseline_config = _apply_runtime_overrides(
        base_config,
        input_path=input_path,
        min_runtime_minutes=0.0,
        target_trials=target_trials,
    )
    _log(f"Starting baseline with dataset {input_path}")
    _run_baseline(baseline_config)
    _append_report("baseline_complete", OUTPUTS_DIR)
    _log("Baseline complete")

    remaining_minutes = max(
        0.0,
        (end_time - datetime.now().astimezone()).total_seconds() / 60.0,
    )
    search_config = _apply_runtime_overrides(
        base_config,
        input_path=input_path,
        min_runtime_minutes=remaining_minutes,
        target_trials=target_trials,
    )
    _write_overrides(
        OVERRIDES_PATH,
        {
            "config": search_config,
            "target_trials": target_trials,
        },
    )

    _log(
        "Starting governed engineering research loop with "
        f"target_cycles={target_trials} and max_runtime_minutes={remaining_minutes:.2f}"
    )
    child_env = os.environ.copy()
    child_env["PYTHONUNBUFFERED"] = "1"
    with SEARCH_LOG_PATH.open("w", encoding="utf-8", buffering=1) as search_log_handle:
        child = subprocess.Popen(
            [sys.executable, "-u", str(Path(__file__).resolve()), "--child-search", str(OVERRIDES_PATH)],
            cwd=str(REPO_ROOT),
            env=child_env,
            stdout=search_log_handle,
            stderr=subprocess.STDOUT,
        )

    next_report_at = datetime.now().astimezone()
    report_delta = timedelta(minutes=report_minutes)
    forced_stop_logged = False

    while True:
        return_code = child.poll()
        now = datetime.now().astimezone()

        if now >= next_report_at:
            _append_report("search_running" if return_code is None else "search_finished", OUTPUTS_DIR)
            next_report_at = now + report_delta

        if return_code is not None:
            break

        if now >= end_time and not forced_stop_logged:
            _log("Reached requested stop time; waiting for current process state.")
            forced_stop_logged = True

        if now >= end_time + timedelta(minutes=5):
            _log("Search exceeded the requested stop time by more than 5 minutes; terminating.")
            child.terminate()
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait(timeout=30)
            break

        time.sleep(15)

    try:
        _finalize_partial_search(OVERRIDES_PATH, OUTPUTS_DIR)
    except Exception as exc:
        _log(f"WARNING partial_finalization_failed | {exc}")

    _append_report("search_complete", OUTPUTS_DIR)
    _log("Search session finished")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Run an AutoCivil-Lab session until a deadline.")
    parser.add_argument("--input-path", type=Path)
    parser.add_argument("--end-time", type=str)
    parser.add_argument("--duration-minutes", type=float)
    parser.add_argument("--target-trials", type=int, default=50)
    parser.add_argument("--report-minutes", type=int, default=5)
    parser.add_argument("--child-search", type=Path)
    args = parser.parse_args()

    if args.child_search is not None:
        return _run_search_child(args.child_search)

    if args.input_path is None:
        raise ValueError("--input-path is required for parent mode.")
    if args.end_time is None and args.duration_minutes is None:
        raise ValueError("Provide either --end-time or --duration-minutes for parent mode.")
    if args.end_time is not None and args.duration_minutes is not None:
        raise ValueError("Use only one of --end-time or --duration-minutes.")

    if args.end_time is not None:
        end_time = datetime.fromisoformat(args.end_time)
    else:
        end_time = datetime.now().astimezone() + timedelta(minutes=float(args.duration_minutes))
    _acquire_session_lock()
    try:
        return _run_parent(args.input_path.resolve(), end_time, args.target_trials, args.report_minutes)
    finally:
        _release_session_lock()


if __name__ == "__main__":
    raise SystemExit(main())
