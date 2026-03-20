import os
import json
import csv
import re
import secrets
import html
from pathlib import Path
from datetime import datetime, timedelta, timezone
from flask import Flask, Response, jsonify, render_template_string, request, send_from_directory, url_for

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
OUTPUTS_DIR = BASE_DIR / "outputs"
DATA_DIR = BASE_DIR / "data"
CONFIG_PATH = BASE_DIR / "config.yaml"
SHARED_SNAPSHOTS_FILENAME = "shared_snapshots.json"
SHARE_EVENTS_FILENAME = "share_events.jsonl"
DASHBOARD_PASSWORD = os.environ.get("DASHBOARD_PASSWORD", "")
if not DASHBOARD_PASSWORD:
    DASHBOARD_PASSWORD = secrets.token_urlsafe(18)
    print(f"[dashboard] DASHBOARD_PASSWORD not set; generated temporary password: {DASHBOARD_PASSWORD}")

# ─── helpers ────────────────────────────────────────────────────────────────

def safe_read_json(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def safe_read_csv(path):
    try:
        rows = []
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(dict(row))
        return rows
    except Exception:
        return None

def utc_now():
    return datetime.now(timezone.utc)

def to_utc_iso(dt):
    return dt.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")

def parse_utc_iso(value):
    if not value:
        return None
    text = str(value).strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    except ValueError:
        return None

def share_link_ttl_hours():
    raw = os.environ.get("SHARE_LINK_TTL_HOURS", "168")
    try:
        ttl = int(raw)
        return max(ttl, 1)
    except (TypeError, ValueError):
        return 168

def get_dataset_rows():
    try:
        with open(DATA_DIR / "concrete_data.csv", encoding="utf-8") as data_file:
            return max(sum(1 for _ in data_file) - 1, 0)
    except Exception:
        return 0

def shared_snapshots_path():
    return OUTPUTS_DIR / SHARED_SNAPSHOTS_FILENAME

def share_events_path():
    return OUTPUTS_DIR / SHARE_EVENTS_FILENAME

def load_shared_snapshots_store():
    payload = safe_read_json(shared_snapshots_path())
    if isinstance(payload, dict) and isinstance(payload.get("links"), dict):
        return payload
    return {"links": {}}

def save_shared_snapshots_store(store):
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    with shared_snapshots_path().open("w", encoding="utf-8") as handle:
        json.dump(store, handle, indent=2, sort_keys=True)

def log_share_event(event_name, *, token="", metadata=None):
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    event_payload = {
        "event": str(event_name),
        "timestamp": to_utc_iso(utc_now()),
        "token": str(token),
        "metadata": metadata if isinstance(metadata, dict) else {},
    }
    with share_events_path().open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event_payload) + "\n")

def build_share_snapshot_payload():
    baseline = normalize_result_payload(load_required_output_json("baseline_metrics.json") or {})
    final_raw = safe_read_json(OUTPUTS_DIR / "final_metrics.json") or {}
    best_source = (
        final_raw.get("best_search_metrics")
        or safe_read_json(OUTPUTS_DIR / "search_state_best_result.json")
        or safe_read_json(OUTPUTS_DIR / "best_search_result.json")
        or {}
    )
    best = normalize_result_payload(best_source)
    final = normalize_final_payload(final_raw)

    return {
        "dataset_rows": get_dataset_rows(),
        "baseline_model": baseline.get("model_name") or "RandomForestRegressor",
        "best_model": best.get("model_name") or "N/A",
        "best_trial": best.get("best_trial"),
        "baseline_composite": baseline.get("cv_composite") or baseline.get("composite"),
        "best_composite": best.get("holdout_composite") or best.get("cv_composite") or best.get("composite"),
        "composite_improvement_pct": final.get("composite_improvement_pct"),
        "validation_verdict": best.get("validation_verdict") or best.get("validation"),
    }

def load_required_output_json(filename):
    path = OUTPUTS_DIR / filename
    if not path.exists():
        raise FileNotFoundError(filename)
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)

def json_not_found(filename):
    return jsonify({"error": "not_found", "file": filename}), 404

def coerce_number(value):
    try:
        if value in (None, ""):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None

def sanitize_dashboard_payload(value):
    if isinstance(value, dict):
        return {key: sanitize_dashboard_payload(item) for key, item in value.items()}
    if isinstance(value, list):
        return [sanitize_dashboard_payload(item) for item in value]
    if isinstance(value, str):
        return html.escape(value, quote=True)
    return value

def normalize_result_payload(payload):
    if not isinstance(payload, dict):
        return {}

    normalized = dict(payload)
    cv_metrics = normalized.get("cv_metrics") or {}
    test_metrics = normalized.get("test_metrics") or {}
    validation_report = normalized.get("validation_report") or {}

    normalized["validation"] = normalized.get("validation_verdict") or normalized.get("validation")
    normalized["cv_rmse"] = coerce_number(normalized.get("cv_rmse") or cv_metrics.get("rmse") or normalized.get("rmse"))
    normalized["cv_mae"] = coerce_number(normalized.get("cv_mae") or cv_metrics.get("mae") or normalized.get("mae"))
    normalized["cv_r2"] = coerce_number(normalized.get("cv_r2") or cv_metrics.get("r2") or normalized.get("r2"))
    normalized["cv_composite"] = coerce_number(
        normalized.get("cv_composite") or cv_metrics.get("composite_score") or normalized.get("composite_score")
    )
    normalized["holdout_rmse"] = coerce_number(
        normalized.get("holdout_rmse") or test_metrics.get("rmse")
    )
    normalized["holdout_mae"] = coerce_number(
        normalized.get("holdout_mae") or test_metrics.get("mae")
    )
    normalized["holdout_r2"] = coerce_number(
        normalized.get("holdout_r2") or test_metrics.get("r2")
    )
    normalized["holdout_composite"] = coerce_number(
        normalized.get("holdout_composite") or test_metrics.get("composite_score")
    )
    normalized["best_trial"] = normalized.get("best_trial", normalized.get("trial_number"))

    hard_failed_count = int(
        coerce_number(
            normalized.get("hard_failed_count")
            or normalized.get("hard_constraint_count")
            or validation_report.get("hard_constraint_count")
            or validation_report.get("hard_failed_count")
            or normalized.get("failed_count")
            or validation_report.get("failed_count")
            or 0
        )
        or 0
    )
    engineering_caution_count = int(
        coerce_number(
            normalized.get("engineering_caution_count")
            or validation_report.get("engineering_caution_count")
            or normalized.get("durability_caution_count")
            or validation_report.get("durability_caution_count")
            or 0
        )
        or 0
    )
    warning_count = int(
        coerce_number(
            normalized.get("warning_count") or validation_report.get("warning_count") or 0
        )
        or 0
    )
    suspicious_count = int(
        coerce_number(
            normalized.get("suspicious_count")
            or validation_report.get("suspicious_count")
            or 0
        )
        or 0
    )
    dataset_anomaly_count = int(
        coerce_number(
            normalized.get("dataset_anomaly_count")
            or validation_report.get("dataset_anomaly_count")
            or normalized.get("data_review_flag_count")
            or validation_report.get("data_review_flag_count")
            or 0
        )
        or 0
    )
    data_review_flag_count = int(
        coerce_number(
            normalized.get("data_review_flag_count")
            or validation_report.get("data_review_flag_count")
            or dataset_anomaly_count
            or suspicious_count
            or 0
        )
        or 0
    )

    normalized["failed_count"] = hard_failed_count
    normalized["hard_failed_count"] = hard_failed_count
    normalized["hard_constraint_count"] = hard_failed_count
    normalized["warning_count"] = warning_count
    normalized["engineering_caution_count"] = engineering_caution_count
    normalized["durability_caution_count"] = engineering_caution_count
    normalized["data_review_flag_count"] = data_review_flag_count
    normalized["dataset_anomaly_count"] = dataset_anomaly_count
    # API responses flatten list-valued artifact fields into counts where the
    # dashboard expects summary numbers.
    normalized["failed_samples"] = hard_failed_count
    normalized["warning_samples"] = warning_count
    normalized["suspicious_samples"] = suspicious_count
    normalized["suspicious_count"] = suspicious_count
    normalized["warn_reasons"] = normalized.get("warn_reasons") or validation_report.get("warn_reasons") or []
    normalized["hard_constraint_reasons"] = (
        normalized.get("hard_constraint_reasons")
        or validation_report.get("hard_constraint_reasons")
        or normalized.get("hard_fail_reasons")
        or validation_report.get("hard_fail_reasons")
        or []
    )
    normalized["hard_fail_reasons"] = normalized["hard_constraint_reasons"]
    normalized["engineering_caution_reasons"] = (
        normalized.get("engineering_caution_reasons")
        or validation_report.get("engineering_caution_reasons")
        or normalized.get("durability_caution_reasons")
        or validation_report.get("durability_caution_reasons")
        or []
    )
    normalized["durability_caution_reasons"] = (
        normalized["engineering_caution_reasons"]
    )
    normalized["data_review_flag_reasons"] = (
        normalized.get("data_review_flag_reasons")
        or validation_report.get("data_review_flag_reasons")
        or normalized.get("dataset_anomaly_reasons")
        or validation_report.get("dataset_anomaly_reasons")
        or []
    )
    normalized["dataset_anomaly_reasons"] = normalized["data_review_flag_reasons"]
    normalized["contextual_summary"] = (
        normalized.get("contextual_summary") or validation_report.get("contextual_summary") or ""
    )
    normalized["confidence_of_warning_assessment"] = (
        normalized.get("confidence_of_warning_assessment")
        or validation_report.get("confidence_of_warning_assessment")
        or ""
    )
    normalized["validation_pass_rate"] = coerce_number(
        normalized.get("validation_pass_rate") or validation_report.get("pass_rate")
    )
    return normalized

def normalize_final_payload(payload):
    if not isinstance(payload, dict):
        return {}
    normalized = dict(payload)
    normalized["composite_improvement_pct"] = coerce_number(
        normalized.get("composite_improvement_pct") or normalized.get("improvement_percentage")
    )
    return normalized

def normalize_optuna_rows(rows):
    normalized_rows = []
    for row in rows or []:
        normalized = dict(row)
        hyperparameters = {}
        if row.get("hyperparameters"):
            try:
                hyperparameters = json.loads(row["hyperparameters"])
            except Exception:
                hyperparameters = {}

        trial_number = int(coerce_number(row.get("trial_number")) or 0)
        normalized["number"] = trial_number
        normalized["trial"] = trial_number
        normalized["params_model_name"] = row.get("display_name") or row.get("model_name") or "Other"
        normalized["params_model"] = row.get("model_name") or row.get("display_name") or "Other"
        for key, value in hyperparameters.items():
            normalized[f"params_{key}"] = value

        normalized["value"] = coerce_number(row.get("rmse"))
        normalized["rmse"] = coerce_number(row.get("rmse"))
        normalized["mae"] = coerce_number(row.get("mae"))
        normalized["r2"] = coerce_number(row.get("r2"))
        normalized["composite"] = coerce_number(row.get("composite_score"))
        normalized["user_attrs_mae"] = normalized["mae"]
        normalized["user_attrs_r2"] = normalized["r2"]
        normalized["user_attrs_composite"] = normalized["composite"]
        normalized["user_attrs_validation"] = row.get("validation_verdict")
        normalized["user_attrs_improved"] = row.get("selection_status") == "new_best"
        normalized_rows.append(normalized)
    return normalized_rows

def parse_research_log(path):
    trials = []
    pattern = re.compile(
        r"\[(?P<ts>[^\]]+)\]\s+Trial\s+(?P<trial>\d+)\s+\|\s+Model:\s+(?P<model>[^\|]+)\|(?P<rest>.+)"
    )
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                m = pattern.match(line)
                if not m:
                    continue
                rest = m.group("rest")
                is_improvement = "✅" in rest or "New best" in rest
                rmse = re.search(r"RMSE:\s*([\d.]+)", rest)
                r2   = re.search(r"R2:\s*([\d.]+)", rest)
                comp = re.search(r"Composite:\s*([\d.]+)", rest)
                val  = re.search(r"Validation:\s*(\w+)", rest)
                hp_part = re.sub(r"(RMSE|R2|Composite|Validation|✅|❌)[^\|]*", "", rest)
                hp_part = hp_part.replace("|", " ").strip()
                trials.append({
                    "trial":       int(m.group("trial")),
                    "timestamp":   m.group("ts").strip(),
                    "model":       m.group("model").strip(),
                    "hyperparams": hp_part.strip(" |"),
                    "rmse":        float(rmse.group(1)) if rmse else None,
                    "r2":          float(r2.group(1))   if r2   else None,
                    "composite":   float(comp.group(1)) if comp else None,
                    "validation":  val.group(1)         if val  else "—",
                    "improvement": is_improvement,
                })
    except Exception:
        pass
    return trials

# ─── API endpoints ──────────────────────────────────────────────────────────

@app.before_request
def require_dashboard_auth():
    if request.path == "/health" or request.path.startswith("/shared/"):
        return None
    auth = request.authorization
    if auth and auth.username == "autocivil" and auth.password == DASHBOARD_PASSWORD:
        return None
    return Response(
        "Authentication required",
        401,
        {"WWW-Authenticate": 'Basic realm="AutoCivil-Lab"'},
    )

@app.route("/health")
def health():
    return jsonify({"status": "ok", "outputs_exist": OUTPUTS_DIR.exists()})

@app.route("/api/overview")
def api_overview():
    try:
        baseline = normalize_result_payload(load_required_output_json("baseline_metrics.json") or {})
        final_raw = safe_read_json(OUTPUTS_DIR / "final_metrics.json") or {}
        best_source = (
            final_raw.get("best_search_metrics")
            or safe_read_json(OUTPUTS_DIR / "search_state_best_result.json")
            or safe_read_json(OUTPUTS_DIR / "best_search_result.json")
            or {}
        )
        best = normalize_result_payload(best_source)
        final = normalize_final_payload(final_raw)
    except FileNotFoundError as exc:
        return json_not_found(exc.args[0])
    dataset_rows = get_dataset_rows()
    return jsonify(sanitize_dashboard_payload({"baseline": baseline, "best": best, "final": final, "dataset_rows": dataset_rows}))

@app.route("/api/research_log")
def api_research_log():
    trials = parse_research_log(OUTPUTS_DIR / "research_log.txt")
    return jsonify(sanitize_dashboard_payload(trials))

@app.route("/api/optuna_results")
def api_optuna_results():
    rows = safe_read_csv(OUTPUTS_DIR / "optuna_results.csv")
    return jsonify(sanitize_dashboard_payload(normalize_optuna_rows(rows)))

@app.route("/api/validation_details")
def api_validation_details():
    try:
        final_raw = safe_read_json(OUTPUTS_DIR / "final_metrics.json") or {}
        best_source = (
            final_raw.get("best_search_metrics")
            or safe_read_json(OUTPUTS_DIR / "search_state_best_result.json")
            or load_required_output_json("best_search_result.json")
            or {}
        )
        best = normalize_result_payload(best_source)
    except FileNotFoundError as exc:
        return json_not_found(exc.args[0])
    return jsonify(sanitize_dashboard_payload(best))

@app.route("/api/field_validation")
def api_field_validation():
    try:
        records = load_required_output_json("field_validation_log.json")
    except FileNotFoundError as exc:
        return json_not_found(exc.args[0])
    return jsonify(sanitize_dashboard_payload(records if isinstance(records, list) else []))

@app.route("/api/design_results")
def api_design_results():
    batch = safe_read_csv(OUTPUTS_DIR / "batch_design_results.csv")
    singles = []
    for f in OUTPUTS_DIR.glob("design_*MPa.json"):
        d = safe_read_json(f)
        if d:
            d["_filename"] = f.name
            singles.append(d)
    return jsonify(sanitize_dashboard_payload({"batch": batch or [], "singles": singles}))

@app.route("/api/plots")
def api_plots():
    known = {
        "search_progress.png":     "Composite score across all search trials with baseline reference",
        "actual_vs_predicted.png": "Model predictions vs actual values on holdout test set",
        "residuals_plot.png":      "Residual errors vs predicted values — checks for systematic bias",
        "feature_importance.png":  "Relative importance of each input feature in the best model",
        "performance_by_range.png":"RMSE breakdown by low, mid, and high strength ranges",
        "uncertainty_plot.png":    "Prediction interval width across the strength range",
    }
    plots = []
    for f in sorted(OUTPUTS_DIR.glob("*.png")):
        plots.append({"filename": f.name, "description": known.get(f.name, f.stem.replace("_", " ").title())})
    return jsonify(sanitize_dashboard_payload(plots))

@app.route("/api/status")
def api_status():
    required = [
        "baseline_metrics.json", "search_state_best_result.json", "best_search_result.json",
        "final_metrics.json", "research_log.txt", "optuna_results.csv",
    ]
    status = {}
    for name in required:
        p = OUTPUTS_DIR / name
        status[name] = {"exists": p.exists(), "modified": datetime.fromtimestamp(p.stat().st_mtime).isoformat() if p.exists() else None}
    return jsonify(status)

@app.route("/api/share_latest", methods=["POST"])
def api_share_latest():
    try:
        snapshot = build_share_snapshot_payload()
    except FileNotFoundError as exc:
        return json_not_found(exc.args[0])

    now = utc_now()
    ttl_hours = share_link_ttl_hours()
    expires_at = now + timedelta(hours=ttl_hours)
    token = secrets.token_urlsafe(12)

    store = load_shared_snapshots_store()
    store.setdefault("links", {})[token] = {
        "created_at": to_utc_iso(now),
        "expires_at": to_utc_iso(expires_at),
        "snapshot": snapshot,
    }
    save_shared_snapshots_store(store)

    log_share_event(
        "share_link_created",
        token=token,
        metadata={
            "best_model": str(snapshot.get("best_model") or ""),
            "composite_improvement_pct": snapshot.get("composite_improvement_pct"),
        },
    )

    share_path = url_for("shared_snapshot", token=token)
    share_url = request.url_root.rstrip("/") + share_path
    return jsonify(
        sanitize_dashboard_payload(
            {
                "share_url": share_url,
                "share_path": share_path,
                "expires_at": to_utc_iso(expires_at),
                "ttl_hours": ttl_hours,
            }
        )
    ), 201

@app.route("/shared/<token>")
def shared_snapshot(token):
    store = load_shared_snapshots_store()
    record = store.get("links", {}).get(token)
    if not isinstance(record, dict):
        return Response("Share link not found.", 404)

    expires_at = parse_utc_iso(record.get("expires_at"))
    if expires_at and utc_now() > expires_at:
        return Response("Share link expired.", 410)

    snapshot = record.get("snapshot")
    if not isinstance(snapshot, dict):
        snapshot = {}

    def _safe_text(value, fallback="N/A"):
        if value in (None, ""):
            return fallback
        return html.escape(str(value), quote=True)

    def _safe_metric(value, decimals=4, suffix=""):
        numeric = coerce_number(value)
        if numeric is None:
            return "N/A"
        return f"{numeric:.{decimals}f}{suffix}"

    improvement_value = coerce_number(snapshot.get("composite_improvement_pct"))
    improvement_text = "N/A"
    if improvement_value is not None:
        sign = "+" if improvement_value >= 0 else ""
        improvement_text = f"{sign}{improvement_value:.2f}%"

    created_at = parse_utc_iso(record.get("created_at"))
    created_text = to_utc_iso(created_at) if created_at else "N/A"
    expires_text = to_utc_iso(expires_at) if expires_at else "N/A"

    log_share_event(
        "share_link_opened",
        token=token,
        metadata={"best_model": str(snapshot.get("best_model") or "")},
    )

    shared_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width,initial-scale=1"/>
  <title>AutoCivil-Lab Shared Snapshot</title>
  <link rel="preconnect" href="https://fonts.googleapis.com"/>
  <link href="https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@500;700;800&display=swap" rel="stylesheet"/>
  <style>
    :root{{--bg:#f4eee4;--card:#ffffff;--border:#d7cfc1;--txt:#1f2937;--muted:#667085;--accent:#0f766e;--green:#15803d;}}
    *{{box-sizing:border-box;margin:0;padding:0}}
    body{{font-family:'Syne',sans-serif;background:linear-gradient(180deg,#fbf7ef 0%,var(--bg) 100%);color:var(--txt);padding:28px}}
    .wrap{{max-width:860px;margin:0 auto}}
    .hero{{margin-bottom:20px}}
    .hero h1{{font-size:30px;font-weight:800;letter-spacing:-.6px}}
    .hero p{{font-family:'Space Mono',monospace;font-size:12px;color:var(--muted);margin-top:8px}}
    .grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:14px}}
    .card{{background:var(--card);border:1px solid var(--border);border-radius:12px;padding:18px}}
    .label{{font-family:'Space Mono',monospace;font-size:10px;color:var(--muted);margin-bottom:10px;text-transform:uppercase;letter-spacing:.8px}}
    .value{{font-family:'Space Mono',monospace;font-size:24px;font-weight:700}}
    .value.green{{color:var(--green)}}
    .meta{{margin-top:16px;padding-top:16px;border-top:1px solid var(--border);font-family:'Space Mono',monospace;font-size:11px;color:var(--muted);line-height:1.7}}
    .cta{{margin-top:18px;font-family:'Space Mono',monospace;font-size:11px;color:var(--muted)}}
    .cta a{{color:var(--accent);text-decoration:none;font-weight:700}}
  </style>
</head>
<body>
  <main class="wrap">
    <section class="hero">
      <h1>AutoCivil-Lab Shared Snapshot</h1>
      <p>Read-only summary shared from the dashboard.</p>
    </section>
    <section class="grid">
      <article class="card">
        <div class="label">Best Model</div>
        <div class="value">{_safe_text(snapshot.get("best_model"))}</div>
      </article>
      <article class="card">
        <div class="label">Composite Improvement</div>
        <div class="value green">{_safe_text(improvement_text)}</div>
      </article>
      <article class="card">
        <div class="label">Best Composite Score</div>
        <div class="value">{_safe_text(_safe_metric(snapshot.get("best_composite"), 4))}</div>
      </article>
      <article class="card">
        <div class="label">Dataset Rows</div>
        <div class="value">{_safe_text(snapshot.get("dataset_rows"))}</div>
      </article>
    </section>
    <section class="meta">
      <div>Baseline model: {_safe_text(snapshot.get("baseline_model"))}</div>
      <div>Baseline composite score: {_safe_text(_safe_metric(snapshot.get("baseline_composite"), 4))}</div>
      <div>Validation verdict: {_safe_text(snapshot.get("validation_verdict"))}</div>
      <div>Created at: {_safe_text(created_text)}</div>
      <div>Expires at: {_safe_text(expires_text)}</div>
    </section>
    <section class="cta">
      Need the full research detail? Open the private dashboard at <a href="/">/</a> (requires authentication).
    </section>
  </main>
</body>
</html>"""
    return render_template_string(shared_html)

@app.route("/outputs/<path:filename>")
def serve_output(filename):
    return send_from_directory(OUTPUTS_DIR, filename)

# ─── Main page ──────────────────────────────────────────────────────────────

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>AutoCivil-Lab · Dashboard</title>
<link rel="preconnect" href="https://fonts.googleapis.com"/>
<link href="https://fonts.googleapis.com/css2?family=Space+Mono:ital,wght@0,400;0,700;1,400&family=Syne:wght@400;600;700;800&display=swap" rel="stylesheet"/>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<style>
:root{
  color-scheme:light;
  --bg:#f4eee4;
  --surface:#fbf7ef;
  --card:#ffffff;
  --border:#d7cfc1;
  --border-strong:#b8ad9b;
  --accent:#0f766e;
  --accent2:#14b8a6;
  --green:#15803d;
  --yellow:#b45309;
  --red:#b42318;
  --txt:#1f2937;
  --muted:#667085;
  --shadow:0 22px 48px rgba(15, 23, 42, .08);
  --shadow-sm:0 10px 26px rgba(15, 23, 42, .05);
  --mono:'Space Mono',monospace;
  --sans:'Syne',sans-serif;
}
*{box-sizing:border-box;margin:0;padding:0}
html{scroll-behavior:smooth}
body{
  background:
    radial-gradient(circle at top right, rgba(20,184,166,.16), transparent 24%),
    radial-gradient(circle at left 18%, rgba(15,118,110,.08), transparent 20%),
    linear-gradient(180deg,#fbf7ef 0%,var(--bg) 100%);
  color:var(--txt);font-family:var(--sans);display:flex;min-height:100vh;overflow-x:hidden
}

/* scrollbar */
::-webkit-scrollbar{width:4px;height:4px}
::-webkit-scrollbar-track{background:rgba(255,255,255,.4)}
::-webkit-scrollbar-thumb{background:var(--border-strong);border-radius:2px}

/* sidebar */
#sidebar{
  width:220px;min-width:220px;background:rgba(251,247,239,.92);
  border-right:1px solid var(--border);
  display:flex;flex-direction:column;
  position:fixed;top:0;left:0;height:100vh;z-index:100;
  padding:0 0 24px;backdrop-filter:blur(20px);
  box-shadow:var(--shadow-sm);
}
.sb-logo{
  padding:24px 20px 20px;
  border-bottom:1px solid var(--border);
  margin-bottom:12px;
}
.sb-logo h1{font-size:18px;font-weight:800;letter-spacing:-.5px;color:var(--txt)}
.sb-logo span{font-family:var(--mono);font-size:10px;color:var(--accent);display:block;margin-top:3px}
.sb-status{
  display:flex;align-items:center;gap:6px;
  margin-top:10px;font-size:11px;font-family:var(--mono);color:var(--muted);
}
.dot{width:7px;height:7px;border-radius:50%;background:var(--green);box-shadow:0 0 0 4px rgba(21,128,61,.12)}
.dot.red{background:var(--red);box-shadow:0 0 0 4px rgba(180,35,24,.12)}

nav{flex:1;padding:0 10px;overflow-y:auto}
nav a{
  display:flex;align-items:center;gap:10px;
  padding:9px 12px;border-radius:7px;
  font-size:13px;font-weight:600;color:var(--muted);
  text-decoration:none;margin-bottom:2px;
  transition:all .18s;letter-spacing:.3px;
}
nav a:hover{color:var(--txt);background:rgba(15,118,110,.06)}
nav a.active{color:var(--accent);background:rgba(15,118,110,.1);border-left:2px solid var(--accent)}
nav a .icon{font-size:15px;width:20px;text-align:center}

/* topbar */
#topbar{
  position:fixed;top:0;left:220px;right:0;height:52px;
  background:rgba(251,247,239,.82);backdrop-filter:blur(16px);
  border-bottom:1px solid var(--border);
  display:flex;align-items:center;justify-content:space-between;
  padding:0 28px;z-index:90;
}
.tb-left{font-family:var(--mono);font-size:11px;color:var(--muted)}
.tb-left span{color:var(--accent);margin-right:16px}
.tb-right{display:flex;align-items:center;gap:8px}
.refresh-btn{
  background:var(--card);border:1px solid var(--border);
  color:var(--txt);padding:6px 14px;border-radius:6px;
  font-family:var(--mono);font-size:11px;cursor:pointer;
  transition:all .15s;box-shadow:var(--shadow-sm);
}
.refresh-btn:hover{border-color:var(--accent);color:var(--accent);transform:translateY(-1px)}
.refresh-btn:disabled{opacity:.65;cursor:not-allowed;transform:none}
.share-btn{border-color:var(--accent);color:var(--accent)}
.share-status{
  max-width:310px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;
  font-family:var(--mono);font-size:10px;color:var(--muted);
}
.share-status.ok{color:var(--green)}
.share-status.warn{color:var(--yellow)}

/* main */
#main{margin-left:220px;padding-top:52px;min-height:100vh}
.section{padding:32px 28px;min-height:100vh}
.section-title{
  font-size:22px;font-weight:800;margin-bottom:6px;
  letter-spacing:-.5px;color:var(--txt);
}
.section-sub{font-family:var(--mono);font-size:11px;color:var(--muted);margin-bottom:28px}

/* cards */
.card-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(240px,1fr));gap:16px;margin-bottom:28px}
.card{
  background:var(--card);border:1px solid var(--border);
  border-radius:12px;padding:20px;position:relative;overflow:hidden;
  transition:border-color .2s, transform .2s, box-shadow .2s;
  box-shadow:var(--shadow-sm);
}
.card:hover{border-color:var(--accent);transform:translateY(-2px);box-shadow:var(--shadow)}
.card::before{
  content:'';position:absolute;top:0;left:0;right:0;height:2px;
  background:linear-gradient(90deg,var(--accent),var(--accent2));
  opacity:0;transition:opacity .2s;
}
.card:hover::before{opacity:1}
.card-label{font-family:var(--mono);font-size:10px;color:var(--muted);text-transform:uppercase;letter-spacing:1px;margin-bottom:12px}
.card-title{font-size:13px;font-weight:700;color:var(--txt);margin-bottom:16px}
.metric{margin-bottom:10px}
.metric-label{font-family:var(--mono);font-size:10px;color:var(--muted);margin-bottom:2px}
.metric-value{font-family:var(--mono);font-size:20px;font-weight:700;color:var(--txt)}
.metric-value.sm{font-size:14px}
.badge{
  display:inline-block;padding:3px 10px;border-radius:4px;
  font-family:var(--mono);font-size:10px;font-weight:700;letter-spacing:.5px;
}
.badge.pass{background:rgba(21,128,61,.1);color:var(--green);border:1px solid rgba(21,128,61,.22)}
.badge.warn{background:rgba(180,83,9,.1);color:var(--yellow);border:1px solid rgba(180,83,9,.22)}
.badge.fail{background:rgba(180,35,24,.1);color:var(--red);border:1px solid rgba(180,35,24,.22)}

/* timeline */
.log-controls{display:flex;gap:10px;margin-bottom:20px;flex-wrap:wrap;align-items:center}
.log-controls input,.log-controls select{
  background:var(--card);border:1px solid var(--border);
  color:var(--txt);padding:8px 12px;border-radius:7px;
  font-family:var(--mono);font-size:11px;outline:none;
  box-shadow:var(--shadow-sm);
}
.log-controls input:focus,.log-controls select:focus{border-color:var(--accent)}
.log-stats{
  display:flex;gap:20px;margin-bottom:20px;
  font-family:var(--mono);font-size:11px;
}
.log-stat span{color:var(--accent)}

.timeline{display:flex;flex-direction:column;gap:8px}
.t-entry{
  background:var(--card);border:1px solid var(--border);
  border-radius:8px;padding:12px 16px;
  display:grid;grid-template-columns:60px 140px 120px 1fr auto auto auto auto;
  align-items:center;gap:12px;
  font-family:var(--mono);font-size:11px;
  transition:border-color .15s, box-shadow .15s;
  box-shadow:var(--shadow-sm);
}
.t-entry:hover{border-color:var(--border-strong);box-shadow:var(--shadow)}
.t-entry.best{border-left:3px solid var(--green)}
.t-num{color:var(--accent);font-weight:700}
.t-ts{color:var(--muted)}
.t-model{font-weight:700}
.t-model.rf{color:#2563eb}
.t-model.gb{color:#c2410c}
.t-model.xgb{color:#be123c}
.t-model.lgbm{color:var(--green)}
.t-model.svr{color:#7c3aed}
.t-model.ridge{color:var(--muted)}
.t-hp{color:var(--muted);font-size:10px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.t-val{color:var(--txt);font-size:12px}
.imp-badge{font-size:13px}

/* tables */
.table-wrap{overflow-x:auto;margin-bottom:24px}
table{width:100%;border-collapse:collapse;font-family:var(--mono);font-size:11px}
thead th{
  background:var(--surface);color:var(--muted);
  padding:10px 12px;text-align:left;border-bottom:1px solid var(--border);
  cursor:pointer;user-select:none;white-space:nowrap;
  font-size:10px;letter-spacing:.5px;text-transform:uppercase;
}
thead th:hover{color:var(--accent)}
tbody tr{border-bottom:1px solid rgba(184,173,155,.45);transition:background .1s}
tbody tr:hover{background:rgba(15,118,110,.05)}
tbody td{padding:9px 12px;color:var(--txt)}
tbody tr.highlight-best td{background:rgba(21,128,61,.06)}
tbody tr.highlight-base td{background:rgba(15,118,110,.06)}
.tag{
  display:inline-block;background:rgba(15,118,110,.08);
  border:1px solid var(--border);padding:1px 6px;border-radius:3px;
  font-size:9px;margin:1px;color:var(--txt);
}
.pagination{display:flex;gap:6px;align-items:center;margin-top:12px;font-family:var(--mono);font-size:11px}
.pagination button{
  background:var(--card);border:1px solid var(--border);
  color:var(--txt);padding:4px 10px;border-radius:5px;cursor:pointer;
  box-shadow:var(--shadow-sm);
}
.pagination button:hover{border-color:var(--accent)}
.pagination button.active{background:var(--accent);color:#fff;border-color:var(--accent)}
.export-btn{
  background:var(--card);border:1px solid var(--border);
  color:var(--muted);padding:6px 14px;border-radius:6px;
  font-family:var(--mono);font-size:10px;cursor:pointer;margin-bottom:12px;
  box-shadow:var(--shadow-sm);
}
.export-btn:hover{border-color:var(--accent);color:var(--accent)}

/* charts */
.chart-row{display:grid;grid-template-columns:1fr 1fr;gap:20px;margin-bottom:28px}
.chart-box{
  background:var(--card);border:1px solid var(--border);
  border-radius:12px;padding:20px;box-shadow:var(--shadow-sm);
}
.chart-box h3{font-size:13px;font-weight:700;margin-bottom:4px}
.chart-box p{font-family:var(--mono);font-size:10px;color:var(--muted);margin-bottom:16px}
.chart-box canvas{max-height:280px}
.chart-full{grid-column:1/-1}

/* gallery */
.plot-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));gap:16px}
.plot-card{
  background:var(--card);border:1px solid var(--border);
  border-radius:10px;overflow:hidden;cursor:pointer;
  transition:border-color .2s,transform .2s,box-shadow .2s;
  box-shadow:var(--shadow-sm);
}
.plot-card:hover{border-color:var(--accent);transform:translateY(-3px);box-shadow:var(--shadow)}
.plot-card img{width:100%;height:180px;object-fit:cover;display:block}
.plot-caption{padding:12px 14px}
.plot-caption strong{font-size:12px;display:block;margin-bottom:4px}
.plot-caption span{font-family:var(--mono);font-size:10px;color:var(--muted)}

/* modal */
.modal-overlay{
  display:none;position:fixed;inset:0;background:rgba(24,41,57,.42);
  z-index:999;align-items:center;justify-content:center;
  backdrop-filter:blur(12px);
}
.modal-overlay.open{display:flex}
.modal-overlay img{max-width:90vw;max-height:90vh;border-radius:8px;border:1px solid var(--border)}
.modal-close{
  position:absolute;top:20px;right:28px;
  background:var(--card);border:1px solid var(--border);
  color:var(--txt);width:36px;height:36px;border-radius:50%;
  font-size:18px;cursor:pointer;display:flex;align-items:center;justify-content:center;
  box-shadow:var(--shadow-sm);
}

/* design section */
.design-placeholder{
  background:var(--card);border:1px dashed var(--border);
  border-radius:12px;padding:40px;text-align:center;box-shadow:var(--shadow-sm);
}
.design-placeholder code{
  display:block;margin-top:12px;font-family:var(--mono);
  font-size:12px;color:var(--accent);
  background:rgba(15,118,110,.08);padding:8px 16px;border-radius:6px;
  display:inline-block;
}

/* validation panels */
.panel-grid{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:24px}
.panel{background:var(--card);border:1px solid var(--border);border-radius:12px;padding:20px;box-shadow:var(--shadow-sm)}
.panel h3{font-size:13px;font-weight:700;margin-bottom:16px}
.verdict-big{font-size:48px;font-weight:800;font-family:var(--mono);margin:8px 0}
.verdict-big.pass{color:var(--green)}
.verdict-big.warn{color:var(--yellow)}
.verdict-big.fail{color:var(--red)}
.rule-item{
  display:flex;justify-content:space-between;align-items:center;
  padding:8px 0;border-bottom:1px solid rgba(184,173,155,.45);
  font-family:var(--mono);font-size:11px;
}
.rule-item:last-child{border-bottom:none}
.warn-list{list-style:none;padding:0}
.warn-list li{
  padding:6px 10px;margin-bottom:4px;
  background:rgba(180,83,9,.08);border-left:2px solid var(--yellow);
  border-radius:0 4px 4px 0;font-family:var(--mono);font-size:11px;color:var(--yellow);
}

/* skeleton */
.skeleton{
  background:linear-gradient(90deg,var(--surface) 25%,var(--border) 50%,var(--surface) 75%);
  background-size:200% 100%;animation:shimmer 1.5s infinite;
  border-radius:6px;
}
@keyframes shimmer{0%{background-position:200% 0}100%{background-position:-200% 0}}

/* responsive */
@media(max-width:900px){
  .chart-row{grid-template-columns:1fr}
  .panel-grid{grid-template-columns:1fr}
  .t-entry{grid-template-columns:40px 1fr auto auto}
  .share-status{display:none}
}
</style>
</head>
<body>

<!-- Sidebar -->
<aside id="sidebar">
  <div class="sb-logo">
    <h1>AutoCivil-Lab</h1>
    <span>AUTOMATED RESEARCH FRAMEWORK</span>
    <div class="sb-status" id="sb-status">
      <div class="dot" id="status-dot"></div>
      <span id="status-text">Checking…</span>
    </div>
  </div>
  <nav>
    <a href="#field-results"     ><span class="icon">#</span> Field Results</a>
    <a href="#overview"   class="active"><span class="icon">◈</span> Overview</a>
    <a href="#log"               ><span class="icon">◎</span> Research Log</a>
    <a href="#models"            ><span class="icon">◫</span> Model Comparison</a>
    <a href="#validation"        ><span class="icon">◉</span> Engineering Validation</a>
    <a href="#design"            ><span class="icon">◧</span> Design Tool</a>
    <a href="#gallery"           ><span class="icon">◰</span> Plots Gallery</a>
  </nav>
</aside>

<!-- Topbar -->
<header id="topbar">
  <div class="tb-left">
    <span id="tb-dataset">—</span>
    <span id="tb-time">—</span>
  </div>
  <div class="tb-right">
    <span class="share-status" id="share-status"></span>
    <button class="refresh-btn share-btn" id="share-btn" onclick="shareLatestRun()">Share Latest Run</button>
    <button class="refresh-btn" onclick="loadAll()">⟳ Refresh</button>
  </div>
</header>

<!-- Main -->
<main id="main">

<!-- ══ OVERVIEW ══════════════════════════════════════════════════════════ -->
<section class="section" id="overview">
  <div class="section-title">Overview</div>
  <div class="section-sub">Pipeline summary · last run results · dataset info</div>
  <div class="card-grid" id="overview-cards">
    <div class="card"><div class="skeleton" style="height:120px"></div></div>
    <div class="card"><div class="skeleton" style="height:120px"></div></div>
    <div class="card"><div class="skeleton" style="height:120px"></div></div>
  </div>
</section>

<!-- ══ RESEARCH LOG ══════════════════════════════════════════════════════ -->
<section class="section" id="log">
  <div class="section-title">Research Log</div>
  <div class="section-sub">Automated experiment timeline — every trial, every decision</div>
  <div class="log-controls">
    <input type="text" id="log-search" placeholder="Search model / hyperparams…" oninput="filterLog()"/>
    <select id="log-filter" onchange="filterLog()">
      <option value="all">All trials</option>
      <option value="best">Improvements only</option>
    </select>
  </div>
  <div class="log-stats" id="log-stats">Loading…</div>
  <div class="timeline" id="timeline">
    <div class="card skeleton" style="height:44px"></div>
    <div class="card skeleton" style="height:44px"></div>
    <div class="card skeleton" style="height:44px"></div>
  </div>
</section>

<!-- ══ MODEL COMPARISON ══════════════════════════════════════════════════ -->
<section class="section" id="models">
  <div class="section-title">Model Comparison</div>
  <div class="section-sub">All search trials — sortable table + performance charts</div>
  <button class="export-btn" onclick="exportCSV()">↓ Export CSV</button>
  <div class="table-wrap">
    <table id="model-table">
      <thead>
        <tr>
          <th onclick="sortTable(0)">Trial ↕</th>
          <th onclick="sortTable(1)">Model ↕</th>
          <th>Hyperparams</th>
          <th onclick="sortTable(3)">RMSE ↕</th>
          <th onclick="sortTable(4)">MAE ↕</th>
          <th onclick="sortTable(5)">R² ↕</th>
          <th onclick="sortTable(6)">Composite ↕</th>
          <th>Validation</th>
          <th>Result</th>
        </tr>
      </thead>
      <tbody id="model-tbody"></tbody>
    </table>
  </div>
  <div class="pagination" id="pagination"></div>
  <div class="chart-row">
    <div class="chart-box chart-full">
      <h3>Search Progress</h3>
      <p>Composite score per trial — dashed line = baseline</p>
      <canvas id="chart-progress"></canvas>
    </div>
  </div>
  <div class="chart-row">
    <div class="chart-box">
      <h3>Model Family Performance</h3>
      <p>Score range per model family</p>
      <canvas id="chart-families"></canvas>
    </div>
    <div class="chart-box">
      <h3>RMSE Distribution</h3>
      <p>Lower is better</p>
      <canvas id="chart-rmse"></canvas>
    </div>
  </div>
</section>

<!-- ══ VALIDATION ═════════════════════════════════════════════════════════ -->
<section class="section" id="validation">
  <div class="section-title">Engineering Validation</div>
  <div class="section-sub">Physical constraint checks — ACI 318 / BS 8500 inspired rules</div>
  <div class="panel-grid">
    <div class="panel" id="val-summary">Loading…</div>
    <div class="panel" id="val-warnings">Loading…</div>
  </div>
  <div class="chart-row">
    <div class="chart-box">
      <h3>RMSE by Strength Range</h3>
      <p>Baseline vs Best Model — low / mid / high targets</p>
      <canvas id="chart-range"></canvas>
    </div>
    <div class="chart-box">
      <h3>Composite Score Improvement</h3>
      <p>Baseline → Best model delta</p>
      <canvas id="chart-improvement"></canvas>
    </div>
  </div>
</section>

<!-- ══ DESIGN TOOL ════════════════════════════════════════════════════════ -->
<section class="section" id="design">
  <div class="section-title">Design Tool Results</div>
  <div class="section-sub">Inverse prediction — find mix design for a target strength</div>
  <div id="design-content">Loading…</div>
</section>

<!-- ══ GALLERY ════════════════════════════════════════════════════════════ -->
<section class="section" id="field-results">
  <div class="section-title">Field Results</div>
  <div class="section-sub">Recorded lab outcomes - compare predicted and measured compressive strength</div>
  <div id="field-results-content">Loading...</div>
</section>

<section class="section" id="gallery">
  <div class="section-title">Plots Gallery</div>
  <div class="section-sub">All generated figures from the pipeline</div>
  <div class="plot-grid" id="plot-grid">
    <div class="card skeleton" style="height:220px"></div>
    <div class="card skeleton" style="height:220px"></div>
    <div class="card skeleton" style="height:220px"></div>
  </div>
</section>

</main>

<!-- Modal -->
<div class="modal-overlay" id="modal" onclick="closeModal()">
  <button class="modal-close" onclick="closeModal()">✕</button>
  <img id="modal-img" src="" alt=""/>
</div>

<script>
// ─── state ──────────────────────────────────────────────────────────────────
let allTrials = [];
let allOptuna = [];
let fieldValidationRecords = [];
let overviewData = {};
let chartProgress, chartFamilies, chartRmse, chartRange, chartImprovement;
let tablePage = 0;
const PAGE_SIZE = 20;
let tableSortCol = 0;
let tableSortAsc = true;

// ─── boot ────────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  loadAll();
  initScrollSpy();
});

async function loadAll() {
  await Promise.all([
    loadStatus(),
    loadOverview(),
    loadLog(),
    loadOptuna(),
    loadValidation(),
    loadDesign(),
    loadFieldValidation(),
    loadPlots(),
  ]);
}

function setShareStatus(message, level = '') {
  const statusEl = document.getElementById('share-status');
  statusEl.className = 'share-status' + (level ? ` ${level}` : '');
  statusEl.textContent = message || '';
}

async function shareLatestRun() {
  const button = document.getElementById('share-btn');
  const originalLabel = button.textContent;
  button.disabled = true;
  button.textContent = 'Sharing...';
  setShareStatus('Generating secure share link...');
  try {
    const response = await fetch('/api/share_latest', { method: 'POST' });
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) {
      const missing = payload && payload.file ? ` (${payload.file})` : '';
      setShareStatus(`Share failed${missing}`, 'warn');
      button.textContent = 'Share Failed';
      return;
    }

    const sharePath = payload.share_path || '';
    const shareUrl = (sharePath ? `${window.location.origin}${sharePath}` : '') || payload.share_url || '';
    let copied = false;
    if (shareUrl && navigator.clipboard?.writeText) {
      try {
        await navigator.clipboard.writeText(shareUrl);
        copied = true;
      } catch (err) {
        copied = false;
      }
    }

    if (copied) {
      setShareStatus('Link copied. Opens are now tracked.', 'ok');
      button.textContent = 'Copied Link';
    } else if (shareUrl) {
      window.prompt('Copy this share link', shareUrl);
      setShareStatus('Share link created.', 'ok');
      button.textContent = 'Link Ready';
    } else {
      setShareStatus('Share link created.', 'ok');
      button.textContent = 'Link Ready';
    }
  } catch (error) {
    setShareStatus('Share failed due to network error.', 'warn');
    button.textContent = 'Share Failed';
  } finally {
    setTimeout(() => {
      button.disabled = false;
      button.textContent = originalLabel;
    }, 1400);
  }
}

// ─── status ──────────────────────────────────────────────────────────────────
function escapeHtml(value){
  return String(value ?? '—')
    .replace(/&/g,'&amp;')
    .replace(/</g,'&lt;')
    .replace(/>/g,'&gt;')
    .replace(/"/g,'&quot;')
    .replace(/'/g,'&#39;');
}

function badgeClass(value){
  const cleaned = String(value || 'warn').toLowerCase().replace(/[^a-z0-9_-]/g,'');
  return ['pass','warn','fail','ok','low','moderate','high'].includes(cleaned) ? cleaned : 'warn';
}

async function loadStatus() {
  const r = await fetch('/api/status').then(r=>r.json()).catch(()=>({}));
  const all = Object.values(r).every(v=>v.exists);
  document.getElementById('status-dot').className = 'dot' + (all?'':' red');
  document.getElementById('status-text').textContent = all ? 'Pipeline Ready' : 'Incomplete';
  // topbar time
  const times = Object.values(r).filter(v=>v.modified).map(v=>new Date(v.modified));
  if(times.length){
    const latest = new Date(Math.max(...times));
    document.getElementById('tb-time').textContent = 'Last run: ' + latest.toLocaleString();
  }
}

// ─── overview ────────────────────────────────────────────────────────────────
async function loadOverview() {
  const d = await fetch('/api/overview').then(r=>r.json()).catch(()=>({}));
  overviewData = d;
  const b = d.baseline || {};
  const best = d.best || {};
  const fin = d.final || {};
  const bestMetricSource = best.holdout_composite != null ? 'Holdout' : ((best.cv_composite != null || best.composite != null) ? 'CV' : 'N/A');
  document.getElementById('tb-dataset').textContent =
    `Dataset: ${d.dataset_rows || '—'} rows`;

  const verdictBadge = v => `<span class="badge ${(v||'').toLowerCase()}">${v||'—'}</span>`;
  const pct = n => n ? (n*100).toFixed(2)+'%' : '—';
  const num = (n,dec=4) => n!=null ? (+n).toFixed(dec) : '—';

  document.getElementById('overview-cards').innerHTML = `
    <div class="card">
      <div class="card-label">01 · Baseline</div>
      <div class="card-title">${escapeHtml(b.model_name||'RandomForestRegressor')}</div>
      <div class="metric"><div class="metric-label">RMSE ?? CV</div><div class="metric-value">${num(b.cv_rmse||b.rmse,4)}</div></div>
      <div class="metric"><div class="metric-label">CV R²</div><div class="metric-value sm">${num(b.cv_r2||b.r2,4)}</div></div>
      <div class="metric"><div class="metric-label">Composite ?? CV</div><div class="metric-value sm">${num(b.cv_composite||b.composite,4)}</div></div>
      <div style="margin-top:8px">${verdictBadge(b.validation_verdict||b.validation)}</div>
    </div>
    <div class="card">
      <div class="card-label">02 · Best Model</div>
      <div class="card-title">${best.model_name||'—'}</div>
      <div class="metric"><div class="metric-label">RMSE ?? CV</div><div class="metric-value">${num(best.cv_rmse||best.rmse,4)}</div></div>
      <div class="metric"><div class="metric-label">Holdout R²</div><div class="metric-value sm">${num(best.holdout_r2||best.r2,4)}</div></div>
      <div class="metric"><div class="metric-label">Composite ?? ${bestMetricSource}</div><div class="metric-value sm">${num(best.holdout_composite||best.val_composite||best.composite,4)}</div></div>
      <div style="margin-top:8px">${verdictBadge(best.validation_verdict||best.validation)}</div>
    </div>
    <div class="card">
      <div class="card-label">03 · Improvement</div>
      <div class="card-title">vs Baseline</div>
      <div class="metric"><div class="metric-label">Composite Δ</div><div class="metric-value" style="color:var(--green)">${fin.composite_improvement_pct ? '+'+num(fin.composite_improvement_pct,2)+'%' : '—'}</div></div>
      <div class="metric"><div class="metric-label">Best found at Trial</div><div class="metric-value sm">#${best.best_trial||'—'}</div></div>
    </div>
    <div class="card">
      <div class="card-label">04 · Dataset</div>
      <div class="card-title">concrete_data.csv</div>
      <div class="metric"><div class="metric-label">Rows</div><div class="metric-value">${d.dataset_rows||'—'}</div></div>
      <div class="metric"><div class="metric-label">Target</div><div class="metric-value sm">compressive_strength</div></div>
    </div>
    <div class="card">
      <div class="card-label">05 · RMSE by Range</div>
      <div class="card-title">Low / Mid / High MPa</div>
      ${fin.rmse_by_range ? `
        <div class="metric"><div class="metric-label">Low</div><div class="metric-value sm">${num(fin.rmse_by_range.low,4)}</div></div>
        <div class="metric"><div class="metric-label">Mid</div><div class="metric-value sm">${num(fin.rmse_by_range.mid,4)}</div></div>
        <div class="metric"><div class="metric-label">High</div><div class="metric-value sm">${num(fin.rmse_by_range.high,4)}</div></div>
      ` : '<div style="color:var(--muted);font-size:12px">Not available</div>'}
    </div>
    <div class="card">
      <div class="card-label">06 · Validation Status</div>
      <div class="card-title">Engineering Checks</div>
      <div class="metric"><div class="metric-label">Hard Constraints</div><div class="metric-value" style="color:${(best.hard_failed_count||0)>0?'var(--red)':'var(--green)'}">${best.hard_failed_count??'—'}</div></div>
      <div class="metric"><div class="metric-label">Engineering Cautions</div><div class="metric-value sm" style="color:${(best.engineering_caution_count||best.durability_caution_count||0)>0?'var(--yellow)':'var(--green)'}">${best.engineering_caution_count??best.durability_caution_count??'—'}</div></div>
      <div class="metric"><div class="metric-label">Data Review Flags</div><div class="metric-value sm">${best.data_review_flag_count??best.dataset_anomaly_count??'—'}</div></div>
      <div style="margin-top:8px">${verdictBadge(best.validation_verdict||best.validation)}</div>
    </div>
  `;
}

// ─── research log ────────────────────────────────────────────────────────────
async function loadLog() {
  allTrials = await fetch('/api/research_log').then(r=>r.json()).catch(()=>[]);
  renderLog();
}

function filterLog() {
  renderLog();
}

function renderLog() {
  const q = document.getElementById('log-search').value.toLowerCase();
  const filt = document.getElementById('log-filter').value;
  let trials = [...allTrials].reverse();
  if(filt==='best') trials = trials.filter(t=>t.improvement);
  if(q) trials = trials.filter(t=>
    (t.model||'').toLowerCase().includes(q) ||
    (t.hyperparams||'').toLowerCase().includes(q) ||
    String(t.trial).includes(q)
  );
  const total = allTrials.length;
  const improvements = allTrials.filter(t=>t.improvement).length;
  const rate = total ? ((improvements/total)*100).toFixed(1) : 0;
  const bestTrial = allTrials.filter(t=>t.improvement).pop();
  document.getElementById('log-stats').innerHTML = `
    <span>Total: <span>${total}</span></span>
    <span>Improvements: <span>${improvements}</span></span>
    <span>Best at Trial: <span>#${bestTrial?.trial||'—'}</span></span>
    <span>Improvement Rate: <span>${rate}%</span></span>
  `;
  const modelClass = m => {
    if(!m) return '';
    const ml = m.toLowerCase();
    if(ml.includes('random')) return 'rf';
    if(ml.includes('gradient')) return 'gb';
    if(ml.includes('xgb')) return 'xgb';
    if(ml.includes('lgbm')||ml.includes('light')) return 'lgbm';
    if(ml.includes('svr')) return 'svr';
    if(ml.includes('ridge')) return 'ridge';
    return '';
  };
  document.getElementById('timeline').innerHTML = trials.length
    ? trials.map(t=>`
      <div class="t-entry ${t.improvement?'best':''}">
        <span class="t-num">#${String(t.trial).padStart(3,'0')}</span>
        <span class="t-ts">${(t.timestamp||'').slice(11,19)}</span>
        <span class="t-model ${modelClass(t.model)}">${t.model||'—'}</span>
        <span class="t-hp">${escapeHtml(t.hyperparams||'')}</span>
        <span class="t-val">${t.rmse!=null?t.rmse.toFixed(4):'—'}</span>
        <span class="t-val" style="color:var(--muted)">${t.r2!=null?t.r2.toFixed(4):'—'}</span>
        <span class="badge ${(t.validation||'').toLowerCase()}">${t.validation||'—'}</span>
        <span class="imp-badge">${t.improvement?'✅':'❌'}</span>
      </div>`).join('')
    : '<div style="color:var(--muted);font-family:var(--mono);font-size:12px;padding:20px">No trials match the current filter.</div>';
}

// ─── model table ─────────────────────────────────────────────────────────────
async function loadOptuna() {
  allOptuna = await fetch('/api/optuna_results').then(r=>r.json()).catch(()=>[]);
  tablePage = 0;
  renderTable();
  renderCharts();
}

function sortTable(col) {
  if(tableSortCol===col) tableSortAsc=!tableSortAsc;
  else { tableSortCol=col; tableSortAsc=true; }
  renderTable();
}

function renderTable() {
  const best = overviewData.best || {};
  const bestTrial = best.best_trial;
  let rows = [...allOptuna];
  // sort
  rows.sort((a,b)=>{
    const keys = ['number','params_model_name','params','value','user_attrs_mae','user_attrs_r2','user_attrs_composite'];
    const k = keys[tableSortCol];
    const av = isNaN(a[k]) ? (a[k]||'') : +a[k];
    const bv = isNaN(b[k]) ? (b[k]||'') : +b[k];
    return tableSortAsc ? (av>bv?1:-1) : (av<bv?1:-1);
  });
  const total = rows.length;
  const pages = Math.ceil(total/PAGE_SIZE);
  const slice = rows.slice(tablePage*PAGE_SIZE,(tablePage+1)*PAGE_SIZE);

  const modelName = row => row.params_model_name || row['params_model'] || Object.entries(row).find(([k])=>k.toLowerCase().includes('model'))?.[1] || '—';
  const hp = row => Object.entries(row)
    .filter(([k])=>k.startsWith('params_') && !k.includes('model'))
    .map(([k,v])=>`<span class="tag">${escapeHtml(k.replace('params_',''))}: ${escapeHtml(isNaN(v)?v:parseFloat(v).toFixed?.(3))}</span>`)
    .join('');

  document.getElementById('model-tbody').innerHTML = slice.map((row,i)=>{
    const trial = row.number || row.trial || (tablePage*PAGE_SIZE+i+1);
    const isBest = String(trial)===String(bestTrial);
    const rmse = row.value || row.rmse || '—';
    const mae  = row.user_attrs_mae || row.mae || '—';
    const r2   = row.user_attrs_r2 || row.r2 || '—';
    const comp = row.user_attrs_composite || row.composite || '—';
    const val  = row.user_attrs_validation || row.validation || '—';
    const improved = row.user_attrs_improved || '';
    return `<tr class="${isBest?'highlight-best':''}">
      <td>${trial}</td>
      <td>${escapeHtml(modelName(row))}</td>
      <td>${hp(row)}</td>
      <td>${isNaN(rmse)?rmse:parseFloat(rmse).toFixed(4)}</td>
      <td>${isNaN(mae)?mae:parseFloat(mae).toFixed(4)}</td>
      <td>${isNaN(r2)?r2:parseFloat(r2).toFixed(4)}</td>
      <td>${isNaN(comp)?comp:parseFloat(comp).toFixed(4)}</td>
      <td><span class="badge ${badgeClass(val)}">${escapeHtml(val)}</span></td>
      <td>${improved?'✅':'❌'}</td>
    </tr>`;
  }).join('');

  // pagination
  let pg = '';
  for(let i=0;i<pages;i++){
    pg += `<button class="${i===tablePage?'active':''}" onclick="goPage(${i})">${i+1}</button>`;
  }
  document.getElementById('pagination').innerHTML = `<span style="color:var(--muted)">${total} trials</span>${pg}`;
}

function goPage(p){ tablePage=p; renderTable(); }

function exportCSV(){
  if(!allOptuna.length) return;
  const keys = Object.keys(allOptuna[0]);
  const rows = [keys.join(','), ...allOptuna.map(r=>keys.map(k=>`"${r[k]||''}"`).join(','))];
  const blob = new Blob([rows.join('\n')],{type:'text/csv'});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'optuna_results.csv';
  a.click();
}

// ─── charts ──────────────────────────────────────────────────────────────────
function theme(name){
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

function withAlpha(color, alpha){
  if(!color) return color;
  if(color.startsWith('rgba(')) return color;
  if(color.startsWith('rgb(')){
    return color.replace('rgb(', 'rgba(').replace(')', `, ${alpha})`);
  }
  const clean = color.replace('#','');
  const normalized = clean.length === 3 ? clean.split('').map(ch=>ch+ch).join('') : clean;
  const parsed = Number.parseInt(normalized, 16);
  if(Number.isNaN(parsed)) return color;
  const r = (parsed >> 16) & 255;
  const g = (parsed >> 8) & 255;
  const b = parsed & 255;
  return `rgba(${r},${g},${b},${alpha})`;
}

function chartLegendOptions(){
  return {labels:{color:theme('--muted'),font:{size:10}}};
}

function renderCharts(){
  const best = overviewData.best || {};
  const baseline = overviewData.baseline || {};
  const baseComp = +(baseline.cv_composite||baseline.composite||0);
  const accent = theme('--accent');
  const yellow = theme('--yellow');
  const green = theme('--green');
  const defaults = chartDefaults();

  // Search progress
  const comps = allOptuna.map(r=>+(r.user_attrs_composite||r.composite||0));
  const labels = allOptuna.map((_,i)=>i+1);
  const bestIdx = comps.indexOf(Math.max(...comps));

  if(chartProgress) chartProgress.destroy();
  chartProgress = new Chart(document.getElementById('chart-progress'),{
    type:'line',
    data:{
      labels,
      datasets:[
        {label:'Composite Score',data:comps,borderColor:accent,backgroundColor:withAlpha(accent,.14),borderWidth:1.5,pointRadius:2,tension:.3,fill:true},
        {label:'Baseline',data:labels.map(()=>baseComp),borderColor:yellow,borderDash:[5,5],borderWidth:1.5,pointRadius:0},
        {label:'Best',data:labels.map((_,i)=>i===bestIdx?comps[i]:null),borderColor:'transparent',backgroundColor:green,pointRadius:8,pointStyle:'circle',showLine:false},
      ]
    },
    options:{...defaults,plugins:{...defaults.plugins,legend:chartLegendOptions()}}
  });

  // Family performance
  const families = {};
  allOptuna.forEach(r=>{
    const m = r.params_model_name||r['params_model']||'Other';
    const v = +(r.user_attrs_composite||r.composite||0);
    if(!families[m]) families[m]=[];
    families[m].push(v);
  });
  const famLabels = Object.keys(families);
  const famMeans = famLabels.map(f=>avg(families[f]));
  const famColors = famLabels.map(modelColor);

  if(chartFamilies) chartFamilies.destroy();
  chartFamilies = new Chart(document.getElementById('chart-families'),{
    type:'bar',
    data:{labels:famLabels,datasets:[{label:'Mean Composite',data:famMeans,backgroundColor:famColors.map(c=>withAlpha(c,.18)),borderColor:famColors,borderWidth:1.5,borderRadius:4}]},
    options:{...defaults,plugins:{...defaults.plugins,legend:{display:false}}}
  });

  // RMSE distribution
  const famRmse = famLabels.map(f=>{
    const vals = allOptuna.filter(r=>(r.params_model_name||r['params_model']||'Other')===f).map(r=>+(r.value||r.rmse||0));
    return avg(vals);
  });
  if(chartRmse) chartRmse.destroy();
  chartRmse = new Chart(document.getElementById('chart-rmse'),{
    type:'bar',
    data:{labels:famLabels,datasets:[{label:'Mean RMSE',data:famRmse,backgroundColor:famColors.map(c=>withAlpha(c,.18)),borderColor:famColors,borderWidth:1.5,borderRadius:4}]},
    options:{...defaults,plugins:{...defaults.plugins,legend:{display:false}}}
  });

  // Range chart
  const fin = overviewData.final || {};
  if(fin.rmse_by_range){
    const rng = fin.rmse_by_range;
    const baseRng = fin.baseline_rmse_by_range || {};
    if(chartRange) chartRange.destroy();
    chartRange = new Chart(document.getElementById('chart-range'),{
      type:'bar',
      data:{
        labels:['Low','Mid','High'],
        datasets:[
          {label:'Baseline',data:[baseRng.low,baseRng.mid,baseRng.high],backgroundColor:withAlpha(accent,.18),borderColor:accent,borderWidth:1.5,borderRadius:4},
          {label:'Best Model',data:[rng.low,rng.mid,rng.high],backgroundColor:withAlpha(green,.18),borderColor:green,borderWidth:1.5,borderRadius:4},
        ]
      },
      options:{...defaults,plugins:{...defaults.plugins,legend:chartLegendOptions()}}
    });
  }

  // Improvement gauge
  const bComp = +(baseline.cv_composite||baseline.composite||0);
  const bestComp = +(best.holdout_composite||best.composite||0);
  if(chartImprovement) chartImprovement.destroy();
  chartImprovement = new Chart(document.getElementById('chart-improvement'),{
    type:'doughnut',
    data:{
      labels:['Baseline','Improvement'],
      datasets:[{data:[bComp, Math.max(0,bestComp-bComp)],backgroundColor:[withAlpha(accent,.24),withAlpha(green,.4)],borderColor:[accent,green],borderWidth:2}]
    },
    options:{
      ...defaults,
      cutout:'72%',
      plugins:{
        ...defaults.plugins,
        legend:chartLegendOptions(),
        tooltip:{
          ...defaults.plugins.tooltip,
          callbacks:{label:c=>`${c.label}: ${(+c.raw).toFixed(4)}`}
        }
      }
    }
  });
}

function modelColor(m){
  if(!m) return theme('--muted');
  const ml = m.toLowerCase();
  if(ml.includes('random')) return '#2563eb';
  if(ml.includes('gradient')) return '#c2410c';
  if(ml.includes('xgb')) return '#be123c';
  if(ml.includes('lgbm')||ml.includes('light')) return theme('--green');
  if(ml.includes('svr')) return '#7c3aed';
  return theme('--muted');
}

function avg(arr){ return arr.length ? arr.reduce((a,b)=>a+b,0)/arr.length : 0; }

function chartDefaults(){
  const card = theme('--card');
  const border = theme('--border');
  const txt = theme('--txt');
  const muted = theme('--muted');
  return {
    responsive:true,
    plugins:{tooltip:{backgroundColor:card,borderColor:border,borderWidth:1,titleColor:txt,bodyColor:muted,titleFont:{family:'Space Mono',size:11},bodyFont:{family:'Space Mono',size:11},padding:12}},
    scales:{x:{grid:{color:withAlpha(border,.75)},ticks:{color:muted,font:{family:'Space Mono',size:10}}},y:{grid:{color:withAlpha(border,.75)},ticks:{color:muted,font:{family:'Space Mono',size:10}}}}
  };
}

async function loadValidation() {
  const d = await fetch('/api/validation_details').then(r=>r.json()).catch(()=>({}));
  const v = d.validation_verdict || d.validation || '—';
  const fail = d.hard_failed_count ?? d.failed_samples ?? 0;
  const warn = d.warning_count ?? 0;
  const cautions = d.engineering_caution_count ?? d.durability_caution_count ?? 0;
  const reviewFlags = d.data_review_flag_count ?? d.dataset_anomaly_count ?? d.suspicious_samples ?? 0;
  const passRate = d.validation_pass_rate;
  const hardFailReasons = d.hard_fail_reasons || [];
  const warns = d.warn_reasons || [];
  const cautionReasons = d.engineering_caution_reasons || d.durability_caution_reasons || [];
  const reviewFlagReasons = d.data_review_flag_reasons || d.dataset_anomaly_reasons || [];
  const contextualSummary = d.contextual_summary || '—';
  const assessmentConfidence = d.confidence_of_warning_assessment || '—';
  const red = theme('--red');
  const yellow = theme('--yellow');
  const accent = theme('--accent');
  const renderReasonGroup = (title, items, emptyText, styles) => `
    <div style="margin-bottom:16px">
      <div class="metric-label" style="margin-bottom:8px">${title}</div>
      ${items.length
        ? `<ul class="warn-list">${items.map(w=>`<li style="background:${styles.bg};border-left-color:${styles.border};color:${styles.text}">${escapeHtml(w)}</li>`).join('')}</ul>`
        : `<div style="color:var(--muted);font-family:var(--mono);font-size:11px">${emptyText}</div>`
      }
    </div>
  `;

  document.getElementById('val-summary').innerHTML = `
    <h3>Validation Summary</h3>
    <div class="verdict-big ${v.toLowerCase()}">${v}</div>
    <div class="metric"><div class="metric-label">Hard Constraints</div><div class="metric-value sm" style="color:${fail>0?'var(--red)':'var(--green)'}">${fail}</div></div>
    <div class="metric"><div class="metric-label">Warning Samples</div><div class="metric-value sm" style="color:${warn>0?'var(--yellow)':'var(--green)'}">${warn}</div></div>
    <div class="metric"><div class="metric-label">Engineering Cautions</div><div class="metric-value sm" style="color:${cautions>0?'var(--yellow)':'var(--green)'}">${cautions}</div></div>
    <div class="metric"><div class="metric-label">Data Review Flags</div><div class="metric-value sm" style="color:${reviewFlags>0?'var(--yellow)':'var(--green)'}">${reviewFlags}</div></div>
    <div class="metric"><div class="metric-label">Pass Rate</div><div class="metric-value sm">${passRate!=null?(passRate*100).toFixed(2)+'%':'—'}</div></div>
    <div class="metric"><div class="metric-label">Assessment Confidence</div><div class="metric-value sm">${assessmentConfidence}</div></div>
    <div class="metric"><div class="metric-label">Model</div><div class="metric-value sm">${d.model_name||'—'}</div></div>
  `;

  document.getElementById('val-warnings').innerHTML = `
    <h3>Validation Breakdown</h3>
    ${renderReasonGroup('Hard-Fail Reasons', hardFailReasons, 'No hard failures triggered', {bg:withAlpha(red,.08), border:red, text:red})}
    ${renderReasonGroup('Warning Reasons', warns, 'No warnings triggered', {bg:withAlpha(yellow,.08), border:yellow, text:yellow})}
    ${renderReasonGroup('Engineering Cautions', cautionReasons, 'No engineering cautions triggered', {bg:withAlpha(yellow,.08), border:yellow, text:yellow})}
    ${renderReasonGroup('Data Review Flags', reviewFlagReasons, 'No data review flags triggered', {bg:withAlpha(accent,.08), border:accent, text:accent})}
    <div style="margin-bottom:16px">
      <div class="metric-label" style="margin-bottom:8px">Contextual Summary</div>
      <div style="color:var(--muted);font-family:var(--mono);font-size:11px;line-height:1.6">${escapeHtml(contextualSummary)}</div>
    </div>
    <div style="margin-top:16px;padding-top:16px;border-top:1px solid var(--border)">
      <div class="metric-label" style="margin-bottom:8px">RULE REFERENCE</div>
      ${[
        ['Exposure class + w/c or w/b','Durability caution uses exposure metadata when available','WARN'],
        ['High w/c + high strength','SCM, age, binder, and w/b context decide whether review is needed','WARN'],
        ['High-volume SCM regime','Triggers age-aware data review instead of automatic anomaly labeling','WARN'],
        ['Binder < 250 kg/m³','Low binder content','WARN'],
        ['Binder > 550 kg/m³','Shrinkage risk','WARN'],
        ['Fly ash > 40%','Exceeds ACI substitution limit','WARN'],
        ['Slag > 70%','Exceeds BS 8500 GGBS limit','WARN'],
        ['Predicted NaN / inf','Numerically invalid model output','FAIL'],
        ['Predicted < 0 MPa','Physical impossibility','FAIL'],
        ['Predicted outside configured bounds','Outside configured engineering range','FAIL'],
      ].map(([rule,desc,sev])=>`
        <div class="rule-item">
          <span style="color:var(--txt)">${rule}</span>
          <span style="color:var(--muted);flex:1;margin:0 12px;font-size:10px">${desc}</span>
          <span class="badge ${sev.toLowerCase()}">${sev}</span>
        </div>`).join('')}
    </div>
  `;
}

// ─── design tool ─────────────────────────────────────────────────────────────
async function loadDesign() {
  const d = await fetch('/api/design_results').then(r=>r.json()).catch(()=>({batch:[],singles:[]}));
  const el = document.getElementById('design-content');
  if(!d.batch.length && !d.singles.length){
    el.innerHTML = `
      <div class="design-placeholder">
        <div style="font-size:32px;margin-bottom:12px">⬡</div>
        <div style="font-size:14px;font-weight:700;margin-bottom:6px">No Design Tool Results Yet</div>
        <div style="color:var(--muted);font-size:12px;margin-bottom:16px">Run the design tool to generate inverse-prediction mix designs</div>
        <code>python design_tool.py --target 35</code>
        <div style="margin-top:8px"></div>
        <code>python design_tool.py --batch 25,30,35,40,45</code>
      </div>`;
    return;
  }
  let html = '';
  if(d.batch.length){
    html += `<div class="table-wrap" style="margin-bottom:24px">
      <table>
        <thead><tr>
          <th>Target MPa</th><th>Predicted MPa</th>
          <th>Cement</th><th>Slag</th><th>Fly Ash</th>
          <th>Water</th><th>w/c</th><th>Verdict</th>
        </tr></thead>
        <tbody>${d.batch.map(r=>`<tr>
          <td>${r.target_strength||'—'}</td>
          <td>${r.predicted_strength?parseFloat(r.predicted_strength).toFixed(2):'—'}</td>
          <td>${r.cement||'—'}</td><td>${r.slag||'—'}</td><td>${r.fly_ash||'—'}</td>
          <td>${r.water||'—'}</td>
          <td>${r.water_cement_ratio?parseFloat(r.water_cement_ratio).toFixed(3):'—'}</td>
          <td><span class="badge ${(r.validation_verdict||'').toLowerCase()}">${r.validation_verdict||'—'}</span></td>
        </tr>`).join('')}</tbody>
      </table></div>`;
  }
  if(d.singles.length){
    html += `<div class="card-grid">${d.singles.map(s=>`
      <div class="card" style="cursor:pointer" onclick="this.nextElementSibling.style.display=this.nextElementSibling.style.display==='none'?'block':'none'">
        <div class="card-label">Design Result</div>
        <div class="card-title">${s._filename||'—'}</div>
        <div class="metric"><div class="metric-label">Target</div><div class="metric-value sm">${s.target_strength||'—'} MPa</div></div>
        <div class="metric"><div class="metric-label">Predicted</div><div class="metric-value sm">${s.predicted_strength?parseFloat(s.predicted_strength).toFixed(2):'—'} MPa</div></div>
        <span class="badge ${(s.validation_verdict||'').toLowerCase()}">${s.validation_verdict||'—'}</span>
      </div>
      <div style="display:none;background:var(--card);border:1px solid var(--border);border-radius:8px;padding:16px;margin-top:-8px;margin-bottom:8px;font-family:var(--mono);font-size:11px">
        <pre style="white-space:pre-wrap;color:var(--muted)">${escapeHtml(JSON.stringify(s,null,2))}</pre>
      </div>
    `).join('')}</div>`;
  }
  el.innerHTML = html;
}

// ─── gallery ─────────────────────────────────────────────────────────────────
async function loadFieldValidation() {
  const el = document.getElementById('field-results-content');
  const response = await fetch('/api/field_validation').catch(()=>null);
  if(!response){
    el.innerHTML = '<div class="design-placeholder">Unable to load field validation records.</div>';
    return;
  }

  const payload = await response.json().catch(()=>[]);
  if(!response.ok || !Array.isArray(payload) || !payload.length){
    const missingFile = payload && payload.file ? payload.file : 'field_validation_log.json';
    el.innerHTML = `
      <div class="design-placeholder">
        <div style="font-size:14px;font-weight:700;margin-bottom:6px">No Field Validation Records Yet</div>
        <div style="color:var(--muted);font-size:12px;margin-bottom:16px">Record lab outcomes after testing saved mix designs.</div>
        <code>python field_tracker.py record --design outputs/design_35MPa.json --actual-strength 33.7 --notes "28-day cube test, batch #4"</code>
        <div style="margin-top:12px;color:var(--muted);font-family:var(--mono);font-size:10px">Expected log file: ${missingFile}</div>
      </div>`;
    return;
  }

  fieldValidationRecords = [...payload].sort((a,b)=>new Date(b.timestamp) - new Date(a.timestamp));
  const errors = fieldValidationRecords.map(r=>+(r.prediction_error_mpa || 0));
  const rmse = Math.sqrt(errors.reduce((sum,value)=>sum + value * value, 0) / fieldValidationRecords.length);
  const meanError = errors.reduce((sum,value)=>sum + value, 0) / fieldValidationRecords.length;
  const withinTolerance = fieldValidationRecords.filter(r=>r.within_tolerance).length / fieldValidationRecords.length * 100;

  el.innerHTML = `
    <div class="card-grid" style="margin-bottom:20px">
      <div class="card"><div class="card-label">Field Tests</div><div class="metric-value">${fieldValidationRecords.length}</div></div>
      <div class="card"><div class="card-label">Mean Error</div><div class="metric-value sm">${meanError.toFixed(3)} MPa</div></div>
      <div class="card"><div class="card-label">RMSE</div><div class="metric-value sm">${rmse.toFixed(3)} MPa</div></div>
      <div class="card"><div class="card-label">Within +/- 2 MPa</div><div class="metric-value sm">${withinTolerance.toFixed(1)}%</div></div>
    </div>
    <div class="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Timestamp</th>
            <th>Design File</th>
            <th>Target</th>
            <th>Predicted</th>
            <th>Actual</th>
            <th>Error</th>
            <th>Tolerance</th>
            <th>Verdict</th>
            <th>Notes</th>
          </tr>
        </thead>
        <tbody>
          ${fieldValidationRecords.map(record=>`
            <tr>
              <td>${record.timestamp || '—'}</td>
              <td>${record.design_file || '—'}</td>
              <td>${record.target_strength != null ? Number(record.target_strength).toFixed(2) : '—'}</td>
              <td>${record.predicted_strength != null ? Number(record.predicted_strength).toFixed(2) : '—'}</td>
              <td>${record.actual_strength != null ? Number(record.actual_strength).toFixed(2) : '—'}</td>
              <td>${record.prediction_error_mpa != null ? Number(record.prediction_error_mpa).toFixed(2) : '—'}</td>
              <td><span class="badge ${record.within_tolerance ? 'pass' : 'warn'}">${record.within_tolerance ? 'WITHIN' : 'OUTSIDE'}</span></td>
              <td><span class="badge ${String(record.validation_verdict || 'warn').toLowerCase()}">${record.validation_verdict || '—'}</span></td>
              <td>${record.notes || '—'}</td>
            </tr>
          `).join('')}
        </tbody>
      </table>
    </div>`;
}

async function loadPlots() {
  const plots = await fetch('/api/plots').then(r=>r.json()).catch(()=>[]);
  document.getElementById('plot-grid').innerHTML = plots.length
    ? plots.map(p=>`
      <div class="plot-card" onclick="openModal('/outputs/${p.filename}')">
        <img src="/outputs/${encodeURIComponent(p.filename)}" alt="${escapeHtml(p.filename)}" loading="lazy"/>
        <div class="plot-caption">
          <strong>${escapeHtml(p.filename)}</strong>
          <span>${escapeHtml(p.description)}</span>
        </div>
      </div>`).join('')
    : '<div style="color:var(--muted);font-family:var(--mono);font-size:12px;padding:20px">No plots found in outputs/ yet. Run the pipeline first.</div>';
}

function openModal(src){ document.getElementById('modal-img').src=src; document.getElementById('modal').classList.add('open'); }
function closeModal(){ document.getElementById('modal').classList.remove('open'); document.getElementById('modal-img').src=''; }
document.addEventListener('keydown',e=>{ if(e.key==='Escape') closeModal(); });

// ─── scrollspy ───────────────────────────────────────────────────────────────
function initScrollSpy(){
  const links = document.querySelectorAll('nav a');
  const sections = [...links].map(a=>document.querySelector(a.getAttribute('href')));
  const obs = new IntersectionObserver(entries=>{
    entries.forEach(e=>{
      if(e.isIntersecting){
        links.forEach(l=>l.classList.remove('active'));
        const a = document.querySelector(`nav a[href="#${e.target.id}"]`);
        if(a) a.classList.add('active');
      }
    });
  },{threshold:.3});
  sections.forEach(s=>s&&obs.observe(s));
}
</script>
</body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(HTML)

if __name__ == "__main__":
    import webbrowser, threading
    port = int(os.environ.get("PORT", 5050))
    def open_browser():
        webbrowser.open(f"http://localhost:{port}")
    threading.Timer(1.2, open_browser).start()
    print("\n  AutoCivil-Lab Dashboard")
    print(f"  http://localhost:{port}\n")
    app.run(debug=False, port=port)
