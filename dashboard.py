import os
import json
import csv
import re
from pathlib import Path
from datetime import datetime
from flask import Flask, jsonify, send_from_directory, render_template_string

app = Flask(__name__)

BASE_DIR = Path(__file__).parent
OUTPUTS_DIR = BASE_DIR / "outputs"
DATA_DIR = BASE_DIR / "data"
CONFIG_PATH = BASE_DIR / "config.yaml"

# ─── helpers ────────────────────────────────────────────────────────────────

def safe_read_json(path):
    try:
        with open(path) as f:
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

@app.route("/api/overview")
def api_overview():
    baseline = safe_read_json(OUTPUTS_DIR / "baseline_metrics.json") or {}
    best     = safe_read_json(OUTPUTS_DIR / "best_search_result.json") or {}
    final    = safe_read_json(OUTPUTS_DIR / "final_metrics.json") or {}
    # dataset info
    dataset_rows = 0
    try:
        with open(DATA_DIR / "concrete_data.csv") as f:
            dataset_rows = sum(1 for _ in f) - 1
    except Exception:
        pass
    return jsonify({"baseline": baseline, "best": best, "final": final, "dataset_rows": dataset_rows})

@app.route("/api/research_log")
def api_research_log():
    trials = parse_research_log(OUTPUTS_DIR / "research_log.txt")
    return jsonify(trials)

@app.route("/api/optuna_results")
def api_optuna_results():
    rows = safe_read_csv(OUTPUTS_DIR / "optuna_results.csv")
    return jsonify(rows or [])

@app.route("/api/validation_details")
def api_validation_details():
    best = safe_read_json(OUTPUTS_DIR / "best_search_result.json") or {}
    return jsonify(best)

@app.route("/api/design_results")
def api_design_results():
    batch = safe_read_csv(OUTPUTS_DIR / "batch_design_results.csv")
    singles = []
    for f in OUTPUTS_DIR.glob("design_*MPa.json"):
        d = safe_read_json(f)
        if d:
            d["_filename"] = f.name
            singles.append(d)
    return jsonify({"batch": batch or [], "singles": singles})

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
    return jsonify(plots)

@app.route("/api/status")
def api_status():
    required = [
        "baseline_metrics.json", "best_search_result.json",
        "final_metrics.json", "research_log.txt", "optuna_results.csv",
    ]
    status = {}
    for name in required:
        p = OUTPUTS_DIR / name
        status[name] = {"exists": p.exists(), "modified": datetime.fromtimestamp(p.stat().st_mtime).isoformat() if p.exists() else None}
    return jsonify(status)

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
  --bg:#080b12;
  --surface:#0e1420;
  --card:#131a27;
  --border:#1e2d42;
  --accent:#00d4ff;
  --accent2:#7c3aed;
  --green:#00e5a0;
  --yellow:#f59e0b;
  --red:#f43f5e;
  --txt:#e2eaf4;
  --muted:#4a6080;
  --mono:'Space Mono',monospace;
  --sans:'Syne',sans-serif;
}
*{box-sizing:border-box;margin:0;padding:0}
html{scroll-behavior:smooth}
body{background:var(--bg);color:var(--txt);font-family:var(--sans);display:flex;min-height:100vh;overflow-x:hidden}

/* scrollbar */
::-webkit-scrollbar{width:4px;height:4px}
::-webkit-scrollbar-track{background:var(--bg)}
::-webkit-scrollbar-thumb{background:var(--border);border-radius:2px}

/* ── sidebar ── */
#sidebar{
  width:220px;min-width:220px;background:var(--surface);
  border-right:1px solid var(--border);
  display:flex;flex-direction:column;
  position:fixed;top:0;left:0;height:100vh;z-index:100;
  padding:0 0 24px;
}
.sb-logo{
  padding:24px 20px 20px;
  border-bottom:1px solid var(--border);
  margin-bottom:12px;
}
.sb-logo h1{font-size:18px;font-weight:800;letter-spacing:-.5px;color:#fff}
.sb-logo span{font-family:var(--mono);font-size:10px;color:var(--accent);display:block;margin-top:3px}
.sb-status{
  display:flex;align-items:center;gap:6px;
  margin-top:10px;font-size:11px;font-family:var(--mono);color:var(--muted);
}
.dot{width:7px;height:7px;border-radius:50%;background:var(--green);box-shadow:0 0 6px var(--green)}
.dot.red{background:var(--red);box-shadow:0 0 6px var(--red)}

nav{flex:1;padding:0 10px;overflow-y:auto}
nav a{
  display:flex;align-items:center;gap:10px;
  padding:9px 12px;border-radius:7px;
  font-size:13px;font-weight:600;color:var(--muted);
  text-decoration:none;margin-bottom:2px;
  transition:all .18s;letter-spacing:.3px;
}
nav a:hover{color:var(--txt);background:var(--card)}
nav a.active{color:var(--accent);background:rgba(0,212,255,.07);border-left:2px solid var(--accent)}
nav a .icon{font-size:15px;width:20px;text-align:center}

/* ── topbar ── */
#topbar{
  position:fixed;top:0;left:220px;right:0;height:52px;
  background:rgba(8,11,18,.85);backdrop-filter:blur(12px);
  border-bottom:1px solid var(--border);
  display:flex;align-items:center;justify-content:space-between;
  padding:0 28px;z-index:90;
}
.tb-left{font-family:var(--mono);font-size:11px;color:var(--muted)}
.tb-left span{color:var(--accent);margin-right:16px}
.refresh-btn{
  background:var(--card);border:1px solid var(--border);
  color:var(--txt);padding:6px 14px;border-radius:6px;
  font-family:var(--mono);font-size:11px;cursor:pointer;
  transition:all .15s;
}
.refresh-btn:hover{border-color:var(--accent);color:var(--accent)}

/* ── main ── */
#main{margin-left:220px;padding-top:52px;min-height:100vh}
.section{padding:32px 28px;min-height:100vh}
.section-title{
  font-size:22px;font-weight:800;margin-bottom:6px;
  letter-spacing:-.5px;color:#fff;
}
.section-sub{font-family:var(--mono);font-size:11px;color:var(--muted);margin-bottom:28px}

/* ── cards ── */
.card-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(240px,1fr));gap:16px;margin-bottom:28px}
.card{
  background:var(--card);border:1px solid var(--border);
  border-radius:12px;padding:20px;position:relative;overflow:hidden;
  transition:border-color .2s, transform .2s;
}
.card:hover{border-color:var(--accent);transform:translateY(-2px)}
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
.metric-value{font-family:var(--mono);font-size:20px;font-weight:700;color:#fff}
.metric-value.sm{font-size:14px}
.badge{
  display:inline-block;padding:3px 10px;border-radius:4px;
  font-family:var(--mono);font-size:10px;font-weight:700;letter-spacing:.5px;
}
.badge.pass{background:rgba(0,229,160,.12);color:var(--green);border:1px solid rgba(0,229,160,.3)}
.badge.warn{background:rgba(245,158,11,.12);color:var(--yellow);border:1px solid rgba(245,158,11,.3)}
.badge.fail{background:rgba(244,63,94,.12);color:var(--red);border:1px solid rgba(244,63,94,.3)}

/* ── timeline ── */
.log-controls{display:flex;gap:10px;margin-bottom:20px;flex-wrap:wrap;align-items:center}
.log-controls input,.log-controls select{
  background:var(--card);border:1px solid var(--border);
  color:var(--txt);padding:8px 12px;border-radius:7px;
  font-family:var(--mono);font-size:11px;outline:none;
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
  transition:border-color .15s;
}
.t-entry:hover{border-color:var(--border)}
.t-entry.best{border-left:3px solid var(--green)}
.t-num{color:var(--accent);font-weight:700}
.t-ts{color:var(--muted)}
.t-model{font-weight:700}
.t-model.rf{color:#60a5fa}
.t-model.gb{color:#fb923c}
.t-model.xgb{color:#f43f5e}
.t-model.lgbm{color:var(--green)}
.t-model.svr{color:#a78bfa}
.t-model.ridge{color:var(--muted)}
.t-hp{color:var(--muted);font-size:10px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.t-val{color:#fff;font-size:12px}
.imp-badge{font-size:13px}

/* ── tables ── */
.table-wrap{overflow-x:auto;margin-bottom:24px}
table{width:100%;border-collapse:collapse;font-family:var(--mono);font-size:11px}
thead th{
  background:var(--surface);color:var(--muted);
  padding:10px 12px;text-align:left;border-bottom:1px solid var(--border);
  cursor:pointer;user-select:none;white-space:nowrap;
  font-size:10px;letter-spacing:.5px;text-transform:uppercase;
}
thead th:hover{color:var(--accent)}
tbody tr{border-bottom:1px solid rgba(30,45,66,.5);transition:background .1s}
tbody tr:hover{background:rgba(255,255,255,.02)}
tbody td{padding:9px 12px;color:var(--txt)}
tbody tr.highlight-best td{background:rgba(0,229,160,.04)}
tbody tr.highlight-base td{background:rgba(0,212,255,.04)}
.tag{
  display:inline-block;background:rgba(255,255,255,.06);
  border:1px solid var(--border);padding:1px 6px;border-radius:3px;
  font-size:9px;margin:1px;
}
.pagination{display:flex;gap:6px;align-items:center;margin-top:12px;font-family:var(--mono);font-size:11px}
.pagination button{
  background:var(--card);border:1px solid var(--border);
  color:var(--txt);padding:4px 10px;border-radius:5px;cursor:pointer;
}
.pagination button:hover{border-color:var(--accent)}
.pagination button.active{background:var(--accent);color:#000;border-color:var(--accent)}
.export-btn{
  background:transparent;border:1px solid var(--border);
  color:var(--muted);padding:6px 14px;border-radius:6px;
  font-family:var(--mono);font-size:10px;cursor:pointer;margin-bottom:12px;
}
.export-btn:hover{border-color:var(--accent);color:var(--accent)}

/* ── charts ── */
.chart-row{display:grid;grid-template-columns:1fr 1fr;gap:20px;margin-bottom:28px}
.chart-box{
  background:var(--card);border:1px solid var(--border);
  border-radius:12px;padding:20px;
}
.chart-box h3{font-size:13px;font-weight:700;margin-bottom:4px}
.chart-box p{font-family:var(--mono);font-size:10px;color:var(--muted);margin-bottom:16px}
.chart-box canvas{max-height:280px}
.chart-full{grid-column:1/-1}

/* ── gallery ── */
.plot-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));gap:16px}
.plot-card{
  background:var(--card);border:1px solid var(--border);
  border-radius:10px;overflow:hidden;cursor:pointer;
  transition:border-color .2s,transform .2s;
}
.plot-card:hover{border-color:var(--accent);transform:translateY(-3px)}
.plot-card img{width:100%;height:180px;object-fit:cover;display:block}
.plot-caption{padding:12px 14px}
.plot-caption strong{font-size:12px;display:block;margin-bottom:4px}
.plot-caption span{font-family:var(--mono);font-size:10px;color:var(--muted)}

/* ── modal ── */
.modal-overlay{
  display:none;position:fixed;inset:0;background:rgba(0,0,0,.85);
  z-index:999;align-items:center;justify-content:center;
}
.modal-overlay.open{display:flex}
.modal-overlay img{max-width:90vw;max-height:90vh;border-radius:8px;border:1px solid var(--border)}
.modal-close{
  position:absolute;top:20px;right:28px;
  background:var(--card);border:1px solid var(--border);
  color:var(--txt);width:36px;height:36px;border-radius:50%;
  font-size:18px;cursor:pointer;display:flex;align-items:center;justify-content:center;
}

/* ── design section ── */
.design-placeholder{
  background:var(--card);border:1px dashed var(--border);
  border-radius:12px;padding:40px;text-align:center;
}
.design-placeholder code{
  display:block;margin-top:12px;font-family:var(--mono);
  font-size:12px;color:var(--accent);
  background:rgba(0,212,255,.06);padding:8px 16px;border-radius:6px;
  display:inline-block;
}

/* ── validation panels ── */
.panel-grid{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:24px}
.panel{background:var(--card);border:1px solid var(--border);border-radius:12px;padding:20px}
.panel h3{font-size:13px;font-weight:700;margin-bottom:16px}
.verdict-big{font-size:48px;font-weight:800;font-family:var(--mono);margin:8px 0}
.verdict-big.pass{color:var(--green)}
.verdict-big.warn{color:var(--yellow)}
.verdict-big.fail{color:var(--red)}
.rule-item{
  display:flex;justify-content:space-between;align-items:center;
  padding:8px 0;border-bottom:1px solid rgba(30,45,66,.5);
  font-family:var(--mono);font-size:11px;
}
.rule-item:last-child{border-bottom:none}
.warn-list{list-style:none;padding:0}
.warn-list li{
  padding:6px 10px;margin-bottom:4px;
  background:rgba(245,158,11,.06);border-left:2px solid var(--yellow);
  border-radius:0 4px 4px 0;font-family:var(--mono);font-size:11px;color:var(--yellow);
}

/* ── skeleton ── */
.skeleton{
  background:linear-gradient(90deg,var(--card) 25%,var(--border) 50%,var(--card) 75%);
  background-size:200% 100%;animation:shimmer 1.5s infinite;
  border-radius:6px;
}
@keyframes shimmer{0%{background-position:200% 0}100%{background-position:-200% 0}}

/* ── responsive ── */
@media(max-width:900px){
  .chart-row{grid-template-columns:1fr}
  .panel-grid{grid-template-columns:1fr}
  .t-entry{grid-template-columns:40px 1fr auto auto}
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
  <button class="refresh-btn" onclick="loadAll()">⟳ Refresh</button>
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
    loadPlots(),
  ]);
}

// ─── status ──────────────────────────────────────────────────────────────────
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
  document.getElementById('tb-dataset').textContent =
    `Dataset: ${d.dataset_rows || '—'} rows`;

  const verdictBadge = v => `<span class="badge ${(v||'').toLowerCase()}">${v||'—'}</span>`;
  const pct = n => n ? (n*100).toFixed(2)+'%' : '—';
  const num = (n,dec=4) => n!=null ? (+n).toFixed(dec) : '—';

  document.getElementById('overview-cards').innerHTML = `
    <div class="card">
      <div class="card-label">01 · Baseline</div>
      <div class="card-title">${b.model_name||'RandomForestRegressor'}</div>
      <div class="metric"><div class="metric-label">CV RMSE</div><div class="metric-value">${num(b.cv_rmse||b.rmse,4)}</div></div>
      <div class="metric"><div class="metric-label">CV R²</div><div class="metric-value sm">${num(b.cv_r2||b.r2,4)}</div></div>
      <div class="metric"><div class="metric-label">Composite</div><div class="metric-value sm">${num(b.cv_composite||b.composite,4)}</div></div>
      <div style="margin-top:8px">${verdictBadge(b.validation_verdict||b.validation)}</div>
    </div>
    <div class="card">
      <div class="card-label">02 · Best Model</div>
      <div class="card-title">${best.model_name||'—'}</div>
      <div class="metric"><div class="metric-label">CV RMSE</div><div class="metric-value">${num(best.cv_rmse||best.rmse,4)}</div></div>
      <div class="metric"><div class="metric-label">Holdout R²</div><div class="metric-value sm">${num(best.holdout_r2||best.r2,4)}</div></div>
      <div class="metric"><div class="metric-label">Composite</div><div class="metric-value sm">${num(best.holdout_composite||best.composite,4)}</div></div>
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
      <div class="metric"><div class="metric-label">Failed Samples</div><div class="metric-value" style="color:${(best.failed_samples||0)>0?'var(--red)':'var(--green)'}">${best.failed_samples??'—'}</div></div>
      <div class="metric"><div class="metric-label">Suspicious Samples</div><div class="metric-value sm">${best.suspicious_samples??'—'}</div></div>
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
        <span class="t-hp">${t.hyperparams||''}</span>
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
    .map(([k,v])=>`<span class="tag">${k.replace('params_','')}: ${isNaN(v)?v:parseFloat(v).toFixed?.(3)}</span>`)
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
      <td>${modelName(row)}</td>
      <td>${hp(row)}</td>
      <td>${isNaN(rmse)?rmse:parseFloat(rmse).toFixed(4)}</td>
      <td>${isNaN(mae)?mae:parseFloat(mae).toFixed(4)}</td>
      <td>${isNaN(r2)?r2:parseFloat(r2).toFixed(4)}</td>
      <td>${isNaN(comp)?comp:parseFloat(comp).toFixed(4)}</td>
      <td><span class="badge ${val.toLowerCase()}">${val}</span></td>
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
function renderCharts(){
  const best = overviewData.best || {};
  const baseline = overviewData.baseline || {};
  const baseComp = +(baseline.cv_composite||baseline.composite||0);

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
        {label:'Composite Score',data:comps,borderColor:'#00d4ff',backgroundColor:'rgba(0,212,255,.06)',borderWidth:1.5,pointRadius:2,tension:.3},
        {label:'Baseline',data:labels.map(()=>baseComp),borderColor:'#f59e0b',borderDash:[5,5],borderWidth:1.5,pointRadius:0},
        {label:'Best',data:labels.map((l,i)=>i===bestIdx?comps[i]:null),borderColor:'transparent',backgroundColor:'#00e5a0',pointRadius:8,pointStyle:'circle',showLine:false},
      ]
    },
    options:{...chartDefaults(),plugins:{legend:{labels:{color:'#94a3b8',font:{size:10}}}}}
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
    data:{labels:famLabels,datasets:[{label:'Mean Composite',data:famMeans,backgroundColor:famColors.map(c=>c+'33'),borderColor:famColors,borderWidth:1.5,borderRadius:4}]},
    options:{...chartDefaults(),plugins:{legend:{display:false}}}
  });

  // RMSE distribution
  const famRmse = famLabels.map(f=>{
    const vals = allOptuna.filter(r=>(r.params_model_name||r['params_model']||'Other')===f).map(r=>+(r.value||r.rmse||0));
    return avg(vals);
  });
  if(chartRmse) chartRmse.destroy();
  chartRmse = new Chart(document.getElementById('chart-rmse'),{
    type:'bar',
    data:{labels:famLabels,datasets:[{label:'Mean RMSE',data:famRmse,backgroundColor:famColors.map(c=>c+'33'),borderColor:famColors,borderWidth:1.5,borderRadius:4}]},
    options:{...chartDefaults(),plugins:{legend:{display:false}}}
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
          {label:'Baseline',data:[baseRng.low,baseRng.mid,baseRng.high],backgroundColor:'rgba(0,212,255,.2)',borderColor:'#00d4ff',borderWidth:1.5,borderRadius:4},
          {label:'Best Model',data:[rng.low,rng.mid,rng.high],backgroundColor:'rgba(0,229,160,.2)',borderColor:'#00e5a0',borderWidth:1.5,borderRadius:4},
        ]
      },
      options:{...chartDefaults(),plugins:{legend:{labels:{color:'#94a3b8',font:{size:10}}}}}
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
      datasets:[{data:[bComp, Math.max(0,bestComp-bComp)],backgroundColor:['rgba(0,212,255,.3)','rgba(0,229,160,.5)'],borderColor:['#00d4ff','#00e5a0'],borderWidth:2}]
    },
    options:{
      cutout:'72%',
      plugins:{legend:{labels:{color:'#94a3b8',font:{size:10}}},
      tooltip:{callbacks:{label:c=>`${c.label}: ${(+c.raw).toFixed(4)}`}}}
    }
  });
}

function modelColor(m){
  if(!m) return '#94a3b8';
  const ml = m.toLowerCase();
  if(ml.includes('random')) return '#60a5fa';
  if(ml.includes('gradient')) return '#fb923c';
  if(ml.includes('xgb')) return '#f43f5e';
  if(ml.includes('lgbm')||ml.includes('light')) return '#00e5a0';
  if(ml.includes('svr')) return '#a78bfa';
  return '#94a3b8';
}

function avg(arr){ return arr.length ? arr.reduce((a,b)=>a+b,0)/arr.length : 0; }

function chartDefaults(){
  return {
    responsive:true,
    plugins:{tooltip:{backgroundColor:'#1e2130',borderColor:'#1e2d42',borderWidth:1,titleColor:'#e2eaf4',bodyColor:'#94a3b8',titleFont:{family:'Space Mono',size:11},bodyFont:{family:'Space Mono',size:11}}},
    scales:{x:{grid:{color:'rgba(30,45,66,.5)'},ticks:{color:'#4a6080',font:{family:'Space Mono',size:10}}},y:{grid:{color:'rgba(30,45,66,.5)'},ticks:{color:'#4a6080',font:{family:'Space Mono',size:10}}}}
  };
}

// ─── validation ───────────────────────────────────────────────────────────────
async function loadValidation() {
  const d = await fetch('/api/validation_details').then(r=>r.json()).catch(()=>({}));
  const v = d.validation_verdict || d.validation || '—';
  const fail = d.failed_samples ?? '—';
  const sus  = d.suspicious_samples ?? '—';
  const warns = d.warn_reasons || [];

  document.getElementById('val-summary').innerHTML = `
    <h3>Validation Summary</h3>
    <div class="verdict-big ${v.toLowerCase()}">${v}</div>
    <div class="metric"><div class="metric-label">Hard Failures</div><div class="metric-value sm" style="color:${fail>0?'var(--red)':'var(--green)'}">${fail}</div></div>
    <div class="metric"><div class="metric-label">Suspicious Samples</div><div class="metric-value sm" style="color:${sus>0?'var(--yellow)':'var(--green)'}">${sus}</div></div>
    <div class="metric"><div class="metric-label">Model</div><div class="metric-value sm">${d.model_name||'—'}</div></div>
  `;

  document.getElementById('val-warnings').innerHTML = `
    <h3>Warning Reasons</h3>
    ${warns.length
      ? `<ul class="warn-list">${warns.map(w=>`<li>${w}</li>`).join('')}</ul>`
      : `<div style="color:var(--green);font-family:var(--mono);font-size:12px;margin-top:8px">✓ No warnings triggered</div>`
    }
    <div style="margin-top:16px;padding-top:16px;border-top:1px solid var(--border)">
      <div class="metric-label" style="margin-bottom:8px">RULE REFERENCE</div>
      ${[
        ['w/c > 0.60','Durability limit (moderate exposure)','WARN'],
        ['w/c > 0.70','Hard engineering flag','WARN'],
        ['Binder < 250 kg/m³','Low binder content','WARN'],
        ['Binder > 550 kg/m³','Shrinkage risk','WARN'],
        ['Fly ash > 40%','Exceeds ACI substitution limit','WARN'],
        ['Slag > 70%','Exceeds BS 8500 GGBS limit','WARN'],
        ['Predicted < 0 MPa','Physical impossibility','FAIL'],
        ['Predicted > 120 MPa','Outside normal concrete range','FAIL'],
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
        <pre style="white-space:pre-wrap;color:var(--muted)">${JSON.stringify(s,null,2)}</pre>
      </div>
    `).join('')}</div>`;
  }
  el.innerHTML = html;
}

// ─── gallery ─────────────────────────────────────────────────────────────────
async function loadPlots() {
  const plots = await fetch('/api/plots').then(r=>r.json()).catch(()=>[]);
  document.getElementById('plot-grid').innerHTML = plots.length
    ? plots.map(p=>`
      <div class="plot-card" onclick="openModal('/outputs/${p.filename}')">
        <img src="/outputs/${p.filename}" alt="${p.filename}" loading="lazy"/>
        <div class="plot-caption">
          <strong>${p.filename}</strong>
          <span>${p.description}</span>
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
    def open_browser():
        webbrowser.open("http://localhost:5050")
    threading.Timer(1.2, open_browser).start()
    print("\n  ◈ AutoCivil-Lab Dashboard")
    print("  → http://localhost:5050\n")
    app.run(debug=False, port=5050)
