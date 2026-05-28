#!/usr/bin/env python3
"""
compare_performance.py — Full Cross-participant Flight Performance Comparison
==============================================================================
Compare-type script: orchestrates all three performance comparison analyses
(aviate, navigate, time) for ALL participants in one pass.
Run from the repository root:
    python HITLS/compare_performance.py

The script will show a summary of what will be generated or overwritten and
ask for confirmation before doing any work.

  1. Ensures aviate_perf, navigate_perf, and time_perf reports exist for
     every participant (regenerates any that are missing).
  2. Runs all three compare scripts' plot functions in sequence:

       Aviate (slip · roll · airspeed during climb)
         aviate_boxplots.png
         aviate_rmse_distributions.png
         aviate_nmae_distributions.png

       Navigate (XTE · ATD · heading · altitude)
         navigate_boxplots.png
         navigate_rmse_distributions.png
         navigate_nmae_distributions.png

       Time (scenario duration · failure→nominal · per-procedure timing)
         time_boxplots.png
         time_distributions.png
         time_mean_task_distributions.png

All figures are saved to HITLS/plots/.
"""

import os
import sys
import json
import importlib.util
import subprocess
import numpy as np
from scipy.stats import friedmanchisquare, wilcoxon as _sp_wilcoxon
from statsmodels.stats.multitest import multipletests

# ── Paths ─────────────────────────────────────────────────────────────────────
HITLS_DIR = os.path.dirname(os.path.abspath(__file__))
PERF_DIR  = os.path.join(HITLS_DIR, "performance")
PLOTS_DIR = os.path.join(HITLS_DIR, "plots")
PYTHON    = sys.executable

# ── Plots produced by this script (across all three domains) ──────────────────
_ALL_PLOTS = [
    "aviate_boxplots.png",
    "aviate_rmse_distributions.png",
    "aviate_nmae_distributions.png",
    "navigate_boxplots.png",
    "navigate_rmse_distributions.png",
    "navigate_nmae_distributions.png",
    "time_boxplots.png",
    "time_distributions.png",
    "time_mean_task_distributions.png",
]

# ── Reports produced by this script ───────────────────────────────────────────
REPORTS_DIR  = os.path.join(HITLS_DIR, "compare_performance")
_ALL_REPORTS = ["aviate_report.txt", "navigate_report.txt", "time_report.txt"]

# ── Pairwise comparison pairs ──────────────────────────────────────────────────
_ALL_PAIRS = [("TARS", "TARP-S"), ("TARS", "TARP-F"), ("TARP-S", "TARP-F"), ("TARS", "TARC")]


# ═══════════════════════════════════════════════════════════════════════════════
#  Statistical helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _wilcoxon_test(a, b):
    """Paired Wilcoxon signed-rank test. Returns (p_value, rank_biserial_r).
    Drops pairs where either value is None.  Minimum 4 valid pairs required.
    """
    pairs = [(float(x), float(y)) for x, y in zip(a, b)
             if x is not None and y is not None]
    if len(pairs) < 4:
        return 1.0, 0.0
    xa, xb = np.array([p[0] for p in pairs]), np.array([p[1] for p in pairs])
    try:
        res = _sp_wilcoxon(xa, xb, alternative="two-sided", zero_method="wilcox")
        n   = len(pairs)
        r   = 1.0 - 2.0 * res.statistic / (n * (n + 1) / 2.0)
        return float(res.pvalue), float(r)
    except Exception:
        return 1.0, 0.0


def _friedman_test(groups):
    """Friedman chi-square for k groups (list of lists, equal length).
    Drops rows with any None across conditions.  Returns (chi2, p).
    """
    rows = list(zip(*groups))
    valid = [r for r in rows if all(x is not None for x in r)]
    if len(valid) < 3:
        return 0.0, 1.0
    aligned = [[float(r[i]) for r in valid] for i in range(len(groups))]
    try:
        res = friedmanchisquare(*aligned)
        return float(res.statistic), float(res.pvalue)
    except Exception:
        return 0.0, 1.0


def _holm_correct(pvals, alpha=0.05):
    """Holm-Bonferroni correction. Returns (reject_array, pvals_corrected)."""
    pvals = list(pvals)
    if not pvals:
        return np.array([], dtype=bool), np.array([])
    reject, pvals_corr, _, _ = multipletests(pvals, alpha=alpha, method="holm")
    return reject, pvals_corr


def _sig_stars(p_raw, sig):
    """Return significance label string (raw p + Holm rejection)."""
    if not sig:
        return ""
    if p_raw < 0.001:
        return "***"
    if p_raw < 0.01:
        return "**"
    if p_raw < 0.05:
        return "*"
    return "†"


def _fmt_stat_section(label, friedman, pairwise, diff_pairs):
    """Format a stats block for one metric.

    friedman : {'chi2': float, 'p': float, 'df': int}
    pairwise : list of dicts per pair (same order as diff_pairs)
    """
    lines = [f"  {label}"]
    chi2, p_f, df = friedman['chi2'], friedman['p'], friedman['df']
    p_f_str = f"{p_f:.4f}" if p_f >= 0.0001 else f"{p_f:.2e}"
    lines.append(f"    Friedman \u03c7\u00b2({df}) = {chi2:.3f},  p = {p_f_str}")
    for pi, (bl, comp) in enumerate(diff_pairs):
        pw = pairwise[pi]
        p_raw = pw['p_raw']
        p_cor = pw['p_corr']
        r_val = pw['r']
        sig   = "\u2713" if pw['reject'] else " "
        p_raw_str = f"{p_raw:.4f}" if p_raw >= 0.0001 else f"{p_raw:.2e}"
        p_cor_str = f"{p_cor:.4f}" if p_cor >= 0.0001 else f"{p_cor:.2e}"
        stars = pw['stars'] or "ns"
        lines.append(
            f"    [{sig}] {comp} \u2212 {bl:<8}  W p={p_raw_str}  p_Holm={p_cor_str}  r={r_val:+.3f}  {stars}"
        )
    return "\n".join(lines) + "\n"


def _run_stats(vals_by_cond, conditions, pairs):
    """Run Friedman + Wilcoxon + Holm for one metric. Returns (frd_dict, pairwise_list)."""
    chi2, p_f = _friedman_test([vals_by_cond[c] for c in conditions])
    frd = {"chi2": chi2, "p": p_f, "df": len(conditions) - 1}
    ps_raw, rs = [], []
    for bl, comp in pairs:
        p, r = _wilcoxon_test(vals_by_cond[bl], vals_by_cond[comp])
        ps_raw.append(p)
        rs.append(r)
    reject, p_corr = _holm_correct(ps_raw)
    pairwise = [
        {"p_raw": ps_raw[i], "r": rs[i], "reject": bool(reject[i]),
         "p_corr": float(p_corr[i]), "stars": _sig_stars(ps_raw[i], bool(reject[i]))}
        for i in range(len(pairs))
    ]
    return frd, pairwise


# ═══════════════════════════════════════════════════════════════════════════════
#  Report helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _desc(vals):
    if not vals:
        return {"n": 0, "mean": None, "sd": None, "median": None, "min": None, "max": None}
    a = np.array([v for v in vals if v is not None], dtype=float)
    if len(a) == 0:
        return {"n": 0, "mean": None, "sd": None, "median": None, "min": None, "max": None}
    return {
        "n":      int(len(a)),
        "mean":   round(float(np.mean(a)), 4),
        "sd":     round(float(np.std(a, ddof=1)), 4) if len(a) > 1 else 0.0,
        "median": round(float(np.median(a)), 4),
        "min":    round(float(np.min(a)), 4),
        "max":    round(float(np.max(a)), 4),
    }


def _collect_vals(all_data, participants, conditions, *keys):
    """Collect per-condition value lists for a given key path."""
    result = {c: [] for c in conditions}
    for pid in participants:
        d = all_data.get(pid, {})
        for cond in conditions:
            node = d.get("conditions", {}).get(cond, {})
            for k in keys:
                if not isinstance(node, dict):
                    node = None
                    break
                node = node.get(k)
            if node is not None:
                try:
                    result[cond].append(float(node))
                except (TypeError, ValueError):
                    pass
    return result


def _write_report(path, title, summary_json, sections):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("--- MACHINE-READABLE SUMMARY (JSON) ---\n")
        f.write(json.dumps(summary_json, indent=2, ensure_ascii=False))
        f.write("\n--- END SUMMARY ---\n")
        f.write("=" * 78 + "\n")
        f.write(f"  {title}\n")
        f.write("=" * 78 + "\n\n")
        for sec in sections:
            f.write(sec)


def _cond_header(label_w, conditions):
    h = " " * (label_w + 2)
    for c in conditions:
        h += f"  {c:^22}"
    return h + "\n" + " " * (label_w + 2) + "  " + ("─" * 22 + "  ") * len(conditions) + "\n"


def _fmt_row(label_w, label, stats_by_cond, conditions):
    row = f"  {label:<{label_w}}"
    for c in conditions:
        s = stats_by_cond.get(c, {})
        if not s or s.get("n", 0) == 0:
            row += f"  {'—':^22}"
        else:
            row += f"  μ={s['mean']:6.3f} σ={s['sd']:5.3f} Md={s['median']:6.3f}"
    return row + "\n"


# ═══════════════════════════════════════════════════════════════════════════════
#  Aviate report
# ═══════════════════════════════════════════════════════════════════════════════

_AVIATE_CONDITIONS = ["TARS", "TARC", "TARP-S", "TARP-F"]

_AVIATE_SIGNALS = [
    # (json_key, label, metrics)
    ("slip",           "Slip",          ["rmse", "nmae", "rmse_mae", "window_s"]),
    ("roll_angle",     "Roll Angle",    ["rmse", "nmae", "rmse_mae", "window_s"]),
    ("airspeed_climb", "Airspeed Climb",["rmse", "mape", "nmae", "rmse_mae", "window_s"]),
]

_AVIATE_METRIC_LABELS = {
    "rmse":     "RMSE",
    "nmae":     "nMAE (MAE/window)",
    "mape":     "MAPE (%)",
    "rmse_mae": "RMSE/MAE ratio",
    "window_s": "Window duration (s)",
}


def write_aviate_report(aviate_data, participants):
    conditions = _AVIATE_CONDITIONS

    # Collect all values up front
    vals_cache = {}  # (sig_key, met) -> {cond: [vals]}
    summary_signals = {}
    for sig_key, sig_label, metrics in _AVIATE_SIGNALS:
        summary_signals[sig_key] = {}
        for met in metrics:
            v = _collect_vals(aviate_data, participants, conditions, sig_key, met)
            vals_cache[(sig_key, met)] = v
            summary_signals[sig_key][met] = {c: _desc(v[c]) for c in conditions}

    summary = {
        "domain": "aviate",
        "n_participants": len(participants),
        "conditions": conditions,
        "signals": summary_signals,
    }

    w = 24
    sections = []
    for sig_key, sig_label, metrics in _AVIATE_SIGNALS:
        sec = f"{'─' * 78}\n  {sig_label.upper()}\n{'─' * 78}\n"
        sec += _cond_header(w, conditions)
        for met in metrics:
            label = _AVIATE_METRIC_LABELS.get(met, met)
            stats_by_cond = summary_signals[sig_key][met]
            sec += _fmt_row(w, label, stats_by_cond, conditions)
        sec += "\n"
        sections.append(sec)

    # ── Statistical analysis: Slip & Roll Angle (RMSE + nMAE) ────────────────
    stat_sec  = f"{'─' * 78}\n"
    stat_sec += "  STATISTICAL ANALYSIS  (Friedman + Wilcoxon + Holm-Bonferroni)\n"
    stat_sec += "  Signals: Slip, Roll Angle  |  Metrics: RMSE, nMAE\n"
    stat_sec += "  Non-parametric within-subjects tests across 4 conditions\n"
    stat_sec += f"{'─' * 78}\n\n"
    for sig_key, sig_label in [("slip", "Slip"), ("roll_angle", "Roll Angle")]:
        stat_sec += f"  ── {sig_label} ──\n"
        for met, met_label in [("rmse", "RMSE"), ("nmae", "nMAE")]:
            frd, pw = _run_stats(vals_cache[(sig_key, met)], conditions, _ALL_PAIRS)
            stat_sec += _fmt_stat_section(f"{sig_label} — {met_label}", frd, pw, _ALL_PAIRS)
        stat_sec += "\n"
    sections.append(stat_sec)

    path = os.path.join(REPORTS_DIR, "aviate_report.txt")
    _write_report(path, "Aviate Performance  (cross-participant)", summary, sections)
    print(f"  → {path}")


# ═══════════════════════════════════════════════════════════════════════════════
#  Navigate report
# ═══════════════════════════════════════════════════════════════════════════════

_NAVIGATE_CONDITIONS = ["TARS", "TARC", "TARP-S", "TARP-F"]

_NAVIGATE_PHASES = [
    # (phase_key, phase_label, [(metric_path, metric_label), ...])
    ("climb", "CLIMB PHASE", [
        (("xte",          "rmse"),    "XTE RMSE (NM)"),
        (("xte",          "nmae"),    "XTE nMAE (NM/s)"),
        (("xte",          "rmse_mae"),"XTE RMSE/MAE ratio"),
        (("atd_error",    "rmse"),    "ATD RMSE (NM)"),
        (("atd_error",    "nmae"),    "ATD nMAE (NM/s)"),
        (("atd_error",    "rmse_mae"),"ATD RMSE/MAE ratio"),
        (("heading_error","rmse"),    "Heading RMSE (deg)"),
        (("heading_error","nmae"),    "Heading nMAE (deg/s)"),
        (("heading_error","rmse_mae"),"Heading RMSE/MAE ratio"),
        (("window_s",),               "Window duration (s)"),
    ]),
    ("leveloff", "LEVEL-OFF PHASE", [
        (("alt_error", "rmse"),    "Alt RMSE (ft)"),
        (("alt_error", "nmae"),    "Alt nMAE (ft/s)"),
        (("alt_error", "rmse_mae"),"Alt RMSE/MAE ratio"),
        (("window_s",),             "Window duration (s)"),
    ]),
]


def write_navigate_report(navigate_data, participants):
    conditions = _NAVIGATE_CONDITIONS

    # Collect all values up front
    vals_cache = {}  # (phase_key, path_keys) -> {cond: [vals]}
    summary_phases = {}
    for phase_key, _, metric_specs in _NAVIGATE_PHASES:
        summary_phases[phase_key] = {}
        for path_keys, label in metric_specs:
            key = "_".join(path_keys)
            v = _collect_vals(navigate_data, participants, conditions, phase_key, *path_keys)
            vals_cache[(phase_key, path_keys)] = v
            summary_phases[phase_key][key] = {c: _desc(v[c]) for c in conditions}

    summary = {
        "domain": "navigate",
        "n_participants": len(participants),
        "conditions": conditions,
        "phases": summary_phases,
    }

    w = 28
    sections = []
    for phase_key, phase_label, metric_specs in _NAVIGATE_PHASES:
        sec = f"{'─' * 78}\n  {phase_label}\n{'─' * 78}\n"
        sec += _cond_header(w, conditions)
        for path_keys, label in metric_specs:
            key = "_".join(path_keys)
            stats_by_cond = summary_phases[phase_key][key]
            sec += _fmt_row(w, label, stats_by_cond, conditions)
        sec += "\n"
        sections.append(sec)

    # ── Statistical analysis: XTE, ATD, Heading in climb phase (RMSE + nMAE) ─
    stat_sec  = f"{'─' * 78}\n"
    stat_sec += "  STATISTICAL ANALYSIS  (Friedman + Wilcoxon + Holm-Bonferroni)\n"
    stat_sec += "  Signals: XTE, ATD error, Heading error (climb phase)  |  Metrics: RMSE, nMAE\n"
    stat_sec += "  Non-parametric within-subjects tests across 4 conditions\n"
    stat_sec += f"{'─' * 78}\n\n"
    for sig_key, sig_label in [
        ("xte",           "XTE"),
        ("atd_error",     "ATD Error"),
        ("heading_error", "Heading Error"),
    ]:
        stat_sec += f"  ── {sig_label} (climb) ──\n"
        for met, met_label in [("rmse", "RMSE"), ("nmae", "nMAE")]:
            v = vals_cache[("climb", (sig_key, met))]
            frd, pw = _run_stats(v, conditions, _ALL_PAIRS)
            stat_sec += _fmt_stat_section(f"{sig_label} — {met_label}", frd, pw, _ALL_PAIRS)
        stat_sec += "\n"
    sections.append(stat_sec)

    path = os.path.join(REPORTS_DIR, "navigate_report.txt")
    _write_report(path, "Navigate Performance  (cross-participant)", summary, sections)
    print(f"  → {path}")


# ═══════════════════════════════════════════════════════════════════════════════
#  Time report
# ═══════════════════════════════════════════════════════════════════════════════

_TIME_CONDITIONS = ["TARS", "TARC", "TARP-S", "TARP-F"]

_TIME_TOP = [
    ("scenario_duration_s",   "Scenario duration (s)"),
    ("failure_to_nominal_s",  "Failure → nominal recovery (s)"),
]

_TIME_PROCS = [
    ("crew_briefing",  "Crew Briefing"),
    ("before_takeoff", "Before Takeoff"),
    ("lineup_hold",    "Lineup & Hold"),
    ("takeoff",        "Takeoff"),
    ("eng_failure",    "ENG Failure during Takeoff"),
    ("engine_fire",    "ENGINE FIRE"),
    ("declare_panpan", "Declare PANPAN"),
    ("after_takeoff",  "After Takeoff"),
]


def write_time_report(time_data, participants):
    conditions = _TIME_CONDITIONS

    # Collect top-level durations
    top_vals_cache = {}  # key -> {cond: [vals]}
    top_stats = {}
    for key, label in _TIME_TOP:
        v = _collect_vals(time_data, participants, conditions, key)
        top_vals_cache[key] = v
        top_stats[key] = {c: _desc(v[c]) for c in conditions}

    # Collect per-procedure metrics
    proc_vals_cache = {}  # (proc_key, met) -> {cond: [vals]}
    proc_stats = {}
    for proc_key, _ in _TIME_PROCS:
        proc_stats[proc_key] = {}
        for met in ("n_tasks", "total_s", "mean_task_s"):
            v = _collect_vals(time_data, participants, conditions, proc_key, met)
            proc_vals_cache[(proc_key, met)] = v
            proc_stats[proc_key][met] = {c: _desc(v[c]) for c in conditions}

    summary = {
        "domain": "time",
        "n_participants": len(participants),
        "conditions": conditions,
        "top_level": top_stats,
        "procedures": proc_stats,
    }

    w = 36
    sections = []

    # Section 1 — top-level descriptives
    sec1 = f"{'─' * 78}\n  SCENARIO-LEVEL DURATIONS\n{'─' * 78}\n"
    sec1 += _cond_header(w, conditions)
    for key, label in _TIME_TOP:
        sec1 += _fmt_row(w, label, top_stats[key], conditions)
    sec1 += "\n"
    sections.append(sec1)

    # Section 2 — per-procedure descriptives
    sec2 = f"{'─' * 78}\n  PER-PROCEDURE TIMING\n{'─' * 78}\n"
    for proc_key, proc_label in _TIME_PROCS:
        sec2 += f"\n  [{proc_label}]\n"
        sec2 += _cond_header(w, conditions)
        for met, met_label in [("n_tasks", "n tasks"), ("total_s", "Total (s)"), ("mean_task_s", "Mean task (s)")]:
            sec2 += _fmt_row(w, met_label, proc_stats[proc_key][met], conditions)
    sec2 += "\n"
    sections.append(sec2)

    # ── Statistical analysis ─────────────────────────────────────────────────
    stat_sec  = f"{'─' * 78}\n"
    stat_sec += "  STATISTICAL ANALYSIS  (Friedman + Wilcoxon + Holm-Bonferroni)\n"
    stat_sec += "  Non-parametric within-subjects tests across 4 conditions\n"
    stat_sec += f"{'─' * 78}\n\n"

    # Scenario-level: scenario_duration_s + failure_to_nominal_s
    stat_sec += "  ── Scenario-level durations ──\n"
    for key, label in [
        ("scenario_duration_s",  "Scenario duration (s)"),
        ("failure_to_nominal_s", "Failure \u2192 nominal (s)"),
    ]:
        frd, pw = _run_stats(top_vals_cache[key], conditions, _ALL_PAIRS)
        stat_sec += _fmt_stat_section(label, frd, pw, _ALL_PAIRS)
    stat_sec += "\n"

    # Per-procedure: total_s (procedure duration) + mean_task_s (avg task duration)
    stat_sec += "  ── Per-procedure: duration & mean task duration ──\n\n"
    for proc_key, proc_label in _TIME_PROCS:
        stat_sec += f"  [{proc_label}]\n"
        for met, met_label in [
            ("total_s",      "Procedure duration (s)"),
            ("mean_task_s",  "Mean task duration (s)"),
        ]:
            frd, pw = _run_stats(proc_vals_cache[(proc_key, met)], conditions, _ALL_PAIRS)
            stat_sec += _fmt_stat_section(met_label, frd, pw, _ALL_PAIRS)
        stat_sec += "\n"
    sections.append(stat_sec)

    path = os.path.join(REPORTS_DIR, "time_report.txt")
    _write_report(path, "Time Performance  (cross-participant)", summary, sections)
    print(f"  → {path}")


def write_all_perf_reports(aviate_data, av_pids, navigate_data, nav_pids, time_data, time_pids):
    os.makedirs(REPORTS_DIR, exist_ok=True)
    print(f"\n  Generating performance reports → {REPORTS_DIR}/\n")
    if av_pids:
        write_aviate_report(aviate_data, av_pids)
    if nav_pids:
        write_navigate_report(navigate_data, nav_pids)
    if time_pids:
        write_time_report(time_data, time_pids)


# ═══════════════════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def find_participants():
    return [
        e for e in sorted(os.listdir(HITLS_DIR))
        if os.path.isdir(os.path.join(HITLS_DIR, e))
        and e.startswith("P") and e[1:].isdigit()
    ]


def _load_json(path):
    try:
        txt   = open(path, encoding="utf-8").read()
        start = txt.index("{")
        end   = txt.index("--- END SUMMARY ---")
        return json.loads(txt[start:end].strip())
    except (FileNotFoundError, ValueError, json.JSONDecodeError):
        return None


def _has_aviate(pid):
    data = _load_json(os.path.join(HITLS_DIR, pid, "cleaned", f"{pid}_aviate_perf_report.txt"))
    if not data:
        return False
    for cond_data in data.get("conditions", {}).values():
        return "nmae" in cond_data.get("slip", {})
    return False


def _has_navigate(pid):
    data = _load_json(os.path.join(HITLS_DIR, pid, "cleaned", f"{pid}_navigate_perf_report.txt"))
    if not data:
        return False
    for cond_data in data.get("conditions", {}).values():
        return "nmae" in cond_data.get("climb", {}).get("xte", {})
    return False


def _has_time(pid):
    data = _load_json(os.path.join(HITLS_DIR, pid, "cleaned", f"{pid}_time_perf_report.txt"))
    if not data:
        return False
    for cond_data in data.get("conditions", {}).values():
        return "scenario_duration_s" in cond_data
    return False


def _missing_for(pid):
    """Return list of report kinds missing for this participant."""
    missing = []
    if not _has_aviate(pid):
        missing.append("aviate")
    if not _has_navigate(pid):
        missing.append("navigate")
    if not _has_time(pid):
        missing.append("time")
    return missing


def _run_script(script_path, participant_number):
    proc = subprocess.run(
        [PYTHON, script_path],
        input=f"{participant_number}\n",
        text=True,
        capture_output=True,
    )
    if proc.returncode != 0:
        name = os.path.basename(script_path)
        print(f"  ⚠  {name} returned code {proc.returncode}")
        if proc.stderr:
            print(proc.stderr[:400])


def _load_module(name, filename):
    path = os.path.join(PERF_DIR, filename)
    spec = importlib.util.spec_from_file_location(name, path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ═══════════════════════════════════════════════════════════════════════════════
#  Pre-run confirmation (compare-type script pattern)
# ═══════════════════════════════════════════════════════════════════════════════

def _confirm_run(participants, missing_map, output_plots):
    """Print a pre-run summary and ask the user to confirm before proceeding."""
    print()
    total_missing = sum(len(v) for v in missing_map.values())
    if total_missing:
        print(f"Reports to generate ({total_missing} missing across participants):")
        for pid in participants:
            kinds = missing_map.get(pid, [])
            if kinds:
                print(f"  + {pid}: {', '.join(k + '_perf' for k in kinds)}")
    else:
        print("All participant reports are up to date.")
    print(f"\nOutput plots that will be written/overwritten ({len(output_plots)}):")
    for name in output_plots:
        path = os.path.join(PLOTS_DIR, name)
        tag  = "[overwrite]" if os.path.exists(path) else "[new     ]"
        print(f"  {tag}  {name}")
    print(f"\nCross-participant reports that will be written/overwritten ({len(_ALL_REPORTS)}):")
    for name in _ALL_REPORTS:
        path = os.path.join(REPORTS_DIR, name)
        tag  = "[overwrite]" if os.path.exists(path) else "[new     ]"
        print(f"  {tag}  compare_performance/{name}")
    print()
    try:
        ans = input("Continue? [Y/n]: ").strip().lower()
    except KeyboardInterrupt:
        print("\nAborted.")
        return False
    return ans in ("", "y", "yes")


# ═══════════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    sep = "=" * 70
    print(f"\n{sep}")
    print("  HITLS — Cross-participant Flight Performance Comparison")
    print(f"{sep}")

    participants = find_participants()
    print(f"\nParticipants found: {', '.join(participants)}")

    missing_map = {pid: _missing_for(pid) for pid in participants}
    missing_map = {pid: kinds for pid, kinds in missing_map.items() if kinds}

    if not _confirm_run(participants, missing_map, _ALL_PLOTS):
        return

    # ── Step 1: Generate any missing per-participant reports ──────────────────
    _scripts = {
        "aviate":   os.path.join(PERF_DIR, "aviate_perf.py"),
        "navigate": os.path.join(PERF_DIR, "navigate_perf.py"),
        "time":     os.path.join(PERF_DIR, "time_perf.py"),
    }

    n_total = sum(len(v) for v in missing_map.values())
    if n_total:
        print(f"\n[1/4] Generating {n_total} missing report(s) …")
        for i, pid in enumerate(participants, start=1):
            for kind in missing_map.get(pid, []):
                print(f"  {pid} / {kind}_perf …", flush=True)
                _run_script(_scripts[kind], str(i))
    else:
        print("\n[1/4] All reports up to date.")

    # ── Step 2: Load the three compare modules ────────────────────────────────
    print("\n[2/4] Loading compare modules …")
    cmp_aviate   = _load_module("compare_aviate",   "compare_aviate.py")
    cmp_navigate = _load_module("compare_navigate", "compare_navigate.py")
    cmp_time     = _load_module("compare_time",     "compare_time.py")

    # ── Step 3: Load data ─────────────────────────────────────────────────────
    print("\n[3/4] Loading data …")
    aviate_data   = cmp_aviate.load_all(participants)
    navigate_data = cmp_navigate.load_all(participants)
    time_data     = cmp_time.load_all(participants)

    av_pids   = list(aviate_data.keys())
    nav_pids  = list(navigate_data.keys())
    time_pids = list(time_data.keys())

    print(f"  Aviate:   {len(av_pids)} participant(s) — {', '.join(av_pids) or 'none'}")
    print(f"  Navigate: {len(nav_pids)} participant(s) — {', '.join(nav_pids) or 'none'}")
    print(f"  Time:     {len(time_pids)} participant(s) — {', '.join(time_pids) or 'none'}")

    if not av_pids and not nav_pids and not time_pids:
        print("  No data available — aborting.")
        return

    os.makedirs(PLOTS_DIR, exist_ok=True)
    print(f"\n[4/4] Generating charts → {PLOTS_DIR}/\n")

    figs = []

    # ── Aviate ────────────────────────────────────────────────────────────────
    if av_pids:
        print("  ── Aviate (slip · roll · airspeed) ──")
        figs.append(cmp_aviate.plot_boxplots(aviate_data, av_pids))
        figs.append(cmp_aviate.plot_rmse_distributions(aviate_data, av_pids))
        figs.append(cmp_aviate.plot_nmae_distributions(aviate_data, av_pids))
    else:
        print("  ── Aviate: no data, skipping.")

    # ── Navigate ──────────────────────────────────────────────────────────────
    if nav_pids:
        print("  ── Navigate (XTE · ATD · heading · altitude) ──")
        figs.append(cmp_navigate.plot_boxplots(navigate_data, nav_pids))
        figs.append(cmp_navigate.plot_rmse_distributions(navigate_data, nav_pids))
        figs.append(cmp_navigate.plot_nmae_distributions(navigate_data, nav_pids))
    else:
        print("  ── Navigate: no data, skipping.")

    # ── Time ──────────────────────────────────────────────────────────────────
    if time_pids:
        print("  ── Time (scenario duration · failure→nominal · per-procedure) ──")
        figs.append(cmp_time.plot_boxplots(time_data, time_pids))
        figs.append(cmp_time.plot_distributions(time_data, time_pids))
        figs.append(cmp_time.plot_mean_task_distributions(time_data, time_pids))
    else:
        print("  ── Time: no data, skipping.")

    import matplotlib.pyplot as plt
    n_saved = sum(1 for f in figs if f is not None)
    print(f"\nDone — {n_saved} figure(s) saved to {PLOTS_DIR}/")

    write_all_perf_reports(aviate_data, av_pids, navigate_data, nav_pids, time_data, time_pids)

    plt.show()


if __name__ == "__main__":
    main()
