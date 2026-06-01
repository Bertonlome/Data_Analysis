#!/usr/bin/env python3
"""
HRV/HRV.py — Heart Rate Variability Analysis for HITLS
=======================================================
Analyses HRV data recorded with a Polar H10 chest strap (via Elite HRV app)
and synchronises it with Ingescape simulator logs to extract per-scenario
and per-procedure HRV features.

Synchronisation
---------------
Each Ingescape log contains ``Aircraft;eliteHRV`` events whose value is the
elapsed time in the Elite HRV recording at the moment the simulator is
resumed (``paused`` transitions true→false ~0.3 s later).
Multiple sync anchors per participant allow a robust median estimate of the
HRV recording start time in Unix epoch seconds.

HRV Features (via NeuroKit2, all domains)
------------------------------------------
  Time domain    : MeanNN, SDNN, RMSSD, pNN50, CVNN, CVSD, MeanHR …
  Frequency dom. : VLF, LF, HF, LFHF, LFNU, HFNU … (requires ≥ 120 s)
  Non-linear     : SD1, SD2, DFA α1, SampEn … (DFA requires ≥ 300 beats)

Dynamic / per-procedure workload
---------------------------------
  • Sliding-window RMSSD (60 s window, 30 s step) within each scenario
  • Per-procedure RMSSD computed from ``Shared Interface;force_state_jump``
    state-machine boundaries in the scenario Ingescape CSV.

Outputs
-------
  HRV/hrv_features_per_scenario.csv  — one row per valid scenario
  HRV/hrv_report.txt                 — statistics report (JSON + text)
  HRV/plots/*.png                    — condition box-plots, per-procedure
                                       profiles, per-participant heatmaps

Usage
-----
  python HITLS/HRV/HRV.py          # interactive (asks for confirmation)
  python HITLS/HRV/HRV.py --force  # skip confirmation prompt

References
----------
  Pham et al. 2021 — HRV in Psychology (NeuroKit2 tutorial)
  Arutyunova et al. 2024 — HRV for cognitive load in driving simulation
  Izzah et al. 2023 — Physiological signals as workload predictors
  Luque-Casado et al. 2016 — HRV and cognitive processing
  Solhjoo et al. 2019 — HRV and clinical reasoning performance
"""

import argparse
import csv
import json
import os
import re
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import neurokit2 as nk
from scipy.stats import friedmanchisquare, wilcoxon as _sp_wilcoxon
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ── Paths ─────────────────────────────────────────────────────────────────────
HRV_DIR   = os.path.dirname(os.path.abspath(__file__))
HITLS_DIR = os.path.dirname(HRV_DIR)
PLOTS_DIR = os.path.join(HRV_DIR, "plots")
REPORT    = os.path.join(HRV_DIR, "hrv_report.txt")
FEATURES_CSV = os.path.join(HRV_DIR, "hrv_features_per_scenario.csv")

# ── Experiment constants ───────────────────────────────────────────────────────
CONDITIONS = ["TARS", "TARC", "TARP-S", "TARP-F"]
_COND_COLORS = {"TARS": "#4C72B0", "TARC": "#DD8452",
                "TARP-S": "#55A868", "TARP-F": "#C44E52"}

_SKIP_SUBSTRINGS = [
    "training", "TRAINING",
    "unfinished", "UNFINISHED", "UNIFNISHED",
    "no_birds_strike", "birds_strike",
]

# Procedure order for display
_PROC_ORDER = [
    "CREW BRIEFING",
    "BEFORE TAKEOFF",
    "LINE-UP AND HOLD",
    "TAKEOFF",
    "ENG FAILURE DURING TAKEOFF",
    "ENGINE FIRE",
    "DECLARE PANPAN",
    "AFTER TAKEOFF",
]

# HRV feature subsets for summary statistics (metrics most sensitive to workload)
_TIME_METRICS    = ["HRV_MeanNN", "HRV_SDNN", "HRV_RMSSD", "HRV_pNN50",
                    "HRV_CVNN", "HRV_MeanHR"]
_FREQ_METRICS    = ["HRV_LF", "HRV_HF", "HRV_LFHF", "HRV_LFNU", "HRV_HFNU"]
_NONLIN_METRICS  = ["HRV_SD1", "HRV_SD2", "HRV_SD1SD2", "HRV_DFA_alpha1",
                    "HRV_SampEn"]
_PRIMARY_METRICS = ["HRV_MeanNN", "HRV_SDNN", "HRV_RMSSD", "HRV_pNN50",
                    "HRV_MeanHR", "HRV_LFHF", "HRV_HFNU", "HRV_SD1"]

# Minimum data requirements
_MIN_BEATS_TIME  = 20      # for time-domain features
_MIN_BEATS_FREQ  = 30      # NeuroKit2 will skip freq if < this
_MIN_DURATION_FREQ_S = 120 # seconds — below this freq domain is unreliable
_WIN_S  = 60               # sliding-window width (seconds)
_STEP_S = 30               # sliding-window step (seconds)


# ══════════════════════════════════════════════════════════════════════════════
#  Parsing helpers
# ══════════════════════════════════════════════════════════════════════════════

def parse_hrv_elapsed(s: str) -> float:
    """Parse 'MM:SS' or 'H:MM:SS' to elapsed seconds."""
    parts = s.strip().split(":")
    if len(parts) == 2:
        return int(parts[0]) * 60 + int(parts[1])
    if len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
    raise ValueError(f"Cannot parse HRV time: {s!r}")


def condition_from_filename(name: str) -> str | None:
    """Extract condition label from a scenario filename."""
    base = os.path.basename(name)
    # Order matters: check TARP-F before TARP-S before TARC before TARS
    for cond in ("TARP-F", "TARP-S", "TARC", "TARS"):
        if cond in base:
            return cond
    return None


def is_valid_scenario(path: str) -> bool:
    """Return True if this scenario should be included in the analysis."""
    name = os.path.basename(path)
    return not any(skip in name for skip in _SKIP_SUBSTRINGS)


def iter_ingescape_rows(path: str):
    """
    Yield parsed rows from an Ingescape CSV as (uuid, timestamp, agent,
    source, itype, igs_ts, value_str) tuples.
    Uses csv.reader to correctly handle semicolon-delimited files where the
    value field may contain CSV-escaped JSON (double-double-quote encoding).
    """
    with open(path, encoding="utf-8", errors="replace") as fh:
        reader = csv.reader(fh, delimiter=";")
        for row in reader:
            if len(row) < 5:
                continue
            if row[0] == "uuid":
                continue
            # csv.reader already unescapes "" → "  inside quoted fields
            uuid   = row[0].strip()
            try:
                ts = float(row[1].strip())
            except ValueError:
                continue
            agent  = row[2].strip()
            source = row[3].strip()
            itype  = row[4].strip()
            igs_ts = row[5].strip() if len(row) > 5 else ""
            value  = row[6].strip() if len(row) > 6 else ""
            yield uuid, ts, agent, source, itype, igs_ts, value


# ══════════════════════════════════════════════════════════════════════════════
#  Phase 1 — Load RR intervals
# ══════════════════════════════════════════════════════════════════════════════

def load_rr_intervals(hrv_file: str) -> np.ndarray:
    """
    Load RR intervals (ms) from a Polar H10 / Elite HRV export file
    (one integer per line).

    Important: returns the raw RR stream (only positive values retained).
    We keep raw timing for synchronisation / windowing and apply physiological
    cleaning only right before HRV feature computation.
    """
    rr = []
    with open(hrv_file, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                val = float(line)
            except ValueError:
                continue
            if val > 0.0:
                rr.append(val)
    return np.array(rr, dtype=float)


def clean_rr_for_features(rr_ms: np.ndarray) -> np.ndarray:
    """Physiological cleaning used for HRV feature extraction only."""
    if rr_ms.size == 0:
        return rr_ms
    return rr_ms[(rr_ms >= 200.0) & (rr_ms <= 3000.0)]


# ══════════════════════════════════════════════════════════════════════════════
#  Phase 2 — Time synchronisation
# ══════════════════════════════════════════════════════════════════════════════

def extract_sync_points(ingescape_csvs: list[str]) -> list[tuple[float, float]]:
    """
    Extract (unix_timestamp, hrv_elapsed_seconds) pairs from all
    ``Aircraft;eliteHRV`` events in the given Ingescape CSV files.
    """
    pairs = []
    for path in ingescape_csvs:
        try:
            for _, ts, agent, source, *_ in iter_ingescape_rows(path):
                if agent == "Aircraft" and source == "eliteHRV":
                    # value is last element — re-read the row
                    pass
        except Exception:
            pass
    # Re-read properly to get value field
    pairs = []
    for path in ingescape_csvs:
        try:
            for _, ts, agent, source, itype, igs_ts, value in iter_ingescape_rows(path):
                if agent == "Aircraft" and source == "eliteHRV" and value:
                    try:
                        elapsed = parse_hrv_elapsed(value)
                        pairs.append((ts, elapsed))
                    except ValueError:
                        pass
        except Exception:
            pass
    return pairs


def estimate_hrv_start(sync_pairs: list[tuple[float, float]]) -> float | None:
    """
    Estimate HRV recording start as Unix timestamp.
    For each sync point (unix_ts, hrv_elapsed_s): start = unix_ts - hrv_elapsed_s.
    Returns median estimate (robust to outliers).
    """
    if not sync_pairs:
        return None
    estimates = [ts - elapsed for ts, elapsed in sync_pairs]
    return float(np.median(estimates))


# ══════════════════════════════════════════════════════════════════════════════
#  Phase 3 — Scenario RR segment extraction
# ══════════════════════════════════════════════════════════════════════════════

def rr_to_peak_times_ms(rr_ms: np.ndarray) -> np.ndarray:
    """Cumulative peak times in ms, starting from 0.  Length = len(rr_ms)+1."""
    return np.concatenate([[0.0], np.cumsum(rr_ms)])


def extract_rr_segment(
    rr_ms: np.ndarray,
    peak_times_ms: np.ndarray,
    offset_start_s: float,
    offset_end_s: float,
) -> np.ndarray:
    """
    Extract the RR intervals whose R-peak falls within
    [offset_start_s, offset_end_s] (seconds from HRV recording start).
    Returns the RR intervals in ms as a 1-D array.
    """
    t0 = offset_start_s * 1000.0
    t1 = offset_end_s   * 1000.0
    # peak_times_ms[i] is the time of the i-th R-peak (i=0 is time 0)
    # rr_ms[i] is the interval *ending* at peak i+1 (i.e. between peak i and i+1)
    # We want intervals whose endpoint peak falls in [t0, t1]
    in_window = (peak_times_ms[1:] >= t0) & (peak_times_ms[1:] <= t1)
    return rr_ms[in_window]


def get_scenario_timestamps(scenario_csv: str) -> tuple[float, float] | None:
    """
    Return (start_unix, end_unix) from the first/last *real* event row timestamps.
    Skips SNAPSHOT rows (agent == "SNAPSHOT") and rows whose timestamp looks like
    a relative time (< 1e9), which appear in scenario CSVs exported with the
    ``relative_time_us`` column format (e.g. P08).
    """
    first_ts = last_ts = None
    for _, ts, agent, *_ in iter_ingescape_rows(scenario_csv):
        if agent == "SNAPSHOT":
            continue
        if ts < 1e9:          # relative time in microseconds or seconds — skip
            continue
        if first_ts is None:
            first_ts = ts
        last_ts = ts
    if first_ts is None or last_ts is None:
        return None
    return first_ts, last_ts


# ══════════════════════════════════════════════════════════════════════════════
#  Phase 4 — HRV feature extraction
# ══════════════════════════════════════════════════════════════════════════════

def compute_hrv_features(rr_ms: np.ndarray, duration_s: float) -> dict:
    """
    Compute NeuroKit2 HRV features (all domains) from a clean RR segment.
    Returns a flat dict of metric_name → float (NaN for unavailable metrics).
    """
    result: dict[str, float] = {}
    n = len(rr_ms)

    if n < _MIN_BEATS_TIME:
        return result

    # Build peak sample indices (at sampling_rate=1000 → 1 sample = 1 ms)
    peaks_idx = np.cumsum(rr_ms).astype(int)
    peaks_info = {"ECG_R_Peaks": peaks_idx}

    # Decide which domains to attempt
    run_freq    = (n >= _MIN_BEATS_FREQ) and (duration_s >= _MIN_DURATION_FREQ_S)
    run_nonlin  = n >= _MIN_BEATS_TIME   # NaN will propagate for DFA if too short

    if run_freq:
        try:
            hrv_all = nk.hrv(peaks_info, sampling_rate=1000, show=False)
            for col in hrv_all.columns:
                val = hrv_all.iloc[0][col]
                result[col] = float(val) if val is not None and not (
                    isinstance(val, float) and np.isnan(val)) else np.nan
            return result
        except Exception:
            pass

    # Frequency domain failed or too short — try individual domains
    try:
        td = nk.hrv_time(peaks_info, sampling_rate=1000, show=False)
        for col in td.columns:
            val = td.iloc[0][col]
            result[col] = float(val) if val is not None else np.nan
    except Exception:
        pass

    if run_nonlin:
        try:
            nl = nk.hrv_nonlinear(peaks_info, sampling_rate=1000, show=False)
            for col in nl.columns:
                if col not in result:
                    val = nl.iloc[0][col]
                    result[col] = float(val) if val is not None else np.nan
        except Exception:
            pass

    return result


# ══════════════════════════════════════════════════════════════════════════════
#  Phase 5 — Dynamic / per-procedure analysis
# ══════════════════════════════════════════════════════════════════════════════

def extract_procedure_boundaries(
    scenario_csv: str,
    hrv_start_unix: float,
    scenario_end_offset_s: float | None = None,
) -> list[dict]:
    """
    Parse procedure transitions to extract procedure time windows.

    Primary source is ``*;current_state`` events (richer and present in all
    conditions). ``Shared Interface;force_state_jump`` is used as a fallback
    for edge cases where current_state payloads are missing.

    Returns list of dicts:
      {"procedure": str, "t_start_s": float, "t_end_s": float}
    where times are in seconds from HRV recording start.
    """
    events: list[tuple[float, str]] = []
    for _, ts, agent, source, itype, igs_ts, value in iter_ingescape_rows(scenario_csv):
        if source in ("current_state", "force_state_jump") and value:
            try:
                d = json.loads(value)
                proc = d.get("procedure", "")
                if proc:
                    hrv_offset = ts - hrv_start_unix
                    events.append((hrv_offset, proc))
            except (json.JSONDecodeError, ValueError):
                pass

    if not events:
        return []

    # Sort by time, then keep only true procedure transitions.
    events.sort(key=lambda x: x[0])
    transitions: list[tuple[float, str]] = []
    prev_proc = None
    for t, proc in events:
        if proc != prev_proc:
            transitions.append((t, proc))
            prev_proc = proc

    if not transitions:
        return []

    # Build procedure windows from transitions.
    boundaries: list[dict] = []
    for i, (t_start, proc) in enumerate(transitions):
        if i + 1 < len(transitions):
            t_end = transitions[i + 1][0]
        else:
            if scenario_end_offset_s is not None and scenario_end_offset_s > t_start:
                t_end = scenario_end_offset_s
            else:
                t_end = t_start + 1.0
        boundaries.append({"procedure": proc, "t_start_s": t_start, "t_end_s": t_end})

    return boundaries


def compute_sliding_window_rmssd(
    rr_ms: np.ndarray,
    peak_times_ms: np.ndarray,
    offset_start_s: float,
    offset_end_s: float,
    win_s: float = _WIN_S,
    step_s: float = _STEP_S,
) -> list[dict]:
    """
    Compute RMSSD in successive windows of ``win_s`` seconds, stepping
    ``step_s`` seconds.  Returns list of {"t_center_s", "rmssd"} dicts
    (t_center_s relative to HRV recording start).
    """
    results = []
    t = offset_start_s
    t_limit = offset_end_s - win_s
    while t <= t_limit:
        seg_raw = extract_rr_segment(rr_ms, peak_times_ms, t, t + win_s)
        seg = clean_rr_for_features(seg_raw)
        if len(seg) >= 10:
            try:
                peaks_idx = np.cumsum(seg).astype(int)
                td = nk.hrv_time({"ECG_R_Peaks": peaks_idx},
                                 sampling_rate=1000, show=False)
                rmssd = float(td["HRV_RMSSD"].iloc[0])
                if not np.isnan(rmssd):
                    results.append({"t_center_s": t + win_s / 2.0, "rmssd": rmssd})
            except Exception:
                pass
        t += step_s
    return results


# ══════════════════════════════════════════════════════════════════════════════
#  Phase 6 — Statistical analysis
# ══════════════════════════════════════════════════════════════════════════════

_PAIRS = [("TARS", "TARP-S"), ("TARS", "TARP-F"), ("TARS", "TARC"),
          ("TARP-S", "TARP-F")]


def _wilcoxon_test(a, b):
    pairs = [(float(x), float(y)) for x, y in zip(a, b)
             if x is not None and y is not None
             and not np.isnan(x) and not np.isnan(y)]
    if len(pairs) < 4:
        return 1.0, 0.0, len(pairs), np.nan
    xa = np.array([p[0] for p in pairs])
    xb = np.array([p[1] for p in pairs])
    try:
        res = _sp_wilcoxon(xa, xb, alternative="two-sided", zero_method="wilcox")
        n = len(pairs)
        r = 1.0 - 2.0 * res.statistic / (n * (n + 1) / 2.0)
        return float(res.pvalue), float(r), n, float(res.statistic)
    except Exception:
        return 1.0, 0.0, len(pairs), np.nan


def _friedman_test(groups):
    rows = list(zip(*groups))
    valid = [r for r in rows if all(
        x is not None and not np.isnan(x) for x in r)]
    if len(valid) < 3:
        return 0.0, 1.0
    aligned = [[float(r[i]) for r in valid] for i in range(len(groups))]
    try:
        res = friedmanchisquare(*aligned)
        return float(res.statistic), float(res.pvalue)
    except Exception:
        return 0.0, 1.0


def _holm_correct(pvals):
    pvals = list(pvals)
    if not pvals:
        return np.array([], dtype=bool), np.array([])
    reject, pvals_corr, _, _ = multipletests(pvals, alpha=0.05, method="holm")
    return reject, pvals_corr


def _sig_stars(p_raw, sig):
    if not sig:
        return ""
    if p_raw < 0.001: return "***"
    if p_raw < 0.01:  return "**"
    if p_raw < 0.05:  return "*"
    return "†"


def _desc(vals):
    a = np.array([v for v in vals if v is not None and not np.isnan(v)], dtype=float)
    if len(a) == 0:
        return {"n": 0, "mean": None, "sd": None, "median": None,
                "min": None, "max": None}
    return {
        "n":      int(len(a)),
        "mean":   round(float(np.mean(a)),   4),
        "sd":     round(float(np.std(a, ddof=1)), 4) if len(a) > 1 else 0.0,
        "median": round(float(np.median(a)), 4),
        "min":    round(float(np.min(a)),    4),
        "max":    round(float(np.max(a)),    4),
    }


def run_stats_for_metric(
    all_data: dict,
    participants: list[str],
    metric: str,
) -> dict:
    """
    Collect per-participant, per-condition values and run Friedman +
    Holm-corrected Wilcoxon for one HRV metric.
    Returns dict suitable for reporting.
    """
    # Build participant-aligned lists (one per condition, None if missing)
    vals_by_cond: dict[str, list] = {c: [] for c in CONDITIONS}
    valid_pids = []
    for pid in participants:
        cond_vals = {}
        for cond in CONDITIONS:
            v = all_data.get(pid, {}).get("conditions", {}).get(cond, {}).get(metric)
            cond_vals[cond] = v
        if any(v is not None and not np.isnan(v)
               for v in cond_vals.values()):
            valid_pids.append(pid)
            for cond in CONDITIONS:
                vals_by_cond[cond].append(cond_vals[cond])

    stats_per_cond = {c: _desc(vals_by_cond[c]) for c in CONDITIONS}

    chi2, p_f = _friedman_test([vals_by_cond[c] for c in CONDITIONS])
    n_complete = len([
        row for row in zip(*[vals_by_cond[c] for c in CONDITIONS])
        if all(v is not None and not np.isnan(v) for v in row)
    ])
    frd = {"chi2": chi2, "p": p_f, "df": len(CONDITIONS) - 1}

    ps_raw, rs, ns, ws = [], [], [], []
    for bl, comp in _PAIRS:
        p, r, n, w = _wilcoxon_test(vals_by_cond[bl], vals_by_cond[comp])
        ps_raw.append(p)
        rs.append(r)
        ns.append(n)
        ws.append(w)
    reject, p_corr = _holm_correct(ps_raw)
    pairwise = [
        {"pair": f"{bl} vs {comp}",
         "p_raw": ps_raw[i], "r": rs[i],
         "n": ns[i], "W": ws[i],
         "reject": bool(reject[i]),
         "p_corr": float(p_corr[i]),
         "stars": _sig_stars(ps_raw[i], bool(reject[i]))}
        for i, (bl, comp) in enumerate(_PAIRS)
    ]
    return {
        "metric": metric,
        "n_participants": len(valid_pids),
        "n_complete_cases": n_complete,
        "stats_per_cond": stats_per_cond,
        "friedman": frd,
        "pairwise": pairwise,
        "vals_by_cond": vals_by_cond,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  Phase 7 — Plotting
# ══════════════════════════════════════════════════════════════════════════════

plt.rcParams.update({
    "font.family":  "DejaVu Sans",
    "font.size":    10,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "figure.dpi":   150,
})


def _strip_jitter(ax, cond, x_pos, vals, color):
    """Overlay individual data points with horizontal jitter."""
    clean = [v for v in vals if v is not None and not np.isnan(v)]
    if not clean:
        return
    xs = x_pos + np.random.default_rng(42).uniform(-0.12, 0.12, len(clean))
    ax.scatter(xs, clean, color=color, alpha=0.55, s=18, zorder=3,
               linewidths=0)


def plot_condition_boxplots(stats_results: list[dict], out_dir: str):
    """Box-plots (violin + strip) of key HRV metrics by condition."""
    metrics = [r for r in stats_results
               if r["metric"] in _PRIMARY_METRICS]
    n = len(metrics)
    if n == 0:
        return

    ncols = 4
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 3.6, nrows * 3.2))
    axes = np.array(axes).flatten()

    for idx, sr in enumerate(metrics):
        ax = axes[idx]
        metric = sr["metric"]
        short  = metric.replace("HRV_", "")
        vals_list = [
            [v for v in sr["vals_by_cond"][c]
             if v is not None and not np.isnan(v)]
            for c in CONDITIONS
        ]
        positions = list(range(len(CONDITIONS)))
        colors = [_COND_COLORS[c] for c in CONDITIONS]

        parts = ax.violinplot(
            [v if v else [np.nan] for v in vals_list],
            positions=positions,
            widths=0.6, showmedians=True,
            showextrema=False,
        )
        for pc, col in zip(parts["bodies"], colors):
            pc.set_facecolor(col)
            pc.set_alpha(0.45)
        parts["cmedians"].set_color("black")
        parts["cmedians"].set_linewidth(2)

        for i, (cond, vlist) in enumerate(zip(CONDITIONS, vals_list)):
            _strip_jitter(ax, cond, i, vlist, colors[i])

        # Significance brackets
        frd = sr["friedman"]
        p_f = frd["p"]
        p_str = (f"Friedman p={p_f:.3f}" if p_f >= 0.001
                 else f"Friedman p={p_f:.2e}")
        ax.set_title(f"{short}\n{p_str}", fontsize=8.5)
        ax.set_xticks(positions)
        ax.set_xticklabels(CONDITIONS, fontsize=8, rotation=20)
        ax.set_ylabel(short, fontsize=8)

    # Hide unused subplots
    for ax in axes[n:]:
        ax.set_visible(False)

    handles = [mpatches.Patch(color=_COND_COLORS[c], label=c)
               for c in CONDITIONS]
    fig.legend(handles=handles, loc="lower center",
               ncol=len(CONDITIONS), fontsize=9,
               bbox_to_anchor=(0.5, 0.0))
    fig.suptitle("HRV Metrics by Condition", fontsize=13, y=1.01)
    fig.tight_layout(rect=[0, 0.06, 1, 1])

    out = os.path.join(out_dir, "hrv_condition_boxplots.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out}")


def plot_procedure_dynamics(all_data: dict, participants: list[str], out_dir: str):
    """
    Per-procedure RMSSD profile averaged across participants × condition.
    One line per condition, x-axis = procedure phase.
    """
    # Collect per-condition per-procedure RMSSD
    proc_data: dict[str, dict[str, list]] = {c: {p: [] for p in _PROC_ORDER}
                                              for c in CONDITIONS}
    for pid in participants:
        for scen in all_data.get(pid, {}).get("scenarios", []):
            cond = scen.get("condition")
            if cond not in CONDITIONS:
                continue
            for pb in scen.get("procedure_boundaries", []):
                proc = pb.get("procedure")
                rmssd = pb.get("rmssd")
                if proc in _PROC_ORDER and rmssd is not None and not np.isnan(rmssd):
                    proc_data[cond][proc].append(rmssd)

    # Only include procedures that have any data
    active_procs = [p for p in _PROC_ORDER
                    if any(proc_data[c][p] for c in CONDITIONS)]
    if not active_procs:
        return

    fig, ax = plt.subplots(figsize=(max(8, len(active_procs) * 1.4), 4.5))
    x = np.arange(len(active_procs))
    for cond in CONDITIONS:
        means = [np.mean(proc_data[cond][p]) if proc_data[cond][p] else np.nan
                 for p in active_procs]
        sems  = [(np.std(proc_data[cond][p], ddof=1) /
                  np.sqrt(len(proc_data[cond][p])))
                 if len(proc_data[cond][p]) > 1 else 0.0
                 for p in active_procs]
        ax.plot(x, means, marker="o", label=cond,
                color=_COND_COLORS[cond], linewidth=2)
        ax.fill_between(x,
                        np.array(means) - np.array(sems),
                        np.array(means) + np.array(sems),
                        alpha=0.15, color=_COND_COLORS[cond])

    short_labels = [p.replace("ENG FAILURE DURING TAKEOFF", "ENG FAIL")
                      .replace("LINE-UP AND HOLD", "LINE-UP")
                      .replace("CREW BRIEFING", "CREW BRIEF")
                      .replace("DECLARE PANPAN", "PANPAN")
                    for p in active_procs]
    ax.set_xticks(x)
    ax.set_xticklabels(short_labels, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("RMSSD (ms)", fontsize=10)
    ax.set_title("RMSSD per Procedure Phase by Condition\n(mean ± SEM across participants)",
                 fontsize=11)
    ax.legend(loc="upper right", fontsize=9)

    fig.tight_layout()
    out = os.path.join(out_dir, "hrv_procedure_dynamics.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out}")


def plot_workload_timeseries(all_data: dict, participants: list[str], out_dir: str):
    """
    Per-participant sliding-window RMSSD time series, arranged by condition.
    Each subplot = one condition; lines = individual participants.
    """
    fig, axes = plt.subplots(1, len(CONDITIONS),
                             figsize=(len(CONDITIONS) * 5, 4.5),
                             sharey=True)
    for ax, cond in zip(axes, CONDITIONS):
        color = _COND_COLORS[cond]
        any_plotted = False
        for pid in participants:
            for scen in all_data.get(pid, {}).get("scenarios", []):
                if scen.get("condition") != cond:
                    continue
                windows = scen.get("sliding_windows", [])
                if not windows:
                    continue
                # Normalize time to scenario-relative seconds
                t0 = windows[0]["t_center_s"]
                ts = [w["t_center_s"] - t0 for w in windows]
                rs = [w["rmssd"] for w in windows]
                ax.plot(ts, rs, alpha=0.4, linewidth=1.2,
                        color=color, label=pid)
                any_plotted = True
        ax.set_title(cond, fontsize=11, color=color, fontweight="bold")
        ax.set_xlabel("Time into scenario (s)", fontsize=9)
        if ax is axes[0]:
            ax.set_ylabel("RMSSD (ms)", fontsize=10)

    fig.suptitle("Sliding-Window RMSSD Within Scenarios (per participant)",
                 fontsize=12)
    fig.tight_layout()
    out = os.path.join(out_dir, "hrv_workload_timeseries.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out}")


def plot_participant_heatmap(all_data: dict, participants: list[str], out_dir: str):
    """
    Heatmap of mean RMSSD per participant × condition (z-scored per row).
    """
    matrix = np.full((len(participants), len(CONDITIONS)), np.nan)
    for r, pid in enumerate(participants):
        for c, cond in enumerate(CONDITIONS):
            v = all_data.get(pid, {}).get("conditions", {}).get(cond, {}).get("HRV_RMSSD")
            if v is not None and not np.isnan(v):
                matrix[r, c] = v

    # Z-score across conditions per participant
    z_matrix = np.full_like(matrix, np.nan)
    for r in range(len(participants)):
        row = matrix[r]
        valid = row[~np.isnan(row)]
        if len(valid) > 1:
            mu, sigma = np.mean(valid), np.std(valid, ddof=1)
            if sigma > 0:
                z_matrix[r] = (row - mu) / sigma

    fig, ax = plt.subplots(figsize=(len(CONDITIONS) * 1.2 + 1.5, len(participants) * 0.55 + 1.5))
    im = ax.imshow(z_matrix, aspect="auto", cmap="RdYlBu",
                   vmin=-2, vmax=2, interpolation="nearest")
    ax.set_xticks(range(len(CONDITIONS)))
    ax.set_xticklabels(CONDITIONS, fontsize=10)
    ax.set_yticks(range(len(participants)))
    ax.set_yticklabels(participants, fontsize=9)
    ax.set_title("RMSSD z-score per participant × condition\n(blue = higher HRV / lower workload)",
                 fontsize=10)
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.04)
    cbar.set_label("z-score", fontsize=9)
    fig.tight_layout()
    out = os.path.join(out_dir, "hrv_participant_heatmap.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out}")


# ══════════════════════════════════════════════════════════════════════════════
#  Phase 8 — Reporting
# ══════════════════════════════════════════════════════════════════════════════

def _fmt_p(p):
    return f"{p:.4f}" if p >= 0.0001 else f"{p:.2e}"


def _fmt_metric_section(sr: dict) -> str:
    metric = sr["metric"].replace("HRV_", "")
    lines  = [f"\n  {metric}"]
    frd    = sr["friedman"]
    lines.append(
        f"    Friedman χ²({frd['df']}) = {frd['chi2']:.3f},  "
        f"p = {_fmt_p(frd['p'])}"
    )
    # Condition descriptive stats
    hdr = "    " + "  ".join(f"{c:^22}" for c in CONDITIONS)
    lines.append(hdr)
    stat_line = "    "
    for c in CONDITIONS:
        s = sr["stats_per_cond"][c]
        if s["n"] == 0:
            stat_line += f"  {'— (no data)':^22}"
        else:
            stat_line += (f"  μ={s['mean']:6.2f} σ={s['sd']:5.2f} "
                          f"Md={s['median']:6.2f}")
    lines.append(stat_line)
    # Pairwise
    for pw in sr["pairwise"]:
        sig   = "✓" if pw["reject"] else " "
        stars = pw["stars"] or "ns"
        lines.append(
            f"    [{sig}] {pw['pair']:<20}  "
            f"W p={_fmt_p(pw['p_raw'])}  "
            f"p_Holm={_fmt_p(pw['p_corr'])}  "
            f"r={pw['r']:+.3f}  {stars}"
        )
    return "\n".join(lines) + "\n"


_METRIC_WHY = {
    "HRV_MeanNN": "Mean beat-to-beat interval; lower values indicate higher heart rate under load.",
    "HRV_SDNN": "Global HRV dispersion; often decreases with sustained cognitive load.",
    "HRV_RMSSD": "Short-term vagal HRV marker; lower RMSSD is consistent with higher workload.",
    "HRV_pNN50": "Proportion of large NN differences; complements RMSSD for parasympathetic activity.",
    "HRV_MeanHR": "Heart rate level; tends to increase with stress and effort.",
    "HRV_LF": "Low-frequency power; mixed autonomic contribution, interpreted jointly with HF.",
    "HRV_HF": "High-frequency power; parasympathetic-linked component.",
    "HRV_LFHF": "Sympathovagal balance proxy; interpreted cautiously in context.",
    "HRV_LFNU": "LF normalized units; relative LF contribution.",
    "HRV_HFNU": "HF normalized units; relative parasympathetic contribution.",
    "HRV_SD1": "Poincare short-axis variability; mirrors short-term vagal modulation.",
    "HRV_SD2": "Poincare long-axis variability; reflects longer-term variability.",
    "HRV_SD1SD2": "Shape ratio from Poincare plot; balance between short and long-term variability.",
    "HRV_DFA_alpha1": "Short-term fractal scaling; non-linear autonomic control index.",
    "HRV_SampEn": "Signal irregularity measure; lower values indicate more regular dynamics.",
    "HRV_CVNN": "Variation normalized by mean NN; scale-independent variability marker.",
}


def _fmt_sig_label(p_adj: float) -> str:
    if np.isnan(p_adj):
        return "n/a"
    if p_adj < 0.001:
        return "***"
    if p_adj < 0.01:
        return "**"
    if p_adj < 0.05:
        return "*"
    return "ns"


def write_report(
    all_data: dict,
    participants: list[str],
    stats_results: list[dict],
):
    os.makedirs(os.path.dirname(REPORT), exist_ok=True)

    # Build machine-readable JSON
    summary = {
        "participants": participants,
        "n_participants": len(participants),
        "conditions": CONDITIONS,
        "per_participant": {
            pid: {
                "hrv_start_unix": all_data[pid].get("hrv_start_unix"),
                "n_sync_points":  all_data[pid].get("n_sync_points"),
                "scenarios":      [
                    {k: v for k, v in s.items()
                     if k not in ("sliding_windows", "procedure_boundaries")}
                    for s in all_data[pid].get("scenarios", [])
                ],
                "conditions": {
                    c: {m: v for m, v in
                        all_data[pid].get("conditions", {}).get(c, {}).items()
                        if not m.startswith("_")}
                    for c in CONDITIONS
                },
            }
            for pid in participants
        },
        "stats": {
            sr["metric"]: {
                "friedman": sr["friedman"],
                "pairwise": sr["pairwise"],
                "stats_per_cond": sr["stats_per_cond"],
            }
            for sr in stats_results
        },
    }

    with open(REPORT, "w", encoding="utf-8") as fh:
        fh.write("--- MACHINE-READABLE SUMMARY (JSON) ---\n")
        fh.write(json.dumps(summary, indent=2, ensure_ascii=False,
                            default=lambda o: None if (isinstance(o, float) and np.isnan(o)) else o))
        fh.write("\n--- END SUMMARY ---\n")
        fh.write("HITLS HRV: Heart-Rate Variability Analysis\n")
        fh.write("=" * 80 + "\n")
        fh.write("Input signal      : RR intervals (ms) from Polar H10 / Elite HRV export\n")
        fh.write("Synchronization   : eliteHRV app elapsed time aligned to ingescape unix timestamps\n")
        fh.write("Scenario slicing  : per scenario from scenario ingescape start/end timestamps\n")
        fh.write("Procedure slicing : procedure transitions from current_state (fallback: force_state_jump)\n")
        fh.write("Domains computed  : NeuroKit2 HRV all domains (time, frequency, non-linear)\n")
        fh.write("Dynamic workload  : RMSSD on 60 s windows (step 30 s) + per-procedure RMSSD\n")
        fh.write("Statistics        : Friedman test + pairwise Wilcoxon (Holm correction)\n")
        fh.write("=" * 80 + "\n\n")

        fh.write(f"Participants with HRV data (n={len(participants)}): {', '.join(participants)}\n")
        fh.write("\n")
        fh.write("Why these calculations:\n")
        fh.write("  • Time-domain metrics (RMSSD, SDNN, MeanNN, pNN50) quantify short- and global variability\n")
        fh.write("    linked to mental effort and stress reactivity.\n")
        fh.write("  • Frequency-domain metrics (LF, HF, LF/HF, LFNU, HFNU) characterize spectral balance\n")
        fh.write("    between autonomic components across conditions.\n")
        fh.write("  • Non-linear metrics (SD1/SD2, DFA alpha1, SampEn) capture complexity changes\n")
        fh.write("    not visible in linear summaries.\n")
        fh.write("  • Repeated-measures non-parametric tests are used because data are paired by participant\n")
        fh.write("    and normality cannot be assumed.\n")

        for sr in stats_results:
            metric = sr["metric"]
            label = metric.replace("HRV_", "")
            why = _METRIC_WHY.get(metric, "Condition-level workload-sensitive HRV marker.")

            fh.write(f"\n\n{'─'*80}\n")
            fh.write(f"METRIC: {label}\n")
            fh.write(f"Why this metric: {why}\n")
            fh.write(f"{'─'*80}\n")

            cond_data_by_pid = {
                pid: {
                    cond: all_data.get(pid, {}).get("conditions", {}).get(cond, {}).get(metric, np.nan)
                    for cond in CONDITIONS
                }
                for pid in participants
            }

            col_w = 16
            fh.write(f"\n{'Participant':<12}")
            for cond in CONDITIONS:
                fh.write(f"  {cond:^{col_w}}")
            fh.write("\n")
            fh.write(f"{'-'*12}" + f"  {'-'*col_w}" * len(CONDITIONS) + "\n")

            for pid in participants:
                fh.write(f"{pid:<12}")
                for cond in CONDITIONS:
                    v = cond_data_by_pid[pid][cond]
                    if v is not None and not np.isnan(v):
                        fh.write(f"  {v:>14.3f}  ")
                    else:
                        fh.write(f"  {'—':^{col_w}}")
                fh.write("\n")

            fh.write(f"\n{'MEAN±SD':<12}")
            for cond in CONDITIONS:
                vals = [
                    cond_data_by_pid[pid][cond]
                    for pid in participants
                    if cond_data_by_pid[pid][cond] is not None
                    and not np.isnan(cond_data_by_pid[pid][cond])
                ]
                if vals:
                    fh.write(f"  {np.mean(vals):>6.2f}±{np.std(vals, ddof=1) if len(vals) > 1 else 0.0:<7.2f}  ")
                else:
                    fh.write(f"  {'—':^{col_w}}")
            fh.write("\n")

            fr = sr["friedman"]
            n_complete = sr.get("n_complete_cases", 0)
            if n_complete >= 3:
                fh.write(
                    f"\nFriedman test (n={n_complete} with all 4 conditions): "
                    f"χ²={fr['chi2']:.3f}  p={_fmt_p(fr['p'])}  {_fmt_sig_label(fr['p'])}\n"
                )
            else:
                fh.write("\nFriedman test: insufficient complete repeated-measures data\n")

            if sr["pairwise"]:
                fh.write("\nPairwise Wilcoxon signed-rank (Holm, k=4 pairs):\n")
                fh.write(
                    f"  {'Pair':<17}  {'n':>4}  {'W':>7}  {'p':>10}  {'p_adj':>10}  {'sig':>4}\n"
                )
                fh.write(
                    f"  {'-'*17}  {'-'*4}  {'-'*7}  {'-'*10}  {'-'*10}  {'-'*4}\n"
                )
                for pw in sr["pairwise"]:
                    w = pw.get("W", np.nan)
                    w_str = f"{w:.0f}" if not np.isnan(w) else "n/a"
                    sig = _fmt_sig_label(pw.get("p_corr", np.nan))
                    fh.write(
                        f"  {pw['pair']:<17}  {pw.get('n', 0):>4}  {w_str:>7}  "
                        f"{_fmt_p(pw['p_raw']):>10}  {_fmt_p(pw['p_corr']):>10}  {sig:>4}\n"
                    )

    print(f"  → {REPORT}")


def write_features_csv(all_data: dict, participants: list[str]):
    """Write one row per valid scenario to hrv_features_per_scenario.csv."""
    # Collect all HRV column names
    all_cols: set[str] = set()
    for pid in participants:
        for scen in all_data.get(pid, {}).get("scenarios", []):
            all_cols.update(scen.get("features", {}).keys())

    hrv_cols = sorted(c for c in all_cols if c.startswith("HRV_"))
    meta_cols = ["participant", "scenario", "condition",
                 "start_unix", "end_unix", "duration_s", "n_beats"]

    os.makedirs(os.path.dirname(FEATURES_CSV), exist_ok=True)
    with open(FEATURES_CSV, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=meta_cols + hrv_cols)
        writer.writeheader()
        for pid in participants:
            for scen in all_data.get(pid, {}).get("scenarios", []):
                row: dict = {
                    "participant": pid,
                    "scenario":   scen["name"],
                    "condition":  scen.get("condition", ""),
                    "start_unix": scen.get("start_unix", ""),
                    "end_unix":   scen.get("end_unix", ""),
                    "duration_s": scen.get("duration_s", ""),
                    "n_beats":    scen.get("n_beats", ""),
                }
                feats = scen.get("features", {})
                for col in hrv_cols:
                    v = feats.get(col, np.nan)
                    row[col] = "" if (v is None or (isinstance(v, float) and np.isnan(v))) else f"{v:.6g}"
                writer.writerow(row)

    print(f"  → {FEATURES_CSV}")


# ══════════════════════════════════════════════════════════════════════════════
#  Per-participant pipeline
# ══════════════════════════════════════════════════════════════════════════════

def find_ingescape_csvs(participant_dir: str) -> list[str]:
    """
    Return all *_ingescape.csv files in the participant root directory
    (not in scenarios/).  Excludes TARC-only and train_then_test files.
    """
    result = []
    pdir = Path(participant_dir)
    for p in sorted(pdir.glob("*_ingescape.csv")):
        name = p.name
        if "scenarios" in str(p.parent):
            continue
        if "train_then_test" in name:
            continue
        result.append(str(p))
    # Also include *_ingescape.csv.csv (double extension, seen in some exports)
    for p in sorted(pdir.glob("*_ingescape.csv.csv")):
        if "train_then_test" not in p.name:
            result.append(str(p))
    return result


def analyze_participant(pid: str, participant_dir: str) -> dict:
    """
    Run the full HRV pipeline for one participant.
    Returns a dict with all extracted data.
    """
    result: dict = {
        "pid": pid,
        "hrv_start_unix": None,
        "n_sync_points": 0,
        "scenarios": [],
        "conditions": {c: {} for c in CONDITIONS},
        "warnings": [],
    }

    # ── Find HRV file ──────────────────────────────────────────────────────
    hrv_files = list(Path(participant_dir).glob("HRV_*.txt"))
    if not hrv_files:
        result["warnings"].append("No HRV file found")
        return result
    hrv_file = str(hrv_files[0])

    # ── Load RR intervals ──────────────────────────────────────────────────
    rr_ms = load_rr_intervals(hrv_file)
    if len(rr_ms) < 50:
        result["warnings"].append(f"Too few RR intervals in file: {len(rr_ms)}")
        return result
    peak_times_ms = rr_to_peak_times_ms(rr_ms)

    # ── Synchronisation ────────────────────────────────────────────────────
    ingescape_csvs = find_ingescape_csvs(participant_dir)
    sync_pairs = extract_sync_points(ingescape_csvs)
    hrv_start_unix = estimate_hrv_start(sync_pairs)
    if hrv_start_unix is None:
        result["warnings"].append("No eliteHRV sync points found")
        return result
    result["hrv_start_unix"] = hrv_start_unix
    result["n_sync_points"]  = len(sync_pairs)

    # Residual quality check (sync consistency)
    residuals = [abs((ts - elapsed) - hrv_start_unix) for ts, elapsed in sync_pairs]
    max_resid = max(residuals) if residuals else 0.0
    if max_resid > 10.0:
        result["warnings"].append(
            f"Sync residual {max_resid:.1f}s > 10s — check eliteHRV timestamps")

    # ── Per-scenario analysis ──────────────────────────────────────────────
    scenarios_dir = os.path.join(participant_dir, "scenarios")
    if not os.path.isdir(scenarios_dir):
        result["warnings"].append("No scenarios/ directory")
        return result

    scenario_csvs = sorted(Path(scenarios_dir).glob("*_ingescape.csv"))
    for scen_csv in scenario_csvs:
        scen_name = scen_csv.stem.replace("_ingescape", "")
        if not is_valid_scenario(str(scen_csv)):
            continue
        cond = condition_from_filename(str(scen_csv))
        if cond is None:
            continue

        ts_bounds = get_scenario_timestamps(str(scen_csv))
        if ts_bounds is None:
            continue
        start_unix, end_unix = ts_bounds
        duration_s = end_unix - start_unix

        offset_start = start_unix - hrv_start_unix
        offset_end   = end_unix   - hrv_start_unix

        # Sanity: HRV recording must cover at least the start of this scenario
        hrv_total_s = peak_times_ms[-1] / 1000.0
        if offset_start < -60:
            result["warnings"].append(
                f"{scen_name}: scenario starts {-offset_start:.0f}s before HRV "
                f"recording — sync error, skipping")
            continue
        if offset_start > hrv_total_s:
            result["warnings"].append(
                f"{scen_name}: scenario starts {offset_start - hrv_total_s:.0f}s "
                f"after HRV recording ended — no data available, skipping")
            continue
        if offset_end > hrv_total_s + 10:
            result["warnings"].append(
                f"{scen_name}: HRV recording ends {offset_end - hrv_total_s:.0f}s "
                f"before scenario end — clipping to available data")
            offset_end = hrv_total_s
        offset_start = max(0.0, offset_start)

        rr_seg_raw = extract_rr_segment(rr_ms, peak_times_ms,
                        max(0.0, offset_start), offset_end)
        rr_seg = clean_rr_for_features(rr_seg_raw)
        n_beats = len(rr_seg)

        # HRV features
        seg_duration_s = max(0.0, offset_end - max(0.0, offset_start))
        features = compute_hrv_features(rr_seg, seg_duration_s)

        # Sliding-window RMSSD
        windows = compute_sliding_window_rmssd(
            rr_ms, peak_times_ms,
            max(0.0, offset_start), offset_end,
        )

        # Per-procedure RMSSD
        proc_bounds = extract_procedure_boundaries(
            str(scen_csv), hrv_start_unix, scenario_end_offset_s=offset_end)
        proc_hrv: list[dict] = []
        for pb in proc_bounds:
            proc_rr_raw = extract_rr_segment(
                rr_ms, peak_times_ms,
                pb["t_start_s"], pb["t_end_s"])
            proc_rr = clean_rr_for_features(proc_rr_raw)
            rmssd = np.nan
            if len(proc_rr) >= 10:
                try:
                    pidx = np.cumsum(proc_rr).astype(int)
                    td = nk.hrv_time({"ECG_R_Peaks": pidx},
                                     sampling_rate=1000, show=False)
                    rmssd = float(td["HRV_RMSSD"].iloc[0])
                except Exception:
                    pass
            proc_hrv.append({
                "procedure":  pb["procedure"],
                "t_start_s":  pb["t_start_s"],
                "t_end_s":    pb["t_end_s"],
                "n_beats":    len(proc_rr),
                "rmssd":      rmssd if not np.isnan(rmssd) else None,
            })

        scen_record = {
            "name":               scen_name,
            "condition":          cond,
            "start_unix":         start_unix,
            "end_unix":           end_unix,
            "duration_s":         duration_s,
            "n_beats":            n_beats,
            "features":           features,
            "sliding_windows":    windows,
            "procedure_boundaries": proc_hrv,
        }
        result["scenarios"].append(scen_record)

    # ── Aggregate per condition ────────────────────────────────────────────
    # Multiple scenarios per condition → take mean across scenarios
    cond_accum: dict[str, dict[str, list]] = {c: {} for c in CONDITIONS}
    for scen in result["scenarios"]:
        cond = scen["condition"]
        for metric, val in scen["features"].items():
            if not metric.startswith("HRV_"):
                continue
            if val is None or (isinstance(val, float) and np.isnan(val)):
                continue
            cond_accum[cond].setdefault(metric, []).append(val)

    for cond in CONDITIONS:
        for metric, vals in cond_accum[cond].items():
            result["conditions"][cond][metric] = float(np.mean(vals))

    return result


# ══════════════════════════════════════════════════════════════════════════════
#  Outputs preview & confirmation
# ══════════════════════════════════════════════════════════════════════════════

def _collect_outputs() -> tuple[list[str], list[str]]:
    """Return (plot_paths, report_paths) that will be written."""
    plots = [
        os.path.join(PLOTS_DIR, "hrv_condition_boxplots.png"),
        os.path.join(PLOTS_DIR, "hrv_procedure_dynamics.png"),
        os.path.join(PLOTS_DIR, "hrv_workload_timeseries.png"),
        os.path.join(PLOTS_DIR, "hrv_participant_heatmap.png"),
    ]
    reports = [REPORT, FEATURES_CSV]
    return plots, reports


def _check_existing(paths: list[str]) -> list[str]:
    return [p for p in paths if os.path.exists(p)]


def confirm_run(plots: list[str], reports: list[str]) -> bool:
    exist_p = _check_existing(plots)
    exist_r = _check_existing(reports)

    print("Plots that will be written/overwritten:")
    for p in plots:
        tag = "[overwrite]" if p in exist_p else "[new]"
        print(f"  {tag:12s}  {os.path.basename(p)}")

    print("\nReports that will be written/overwritten:")
    for r in reports:
        tag = "[overwrite]" if r in exist_r else "[new]"
        print(f"  {tag:12s}  {os.path.relpath(r, HITLS_DIR)}")

    ans = input("\nContinue? [Y/n]: ").strip().lower()
    return ans in ("", "y", "yes")


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="HRV analysis for HITLS participants")
    parser.add_argument("--force", action="store_true",
                        help="Skip confirmation prompt")
    args = parser.parse_args()

    print("=" * 70)
    print("  HITLS — HRV Analysis")
    print("=" * 70)

    # ── Discover participants with HRV files ───────────────────────────────
    hitls_dir = Path(HITLS_DIR)
    all_pids = sorted(
        p.name for p in hitls_dir.iterdir()
        if p.is_dir() and re.match(r"P\d+$", p.name)
        and list(p.glob("HRV_*.txt"))
    )
    if not all_pids:
        print("  No participants with HRV files found.")
        sys.exit(0)
    print(f"  Participants with HRV data: {', '.join(all_pids)}")

    plots, reports = _collect_outputs()
    if not args.force:
        if not confirm_run(plots, reports):
            print("Aborted.")
            sys.exit(0)

    os.makedirs(PLOTS_DIR, exist_ok=True)

    # ── Per-participant analysis ───────────────────────────────────────────
    print(f"\n[1/4] Analysing per-participant HRV …")
    all_data: dict[str, dict] = {}
    for pid in all_pids:
        pdir = str(hitls_dir / pid)
        print(f"  {pid} …", end=" ", flush=True)
        data = analyze_participant(pid, pdir)
        all_data[pid] = data
        n_scen = len(data.get("scenarios", []))
        n_sync = data.get("n_sync_points", 0)
        warns  = data.get("warnings", [])
        if warns:
            print(f"  ⚠ {n_scen} scenarios  ({n_sync} sync pts) "
                  f"— warnings: {'; '.join(warns)}")
        else:
            print(f"  ✓ {n_scen} valid scenarios  ({n_sync} sync pts)")

    # Only keep participants with at least one analysed scenario
    participants = [pid for pid in all_pids
                    if all_data[pid].get("scenarios")]
    if not participants:
        print("  No valid HRV scenario data found after filtering.")
        sys.exit(0)

    # ── Write features CSV ─────────────────────────────────────────────────
    print(f"\n[2/4] Writing features CSV …")
    write_features_csv(all_data, participants)

    # ── Statistical analysis ───────────────────────────────────────────────
    print(f"\n[3/4] Running statistics …")
    # Collect all HRV metrics that appear in the data
    all_metrics: set[str] = set()
    for pid in participants:
        for cond in CONDITIONS:
            all_metrics.update(all_data[pid]["conditions"][cond].keys())
    metrics_to_test = [m for m in sorted(all_metrics)
                       if m in (_TIME_METRICS + _FREQ_METRICS + _NONLIN_METRICS)]

    stats_results = []
    for metric in metrics_to_test:
        sr = run_stats_for_metric(all_data, participants, metric)
        if sr["n_participants"] >= 3:
            stats_results.append(sr)
            print(f"  {metric:<30s}  "
                  f"Friedman χ²={sr['friedman']['chi2']:.2f}  "
                  f"p={_fmt_p(sr['friedman']['p'])}")

    # ── Write report ───────────────────────────────────────────────────────
    print(f"\n  Writing report …")
    write_report(all_data, participants, stats_results)

    # ── Plots ──────────────────────────────────────────────────────────────
    print(f"\n[4/4] Generating plots → {PLOTS_DIR}/")
    plot_condition_boxplots(stats_results, PLOTS_DIR)
    plot_procedure_dynamics(all_data, participants, PLOTS_DIR)
    plot_workload_timeseries(all_data, participants, PLOTS_DIR)
    plot_participant_heatmap(all_data, participants, PLOTS_DIR)

    print(f"\nDone — {len(participants)} participants analysed.")


if __name__ == "__main__":
    main()
