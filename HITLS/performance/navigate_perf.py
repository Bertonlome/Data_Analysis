#!/usr/bin/env python3
"""
navigate_perf.py — Navigate Performance Analysis for a single HITLS participant.

Metrics extracted from Aircraft agent signals (latitude, longitude, altitude,
heading) in scenario ingescape CSV files, compared against the theoretical
flight path defined by the experiment scenario design.

──────────────────────────────────────────────────────────────────────────────
Theoretical flight path
──────────────────────────────────────────────────────────────────────────────
  Phase 1 — Climb  (positive-rate detected → DECLARE PANPAN)
    The aircraft should maintain runway heading from the threshold and climb
    toward 5 000 ft at 120 KIAS.

    Signals   : latitude, longitude, heading
    Metrics   : XTE   — cross-track error from runway heading line (NM)
                ATD   — along-track error vs expected 120 kt (NM)
                HDG   — heading deviation from runway heading (°)

  Level-off  (first sample ≥ 5 000 ft → DECLARE PANPAN)
    The aircraft should hold 5 000 ft.

    Signals   : altitude
    Metrics   : ALT   — altitude deviation from 5 000 ft (ft)

──────────────────────────────────────────────────────────────────────────────
Geometry
──────────────────────────────────────────────────────────────────────────────
  Flat-earth approximation (valid for the <20 NM distances involved):

    Δx (E, m) = (lon − lon₀) × cos(lat₀) × 111 320
    Δy (N, m) = (lat − lat₀) × 111 320

  For track heading θ (aviation convention, CW from north):
    track unit vector = (sin θ, cos θ)   in (E, N) components

    ATD (m)  =  Δx·sin θ + Δy·cos θ     (along-track distance)
    XTE (m)  =  Δx·cos θ − Δy·sin θ     (cross-track; +ve = right of track)

──────────────────────────────────────────────────────────────────────────────
Runways in experiment
──────────────────────────────────────────────────────────────────────────────
  Runway  True HDG
  06R       057°
  24R       237°
  24L       237°
  (06L does not appear in the experiment scenarios)

Conditions  : TARS | TARC | TARP-S | TARP-F
Output      : {PID}/cleaned/{PID}_navigate_perf_report.txt
"""

import os
import re
import sys
import json
import glob
import math
import numpy as np

# ── Paths ─────────────────────────────────────────────────────────────────────
PERF_DIR  = os.path.dirname(os.path.abspath(__file__))
HITLS_DIR = os.path.dirname(PERF_DIR)

# ── Experiment constants ───────────────────────────────────────────────────────
CONDITIONS      = ["TARS", "TARC", "TARP-S", "TARP-F"]
TARGET_ALT      = 5000.0    # ft   — level-off altitude
TARGET_AIRSPEED = 120.0     # kt   — expected ground speed
NM_PER_M        = 1.0 / 1852.0
KT_TO_MPS       = 1852.0 / 3600.0   # 1 kt → m/s  (0.5144 m/s)

# ── Runway geometry ────────────────────────────────────────────────────────────
RUNWAY_START = {
    "06L": (45.461222,  -73.764740),
    "06R": (45.457832,  -73.741171),
    "24R": (45.483156,  -73.736070),
    "24L": (45.476887,  -73.716188),
}

RUNWAY_HEADING = {
    "06L":  57.0,
    "06R":  57.0,
    "24R": 237.0,
    "24L": 237.0,
}

# ── Phase-detection strings ────────────────────────────────────────────────────
_POSITIVE_RATE_TASK_OBJ = '""Climb rate""'
_POSITIVE_RATE_VALUE    = '""CHECK POSITIVE""'

# ── Filename parsing ───────────────────────────────────────────────────────────
_COND_RE   = re.compile(r"_(TARS|TARC|TARP-S|TARP-F)_", re.IGNORECASE)
_RUNWAY_RE = re.compile(r"_(06L|06R|24L|24R)[_.]")
_SKIP_SUBSTRINGS = [
    "training", "TRAINING",
    "unfinished", "UNFINISHED",
    "UNIFNISHED",
    "no_birds_strike",
    "birds",
]

# ── Geometry helpers ───────────────────────────────────────────────────────────
_DEG_TO_RAD = math.pi / 180.0
_LAT_M_PER_DEG = 111_320.0  # metres per degree of latitude


def _latlon_to_xy(lat: float, lon: float, lat0: float, lon0: float):
    """Return (dx_east_m, dy_north_m) relative to reference point."""
    dy = (lat - lat0) * _LAT_M_PER_DEG
    dx = (lon - lon0) * math.cos(lat0 * _DEG_TO_RAD) * _LAT_M_PER_DEG
    return dx, dy


def _xte_atd(lat: float, lon: float, lat0: float, lon0: float, heading_deg: float):
    """
    Cross-track error (signed, metres) and along-track distance (metres)
    for a point (lat, lon) relative to a track starting at (lat0, lon0)
    with the given true heading.

    XTE > 0 means the aircraft is to the RIGHT of the track.
    """
    dx, dy = _latlon_to_xy(lat, lon, lat0, lon0)
    θ = heading_deg * _DEG_TO_RAD
    sin_θ = math.sin(θ)
    cos_θ = math.cos(θ)
    atd = dx * sin_θ + dy * cos_θ   # along-track distance (m)
    xte = dx * cos_θ - dy * sin_θ   # cross-track error (m), signed
    return xte, atd


def _hdg_error(actual: float, target: float) -> float:
    """Minimum angular difference between two headings (0–180°)."""
    diff = abs(actual - target) % 360.0
    return diff if diff <= 180.0 else 360.0 - diff


# ── File helpers ───────────────────────────────────────────────────────────────

def find_participants():
    return [
        e for e in sorted(os.listdir(HITLS_DIR))
        if os.path.isdir(os.path.join(HITLS_DIR, e))
        and re.match(r"^P\d+$", e)
    ]


def _is_valid_scenario(path: str) -> bool:
    name = os.path.basename(path)
    return not any(s in name for s in _SKIP_SUBSTRINGS)


def _condition_from_filename(path: str):
    m = _COND_RE.search(os.path.basename(path))
    return m.group(1).upper() if m else None


def _runway_from_filename(path: str):
    m = _RUNWAY_RE.search(os.path.basename(path))
    return m.group(1).upper() if m else None


def find_scenario_files(pid: str) -> dict:
    """Return {condition: [one_file_path]} — at most one valid file per condition."""
    scen_dir = os.path.join(HITLS_DIR, pid, "scenarios")
    if not os.path.isdir(scen_dir):
        return {}
    out = {c: [] for c in CONDITIONS}
    for path in sorted(glob.glob(os.path.join(scen_dir, "*_ingescape.csv"))):
        if _is_valid_scenario(path):
            cond = _condition_from_filename(path)
            if cond in out and len(out[cond]) == 0:
                out[cond].append(path)
    return {c: v for c, v in out.items() if v}


# ═══════════════════════════════════════════════════════════════════════════════
#  CSV parsing
# ═══════════════════════════════════════════════════════════════════════════════

_NAV_SIGNALS = {"latitude", "longitude", "altitude", "heading"}


def _extract_nav_signals(path: str) -> dict:
    """
    Parse scenario CSV and return signal time-series plus phase markers.

    Returns:
      {
        "latitude":            [(t, v), ...],
        "longitude":           [(t, v), ...],
        "altitude":            [(t, v), ...],
        "heading":             [(t, v), ...],
        "positive_rate_time":  float | None,

        "panpan_time":         float | None,
        "t0":                  float,
        "t_last":              float,
      }

    CSV columns (semicolon-delimited):
      [0]uuid  [1]timestamp  [2]agent  [3]source  [4]type  [5]igs_ts  [6]value
    """
    signals = {k: [] for k in _NAV_SIGNALS}
    positive_rate_time = None
    panpan_time        = None
    t0                 = None
    t_last             = None

    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            parts = line.rstrip("\n").split(";")
            if len(parts) < 7:
                continue
            try:
                ts = float(parts[1])
            except ValueError:
                continue

            if t0 is None:
                t0 = ts
            t_last = ts

            agent  = parts[2]
            source = parts[3]

            # ── Aircraft navigation signals ───────────────────────────────────
            if agent == "Aircraft" and source in _NAV_SIGNALS:
                try:
                    signals[source].append((ts, float(parts[6])))
                except (ValueError, IndexError):
                    pass

            # ── Positive rate detection ───────────────────────────────────────
            if (positive_rate_time is None
                    and source == "current_state"
                    and '""TAKEOFF""' in line
                    and _POSITIVE_RATE_TASK_OBJ in line
                    and _POSITIVE_RATE_VALUE in line):
                positive_rate_time = ts

            # ── DECLARE PANPAN detection ──────────────────────────────────────
            if panpan_time is None and source == "current_state":
                if '""DECLARE PANPAN""' in line or '"DECLARE PANPAN"' in line:
                    panpan_time = ts

    result = {k: sorted(v) for k, v in signals.items()}
    result["positive_rate_time"] = positive_rate_time
    result["panpan_time"]        = panpan_time
    result["t0"]                 = t0     if t0     is not None else 0.0
    result["t_last"]             = t_last if t_last is not None else 0.0
    return result


# ═══════════════════════════════════════════════════════════════════════════════
#  Metric computation
# ═══════════════════════════════════════════════════════════════════════════════

def _rmse_mae(values):
    """Return (rmse, mae) for a list of floats, or (None, None) if empty."""
    if not values:
        return None, None
    a = np.array(values, dtype=float)
    rmse = float(np.sqrt(np.mean(a ** 2)))
    mae  = float(np.mean(np.abs(a)))
    return round(rmse, 6), round(mae, 6)


def _nmae(mae, window_s):
    if mae is None or window_s is None:
        return None
    return round(mae / max(window_s, 1e-6), 6)


def _ratio(rmse, mae):
    if rmse is None or mae is None or mae == 0:
        return None
    return round(rmse / mae, 4)


def _interp_series(primary, secondary):
    """
    For each (t, v1) in primary, interpolate secondary at time t.
    Returns array of (t, v1, v2_interp) only where interpolation is valid
    (i.e., t is within the time range of secondary).
    """
    if not secondary or not primary:
        return []
    sec = np.array(secondary)          # (N, 2): [[t, v], ...]
    pri = np.array(primary)            # (M, 2): [[t, v], ...]
    v2_interp = np.interp(pri[:, 0], sec[:, 0], sec[:, 1])
    return list(zip(pri[:, 0].tolist(), pri[:, 1].tolist(), v2_interp.tolist()))


def compute_scenario_metrics(path: str) -> dict:
    """
    Compute navigate performance metrics for a single scenario file.
    Returns a dict with all metrics, or {'notes': ..., 'error': True} on failure.
    """
    runway = _runway_from_filename(path)
    if runway is None:
        return {"notes": "runway not found in filename", "error": True}
    if runway not in RUNWAY_HEADING:
        return {"notes": f"unknown runway: {runway}", "error": True}

    rwy_hdg    = RUNWAY_HEADING[runway]
    rwy_lat, rwy_lon = RUNWAY_START[runway]

    sig = _extract_nav_signals(path)

    t0     = sig["t0"]
    t_last = sig["t_last"]
    t_pos  = sig["positive_rate_time"]  if sig["positive_rate_time"] else t0
    t_pan  = sig["panpan_time"]

    # Build synchronized (t, lat, lon) series using lat as master clock,
    # interpolating lon at lat timestamps.
    lat_lon_series = _interp_series(sig["latitude"], sig["longitude"])
    # (t, lat, lon_interp)

    # Build (t, alt) and (t, hdg) series
    alt_ts = sig["altitude"]
    hdg_ts = sig["heading"]

    notes = []

    # ── Detect altitude_5000_time (first sample ≥ 5 000 ft) ─────────────────
    t_5000 = None
    for t, alt in alt_ts:
        if alt >= TARGET_ALT:
            t_5000 = t
            break

    # ── Phase window boundaries ──────────────────────────────────────────────
    # Climb: t_pos → t_pan (or end of file)
    phase1_end = t_pan if t_pan is not None else t_last

    # Level-off: t_5000 → t_pan (or end of file)
    leveloff_start = t_5000
    leveloff_end   = phase1_end

    # ── Phase 1: XTE, ATD-error, HDG from lat/lon/hdg ───────────────────────
    climb_xte = []   # NM, signed
    climb_atd = []   # NM, error vs expected 120 kt
    climb_hdg = []   # deg, heading error

    for t, lat, lon in lat_lon_series:
        if t < t_pos or t > phase1_end:
            continue
        xte_m, atd_m = _xte_atd(lat, lon, rwy_lat, rwy_lon, rwy_hdg)
        expected_atd_m = TARGET_AIRSPEED * KT_TO_MPS * (t - t_pos)
        climb_xte.append(xte_m * NM_PER_M)
        climb_atd.append((atd_m - expected_atd_m) * NM_PER_M)

    hdg_interp_series = _interp_series(
        [(t, v) for t, v in hdg_ts if t_pos <= t <= phase1_end], []
    ) if False else [(t, v) for t, v in hdg_ts if t_pos <= t <= phase1_end]
    for t, hdg in hdg_interp_series:
        climb_hdg.append(_hdg_error(hdg, rwy_hdg))

    climb_window_s = phase1_end - t_pos if phase1_end > t_pos else None

    # ── Level-off: altitude error vs 5 000 ft ────────────────────────────────
    leveloff_alt = []
    if leveloff_start is not None and leveloff_end > leveloff_start:
        for t, alt in alt_ts:
            if leveloff_start <= t <= leveloff_end:
                leveloff_alt.append(alt - TARGET_ALT)   # signed (+ = above)
    else:
        notes.append("level-off window not detected (aircraft may not have reached 5000 ft before panpan/end of file)")

    leveloff_window_s = (
        leveloff_end - leveloff_start
        if (leveloff_start is not None and leveloff_end > leveloff_start)
        else None
    )

    # ── Compute RMSE/MAE/nMAE/ratio for each metric ───────────────────────────
    def _metrics(values, window_s, label):
        rmse, mae = _rmse_mae(values)
        nm = _nmae(mae, window_s)
        r  = _ratio(rmse, mae)
        return {
            f"{label}_rmse":   rmse,
            f"{label}_mae":    mae,
            f"{label}_nmae":   nm,
            f"{label}_ratio":  r,
        }

    result = {}
    result.update(_metrics(climb_xte,    climb_window_s,    "climb_xte"))
    result.update(_metrics(climb_atd,    climb_window_s,    "climb_atd"))
    result.update(_metrics(climb_hdg,    climb_window_s,    "climb_hdg"))
    result.update(_metrics(leveloff_alt, leveloff_window_s, "leveloff_alt"))

    result["n_climb"]    = len(climb_xte)
    result["n_leveloff"] = len(leveloff_alt)

    result["climb_window_s"]    = round(climb_window_s,    2) if climb_window_s else None
    result["leveloff_window_s"] = round(leveloff_window_s, 2) if leveloff_window_s else None

    result["runway"]  = runway
    result["rwy_hdg"] = rwy_hdg

    result["t0"]                     = round(t0, 3)
    result["positive_rate_time_rel"]  = round(t_pos - t0, 3) if sig["positive_rate_time"] else None
    result["altitude_5000_time_rel"]  = round(t_5000 - t0, 3) if t_5000 else None
    result["panpan_time_rel"]         = round(t_pan - t0, 3) if t_pan else None

    result["notes"] = notes
    return result


# ═══════════════════════════════════════════════════════════════════════════════
#  Per-condition aggregation
# ═══════════════════════════════════════════════════════════════════════════════

_NUM_KEYS = [
    "climb_xte_rmse",  "climb_xte_mae",  "climb_xte_nmae",  "climb_xte_ratio",
    "climb_atd_rmse",  "climb_atd_mae",  "climb_atd_nmae",  "climb_atd_ratio",
    "climb_hdg_rmse",  "climb_hdg_mae",  "climb_hdg_nmae",  "climb_hdg_ratio",
    "leveloff_alt_rmse","leveloff_alt_mae","leveloff_alt_nmae","leveloff_alt_ratio",
    "climb_window_s", "leveloff_window_s",
    "n_climb", "n_leveloff",
]


def _aggregate(metrics_list: list) -> dict:
    """Average numeric metrics over a list of per-scenario metric dicts."""
    if not metrics_list:
        return {k: None for k in _NUM_KEYS}
    result = {}
    for k in _NUM_KEYS:
        vals = [m[k] for m in metrics_list if m.get(k) is not None]
        result[k] = round(float(np.mean(vals)), 6) if vals else None
    result["scenario_count"] = len(metrics_list)
    result["notes"] = [n for m in metrics_list for n in m.get("notes", [])]
    return result


# ═══════════════════════════════════════════════════════════════════════════════
#  Reporting helpers
# ═══════════════════════════════════════════════════════════════════════════════

def cleaned_dir(pid: str) -> str:
    return os.path.join(HITLS_DIR, pid, "cleaned")


def _bar(value, max_val: float, width: int = 20) -> str:
    """ASCII progress bar — more fill = larger error (lower is better)."""
    if value is None:
        return "[" + "?" * width + "]"
    ratio  = min(abs(float(value)) / max_val, 1.0) if max_val > 0 else 0.0
    filled = int(ratio * width)
    return "[" + "█" * filled + "░" * (width - filled) + "]"


def _fmt(val, unit: str = "", decimals: int = 3) -> str:
    if val is None:
        return "N/A"
    return f"{val:.{decimals}f}{(' ' + unit) if unit else ''}"


def _ts_fmt(ts_rel) -> str:
    """Format a relative timestamp (seconds from scenario start)."""
    if ts_rel is None:
        return "not detected"
    return f"+{ts_rel:.3f} s"


_BAR_MAX = {
    "climb_xte_rmse":     2.0,
    "climb_xte_mae":      2.0,
    "climb_xte_nmae":     0.01,
    "climb_atd_rmse":     2.0,
    "climb_atd_mae":      2.0,
    "climb_atd_nmae":     0.01,
    "climb_hdg_rmse":    30.0,
    "climb_hdg_mae":     30.0,
    "climb_hdg_nmae":     0.15,
    "leveloff_alt_rmse": 200.0,
    "leveloff_alt_mae":  200.0,
    "leveloff_alt_nmae":   5.0,
}


def build_report(participant_id: str, condition_data: dict) -> str:
    sep  = "=" * 78
    dash = "─" * 78
    lines = []

    lines.append(f"\n{sep}")
    lines.append(f"  NAVIGATE PERFORMANCE REPORT — {participant_id}")
    lines.append(sep)
    lines.append(
        "\n  Metrics computed from Aircraft agent signals in scenario ingescape CSV files.\n"
        "  All phases start at the TAKEOFF \"Climb rate CHECK POSITIVE\" state.\n"
        "  Timestamps shown as +s.mmm relative to the first record in each scenario file.\n"
        "\n  Two measurement phases:\n"
        "    Climb     : pos-rate → DECLARE PANPAN\n"
        "    Level-off : first ≥ 5 000 ft → DECLARE PANPAN (altitude vs 5 000 ft)\n"
        "\n  XTE (NM) = lateral deviation from track  |  ATD (NM) = along-track vs 120 kt\n"
        "  HDG (°) = heading deviation                |  ALT (ft) = altitude vs 5 000 ft\n"
        "  nMAE = MAE / window_duration  |  RMSE/MAE: near 1.0 = uniform, higher = spiky\n"
    )

    for cond in CONDITIONS:
        if cond not in condition_data:
            continue

        m   = condition_data[cond]["aggregate"]
        scn = condition_data[cond]["scenarios"]

        _, first_sm = scn[0]
        rwy       = first_sm.get("runway", "?")
        rwy_hdg_v = first_sm.get("rwy_hdg")

        lines.append(f"\n{dash}")
        lines.append(
            f"  CONDITION: {cond}  ({len(scn)} scenario file(s))  "
            f"runway {rwy}  hdg {rwy_hdg_v}°"
        )
        lines.append(f"{dash}\n")

        # Per-scenario detail
        for i, (path, sm) in enumerate(scn, 1):
            bname = os.path.basename(path)
            cw = sm.get("climb_window_s")
            lw = sm.get("leveloff_window_s")
            lines.append(f"  Scenario {i}: {bname}")
            lines.append(
                f"    Climb rate CHECK POSITIVE : "
                f"{_ts_fmt(sm.get('positive_rate_time_rel'))}"
            )
            lines.append(f"    5 000 ft reached          : {_ts_fmt(sm.get('altitude_5000_time_rel'))}")
            lines.append(f"    DECLARE PANPAN            : {_ts_fmt(sm.get('panpan_time_rel'))}")
            lines.append(
                f"    Window durations          : "
                f"climb={_fmt(cw, 's', 1)}  "
                f"level-off={_fmt(lw, 's', 1)}"
            )
            lines.append(
                f"    Samples                   : "
                f"climb={sm.get('n_climb', 0)}  "
                f"level-off={sm.get('n_leveloff', 0)}"
            )
            for note in sm.get("notes", []):
                lines.append(f"    ⚠  {note}")
            lines.append("")

        if len(scn) > 1:
            lines.append(f"  (metrics below are means across {len(scn)} scenarios)\n")

        rwy_hdg_str = f"{rwy_hdg_v}°" if rwy_hdg_v else "?"

        col_w_label = 44
        col_w_val   = 11
        lines.append(
            f"  {'Metric':<{col_w_label}} {'Value':>{col_w_val}}   Bar (lower = better)"
        )
        lines.append(
            f"  {'─' * col_w_label}   {'─' * col_w_val}   {'─' * 22}"
        )

        metric_rows = [
            (f"── Climb (runway hdg {rwy_hdg_str}) ─────────────────────────", None,                    "",      False),
            ("Climb XTE RMSE",                             "climb_xte_rmse",    "NM",    True),
            ("Climb XTE MAE",                              "climb_xte_mae",     "NM",    True),
            ("Climb XTE nMAE (MAE/window)",                "climb_xte_nmae",    "NM/s",  True),
            ("Climb XTE RMSE/MAE",                         "climb_xte_ratio",   "",      False),
            ("Climb ATD RMSE",                             "climb_atd_rmse",    "NM",    True),
            ("Climb ATD MAE",                              "climb_atd_mae",     "NM",    True),
            ("Climb ATD nMAE (MAE/window)",                "climb_atd_nmae",    "NM/s",  True),
            ("Climb ATD RMSE/MAE",                         "climb_atd_ratio",   "",      False),
            ("Climb Heading RMSE",                         "climb_hdg_rmse",    "deg",   True),
            ("Climb Heading MAE",                          "climb_hdg_mae",     "deg",   True),
            ("Climb Heading nMAE (MAE/window)",             "climb_hdg_nmae",    "deg/s", True),
            ("Climb Heading RMSE/MAE",                     "climb_hdg_ratio",   "",      False),
            (f"── Level-off (alt vs 5 000 ft) ────────────────────────────", None,                    "",      False),
            ("Level-off Alt RMSE",                         "leveloff_alt_rmse", "ft",    True),
            ("Level-off Alt MAE",                          "leveloff_alt_mae",  "ft",    True),
            ("Level-off Alt nMAE (MAE/window)",             "leveloff_alt_nmae", "ft/s",  True),
            ("Level-off Alt RMSE/MAE",                     "leveloff_alt_ratio","",      False),
        ]

        for label, key, unit, has_bar in metric_rows:
            if key is None:
                lines.append(f"\n  {label}")
                continue
            val = m.get(key)
            bar = _bar(val, _BAR_MAX[key]) if has_bar and key in _BAR_MAX else ""
            lines.append(
                f"  {label:<{col_w_label}} {_fmt(val, unit, 3):>{col_w_val}}   {bar}"
            )

        unique_notes = sorted(set(m.get("notes", [])))
        if unique_notes:
            lines.append("")
            for note in unique_notes:
                lines.append(f"  ⚠  {note}")

    # ── Cross-condition summary table ─────────────────────────────────────────
    lines.append(f"\n\n{dash}")
    lines.append("  CROSS-CONDITION SUMMARY")
    lines.append(f"{dash}\n")

    present_conds = [c for c in CONDITIONS if c in condition_data]
    col_w = 12

    hdr = f"  {'Metric':<44} " + "  ".join(f"{c:>{col_w}}" for c in present_conds)
    lines.append(hdr)
    lines.append(f"  {'─' * 44} " + "  ".join("─" * col_w for _ in present_conds))

    def _summary_row(label, key, unit="", decimals=3):
        vals = []
        for c in present_conds:
            v = condition_data[c]["aggregate"].get(key)
            vals.append(_fmt(v, unit, decimals))
        return f"  {label:<44} " + "  ".join(f"{v:>{col_w}}" for v in vals)

    lines.append(_summary_row("Climb XTE RMSE (NM)",        "climb_xte_rmse",    "NM"))
    lines.append(_summary_row("Climb XTE MAE (NM)",         "climb_xte_mae",     "NM"))
    lines.append(_summary_row("Climb ATD RMSE (NM)",        "climb_atd_rmse",    "NM"))
    lines.append(_summary_row("Climb ATD MAE (NM)",         "climb_atd_mae",     "NM"))
    lines.append(_summary_row("Climb Heading RMSE (deg)",   "climb_hdg_rmse",    "deg"))
    lines.append(_summary_row("Climb Heading MAE (deg)",    "climb_hdg_mae",     "deg"))
    lines.append(_summary_row("Level-off Alt RMSE (ft)",    "leveloff_alt_rmse", "ft"))
    lines.append(_summary_row("Level-off Alt MAE (ft)",     "leveloff_alt_mae",  "ft"))

    lines.append(f"\n{sep}\n")
    return "\n".join(lines)


def build_summary(participant_id: str, condition_data: dict) -> dict:
    """Build the JSON-serialisable summary dict."""
    summary = {"participant": participant_id, "conditions": {}}

    for cond in CONDITIONS:
        if cond not in condition_data:
            continue

        m   = condition_data[cond]["aggregate"]
        scn = condition_data[cond]["scenarios"]
        _, first_sm = scn[0]

        summary["conditions"][cond] = {
            "scenario_files":     [os.path.basename(p) for p, _ in scn],
            "scenario_count":     len(scn),
            "runway":             first_sm.get("runway"),
            "runway_heading_deg": first_sm.get("rwy_hdg"),
            "climb": {
                "xte": {
                    "rmse":     m.get("climb_xte_rmse"),
                    "mae":      m.get("climb_xte_mae"),
                    "nmae":     m.get("climb_xte_nmae"),
                    "rmse_mae": m.get("climb_xte_ratio"),
                    "unit":     "NM",
                },
                "atd_error": {
                    "rmse":     m.get("climb_atd_rmse"),
                    "mae":      m.get("climb_atd_mae"),
                    "nmae":     m.get("climb_atd_nmae"),
                    "rmse_mae": m.get("climb_atd_ratio"),
                    "unit":     "NM",
                },
                "heading_error": {
                    "rmse":     m.get("climb_hdg_rmse"),
                    "mae":      m.get("climb_hdg_mae"),
                    "nmae":     m.get("climb_hdg_nmae"),
                    "rmse_mae": m.get("climb_hdg_ratio"),
                    "unit":     "deg",
                },
                "window_s": m.get("climb_window_s"),
            },
            "leveloff": {
                "alt_error": {
                    "rmse":     m.get("leveloff_alt_rmse"),
                    "mae":      m.get("leveloff_alt_mae"),
                    "nmae":     m.get("leveloff_alt_nmae"),
                    "rmse_mae": m.get("leveloff_alt_ratio"),
                    "unit":     "ft",
                },
                "window_s": m.get("leveloff_window_s"),
            },
        }

    return summary


def save_report(participant_id: str, report_text: str, summary_dict: dict):
    cdir = cleaned_dir(participant_id)
    os.makedirs(cdir, exist_ok=True)
    path = os.path.join(cdir, f"{participant_id}_navigate_perf_report.txt")
    content = (
        json.dumps(summary_dict, indent=2, ensure_ascii=False)
        + "\n--- END SUMMARY ---\n"
        + report_text
    )
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"\n  Report saved → {path}")


# ═══════════════════════════════════════════════════════════════════════════════
#  Entry point
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    participants = find_participants()
    if not participants:
        print("No participant folders found.")
        return

    sep = "=" * 78
    print(f"\n{sep}")
    print("  HITLS — Navigate Performance Analysis")
    print(f"{sep}")
    print("\nAvailable participants:")
    for i, p in enumerate(participants, 1):
        print(f"  {i}. {p}")

    while True:
        choice = input("\nSelect a participant (number or ID, e.g. 1 or P02): ").strip()
        if choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(participants):
                participant_id = participants[idx]
                break
        elif choice.upper() in participants:
            participant_id = choice.upper()
            break
        print("  Invalid choice.")

    print(f"\n  Analysing {participant_id} …\n")

    files_by_cond = find_scenario_files(participant_id)
    if not files_by_cond:
        print(f"  No valid scenario files found for {participant_id}.")
        return

    condition_data = {}

    for cond in CONDITIONS:
        paths = files_by_cond.get(cond, [])
        if not paths:
            continue

        print(f"  {cond}: {len(paths)} scenario file(s)")
        scn_results = []

        for path in paths:
            bname = os.path.basename(path)
            print(f"    Parsing {bname} …", end="", flush=True)
            try:
                metrics = compute_scenario_metrics(path)
                scn_results.append((path, metrics))
                print(
                    f" ✓  "
                    f"climb n={metrics.get('n_climb', 0)}  "
                    f"leveloff n={metrics.get('n_leveloff', 0)}"
                )
            except Exception as exc:
                print(f" ✗  {exc}")

        if scn_results:
            agg = _aggregate([sm for _, sm in scn_results])
            condition_data[cond] = {"scenarios": scn_results, "aggregate": agg}

    if not condition_data:
        print(f"\n  No metrics could be computed for {participant_id}.")
        return

    report_text  = build_report(participant_id, condition_data)
    summary_dict = build_summary(participant_id, condition_data)

    print(report_text)
    save_report(participant_id, report_text, summary_dict)


if __name__ == "__main__":
    main()
