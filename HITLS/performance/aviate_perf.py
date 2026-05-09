#!/usr/bin/env python3
"""
aviate_perf.py — Aviate Performance Analysis for a single HITLS participant.

Metrics extracted from Aircraft agent signals in scenario ingescape CSV files:

  1. Slip / Skid  (stability)
       Signal  : Aircraft;slip          (degrees, ball displacement)
       Target  : 0 deg  (ball centred — no adverse yaw / sideslip)
       Window  : positive rate (Climb rate CHECK POSITIVE) → end of scenario
       Reported: RMSE, MAE
       Note    : MAPE undefined for zero-target; MAE expressed in same unit.

  2. Roll Angle  (wings-level stability)
       Signal  : Aircraft;roll  (degrees)
       Target  : 0 deg  (wings level — no bank before ATC vectors the crew)
       Window  : positive rate → first DECLARE PANPAN procedure state
                 (if DECLARE PANPAN not detected, window extends to end of file)
       Reported: RMSE, MAE

  3. Airspeed During Climb
       Signal  : Aircraft;airspeed  (knots)
       Target  : 120 kt  (initial climb speed — maintain until 5 000 ft)
       Window  : positive rate → first time altitude ≥ 5 000 ft
                 (if 5 000 ft never reached, window extends to end of file)
       Reported: RMSE, MAPE (%)

Conditions  : TARS | TARC | TARP-S | TARP-F
              Detected automatically from the scenario filename.
              e.g. scenario_07_TARS_24L_ingescape.csv → TARS

Per-condition metrics are averaged over all valid (non-training, non-unfinished)
scenario files found for that participant × condition.

Output
------
  • Rich terminal report printed to stdout.
  • JSON summary saved to  {PID}/cleaned/{PID}_aviate_perf_report.txt
    (same format convention as questionnaire reports: JSON block followed by
     human-readable text, delimited by "--- END SUMMARY ---").
"""

import os
import re
import sys
import json
import glob
import numpy as np

# ── Paths ─────────────────────────────────────────────────────────────────────
PERF_DIR  = os.path.dirname(os.path.abspath(__file__))
HITLS_DIR = os.path.dirname(PERF_DIR)

# ── Experiment constants ───────────────────────────────────────────────────────
CONDITIONS        = ["TARS", "TARC", "TARP-S", "TARP-F"]
TARGET_AIRSPEED   = 120.0    # kt   — climb speed target
TARGET_ZERO       = 0.0      # deg  — ideal slip and roll rate
ALTITUDE_LEVEL    = 5000.0   # ft   — level-off altitude
_POSITIVE_RATE_TASK_OBJ = '""Climb rate""'
_POSITIVE_RATE_VALUE    = '""CHECK POSITIVE""'

# ── Filename parsing ───────────────────────────────────────────────────────────
_COND_RE      = re.compile(r"_(TARS|TARC|TARP-S|TARP-F)_", re.IGNORECASE)
_SKIP_SUBSTRINGS = [
    "training", "TRAINING",
    "unfinished", "UNFINISHED",
    "UNIFNISHED",       # observed typo in raw data
    "no_birds_strike",  # aborted / incomplete variant — excluded
    "birds",            # bird-strike event variant — excluded
]


# ═══════════════════════════════════════════════════════════════════════════════
#  File discovery
# ═══════════════════════════════════════════════════════════════════════════════

def find_participants():
    return [
        e for e in sorted(os.listdir(HITLS_DIR))
        if os.path.isdir(os.path.join(HITLS_DIR, e))
        and e.startswith("P") and e[1:].isdigit()
    ]


def cleaned_dir(pid):
    return os.path.join(HITLS_DIR, pid, "cleaned")


def _is_valid_scenario(path: str) -> bool:
    name = os.path.basename(path)
    if not name.endswith("_ingescape.csv"):
        return False
    for skip in _SKIP_SUBSTRINGS:
        if skip in name:
            return False
    return _COND_RE.search(name) is not None


def _condition_from_filename(path: str) -> str:
    m = _COND_RE.search(os.path.basename(path))
    return m.group(1).upper() if m else None


def find_scenario_files(pid: str) -> dict:
    """
    Return {condition: [one file path]} — at most one file per condition.
    Files matching any skip substring (training, unfinished, birds, etc.) are excluded.
    When multiple valid files exist for a condition the first (alphabetically) is used.
    """
    scen_dir = os.path.join(HITLS_DIR, pid, "scenarios")
    if not os.path.isdir(scen_dir):
        return {}
    out = {c: [] for c in CONDITIONS}
    for path in sorted(glob.glob(os.path.join(scen_dir, "*_ingescape.csv"))):
        if _is_valid_scenario(path):
            cond = _condition_from_filename(path)
            if cond in out and len(out[cond]) == 0:  # take only first valid file
                out[cond].append(path)
    return {c: v for c, v in out.items() if v}


# ═══════════════════════════════════════════════════════════════════════════════
#  CSV parsing
# ═══════════════════════════════════════════════════════════════════════════════

_AIRCRAFT_SIGNALS = {"slip", "roll", "airspeed", "altitude", "verticalSpeed"}


def _extract_signals(path: str) -> dict:
    """
    Parse a scenario ingescape CSV and return:
      {
        "slip":               [(timestamp, value), ...],
        "roll":               [(timestamp, value), ...],
        "airspeed":           [(timestamp, value), ...],
        "altitude":           [(timestamp, value), ...],
        "verticalSpeed":      [(timestamp, value), ...],
        "positive_rate_time": float | None,
        "panpan_time":        float | None,
        "t0":                 float,   # first timestamp in file
        "t_last":             float,   # last timestamp in file
      }
    The CSV columns (semicolon-delimited):
      [0] uuid  [1] timestamp  [2] agent  [3] source  [4] type  [5] (empty)  [6] value

    Phase markers are detected from current_state procedure lines:
      positive_rate_time : TAKEOFF procedure, task_object "Climb rate",
                           value "CHECK POSITIVE"
      panpan_time        : DECLARE PANPAN procedure (any task)
    """
    signals = {k: [] for k in _AIRCRAFT_SIGNALS}
    panpan_time        = None
    positive_rate_time = None
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

            # ── Aircraft numeric signals ──────────────────────────────────────
            if agent == "Aircraft" and source in _AIRCRAFT_SIGNALS:
                try:
                    signals[source].append((ts, float(parts[6])))
                except (ValueError, IndexError):
                    pass

            # ── Phase marker detection from current_state ─────────────────────
            # current_state rows carry a JSON blob with doubled-quote CSV encoding.
            if source == "current_state":
                # Positive rate of climb: TAKEOFF procedure, Climb rate CHECK POSITIVE
                if (positive_rate_time is None
                        and '""TAKEOFF""' in line
                        and _POSITIVE_RATE_TASK_OBJ in line
                        and _POSITIVE_RATE_VALUE in line):
                    positive_rate_time = ts

                # DECLARE PANPAN: first entry into that procedure
                if panpan_time is None:
                    if '""DECLARE PANPAN""' in line or '"DECLARE PANPAN"' in line:
                        panpan_time = ts

    result = {k: sorted(v) for k, v in signals.items()}
    result["positive_rate_time"] = positive_rate_time
    result["panpan_time"]        = panpan_time
    result["t0"]                 = t0     if t0     is not None else 0.0
    result["t_last"]             = t_last if t_last is not None else 0.0
    return result


# ═══════════════════════════════════════════════════════════════════════════════
#  Phase / window helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _altitude_5000_time(signals: dict):
    """Return the first timestamp when altitude ≥ ALTITUDE_LEVEL (5 000 ft)."""
    for ts, val in signals["altitude"]:
        if val >= ALTITUDE_LEVEL:
            return ts
    return None


def _window_values(series: list, t_start, t_end) -> np.ndarray:
    """Extract the float values from a (timestamp, value) series within [t_start, t_end].
    None bounds are treated as open (no restriction on that side)."""
    out = []
    for ts, val in series:
        if t_start is not None and ts < t_start:
            continue
        if t_end is not None and ts > t_end:
            break
        out.append(val)
    return np.array(out, dtype=float)





# ═══════════════════════════════════════════════════════════════════════════════
#  Statistics
# ═══════════════════════════════════════════════════════════════════════════════

def _rmse(arr: np.ndarray, target: float = 0.0):
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return None
    return float(np.sqrt(np.mean((arr - target) ** 2)))


def _mae(arr: np.ndarray, target: float = 0.0):
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return None
    return float(np.mean(np.abs(arr - target)))


def _mape(arr: np.ndarray, target: float):
    """MAPE only valid when target ≠ 0."""
    if target == 0.0:
        return None
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return None
    return float(np.mean(np.abs(arr - target) / abs(target)) * 100.0)


# ═══════════════════════════════════════════════════════════════════════════════
#  Per-scenario metrics
# ═══════════════════════════════════════════════════════════════════════════════

def compute_scenario_metrics(path: str) -> dict:
    """
    Compute all aviate metrics for a single scenario file.

    Returns a dict with the following keys:
      slip_rmse, slip_mae, slip_nmae, slip_ratio       — slip/skid analysis
      roll_rmse, roll_mae, roll_nmae, roll_ratio        — roll angle (straight climb)
      airspeed_rmse, airspeed_mape,
        airspeed_nmae, airspeed_ratio                   — airspeed during climb
      slip_window_s, roll_window_s, airspeed_window_s   — actual window durations (s)
      n_slip, n_roll, n_airspeed                        — sample counts
      t0                                                — scenario start (abs)
      positive_rate_time_rel, panpan_time_rel,
        altitude_5000_time_rel                          — relative timestamps (s)
      notes                                             — list of warning strings

    *_nmae  = MAE / window_duration_s  (error per second — comparable across conditions)
    *_ratio = RMSE / MAE               (>1 — closer to 1 = uniform errors; higher = spiky)
    """
    sig    = _extract_signals(path)
    t0     = sig["t0"]
    t_last = sig["t_last"]
    t_pos  = sig["positive_rate_time"]
    t_5000 = _altitude_5000_time(sig)
    t_pan  = sig["panpan_time"]
    notes  = []
    t_start = t_pos if t_pos is not None else t0   # window start for all metrics

    if t_pos is None:
        notes.append(
            "No TAKEOFF Climb rate CHECK POSITIVE state detected — "
            "slip / roll angle / airspeed windows start from beginning of file"
        )

    # ── 1. Slip / Skid ────────────────────────────────────────────────────────
    # Window: positive rate → end of scenario
    # Window endpoints (actual timestamps used as upper bounds)
    t_slip_end    = t_last
    t_roll_end    = t_pan   if t_pan   is not None else t_last
    t_airspd_end  = t_5000  if t_5000  is not None else t_last

    slip_arr = _window_values(sig["slip"], t_pos, None)
    s_rmse   = _rmse(slip_arr, TARGET_ZERO)
    s_mae    = _mae(slip_arr,  TARGET_ZERO)
    s_win    = max(t_slip_end - t_start, 1e-6)         # avoid /0
    s_nmae   = round(s_mae / s_win, 6) if s_mae is not None else None
    s_ratio  = round(s_rmse / s_mae, 4) if (s_rmse and s_mae) else None

    # ── 2. Roll Angle ─────────────────────────────────────────────────────────
    # Window: positive rate → DECLARE PANPAN (first intentional bank by ATC)
    if t_pan is None:
        notes.append(
            "No DECLARE PANPAN detected — roll angle window extends to end of file"
        )
    roll_arr = _window_values(sig["roll"], t_pos, t_pan)
    r_rmse   = _rmse(roll_arr, TARGET_ZERO)
    r_mae    = _mae(roll_arr,  TARGET_ZERO)
    r_win    = max(t_roll_end - t_start, 1e-6)
    r_nmae   = round(r_mae / r_win, 6) if r_mae is not None else None
    r_ratio  = round(r_rmse / r_mae, 4) if (r_rmse and r_mae) else None

    # ── 3. Airspeed During Climb ──────────────────────────────────────────────
    # Window: positive rate → 5 000 ft
    if t_5000 is None:
        notes.append(
            f"Aircraft never reached {ALTITUDE_LEVEL:.0f} ft — "
            "airspeed window extends to end of file"
        )
    asp_arr = _window_values(sig["airspeed"], t_pos, t_5000)
    a_rmse  = _rmse(asp_arr, TARGET_AIRSPEED)
    a_mape  = _mape(asp_arr, TARGET_AIRSPEED)
    a_mae   = _mae(asp_arr, TARGET_AIRSPEED)
    a_win   = max(t_airspd_end - t_start, 1e-6)
    a_nmae  = round(a_mae / a_win, 6) if a_mae is not None else None
    a_ratio = round(a_rmse / a_mae, 4) if (a_rmse and a_mae) else None

    def _rel(ts):
        """Convert absolute timestamp to seconds relative to scenario start."""
        return round(ts - t0, 3) if ts is not None else None

    return {
        "slip_rmse":        s_rmse,
        "slip_mae":         s_mae,
        "slip_nmae":        s_nmae,
        "slip_ratio":       s_ratio,
        "slip_window_s":    round(s_win, 2),
        "roll_rmse":        r_rmse,
        "roll_mae":         r_mae,
        "roll_nmae":        r_nmae,
        "roll_ratio":       r_ratio,
        "roll_window_s":    round(r_win, 2),
        "airspeed_rmse":    a_rmse,
        "airspeed_mape":    a_mape,
        "airspeed_nmae":    a_nmae,
        "airspeed_ratio":   a_ratio,
        "airspeed_window_s": round(a_win, 2),
        "n_slip":           len(slip_arr),
        "n_roll":           len(roll_arr),
        "n_airspeed":       len(asp_arr),
        "t0":                     t0,
        "positive_rate_time_rel": _rel(t_pos),
        "panpan_time_rel":        _rel(t_pan),
        "altitude_5000_time_rel": _rel(t_5000),
        "notes":                  notes,
    }


def _aggregate(metrics_list: list) -> dict:
    """
    Average numeric metrics across multiple scenario files for the same condition.
    """
    if not metrics_list:
        return {}

    num_keys = [
        "slip_rmse",  "slip_mae",  "slip_nmae",  "slip_ratio",  "slip_window_s",
        "roll_rmse",  "roll_mae",  "roll_nmae",  "roll_ratio",  "roll_window_s",
        "airspeed_rmse", "airspeed_mape", "airspeed_nmae", "airspeed_ratio", "airspeed_window_s",
    ]
    agg = {}
    for k in num_keys:
        vals = [m[k] for m in metrics_list if m.get(k) is not None]
        agg[k] = round(float(np.mean(vals)), 4) if vals else None

    agg["scenario_count"] = len(metrics_list)
    agg["notes"] = [n for m in metrics_list for n in m.get("notes", [])]
    return agg


# ═══════════════════════════════════════════════════════════════════════════════
#  Report builders
# ═══════════════════════════════════════════════════════════════════════════════

def _bar(value, max_val: float, width: int = 20) -> str:
    """ASCII bar where more fill = larger error (lower is better)."""
    if value is None:
        return "[" + "?" * width + "]"
    ratio  = min(float(value) / max_val, 1.0) if max_val > 0 else 0.0
    filled = int(ratio * width)
    return "[" + "█" * filled + "░" * (width - filled) + "]"


def _fmt(val, unit: str = "", decimals: int = 3) -> str:
    if val is None:
        return "N/A"
    return f"{val:.{decimals}f}{(' ' + unit) if unit else ''}"


def _ts_fmt(ts_rel) -> str:
    """Format a relative timestamp (seconds from scenario start) as s.mmm."""
    if ts_rel is None:
        return "not detected"
    return f"+{ts_rel:.3f} s"


def build_report(participant_id: str, condition_data: dict) -> str:
    sep  = "=" * 78
    dash = "─" * 78
    lines = []

    lines.append(f"\n{sep}")
    lines.append(f"  AVIATE PERFORMANCE REPORT — {participant_id}")
    lines.append(sep)
    lines.append(
        "\n  Metrics computed from Aircraft agent signals in scenario ingescape CSV files.\n"
        "  All flight metrics start at the TAKEOFF \"Climb rate CHECK POSITIVE\" state.\n"
        "  Timestamps shown as +s.mmm relative to the first record in each scenario file.\n"
        "\n  Targets:  Slip = 0 deg  |  Roll Angle = 0 deg  |  Airspeed = "
        f"{TARGET_AIRSPEED:.0f} kt (climb)\n"
        f"  Roll angle window ends at DECLARE PANPAN (first intentional ATC turn).\n"
        "  nMAE = MAE / window_duration (deg/s or kt/s — comparable across conditions).\n"
        "  RMSE/MAE ratio: near 1.0 = uniform errors; higher = a few large spikes.\n"
        f"  Airspeed window ends when altitude reaches {ALTITUDE_LEVEL:.0f} ft.\n"
    )

    # Reference max values for bar scaling (choose representative upper bounds)
    _BAR_MAX = {
        "slip_rmse":      2.0,
        "slip_mae":       2.0,
        "slip_nmae":      0.02,
        "roll_rmse":      15.0,
        "roll_mae":       15.0,
        "roll_nmae":      0.1,
        "airspeed_rmse":  30.0,
        "airspeed_mape":  25.0,
        "airspeed_nmae":  0.2,
    }

    for cond in CONDITIONS:
        if cond not in condition_data:
            continue

        m   = condition_data[cond]["aggregate"]
        scn = condition_data[cond]["scenarios"]

        lines.append(f"\n{dash}")
        lines.append(f"  CONDITION: {cond}  ({len(scn)} scenario file(s))")
        lines.append(f"{dash}\n")

        # Per-scenario detail
        for i, (path, sm) in enumerate(scn, 1):
            bname = os.path.basename(path)
            lines.append(f"  Scenario {i}: {bname}")
            lines.append(
                f"    Climb rate CHECK POSITIVE : "
                f"{_ts_fmt(sm['positive_rate_time_rel'])}"
            )
            lines.append(f"    DECLARE PANPAN            : {_ts_fmt(sm['panpan_time_rel'])}")
            lines.append(f"    5 000 ft reached          : {_ts_fmt(sm['altitude_5000_time_rel'])}")
            lines.append(
                f"    Window durations          : "
                f"slip={sm['slip_window_s']:.1f}s  "
                f"roll={sm['roll_window_s']:.1f}s  "
                f"airspeed={sm['airspeed_window_s']:.1f}s"
            )
            lines.append(
                f"    Samples                   : "
                f"slip={sm['n_slip']}  roll={sm['n_roll']}  airspeed={sm['n_airspeed']}"
            )
            for note in sm.get("notes", []):
                lines.append(f"    ⚠  {note}")
            lines.append("")

        # Aggregated metric table
        if len(scn) > 1:
            lines.append(f"  (metrics below are means across {len(scn)} scenarios)\n")

        col_w_label = 40
        col_w_val   = 11
        lines.append(
            f"  {'Metric':<{col_w_label}} {'Value':>{col_w_val}}   Bar (lower = better)"
        )
        lines.append(
            f"  {'─' * col_w_label}   {'─' * col_w_val}   {'─' * 22}"
        )

        metric_rows = [
            ("Slip RMSE",                       "slip_rmse",      "deg",   True),
            ("Slip MAE",                        "slip_mae",       "deg",   True),
            ("Slip nMAE (MAE/window)",          "slip_nmae",      "deg/s", True),
            ("Slip RMSE/MAE",                   "slip_ratio",     "",      False),
            ("Roll Angle RMSE  [pre-panpan]",   "roll_rmse",      "deg",   True),
            ("Roll Angle MAE   [pre-panpan]",   "roll_mae",       "deg",   True),
            ("Roll nMAE (MAE/window)",          "roll_nmae",      "deg/s", True),
            ("Roll RMSE/MAE",                   "roll_ratio",     "",      False),
            (f"Airspeed RMSE  (target {TARGET_AIRSPEED:.0f} kt)", "airspeed_rmse",  "kt", True),
            (f"Airspeed MAPE  (target {TARGET_AIRSPEED:.0f} kt)", "airspeed_mape",  "%",  True),
            ("Airspeed nMAE (MAE/window)",      "airspeed_nmae",  "kt/s",  True),
            ("Airspeed RMSE/MAE",               "airspeed_ratio", "",      False),
        ]
        for label, key, unit, has_bar in metric_rows:
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

    hdr = f"  {'Metric':<40} " + "  ".join(f"{c:>{col_w}}" for c in present_conds)
    lines.append(hdr)
    lines.append(f"  {'─' * 40} " + "  ".join("─" * col_w for _ in present_conds))

    def _summary_row(label, key, unit="", decimals=3):
        vals = []
        for c in present_conds:
            v = condition_data[c]["aggregate"].get(key)
            vals.append(_fmt(v, unit, decimals))
        return f"  {label:<40} " + "  ".join(f"{v:>{col_w}}" for v in vals)

    lines.append(_summary_row("Slip RMSE (deg)",             "slip_rmse",  "deg"))
    lines.append(_summary_row("Slip MAE  (deg)",             "slip_mae",   "deg"))
    lines.append(_summary_row("Slip nMAE (deg/s)",           "slip_nmae",  "deg/s", 4))
    lines.append(_summary_row("Slip RMSE/MAE",               "slip_ratio", "",      3))
    lines.append(_summary_row("Roll Angle RMSE (deg)",       "roll_rmse",  "deg"))
    lines.append(_summary_row("Roll Angle MAE  (deg)",       "roll_mae",   "deg"))
    lines.append(_summary_row("Roll nMAE (deg/s)",           "roll_nmae",  "deg/s", 4))
    lines.append(_summary_row("Roll RMSE/MAE",               "roll_ratio", "",      3))
    lines.append(_summary_row(f"Airspeed RMSE vs {TARGET_AIRSPEED:.0f} kt (kt)",
                               "airspeed_rmse",  "kt"))
    lines.append(_summary_row(f"Airspeed MAPE vs {TARGET_AIRSPEED:.0f} kt (%)",
                               "airspeed_mape",  "%",  1))
    lines.append(_summary_row("Airspeed nMAE (kt/s)",
                               "airspeed_nmae",  "kt/s", 4))
    lines.append(_summary_row("Airspeed RMSE/MAE",
                               "airspeed_ratio", "",     3))

    lines.append(f"\n{sep}\n")
    return "\n".join(lines)


def build_summary(participant_id: str, condition_data: dict) -> dict:
    """Build the JSON-serialisable summary dict (same convention as questionnaire reports)."""
    summary = {"participant": participant_id, "conditions": {}}

    for cond in CONDITIONS:
        if cond not in condition_data:
            continue

        m   = condition_data[cond]["aggregate"]
        scn = condition_data[cond]["scenarios"]

        summary["conditions"][cond] = {
            "scenario_files":  [os.path.basename(p) for p, _ in scn],
            "scenario_count":  len(scn),
            "slip": {
                "rmse":      m.get("slip_rmse"),
                "mae":       m.get("slip_mae"),
                "nmae":      m.get("slip_nmae"),
                "rmse_mae":  m.get("slip_ratio"),
                "window_s":  m.get("slip_window_s"),
                "target":    TARGET_ZERO,
                "unit":      "deg",
                "window":    "climb-rate-check-positive to end of scenario",
            },
            "roll_angle": {
                "rmse":      m.get("roll_rmse"),
                "mae":       m.get("roll_mae"),
                "nmae":      m.get("roll_nmae"),
                "rmse_mae":  m.get("roll_ratio"),
                "window_s":  m.get("roll_window_s"),
                "target":    TARGET_ZERO,
                "unit":      "deg",
                "window":    "climb-rate-check-positive to DECLARE PANPAN (or end of file if not detected)",
            },
            "airspeed_climb": {
                "rmse":      m.get("airspeed_rmse"),
                "mape":      m.get("airspeed_mape"),
                "nmae":      m.get("airspeed_nmae"),
                "rmse_mae":  m.get("airspeed_ratio"),
                "window_s":  m.get("airspeed_window_s"),
                "target":    TARGET_AIRSPEED,
                "unit":      "kt",
                "window":    f"climb-rate-check-positive to {ALTITUDE_LEVEL:.0f} ft (or end of file if not reached)",
            },
        }

    return summary


# ═══════════════════════════════════════════════════════════════════════════════
#  I/O
# ═══════════════════════════════════════════════════════════════════════════════

def save_report(participant_id: str, report_text: str, summary_dict: dict):
    cdir = cleaned_dir(participant_id)
    os.makedirs(cdir, exist_ok=True)
    path = os.path.join(cdir, f"{participant_id}_aviate_perf_report.txt")
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
    print("  HITLS — Aviate Performance Analysis")
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
                    f"slip n={metrics['n_slip']}  "
                    f"roll n={metrics['n_roll']}  "
                    f"as n={metrics['n_airspeed']}"
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
