#!/usr/bin/env python3
"""
time_perf.py — Timing Performance Analysis for a single HITLS participant.

Metrics extracted from current_state events in scenario ingescape CSV files.
Each state-machine row carries a JSON payload describing the active task
(procedure, task_object, value, transition_kind …).

──────────────────────────────────────────────────────────────────────────────
Measurement windows
──────────────────────────────────────────────────────────────────────────────
  Scenario duration
    From : BEFORE TAKEOFF  — task "Takeoff clearance"
    To   : AFTER TAKEOFF   — task "Checklist" (ORDER START)
    Unit : seconds

  Failure-to-nominal
    From : first task of procedure "ENG FAILURE DURING TAKEOFF"
    To   : ENGINE FIRE — task "Next Checklist" value "ENGINE FAILURE/
           PRECAUTIONARY SHUTDOWN"
    Unit : seconds

  Per-task timing (for every procedure)
    Duration = wall-clock time between consecutive state-machine events
               (tasks from different procedures may interleave; all
               durations are measured in global time order).

──────────────────────────────────────────────────────────────────────────────
Procedure sequence observed in experiment scenarios
──────────────────────────────────────────────────────────────────────────────
  CREW BRIEFING  →  BEFORE TAKEOFF  →  LINE-UP AND HOLD  →  TAKEOFF
  →  ENG FAILURE DURING TAKEOFF  (interleaved with)  ENGINE FIRE
  →  DECLARE PANPAN  →  AFTER TAKEOFF

──────────────────────────────────────────────────────────────────────────────
CSV format
──────────────────────────────────────────────────────────────────────────────
  Semicolon-delimited, 7 cols (split with maxsplit=6):
    [0] uuid  [1] timestamp  [2] agent  [3] source  [4] type
    [5] igs_ts  [6] value

  current_state rows have source == "current_state" and a JSON value
  wrapped in CSV double-quotes (outer " stripped; inner "" → ").

Conditions  : TARS | TARC | TARP-S | TARP-F
Output      : {PID}/cleaned/{PID}_time_perf_report.txt

Usage
-----
  python performance/time_perf.py           # interactive participant selection
  python performance/time_perf.py P02       # run directly for participant P02
  python performance/time_perf.py 3         # run for 3rd participant in the list
"""

import os
import re
import sys
import json
import glob
import argparse
import numpy as np

# ── Paths ─────────────────────────────────────────────────────────────────────
PERF_DIR  = os.path.dirname(os.path.abspath(__file__))
HITLS_DIR = os.path.dirname(PERF_DIR)

# ── Experiment constants ───────────────────────────────────────────────────────
CONDITIONS = ["TARS", "TARC", "TARP-S", "TARP-F"]

# ── Procedure order for display ────────────────────────────────────────────────
_PROC_KEY = {
    "CREW BRIEFING":               "crew_briefing",
    "BEFORE TAKEOFF":              "before_takeoff",
    "LINE-UP AND HOLD":            "lineup_hold",
    "TAKEOFF":                     "takeoff",
    "ENG FAILURE DURING TAKEOFF":  "eng_failure",
    "ENGINE FIRE":                 "engine_fire",
    "DECLARE PANPAN":              "declare_panpan",
    "AFTER TAKEOFF":               "after_takeoff",
}

# ── Milestone detection strings ────────────────────────────────────────────────
_START_PROC   = "BEFORE TAKEOFF"
_START_TASK   = "Takeoff clearance"

_FAILURE_PROC = "ENG FAILURE DURING TAKEOFF"

_NOMINAL_PROC = "ENGINE FIRE"
_NOMINAL_TASK = "Next Checklist"
_NOMINAL_VAL  = "ENGINE FAILURE"           # substring — matches full value

_END_PROC  = "AFTER TAKEOFF"
_END_TASK  = "Checklist"

_END_FALLBACK_PROC = "DECLARE PANPAN"   # used when AFTER TAKEOFF Checklist absent
_END_FALLBACK_TASK = "Heading"
_END_FALLBACK_VAL  = "SET ACCORDINGLY"

_END_FALLBACK2_PROC = "DECLARE PANPAN"  # last-resort: last DECLARE PANPAN task

# ── Skip patterns (same as navigate_perf) ─────────────────────────────────────
_SKIP_SUBSTRINGS = [
    "training", "TRAINING",
    "unfinished", "UNFINISHED", "UNIFNISHED",
    "no_birds_strike", "birds",
]

# ── Aggregatable numeric keys ──────────────────────────────────────────────────
_NUM_KEYS = [
    "scenario_duration_s",
    "failure_to_nominal_s",
    "crew_briefing_n_tasks",    "crew_briefing_total_s",    "crew_briefing_mean_task_s",
    "before_takeoff_n_tasks",   "before_takeoff_total_s",   "before_takeoff_mean_task_s",
    "lineup_hold_n_tasks",      "lineup_hold_total_s",      "lineup_hold_mean_task_s",
    "takeoff_n_tasks",          "takeoff_total_s",          "takeoff_mean_task_s",
    "eng_failure_n_tasks",      "eng_failure_total_s",      "eng_failure_mean_task_s",
    "engine_fire_n_tasks",      "engine_fire_total_s",      "engine_fire_mean_task_s",
    "declare_panpan_n_tasks",   "declare_panpan_total_s",   "declare_panpan_mean_task_s",
    "after_takeoff_n_tasks",    "after_takeoff_total_s",    "after_takeoff_mean_task_s",
]

# ── Bar chart scale (upper bound for full bar) ─────────────────────────────────
_BAR_MAX = {
    "scenario_duration_s":          900.0,
    "failure_to_nominal_s":         300.0,
    "crew_briefing_total_s":        600.0,
    "crew_briefing_mean_task_s":     60.0,
    "before_takeoff_total_s":       300.0,
    "before_takeoff_mean_task_s":    30.0,
    "lineup_hold_total_s":          120.0,
    "lineup_hold_mean_task_s":       30.0,
    "takeoff_total_s":              120.0,
    "takeoff_mean_task_s":           30.0,
    "eng_failure_total_s":          300.0,
    "eng_failure_mean_task_s":       60.0,
    "engine_fire_total_s":          300.0,
    "engine_fire_mean_task_s":       60.0,
    "declare_panpan_total_s":       120.0,
    "declare_panpan_mean_task_s":    30.0,
    "after_takeoff_total_s":        300.0,
    "after_takeoff_mean_task_s":     60.0,
}

# ── File-name patterns (shared with navigate_perf) ────────────────────────────
_COND_RE   = re.compile(r"_(TARS|TARC|TARP-S|TARP-F)_", re.IGNORECASE)
_RUNWAY_RE = re.compile(r"_(06L|06R|24L|24R)[_.]")


# ═══════════════════════════════════════════════════════════════════════════════
#  File helpers
# ═══════════════════════════════════════════════════════════════════════════════

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
#  CSV / JSON parsing
# ═══════════════════════════════════════════════════════════════════════════════

def _parse_current_state_rows(path: str) -> list:
    """
    Return list of (timestamp_float, state_dict) for all current_state rows.

    current_state rows contain a JSON value wrapped in CSV double-quotes:
      outer "..." stripped, internal "" → "
    Splitting with maxsplit=6 keeps the full value even when it contains ';'.
    """
    rows = []
    ts_scale = 1.0  # will be set to 1e-6 for relative_time_us columns
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            # Detect header to determine timestamp unit
            if line.startswith("uuid;"):
                header_col1 = line.rstrip("\r\n").split(";", 2)[1]
                if header_col1 == "relative_time_us":
                    ts_scale = 1e-6  # microseconds → seconds
                continue
            if "current_state" not in line:
                continue
            parts = line.rstrip("\r\n").split(";", 6)
            if len(parts) < 7:
                continue
            try:
                ts = float(parts[1]) * ts_scale
            except ValueError:
                continue
            val = parts[6]
            # CSV-unquote: strip surrounding " and unescape "" → "
            if val.startswith('"') and val.endswith('"'):
                val = val[1:-1].replace('""', '"')
            try:
                d = json.loads(val)
                rows.append((ts, d))
            except (json.JSONDecodeError, ValueError):
                pass
    return rows


# ═══════════════════════════════════════════════════════════════════════════════
#  Metric computation
# ═══════════════════════════════════════════════════════════════════════════════

def compute_scenario_metrics(path: str) -> dict:
    """
    Parse one scenario ingescape CSV and return a metrics dict.

    Returns
    -------
    dict with keys:
      t0, takeoff_clearance_rel, failure_onset_rel, engine_fire_start_rel,
      nominal_recovery_rel, after_takeoff_start_rel,
      scenario_duration_s, failure_to_nominal_s,
      procedures: {proc_name: {n_tasks, total_s, mean_task_s, tasks: [...]}},
      notes: [str],
      + flat keys per procedure for aggregation (e.g. eng_failure_n_tasks).
    """
    rows = _parse_current_state_rows(path)
    if not rows:
        raise ValueError("no current_state rows found in file")

    t0 = rows[0][0]
    notes = []

    # ── Build global task list with wall-clock duration to next event ──────────
    global_tasks = []
    for i, (ts, d) in enumerate(rows):
        dur = rows[i + 1][0] - ts if i < len(rows) - 1 else None
        global_tasks.append({
            "ts":                 ts,
            "ts_rel":             ts - t0,
            "procedure":          d.get("procedure", ""),
            "task_object":        d.get("task_object", ""),
            "value":              d.get("value", ""),
            "transition_kind":    d.get("transition_kind", ""),
            "duration_to_next_s": dur,
        })

    # ── Milestone timestamps ───────────────────────────────────────────────────
    t_start           = None   # BEFORE TAKEOFF / Takeoff clearance
    t_start_fallback  = None   # first BEFORE TAKEOFF task (any)
    t_failure         = None   # first ENG FAILURE DURING TAKEOFF
    t_engine_fire_start = None # first ENGINE FIRE
    t_nominal         = None   # ENGINE FIRE / Next Checklist / ENGINE FAILURE…
    t_after_takeoff   = None   # AFTER TAKEOFF / Checklist
    t_fallback_end    = None   # DECLARE PANPAN / Heading SET ACCORDINGLY
    t_fallback2_end   = None   # last DECLARE PANPAN task (any)

    for task in global_tasks:
        proc = task["procedure"]
        obj  = task["task_object"]
        val  = task["value"]
        ts   = task["ts"]

        if t_start is None and proc == _START_PROC and obj == _START_TASK:
            t_start = ts

        if t_start_fallback is None and proc == _START_PROC:
            t_start_fallback = ts  # first BEFORE TAKEOFF task, whatever it is

        if t_failure is None and proc == _FAILURE_PROC:
            t_failure = ts

        if t_engine_fire_start is None and proc == _NOMINAL_PROC:
            t_engine_fire_start = ts

        if (t_nominal is None
                and proc == _NOMINAL_PROC
                and obj  == _NOMINAL_TASK
                and _NOMINAL_VAL in val):
            t_nominal = ts

        if (t_after_takeoff is None
                and proc == _END_PROC
                and obj  == _END_TASK):
            t_after_takeoff = ts

        if (t_fallback_end is None
                and proc == _END_FALLBACK_PROC
                and obj  == _END_FALLBACK_TASK
                and _END_FALLBACK_VAL in val):
            t_fallback_end = ts

        if proc == _END_FALLBACK2_PROC:
            t_fallback2_end = ts  # updated each time → keeps last DECLARE PANPAN task

    # ── Apply fallbacks if primary end marker missing ──────────────────────────
    if t_after_takeoff is None and t_fallback_end is not None:
        t_after_takeoff = t_fallback_end
        notes.append("scenario end: fallback to PANPAN 'Heading SET ACCORDINGLY' (no AFTER TAKEOFF Checklist)")
    elif t_after_takeoff is None and t_fallback2_end is not None:
        t_after_takeoff = t_fallback2_end
        notes.append("scenario end: fallback to last DECLARE PANPAN task (no AFTER TAKEOFF Checklist or Heading SET ACCORDINGLY)")

    # ── Apply fallback for start marker ───────────────────────────────────────
    if t_start is None and t_start_fallback is not None:
        t_start = t_start_fallback
        notes.append("scenario start: fallback to first BEFORE TAKEOFF task (no Takeoff clearance found)")

    # ── Derived durations ──────────────────────────────────────────────────────
    scenario_duration_s   = None
    failure_to_nominal_s  = None

    if t_start is not None and t_after_takeoff is not None:
        scenario_duration_s = round(t_after_takeoff - t_start, 2)
    else:
        if t_start is None:
            notes.append("scenario start marker not found (no BEFORE TAKEOFF task)")
        if t_after_takeoff is None:
            notes.append("scenario end marker not found (no AFTER TAKEOFF Checklist or PANPAN Heading)")

    if t_failure is not None and t_nominal is not None:
        failure_to_nominal_s = round(t_nominal - t_failure, 2)
    else:
        if t_failure is None:
            notes.append("ENG FAILURE DURING TAKEOFF marker not found")
        if t_nominal is None:
            notes.append("ENGINE FIRE nominal recovery marker not found")

    # ── Per-procedure data ─────────────────────────────────────────────────────
    proc_buckets: dict = {}
    for task in global_tasks:
        p = task["procedure"]
        proc_buckets.setdefault(p, []).append(task)

    proc_metrics: dict = {}
    for proc, tasks in proc_buckets.items():
        n      = len(tasks)
        t_proc = tasks[0]["ts"]
        # Total = last task ts – first task ts (0 if single task)
        total_s = round(tasks[-1]["ts"] - tasks[0]["ts"], 2) if n > 1 else 0.0
        # Mean wall-clock duration per task (exclude last global task if no next)
        durs = [
            t["duration_to_next_s"]
            for t in tasks
            if t["duration_to_next_s"] is not None
        ]
        mean_s = round(float(np.mean(durs)), 2) if durs else None

        proc_metrics[proc] = {
            "n_tasks":     n,
            "total_s":     total_s,
            "mean_task_s": mean_s,
            "tasks": [
                {
                    "task_object":        t["task_object"],
                    "value":              t["value"],
                    "elapsed_s":          round(t["ts"] - t_proc, 2),
                    "duration_to_next_s": (
                        round(t["duration_to_next_s"], 2)
                        if t["duration_to_next_s"] is not None else None
                    ),
                }
                for t in tasks
            ],
        }

    # ── Assemble result ────────────────────────────────────────────────────────
    result = {
        "t0":                    round(t0, 3),
        "takeoff_clearance_rel": round(t_start - t0, 2)          if t_start            else None,
        "failure_onset_rel":     round(t_failure - t0, 2)        if t_failure          else None,
        "engine_fire_start_rel": round(t_engine_fire_start - t0, 2) if t_engine_fire_start else None,
        "nominal_recovery_rel":  round(t_nominal - t0, 2)        if t_nominal          else None,
        "after_takeoff_start_rel": round(t_after_takeoff - t0, 2) if t_after_takeoff   else None,
        "scenario_duration_s":   scenario_duration_s,
        "failure_to_nominal_s":  failure_to_nominal_s,
        "procedures":            proc_metrics,
        "notes":                 notes,
    }

    # Flat keys for _aggregate()
    for proc_name, proc_key in _PROC_KEY.items():
        pm = proc_metrics.get(proc_name, {})
        result[f"{proc_key}_n_tasks"]    = pm.get("n_tasks",     0)
        result[f"{proc_key}_total_s"]    = pm.get("total_s")
        result[f"{proc_key}_mean_task_s"]= pm.get("mean_task_s")

    return result


def _aggregate(metrics_list: list) -> dict:
    """Average numeric keys across multiple scenario metrics dicts."""
    if not metrics_list:
        return {k: None for k in _NUM_KEYS}
    result = {}
    for k in _NUM_KEYS:
        vals = [m[k] for m in metrics_list if isinstance(m.get(k), (int, float))]
        result[k] = round(float(np.mean(vals)), 2) if vals else None
    result["scenario_count"] = len(metrics_list)
    result["notes"] = [n for m in metrics_list for n in m.get("notes", [])]
    return result


# ═══════════════════════════════════════════════════════════════════════════════
#  Reporting helpers
# ═══════════════════════════════════════════════════════════════════════════════

def cleaned_dir(pid: str) -> str:
    return os.path.join(HITLS_DIR, pid, "cleaned")


def _bar(value, max_val: float, width: int = 20) -> str:
    """ASCII progress bar — more fill = larger value (lower = better)."""
    if value is None or max_val <= 0:
        return "[" + "?" * width + "]"
    filled = int(round(min(value / max_val, 1.0) * width))
    return "[" + "█" * filled + "░" * (width - filled) + "]"


def _fmt(val, unit: str = "", decimals: int = 1) -> str:
    if val is None:
        return "N/A"
    s = f"{val:.{decimals}f}"
    return f"{s} {unit}" if unit else s


def _ts_fmt(rel: float | None) -> str:
    """Format a relative timestamp as +s.s"""
    if rel is None:
        return "N/A"
    return f"+{rel:.1f} s"


# ═══════════════════════════════════════════════════════════════════════════════
#  Report builder
# ═══════════════════════════════════════════════════════════════════════════════

def build_report(participant_id: str, condition_data: dict) -> str:
    sep  = "=" * 78
    dash = "─" * 78
    lines = []

    lines.append(f"\n{sep}")
    lines.append(f"  TIME PERFORMANCE REPORT — {participant_id}")
    lines.append(sep)
    lines.append(
        "\n  Timing metrics extracted from current_state events in scenario ingescape CSV.\n"
        "  All timestamps are +s relative to the first record in each scenario file.\n"
        "\n  Two key measurements:\n"
        "    Scenario duration   : BEFORE TAKEOFF 'Takeoff clearance'"
        " → AFTER TAKEOFF 'Checklist'\n"
        "    Failure → nominal   : first ENG FAILURE DURING TAKEOFF task"
        " → ENGINE FIRE 'Next Checklist'\n"
        "\n  Task durations = wall-clock time to the next state-machine event\n"
        "  (tasks from different procedures interleave; all durations are in global time order).\n"
    )

    for cond in CONDITIONS:
        if cond not in condition_data:
            continue

        m   = condition_data[cond]["aggregate"]
        scn = condition_data[cond]["scenarios"]

        lines.append(f"\n{dash}")
        lines.append(f"  CONDITION: {cond}  ({len(scn)} scenario file(s))")
        lines.append(f"{dash}\n")

        # ── Per-scenario detail ────────────────────────────────────────────────
        for i, (path, sm) in enumerate(scn, 1):
            bname = os.path.basename(path)
            lines.append(f"  Scenario {i}: {bname}")
            lines.append(f"    Takeoff clearance           : {_ts_fmt(sm.get('takeoff_clearance_rel'))}")
            lines.append(f"    Engine failure onset        : {_ts_fmt(sm.get('failure_onset_rel'))}")
            lines.append(f"    ENGINE FIRE checklist start : {_ts_fmt(sm.get('engine_fire_start_rel'))}")
            lines.append(f"    Nominal recovery            : {_ts_fmt(sm.get('nominal_recovery_rel'))}")
            lines.append(f"    AFTER TAKEOFF ORDER START   : {_ts_fmt(sm.get('after_takeoff_start_rel'))}")
            lines.append(f"    Scenario duration           : {_fmt(sm.get('scenario_duration_s'), 's', 1)}")
            lines.append(f"    Failure → nominal           : {_fmt(sm.get('failure_to_nominal_s'), 's', 1)}")
            for note in sm.get("notes", []):
                lines.append(f"    ⚠  {note}")
            lines.append("")

            # Task tables per procedure (in defined order)
            procs = sm.get("procedures", {})
            col_o = 33   # task_object column width
            col_v = 22   # value column width
            for proc_name in _PROC_KEY:
                if proc_name not in procs:
                    continue
                pd    = procs[proc_name]
                n     = pd["n_tasks"]
                total = pd["total_s"]
                mean  = pd["mean_task_s"]
                lines.append(
                    f"  ── {proc_name}  "
                    f"({n} tasks · total {_fmt(total,'s',1)} · mean {_fmt(mean,'s',1)}/task)"
                )
                lines.append(
                    f"  {'Task / Object':<{col_o}} {'Value':<{col_v}} {'Elapsed':>9}  {'Dur-next':>9}"
                )
                lines.append(
                    f"  {'─'*col_o} {'─'*col_v} {'─'*9}  {'─'*9}"
                )
                for t in pd["tasks"]:
                    obj = t["task_object"][:col_o - 1]
                    val = t["value"][:col_v - 1]
                    ela = _ts_fmt(t["elapsed_s"])
                    dur = (
                        _fmt(t["duration_to_next_s"], "s", 1)
                        if t["duration_to_next_s"] is not None else "—"
                    )
                    lines.append(
                        f"  {obj:<{col_o}} {val:<{col_v}} {ela:>9}  {dur:>9}"
                    )
                lines.append("")

        if len(scn) > 1:
            lines.append(f"  (aggregate metrics below are means across {len(scn)} scenarios)\n")

        # ── Aggregate metric table with bars ───────────────────────────────────
        col_w_label = 42
        col_w_val   = 10
        lines.append(
            f"  {'Metric':<{col_w_label}} {'Value':>{col_w_val}}   Bar (lower = better)"
        )
        lines.append(
            f"  {'─' * col_w_label}   {'─' * col_w_val}   {'─' * 22}"
        )

        def _row(label, key, unit="", decimals=1, has_bar=True):
            if key is None:
                return f"\n  {label}"
            val = m.get(key)
            bar = _bar(val, _BAR_MAX[key]) if has_bar and key in _BAR_MAX else ""
            return (
                f"  {label:<{col_w_label}} {_fmt(val, unit, decimals):>{col_w_val}}   {bar}"
            )

        lines += [
            _row("Scenario duration",                 "scenario_duration_s",         "s"),
            _row("Failure → nominal",                 "failure_to_nominal_s",        "s"),
            _row("── ENG FAILURE DURING TAKEOFF",     None),
            _row("  N tasks",                         "eng_failure_n_tasks",         "",  0, False),
            _row("  Total duration",                  "eng_failure_total_s",         "s"),
            _row("  Mean duration / task",            "eng_failure_mean_task_s",     "s"),
            _row("── ENGINE FIRE",                    None),
            _row("  N tasks",                         "engine_fire_n_tasks",         "",  0, False),
            _row("  Total duration",                  "engine_fire_total_s",         "s"),
            _row("  Mean duration / task",            "engine_fire_mean_task_s",     "s"),
            _row("── BEFORE TAKEOFF",                 None),
            _row("  N tasks",                         "before_takeoff_n_tasks",      "",  0, False),
            _row("  Total duration",                  "before_takeoff_total_s",      "s"),
            _row("  Mean duration / task",            "before_takeoff_mean_task_s",  "s"),
            _row("── TAKEOFF",                        None),
            _row("  N tasks",                         "takeoff_n_tasks",             "",  0, False),
            _row("  Total duration",                  "takeoff_total_s",             "s"),
            _row("  Mean duration / task",            "takeoff_mean_task_s",         "s"),
            _row("── AFTER TAKEOFF",                  None),
            _row("  N tasks",                         "after_takeoff_n_tasks",       "",  0, False),
            _row("  Total duration",                  "after_takeoff_total_s",       "s"),
            _row("  Mean duration / task",            "after_takeoff_mean_task_s",   "s"),
        ]

        unique_notes = sorted(set(m.get("notes", [])))
        if unique_notes:
            lines.append("")
            for note in unique_notes:
                lines.append(f"  ⚠  {note}")

    # ── Cross-condition summary table ──────────────────────────────────────────
    lines.append(f"\n\n{dash}")
    lines.append("  CROSS-CONDITION SUMMARY")
    lines.append(f"{dash}\n")

    present_conds = [c for c in CONDITIONS if c in condition_data]
    col_w = 12

    hdr = f"  {'Metric':<46} " + "  ".join(f"{c:>{col_w}}" for c in present_conds)
    lines.append(hdr)
    lines.append(f"  {'─' * 46} " + "  ".join("─" * col_w for _ in present_conds))

    def _summary_row(label, key, unit="", decimals=1):
        vals = [
            _fmt(condition_data[c]["aggregate"].get(key), unit, decimals)
            for c in present_conds
        ]
        return f"  {label:<46} " + "  ".join(f"{v:>{col_w}}" for v in vals)

    lines += [
        _summary_row("Scenario duration (s)",              "scenario_duration_s",         "s"),
        _summary_row("Failure → nominal (s)",              "failure_to_nominal_s",        "s"),
        _summary_row("ENG FAILURE n tasks",                "eng_failure_n_tasks",         "",  0),
        _summary_row("ENG FAILURE total (s)",              "eng_failure_total_s",         "s"),
        _summary_row("ENG FAILURE mean / task (s)",        "eng_failure_mean_task_s",     "s"),
        _summary_row("ENGINE FIRE n tasks",                "engine_fire_n_tasks",         "",  0),
        _summary_row("ENGINE FIRE total (s)",              "engine_fire_total_s",         "s"),
        _summary_row("ENGINE FIRE mean / task (s)",        "engine_fire_mean_task_s",     "s"),
        _summary_row("Before Takeoff total (s)",           "before_takeoff_total_s",      "s"),
        _summary_row("Before Takeoff mean / task (s)",     "before_takeoff_mean_task_s",  "s"),
        _summary_row("Takeoff total (s)",                  "takeoff_total_s",             "s"),
        _summary_row("Takeoff mean / task (s)",            "takeoff_mean_task_s",         "s"),
        _summary_row("After Takeoff total (s)",            "after_takeoff_total_s",       "s"),
        _summary_row("After Takeoff mean / task (s)",      "after_takeoff_mean_task_s",   "s"),
    ]

    lines.append(f"\n{sep}\n")
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
#  JSON summary builder
# ═══════════════════════════════════════════════════════════════════════════════

def build_summary(participant_id: str, condition_data: dict) -> dict:
    """Build the JSON-serialisable summary dict."""
    summary = {"participant": participant_id, "conditions": {}}

    for cond in CONDITIONS:
        if cond not in condition_data:
            continue
        m   = condition_data[cond]["aggregate"]
        scn = condition_data[cond]["scenarios"]

        cond_summary = {
            "scenario_files":       [os.path.basename(p) for p, _ in scn],
            "scenario_count":       len(scn),
            "scenario_duration_s":  m.get("scenario_duration_s"),
            "failure_to_nominal_s": m.get("failure_to_nominal_s"),
        }
        for proc_key in _PROC_KEY.values():
            cond_summary[proc_key] = {
                "n_tasks":     m.get(f"{proc_key}_n_tasks"),
                "total_s":     m.get(f"{proc_key}_total_s"),
                "mean_task_s": m.get(f"{proc_key}_mean_task_s"),
            }

        summary["conditions"][cond] = cond_summary

    return summary


def save_report(participant_id: str, report_text: str, summary_dict: dict):
    cdir = cleaned_dir(participant_id)
    os.makedirs(cdir, exist_ok=True)
    path = os.path.join(cdir, f"{participant_id}_time_perf_report.txt")
    content = (
        json.dumps(summary_dict, indent=2, ensure_ascii=False)
        + "\n--- END SUMMARY ---\n"
        + report_text
    )
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(content)
    print(f"\n  Report saved → {path}")


# ═══════════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    sep = "=" * 78
    print(f"\n{sep}")
    print("  HITLS — Time Performance Analysis")
    print(f"{sep}")

    participants = find_participants()

    # ── Resolve participant from CLI arg or interactive prompt ────────────────
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("participant", nargs="?", default=None)
    args, _ = parser.parse_known_args()

    if args.participant is not None:
        raw = args.participant.strip()
        if raw.isdigit():
            idx = int(raw) - 1
            if not (0 <= idx < len(participants)):
                print("Invalid participant number.")
                sys.exit(1)
            participant_id = participants[idx]
        else:
            participant_id = raw.upper()
            if participant_id not in participants:
                print(f"Participant '{participant_id}' not found.")
                sys.exit(1)
    else:
        print("\nAvailable participants:")
        for i, pid in enumerate(participants, 1):
            print(f"  {i:2d}. {pid}")

        choice = input("\nSelect a participant (number or ID, e.g. 1 or P02): ").strip()
        if choice.isdigit():
            idx = int(choice) - 1
            if not (0 <= idx < len(participants)):
                print("Invalid selection.")
                sys.exit(1)
            participant_id = participants[idx]
        else:
            participant_id = choice.upper()
            if participant_id not in participants:
                print(f"Participant '{participant_id}' not found.")
                sys.exit(1)

    print(f"\n  Analysing {participant_id} …")

    scenario_files = find_scenario_files(participant_id)
    if not scenario_files:
        print("  No valid scenario files found.")
        sys.exit(1)

    condition_data = {}

    for cond in CONDITIONS:
        files = scenario_files.get(cond, [])
        if not files:
            continue
        print(f"  {cond}: {len(files)} scenario file(s)")
        scn_results = []
        for path in files:
            bname = os.path.basename(path)
            print(f"    Parsing {bname} … ", end="", flush=True)
            try:
                metrics = compute_scenario_metrics(path)
                scn_results.append((path, metrics))
                print(
                    f"✓  "
                    f"scenario={_fmt(metrics.get('scenario_duration_s'),'s',1)}  "
                    f"failure→nominal={_fmt(metrics.get('failure_to_nominal_s'),'s',1)}"
                )
            except Exception as exc:
                print(f"✗  {exc}")

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
