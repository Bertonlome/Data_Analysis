#!/usr/bin/env python3
"""
error_perf.py — Error Detection & Correction Analysis for a single HITLS participant.

Analyses the three hidden errors present in non-training, non-TARC scenarios
starting from P05.  The error type is encoded in the scenario filename:

  alt   : LINE-UP AND HOLD — task "Select Altitude PRESET AS CLEARED"
          • TARS    : suggests alt_sel = 3000 ft (should be 5000 ft)
          • TARP-S/F: actually enters alt_sel = 3000 ft via Aircraft output
          Corrected   : Aircraft;alt_sel == 50 (= 5000 ft) at "Airspeed's alive"
          Crosschecked: eye on "PFD" with filtered_closest_world_count > 0
                        during the Select Altitude task window

  flaps : BEFORE TAKEOFF — task "Takeoff clearance CONFIRM"
          • TARS    : falsely reports flaps already at takeoff position
          • TARP-S/F: announces setting flaps but does NOT issue the command
          Corrected   : Aircraft;controlFlaps ≈ 0.5 at "Airspeed's alive"
          Crosschecked: eye on "ND" with filtered_closest_world_count > 0
                        during the Takeoff clearance task window

  winds : LINE-UP AND HOLD — task "Winds CHECK"
          Suggests wrong wind indication in all conditions.
          Corrected / Crosschecked : MANUAL — edit the text report fields

──────────────────────────────────────────────────────────────────────────────
Aircraft output units
──────────────────────────────────────────────────────────────────────────────
  Aircraft;alt_sel  → value in units of 100 ft  (30 = 3000 ft, 50 = 5000 ft)
  Aircraft;controlFlaps → 0.000000 = retracted, 0.500000 = takeoff (15 °)

──────────────────────────────────────────────────────────────────────────────
Report format
──────────────────────────────────────────────────────────────────────────────
  JSON summary block  (alt + flaps auto-computed, winds fields null)
  --- END SUMMARY ---
  Human-readable text report (winds manual-entry fields clearly marked)

Output: {PID}/cleaned/{PID}_error_perf_report.txt

Usage
-----
  python performance/error_perf.py           # interactive participant selection
  python performance/error_perf.py P02       # run directly for participant P02
  python performance/error_perf.py A         # run for ALL participants
"""

import os
import re
import sys
import json
import glob
import argparse

# ── Paths ─────────────────────────────────────────────────────────────────────
PERF_DIR  = os.path.dirname(os.path.abspath(__file__))
HITLS_DIR = os.path.dirname(PERF_DIR)

# ── Participants with error scenarios (P05 onward) ─────────────────────────────
_PARTICIPANTS_WITH_ERRORS = [
    "P05", "P06", "P07", "P08", "P09", "P10",
    "P11", "P12", "P13", "P16", "P17",
]

# ── Skip patterns ──────────────────────────────────────────────────────────────
_SKIP_SUBSTRINGS = [
    "training", "TRAINING",
    "unfinished", "UNFINISHED", "UNIFNISHED",
    "no_birds_strike", "birds",
]

# ── Error type & condition detection from filename ────────────────────────────
_ALT_RE   = re.compile(r"_alti?_",  re.IGNORECASE)
_FLAPS_RE = re.compile(r"_flaps_",  re.IGNORECASE)
_WINDS_RE = re.compile(r"_winds_",  re.IGNORECASE)
_COND_RE  = re.compile(r"_(TARS|TARC|TARP-S|TARP-F)_", re.IGNORECASE)

# ── Task / milestone definitions ───────────────────────────────────────────────
# Error task windows — (procedure, task_object substring)
_ALT_TASK_PROC  = "LINE-UP AND HOLD"
_ALT_TASK_OBJ   = "Select Altitude"   # substring match

_FLAPS_TASK_PROC = "BEFORE TAKEOFF"
_FLAPS_TASK_OBJ  = "Takeoff clearance"

_WINDS_TASK_PROC = "LINE-UP AND HOLD"
_WINDS_TASK_OBJ  = "Winds"

# Correction check event
_ALIVE_PROC = "TAKEOFF"
_ALIVE_OBJ  = "Airspeed"   # substring match (catches "\"Airspeed's alive\"")

# Aircraft output sources
_ALT_SEL_AGENT = "Aircraft"
_ALT_SEL_SRC   = "alt_sel"
_FLAPS_AGENT   = "Aircraft"
_FLAPS_SRC     = "controlFlaps"

# Expected alt_sel values (Aircraft units: 100 ft increments)
_ALT_ERROR_VAL     = 30   # 3000 ft — erroneous
_ALT_CORRECTED_VAL = 50   # 5000 ft — correct

# Expected flaps value at takeoff
_FLAPS_TAKEOFF     = 0.5  # 15°

# Eye-tracking sources (SmartEyeProBridge agent)
_EYE_AGENT     = "SmartEyeProBridge"
_EYE_NAME_SRC  = "filtered_closest_world_object_name"
_EYE_COUNT_SRC = "filtered_closest_world_count"

# Instruments that indicate crosscheck per error type
_ALT_CROSSCHECK_OBJ   = "PFD"
_FLAPS_CROSSCHECK_OBJ = "ND"

# Manual-entry placeholder text
_MANUAL = "[FILL: YES / NO]"


# ═══════════════════════════════════════════════════════════════════════════════
#  File helpers
# ═══════════════════════════════════════════════════════════════════════════════

def find_participants():
    return [
        e for e in sorted(os.listdir(HITLS_DIR))
        if os.path.isdir(os.path.join(HITLS_DIR, e))
        and re.match(r"^P\d+$", e)
        and e in _PARTICIPANTS_WITH_ERRORS
    ]


def _error_type(path: str):
    name = os.path.basename(path)
    if _ALT_RE.search(name):
        return "alt"
    if _FLAPS_RE.search(name):
        return "flaps"
    if _WINDS_RE.search(name):
        return "winds"
    return None


def _condition(path: str):
    m = _COND_RE.search(os.path.basename(path))
    return m.group(1).upper() if m else None


def find_error_scenarios(pid: str) -> list:
    """Return list of (filepath, error_type, condition) for valid error scenarios.

    Excludes TARC, training, and unfinished scenarios.
    Only includes files with a recognised error suffix (alt/alti/flaps/winds).
    Sorted by scenario number (first integer in filename).
    """
    scen_dir = os.path.join(HITLS_DIR, pid, "scenarios")
    if not os.path.isdir(scen_dir):
        return []
    results = []
    for path in sorted(glob.glob(os.path.join(scen_dir, "*_ingescape.csv"))):
        name = os.path.basename(path)
        if any(s in name for s in _SKIP_SUBSTRINGS):
            continue
        cond = _condition(path)
        if cond == "TARC":
            continue
        etype = _error_type(path)
        if etype is None:
            continue
        results.append((path, etype, cond))
    return results


def report_path(pid: str) -> str:
    return os.path.join(HITLS_DIR, pid, "cleaned", f"{pid}_error_perf_report.txt")


# ═══════════════════════════════════════════════════════════════════════════════
#  CSV parsing helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _parse_state_value(val_str: str):
    """Unquote a CSV-wrapped JSON string and parse it.  Returns dict or None."""
    v = val_str
    if v.startswith('"') and v.endswith('"'):
        v = v[1:-1].replace('""', '"')
    try:
        return json.loads(v)
    except (json.JSONDecodeError, ValueError):
        return None


# ═══════════════════════════════════════════════════════════════════════════════
#  Single-scenario analysis
# ═══════════════════════════════════════════════════════════════════════════════

def analyse_scenario(filepath: str, error_type: str, condition: str) -> dict:
    """
    Parse one scenario ingescape CSV and return an analysis dict.

    Returns
    -------
    dict with keys (fields not applicable to this error_type are None):
      scenario, condition, error_type
      alt_task_start_rel, alt_task_end_rel           — task window (s from t0)
      alt_val_at_alive                               — Aircraft;alt_sel value
      alt_val_at_alive_ft                            — × 100 for display
      alt_corrected                                  — bool or None
      alt_crosschecked_eye                           — bool or None
      alt_eye_objects                                — set of objects seen
      flaps_task_start_rel, flaps_task_end_rel
      flaps_val_at_alive                             — Aircraft;controlFlaps
      flaps_corrected                                — bool or None
      flaps_crosschecked_eye                         — bool or None
      flaps_eye_objects
      winds_task_start_rel, winds_task_end_rel
      winds_detected                                 — None (manual)
      winds_crosschecked                             — None (manual)
      alive_rel                                      — Airspeed's alive (s)
      notes                                          — list[str]
    """
    result = {
        "scenario":              os.path.basename(filepath),
        "condition":             condition,
        "error_type":            error_type,
        # alt
        "alt_task_start_rel":    None,
        "alt_task_end_rel":      None,
        "alt_val_at_alive":      None,
        "alt_val_at_alive_ft":   None,
        "alt_corrected":         None,
        "alt_crosschecked_eye":  None,
        "alt_eye_objects":       [],
        # flaps
        "flaps_task_start_rel":  None,
        "flaps_task_end_rel":    None,
        "flaps_val_at_alive":    None,
        "flaps_corrected":       None,
        "flaps_crosschecked_eye": None,
        "flaps_eye_objects":     [],
        # winds
        "winds_task_start_rel":  None,
        "winds_task_end_rel":    None,
        "winds_detected":        None,
        "winds_crosschecked":    None,
        # misc
        "alive_rel":             None,
        "notes":                 [],
    }

    # ── Load lines ─────────────────────────────────────────────────────────────
    try:
        with open(filepath, encoding="utf-8-sig", errors="replace") as fh:
            lines = fh.readlines()
    except OSError as exc:
        result["notes"].append(f"ERROR reading file: {exc}")
        return result

    if len(lines) < 2:
        result["notes"].append("ERROR: file too short")
        return result

    # t0 = first valid timestamp in file
    t0 = None
    for line in lines[1:]:
        p = line.rstrip("\r\n").split(";", 6)
        if len(p) >= 7:
            try:
                t0 = float(p[1])
                break
            except ValueError:
                pass
    if t0 is None:
        result["notes"].append("ERROR: could not determine t0")
        return result

    # ── State variables (updated in one pass) ─────────────────────────────────
    last_alt_sel   = None   # most recent Aircraft;alt_sel (int)
    last_flaps     = None   # most recent Aircraft;controlFlaps (float)
    last_eye_count = 0      # most recent SmartEyeProBridge;filtered_closest_world_count

    # Task window bookkeeping
    alt_task_open   = False;  t_alt_start   = None;  t_alt_end   = None
    flaps_task_open = False;  t_flaps_start = None;  t_flaps_end = None
    winds_task_open = False;  t_winds_start = None;  t_winds_end = None

    t_alive = None

    # Eye-tracking samples during task windows
    # Each entry: (t_rel, object_name, count)
    alt_eye   = []
    flaps_eye = []

    # ── One-pass CSV parse ─────────────────────────────────────────────────────
    for line in lines:
        p = line.rstrip("\r\n").split(";", 6)
        if len(p) < 7:
            continue
        try:
            t = float(p[1])
        except ValueError:
            continue
        t_rel  = t - t0
        agent  = p[2]
        source = p[3]
        val    = p[6]

        # ── Aircraft outputs ───────────────────────────────────────────────────
        if agent == _ALT_SEL_AGENT and source == _ALT_SEL_SRC:
            try:
                last_alt_sel = int(val)
            except ValueError:
                pass

        elif agent == _FLAPS_AGENT and source == _FLAPS_SRC:
            try:
                last_flaps = float(val)
            except ValueError:
                pass

        # ── Eye-tracking ───────────────────────────────────────────────────────
        elif agent == _EYE_AGENT:
            if source == _EYE_COUNT_SRC:
                try:
                    last_eye_count = int(val)
                except ValueError:
                    last_eye_count = 0
            elif source == _EYE_NAME_SRC:
                obj_name = val
                cnt = last_eye_count
                if alt_task_open:
                    alt_eye.append((t_rel, obj_name, cnt))
                if flaps_task_open:
                    flaps_eye.append((t_rel, obj_name, cnt))

        # ── State-machine events ───────────────────────────────────────────────
        elif source == "current_state":
            d = _parse_state_value(val)
            if d is None:
                continue
            proc     = d.get("procedure", "")
            task_obj = d.get("task_object", "")

            # ── Open / close task windows ──────────────────────────────────────

            # alt: LINE-UP AND HOLD / Select Altitude
            if proc == _ALT_TASK_PROC and _ALT_TASK_OBJ in task_obj:
                if t_alt_start is None:
                    t_alt_start = t_rel
                    alt_task_open = True
            elif alt_task_open and t_alt_end is None:
                t_alt_end = t_rel
                alt_task_open = False

            # flaps: BEFORE TAKEOFF / Takeoff clearance
            if proc == _FLAPS_TASK_PROC and _FLAPS_TASK_OBJ in task_obj:
                if t_flaps_start is None:
                    t_flaps_start = t_rel
                    flaps_task_open = True
            elif flaps_task_open and t_flaps_end is None:
                t_flaps_end = t_rel
                flaps_task_open = False

            # winds: LINE-UP AND HOLD / Winds
            if proc == _WINDS_TASK_PROC and _WINDS_TASK_OBJ in task_obj:
                if t_winds_start is None:
                    t_winds_start = t_rel
                    winds_task_open = True
            elif winds_task_open and t_winds_end is None:
                t_winds_end = t_rel
                winds_task_open = False

            # Airspeed's alive (correction check point)
            if t_alive is None and proc == _ALIVE_PROC and _ALIVE_OBJ in task_obj:
                t_alive = t_rel
                # Snapshot current aircraft state for correction assessment
                result["alt_val_at_alive"]  = last_alt_sel
                result["flaps_val_at_alive"] = last_flaps

    # ── Close still-open windows (end of file) ────────────────────────────────
    # (edge case: task is the very last event in the file)
    # Windows left open are treated as "no closing event found"

    # ── Store task window timestamps ───────────────────────────────────────────
    result["alt_task_start_rel"]   = t_alt_start
    result["alt_task_end_rel"]     = t_alt_end
    result["flaps_task_start_rel"] = t_flaps_start
    result["flaps_task_end_rel"]   = t_flaps_end
    result["winds_task_start_rel"] = t_winds_start
    result["winds_task_end_rel"]   = t_winds_end
    result["alive_rel"]            = t_alive

    # ── Correction assessment ──────────────────────────────────────────────────
    if t_alive is None and error_type in ("alt", "flaps"):
        result["notes"].append("WARNING: 'Airspeed's alive' not found — correction cannot be assessed")

    if error_type == "alt":
        av = result["alt_val_at_alive"]
        if av is not None:
            result["alt_val_at_alive_ft"] = av * 100
            if av == _ALT_CORRECTED_VAL:
                result["alt_corrected"] = True
            elif av == _ALT_ERROR_VAL:
                result["alt_corrected"] = False
            else:
                result["notes"].append(
                    f"alt_sel at 'Airspeed\\'s alive' = {av} (unexpected value; "
                    f"expected {_ALT_ERROR_VAL} or {_ALT_CORRECTED_VAL})"
                )
        else:
            result["notes"].append("WARNING: no Aircraft;alt_sel reading found before 'Airspeed\\'s alive'")

    elif error_type == "flaps":
        fv = result["flaps_val_at_alive"]
        if fv is not None:
            result["flaps_corrected"] = abs(fv - _FLAPS_TAKEOFF) < 0.05
        else:
            result["notes"].append("WARNING: no Aircraft;controlFlaps reading found before 'Airspeed\\'s alive'")

    # ── Eye-tracking crosscheck assessment ────────────────────────────────────
    if error_type == "alt":
        if t_alt_start is None:
            result["notes"].append("WARNING: 'Select Altitude' task not found — eye crosscheck cannot be assessed")
        else:
            obj_set = set(obj for _, obj, _ in alt_eye)
            result["alt_eye_objects"] = sorted(obj_set)
            crosschecked = any(
                obj == _ALT_CROSSCHECK_OBJ and cnt > 0
                for _, obj, cnt in alt_eye
            )
            result["alt_crosschecked_eye"] = crosschecked

    elif error_type == "flaps":
        if t_flaps_start is None:
            result["notes"].append("WARNING: 'Takeoff clearance' task not found — eye crosscheck cannot be assessed")
        else:
            obj_set = set(obj for _, obj, _ in flaps_eye)
            result["flaps_eye_objects"] = sorted(obj_set)
            crosschecked = any(
                obj == _FLAPS_CROSSCHECK_OBJ and cnt > 0
                for _, obj, cnt in flaps_eye
            )
            result["flaps_crosschecked_eye"] = crosschecked

    elif error_type == "winds":
        if t_winds_start is None:
            result["notes"].append("WARNING: 'Winds CHECK' task not found in LINE-UP AND HOLD")
        # Winds detection / crosscheck remain None → manual entry

    return result


# ═══════════════════════════════════════════════════════════════════════════════
#  Reporting helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _yn(val) -> str:
    """Format a bool/None as YES / NO / N/A."""
    if val is True:
        return "YES"
    if val is False:
        return "NO"
    return "N/A"


def _fmt_t(t) -> str:
    """Format a relative timestamp in seconds, or '—' if None."""
    if t is None:
        return "—"
    return f"+{t:.1f} s"


def _fmt_window(t_start, t_end) -> str:
    """Format a task window as 'start → end  (duration)'."""
    if t_start is None:
        return "not found"
    dur = f"{t_end - t_start:.1f} s" if t_end is not None else "open (end of file)"
    end_str = f"+{t_end:.1f} s" if t_end is not None else "—"
    return f"+{t_start:.1f} s  →  {end_str}  (duration: {dur})"


def build_json_summary(pid: str, scenarios: list) -> dict:
    """Build the JSON block written at the top of the report."""
    out = {"participant": pid, "scenarios": []}
    for s in scenarios:
        entry = {
            "file":               s["scenario"],
            "condition":          s["condition"],
            "error_type":         s["error_type"],
        }
        if s["error_type"] == "alt":
            entry.update({
                "alt_task_window_s":     [s["alt_task_start_rel"], s["alt_task_end_rel"]],
                "alt_val_at_alive_raw":  s["alt_val_at_alive"],
                "alt_val_at_alive_ft":   s["alt_val_at_alive_ft"],
                "alt_corrected":         s["alt_corrected"],
                "alt_crosschecked_eye":  s["alt_crosschecked_eye"],
                "alt_eye_objects":       s["alt_eye_objects"],
            })
        elif s["error_type"] == "flaps":
            entry.update({
                "flaps_task_window_s":      [s["flaps_task_start_rel"], s["flaps_task_end_rel"]],
                "flaps_val_at_alive":       s["flaps_val_at_alive"],
                "flaps_corrected":          s["flaps_corrected"],
                "flaps_crosschecked_eye":   s["flaps_crosschecked_eye"],
                "flaps_eye_objects":        s["flaps_eye_objects"],
            })
        elif s["error_type"] == "winds":
            entry.update({
                "winds_task_window_s":   [s["winds_task_start_rel"], s["winds_task_end_rel"]],
                "winds_detected":        s["winds_detected"],    # null (manual)
                "winds_crosschecked":    s["winds_crosschecked"], # null (manual)
            })
        if s["notes"]:
            entry["notes"] = s["notes"]
        out["scenarios"].append(entry)
    return out


def build_text_report(pid: str, scenarios: list) -> str:
    W = 78
    SEP = "─" * W

    lines = []
    lines.append("=" * W)
    lines.append(f"  ERROR DETECTION & CORRECTION REPORT — {pid}")
    lines.append("=" * W)
    lines.append(
        "  alt   : corrected = alt_sel == 5000 ft at 'Airspeed\\'s alive'"
        "  |  crosscheck via PFD"
    )
    lines.append(
        "  flaps : corrected = controlFlaps == 0.5 at 'Airspeed\\'s alive'"
        "  |  crosscheck via ND"
    )
    lines.append(
        "  winds : MANUAL ENTRY REQUIRED  "
        "(edit the [FILL: YES / NO] fields below)"
    )
    lines.append("=" * W)
    lines.append("")

    for s in scenarios:
        etype = s["error_type"]
        lines.append(SEP)
        lines.append(f"  SCENARIO  : {s['scenario']}")
        lines.append(f"  Condition : {s['condition']}    Error type : {etype.upper()}")
        lines.append(SEP)

        if etype == "alt":
            lines.append(f"  Task window (LINE-UP AND HOLD — 'Select Altitude'):")
            lines.append(f"    {_fmt_window(s['alt_task_start_rel'], s['alt_task_end_rel'])}")
            lines.append(f"  'Airspeed\\'s alive' milestone : {_fmt_t(s['alive_rel'])}")
            lines.append(f"  Aircraft alt_sel at milestone :"
                         f"  raw = {s['alt_val_at_alive']}  "
                         f"(= {s['alt_val_at_alive_ft']} ft)"
                         if s["alt_val_at_alive"] is not None
                         else "  Aircraft alt_sel at milestone :  N/A")
            lines.append(f"  Error corrected               :  {_yn(s['alt_corrected'])}")
            lines.append(f"  Eye crosscheck (PFD, count>0) :  {_yn(s['alt_crosschecked_eye'])}")
            objs = ", ".join(s["alt_eye_objects"]) if s["alt_eye_objects"] else "none"
            lines.append(f"  Objects seen during task      :  {objs}")

        elif etype == "flaps":
            lines.append(f"  Task window (BEFORE TAKEOFF — 'Takeoff clearance'):")
            lines.append(f"    {_fmt_window(s['flaps_task_start_rel'], s['flaps_task_end_rel'])}")
            lines.append(f"  'Airspeed\\'s alive' milestone : {_fmt_t(s['alive_rel'])}")
            fv = s["flaps_val_at_alive"]
            fv_str = f"{fv:.6f}" if fv is not None else "N/A"
            lines.append(f"  Aircraft controlFlaps at milestone :  {fv_str}")
            lines.append(f"  Error corrected                    :  {_yn(s['flaps_corrected'])}")
            lines.append(f"  Eye crosscheck (ND, count>0)       :  {_yn(s['flaps_crosschecked_eye'])}")
            objs = ", ".join(s["flaps_eye_objects"]) if s["flaps_eye_objects"] else "none"
            lines.append(f"  Objects seen during task           :  {objs}")

        elif etype == "winds":
            lines.append(f"  Task window (LINE-UP AND HOLD — 'Winds CHECK'):")
            lines.append(f"    {_fmt_window(s['winds_task_start_rel'], s['winds_task_end_rel'])}")
            lines.append(f"  'Airspeed\\'s alive' milestone : {_fmt_t(s['alive_rel'])}")
            lines.append(f"")
            lines.append(f"  ┌─ MANUAL ENTRY ─────────────────────────────────────┐")
            winds_det  = s["winds_detected"]   if s["winds_detected"]   is not None else _MANUAL
            winds_cc   = s["winds_crosschecked"] if s["winds_crosschecked"] is not None else _MANUAL
            lines.append(f"  │  Winds error detected      : {winds_det:<25}│")
            lines.append(f"  │  Winds error crosschecked  : {winds_cc:<25}│")
            lines.append(f"  └────────────────────────────────────────────────────┘")

        if s["notes"]:
            lines.append(f"")
            lines.append(f"  Notes:")
            for n in s["notes"]:
                lines.append(f"    ! {n}")

        lines.append("")

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
#  Main analysis driver
# ═══════════════════════════════════════════════════════════════════════════════

def analyse_participant(pid: str) -> list:
    """
    Find and analyse all error scenarios for one participant.
    Writes the report file and returns list of scenario result dicts.
    """
    scenarios_list = find_error_scenarios(pid)
    if not scenarios_list:
        print(f"  No error scenarios found for {pid}")
        return []

    results = []
    for filepath, etype, cond in scenarios_list:
        fname = os.path.basename(filepath)
        print(f"  {cond} [{etype:5s}] Parsing {fname} … ", end="", flush=True)
        r = analyse_scenario(filepath, etype, cond)
        results.append(r)

        # Brief status on stdout
        parts = []
        if etype == "alt":
            parts.append(f"corrected={_yn(r['alt_corrected'])}  "
                         f"crosscheck={_yn(r['alt_crosschecked_eye'])}")
        elif etype == "flaps":
            parts.append(f"corrected={_yn(r['flaps_corrected'])}  "
                         f"crosscheck={_yn(r['flaps_crosschecked_eye'])}")
        elif etype == "winds":
            parts.append("manual entry required")
        if r["notes"]:
            parts.append(f"({len(r['notes'])} note(s))")
        print("  ".join(parts) if parts else "ok")

    # ── Write report ─────────────────────────────────────────────────────────
    rpt_path = report_path(pid)
    os.makedirs(os.path.dirname(rpt_path), exist_ok=True)

    summary = build_json_summary(pid, results)
    text    = build_text_report(pid, results)

    with open(rpt_path, "w", encoding="utf-8") as fh:
        fh.write(json.dumps(summary, indent=2, ensure_ascii=False))
        fh.write("\n--- END SUMMARY ---\n\n")
        fh.write(text)
        fh.write("\n")

    print(f"  Report saved → {rpt_path}")
    return results


# ═══════════════════════════════════════════════════════════════════════════════
#  Cross-participant summary table
# ═══════════════════════════════════════════════════════════════════════════════

def print_summary_table(all_results: dict):
    """Print a cross-participant overview table.

    all_results : {pid: [scenario_result_dict, ...]}
    """
    W = 78
    print()
    print("=" * W)
    print("  CROSS-PARTICIPANT SUMMARY")
    print("=" * W)
    HDR = (
        f"  {'Participant':<8}  {'Condition':<8}  {'Error':<6}  "
        f"{'Corrected':<10}  {'Eye Crosscheck':<15}  Notes"
    )
    print(HDR)
    print("  " + "─" * (W - 2))

    for pid in sorted(all_results):
        for r in all_results[pid]:
            etype = r["error_type"]
            if etype == "alt":
                corr = _yn(r["alt_corrected"])
                cc   = _yn(r["alt_crosschecked_eye"])
            elif etype == "flaps":
                corr = _yn(r["flaps_corrected"])
                cc   = _yn(r["flaps_crosschecked_eye"])
            else:  # winds
                corr = "MANUAL"
                cc   = "MANUAL"
            note_flag = "!" if r["notes"] else ""
            print(
                f"  {pid:<8}  {r['condition']:<8}  {etype:<6}  "
                f"{corr:<10}  {cc:<15}  {note_flag}"
            )

    print("=" * W)


# ═══════════════════════════════════════════════════════════════════════════════
#  Entry point
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print()
    print("=" * 78)
    print("  HITLS — Error Detection & Correction Analysis")
    print("=" * 78)

    participants = find_participants()
    if not participants:
        print("  No eligible participants found.")
        sys.exit(1)

    # ── Participant selection ─────────────────────────────────────────────────
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("participant", nargs="?", default=None)
    args, _ = parser.parse_known_args()

    if args.participant is not None:
        raw = args.participant.strip().upper()
        if raw == "A":
            selected = participants
        elif raw.isdigit() and 1 <= int(raw) <= len(participants):
            selected = [participants[int(raw) - 1]]
        elif raw in participants:
            selected = [raw]
        else:
            print(f"  Invalid participant: {raw!r}")
            sys.exit(1)
    else:
        print("\nAvailable participants:")
        for i, pid in enumerate(participants, 1):
            print(f"  {i:>3}. {pid}")
        print(f"  {'A'.rjust(3)}. ALL participants")
        print()
        raw = input("Select a participant (number, ID, or A for all): ").strip()

        if raw.upper() == "A":
            selected = participants
        elif raw.isdigit() and 1 <= int(raw) <= len(participants):
            selected = [participants[int(raw) - 1]]
        elif raw.upper() in participants:
            selected = [raw.upper()]
        else:
            print(f"  Invalid selection: {raw!r}")
            sys.exit(1)

    # ── Run analysis ──────────────────────────────────────────────────────────
    all_results = {}
    for pid in selected:
        print(f"\n  Analysing {pid} …")
        res = analyse_participant(pid)
        if res:
            all_results[pid] = res

    # ── Cross-participant table (when >1 participant) ─────────────────────────
    if len(all_results) > 1:
        print_summary_table(all_results)


if __name__ == "__main__":
    main()
