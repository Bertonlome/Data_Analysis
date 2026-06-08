#!/usr/bin/env python3
"""
crosscheck_perf.py — Comprehensive crosscheck-behavior analysis for HITLS.

Uses task-to-AoI mappings from:
  HITLS/performance/task_crosscheck_aoi_table.csv

Window rules
------------
- TARS (and TARC fallback):
    accepted crosscheck only during task period
    [task_onset, next_task_onset]
- TARP-S / TARP-F:
    accepted during task period OR up to onset + 10 s
    [task_onset, max(next_task_onset, task_onset + 10)]
    Overlap between multiple active tasks is intentionally allowed.

Output
------
Per participant report:
  HITLS/{PID}/cleaned/{PID}_crosscheck_perf_report.txt
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
from dataclasses import dataclass
from typing import Optional

PERF_DIR = os.path.dirname(os.path.abspath(__file__))
HITLS_DIR = os.path.dirname(PERF_DIR)
TASK_TABLE_PATH = os.path.join(PERF_DIR, "task_crosscheck_aoi_table.csv")

# ── Fixation criterion ────────────────────────────────────────────────────────
# Minimum continuous dwell time (seconds) that a single fixation (same
# SmartEyeProBridge fixation-ID) must accumulate on the target AoI before it
# is counted as a successful crosscheck.  Mere scanning (short glances) is
# excluded.  Adjust here to try different thresholds.
MIN_FIXATION_S: float = 0.5

_SKIP_SUBSTRINGS = [
    "training", "TRAINING",
    "unfinished", "UNFINISHED", "UNIFNISHED",
    "no_birds_strike", "birds",
]
_COND_RE = re.compile(r"_(TARS|TARC|TARP-S|TARP-F)_", re.IGNORECASE)

_EYE_AGENT = "SmartEyeProBridge"
_EYE_NAME_SRC  = "filtered_closest_world_object_name"
_EYE_COUNT_SRC = "filtered_closest_world_count"
_EYE_FIX_SRC   = "an_fixation"
_STATE_SRC = "current_state"

_VALID_AOI = {"TARS", "PFD", "ND", "pedestal", "Outside_Window", "no_intersection"}

# Alias hints improve robust matching between table tasks and checklist labels.
_TASK_ALIAS_SUBSTRINGS = {
    "takeoff clearance confirm": ["takeoff clearance"],
    "flaps takeoff": ["flaps", "set for takeoff"],
    "pitot static on": ["pitot", "static"],
    "engine anti ice": ["engine anti", "anti ice"],
    "windshield anti ice": ["windshield", "anti ice"],
    "pax safety": ["pax safety"],
    "landing light": ["landing light"],
    "anti coll light": ["anti coll"],
    "eicas check": ["eicas"],
    "winds checked": ["winds"],
    "select altitude": ["select altitude"],
    "cas check": ["cas"],
    "throttle toga": ["throttle", "toga"],
    "fadec bug to": ["fadec", "bug"],
    "engine spool": ["engine spool"],
    "airspeed s alive": ["airspeed", "alive"],
    "70kts": ["70kts", "70 kts"],
    "v1": ["v1"],
    "rotate": ["rotate"],
    "climb rate": ["climb rate"],
    "landing gear up": ["l g up", "landing gear up"],
    "climb to a safe altitude": ["safe altitude"],
    "flight director": ["flight director"],
    "pitch 10 deg": ["pitch 10"],
    "landing gear check": ["landing gear", "check"],
    "airspeed check v2": ["airspeed", "v2"],
    "rudder trim": ["rudder trim"],
    "autopilot spd mode": ["autopilot", "spd"],
    "autopilot hdg mode": ["autopilot", "hdg"],
    "autopilot engage": ["autopilot", "engage"],
    "alarm announce": ["alarm", "announce"],
    "check altitude": ["check altitude"],
    "throttle idle": ["throttle", "idle"],
    "chrono start": ["chrono", "start"],
    "check on after 15 s": ["after 15"],
    "lift cover and push": ["lift", "cover", "push"],
    "check safe altitude": ["safe altitude"],
    "obstacles check clear": ["obstacles", "clear"],
    "flap handle up": ["flap handle", "up"],
    "accelerate to venr": ["accelerate", "venr", "v enr"],
    "atc contact": ["atc", "contact"],
    "atc readback": ["atc", "readback"],
    "engine fire checklist order start": ["engine fire", "checklist", "order start"],
    "check immediate action": ["immediate action"],
    "fuel boost switch off": ["fuel boost", "switch off"],
    "engine fire light check": ["engine fire", "light check"],
    "illuminated bottle armed switch push": ["bottle", "armed", "switch", "push"],
    "rotary test": ["rotary test"],
    "engine fire lights check both illuminate": ["engine fire", "lights", "illuminate"],
    "next checklist prompt": ["next checklist"],
    "atc announce panpan request vector": ["panpan", "request vector"],
    "heading set": ["heading set"],
    "altitude set": ["altitude set"],
}


@dataclass(frozen=True)
class TaskRule:
    task: str
    crosscheck_aoi_raw: str
    crosscheck_aois: tuple[str, ...]
    norm_task: str


@dataclass
class TaskEvent:
    t_rel: float
    procedure: str
    task_object: str
    value: str


@dataclass
class EyeEvent:
    t_rel: float
    obj_name: str
    count: int
    fix_id: int = 0   # SmartEyeProBridge an_fixation ID; 0 = not in a fixation


def _normalize_text(s: str) -> str:
    s = (s or "").lower().strip()
    s = s.replace("t/o", "takeoff")
    s = s.replace("w/s", "windshield")
    s = s.replace("l/g", "landing gear")
    s = s.replace("v_enr", "venr")
    s = s.replace("throttles", "throttle")
    s = s.replace("checked", "check")
    s = re.sub(r"\b(\d+)\s*kts\b", r"\1kts", s)
    s = s.replace("'", " ")
    s = s.replace('"', " ")
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _parse_state_value(val_str: str) -> Optional[dict]:
    v = val_str
    if v.startswith('"') and v.endswith('"'):
        v = v[1:-1].replace('""', '"')
    try:
        return json.loads(v)
    except (json.JSONDecodeError, ValueError):
        return None


def _condition(path: str) -> Optional[str]:
    m = _COND_RE.search(os.path.basename(path))
    return m.group(1).upper() if m else None


def _is_valid_scenario(path: str) -> bool:
    name = os.path.basename(path)
    return not any(s in name for s in _SKIP_SUBSTRINGS)


def find_participants() -> list[str]:
    return [
        e for e in sorted(os.listdir(HITLS_DIR))
        if os.path.isdir(os.path.join(HITLS_DIR, e)) and re.match(r"^P\d+$", e)
    ]


def find_scenario_files(pid: str) -> list[tuple[str, str]]:
    scen_dir = os.path.join(HITLS_DIR, pid, "scenarios")
    if not os.path.isdir(scen_dir):
        return []

    out = []
    for path in sorted(glob.glob(os.path.join(scen_dir, "*_ingescape.csv"))):
        if not _is_valid_scenario(path):
            continue
        cond = _condition(path)
        if cond is not None:
            out.append((path, cond))
    return out


def load_task_rules(csv_path: str) -> list[TaskRule]:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Task table not found: {csv_path}")

    dedup: dict[tuple[str, tuple[str, ...]], TaskRule] = {}
    with open(csv_path, encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            task = (row.get("task") or "").strip()
            if not task:
                continue

            raw = (row.get("crosscheck_aoi") or "").strip()
            aois = tuple(a.strip() for a in raw.split("|") if a.strip() in _VALID_AOI)

            norm_task = _normalize_text(task)
            rule = TaskRule(task=task, crosscheck_aoi_raw=(row.get("crosscheck_aoi_raw") or "").strip(), crosscheck_aois=aois, norm_task=norm_task)
            dedup[(norm_task, aois)] = rule

    return list(dedup.values())


def parse_scenario_stream(filepath: str) -> tuple[list[TaskEvent], list[EyeEvent], float]:
    with open(filepath, encoding="utf-8-sig", errors="replace") as fh:
        lines = fh.readlines()

    if len(lines) < 2:
        return [], [], 0.0

    ts_scale = 1.0
    header = lines[0].rstrip("\r\n").split(";", 2)
    if len(header) > 1 and header[1] == "relative_time_us":
        ts_scale = 1e-6

    t0 = None
    for line in lines[1:]:
        p = line.rstrip("\r\n").split(";", 6)
        if len(p) < 7:
            continue
        try:
            t0 = float(p[1]) * ts_scale
            break
        except ValueError:
            continue
    if t0 is None:
        return [], [], 0.0

    task_events: list[TaskEvent] = []
    eye_events: list[EyeEvent] = []

    last_eye_obj = ""
    last_eye_count = 0
    last_fix_id = 0
    stream_end_rel = 0.0

    for line in lines[1:]:
        p = line.rstrip("\r\n").split(";", 6)
        if len(p) < 7:
            continue
        try:
            t = float(p[1]) * ts_scale
        except ValueError:
            continue

        t_rel = t - t0
        stream_end_rel = max(stream_end_rel, t_rel)
        agent = p[2]
        source = p[3]
        val = p[6]

        if source == _STATE_SRC:
            d = _parse_state_value(val)
            if d is None:
                continue
            task_events.append(
                TaskEvent(
                    t_rel=t_rel,
                    procedure=str(d.get("procedure", "")),
                    task_object=str(d.get("task_object", "")),
                    value=str(d.get("value", "")),
                )
            )
            continue

        if agent != _EYE_AGENT:
            continue

        if source == _EYE_FIX_SRC:
            try:
                last_fix_id = int(val)
            except ValueError:
                last_fix_id = 0
            eye_events.append(EyeEvent(t_rel=t_rel, obj_name=last_eye_obj,
                                       count=last_eye_count, fix_id=last_fix_id))
        elif source == _EYE_COUNT_SRC:
            try:
                last_eye_count = int(val)
            except ValueError:
                last_eye_count = 0
            eye_events.append(EyeEvent(t_rel=t_rel, obj_name=last_eye_obj,
                                       count=last_eye_count, fix_id=last_fix_id))
        elif source == _EYE_NAME_SRC:
            last_eye_obj = val.strip().strip('"')
            eye_events.append(EyeEvent(t_rel=t_rel, obj_name=last_eye_obj,
                                       count=last_eye_count, fix_id=last_fix_id))

    return task_events, eye_events, stream_end_rel


def _match_score(rule: TaskRule, task_text: str) -> float:
    obj = _normalize_text(task_text)
    if not obj:
        return 0.0

    rule_tokens = [t for t in rule.norm_task.split() if len(t) >= 2]
    if not rule_tokens:
        return 0.0
    obj_tokens = set(obj.split())
    overlap = sum(1 for t in rule_tokens if t in obj_tokens)
    score = overlap / max(1, len(rule_tokens))

    # Exact normalized phrase match is a strong signal.
    if rule.norm_task and rule.norm_task in obj:
        score = max(score, 0.95)

    return min(score, 1.0)


def _find_best_rule(task_text: str, rules: list[TaskRule]) -> Optional[TaskRule]:
    best = None
    best_score = 0.0
    for rule in rules:
        s = _match_score(rule, task_text)
        if s > best_score:
            best = rule
            best_score = s
    return best if best is not None and best_score >= 0.60 else None


def _find_rule_by_norm_task(norm_task: str, rules: list[TaskRule]) -> Optional[TaskRule]:
    for r in rules:
        if r.norm_task == norm_task:
            return r
    return None


def _manual_task_override(
    procedure: str,
    task_object: str,
    task_value: str,
    rules: list[TaskRule],
) -> Optional[TaskRule]:
    """Hard overrides for known checklist label variants in HITLS logs."""
    obj = _normalize_text(task_object)
    val = _normalize_text(task_value)
    proc = _normalize_text(procedure)

    if obj == "throttle" and "to detent" in val:
        return _find_rule_by_norm_task("throttle toga", rules)

    if obj == "test" and "fire warn" in val and proc == "engine fire":
        return _find_rule_by_norm_task("rotary test", rules)

    return None


def _accepted_window_end(condition: str, t_start: float, t_natural_end: float) -> float:
    if condition in {"TARP-S", "TARP-F"}:
        return max(t_natural_end, t_start + 10.0)
    return t_natural_end


def _assess_crosscheck(
    samples: list[EyeEvent],
    targets: tuple[str, ...],
    min_fix_s: float = MIN_FIXATION_S,
) -> tuple[bool, list[str], float]:
    """Return (hit, seen_aois, max_fix_dwell_s).

    A crosscheck is positive only when a single SmartEyeProBridge fixation ID
    accumulates at least *min_fix_s* seconds of dwell time while the gaze is
    on a target AoI.  Saccadic scanning (fix_id == 0) is ignored.
    """
    seen: set[str] = set()
    # fix_id -> total seconds spent on *any* target AoI within that fixation
    fix_dwell: dict[int, float] = {}

    for idx, s in enumerate(samples):
        # Resolve AoI for this sample
        if s.count == 0:
            aoi = "no_intersection"
        elif s.obj_name in _VALID_AOI:
            aoi = s.obj_name
        else:
            aoi = ""

        if aoi:
            seen.add(aoi)

        # Only credit fixations (fix_id != 0) on a target AoI
        if s.fix_id != 0 and aoi in targets:
            # Duration of this sample's validity = interval to the next event
            dt = (samples[idx + 1].t_rel - s.t_rel) if idx + 1 < len(samples) else 0.0
            dt = max(0.0, dt)
            fix_dwell[s.fix_id] = fix_dwell.get(s.fix_id, 0.0) + dt

    max_dwell = max(fix_dwell.values()) if fix_dwell else 0.0
    hit = max_dwell >= min_fix_s
    return hit, sorted(seen), round(max_dwell, 4)


def analyse_scenario(filepath: str, condition: str, rules: list[TaskRule]) -> dict:
    task_events, eye_events, stream_end = parse_scenario_stream(filepath)
    has_eye_data = len(eye_events) > 0
    notes = []

    if not task_events:
        return {
            "file": os.path.basename(filepath),
            "condition": condition,
            "n_eye_events": len(eye_events),
            "eye_data_available": has_eye_data,
            "n_task_events": 0,
            "n_matched": 0,
            "n_crosschecked": 0,
            "unmatched_task_objects": [],
            "task_results": [],
            "notes": ["No current_state events found"],
        }

    if not has_eye_data:
        notes.append("No eye-tracking data found (SmartEyeProBridge); crosscheck cannot be assessed in this scenario.")

    results = []
    unmatched = []

    for i, ev in enumerate(task_events):
        t_start = ev.t_rel
        t_natural_end = task_events[i + 1].t_rel if i < len(task_events) - 1 else stream_end
        t_accept_end = _accepted_window_end(condition, t_start, t_natural_end)

        # Explicit overrides for known label variants, then generic matcher.
        rule = _manual_task_override(ev.procedure, ev.task_object, ev.value, rules)
        if rule is None:
            # Use both task_object and task value to improve mapping fidelity.
            rule = _find_best_rule(f"{ev.task_object} {ev.value}", rules)
        if rule is None:
            unmatched.append(ev.task_object)
            continue

        in_window = [e for e in eye_events if t_start <= e.t_rel <= t_accept_end]
        if not has_eye_data:
            hit, seen, max_dwell = None, [], 0.0
        elif rule.crosscheck_aois:
            hit, seen, max_dwell = _assess_crosscheck(in_window, rule.crosscheck_aois)
        else:
            hit, seen, max_dwell = None, [], 0.0

        results.append({
            "procedure": ev.procedure,
            "task_object": ev.task_object,
            "task_value": ev.value,
            "task_start_s": round(t_start, 3),
            "task_end_natural_s": round(t_natural_end, 3),
            "task_end_accepted_s": round(t_accept_end, 3),
            "task_from_table": rule.task,
            "crosscheck_aoi": list(rule.crosscheck_aois),
            "crosscheck_aoi_raw": rule.crosscheck_aoi_raw,
            "crosschecked": hit,
            "max_fix_dwell_s": max_dwell,
            "seen_aoi": seen,
            "n_eye_samples_in_window": len(in_window),
        })

    n_assessable = sum(1 for r in results if r["crosschecked"] is not None)
    n_crosschecked = sum(1 for r in results if r["crosschecked"] is True)

    return {
        "file": os.path.basename(filepath),
        "condition": condition,
        "n_eye_events": len(eye_events),
        "eye_data_available": has_eye_data,
        "n_task_events": len(task_events),
        "n_matched": len(results),
        "n_assessable": n_assessable,
        "n_crosschecked": n_crosschecked,
        "unmatched_task_objects": sorted(set(unmatched)),
        "task_results": results,
        "notes": notes,
    }


def _pct(n: int, d: int) -> str:
    return "N/A" if d == 0 else f"{(100.0 * n / d):.1f}%"


def report_path(pid: str) -> str:
    return os.path.join(HITLS_DIR, pid, "cleaned", f"{pid}_crosscheck_perf_report.txt")


def build_report_text(pid: str, scenarios: list[dict]) -> str:
    lines = []
    W = 96
    lines.append("=" * W)
    lines.append(f"  COMPREHENSIVE TASK CROSSCHECK REPORT — {pid}")
    lines.append("=" * W)
    lines.append("Rules:")
    lines.append("  TARS/TARC: accepted only during task period")
    lines.append("  TARP-S/F : accepted during task period OR up to onset+10s (overlap allowed)")
    lines.append(f"  Fixation criterion: ≥{MIN_FIXATION_S:.2f} s dwell within a single fixation on target AoI")
    lines.append("=" * W)
    lines.append("")

    total_matched = sum(s["n_matched"] for s in scenarios)
    total_assessable = sum(s["n_assessable"] for s in scenarios)
    total_cross = sum(s["n_crosschecked"] for s in scenarios)
    lines.append(f"Matched occurrences:     {total_matched}")
    lines.append(f"Assessable occurrences:  {total_assessable}")
    lines.append(f"Crosschecked occurrences:{total_cross} ({_pct(total_cross, total_assessable)})")
    lines.append("")

    for s in scenarios:
        lines.append("-" * W)
        lines.append(f"SCENARIO: {s['file']}  |  Condition: {s['condition']}")
        lines.append(
            f"Task events={s['n_task_events']}  Matched={s['n_matched']}  "
            f"Assessable={s['n_assessable']}  "
            f"Crosschecked={s['n_crosschecked']} ({_pct(s['n_crosschecked'], s['n_assessable'])})"
        )
        lines.append(
            f"Eye events={s.get('n_eye_events', 0)}  Eye data available={s.get('eye_data_available', False)}"
        )

        if s.get("notes"):
            lines.append("Notes:")
            for n in s["notes"]:
                lines.append(f"  - {n}")

        if s["unmatched_task_objects"]:
            lines.append("Unmatched task_object labels:")
            for u in s["unmatched_task_objects"]:
                lines.append(f"  - {u}")

        lines.append("Matched task occurrences:")
        for r in s["task_results"]:
            cc = "YES" if r["crosschecked"] is True else ("NO" if r["crosschecked"] is False else "N/A")
            dwell = r.get("max_fix_dwell_s", 0.0)
            lines.append(
                f"  - [{cc}] {r['procedure']} :: {r['task_object']}"
                f" -> table='{r['task_from_table']}'"
                f" aoi={'|'.join(r['crosscheck_aoi'])}"
                f" win=[{r['task_start_s']:.1f},{r['task_end_accepted_s']:.1f}]s"
                f" seen={('|'.join(r['seen_aoi']) if r['seen_aoi'] else 'none')}"
                f" fix_dwell={dwell:.3f}s"
            )
        lines.append("")

    return "\n".join(lines)


def analyse_participant(pid: str, rules: list[TaskRule]) -> list[dict]:
    scen_files = find_scenario_files(pid)
    if not scen_files:
        print(f"  No scenarios found for {pid}")
        return []

    results = []
    for path, cond in scen_files:
        print(f"  {cond:6s}  parsing {os.path.basename(path)}")
        results.append(analyse_scenario(path, cond, rules))

    out_path = report_path(pid)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    summary = {
        "participant": pid,
        "min_fixation_s": MIN_FIXATION_S,
        "scenarios": results,
    }

    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(json.dumps(summary, indent=2, ensure_ascii=False))
        fh.write("\n--- END SUMMARY ---\n\n")
        fh.write(build_report_text(pid, results))
        fh.write("\n")

    print(f"  Report saved -> {out_path}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Comprehensive crosscheck performance analysis")
    parser.add_argument("participant", nargs="?", default=None, help="Participant ID (e.g., P02), index, or A")
    args = parser.parse_args()

    rules = load_task_rules(TASK_TABLE_PATH)
    participants = find_participants()
    if not participants:
        print("No participant folders found.")
        return

    if args.participant:
        choice = args.participant.strip()
    else:
        print("Available participants:")
        for i, p in enumerate(participants, 1):
            print(f"  {i}. {p}")
        print("  A. ALL")
        choice = input("\nSelect participant (number, ID, or A): ").strip()

    if choice.upper() == "A":
        for pid in participants:
            print(f"\n=== {pid} ===")
            analyse_participant(pid, rules)
        return

    if choice.isdigit():
        idx = int(choice) - 1
        if not (0 <= idx < len(participants)):
            print(f"Invalid participant number: {choice}")
            return
        pid = participants[idx]
        analyse_participant(pid, rules)
        return

    pid = choice.upper()
    if pid not in participants:
        print(f"Participant '{pid}' not found.")
        return

    analyse_participant(pid, rules)


if __name__ == "__main__":
    main()
