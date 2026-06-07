"""
coverage.py — MITLS procedure/task coverage map.

Single source of truth for which HITLS procedures and tasks have been
implemented in the current MITLS build.  Every analysis module imports
this to gate statistics and generate the scope note in Tier 1 reports.

To extend coverage when new procedures are modelled in MITLS, add entries
to PROCEDURE_STATUS and TASKS_BY_PROCEDURE only — nothing else needs to change.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import pandas as pd

# ---------------------------------------------------------------------------
# Status type
# ---------------------------------------------------------------------------

ProcedureStatus = Literal["implemented", "not_modelled"]

# ---------------------------------------------------------------------------
# Procedure-level coverage
# ---------------------------------------------------------------------------
# Keys match the `procedure` field in HITLS Ingescape CSVs (TARS Agent
# current_state events) and the `procedures` dict in compare_performance
# time_report.txt JSON (lowercased / underscored variant stored separately).
#
# HITLS canonical name  →  time_report JSON key
# "BEFORE TAKEOFF"      →  "before_takeoff"
# "LINE-UP AND HOLD"    →  "lineup_hold"
# "TAKEOFF"             →  "takeoff"
# ...

PROCEDURE_STATUS: dict[str, ProcedureStatus] = {
    "CREW BRIEFING":                "not_modelled",
    "BEFORE TAKEOFF":               "implemented",
    "LINE-UP AND HOLD":             "implemented",
    "TAKEOFF":                      "not_modelled",
    "ENG FAILURE DURING TAKEOFF":   "not_modelled",
    "ENGINE FIRE":                  "not_modelled",
    "DECLARE PANPAN":               "not_modelled",
    "AFTER TAKEOFF":                "not_modelled",
}

# Mapping from HITLS canonical procedure name to its time_report JSON key
PROCEDURE_JSON_KEY: dict[str, str] = {
    "CREW BRIEFING":                "crew_briefing",
    "BEFORE TAKEOFF":               "before_takeoff",
    "LINE-UP AND HOLD":             "lineup_hold",
    "TAKEOFF":                      "takeoff",
    "ENG FAILURE DURING TAKEOFF":   "eng_failure",
    "ENGINE FIRE":                  "engine_fire",
    "DECLARE PANPAN":               "declare_panpan",
    "AFTER TAKEOFF":                "after_takeoff",
}

# ---------------------------------------------------------------------------
# Task-level coverage
# ---------------------------------------------------------------------------
# Each entry is (Task_Object, Task_Value) exactly as they appear in:
#   - HITLS: TARS Agent current_state JSON field {"task_object": ..., "value": ...}
#   - MITLS: team-analyzer/task_summary.csv columns Task_Object, Task_Value
#
# Tasks present in HITLS but NOT yet in MITLS are listed as not_modelled.

@dataclass
class TaskCoverage:
    task_object: str
    task_value: str
    procedure: str          # HITLS canonical procedure name
    status: ProcedureStatus


TASK_COVERAGE: list[TaskCoverage] = [
    # ── CREW BRIEFING (not modelled) ────────────────────────────────────────
    TaskCoverage("START",          "BRIEFING",         "CREW BRIEFING",      "not_modelled"),
    TaskCoverage("Weather",        "BRIEF",             "CREW BRIEFING",      "not_modelled"),
    TaskCoverage("Aircraft",       "BRIEF",             "CREW BRIEFING",      "not_modelled"),
    TaskCoverage("NOTAMs",         "BRIEF",             "CREW BRIEFING",      "not_modelled"),
    TaskCoverage("Routing",        "BRIEF",             "CREW BRIEFING",      "not_modelled"),
    TaskCoverage("Automation",     "BRIEF",             "CREW BRIEFING",      "not_modelled"),
    TaskCoverage("Miscellaneous",  "BRIEF",             "CREW BRIEFING",      "not_modelled"),

    # ── BEFORE TAKEOFF ──────────────────────────────────────────────────────
    TaskCoverage("Takeoff clearance",          "CONFIRM",           "BEFORE TAKEOFF", "implemented"),
    TaskCoverage("FLAPS",                      "SET FOR TAKEOFF",   "BEFORE TAKEOFF", "not_modelled"),  # in HITLS, missing from MITLS
    TaskCoverage("Pitot-Static Switch",        "PITOT-STATIC",      "BEFORE TAKEOFF", "implemented"),
    TaskCoverage("ENGINE ANTI-ICE Switches",   "AS REQUIRED",       "BEFORE TAKEOFF", "implemented"),
    TaskCoverage("WINDSHIELD ANTI-ICE Switches","AS REQUIRED",      "BEFORE TAKEOFF", "implemented"),
    TaskCoverage("PAX SAFETY Switch",          "PAX SAFETY",        "BEFORE TAKEOFF", "implemented"),
    TaskCoverage("LANDING Light Switch",       "AS DESIRED",        "BEFORE TAKEOFF", "implemented"),
    TaskCoverage("ANTI-COLL Light Switch",     "ON",                "BEFORE TAKEOFF", "implemented"),
    TaskCoverage("EICAS",                      "CHECKED",           "BEFORE TAKEOFF", "implemented"),

    # ── LINE-UP AND HOLD ────────────────────────────────────────────────────
    TaskCoverage("Winds",           "CHECK",              "LINE-UP AND HOLD", "implemented"),
    TaskCoverage("Select Altitude", "PRESET AS CLEARED",  "LINE-UP AND HOLD", "implemented"),

    # ── TAKEOFF (not modelled) ───────────────────────────────────────────────
    TaskCoverage("CAS",             "CHECK CLEAR",   "TAKEOFF", "not_modelled"),
    TaskCoverage("THROTTLES",       "TO Detent",     "TAKEOFF", "not_modelled"),
    TaskCoverage("FADEC bug",       "CHECK TO",      "TAKEOFF", "not_modelled"),
    TaskCoverage("Engine spool",    "CHECK EVEN",    "TAKEOFF", "not_modelled"),
    TaskCoverage('"Airspeed\'s alive"', "ANNOUNCE",  "TAKEOFF", "not_modelled"),
    TaskCoverage('"70 kts"',        "ANNOUNCE",      "TAKEOFF", "not_modelled"),
    TaskCoverage('"V1"',            "ANNOUNCE",      "TAKEOFF", "not_modelled"),
    TaskCoverage('"Rotate"',        "ANNOUNCE",      "TAKEOFF", "not_modelled"),
    TaskCoverage("Pitch",           "MAINTAIN 10°",  "TAKEOFF", "not_modelled"),
    TaskCoverage("Climb rate",      "CHECK POSITIVE","TAKEOFF", "not_modelled"),
    TaskCoverage("LANDING GEAR",    "UP",            "TAKEOFF", "not_modelled"),

    # ── ENG FAILURE DURING TAKEOFF (not modelled) ───────────────────────────
    TaskCoverage("Climb",           "TO A SAFE ALTITUDE", "ENG FAILURE DURING TAKEOFF", "not_modelled"),
    TaskCoverage("Flight Director", "SET TO MODE",         "ENG FAILURE DURING TAKEOFF", "not_modelled"),
    TaskCoverage("Pitch",           "MAINTAIN 10°",        "ENG FAILURE DURING TAKEOFF", "not_modelled"),
    TaskCoverage("LANDING GEAR",    "UP",                  "ENG FAILURE DURING TAKEOFF", "not_modelled"),
    TaskCoverage("Airspeed",        "CHECK V2",            "ENG FAILURE DURING TAKEOFF", "not_modelled"),
    TaskCoverage("Rudder",          "TRIM",                "ENG FAILURE DURING TAKEOFF", "not_modelled"),
    TaskCoverage("Autopilot",       "SET SPD MODE",        "ENG FAILURE DURING TAKEOFF", "not_modelled"),
    TaskCoverage("Autopilot",       "SET HDG MODE",        "ENG FAILURE DURING TAKEOFF", "not_modelled"),
    TaskCoverage("Autopilot",       "ENGAGE",              "ENG FAILURE DURING TAKEOFF", "not_modelled"),
    TaskCoverage("Alarm",           "ANNOUNCE",            "ENG FAILURE DURING TAKEOFF", "not_modelled"),
    TaskCoverage("Check",           "SAFE ALTITUDE REACHED","ENG FAILURE DURING TAKEOFF","not_modelled"),
    TaskCoverage("Altitude",        "CHECK 1500ft AGL",    "ENG FAILURE DURING TAKEOFF", "not_modelled"),
    TaskCoverage("Airspeed",        "CHECK V2+10",         "ENG FAILURE DURING TAKEOFF", "not_modelled"),
    TaskCoverage("Obstacles",       "CHECK Clear",         "ENG FAILURE DURING TAKEOFF", "not_modelled"),
    TaskCoverage("FLAP Handle",     "UP",                  "ENG FAILURE DURING TAKEOFF", "not_modelled"),
    TaskCoverage("ATC",             "CONTACT",             "ENG FAILURE DURING TAKEOFF", "not_modelled"),
    TaskCoverage("ATC",             "READBACK",            "ENG FAILURE DURING TAKEOFF", "not_modelled"),

    # ── ENGINE FIRE (not modelled) ───────────────────────────────────────────
    TaskCoverage("Throttle (affected engine)",    "IDLE",                             "ENGINE FIRE", "not_modelled"),
    TaskCoverage("Chrono",                        "START",                            "ENGINE FIRE", "not_modelled"),
    TaskCoverage("Engine FIRE LIGHT",             "CHECK ON AFTER 15s",               "ENGINE FIRE", "not_modelled"),
    TaskCoverage("Illuminated ENGINE FIRE Switch","LIFT COVER AND PUSH",              "ENGINE FIRE", "not_modelled"),
    TaskCoverage("Checklist",                     "ORDER START",                      "ENGINE FIRE", "not_modelled"),
    TaskCoverage("Immediate Action Item",         "CHECK DONE",                       "ENGINE FIRE", "not_modelled"),
    TaskCoverage("FUEL BOOST Switch (affected side)", "OFF",                          "ENGINE FIRE", "not_modelled"),
    TaskCoverage("Engine FIRE LIGHT",             "CHECK ON AFTER 30s",               "ENGINE FIRE", "not_modelled"),
    TaskCoverage("Illuminated BOTTLE ARMED Switch","PUSH",                            "ENGINE FIRE", "not_modelled"),
    TaskCoverage("Test",                          "FIRE WARN",                        "ENGINE FIRE", "not_modelled"),
    TaskCoverage("Engine fire lights",            "Check both illuminate",            "ENGINE FIRE", "not_modelled"),
    TaskCoverage("Next Checklist",                "ENGINE FAILURE/PRECAUTIONARY SHUTDOWN","ENGINE FIRE","not_modelled"),

    # ── DECLARE PANPAN (not modelled) ────────────────────────────────────────
    TaskCoverage("ATC",      "ANNOUNCE PANPAN AND REQUEST VECTOR", "DECLARE PANPAN", "not_modelled"),
    TaskCoverage("ATC",      "READBACK",                           "DECLARE PANPAN", "not_modelled"),
    TaskCoverage("Heading",  "SET ACCORDINGLY",                    "DECLARE PANPAN", "not_modelled"),
    TaskCoverage("Altitude", "SET ACCORDINGLY",                    "DECLARE PANPAN", "not_modelled"),

    # ── AFTER TAKEOFF (not modelled) ─────────────────────────────────────────
    TaskCoverage("Checklist",       "ORDER START", "AFTER TAKEOFF", "not_modelled"),
    TaskCoverage("LANDING GEAR Handle", "UP",      "AFTER TAKEOFF", "not_modelled"),
]

# ---------------------------------------------------------------------------
# Derived helpers
# ---------------------------------------------------------------------------

COVERED_PROCEDURES: frozenset[str] = frozenset(
    p for p, s in PROCEDURE_STATUS.items() if s == "implemented"
)

COVERED_TASKS: frozenset[tuple[str, str]] = frozenset(
    (t.task_object, t.task_value)
    for t in TASK_COVERAGE
    if t.status == "implemented"
)


def get_covered_tasks(
    df: pd.DataFrame,
    task_object_col: str = "Task_Object",
    task_value_col: str = "Task_Value",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split *df* into (covered, uncovered) based on current MITLS coverage.

    Parameters
    ----------
    df:
        DataFrame with at least *task_object_col* and *task_value_col* columns.
    task_object_col, task_value_col:
        Column names for the task key fields.

    Returns
    -------
    covered_df, uncovered_df : tuple[pd.DataFrame, pd.DataFrame]
    """
    mask = df.apply(
        lambda row: (row[task_object_col], row[task_value_col]) in COVERED_TASKS,
        axis=1,
    )
    return df[mask].copy(), df[~mask].copy()


def coverage_summary() -> dict[str, str]:
    """Return a dict mapping each procedure to its status string."""
    return {proc: status for proc, status in PROCEDURE_STATUS.items()}


def scope_note() -> str:
    """Return a one-line scope note for Tier 1 report headers."""
    covered = sorted(COVERED_PROCEDURES)
    n_covered = len([t for t in TASK_COVERAGE if t.status == "implemented"])
    n_total = len(TASK_COVERAGE)
    procs_str = " + ".join(covered)
    return (
        f"Analysis restricted to MITLS-covered procedures: {procs_str} "
        f"({n_covered}/{n_total} tasks implemented)."
    )


def task_coverage_dataframe() -> pd.DataFrame:
    """Return the full task coverage list as a tidy DataFrame."""
    return pd.DataFrame(
        [
            {
                "procedure": t.procedure,
                "task_object": t.task_object,
                "task_value": t.task_value,
                "status": t.status,
            }
            for t in TASK_COVERAGE
        ]
    )


# ---------------------------------------------------------------------------
# CLI smoke-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print(scope_note())
    print()
    df = task_coverage_dataframe()
    for proc, grp in df.groupby("procedure", sort=False):
        impl = grp[grp.status == "implemented"]
        not_impl = grp[grp.status == "not_modelled"]
        sym = "✅" if PROCEDURE_STATUS.get(proc) == "implemented" else "⬜"
        print(f"{sym}  {proc}  ({len(impl)} implemented, {len(not_impl)} not modelled)")
        for _, row in grp.iterrows():
            mark = "  ✓" if row.status == "implemented" else "  ✗"
            print(f"    {mark}  {row.task_object} / {row.task_value}")
        print()
