"""
hitls_loader.py — Load and normalise HITLS data for model validation.

All paths are resolved relative to the workspace root (DATA_ANALYSIS/).
Functions return pandas DataFrames or plain dicts, never raw file handles.
"""

from __future__ import annotations

import csv
import glob
import json
import os
from pathlib import Path
from typing import Optional

import pandas as pd

# ---------------------------------------------------------------------------
# Workspace root
# ---------------------------------------------------------------------------

_HERE = Path(__file__).resolve()
WORKSPACE_ROOT = _HERE.parents[2]          # …/DATA_ANALYSIS
HITLS_DIR = WORKSPACE_ROOT / "HITLS"

# ---------------------------------------------------------------------------
# Condition mapping
# HITLS uses 4 condition labels; MITLS covers 3 of them.
# Use HITLS_TO_MITLS to filter or align DataFrames.
# ---------------------------------------------------------------------------

HITLS_CONDITIONS = ["TARS", "TARC", "TARP-S", "TARP-F"]

# HITLS label  →  MITLS Run_Name (None = no MITLS equivalent)
HITLS_TO_MITLS: dict[str, Optional[str]] = {
    "TARS":   "C1",
    "TARC":   None,
    "TARP-S": "C3",
    "TARP-F": "C2",
}

# Conditions shared by both systems (TARC excluded)
SHARED_CONDITIONS = [c for c, m in HITLS_TO_MITLS.items() if m is not None]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _parse_report_json(path: Path) -> dict:
    """Parse the first JSON block in a HITLS report .txt file."""
    content = path.read_text(encoding="utf-8")
    marker = "--- MACHINE-READABLE SUMMARY (JSON) ---"
    start = content.index(marker) + len(marker)
    decoder = json.JSONDecoder()
    data, _ = decoder.raw_decode(content[start:].strip())
    return data


def _find_participants(pids: Optional[list[str]] = None) -> list[str]:
    """Return sorted participant IDs found in HITLS_DIR, optionally filtered."""
    found = sorted(
        e for e in os.listdir(HITLS_DIR)
        if (HITLS_DIR / e).is_dir() and e.startswith("P") and e[1:].isdigit()
    )
    if pids is not None:
        return [p for p in found if p in pids]
    return found


# ---------------------------------------------------------------------------
# NASA-TLX
# ---------------------------------------------------------------------------

def load_nasa_tlx_report() -> dict:
    """Load condition-level NASA-TLX aggregate stats from the group report.

    Returns
    -------
    dict with keys:
        'weighted_score'  → {condition: {n, mean, sd, median, min, max}}
        'dimensions'      → {dim_name: {condition: {n, mean, sd, …}}}
        'conditions'      → list of condition labels present
    """
    path = HITLS_DIR / "compare_forms" / "nasa_tlx_report.txt"
    data = _parse_report_json(path)
    return {
        "weighted_score": data["nasa_tlx_weighted_score"],
        "dimensions": data["dimensions"],
        "conditions": data["conditions"],
    }


def load_nasa_tlx_per_participant(
    pids: Optional[list[str]] = None,
) -> pd.DataFrame:
    """Load per-participant NASA-TLX weighted scores per condition.

    Replicates the logic from HITLS/forms/nasa-tlx.py: loads each
    participant's HAT_study.csv, extracts subscale weights and ratings,
    and computes the weighted score for each condition.

    Returns
    -------
    DataFrame with columns: participant, condition, weighted_score,
        mental_demand, physical_demand, temporal_demand,
        performance, effort, frustration   (raw 0–100 weighted values)
    """
    SUBSCALE_KEYS = {
        "mental_demand":   "Mental Demand",
        "physical_demand": "Physical Demand",
        "temporal_demand": "Temporal Demand",
        "performance":     "Performance",
        "effort":          "Effort",
        "frustration":     "Frustration",
    }
    TARGET_CONDITIONS = HITLS_CONDITIONS[1:]  # exclude baseline_no_system

    rows_out = []
    for pid in _find_participants(pids):
        # Locate HAT study CSV
        candidates = (
            glob.glob(str(HITLS_DIR / pid / f"{pid}_HAT_study.csv"))
            + glob.glob(str(HITLS_DIR / pid / "HAT_study.csv"))
        )
        if not candidates:
            continue
        with open(candidates[0], newline="", encoding="utf-8") as fh:
            all_rows = list(csv.DictReader(fh))

        # Extract subscale weights (from after_familiarization pairwise ranking)
        weight_votes: dict[str, int] = {k: 0 for k in SUBSCALE_KEYS}
        for r in all_rows:
            if r.get("questionnaire_id") == "nasa_tlx_subscale_ranking" and \
               r.get("condition") == "after_familiarization":
                winner = r.get("value", "").strip()
                for key, label in SUBSCALE_KEYS.items():
                    if label == winner:
                        weight_votes[key] += 1

        # Extract per-condition raw ratings (0–20 scale → convert to 0–100)
        ratings: dict[str, dict[str, float]] = {}
        for r in all_rows:
            if r.get("questionnaire_id") != "nasa_tlx":
                continue
            cond = r.get("condition", "").strip()
            q_id = r.get("question_id", "").strip()
            val = r.get("value", "").strip()
            if cond not in TARGET_CONDITIONS or not val:
                continue
            if cond not in ratings:
                ratings[cond] = {}
            for key, label in SUBSCALE_KEYS.items():
                if q_id == key or q_id == label.lower().replace(" ", "_"):
                    try:
                        ratings[cond][key] = float(val)
                    except ValueError:
                        pass

        # Compute weighted score per condition
        for cond in TARGET_CONDITIONS:
            cond_ratings = ratings.get(cond, {})
            if not cond_ratings:
                continue
            total = 0.0
            subscale_vals = {}
            for key in SUBSCALE_KEYS:
                w = weight_votes.get(key, 0)
                r_raw = cond_ratings.get(key)
                if r_raw is None:
                    continue
                r100 = r_raw * 5.0   # 0–20 → 0–100
                weighted = w * r100
                total += weighted
                subscale_vals[key] = weighted
            score = total / 15.0 if total > 0 else None
            rows_out.append({
                "participant": pid,
                "condition": cond,
                "weighted_score": score,
                **subscale_vals,
            })

    return pd.DataFrame(rows_out)


# ---------------------------------------------------------------------------
# HRV
# ---------------------------------------------------------------------------

def load_hrv_features() -> pd.DataFrame:
    """Load per-scenario HRV features for all participants.

    Returns the full CSV as a DataFrame. Key columns:
        participant, scenario, condition, duration_s,
        HRV_RMSSD, HRV_SDNN, HRV_MeanNN, …  (100+ HRV columns)
    """
    path = HITLS_DIR / "HRV" / "hrv_features_per_scenario.csv"
    return pd.read_csv(path)


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------

def load_timing_report() -> dict:
    """Load the HITLS task-timing group report JSON.

    Returns
    -------
    dict with keys:
        'conditions'    → list of condition labels
        'top_level'     → {metric: {condition: {n, mean, sd, …}}}
        'procedures'    → {proc_key: {metric: {condition: {n, mean, sd, …}}}}
    """
    path = HITLS_DIR / "compare_performance" / "time_report.txt"
    data = _parse_report_json(path)
    return {
        "conditions": data["conditions"],
        "top_level": data["top_level"],
        "procedures": data["procedures"],
    }


def load_procedure_timing(
    procedure_key: str,
    metric: str = "total_s",
) -> pd.DataFrame:
    """Return a tidy DataFrame of {condition, n, mean, sd, median, min, max}
    for a single procedure and metric from the timing report.

    Parameters
    ----------
    procedure_key:
        One of the keys in the 'procedures' dict, e.g. 'before_takeoff'.
    metric:
        Sub-metric to extract, e.g. 'total_s', 'mean_task_s', 'n_tasks'.
    """
    report = load_timing_report()
    proc_data = report["procedures"].get(procedure_key, {})
    metric_data = proc_data.get(metric, {})
    rows = []
    for cond, stats in metric_data.items():
        rows.append({"condition": cond, **stats})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Task allocation (TARC)
# ---------------------------------------------------------------------------

def load_tarc(pid: str) -> pd.DataFrame:
    """Load a single participant's TARC allocation table.

    Returns DataFrame with columns:
        Procedure, Classification, Type, Category, Task Object, Value,
        Human Role, Autonomy Role, …
    Adds a 'participant' column.
    """
    candidates = glob.glob(str(HITLS_DIR / pid / f"{pid}_TARC.csv"))
    if not candidates:
        raise FileNotFoundError(f"No TARC CSV found for {pid}")
    df = pd.read_csv(candidates[0])
    df.insert(0, "participant", pid)
    return df


def load_all_tarc(pids: Optional[list[str]] = None) -> pd.DataFrame:
    """Load and concatenate TARC tables for all (or specified) participants.

    Returns combined DataFrame with a 'participant' column prepended.
    """
    frames = []
    for pid in _find_participants(pids):
        try:
            frames.append(load_tarc(pid))
        except FileNotFoundError:
            pass
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Ingescape scenario CSV
# ---------------------------------------------------------------------------

def load_ingescape_scenario(path: str | Path) -> pd.DataFrame:
    """Parse a TARS Agent current_state event stream from an Ingescape CSV.

    Only rows where agent == 'TARS Agent' and event_type == 'current_state'
    are returned.  The JSON payload is expanded into columns.

    Returns
    -------
    DataFrame with columns:
        timestamp, procedure, classification, type, category,
        task_object, value, human_role, autonomy_role,
        delay_before_action, delay_after_action, callout
    """
    rows_out = []
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh, delimiter=";")
        for row in reader:
            if len(row) < 7:
                continue
            if row[2] != "TARS Agent" or row[3] != "current_state":
                continue
            try:
                payload = json.loads(row[6])
            except (json.JSONDecodeError, IndexError):
                continue
            rows_out.append(
                {
                    "timestamp": float(row[1]),
                    "procedure": payload.get("procedure", ""),
                    "classification": payload.get("classification", ""),
                    "type": payload.get("type", ""),
                    "category": payload.get("category", ""),
                    "task_object": payload.get("task_object", ""),
                    "value": payload.get("value", ""),
                    "human_role": payload.get("human_role", ""),
                    "autonomy_role": payload.get("autonomy_role", ""),
                    "delay_before_action": payload.get("delay_before_action"),
                    "delay_after_action": payload.get("delay_after_action"),
                    "callout": payload.get("callout", ""),
                }
            )
    return pd.DataFrame(rows_out)


# ---------------------------------------------------------------------------
# CLI smoke-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== NASA-TLX report (weighted score) ===")
    tlx = load_nasa_tlx_report()
    for cond, stats in tlx["weighted_score"].items():
        print(f"  {cond}: mean={stats['mean']:.2f}  sd={stats['sd']:.2f}  "
              f"min={stats['min']:.2f}  max={stats['max']:.2f}")

    print("\n=== HRV features (first 3 rows, RMSSD) ===")
    hrv = load_hrv_features()
    print(hrv[["participant", "condition", "HRV_RMSSD"]].head(3).to_string(index=False))

    print("\n=== Timing report — before_takeoff total_s ===")
    bt = load_procedure_timing("before_takeoff", "total_s")
    print(bt[["condition", "mean", "sd", "min", "max"]].to_string(index=False))

    print("\n=== TARC tables (participants found) ===")
    tarc = load_all_tarc()
    if not tarc.empty:
        print(f"  {tarc['participant'].nunique()} participants, "
              f"{len(tarc)} total task rows")
        print(tarc[["participant", "Procedure", "Task Object", "Value", "Human Role"]].head(4).to_string(index=False))
