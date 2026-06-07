"""
mitls_loader.py — Load and normalise MITLS cognitive-model outputs.

All paths are resolved relative to the workspace root (DATA_ANALYSIS/).
The MITLS output lives under:
    MITLS/eye_movement/output/

Condition mapping (MITLS → HITLS):
    C1 (run_1) → TARS
    C2 (run_2) → TARP-F
    C3 (run_3) → TARP-S
"""

from __future__ import annotations

import glob
import re
from pathlib import Path
from typing import Generator, Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_HERE = Path(__file__).resolve()
WORKSPACE_ROOT = _HERE.parents[2]
MITLS_EYE_DIR = WORKSPACE_ROOT / "MITLS" / "eye_movement"
MITLS_OUTPUT_DIR = MITLS_EYE_DIR / "output"

# ---------------------------------------------------------------------------
# Condition / run mapping
# ---------------------------------------------------------------------------

# MITLS Run_Name  →  HITLS condition label
MITLS_TO_HITLS: dict[str, str] = {
    "C1": "TARS",
    "C2": "TARP-F",
    "C3": "TARP-S",
}

# HITLS condition label  →  MITLS Run_Name
HITLS_TO_MITLS: dict[str, str] = {v: k for k, v in MITLS_TO_HITLS.items()}

# MITLS Run_Name  →  run folder prefix number (e.g. C1 → run_1)
CONDITION_RUN_NUMBER: dict[str, int] = {"C1": 1, "C2": 2, "C3": 3}


# ---------------------------------------------------------------------------
# Condition run iteration
# ---------------------------------------------------------------------------

def iter_condition_runs(condition: str) -> Generator[Path, None, None]:
    """Yield all per-repetition run directories for a given condition.

    Parameters
    ----------
    condition : str
        MITLS condition label: 'C1', 'C2', or 'C3'.

    Yields
    ------
    Path
        Directory path for each run repetition, sorted chronologically.
    """
    run_num = CONDITION_RUN_NUMBER[condition]
    pattern = str(MITLS_OUTPUT_DIR / f"run_{run_num}_results_*")
    dirs = sorted(
        Path(d) for d in glob.glob(pattern)
        if Path(d).is_dir() and not Path(d).name.endswith("__pycache__")
    )
    yield from dirs


def count_condition_runs(condition: str) -> int:
    """Return the number of repetitions available for *condition*."""
    return sum(1 for _ in iter_condition_runs(condition))


# ---------------------------------------------------------------------------
# Aggregated summary (across all repetitions)
# ---------------------------------------------------------------------------

def load_comparison_summary() -> pd.DataFrame:
    """Load the aggregated per-condition summary (12 reps each).

    Source: output/comparison_summary_metrics.csv

    Returns
    -------
    DataFrame with one row per condition (C1 / C2 / C3) and columns:
        Run_Name, Run_Number, N_Repetitions,
        Total_FSM_Time_{Mean,CI,Std},
        Active_Time_{Mean,CI,Std},
        Coordination_Time_{Mean,CI,Std},
        Active_Percentage, Coordination_Percentage,
        Workload_Overall_{Mean,CI,Std},
        Workload_Perceptual_{Mean,CI},
        Workload_Cognitive_{Mean,CI},
        Workload_Motor_{Mean,CI},
        JAE_Data_{Mean,CI,Std},
        Team_Efficiency_Pct,
        hitls_condition   ← added: TARS / TARP-F / TARP-S
    """
    path = MITLS_OUTPUT_DIR / "comparison_summary_metrics.csv"
    df = pd.read_csv(path)
    df["hitls_condition"] = df["Run_Name"].map(MITLS_TO_HITLS)
    return df


def load_convergence(run_number: int) -> pd.DataFrame:
    """Load the convergence analysis for a single run number (1, 2, or 3).

    Source: output/convergence_analysis_run{N}.csv

    Returns
    -------
    DataFrame indexed by n_replications with convergence metrics per step.
    """
    path = MITLS_OUTPUT_DIR / f"convergence_analysis_run{run_number}.csv"
    return pd.read_csv(path)


# ---------------------------------------------------------------------------
# Per-run team-analyzer outputs
# ---------------------------------------------------------------------------

def load_task_summary(run_dir: str | Path) -> pd.DataFrame:
    """Load per-task FSM/human/TARS durations from a single run directory.

    Source: <run_dir>/team-analyzer/task_summary.csv

    Returns
    -------
    DataFrame with columns:
        Task_Object, Task_Value,
        FSM_Duration_s, Human_Active_Time_s, TARS_Execution_Time_s,
        Active_Duration_s, Coordination_Time_s
    """
    path = Path(run_dir) / "team-analyzer" / "task_summary.csv"
    return pd.read_csv(path)


def load_scenario_summary(run_dir: str | Path) -> pd.DataFrame:
    """Load per-scenario summary metrics from a single run directory.

    Source: <run_dir>/team-analyzer/scenario_summary.csv

    Returns
    -------
    DataFrame with columns:
        Metric, Value
    """
    path = Path(run_dir) / "team-analyzer" / "scenario_summary.csv"
    return pd.read_csv(path)


def load_condition_task_summaries(condition: str) -> pd.DataFrame:
    """Load and stack task_summary.csv for all repetitions of *condition*.

    Adds columns: run_dir (str), repetition (int, 0-based index).

    Returns
    -------
    Tall DataFrame with all repetitions stacked vertically.
    """
    frames = []
    for i, run_dir in enumerate(iter_condition_runs(condition)):
        df = load_task_summary(run_dir)
        df["run_dir"] = str(run_dir)
        df["repetition"] = i
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Per-run workload output
# ---------------------------------------------------------------------------

def load_workload_timeseries(run_dir: str | Path) -> pd.DataFrame:
    """Load the ACT-R module utilization time-series from a run directory.

    Source: <run_dir>/workload_analyzer/results_mental_workload.txt
    Format: tab-delimited, first column is ClockTime(s), then 10 module columns.

    Returns
    -------
    DataFrame with columns:
        time_s,
        Vision_Module, Audio_Module, Perceptual_SubNetwork,
        Production_Module, Declarative_Module, Imaginary_Module,
        Cognitive_SubNetwork,
        Motor_Module, Speech_Module, Motor_SubNetwork,
        Overall_Utilization
    """
    path = Path(run_dir) / "workload_analyzer" / "results_mental_workload.txt"
    df = pd.read_csv(path, sep="\t", header=0)
    # Rename the first column (contains a long descriptor string) to 'time_s'
    df = df.rename(columns={df.columns[0]: "time_s"})
    return df


def load_condition_workload(condition: str) -> pd.DataFrame:
    """Load and stack workload time-series for all repetitions of *condition*.

    Adds columns: run_dir (str), repetition (int, 0-based).
    """
    frames = []
    for i, run_dir in enumerate(iter_condition_runs(condition)):
        df = load_workload_timeseries(run_dir)
        df["run_dir"] = str(run_dir)
        df["repetition"] = i
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Per-run AoI metrics
# ---------------------------------------------------------------------------

# AoI bounding boxes (pixel coordinates, same as eye-movement-analyzer.py)
_AOI_BOXES: dict[str, object] = {
    "TARS":            (0, 865, 800, 1374),
    "PFD":             (1041, 636, 1532, 963),
    "E_W_CAS":         (1592, 681, 1674, 1038),
    "ND":              (1719, 682, 2126, 1036),
    "Central_Console": (1724, 1233, 2282, 1374),
    "Outside_Window":  [
        (0, 0, 2555, 344),   # upper window
        (0, 0, 924, 852),    # left window
    ],
}


def _point_in_aoi(x: int, y: int, box: object) -> bool:
    """Return True if (x, y) falls inside a bounding box (or list of boxes)."""
    if isinstance(box, list):
        return any(_point_in_aoi(x, y, b) for b in box)
    x0, y0, x1, y1 = box
    return x0 <= x <= x1 and y0 <= y <= y1


def _assign_aoi(x: int, y: int) -> str:
    for name, box in _AOI_BOXES.items():
        if _point_in_aoi(x, y, box):
            return name
    return "Other"


def load_aoi_metrics(run_dir: str | Path) -> pd.DataFrame:
    """Compute AoI metrics from the raw eye-movement trace of a run directory.

    Parses <run_dir>/eye_movement/results_eye_movement.txt and reproduces
    the same metrics as eye-movement-analyzer.py's export_aoi_metrics_to_csv().

    Returns
    -------
    DataFrame with columns:
        AoI_Name, Fixation_Count, Percentage, Avg_Dwell_Time_s, Total_Dwell_Time_s
    """
    path = Path(run_dir) / "eye_movement" / "results_eye_movement.txt"
    timestamps, x_coords, y_coords, dwell_times = _parse_eye_movement_file(path)

    if len(timestamps) == 0:
        return pd.DataFrame(columns=[
            "AoI_Name", "Fixation_Count", "Percentage",
            "Avg_Dwell_Time_s", "Total_Dwell_Time_s",
        ])

    # Assign each fixation to an AoI
    aoi_labels = [_assign_aoi(x, y) for x, y in zip(x_coords, y_coords)]
    total_fixations = len(aoi_labels)

    aoi_names = list(_AOI_BOXES.keys()) + ["Other"]
    rows = []
    for name in aoi_names:
        indices = [i for i, a in enumerate(aoi_labels) if a == name]
        count = len(indices)
        total_dwell = sum(dwell_times[i] for i in indices)
        avg_dwell = total_dwell / count if count > 0 else 0.0
        pct = (count / total_fixations * 100) if total_fixations > 0 else 0.0
        rows.append({
            "AoI_Name": name,
            "Fixation_Count": count,
            "Percentage": round(pct, 2),
            "Avg_Dwell_Time_s": round(avg_dwell, 6),
            "Total_Dwell_Time_s": round(total_dwell, 6),
        })
    return pd.DataFrame(rows)


def _parse_eye_movement_file(
    path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[float]]:
    """Parse raw fixation trace file.

    Format (tab-separated):
        timestamp  visual-location  x  y  text_label  "AOI_LABEL"  dwell_ms

    Returns (timestamps, x_coords, y_coords, dwell_times_s).
    """
    timestamps, xs, ys, dwells = [], [], [], []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            parts = line.strip().split("\t")
            if len(parts) < 7:
                continue
            try:
                timestamps.append(float(parts[0]))
                xs.append(int(parts[2]))
                ys.append(int(parts[3]))
                dwells.append(float(parts[6]) / 1000.0)  # ms → s
            except (ValueError, IndexError):
                continue
    return (
        np.array(timestamps),
        np.array(xs),
        np.array(ys),
        dwells,
    )


def load_condition_aoi_metrics(condition: str) -> pd.DataFrame:
    """Load and stack AoI metrics for all repetitions of *condition*.

    Adds columns: run_dir (str), repetition (int, 0-based).
    """
    frames = []
    for i, run_dir in enumerate(iter_condition_runs(condition)):
        df = load_aoi_metrics(run_dir)
        df["run_dir"] = str(run_dir)
        df["repetition"] = i
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# CLI smoke-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== Comparison summary ===")
    summary = load_comparison_summary()
    print(
        summary[
            ["Run_Name", "hitls_condition", "N_Repetitions",
             "Total_FSM_Time_Mean", "JAE_Data_Mean", "Team_Efficiency_Pct"]
        ].to_string(index=False)
    )

    print("\n=== Available repetitions per condition ===")
    for cond in ["C1", "C2", "C3"]:
        n = count_condition_runs(cond)
        print(f"  {cond} ({MITLS_TO_HITLS[cond]}): {n} runs")

    print("\n=== Task summary for first C1 run ===")
    c1_runs = list(iter_condition_runs("C1"))
    if c1_runs:
        ts = load_task_summary(c1_runs[0])
        print(ts[["Task_Object", "Task_Value", "FSM_Duration_s",
                   "Human_Active_Time_s"]].to_string(index=False))

    print("\n=== Workload time-series (first 3 rows, C1 run 0) ===")
    if c1_runs:
        wl = load_workload_timeseries(c1_runs[0])
        print(wl[["time_s", "Cognitive_SubNetwork",
                   "Overall_Utilization"]].head(3).to_string(index=False))

    print("\n=== AoI metrics for first C1 run ===")
    if c1_runs:
        aoi = load_aoi_metrics(c1_runs[0])
        print(aoi.to_string(index=False))

    print("\n=== Convergence — C1 (run 1), first 3 rows ===")
    conv = load_convergence(1)
    print(conv[["n_replications", "jae_data_mean",
                "total_fsm_time_mean"]].head(3).to_string(index=False))
