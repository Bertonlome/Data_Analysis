#!/usr/bin/env python3
"""
Step 5: Timing Validation
=========================
Compare MITLS per-procedure task-completion times to HITLS empirical procedure
durations.

MITLS source : per-run team-analyzer/task_summary.csv  →  FSM_Duration_s
HITLS source : PXX/cleaned/PXX_time_perf_report.txt  →  before_takeoff /
               lineup_hold / total_s per condition per participant

Coverage
--------
  BEFORE TAKEOFF  : 8 / 9 tasks modelled  (FLAPS / SET FOR TAKEOFF absent)
  LINE-UP AND HOLD: 2 / 2 tasks modelled

Tier 1 outputs (publication)
  plots/pub/procedure_timing_comparison.png
  timing/timing_report.txt

Tier 2 outputs (diagnostic)
  plots/debug/timing_per_task_envelope.png
  plots/debug/timing_nmae_sorted.png
  timing/timing_debug_report.txt
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve()
WORKSPACE_ROOT = _HERE.parents[2]

sys.path.insert(0, str(WORKSPACE_ROOT))

from model_validation.shared.hitls_loader import (
    HITLS_DIR, SHARED_CONDITIONS, load_procedure_timing,
)
from model_validation.shared.mitls_loader import (
    MITLS_TO_HITLS, iter_condition_runs, load_task_summary,
)
from model_validation.shared.stats import (
    friedman_test, wilcoxon_pairwise, nmae, fmt_p, sig_stars,
)

VAL_DIR   = WORKSPACE_ROOT / "model_validation"
PUB_DIR   = VAL_DIR / "plots" / "pub"
DEBUG_DIR = VAL_DIR / "plots" / "debug"
PUB_DIR.mkdir(parents=True, exist_ok=True)
DEBUG_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Procedure / task definitions
# ---------------------------------------------------------------------------

# BEFORE TAKEOFF — 8 of 9 modelled (FLAPS absent)
BEFORE_TAKEOFF_TASKS = [
    "Takeoff clearance",
    "Pitot-Static Switch",
    "ENGINE ANTI-ICE Switches",
    "WINDSHIELD ANTI-ICE Switches",
    "PAX SAFETY Switch",
    "LANDING Light Switch",
    "ANTI-COLL Light Switch",
    "EICAS",
]

# LINE-UP AND HOLD — 2 of 2 modelled
LINEUP_HOLD_TASKS = ["Winds", "Select Altitude"]

# NOT MODELLED slot that appears in the full BEFORE TAKEOFF task list
BEFORE_TAKEOFF_NOT_MODELLED = ["FLAPS / SET FOR TAKEOFF"]

PROC_INFO: dict[str, dict] = {
    "before_takeoff": {
        "label":      "BEFORE TAKEOFF",
        "tasks":      BEFORE_TAKEOFF_TASKS,
        "not_modelled": BEFORE_TAKEOFF_NOT_MODELLED,
        "n_modelled": 8,
        "n_total":    9,
        "hitls_key":  "before_takeoff",
    },
    "lineup_hold": {
        "label":      "LINE-UP AND HOLD",
        "tasks":      LINEUP_HOLD_TASKS,
        "not_modelled": [],
        "n_modelled": 2,
        "n_total":    2,
        "hitls_key":  "lineup_hold",
    },
}

_COND_ORDER = SHARED_CONDITIONS           # TARS, TARP-S, TARP-F
_COLORS     = {                            # consistent with stats.py palette
    "TARS":   "#4878CF",
    "TARP-S": "#6ACC65",
    "TARP-F": "#D65F5F",
}
_HUMAN_ALPHA = 0.30
_MODEL_COLOR = "#D65F5F"                  # red for model
_NOT_MODELLED_COLOR = "#BBBBBB"

# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def _find_participants() -> list[str]:
    return sorted(
        e for e in HITLS_DIR.iterdir()
        if e.is_dir() and e.name.startswith("P") and e.name[1:].isdigit()
    )


def _load_per_participant_timing() -> dict[str, dict]:
    """Load per-participant timing from cleaned reports.

    Returns
    -------
    dict: pid → condition → proc_key → Optional[float]
        Values are procedure total_s, None if missing.
    """
    result: dict[str, dict] = {}
    for pid_dir in _find_participants():
        pid = pid_dir.name
        report_path = pid_dir / "cleaned" / f"{pid}_time_perf_report.txt"
        if not report_path.exists():
            continue
        with open(report_path, encoding="utf-8") as fh:
            try:
                decoder = json.JSONDecoder()
                report, _ = decoder.raw_decode(fh.read().strip())
            except (json.JSONDecodeError, ValueError):
                continue
        result[pid] = {}
        for cond, cond_data in report.get("conditions", {}).items():
            if cond not in SHARED_CONDITIONS:
                continue
            result[pid][cond] = {}
            for proc_key in PROC_INFO:
                proc_data = cond_data.get(proc_key, {})
                result[pid][cond][proc_key] = proc_data.get("total_s")  # may be None
    return result


def _hitls_procedure_arrays(
    per_participant: dict[str, dict],
) -> dict[str, dict[str, list[float]]]:
    """Extract paired arrays per condition per procedure (NaN-dropped within proc).

    Returns
    -------
    dict: proc_key → condition → list[float]   (only non-None values)
    """
    out: dict[str, dict[str, list[float]]] = {pk: {} for pk in PROC_INFO}
    for proc_key in PROC_INFO:
        for cond in SHARED_CONDITIONS:
            vals = [
                v
                for p_data in per_participant.values()
                for v in [p_data.get(cond, {}).get(proc_key)]
                if v is not None
            ]
            out[proc_key][cond] = vals
    return out


def _mitls_procedure_totals(n: int = 12) -> dict[str, dict[str, list[float]]]:
    """Compute per-rep procedure totals from task_summary FSM_Duration_s.

    Returns
    -------
    dict: condition (HITLS label) → proc_key → list[float]  (one per rep)
    """
    out: dict[str, dict[str, list[float]]] = {
        cond: {pk: [] for pk in PROC_INFO}
        for cond in SHARED_CONDITIONS
    }

    bt_set  = set(BEFORE_TAKEOFF_TASKS)
    lh_set  = set(LINEUP_HOLD_TASKS)

    for mitls_cond in ["C1", "C2", "C3"]:
        hitls_cond = MITLS_TO_HITLS[mitls_cond]
        runs = list(iter_condition_runs(mitls_cond))[:n]
        for run_dir in runs:
            try:
                ts = load_task_summary(run_dir)
            except FileNotFoundError:
                continue
            bt_total = ts.loc[
                ts["Task_Object"].isin(bt_set), "FSM_Duration_s"
            ].sum()
            lh_total = ts.loc[
                ts["Task_Object"].isin(lh_set), "FSM_Duration_s"
            ].sum()
            if bt_total > 0:
                out[hitls_cond]["before_takeoff"].append(float(bt_total))
            if lh_total > 0:
                out[hitls_cond]["lineup_hold"].append(float(lh_total))

    return out


def _mitls_per_task_fsm(n: int = 12) -> pd.DataFrame:
    """Return tidy DataFrame of FSM_Duration_s per task per rep per condition.

    Columns: hitls_condition, rep_idx, Task_Object, FSM_Duration_s
    """
    rows = []
    for mitls_cond in ["C1", "C2", "C3"]:
        hitls_cond = MITLS_TO_HITLS[mitls_cond]
        runs = list(iter_condition_runs(mitls_cond))[:n]
        for rep_idx, run_dir in enumerate(runs):
            try:
                ts = load_task_summary(run_dir)
            except FileNotFoundError:
                continue
            for _, row in ts.iterrows():
                rows.append({
                    "hitls_condition": hitls_cond,
                    "rep_idx":         rep_idx,
                    "Task_Object":     row["Task_Object"],
                    "FSM_Duration_s":  float(row["FSM_Duration_s"]),
                })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def _compute_stats(
    hitls_arrays: dict[str, dict[str, list[float]]],
    mitls_totals: dict[str, dict[str, list[float]]],
) -> dict:
    """Compute Friedman + pairwise for HITLS & MITLS, plus NMAE per cell."""
    results = {}

    for proc_key, info in PROC_INFO.items():
        h_arrays = hitls_arrays[proc_key]          # cond → list[float]
        m_arrays = mitls_totals                     # hitls_cond → proc_key → list

        # --- HITLS Friedman (per-participant paired data) ---
        h_groups = {c: h_arrays[c] for c in _COND_ORDER}
        h_fried  = friedman_test(h_groups)
        h_pairs  = wilcoxon_pairwise(h_groups)

        # --- MITLS Friedman (per-rep) ---
        m_groups = {c: m_arrays[c][proc_key] for c in _COND_ORDER}
        m_fried  = friedman_test(m_groups)
        m_pairs  = wilcoxon_pairwise(m_groups)

        # --- NMAE per condition ---
        nmae_vals: dict[str, float] = {}
        for cond in _COND_ORDER:
            h_vals = h_arrays[cond]
            m_vals = m_arrays[cond][proc_key]
            if h_vals and m_vals:
                h_mean = float(np.mean(h_vals))
                m_mean = float(np.mean(m_vals))
                nmae_vals[cond] = float(nmae(m_mean, h_mean))
            else:
                nmae_vals[cond] = float("nan")

        # --- Direction match ---
        h_means = {c: np.mean(h_arrays[c]) if h_arrays[c] else np.nan
                   for c in _COND_ORDER}
        m_means = {c: np.mean(m_arrays[c][proc_key]) if m_arrays[c][proc_key]
                   else np.nan for c in _COND_ORDER}
        h_rank  = sorted(_COND_ORDER, key=lambda c: h_means[c])
        m_rank  = sorted(_COND_ORDER, key=lambda c: m_means[c])
        dir_match = (h_rank == m_rank)

        results[proc_key] = {
            "h_fried":   h_fried,
            "h_pairs":   h_pairs,
            "m_fried":   m_fried,
            "m_pairs":   m_pairs,
            "nmae_vals": nmae_vals,
            "dir_match": dir_match,
            "h_means":   h_means,
            "m_means":   m_means,
            "h_sds":     {c: float(np.std(h_arrays[c], ddof=1))
                          if len(h_arrays[c]) > 1 else 0.0
                          for c in _COND_ORDER},
            "m_sds":     {c: float(np.std(m_arrays[c][proc_key], ddof=1))
                          if len(m_arrays[c][proc_key]) > 1 else 0.0
                          for c in _COND_ORDER},
            "h_ns":      {c: len(h_arrays[c]) for c in _COND_ORDER},
            "m_ns":      {c: len(m_arrays[c][proc_key]) for c in _COND_ORDER},
        }

    return results


# ---------------------------------------------------------------------------
# Tier 1 — publication figure
# ---------------------------------------------------------------------------

def _plot_pub_procedure_comparison(
    stats: dict,
    hitls_arrays: dict[str, dict[str, list[float]]],
    mitls_totals: dict[str, dict[str, list[float]]],
    save_path: Path,
) -> None:
    """2-panel figure: BEFORE TAKEOFF and LINE-UP AND HOLD.

    Per panel: side-by-side box plots (HITLS=blue, MITLS=red) per condition.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        "Procedure Duration: Model vs. Human Participants\n"
        "(covered procedures only)",
        fontsize=13, fontweight="bold",
    )

    proc_keys = list(PROC_INFO.keys())  # [before_takeoff, lineup_hold]

    for ax, proc_key in zip(axes, proc_keys):
        info   = PROC_INFO[proc_key]
        st     = stats[proc_key]
        n_cond = len(_COND_ORDER)
        x      = np.arange(n_cond)
        width  = 0.35

        for i, cond in enumerate(_COND_ORDER):
            h_vals = hitls_arrays[proc_key][cond]
            m_vals = mitls_totals[cond][proc_key]

            # HITLS box
            if h_vals:
                bp = ax.boxplot(
                    h_vals,
                    positions=[x[i] - width / 2],
                    widths=[width * 0.9],
                    patch_artist=True,
                    medianprops=dict(color="white", linewidth=2),
                    boxprops=dict(facecolor="#4878CF", alpha=0.7),
                    whiskerprops=dict(color="#4878CF"),
                    capprops=dict(color="#4878CF"),
                    flierprops=dict(marker="o", color="#4878CF",
                                   markerfacecolor="#4878CF", markersize=3),
                )

            # MITLS box
            if m_vals:
                ax.boxplot(
                    m_vals,
                    positions=[x[i] + width / 2],
                    widths=[width * 0.9],
                    patch_artist=True,
                    medianprops=dict(color="white", linewidth=2),
                    boxprops=dict(facecolor="#D65F5F", alpha=0.7),
                    whiskerprops=dict(color="#D65F5F"),
                    capprops=dict(color="#D65F5F"),
                    flierprops=dict(marker="o", color="#D65F5F",
                                   markerfacecolor="#D65F5F", markersize=3),
                )

        ax.set_xticks(x)
        ax.set_xticklabels(_COND_ORDER, fontsize=10)
        ax.set_ylabel("Procedure duration (s)", fontsize=10)
        ax.set_title(
            f"{info['label']}\n"
            f"({info['n_modelled']}/{info['n_total']} tasks modelled)",
            fontsize=11,
        )
        ax.grid(axis="y", linestyle="--", alpha=0.4)

        # NMAE annotation per condition
        for i, cond in enumerate(_COND_ORDER):
            nv = st["nmae_vals"][cond]
            if not np.isnan(nv):
                ax.text(
                    x[i], ax.get_ylim()[1] * 0.92 if ax.get_ylim()[1] > 0 else 1,
                    f"NMAE={nv:.0%}",
                    ha="center", va="bottom", fontsize=7,
                    color="darkred" if nv > 0.5 else "black",
                )

    # Legend
    human_patch = mpatches.Patch(color="#4878CF", alpha=0.7, label="Human (HITLS)")
    model_patch = mpatches.Patch(color="#D65F5F", alpha=0.7, label="Model (MITLS)")
    fig.legend(
        handles=[human_patch, model_patch],
        loc="lower center", ncol=2, fontsize=10,
        bbox_to_anchor=(0.5, -0.04),
    )

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [pub] Saved: {save_path.name}")


# ---------------------------------------------------------------------------
# Tier 2 — per-task envelope debug figure
# ---------------------------------------------------------------------------

def _plot_debug_per_task_envelope(
    per_task_df: pd.DataFrame,
    hitls_arrays: dict[str, dict[str, list[float]]],
    save_path: Path,
) -> None:
    """One row per task, violin of MITLS FSM_Duration_s per condition.
    NOT MODELLED tasks shown as grey placeholder.
    HITLS procedure mean_task_s shown as reference.
    """
    # Full task list with NOT MODELLED slots
    task_list_bt = BEFORE_TAKEOFF_TASKS[:]
    task_list_bt.insert(1, "FLAPS / SET FOR TAKEOFF")  # actual order position
    task_list_lh = LINEUP_HOLD_TASKS[:]
    all_tasks = task_list_bt + task_list_lh  # 11 slots total

    n_tasks = len(all_tasks)
    n_conds = len(_COND_ORDER)
    cond_colors = ["#4878CF", "#6ACC65", "#D65F5F"]  # TARS, TARP-S, TARP-F

    fig, ax = plt.subplots(figsize=(12, max(6, n_tasks * 0.8)))

    # Reference lines: HITLS mean_task_s (procedure level, from report)
    from model_validation.shared.hitls_loader import load_procedure_timing
    hitls_ref = {}  # proc_key → cond → mean_task_s
    for proc_key, info in PROC_INFO.items():
        df_mt = load_procedure_timing(info["hitls_key"], "mean_task_s")
        hitls_ref[proc_key] = {
            row["condition"]: row["mean"]
            for _, row in df_mt.iterrows()
            if row["condition"] in SHARED_CONDITIONS
        }

    y_positions = {}   # task_name → y_center
    y = n_tasks - 1
    for task_name in all_tasks:
        y_positions[task_name] = y
        y -= 1

    offset_step = 0.22
    offsets = [-(n_conds - 1) * offset_step / 2 + i * offset_step
               for i in range(n_conds)]

    for ti, task_name in enumerate(all_tasks):
        y_center = y_positions[task_name]
        is_modelled = task_name in set(BEFORE_TAKEOFF_TASKS) or \
                      task_name in set(LINEUP_HOLD_TASKS)

        if not is_modelled:
            # Grey placeholder bar
            ax.barh(
                y_center, 15, height=0.6,
                color=_NOT_MODELLED_COLOR, alpha=0.5,
                left=0, zorder=1,
            )
            ax.text(
                0.5, y_center, "[NOT MODELLED]",
                va="center", ha="left", fontsize=8, color="grey",
            )
            continue

        # Determine procedure key for this task
        proc_key = "before_takeoff" if task_name in set(BEFORE_TAKEOFF_TASKS) \
                   else "lineup_hold"

        for ci, (cond, color) in enumerate(zip(_COND_ORDER, cond_colors)):
            task_vals = per_task_df.loc[
                (per_task_df["hitls_condition"] == cond)
                & (per_task_df["Task_Object"] == task_name),
                "FSM_Duration_s",
            ].dropna().values

            y_off = y_center + offsets[ci]

            if len(task_vals) >= 3:
                # Horizontal violin (manually drawn as patch)
                from scipy.stats import gaussian_kde
                kde = gaussian_kde(task_vals, bw_method=0.5)
                x_range = np.linspace(task_vals.min(), task_vals.max(), 100)
                density = kde(x_range)
                density = density / density.max() * 0.18  # scale height

                ax.fill_between(
                    x_range,
                    y_off - density,
                    y_off + density,
                    alpha=0.5, color=color,
                )
                ax.plot(
                    [np.mean(task_vals), np.mean(task_vals)],
                    [y_off - 0.15, y_off + 0.15],
                    color=color, lw=2, zorder=3,
                )
            elif len(task_vals) > 0:
                ax.scatter(
                    task_vals, [y_off] * len(task_vals),
                    color=color, s=20, alpha=0.7, zorder=3,
                )

    # HITLS procedure mean_task_s reference lines (dashed)
    from model_validation.shared.hitls_loader import _parse_report_json
    time_report_path = HITLS_DIR / "compare_performance" / "time_report.txt"
    time_data = _parse_report_json(time_report_path)
    for ci, cond in enumerate(_COND_ORDER):
        color = cond_colors[ci]
        for proc_key in PROC_INFO:
            ref_val = (time_data["procedures"]
                       .get(PROC_INFO[proc_key]["hitls_key"], {})
                       .get("mean_task_s", {})
                       .get(cond, {})
                       .get("mean"))
            if ref_val is None:
                continue
            # Determine y-range for this procedure
            proc_tasks = [t for t in all_tasks
                          if t in set(PROC_INFO[proc_key]["tasks"])]
            if not proc_tasks:
                continue
            y_lo = min(y_positions[t] for t in proc_tasks) - 0.4
            y_hi = max(y_positions[t] for t in proc_tasks) + 0.4
            ax.plot(
                [ref_val, ref_val], [y_lo, y_hi],
                linestyle="--", color=color, lw=1.2, alpha=0.6,
                label=f"HITLS {cond} mean_task" if ci == 0 else None,
            )

    ax.set_yticks(list(y_positions.values()))
    ax.set_yticklabels(list(y_positions.keys()), fontsize=9)
    ax.set_xlabel("FSM Duration (s)", fontsize=10)
    ax.set_title(
        "Per-Task MITLS FSM Duration (violin) with HITLS procedure mean_task_s (dashed)",
        fontsize=11,
    )
    ax.grid(axis="x", linestyle="--", alpha=0.3)

    # Add procedure separators
    sep_y = y_positions["Winds"] + 0.5 + 0.5
    ax.axhline(sep_y, color="grey", lw=0.8, linestyle=":")
    ax.text(ax.get_xlim()[1] * 0.98 if ax.get_xlim()[1] > 0 else 20,
            sep_y + 0.1, "↑ L.U.H.  ↓ B.T.", ha="right", fontsize=7, color="grey")

    # Legend
    legend_patches = [
        mpatches.Patch(color=cond_colors[i], alpha=0.7, label=cond)
        for i, cond in enumerate(_COND_ORDER)
    ]
    legend_patches.append(
        mpatches.Patch(color=_NOT_MODELLED_COLOR, alpha=0.5, label="[NOT MODELLED]")
    )
    ax.legend(handles=legend_patches, loc="lower right", fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [debug] Saved: {save_path.name}")


def _plot_debug_nmae_sorted(stats: dict, save_path: Path) -> None:
    """Sorted NMAE bar chart (procedure × condition).  |NMAE| > 0.5 in red."""
    bars = []
    for proc_key, info in PROC_INFO.items():
        st = stats[proc_key]
        for cond in _COND_ORDER:
            nv = st["nmae_vals"][cond]
            if not np.isnan(nv):
                bars.append((f"{cond}\n({info['label'][:6]})", nv))

    bars.sort(key=lambda x: x[1])
    labels = [b[0] for b in bars]
    values = [b[1] for b in bars]
    colors = ["#D65F5F" if v > 0.5 else "#4878CF" for v in values]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(labels, values, color=colors, alpha=0.75)
    ax.axvline(0.5, color="darkred", linestyle="--", lw=1.5, label="|NMAE| = 0.5")
    ax.set_xlabel("NMAE (fraction)", fontsize=10)
    ax.set_title("Timing NMAE per Condition × Procedure (sorted)", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(axis="x", linestyle="--", alpha=0.4)

    # Value labels
    for i, (label, val) in enumerate(zip(labels, values)):
        ax.text(val + 0.01, i, f"{val:.2f}", va="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [debug] Saved: {save_path.name}")


# ---------------------------------------------------------------------------
# Report writers
# ---------------------------------------------------------------------------

def _write_tier1_report(stats: dict, save_path: Path) -> None:
    lines = [
        "=" * 60,
        "TIMING VALIDATION — TIER 1 REPORT",
        "=" * 60,
        "",
        "Scope: BEFORE TAKEOFF (8/9 tasks) + LINE-UP AND HOLD (2/2 tasks)",
        "MITLS: FSM_Duration_s per task, summed per procedure (12 reps/condition)",
        "HITLS: per-participant procedure total_s from cleaned time reports",
        "",
    ]

    for proc_key, info in PROC_INFO.items():
        st = stats[proc_key]
        lines += [
            "-" * 50,
            f"{info['label']}  ({info['n_modelled']}/{info['n_total']} tasks modelled)",
            "-" * 50,
            "",
            "HITLS (participants):",
        ]
        for cond in _COND_ORDER:
            h_mean = st["h_means"][cond]
            h_sd   = st["h_sds"][cond]
            n      = st["h_ns"][cond]
            lines.append(f"  {cond:8s}  n={n:2d}  mean={h_mean:6.1f}s  SD={h_sd:.1f}s")

        lines += ["", "MITLS (model repetitions):"]
        for cond in _COND_ORDER:
            m_mean = st["m_means"][cond]
            m_sd   = st["m_sds"][cond]
            n      = st["m_ns"][cond]
            lines.append(f"  {cond:8s}  n={n:2d}  mean={m_mean:6.1f}s  SD={m_sd:.1f}s")

        lines += ["", "NMAE (|model_mean − human_mean| / human_mean):"]
        for cond in _COND_ORDER:
            nv = st["nmae_vals"][cond]
            flag = "  ← HIGH" if not np.isnan(nv) and nv > 0.5 else ""
            lines.append(f"  {cond:8s}  {nv:.1%}{flag}")

        lines += ["",
                  f"Direction match (condition ranking): {st['dir_match']}"]

        h_fried = st["h_fried"]
        m_fried = st["m_fried"]
        lines += [
            "",
            "Friedman (HITLS):",
            f"  χ²({h_fried.df}, N={h_fried.n}) = {h_fried.chi2:.3f}  "
            f"p = {fmt_p(h_fried.p)}  W = {h_fried.kendall_w:.3f}",
            "",
            "Friedman (MITLS):",
            f"  χ²({m_fried.df}, N={m_fried.n}) = {m_fried.chi2:.3f}  "
            f"p = {fmt_p(m_fried.p)}  W = {m_fried.kendall_w:.3f}",
            "",
            "Pairwise Wilcoxon (HITLS, Holm-corrected):",
        ]
        for pw in st["h_pairs"]:
            stars = sig_stars(pw.p_corrected, pw.reject)
            lines.append(
                f"  {pw.condition_a} vs {pw.condition_b}: "
                f"p_raw={fmt_p(pw.p_raw)}  p_corr={fmt_p(pw.p_corrected)} "
                f"r={pw.r:+.3f}  {stars}"
            )
        lines += [
            "",
            "Pairwise Wilcoxon (MITLS, Holm-corrected):",
        ]
        for pw in st["m_pairs"]:
            stars = sig_stars(pw.p_corrected, pw.reject)
            lines.append(
                f"  {pw.condition_a} vs {pw.condition_b}: "
                f"p_raw={fmt_p(pw.p_raw)}  p_corr={fmt_p(pw.p_corrected)} "
                f"r={pw.r:+.3f}  {stars}"
            )
        lines.append("")

    lines += [
        "=" * 60,
        "NOTE: BEFORE TAKEOFF comparison is slightly conservative — MITLS",
        "models 8/9 tasks; FLAPS / SET FOR TAKEOFF is absent, so model",
        "durations are systematically shorter by ≈1 task.",
        "=" * 60,
    ]

    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  [pub] Saved: {save_path.name}")


def _write_tier2_report(
    stats: dict,
    per_task_df: pd.DataFrame,
    save_path: Path,
) -> None:
    lines = [
        "=" * 60,
        "TIMING VALIDATION — TIER 2 DEBUG REPORT",
        "=" * 60,
        "",
        "Coverage note: BEFORE TAKEOFF 8/9 | LINE-UP AND HOLD 2/2",
        "",
    ]

    # Per-task FSM stats
    lines += ["-" * 50, "Per-Task MITLS FSM_Duration_s (mean ± SD across reps)", "-" * 50]
    for proc_key, info in PROC_INFO.items():
        lines.append(f"\n{info['label']}:")
        all_tasks = list(BEFORE_TAKEOFF_TASKS) if proc_key == "before_takeoff" \
                    else list(LINEUP_HOLD_TASKS)
        for task in all_tasks:
            line_parts = [f"  {task:<35s}"]
            for cond in _COND_ORDER:
                vals = per_task_df.loc[
                    (per_task_df["hitls_condition"] == cond)
                    & (per_task_df["Task_Object"] == task),
                    "FSM_Duration_s"
                ].dropna()
                if len(vals) > 0:
                    line_parts.append(
                        f"{cond}: {vals.mean():.2f}s ± {vals.std(ddof=1):.2f}"
                    )
                else:
                    line_parts.append(f"{cond}: —")
            lines.append("  ".join(line_parts))

        for task in info["not_modelled"]:
            lines.append(f"  {task:<35s}  [NOT MODELLED]")

    # NMAE table
    lines += [
        "",
        "-" * 50,
        "NMAE Summary",
        "-" * 50,
        f"{'Procedure':<20s}  {'Condition':<10s}  NMAE",
    ]
    for proc_key, info in PROC_INFO.items():
        st = stats[proc_key]
        for cond in _COND_ORDER:
            nv = st["nmae_vals"][cond]
            flag = " ← HIGH" if not np.isnan(nv) and nv > 0.5 else ""
            lines.append(
                f"  {info['label']:<20s}  {cond:<10s}  {nv:.1%}{flag}"
            )

    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  [debug] Saved: {save_path.name}")


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_timing_validation(n: int = 12) -> dict:
    """Run full timing validation and write all outputs.

    Returns a metrics dict for use by run_all.py.
    """
    print("=" * 60)
    print("TIMING VALIDATION")
    print("=" * 60)

    print("\n[1/5] Loading data...")
    per_participant = _load_per_participant_timing()
    print(f"  HITLS: {len(per_participant)} participants loaded")
    hitls_arrays  = _hitls_procedure_arrays(per_participant)
    mitls_totals  = _mitls_procedure_totals(n)
    per_task_df   = _mitls_per_task_fsm(n)

    for proc_key, info in PROC_INFO.items():
        for cond in _COND_ORDER:
            hn = len(hitls_arrays[proc_key][cond])
            mn = len(mitls_totals[cond][proc_key])
            print(f"  {info['label']} / {cond}: HITLS n={hn}, MITLS n={mn}")

    print("\n[2/5] Running statistics...")
    stats = _compute_stats(hitls_arrays, mitls_totals)
    for proc_key, info in PROC_INFO.items():
        st = stats[proc_key]
        print(f"  {info['label']}:")
        print(
            f"    HITLS Friedman: χ²({st['h_fried'].df}, N={st['h_fried'].n})"
            f" = {st['h_fried'].chi2:.3f}  p={fmt_p(st['h_fried'].p)}"
            f"  W={st['h_fried'].kendall_w:.3f}"
        )
        print(
            f"    MITLS Friedman: χ²({st['m_fried'].df}, N={st['m_fried'].n})"
            f" = {st['m_fried'].chi2:.3f}  p={fmt_p(st['m_fried'].p)}"
            f"  W={st['m_fried'].kendall_w:.3f}"
        )
        for cond in _COND_ORDER:
            nv = st["nmae_vals"][cond]
            print(f"    NMAE {cond}: {nv:.1%}")
        print(f"    Direction match: {st['dir_match']}")

    print("\n[3/5] Writing Tier 1 outputs (publication)...")
    _plot_pub_procedure_comparison(
        stats, hitls_arrays, mitls_totals,
        PUB_DIR / "procedure_timing_comparison.png",
    )
    _write_tier1_report(
        stats,
        VAL_DIR / "timing" / "timing_report.txt",
    )

    print("\n[4/5] Writing Tier 2 outputs (diagnostic)...")
    _plot_debug_per_task_envelope(
        per_task_df,
        hitls_arrays,
        DEBUG_DIR / "timing_per_task_envelope.png",
    )
    _plot_debug_nmae_sorted(
        stats,
        DEBUG_DIR / "timing_nmae_sorted.png",
    )
    _write_tier2_report(
        stats,
        per_task_df,
        VAL_DIR / "timing" / "timing_debug_report.txt",
    )

    print("\n[5/5] Done.")

    # Build summary metrics dict
    metrics: dict = {}
    for proc_key, info in PROC_INFO.items():
        st     = stats[proc_key]
        prefix = proc_key
        metrics[f"{prefix}_hitls_friedman_chi2"] = st["h_fried"].chi2
        metrics[f"{prefix}_hitls_friedman_p"]    = st["h_fried"].p
        metrics[f"{prefix}_hitls_kendall_w"]     = st["h_fried"].kendall_w
        metrics[f"{prefix}_mitls_friedman_chi2"] = st["m_fried"].chi2
        metrics[f"{prefix}_mitls_friedman_p"]    = st["m_fried"].p
        metrics[f"{prefix}_mitls_kendall_w"]     = st["m_fried"].kendall_w
        metrics[f"{prefix}_dir_match"]           = st["dir_match"]
        for cond in _COND_ORDER:
            metrics[f"{prefix}_nmae_{cond}"] = st["nmae_vals"][cond]

    print("\nSummary metrics:")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    return metrics


if __name__ == "__main__":
    run_timing_validation()
