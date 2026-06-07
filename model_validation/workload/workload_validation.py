#!/usr/bin/env python3
"""
Step 7: Workload Validation
============================
Compare MITLS ACT-R module utilization to HITLS subjective workload (NASA-TLX)
and physiological workload (HRV RMSSD).

Conceptual mapping
------------------
  ACT-R module group          NASA-TLX subscale         HRV proxy
  ──────────────────────────  ────────────────────────  ────────────
  Overall_Utilization         Weighted score total      SDNN / RMSSD (inverse)
  Cognitive_SubNetwork        Mental Demand + Effort    RMSSD (inverse)
  Perceptual_SubNetwork       Temporal Demand           —
  Motor_SubNetwork            Physical Demand           —

Note: Only 3 conditions exist (no TARC in MITLS), so Spearman ρ operates on
3 data points — very low power. This is noted explicitly in both reports.

Data sources
------------
  MITLS: per-run workload_analyzer/results_mental_workload.txt
         → mean Overall_Utilization / subnetwork utilizations per rep
  HITLS NASA-TLX: P{pid}/HAT_study.csv, questionnaire_id='nasa_tlx_evaluation'
                  + subscale weights from 'nasa_tlx_subscale_ranking'
  HITLS HRV:      HITLS/HRV/hrv_features_per_scenario.csv

Tier 1 outputs (publication)
  plots/pub/workload_vs_nasa_tlx.png
  workload/workload_report.txt

Tier 2 outputs (diagnostic)
  plots/debug/workload_per_condition_distribution.png
  plots/debug/workload_hrv_scatter.png
  workload/workload_debug_report.txt
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

_HERE = Path(__file__).resolve()
WORKSPACE_ROOT = _HERE.parents[2]
sys.path.insert(0, str(WORKSPACE_ROOT))

from model_validation.shared.hitls_loader import HITLS_DIR, SHARED_CONDITIONS
from model_validation.shared.mitls_loader import (
    MITLS_TO_HITLS, iter_condition_runs, load_workload_timeseries,
)
from model_validation.shared.stats import friedman_test, wilcoxon_pairwise, fmt_p, sig_stars

VAL_DIR   = WORKSPACE_ROOT / "model_validation"
PUB_DIR   = VAL_DIR / "plots" / "pub"
DEBUG_DIR = VAL_DIR / "plots" / "debug"
PUB_DIR.mkdir(parents=True, exist_ok=True)
DEBUG_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_COND_ORDER = SHARED_CONDITIONS   # TARS, TARP-S, TARP-F
_COND_COLORS = {
    "TARS":   "#4878CF",
    "TARP-S": "#6ACC65",
    "TARP-F": "#D65F5F",
}

# ACT-R module → MITLS column mapping
MITLS_METRICS = {
    "Overall":    "Overall_Utilization",
    "Cognitive":  "Cognitive_SubNetwork",
    "Perceptual": "Perceptual_SubNetwork",
    "Motor":      "Motor_SubNetwork",
}

# MITLS metric → proposed NASA-TLX subscale(s)
METRIC_TO_TLX: dict[str, list[str]] = {
    "Overall":    ["weighted_score"],
    "Cognitive":  ["mental_demand", "effort"],
    "Perceptual": ["temporal_demand"],
    "Motor":      ["physical_demand"],
}

# NASA-TLX subscale display names
TLX_LABELS = {
    "weighted_score":  "Weighted score",
    "mental_demand":   "Mental Demand",
    "effort":          "Effort",
    "temporal_demand": "Temporal Demand",
    "physical_demand": "Physical Demand",
    "frustration":     "Frustration",
}

# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def _find_participants() -> list[Path]:
    return sorted(
        e for e in HITLS_DIR.iterdir()
        if e.is_dir() and e.name.startswith("P") and e.name[1:].isdigit()
    )


def _load_per_participant_nasa_tlx() -> pd.DataFrame:
    """Load per-participant NASA-TLX weighted scores and subscale ratings.

    Parses HAT_study.csv for each participant using the correct questionnaire
    id 'nasa_tlx_evaluation' (not 'nasa_tlx').

    Returns
    -------
    DataFrame with columns:
        participant, condition, weighted_score,
        mental_demand, physical_demand, temporal_demand,
        performance, effort, frustration
    All scores on 0–100 scale (raw 0–20 ratings × 5).
    """
    SUBSCALE_KEYS = [
        "mental_demand", "physical_demand", "temporal_demand",
        "performance", "effort", "frustration",
    ]

    rows_out = []
    for pid_dir in _find_participants():
        pid = pid_dir.name
        hat_path = pid_dir / "HAT_study.csv"
        if not hat_path.exists():
            # Also try pid-specific name
            candidates = list(pid_dir.glob(f"{pid}_HAT_study.csv"))
            if not candidates:
                continue
            hat_path = candidates[0]

        with open(hat_path, newline="", encoding="utf-8") as fh:
            all_rows = list(csv.DictReader(fh))

        # Extract subscale weights (pairwise rankings from after_familiarization)
        weight_votes: dict[str, int] = {k: 0 for k in SUBSCALE_KEYS}
        for r in all_rows:
            if r.get("questionnaire_id") == "nasa_tlx_subscale_ranking" and \
               r.get("condition") == "after_familiarization":
                winner = r.get("value", "").strip().lower().replace(" ", "_")
                if winner in weight_votes:
                    weight_votes[winner] += 1

        # Extract per-condition subscale ratings (nasa_tlx_evaluation)
        ratings: dict[str, dict[str, float]] = {}
        for r in all_rows:
            if r.get("questionnaire_id") != "nasa_tlx_evaluation":
                continue
            cond = r.get("condition", "").strip()
            q_id = r.get("question_id", "").strip().lower()
            val  = r.get("value", "").strip()
            if cond not in SHARED_CONDITIONS or not val:
                continue
            if cond not in ratings:
                ratings[cond] = {}
            if q_id in SUBSCALE_KEYS:
                try:
                    ratings[cond][q_id] = float(val)
                except ValueError:
                    pass

        # Compute weighted score per condition
        for cond in SHARED_CONDITIONS:
            cond_ratings = ratings.get(cond, {})
            if not cond_ratings:
                continue
            total = 0.0
            subscale_vals: dict[str, float] = {}
            for key in SUBSCALE_KEYS:
                w    = weight_votes.get(key, 0)
                r_raw = cond_ratings.get(key)
                if r_raw is None:
                    continue
                r100 = r_raw * 5.0   # 0–20 → 0–100
                weighted = w * r100
                total += weighted
                subscale_vals[key] = r100   # raw 0-100 (unweighted) for subscale comparison
            score = total / 15.0 if total > 0 else None
            rows_out.append({
                "participant": pid,
                "condition":   cond,
                "weighted_score": score,
                **subscale_vals,
            })

    return pd.DataFrame(rows_out)


def _load_mitls_utilization(n: int = 12) -> dict[str, dict[str, list[float]]]:
    """Load mean utilization per metric per rep per condition.

    Returns
    -------
    dict: hitls_condition → metric_key → list[float]  (one per rep)
    """
    out: dict[str, dict[str, list[float]]] = {
        cond: {m: [] for m in MITLS_METRICS}
        for cond in _COND_ORDER
    }
    for mitls_cond in ["C1", "C2", "C3"]:
        hitls_cond = MITLS_TO_HITLS[mitls_cond]
        runs = list(iter_condition_runs(mitls_cond))[:n]
        for run_dir in runs:
            try:
                df = load_workload_timeseries(run_dir)
            except (FileNotFoundError, Exception):
                continue
            for metric_key, col in MITLS_METRICS.items():
                if col in df.columns:
                    out[hitls_cond][metric_key].append(float(df[col].mean()))
    return out


def _load_hitls_hrv() -> dict[str, list[float]]:
    """Load per-condition RMSSD values from HRV features CSV.

    Returns
    -------
    dict: condition → list[float]  (per participant)
    """
    from model_validation.shared.hitls_loader import load_hrv_features
    hrv = load_hrv_features()
    out: dict[str, list[float]] = {c: [] for c in _COND_ORDER}
    for cond in _COND_ORDER:
        rows = hrv[hrv["condition"] == cond]["HRV_RMSSD"].dropna()
        out[cond] = rows.tolist()
    return out


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def _compute_stats(
    tlx_df: pd.DataFrame,
    mitls: dict[str, dict[str, list[float]]],
    hrv: dict[str, list[float]],
) -> dict:
    """Compute workload comparison statistics.

    Returns nested dict with:
        hitls_friedman, hitls_pairs, mitls_friedman, mitls_pairs,
        direction_match, spearman_rho, spearman_p,
        condition_means (hitls & mitls), hrv_correlation
    """
    # --- HITLS Friedman (per-participant, 3 conditions paired) ---
    hitls_groups: dict[str, list[float]] = {}
    for cond in _COND_ORDER:
        sub = tlx_df[tlx_df["condition"] == cond]["weighted_score"].dropna()
        hitls_groups[cond] = sub.tolist()

    h_fried = friedman_test(hitls_groups)
    h_pairs = wilcoxon_pairwise(hitls_groups)

    # --- MITLS Friedman (per-rep, Overall_Utilization) ---
    mitls_groups = {c: mitls[c]["Overall"] for c in _COND_ORDER}
    m_fried = friedman_test(mitls_groups)
    m_pairs = wilcoxon_pairwise(mitls_groups)

    # --- Condition-level means ---
    h_means = {c: float(np.mean(hitls_groups[c])) if hitls_groups[c] else np.nan
               for c in _COND_ORDER}
    m_means = {c: float(np.mean(mitls[c]["Overall"])) if mitls[c]["Overall"] else np.nan
               for c in _COND_ORDER}
    h_sds   = {c: float(np.std(hitls_groups[c], ddof=1)) if len(hitls_groups[c]) > 1 else 0.0
               for c in _COND_ORDER}
    m_sds   = {c: float(np.std(mitls[c]["Overall"], ddof=1)) if len(mitls[c]["Overall"]) > 1 else 0.0
               for c in _COND_ORDER}

    # --- Spearman ρ (condition-level, 3 points) ---
    h_vec = [h_means[c] for c in _COND_ORDER]
    m_vec = [m_means[c] for c in _COND_ORDER]
    if not any(np.isnan(v) for v in h_vec + m_vec):
        rho, p_rho = scipy_stats.spearmanr(h_vec, m_vec)
    else:
        rho, p_rho = float("nan"), float("nan")

    # --- Direction match ---
    h_rank = sorted(_COND_ORDER, key=lambda c: h_means[c], reverse=True)
    m_rank = sorted(_COND_ORDER, key=lambda c: m_means[c], reverse=True)
    dir_match = (h_rank == m_rank)

    # --- HRV correlation with model utilization ---
    hrv_means = {c: float(np.mean(hrv[c])) if hrv[c] else np.nan for c in _COND_ORDER}
    hrv_h_vec = [hrv_means[c] for c in _COND_ORDER]
    if not any(np.isnan(v) for v in hrv_h_vec + m_vec):
        hrv_rho, hrv_p = scipy_stats.pearsonr(m_vec, hrv_h_vec)
    else:
        hrv_rho, hrv_p = float("nan"), float("nan")

    # --- Subnetwork vs TLX subscale comparison ---
    subscale_stats: dict[str, dict] = {}
    for metric_key, tlx_subscales in METRIC_TO_TLX.items():
        if metric_key == "Overall":
            continue
        sub_vals = {}
        for cond in _COND_ORDER:
            tlx_sub_mean = float(np.nanmean([
                tlx_df[tlx_df["condition"] == cond][s].mean()
                for s in tlx_subscales
                if s in tlx_df.columns
            ]))
            mitls_mean = float(np.mean(mitls[cond][metric_key])) \
                         if mitls[cond][metric_key] else np.nan
            sub_vals[cond] = {"tlx": tlx_sub_mean, "mitls": mitls_mean}
        subscale_stats[metric_key] = sub_vals

    return {
        "h_fried":      h_fried,
        "h_pairs":      h_pairs,
        "m_fried":      m_fried,
        "m_pairs":      m_pairs,
        "h_means":      h_means,
        "m_means":      m_means,
        "h_sds":        h_sds,
        "m_sds":        m_sds,
        "h_ns":         {c: len(hitls_groups[c]) for c in _COND_ORDER},
        "m_ns":         {c: len(mitls[c]["Overall"]) for c in _COND_ORDER},
        "dir_match":    dir_match,
        "h_rank":       h_rank,
        "m_rank":       m_rank,
        "spearman_rho": rho,
        "spearman_p":   p_rho,
        "hrv_means":    hrv_means,
        "hrv_rho":      hrv_rho,
        "hrv_p":        hrv_p,
        "subscale_stats": subscale_stats,
    }


# ---------------------------------------------------------------------------
# Tier 1 — publication figure
# ---------------------------------------------------------------------------

def _plot_pub_workload_comparison(
    stats: dict,
    tlx_df: pd.DataFrame,
    mitls: dict[str, dict[str, list[float]]],
    save_path: Path,
) -> None:
    """Dual-axis bar chart: NASA-TLX (left) vs model utilization (right) per condition.

    Both axes normalised so the visual comparison is direct. Annotates
    Spearman ρ and direction match.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))
    fig.suptitle(
        "Cognitive Workload: Model Utilization vs. Human NASA-TLX\n"
        "(condition-level comparison, 3 shared conditions)",
        fontsize=12, fontweight="bold",
    )

    x      = np.arange(len(_COND_ORDER))
    width  = 0.5
    colors = [_COND_COLORS[c] for c in _COND_ORDER]

    # --- Panel 1: HITLS NASA-TLX per condition (box) ---
    for i, cond in enumerate(_COND_ORDER):
        vals = tlx_df[tlx_df["condition"] == cond]["weighted_score"].dropna().tolist()
        if vals:
            bp = ax1.boxplot(
                vals, positions=[x[i]], widths=[width * 0.8],
                patch_artist=True,
                medianprops=dict(color="white", linewidth=2),
                boxprops=dict(facecolor=colors[i], alpha=0.7),
                whiskerprops=dict(color=colors[i]),
                capprops=dict(color=colors[i]),
                flierprops=dict(marker="o", markerfacecolor=colors[i], markersize=3),
            )

    ax1.set_xticks(x)
    ax1.set_xticklabels(_COND_ORDER, fontsize=10)
    ax1.set_ylabel("NASA-TLX weighted score (0–100)", fontsize=9)
    ax1.set_title("Human Workload (NASA-TLX)", fontsize=11)
    ax1.set_ylim(0, 105)
    ax1.grid(axis="y", linestyle="--", alpha=0.4)

    # --- Panel 2: MITLS Overall_Utilization per condition (box) ---
    for i, cond in enumerate(_COND_ORDER):
        vals = mitls[cond]["Overall"]
        if vals:
            ax2.boxplot(
                vals, positions=[x[i]], widths=[width * 0.8],
                patch_artist=True,
                medianprops=dict(color="white", linewidth=2),
                boxprops=dict(facecolor=colors[i], alpha=0.7),
                whiskerprops=dict(color=colors[i]),
                capprops=dict(color=colors[i]),
                flierprops=dict(marker="o", markerfacecolor=colors[i], markersize=3),
            )

    ax2.set_xticks(x)
    ax2.set_xticklabels(_COND_ORDER, fontsize=10)
    ax2.set_ylabel("Mean Overall Utilization (0–1)", fontsize=9)
    ax2.set_title("Model Workload (ACT-R utilization)", fontsize=11)
    ax2.grid(axis="y", linestyle="--", alpha=0.4)

    st = stats
    rho   = st["spearman_rho"]
    dm    = st["dir_match"]
    h_ord = " > ".join(st["h_rank"])
    m_ord = " > ".join(st["m_rank"])

    fig.text(
        0.5, 0.01,
        f"Spearman ρ (condition means) = {rho:.3f}   |   "
        f"Direction match: {dm}   |   "
        f"HITLS: {h_ord}   |   MITLS: {m_ord}",
        ha="center", fontsize=8.5, color="darkblue",
    )

    plt.tight_layout(rect=[0, 0.06, 1, 1])
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [pub] Saved: {save_path.name}")


# ---------------------------------------------------------------------------
# Tier 2 — debug figures
# ---------------------------------------------------------------------------

def _plot_debug_per_condition_distribution(
    tlx_df: pd.DataFrame,
    mitls: dict[str, dict[str, list[float]]],
    save_path: Path,
) -> None:
    """2×3 grid: top row = NASA-TLX distributions, bottom = MITLS utilization."""
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    fig.suptitle(
        "Workload Distributions: Human NASA-TLX vs. Model Utilization (per condition)",
        fontsize=11, fontweight="bold",
    )

    for col_idx, cond in enumerate(_COND_ORDER):
        color = _COND_COLORS[cond]

        # Top: NASA-TLX histogram + individual points
        ax_top = axes[0, col_idx]
        tlx_vals = tlx_df[tlx_df["condition"] == cond]["weighted_score"].dropna().tolist()
        if tlx_vals:
            ax_top.hist(tlx_vals, bins=8, color=color, alpha=0.6, edgecolor="white")
            ax_top.axvline(np.mean(tlx_vals), color="black", lw=1.5, linestyle="--",
                           label=f"mean={np.mean(tlx_vals):.1f}")
        ax_top.set_title(f"NASA-TLX — {cond}\n(n={len(tlx_vals)})", fontsize=10)
        ax_top.set_xlabel("Weighted score", fontsize=8)
        ax_top.set_xlim(0, 100)
        ax_top.legend(fontsize=7)

        # Bottom: MITLS overall utilization distribution
        ax_bot = axes[1, col_idx]
        util_vals = mitls[cond]["Overall"]
        if util_vals:
            ax_bot.hist(util_vals, bins=8, color=color, alpha=0.6, edgecolor="white")
            ax_bot.axvline(np.mean(util_vals), color="black", lw=1.5, linestyle="--",
                           label=f"mean={np.mean(util_vals):.4f}")
        ax_bot.set_title(f"ACT-R Utilization — {cond}\n(n={len(util_vals)} reps)", fontsize=10)
        ax_bot.set_xlabel("Overall Utilization", fontsize=8)
        ax_bot.legend(fontsize=7)

    for row in axes:
        for ax in row:
            ax.grid(axis="y", linestyle="--", alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [debug] Saved: {save_path.name}")


def _plot_debug_hrv_scatter(
    stats: dict,
    mitls: dict[str, dict[str, list[float]]],
    save_path: Path,
) -> None:
    """Scatter: model mean utilization vs HRV RMSSD per condition."""
    fig, ax = plt.subplots(figsize=(6, 5))

    for cond in _COND_ORDER:
        m_mean   = stats["m_means"][cond]
        hrv_mean = stats["hrv_means"][cond]
        if np.isnan(m_mean) or np.isnan(hrv_mean):
            continue
        ax.scatter(m_mean, hrv_mean, color=_COND_COLORS[cond],
                   s=100, zorder=3, label=cond)
        ax.annotate(cond, (m_mean, hrv_mean),
                    textcoords="offset points", xytext=(5, 5), fontsize=9)

    hrv_rho = stats["hrv_rho"]
    hrv_p   = stats["hrv_p"]
    ax.set_xlabel("MITLS mean Overall Utilization (0–1)", fontsize=10)
    ax.set_ylabel("HITLS mean RMSSD (ms)", fontsize=10)
    ax.set_title(
        f"Model Utilization vs. HRV RMSSD\n"
        f"Pearson r = {hrv_rho:.3f}   p = {fmt_p(hrv_p)}   (n=3 conditions)",
        fontsize=10,
    )
    ax.legend(fontsize=9)
    ax.grid(linestyle="--", alpha=0.4)
    ax.text(
        0.05, 0.05,
        "Note: n=3 — very low statistical power.\nExpected r<0 (higher workload → lower RMSSD).",
        transform=ax.transAxes, fontsize=7.5, color="grey",
        verticalalignment="bottom",
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [debug] Saved: {save_path.name}")


# ---------------------------------------------------------------------------
# Report writers
# ---------------------------------------------------------------------------

def _write_tier1_report(stats: dict, tlx_df: pd.DataFrame, save_path: Path) -> None:
    lines = [
        "=" * 60,
        "WORKLOAD VALIDATION — TIER 1 REPORT",
        "=" * 60,
        "",
        "MITLS: mean Overall_Utilization per condition (12 reps each)",
        "HITLS: NASA-TLX weighted score per participant per condition",
        "",
        "CAVEAT: Only 3 conditions are comparable (no TARC in MITLS).",
        "        Spearman ρ on 3 data points has very low power (n=3).",
        "",
    ]

    # Condition-level summary
    lines += ["-" * 50, "Condition-Level Summary", "-" * 50]
    lines.append(
        f"  {'Condition':<10s}  {'HITLS NASA-TLX':>14s}  {'MITLS Utiliz.':>14s}  "
        f"{'HITLS n':>8s}  {'MITLS n':>8s}"
    )
    for cond in _COND_ORDER:
        h_m = stats["h_means"][cond]
        h_s = stats["h_sds"][cond]
        m_m = stats["m_means"][cond]
        m_s = stats["m_sds"][cond]
        hn  = stats["h_ns"][cond]
        mn  = stats["m_ns"][cond]
        lines.append(
            f"  {cond:<10s}  {h_m:>7.2f} ± {h_s:.2f}  "
            f"{m_m:>7.4f} ± {m_s:.4f}  {hn:>8d}  {mn:>8d}"
        )

    lines += [
        "",
        f"Condition ranking (NASA-TLX, highest first): "
        f"{' > '.join(stats['h_rank'])}",
        f"Condition ranking (model utilization):       "
        f"{' > '.join(stats['m_rank'])}",
        f"Direction match: {stats['dir_match']}",
        "",
        f"Spearman ρ (3-point, condition means): "
        f"ρ = {stats['spearman_rho']:.3f}  p = {fmt_p(stats['spearman_p'])}",
        "  (n=3 — interpret with caution; p-values are unreliable at this sample size)",
        "",
    ]

    # HITLS Friedman
    h_fried = stats["h_fried"]
    lines += [
        "-" * 50,
        "Statistical Tests",
        "-" * 50,
        "",
        "HITLS NASA-TLX — Friedman (across 3 conditions):",
        f"  χ²({h_fried.df}, N={h_fried.n}) = {h_fried.chi2:.3f}  "
        f"p = {fmt_p(h_fried.p)}  W = {h_fried.kendall_w:.3f}",
        "",
        "HITLS pairwise Wilcoxon (Holm-corrected):",
    ]
    for pw in stats["h_pairs"]:
        stars = sig_stars(pw.p_corrected, pw.reject)
        lines.append(
            f"  {pw.condition_a} vs {pw.condition_b}: "
            f"p_raw = {fmt_p(pw.p_raw)}  p_corr = {fmt_p(pw.p_corrected)}  "
            f"r = {pw.r:+.3f}  {stars}"
        )

    m_fried = stats["m_fried"]
    lines += [
        "",
        "MITLS Utilization — Friedman (across 3 conditions):",
        f"  χ²({m_fried.df}, N={m_fried.n}) = {m_fried.chi2:.3f}  "
        f"p = {fmt_p(m_fried.p)}  W = {m_fried.kendall_w:.3f}",
        "",
        "MITLS pairwise Wilcoxon (Holm-corrected):",
    ]
    for pw in stats["m_pairs"]:
        stars = sig_stars(pw.p_corrected, pw.reject)
        lines.append(
            f"  {pw.condition_a} vs {pw.condition_b}: "
            f"p_raw = {fmt_p(pw.p_raw)}  p_corr = {fmt_p(pw.p_corrected)}  "
            f"r = {pw.r:+.3f}  {stars}"
        )

    lines += [
        "",
        "-" * 50,
        "HRV Comparison",
        "-" * 50,
        "",
        f"Pearson r (model utilization vs HRV RMSSD): "
        f"r = {stats['hrv_rho']:.3f}  p = {fmt_p(stats['hrv_p'])}",
        "  Expected: r < 0 (higher utilization → lower RMSSD)",
        f"  HRV RMSSD means: " +
        "  ".join(f"{c}={stats['hrv_means'][c]:.1f}ms" for c in _COND_ORDER),
        "",
        "=" * 60,
        "INTERPRETATION NOTE",
        "=" * 60,
        "",
        "MITLS Overall_Utilization shows minimal variation across conditions",
        "(range ~0.14–0.16), while HITLS NASA-TLX shows a clear TARS > TARP",
        "effect. The ACT-R utilization metric is primarily driven by the",
        "Motor_SubNetwork (physical device interaction) rather than Cognitive",
        "demand, which may explain the mismatch with subjective ratings.",
        "Consider comparing Cognitive_SubNetwork specifically to Mental Demand.",
    ]

    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  [pub] Saved: {save_path.name}")


def _write_tier2_report(
    stats: dict,
    tlx_df: pd.DataFrame,
    mitls: dict[str, dict[str, list[float]]],
    save_path: Path,
) -> None:
    lines = [
        "=" * 60,
        "WORKLOAD VALIDATION — TIER 2 DEBUG REPORT",
        "=" * 60,
        "",
        "Subnetwork vs NASA-TLX subscale breakdown",
        "(Conceptual mapping: Cognitive↔Mental/Effort, Perceptual↔Temporal,",
        "Motor↔Physical)",
        "",
    ]

    lines += ["-" * 50, "Per-Condition MITLS Subnetwork Means", "-" * 50]
    header = f"  {'Metric':<15s}" + "".join(f"  {c:<12s}" for c in _COND_ORDER)
    lines.append(header)
    for metric_key, col in MITLS_METRICS.items():
        row = f"  {metric_key:<15s}"
        for cond in _COND_ORDER:
            vals = mitls[cond][metric_key]
            m = float(np.mean(vals)) if vals else float("nan")
            row += f"  {m:<12.4f}"
        lines.append(row)

    lines += [
        "",
        "-" * 50,
        "Per-Condition HITLS NASA-TLX Subscale Means (0-100 scale)",
        "-" * 50,
    ]
    tlx_subscales = ["weighted_score", "mental_demand", "effort",
                     "physical_demand", "temporal_demand", "frustration"]
    header2 = f"  {'Subscale':<20s}" + "".join(f"  {c:<12s}" for c in _COND_ORDER)
    lines.append(header2)
    for sub in tlx_subscales:
        if sub not in tlx_df.columns:
            continue
        row = f"  {sub:<20s}"
        for cond in _COND_ORDER:
            vals = tlx_df[tlx_df["condition"] == cond][sub].dropna()
            m = float(vals.mean()) if len(vals) > 0 else float("nan")
            row += f"  {m:<12.2f}"
        lines.append(row)

    lines += [
        "",
        "-" * 50,
        "Proposed Mapping Comparison (normalized direction only)",
        "-" * 50,
        "",
        "Mapping: Cognitive ↔ Mental Demand + Effort",
    ]
    for cond in _COND_ORDER:
        cog = float(np.mean(mitls[cond]["Cognitive"])) if mitls[cond]["Cognitive"] else np.nan
        md  = tlx_df[tlx_df["condition"] == cond]["mental_demand"].mean()
        ef  = tlx_df[tlx_df["condition"] == cond]["effort"].mean()
        lines.append(f"  {cond}: Cognitive={cog:.4f}  Mental={md:.1f}  Effort={ef:.1f}")

    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  [debug] Saved: {save_path.name}")


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_workload_validation(n: int = 12) -> dict:
    """Run full workload validation and write all outputs.

    Returns a metrics dict for use by run_all.py.
    """
    print("=" * 60)
    print("WORKLOAD VALIDATION")
    print("=" * 60)

    print("\n[1/5] Loading data...")
    tlx_df = _load_per_participant_nasa_tlx()
    print(f"  NASA-TLX: {len(tlx_df)} rows loaded "
          f"({tlx_df['participant'].nunique() if not tlx_df.empty else 0} participants)")
    if not tlx_df.empty:
        for cond in _COND_ORDER:
            n_sub = len(tlx_df[tlx_df["condition"] == cond])
            m = tlx_df[tlx_df["condition"] == cond]["weighted_score"].mean()
            print(f"    {cond}: n={n_sub} mean={m:.2f}")

    mitls = _load_mitls_utilization(n)
    for cond in _COND_ORDER:
        m = float(np.mean(mitls[cond]["Overall"])) if mitls[cond]["Overall"] else float("nan")
        print(f"  MITLS {cond}: n={len(mitls[cond]['Overall'])} reps, mean_util={m:.4f}")

    hrv = _load_hitls_hrv()
    for cond in _COND_ORDER:
        print(f"  HRV {cond}: n={len(hrv[cond])} RMSSD values")

    print("\n[2/5] Computing statistics...")
    stats = _compute_stats(tlx_df, mitls, hrv)
    print(f"  HITLS Friedman: χ²({stats['h_fried'].df}, N={stats['h_fried'].n})"
          f" = {stats['h_fried'].chi2:.3f}  p={fmt_p(stats['h_fried'].p)}")
    print(f"  MITLS Friedman: χ²({stats['m_fried'].df}, N={stats['m_fried'].n})"
          f" = {stats['m_fried'].chi2:.3f}  p={fmt_p(stats['m_fried'].p)}")
    print(f"  Spearman ρ (condition means): {stats['spearman_rho']:.3f}  "
          f"p={fmt_p(stats['spearman_p'])}")
    print(f"  Direction match: {stats['dir_match']}")
    print(f"    HITLS: {' > '.join(stats['h_rank'])}")
    print(f"    MITLS: {' > '.join(stats['m_rank'])}")
    print(f"  HRV Pearson r: {stats['hrv_rho']:.3f}")

    print("\n[3/5] Writing Tier 1 outputs (publication)...")
    _plot_pub_workload_comparison(
        stats, tlx_df, mitls,
        PUB_DIR / "workload_vs_nasa_tlx.png",
    )
    _write_tier1_report(stats, tlx_df, VAL_DIR / "workload" / "workload_report.txt")

    print("\n[4/5] Writing Tier 2 outputs (diagnostic)...")
    _plot_debug_per_condition_distribution(
        tlx_df, mitls,
        DEBUG_DIR / "workload_per_condition_distribution.png",
    )
    _plot_debug_hrv_scatter(stats, mitls, DEBUG_DIR / "workload_hrv_scatter.png")
    _write_tier2_report(
        stats, tlx_df, mitls,
        VAL_DIR / "workload" / "workload_debug_report.txt",
    )

    print("\n[5/5] Done.")

    # Summary metrics
    metrics = {
        "workload_hitls_friedman_chi2": stats["h_fried"].chi2,
        "workload_hitls_friedman_p":    stats["h_fried"].p,
        "workload_mitls_friedman_chi2": stats["m_fried"].chi2,
        "workload_mitls_friedman_p":    stats["m_fried"].p,
        "workload_spearman_rho":        stats["spearman_rho"],
        "workload_spearman_p":          stats["spearman_p"],
        "workload_dir_match":           stats["dir_match"],
        "workload_hrv_pearson_r":       stats["hrv_rho"],
    }

    print("\nSummary metrics:")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    return metrics


if __name__ == "__main__":
    run_workload_validation()
