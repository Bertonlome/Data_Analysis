#!/usr/bin/env python3
"""
Step 6: Eye Movement / AoI Validation
======================================
Compare MITLS simulated gaze (AoI proportions) to HITLS empirical eye-tracking
data collected across 16–18 participants per condition.

AoI alignment
-------------
  HITLS         MITLS (mapped)
  ──────────    ──────────────────────────
  TARS          TARS
  PFD           PFD
  ND            ND  +  E_W_CAS   (merged)
  pedestal      Central_Console
  Outside_Window Outside_Window
    no_intersection Other             (count==0 / no world-object intersection)

Data sources
------------
  HITLS : HITLS/P{pid}/cleaned/P{pid}_{condition}_aoi_summary.csv
  MITLS : per-run eye_movement/results_eye_movement.txt
          (parsed by mitls_loader.load_aoi_metrics)

Tier 1 outputs (publication)
  plots/pub/aoi_proportions_comparison.png
  eye_movement/aoi_report.txt

Tier 2 outputs (diagnostic)
  plots/debug/aoi_stacked_bar.png
  plots/debug/aoi_model_vs_human_scatter.png
  eye_movement/aoi_debug_report.txt
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from scipy.special import rel_entr  # for KL divergence

_HERE = Path(__file__).resolve()
WORKSPACE_ROOT = _HERE.parents[2]
sys.path.insert(0, str(WORKSPACE_ROOT))

from model_validation.shared.hitls_loader import HITLS_DIR, SHARED_CONDITIONS
from model_validation.shared.mitls_loader import (
    MITLS_TO_HITLS, iter_condition_runs, load_aoi_metrics,
)
from model_validation.shared.stats import nmae, fmt_p

VAL_DIR   = WORKSPACE_ROOT / "model_validation"
PUB_DIR   = VAL_DIR / "plots" / "pub"
DEBUG_DIR = VAL_DIR / "plots" / "debug"
PUB_DIR.mkdir(parents=True, exist_ok=True)
DEBUG_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# AoI definitions
# ---------------------------------------------------------------------------

# HITLS AoI name → list of MITLS AoI names to sum
AOI_MAP: dict[str, list[str]] = {
    "TARS":           ["TARS"],
    "PFD":            ["PFD"],
    "ND":             ["ND", "E_W_CAS"],   # HITLS ND includes E_W_CAS region
    "pedestal":       ["Central_Console"],
    "Outside_Window": ["Outside_Window"],
    "no_intersection": ["Other"],
}

COMPARISON_AOIS = list(AOI_MAP.keys())   # 6 AoIs

_COND_ORDER = SHARED_CONDITIONS          # TARS, TARP-S, TARP-F
_COND_COLORS = {
    "TARS":   "#4878CF",
    "TARP-S": "#6ACC65",
    "TARP-F": "#D65F5F",
}

# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def _find_participants() -> list[Path]:
    return sorted(
        e for e in HITLS_DIR.iterdir()
        if e.is_dir() and e.name.startswith("P") and e.name[1:].isdigit()
    )


def _load_hitls_aoi() -> dict[str, dict[str, list[float]]]:
    """Load per-participant AoI percentages from cleaned summary CSVs.

    Returns
    -------
    dict: condition → aoi_name → list[float]  (% values across participants)
    """
    out: dict[str, dict[str, list[float]]] = {
        cond: {aoi: [] for aoi in COMPARISON_AOIS}
        for cond in _COND_ORDER
    }
    for pid_dir in _find_participants():
        pid = pid_dir.name
        for cond in _COND_ORDER:
            fp = pid_dir / "cleaned" / f"{pid}_{cond}_aoi_summary.csv"
            if not fp.exists():
                continue
            df = pd.read_csv(fp)
            aoi_dict = dict(zip(df["AoI_Name"], df["Percentage"]))
            for aoi in COMPARISON_AOIS:
                val = aoi_dict.get(aoi)
                if val is not None:
                    out[cond][aoi].append(float(val))
    return out


def _load_mitls_aoi(n: int = 12) -> dict[str, dict[str, list[float]]]:
    """Load per-rep AoI percentages from MITLS model outputs.

    Applies AOI_MAP: MITLS AoIs are merged to match HITLS categories.

    Returns
    -------
    dict: hitls_condition → aoi_name → list[float]  (% across reps)
    """
    out: dict[str, dict[str, list[float]]] = {
        cond: {aoi: [] for aoi in COMPARISON_AOIS}
        for cond in _COND_ORDER
    }
    for mitls_cond in ["C1", "C2", "C3"]:
        hitls_cond = MITLS_TO_HITLS[mitls_cond]
        runs = list(iter_condition_runs(mitls_cond))[:n]
        for run_dir in runs:
            try:
                df = load_aoi_metrics(run_dir)
            except FileNotFoundError:
                continue
            mitls_dict: dict[str, float] = {}
            for _, row in df.iterrows():
                mitls_dict[row["AoI_Name"]] = float(row["Percentage"])

            for hitls_aoi, mitls_aois in AOI_MAP.items():
                mapped_val = sum(mitls_dict.get(m, 0.0) for m in mitls_aois)
                out[hitls_cond][hitls_aoi].append(mapped_val)
    return out


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def _kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """KL divergence D_KL(p || q) with epsilon smoothing."""
    eps = 1e-9
    p_ = np.maximum(p, eps); p_ /= p_.sum()
    q_ = np.maximum(q, eps); q_.max(); q_ /= q_.sum()
    return float(np.sum(rel_entr(p_, q_)))


def _compute_stats(
    hitls: dict[str, dict[str, list[float]]],
    mitls: dict[str, dict[str, list[float]]],
) -> dict:
    """Compute per-condition per-AoI NMAE plus distribution-level KL divergence.

    Returns
    -------
    dict: condition → {
        'nmae': {aoi: float},
        'h_means': {aoi: float},
        'h_sds': {aoi: float},
        'm_means': {aoi: float},
        'm_sds': {aoi: float},
        'h_ns': {aoi: int},
        'm_ns': {aoi: int},
        'kl_div': float,   # model vs human distribution similarity
    }
    """
    results = {}
    for cond in _COND_ORDER:
        h = hitls[cond]
        m = mitls[cond]

        h_means = {a: float(np.mean(h[a])) if h[a] else float("nan")
                   for a in COMPARISON_AOIS}
        h_sds   = {a: float(np.std(h[a], ddof=1)) if len(h[a]) > 1 else 0.0
                   for a in COMPARISON_AOIS}
        m_means = {a: float(np.mean(m[a])) if m[a] else float("nan")
                   for a in COMPARISON_AOIS}
        m_sds   = {a: float(np.std(m[a], ddof=1)) if len(m[a]) > 1 else 0.0
                   for a in COMPARISON_AOIS}

        nmae_vals = {}
        for aoi in COMPARISON_AOIS:
            if h_means[aoi] != 0 and not np.isnan(h_means[aoi]) \
                    and not np.isnan(m_means[aoi]):
                nmae_vals[aoi] = float(nmae(m_means[aoi], h_means[aoi]))
            else:
                nmae_vals[aoi] = float("nan")

        # KL divergence of full distributions (model mean dist vs human mean dist)
        h_dist = np.array([h_means[a] for a in COMPARISON_AOIS])
        m_dist = np.array([m_means[a] for a in COMPARISON_AOIS])
        kl = _kl_divergence(m_dist, h_dist) if not np.any(np.isnan(h_dist)) \
             and not np.any(np.isnan(m_dist)) else float("nan")

        results[cond] = {
            "nmae":    nmae_vals,
            "h_means": h_means,
            "h_sds":   h_sds,
            "m_means": m_means,
            "m_sds":   m_sds,
            "h_ns":    {a: len(h[a]) for a in COMPARISON_AOIS},
            "m_ns":    {a: len(m[a]) for a in COMPARISON_AOIS},
            "kl_div":  kl,
        }
    return results


# ---------------------------------------------------------------------------
# Tier 1 — publication figure
# ---------------------------------------------------------------------------

def _plot_pub_grouped_bars(
    stats: dict,
    save_path: Path,
) -> None:
    """Per-condition grouped bar chart: human mean±SD vs model mean±SD per AoI."""
    n_aois  = len(COMPARISON_AOIS)
    n_conds = len(_COND_ORDER)
    fig, axes = plt.subplots(1, n_conds, figsize=(14, 5), sharey=False)
    fig.suptitle(
        "AoI Proportions: Model vs. Human Participants",
        fontsize=13, fontweight="bold",
    )

    x = np.arange(n_aois)
    width = 0.35

    for ax, cond in zip(axes, _COND_ORDER):
        st  = stats[cond]
        h_m = [st["h_means"][a] for a in COMPARISON_AOIS]
        h_e = [st["h_sds"][a]   for a in COMPARISON_AOIS]
        m_m = [st["m_means"][a] for a in COMPARISON_AOIS]
        m_e = [st["m_sds"][a]   for a in COMPARISON_AOIS]

        ax.bar(x - width / 2, h_m, width, label="Human (HITLS)",
               color="#4878CF", alpha=0.75, yerr=h_e,
               error_kw=dict(elinewidth=1, capsize=3, ecolor="#1F3A6E"))
        ax.bar(x + width / 2, m_m, width, label="Model (MITLS)",
               color="#D65F5F", alpha=0.75, yerr=m_e,
               error_kw=dict(elinewidth=1, capsize=3, ecolor="#8B0000"))

        ax.set_xticks(x)
        ax.set_xticklabels(
            [a.replace("_", "\n") for a in COMPARISON_AOIS],
            fontsize=8,
        )
        ax.set_ylabel("Fixation proportion (%)", fontsize=9)
        ax.set_title(
            f"{cond}\n(KL div = {st['kl_div']:.2f})",
            fontsize=10,
        )
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.set_ylim(0, 100)
        if ax is axes[0]:
            ax.legend(fontsize=8, loc="upper right")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [pub] Saved: {save_path.name}")


# ---------------------------------------------------------------------------
# Tier 2 — debug figures
# ---------------------------------------------------------------------------

def _plot_debug_stacked_bar(
    stats: dict,
    save_path: Path,
) -> None:
    """Stacked bar chart: model vs human mean AoI distribution per condition."""
    aoi_colors = {
        "TARS":           "#E07B39",
        "PFD":            "#4878CF",
        "ND":             "#6ACC65",
        "pedestal":       "#8B4513",
        "Outside_Window": "#888888",
        "no_intersection": "#B07AA1",
    }

    n_conds = len(_COND_ORDER)
    fig, axes = plt.subplots(1, n_conds, figsize=(12, 5))
    fig.suptitle(
        "AoI Distribution: Model vs. Human Mean (Stacked)",
        fontsize=12, fontweight="bold",
    )

    for ax, cond in zip(axes, _COND_ORDER):
        st    = stats[cond]
        x_pos = [0, 1]  # human at 0, model at 1
        bottom_h = 0.0
        bottom_m = 0.0

        for aoi in COMPARISON_AOIS:
            h_val = st["h_means"][aoi]
            m_val = st["m_means"][aoi]
            color = aoi_colors.get(aoi, "#CCCCCC")
            ax.bar(0, h_val, bottom=bottom_h, color=color, alpha=0.85,
                   edgecolor="white", linewidth=0.5)
            ax.bar(1, m_val, bottom=bottom_m, color=color, alpha=0.85,
                   edgecolor="white", linewidth=0.5)
            bottom_h += h_val
            bottom_m += m_val

        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Human", "Model"], fontsize=10)
        ax.set_title(cond, fontsize=11)
        ax.set_ylabel("Fixation proportion (%)" if ax is axes[0] else "")
        ax.set_ylim(0, 115)
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        ax.text(
            0.5, 108,
            f"KL div = {st['kl_div']:.2f}",
            ha="center", fontsize=8, color="darkred",
            transform=ax.get_xaxis_transform(),
        )

    # Legend
    legend_patches = [
        mpatches.Patch(color=aoi_colors[a], alpha=0.85, label=a)
        for a in COMPARISON_AOIS
    ]
    fig.legend(
        handles=legend_patches, loc="lower center",
        ncol=5, fontsize=9, bbox_to_anchor=(0.5, -0.06),
    )

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [debug] Saved: {save_path.name}")


def _plot_debug_scatter(
    stats: dict,
    save_path: Path,
) -> None:
    """Scatter: model mean % vs human mean % per AoI, colored by condition.
    Diagonal = perfect agreement.
    """
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 100], [0, 100], "k--", lw=1, alpha=0.5, label="Perfect agreement")

    aoi_markers = {
        "TARS":           "o",
        "PFD":            "s",
        "ND":             "^",
        "pedestal":       "D",
        "Outside_Window": "P",
        "no_intersection": "X",
    }

    for cond in _COND_ORDER:
        st    = stats[cond]
        color = _COND_COLORS[cond]
        for aoi in COMPARISON_AOIS:
            h_m = st["h_means"][aoi]
            m_m = st["m_means"][aoi]
            if np.isnan(h_m) or np.isnan(m_m):
                continue
            marker = aoi_markers.get(aoi, "o")
            ax.scatter(
                h_m, m_m,
                color=color, marker=marker, s=80, alpha=0.85, zorder=3,
            )
            # Error bars (human SD in x, model SD in y)
            ax.errorbar(
                h_m, m_m,
                xerr=st["h_sds"][aoi], yerr=st["m_sds"][aoi],
                fmt="none", color=color, alpha=0.3, lw=1,
            )

    # Condition legend
    cond_patches = [
        mpatches.Patch(color=_COND_COLORS[c], label=c) for c in _COND_ORDER
    ]
    # AoI marker legend
    aoi_lines = [
        plt.Line2D([0], [0], marker=aoi_markers[a], color="grey",
                   linestyle="none", markersize=7, label=a)
        for a in COMPARISON_AOIS
    ]
    leg1 = ax.legend(handles=cond_patches, loc="upper left", fontsize=9,
                     title="Condition")
    ax.add_artist(leg1)
    ax.legend(handles=aoi_lines, loc="lower right", fontsize=9, title="AoI")

    ax.set_xlabel("Human mean fixation proportion (%)", fontsize=10)
    ax.set_ylabel("Model mean fixation proportion (%)", fontsize=10)
    ax.set_title(
        "Model vs. Human AoI Proportions\n(per AoI × per condition)",
        fontsize=11,
    )
    ax.set_xlim(0, 80)
    ax.set_ylim(0, 100)
    ax.grid(linestyle="--", alpha=0.3)

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
        "EYE MOVEMENT / AoI VALIDATION — TIER 1 REPORT",
        "=" * 60,
        "",
        "Scope: BEFORE TAKEOFF (8/9 tasks) + LINE-UP AND HOLD (2/2 tasks)",
        "HITLS source: P{pid}/cleaned/P{pid}_{cond}_aoi_summary.csv",
        "MITLS source: per-run eye_movement/results_eye_movement.txt",
        "",
        "AoI mapping:",
        "  HITLS ND      ← MITLS ND + E_W_CAS (merged)",
        "  HITLS pedestal ← MITLS Central_Console",
        "  HITLS no_intersection (count=0) ← MITLS Other",
        "",
    ]

    for cond in _COND_ORDER:
        st = stats[cond]
        lines += [
            "-" * 50,
            f"Condition: {cond}",
            f"  KL divergence (model || human) = {st['kl_div']:.4f}",
            "",
            f"  {'AoI':<18s}  {'HITLS mean':>10s}  {'MITLS mean':>10s}  "
            f"{'HITLS SD':>9s}  {'MITLS SD':>9s}  {'NMAE':>8s}",
        ]
        for aoi in COMPARISON_AOIS:
            h_m = st["h_means"][aoi]
            m_m = st["m_means"][aoi]
            h_s = st["h_sds"][aoi]
            m_s = st["m_sds"][aoi]
            nv  = st["nmae"][aoi]
            flag = " ← HIGH" if not np.isnan(nv) and abs(nv) > 0.5 else ""
            lines.append(
                f"  {aoi:<18s}  {h_m:>9.2f}%  {m_m:>9.2f}%  "
                f"{h_s:>8.2f}%  {m_s:>8.2f}%  {nv:>+7.1%}{flag}"
            )
        lines.append("")

    lines += [
        "=" * 60,
        "INTERPRETATION NOTE",
        "=" * 60,
        "",
        "The model shows a strong bias towards TARS fixation (~75-85%)",
        "compared to human pilots (~10-45% depending on condition).",
        "Human pilots spend significant time looking at PFD (~30-40%)",
        "and Outside_Window (~10-45%) — ambient monitoring behaviour not",
        "currently modelled in MITLS. The ND/E_W_CAS comparison is also",
        "affected because MITLS treats it as task-driven (checklist cross-",
        "check) while humans use it for continuous navigation monitoring.",
        "",
        "This analysis confirms that MITLS currently models only",
        "task-execution gaze (TARS interaction) and lacks ambient",
        "monitoring productions. Adding scan-pattern productions for",
        "PFD and Out-of-Window in between checklist items is recommended.",
    ]

    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  [pub] Saved: {save_path.name}")


def _write_tier2_report(stats: dict, save_path: Path) -> None:
    lines = [
        "=" * 60,
        "EYE MOVEMENT / AoI VALIDATION — TIER 2 DEBUG REPORT",
        "=" * 60,
        "",
        "Full per-AoI breakdown with participant-level counts.",
        "",
    ]

    lines += [
        "-" * 50,
        "Sample Sizes",
        "-" * 50,
    ]
    for cond in _COND_ORDER:
        st = stats[cond]
        lines.append(f"\n{cond}:")
        for aoi in COMPARISON_AOIS:
            lines.append(
                f"  {aoi:<18s}  HITLS n={st['h_ns'][aoi]:2d}  "
                f"MITLS n_reps={st['m_ns'][aoi]:2d}"
            )

    lines += [
        "",
        "-" * 50,
        "NMAE Summary (all conditions)",
        "-" * 50,
        f"  {'AoI':<18s}  " + "  ".join(f"{c:<10s}" for c in _COND_ORDER),
    ]
    for aoi in COMPARISON_AOIS:
        row_parts = [f"  {aoi:<18s}"]
        for cond in _COND_ORDER:
            nv = stats[cond]["nmae"][aoi]
            flag = "*" if not np.isnan(nv) and abs(nv) > 0.5 else " "
            row_parts.append(f"  {nv:>+7.1%}{flag}   ")
        lines.append("".join(row_parts))

    lines += [
        "",
        "-" * 50,
        "KL Divergence (model distribution vs. human mean distribution)",
        "-" * 50,
    ]
    for cond in _COND_ORDER:
        kl = stats[cond]["kl_div"]
        lines.append(f"  {cond}: {kl:.4f}")

    lines += [
        "",
        "* = |NMAE| > 0.5 (model deviates >50% from human mean)",
        "",
        "Note: TARS aoi NMAE is typically large and positive (model over-estimates",
        "TARS fixation). PFD and Outside_Window NMAE are large and negative",
        "(model under-estimates these). This pattern is expected given that MITLS",
        "does not model ambient monitoring gaze between checklist tasks.",
    ]

    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  [debug] Saved: {save_path.name}")


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_aoi_validation(n: int = 12) -> dict:
    """Run full AoI validation and write all outputs.

    Returns a metrics dict for use by run_all.py.
    """
    print("=" * 60)
    print("EYE MOVEMENT / AoI VALIDATION")
    print("=" * 60)

    print("\n[1/5] Loading data...")
    hitls = _load_hitls_aoi()
    mitls = _load_mitls_aoi(n)

    for cond in _COND_ORDER:
        print(f"  {cond}: HITLS TARS n={len(hitls[cond]['TARS'])}, "
              f"MITLS n_reps={len(mitls[cond]['TARS'])}")

    print("\n[2/5] Computing statistics...")
    stats = _compute_stats(hitls, mitls)
    for cond in _COND_ORDER:
        st = stats[cond]
        print(f"  {cond}:")
        print(f"    KL divergence = {st['kl_div']:.4f}")
        for aoi in COMPARISON_AOIS:
            nv = st["nmae"][aoi]
            print(f"    NMAE {aoi:<18s}: {nv:+.1%}")

    print("\n[3/5] Writing Tier 1 outputs (publication)...")
    _plot_pub_grouped_bars(stats, PUB_DIR / "aoi_proportions_comparison.png")
    _write_tier1_report(stats, VAL_DIR / "eye_movement" / "aoi_report.txt")

    print("\n[4/5] Writing Tier 2 outputs (diagnostic)...")
    _plot_debug_stacked_bar(stats, DEBUG_DIR / "aoi_stacked_bar.png")
    _plot_debug_scatter(stats, DEBUG_DIR / "aoi_model_vs_human_scatter.png")
    _write_tier2_report(stats, VAL_DIR / "eye_movement" / "aoi_debug_report.txt")

    print("\n[5/5] Done.")

    # Summary metrics dict
    metrics: dict = {}
    for cond in _COND_ORDER:
        st = stats[cond]
        metrics[f"aoi_kl_div_{cond}"] = st["kl_div"]
        for aoi in COMPARISON_AOIS:
            key = f"aoi_nmae_{cond}_{aoi.replace(' ', '_')}"
            metrics[key] = st["nmae"][aoi]

    print("\nSummary metrics:")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    return metrics


if __name__ == "__main__":
    run_aoi_validation()
