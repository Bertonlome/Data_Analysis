#!/usr/bin/env python3
"""
compare_navigate.py — Cross-participant Navigate Performance Comparison
=======================================================================
Mirrors compare_aviate.py for navigation-performance data:

  1. Ensures navigate_perf reports exist for every participant
     (regenerates if missing).
  2. Loads all navigate_perf JSON summaries.
  3. Generates three figures:

       navigate_boxplots.png
         Box-plot grid — RMSE, nMAE, RMSE/MAE ratio, window duration
         for each metric (climb XTE · ATD · Heading · level-off ALT).

       navigate_rmse_distributions.png
         Histogram + KDE + mean ◆ + 95% CI strip per condition,
         one subplot per metric (raw RMSE values).

       navigate_nmae_distributions.png
         Same layout for nMAE (MAE / window_duration) — the
         window-duration-normalised metric directly comparable
         across conditions regardless of scenario length.

Run from the repo root:
    python HITLS/performance/compare_navigate.py
"""

import os
import sys
import json
import subprocess

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde

# ── Paths ─────────────────────────────────────────────────────────────────────
PERF_DIR         = os.path.dirname(os.path.abspath(__file__))
HITLS_DIR        = os.path.dirname(PERF_DIR)
NAVIGATE_SCRIPT  = os.path.join(PERF_DIR, "navigate_perf.py")
PLOTS_DIR        = os.path.join(HITLS_DIR, "plots")
PYTHON           = sys.executable

# ── Conditions ────────────────────────────────────────────────────────────────
CONDITIONS = ["TARS", "TARC", "TARP-S", "TARP-F"]

COND_COLOR = {
    "TARS":   "#4472C4",
    "TARC":   "#ED7D31",
    "TARP-S": "#70AD47",
    "TARP-F": "#C00000",
}

# ── Visual parameters ─────────────────────────────────────────────────────────
FIGURE_DPI         = 150

# Box plots
BOX_WIDTH          = 0.5
BOX_JITTER         = 0.15
BOX_DOT_SIZE       = 20
BOX_ALPHA          = 0.65
BOX_HEIGHT_PER_ROW = 3.5
BOX_FIGW           = 12

# Distribution plots (histogram + KDE)
DIST_FIGW          = 16
DIST_FIGH          = 5.0
DIST_HIST_BINS     = 10
DIST_HIST_ALPHA    = 0.20
DIST_KDE_LW        = 2.0
DIST_CI_STRIP_FRAC = 0.28   # fraction of plot height reserved for CI strip

# Typography
FONT_SUPTITLE   = 11
FONT_TITLE      = 10
FONT_LABEL      = 9
FONT_TICK       = 8
FONT_TICK_SM    = 7
FONT_ANNOTATION = 6


# ═══════════════════════════════════════════════════════════════════════════════
#  I/O helpers
# ═══════════════════════════════════════════════════════════════════════════════

def find_participants():
    return [
        e for e in sorted(os.listdir(HITLS_DIR))
        if os.path.isdir(os.path.join(HITLS_DIR, e))
        and e.startswith("P") and e[1:].isdigit()
    ]


def _report_path(pid):
    return os.path.join(HITLS_DIR, pid, "cleaned", f"{pid}_navigate_perf_report.txt")


def load_json(path):
    try:
        txt   = open(path, encoding="utf-8").read()
        start = txt.index("{")
        end   = txt.index("--- END SUMMARY ---")
        return json.loads(txt[start:end].strip())
    except (FileNotFoundError, ValueError, json.JSONDecodeError):
        return None


def has_valid_report(pid):
    """True if report exists and contains the navigate_perf nmae field."""
    data = load_json(_report_path(pid))
    if not data:
        return False
    for cond_data in data.get("conditions", {}).values():
        return "nmae" in cond_data.get("climb", {}).get("xte", {})
    return False


def run_navigate_perf(pid, participant_number):
    print(f"  Generating navigate_perf report for {pid} …", flush=True)
    proc = subprocess.run(
        [PYTHON, NAVIGATE_SCRIPT],
        input=f"{participant_number}\n",
        text=True,
        capture_output=True,
    )
    if proc.returncode != 0:
        print(f"  ⚠  navigate_perf.py returned code {proc.returncode} for {pid}")
        if proc.stderr:
            print(proc.stderr[:400])


def load_all(participants):
    data = {}
    for pid in participants:
        d = load_json(_report_path(pid))
        if d is not None:
            data[pid] = d
    return data


def save_fig(fig, filename):
    os.makedirs(PLOTS_DIR, exist_ok=True)
    path = os.path.join(PLOTS_DIR, filename)
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    print(f"  Saved → {path}")


def _get(d, *keys):
    for k in keys:
        if not isinstance(d, dict):
            return None
        d = d.get(k)
    return d


def _collect(all_data, participants, extract_fn):
    result = {c: [] for c in CONDITIONS}
    for pid in participants:
        pdata = all_data.get(pid, {})
        for cond in CONDITIONS:
            v = extract_fn(pdata, cond)
            if v is not None:
                result[cond].append(float(v))
    return result


# ═══════════════════════════════════════════════════════════════════════════════
#  Box-plot grid
# ═══════════════════════════════════════════════════════════════════════════════

def _box_panel(ax, data_by_cond, title, ylabel, floor_zero=True):
    present   = [c for c in CONDITIONS if data_by_cond.get(c)]
    positions = list(range(len(present)))
    boxes     = [data_by_cond.get(c, []) for c in present]

    bp = ax.boxplot(
        boxes, positions=positions, widths=BOX_WIDTH, patch_artist=True,
        medianprops=dict(color="black", linewidth=2),
        flierprops=dict(marker="o", markersize=4, alpha=0.5),
    )
    for patch, cond in zip(bp["boxes"], present):
        patch.set_facecolor(COND_COLOR[cond])
        patch.set_alpha(BOX_ALPHA)

    rng = np.random.default_rng(42)
    for i, (cond, vals) in enumerate(zip(present, boxes)):
        if vals:
            jitter = rng.uniform(-BOX_JITTER, BOX_JITTER, len(vals))
            ax.scatter(
                np.array([i] * len(vals)) + jitter, vals,
                color="black", s=BOX_DOT_SIZE, zorder=5, alpha=0.75,
            )

    ax.set_xticks(positions)
    ax.set_xticklabels(present, fontsize=FONT_TICK)
    ax.set_title(title, fontsize=FONT_TITLE, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=FONT_TICK)
    ax.grid(axis="y", linestyle="--", alpha=0.45)
    if floor_zero:
        ax.set_ylim(bottom=0)


def plot_boxplots(all_data, participants):
    """Box-plot grid for all navigate performance metrics."""

    def g(*path):
        return lambda d, c: _get(d, "conditions", c, *path)

    # (title, ylabel, floor_zero, extract_fn)
    metrics = [
        ("Climb XTE RMSE",                  "NM",    True,  g("climb",   "xte",           "rmse")),
        ("Climb XTE nMAE  (MAE / window)",   "NM/s",  True,  g("climb",   "xte",           "nmae")),
        ("Climb ATD RMSE",                  "NM",    True,  g("climb",   "atd_error",      "rmse")),
        ("Climb ATD nMAE  (MAE / window)",   "NM/s",  True,  g("climb",   "atd_error",      "nmae")),
        ("Climb Heading RMSE",              "deg",   True,  g("climb",   "heading_error",  "rmse")),
        ("Climb Heading nMAE (MAE / window)","deg/s", True,  g("climb",   "heading_error",  "nmae")),
        ("Level-off Alt RMSE",              "ft",    True,  g("leveloff","alt_error",      "rmse")),
        ("Level-off Alt nMAE (MAE / window)","ft/s",  True,  g("leveloff","alt_error",      "nmae")),
        ("Climb XTE RMSE / MAE ratio",      "",      False, g("climb",   "xte",           "rmse_mae")),
        ("Climb ATD RMSE / MAE ratio",      "",      False, g("climb",   "atd_error",      "rmse_mae")),
        ("Climb Heading RMSE / MAE ratio",  "",      False, g("climb",   "heading_error",  "rmse_mae")),
        ("Level-off Alt RMSE / MAE ratio",  "",      False, g("leveloff","alt_error",      "rmse_mae")),
        ("Climb window duration",           "s",     True,  g("climb",   "window_s")),
        ("Level-off window duration",       "s",     True,  g("leveloff","window_s")),
    ]

    ncols = 2
    nrows = (len(metrics) + 1) // 2
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(BOX_FIGW, nrows * BOX_HEIGHT_PER_ROW),
    )
    fig.suptitle(
        "Navigate Performance — Box Plots across Conditions\n"
        "(raw dots = individual participants  ·  nMAE = MAE / window_duration"
        "  ·  ratio near 1.0 = uniform errors)",
        fontsize=FONT_SUPTITLE, fontweight="bold",
    )
    axes_flat = axes.flatten()

    for i, (title, ylabel, floor_zero, fn) in enumerate(metrics):
        _box_panel(
            axes_flat[i],
            _collect(all_data, participants, fn),
            title, ylabel, floor_zero,
        )

    for j in range(len(metrics), len(axes_flat)):
        axes_flat[j].set_visible(False)

    patches = [mpatches.Patch(color=COND_COLOR[c], label=c) for c in CONDITIONS]
    fig.legend(handles=patches, loc="lower right", fontsize=FONT_TICK, title="Condition")
    plt.tight_layout()
    save_fig(fig, "navigate_boxplots.png")
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
#  KDE distribution figures  (histogram + curve + mean◆ + CI strip)
# ═══════════════════════════════════════════════════════════════════════════════

def _plot_metric_distributions(all_data, participants, metric_specs, suptitle, filename):
    """
    One figure, one subplot per metric_spec.

    metric_specs : list of (path, subplot_title, xlabel, x_max_hint)
      path        : sequence of dict keys inside data["conditions"][cond]
      x_max_hint  : upper bound for the x-axis (None = auto)
    """
    n = len(metric_specs)
    fig, axes = plt.subplots(1, n, figsize=(DIST_FIGW, DIST_FIGH), sharey=False)
    if n == 1:
        axes = [axes]
    fig.suptitle(suptitle, fontsize=FONT_SUPTITLE, fontweight="bold")

    for ax, (path, subtitle, xlabel, x_max_hint) in zip(axes, metric_specs):
        ax.set_title(subtitle, fontsize=FONT_TITLE, fontweight="bold")

        cond_vals = {}
        for cond in CONDITIONS:
            vals = []
            for pid in participants:
                v = _get(all_data.get(pid, {}), "conditions", cond, *path)
                if v is not None:
                    vals.append(float(v))
            cond_vals[cond] = vals

        all_vals = [v for vs in cond_vals.values() for v in vs]
        if not all_vals:
            ax.text(0.5, 0.5, "no data", ha="center", va="center",
                    transform=ax.transAxes, fontsize=FONT_LABEL)
            continue

        x_max = max(
            x_max_hint if x_max_hint else max(all_vals) * 1.2,
            max(all_vals) * 1.05,
        )
        x_margin = x_max * 0.04
        x_kde    = np.linspace(0, x_max, 400)

        for cond in CONDITIONS:
            vals = cond_vals[cond]
            if not vals:
                continue
            color = COND_COLOR[cond]

            ax.hist(
                vals, bins=DIST_HIST_BINS, range=(0, x_max),
                density=True, color=color, alpha=DIST_HIST_ALPHA,
                edgecolor=color, linewidth=0.5,
            )

            if len(vals) >= 2:
                try:
                    kde   = gaussian_kde(vals, bw_method="scott")
                    y_kde = kde(x_kde)
                    ax.plot(x_kde, y_kde, color=color, linewidth=DIST_KDE_LW, alpha=0.90)
                except Exception:
                    pass
            else:
                ax.axvline(vals[0], color=color, linewidth=1.5, linestyle="--", alpha=0.75)

        # ── Mean ◆ + 95% CI strip below x-axis ────────────────────────────
        y_top     = ax.get_ylim()[1]
        n_present = sum(1 for c in CONDITIONS if cond_vals.get(c))
        strip_h   = DIST_CI_STRIP_FRAC * y_top
        row_h     = strip_h / max(n_present, 1)
        ax.set_ylim(bottom=-strip_h, top=y_top)
        ax.axhline(0, color="black", linewidth=0.5, linestyle=":", alpha=0.35)

        ci_idx = 0
        for cond in CONDITIONS:
            vals = cond_vals.get(cond, [])
            if not vals:
                continue
            n_v    = len(vals)
            mean_v = float(np.mean(vals))
            sem    = float(np.std(vals, ddof=1) / np.sqrt(n_v)) if n_v > 1 else 0.0
            ci_lo  = max(0.0, mean_v - 1.96 * sem)
            ci_hi  = mean_v + 1.96 * sem
            color  = COND_COLOR[cond]
            y_pos  = -(ci_idx + 0.5) * row_h
            cap    = row_h * 0.20

            ax.axvline(mean_v, color=color, linewidth=1.0,
                       linestyle="--", alpha=0.50, zorder=3)
            ax.plot([ci_lo, ci_hi], [y_pos, y_pos],
                    color=color, linewidth=1.8, alpha=0.90, zorder=5)
            for xc in (ci_lo, ci_hi):
                ax.plot([xc, xc], [y_pos - cap, y_pos + cap],
                        color=color, linewidth=1.5, alpha=0.90, zorder=5)
            ax.scatter(mean_v, y_pos, color=color, marker="D", s=45,
                       edgecolors="black", linewidths=0.7, zorder=6)
            ax.text(-x_margin * 0.8, y_pos, cond,
                    ha="right", va="center",
                    fontsize=FONT_TICK_SM, color=color, fontweight="bold")
            ci_idx += 1

        ax.set_xlim(left=-x_margin, right=x_max)
        ax.set_xlabel(xlabel, fontsize=FONT_TICK)
        ax.set_ylabel("Density", fontsize=FONT_TICK)
        ax.set_yticks([t for t in ax.get_yticks() if t >= 0])
        ax.grid(axis="x", linestyle=":", alpha=0.20)
        ax.grid(axis="y", linestyle=":", alpha=0.35)

    cond_handles = [
        Line2D([0], [0], color=COND_COLOR[c], linewidth=2, label=c)
        for c in CONDITIONS
    ]
    extra_handles = [
        Line2D([0], [0], color="gray", marker="D", linestyle="None",
               markersize=6, markeredgecolor="black", label="Mean (◆)"),
        Line2D([0], [0], color="gray", linewidth=1.8, label="95% CI"),
    ]
    fig.legend(
        handles=cond_handles + extra_handles,
        loc="lower center", ncol=len(CONDITIONS) + 2,
        fontsize=FONT_TICK, bbox_to_anchor=(0.5, 0),
    )
    fig.tight_layout(rect=[0, 0.10, 1, 1])
    save_fig(fig, filename)
    return fig


def plot_rmse_distributions(all_data, participants):
    """RMSE distribution per condition for each navigate metric."""
    return _plot_metric_distributions(
        all_data, participants,
        metric_specs=[
            (("climb",    "xte",          "rmse"), "Climb XTE RMSE",       "RMSE (NM)",  None),
            (("climb",    "atd_error",    "rmse"), "Climb ATD RMSE",       "RMSE (NM)",  None),
            (("climb",    "heading_error","rmse"), "Climb Heading RMSE",   "RMSE (deg)", None),
            (("leveloff", "alt_error",    "rmse"), "Level-off Alt RMSE",   "RMSE (ft)",  None),
        ],
        suptitle=(
            "Navigate Performance — RMSE Distribution per Condition\n"
            "(histogram · KDE · ◆ = mean · ─── = 95% CI)"
        ),
        filename="navigate_rmse_distributions.png",
    )


def plot_nmae_distributions(all_data, participants):
    """nMAE (MAE / window_duration) distribution — comparable across conditions."""
    return _plot_metric_distributions(
        all_data, participants,
        metric_specs=[
            (("climb",    "xte",          "nmae"), "Climb XTE nMAE",       "nMAE (NM/s)",  None),
            (("climb",    "atd_error",    "nmae"), "Climb ATD nMAE",       "nMAE (NM/s)",  None),
            (("climb",    "heading_error","nmae"), "Climb Heading nMAE",   "nMAE (deg/s)", None),
            (("leveloff", "alt_error",    "nmae"), "Level-off Alt nMAE",   "nMAE (ft/s)",  None),
        ],
        suptitle=(
            "Navigate Performance — nMAE Distribution per Condition  "
            "(MAE / window duration — normalised for scenario length)\n"
            "(histogram · KDE · ◆ = mean · ─── = 95% CI)"
        ),
        filename="navigate_nmae_distributions.png",
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    sep = "=" * 70
    print(f"\n{sep}")
    print("  HITLS — Cross-participant Navigate Performance Comparison")
    print(f"{sep}")

    participants = find_participants()
    print(f"\nParticipants found: {', '.join(participants)}")

    print("\n[1/3] Checking / generating navigate_perf reports …")
    for i, pid in enumerate(participants, start=1):
        if has_valid_report(pid):
            print(f"  {pid}: ✓ report exists")
        else:
            run_navigate_perf(pid, i)

    print("\n[2/3] Loading data …")
    all_data = load_all(participants)
    loaded   = list(all_data.keys())
    print(f"  Loaded: {', '.join(loaded)}")

    if not loaded:
        print("  No data available — aborting.")
        return

    os.makedirs(PLOTS_DIR, exist_ok=True)
    print(f"\n[3/3] Generating charts → {PLOTS_DIR}/\n")

    figs = [
        plot_boxplots(all_data, loaded),
        plot_rmse_distributions(all_data, loaded),
        plot_nmae_distributions(all_data, loaded),
    ]

    n_saved = sum(1 for f in figs if f is not None)
    print(f"\nDone — {n_saved} figures saved to {PLOTS_DIR}/")
    plt.show()


if __name__ == "__main__":
    main()
