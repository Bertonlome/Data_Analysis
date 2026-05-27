#!/usr/bin/env python3
"""
HITLS — TARC Allocation Similarity Analysis
============================================
Compare-type script: operates on ALL participants at once.
Run from the repository root:
    python HITLS/allocation.py

The script will show a summary of what will be generated or overwritten and
ask for confirmation before doing any work.

Reads each participant's TARC CSV (P0x_TARC.csv or briefing_export_*.csv) and
encodes each task as one of four states:

  HUMAN     (0) — Human Role == performer
  AUTO_FAST (1) — Autonomy Role == performer, no delays
  AUTO_SLOW (2) — Autonomy Role == performer, delay before OR after action
                  (TiA > 0 OR TaEA is a positive integer; "is_acked" /
                  "is_sensed" are NOT counted as numeric delays)
  SHARED    (3) — neither sole performer (both supporter / both empty)

Feature vector per participant: one integer (0–3) per task, shape (n_tasks,).

Similarity metric: Hamming similarity = 1 − (differing positions / total)

Outputs (saved to HITLS/plots/):
  allocation_river.png              — river/branch view (tasks branch up=auto / down=human)
  allocation_task_heatmap.png       — per-task per-participant grid, grouped by category
  allocation_category_breakdown.png — stacked bar chart per task category
"""

import os, csv, glob, textwrap
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.image as mpimg
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # one level behind the script's directory, to find participant folders
IMAGE_DIR = os.path.join(ROOT_DIR, "images")  # for icons

HITLS_DIR   = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR   = os.path.join(HITLS_DIR, "plots")
REPORT_DIR  = os.path.join(HITLS_DIR, "allocation")
REPORT_PATH = os.path.join(REPORT_DIR, "allocation_report.txt")
IA_V8_PATH  = os.path.join(HITLS_DIR, "IA_V8.csv")

# ── Encoding constants ────────────────────────────────────────────────────────
HUMAN     = 0   # human is performer
AUTO_FAST = 1   # autonomy performer, no delays
AUTO_SLOW = 2   # autonomy performer, delay before OR after
SHARED    = 3   # neither sole performer (both supporter / both empty)

ALLOC_COLORS = {
    HUMAN:     "#4472C4",  # blue
    AUTO_FAST: "#C00000",  # red
    AUTO_SLOW: "#ED7D31",  # orange
    SHARED:    "#D0D0D0",  # grey
}
ALLOC_LABELS = {
    HUMAN:     "Human",
    AUTO_FAST: "Auto (no delay)",
    AUTO_SLOW: "Auto (slow)",
    SHARED:    "Shared / neither",
}


# ── File discovery ────────────────────────────────────────────────────────────

def find_participants():
    return sorted(
        e for e in os.listdir(HITLS_DIR)
        if os.path.isdir(os.path.join(HITLS_DIR, e))
        and e.startswith("P") and e[1:].isdigit()
    )


def find_tarc_csv(pid):
    pdir = os.path.join(HITLS_DIR, pid)
    for candidate in [
        os.path.join(pdir, f"{pid}_TARC.csv"),
        *glob.glob(os.path.join(pdir, "briefing_export_*.csv")),
    ]:
        if os.path.isfile(candidate):
            return candidate
    return None


def load_rows(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_template():
    """Load IA_V8.csv. Return is_choice[n_tasks]: True when TARS allocation
    was available (TARS* is not empty / red)."""
    with open(IA_V8_PATH, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    is_choice = np.array(
        [r["TARS*"].strip() not in ("", "red") for r in rows], dtype=bool
    )
    print(f"  Template: {is_choice.sum()} choice tasks, "
          f"{(~is_choice).sum()} fixed (human-only) tasks")
    return is_choice


# ── Feature encoding ──────────────────────────────────────────────────────────

def _is_delay(val: str) -> bool:
    """Return True if val is a numeric integer strictly > 0."""
    v = val.strip()
    try:
        return int(v) > 0
    except ValueError:
        return False


def encode_row(row: dict) -> int:
    """Return allocation state (HUMAN/AUTO_FAST/AUTO_SLOW/SHARED) for one task."""
    hr = row["Human Role"].strip().lower()
    ar = row["Autonomy Role"].strip().lower()

    if ar == "performer":
        has_delay = (_is_delay(row["Time to Initiate Action"])
                     or _is_delay(row["Time after Ending Action"]))
        return AUTO_SLOW if has_delay else AUTO_FAST
    elif hr == "performer":
        return HUMAN
    else:
        return SHARED


def build_feature_vector(rows: list[dict]) -> np.ndarray:
    """Return integer feature array of shape (n_tasks,)."""
    return np.array([encode_row(r) for r in rows], dtype=np.int8)


def build_task_labels(rows: list[dict]) -> list[str]:
    return [f"{r['Procedure'][:18]}\n{r['Task Object'][:15]}" for r in rows]


def build_categories(rows: list[dict]) -> list[str]:
    return [r["Category"].strip() for r in rows]


def build_procedures(rows: list[dict]) -> list[str]:
    return [r["Procedure"].strip() for r in rows]


# ── Load all participants ─────────────────────────────────────────────────────

def load_all():
    participants, vectors, ref_rows = [], [], None
    for pid in find_participants():
        path = find_tarc_csv(pid)
        if path is None:
            print(f"  ⚠  {pid}: no TARC CSV found — skipping")
            continue
        rows = load_rows(path)
        if ref_rows is None:
            ref_rows = rows
        participants.append(pid)
        vectors.append(build_feature_vector(rows))
        print(f"  {pid}: {os.path.basename(path)}  ({len(rows)} tasks)")
    return participants, np.array(vectors, dtype=np.int8), ref_rows


def _unique_ordered(seq):
    """Return unique values in first-seen order."""
    seen, out = set(), []
    for v in seq:
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out


# ── River plot helpers ────────────────────────────────────────────────────────

def _river_color(ratio: float) -> tuple:
    """RGB colour: deep-blue (ratio=1 → autonomy), grey (ratio=0.5), deep-green (ratio=0 → human).
    Saturation is proportional to distance from 0.5."""
    deep_blue  = np.array([0.086, 0.396, 0.753])   # #1565C0
    mid_grey   = np.array([0.62,  0.62,  0.62 ])   # #9E9E9E
    deep_green = np.array([0.180, 0.490, 0.196])   # #2E7D32
    if ratio >= 0.5:
        t = (ratio - 0.5) * 2.0
        c = mid_grey + t * (deep_blue - mid_grey)
    else:
        t = (0.5 - ratio) * 2.0
        c = mid_grey + t * (deep_green - mid_grey)
    return tuple(np.clip(c, 0.0, 1.0))


def build_task_objects(rows: list[dict]) -> list[str]:
    return [r["Task Object"] for r in rows]


def build_task_values(rows: list[dict]) -> list[str]:
    return [r["Value"] for r in rows]


# ── Plots ─────────────────────────────────────────────────────────────────────

# ── Shared river drawing engine ───────────────────────────────────────────────

def _draw_river(fig_title: str, out_filename: str,
                ordered_indices: list,
                cat_meta: list,
                auto_ratio: np.ndarray,
                task_objects: list,
                task_values: list,
                categories: list,
                tasks_per_row: int = 22):
    """Shared engine for both river variants.

    ordered_indices : task indices in the desired left-to-right order
                      (fixed tasks already removed by the caller).
    cat_meta        : list of (first_plot_pos, label, count) for category
                      background shading. For the chronological view this
                      describes runs of the same category in CSV order;
                      for the category view it's the grouped blocks.
    """
    n_visible  = len(ordered_indices)
    x_spacing  = 1.0
    max_branch = 4.5
    # extra vertical space reserved for the category title strip at the bottom
    cat_strip  = 1.6
    y_lo       = -(max_branch + cat_strip)
    y_hi       =   max_branch + 1.0
    y_cat      = -(max_branch + cat_strip * 0.55)   # y position for cat labels

    n_rows  = (n_visible + tasks_per_row - 1) // tasks_per_row
    rows    = [ordered_indices[r * tasks_per_row:(r + 1) * tasks_per_row]
               for r in range(n_rows)]

    row_width = tasks_per_row * x_spacing
    fig_w     = min(max(18, row_width * 0.44 + 3), 44)
    fig_h     = n_rows * (y_hi - y_lo) * 0.40 + 2.5

    fig, axes = plt.subplots(n_rows, 1, figsize=(fig_w, fig_h))
    if n_rows == 1:
        axes = [axes]

    fig.suptitle(fig_title, fontsize=9, fontweight="bold")

    ytick_vals   = [-max_branch, -max_branch * 0.5, 0,
                     max_branch * 0.5, max_branch]
    ytick_labels = ["100%\nHuman", "75%\nHuman", "50/50",
                    "75%\nAuto", "100%\nAuto"]
    cat_bg_colors = ["#ffffff", "#eeeeee"]

    # Load icons once (fail silently if files are missing)
    _icon_agent = _icon_human = None
    try:
        _icon_agent = mpimg.imread(os.path.join(IMAGE_DIR, "AGENT_ICON.png"))
        _icon_human = mpimg.imread(os.path.join(IMAGE_DIR, "HUMAN_PILOT_ICON.png"))
    except Exception:
        pass

    for row_idx, (ax, row_tasks) in enumerate(zip(axes, rows)):
        n_col           = len(row_tasks)
        row_start_g     = row_idx * tasks_per_row
        row_end_g       = row_start_g + n_col

        # ── Category background bands and title strips ─────────────────────
        bg_segs = []
        for cat_start_g, cat_label, cat_count in cat_meta:
            cat_end_g = cat_start_g + cat_count
            seg_s = max(cat_start_g, row_start_g)
            seg_e = min(cat_end_g,   row_end_g)
            if seg_s < seg_e:
                bg_segs.append((seg_s - row_start_g,
                                 seg_e - row_start_g,
                                 cat_label))

        for bi, (cs, ce, clabel) in enumerate(bg_segs):
            x_lo = (cs - 0.5) * x_spacing
            x_hi = (ce - 0.5) * x_spacing
            bgc  = cat_bg_colors[bi % 2]

            # Shaded column band
            ax.axvspan(x_lo, x_hi, ymin=0, ymax=1,
                       facecolor=bgc, alpha=0.6, zorder=0, linewidth=0)

            # Hard separator on left edge
            if cs > 0:
                ax.axvline((cs - 0.5) * x_spacing,
                           color="#888888", linewidth=1.2, zorder=2)

            # Category title: two rules + label in the bottom strip
            mid_x  = (cs + (ce - cs) / 2.0 - 0.5) * x_spacing
            half_w = (ce - cs) * x_spacing / 2.0 * 0.85
            ax.plot([mid_x - half_w, mid_x + half_w],
                    [y_cat, y_cat], color="#555555", linewidth=1.1, zorder=5)
            ax.plot([mid_x - half_w, mid_x + half_w],
                    [y_cat - 0.35, y_cat - 0.35],
                    color="#555555", linewidth=1.1, zorder=5)
            ax.text(mid_x, y_cat - 0.17, clabel,
                    ha="center", va="center",
                    fontsize=8, fontweight="bold", color="#111111",
                    bbox=dict(facecolor=bgc, edgecolor="none", pad=1.5),
                    zorder=6)

        # ── Spine and reference lines ──────────────────────────────────────
        ax.axhline(0, color="#333333", linewidth=1.8, zorder=4)
        for r_ref, ls in [(0.75, ":"), (0.25, ":")]:
            y_ref = (r_ref - 0.5) * 2.0 * max_branch
            ax.axhline(y_ref, color="#cccccc", linewidth=0.8,
                       linestyle=ls, zorder=1)

        # ── Draw each task ─────────────────────────────────────────────────
        for col, ti in enumerate(row_tasks):
            x     = col * x_spacing
            ratio = float(auto_ratio[ti])
            y     = (ratio - 0.5) * 2.0 * max_branch
            color = _river_color(ratio)

            if abs(y) > 0.05:
                ax.plot([x, x], [0, y], color=color, linewidth=1.6,
                        solid_capstyle="round", zorder=3)

            _wrap = 16
            obj_w = textwrap.fill(task_objects[ti], _wrap)
            val_w = textwrap.fill(task_values[ti],   _wrap)
            pct   = f"{ratio * 100:.0f}%"
            label = f"{obj_w}\n{val_w}\n{pct}"

            ax.text(x, y, label,
                    ha="center", va="center",
                    fontsize=9, linespacing=1.3,
                    bbox=dict(boxstyle="round,pad=0.32",
                              facecolor=color,
                              edgecolor="white", linewidth=0.5,
                              alpha=0.95),
                    zorder=9)

        # ── Axes styling ───────────────────────────────────────────────────
        ax.set_ylim(y_lo, y_hi)
        ax.set_yticks(ytick_vals)
        ax.set_yticklabels(ytick_labels, fontsize=7)
        ax.set_xlim(-2.0, (n_col - 0.5) * x_spacing + 0.8)

        # ── Icons in left margin ───────────────────────────────────────────
        if _icon_agent is not None:
            icon_x = -1.5
            for img, icon_y in [(_icon_agent,  max_branch * 0.70),
                                 (_icon_human, -max_branch * 0.70)]:
                ax.add_artist(
                    AnnotationBbox(OffsetImage(img, zoom=0.037, resample=True),
                                   (icon_x, icon_y),
                                   frameon=False, zorder=10))
        ax.set_xticks([])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)

        start_task = row_idx * tasks_per_row + 1
        end_task   = row_idx * tasks_per_row + n_col
        ax.set_ylabel(f"#{start_task}–{end_task}", fontsize=7, labelpad=4)

    # ── Shared legend ──────────────────────────────────────────────────────
    legend_patches = [
        mpatches.Patch(color=_river_color(r),
                       label={1.0: "100% Auto", 0.75: "≥75% Auto",
                              0.5: "50/50", 0.25: "≥75% Human",
                              0.0: "100% Human"}[r])
        for r in [1.0, 0.75, 0.5, 0.25, 0.0]
    ]
    fig.legend(handles=legend_patches, loc="lower center",
               bbox_to_anchor=(0.5, 0), fontsize=7,
               title="Allocation consensus", title_fontsize=8,
               ncol=len(legend_patches), frameon=True)
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    _save(fig, out_filename)
    return fig


def plot_allocation_river_category(vectors, task_objects, task_values, categories,
                                   is_choice, tasks_per_row: int = 22):
    """River grouped and sorted by task category."""
    n_p = vectors.shape[0]
    auto_counts = np.sum((vectors == AUTO_FAST) | (vectors == AUTO_SLOW), axis=0)
    auto_ratio  = auto_counts / float(n_p)

    cat_order = _unique_ordered(categories)
    ordered   = []
    cat_meta  = []
    for c in cat_order:
        idxs = [i for i, cat in enumerate(categories)
                if cat == c and is_choice[i]]
        if idxs:
            cat_meta.append((len(ordered), c, len(idxs)))
            ordered.extend(idxs)

    title = (
        "TARC Allocation - (grouped by category)"
    )
    return _draw_river(title, "allocation_river_category.png",
                       ordered, cat_meta, auto_ratio,
                       task_objects, task_values, categories, tasks_per_row)


def plot_allocation_river_chronological(vectors, task_objects, task_values,
                                        categories, is_choice,
                                        procedures=None,
                                        tasks_per_row: int = 22):
    """River in CSV (chronological / procedural) order.

    Sections (background bands + title strips) are drawn by Procedure when
    `procedures` is provided, otherwise fall back to Category.
    """
    n_p = vectors.shape[0]
    auto_counts = np.sum((vectors == AUTO_FAST) | (vectors == AUTO_SLOW), axis=0)
    auto_ratio  = auto_counts / float(n_p)

    # Keep only choice tasks, in CSV row order
    ordered = [i for i in range(len(task_objects)) if is_choice[i]]

    # Group labels: Procedure when available, else Category
    group_labels = procedures if procedures is not None else categories

    # Build cat_meta as consecutive runs of the same group label in ordered list
    cat_meta = []
    for plot_pos, ti in enumerate(ordered):
        grp = group_labels[ti]
        if cat_meta and cat_meta[-1][1] == grp:
            start, lbl, cnt = cat_meta[-1]
            cat_meta[-1] = (start, lbl, cnt + 1)
        else:
            cat_meta.append((plot_pos, grp, 1))

    title = (
        "TARC Allocation - Chronological / Procedural Order"
    )
    return _draw_river(title, "allocation_river_chronological.png",
                       ordered, cat_meta, auto_ratio,
                       task_objects, task_values, categories, tasks_per_row)


def _hex_to_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i+2], 16) / 255 for i in (0, 2, 4))


def plot_category_breakdown(participants, vectors, categories, is_choice):
    """Stacked bar chart per task category showing allocation type counts.

    Only choice tasks (is_choice==True) are included.
    Segment labels: H* = human, P*-F = auto fast, P*-S = auto slow.
    Participants sorted human-most on the left within each subplot.
    """
    cat_order = _unique_ordered(categories)
    cat_indices = {c: [i for i, cat in enumerate(categories)
                       if cat == c and is_choice[i]]
                   for c in cat_order}
    cat_order = [c for c in cat_order if cat_indices[c]]
    seg_labels = {HUMAN: "H*", AUTO_FAST: "P*-F", AUTO_SLOW: "P*-S", SHARED: ""}

    n_cats = len(cat_order)
    ncols  = 4
    nrows  = (n_cats + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(ncols * 3.5, nrows * 2.8),
                              sharey=False)
    fig.suptitle("TARC Allocation by Task Category\n"
                 "(choice tasks only — stacked counts per participant)",
                 fontsize=12, fontweight="bold")

    axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for ax_idx, cat in enumerate(cat_order):
        ax = axes_flat[ax_idx]
        task_indices = cat_indices[cat]
        counts = np.zeros((len(participants), 4), dtype=int)
        for pi in range(len(participants)):
            for ti in task_indices:
                counts[pi, vectors[pi, ti]] += 1

        # Sort participants: human-most on the left
        order = np.argsort(-counts[:, HUMAN])
        counts_sorted = counts[order]
        labels_sorted = [participants[i] for i in order]
        x = np.arange(len(participants))

        bottom = np.zeros(len(participants))
        for state in [HUMAN, AUTO_FAST, AUTO_SLOW, SHARED]:
            heights = counts_sorted[:, state].astype(float)
            ax.bar(x, heights, bottom=bottom,
                   color=ALLOC_COLORS[state], label=ALLOC_LABELS[state],
                   width=0.7, edgecolor="white", linewidth=0.4)
            lbl = seg_labels[state]
            if lbl:
                for xi, (h, b) in enumerate(zip(heights, bottom)):
                    if h >= 1:
                        ax.text(xi, b + h / 2, lbl,
                                ha="center", va="center",
                                fontsize=6.5, fontweight="bold",
                                color="white")
            bottom += heights

        total = len(task_indices)
        ax.set_title(f"{cat}  (n={total})", fontsize=8, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(labels_sorted, rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("Tasks", fontsize=7)
        ax.set_ylim(0, total + 0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="y", labelsize=7)

    for ax_idx in range(n_cats, len(axes_flat)):
        axes_flat[ax_idx].set_visible(False)

    legend_patches = [
        mpatches.Patch(color=ALLOC_COLORS[k], label=f"{ALLOC_LABELS[k]}  ({seg_labels[k]})" if seg_labels[k] else ALLOC_LABELS[k])
        for k in [HUMAN, AUTO_FAST, AUTO_SLOW, SHARED]
    ]
    fig.legend(handles=legend_patches, loc="lower center", ncol=4,
               fontsize=9, bbox_to_anchor=(0.5, 0))
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    _save(fig, "allocation_category_breakdown.png")
    return fig


def plot_task_breakdown(vectors, task_objects, task_values, categories, is_choice):
    """Stacked % bar per task, grouped by category.

    Each subplot = one category; x = individual tasks sorted human-most on left;
    bars = % of participants who chose Human / Auto-Fast / Auto-Slow / Shared.
    Column widths are proportional to the number of tasks in that column.
    Only choice tasks are shown.
    """
    n_p = vectors.shape[0]
    seg_labels = {HUMAN: "H*", AUTO_FAST: "P*-F", AUTO_SLOW: "P*-S", SHARED: ""}

    cat_order = _unique_ordered(categories)
    cat_indices = {c: [i for i, cat in enumerate(categories)
                       if cat == c and is_choice[i]]
                   for c in cat_order}
    cat_order = [c for c in cat_order if cat_indices[c]]

    n_cats = len(cat_order)
    ncols  = 4
    nrows  = (n_cats + ncols - 1) // ncols

    # Column width ratios: proportional to max task count in each column
    col_widths = []
    for j in range(ncols):
        cats_in_col = [cat_order[i] for i in range(j, n_cats, ncols)]
        col_widths.append(max((len(cat_indices[c]) for c in cats_in_col), default=1))

    bar_w_inch = 0.55   # inches per task slot
    fig_w = sum(w * bar_w_inch + 1.4 for w in col_widths)
    fig_h = nrows * 3.2

    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(fig_w, fig_h),
                              gridspec_kw={"width_ratios": col_widths},
                              sharey=True)
    fig.suptitle("TARC Allocation by Task  (choice tasks only)\n"
                 "% of participants per allocation choice",
                 fontsize=12, fontweight="bold")

    axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for ax_idx, cat in enumerate(cat_order):
        ax = axes_flat[ax_idx]
        task_indices = cat_indices[cat]

        # Sort tasks: human-most % on the left
        task_indices = sorted(task_indices,
                              key=lambda ti: np.sum(vectors[:, ti] == HUMAN),
                              reverse=True)
        n_tasks = len(task_indices)
        x = np.arange(n_tasks)

        # pct[task, state]  — percentage of participants
        pct = np.zeros((n_tasks, 4))
        for ti_pos, ti in enumerate(task_indices):
            for state in range(4):
                pct[ti_pos, state] = np.sum(vectors[:, ti] == state) / n_p * 100

        bottom = np.zeros(n_tasks)
        for state in [HUMAN, AUTO_FAST, AUTO_SLOW, SHARED]:
            heights = pct[:, state]
            ax.bar(x, heights, bottom=bottom,
                   color=ALLOC_COLORS[state], label=ALLOC_LABELS[state],
                   width=0.75, edgecolor="white", linewidth=0.4)
            lbl = seg_labels[state]
            if lbl:
                for xi, (h, b) in enumerate(zip(heights, bottom)):
                    if h >= 12:
                        ax.text(xi, b + h / 2, lbl,
                                ha="center", va="center",
                                fontsize=6, fontweight="bold",
                                color="white")
            bottom += heights

        x_labels = [f"{task_objects[ti]}\n{task_values[ti]}"
                    for ti in task_indices]
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=6)
        ax.set_xlim(-0.5, n_tasks - 0.5)
        ax.set_title(f"{cat}  (n={n_tasks})", fontsize=9, fontweight="bold",
                     pad=4, color="white",
                     bbox=dict(facecolor="#333333", edgecolor="none",
                               boxstyle="round,pad=0.35"))
        ax.set_ylabel("%", fontsize=7)
        ax.set_ylim(0, 105)
        ax.axhline(50, color="#cccccc", linewidth=0.7, linestyle=":")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="y", labelsize=7)

    for ax_idx in range(n_cats, len(axes_flat)):
        axes_flat[ax_idx].set_visible(False)

    legend_patches = [
        mpatches.Patch(color=ALLOC_COLORS[k],
                       label=f"{ALLOC_LABELS[k]}  ({seg_labels[k]})" if seg_labels[k]
                             else ALLOC_LABELS[k])
        for k in [HUMAN, AUTO_FAST, AUTO_SLOW, SHARED]
    ]
    fig.legend(handles=legend_patches, loc="lower center", ncol=4,
               fontsize=9, bbox_to_anchor=(0.5, 0))
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    _save(fig, "allocation_task_breakdown.png")
    return fig


def _save(fig, filename):
    os.makedirs(PLOTS_DIR, exist_ok=True)
    path = os.path.join(PLOTS_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved → {path}")


# ── Report ────────────────────────────────────────────────────────────────────

_STATE_KEYS = {
    HUMAN:     "human",
    AUTO_FAST: "auto_fast",
    AUTO_SLOW: "auto_slow",
}


def _bar(value: float, max_value: float, width: int = 20) -> str:
    filled = round(value / max_value * width) if max_value else 0
    return "█" * filled + "░" * (width - filled)


def write_allocation_report(participants: list, vectors: np.ndarray,
                            ref_rows: list, is_choice: np.ndarray) -> None:
    """Compute descriptive statistics and write allocation_report.txt."""
    import json
    from datetime import date

    n_p      = len(participants)
    n_tasks  = vectors.shape[1]
    n_choice = int(is_choice.sum())
    n_fixed  = n_tasks - n_choice
    choice_idx = [i for i in range(n_tasks) if is_choice[i]]

    task_objects = build_task_objects(ref_rows)
    categories   = build_categories(ref_rows)

    v      = vectors[:, choice_idx]          # (n_p, n_choice)
    cats_c = [categories[i]   for i in choice_idx]
    objs_c = [task_objects[i] for i in choice_idx]
    total_cells = n_p * n_choice

    # ── Overall counts ────────────────────────────────────────────────────────
    overall = {}
    for s, key in _STATE_KEYS.items():
        cnt = int(np.sum(v == s))
        overall[key] = {"count": cnt, "pct": round(cnt / total_cells * 100, 1)}

    # ── Per-participant ───────────────────────────────────────────────────────
    per_part = {}
    for pi, pid in enumerate(participants):
        d = {key: int(np.sum(v[pi] == s)) for s, key in _STATE_KEYS.items()}
        d["auto_pct"] = round((d["auto_fast"] + d["auto_slow"]) / n_choice * 100, 1)
        per_part[pid] = d

    # ── Per-category ──────────────────────────────────────────────────────────
    cat_order = _unique_ordered(cats_c)
    per_cat = {}
    for cat in cat_order:
        idxs = [i for i, c in enumerate(cats_c) if c == cat]
        vc   = v[:, idxs]
        total_cat = n_p * len(idxs)
        d = {"n_tasks": len(idxs)}
        for s, key in _STATE_KEYS.items():
            d[key + "_pct"] = round(int(np.sum(vc == s)) / total_cat * 100, 1)
        per_cat[cat] = d

    # ── Per-task agreement ────────────────────────────────────────────────────
    task_agr = []
    for ti in range(n_choice):
        col = v[:, ti]
        unique, counts = np.unique(col, return_counts=True)
        maj_s   = int(unique[np.argmax(counts)])
        maj_pct = round(int(np.max(counts)) / n_p * 100, 1)
        task_agr.append({
            "task":          objs_c[ti],
            "category":      cats_c[ti],
            "majority_state": _STATE_KEYS[maj_s],
            "majority_pct":  maj_pct,
        })

    n_consensus = sum(1 for t in task_agr if t["majority_pct"] >= 80)
    n_majority  = sum(1 for t in task_agr if t["majority_pct"] >= 65)
    n_split     = sum(1 for t in task_agr if t["majority_pct"] < 50)
    top_agreed   = sorted(task_agr, key=lambda t: -t["majority_pct"])[:5]
    top_disputed = sorted(task_agr, key=lambda t:  t["majority_pct"])[:5]

    # ── Pairwise Hamming similarity ───────────────────────────────────────────
    pairs = [
        (participants[i], participants[j],
         round(float(np.mean(v[i] == v[j])), 3))
        for i in range(n_p)
        for j in range(i + 1, n_p)
    ]
    mean_sim  = round(float(np.mean([s for *_, s in pairs])), 3) if pairs else None
    most_sim  = max(pairs, key=lambda x: x[2]) if pairs else None
    most_diff = min(pairs, key=lambda x: x[2]) if pairs else None

    # ── JSON data ─────────────────────────────────────────────────────────────
    json_data = {
        "generated":      str(date.today()),
        "n_participants": n_p,
        "participants":   participants,
        "n_choice_tasks": n_choice,
        "n_fixed_tasks":  n_fixed,
        "overall_allocation": overall,
        "per_participant":    per_part,
        "per_category":       per_cat,
        "agreement": {
            "consensus_tasks_ge80pct": n_consensus,
            "majority_tasks_ge65pct":  n_majority,
            "split_tasks_lt50pct":     n_split,
        },
        "hamming_similarity": {
            "mean": mean_sim,
            "most_similar_pair":   {"pair": [most_sim[0],  most_sim[1]],  "similarity": most_sim[2]}  if most_sim  else None,
            "most_different_pair": {"pair": [most_diff[0], most_diff[1]], "similarity": most_diff[2]} if most_diff else None,
        },
    }

    # ── Formatted output ──────────────────────────────────────────────────────
    SEP  = "=" * 72

    lines = []
    a = lines.append

    a("--- MACHINE-READABLE SUMMARY (JSON) ---")
    a(json.dumps(json_data, indent=2))
    a("--- END SUMMARY ---")
    a("")
    a(SEP)
    a("  TARC ALLOCATION REPORT")
    a(f"  N = {n_p} participants  |  {n_choice} choice tasks  |  {n_fixed} fixed tasks")
    a(f"  Generated: {date.today()}")
    a(SEP)
    a("")

    # Overall allocation
    a("\u2500\u2500 OVERALL ALLOCATION (choice tasks only) " + "\u2500" * 29)
    a("")
    a(f"  {'State':<22}  {'Count':>6}  {'%':>6}  Bar (0\u2013100%)")
    a(f"  {'\u2500'*22}  {'\u2500'*6}  {'\u2500'*6}  {'\u2500'*20}")
    for s, key in _STATE_KEYS.items():
        cnt = overall[key]["count"]
        pct = overall[key]["pct"]
        a(f"  {ALLOC_LABELS[s]:<22}  {cnt:>6}  {pct:>5.1f}%  [{_bar(pct, 100)}]")
    a("")
    a(f"  Total cells (N={n_p} \u00d7 {n_choice} choice tasks): {total_cells}")
    a("")

    # Per-participant
    a("\u2500\u2500 PER-PARTICIPANT SUMMARY " + "\u2500" * 45)
    a("")
    a(f"  {'Participant':<12}  {'Human':>6}  {'Auto-F':>7}  {'Auto-S':>7}  {'Auto%':>6}")
    a(f"  {'\u2500'*12}  {'\u2500'*6}  {'\u2500'*7}  {'\u2500'*7}  {'\u2500'*6}")
    for pid in participants:
        d = per_part[pid]
        a(f"  {pid:<12}  {d['human']:>6}  {d['auto_fast']:>7}  "
          f"{d['auto_slow']:>7}  {d['auto_pct']:>5.1f}%")
    sorted_auto = sorted(participants, key=lambda p: per_part[p]["auto_pct"])
    a("")
    a(f"  Least automation: {sorted_auto[0]}  ({per_part[sorted_auto[0]]['auto_pct']:.1f}%)")
    a(f"  Most automation:  {sorted_auto[-1]}  ({per_part[sorted_auto[-1]]['auto_pct']:.1f}%)")
    a("")

    # Per-category
    a("\u2500\u2500 PER-CATEGORY BREAKDOWN " + "\u2500" * 46)
    a("")
    a(f"  {'Category':<22}  {'Tasks':>5}  {'Human%':>7}  {'Auto-F%':>8}  {'Auto-S%':>8}")
    a(f"  {'\u2500'*22}  {'\u2500'*5}  {'\u2500'*7}  {'\u2500'*8}  {'\u2500'*8}")
    for cat in cat_order:
        d = per_cat[cat]
        a(f"  {cat:<22}  {d['n_tasks']:>5}  {d['human_pct']:>6.1f}%  "
          f"{d['auto_fast_pct']:>7.1f}%  {d['auto_slow_pct']:>7.1f}%")
    a("")

    # Agreement
    a("\u2500\u2500 INTER-PARTICIPANT AGREEMENT " + "\u2500" * 41)
    a("")
    a(f"  Consensus tasks (\u226580% same choice):  {n_consensus:>2} / {n_choice}")
    a(f"  Majority tasks  (\u226565% same choice):  {n_majority:>2} / {n_choice}")
    a(f"  Split tasks     (<50% majority):     {n_split:>2} / {n_choice}")
    a("")
    a("  Top-5 most agreed tasks:")
    for t in top_agreed:
        a(f"    {t['majority_pct']:>5.1f}%  {t['majority_state']:<16}  "
          f"{t['task']}  [{t['category']}]")
    a("")
    a("  Top-5 most disputed tasks:")
    for t in top_disputed:
        a(f"    {t['majority_pct']:>5.1f}%  {t['majority_state']:<16}  "
          f"{t['task']}  [{t['category']}]")
    a("")

    # Hamming similarity
    a("\u2500\u2500 PAIRWISE HAMMING SIMILARITY " + "\u2500" * 41)
    a("")
    if mean_sim is not None:
        a(f"  Mean similarity across all {len(pairs)} pairs: "
          f"{mean_sim:.3f}  [{_bar(mean_sim * 100, 100)}]")
    if most_sim:
        a(f"  Most similar pair:    {most_sim[0]} & {most_sim[1]}  "
          f"(sim = {most_sim[2]:.3f})")
    if most_diff:
        a(f"  Most different pair:  {most_diff[0]} & {most_diff[1]}  "
          f"(sim = {most_diff[2]:.3f})")
    a("")
    a(SEP)
    a("")

    os.makedirs(REPORT_DIR, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    print(f"  Report \u2192 {REPORT_PATH}")


# ── Main ──────────────────────────────────────────────────────────────────────

_ALLOCATION_PLOTS = [
    "allocation_river_category.png",
    "allocation_river_chronological.png",
    "allocation_category_breakdown.png",
    "allocation_task_breakdown.png",
]


def _confirm_run(participants, output_files):
    """Print a pre-run summary and ask the user to confirm before proceeding.

    output_files: list of (display_name, abs_path) tuples.
    """
    print(f"\nParticipants:  {', '.join(participants)}")
    print(f"\nOutput files that will be written/overwritten ({len(output_files)}):")
    for label, path in output_files:
        tag = "[overwrite]" if os.path.exists(path) else "[new     ]"
        print(f"  {tag}  {label}")
    print()
    try:
        ans = input("Continue? [Y/n]: ").strip().lower()
    except KeyboardInterrupt:
        print("\nAborted.")
        return False
    return ans in ("", "y", "yes")


def main():
    print("=" * 65)
    print("  HITLS — TARC Allocation Similarity Analysis")
    print("=" * 65)

    participants, vectors, ref_rows = load_all()

    if len(participants) < 2:
        print("Need at least 2 participants — aborting.")
        return

    output_files = [(name, os.path.join(PLOTS_DIR, name)) for name in _ALLOCATION_PLOTS]
    output_files.append(("allocation/allocation_report.txt", REPORT_PATH))
    if not _confirm_run(participants, output_files):
        return

    task_objects = build_task_objects(ref_rows)
    task_values  = build_task_values(ref_rows)
    categories   = build_categories(ref_rows)
    procedures   = build_procedures(ref_rows)
    is_choice    = load_template()

    os.makedirs(PLOTS_DIR, exist_ok=True)
    print(f"\nGenerating plots → {PLOTS_DIR}/\n")

    plot_allocation_river_category(vectors, task_objects, task_values, categories, is_choice)
    plot_allocation_river_chronological(vectors, task_objects, task_values, categories,
                                        is_choice, procedures=procedures)
    plot_category_breakdown(participants, vectors, categories, is_choice)
    plot_task_breakdown(vectors, task_objects, task_values, categories, is_choice)

    print(f"\nGenerating report ...")
    write_allocation_report(participants, vectors, ref_rows, is_choice)

    print(f"\nDone — 4 figures + 1 report saved.")
    plt.show()


if __name__ == "__main__":
    main()
