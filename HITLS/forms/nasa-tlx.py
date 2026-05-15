"""
nasa-tlx.py — NASA Task Load Index (TLX) Processor
===================================================
Computes weighted NASA-TLX workload scores from 6 subscales
(Mental Demand, Physical Demand, Temporal Demand, Performance,
Effort, Frustration) with pairwise comparison weights.
Reads from HAT_study.csv (questionnaire_id = "nasa_tlx").

Typically invoked by `forms/forms.py`, but can be run interactively.
Output: {PID}/cleaned/{PID}_nasa_tlx_report.txt
"""

import os
import csv
import json
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec

HITLS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Maps display names used in pairwise comparisons to question_id keys
DISPLAY_TO_KEY = {
    "Mental Demand":    "mental_demand",
    "Physical Demand":  "physical_demand",
    "Temporal Demand":  "temporal_demand",
    "Performance":      "performance",
    "Effort":           "effort",
    "Frustration":      "frustration",
}

SUBSCALE_LABELS = {
    "mental_demand":   "Mental Demand",
    "physical_demand": "Physical Demand",
    "temporal_demand": "Temporal Demand",
    "performance":     "Performance",
    "effort":          "Effort",
    "frustration":     "Frustration",
}

CONDITION_ORDER = [
    "baseline_no_system",
    "TARS",
    "TARC",
    "TARP-S",
    "TARP-F",
]


def find_participants():
    return [
        e for e in sorted(os.listdir(HITLS_DIR))
        if os.path.isdir(os.path.join(HITLS_DIR, e))
        and e.startswith("P") and e[1:].isdigit()
    ]


def find_forms_csv(participant_id):
    folder = os.path.join(HITLS_DIR, participant_id)
    # Priority 1: participant-specific naming P0X_forms.csv
    specific = os.path.join(folder, f"{participant_id}_forms.csv")
    if os.path.isfile(specific):
        return specific
    # Priority 2: any file matching HAT_study*.csv or briefing_export*.csv
    # (HAT_study files contain the same long-form data)
    for name in sorted(os.listdir(folder)):
        lower = name.lower()
        if "hat_study" in lower and name.endswith(".csv"):
            return os.path.join(folder, name)
    return None


def load_rows(csv_path):
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def extract_weights(rows):
    """
    Tally how many times each subscale was chosen across the 15 pairwise
    comparisons (nasa_tlx_subscale_ranking, condition after_familiarization).
    Returns dict {question_key: weight (0-5)}.
    """
    tally = defaultdict(int)
    for row in rows:
        if (row["questionnaire_id"] == "nasa_tlx_subscale_ranking"
                and row["condition"] == "after_familiarization"):
            chosen = row["value"].strip()
            key = DISPLAY_TO_KEY.get(chosen)
            if key:
                tally[key] += 1
    return dict(tally)


def extract_ratings(rows):
    """
    Returns dict {condition: {question_key: rating_float}}.
    """
    ratings = defaultdict(dict)
    for row in rows:
        if row["questionnaire_id"] == "nasa_tlx_evaluation":
            condition = row["condition"].strip()
            q = row["question_id"].strip()
            try:
                ratings[condition][q] = float(row["value"])
            except ValueError:
                pass
    return dict(ratings)


def compute_nasa_tlx(weights, ratings_for_condition):
    """
    NASA-TLX weighted score = sum(weight_i * (rating_i * 5)) / 15
    Ratings are stored as 0-20 interval indices; multiply by 5 to get 0-100.
    Returns (score, per_subscale_detail) or (None, {}) if data missing.
    """
    details = {}
    total_weighted = 0.0
    for key in SUBSCALE_LABELS:
        w = weights.get(key, 0)
        r = ratings_for_condition.get(key)
        if r is None:
            continue
        r100 = r * 5          # convert 0-20 → 0-100
        weighted = w * r100
        total_weighted += weighted
        details[key] = {"weight": w, "rating": r, "rating100": r100, "weighted": weighted}
    if not details:
        return None, {}
    score = total_weighted / 15.0
    return score, details


def bar_rating(value, max_val=20, width=20):
    filled = int((value / max_val) * width)
    return f"[{'█' * filled}{'░' * (width - filled)}] {value:>4.0f}/20"


def bar_weight(value, max_val=5, width=10):
    filled = int((value / max_val) * width) if max_val > 0 else 0
    return f"[{'█' * filled}{'░' * (width - filled)}] {value}/5"


def build_summary(participant_id, weights, all_ratings, conditions):
    """Return a dict of key scalars suitable for cross-subject aggregation."""
    summary = {
        "participant": participant_id,
        "weights": {key: weights.get(key, 0) for key in SUBSCALE_LABELS},
        "conditions": {},
    }
    for condition in conditions:
        score, details = compute_nasa_tlx(weights, all_ratings[condition])
        summary["conditions"][condition] = {
            "nasa_tlx_weighted_score": round(score, 4) if score is not None else None,
            "subscales": {
                key: {
                    "rating_0_20": details[key]["rating"],
                    "rating_0_100": details[key]["rating100"],
                    "weight": details[key]["weight"],
                    "weighted": round(details[key]["weighted"], 4),
                }
                for key in SUBSCALE_LABELS
                if key in details
            },
        }
    return summary


def build_report(participant_id, weights, all_ratings):
    lines = []
    sep = "=" * 72

    lines.append(f"\n{sep}")
    lines.append(f"  NASA-TLX REPORT — {participant_id}")
    lines.append(sep)

    # ── Weights ──────────────────────────────────────────────────────────────
    lines.append("\n── PAIRWISE WEIGHTS (collected after familiarization) ───────────────────\n")
    lines.append(f"  {'Subscale':<20} {'Weight (tally/15 pairs)'}")
    lines.append(f"  {'─'*20}   {'─'*35}")
    for key, label in SUBSCALE_LABELS.items():
        w = weights.get(key, 0)
        lines.append(f"  {label:<20}   {bar_weight(w)}  ({w} selections)")
    lines.append(f"\n  Total tallied: {sum(weights.get(k,0) for k in SUBSCALE_LABELS)} / 15")

    # ── Per-condition scores ──────────────────────────────────────────────────
    conditions = [c for c in CONDITION_ORDER if c in all_ratings]
    # Also include any conditions present but not in our order list
    extra = [c for c in sorted(all_ratings) if c not in CONDITION_ORDER]
    conditions += extra

    for condition in conditions:
        ratings = all_ratings[condition]
        score, details = compute_nasa_tlx(weights, ratings)

        lines.append(f"\n── CONDITION: {condition} {'─' * (55 - len(condition))}\n")
        lines.append(
            f"  {'Subscale':<20} {'Rating':>14}   {'Weight':>7}   {'Weighted':>9}   {'Bar (rating/20)'}"
        )
        lines.append(f"  {'─'*20}   {'─'*14}   {'─'*7}   {'─'*9}   {'─'*25}")

        for key, label in SUBSCALE_LABELS.items():
            if key not in details:
                continue
            d = details[key]
            lines.append(
                f"  {label:<20}   {d['rating']:>4.0f}/20 ({d['rating100']:>3.0f}/100)"
                f"   {d['weight']:>5}/5   {d['weighted']:>7.1f}    {bar_rating(d['rating'])}"
            )

        if score is not None:
            lines.append(f"\n  {'─'*60}")
            lines.append(f"  NASA-TLX Weighted Score : {score:.2f} / 100")
            if score >= 80:
                interp = "Very high workload"
            elif score >= 60:
                interp = "High workload"
            elif score >= 40:
                interp = "Moderate workload"
            elif score >= 20:
                interp = "Low workload"
            else:
                interp = "Very low workload"
            lines.append(f"  Interpretation          : {interp}")

    # ── Summary table ─────────────────────────────────────────────────────────
    lines.append(f"\n── SUMMARY ─────────────────────────────────────────────────────────────\n")
    lines.append(f"  {'Condition':<25} {'NASA-TLX Score':>16}")
    lines.append(f"  {'─'*25}   {'─'*16}")
    for condition in conditions:
        ratings = all_ratings[condition]
        score, _ = compute_nasa_tlx(weights, ratings)
        score_str = f"{score:.2f} / 100" if score is not None else "N/A"
        lines.append(f"  {condition:<25}   {score_str:>16}")

    lines.append(f"\n{sep}\n")
    return "\n".join(lines)


def generate_visual_report(participant_id, weights, all_ratings, conditions,
                           save_dir=None):
    """
    Visual per-participant NASA-TLX report — three sections:

      Top-left  : dimension ranking (pairwise weights, sorted best → worst)
      Top-right : per-dimension horizontal 0-20 strip with one marker per condition
      Bottom    : computed-score formula + breakdown table

    Saved as <participant_id>_nasa_tlx_visual.png in *save_dir* if provided.
    Returns the matplotlib Figure.
    """
    COND_COLORS = {
        "baseline_no_system": "#888888",
        "TARS":               "#4472C4",
        "TARC":               "#ED7D31",
        "TARP-S":             "#70AD47",
        "TARP-F":             "#C00000",
    }
    MARKERS      = ["o", "s", "D", "^", "v"]
    DEFAULT_COL  = "#999999"

    dims    = list(SUBSCALE_LABELS.keys())
    n_dims  = len(dims)
    n_conds = len(conditions)

    # Sorted order for the ranking panel (most selected = top)
    sorted_dims    = sorted(dims, key=lambda k: weights.get(k, 0), reverse=True)
    sorted_labels  = [SUBSCALE_LABELS[k] for k in sorted_dims]
    sorted_weights = [weights.get(k, 0) for k in sorted_dims]

    cond_style = {
        c: {"color": COND_COLORS.get(c, DEFAULT_COL),
            "marker": MARKERS[i % len(MARKERS)]}
        for i, c in enumerate(conditions)
    }

    # ── Figure layout ─────────────────────────────────────────────────────────
    formula_h = n_conds * 0.60 + 2.0
    strip_h   = max(5.0, n_dims * 0.90)
    fig = plt.figure(figsize=(14, strip_h + formula_h + 0.8))
    fig.suptitle(f"NASA-TLX Visual Report — {participant_id}",
                 fontsize=13, fontweight="bold")

    gs_outer = GridSpec(2, 1, figure=fig,
                        height_ratios=[strip_h, formula_h],
                        hspace=0.40)

    # Top half: ranking (left) + rating strips (right)
    gs_top = gs_outer[0].subgridspec(n_dims, 2,
                                     width_ratios=[1, 2.5],
                                     hspace=0.05, wspace=0.42)

    # Bottom half: formula text
    ax_formula = fig.add_subplot(gs_outer[1])
    ax_formula.axis("off")

    # ── Ranking panel ─────────────────────────────────────────────────────────
    ax_rank = fig.add_subplot(gs_top[:, 0])
    rank_colors = plt.cm.RdYlGn(np.linspace(0.15, 0.85, n_dims))[::-1]
    y_pos = list(range(n_dims))

    bars = ax_rank.barh(y_pos, sorted_weights,
                        color=rank_colors, edgecolor="white", height=0.62)
    for bar, val in zip(bars, sorted_weights):
        ax_rank.text(val + 0.06, bar.get_y() + bar.get_height() / 2,
                     str(val), va="center", ha="left", fontsize=9,
                     fontweight="bold")

    ax_rank.set_yticks(y_pos)
    ax_rank.set_yticklabels(sorted_labels, fontsize=9)
    ax_rank.set_xlim(0, 6.2)
    ax_rank.set_xticks(range(6))
    ax_rank.set_xlabel("Weight (0–5 pairwise selections)", fontsize=8)
    ax_rank.set_title("Dimension Ranking\n(pairwise weights)", fontsize=9,
                      fontweight="bold")
    ax_rank.invert_yaxis()
    ax_rank.grid(axis="x", linestyle=":", alpha=0.4)

    # ── Rating strips (one per dimension, right column) ───────────────────────
    ax_strips = []
    for ki, key in enumerate(dims):
        share = ax_strips[0] if ki > 0 else None
        ax = fig.add_subplot(gs_top[ki, 1], sharex=share)
        ax_strips.append(ax)

        # Faint bin lines at every integer 0-20
        for x in range(0, 21):
            ax.axvline(x, color="#e6e6e6", linewidth=0.4, zorder=0)

        for i, cond in enumerate(conditions):
            rating = all_ratings.get(cond, {}).get(key)
            if rating is None:
                continue
            style = cond_style[cond]
            y_off = (i - (n_conds - 1) / 2.0) * 0.16
            ax.scatter(rating, y_off,
                       color=style["color"], marker=style["marker"],
                       s=110, zorder=5, edgecolors="white", linewidths=0.8)
            ax.text(rating, y_off + 0.30, f"{int(rating)}",
                    ha="center", va="bottom", fontsize=6.5,
                    color=style["color"], fontweight="bold")

        ax.set_xlim(-0.5, 20.5)
        ax.set_ylim(-0.70, 0.70)
        ax.set_yticks([])
        ax.set_ylabel(SUBSCALE_LABELS[key], fontsize=8.5,
                      rotation=0, ha="right", va="center", labelpad=6)
        ax.grid(axis="x", linestyle=":", alpha=0.3)

        if ki < n_dims - 1:
            ax.tick_params(labelbottom=False)
        else:
            ax.set_xticks(range(0, 21))
            ax.tick_params(axis="x", labelsize=7)
            ax.set_xlabel("Rating (0–20)", fontsize=8)

    ax_strips[0].set_title("Ratings per Condition  (0–20 · 20 bins)",
                            fontsize=9, fontweight="bold")

    # ── Formula + breakdown table ─────────────────────────────────────────────
    abbr = {"mental_demand": "MD", "physical_demand": "PD",
            "temporal_demand": "TD", "performance": "Perf",
            "effort": "Eff", "frustration": "Frust"}

    lines = ["Formula:  Score = Σ(weight_i × rating_i × 5) / 15\n"]
    for cond in conditions:
        score, details = compute_nasa_tlx(weights, all_ratings.get(cond, {}))
        if score is None:
            lines.append(f"  {cond:<14}: no data")
            continue
        parts = "  ".join(
            f"{abbr[k]}: {details[k]['weight']}×{details[k]['rating']:.0f}"
            for k in dims if k in details
        )
        lines.append(f"  {cond:<14}: {parts}  →  {score:.1f} / 100")

    ax_formula.text(0.02, 0.96, "\n".join(lines),
                    transform=ax_formula.transAxes,
                    fontsize=8.5, va="top", ha="left", family="monospace",
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="#f5f5f5",
                              edgecolor="#cccccc", alpha=0.9))
    ax_formula.set_title("Computed Score", fontsize=9, fontweight="bold",
                         loc="left", pad=4)

    # ── Legend ────────────────────────────────────────────────────────────────
    legend_handles = [
        Line2D([0], [0], color=cond_style[c]["color"],
               marker=cond_style[c]["marker"], linestyle="None",
               markersize=9, markeredgecolor="white", markeredgewidth=0.8,
               label=c)
        for c in conditions
    ]
    fig.legend(handles=legend_handles, loc="lower left",
               bbox_to_anchor=(0.01, 0.01), fontsize=8.5,
               ncol=min(len(conditions), 5),
               title="Condition", title_fontsize=8, framealpha=0.95)

    fig.tight_layout()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, f"{participant_id}_nasa_tlx_visual.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Visual report saved → {os.path.relpath(path)}")

    return fig


def main():
    participants = find_participants()
    if not participants:
        print("No participant folders found in HITLS directory.")
        return

    print("\nAvailable participants:")
    for i, p in enumerate(participants, 1):
        print(f"  {i}. {p}")

    while True:
        choice = input("\nSelect a participant (number or ID, e.g. 1 or P02): ").strip()
        if choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(participants):
                participant_id = participants[idx]
                break
        elif choice.upper() in participants:
            participant_id = choice.upper()
            break
        print(f"  Invalid choice. Please enter a number between 1 and {len(participants)} or a valid ID.")

    csv_path = find_forms_csv(participant_id)
    if csv_path is None:
        print(f"No forms CSV found for {participant_id}.")
        return

    rows = load_rows(csv_path)
    weights = extract_weights(rows)
    all_ratings = extract_ratings(rows)

    if not weights:
        print(f"No pairwise weight data found for {participant_id}.")
        return
    if not all_ratings:
        print(f"No NASA-TLX rating data found for {participant_id}.")
        return

    # Determine ordered conditions list (same logic as build_report)
    conditions = [c for c in CONDITION_ORDER if c in all_ratings]
    conditions += [c for c in sorted(all_ratings) if c not in CONDITION_ORDER]

    summary = build_summary(participant_id, weights, all_ratings, conditions)
    report = build_report(participant_id, weights, all_ratings)
    print(report)

    json_block = (
        "--- MACHINE-READABLE SUMMARY (JSON) ---\n"
        + json.dumps(summary, indent=2, ensure_ascii=False)
        + "\n--- END SUMMARY ---\n"
    )
    out_path = os.path.join(HITLS_DIR, participant_id, "cleaned", f"{participant_id}_nasa_tlx_report.txt")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(json_block)
        f.write(report)
    print(f"  Report saved to: {os.path.relpath(out_path)}")

    cleaned_dir = os.path.dirname(out_path)
    generate_visual_report(participant_id, weights, all_ratings,
                           conditions, save_dir=cleaned_dir)
    plt.show()


if __name__ == "__main__":
    main()
