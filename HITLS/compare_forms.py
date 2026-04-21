#!/usr/bin/env python3
"""
HITLS — Cross-participant questionnaire comparison
==================================================
1. Runs forms.py for every participant whose cleaned reports are missing.
2. Generates per-questionnaire DIVERGING STACKED BAR charts (Likert-style):
     - One bar per item, bars split left/right from a centre axis (x=0).
     - Left side  = low / negative responses  → red shades.
     - Right side = high / positive responses → green shades.
     - Count labels appear inside each coloured segment.
     - One subplot per condition (2×2 grid) per figure.
3. Generates box-plot figures for all computed scores across conditions.

Colour palette
--------------
  5-point Likert  : deep-red | orange | yellow | light-green | deep-green
  7-point Likert  : deep-red | light-red | orange | yellow | light-green | medium-green | deep-green
  Negative-valence items use already-recoded (inverted) values from the JSON,
  so colour always means:  low (bad) = red  /  high (good) = green.
  For NASA-TLX and Risk VAS where high = bad, the palette is reversed (inverted=True).

Box plots
---------
  Purely descriptive — no statistical test is embedded.
  The choice between t-test, Wilcoxon, ANOVA, Friedman, etc. is a separate step.
"""

import os, sys, json, subprocess, glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Paths ────────────────────────────────────────────────────────────────────
HITLS_DIR    = os.path.dirname(os.path.abspath(__file__))
FORMS_SCRIPT = os.path.join(HITLS_DIR, "forms", "forms.py")
PLOTS_DIR    = os.path.join(HITLS_DIR, "plots")
PYTHON       = sys.executable

# ── Experiment constants ─────────────────────────────────────────────────────
CONDITIONS      = ["TARS", "TARC", "TARP-S", "TARP-F"]   # all four (used by NASA + box plots)
CONDITIONS_MAIN = ["TARS", "TARP-S", "TARP-F"]           # TARC excluded (all other questionnaires)
_BASELINE    = "TARS"
_COMPARISONS = ["TARP-S", "TARP-F"]   # conditions compared vs _BASELINE
# Diff pairs: (reference, compare) — used for the diff panels
_ALL_PAIRS  = [("TARS", "TARP-S"), ("TARS", "TARP-F"), ("TARP-S", "TARP-F")]
_NASA_PAIRS = [("TARS", "TARC"),   ("TARS", "TARP-S"), ("TARS", "TARP-F"), ("TARP-S", "TARP-F")]

COND_COLOR = {
    "TARS":   "#4472C4",
    "TARC":   "#ED7D31",
    "TARP-S": "#70AD47",
    "TARP-F": "#C00000",
}

# ── Likert colour palettes ───────────────────────────────────────────────────
#   Index 0  →  lowest score  →  deep red
#   Index -1 →  highest score →  deep green
COLORS_5 = [
    "#CC0000",   # 1  deep red
    "#FF8000",   # 2  orange
    "#FFD700",   # 3  yellow
    "#66CC00",   # 4  light green
    "#006600",   # 5  deep green
]

COLORS_7 = [
    "#CC0000",   # 1  deep red
    "#FF4000",   # 2  light red
    "#FF8000",   # 3  orange
    "#FFD700",   # 4  yellow
    "#66CC00",   # 5  light green
    "#00AA00",   # 6  medium green
    "#006600",   # 7  deep green
]

# ═══════════════════════════════════════════════════════════════════════════════
#  Low-level helpers
# ═══════════════════════════════════════════════════════════════════════════════

def find_participants():
    dirs = sorted(glob.glob(os.path.join(HITLS_DIR, "P0*")))
    return [os.path.basename(d) for d in dirs if os.path.isdir(d)]


def cleaned_dir(pid):
    return os.path.join(HITLS_DIR, pid, "cleaned")


def has_all_reports(pid):
    cdir = cleaned_dir(pid)
    needed = [
        f"{pid}_nasa_tlx_report.txt",
        f"{pid}_oversight_bespoke_report.txt",
        f"{pid}_sus_report.txt",
        f"{pid}_tia_report.txt",
        f"{pid}_trust_risk_report.txt",
    ]
    return all(os.path.exists(os.path.join(cdir, f)) for f in needed)


def run_forms(pid, participant_number):
    print(f"  Generating reports for {pid} …", flush=True)
    proc = subprocess.run(
        [PYTHON, FORMS_SCRIPT],
        input=f"{participant_number}\n",
        text=True, capture_output=True,
    )
    if proc.returncode != 0:
        print(f"  ⚠  forms.py returned code {proc.returncode} for {pid}")
        if proc.stderr:
            print(proc.stderr[:400])


def load_json(path):
    try:
        txt = open(path).read()
        start = txt.index("{")
        end   = txt.index("--- END SUMMARY ---")
        return json.loads(txt[start:end].strip())
    except (FileNotFoundError, ValueError, json.JSONDecodeError):
        return None


def load_all(participants):
    data = {}
    tags = ("sus", "tia", "trust_risk", "nasa_tlx",
            "oversight_bespoke", "pre_experiment_form")
    for pid in participants:
        cdir = cleaned_dir(pid)
        reps = {}
        for tag in tags:
            d = load_json(os.path.join(cdir, f"{pid}_{tag}_report.txt"))
            if d is not None:
                reps[tag] = d
        if reps:
            data[pid] = reps
    return data


def save_fig(fig, filename):
    os.makedirs(PLOTS_DIR, exist_ok=True)
    path = os.path.join(PLOTS_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved → {path}")


# ═══════════════════════════════════════════════════════════════════════════════
#  Diverging stacked bar scaffold
# ═══════════════════════════════════════════════════════════════════════════════

def _draw_diverging_row(ax, y, counts, palette, inverted=False, height=0.72):
    """Draw one diverging bar at vertical position *y*.

    *counts* : list of ints, one per score level (index 0 = lowest score).
    Bars split at the centre: low scores extend left (red), high scores right (green).
    The middle level (odd-length scales) is split half-left / half-right.
    Segment count labels are printed in white inside each segment.
    """
    cnt = list(counts)
    pal = list(palette)
    if inverted:
        cnt = cnt[::-1]
        pal = pal[::-1]

    n = len(cnt)
    mid = n // 2 if n % 2 == 1 else None
    half_mid = cnt[mid] / 2.0 if mid is not None else 0.0

    # Left side — scores below the midpoint, drawn outward from centre
    cursor = -half_mid
    left_range = range(mid - 1, -1, -1) if mid is not None else range(n // 2 - 1, -1, -1)
    for i in left_range:
        w = cnt[i]
        if w > 0:
            ax.barh(y, -w, left=cursor, height=height, color=pal[i],
                    edgecolor="white", linewidth=0.5)
            ax.text(cursor - w / 2, y, str(int(w)),
                    ha="center", va="center", fontsize=7.5,
                    color="white", fontweight="bold")
        cursor -= w

    # Middle (neutral) — split half-left / half-right
    if mid is not None and cnt[mid] > 0:
        color = pal[mid]
        if half_mid > 0:
            ax.barh(y, -half_mid, left=0,        height=height, color=color,
                    edgecolor="white", linewidth=0.5)
            ax.barh(y,  half_mid, left=0,        height=height, color=color,
                    edgecolor="white", linewidth=0.5)
        ax.text(0, y, str(int(cnt[mid])),
                ha="center", va="center", fontsize=7.5,
                color="black", fontweight="bold")

    # Right side — scores above the midpoint, drawn outward from centre
    cursor = half_mid
    right_start = mid + 1 if mid is not None else n // 2
    for i in range(right_start, n):
        w = cnt[i]
        if w > 0:
            ax.barh(y, w, left=cursor, height=height, color=pal[i],
                    edgecolor="white", linewidth=0.5)
            ax.text(cursor + w / 2, y, str(int(w)),
                    ha="center", va="center", fontsize=7.5,
                    color="white", fontweight="bold")
        cursor += w


def _build_counts(all_data, participants, item_keys, extract_fn,
                  scale_min, scale_max, conditions=None):
    """Return {cond: [[count_per_level] for each item_key]}.

    *extract_fn(pdata, cond, item_key)* → numeric score or None.
    *conditions* defaults to CONDITIONS_MAIN (TARC excluded).
    """
    if conditions is None:
        conditions = CONDITIONS_MAIN
    n_levels = scale_max - scale_min + 1
    result = {c: [[0] * n_levels for _ in item_keys] for c in conditions}
    for pid in participants:
        pdata = all_data.get(pid, {})
        for cond in conditions:
            for ki, key in enumerate(item_keys):
                score = extract_fn(pdata, cond, key)
                if score is not None:
                    idx = int(round(float(score))) - scale_min
                    if 0 <= idx < n_levels:
                        result[cond][ki][idx] += 1
    return result


def _build_raw(all_data, participants, item_keys, extract_fn, conditions):
    """Return {cond: [[score_or_None per participant] per item]}.

    Preserves per-participant scores for paired bootstrap CI computation.
    Participant order is the same across all conditions.
    """
    result = {c: [[] for _ in item_keys] for c in conditions}
    for pid in participants:
        pdata = all_data.get(pid, {})
        for cond in conditions:
            for ki, key in enumerate(item_keys):
                result[cond][ki].append(extract_fn(pdata, cond, key))
    return result


def _studentized_bootstrap_ci(diffs, B=9999, rng=None):
    """95 % studentized (bootstrap-t) CI for the mean of paired differences.

    Each bootstrap resample b gives:
        t*_b = (d̄*_b − d̄) / SE*_b
    The CI is then:
        [d̄ − q_0.975 · SE_obs ,  d̄ − q_0.025 · SE_obs]
    where q are empirical percentiles of the t* distribution.

    Second-order accurate — adapts to skewness and scale, recommended for n < 20.
    Returns (mean_diff, ci_lower, ci_upper).
    """
    d = np.asarray([x for x in diffs if x is not None], dtype=float)
    n = len(d)
    if n == 0:
        return 0.0, 0.0, 0.0
    if n == 1:
        return float(d[0]), float(d[0]), float(d[0])

    d_bar  = d.mean()
    se_obs = d.std(ddof=1) / np.sqrt(n)

    if se_obs == 0.0:
        return d_bar, d_bar, d_bar

    if rng is None:
        rng = np.random.default_rng(42)

    t_star = []
    for _ in range(B):
        resamp = rng.choice(d, size=n, replace=True)
        se_b   = resamp.std(ddof=1) / np.sqrt(n)
        if se_b > 0.0:
            t_star.append((resamp.mean() - d_bar) / se_b)

    if len(t_star) < 20:                       # degenerate → fall back to normal
        return d_bar, d_bar - 1.96 * se_obs, d_bar + 1.96 * se_obs

    t_arr = np.asarray(t_star)
    q025  = np.percentile(t_arr, 2.5)
    q975  = np.percentile(t_arr, 97.5)
    return d_bar, d_bar - q975 * se_obs, d_bar - q025 * se_obs


def _draw_diff_panel(ax, raw_by_cond, baseline, compare, n_items, rng,
                     inverted=False):
    """Horizontal mean-difference plot (compare − baseline) with 95 % CI.

    Dot colour:
      green   = CI entirely above 0 and effect is positive (compare > baseline for
                good scales, compare < baseline for inverted scales)
      red     = CI entirely below 0 and effect is negative
      grey    = CI crosses 0 (inconclusive)
    """
    means, lo_list, hi_list = [], [], []
    for ki in range(n_items):
        base_s = raw_by_cond[baseline][ki]
        comp_s = raw_by_cond[compare][ki]
        diffs = [float(c) - float(b)
                 for b, c in zip(base_s, comp_s) if b is not None and c is not None]
        m, lo, hi = _studentized_bootstrap_ci(diffs, rng=rng)
        means.append(m)
        lo_list.append(lo)
        hi_list.append(hi)

    for ki in range(n_items):
        m, lo, hi = means[ki], lo_list[ki], hi_list[ki]
        crosses = lo <= 0.0 <= hi
        if crosses:
            color = "#888888"
        elif bool(m > 0.0) ^ bool(inverted):   # XOR: pos is good iff not inverted
            color = "#006600"
        else:
            color = "#CC0000"
        ax.errorbar(m, ki,
                    xerr=[[max(0.0, m - lo)], [max(0.0, hi - m)]],
                    fmt="o", color=color, ecolor=color,
                    capsize=3, markersize=5, linewidth=1.5, zorder=4)

    all_abs = [abs(v) for lst in (means, lo_list, hi_list) for v in lst]
    x_max   = max(all_abs) * 1.3 if any(v > 0 for v in all_abs) else 1.0
    x_max   = max(x_max, 0.4)

    ax.set_xlim(-x_max, x_max)
    ax.axvline(0, color="black", linewidth=0.9, linestyle="--", zorder=5)
    ax.tick_params(labelleft=False)
    ax.set_title(f"{compare} − {baseline}", fontsize=9, fontweight="bold")
    ax.set_xlabel("Mean diff.\n(95 % stud. boot. CI)", fontsize=7)
    ax.grid(axis="both", linestyle=":", alpha=0.3)


def _make_combined_figure(suptitle, item_labels, counts_by_cond, raw_by_cond,
                           palette, scale_labels, conditions, diff_pairs,
                           inverted=False, label_width=42):
    """One-row figure: diverging Likert bars per condition + diff panels.

    Layout: [cond_0 bars] … [cond_n bars] [diff_pair_0] … [diff_pair_m]

    *conditions* : ordered list of conditions for the diverging-bar columns.
    *diff_pairs* : list of (reference, compare) tuples for the diff panels.
    *inverted*   : True when high raw score = bad outcome (NASA-TLX, Risk VAS).
    """
    n_items = len(item_labels)
    n_div   = len(conditions)
    n_diff  = len(diff_pairs)
    n_cols  = n_div + n_diff

    n_total = max(
        (sum(c) for cond_list in counts_by_cond.values() for c in cond_list),
        default=7,
    )

    width_ratios = [3] * n_div + [2] * n_diff
    fig_w = 3.2 * n_div + 2.2 * n_diff
    fig_h = max(5, n_items * 0.52 + 3.4)

    fig, axes = plt.subplots(
        1, n_cols, figsize=(fig_w, fig_h), sharey=True,
        gridspec_kw={"width_ratios": width_ratios, "wspace": 0.06},
    )
    axes = list(axes) if n_cols > 1 else [axes]
    fig.suptitle(suptitle, fontsize=11, fontweight="bold")

    # ── Diverging bar panels ──────────────────────────────────────────────────
    for k, cond in enumerate(conditions):
        ax = axes[k]
        for row, counts in enumerate(counts_by_cond.get(cond, [])):
            _draw_diverging_row(ax, row, counts, palette, inverted=inverted)

        ax.set_xlim(-n_total - 0.4, n_total + 0.4)
        ax.axvline(0, color="black", linewidth=1.0, zorder=5)
        ticks = list(range(-n_total, n_total + 1))
        ax.set_xticks(ticks)
        ax.set_xticklabels([str(abs(t)) for t in ticks], fontsize=7)
        ax.set_yticks(range(n_items))
        ax.set_title(cond, fontsize=10, fontweight="bold")
        ax.set_xlabel("← low     n     high →", fontsize=7)
        ax.grid(axis="x", linestyle=":", alpha=0.3)
        if k == 0:
            ax.set_yticklabels([l[:label_width] for l in item_labels], fontsize=8)
            ax.set_ylim(-0.5, n_items - 0.5)
            ax.invert_yaxis()   # shared y — inverts all panels
        else:
            ax.tick_params(labelleft=False)

    # ── Diff panels ───────────────────────────────────────────────────────────
    rng = np.random.default_rng(42)
    for j, (bl, comp) in enumerate(diff_pairs):
        _draw_diff_panel(axes[n_div + j], raw_by_cond, bl, comp,
                         n_items, rng, inverted=inverted)

    # ── Combined legend ───────────────────────────────────────────────────────
    pal_leg = list(reversed(palette)) if inverted else list(palette)
    lab_leg = list(reversed(scale_labels)) if inverted else list(scale_labels)
    scale_patches = [mpatches.Patch(color=c, label=l)
                     for c, l in zip(pal_leg, lab_leg)]
    diff_patches = [
        mpatches.Patch(color="#006600", label="CI > 0: favours right (compare)"),
        mpatches.Patch(color="#CC0000", label="CI < 0: favours left (reference)"),
        mpatches.Patch(color="#888888", label="CI crosses 0  (inconclusive)"),
    ]
    fig.legend(handles=scale_patches + diff_patches,
               loc="lower center", ncol=len(scale_patches) + len(diff_patches),
               fontsize=7.5, bbox_to_anchor=(0.5, 0))
    fig.tight_layout(rect=[0, 0.10, 1, 1])
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
#  SUS  (converted scores 0–4)
# ═══════════════════════════════════════════════════════════════════════════════
_SUS_KEYS = [f"sus_{i}" for i in range(1, 11)]
_SUS_LABELS = [
    "1(+) I would like to use this system frequently",
    "2(–) I found the system unnecessarily complex",
    "3(+) I thought the system was easy to use",
    "4(–) I would need support of a technical person",
    "5(+) The various functions were well integrated",
    "6(–) There was too much inconsistency",
    "7(+) Most people would learn to use this quickly",
    "8(–) I found the system very cumbersome to use",
    "9(+) I felt very confident using the system",
    "10(–) I needed to learn a lot before I could use it",
]


def plot_sus(all_data, participants):
    def extract(pdata, cond, key):
        try:
            return pdata["sus"]["conditions"][cond]["items"][key]["converted"]
        except (KeyError, TypeError):
            return None

    counts = _build_counts(all_data, participants, _SUS_KEYS, extract, 0, 4)
    raw    = _build_raw(all_data, participants, _SUS_KEYS, extract, CONDITIONS_MAIN)
    fig = _make_combined_figure(
        "SUS — Item Response Distribution  (converted 0–4 · TARS is reference)",
        _SUS_LABELS, counts, raw, COLORS_5,
        ["0 — strongly disagree", "1", "2 — neutral", "3", "4 — strongly agree"],
        conditions=CONDITIONS_MAIN, diff_pairs=_ALL_PAIRS,
    )
    save_fig(fig, "sus_items.png")
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
#  TiA  (recoded 1–5, grouped by subscale)
# ═══════════════════════════════════════════════════════════════════════════════
_TIA_SPECS = [
    # (subscale_key, item_key, label)
    ("reliability_competence",       "kta_01", "[RC]  1(+) Interprets situations correctly"),
    ("reliability_competence",       "kta_03", "[RC]  6(+) Works reliably"),
    ("reliability_competence",       "kta_06", "[RC] 10(–) System malfunction likely  *"),
    ("reliability_competence",       "kta_08", "[RC] 13(+) Can take over complex tasks"),
    ("reliability_competence",       "kta_10", "[RC] 15(–) Sporadic errors likely  *"),
    ("reliability_competence",       "kta_12", "[RC] 19(+) Confident about capabilities"),
    ("understanding_predictability", "kta_02", "[UP]  2(+) System state always clear"),
    ("understanding_predictability", "kta_04", "[UP]  7(–) Reacts unpredictably  *"),
    ("understanding_predictability", "kta_07", "[UP] 11(+) Understood why things happened"),
    ("understanding_predictability", "kta_11", "[UP] 16(–) Difficult to predict next action  *"),
    ("trust_in_automation",          "kta_05", "[TA]  9(+) I trust the system"),
    ("trust_in_automation",          "kta_09", "[TA] 14(+) I can rely on the system"),
]
_TIA_KEYS   = [s for _, s, _ in _TIA_SPECS]
_TIA_LABELS = [l for _, _, l in _TIA_SPECS]
_TIA_SUB_MAP = {key: sub for sub, key, _ in _TIA_SPECS}


def plot_tia(all_data, participants):
    def extract(pdata, cond, key):
        sub = _TIA_SUB_MAP.get(key)
        if sub is None:
            return None
        try:
            return pdata["tia"]["conditions"][cond]["subscales"][sub]["items"][key]["recoded"]
        except (KeyError, TypeError):
            return None

    counts = _build_counts(all_data, participants, _TIA_KEYS, extract, 1, 5)
    raw    = _build_raw(all_data, participants, _TIA_KEYS, extract, CONDITIONS_MAIN)
    fig = _make_combined_figure(
        "TiA — Item Response Distribution  (recoded 1–5 · * items already inverted · TARS is reference)",
        _TIA_LABELS, counts, raw, COLORS_5,
        ["1 — strongly disagree", "2", "3 — neutral", "4", "5 — strongly agree"],
        conditions=CONDITIONS_MAIN, diff_pairs=_ALL_PAIRS,
    )
    save_fig(fig, "tia_items.png")
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
#  Oversight Bespoke  (scored 1–5)
# ═══════════════════════════════════════════════════════════════════════════════
_OB_KEYS = ["ob_a2", "ob_a3", "ob_b1", "ob_b2", "ob_b3",
            "ob_c1", "ob_c2", "ob_d1", "ob_d2"]
_OB_LABELS = [
    "A2(+) Enough time / attention to oversee system",
    "A3(–) Difficult to follow system actions  *",
    "B1(+) Actively verified system's actions",
    "B2(+) No need to monitor closely  [ext]",
    "B3(–) May have missed check opportunities  *",
    "C1(+) Could detect system errors quickly",
    "C2(+) Enough time to react to actions",
    "D1(+) System helped work efficiently  [ext]",
    "D2(–) Efficiency came at cost of oversight  * [ext]",
]


def plot_ob(all_data, participants):
    def extract(pdata, cond, key):
        try:
            return pdata["oversight_bespoke"]["conditions"][cond]["extended_score"]["items"][key]["scored"]
        except (KeyError, TypeError):
            return None

    counts = _build_counts(all_data, participants, _OB_KEYS, extract, 1, 5)
    raw    = _build_raw(all_data, participants, _OB_KEYS, extract, CONDITIONS_MAIN)
    fig = _make_combined_figure(
        "Oversight Bespoke — Item Response Distribution  (scored 1–5 · * reversed · TARS is reference)",
        _OB_LABELS, counts, raw, COLORS_5,
        ["1 — strongly disagree", "2", "3 — neutral", "4", "5 — strongly agree"],
        conditions=CONDITIONS_MAIN, diff_pairs=_ALL_PAIRS,
    )
    save_fig(fig, "ob_items.png")
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
#  NASA-TLX  (0–20, binned into 5 levels, INVERTED: high = more workload = red)
# ═══════════════════════════════════════════════════════════════════════════════
_NASA_BINS   = [(0, 3), (4, 7), (8, 11), (12, 15), (16, 20)]
_NASA_KEYS   = ["mental_demand", "physical_demand", "temporal_demand",
                "performance", "effort", "frustration"]
_NASA_LABELS = ["Mental Demand", "Physical Demand", "Temporal Demand",
                "Performance", "Effort", "Frustration"]


def _nasa_bin(val):
    for idx, (lo, hi) in enumerate(_NASA_BINS):
        if lo <= val <= hi:
            return idx + 1   # 1-indexed
    return None


def plot_nasa(all_data, participants):
    def extract(pdata, cond, key):
        try:
            raw = pdata["nasa_tlx"]["conditions"][cond]["subscales"][key]["rating_0_20"]
            return _nasa_bin(raw)
        except (KeyError, TypeError):
            return None

    # NASA-TLX keeps all 4 conditions (including TARC) per user request
    counts = _build_counts(all_data, participants, _NASA_KEYS, extract, 1, 5,
                           conditions=CONDITIONS)
    raw    = _build_raw(all_data, participants, _NASA_KEYS, extract, CONDITIONS)
    bin_labels = [f"{lo}–{hi}" for lo, hi in _NASA_BINS]
    fig = _make_combined_figure(
        "NASA-TLX — Rating Distribution  (0–20 binned · high workload = left/red · TARS is reference)",
        _NASA_LABELS, counts, raw, COLORS_5, bin_labels,
        conditions=CONDITIONS, diff_pairs=_NASA_PAIRS,
        inverted=True,
    )
    save_fig(fig, "nasa_tlx_items.png")
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
#  Trust & Risk VAS  (0–100, binned into 5 levels)
#  Trust : high = good  (normal orientation)
#  Risk  : high = bad   (inverted)
# ═══════════════════════════════════════════════════════════════════════════════
_VAS_BINS = [(0, 19), (20, 39), (40, 59), (60, 79), (80, 100)]


def _vas_bin(val):
    for idx, (lo, hi) in enumerate(_VAS_BINS):
        if lo <= val <= hi:
            return idx + 1
    return None


def plot_trust_risk(all_data, participants):
    figs = []
    for key, title, inverted in [
        ("trust_vas", "Trust VAS  (0–100 binned · high trust = right/green · TARS is reference)", False),
        ("risk_vas",  "Perceived Risk VAS  (0–100 binned · high risk = left/red · TARS is reference)",  True),
    ]:
        k = key  # capture for closure
        def extract(pdata, cond, item_key, _k=k):
            try:
                raw = pdata["trust_risk"]["conditions"][cond][_k]
                return _vas_bin(raw)
            except (KeyError, TypeError):
                return None

        counts = _build_counts(all_data, participants, [key], extract, 1, 5)
        raw    = _build_raw(all_data, participants, [key], extract, CONDITIONS_MAIN)
        bin_labels = [f"{lo}–{hi}" for lo, hi in _VAS_BINS]
        item_label = [title.split("(")[0].strip()]
        fig = _make_combined_figure(
            title, item_label, counts, raw, COLORS_5, bin_labels,
            conditions=CONDITIONS_MAIN, diff_pairs=_ALL_PAIRS,
            inverted=inverted,
        )
        fname = "trust_vas_items.png" if not inverted else "risk_vas_items.png"
        save_fig(fig, fname)
        figs.append(fig)
    return figs


# ═══════════════════════════════════════════════════════════════════════════════
#  PTS  (trait — no condition, one diverging chart for all participants)
# ═══════════════════════════════════════════════════════════════════════════════
_PTS_KEYS   = ["Q1", "Q2", "Q3", "Q4", "Q5", "Q6", "Q7"]
_PTS_LABELS = [
    "Q1(+) Automation is generally reliable",
    "Q2(–) Humans outperform automation  [inv]",
    "Q3(+) I trust new technologies quickly",
    "Q4(+) I prefer automated solutions",
    "Q5(+) Automation is generally trustworthy",
    "Q6(+) I rely on automation for decisions",
    "Q7(+) Automation handles complex tasks well",
]


def plot_pts(all_data, participants):
    n = len(_PTS_KEYS)
    n_participants = len(participants)

    # Collect counts (trait — ignore conditions)
    counts_list = [[0] * 5 for _ in _PTS_KEYS]
    for pid in participants:
        pdata = all_data.get(pid, {})
        for ki, key in enumerate(_PTS_KEYS):
            try:
                score = pdata["pre_experiment_form"]["pts"][key]["score"]
                idx = int(round(float(score))) - 1
                if 0 <= idx < 5:
                    counts_list[ki][idx] += 1
            except (KeyError, TypeError):
                pass

    fig_h = max(4, n * 0.55 + 2.5)
    fig, ax = plt.subplots(figsize=(8, fig_h))
    fig.suptitle("PTS — Propensity to Trust Automation  (scored 1–5 · Q2 already inverted)",
                 fontsize=12, fontweight="bold")

    for row, counts in enumerate(counts_list):
        _draw_diverging_row(ax, row, counts, COLORS_5)

    ax.set_xlim(-n_participants - 0.4, n_participants + 0.4)
    ax.axvline(0, color="black", linewidth=1.0, zorder=5)
    ticks = list(range(-n_participants, n_participants + 1))
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(abs(t)) for t in ticks], fontsize=8)
    ax.set_yticks(range(n))
    ax.set_yticklabels(_PTS_LABELS, fontsize=9)
    ax.set_ylim(-0.5, n - 0.5)
    ax.invert_yaxis()
    ax.set_xlabel("← low / disagree     n     agree / high →", fontsize=8)
    ax.grid(axis="x", linestyle=":", alpha=0.3)
    ax.set_title("All participants (trait constant — no condition)", fontsize=9)

    patches = [mpatches.Patch(color=c, label=l) for c, l in
               zip(COLORS_5, ["1 — strongly disagree", "2", "3 — neutral", "4", "5 — strongly agree"])]
    fig.legend(handles=patches, loc="lower center", ncol=5,
               fontsize=8, bbox_to_anchor=(0.5, 0))
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    save_fig(fig, "pts_items.png")
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
#  BOX PLOTS — Computed scores per condition
# ═══════════════════════════════════════════════════════════════════════════════

def _get(d, *keys):
    for k in keys:
        if not isinstance(d, dict):
            return None
        d = d.get(k)
    return d


def _collect(all_data, participants, extract_fn, conditions=None):
    if conditions is None:
        conditions = CONDITIONS
    result = {c: [] for c in conditions}
    for pid in participants:
        pdata = all_data.get(pid, {})
        vals  = extract_fn(pdata)
        for cond in conditions:
            v = vals.get(cond)
            if v is not None:
                result[cond].append(float(v))
    return result


def _box_panel(ax, data_by_cond, title, ylabel, ylim=None, conditions=None):
    if conditions is None:
        conditions = CONDITIONS
    positions = list(range(len(conditions)))
    boxes     = [data_by_cond.get(c, []) for c in conditions]

    bp = ax.boxplot(boxes, positions=positions, widths=0.5, patch_artist=True,
                    medianprops=dict(color="black", linewidth=2),
                    flierprops=dict(marker="o", markersize=4, alpha=0.5))
    for patch, cond in zip(bp["boxes"], conditions):
        patch.set_facecolor(COND_COLOR[cond])
        patch.set_alpha(0.65)

    rng = np.random.default_rng(42)
    for i, (cond, vals) in enumerate(zip(conditions, boxes)):
        if vals:
            jitter = rng.uniform(-0.15, 0.15, len(vals))
            ax.scatter(np.array([i] * len(vals)) + jitter, vals,
                       color="black", s=20, zorder=5, alpha=0.75)

    ax.set_xticks(positions)
    ax.set_xticklabels(conditions, fontsize=8)
    ax.set_title(title, fontsize=9, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=8)
    ax.grid(axis="y", linestyle="--", alpha=0.45)
    if ylim:
        ax.set_ylim(ylim)


def plot_boxplots(all_data, participants):
    # Each entry: (title, ylabel, ylim, extract_fn, conditions)
    metrics = [
        (
            "SUS Score", "Score (0–100)", (0, 100),
            lambda d, _c=CONDITIONS_MAIN: {c: _get(d, "sus", "conditions", c, "sus_score") for c in _c},
            CONDITIONS_MAIN,
        ),
        (
            "TiA — Global mean (excl. traits)", "Mean (1–5)", (1, 5),
            lambda d, _c=CONDITIONS_MAIN: {c: _get(d, "tia", "conditions", c, "global_mean_excl_traits") for c in _c},
            CONDITIONS_MAIN,
        ),
        (
            "TiA — Reliability / Competence", "Mean (1–5)", (1, 5),
            lambda d, _c=CONDITIONS_MAIN: {c: _get(d, "tia", "conditions", c,
                               "subscales", "reliability_competence", "mean") for c in _c},
            CONDITIONS_MAIN,
        ),
        (
            "TiA — Understanding / Predictability", "Mean (1–5)", (1, 5),
            lambda d, _c=CONDITIONS_MAIN: {c: _get(d, "tia", "conditions", c,
                               "subscales", "understanding_predictability", "mean") for c in _c},
            CONDITIONS_MAIN,
        ),
        (
            "TiA — Trust in Automation", "Mean (1–5)", (1, 5),
            lambda d, _c=CONDITIONS_MAIN: {c: _get(d, "tia", "conditions", c,
                               "subscales", "trust_in_automation", "mean") for c in _c},
            CONDITIONS_MAIN,
        ),
        (
            "OB Base Score", "Mean (1–5)", (1, 5),
            lambda d, _c=CONDITIONS_MAIN: {c: _get(d, "oversight_bespoke", "conditions", c, "base_score", "mean") for c in _c},
            CONDITIONS_MAIN,
        ),
        (
            "OB Extended Score", "Mean (1–5)", (1, 5),
            lambda d, _c=CONDITIONS_MAIN: {c: _get(d, "oversight_bespoke", "conditions", c, "extended_score", "mean") for c in _c},
            CONDITIONS_MAIN,
        ),
        (
            "NASA-TLX Weighted Score", "Score (0–100)", (0, 100),
            lambda d, _c=CONDITIONS: {c: _get(d, "nasa_tlx", "conditions", c, "nasa_tlx_weighted_score") for c in _c},
            CONDITIONS,
        ),
        (
            "Trust VAS", "Score (0–100)", (0, 100),
            lambda d, _c=CONDITIONS_MAIN: {c: _get(d, "trust_risk", "conditions", c, "trust_vas") for c in _c},
            CONDITIONS_MAIN,
        ),
        (
            "Risk VAS", "Score (0–100)", (0, 100),
            lambda d, _c=CONDITIONS_MAIN: {c: _get(d, "trust_risk", "conditions", c, "risk_vas") for c in _c},
            CONDITIONS_MAIN,
        ),
    ]

    ncols  = 2
    nrows  = (len(metrics) + 1) // 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, nrows * 4))
    fig.suptitle(
        "Computed Scores — Box Plots across Conditions\n",
        fontsize=11, fontweight="bold",
    )
    axes_flat = axes.flatten()

    for i, (title, ylabel, ylim, fn, conds) in enumerate(metrics):
        _box_panel(axes_flat[i], _collect(all_data, participants, fn, conds),
                   title, ylabel, ylim, conditions=conds)
    for j in range(len(metrics), len(axes_flat)):
        axes_flat[j].set_visible(False)

    patches = [mpatches.Patch(color=COND_COLOR[c], label=c) for c in CONDITIONS]
    fig.legend(handles=patches, loc="lower right", fontsize=9, title="Condition")
    plt.tight_layout()
    save_fig(fig, "boxplots_scores.png")
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("  HITLS — Cross-participant Questionnaire Comparison")
    print("=" * 70)

    participants = find_participants()
    print(f"\nParticipants found: {', '.join(participants)}")

    print("\n[1/3] Checking / generating participant reports …")
    for i, pid in enumerate(participants, start=1):
        if has_all_reports(pid):
            print(f"  {pid}: ✓ reports already exist")
        else:
            run_forms(pid, i)

    print("\n[2/3] Loading data …")
    all_data = load_all(participants)
    loaded   = list(all_data.keys())
    print(f"  Loaded: {', '.join(loaded)}")

    os.makedirs(PLOTS_DIR, exist_ok=True)
    print(f"\n[3/3] Generating charts → {PLOTS_DIR}/\n")

    figs = []
    figs.append(plot_sus(all_data, loaded))
    figs.append(plot_tia(all_data, loaded))
    figs.append(plot_ob(all_data, loaded))
    figs.append(plot_nasa(all_data, loaded))
    figs.ext= [mpatches.Patch(color=COND_COLOR[c], label=c) for c in CONDITIONS]
    fig.legend(handles=patches, loc="lower right", fontsize=9, title="Condition")
    plt.tight_layout()
    save_fig(fig, "boxplots_scores.png")
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("  HITLS — Cross-participant Questionnaire Comparison")
    print("=" * 70)

    participants = find_participants()
    print(f"\nParticipants found: {', '.join(participants)}")

    print("\n[1/3] Checking / generating participant reports …")
    for i, pid in enumerate(participants, start=1):
        if has_all_reports(pid):
            print(f"  {pid}: ✓ reports already exist")
        else:
            run_forms(pid, i)

    print("\n[2/3] Loading data …")
    all_data = load_all(participants)
    loaded   = list(all_data.keys())
    print(f"  Loaded: {', '.join(loaded)}")

    os.makedirs(PLOTS_DIR, exist_ok=True)
    print(f"\n[3/3] Generating charts → {PLOTS_DIR}/\n")

    figs = []
    figs.append(plot_sus(all_data, loaded))
    figs.append(plot_tia(all_data, loaded))
    figs.append(plot_ob(all_data, loaded))
    figs.append(plot_nasa(all_data, loaded))
    figs.extend(plot_trust_risk(all_data, loaded))
    figs.append(plot_pts(all_data, loaded))
    figs.append(plot_boxplots(all_data, loaded))

    print(f"\nDone — {len(figs)} figures saved to {PLOTS_DIR}/")
    plt.show()


if __name__ == "__main__":
    main()
