#!/usr/bin/env python3
"""
HITLS — Cross-participant questionnaire comparison
==================================================
Compare-type script: operates on ALL participants at once.
Run from the repository root:
    python HITLS/compare_forms.py

The script will show a summary of what will be generated or overwritten and
ask for confirmation before doing any work.

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

import os, sys, json, subprocess, glob, textwrap
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import gaussian_kde
from scipy.stats import friedmanchisquare, wilcoxon as _sp_wilcoxon
from statsmodels.stats.multitest import multipletests

# ── Paths ────────────────────────────────────────────────────────────────────
HITLS_DIR    = os.path.dirname(os.path.abspath(__file__))
FORMS_SCRIPT = os.path.join(HITLS_DIR, "forms", "forms.py")
PLOTS_DIR    = os.path.join(HITLS_DIR, "plots")
PYTHON       = sys.executable

# ── Experiment constants ─────────────────────────────────────────────────────
CONDITIONS      = ["TARS", "TARC", "TARP-S", "TARP-F"]   # all four (used by NASA + box plots)
CONDITIONS_MAIN = ["TARS", "TARC", "TARP-S", "TARP-F"]   # all four questionnaire conditions
_BASELINE    = "TARS"
_COMPARISONS = ["TARC", "TARP-S", "TARP-F"]   # conditions compared vs _BASELINE
# Diff pairs: (reference, compare) — used for the diff panels
_ALL_PAIRS  = [("TARS", "TARP-S"), ("TARS", "TARP-F"), ("TARP-S", "TARP-F"), ("TARS", "TARC")]
_NASA_PAIRS = [("TARS", "TARC"),   ("TARS", "TARP-S"), ("TARS", "TARP-F"), ("TARP-S", "TARP-F")]

COND_COLOR = {
    "TARS":   "#4472C4",
    "TARC":   "#ED7D31",
    "TARP-S": "#70AD47",
    "TARP-F": "#C00000",
}

# Per-pair visual styles (color, marker) — index aligned with _ALL_PAIRS order
# TARP-S − TARS | TARP-F − TARS | TARP-F − TARP-S | TARC − TARS
_PAIR_STYLES = [
    ("#70AD47", "o"),   # TARP-S − TARS   (green  / circle)
    ("#C00000", "s"),   # TARP-F − TARS   (red    / square)
    ("#7030A0", "^"),   # TARP-F − TARP-S (purple / triangle)
    ("#ED7D31", "D"),   # TARC  − TARS    (orange / diamond)
]

# ── Likert colour palettes ───────────────────────────────────────────────────
#   Index 0  →  lowest score  →  dark orange (strongly disagree)
#   Index -1 →  highest score →  dark blue   (strongly agree)
COLORS_5 = [
    "#C44E00",   # 0  strongly disagree  (dark orange)
    "#E6730D",   # 1  disagree           (orange)
    "#D0D0D0",   # 2  neutral            (light grey)
    "#93C4DE",   # 3  agree              (light blue)
    "#2E75B6",   # 4  strongly agree     (dark blue)
]

COLORS_7 = [
    "#8B3300",   # 0  strongly disagree  (very dark orange)
    "#C44E00",   # 1                      (dark orange)
    "#E6730D",   # 2                      (orange)
    "#D0D0D0",   # 3  neutral            (light grey)
    "#93C4DE",   # 4                      (light blue)
    "#5A9EC5",   # 5                      (medium blue)
    "#2E75B6",   # 6  strongly agree     (dark blue)
]

# ── Visual parameters — edit these to tune the appearance ────────────────────
# Output
FIGURE_DPI           = 150      # saved figure resolution (DPI)

# Stacked bar charts (all Likert / bespoke questionnaires)
SBAR_HEIGHT          = 0.72     # bar height in data units  (gap = 1 − height)
SBAR_ITEM_PITCH      = 1.0      # y-spacing between bar centres; increase for more gap between rows
SBAR_COUNT_FONTSIZE  = 7.5      # font size of count labels inside bar segments
SBAR_LABEL_WIDTH     = 42       # max chars per line when wrapping y-axis item labels
SBAR_COL_RATIO       = 3        # relative width of each condition column
SBAR_DIFF_RATIO      = 2.5      # relative width of the consolidated diff panel
SBAR_WSPACE          = 0.06     # horizontal spacing between subplots (fraction)
SBAR_FIGW_PER_COL    = 3.2      # figure width added per condition column (inches)
SBAR_FIGW_OFFSET     = 2.8      # fixed extra width for the diff panel (inches)
SBAR_FIGH_PER_ITEM   = 0.52     # figure height per item row (inches)
SBAR_FIGH_MULTILINE  = 0.22     # extra height per additional wrapped label line (inches)
SBAR_FIGH_MIN        = 5.0      # minimum figure height (inches)
SBAR_FIGH_OFFSET     = 3.4      # fixed top/bottom figure height margin (inches)

# NASA-TLX strip-plot  (Figure 1 of plot_nasa)
NASA_STRIP_FIGW      = 12       # figure width (inches)
NASA_STRIP_H_PER_DIM = 2.6      # height per dimension subplot (inches)
NASA_STRIP_H_OFFSET  = 1.8      # fixed top/bottom margin (inches)
NASA_KDE_HALF        = 0.34     # KDE curve max half-height (data units)
NASA_JITTER          = 0.22     # ±vertical jitter for raw dots (data units)
NASA_DOT_SIZE        = 40       # raw dot marker area (pt²)
NASA_MEAN_SIZE       = 130      # mean diamond marker area (pt²)
NASA_MEAN_HALF       = 0.44     # mean label vertical offset above row centre (data units)
NASA_MEDIAN_HALF     = 0.36     # median tick half-height (data units)
NASA_MEDIAN_OFFSET   = 0.24     # median label vertical offset below row centre (data units)
NASA_MIDPT_LW        = 1.6      # midpoint dashed line width
NASA_SCORES_FIGW     = 7        # figure width for weighted score box plot (inches)
NASA_SCORES_FIGH     = 5        # figure height for weighted score box plot (inches)

# Box plots
BOX_WIDTH            = 0.5      # box width (data units)
BOX_JITTER           = 0.15     # ±horizontal jitter for raw dots (data units)
BOX_DOT_SIZE         = 20       # raw dot marker area (pt²)
BOX_ALPHA            = 0.65     # box fill alpha
BOX_HEIGHT_PER_ROW   = 4        # figure height per subplot row (inches)
BOX_FIGW             = 12       # box-plot grid figure width (inches)

# Trust/Risk continuous distribution
DIST_FIGW            = 12       # figure width (inches)
DIST_FIGH            = 5        # figure height (inches)
DIST_HIST_BINS       = 10       # number of histogram bins
DIST_HIST_ALPHA      = 0.20     # histogram fill alpha
DIST_KDE_LW          = 2.0      # KDE curve line width
DIST_CI_STRIP_FRAC   = 0.28     # fraction of plot height reserved for the CI strip below x-axis

# Diff / forest-plot panels
DIFF_CAPSIZE         = 3        # error-bar cap length (pt)
DIFF_MARKER_SIZE     = 5        # dot diameter for single-pair diff panel (pt)
DIFF_DOT_SIZE        = 28       # dot area for multi-pair diff panel (pt²)
DIFF_OFFSET_RANGE    = 0.28     # ±vertical offset spread between diff pairs (data units; keep < 0.5 × SBAR_ITEM_PITCH)
DIFF_CAP_HALF        = 0.09     # CI cap tick half-height (data units)

# PTS trait chart (pre-experiment)
PTS_FIGW             = 9        # figure width (inches)
PTS_LABEL_WIDTH      = 40       # max chars per line when wrapping PTS item labels
PTS_FIGH_PER_ITEM    = 0.55     # height per item (inches)
PTS_FIGH_OFFSET      = 2.5      # fixed margin (inches)
PTS_FIGH_MIN         = 4.0      # minimum figure height (inches)

# Preference cluster analysis figure
CLUSTER_FIGW         = 13       # figure width (inches)
CLUSTER_FIGH         = 5.5      # figure height (inches)

# Preference profile coherence bump chart
COHERENCE_FIGW       = 11       # figure width (inches)
COHERENCE_FIGH       = 5.5      # figure height (inches)

# Typography  (font sizes in pt)
FONT_SUPTITLE        = 11       # top-level figure title
FONT_TITLE           = 10       # panel / condition subtitle
FONT_LABEL           = 9        # axis labels and larger annotations
FONT_TICK            = 8        # tick labels and legend entries
FONT_TICK_SM         = 7        # dense tick labels (x-ticks in stacked bars / diff xlabel)
FONT_ANNOTATION      = 6        # in-plot text (mean / median value labels)

# ═══════════════════════════════════════════════════════════════════════════════
#  Low-level helpers
# ═══════════════════════════════════════════════════════════════════════════════

def find_participants():
    dirs = sorted(glob.glob(os.path.join(HITLS_DIR, "P*")))
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
            "oversight_bespoke", "perceived_control", "pre_experiment_form")
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
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    print(f"  Saved → {path}")


# ═══════════════════════════════════════════════════════════════════════════════
#  Stacked bar scaffold
# ═══════════════════════════════════════════════════════════════════════════════

# Light colours where white text would be invisible → use black text instead
_LIGHT_COLORS = {"#D0D0D0", "#FFD700", "#FFFFFF"}


def _draw_stacked_row(ax, y, counts, palette, inverted=False, height=SBAR_HEIGHT):
    """Draw one left-to-right stacked bar at vertical position *y*.

    *counts* : list of ints, index 0 = strongly disagree (orange end),
               index -1 = strongly agree (blue end).
    Stacks from x=0 rightward; segment width = count.
    Count labels are printed inside each segment (white or black by luminance).
    """
    cnt = list(counts)
    pal = list(palette)
    if inverted:
        cnt = cnt[::-1]
        pal = pal[::-1]

    cursor = 0.0
    for i, w in enumerate(cnt):
        if w > 0:
            ax.barh(y, w, left=cursor, height=height, color=pal[i],
                    edgecolor="white", linewidth=0.5)
            txt_color = "black" if pal[i] in _LIGHT_COLORS else "white"
            ax.text(cursor + w / 2, y, str(int(w)),
                    ha="center", va="center", fontsize=SBAR_COUNT_FONTSIZE,
                    color=txt_color, fontweight="bold")
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


# ── Statistical testing helpers ───────────────────────────────────────────────

def _wilcoxon_test(a, b):
    """Paired Wilcoxon signed-rank test. Returns (p_value, rank_biserial_r).
    Drops pairs where either value is None.  Minimum 4 valid pairs required.
    """
    pairs = [(float(x), float(y)) for x, y in zip(a, b)
             if x is not None and y is not None]
    if len(pairs) < 4:
        return 1.0, 0.0
    xa, xb = np.array([p[0] for p in pairs]), np.array([p[1] for p in pairs])
    try:
        res = _sp_wilcoxon(xa, xb, alternative="two-sided", zero_method="wilcox")
        n   = len(pairs)
        # rank-biserial effect size: r = 1 - 2W / (n*(n+1)/2)
        r   = 1.0 - 2.0 * res.statistic / (n * (n + 1) / 2.0)
        return float(res.pvalue), float(r)
    except Exception:
        return 1.0, 0.0


def _friedman_test(groups):
    """Friedman chi-square for k groups (list of arrays/lists, equal length).
    Drops participants (rows) with any None across conditions.
    Returns (chi2, p).
    """
    rows = list(zip(*groups))
    valid = [r for r in rows if all(x is not None for x in r)]
    if len(valid) < 3:
        return 0.0, 1.0
    aligned = [[float(r[i]) for r in valid] for i in range(len(groups))]
    try:
        res = friedmanchisquare(*aligned)
        return float(res.statistic), float(res.pvalue)
    except Exception:
        return 0.0, 1.0


def _holm_correct(pvals, alpha=0.05):
    """Holm-Bonferroni correction via statsmodels multipletests.
    Returns (reject_array, pvals_corrected).
    """
    pvals = list(pvals)
    if not pvals:
        return np.array([], dtype=bool), np.array([])
    reject, pvals_corr, _, _ = multipletests(pvals, alpha=alpha, method="holm")
    return reject, pvals_corr


def _sig_stars(p_raw, sig):
    """Return significance label string (based on raw p and Holm rejection)."""
    if not sig:
        return ""
    if p_raw < 0.001:
        return "***"
    if p_raw < 0.01:
        return "**"
    if p_raw < 0.05:
        return "*"
    return "†"   # Holm-rejected but raw p >= .05 (shouldn't happen, kept for safety)


def _compute_pairwise_stats(raw_by_cond, diff_pairs, n_items):
    """Compute Wilcoxon pairwise tests with Holm correction within each item.

    Returns pairwise_by_item[ki][pi] = {
        'p_raw': float, 'r': float, 'reject': bool, 'p_corr': float, 'stars': str
    }
    """
    pairwise = [[None] * len(diff_pairs) for _ in range(n_items)]
    for ki in range(n_items):
        raw_ps, raw_rs = [], []
        for bl, comp in diff_pairs:
            p, r = _wilcoxon_test(raw_by_cond[bl][ki], raw_by_cond[comp][ki])
            raw_ps.append(p)
            raw_rs.append(r)
        reject, p_corr = _holm_correct(raw_ps)
        for pi in range(len(diff_pairs)):
            pairwise[ki][pi] = {
                "p_raw":  raw_ps[pi],
                "r":      raw_rs[pi],
                "reject": bool(reject[pi]),
                "p_corr": float(p_corr[pi]),
                "stars":  _sig_stars(raw_ps[pi], bool(reject[pi])),
            }
    return pairwise


def _compute_friedman_stats(raw_by_cond, n_items, conditions):
    """Compute Friedman test per item across all conditions.

    Returns friedman_by_item[ki] = {'chi2': float, 'p': float, 'df': int}
    """
    friedman = []
    for ki in range(n_items):
        groups = [raw_by_cond[c][ki] for c in conditions]
        chi2, p = _friedman_test(groups)
        friedman.append({"chi2": chi2, "p": p, "df": len(conditions) - 1})
    return friedman


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
                    capsize=DIFF_CAPSIZE, markersize=DIFF_MARKER_SIZE, linewidth=1.5, zorder=4)

    all_abs = [abs(v) for lst in (means, lo_list, hi_list) for v in lst]
    x_max   = max(all_abs) * 1.3 if any(v > 0 for v in all_abs) else 1.0
    x_max   = max(x_max, 0.4)

    ax.set_xlim(-x_max, x_max)
    ax.axvline(0, color="black", linewidth=0.9, linestyle="--", zorder=5)
    ax.tick_params(labelleft=False)
    ax.set_title(f"{compare} − {baseline}", fontsize=FONT_TITLE, fontweight="bold")
    ax.set_xlabel("Mean diff.\n(95 % stud. boot. CI)", fontsize=FONT_TICK_SM)
    ax.grid(axis="both", linestyle=":", alpha=0.3)


def _draw_multi_diff_panel(ax, raw_by_cond, diff_pairs, n_items, rng,
                           inverted=False, pairwise_stats=None):
    """Single consolidated forest-plot panel: all comparison pairs in one column.

    For each item row, pairs are offset vertically (±0.28 spacing).
    Filled marker = CI entirely on one side (reliable effect).
    Open  marker  = CI crosses 0 (inconclusive).
    When pairwise_stats is provided, significance stars are annotated.
    """
    n_pairs = len(diff_pairs)
    offsets = np.linspace(-DIFF_OFFSET_RANGE, DIFF_OFFSET_RANGE, n_pairs) if n_pairs > 1 else [0.0]
    cap     = DIFF_CAP_HALF
    all_ext = []

    for pi, (bl, comp) in enumerate(diff_pairs):
        color, marker = _PAIR_STYLES[pi % len(_PAIR_STYLES)]
        offset = offsets[pi]

        for ki in range(n_items):
            base_s = raw_by_cond[bl][ki]
            comp_s = raw_by_cond[comp][ki]
            diffs  = [float(c) - float(b)
                      for b, c in zip(base_s, comp_s)
                      if b is not None and c is not None]
            m, lo, hi = _studentized_bootstrap_ci(diffs, rng=rng)
            y_pos   = ki * SBAR_ITEM_PITCH + offset
            crosses = lo <= 0.0 <= hi
            alpha   = 0.35 if crosses else 0.90

            # CI bar + caps
            ax.plot([lo, hi], [y_pos, y_pos],
                    color=color, linewidth=1.5, alpha=alpha, zorder=3)
            for xc in (lo, hi):
                ax.plot([xc, xc], [y_pos - cap, y_pos + cap],
                        color=color, linewidth=1.2, alpha=alpha, zorder=3)

            # dot: filled = significant, open = inconclusive
            if crosses:
                ax.scatter(m, y_pos, color="white", edgecolors=color,
                           marker=marker, s=DIFF_DOT_SIZE, linewidths=1.2, zorder=5)
            else:
                ax.scatter(m, y_pos, color=color, edgecolors="black",
                           marker=marker, s=DIFF_DOT_SIZE, linewidths=0.5, zorder=5)

            # significance stars from Wilcoxon + Holm correction
            if pairwise_stats is not None:
                stars = pairwise_stats[ki][pi]["stars"]
                if stars:
                    ax.text(hi, y_pos, f" {stars}", va="center", ha="left",
                            fontsize=6, color=color, zorder=7)

            all_ext.extend([abs(lo), abs(hi), abs(m)])

    x_max = max(all_ext) * 1.3 if all_ext else 1.0
    x_max = max(x_max, 0.5)
    ax.set_xlim(-x_max, x_max)
    ax.axvline(0, color="black", linewidth=1.0, linestyle="--", zorder=6)
    ax.tick_params(labelleft=False)
    ax.set_title("Δ mean  (95 % boot. CI)", fontsize=FONT_TITLE, fontweight="bold")
    ax.set_xlabel("compare − reference", fontsize=FONT_TICK_SM)
    ax.grid(axis="x", linestyle=":", alpha=0.30)
    for ki in range(n_items):
        ax.axhline(ki * SBAR_ITEM_PITCH - 0.5 * SBAR_ITEM_PITCH,
                   color="#dddddd", linewidth=0.4, zorder=1)


def _make_combined_figure(suptitle, item_labels, counts_by_cond, raw_by_cond,
                           palette, scale_labels, conditions, diff_pairs,
                           inverted=False, label_width=SBAR_LABEL_WIDTH):
    """One-row figure: diverging Likert bars per condition + diff panels.

    Layout: [cond_0 bars] … [cond_n bars] [diff_pair_0] … [diff_pair_m]

    *conditions* : ordered list of conditions for the diverging-bar columns.
    *diff_pairs* : list of (reference, compare) tuples for the diff panels.
    *inverted*   : True when high raw score = bad outcome (NASA-TLX, Risk VAS).
    """
    n_items = len(item_labels)
    n_div   = len(conditions)
    n_cols  = n_div + 1    # one consolidated diff panel for all pairs

    n_total = max(
        (sum(c) for cond_list in counts_by_cond.values() for c in cond_list),
        default=7,
    )

    # Wrap labels — compute ahead of fig_h so we can scale height by line count
    wrapped_labels = ["\n".join(textwrap.wrap(l, width=label_width))
                      for l in item_labels]
    max_lines = max((l.count("\n") + 1 for l in wrapped_labels), default=1)

    width_ratios = [SBAR_COL_RATIO] * n_div + [SBAR_DIFF_RATIO]
    fig_w = SBAR_FIGW_PER_COL * n_div + SBAR_FIGW_OFFSET
    fig_h = max(SBAR_FIGH_MIN, n_items * (SBAR_FIGH_PER_ITEM + SBAR_FIGH_MULTILINE * (max_lines - 1)) + SBAR_FIGH_OFFSET)

    fig, axes = plt.subplots(
        1, n_cols, figsize=(fig_w, fig_h), sharey=True,
        gridspec_kw={"width_ratios": width_ratios, "wspace": SBAR_WSPACE},
    )
    axes = list(axes) if n_cols > 1 else [axes]
    fig.suptitle(suptitle, fontsize=FONT_SUPTITLE, fontweight="bold")

    # ── Stacked bar panels ────────────────────────────────────────────────────
    for k, cond in enumerate(conditions):
        ax = axes[k]
        for row, counts in enumerate(counts_by_cond.get(cond, [])):
            _draw_stacked_row(ax, row * SBAR_ITEM_PITCH, counts, palette, inverted=inverted)

        ax.set_xlim(0, n_total + 0.4)
        ticks = list(range(0, n_total + 1))
        ax.set_xticks(ticks)
        ax.set_xticklabels([str(t) for t in ticks], fontsize=FONT_TICK_SM)
        ax.set_yticks([r * SBAR_ITEM_PITCH for r in range(n_items)])
        ax.set_title(cond, fontsize=FONT_TITLE, fontweight="bold")
        ax.set_xlabel("n respondents", fontsize=FONT_TICK_SM)
        ax.grid(axis="x", linestyle=":", alpha=0.3)
        if k == 0:
            ax.set_yticklabels(wrapped_labels, fontsize=FONT_TICK)
            ax.set_ylim(-0.5 * SBAR_ITEM_PITCH, (n_items - 0.5) * SBAR_ITEM_PITCH)
            ax.invert_yaxis()   # shared y — inverts all panels
        else:
            ax.tick_params(labelleft=False)

    # ── Consolidated diff panel (all pairs overlaid in one column) ──────────
    rng = np.random.default_rng(42)
    pairwise_stats = _compute_pairwise_stats(raw_by_cond, diff_pairs, n_items)
    _draw_multi_diff_panel(axes[n_div], raw_by_cond, diff_pairs,
                           n_items, rng, inverted=inverted,
                           pairwise_stats=pairwise_stats)

    # ── Combined legend ───────────────────────────────────────────────────────
    from matplotlib.lines import Line2D
    pal_leg = list(reversed(palette)) if inverted else list(palette)
    lab_leg = list(reversed(scale_labels)) if inverted else list(scale_labels)
    scale_patches = [mpatches.Patch(color=c, label=l)
                     for c, l in zip(pal_leg, lab_leg)]
    pair_handles = [
        Line2D([0], [0],
               color=_PAIR_STYLES[pi][0], marker=_PAIR_STYLES[pi][1],
               linestyle="-", linewidth=1.5, markersize=5,
               markerfacecolor=_PAIR_STYLES[pi][0], markeredgecolor="black",
               label=f"{comp} − {bl}")
        for pi, (bl, comp) in enumerate(diff_pairs)
    ]
    open_handle = Line2D([0], [0], color="#555555", marker="o", linestyle="None",
                         markersize=5, markerfacecolor="white",
                         markeredgecolor="#555555",
                         label="open = CI crosses 0")
    all_handles = scale_patches + pair_handles + [open_handle]
    fig.legend(handles=all_handles,
               loc="lower center", ncol=len(all_handles),
               fontsize=FONT_TICK, bbox_to_anchor=(0.5, 0))
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
        "System Usability Scale",
        _SUS_LABELS, counts, raw, COLORS_5,
        ["strongly disagree", "disagree", "neutral", "agree", "strongly agree"],
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
        "Trust in Automation checklist",
        _TIA_LABELS, counts, raw, COLORS_5,
        ["strongly disagree", "disagree", "neutral", "agree", "strongly agree"],
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
        ["strongly disagree", "disagree", "neutral", "agree", "strongly agree"],
        conditions=CONDITIONS_MAIN, diff_pairs=_ALL_PAIRS,
    )
    save_fig(fig, "ob_items.png")
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
#  Perceived Control  (scored 1–5, all positive valence)
# ═══════════════════════════════════════════════════════════════════════════════
_PC_KEYS   = ["pc_01", "pc_02", "pc_03", "pc_04"]
_PC_LABELS = [
    "PC1(+) I feel in control while using this autonomous system",
    "PC2(+) I feel I can control how the autonomous system behaves",
    "PC3(+) I have the resources and ability to make use of this system",
    "PC4(+) Team was effective in accomplishing the mission",
]


def plot_perceived_control(all_data, participants):
    def extract(pdata, cond, key):
        try:
            return pdata["perceived_control"]["conditions"][cond]["items"][key]["scored"]
        except (KeyError, TypeError):
            return None

    counts = _build_counts(all_data, participants, _PC_KEYS, extract, 1, 5)
    raw    = _build_raw(all_data, participants, _PC_KEYS, extract, CONDITIONS_MAIN)
    fig = _make_combined_figure(
        "Perceived Control — Item Response Distribution  (scored 1–5 · TARS is reference)",
        _PC_LABELS, counts, raw, COLORS_5,
        ["strongly disagree", "disagree", "neutral", "agree", "strongly agree"],
        conditions=CONDITIONS_MAIN, diff_pairs=_ALL_PAIRS,
    )
    save_fig(fig, "perceived_control_items.png")
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
    """
    NASA-TLX cross-participant visualization — two figures:

      Figure 1  (nasa_tlx_ratings.png):
        Strip plot — one subplot per dimension, one horizontal row per condition.
        Each participant's 0-20 rating is shown as a jittered dot.
        ◆ = mean (coloured, per-condition),  │ = median (black tick).

      Figure 2  (nasa_tlx_scores.png):
        Box plot of NASA-TLX weighted scores (0-100) per condition.

    Returns [fig1, fig2].
    """
    from matplotlib.lines import Line2D

    dims       = ["mental_demand", "physical_demand", "temporal_demand",
                  "performance", "effort", "frustration"]
    dim_labels = ["Mental Demand", "Physical Demand", "Temporal Demand",
                  "Performance", "Effort", "Frustration"]
    n_dims  = len(dims)
    n_conds = len(CONDITIONS)

    def get_rating(pdata, cond, key):
        try:
            return float(pdata["nasa_tlx"]["conditions"][cond]["subscales"][key]["rating_0_20"])
        except (KeyError, TypeError):
            return None

    def get_score(pdata, cond):
        try:
            return float(pdata["nasa_tlx"]["conditions"][cond]["nasa_tlx_weighted_score"])
        except (KeyError, TypeError):
            return None

    rng = np.random.default_rng(42)

    # ── Figure 1: strip plots ─────────────────────────────────────────────────
    fig1, axes = plt.subplots(n_dims, 1,
                              figsize=(NASA_STRIP_FIGW, n_dims * NASA_STRIP_H_PER_DIM + NASA_STRIP_H_OFFSET))
    fig1.suptitle(
        "NASA-TLX — Dimensions",
        fontsize=FONT_SUPTITLE, fontweight="bold",
    )

    for di, (key, label) in enumerate(zip(dims, dim_labels)):
        ax = axes[di]

        # Faint bin lines at every integer 0-20
        for x in range(0, 21):
            ax.axvline(x, color="#eeeeee", linewidth=0.4, zorder=0)

        for ci, cond in enumerate(CONDITIONS):
            vals = [get_rating(all_data.get(pid, {}), cond, key)
                    for pid in participants]
            vals = [v for v in vals if v is not None]
            if not vals:
                continue

            # KDE overlay — mini violin centred at y=ci
            if len(vals) >= 2:
                x_kde = np.linspace(0, 20, 300)
                try:
                    kde     = gaussian_kde(vals, bw_method="scott")
                    y_kde   = kde(x_kde)
                    y_scale = y_kde / y_kde.max() * NASA_KDE_HALF
                    ax.plot(x_kde, ci + y_scale,
                            color=COND_COLOR[cond], linewidth=1.4,
                            alpha=0.85, zorder=2)
                except Exception:
                    pass

            jitter = rng.uniform(-NASA_JITTER, NASA_JITTER, len(vals))
            ax.scatter(vals, np.full(len(vals), ci) + jitter,
                       color=COND_COLOR[cond], alpha=0.80, s=NASA_DOT_SIZE,
                       edgecolors="white", linewidths=0.5, zorder=4)

            mean_v = float(np.mean(vals))
            med_v  = float(np.median(vals))

            # Mean — filled diamond
            ax.scatter(mean_v, ci,
                       color=COND_COLOR[cond], marker="D", s=NASA_MEAN_SIZE,
                       edgecolors="black", linewidths=1.2, zorder=7)
            ax.text(mean_v, ci + NASA_MEAN_HALF, f"μ={mean_v:.1f}",
                    ha="center", va="bottom", fontsize=FONT_ANNOTATION,
                    color=COND_COLOR[cond], fontweight="bold")

            # Median — thick black vertical tick
            ax.vlines(med_v, ci - NASA_MEDIAN_HALF, ci + NASA_MEDIAN_HALF,
                      colors="black", linewidth=2.0, zorder=6)
            ax.text(med_v, ci - NASA_MEDIAN_OFFSET, f"M={med_v:.1f}",
                    ha="center", va="top", fontsize=FONT_ANNOTATION, color="black")

        ax.set_xlim(-0.5, 20.5)
        ax.set_ylim(-0.68, n_conds - 0.32)
        ax.set_yticks(range(n_conds))
        ax.set_yticklabels(CONDITIONS, fontsize=FONT_TICK)
        ax.set_ylabel(label, fontsize=FONT_LABEL, rotation=0,
                      ha="right", va="center", labelpad=8)
        ax.set_xticks(range(0, 21))
        ax.set_xticklabels(
            [str(x) if x != 10 else "" for x in range(0, 21)],
            fontsize=FONT_TICK_SM, color="#555555")
        # Midpoint marker: bold label above the axis
        ax.text(10, -0.68 - 0.02, "10",
                ha="center", va="top", fontsize=FONT_TICK,
                fontweight="bold", color="#222222",
                transform=ax.get_xaxis_transform())
        # Prominent midpoint line
        ax.axvline(10, color="#555555", linewidth=NASA_MIDPT_LW,
                   linestyle="--", zorder=3, alpha=0.7)
        ax.grid(axis="x", linestyle=":", alpha=0.30)

    axes[-1].set_xlabel("Rating (0–20)", fontsize=FONT_LABEL)

    cond_patches  = [mpatches.Patch(color=COND_COLOR[c], label=c)
                     for c in CONDITIONS]
    legend_extras = [
        Line2D([0], [0], color="gray", marker="D", linestyle="None",
               markersize=8, markeredgecolor="black", label="Mean"),
        Line2D([0], [0], color="black", linewidth=2, label="Median"),
        Line2D([0], [0], color="gray", linewidth=1.4, label="KDE curve"),
    ]
    fig1.legend(handles=cond_patches + legend_extras,
                loc="lower center", ncol=n_conds + 2,
                fontsize=FONT_TICK, bbox_to_anchor=(0.5, 0))
    fig1.tight_layout(rect=[0, 0.04, 1, 1])
    save_fig(fig1, "nasa_tlx_ratings.png")

    # ── Figure 2: box plot of weighted scores ─────────────────────────────────
    fig2, ax2 = plt.subplots(figsize=(NASA_SCORES_FIGW, NASA_SCORES_FIGH))
    fig2.suptitle("NASA-TLX — Weighted Score per Condition",
                  fontsize=FONT_SUPTITLE, fontweight="bold")

    scores = {c: [] for c in CONDITIONS}
    for pid in participants:
        pdata = all_data.get(pid, {})
        for cond in CONDITIONS:
            v = get_score(pdata, cond)
            if v is not None:
                scores[cond].append(v)

    _box_panel(ax2, scores, "NASA-TLX Weighted Score", "Score (0–100)",
               ylim=(0, 100), conditions=CONDITIONS)
    fig2.tight_layout()
    save_fig(fig2, "nasa_tlx_scores.png")

    return [fig1, fig2]


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
#  Trust & Risk VAS — histogram + KDE distribution  (0–100 continuous)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_trust_risk_distribution(all_data, participants):
    """
    One figure with two subplots (trust_vas | risk_vas).
    Each subplot overlays, per condition:
      • a semi-transparent histogram (density-normalised)
      • a KDE curve (line only — no fill)
      • a mean dashed vertical line
      • a 95% CI strip at the bottom (SEM-based: mean ± 1.96 × SEM)
    Conditions drawn: CONDITIONS_MAIN.
    """
    from matplotlib.lines import Line2D

    items = [
        ("trust_vas", "Trust VAS  (0–100)", False),
        ("risk_vas",  "Perceived Risk VAS  (0–100)", True),
    ]

    fig, axes = plt.subplots(1, len(items), figsize=(DIST_FIGW, DIST_FIGH), sharey=False)
    fig.suptitle(
        "Trust & Risk VAS — Distribution per Condition\n"
        "(histogram · KDE curve · ◆ = mean · ─── = 95% CI)",
        fontsize=FONT_SUPTITLE, fontweight="bold",
    )

    x_kde = np.linspace(0, 100, 400)

    for ax, (key, subtitle, inverted) in zip(axes, items):
        ax.set_title(subtitle, fontsize=FONT_TITLE, fontweight="bold")

        cond_vals = {}  # store for CI strip

        for cond in CONDITIONS_MAIN:
            vals = []
            for pid in participants:
                v = _get(all_data.get(pid, {}), "trust_risk", "conditions", cond, key)
                if v is not None:
                    vals.append(float(v))
            cond_vals[cond] = vals
            if not vals:
                continue

            color = COND_COLOR[cond]

            # Histogram (density-normalised, semi-transparent)
            ax.hist(vals, bins=DIST_HIST_BINS, range=(0, 100),
                    density=True, color=color, alpha=DIST_HIST_ALPHA,
                    edgecolor=color, linewidth=0.5)

            # KDE curve — line only, no fill
            if len(vals) >= 2:
                try:
                    kde   = gaussian_kde(vals, bw_method="scott")
                    y_kde = kde(x_kde)
                    ax.plot(x_kde, y_kde,
                            color=color, linewidth=DIST_KDE_LW, alpha=0.90)
                except Exception:
                    pass
            else:
                ax.axvline(vals[0], color=color, linewidth=1.5,
                           linestyle="--", alpha=0.75)

        # ── Mean + 95% CI strip at the bottom ─────────────────────────────
        y_top     = ax.get_ylim()[1]
        n_main    = len(CONDITIONS_MAIN)
        strip_h   = DIST_CI_STRIP_FRAC * y_top
        row_h     = strip_h / n_main
        ax.set_ylim(bottom=-strip_h, top=y_top)

        # faint separator between KDE area and CI strip
        ax.axhline(0, color="black", linewidth=0.5, linestyle=":", alpha=0.35)

        for ci_idx, cond in enumerate(CONDITIONS_MAIN):
            vals  = cond_vals.get(cond, [])
            if not vals:
                continue
            n      = len(vals)
            mean_v = float(np.mean(vals))
            sem    = float(np.std(vals, ddof=1) / np.sqrt(n)) if n > 1 else 0.0
            ci_lo  = mean_v - 1.96 * sem
            ci_hi  = mean_v + 1.96 * sem
            color  = COND_COLOR[cond]
            y_pos  = -(ci_idx + 0.5) * row_h
            cap    = row_h * 0.20

            # dashed vertical mean line (full height)
            ax.axvline(mean_v, color=color, linewidth=1.0,
                       linestyle="--", alpha=0.50, zorder=3)

            # CI horizontal bar + caps
            ax.plot([ci_lo, ci_hi], [y_pos, y_pos],
                    color=color, linewidth=1.8, alpha=0.90, zorder=5)
            ax.plot([ci_lo, ci_lo], [y_pos - cap, y_pos + cap],
                    color=color, linewidth=1.5, alpha=0.90, zorder=5)
            ax.plot([ci_hi, ci_hi], [y_pos - cap, y_pos + cap],
                    color=color, linewidth=1.5, alpha=0.90, zorder=5)

            # mean diamond
            ax.scatter(mean_v, y_pos, color=color, marker="D", s=45,
                       edgecolors="black", linewidths=0.7, zorder=6)

            # condition label on the left
            ax.text(-2, y_pos, cond, ha="right", va="center",
                    fontsize=FONT_TICK_SM, color=color, fontweight="bold")

        ax.set_xlim(-1, 101)
        ax.set_xlabel("Score (0–100)", fontsize=FONT_TICK)
        ax.set_ylabel("Density", fontsize=FONT_TICK)
        # suppress y-tick labels in the negative strip
        ax.set_yticks([t for t in ax.get_yticks() if t >= 0])
        ax.grid(axis="x", linestyle=":", alpha=0.20)
        ax.grid(axis="y", linestyle=":", alpha=0.35)

    # Shared legend
    cond_handles  = [Line2D([0], [0], color=COND_COLOR[c], linewidth=2, label=c)
                     for c in CONDITIONS_MAIN]
    extra_handles = [
        Line2D([0], [0], color="gray", marker="D", linestyle="None",
               markersize=6, markeredgecolor="black", label="Mean (◆)"),
        Line2D([0], [0], color="gray", linewidth=1.8, label="95% CI"),
    ]
    fig.legend(handles=cond_handles + extra_handles,
               loc="lower center", ncol=len(CONDITIONS_MAIN) + 2,
               fontsize=FONT_TICK, bbox_to_anchor=(0.5, 0))
    fig.tight_layout(rect=[0, 0.07, 1, 1])
    save_fig(fig, "trust_risk_distribution.png")
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
#  PTS  (trait — no condition, one diverging chart for all participants)
# ═══════════════════════════════════════════════════════════════════════════════
_PTS_KEYS   = ["Q1", "Q2", "Q3", "Q4", "Q5", "Q6", "Q7"]
_PTS_LABELS = [
    "Q1(+) I usually trust machines until there is a reason not to",
    "Q2(–) For the most part, I distrust machines",
    "Q3(+) In general, I would rely on a machine to assist me",
    "Q4(+) My tendency to trust machines is high",
    "Q5(+) It is easy for me to trust machines to do their job.",
    "Q6(+) I am likely to trust a machine even when I have little knowledge about it",
    "Q7(+) I have a generally positive attitude toward advanced automation in aviation",
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

    wrapped_pts = ["\n".join(textwrap.wrap(l, width=PTS_LABEL_WIDTH)) for l in _PTS_LABELS]
    max_lines   = max(l.count("\n") + 1 for l in wrapped_pts)
    fig_h = max(PTS_FIGH_MIN, n * (PTS_FIGH_PER_ITEM + SBAR_FIGH_MULTILINE * (max_lines - 1)) + PTS_FIGH_OFFSET)
    fig, ax = plt.subplots(figsize=(PTS_FIGW, fig_h))
    fig.suptitle("Propensity to Trust Automation",
                 fontsize=FONT_SUPTITLE, fontweight="bold")

    for row, counts in enumerate(counts_list):
        _draw_stacked_row(ax, row * SBAR_ITEM_PITCH, counts, COLORS_5)

    ax.set_xlim(0, n_participants + 0.4)
    ticks = list(range(0, n_participants + 1))
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(t) for t in ticks], fontsize=FONT_TICK)
    ax.set_yticks([r * SBAR_ITEM_PITCH for r in range(n)])
    ax.set_yticklabels(wrapped_pts, fontsize=FONT_LABEL)
    ax.set_ylim(-0.5 * SBAR_ITEM_PITCH, (n - 0.5) * SBAR_ITEM_PITCH)
    ax.invert_yaxis()
    ax.set_xlabel("n respondents", fontsize=FONT_TICK)
    ax.grid(axis="x", linestyle=":", alpha=0.3)
    ax.set_title("All participants (pre-experiment)", fontsize=FONT_TITLE)

    patches = [mpatches.Patch(color=c, label=l) for c, l in
               zip(COLORS_5, ["strongly disagree", "disagree", "neutral", "agree", "strongly agree"])]
    fig.legend(handles=patches, loc="lower center", ncol=5,
               fontsize=FONT_TICK, bbox_to_anchor=(0.5, 0))
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

    bp = ax.boxplot(boxes, positions=positions, widths=BOX_WIDTH, patch_artist=True,
                    medianprops=dict(color="black", linewidth=2),
                    flierprops=dict(marker="o", markersize=4, alpha=0.5))
    for patch, cond in zip(bp["boxes"], conditions):
        patch.set_facecolor(COND_COLOR[cond])
        patch.set_alpha(BOX_ALPHA)

    rng = np.random.default_rng(42)
    for i, (cond, vals) in enumerate(zip(conditions, boxes)):
        if vals:
            jitter = rng.uniform(-BOX_JITTER, BOX_JITTER, len(vals))
            ax.scatter(np.array([i] * len(vals)) + jitter, vals,
                       color="black", s=BOX_DOT_SIZE, zorder=5, alpha=0.75)

    ax.set_xticks(positions)
    ax.set_xticklabels(conditions, fontsize=FONT_TICK)
    ax.set_title(title, fontsize=FONT_TITLE, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=FONT_TICK)
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
    fig, axes = plt.subplots(nrows, ncols, figsize=(BOX_FIGW, nrows * BOX_HEIGHT_PER_ROW))
    fig.suptitle(
        "Computed Scores — Box Plots across Conditions\n",
        fontsize=FONT_SUPTITLE, fontweight="bold",
    )
    axes_flat = axes.flatten()

    for i, (title, ylabel, ylim, fn, conds) in enumerate(metrics):
        _box_panel(axes_flat[i], _collect(all_data, participants, fn, conds),
                   title, ylabel, ylim, conditions=conds)
    for j in range(len(metrics), len(axes_flat)):
        axes_flat[j].set_visible(False)

    patches = [mpatches.Patch(color=COND_COLOR[c], label=c) for c in CONDITIONS]
    fig.legend(handles=patches, loc="lower right", fontsize=FONT_TICK, title="Condition")
    plt.tight_layout()
    save_fig(fig, "boxplots_scores.png")
    return fig



# ═══════════════════════════════════════════════════════════════════════════════
#  PREFERENCE CLUSTER ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

# NASA-TLX polarity: +1 = high score is GOOD, -1 = high score is BAD
# All workload dims: high = bad.  Performance: high = good (inverted valence).
_NASA_POLARITY = {
    "mental_demand":   -1,
    "physical_demand": -1,
    "temporal_demand": -1,
    "performance":     +1,   # ← reversed: high score = performed well
    "effort":          -1,
    "frustration":     -1,
}


def _build_cluster_data(all_data, participants, include_ob=True, include_nasa=False,
                        include_vas=False, exclude=None):
    """Return (valid_pids, scatter_xs, scatter_ys, feature_matrix).

    Feature vector per participant: polarity-corrected per-item differences
    [TARP-S − TARS, TARP-F − TARS] so that **positive always = better**.

    SUS/TiA/OB items are already polarity-corrected in the JSON (recoded/converted).
    NASA-TLX items are corrected here using _NASA_POLARITY.
    Trust VAS (0–100): high = good → polarity +1.
    Risk  VAS (0–100): high = bad  → polarity −1.
    """
    def _sus_items(pdata, cond):
        out = []
        for key in _SUS_KEYS:
            try:
                out.append(float(pdata["sus"]["conditions"][cond]["items"][key]["converted"]))
            except (KeyError, TypeError):
                out.append(np.nan)
        return out   # polarity already +1 (high = good)

    def _tia_items(pdata, cond):
        out = []
        for key, sub in _TIA_SUB_MAP.items():
            try:
                out.append(float(pdata["tia"]["conditions"][cond]["subscales"][sub]["items"][key]["recoded"]))
            except (KeyError, TypeError):
                out.append(np.nan)
        return out   # polarity already +1

    def _ob_items(pdata, cond):
        out = []
        for key in _OB_KEYS:
            try:
                out.append(float(pdata["oversight_bespoke"]["conditions"][cond]["extended_score"]["items"][key]["scored"]))
            except (KeyError, TypeError):
                out.append(np.nan)
        return out   # polarity already +1

    def _nasa_items(pdata, cond):
        """Return polarity-corrected NASA-TLX ratings (positive = better)."""
        out = []
        for key, pol in _NASA_POLARITY.items():
            try:
                v = float(pdata["nasa_tlx"]["conditions"][cond]["subscales"][key]["rating_0_20"])
                out.append(pol * v)   # flip so positive = better
            except (KeyError, TypeError):
                out.append(np.nan)
        return out

    def _vas_items(pdata, cond):
        """Return [trust, -risk] so that positive = better for both."""
        out = []
        try:
            out.append(+1.0 * float(pdata["trust_risk"]["conditions"][cond]["trust_vas"]))
        except (KeyError, TypeError):
            out.append(np.nan)
        try:
            out.append(-1.0 * float(pdata["trust_risk"]["conditions"][cond]["risk_vas"]))
        except (KeyError, TypeError):
            out.append(np.nan)
        return out

    def _pc_items(pdata, cond):
        """Return perceived control items (all positive valence, 1–5)."""
        out = []
        for key in _PC_KEYS:
            try:
                out.append(float(pdata["perceived_control"]["conditions"][cond]["items"][key]["scored"]))
            except (KeyError, TypeError):
                out.append(np.nan)
        return out

    def _all_items(pdata, cond):
        items = _sus_items(pdata, cond) + _tia_items(pdata, cond)
        if include_ob:
            items += _ob_items(pdata, cond)
        if include_nasa:
            items += _nasa_items(pdata, cond)
        if include_vas:
            items += _vas_items(pdata, cond)
        items += _pc_items(pdata, cond)
        return np.array(items, dtype=float)

    valid_pids, scatter_xs, scatter_ys, feature_matrix = [], [], [], []
    _exclude = set(exclude) if exclude else set()

    for pid in participants:
        if pid in _exclude:
            continue
        pdata = all_data.get(pid, {})
        tars_vec  = _all_items(pdata, "TARS")
        tarps_vec = _all_items(pdata, "TARP-S")
        tarpf_vec = _all_items(pdata, "TARP-F")

        diff_s = tarps_vec - tars_vec
        diff_f = tarpf_vec - tars_vec

        x = float(np.nanmean(diff_s))
        y = float(np.nanmean(diff_f))

        if np.isnan(x) or np.isnan(y):
            continue

        # Use item-level TARP-S − TARP-F as the clustering feature.
        # This cancels the TARS baseline and captures *relative* preference
        # between the two autopilot modes directly, so the clustering finds
        # "S-lovers" vs "F-lovers" rather than "overall approve/disapprove".
        feat = tarps_vec - tarpf_vec
        feat = np.where(np.isnan(feat), 0.0, feat)   # impute missing items with 0

        scatter_xs.append(x)
        scatter_ys.append(y)
        feature_matrix.append(feat)
        valid_pids.append(pid)

    return valid_pids, scatter_xs, scatter_ys, feature_matrix


def _draw_cluster_figure(valid_pids, scatter_xs, scatter_ys, feature_matrix,
                         suptitle, filename):
    """Draw and save the quadrant cluster figure (scatter only)."""
    from matplotlib.lines import Line2D

    if len(valid_pids) < 3:
        print(f"  ⚠  Not enough data for {filename}")
        return None

    mode_colors = {
        "TARP-S": "#2E75B6",
        "TARP-F": "#E6730D",
        "Parity": "#888888",
    }

    fig, ax_scatter = plt.subplots(1, 1, figsize=(CLUSTER_FIGW, CLUSTER_FIGH))
    fig.suptitle(suptitle, fontsize=FONT_SUPTITLE, fontweight="bold")

    # ── Scatter ───────────────────────────────────────────────────────────────
    lim = max(abs(v) for v in scatter_xs + scatter_ys) * 1.35
    lim = max(lim, 0.3)
    ax_scatter.axhline(0, color="#aaaaaa", linewidth=0.8, zorder=1)
    ax_scatter.axvline(0, color="#aaaaaa", linewidth=0.8, zorder=1)
    ax_scatter.plot([-lim, lim], [-lim, lim], color="#cccccc",
                    linewidth=1.0, linestyle="--", zorder=1)

    for i, pid in enumerate(valid_pids):
        mode_balance = scatter_ys[i] - scatter_xs[i]
        if mode_balance > 0.05:
            pref = "TARP-F"
        elif mode_balance < -0.05:
            pref = "TARP-S"
        else:
            pref = "Parity"
        c = mode_colors[pref]
        ax_scatter.scatter(scatter_xs[i], scatter_ys[i],
                           color=c, s=80, edgecolors="black",
                           linewidths=0.8, zorder=4)
        ax_scatter.annotate(pid, (scatter_xs[i], scatter_ys[i]),
                            textcoords="offset points", xytext=(6, 4),
                            fontsize=FONT_TICK, color=c, fontweight="bold")

    ax_scatter.set_xlim(-lim, lim)
    ax_scatter.set_ylim(-lim, lim)
    ax_scatter.set_xlabel("Mean Δ  (TARP-S − TARS)  ↑ better", fontsize=FONT_LABEL)
    ax_scatter.set_ylabel("Mean Δ  (TARP-F − TARS)  ↑ better", fontsize=FONT_LABEL)
    ax_scatter.set_title("(A)  Per-participant preference", fontsize=FONT_TITLE, fontweight="bold")
    ax_scatter.grid(linestyle=":", alpha=0.35)

    qpad = lim * 0.06
    for (tx, ty, ha, va, txt) in [
        ( lim - qpad,  lim - qpad, "right", "top",    "Q1\nBoth modes better\nthan baseline"),
        (-lim + qpad,  lim - qpad, "left",  "top",    "Q2\nOnly TARP-F\nbetter than baseline"),
        ( lim - qpad, -lim + qpad, "right", "bottom", "Q3\nOnly TARP-S\nbetter than baseline"),
        (-lim + qpad, -lim + qpad, "left",  "bottom", "Q4\nBoth modes worse\nthan baseline"),
    ]:
        ax_scatter.text(tx, ty, txt, ha=ha, va=va, fontsize=FONT_TICK_SM, color="#444444",
                        bbox=dict(boxstyle="round,pad=0.2", fc="#f0f0f0", ec="none", alpha=0.7))
    ax_scatter.text(lim * 0.50, lim * 0.70, "above: TARP-F > TARP-S",
                    ha="left", va="bottom", fontsize=FONT_TICK_SM, color="#888888", rotation=45)
    ax_scatter.text(lim * 0.05, -lim * 0.28, "below: TARP-S > TARP-F",
                    ha="left", va="top", fontsize=FONT_TICK_SM, color="#888888", rotation=45)

    legend_handles = [
         Line2D([0], [0], marker="o", color="w", markerfacecolor=mode_colors["TARP-S"],
             markeredgecolor="black", markersize=9, label="Relative tilt: TARP-S"),
         Line2D([0], [0], marker="o", color="w", markerfacecolor=mode_colors["TARP-F"],
             markeredgecolor="black", markersize=9, label="Relative tilt: TARP-F"),
         Line2D([0], [0], marker="o", color="w", markerfacecolor=mode_colors["Parity"],
             markeredgecolor="black", markersize=9, label="Near parity"),
        Line2D([0], [0], color="#cccccc", linestyle="--", linewidth=1.2,
               label="y = x  (TARP-S = TARP-F)"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=4,
               fontsize=FONT_TICK, bbox_to_anchor=(0.5, 0))
    fig.tight_layout(rect=[0, 0.07, 1, 0.95])
    save_fig(fig, filename)
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
#  PREFERENCE PROFILE COHERENCE  ("bumpchart" across questionnaires)
# ═══════════════════════════════════════════════════════════════════════════════

# Ordered questionnaire groups for the coherence chart.
# Each entry: (label, extract_fn) where extract_fn(pdata, cond) → float or nan
# All values are polarity-corrected so positive = participant preferred that condition.
_COHERENCE_GROUPS = [
    ("NASA-TLX",          None),   # filled dynamically below
    ("Perceived\nControl", None),
    ("Oversight\nBespoke", None),
    ("Trust/Risk\nVAS",    None),
    ("TiA",               None),
    ("SUS",               None),
]


def _pc_mean(pdata, cond):
    vals = []
    for key in _PC_KEYS:
        try:
            vals.append(float(pdata["perceived_control"]["conditions"][cond]["items"][key]["scored"]))
        except (KeyError, TypeError):
            pass
    return float(np.nanmean(vals)) if vals else np.nan


def _ob_mean(pdata, cond):
    vals = []
    for key in _OB_KEYS:
        try:
            vals.append(float(pdata["oversight_bespoke"]["conditions"][cond]["extended_score"]["items"][key]["scored"]))
        except (KeyError, TypeError):
            pass
    return float(np.nanmean(vals)) if vals else np.nan


def _vas_mean(pdata, cond):
    """Trust (positive) and -Risk (negative polarity) averaged."""
    vals = []
    try:
        vals.append(+1.0 * float(pdata["trust_risk"]["conditions"][cond]["trust_vas"]))
    except (KeyError, TypeError):
        pass
    try:
        vals.append(-1.0 * float(pdata["trust_risk"]["conditions"][cond]["risk_vas"]))
    except (KeyError, TypeError):
        pass
    return float(np.nanmean(vals)) if vals else np.nan


def _tia_mean(pdata, cond):
    vals = []
    for key, sub in _TIA_SUB_MAP.items():
        try:
            vals.append(float(pdata["tia"]["conditions"][cond]["subscales"][sub]["items"][key]["recoded"]))
        except (KeyError, TypeError):
            pass
    return float(np.nanmean(vals)) if vals else np.nan


def _sus_mean(pdata, cond):
    vals = []
    for key in _SUS_KEYS:
        try:
            vals.append(float(pdata["sus"]["conditions"][cond]["items"][key]["converted"]))
        except (KeyError, TypeError):
            pass
    return float(np.nanmean(vals)) if vals else np.nan


def _nasa_mean_polarity(pdata, cond):
    vals = []
    for key, pol in _NASA_POLARITY.items():
        try:
            v = float(pdata["nasa_tlx"]["conditions"][cond]["subscales"][key]["rating_0_20"])
            vals.append(pol * v)
        except (KeyError, TypeError):
            pass
    return float(np.nanmean(vals)) if vals else np.nan


_COHERENCE_EXTRACTORS = [
    ("NASA-TLX",          _nasa_mean_polarity),
    ("Perceived\nControl", _pc_mean),
    ("Oversight\nBespoke", _ob_mean),
    ("Trust/Risk\nVAS",   _vas_mean),
    ("TiA",               _tia_mean),
    ("SUS",               _sus_mean),
]


def plot_preference_profile(all_data, participants):
    """Preference Profile Coherence chart.

    For each questionnaire group (x-axis) and each participant (one line),
    show  mean(TARP-S) − mean(TARP-F)  on the y-axis, **z-score normalised
    per instrument column** so that Trust/Risk VAS (0-100) and Likert scales
    (1-5) are on the same resolution.

    Coloring:
      • Participants whose line never crosses zero → same grey (#999999)
        (coherent preference throughout)
      • Participants whose line crosses zero at least once → a distinct
        vivid color per switcher (up to 8, then cycling)
    """
    from matplotlib.lines import Line2D

    labels = [lbl for lbl, _ in _COHERENCE_EXTRACTORS]
    n_groups = len(labels)
    x_pos = list(range(n_groups))

    # ── Build raw diffs (TARP-S − TARP-F, polarity-corrected) ────────────────
    raw_lines = {}
    for pid in participants:
        pdata = all_data.get(pid, {})
        row = []
        for _, fn in _COHERENCE_EXTRACTORS:
            s = fn(pdata, "TARP-S")
            f = fn(pdata, "TARP-F")
            row.append(np.nan if (np.isnan(s) or np.isnan(f)) else s - f)
        raw_lines[pid] = row

    # ── Z-score normalise per column (instrument) ─────────────────────────────
    # Compute mean & std across all participants for each instrument column,
    # ignoring NaNs.  Divide each diff by the column std so all instruments
    # have the same unit on the y-axis.
    arr = np.array([raw_lines[pid] for pid in participants], dtype=float)  # (n_pid, n_groups)
    col_std = np.nanstd(arr, axis=0, ddof=1)
    col_std[col_std == 0] = 1.0   # avoid division by zero for constant columns

    norm_lines = {}
    for pid in participants:
        raw = np.array(raw_lines[pid], dtype=float)
        norm_lines[pid] = raw / col_std   # sign preserved, only scale removed

    # ── Classify: switcher = line crosses zero between any two adjacent valid pts
    def _crosses_zero(ys):
        valid = [v for v in ys if not np.isnan(v)]
        if len(valid) < 2:
            return False
        signs = [1 if v > 0 else (-1 if v < 0 else 0) for v in valid]
        signs = [s for s in signs if s != 0]  # ignore exact zeros
        return any(signs[i] != signs[i + 1] for i in range(len(signs) - 1))

    SWITCHER_COLORS = [
        "#E63946", "#2A9D8F", "#E9C46A", "#9B5DE5",
        "#F4A261", "#06D6A0", "#EF476F", "#118AB2",
    ]
    switcher_idx = 0
    pid_colors   = {}
    COHERENT_COLOR = "#999999"

    for pid in participants:
        if _crosses_zero(norm_lines[pid]):
            pid_colors[pid] = SWITCHER_COLORS[switcher_idx % len(SWITCHER_COLORS)]
            switcher_idx += 1
        else:
            pid_colors[pid] = COHERENT_COLOR

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(COHERENCE_FIGW, COHERENCE_FIGH))
    fig.suptitle(
        "Preference Profile Coherence  (TARP-S − TARP-F · z-score normalised per instrument)",
        fontsize=FONT_SUPTITLE, fontweight="bold",
    )

    y_all = [v for row in norm_lines.values() for v in row if not np.isnan(v)]
    y_lim = max(abs(v) for v in y_all) * 1.15 if y_all else 2.0

    ax.axhline(0, color="#888888", linewidth=1.0, linestyle="--", zorder=1)
    ax.fill_between([-0.5, n_groups - 0.5], 0,  y_lim,
                    color="#E6730D", alpha=0.04, zorder=0)
    ax.fill_between([-0.5, n_groups - 0.5], -y_lim, 0,
                    color="#2E75B6", alpha=0.04, zorder=0)
    ax.text(n_groups - 0.52,  y_lim * 0.06, "TARP-S preferred",
            ha="right", va="bottom", fontsize=FONT_TICK, color="#E6730D", fontstyle="italic")
    ax.text(n_groups - 0.52, -y_lim * 0.06, "TARP-F preferred",
            ha="right", va="top",    fontsize=FONT_TICK, color="#2E75B6", fontstyle="italic")

    # Draw coherent lines first (grey, behind), then switchers on top
    for draw_switcher in (False, True):
        for pid in participants:
            row = norm_lines[pid]
            xs = [x_pos[i] for i, v in enumerate(row) if not np.isnan(v)]
            ys = [v         for v in row if not np.isnan(v)]
            if len(xs) < 2:
                continue
            is_sw = _crosses_zero(row)
            if is_sw != draw_switcher:
                continue
            c     = pid_colors[pid]
            lw    = 2.2 if is_sw else 1.2
            alpha = 0.90 if is_sw else 0.45
            zord  = 5    if is_sw else 3
            ax.plot(xs, ys, color=c, linewidth=lw, alpha=alpha, zorder=zord,
                    marker="o", markersize=4 if not is_sw else 5,
                    markeredgecolor="white", markeredgewidth=0.5)
            ax.annotate(pid, (xs[-1], ys[-1]), textcoords="offset points",
                        xytext=(5, 0), fontsize=FONT_TICK_SM, color=c,
                        fontweight="bold" if is_sw else "normal", va="center")

    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, fontsize=FONT_LABEL)
    ax.set_xlim(-0.5, n_groups - 0.3)
    ax.set_ylim(-y_lim, y_lim)
    ax.set_ylabel("Δ  (z-score)  ↑ prefers TARP-S  /  ↓ prefers TARP-F", fontsize=FONT_TICK)
    ax.grid(axis="y", linestyle=":", alpha=0.35)
    ax.grid(axis="x", linestyle=":", alpha=0.20)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Legend: one entry per switcher + one grey entry for coherent group
    switcher_pids  = [p for p in participants if _crosses_zero(norm_lines[p])]
    coherent_pids  = [p for p in participants if not _crosses_zero(norm_lines[p])]
    legend_handles = [
        Line2D([0], [0], color=pid_colors[p], linewidth=2.2, marker="o",
               markersize=5, label=p)
        for p in switcher_pids
    ]
    if coherent_pids:
        legend_handles.append(
            Line2D([0], [0], color=COHERENT_COLOR, linewidth=1.2, marker="o",
                   markersize=4, alpha=0.6,
                   label=f"Coherent: {', '.join(coherent_pids)}")
        )
    fig.legend(handles=legend_handles, loc="lower center",
               ncol=min(len(legend_handles), 5), fontsize=FONT_TICK,
               bbox_to_anchor=(0.5, 0))
    fig.tight_layout(rect=[0, 0.10, 1, 1])
    save_fig(fig, "preference_profile_coherence.png")
    return fig


def plot_preference_clusters(all_data, participants):
    """Quadrant scatter (TARP-S vs TARP-F relative to TARS baseline)."""
    figs = []

    # All instruments, P07 included
    pids, xs, ys, feat = _build_cluster_data(
        all_data, participants, include_ob=True, include_nasa=True, include_vas=True)
    figs.append(_draw_cluster_figure(
        pids, xs, ys, feat,
        "TARP-S vs TARP-F Preference Quadrant — SUS + TiA + OB + PC + NASA-TLX + Trust/Risk VAS\n"
        "(TARS baseline · NASA polarity-corrected · Trust +1 · Risk −1)",
        "preference_clusters_all_vas.png"))

    return [f for f in figs if f is not None]


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════════════
#  CROSS-PARTICIPANT REPORTS  (one .txt per form type)
# ═══════════════════════════════════════════════════════════════════════════════

REPORTS_DIR = os.path.join(HITLS_DIR, "compare_forms")


def _desc(vals):
    """Return dict of descriptive stats for a list of floats."""
    if not vals:
        return {"n": 0, "mean": None, "sd": None, "median": None, "min": None, "max": None}
    a = np.array(vals, dtype=float)
    return {
        "n":      int(len(a)),
        "mean":   round(float(np.mean(a)), 3),
        "sd":     round(float(np.std(a, ddof=1)), 3) if len(a) > 1 else 0.0,
        "median": round(float(np.median(a)), 3),
        "min":    round(float(np.min(a)), 3),
        "max":    round(float(np.max(a)), 3),
    }


def _bar_r(value, min_val, max_val, width=20):
    """ASCII bar proportional to value within [min_val, max_val]."""
    if max_val <= min_val:
        return "░" * width
    frac = (value - min_val) / (max_val - min_val)
    filled = int(round(frac * width))
    return "█" * filled + "░" * (width - filled)


def _write_report(path, title, summary_json, sections):
    """Write a report in the standard format (JSON block + human-readable)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("--- MACHINE-READABLE SUMMARY (JSON) ---\n")
        f.write(json.dumps(summary_json, indent=2, ensure_ascii=False))
        f.write("\n--- END SUMMARY ---\n")
        f.write("=" * 78 + "\n")
        f.write(f"  {title}\n")
        f.write("=" * 78 + "\n\n")
        for section_text in sections:
            f.write(section_text)


def _fmt_desc_table(label_col_w, stats_by_cond, conditions, label):
    """Format a single-item stats row across conditions."""
    row = f"  {label:<{label_col_w}}"
    for cond in conditions:
        s = stats_by_cond.get(cond, {})
        if s.get("n", 0) == 0:
            row += f"  {'—':>22}"
        else:
            row += f"  μ={s['mean']:5.2f} σ={s['sd']:4.2f} Md={s['median']:5.2f}"
    return row + "\n"


def _cond_header(label_col_w, conditions):
    header = " " * (label_col_w + 2)
    for cond in conditions:
        header += f"  {cond:^22}"
    return header + "\n" + " " * (label_col_w + 2) + "  " + ("─" * 22 + "  ") * len(conditions) + "\n"


def _fmt_stat_section(label, friedman, pairwise, diff_pairs):
    """Format a statistical-analysis block for one composite score or item.

    friedman : {'chi2': float, 'p': float, 'df': int}
    pairwise : list of dicts per pair (same order as diff_pairs)
    """
    lines = []
    lines.append(f"  {label}")
    chi2, p_f, df = friedman['chi2'], friedman['p'], friedman['df']
    p_f_str = f"{p_f:.4f}" if p_f >= 0.0001 else f"{p_f:.2e}"
    lines.append(f"    Friedman χ²({df}) = {chi2:.3f},  p = {p_f_str}")
    for pi, (bl, comp) in enumerate(diff_pairs):
        pw = pairwise[pi]
        p_raw = pw['p_raw']
        p_cor = pw['p_corr']
        r_val = pw['r']
        sig   = "✓" if pw['reject'] else " "
        p_raw_str = f"{p_raw:.4f}" if p_raw >= 0.0001 else f"{p_raw:.2e}"
        p_cor_str = f"{p_cor:.4f}" if p_cor >= 0.0001 else f"{p_cor:.2e}"
        stars = pw['stars'] or "ns"
        lines.append(
            f"    [{sig}] {comp} − {bl:<8}  W p={p_raw_str}  p_Holm={p_cor_str}  r={r_val:+.3f}  {stars}"
        )
    return "\n".join(lines) + "\n"


# ── SUS report ────────────────────────────────────────────────────────────────

def write_sus_report(all_data, participants):
    conditions = CONDITIONS_MAIN

    # Per-item, per-condition stats
    item_stats = {}
    for key, label in zip(_SUS_KEYS, _SUS_LABELS):
        item_stats[key] = {}
        for cond in conditions:
            vals = []
            for pid in participants:
                try:
                    v = all_data[pid]["sus"]["conditions"][cond]["items"][key]["converted"]
                    if v is not None:
                        vals.append(float(v))
                except (KeyError, TypeError):
                    pass
            item_stats[key][cond] = _desc(vals)

    # Overall SUS score per condition
    score_stats = {}
    for cond in conditions:
        vals = []
        for pid in participants:
            try:
                v = all_data[pid]["sus"]["conditions"][cond]["sus_score"]
                if v is not None:
                    vals.append(float(v))
            except (KeyError, TypeError):
                pass
        score_stats[cond] = _desc(vals)

    summary = {
        "form": "sus",
        "n_participants": len(participants),
        "conditions": conditions,
        "sus_score": score_stats,
        "items": item_stats,
    }

    # Human-readable
    w = 50
    sec1 = f"{'─' * 78}\n  OVERALL SUS SCORE (0–100)\n{'─' * 78}\n"
    sec1 += _cond_header(w, conditions)
    sec1 += _fmt_desc_table(w, score_stats, conditions, "SUS Score (0–100)")
    sec1 += "\n"

    sec2 = f"{'─' * 78}\n  ITEMS (converted 0–4, high = agree)\n{'─' * 78}\n"
    sec2 += _cond_header(w, conditions)
    for key, label in zip(_SUS_KEYS, _SUS_LABELS):
        sec2 += _fmt_desc_table(w, item_stats[key], conditions, label[:w])
    sec2 += "\n"

    # ── Statistical analysis ──────────────────────────────────────────────────
    diff_pairs = _ALL_PAIRS
    # Raw SUS scores per condition per participant
    raw_sus = {c: [None] * len(participants) for c in conditions}
    for ci, cond in enumerate(conditions):
        for pi, pid in enumerate(participants):
            try:
                v = all_data[pid]["sus"]["conditions"][cond]["sus_score"]
                raw_sus[cond][pi] = float(v) if v is not None else None
            except (KeyError, TypeError):
                pass
    friedman_sus, _ = _friedman_test([raw_sus[c] for c in conditions]), None
    friedman_sus = {"chi2": friedman_sus[0], "p": friedman_sus[1], "df": len(conditions) - 1}
    pairwise_sus_raw = []
    for bl, comp in diff_pairs:
        p, r = _wilcoxon_test(raw_sus[bl], raw_sus[comp])
        pairwise_sus_raw.append(p)
    reject_sus, p_corr_sus = _holm_correct(pairwise_sus_raw)
    pairwise_sus = []
    for pi2, (bl, comp) in enumerate(diff_pairs):
        p_r = pairwise_sus_raw[pi2]
        pairwise_sus.append({
            "p_raw": p_r, "r": _wilcoxon_test(raw_sus[bl], raw_sus[comp])[1],
            "reject": bool(reject_sus[pi2]), "p_corr": float(p_corr_sus[pi2]),
            "stars": _sig_stars(p_r, bool(reject_sus[pi2])),
        })
    sec3 = f"{'─' * 78}\n  STATISTICAL ANALYSIS  (Friedman + Wilcoxon + Holm-Bonferroni)\n{'─' * 78}\n"
    sec3 += _fmt_stat_section("SUS Score (0–100)", friedman_sus, pairwise_sus, diff_pairs)
    sec3 += "\n  [✓] = Holm-Bonferroni corrected rejection at α=.05\n"
    sec3 += "  r = rank-biserial effect size\n\n"

    path = os.path.join(REPORTS_DIR, "sus_report.txt")
    _write_report(path, "SUS — System Usability Scale  (cross-participant)", summary, [sec1, sec2, sec3])
    print(f"  → {path}")


# ── TiA report ────────────────────────────────────────────────────────────────

_TIA_SUBSCALES = ["reliability_competence", "understanding_predictability", "trust_in_automation"]
_TIA_SUB_LABELS = {
    "reliability_competence":       "Reliability / Competence",
    "understanding_predictability": "Understanding / Predictability",
    "trust_in_automation":          "Trust in Automation",
}


def write_tia_report(all_data, participants):
    conditions = CONDITIONS_MAIN

    item_stats = {}
    for key, sub, label in [(k, s, l) for s, k, l in _TIA_SPECS]:
        item_stats[key] = {}
        for cond in conditions:
            vals = []
            for pid in participants:
                try:
                    v = all_data[pid]["tia"]["conditions"][cond]["subscales"][sub]["items"][key]["recoded"]
                    if v is not None:
                        vals.append(float(v))
                except (KeyError, TypeError):
                    pass
            item_stats[key][cond] = _desc(vals)

    subscale_stats = {}
    for sub in _TIA_SUBSCALES:
        subscale_stats[sub] = {}
        for cond in conditions:
            vals = []
            for pid in participants:
                try:
                    v = all_data[pid]["tia"]["conditions"][cond]["subscales"][sub]["mean"]
                    if v is not None:
                        vals.append(float(v))
                except (KeyError, TypeError):
                    pass
            subscale_stats[sub][cond] = _desc(vals)

    global_stats = {}
    for cond in conditions:
        vals = []
        for pid in participants:
            try:
                v = all_data[pid]["tia"]["conditions"][cond]["global_mean_excl_traits"]
                if v is not None:
                    vals.append(float(v))
            except (KeyError, TypeError):
                pass
        global_stats[cond] = _desc(vals)

    summary = {
        "form": "tia",
        "n_participants": len(participants),
        "conditions": conditions,
        "global_mean_excl_traits": global_stats,
        "subscales": subscale_stats,
        "items": item_stats,
    }

    w = 50
    sec1 = f"{'─' * 78}\n  GLOBAL MEAN (excl. traits, 1–5)\n{'─' * 78}\n"
    sec1 += _cond_header(w, conditions)
    sec1 += _fmt_desc_table(w, global_stats, conditions, "Global mean (1–5)")
    sec1 += "\n"

    sec2 = f"{'─' * 78}\n  SUBSCALE MEANS (1–5)\n{'─' * 78}\n"
    sec2 += _cond_header(w, conditions)
    for sub in _TIA_SUBSCALES:
        sec2 += _fmt_desc_table(w, subscale_stats[sub], conditions, _TIA_SUB_LABELS[sub])
    sec2 += "\n"

    sec3 = f"{'─' * 78}\n  ITEMS (recoded 1–5, high = agree)\n{'─' * 78}\n"
    for sub in _TIA_SUBSCALES:
        sec3 += f"\n  [{_TIA_SUB_LABELS[sub]}]\n"
        sec3 += _cond_header(w, conditions)
        for s2, key, label in _TIA_SPECS:
            if s2 == sub:
                sec3 += _fmt_desc_table(w, item_stats[key], conditions, label[:w])
    sec3 += "\n"

    # ── Statistical analysis ──────────────────────────────────────────────────
    diff_pairs = _ALL_PAIRS

    def _tia_raw(path_fn):
        raw = {c: [None] * len(participants) for c in conditions}
        for ci, cond in enumerate(conditions):
            for pi2, pid in enumerate(participants):
                try:
                    raw[cond][pi2] = float(path_fn(pid, cond))
                except (KeyError, TypeError, ValueError):
                    pass
        return raw

    sec4 = f"{'─' * 78}\n  STATISTICAL ANALYSIS  (Friedman + Wilcoxon + Holm-Bonferroni)\n{'─' * 78}\n"
    sec4 += "  [✓] = Holm-Bonferroni corrected rejection at α=.05\n  r = rank-biserial\n\n"

    tia_metrics = [
        ("TiA Global mean (1–5)",
         lambda pid, c: all_data[pid]["tia"]["conditions"][c]["global_mean_excl_traits"]),
        ("Reliability/Competence (1–5)",
         lambda pid, c: all_data[pid]["tia"]["conditions"][c]["subscales"]["reliability_competence"]["mean"]),
        ("Understanding/Predictability (1–5)",
         lambda pid, c: all_data[pid]["tia"]["conditions"][c]["subscales"]["understanding_predictability"]["mean"]),
        ("Trust in Automation (1–5)",
         lambda pid, c: all_data[pid]["tia"]["conditions"][c]["subscales"]["trust_in_automation"]["mean"]),
    ]
    for metric_label, path_fn in tia_metrics:
        raw_m = _tia_raw(path_fn)
        chi2, p_f = _friedman_test([raw_m[c] for c in conditions])
        frd = {"chi2": chi2, "p": p_f, "df": len(conditions) - 1}
        ps_raw = [_wilcoxon_test(raw_m[bl], raw_m[comp])[0] for bl, comp in diff_pairs]
        rs_raw = [_wilcoxon_test(raw_m[bl], raw_m[comp])[1] for bl, comp in diff_pairs]
        reject_m, p_corr_m = _holm_correct(ps_raw)
        pw_m = [{"p_raw": ps_raw[i], "r": rs_raw[i], "reject": bool(reject_m[i]),
                 "p_corr": float(p_corr_m[i]),
                 "stars": _sig_stars(ps_raw[i], bool(reject_m[i]))} for i in range(len(diff_pairs))]
        sec4 += _fmt_stat_section(metric_label, frd, pw_m, diff_pairs)
        sec4 += "\n"

    path = os.path.join(REPORTS_DIR, "tia_report.txt")
    _write_report(path, "TiA — Trust in Automation Checklist  (cross-participant)", summary, [sec1, sec2, sec3, sec4])
    print(f"  → {path}")


# ── NASA-TLX report ───────────────────────────────────────────────────────────

def write_nasa_report(all_data, participants):
    conditions = CONDITIONS  # NASA uses all 4 including TARC

    dim_stats = {}
    for key, label in zip(_NASA_KEYS, _NASA_LABELS):
        dim_stats[key] = {}
        for cond in conditions:
            vals = []
            for pid in participants:
                try:
                    v = all_data[pid]["nasa_tlx"]["conditions"][cond]["subscales"][key]["rating_0_20"]
                    if v is not None:
                        vals.append(float(v))
                except (KeyError, TypeError):
                    pass
            dim_stats[key][cond] = _desc(vals)

    score_stats = {}
    for cond in conditions:
        vals = []
        for pid in participants:
            try:
                v = all_data[pid]["nasa_tlx"]["conditions"][cond]["nasa_tlx_weighted_score"]
                if v is not None:
                    vals.append(float(v))
            except (KeyError, TypeError):
                pass
        score_stats[cond] = _desc(vals)

    summary = {
        "form": "nasa_tlx",
        "n_participants": len(participants),
        "conditions": conditions,
        "nasa_tlx_weighted_score": score_stats,
        "dimensions": dim_stats,
    }

    w = 24
    sec1 = f"{'─' * 78}\n  WEIGHTED SCORE (0–100)\n{'─' * 78}\n"
    sec1 += _cond_header(w, conditions)
    sec1 += _fmt_desc_table(w, score_stats, conditions, "Weighted Score (0–100)")
    sec1 += "\n"

    sec2 = f"{'─' * 78}\n  DIMENSIONS (ratings 0–20, high = more workload)\n{'─' * 78}\n"
    sec2 += _cond_header(w, conditions)
    for key, label in zip(_NASA_KEYS, _NASA_LABELS):
        sec2 += _fmt_desc_table(w, dim_stats[key], conditions, label)
    sec2 += "\n"

    # ── Statistical analysis ──────────────────────────────────────────────────
    diff_pairs = _NASA_PAIRS
    raw_nasa = {c: [None] * len(participants) for c in conditions}
    for cond in conditions:
        for pi2, pid in enumerate(participants):
            try:
                v = all_data[pid]["nasa_tlx"]["conditions"][cond]["nasa_tlx_weighted_score"]
                raw_nasa[cond][pi2] = float(v) if v is not None else None
            except (KeyError, TypeError):
                pass
    chi2_n, p_n = _friedman_test([raw_nasa[c] for c in conditions])
    frd_nasa = {"chi2": chi2_n, "p": p_n, "df": len(conditions) - 1}
    ps_n = [_wilcoxon_test(raw_nasa[bl], raw_nasa[comp])[0] for bl, comp in diff_pairs]
    rs_n = [_wilcoxon_test(raw_nasa[bl], raw_nasa[comp])[1] for bl, comp in diff_pairs]
    rej_n, pc_n = _holm_correct(ps_n)
    pw_n = [{"p_raw": ps_n[i], "r": rs_n[i], "reject": bool(rej_n[i]),
              "p_corr": float(pc_n[i]), "stars": _sig_stars(ps_n[i], bool(rej_n[i]))}
            for i in range(len(diff_pairs))]
    sec3 = f"{'─' * 78}\n  STATISTICAL ANALYSIS  (Friedman + Wilcoxon + Holm-Bonferroni)\n{'─' * 78}\n"
    sec3 += "  [✓] = Holm-Bonferroni corrected rejection at α=.05\n  r = rank-biserial\n\n"
    sec3 += _fmt_stat_section("NASA-TLX Weighted Score (0–100, high=bad)", frd_nasa, pw_n, diff_pairs)
    sec3 += "\n"

    path = os.path.join(REPORTS_DIR, "nasa_tlx_report.txt")
    _write_report(path, "NASA-TLX — Cross-participant", summary, [sec1, sec2, sec3])
    print(f"  → {path}")


# ── Trust & Risk VAS report ───────────────────────────────────────────────────

def write_trust_risk_report(all_data, participants):
    conditions = CONDITIONS_MAIN

    stats = {}
    for key in ("trust_vas", "risk_vas"):
        stats[key] = {}
        for cond in conditions:
            vals = []
            for pid in participants:
                try:
                    v = all_data[pid]["trust_risk"]["conditions"][cond][key]
                    if v is not None:
                        vals.append(float(v))
                except (KeyError, TypeError):
                    pass
            stats[key][cond] = _desc(vals)

    summary = {
        "form": "trust_risk",
        "n_participants": len(participants),
        "conditions": conditions,
        "trust_vas": stats["trust_vas"],
        "risk_vas":  stats["risk_vas"],
    }

    w = 20
    sec1 = f"{'─' * 78}\n  TRUST VAS (0–100, high = more trust)\n{'─' * 78}\n"
    sec1 += _cond_header(w, conditions)
    sec1 += _fmt_desc_table(w, stats["trust_vas"], conditions, "Trust VAS (0–100)")
    sec1 += "\n"

    sec2 = f"{'─' * 78}\n  RISK VAS (0–100, high = more perceived risk)\n{'─' * 78}\n"
    sec2 += _cond_header(w, conditions)
    sec2 += _fmt_desc_table(w, stats["risk_vas"], conditions, "Risk VAS (0–100)")
    sec2 += "\n"

    # ── Statistical analysis ──────────────────────────────────────────────────
    diff_pairs = _ALL_PAIRS
    sec3 = f"{'─' * 78}\n  STATISTICAL ANALYSIS  (Friedman + Wilcoxon + Holm-Bonferroni)\n{'─' * 78}\n"
    sec3 += "  [✓] = Holm-Bonferroni corrected rejection at α=.05\n  r = rank-biserial\n\n"
    for vas_key, vas_label in [("trust_vas", "Trust VAS (0–100, high=good)"),
                                ("risk_vas",  "Risk VAS (0–100, high=bad)")]:
        raw_v = {c: [None] * len(participants) for c in conditions}
        for cond in conditions:
            for pi2, pid in enumerate(participants):
                try:
                    v = all_data[pid]["trust_risk"]["conditions"][cond][vas_key]
                    raw_v[cond][pi2] = float(v) if v is not None else None
                except (KeyError, TypeError):
                    pass
        chi2_v, p_v = _friedman_test([raw_v[c] for c in conditions])
        frd_v = {"chi2": chi2_v, "p": p_v, "df": len(conditions) - 1}
        ps_v = [_wilcoxon_test(raw_v[bl], raw_v[comp])[0] for bl, comp in diff_pairs]
        rs_v = [_wilcoxon_test(raw_v[bl], raw_v[comp])[1] for bl, comp in diff_pairs]
        rej_v, pc_v = _holm_correct(ps_v)
        pw_v = [{"p_raw": ps_v[i], "r": rs_v[i], "reject": bool(rej_v[i]),
                  "p_corr": float(pc_v[i]), "stars": _sig_stars(ps_v[i], bool(rej_v[i]))}
                for i in range(len(diff_pairs))]
        sec3 += _fmt_stat_section(vas_label, frd_v, pw_v, diff_pairs)
        sec3 += "\n"

    path = os.path.join(REPORTS_DIR, "trust_risk_report.txt")
    _write_report(path, "Trust & Risk VAS  (cross-participant)", summary, [sec1, sec2, sec3])
    print(f"  → {path}")


# ── Oversight Bespoke report ──────────────────────────────────────────────────

_OB_SCORES = [
    ("base_score",     "mean", "Base Score mean (1–5)"),
    ("extended_score", "mean", "Extended Score mean (1–5)"),
]


def write_ob_report(all_data, participants):
    conditions = CONDITIONS_MAIN

    item_stats = {}
    for key, label in zip(_OB_KEYS, _OB_LABELS):
        item_stats[key] = {}
        for cond in conditions:
            vals = []
            for pid in participants:
                try:
                    v = all_data[pid]["oversight_bespoke"]["conditions"][cond]["extended_score"]["items"][key]["scored"]
                    if v is not None:
                        vals.append(float(v))
                except (KeyError, TypeError):
                    pass
            item_stats[key][cond] = _desc(vals)

    score_stats = {}
    for score_key, sub_key, _ in _OB_SCORES:
        score_stats[score_key] = {}
        for cond in conditions:
            vals = []
            for pid in participants:
                try:
                    v = all_data[pid]["oversight_bespoke"]["conditions"][cond][score_key][sub_key]
                    if v is not None:
                        vals.append(float(v))
                except (KeyError, TypeError):
                    pass
            score_stats[score_key][cond] = _desc(vals)

    summary = {
        "form": "oversight_bespoke",
        "n_participants": len(participants),
        "conditions": conditions,
        "scores": score_stats,
        "items": item_stats,
    }

    w = 50
    sec1 = f"{'─' * 78}\n  COMPUTED SCORES (1–5)\n{'─' * 78}\n"
    sec1 += _cond_header(w, conditions)
    for score_key, sub_key, label in _OB_SCORES:
        sec1 += _fmt_desc_table(w, score_stats[score_key], conditions, label)
    sec1 += "\n"

    sec2 = f"{'─' * 78}\n  ITEMS (scored 1–5, * = reversed)\n{'─' * 78}\n"
    sec2 += _cond_header(w, conditions)
    for key, label in zip(_OB_KEYS, _OB_LABELS):
        sec2 += _fmt_desc_table(w, item_stats[key], conditions, label[:w])
    sec2 += "\n"

    # ── Statistical analysis — per item (no composite score for OB) ───────────
    diff_pairs = _ALL_PAIRS
    sec3 = f"{'─' * 78}\n  STATISTICAL ANALYSIS — PER ITEM  (Friedman + Wilcoxon + Holm-Bonferroni)\n{'─' * 78}\n"
    sec3 += "  Note: OB is an in-house questionnaire — no validated composite score.\n"
    sec3 += "  Tests run per item; Holm correction applied within each item (4 pairs).\n"
    sec3 += "  [✓] = Holm-Bonferroni corrected rejection at α=.05\n  r = rank-biserial\n\n"
    for ob_key, ob_label in zip(_OB_KEYS, _OB_LABELS):
        raw_ob = {c: [None] * len(participants) for c in conditions}
        for cond in conditions:
            for pi2, pid in enumerate(participants):
                try:
                    v = all_data[pid]["oversight_bespoke"]["conditions"][cond]["extended_score"]["items"][ob_key]["scored"]
                    raw_ob[cond][pi2] = float(v) if v is not None else None
                except (KeyError, TypeError):
                    pass
        chi2_ob, p_ob = _friedman_test([raw_ob[c] for c in conditions])
        frd_ob = {"chi2": chi2_ob, "p": p_ob, "df": len(conditions) - 1}
        ps_ob = [_wilcoxon_test(raw_ob[bl], raw_ob[comp])[0] for bl, comp in diff_pairs]
        rs_ob = [_wilcoxon_test(raw_ob[bl], raw_ob[comp])[1] for bl, comp in diff_pairs]
        rej_ob, pc_ob = _holm_correct(ps_ob)
        pw_ob = [{"p_raw": ps_ob[i], "r": rs_ob[i], "reject": bool(rej_ob[i]),
                   "p_corr": float(pc_ob[i]), "stars": _sig_stars(ps_ob[i], bool(rej_ob[i]))}
                 for i in range(len(diff_pairs))]
        sec3 += _fmt_stat_section(ob_label[:60], frd_ob, pw_ob, diff_pairs)
        sec3 += "\n"

    path = os.path.join(REPORTS_DIR, "oversight_bespoke_report.txt")
    _write_report(path, "Oversight Bespoke  (cross-participant)", summary, [sec1, sec2, sec3])
    print(f"  → {path}")


# ── Perceived Control report ──────────────────────────────────────────────────

def write_perceived_control_report(all_data, participants):
    conditions = CONDITIONS_MAIN

    item_stats = {}
    for key, label in zip(_PC_KEYS, _PC_LABELS):
        item_stats[key] = {}
        for cond in conditions:
            vals = []
            for pid in participants:
                try:
                    v = all_data[pid]["perceived_control"]["conditions"][cond]["items"][key]["scored"]
                    if v is not None:
                        vals.append(float(v))
                except (KeyError, TypeError):
                    pass
            item_stats[key][cond] = _desc(vals)

    mean_stats = {}
    for cond in conditions:
        vals = []
        for pid in participants:
            try:
                v = all_data[pid]["perceived_control"]["conditions"][cond]["mean"]
                if v is not None:
                    vals.append(float(v))
            except (KeyError, TypeError):
                pass
        mean_stats[cond] = _desc(vals)

    summary = {
        "form": "perceived_control",
        "n_participants": len(participants),
        "conditions": conditions,
        "mean": mean_stats,
        "items": item_stats,
    }

    w = 70
    sec1 = f"{'─' * 78}\n  MEAN SCORE (1–5)\n{'─' * 78}\n"
    sec1 += _cond_header(w, conditions)
    sec1 += _fmt_desc_table(w, mean_stats, conditions, "Mean (1–5)")
    sec1 += "\n"

    sec2 = f"{'─' * 78}\n  ITEMS (scored 1–5)\n{'─' * 78}\n"
    sec2 += _cond_header(w, conditions)
    for key, label in zip(_PC_KEYS, _PC_LABELS):
        sec2 += _fmt_desc_table(w, item_stats[key], conditions, label[:w])
    sec2 += "\n"

    # ── Statistical analysis — per item (no validated composite for PC) ─────
    diff_pairs = _ALL_PAIRS
    sec3 = f"{'─' * 78}\n  STATISTICAL ANALYSIS — PER ITEM  (Friedman + Wilcoxon + Holm-Bonferroni)\n{'─' * 78}\n"
    sec3 += "  Note: Perceived Control is an in-house questionnaire — no validated composite score.\n"
    sec3 += "  Tests run per item; Holm correction applied within each item (4 pairs).\n"
    sec3 += "  [✓] = Holm-Bonferroni corrected rejection at α=.05\n  r = rank-biserial\n\n"
    for pc_key, pc_label in zip(_PC_KEYS, _PC_LABELS):
        raw_pc_item = {c: [None] * len(participants) for c in conditions}
        for cond in conditions:
            for pi2, pid in enumerate(participants):
                try:
                    v = all_data[pid]["perceived_control"]["conditions"][cond]["items"][pc_key]["scored"]
                    raw_pc_item[cond][pi2] = float(v) if v is not None else None
                except (KeyError, TypeError):
                    pass
        chi2_pc, p_pc = _friedman_test([raw_pc_item[c] for c in conditions])
        frd_pc = {"chi2": chi2_pc, "p": p_pc, "df": len(conditions) - 1}
        ps_pc = [_wilcoxon_test(raw_pc_item[bl], raw_pc_item[comp])[0] for bl, comp in diff_pairs]
        rs_pc = [_wilcoxon_test(raw_pc_item[bl], raw_pc_item[comp])[1] for bl, comp in diff_pairs]
        rej_pc, pc_corr = _holm_correct(ps_pc)
        pw_pc = [{"p_raw": ps_pc[i], "r": rs_pc[i], "reject": bool(rej_pc[i]),
                   "p_corr": float(pc_corr[i]), "stars": _sig_stars(ps_pc[i], bool(rej_pc[i]))}
                 for i in range(len(diff_pairs))]
        sec3 += _fmt_stat_section(pc_label[:70], frd_pc, pw_pc, diff_pairs)
        sec3 += "\n"

    path = os.path.join(REPORTS_DIR, "perceived_control_report.txt")
    _write_report(path, "Perceived Control  (cross-participant)", summary, [sec1, sec2, sec3])
    print(f"  → {path}")


# ── Pre-experiment / PTS report ───────────────────────────────────────────────

def write_pts_report(all_data, participants):
    """Propensity to Trust Automation — trait questionnaire (no conditions)."""
    item_stats = {}
    for key, label in zip(_PTS_KEYS, _PTS_LABELS):
        vals = []
        for pid in participants:
            try:
                v = all_data[pid]["pre_experiment_form"]["pts"][key]["score"]
                if v is not None:
                    vals.append(float(v))
            except (KeyError, TypeError):
                pass
        item_stats[key] = _desc(vals)

    total_stats = {}
    total_vals = []
    for pid in participants:
        try:
            v = all_data[pid]["pre_experiment_form"]["pts_total"]
            if v is not None:
                total_vals.append(float(v))
        except (KeyError, TypeError):
            pass
    total_stats = _desc(total_vals)

    summary = {
        "form": "pre_experiment_form_pts",
        "n_participants": len(participants),
        "pts_total": total_stats,
        "items": item_stats,
    }

    w = 70
    sep = "─" * 78
    sec1 = f"{sep}\n  PTS TOTAL SCORE\n{sep}\n"
    s = total_stats
    if s["n"] > 0:
        sec1 += f"  n={s['n']}  μ={s['mean']:.2f}  σ={s['sd']:.2f}  Md={s['median']:.2f}  [{s['min']}–{s['max']}]\n"
    else:
        sec1 += "  No data\n"
    sec1 += "\n"

    sec2 = f"{sep}\n  ITEMS (scored 1–5, high = trusting)\n{sep}\n"
    col_h = f"  {'Item':<{w}}  {'n':>3}  {'mean':>5}  {'SD':>4}  {'Md':>5}  [min–max]\n"
    sec2 += col_h + "  " + "─" * (w + 35) + "\n"
    for key, label in zip(_PTS_KEYS, _PTS_LABELS):
        s = item_stats[key]
        if s["n"] > 0:
            bar = _bar_r(s["mean"], 1.0, 5.0, width=15)
            sec2 += f"  {label[:w]:<{w}}  {s['n']:>3}  {s['mean']:>5.2f}  {s['sd']:>4.2f}  {s['median']:>5.2f}  [{s['min']}–{s['max']}]  {bar}\n"
        else:
            sec2 += f"  {label[:w]:<{w}}  {'—':>3}\n"
    sec2 += "\n"

    path = os.path.join(REPORTS_DIR, "pts_report.txt")
    _write_report(path, "Propensity to Trust Automation (PTS) — Pre-experiment  (cross-participant)", summary, [sec1, sec2])
    print(f"  → {path}")


def write_preference_cluster_report(all_data, participants):
    """Quadrant preference interpretation report (cross-participant)."""

    pids, xs, ys, feat = _build_cluster_data(
        all_data, participants, include_ob=True, include_nasa=True, include_vas=True
    )

    def _quadrant(x, y):
        if x >= 0 and y >= 0:
            return "Q1", "Both modes above TARS baseline"
        if x < 0 and y >= 0:
            return "Q2", "TARP-F above baseline; TARP-S below"
        if x >= 0 and y < 0:
            return "Q3", "TARP-S above baseline; TARP-F below"
        return "Q4", "Both modes below TARS baseline"

    rows = []
    for i, pid in enumerate(pids):
        x = float(xs[i])
        y = float(ys[i])
        q, q_txt = _quadrant(x, y)
        mode_balance = y - x
        if mode_balance > 0.05:
            pref = "Relative tilt toward TARP-F"
        elif mode_balance < -0.05:
            pref = "Relative tilt toward TARP-S"
        else:
            pref = "Near parity between TARP-S and TARP-F"
        rows.append({
            "participant": pid,
            "x_tarps_minus_tars": x,
            "y_tarpf_minus_tars": y,
            "quadrant": q,
            "quadrant_meaning": q_txt,
            "relative_mode_tilt": pref,
            "mode_balance_y_minus_x": float(mode_balance),
        })

    quad_counts = {"Q1": 0, "Q2": 0, "Q3": 0, "Q4": 0}
    mode_counts = {"TARP-S": 0, "TARP-F": 0, "Parity": 0}
    for r in rows:
        quad_counts[r["quadrant"]] += 1
        if r["mode_balance_y_minus_x"] > 0.05:
            mode_counts["TARP-F"] += 1
        elif r["mode_balance_y_minus_x"] < -0.05:
            mode_counts["TARP-S"] += 1
        else:
            mode_counts["Parity"] += 1

    if rows:
        balances = [r["mode_balance_y_minus_x"] for r in rows]
        mean_balance = float(np.mean(balances))
        med_balance = float(np.median(balances))
    else:
        mean_balance = float("nan")
        med_balance = float("nan")

    summary = {
        "analysis": "preference_quadrant",
        "n_participants_total": len(participants),
        "n_participants_used": len(pids),
        "feature_definition": "Per-item polarity-corrected means projected as x=(TARP-S-TARS), y=(TARP-F-TARS)",
        "scatter_axes": {
            "x": "mean(TARP-S - TARS)",
            "y": "mean(TARP-F - TARS)",
        },
        "quadrant_counts": quad_counts,
        "mode_preference_counts": mode_counts,
        "mode_balance": {
            "definition": "y - x (positive=more TARP-F, negative=more TARP-S)",
            "mean": mean_balance,
            "median": med_balance,
        },
        "participants": rows,
    }

    sep = "─" * 78
    sec1 = f"{sep}\n  WHAT THIS FIGURE MEASURES\n{sep}\n"
    sec1 += "  The quadrant plot combines all questionnaire families (SUS, TiA, Oversight,\n"
    sec1 += "  Perceived Control, NASA-TLX polarity-corrected, Trust/Risk VAS) into two\n"
    sec1 += "  participant-level coordinates:\n"
    sec1 += "    x = mean(TARP-S - TARS),  y = mean(TARP-F - TARS).\n"
    sec1 += "  Positive values mean better than TARS baseline.\n"
    sec1 += "\n"
    sec1 += "  Relative mode preference is summarized by mode balance:  (y - x).\n"
    sec1 += "  Positive values indicate stronger tilt toward TARP-F; negative values toward TARP-S.\n\n"

    sec2 = f"{sep}\n  HOW TO READ THE QUADRANT\n{sep}\n"
    sec2 += "  Q1 (+x, +y): both modes better than baseline TARS.\n"
    sec2 += "  Q2 (-x, +y): only TARP-F better than baseline.\n"
    sec2 += "  Q3 (+x, -y): only TARP-S better than baseline.\n"
    sec2 += "  Q4 (-x, -y): both modes worse than baseline.\n"
    sec2 += "\n"
    sec2 += "  Relative mode preference is read against the diagonal y=x:\n"
    sec2 += "    y > x  -> tilt toward TARP-F\n"
    sec2 += "    y < x  -> tilt toward TARP-S\n"
    sec2 += "\n"

    sec3 = f"{sep}\n  RESULTS SUMMARY\n{sep}\n"
    sec3 += f"  Participants in analysis: {len(pids)} / {len(participants)}\n"
    sec3 += (
        f"  Quadrant counts: Q1={quad_counts['Q1']}, Q2={quad_counts['Q2']}, "
        f"Q3={quad_counts['Q3']}, Q4={quad_counts['Q4']}\n"
    )
    sec3 += (
        f"  Mode preference counts: TARP-S tilt={mode_counts['TARP-S']}, "
        f"TARP-F tilt={mode_counts['TARP-F']}, parity={mode_counts['Parity']}\n"
    )
    if not np.isnan(mean_balance):
        sec3 += f"  Mean mode balance (y-x): {mean_balance:.3f}\n"
        sec3 += f"  Median mode balance (y-x): {med_balance:.3f}\n"
    sec3 += "\n"

    sec4 = f"{sep}\n  PARTICIPANT-LEVEL INTERPRETATION\n{sep}\n"
    sec4 += (
        f"  {'Participant':<11}  {'x(TS-TARS)':>10}  {'y(TF-TARS)':>10}  {'y-x':>7}  {'Quad':>4}  Interpretation\n"
    )
    sec4 += "  " + "─" * 74 + "\n"
    for r in rows:
        sec4 += (
            f"  {r['participant']:<11}  {r['x_tarps_minus_tars']:>10.3f}  "
            f"{r['y_tarpf_minus_tars']:>10.3f}  {r['mode_balance_y_minus_x']:>7.3f}  {r['quadrant']:>4}  "
            f"{r['quadrant_meaning']}; {r['relative_mode_tilt']}\n"
        )
    sec4 += "\n"

    path = os.path.join(REPORTS_DIR, "preference_clusters_report.txt")
    _write_report(path, "Preference Quadrant — TARP-S vs TARP-F Interpretation  (cross-participant)", summary,
                  [sec1, sec2, sec3, sec4])
    print(f"  → {path}")


def write_all_reports(all_data, participants):
    os.makedirs(REPORTS_DIR, exist_ok=True)
    print(f"\n[4/4] Generating cross-participant reports → {REPORTS_DIR}/\n")
    write_sus_report(all_data, participants)
    write_tia_report(all_data, participants)
    write_nasa_report(all_data, participants)
    write_trust_risk_report(all_data, participants)
    write_ob_report(all_data, participants)
    write_perceived_control_report(all_data, participants)
    write_pts_report(all_data, participants)
    write_preference_cluster_report(all_data, participants)


_COMPARE_FORMS_PLOTS = [
    "sus_items.png",
    "tia_items.png",
    "ob_items.png",
    "nasa_tlx_ratings.png",
    "nasa_tlx_scores.png",
    "perceived_control_items.png",
    "trust_risk_distribution.png",
    "pts_items.png",
    "preference_profile_coherence.png",
    "preference_clusters_all_vas.png",
]


_COMPARE_FORMS_REPORTS = [
    "sus_report.txt",
    "tia_report.txt",
    "nasa_tlx_report.txt",
    "trust_risk_report.txt",
    "oversight_bespoke_report.txt",
    "perceived_control_report.txt",
    "pts_report.txt",
    "preference_clusters_report.txt",
]


def _confirm_run(participants, missing_pids, output_plots):
    """Print a pre-run summary and ask the user to confirm before proceeding."""
    print()
    if missing_pids:
        print(f"Reports to generate ({len(missing_pids)} participant(s) missing reports):")
        for pid in missing_pids:
            print(f"  + {pid}")
    else:
        print("All participant reports are up to date.")
    print(f"\nOutput plots that will be written/overwritten ({len(output_plots)}):")
    for name in output_plots:
        path = os.path.join(PLOTS_DIR, name)
        tag  = "[overwrite]" if os.path.exists(path) else "[new     ]"
        print(f"  {tag}  {name}")
    print(f"\nCross-participant reports that will be written/overwritten ({len(_COMPARE_FORMS_REPORTS)}):")
    for name in _COMPARE_FORMS_REPORTS:
        path = os.path.join(REPORTS_DIR, name)
        tag  = "[overwrite]" if os.path.exists(path) else "[new     ]"
        print(f"  {tag}  compare_forms/{name}")
    print()
    try:
        ans = input("Continue? [Y/n]: ").strip().lower()
    except KeyboardInterrupt:
        print("\nAborted.")
        return False
    return ans in ("", "y", "yes")


def main():
    print("=" * 70)
    print("  HITLS — Cross-participant Questionnaire Comparison")
    print("=" * 70)

    participants = find_participants()
    print(f"\nParticipants found: {', '.join(participants)}")

    missing = [pid for pid in participants if not has_all_reports(pid)]
    if not _confirm_run(participants, missing, _COMPARE_FORMS_PLOTS):
        return

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
    figs.extend(plot_nasa(all_data, loaded))
    figs.append(plot_perceived_control(all_data, loaded))
    figs.append(plot_trust_risk_distribution(all_data, loaded))
    figs.append(plot_pts(all_data, loaded))
    figs.append(plot_preference_profile(all_data, loaded))
    figs.extend(plot_preference_clusters(all_data, loaded))

    print(f"\nDone — {len(figs)} figures saved to {PLOTS_DIR}/")

    write_all_reports(all_data, loaded)

    plt.show()


if __name__ == "__main__":
    main()
