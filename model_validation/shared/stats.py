"""
stats.py — Shared statistical utilities for model validation.

Conventions follow the HITLS compare_forms.py / compare_performance.py codebase:
  - Friedman + pairwise Wilcoxon signed-rank + Holm-Bonferroni correction
  - Effect sizes: rank-biserial r (pairwise), Kendall's W (Friedman)
  - All tests drop None / NaN values before computing
  - Minimum sample requirements enforced silently (returns sentinel values)
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from scipy.stats import (
    friedmanchisquare,
    ks_2samp,
    pearsonr,
    spearmanr,
    wilcoxon as _sp_wilcoxon,
)
from statsmodels.stats.multitest import multipletests

# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class WilcoxonResult:
    condition_a: str
    condition_b: str
    p_raw: float
    p_corrected: float       # after Holm-Bonferroni (set by wilcoxon_pairwise)
    reject: bool             # H0 rejected at α=0.05 after correction
    r: float                 # rank-biserial effect size
    n: int                   # number of valid pairs used


@dataclass
class FriedmanResult:
    chi2: float
    p: float
    df: int                  # k − 1
    kendall_w: float         # = chi2 / (n * df)
    n: int                   # number of valid participants


@dataclass
class HitlsDistribution:
    """Descriptive summary of a metric across HITLS participants.

    Used for participant-envelope plots in Tier 2 diagnostic figures:
    the model value is overlaid on the human [min, mean ± SD, max] band.
    """
    group: str               # e.g. task label, condition, AoI name
    mean: float
    sd: float
    min_val: float
    max_val: float
    pid_min: str             # participant ID with the lowest value
    pid_max: str             # participant ID with the highest value
    n: int
    raw: list[float] = field(default_factory=list)   # individual values


# ---------------------------------------------------------------------------
# Basic metrics
# ---------------------------------------------------------------------------

def rmse(predicted: Sequence[float], observed: Sequence[float]) -> float:
    """Root Mean Squared Error between predicted and observed arrays."""
    p = np.asarray(predicted, dtype=float)
    o = np.asarray(observed, dtype=float)
    mask = np.isfinite(p) & np.isfinite(o)
    if mask.sum() == 0:
        return float("nan")
    return float(np.sqrt(np.mean((p[mask] - o[mask]) ** 2)))


def nmae(predicted: float, observed_mean: float) -> float:
    """Normalised Mean Absolute Error for a single model value vs human mean.

    NMAE = (predicted − observed_mean) / observed_mean
    Returns nan if observed_mean == 0.
    """
    if observed_mean == 0:
        return float("nan")
    return (predicted - observed_mean) / observed_mean


def pearson_r(a: Sequence[float], b: Sequence[float]) -> tuple[float, float]:
    """Pearson correlation.  Returns (r, p_value)."""
    arr_a = np.asarray(a, dtype=float)
    arr_b = np.asarray(b, dtype=float)
    mask = np.isfinite(arr_a) & np.isfinite(arr_b)
    if mask.sum() < 3:
        return float("nan"), float("nan")
    r, p = pearsonr(arr_a[mask], arr_b[mask])
    return float(r), float(p)


def spearman_r(a: Sequence[float], b: Sequence[float]) -> tuple[float, float]:
    """Spearman rank correlation.  Returns (rho, p_value)."""
    arr_a = np.asarray(a, dtype=float)
    arr_b = np.asarray(b, dtype=float)
    mask = np.isfinite(arr_a) & np.isfinite(arr_b)
    if mask.sum() < 3:
        return float("nan"), float("nan")
    rho, p = spearmanr(arr_a[mask], arr_b[mask])
    return float(rho), float(p)


def ks_2samp_test(
    a: Sequence[float], b: Sequence[float]
) -> tuple[float, float]:
    """Two-sample Kolmogorov-Smirnov test.  Returns (statistic, p_value)."""
    arr_a = np.asarray(a, dtype=float)
    arr_b = np.asarray(b, dtype=float)
    arr_a = arr_a[np.isfinite(arr_a)]
    arr_b = arr_b[np.isfinite(arr_b)]
    if len(arr_a) < 2 or len(arr_b) < 2:
        return float("nan"), float("nan")
    stat, p = ks_2samp(arr_a, arr_b)
    return float(stat), float(p)


def kl_divergence(p: Sequence[float], q: Sequence[float]) -> float:
    """KL divergence D_KL(P ‖ Q).

    Both sequences are treated as probability masses and normalised.
    A small epsilon is added to Q to avoid division by zero.
    """
    p_arr = np.asarray(p, dtype=float)
    q_arr = np.asarray(q, dtype=float)
    p_arr = p_arr / p_arr.sum()
    q_arr = q_arr / (q_arr.sum() + 1e-12) + 1e-12
    q_arr = q_arr / q_arr.sum()
    return float(np.sum(p_arr * np.log(p_arr / q_arr + 1e-12)))


# ---------------------------------------------------------------------------
# Inferential tests (matching HITLS compare_forms.py conventions)
# ---------------------------------------------------------------------------

def _clean_pairs(a: Sequence, b: Sequence) -> tuple[np.ndarray, np.ndarray]:
    """Return arrays of finite paired values, dropping any pair with a None/NaN."""
    pairs = [
        (float(x), float(y))
        for x, y in zip(a, b)
        if x is not None and y is not None
        and np.isfinite(float(x)) and np.isfinite(float(y))
    ]
    if not pairs:
        return np.array([]), np.array([])
    return np.array([p[0] for p in pairs]), np.array([p[1] for p in pairs])


def _wilcoxon_pair(a: Sequence, b: Sequence) -> tuple[float, float, int]:
    """Paired Wilcoxon signed-rank test.  Returns (p_value, rank_biserial_r, n).
    Requires ≥ 4 valid pairs; returns (1.0, 0.0, 0) otherwise.
    """
    xa, xb = _clean_pairs(a, b)
    n = len(xa)
    if n < 4:
        return 1.0, 0.0, n
    try:
        res = _sp_wilcoxon(xa, xb, alternative="two-sided", zero_method="wilcox")
        r = 1.0 - 2.0 * res.statistic / (n * (n + 1) / 2.0)
        return float(res.pvalue), float(r), n
    except Exception:
        return 1.0, 0.0, n


def friedman_test(
    groups: dict[str, Sequence[float]],
) -> FriedmanResult:
    """Friedman chi-square test for k repeated-measures groups.

    Parameters
    ----------
    groups : dict mapping condition label → list of per-participant values.
        Lists must be in the same participant order.

    Returns
    -------
    FriedmanResult
    """
    conditions = list(groups.keys())
    arrays = [groups[c] for c in conditions]
    # Drop participants (rows) with any None/NaN across conditions
    rows = list(zip(*arrays))
    valid = [
        r for r in rows
        if all(x is not None and np.isfinite(float(x)) for x in r)
    ]
    n = len(valid)
    k = len(conditions)
    if n < 3 or k < 2:
        return FriedmanResult(chi2=0.0, p=1.0, df=k - 1, kendall_w=0.0, n=n)
    aligned = [[float(r[i]) for r in valid] for i in range(k)]
    try:
        stat, p = friedmanchisquare(*aligned)
        w = stat / (n * (k - 1)) if n * (k - 1) > 0 else 0.0
        return FriedmanResult(
            chi2=float(stat), p=float(p), df=k - 1, kendall_w=float(w), n=n
        )
    except Exception:
        return FriedmanResult(chi2=0.0, p=1.0, df=k - 1, kendall_w=0.0, n=n)


def wilcoxon_pairwise(
    groups: dict[str, Sequence[float]],
    alpha: float = 0.05,
) -> list[WilcoxonResult]:
    """All pairwise Wilcoxon signed-rank tests with Holm-Bonferroni correction.

    Parameters
    ----------
    groups : dict mapping condition label → list of per-participant values.
    alpha  : family-wise error rate for Holm correction.

    Returns
    -------
    List of WilcoxonResult, one per pair, with p_corrected filled in.
    """
    pairs = list(itertools.combinations(groups.keys(), 2))
    raw_results = []
    for cond_a, cond_b in pairs:
        p, r, n = _wilcoxon_pair(groups[cond_a], groups[cond_b])
        raw_results.append(
            WilcoxonResult(
                condition_a=cond_a,
                condition_b=cond_b,
                p_raw=p,
                p_corrected=p,   # filled below
                reject=False,
                r=r,
                n=n,
            )
        )
    if not raw_results:
        return raw_results
    pvals = [r.p_raw for r in raw_results]
    reject_arr, pvals_corr, _, _ = multipletests(
        pvals, alpha=alpha, method="holm"
    )
    for res, rej, p_corr in zip(raw_results, reject_arr, pvals_corr):
        res.reject = bool(rej)
        res.p_corrected = float(p_corr)
    return raw_results


def sig_stars(p_raw: float, reject: bool) -> str:
    """Return APA-style significance asterisks (or '' if not significant)."""
    if not reject:
        return ""
    if p_raw < 0.001:
        return "***"
    if p_raw < 0.01:
        return "**"
    return "*"


# ---------------------------------------------------------------------------
# HitlsDistribution construction
# ---------------------------------------------------------------------------

def build_hitls_distributions(
    df: pd.DataFrame,
    group_col: str,
    value_col: str,
    pid_col: str = "participant",
) -> list[HitlsDistribution]:
    """Build one HitlsDistribution per unique value of *group_col*.

    Parameters
    ----------
    df        : DataFrame with at least [group_col, value_col, pid_col].
    group_col : Column that identifies groups (e.g. 'Task_Object', 'AoI_Name').
    value_col : Numeric metric column.
    pid_col   : Column with participant IDs (used to record pid_min / pid_max).

    Returns
    -------
    List of HitlsDistribution, one per group, in original order.
    """
    results: list[HitlsDistribution] = []
    for group, sub in df.groupby(group_col, sort=False):
        vals = sub[value_col].dropna().astype(float)
        pids = sub.loc[vals.index, pid_col] if pid_col in sub.columns else pd.Series(["?"] * len(vals))
        if vals.empty:
            results.append(
                HitlsDistribution(
                    group=str(group), mean=float("nan"), sd=float("nan"),
                    min_val=float("nan"), max_val=float("nan"),
                    pid_min="?", pid_max="?", n=0,
                )
            )
            continue
        idx_min = vals.idxmin()
        idx_max = vals.idxmax()
        results.append(
            HitlsDistribution(
                group=str(group),
                mean=float(vals.mean()),
                sd=float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
                min_val=float(vals.min()),
                max_val=float(vals.max()),
                pid_min=str(pids.loc[idx_min]) if idx_min in pids.index else "?",
                pid_max=str(pids.loc[idx_max]) if idx_max in pids.index else "?",
                n=int(len(vals)),
                raw=vals.tolist(),
            )
        )
    return results


# ---------------------------------------------------------------------------
# Diagnostic envelope plot helper
# ---------------------------------------------------------------------------

# Shared color scheme (consistent across all debug figures)
_ENVELOPE_COLOR = "#4878CF"      # HITLS human band
_MODEL_COLOR    = "#D65F5F"      # model mean
_REP_COLOR      = "#D65F5F"      # model repetitions (lighter)
_NOTMODELLED_COLOR = "#BBBBBB"   # grey for [NOT MODELLED] rows


def plot_participant_envelope(
    ax: plt.Axes,
    distributions: list[HitlsDistribution],
    model_values: Sequence[float | None],
    model_reps: Optional[list[list[float]]] = None,
    y_label: str = "Value",
    orientation: str = "horizontal",
    highlight_threshold: Optional[float] = None,
) -> None:
    """Draw the HITLS participant envelope with model values overlaid.

    Intended for Tier 2 per-task diagnostic figures.

    Layout
    ------
    For each group (row when orientation='horizontal'):
      • Shaded band: human mean ± SD (blue)
      • Whiskers:    human min and max (thin lines)
      • ●  blue dot:  human mean
      • ●  red dot:   model mean
      • Small grey dots (optional): individual model repetitions
      • Grey row: [NOT MODELLED] — distribution with NaN mean

    Parameters
    ----------
    ax               : Matplotlib Axes to draw on.
    distributions    : List of HitlsDistribution, one per group/row.
    model_values     : Model mean (or None) for each distribution group,
                       in matching order.
    model_reps       : Optional list of per-group repetition lists
                       (12 values each). Plotted as small jittered dots.
    y_label          : Axis label for the value axis.
    orientation      : 'horizontal' (groups on y-axis) or 'vertical'
                       (groups on x-axis).
    highlight_threshold : If set, model points with |NMAE| > threshold
                          are outlined in red.
    """
    n = len(distributions)
    positions = np.arange(n)
    group_labels = [d.group for d in distributions]

    horiz = orientation == "horizontal"

    for i, (dist, mv) in enumerate(zip(distributions, model_values)):
        pos = i

        if np.isnan(dist.mean):
            # Not modelled — draw grey placeholder
            if horiz:
                ax.barh(pos, 0, height=0.4, color=_NOTMODELLED_COLOR,
                        alpha=0.5, zorder=1)
                ax.text(0.02, pos, "[NOT MODELLED]", va="center",
                        ha="left", fontsize=7, color="#888888",
                        transform=ax.get_yaxis_transform())
            else:
                ax.bar(pos, 0, width=0.4, color=_NOTMODELLED_COLOR,
                       alpha=0.5, zorder=1)
            continue

        lo = dist.mean - dist.sd
        hi = dist.mean + dist.sd

        if horiz:
            # SD band
            ax.barh(pos, hi - lo, left=lo, height=0.5,
                    color=_ENVELOPE_COLOR, alpha=0.25, zorder=2)
            # min–max whiskers
            ax.plot([dist.min_val, dist.min_val], [pos - 0.25, pos + 0.25],
                    color=_ENVELOPE_COLOR, lw=1.0, zorder=3)
            ax.plot([dist.max_val, dist.max_val], [pos - 0.25, pos + 0.25],
                    color=_ENVELOPE_COLOR, lw=1.0, zorder=3)
            ax.plot([dist.min_val, dist.max_val], [pos, pos],
                    color=_ENVELOPE_COLOR, lw=0.8, alpha=0.5, zorder=2)
            # Human mean dot
            ax.scatter(dist.mean, pos, color=_ENVELOPE_COLOR, s=40,
                       zorder=5, label="Human mean" if i == 0 else "")
            # Model repetitions
            if model_reps and model_reps[i]:
                reps = np.array(model_reps[i], dtype=float)
                jitter = np.random.default_rng(42).uniform(
                    -0.15, 0.15, size=len(reps)
                )
                ax.scatter(reps, pos + jitter, color=_REP_COLOR, s=12,
                           alpha=0.4, zorder=4)
            # Model mean dot
            if mv is not None and np.isfinite(mv):
                outside = (mv < dist.min_val or mv > dist.max_val)
                edge = "black" if outside else _MODEL_COLOR
                ax.scatter(mv, pos, color=_MODEL_COLOR, s=60, zorder=6,
                           edgecolors=edge, linewidths=1.5,
                           label="Model mean" if i == 0 else "")
        else:
            ax.bar(pos, hi - lo, bottom=lo, width=0.5,
                   color=_ENVELOPE_COLOR, alpha=0.25, zorder=2)
            ax.plot([pos - 0.25, pos + 0.25],
                    [dist.min_val, dist.min_val],
                    color=_ENVELOPE_COLOR, lw=1.0, zorder=3)
            ax.plot([pos - 0.25, pos + 0.25],
                    [dist.max_val, dist.max_val],
                    color=_ENVELOPE_COLOR, lw=1.0, zorder=3)
            ax.scatter(pos, dist.mean, color=_ENVELOPE_COLOR, s=40,
                       zorder=5, label="Human mean" if i == 0 else "")
            if model_reps and model_reps[i]:
                reps = np.array(model_reps[i], dtype=float)
                jitter = np.random.default_rng(42).uniform(
                    -0.15, 0.15, size=len(reps)
                )
                ax.scatter(pos + jitter, reps, color=_REP_COLOR, s=12,
                           alpha=0.4, zorder=4)
            if mv is not None and np.isfinite(mv):
                outside = (mv < dist.min_val or mv > dist.max_val)
                edge = "black" if outside else _MODEL_COLOR
                ax.scatter(pos, mv, color=_MODEL_COLOR, s=60, zorder=6,
                           edgecolors=edge, linewidths=1.5,
                           label="Model mean" if i == 0 else "")

    # Axis labels and ticks
    if horiz:
        ax.set_yticks(positions)
        ax.set_yticklabels(group_labels, fontsize=8)
        ax.set_xlabel(y_label, fontsize=9)
        ax.invert_yaxis()
    else:
        ax.set_xticks(positions)
        ax.set_xticklabels(group_labels, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel(y_label, fontsize=9)

    # Legend (deduplicated)
    handles = [
        mpatches.Patch(color=_ENVELOPE_COLOR, alpha=0.4, label="Human mean ± SD"),
        plt.Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=_ENVELOPE_COLOR, markersize=7,
                   label="Human mean"),
        plt.Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=_MODEL_COLOR, markersize=9,
                   label="Model mean"),
    ]
    if model_reps:
        handles.append(
            plt.Line2D([0], [0], marker="o", color="w",
                       markerfacecolor=_REP_COLOR, markersize=5,
                       alpha=0.4, label="Model repetitions")
        )
    ax.legend(handles=handles, fontsize=7, loc="lower right")


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def fmt_p(p: float) -> str:
    """Format a p-value for reports (e.g. p = .034, p < .001)."""
    if np.isnan(p):
        return "p = n/a"
    if p < 0.001:
        return "p < .001"
    return f"p = {p:.3f}".replace("0.", ".")


def fmt_stat_line(
    label: str,
    friedman: FriedmanResult,
    pairwise: list[WilcoxonResult],
) -> str:
    """Return a one-block human-readable stat summary for a metric."""
    stars = sig_stars(friedman.p, friedman.p < 0.05)
    lines = [
        f"{label}",
        f"  Friedman: χ²({friedman.df}, N={friedman.n}) = {friedman.chi2:.3f}, "
        f"{fmt_p(friedman.p)} {stars}  W = {friedman.kendall_w:.3f}",
    ]
    for pw in pairwise:
        s = sig_stars(pw.p_raw, pw.reject)
        lines.append(
            f"  {pw.condition_a} vs {pw.condition_b}: "
            f"{fmt_p(pw.p_raw)} (corr: {fmt_p(pw.p_corrected)}) "
            f"r = {pw.r:+.3f}  n = {pw.n} {s}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI smoke-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(0)

    # Synthetic data: 18 participants, 3 conditions
    data = {
        "TARS":   rng.normal(55, 16, 18).tolist(),
        "TARP-S": rng.normal(47, 17, 18).tolist(),
        "TARP-F": rng.normal(46, 18, 18).tolist(),
    }

    print("=== Friedman test ===")
    fr = friedman_test(data)
    print(f"  χ²({fr.df}, N={fr.n}) = {fr.chi2:.3f}, {fmt_p(fr.p)}  W = {fr.kendall_w:.3f}")

    print("\n=== Pairwise Wilcoxon + Holm ===")
    pw = wilcoxon_pairwise(data)
    for r in pw:
        print(f"  {r.condition_a} vs {r.condition_b}: p_raw={r.p_raw:.3f}  "
              f"p_corr={r.p_corrected:.3f}  r={r.r:+.3f}  reject={r.reject}")

    print("\n=== RMSE / NMAE ===")
    obs = np.array([83.8, 71.8, 35.9])
    pred = np.array([62.8, 64.7, 47.5])
    print(f"  RMSE = {rmse(pred, obs):.2f}")
    for p_val, o_val in zip(pred, obs):
        print(f"  NMAE({p_val:.1f}, {o_val:.1f}) = {nmae(p_val, o_val):+.2f}")

    print("\n=== HitlsDistribution ===")
    df_test = pd.DataFrame({
        "participant": [f"P{i:02d}" for i in range(2, 20)],
        "condition":   ["TARS"] * 18,
        "score":       data["TARS"],
    })
    dists = build_hitls_distributions(df_test, "condition", "score")
    d = dists[0]
    print(f"  {d.group}: mean={d.mean:.2f}  sd={d.sd:.2f}  "
          f"min={d.min_val:.2f} ({d.pid_min})  max={d.max_val:.2f} ({d.pid_max})")

    print("\n=== Envelope plot (saved to /tmp/envelope_test.png) ===")
    fig, ax = plt.subplots(figsize=(6, 4))
    tasks = ["Task A", "Task B", "Task C", "Task D"]
    fake_dists = [
        HitlsDistribution("Task A", 83.8, 40.7, 38.6, 223.1, "P07", "P12", 17),
        HitlsDistribution("Task B", 71.8, 19.9, 53.2, 110.1, "P03", "P15", 17),
        HitlsDistribution("Task C", 35.9, 18.2, 13.4,  64.3, "P06", "P10", 17),
        HitlsDistribution("Task D", float("nan"), float("nan"),
                          float("nan"), float("nan"), "?", "?", 0),
    ]
    model_vals = [62.8, 64.7, 47.5, None]
    model_reps_data = [
        rng.normal(62.8, 3, 12).tolist(),
        rng.normal(64.7, 5, 12).tolist(),
        rng.normal(47.5, 4, 12).tolist(),
        None,
    ]
    plot_participant_envelope(ax, fake_dists, model_vals,
                              model_reps=model_reps_data,
                              y_label="Duration (s)")
    ax.set_title("Tier 2 diagnostic — participant envelope (smoke-test)")
    fig.tight_layout()
    fig.savefig("/tmp/envelope_test.png", dpi=100)
    print("  Saved.")
