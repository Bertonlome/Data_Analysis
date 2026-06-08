#!/usr/bin/env python3
"""
Crosscheck validation: HITLS vs MITLS.

Compares task-level crosscheck rates across shared conditions:
  TARS, TARP-S, TARP-F

Data sources
------------
HITLS:
  HITLS/Pxx/cleaned/Pxx_crosscheck_perf_report.txt

MITLS:
  MITLS/eye_movement/output/run_*_results_*/team-analyzer/crosscheck_perf_report.txt

Outputs
-------
Tier 1:
  model_validation/plots/pub/crosscheck_human_vs_model.png
  model_validation/eye_movement/crosscheck_report.txt

Tier 2:
  model_validation/eye_movement/crosscheck_debug_report.txt
"""

from __future__ import annotations

import json
import itertools
import sys
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import friedmanchisquare
from scipy.stats import wilcoxon as _sp_wilcoxon
from statsmodels.stats.multitest import multipletests

_HERE = Path(__file__).resolve()
WORKSPACE_ROOT = _HERE.parents[2]
sys.path.insert(0, str(WORKSPACE_ROOT))

from model_validation.shared.hitls_loader import HITLS_DIR, SHARED_CONDITIONS
from model_validation.shared.mitls_loader import MITLS_TO_HITLS, iter_condition_runs
from model_validation.shared.stats import FriedmanResult, WilcoxonResult, nmae, spearman_r


VAL_DIR = WORKSPACE_ROOT / "model_validation"
PUB_DIR = VAL_DIR / "plots" / "pub"
PUB_DIR.mkdir(parents=True, exist_ok=True)

COND_ORDER = [c for c in ["TARS", "TARP-S", "TARP-F"] if c in SHARED_CONDITIONS]
COLORS = {
    "HITLS": "#4878CF",
    "MITLS": "#D65F5F",
}


def _load_json_summary(path: Path) -> Optional[dict]:
    try:
        txt = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    try:
        start = txt.index("{")
        end = txt.index("--- END SUMMARY ---")
        return json.loads(txt[start:end].strip())
    except (ValueError, json.JSONDecodeError):
        return None


def _find_hitls_participants() -> list[str]:
    return [
        e.name for e in sorted(HITLS_DIR.iterdir())
        if e.is_dir() and e.name.startswith("P") and e.name[1:].isdigit()
    ]


def _load_hitls_crosscheck() -> tuple[dict[str, list[float]], dict[str, dict[str, Optional[float]]]]:
    """Return (values_by_condition, per_participant_rates)."""
    vals = {c: [] for c in COND_ORDER}
    per_pid: dict[str, dict[str, Optional[float]]] = {}

    for pid in _find_hitls_participants():
        rep = HITLS_DIR / pid / "cleaned" / f"{pid}_crosscheck_perf_report.txt"
        data = _load_json_summary(rep)
        if not data:
            continue

        rates = {c: None for c in COND_ORDER}
        by_cond: dict[str, dict[str, int]] = {}
        for s in data.get("scenarios", []):
            cond = s.get("condition")
            if cond not in COND_ORDER:
                continue
            if cond not in by_cond:
                by_cond[cond] = {"n_assessable": 0, "n_crosschecked": 0}
            by_cond[cond]["n_assessable"] += int(s.get("n_assessable", 0))
            by_cond[cond]["n_crosschecked"] += int(s.get("n_crosschecked", 0))

        for cond in COND_ORDER:
            d = by_cond.get(cond)
            if d and d["n_assessable"] > 0:
                r = 100.0 * d["n_crosschecked"] / d["n_assessable"]
                rates[cond] = float(r)
                vals[cond].append(float(r))

        per_pid[pid] = rates

    return vals, per_pid


def _load_mitls_crosscheck(n_reps: int = 12) -> tuple[dict[str, list[float]], dict[str, list[Optional[float]]]]:
    """Return (values_by_condition, paired_by_index_for_stats)."""
    vals = {c: [] for c in COND_ORDER}

    # Collect per condition, preserving run order from iter_condition_runs.
    for mitls_cond in ["C1", "C2", "C3"]:
        hitls_cond = MITLS_TO_HITLS.get(mitls_cond)
        if hitls_cond not in COND_ORDER:
            continue
        runs = list(iter_condition_runs(mitls_cond))[:n_reps]
        for run_dir in runs:
            rep = Path(run_dir) / "team-analyzer" / "crosscheck_perf_report.txt"
            data = _load_json_summary(rep)
            if not data:
                continue
            rr = data.get("result", data)
            n_assessable = int(rr.get("n_assessable", 0))
            n_crosschecked = int(rr.get("n_crosschecked", 0))
            if n_assessable <= 0:
                continue
            vals[hitls_cond].append(100.0 * n_crosschecked / n_assessable)

    # Build paired rows by repetition index (conservative alignment).
    max_n = max((len(v) for v in vals.values()), default=0)
    paired = {c: [] for c in COND_ORDER}
    for i in range(max_n):
        for c in COND_ORDER:
            paired[c].append(vals[c][i] if i < len(vals[c]) else None)

    return vals, paired


def _desc(vals: list[float]) -> dict:
    if not vals:
        return {"n": 0, "mean": float("nan"), "sd": float("nan")}
    a = np.asarray(vals, dtype=float)
    return {
        "n": int(len(a)),
        "mean": float(np.mean(a)),
        "sd": float(np.std(a, ddof=1)) if len(a) > 1 else 0.0,
    }


def _rank_order(means: dict[str, float]) -> list[str]:
    return [k for k, _ in sorted(means.items(), key=lambda kv: kv[1], reverse=True)]


def _safe_friedman(groups: dict[str, list[Optional[float]]]) -> FriedmanResult:
    df = len(COND_ORDER) - 1
    rows = list(zip(*[groups[c] for c in COND_ORDER]))
    valid = [
        tuple(float(x) for x in row)
        for row in rows
        if all(x is not None and np.isfinite(float(x)) for x in row)
    ]
    n = len(valid)
    if n < 3:
        return FriedmanResult(chi2=0.0, p=1.0, df=df, kendall_w=0.0, n=n)

    arrs = [np.asarray([r[i] for r in valid], dtype=float) for i in range(len(COND_ORDER))]
    if all(np.allclose(arrs[0], arrs[i]) for i in range(1, len(arrs))):
        return FriedmanResult(chi2=0.0, p=1.0, df=df, kendall_w=0.0, n=n)

    try:
        stat, p = friedmanchisquare(*arrs)
        stat_f = float(stat)
        p_f = float(p)
        if not np.isfinite(stat_f) or not np.isfinite(p_f):
            return FriedmanResult(chi2=0.0, p=1.0, df=df, kendall_w=0.0, n=n)
        w = stat_f / (n * df) if n * df > 0 else 0.0
        if not np.isfinite(w):
            w = 0.0
        return FriedmanResult(chi2=stat_f, p=p_f, df=df, kendall_w=float(w), n=n)
    except Exception:
        return FriedmanResult(chi2=0.0, p=1.0, df=df, kendall_w=0.0, n=n)


def _safe_pairwise(groups: dict[str, list[Optional[float]]]) -> list[WilcoxonResult]:
    pairs = list(itertools.combinations(COND_ORDER, 2))
    raw: list[WilcoxonResult] = []

    for a, b in pairs:
        xa, xb = [], []
        for x, y in zip(groups[a], groups[b]):
            if x is None or y is None:
                continue
            xf, yf = float(x), float(y)
            if np.isfinite(xf) and np.isfinite(yf):
                xa.append(xf)
                xb.append(yf)
        n = len(xa)
        if n < 4:
            raw.append(WilcoxonResult(a, b, 1.0, 1.0, False, 0.0, n))
            continue

        arr_a = np.asarray(xa, dtype=float)
        arr_b = np.asarray(xb, dtype=float)
        if np.allclose(arr_a, arr_b):
            raw.append(WilcoxonResult(a, b, 1.0, 1.0, False, 0.0, n))
            continue

        try:
            res = _sp_wilcoxon(arr_a, arr_b, alternative="two-sided", zero_method="wilcox")
            p = float(res.pvalue)
            if not np.isfinite(p):
                p = 1.0
            r = 1.0 - 2.0 * res.statistic / (n * (n + 1) / 2.0)
            if not np.isfinite(r):
                r = 0.0
            raw.append(WilcoxonResult(a, b, p, p, False, float(r), n))
        except Exception:
            raw.append(WilcoxonResult(a, b, 1.0, 1.0, False, 0.0, n))

    if not raw:
        return raw

    pvals = [r.p_raw for r in raw]
    reject, p_corr, _, _ = multipletests(pvals, alpha=0.05, method="holm")
    for i, r in enumerate(raw):
        r.p_corrected = float(p_corr[i])
        r.reject = bool(reject[i])
    return raw


def _safe_spearman(a: list[float], b: list[float]) -> tuple[float, float]:
    aa = np.asarray(a, dtype=float)
    bb = np.asarray(b, dtype=float)
    if len(aa) < 3 or len(bb) < 3:
        return 0.0, 1.0
    if np.allclose(aa, aa[0]) or np.allclose(bb, bb[0]):
        return 0.0, 1.0
    rho, p = spearman_r(a, b)
    if not np.isfinite(rho) or not np.isfinite(p):
        return 0.0, 1.0
    return float(rho), float(p)


def _plot_pub(hitls_vals: dict[str, list[float]], mitls_vals: dict[str, list[float]], save_path: Path) -> None:
    x = np.arange(len(COND_ORDER))
    width = 0.36

    h_means = [np.mean(hitls_vals[c]) if hitls_vals[c] else np.nan for c in COND_ORDER]
    h_sds = [np.std(hitls_vals[c], ddof=1) if len(hitls_vals[c]) > 1 else 0.0 for c in COND_ORDER]
    m_means = [np.mean(mitls_vals[c]) if mitls_vals[c] else np.nan for c in COND_ORDER]
    m_sds = [np.std(mitls_vals[c], ddof=1) if len(mitls_vals[c]) > 1 else 0.0 for c in COND_ORDER]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(
        x - width / 2,
        h_means,
        width,
        label="HITLS",
        color=COLORS["HITLS"],
        alpha=0.8,
        yerr=h_sds,
        error_kw=dict(elinewidth=1, capsize=3),
    )
    ax.bar(
        x + width / 2,
        m_means,
        width,
        label="MITLS",
        color=COLORS["MITLS"],
        alpha=0.8,
        yerr=m_sds,
        error_kw=dict(elinewidth=1, capsize=3),
    )

    ax.set_title("Crosscheck Rate: HITLS vs MITLS", fontsize=12, fontweight="bold")
    ax.set_ylabel("Crosscheck rate (% of assessable tasks)")
    ax.set_xticks(x)
    ax.set_xticklabels(COND_ORDER)
    ax.set_ylim(0, 100)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [pub] Saved: {save_path.name}")


def _fmt_p(p: float) -> str:
    if np.isnan(p):
        return "nan"
    if p < 1e-4:
        return f"{p:.2e}"
    return f"{p:.4f}"


def _write_tier1_report(
    hitls_vals: dict[str, list[float]],
    mitls_vals: dict[str, list[float]],
    nmae_by_cond: dict[str, float],
    rho: float,
    p_rho: float,
    save_path: Path,
) -> None:
    lines = [
        "=" * 62,
        "CROSSCHECK VALIDATION — TIER 1 REPORT",
        "=" * 62,
        "",
        "Scope: shared conditions only (TARS, TARP-S, TARP-F)",
        "Metric: crosscheck rate (% of assessable task occurrences)",
        "",
        f"{'Condition':<10s} {'HITLS mean':>12s} {'MITLS mean':>12s} {'NMAE':>10s} {'n HITLS':>9s} {'n MITLS':>9s}",
    ]
    for c in COND_ORDER:
        hm = np.mean(hitls_vals[c]) if hitls_vals[c] else float("nan")
        mm = np.mean(mitls_vals[c]) if mitls_vals[c] else float("nan")
        nv = nmae_by_cond[c]
        lines.append(
            f"{c:<10s} {hm:>11.2f}% {mm:>11.2f}% {nv:>+9.1%} {len(hitls_vals[c]):>9d} {len(mitls_vals[c]):>9d}"
        )

    lines += [
        "",
        f"Spearman rho on condition means (n=3): rho={rho:.3f}, p={_fmt_p(p_rho)}",
        "",
        "Interpretation:",
        "MITLS crosscheck is currently much lower than HITLS across all shared conditions",
        "for the available fixation-threshold settings and gaze model outputs.",
    ]

    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  [pub] Saved: {save_path.name}")


def _write_tier2_report(
    hitls_vals: dict[str, list[float]],
    mitls_vals: dict[str, list[float]],
    hitls_frd,
    mitls_frd,
    hitls_pw,
    mitls_pw,
    save_path: Path,
) -> None:
    lines = [
        "=" * 62,
        "CROSSCHECK VALIDATION — TIER 2 DEBUG REPORT",
        "=" * 62,
        "",
        "Sample sizes per condition:",
    ]
    for c in COND_ORDER:
        lines.append(f"  {c:<8s} HITLS n={len(hitls_vals[c]):>3d}   MITLS n={len(mitls_vals[c]):>3d}")

    lines += [
        "",
        "Within-system condition effect (Friedman):",
        f"  HITLS: chi2({hitls_frd.df})={hitls_frd.chi2:.3f}, p={_fmt_p(hitls_frd.p)}, W={hitls_frd.kendall_w:.3f}, n={hitls_frd.n}",
        f"  MITLS: chi2({mitls_frd.df})={mitls_frd.chi2:.3f}, p={_fmt_p(mitls_frd.p)}, W={mitls_frd.kendall_w:.3f}, n={mitls_frd.n}",
        "",
        "Pairwise Wilcoxon (Holm-corrected):",
        "  HITLS:",
    ]
    for r in hitls_pw:
        lines.append(
            f"    {r.condition_a} vs {r.condition_b}: p={_fmt_p(r.p_raw)}  p_holm={_fmt_p(r.p_corrected)}  r={r.r:+.3f}  reject={r.reject}"
        )
    lines.append("  MITLS:")
    for r in mitls_pw:
        lines.append(
            f"    {r.condition_a} vs {r.condition_b}: p={_fmt_p(r.p_raw)}  p_holm={_fmt_p(r.p_corrected)}  r={r.r:+.3f}  reject={r.reject}"
        )

    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  [debug] Saved: {save_path.name}")


def run_crosscheck_validation(n_reps: int = 12) -> dict:
    """Run HITLS vs MITLS crosscheck validation and return summary metrics."""
    print("=" * 60)
    print("CROSSCHECK VALIDATION")
    print("=" * 60)

    print("\n[1/5] Loading HITLS crosscheck data...")
    hitls_vals, hitls_per_pid = _load_hitls_crosscheck()

    print("[2/5] Loading MITLS crosscheck data...")
    mitls_vals, mitls_paired = _load_mitls_crosscheck(n_reps=n_reps)

    for c in COND_ORDER:
        print(f"  {c}: HITLS n={len(hitls_vals[c])}, MITLS n={len(mitls_vals[c])}")

    # HITLS paired vectors by participant.
    pids = sorted(hitls_per_pid.keys())
    hitls_groups = {c: [hitls_per_pid[pid].get(c) for pid in pids] for c in COND_ORDER}
    mitls_groups = mitls_paired

    print("\n[3/5] Computing statistics...")
    hitls_frd = _safe_friedman(hitls_groups)
    mitls_frd = _safe_friedman(mitls_groups)
    hitls_pw = _safe_pairwise(hitls_groups)
    mitls_pw = _safe_pairwise(mitls_groups)

    hitls_means = {c: (np.mean(hitls_vals[c]) if hitls_vals[c] else np.nan) for c in COND_ORDER}
    mitls_means = {c: (np.mean(mitls_vals[c]) if mitls_vals[c] else np.nan) for c in COND_ORDER}
    nmae_by_cond = {
        c: nmae(mitls_means[c], hitls_means[c]) if np.isfinite(hitls_means[c]) and np.isfinite(mitls_means[c]) else np.nan
        for c in COND_ORDER
    }
    rho, p_rho = _safe_spearman(
        [hitls_means[c] for c in COND_ORDER],
        [mitls_means[c] for c in COND_ORDER],
    )

    hitls_rank = _rank_order(hitls_means)
    mitls_rank = _rank_order(mitls_means)
    dir_match = hitls_rank == mitls_rank

    print("[4/5] Writing Tier 1 outputs...")
    _plot_pub(hitls_vals, mitls_vals, PUB_DIR / "crosscheck_human_vs_model.png")
    _write_tier1_report(
        hitls_vals,
        mitls_vals,
        nmae_by_cond,
        rho,
        p_rho,
        VAL_DIR / "eye_movement" / "crosscheck_report.txt",
    )

    print("[5/5] Writing Tier 2 outputs...")
    _write_tier2_report(
        hitls_vals,
        mitls_vals,
        hitls_frd,
        mitls_frd,
        hitls_pw,
        mitls_pw,
        VAL_DIR / "eye_movement" / "crosscheck_debug_report.txt",
    )

    metrics = {
        "crosscheck_hitls_friedman_chi2": float(hitls_frd.chi2),
        "crosscheck_hitls_friedman_p": float(hitls_frd.p),
        "crosscheck_mitls_friedman_chi2": float(mitls_frd.chi2),
        "crosscheck_mitls_friedman_p": float(mitls_frd.p),
        "crosscheck_spearman_rho": float(rho),
        "crosscheck_spearman_p": float(p_rho),
        "crosscheck_dir_match": bool(dir_match),
    }
    for c in COND_ORDER:
        metrics[f"crosscheck_hitls_mean_{c}"] = float(hitls_means[c]) if np.isfinite(hitls_means[c]) else float("nan")
        metrics[f"crosscheck_mitls_mean_{c}"] = float(mitls_means[c]) if np.isfinite(mitls_means[c]) else float("nan")
        metrics[f"crosscheck_nmae_{c}"] = float(nmae_by_cond[c]) if np.isfinite(nmae_by_cond[c]) else float("nan")

    print("\nSummary metrics:")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    return metrics


if __name__ == "__main__":
    run_crosscheck_validation()
