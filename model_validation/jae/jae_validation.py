"""
jae_validation.py — Joint Activity Efficiency (JAE) model validation.

Compares MITLS model JAE (C1 / C2 / C3) against the HITLS scenario-duration
proxy.  Produces Tier 1 (publication) and Tier 2 (diagnostic) outputs.

Run:
    python model_validation/jae/jae_validation.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
WORKSPACE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WORKSPACE / "model_validation"))

from shared.mitls_loader import (
    MITLS_OUTPUT_DIR,
    MITLS_TO_HITLS,
    CONDITION_RUN_NUMBER,
    iter_condition_runs,
    load_comparison_summary,
    load_convergence,
    load_task_summary,
)
from shared.hitls_loader import load_timing_report, SHARED_CONDITIONS
from shared.stats import (
    friedman_test,
    wilcoxon_pairwise,
    fmt_p,
    fmt_stat_line,
    sig_stars,
)

# ---------------------------------------------------------------------------
# Output directories
# ---------------------------------------------------------------------------
PLOTS_PUB   = WORKSPACE / "model_validation" / "plots" / "pub"
PLOTS_DEBUG = WORKSPACE / "model_validation" / "plots" / "debug"
REPORT_DIR  = WORKSPACE / "model_validation" / "jae"
for d in [PLOTS_PUB, PLOTS_DEBUG, REPORT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Condition order / colors
CONDITIONS   = ["C1", "C2", "C3"]
HITLS_LABELS = [MITLS_TO_HITLS[c] for c in CONDITIONS]
COND_COLORS  = {"C1": "#4878CF", "C2": "#6ACC65", "C3": "#D65F5F"}
N_REPS       = 12   # reps used in official summary (first 12 chronological)

# ---------------------------------------------------------------------------
# JAE computation (replicates compare.py logic)
# ---------------------------------------------------------------------------

def _load_task_summaries_n(condition: str, n: int = N_REPS) -> list[pd.DataFrame]:
    """Load the first *n* task_summary CSVs for *condition*."""
    runs = list(iter_condition_runs(condition))[:n]
    frames = []
    for run_dir in runs:
        try:
            frames.append(load_task_summary(run_dir))
        except FileNotFoundError:
            pass
    return frames


def _compute_ed_baseline(
    all_frames: dict[str, list[pd.DataFrame]],
) -> dict[str, float]:
    """ED (Expected Duration) = minimum Active_Duration_s per task across
    all repetitions AND all conditions.
    """
    task_ads: dict[str, list[float]] = {}
    for frames in all_frames.values():
        for df in frames:
            for _, row in df.iterrows():
                key = row["Task_Object"]
                ad = float(row["Active_Duration_s"])
                task_ads.setdefault(key, []).append(ad)
    return {task: min(ads) for task, ads in task_ads.items()}


def _jae_for_rep(df: pd.DataFrame, ed_baseline: dict[str, float]) -> float:
    """Compute scenario-level JAE for one repetition DataFrame.
    JAE_task = ED / AD;  JAE_scenario = mean across tasks.
    """
    jae_tasks = []
    for _, row in df.iterrows():
        key = row["Task_Object"]
        ad = float(row["Active_Duration_s"])
        if key in ed_baseline and ad > 0:
            jae_tasks.append(ed_baseline[key] / ad)
    return float(np.mean(jae_tasks)) if jae_tasks else float("nan")


def compute_per_rep_jae(n: int = N_REPS) -> dict[str, list[float]]:
    """Return {condition: [jae_rep_0, …, jae_rep_n-1]} for all conditions."""
    all_frames = {c: _load_task_summaries_n(c, n) for c in CONDITIONS}
    ed_baseline = _compute_ed_baseline(all_frames)
    return {
        cond: [_jae_for_rep(df, ed_baseline) for df in frames]
        for cond, frames in all_frames.items()
    }


# ---------------------------------------------------------------------------
# HITLS proxy (scenario duration — shorter = more efficient → invert)
# ---------------------------------------------------------------------------

def _hitls_duration_proxy() -> dict[str, dict]:
    """Return scenario duration stats {hitls_condition: {mean, sd, min, max}}
    for the three conditions shared with MITLS.
    """
    report = load_timing_report()
    top = report["top_level"]["scenario_duration_s"]
    return {cond: top[cond] for cond in SHARED_CONDITIONS if cond in top}


# ---------------------------------------------------------------------------
# Tier 1 — publication figure
# ---------------------------------------------------------------------------

def _plot_pub_jae(summary: pd.DataFrame, save_path: Path) -> None:
    """Bar chart: model JAE mean ± CI per condition, with HITLS proxy."""
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))

    # --- Left panel: model JAE ---
    ax = axes[0]
    means = summary["JAE_Data_Mean"].values
    cis   = summary["JAE_Data_CI"].values
    colors = [COND_COLORS[c] for c in summary["Run_Name"]]
    x = np.arange(len(summary))
    bars = ax.bar(x, means, yerr=cis, capsize=5, color=colors,
                  edgecolor="white", linewidth=0.5, error_kw={"elinewidth": 1.5})
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{r}\n({MITLS_TO_HITLS[r]})" for r in summary["Run_Name"]],
        fontsize=9,
    )
    ax.set_ylabel("JAE  (ED / AD, 0–1)", fontsize=9)
    ax.set_title("Model JAE per condition\n(mean ± 95% CI, 12 reps)", fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.spines[["top", "right"]].set_visible(False)

    # value labels
    for bar, m, ci in zip(bars, means, cis):
        ax.text(bar.get_x() + bar.get_width() / 2, m + ci + 0.02,
                f"{m:.3f}", ha="center", va="bottom", fontsize=8)

    # --- Right panel: HITLS scenario duration proxy ---
    ax2 = axes[1]
    proxy = _hitls_duration_proxy()
    cond_order = ["TARS", "TARP-F", "TARP-S"]   # C1, C2, C3 order
    p_means = [proxy[c]["mean"] for c in cond_order if c in proxy]
    p_sds   = [proxy[c]["sd"]   for c in cond_order if c in proxy]
    p_colors = [COND_COLORS[c] for c in CONDITIONS]
    x2 = np.arange(len(p_means))
    ax2.bar(x2, p_means, yerr=p_sds, capsize=5, color=p_colors,
            edgecolor="white", linewidth=0.5, error_kw={"elinewidth": 1.5})
    ax2.set_xticks(x2)
    ax2.set_xticklabels(
        [f"{m}\n({c})" for m, c in zip(cond_order, CONDITIONS)],
        fontsize=9,
    )
    ax2.set_ylabel("Scenario duration (s)", fontsize=9)
    ax2.set_title("HITLS scenario duration proxy\n(mean ± SD, inverted efficiency)",
                  fontsize=9)
    ax2.spines[["top", "right"]].set_visible(False)
    note = "Shorter duration ≈ higher efficiency\n(inverse proxy for JAE)"
    ax2.text(0.98, 0.97, note, transform=ax2.transAxes, fontsize=7,
             ha="right", va="top", color="#555555",
             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none"))

    fig.suptitle("JAE Validation — Model vs HITLS efficiency proxy", fontsize=10,
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [pub] Saved: {save_path.name}")


# ---------------------------------------------------------------------------
# Tier 2 — diagnostic figures
# ---------------------------------------------------------------------------

def _plot_debug_convergence(save_dir: Path) -> None:
    """One figure per condition: JAE mean ± CI as n_replications grows."""
    for cond, run_num in CONDITION_RUN_NUMBER.items():
        try:
            conv = load_convergence(run_num)
        except FileNotFoundError:
            print(f"  [debug] convergence file not found for run {run_num}")
            continue

        fig, ax = plt.subplots(figsize=(6, 3.5))
        n_reps = conv["n_replications"].values
        means  = conv["jae_data_mean"].values
        cis    = conv["jae_data_ci"].values

        ax.fill_between(n_reps, means - cis, means + cis,
                        alpha=0.2, color=COND_COLORS[cond], label="95% CI")
        ax.plot(n_reps, means, color=COND_COLORS[cond], lw=2, label="Mean JAE")
        ax.axvline(N_REPS, color="grey", ls="--", lw=0.8, label=f"n={N_REPS} (used)")
        ax.set_xlabel("Number of repetitions", fontsize=9)
        ax.set_ylabel("JAE (scenario mean)", fontsize=9)
        ax.set_title(
            f"JAE convergence — {cond} ({MITLS_TO_HITLS[cond]})", fontsize=9
        )
        ax.legend(fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
        # Convergence flag: relative precision of last row
        rel_prec_col = "jae_data_rel_precision_pct"
        if rel_prec_col in conv.columns:
            last_prec = conv[rel_prec_col].iloc[-1]
            flag = "⚠ not converged" if last_prec > 5 else "✓ converged"
            ax.text(0.02, 0.97, f"Rel. precision at n={n_reps[-1]}: "
                    f"{last_prec:.2f}%  {flag}",
                    transform=ax.transAxes, fontsize=7, va="top",
                    color="darkred" if last_prec > 5 else "darkgreen")
        fig.tight_layout()
        path = save_dir / f"jae_convergence_{cond.lower()}.png"
        fig.savefig(path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"  [debug] Saved: {path.name}")


def _plot_debug_coordination(
    summary: pd.DataFrame,
    per_rep_jae: dict[str, list[float]],
    save_path: Path,
) -> None:
    """Stacked bar: Active % / Coordination % / Idle % per condition,
    with individual 12-rep JAE scatter.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # --- Left: stacked bar ---
    ax = axes[0]
    conds_sorted = summary["Run_Name"].tolist()
    active = summary["Active_Percentage"].values
    coord  = summary["Coordination_Percentage"].values
    idle   = 100 - active - coord
    idle   = np.clip(idle, 0, None)
    x = np.arange(len(conds_sorted))
    ax.bar(x, active, color="#4878CF", label="Active %")
    ax.bar(x, coord,  bottom=active, color="#D65F5F", label="Coordination %")
    ax.bar(x, idle,   bottom=active + coord, color="#BBBBBB", label="Idle %")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{c}\n({MITLS_TO_HITLS[c]})" for c in conds_sorted], fontsize=9
    )
    ax.set_ylabel("Percentage of total FSM time", fontsize=9)
    ax.set_title("Time breakdown per condition\n(model means)", fontsize=9)
    ax.legend(fontsize=8, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)

    # --- Right: per-rep JAE strip ---
    ax2 = axes[1]
    rng = np.random.default_rng(42)
    for i, cond in enumerate(conds_sorted):
        reps = per_rep_jae.get(cond, [])
        if not reps:
            continue
        jitter = rng.uniform(-0.15, 0.15, size=len(reps))
        ax2.scatter(
            np.full(len(reps), i) + jitter, reps,
            color=COND_COLORS[cond], s=30, alpha=0.7, zorder=3,
        )
        m = np.nanmean(reps)
        ax2.hlines(m, i - 0.25, i + 0.25, colors=COND_COLORS[cond],
                   linewidths=2, zorder=4, label=f"{cond} mean={m:.3f}")
    ax2.set_xticks(np.arange(len(conds_sorted)))
    ax2.set_xticklabels(
        [f"{c}\n({MITLS_TO_HITLS[c]})" for c in conds_sorted], fontsize=9
    )
    ax2.set_ylabel("JAE per repetition", fontsize=9)
    ax2.set_title("Per-repetition JAE\n(12 reps per condition)", fontsize=9)
    ax2.legend(fontsize=7)
    ax2.spines[["top", "right"]].set_visible(False)

    fig.suptitle("JAE Diagnostic — Coordination breakdown & repetition scatter",
                 fontsize=10, fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  [debug] Saved: {save_path.name}")


# ---------------------------------------------------------------------------
# Report writers
# ---------------------------------------------------------------------------

def _direction_match(
    per_rep_jae: dict[str, list[float]],
    proxy: dict[str, dict],
) -> str:
    """Check if model JAE ranking matches HITLS inverse-duration ranking."""
    model_rank = sorted(CONDITIONS, key=lambda c: np.nanmean(per_rep_jae[c]),
                        reverse=True)
    # HITLS: shorter duration = more efficient → sort ascending
    hitls_rank_hitls = sorted(
        SHARED_CONDITIONS,
        key=lambda c: proxy[c]["mean"] if c in proxy else float("inf"),
    )
    # Map HITLS labels back to C1/C2/C3
    from shared.hitls_loader import HITLS_TO_MITLS as _H2M
    hitls_rank_mitls = [_H2M[c] for c in hitls_rank_hitls if _H2M.get(c)]

    model_str  = " > ".join(model_rank)
    hitls_str  = " > ".join(hitls_rank_mitls)
    match = model_rank == hitls_rank_mitls
    return (
        f"  Model ranking (high→low JAE): {model_str}\n"
        f"  HITLS ranking (short→long duration, =high efficiency): {hitls_str}\n"
        f"  Direction match: {'YES ✓' if match else 'NO ✗'}"
    )


def _write_tier1_report(
    fr,
    pw_results,
    per_rep_jae: dict[str, list[float]],
    summary: pd.DataFrame,
    proxy: dict[str, dict],
    save_path: Path,
) -> None:
    lines = [
        "=" * 72,
        "JAE VALIDATION REPORT — Tier 1 (Publication)",
        "=" * 72,
        "",
        "Note: JAE = mean(ED/AD) per scenario, where ED (Expected Duration) is",
        "the minimum Active_Duration_s observed per task across ALL reps and",
        "conditions.  JAE = 1.0 means perfect efficiency (equal to best run).",
        "",
        "--- Model condition summary ---",
    ]
    for _, row in summary.iterrows():
        lines.append(
            f"  {row['Run_Name']} ({MITLS_TO_HITLS[row['Run_Name']]:<6})  "
            f"JAE = {row['JAE_Data_Mean']:.4f} ± {row['JAE_Data_CI']:.4f} (CI)  "
            f"n = {int(row['N_Repetitions'])}"
        )
    lines += [
        "",
        "--- Inferential statistics (model repetitions as observations) ---",
        fmt_stat_line("JAE across conditions", fr, pw_results),
        "",
        "--- HITLS scenario duration proxy ---",
    ]
    for cond in SHARED_CONDITIONS:
        if cond not in proxy:
            continue
        p = proxy[cond]
        lines.append(
            f"  {cond:<7} duration: {p['mean']:.1f} ± {p['sd']:.1f} s  "
            f"[{p['min']:.1f} – {p['max']:.1f}]  n = {p['n']}"
        )
    lines += [
        "",
        "--- Direction match ---",
        _direction_match(per_rep_jae, proxy),
        "",
        "=" * 72,
    ]
    text = "\n".join(lines)
    save_path.write_text(text, encoding="utf-8")
    print(f"  [pub] Saved: {save_path.name}")


def _write_tier2_report(
    per_rep_jae: dict[str, list[float]],
    summary: pd.DataFrame,
    convergence: dict[int, pd.DataFrame],
    save_path: Path,
) -> None:
    lines = [
        "=" * 72,
        "JAE VALIDATION REPORT — Tier 2 (Diagnostic)",
        "=" * 72,
        "",
        "--- Per-repetition JAE values ---",
    ]
    for cond in CONDITIONS:
        reps = per_rep_jae.get(cond, [])
        lines.append(f"  {cond} ({MITLS_TO_HITLS[cond]}):")
        for i, v in enumerate(reps):
            lines.append(f"    rep {i+1:2d}: JAE = {v:.6f}")
        if reps:
            lines.append(f"    → mean = {np.nanmean(reps):.6f}  "
                         f"std = {np.nanstd(reps):.6f}")
        lines.append("")

    lines.append("--- Coordination overhead ---")
    for _, row in summary.iterrows():
        coord_pct = row["Coordination_Percentage"]
        active_pct = row["Active_Percentage"]
        idle_pct = max(0, 100 - active_pct - coord_pct)
        lines.append(
            f"  {row['Run_Name']} ({MITLS_TO_HITLS[row['Run_Name']]:<6})  "
            f"Active={active_pct:.1f}%  "
            f"Coordination={coord_pct:.1f}%  "
            f"Idle={idle_pct:.1f}%"
        )

    lines += ["", "--- Convergence flags ---"]
    for cond, run_num in CONDITION_RUN_NUMBER.items():
        conv = convergence.get(run_num)
        if conv is None:
            continue
        col = "jae_data_rel_precision_pct"
        if col in conv.columns:
            prec = conv[col].iloc[-1]
            flag = "⚠ NOT CONVERGED (>5%)" if prec > 5 else "✓ converged"
            lines.append(f"  {cond} run_{run_num}: rel_precision = {prec:.2f}%  {flag}")

    lines += ["", "=" * 72]
    save_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  [debug] Saved: {save_path.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_jae_validation() -> dict:
    """Run full JAE validation.  Returns summary metrics dict."""
    print("\n" + "=" * 60)
    print("JAE VALIDATION")
    print("=" * 60)

    # 1. Load data
    print("\n[1/5] Loading MITLS data...")
    summary = load_comparison_summary()
    per_rep_jae = compute_per_rep_jae(N_REPS)
    for cond, reps in per_rep_jae.items():
        print(f"  {cond}: {len(reps)} reps loaded, "
              f"mean JAE = {np.nanmean(reps):.4f}")

    print("\n[2/5] Running statistics...")
    groups = {c: per_rep_jae[c] for c in CONDITIONS}
    fr = friedman_test({MITLS_TO_HITLS[c]: groups[c] for c in CONDITIONS})
    pw = wilcoxon_pairwise({MITLS_TO_HITLS[c]: groups[c] for c in CONDITIONS})
    print(f"  Friedman: χ²({fr.df}, N={fr.n}) = {fr.chi2:.3f}  {fmt_p(fr.p)}  W={fr.kendall_w:.3f}")
    for r in pw:
        s = sig_stars(r.p_raw, r.reject)
        print(f"  {r.condition_a} vs {r.condition_b}: {fmt_p(r.p_raw)}  "
              f"r={r.r:+.3f}  {s or 'ns'}")

    proxy = _hitls_duration_proxy()

    # 3. Load convergence
    convergence = {}
    for run_num in CONDITION_RUN_NUMBER.values():
        try:
            convergence[run_num] = load_convergence(run_num)
        except FileNotFoundError:
            pass

    # 4. Tier 1 outputs
    print("\n[3/5] Writing Tier 1 outputs (publication)...")
    _plot_pub_jae(summary, PLOTS_PUB / "jae_condition_comparison.png")
    _write_tier1_report(fr, pw, per_rep_jae, summary, proxy,
                        REPORT_DIR / "jae_report.txt")

    # 5. Tier 2 outputs
    print("\n[4/5] Writing Tier 2 outputs (diagnostic)...")
    _plot_debug_convergence(PLOTS_DEBUG)
    _plot_debug_coordination(
        summary, per_rep_jae, PLOTS_DEBUG / "jae_coordination_breakdown.png"
    )
    _write_tier2_report(per_rep_jae, summary, convergence,
                        REPORT_DIR / "jae_debug_report.txt")

    print("\n[5/5] Done.")
    return {
        "jae_friedman_chi2": fr.chi2,
        "jae_friedman_p": fr.p,
        "jae_kendall_w": fr.kendall_w,
        "jae_direction_match": (
            sorted(CONDITIONS, key=lambda c: np.nanmean(per_rep_jae[c]), reverse=True)
            == ["C2", "C1", "C3"]
        ),
    }


if __name__ == "__main__":
    metrics = run_jae_validation()
    print("\nSummary metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
