#!/usr/bin/env python3
"""
crosscheck_perf.py — MITLS crosscheck analysis (HITLS-compatible logic).

Computes task-level crosscheck success for each MITLS repetition run from:
  - team-analyzer/cassandra_converted.csv  (TARS Agent current_state stream)
  - eye_movement/results_eye_movement.txt  (fixation trace)

Crosscheck criterion:
  a task is crosschecked if at least one fixation on an expected AoI has
  dwell >= MIN_FIXATION_S during the accepted task window.

Accepted window rules (same policy as HITLS):
  - TARS (C1): [task_onset, next_task_onset]
  - TARP-S / TARP-F (C3 / C2): [task_onset, max(next_task_onset, task_onset + 10s)]

Outputs:
  - Per-run report:
      <run_dir>/team-analyzer/crosscheck_perf_report.txt
  - Cross-run aggregate report:
      MITLS/eye_movement/output/crosscheck_compare_report.txt
  - Aggregate plot:
      MITLS/eye_movement/output/crosscheck_boxplots_mitls.png
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import friedmanchisquare, wilcoxon as _sp_wilcoxon
from statsmodels.stats.multitest import multipletests

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT_DIR = HERE / "output"
TASK_TABLE_PATH = ROOT / "HITLS" / "performance" / "task_crosscheck_aoi_table.csv"

MIN_FIXATION_S: float = 0.1

CONDITION_TO_RUNNUM = {"TARS": 1, "TARP-F": 2, "TARP-S": 3}
CONDITIONS = ["TARS", "TARP-F", "TARP-S"]
PAIRWISE = [("TARS", "TARP-S"), ("TARS", "TARP-F"), ("TARP-S", "TARP-F")]

_VALID_AOI = {"TARS", "PFD", "ND", "pedestal", "Outside_Window", "no_intersection"}

_AOI_BOXES: dict[str, object] = {
    "TARS":            (0, 865, 800, 1374),
    "PFD":             (1041, 636, 1532, 963),
    "E_W_CAS":         (1592, 681, 1674, 1038),
    "ND":              (1719, 682, 2126, 1036),
    "Central_Console": (1724, 1233, 2282, 1374),
    "Outside_Window":  [(0, 0, 2555, 344), (0, 0, 924, 852)],
}

_MITLS_TO_CANON_AOI = {
    "TARS": "TARS",
    "PFD": "PFD",
    "E_W_CAS": "PFD",
    "ND": "ND",
    "Central_Console": "pedestal",
    "Outside_Window": "Outside_Window",
    "Other": "no_intersection",
}


@dataclass(frozen=True)
class TaskRule:
    task: str
    crosscheck_aoi_raw: str
    crosscheck_aois: tuple[str, ...]
    norm_task: str


@dataclass
class TaskEvent:
    t_rel: float
    procedure: str
    task_object: str
    value: str


@dataclass
class EyeFix:
    t_rel: float
    dwell_s: float
    aoi: str


def _normalize_text(s: str) -> str:
    s = (s or "").lower().strip()
    s = s.replace("t/o", "takeoff")
    s = s.replace("w/s", "windshield")
    s = s.replace("l/g", "landing gear")
    s = s.replace("v_enr", "venr")
    s = s.replace("throttles", "throttle")
    s = s.replace("checked", "check")
    s = re.sub(r"\b(\d+)\s*kts\b", r"\1kts", s)
    s = s.replace("'", " ")
    s = s.replace('"', " ")
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _parse_state_value(val_str: str) -> Optional[dict]:
    v = val_str
    if v.startswith('"') and v.endswith('"'):
        v = v[1:-1].replace('""', '"')
    try:
        return json.loads(v)
    except (json.JSONDecodeError, ValueError):
        return None


def _point_in_aoi(x: int, y: int, box: object) -> bool:
    if isinstance(box, list):
        return any(_point_in_aoi(x, y, b) for b in box)
    x0, y0, x1, y1 = box
    return x0 <= x <= x1 and y0 <= y <= y1


def _assign_aoi(x: int, y: int) -> str:
    for name, box in _AOI_BOXES.items():
        if _point_in_aoi(x, y, box):
            return _MITLS_TO_CANON_AOI.get(name, "no_intersection")
    return _MITLS_TO_CANON_AOI["Other"]


def load_task_rules(csv_path: Path) -> list[TaskRule]:
    dedup: dict[tuple[str, tuple[str, ...]], TaskRule] = {}
    with open(csv_path, encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            task = (row.get("task") or "").strip()
            if not task:
                continue
            raw = (row.get("crosscheck_aoi") or "").strip()
            aois = tuple(a.strip() for a in raw.split("|") if a.strip() in _VALID_AOI)
            norm_task = _normalize_text(task)
            dedup[(norm_task, aois)] = TaskRule(
                task=task,
                crosscheck_aoi_raw=(row.get("crosscheck_aoi_raw") or "").strip(),
                crosscheck_aois=aois,
                norm_task=norm_task,
            )
    return list(dedup.values())


def _match_score(rule: TaskRule, task_text: str) -> float:
    obj = _normalize_text(task_text)
    if not obj:
        return 0.0
    rule_tokens = [t for t in rule.norm_task.split() if len(t) >= 2]
    if not rule_tokens:
        return 0.0
    obj_tokens = set(obj.split())
    overlap = sum(1 for t in rule_tokens if t in obj_tokens)
    score = overlap / max(1, len(rule_tokens))
    if rule.norm_task and rule.norm_task in obj:
        score = max(score, 0.95)
    return min(score, 1.0)


def _find_best_rule(task_text: str, rules: list[TaskRule]) -> Optional[TaskRule]:
    best = None
    best_score = 0.0
    for rule in rules:
        s = _match_score(rule, task_text)
        if s > best_score:
            best = rule
            best_score = s
    return best if best is not None and best_score >= 0.60 else None


def _accepted_window_end(condition: str, t_start: float, t_natural_end: float) -> float:
    if condition in {"TARP-S", "TARP-F"}:
        return max(t_natural_end, t_start + 10.0)
    return t_natural_end


def _load_task_events(cassandra_csv: Path) -> tuple[list[TaskEvent], float]:
    events: list[TaskEvent] = []
    t0 = None
    t_end = 0.0
    with open(cassandra_csv, encoding="utf-8") as fh:
        reader = csv.reader(fh, delimiter=";")
        next(reader, None)
        for row in reader:
            if len(row) < 7:
                continue
            try:
                t = float(row[1])
            except ValueError:
                continue
            if t0 is None:
                t0 = t
            t_end = max(t_end, t)
            if row[2] != "TARS Agent" or row[3] != "current_state":
                continue
            d = _parse_state_value(row[6])
            if d is None:
                continue
            events.append(TaskEvent(t_rel=t, procedure=str(d.get("procedure", "")), task_object=str(d.get("task_object", "")), value=str(d.get("value", ""))))
    if t0 is None:
        return [], 0.0
    for ev in events:
        ev.t_rel -= t0
    return events, (t_end - t0)


def _load_eye_fixations(eye_txt: Path) -> list[EyeFix]:
    fixes: list[EyeFix] = []
    with open(eye_txt, encoding="utf-8") as fh:
        for line in fh:
            parts = line.strip().split("\t")
            if len(parts) < 7:
                continue
            try:
                t_rel = float(parts[0])
                x = int(parts[2])
                y = int(parts[3])
                dwell_s = float(parts[6]) / 1000.0
            except (ValueError, IndexError):
                continue
            fixes.append(EyeFix(t_rel=t_rel, dwell_s=dwell_s, aoi=_assign_aoi(x, y)))
    return fixes


def _assess_crosscheck(samples: list[EyeFix], targets: tuple[str, ...]) -> tuple[bool, list[str], float]:
    seen = sorted(set(s.aoi for s in samples if s.aoi))
    target_dwells = [s.dwell_s for s in samples if s.aoi in targets]
    max_dwell = max(target_dwells) if target_dwells else 0.0
    return max_dwell >= MIN_FIXATION_S, seen, round(max_dwell, 4)


def analyze_run(run_dir: Path, condition: str, rules: list[TaskRule]) -> dict:
    cassandra_csv = run_dir / "team-analyzer" / "cassandra_converted.csv"
    eye_txt = run_dir / "eye_movement" / "results_eye_movement.txt"

    if not cassandra_csv.exists():
        return {
            "run_dir": run_dir.name,
            "condition": condition,
            "n_eye_events": 0,
            "eye_data_available": False,
            "n_task_events": 0,
            "n_matched": 0,
            "n_assessable": 0,
            "n_crosschecked": 0,
            "task_results": [],
            "unmatched_task_objects": [],
            "notes": ["Missing team-analyzer/cassandra_converted.csv"],
        }

    task_events, stream_end = _load_task_events(cassandra_csv)
    eye_events = _load_eye_fixations(eye_txt) if eye_txt.exists() else []
    has_eye_data = len(eye_events) > 0

    results = []
    unmatched = []

    for i, ev in enumerate(task_events):
        t_start = ev.t_rel
        t_natural_end = task_events[i + 1].t_rel if i < len(task_events) - 1 else stream_end
        t_accept_end = _accepted_window_end(condition, t_start, t_natural_end)

        rule = _find_best_rule(f"{ev.task_object} {ev.value}", rules)
        if rule is None:
            unmatched.append(ev.task_object)
            continue

        in_window = [e for e in eye_events if t_start <= e.t_rel <= t_accept_end]
        if not has_eye_data:
            hit, seen, max_dwell = None, [], 0.0
        elif rule.crosscheck_aois:
            hit, seen, max_dwell = _assess_crosscheck(in_window, rule.crosscheck_aois)
        else:
            hit, seen, max_dwell = None, [], 0.0

        results.append({
            "procedure": ev.procedure,
            "task_object": ev.task_object,
            "task_value": ev.value,
            "task_start_s": round(t_start, 3),
            "task_end_accepted_s": round(t_accept_end, 3),
            "task_from_table": rule.task,
            "crosscheck_aoi": list(rule.crosscheck_aois),
            "crosschecked": hit,
            "max_fix_dwell_s": max_dwell,
            "seen_aoi": seen,
            "n_eye_samples_in_window": len(in_window),
        })

    n_assessable = sum(1 for r in results if r["crosschecked"] is not None)
    n_crosschecked = sum(1 for r in results if r["crosschecked"] is True)

    return {
        "run_dir": run_dir.name,
        "condition": condition,
        "n_eye_events": len(eye_events),
        "eye_data_available": has_eye_data,
        "n_task_events": len(task_events),
        "n_matched": len(results),
        "n_assessable": n_assessable,
        "n_crosschecked": n_crosschecked,
        "task_results": results,
        "unmatched_task_objects": sorted(set(unmatched)),
    }


def _save_run_report(run_res: dict) -> None:
    run_dir = OUT_DIR / run_res["run_dir"]
    out = run_dir / "team-analyzer" / "crosscheck_perf_report.txt"
    payload = {
        "run_dir": run_res["run_dir"],
        "condition": run_res["condition"],
        "min_fixation_s": MIN_FIXATION_S,
        "result": run_res,
    }
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, indent=2, ensure_ascii=False))
        fh.write("\n--- END SUMMARY ---\n")


def _wilcoxon_test(a: list[Optional[float]], b: list[Optional[float]]) -> tuple[float, float]:
    pairs = [(float(x), float(y)) for x, y in zip(a, b) if x is not None and y is not None]
    if len(pairs) < 4:
        return 1.0, 0.0
    xa, xb = np.array([p[0] for p in pairs]), np.array([p[1] for p in pairs])
    if np.allclose(xa, xb):
        return 1.0, 0.0
    try:
        res = _sp_wilcoxon(xa, xb, alternative="two-sided", zero_method="wilcox")
        pval = float(res.pvalue)
        if not np.isfinite(pval):
            return 1.0, 0.0
        n = len(pairs)
        r = 1.0 - 2.0 * res.statistic / (n * (n + 1) / 2.0)
        if not np.isfinite(r):
            r = 0.0
        return pval, float(r)
    except Exception:
        return 1.0, 0.0


def _friedman_test(groups: list[list[Optional[float]]]) -> tuple[float, float]:
    rows = list(zip(*groups))
    valid = [r for r in rows if all(x is not None for x in r)]
    if len(valid) < 3:
        return 0.0, 1.0
    aligned = [[float(r[i]) for r in valid] for i in range(len(groups))]
    # Degenerate case: identical values across all conditions for every row.
    if all(np.allclose(np.array(aligned[0]), np.array(aligned[i])) for i in range(1, len(aligned))):
        return 0.0, 1.0
    try:
        res = friedmanchisquare(*aligned)
        chi2 = float(res.statistic)
        pval = float(res.pvalue)
        if not np.isfinite(chi2) or not np.isfinite(pval):
            return 0.0, 1.0
        return chi2, pval
    except Exception:
        return 0.0, 1.0


def _holm_correct(pvals: list[float], alpha: float = 0.05):
    if not pvals:
        return np.array([], dtype=bool), np.array([])
    safe = [p if np.isfinite(p) else 1.0 for p in pvals]
    reject, p_corr, _, _ = multipletests(safe, alpha=alpha, method="holm")
    return reject, p_corr


def _desc(vals: list[Optional[float]]) -> dict:
    a = np.array([v for v in vals if v is not None], dtype=float)
    if len(a) == 0:
        return {"n": 0, "mean": None, "sd": None, "median": None, "min": None, "max": None}
    return {
        "n": int(len(a)),
        "mean": round(float(np.mean(a)), 2),
        "sd": round(float(np.std(a, ddof=1)), 2) if len(a) > 1 else 0.0,
        "median": round(float(np.median(a)), 2),
        "min": round(float(np.min(a)), 2),
        "max": round(float(np.max(a)), 2),
    }


def _write_compare_report(run_results: list[dict]) -> Path:
    by_cond: dict[str, list[Optional[float]]] = {c: [] for c in CONDITIONS}
    for cond in CONDITIONS:
        for r in [x for x in run_results if x["condition"] == cond]:
            rate = None
            if r["n_assessable"] > 0:
                rate = 100.0 * r["n_crosschecked"] / r["n_assessable"]
            by_cond[cond].append(rate)

    desc = {c: _desc(by_cond[c]) for c in CONDITIONS}
    chi2, p_f = _friedman_test([by_cond[c] for c in CONDITIONS])

    p_raw, rs = [], []
    for bl, cp in PAIRWISE:
        p, r = _wilcoxon_test(by_cond[bl], by_cond[cp])
        p_raw.append(p)
        rs.append(r)
    reject, p_corr = _holm_correct(p_raw)

    summary = {
        "domain": "mitls_crosscheck",
        "n_runs": len(run_results),
        "min_fixation_s": MIN_FIXATION_S,
        "conditions": CONDITIONS,
        "crosscheck_rate_pct": desc,
        "friedman": {"chi2": chi2, "p": p_f, "df": 2},
        "pairwise": [
            {
                "pair": f"{b}-{a}",
                "p_raw": float(p_raw[i]),
                "p_corr": float(p_corr[i]),
                "r": float(rs[i]),
                "reject": bool(reject[i]),
            }
            for i, (b, a) in enumerate(PAIRWISE)
        ],
    }

    out = OUT_DIR / "crosscheck_compare_report.txt"
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(json.dumps(summary, indent=2, ensure_ascii=False))
        fh.write("\n--- END SUMMARY ---\n")
    return out


def _plot_compare(run_results: list[dict]) -> Path:
    vals = {c: [] for c in CONDITIONS}
    for c in CONDITIONS:
        for r in [x for x in run_results if x["condition"] == c]:
            if r["n_assessable"] > 0:
                vals[c].append(100.0 * r["n_crosschecked"] / r["n_assessable"])

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(CONDITIONS))
    colors = {"TARS": "#4878CF", "TARP-F": "#6ACC65", "TARP-S": "#D65F5F"}

    for i, c in enumerate(CONDITIONS):
        v = vals[c]
        if len(v) >= 2:
            ax.boxplot(
                v, positions=[i], widths=[0.5], patch_artist=True,
                medianprops=dict(color="white", linewidth=2),
                boxprops=dict(facecolor=colors[c], alpha=0.75),
                whiskerprops=dict(color=colors[c]),
                capprops=dict(color=colors[c]),
            )
            jitter = np.linspace(-0.1, 0.1, len(v))
            ax.scatter(np.full(len(v), i) + jitter, v, s=16, color=colors[c], alpha=0.35, zorder=3)
        elif len(v) == 1:
            ax.scatter([i], v, s=50, color=colors[c], edgecolors="black", zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(CONDITIONS)
    ax.set_ylabel("Crosscheck rate (% of assessable tasks)")
    ax.set_title("MITLS Crosscheck by Condition")
    ax.set_ylim(0, 105)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    plt.tight_layout()

    out = OUT_DIR / "crosscheck_boxplots_mitls.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white", transparent=False)
    plt.close(fig)
    return out


def _run_dirs_for_condition(cond: str) -> list[Path]:
    run_num = CONDITION_TO_RUNNUM[cond]
    patt = f"run_{run_num}_results_*"
    return sorted([p for p in OUT_DIR.glob(patt) if p.is_dir()])


def main() -> None:
    parser = argparse.ArgumentParser(description="MITLS crosscheck performance analysis")
    parser.add_argument("--condition", choices=CONDITIONS, default=None)
    parser.add_argument("--max-runs", type=int, default=None)
    args = parser.parse_args()

    rules = load_task_rules(TASK_TABLE_PATH)
    conds = [args.condition] if args.condition else CONDITIONS

    run_results = []
    for cond in conds:
        run_dirs = _run_dirs_for_condition(cond)
        if args.max_runs is not None:
            run_dirs = run_dirs[: max(0, args.max_runs)]
        print(f"\n=== {cond} ===")
        print(f"  Runs found: {len(run_dirs)}")
        for run_dir in run_dirs:
            print(f"  parsing {run_dir.name}")
            rr = analyze_run(run_dir, cond, rules)
            _save_run_report(rr)
            run_results.append(rr)

    if not run_results:
        print("No MITLS runs were analyzed.")
        return

    compare_path = _write_compare_report(run_results)
    plot_path = _plot_compare(run_results)
    print(f"\nSaved: {compare_path}")
    print(f"Saved: {plot_path}")


if __name__ == "__main__":
    main()
