#!/usr/bin/env python3
"""
compare_performance.py — Full Cross-participant Flight Performance Comparison
==============================================================================
Compare-type script: orchestrates all three performance comparison analyses
(aviate, navigate, time) for ALL participants in one pass.
Run from the repository root:
    python HITLS/compare_performance.py

The script will show a summary of what will be generated or overwritten and
ask for confirmation before doing any work.

  1. Ensures aviate_perf, navigate_perf, and time_perf reports exist for
     every participant (regenerates any that are missing).
  2. Runs all three compare scripts' plot functions in sequence:

       Aviate (slip · roll · airspeed during climb)
         aviate_boxplots.png
         aviate_rmse_distributions.png
         aviate_nmae_distributions.png

       Navigate (XTE · ATD · heading · altitude)
         navigate_boxplots.png
         navigate_rmse_distributions.png
         navigate_nmae_distributions.png

       Time (scenario duration · failure→nominal · per-procedure timing)
         time_boxplots.png
         time_distributions.png
         time_mean_task_distributions.png

All figures are saved to HITLS/plots/.
"""

import os
import sys
import json
import importlib.util
import subprocess

# ── Paths ─────────────────────────────────────────────────────────────────────
HITLS_DIR = os.path.dirname(os.path.abspath(__file__))
PERF_DIR  = os.path.join(HITLS_DIR, "performance")
PLOTS_DIR = os.path.join(HITLS_DIR, "plots")
PYTHON    = sys.executable

# ── Plots produced by this script (across all three domains) ──────────────────
_ALL_PLOTS = [
    "aviate_boxplots.png",
    "aviate_rmse_distributions.png",
    "aviate_nmae_distributions.png",
    "navigate_boxplots.png",
    "navigate_rmse_distributions.png",
    "navigate_nmae_distributions.png",
    "time_boxplots.png",
    "time_distributions.png",
    "time_mean_task_distributions.png",
]


# ═══════════════════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def find_participants():
    return [
        e for e in sorted(os.listdir(HITLS_DIR))
        if os.path.isdir(os.path.join(HITLS_DIR, e))
        and e.startswith("P") and e[1:].isdigit()
    ]


def _load_json(path):
    try:
        txt   = open(path, encoding="utf-8").read()
        start = txt.index("{")
        end   = txt.index("--- END SUMMARY ---")
        return json.loads(txt[start:end].strip())
    except (FileNotFoundError, ValueError, json.JSONDecodeError):
        return None


def _has_aviate(pid):
    data = _load_json(os.path.join(HITLS_DIR, pid, "cleaned", f"{pid}_aviate_perf_report.txt"))
    if not data:
        return False
    for cond_data in data.get("conditions", {}).values():
        return "nmae" in cond_data.get("slip", {})
    return False


def _has_navigate(pid):
    data = _load_json(os.path.join(HITLS_DIR, pid, "cleaned", f"{pid}_navigate_perf_report.txt"))
    if not data:
        return False
    for cond_data in data.get("conditions", {}).values():
        return "nmae" in cond_data.get("climb", {}).get("xte", {})
    return False


def _has_time(pid):
    data = _load_json(os.path.join(HITLS_DIR, pid, "cleaned", f"{pid}_time_perf_report.txt"))
    if not data:
        return False
    for cond_data in data.get("conditions", {}).values():
        return "scenario_duration_s" in cond_data
    return False


def _missing_for(pid):
    """Return list of report kinds missing for this participant."""
    missing = []
    if not _has_aviate(pid):
        missing.append("aviate")
    if not _has_navigate(pid):
        missing.append("navigate")
    if not _has_time(pid):
        missing.append("time")
    return missing


def _run_script(script_path, participant_number):
    proc = subprocess.run(
        [PYTHON, script_path],
        input=f"{participant_number}\n",
        text=True,
        capture_output=True,
    )
    if proc.returncode != 0:
        name = os.path.basename(script_path)
        print(f"  ⚠  {name} returned code {proc.returncode}")
        if proc.stderr:
            print(proc.stderr[:400])


def _load_module(name, filename):
    path = os.path.join(PERF_DIR, filename)
    spec = importlib.util.spec_from_file_location(name, path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ═══════════════════════════════════════════════════════════════════════════════
#  Pre-run confirmation (compare-type script pattern)
# ═══════════════════════════════════════════════════════════════════════════════

def _confirm_run(participants, missing_map, output_plots):
    """Print a pre-run summary and ask the user to confirm before proceeding."""
    print()
    total_missing = sum(len(v) for v in missing_map.values())
    if total_missing:
        print(f"Reports to generate ({total_missing} missing across participants):")
        for pid in participants:
            kinds = missing_map.get(pid, [])
            if kinds:
                print(f"  + {pid}: {', '.join(k + '_perf' for k in kinds)}")
    else:
        print("All participant reports are up to date.")
    print(f"\nOutput plots that will be written/overwritten ({len(output_plots)}):")
    for name in output_plots:
        path = os.path.join(PLOTS_DIR, name)
        tag  = "[overwrite]" if os.path.exists(path) else "[new     ]"
        print(f"  {tag}  {name}")
    print()
    try:
        ans = input("Continue? [Y/n]: ").strip().lower()
    except KeyboardInterrupt:
        print("\nAborted.")
        return False
    return ans in ("", "y", "yes")


# ═══════════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    sep = "=" * 70
    print(f"\n{sep}")
    print("  HITLS — Cross-participant Flight Performance Comparison")
    print(f"{sep}")

    participants = find_participants()
    print(f"\nParticipants found: {', '.join(participants)}")

    missing_map = {pid: _missing_for(pid) for pid in participants}
    missing_map = {pid: kinds for pid, kinds in missing_map.items() if kinds}

    if not _confirm_run(participants, missing_map, _ALL_PLOTS):
        return

    # ── Step 1: Generate any missing per-participant reports ──────────────────
    _scripts = {
        "aviate":   os.path.join(PERF_DIR, "aviate_perf.py"),
        "navigate": os.path.join(PERF_DIR, "navigate_perf.py"),
        "time":     os.path.join(PERF_DIR, "time_perf.py"),
    }

    n_total = sum(len(v) for v in missing_map.values())
    if n_total:
        print(f"\n[1/4] Generating {n_total} missing report(s) …")
        for i, pid in enumerate(participants, start=1):
            for kind in missing_map.get(pid, []):
                print(f"  {pid} / {kind}_perf …", flush=True)
                _run_script(_scripts[kind], str(i))
    else:
        print("\n[1/4] All reports up to date.")

    # ── Step 2: Load the three compare modules ────────────────────────────────
    print("\n[2/4] Loading compare modules …")
    cmp_aviate   = _load_module("compare_aviate",   "compare_aviate.py")
    cmp_navigate = _load_module("compare_navigate", "compare_navigate.py")
    cmp_time     = _load_module("compare_time",     "compare_time.py")

    # ── Step 3: Load data ─────────────────────────────────────────────────────
    print("\n[3/4] Loading data …")
    aviate_data   = cmp_aviate.load_all(participants)
    navigate_data = cmp_navigate.load_all(participants)
    time_data     = cmp_time.load_all(participants)

    av_pids   = list(aviate_data.keys())
    nav_pids  = list(navigate_data.keys())
    time_pids = list(time_data.keys())

    print(f"  Aviate:   {len(av_pids)} participant(s) — {', '.join(av_pids) or 'none'}")
    print(f"  Navigate: {len(nav_pids)} participant(s) — {', '.join(nav_pids) or 'none'}")
    print(f"  Time:     {len(time_pids)} participant(s) — {', '.join(time_pids) or 'none'}")

    if not av_pids and not nav_pids and not time_pids:
        print("  No data available — aborting.")
        return

    os.makedirs(PLOTS_DIR, exist_ok=True)
    print(f"\n[4/4] Generating charts → {PLOTS_DIR}/\n")

    figs = []

    # ── Aviate ────────────────────────────────────────────────────────────────
    if av_pids:
        print("  ── Aviate (slip · roll · airspeed) ──")
        figs.append(cmp_aviate.plot_boxplots(aviate_data, av_pids))
        figs.append(cmp_aviate.plot_rmse_distributions(aviate_data, av_pids))
        figs.append(cmp_aviate.plot_nmae_distributions(aviate_data, av_pids))
    else:
        print("  ── Aviate: no data, skipping.")

    # ── Navigate ──────────────────────────────────────────────────────────────
    if nav_pids:
        print("  ── Navigate (XTE · ATD · heading · altitude) ──")
        figs.append(cmp_navigate.plot_boxplots(navigate_data, nav_pids))
        figs.append(cmp_navigate.plot_rmse_distributions(navigate_data, nav_pids))
        figs.append(cmp_navigate.plot_nmae_distributions(navigate_data, nav_pids))
    else:
        print("  ── Navigate: no data, skipping.")

    # ── Time ──────────────────────────────────────────────────────────────────
    if time_pids:
        print("  ── Time (scenario duration · failure→nominal · per-procedure) ──")
        figs.append(cmp_time.plot_boxplots(time_data, time_pids))
        figs.append(cmp_time.plot_distributions(time_data, time_pids))
        figs.append(cmp_time.plot_mean_task_distributions(time_data, time_pids))
    else:
        print("  ── Time: no data, skipping.")

    import matplotlib.pyplot as plt
    n_saved = sum(1 for f in figs if f is not None)
    print(f"\nDone — {n_saved} figure(s) saved to {PLOTS_DIR}/")
    plt.show()


if __name__ == "__main__":
    main()
