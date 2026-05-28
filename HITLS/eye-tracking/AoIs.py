#!/usr/bin/env python3
"""
AoIs.py — Eye-tracking Area of Interest analysis for HITLS scenarios.

Data source : SmartEyeProBridge signals in scenario ingescape CSV files.
Method      : filtered_closest_world_object_name when
              filtered_closest_world_count == 1.
              Consecutive same-AoI samples are grouped into fixation bouts.

Output (per scenario) :
  PXX/cleaned/PXX_<scenario>_aoi.png / .eps  — two-panel bar chart
  PXX/cleaned/PXX_<scenario>_aoi.csv         — metrics CSV

Output (per participant, per condition) :
  PXX/cleaned/PXX_<COND>_aoi_summary.png / .eps
  PXX/cleaned/PXX_<COND>_aoi_summary.csv

Cross-participant report :
  compare_performance/aoi_report.txt
  compare_performance/aoi_<COND>.png / .eps

Chart format is intentionally compatible with MITLS/eye_movement/eye-movement-analyzer.py.

Usage:
  python3 AoIs.py          — interactive menu
  python3 AoIs.py P05      — single participant
  python3 AoIs.py ALL      — all participants + cross-participant report
"""

import os
import sys
import csv
import glob
import re
import numpy as np
try:
    from scipy import stats as sp_stats
    _SCIPY = True
except ImportError:
    _SCIPY = False
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict  # noqa: F401 (kept for potential extension)

# ── Directory layout ────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
HITLS_DIR  = os.path.dirname(SCRIPT_DIR)
CMP_DIR    = os.path.join(HITLS_DIR, 'compare_performance')

# ── AoI definitions ─────────────────────────────────────────────────────────────
# 3D-world object names from SmartEyeProBridge → grouped AoI labels.
# OTW variants are merged into a single Outside_Window category to match MITLS.
AOI_MAP = {
    'TARS':           'TARS',
    'PFD':            'PFD',
    'ND':             'ND',
    'pedestal':       'pedestal',
    'center_OTW':     'Outside_Window',
    'left_OTW':       'Outside_Window',
    'leftmost_OTW':   'Outside_Window',
    'right_OTW':      'Outside_Window',
    'rightmost_OTW':  'Outside_Window',
}

# Display order (left-to-right on bar charts)
AOI_ORDER = ['TARS', 'PFD', 'ND', 'pedestal', 'Outside_Window']

CONDITIONS  = ['TARS', 'TARC', 'TARP-S', 'TARP-F']

# ── Visual style (matching MITLS) ───────────────────────────────────────────────
BAR_COLOR = '#7FB3D5'

# ── Bout-grouping threshold ─────────────────────────────────────────────────────
# Two consecutive same-AoI samples are in the same fixation bout if their
# timestamp gap is ≤ this value.  At ~60 Hz a normal gap is ~16 ms; 150 ms
# (≈ 9 missed frames) is a conservative split threshold.
GAP_THRESHOLD = 0.150  # seconds


# ═══════════════════════════════════════════════════════════════════════════════
#  Data extraction
# ═══════════════════════════════════════════════════════════════════════════════

def extract_condition(filename):
    """Return condition label from scenario filename, or None (training/unfinished)."""
    fname = os.path.basename(filename).lower()
    if 'training' in fname or 'unfinish' in fname:
        return None
    for cond in CONDITIONS:
        if cond.lower() in fname or cond.replace('-', '_').lower() in fname:
            return cond
    return None


def parse_eye_tracking(filepath):
    """
    Parse eye-tracking data from an ingescape CSV file.

    Returns a sorted list of (timestamp_sec, aoi_label) tuples for every
    frame where filtered_closest_world_count == 1 and the object name maps
    to a known AoI.

    Supports both old format (column 'timestamp', Unix seconds) and new
    format (column 'relative_time_us', relative microseconds).

    Signals are processed sequentially in file order.  Within each frame
    SmartEyeProBridge always emits filtered_closest_world_count before
    filtered_closest_world_object_name, so we track current count state and
    emit a sample whenever we see a valid object_name while count == 1.
    This correctly handles both formats:
      - Old (P02–P13, P15–P17): count and name share nearly the same timestamp.
      - New (P14, P20): each signal gets its own timestamp a few µs apart.
    """
    with open(filepath, newline='', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=';')
        header = next(reader)

    if 'relative_time_us' in header:
        ts_col   = 'relative_time_us'
        ts_scale = 1e-6
    else:
        ts_col   = 'timestamp'
        ts_scale = 1.0

    try:
        ts_idx     = header.index(ts_col)
        agent_idx  = header.index('agent')
        source_idx = header.index('source')
        val_idx    = header.index('value')
    except ValueError as exc:
        print(f"    WARNING: missing column in {os.path.basename(filepath)}: {exc}")
        return []

    KEY_C = 'filtered_closest_world_count'
    KEY_N = 'filtered_closest_world_object_name'

    samples   = []
    cur_count = '0'  # last seen count value

    with open(filepath, newline='', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=';')
        next(reader)  # skip header
        for row in reader:
            if len(row) <= val_idx:
                continue
            if row[agent_idx] != 'SmartEyeProBridge':
                continue
            src = row[source_idx]

            if src == KEY_C:
                cur_count = row[val_idx]

            elif src == KEY_N and cur_count == '1':
                aoi = AOI_MAP.get(row[val_idx])
                if aoi:
                    ts_sec = float(row[ts_idx]) * ts_scale
                    samples.append((ts_sec, aoi))

    return samples


# ═══════════════════════════════════════════════════════════════════════════════
#  Bout grouping & metrics
# ═══════════════════════════════════════════════════════════════════════════════

def group_into_bouts(samples):
    """
    Group consecutive same-AoI samples into fixation bouts.

    A new bout starts when the AoI label changes or there is a temporal gap
    larger than GAP_THRESHOLD seconds (accounting for blinks / saccades that
    briefly break continuous tracking).

    Returns a list of (aoi_label, duration_sec) tuples.
    The duration of each bout is  (last_sample_ts − first_sample_ts) plus one
    estimated frame interval so that single-sample bouts get a non-zero
    duration (~16 ms at 60 Hz).
    """
    if not samples:
        return []

    bouts      = []
    aoi_cur    = samples[0][1]
    ts_start   = samples[0][0]
    ts_end     = samples[0][0]
    frame_intervals = []

    prev_ts = samples[0][0]

    for ts, aoi in samples[1:]:
        gap = ts - prev_ts

        # Accumulate inter-frame intervals to estimate frame rate
        if gap < GAP_THRESHOLD:
            frame_intervals.append(gap)

        if aoi == aoi_cur and gap < GAP_THRESHOLD:
            ts_end = ts
        else:
            # Close current bout
            avg_fi   = np.mean(frame_intervals) if frame_intervals else 0.016
            duration = max(ts_end - ts_start + avg_fi, avg_fi)
            bouts.append((aoi_cur, duration))
            # Start new bout
            aoi_cur  = aoi
            ts_start = ts
            ts_end   = ts

        prev_ts = ts

    # Close final bout
    avg_fi   = np.mean(frame_intervals) if frame_intervals else 0.016
    duration = max(ts_end - ts_start + avg_fi, avg_fi)
    bouts.append((aoi_cur, duration))

    return bouts


def calculate_aoi_metrics(bouts):
    """
    Compute per-AoI metrics from a list of fixation bouts.

    Returns a dict: aoi_label → {frequency, avg_dwell_time,
                                  total_dwell_time, percentage}
    """
    data        = {aoi: {'count': 0, 'total': 0.0} for aoi in AOI_ORDER}
    total_bouts = len(bouts)

    for aoi, dur in bouts:
        if aoi in data:
            data[aoi]['count'] += 1
            data[aoi]['total'] += dur

    metrics = {}
    for aoi in AOI_ORDER:
        count      = data[aoi]['count']
        total_dwell = data[aoi]['total']
        avg_dwell  = total_dwell / count if count > 0 else 0.0
        pct        = (count / total_bouts * 100) if total_bouts > 0 else 0.0
        metrics[aoi] = {
            'frequency':        count,
            'avg_dwell_time':   avg_dwell,
            'total_dwell_time': total_dwell,
            'percentage':       pct,
        }
    return metrics


def aggregate_metrics(metrics_list):
    """
    Aggregate a list of per-scenario metric dicts into a single dict.
    Frequencies and dwell times are summed; averages and percentages
    are recalculated from the aggregated totals.
    """
    combined    = {aoi: {'count': 0, 'total': 0.0} for aoi in AOI_ORDER}
    total_bouts = 0

    for m in metrics_list:
        for aoi in AOI_ORDER:
            combined[aoi]['count'] += m[aoi]['frequency']
            combined[aoi]['total'] += m[aoi]['total_dwell_time']
            total_bouts            += m[aoi]['frequency']

    agg = {}
    for aoi in AOI_ORDER:
        count       = combined[aoi]['count']
        total_dwell = combined[aoi]['total']
        avg_dwell   = total_dwell / count if count > 0 else 0.0
        pct         = (count / total_bouts * 100) if total_bouts > 0 else 0.0
        agg[aoi] = {
            'frequency':        count,
            'avg_dwell_time':   avg_dwell,
            'total_dwell_time': total_dwell,
            'percentage':       pct,
        }
    return agg


# ═══════════════════════════════════════════════════════════════════════════════
#  Output functions
# ═══════════════════════════════════════════════════════════════════════════════

def print_aoi_statistics(metrics, title=''):
    """Print AoI statistics table (matches MITLS format)."""
    print(f"\n{'='*80}")
    header_txt = f"AREA OF INTEREST (AoI) ANALYSIS"
    if title:
        header_txt += f" — {title}"
    print(header_txt)
    print(f"{'='*80}")
    print(f"{'AoI Name':<20} {'Fixations':<12} {'Percentage':<12} "
          f"{'Avg Dwell (s)':<15} {'Total Dwell (s)':<15}")
    print(f"{'-'*80}")

    sorted_aois = sorted(metrics.items(),
                         key=lambda x: x[1]['frequency'], reverse=True)
    for aoi, m in sorted_aois:
        if m['frequency'] > 0:
            print(f"{aoi:<20} {m['frequency']:<12} {m['percentage']:>6.2f}%     "
                  f"{m['avg_dwell_time']:>8.3f}        {m['total_dwell_time']:>8.3f}")
    print(f"{'='*80}\n")


def plot_aoi_metrics(metrics, save_as, title=''):
    """
    Two-panel bar chart matching MITLS eye-movement-analyzer.py:
      Left panel  — fixation count with percentage labels
      Right panel — average dwell time in seconds
    """
    aois       = [a for a in AOI_ORDER if metrics[a]['frequency'] > 0]
    freqs      = [metrics[a]['frequency']     for a in aois]
    pcts       = [metrics[a]['percentage']    for a in aois]
    avg_dwells = [metrics[a]['avg_dwell_time'] for a in aois]

    if not aois:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # ── Left: fixation count ──────────────────────────────────────────────────
    bars1 = ax1.bar(range(len(aois)), freqs,
                    color=BAR_COLOR, edgecolor='black', linewidth=1.5)
    ax1.set_xlabel('Area of Interest', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Number of Fixations', fontsize=12, fontweight='bold')
    ax1.set_xticks(range(len(aois)))
    ax1.set_xticklabels(aois, rotation=45, ha='right')
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    if title:
        ax1.set_title(title, fontsize=13, fontweight='bold', pad=10)

    for bar, freq, pct in zip(bars1, freqs, pcts):
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2., h,
                 f'{freq}\n({pct:.1f}%)',
                 ha='center', va='bottom', fontsize=10, fontweight='bold')

    # ── Right: average dwell time ─────────────────────────────────────────────
    bars2 = ax2.bar(range(len(aois)), avg_dwells,
                    color=BAR_COLOR, edgecolor='black', linewidth=1.5)
    ax2.set_xlabel('Area of Interest', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Average Dwell Time (seconds)', fontsize=12, fontweight='bold')
    ax2.set_xticks(range(len(aois)))
    ax2.set_xticklabels(aois, rotation=45, ha='right')
    ax2.grid(axis='y', alpha=0.3, linestyle='--')

    for bar, dwell in zip(bars2, avg_dwells):
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2., h,
                 f'{dwell:.3f}s',
                 ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_as, dpi=300, bbox_inches='tight')
    eps_path = save_as.replace('.png', '.eps')
    plt.savefig(eps_path, format='eps', bbox_inches='tight')
    print(f"    Saved {os.path.basename(save_as)}")
    plt.close()


def export_aoi_metrics_to_csv(metrics, output_path):
    """Export metrics to CSV (matches MITLS column names)."""
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['AoI_Name', 'Fixation_Count', 'Percentage',
                         'Avg_Dwell_Time_s', 'Total_Dwell_Time_s'])
        for aoi in AOI_ORDER:
            m = metrics[aoi]
            writer.writerow([
                aoi,
                m['frequency'],
                f"{m['percentage']:.2f}",
                f"{m['avg_dwell_time']:.6f}",
                f"{m['total_dwell_time']:.6f}",
            ])
    print(f"    Exported {os.path.basename(output_path)}")


# ═══════════════════════════════════════════════════════════════════════════════
#  Per-participant analysis
# ═══════════════════════════════════════════════════════════════════════════════

def analyse_participant(pid, participant_dir, verbose=True):
    """
    Process all (non-training) scenario ingescape CSVs for one participant.

    Returns  {condition: [{'file': str, 'metrics': dict}, ...]}
    """
    scenarios_dir = os.path.join(participant_dir, 'scenarios')
    cleaned_dir   = os.path.join(participant_dir, 'cleaned')
    os.makedirs(cleaned_dir, exist_ok=True)

    pattern = os.path.join(scenarios_dir, '*ingescape*.csv')
    scenario_files = sorted(glob.glob(pattern))

    if not scenario_files:
        print(f"  No ingescape scenario CSVs found in {scenarios_dir}")
        return {}

    results = {}  # condition → list

    for filepath in scenario_files:
        cond = extract_condition(filepath)
        if cond is None:
            continue  # skip training / unfinished

        fname = os.path.basename(filepath)
        if verbose:
            print(f"\n  ── {fname}")

        samples = parse_eye_tracking(filepath)

        if not samples:
            if verbose:
                print(f"    → no valid eye-tracking samples")
            continue

        bouts   = group_into_bouts(samples)
        metrics = calculate_aoi_metrics(bouts)

        total_tracked = sum(m['total_dwell_time'] for m in metrics.values())
        if verbose:
            print(f"    {len(samples)} samples → {len(bouts)} fixation bouts "
                  f"({total_tracked:.1f}s tracked)")
            print_aoi_statistics(metrics, title=f"{pid} – {cond} – {fname}")

        # Strip trailing _ingescape.csv for a cleaner stem
        stem = re.sub(r'_ingescape\.csv$', '', fname, flags=re.IGNORECASE)
        stem = re.sub(r'\.csv$', '', stem, flags=re.IGNORECASE)

        plot_path = os.path.join(cleaned_dir, f"{pid}_{stem}_aoi.png")
        csv_path  = os.path.join(cleaned_dir, f"{pid}_{stem}_aoi.csv")

        plot_aoi_metrics(metrics, save_as=plot_path,
                         title=f"{pid} – {cond} – {stem}")
        export_aoi_metrics_to_csv(metrics, csv_path)

        results.setdefault(cond, []).append({'file': fname, 'metrics': metrics})

    return results


def run_participant(pid):
    """Full analysis for a single participant including per-condition summaries."""
    participant_dir = os.path.join(HITLS_DIR, pid)
    if not os.path.isdir(participant_dir):
        print(f"Participant directory not found: {participant_dir}")
        return

    print(f"\n{'='*60}")
    print(f"  {pid}")
    print(f"{'='*60}")

    results     = analyse_participant(pid, participant_dir, verbose=True)
    cleaned_dir = os.path.join(participant_dir, 'cleaned')

    if not results:
        print(f"  No usable scenarios found for {pid}")
        return

    # Per-condition aggregated summary
    for cond in CONDITIONS:
        scenario_list = results.get(cond, [])
        if not scenario_list:
            continue

        agg       = aggregate_metrics([s['metrics'] for s in scenario_list])
        plot_path = os.path.join(cleaned_dir, f"{pid}_{cond}_aoi_summary.png")
        csv_path  = os.path.join(cleaned_dir, f"{pid}_{cond}_aoi_summary.csv")

        print(f"\n  ══ {pid} – {cond} – {len(scenario_list)} scenario(s) aggregated ══")
        print_aoi_statistics(agg, title=f"{pid} – {cond} (all scenarios)")
        plot_aoi_metrics(agg, save_as=plot_path,
                         title=f"{pid} – {cond} (all scenarios)")
        export_aoi_metrics_to_csv(agg, csv_path)


# ═══════════════════════════════════════════════════════════════════════════════
#  Statistical helpers
# ═══════════════════════════════════════════════════════════════════════════════

def tars_vs_nontars_stats(cond_data):
    """
    For a condition dict {pid: aggregate_metrics}, compute TARS dwell vs.
    combined non-TARS dwell (PFD + ND + pedestal + Outside_Window).

    Returns a dict with descriptive stats and Wilcoxon signed-rank result.
    Uses population std (ddof=0) to match the MEAN±SD rows in the report.
    """
    tars_vals    = []
    nontars_vals = []
    for pid in sorted(cond_data):
        m  = cond_data[pid]
        t  = m['TARS']['total_dwell_time']
        nt = sum(m[aoi]['total_dwell_time'] for aoi in AOI_ORDER if aoi != 'TARS')
        tars_vals.append(t)
        nontars_vals.append(nt)

    ta = np.array(tars_vals)
    na = np.array(nontars_vals)

    result = {
        'n':              len(ta),
        'tars_vals':      tars_vals,
        'nontars_vals':   nontars_vals,
        'mean_tars':      float(np.mean(ta)),
        'sd_tars':        float(np.std(ta)),
        'median_tars':    float(np.median(ta)),
        'mean_nontars':   float(np.mean(na)),
        'sd_nontars':     float(np.std(na)),
        'median_nontars': float(np.median(na)),
        'W':              float('nan'),
        'p':              float('nan'),
    }

    if _SCIPY and len(ta) >= 6:
        try:
            W, p = sp_stats.wilcoxon(ta, na)
            result['W'] = float(W)
            result['p'] = float(p)
        except Exception:
            pass

    return result


def _sig_stars(p):
    """Return significance stars for a p-value, or 'n/a' if NaN."""
    if np.isnan(p):
        return 'n/a'
    if p < 0.001:
        return '***'
    if p < 0.01:
        return '**'
    if p < 0.05:
        return '*'
    return 'ns'


def plot_tars_vs_nontars_comparison(stats_by_cond, save_as):
    """
    Grouped bar chart: mean TARS vs mean non-TARS total dwell time per condition,
    with ±1 SD error bars and Wilcoxon significance annotations above each pair.
    """
    conds    = [c for c in CONDITIONS if c in stats_by_cond]
    means_t  = [stats_by_cond[c]['mean_tars']    for c in conds]
    sds_t    = [stats_by_cond[c]['sd_tars']      for c in conds]
    means_nt = [stats_by_cond[c]['mean_nontars'] for c in conds]
    sds_nt   = [stats_by_cond[c]['sd_nontars']   for c in conds]

    x     = np.arange(len(conds))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width / 2, means_t,  width, yerr=sds_t,
           label='TARS', color='#7FB3D5', edgecolor='black', linewidth=1.2,
           capsize=5, error_kw={'linewidth': 1.5})
    ax.bar(x + width / 2, means_nt, width, yerr=sds_nt,
           label='Non-TARS', color='#F0B27A', edgecolor='black', linewidth=1.2,
           capsize=5, error_kw={'linewidth': 1.5})

    # Significance annotation above each group
    all_tops = [m + s for m, s in zip(means_t + means_nt, sds_t + sds_nt)]
    y_max = max(all_tops) if all_tops else 1.0
    for i, cond in enumerate(conds):
        st    = stats_by_cond[cond]
        sig   = _sig_stars(st['p'])
        if not np.isnan(st['W']):
            annot = f"W={st['W']:.0f}\np={st['p']:.4f} {sig}"
        else:
            annot = sig
        ax.text(i, y_max * 1.04, annot,
                ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_xlabel('Condition', fontsize=12, fontweight='bold')
    ax.set_ylabel('Mean Total Dwell Time (s)', fontsize=12, fontweight='bold')
    ax.set_title(
        'TARS vs. Non-TARS Dwell Time by Condition\n'
        '(Paired Wilcoxon signed-rank test, mean \u00b1 SD)',
        fontsize=13, fontweight='bold'
    )
    ax.set_xticks(x)
    ax.set_xticklabels(conds)
    ax.legend()
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim(0, y_max * 1.3)

    plt.tight_layout()
    plt.savefig(save_as, dpi=300, bbox_inches='tight')
    eps_path = save_as.replace('.png', '.eps')
    plt.savefig(eps_path, format='eps', bbox_inches='tight')
    print(f"    Saved {os.path.basename(save_as)}")
    plt.close()


# ═══════════════════════════════════════════════════════════════════════════════
#  Cross-participant analysis
# ═══════════════════════════════════════════════════════════════════════════════

def run_all():
    """Run analysis for all participants and produce cross-participant report."""
    all_pids = sorted([
        d for d in os.listdir(HITLS_DIR)
        if re.match(r'^P\d+$', d)
        and os.path.isdir(os.path.join(HITLS_DIR, d))
    ])

    all_results = {}  # pid → {cond → [scenario dicts]}

    for pid in all_pids:
        participant_dir = os.path.join(HITLS_DIR, pid)
        print(f"\n{'='*60}")
        print(f"  {pid}")
        print(f"{'='*60}")
        results = analyse_participant(pid, participant_dir, verbose=False)
        all_results[pid] = results

        # Per-condition summary charts even in batch mode
        cleaned_dir = os.path.join(participant_dir, 'cleaned')
        os.makedirs(cleaned_dir, exist_ok=True)
        for cond in CONDITIONS:
            scenario_list = results.get(cond, [])
            if not scenario_list:
                continue
            agg = aggregate_metrics([s['metrics'] for s in scenario_list])
            plot_aoi_metrics(
                agg,
                save_as=os.path.join(cleaned_dir, f"{pid}_{cond}_aoi_summary.png"),
                title=f"{pid} – {cond} (all scenarios)",
            )
            export_aoi_metrics_to_csv(
                agg,
                os.path.join(cleaned_dir, f"{pid}_{cond}_aoi_summary.csv"),
            )

    # ── Cross-participant text report ─────────────────────────────────────────
    os.makedirs(CMP_DIR, exist_ok=True)
    report_path = os.path.join(CMP_DIR, 'aoi_report.txt')

    with open(report_path, 'w', encoding='utf-8') as rpt:
        rpt.write("HITLS Eye-Tracking: Area of Interest (AoI) Analysis\n")
        rpt.write("=" * 80 + "\n")
        rpt.write("Method : fixation bouts on filtered_closest_world_object_name\n")
        rpt.write("         (SmartEyeProBridge, filtered_closest_world_count == 1)\n")
        rpt.write("Columns: Total dwell time (s) and percentage of fixation bouts\n")
        rpt.write("=" * 80 + "\n")

        stats_by_cond = {}

        for cond in CONDITIONS:
            # Collect per-participant aggregates for this condition
            cond_data = {}
            for pid in all_pids:
                scenario_list = all_results.get(pid, {}).get(cond, [])
                if scenario_list:
                    cond_data[pid] = aggregate_metrics(
                        [s['metrics'] for s in scenario_list]
                    )

            if not cond_data:
                continue

            rpt.write(f"\n\n{'─'*80}\n")
            rpt.write(f"CONDITION: {cond}   (n={len(cond_data)} participants)\n")
            rpt.write(f"{'─'*80}\n")

            # Header row
            col_w = 18
            rpt.write(f"{'Participant':<12}")
            for aoi in AOI_ORDER:
                rpt.write(f"  {aoi:^{col_w}}")
            rpt.write("\n")
            rpt.write(f"{'':12}")
            for aoi in AOI_ORDER:
                rpt.write(f"  {'dwell(s)  %':^{col_w}}")
            rpt.write("\n")
            rpt.write(f"{'-'*12}" + (f"  {'─'*col_w}" * len(AOI_ORDER)) + "\n")

            # Per-participant rows
            for pid in all_pids:
                if pid not in cond_data:
                    continue
                m = cond_data[pid]
                rpt.write(f"{pid:<12}")
                for aoi in AOI_ORDER:
                    rpt.write(f"  {m[aoi]['total_dwell_time']:>6.2f}s {m[aoi]['percentage']:>5.1f}%")
                rpt.write("\n")

            # Mean ± SD row
            rpt.write(f"\n{'MEAN±SD':<12}")
            for aoi in AOI_ORDER:
                vals = [cond_data[p][aoi]['total_dwell_time'] for p in cond_data]
                rpt.write(f"  {np.mean(vals):>5.2f}±{np.std(vals):<5.2f}   ")
            rpt.write("\n")

            # TARS vs. non-TARS inline summary
            st = tars_vs_nontars_stats(cond_data)
            stats_by_cond[cond] = st
            rpt.write(f"\nTARS vs non-TARS (Wilcoxon): "
                      f" TARS={st['mean_tars']:.2f}\u00b1{st['sd_tars']:.2f}s"
                      f"  non-TARS={st['mean_nontars']:.2f}\u00b1{st['sd_nontars']:.2f}s")
            if not np.isnan(st['W']):
                rpt.write(f"  W={st['W']:.0f}  p={st['p']:.4f}  {_sig_stars(st['p'])}")
            rpt.write("\n")

            # Cross-participant aggregated chart
            agg_all   = aggregate_metrics(list(cond_data.values()))
            plot_path = os.path.join(CMP_DIR, f"aoi_{cond}.png")
            plot_aoi_metrics(agg_all, save_as=plot_path,
                             title=f"All participants – {cond} (n={len(cond_data)})")
            print_aoi_statistics(agg_all,
                                 title=f"ALL – {cond} (n={len(cond_data)})")

        # ── TARS vs. non-TARS summary table ──────────────────────────────────
        rpt.write(f"\n\n{'='*80}\n")
        rpt.write("TARS vs. NON-TARS DWELL TIME — STATISTICAL COMPARISON\n")
        rpt.write("=" * 80 + "\n")
        rpt.write("Test   : Paired Wilcoxon signed-rank test\n")
        rpt.write("         (TARS total dwell vs. PFD+ND+pedestal+Outside_Window combined)\n")
        rpt.write("Stars  : *** p<0.001  ** p<0.01  * p<0.05  ns p\u22650.05\n")
        rpt.write(f"\n{'Condition':<10}  {'n':>4}  "
                  f"{'TARS mean\u00b1SD (s)':<22}  "
                  f"{'non-TARS mean\u00b1SD (s)':<22}  "
                  f"{'W':>7}  {'p-value':>10}  {'sig':>4}\n")
        rpt.write(f"{'-'*10}  {'-'*4}  {'-'*22}  {'-'*22}  {'-'*7}  {'-'*10}  {'-'*4}\n")
        for cond in CONDITIONS:
            if cond not in stats_by_cond:
                continue
            st    = stats_by_cond[cond]
            tstr  = f"{st['mean_tars']:.2f}\u00b1{st['sd_tars']:.2f}"
            ntstr = f"{st['mean_nontars']:.2f}\u00b1{st['sd_nontars']:.2f}"
            w_str = f"{st['W']:.0f}" if not np.isnan(st['W']) else 'n/a'
            p_str = f"{st['p']:.4f}" if not np.isnan(st['p']) else 'n/a'
            rpt.write(f"{cond:<10}  {st['n']:>4}  "
                      f"{tstr:<22}  {ntstr:<22}  "
                      f"{w_str:>7}  {p_str:>10}  {_sig_stars(st['p']):>4}\n")

    # ── Comparison chart ──────────────────────────────────────────────────────
    if stats_by_cond:
        cmp_plot = os.path.join(CMP_DIR, 'aoi_tars_vs_nontars.png')
        plot_tars_vs_nontars_comparison(stats_by_cond, cmp_plot)

    print(f"\n{'='*60}")
    print(f"Cross-participant report: {report_path}")
    print(f"{'='*60}")


# ═══════════════════════════════════════════════════════════════════════════════
#  Entry point
# ═══════════════════════════════════════════════════════════════════════════════

def _list_participants():
    return sorted([
        d for d in os.listdir(HITLS_DIR)
        if re.match(r'^P\d+$', d)
        and os.path.isdir(os.path.join(HITLS_DIR, d))
    ])


def main():
    if len(sys.argv) > 1:
        arg = sys.argv[1].strip().upper()
        if arg == 'ALL':
            run_all()
        elif re.match(r'^P\d+$', arg):
            run_participant(arg)
        else:
            print("Usage: python3 AoIs.py [PXX | ALL]")
            sys.exit(1)
        return

    # ── Interactive menu ───────────────────────────────────────────────────────
    all_pids = _list_participants()

    print("\nHITLS Eye-Tracking AoI Analysis")
    print("─" * 40)
    print("  0. ALL participants (+ cross-participant report)")
    for i, pid in enumerate(all_pids, 1):
        print(f"  {i:>2}. {pid}")
    print()

    try:
        choice = int(input("Select (0 = ALL): ").strip())
    except (ValueError, KeyboardInterrupt):
        print("Aborted.")
        sys.exit(0)

    if choice == 0:
        run_all()
    elif 1 <= choice <= len(all_pids):
        run_participant(all_pids[choice - 1])
    else:
        print("Invalid choice.")
        sys.exit(1)


if __name__ == '__main__':
    main()
