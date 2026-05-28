#!/usr/bin/env python3
"""
eyemetrics.py — Pupil diameter, blink rate, fixation rate, mean fixation
duration and saccade rate analysis for HITLS scenarios.

Source signals (SmartEyeProBridge, grouped by frame_number):
  filtered_pupil_diameter          – pupil diameter in metres  (→ ×1000 mm)
  filtered_pupil_diameter_quality  – quality score 0–1
  an_blink                         – 0 = no blink, non-zero = blink event ID
  an_fixation                      – 0 = not in fixation, non-zero = fixation ID
  an_saccade                       – 0 = not in saccade, non-zero = saccade ID

Analysis: within-subject comparison across conditions (TARS, TARC, TARP-S,
TARP-F) using the Friedman test with pairwise Wilcoxon signed-rank follow-up
(Bonferroni corrected).

Per-scenario time-series plots use a configurable sliding window.

Usage:
  python3 eyemetrics.py          — interactive menu
  python3 eyemetrics.py P05      — single participant
  python3 eyemetrics.py ALL      — all participants + cross-participant report
"""

# ═══════════════════════════════════════════════════════════════════════════════
#  TOP-OF-FILE PARAMETERS  (edit these to change behaviour)
# ═══════════════════════════════════════════════════════════════════════════════

# Minimum filtered_pupil_diameter_quality score (0–1) for a sample to be
# included in pupil diameter metrics.  Frames below this value are treated as
# missing (NaN) for all pupil calculations.
QUALITY_THRESHOLD = 0.9

# Width of the sliding window used for the per-scenario time-series plots (s).
# Wider → smoother curves; narrower → more temporal detail.
SLIDING_WINDOW_S = 30

# Step between consecutive sliding-window centres (s).
# A smaller step gives more temporal resolution (and more overlap).
SLIDE_STEP_S = 5

# Minimum eye-tracking frame duration (s) for a procedure window to be
# included in the per-procedure breakdown.  Windows shorter than this
# (e.g. LINE-UP AND HOLD where current_state events span only milliseconds)
# are discarded to avoid artefactually inflated blink/fixation/saccade rates.
MIN_PROC_DURATION_S = 10.0

# ═══════════════════════════════════════════════════════════════════════════════

import os
import sys
import csv
import glob
import json
import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict

try:
    from scipy import stats as sp_stats
    _SCIPY = True
except ImportError:
    _SCIPY = False

# ── Directory layout ──────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
HITLS_DIR  = os.path.dirname(SCRIPT_DIR)
CMP_DIR    = os.path.join(HITLS_DIR, 'compare_performance')

CONDITIONS = ['TARS', 'TARC', 'TARP-S', 'TARP-F']

METRICS = [
    'mean_pupil_mm',
    'blink_rate',
    'fixation_rate',
    'mean_fixation_dur_ms',
    'saccade_rate',
]

METRIC_LABELS = {
    'mean_pupil_mm':          'Mean Pupil Diameter (mm)',
    'blink_rate':             'Blink Rate (blinks/min)',
    'fixation_rate':          'Fixation Rate (fix/min)',
    'mean_fixation_dur_ms':   'Mean Fixation Duration (ms)',
    'saccade_rate':           'Saccade Rate (sacc/min)',
}

# Colour palette
BAR_COLOR   = '#7FB3D5'
COND_COLORS = {
    'TARS':   '#4878CF',
    'TARC':   '#6ACC65',
    'TARP-S': '#D65F5F',
    'TARP-F': '#B47CC7',
}

# Procedure sequence (mirrors time_perf.py)
_PROC_KEY = {
    'CREW BRIEFING':               'crew_briefing',
    'BEFORE TAKEOFF':              'before_takeoff',
    'LINE-UP AND HOLD':            'lineup_hold',
    'TAKEOFF':                     'takeoff',
    'ENG FAILURE DURING TAKEOFF':  'eng_failure',
    'ENGINE FIRE':                 'engine_fire',
    'DECLARE PANPAN':              'declare_panpan',
    'AFTER TAKEOFF':               'after_takeoff',
}

PROC_ORDER = [
    'crew_briefing', 'before_takeoff', 'lineup_hold', 'takeoff',
    'eng_failure', 'engine_fire', 'declare_panpan', 'after_takeoff',
]

PROC_LABELS = {
    'crew_briefing':  'Crew\nBriefing',
    'before_takeoff': 'Before\nTakeoff',
    'lineup_hold':    'Lineup\n& Hold',
    'takeoff':        'Takeoff',
    'eng_failure':    'ENG\nFailure',
    'engine_fire':    'Engine\nFire',
    'declare_panpan': 'Declare\nPANPAN',
    'after_takeoff':  'After\nTakeoff',
}


# ═══════════════════════════════════════════════════════════════════════════════
#  Helpers
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


def _sig_stars(p):
    """Significance stars, or 'n/a' for NaN."""
    if np.isnan(p): return 'n/a'
    if p < 0.001:   return '***'
    if p < 0.01:    return '**'
    if p < 0.05:    return '*'
    return 'ns'


# ═══════════════════════════════════════════════════════════════════════════════
#  Data extraction
# ═══════════════════════════════════════════════════════════════════════════════

def parse_eyemetrics(filepath):
    """
    Extract per-frame eye metrics from a scenario ingescape CSV.

    Uses frame_number as the canonical frame key so that signals with slightly
    different timestamps (old format: ±1 ms; new format: ±hundreds µs) are
    correctly grouped into the same frame.

    Supports both timestamp formats:
      Old (P02–P17 except P14): column 'timestamp', Unix seconds.
      New (P14, P20): column 'relative_time_us', relative microseconds.

    Returns a dict:
      ts           : float array – timestamps (s, relative to first frame)
      pupil_mm     : float array – filtered pupil diameter in mm
                                   (NaN where quality < QUALITY_THRESHOLD)
      quality      : float array – filtered_pupil_diameter_quality (0–1)
      blink_id     : int array   – 0 = not blinking, non-zero = blink event ID
      fixation_id  : int array   – 0 = not fixating, non-zero = fixation ID
      saccade_id   : int array   – 0 = not saccading, non-zero = saccade ID
      duration_s   : float       – scenario duration in seconds
      n_frames     : int         – number of frames with a frame_number signal

    Returns None if no SmartEyeProBridge data are found.
    """
    with open(filepath, newline='', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=';')
        header = next(reader)

    if 'relative_time_us' in header:
        ts_col, ts_scale = 'relative_time_us', 1e-6
    else:
        ts_col, ts_scale = 'timestamp', 1.0

    try:
        ts_idx     = header.index(ts_col)
        agent_idx  = header.index('agent')
        source_idx = header.index('source')
        val_idx    = header.index('value')
    except ValueError as exc:
        print(f"    WARNING: missing column in {os.path.basename(filepath)}: {exc}")
        return None

    SIG_FRAME    = 'frame_number'
    SIG_PUPIL    = 'filtered_pupil_diameter'
    SIG_QUALITY  = 'filtered_pupil_diameter_quality'
    SIG_BLINK    = 'an_blink'
    SIG_FIXATION = 'an_fixation'
    SIG_SACCADE  = 'an_saccade'
    WANTED = {SIG_FRAME, SIG_PUPIL, SIG_QUALITY, SIG_BLINK, SIG_FIXATION, SIG_SACCADE}

    # frames[frame_num] accumulates signal values for that frame.
    # The 'ts' field is set when frame_number is first seen.
    frames    = {}   # int → dict
    cur_frame = None

    with open(filepath, newline='', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=';')
        next(reader)
        for row in reader:
            if len(row) <= val_idx:
                continue
            if row[agent_idx] != 'SmartEyeProBridge':
                continue
            src = row[source_idx]
            if src not in WANTED:
                continue

            try:
                ts_sec = float(row[ts_idx]) * ts_scale
                raw_val = row[val_idx]
            except (ValueError, IndexError):
                continue

            if src == SIG_FRAME:
                cur_frame = int(float(raw_val))
                if cur_frame not in frames:
                    frames[cur_frame] = {'ts': ts_sec}
            else:
                if cur_frame is None:
                    continue
                try:
                    frames[cur_frame][src] = float(raw_val)
                except ValueError:
                    pass

    if not frames:
        return None

    # Sort frames by their timestamp
    sorted_frames = sorted(frames.values(), key=lambda f: f['ts'])
    n = len(sorted_frames)
    t0 = sorted_frames[0]['ts']

    ts          = np.array([f['ts'] - t0              for f in sorted_frames])
    pupil_raw   = np.array([f.get(SIG_PUPIL,    np.nan) for f in sorted_frames])
    quality     = np.array([f.get(SIG_QUALITY,  np.nan) for f in sorted_frames])
    blink_id    = np.array([int(f.get(SIG_BLINK,    0)) for f in sorted_frames])
    fixation_id = np.array([int(f.get(SIG_FIXATION, 0)) for f in sorted_frames])
    saccade_id  = np.array([int(f.get(SIG_SACCADE,  0)) for f in sorted_frames])

    # Convert pupil diameter metres → mm; mask low-quality frames
    pupil_mm = pupil_raw * 1000.0
    bad = np.isnan(quality) | (quality < QUALITY_THRESHOLD)
    pupil_mm[bad] = np.nan

    duration_s = float(ts[-1] - ts[0]) if n > 1 else 0.0

    return {
        'ts':           ts,
        'pupil_mm':     pupil_mm,
        'quality':      quality,
        'blink_id':     blink_id,
        'fixation_id':  fixation_id,
        'saccade_id':   saccade_id,
        't0_abs':       float(t0),
        'duration_s':   duration_s,
        'n_frames':     n,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  Procedure window extraction
# ═══════════════════════════════════════════════════════════════════════════════

def parse_procedure_windows(filepath):
    """
    Parse current_state events in an ingescape CSV to extract absolute
    time windows (min / max timestamp) for each procedure.

    Returns {proc_key: (t_abs_start, t_abs_end)} for every procedure that
    appears at least once.  Timestamps are in the same unit/scale as those
    returned by parse_eyemetrics (seconds, either Unix or relative).
    """
    proc_times = {}   # proc_key → list[float]
    ts_scale   = 1.0

    with open(filepath, newline='', encoding='utf-8') as fh:
        for line in fh:
            if line.startswith('uuid;'):
                if 'relative_time_us' in line:
                    ts_scale = 1e-6
                continue
            if 'current_state' not in line:
                continue
            parts = line.rstrip('\r\n').split(';', 6)
            if len(parts) < 7:
                continue
            try:
                ts = float(parts[1]) * ts_scale
            except ValueError:
                continue
            val = parts[6]
            if val.startswith('"') and val.endswith('"'):
                val = val[1:-1].replace('""', '"')
            try:
                d = json.loads(val)
            except (ValueError, KeyError):
                continue
            key = _PROC_KEY.get(d.get('procedure', ''))
            if key:
                proc_times.setdefault(key, []).append(ts)

    return {k: (min(v), max(v)) for k, v in proc_times.items() if v}


def compute_procedure_metrics(data, proc_windows):
    """
    Compute eye metrics for each procedure window.

    data        : output of parse_eyemetrics (must contain 't0_abs')
    proc_windows: {proc_key: (t_abs_start, t_abs_end)}

    Returns {proc_key: {metric: value}}.  Procedures with < 2 frames are
    silently skipped.
    """
    if data is None or not proc_windows:
        return {}

    ts_abs = data['ts'] + data['t0_abs']
    result = {}

    for proc_key, (t_start, t_end) in proc_windows.items():
        mask = (ts_abs >= t_start) & (ts_abs <= t_end)
        if mask.sum() < 2:
            continue
        idx = np.where(mask)[0]
        dur_s = float(data['ts'][idx[-1]] - data['ts'][idx[0]])
        if dur_s < MIN_PROC_DURATION_S:
            continue
        sub = {
            'ts':          data['ts'][mask] - data['ts'][idx[0]],
            'pupil_mm':    data['pupil_mm'][mask].copy(),
            'quality':     data['quality'][mask].copy(),
            'blink_id':    data['blink_id'][mask].copy(),
            'fixation_id': data['fixation_id'][mask].copy(),
            'saccade_id':  data['saccade_id'][mask].copy(),
            'duration_s':  float(data['ts'][idx[-1]] - data['ts'][idx[0]]),
            'n_frames':    int(mask.sum()),
        }
        sc = compute_scalar_metrics(sub)
        if any(not np.isnan(v) for v in sc.values()):
            result[proc_key] = sc

    return result


# ═══════════════════════════════════════════════════════════════════════════════
#  Scalar metrics
# ═══════════════════════════════════════════════════════════════════════════════

def compute_scalar_metrics(data):
    """
    Compute overall scalar metrics for one scenario (or sub-window).

    Blink / fixation / saccade counts are obtained by counting unique non-zero
    event IDs.  Mean fixation duration is computed per unique fixation ID.

    Returns dict: {metric_name: value}
    """
    if data is None or data['n_frames'] < 2 or data['duration_s'] <= 0:
        return {m: np.nan for m in METRICS}

    dur_min = data['duration_s'] / 60.0

    # ── Pupil diameter ────────────────────────────────────────────────────────
    valid_pupil = data['pupil_mm'][~np.isnan(data['pupil_mm'])]
    mean_pupil  = float(np.mean(valid_pupil)) if len(valid_pupil) > 0 else np.nan

    # ── Blink rate ────────────────────────────────────────────────────────────
    blink_ids  = set(data['blink_id'].tolist()) - {0}
    blink_rate = len(blink_ids) / dur_min

    # ── Fixation rate & mean fixation duration ────────────────────────────────
    fix_ids   = set(data['fixation_id'].tolist()) - {0}
    fix_rate  = len(fix_ids) / dur_min

    if fix_ids:
        # Median inter-frame interval used to convert frame counts → ms
        ifi_s = float(np.median(np.diff(data['ts']))) if len(data['ts']) > 1 else 1 / 60.0
        frame_counts = {}
        for fid in data['fixation_id']:
            if fid != 0:
                frame_counts[fid] = frame_counts.get(fid, 0) + 1
        mean_fix_dur_ms = float(np.mean(list(frame_counts.values()))) * ifi_s * 1000.0
    else:
        mean_fix_dur_ms = np.nan

    # ── Saccade rate ──────────────────────────────────────────────────────────
    sacc_ids  = set(data['saccade_id'].tolist()) - {0}
    sacc_rate = len(sacc_ids) / dur_min

    return {
        'mean_pupil_mm':        mean_pupil,
        'blink_rate':           blink_rate,
        'fixation_rate':        fix_rate,
        'mean_fixation_dur_ms': mean_fix_dur_ms,
        'saccade_rate':         sacc_rate,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  Sliding-window time series
# ═══════════════════════════════════════════════════════════════════════════════

def compute_window_series(data):
    """
    Compute per-metric sliding-window time series.

    Window width  = SLIDING_WINDOW_S (s)
    Step between windows = SLIDE_STEP_S (s)

    Returns dict: {metric_name: (t_centers_array, values_array)}
    """
    empty = {m: (np.array([]), np.array([])) for m in METRICS}
    if data is None or data['n_frames'] < 2:
        return empty

    ts    = data['ts']
    t_max = ts[-1]
    if t_max < SLIDING_WINDOW_S / 2:
        return empty

    t_starts  = np.arange(0, t_max - SLIDING_WINDOW_S / 2, SLIDE_STEP_S)
    t_centers = t_starts + SLIDING_WINDOW_S / 2.0

    results = {m: [] for m in METRICS}

    for t_start in t_starts:
        t_end = t_start + SLIDING_WINDOW_S
        mask  = (ts >= t_start) & (ts < t_end)
        if mask.sum() < 2:
            for m in METRICS:
                results[m].append(np.nan)
            continue

        idx = np.where(mask)[0]
        sub = {
            'ts':           ts[mask] - ts[idx[0]],
            'pupil_mm':     data['pupil_mm'][mask].copy(),
            'quality':      data['quality'][mask].copy(),
            'blink_id':     data['blink_id'][mask].copy(),
            'fixation_id':  data['fixation_id'][mask].copy(),
            'saccade_id':   data['saccade_id'][mask].copy(),
            'duration_s':   float(ts[idx[-1]] - ts[idx[0]]),
            'n_frames':     int(mask.sum()),
        }
        sc = compute_scalar_metrics(sub)
        for m in METRICS:
            results[m].append(sc[m])

    return {m: (t_centers, np.array(results[m])) for m in METRICS}


# ═══════════════════════════════════════════════════════════════════════════════
#  Plotting helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _save(fig, path):
    """Save a figure as .png and .eps."""
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.savefig(path.replace('.png', '.eps'), format='eps', bbox_inches='tight')
    print(f"    Saved {os.path.basename(path)}")
    plt.close(fig)


def plot_scenario_timeseries(window_series, scalar_metrics, save_as, title=''):
    """
    5-panel time-series figure, one panel per metric.
    Sliding-window curve (blue) + overall mean dashed line (red).
    """
    fig, axes = plt.subplots(len(METRICS), 1,
                              figsize=(14, 3 * len(METRICS)), sharex=True)
    for ax, metric in zip(axes, METRICS):
        t_c, vals = window_series[metric]
        valid = ~np.isnan(vals)
        if valid.any():
            ax.plot(t_c[valid], vals[valid],
                    color=BAR_COLOR, linewidth=1.5, alpha=0.9,
                    label=f'window={SLIDING_WINDOW_S:.0f} s')
        mean_val = scalar_metrics.get(metric, np.nan)
        if not np.isnan(mean_val):
            ax.axhline(mean_val, color='tomato', linewidth=1.0,
                       linestyle='--', alpha=0.8,
                       label=f'mean = {mean_val:.2f}')
        ax.set_ylabel(METRIC_LABELS[metric], fontsize=9, fontweight='bold')
        ax.legend(fontsize=7, loc='upper right')
        ax.grid(axis='y', alpha=0.3, linestyle='--')

    axes[-1].set_xlabel('Time (s)', fontsize=10, fontweight='bold')
    if title:
        axes[0].set_title(title, fontsize=11, fontweight='bold')

    plt.tight_layout()
    _save(fig, save_as)


def plot_condition_comparison(metrics_by_cond, save_as, pid=''):
    """
    Per-participant: grouped bar chart of all 5 metrics × 4 conditions.
    """
    conds = [c for c in CONDITIONS if c in metrics_by_cond]
    if not conds:
        return

    x     = np.arange(len(METRICS))
    width = 0.8 / len(conds)

    fig, ax = plt.subplots(figsize=(14, 6))
    for i, cond in enumerate(conds):
        offsets = (i - (len(conds) - 1) / 2.0) * width
        vals    = [metrics_by_cond[cond].get(m, np.nan) for m in METRICS]
        bars    = ax.bar(x + offsets, vals, width,
                         label=cond,
                         color=COND_COLORS.get(cond, BAR_COLOR),
                         edgecolor='black', linewidth=1.0)
        for bar, val in zip(bars, vals):
            if not np.isnan(val):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height(),
                        f'{val:.2f}', ha='center', va='bottom',
                        fontsize=6, rotation=75)

    ax.set_xticks(x)
    ax.set_xticklabels([METRIC_LABELS[m] for m in METRICS], rotation=25, ha='right')
    ax.set_ylabel('Value', fontsize=11, fontweight='bold')
    ax.legend(title='Condition')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    title = f'{pid} — Eye Metrics by Condition' if pid else 'Eye Metrics by Condition'
    ax.set_title(title, fontsize=12, fontweight='bold')

    plt.tight_layout()
    _save(fig, save_as)


def plot_cross_participant_metric(metric, cond_data_by_pid, save_as):
    """
    Box plot across conditions for one metric (all participants pooled).
    """
    conds  = [c for c in CONDITIONS
              if any(c in v for v in cond_data_by_pid.values())]
    groups = [
        [cond_data_by_pid[pid][c]
         for pid in cond_data_by_pid
         if c in cond_data_by_pid[pid] and not np.isnan(cond_data_by_pid[pid][c])]
        for c in conds
    ]

    # Annotate with n
    labels = [f"{c}\n(n={len(g)})" for c, g in zip(conds, groups)]

    fig, ax = plt.subplots(figsize=(8, 5))
    bp = ax.boxplot(groups, labels=labels, patch_artist=True,
                    medianprops={'color': 'black', 'linewidth': 2})
    for patch, cond in zip(bp['boxes'], conds):
        patch.set_facecolor(COND_COLORS.get(cond, BAR_COLOR))
        patch.set_alpha(0.7)

    ax.set_xlabel('Condition', fontsize=11, fontweight='bold')
    ax.set_ylabel(METRIC_LABELS[metric], fontsize=11, fontweight='bold')
    ax.set_title(f'{METRIC_LABELS[metric]} — All Participants\n'
                 f'(quality threshold = {QUALITY_THRESHOLD})',
                 fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    plt.tight_layout()
    _save(fig, save_as)


def plot_procedure_breakdown(all_proc_data, save_as):
    """
    Cross-procedure breakdown: 5 metric subplots, one coloured line per
    condition, procedures on the x-axis.  Values are means across participants.

    all_proc_data : {pid: {cond: {proc_key: {metric: value}}}}
    """
    # Collect values per (cond, proc, metric)
    cond_proc_vals = {
        c: {p: {m: [] for m in METRICS} for p in PROC_ORDER}
        for c in CONDITIONS
    }
    for pid, cond_dict in all_proc_data.items():
        for cond, proc_dict in cond_dict.items():
            if cond not in cond_proc_vals:
                continue
            for proc_key, metrics in proc_dict.items():
                if proc_key not in cond_proc_vals[cond]:
                    continue
                for m in METRICS:
                    v = metrics.get(m, np.nan)
                    if not np.isnan(v):
                        cond_proc_vals[cond][proc_key][m].append(v)

    active_procs = [
        p for p in PROC_ORDER
        if any(cond_proc_vals[c][p][m]
               for c in CONDITIONS for m in METRICS)
    ]
    if not active_procs:
        return

    x   = np.arange(len(active_procs))
    fig, axes = plt.subplots(
        len(METRICS), 1,
        figsize=(max(12, len(active_procs) * 1.8), 3.2 * len(METRICS)),
        sharex=True,
    )

    for ax, metric in zip(axes, METRICS):
        for cond in CONDITIONS:
            means = np.array([
                np.mean(cond_proc_vals[cond][p][metric])
                if cond_proc_vals[cond][p][metric] else np.nan
                for p in active_procs
            ])
            valid = ~np.isnan(means)
            if valid.any():
                ax.plot(x[valid], means[valid], 'o-',
                        color=COND_COLORS[cond], label=cond,
                        linewidth=1.8, markersize=6, alpha=0.9)
        ax.set_ylabel(METRIC_LABELS[metric], fontsize=9, fontweight='bold')
        ax.legend(fontsize=7, loc='best', ncol=2)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        for xi in x:
            ax.axvline(xi, color='gray', linewidth=0.3, alpha=0.35, zorder=0)

    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels([PROC_LABELS[p] for p in active_procs], fontsize=9)
    axes[0].set_title(
        'Eye Metrics by Procedure — Mean Across Participants\n'
        '(each line = one condition; quality threshold = '
        f'{QUALITY_THRESHOLD})',
        fontsize=12, fontweight='bold',
    )

    plt.tight_layout()
    _save(fig, save_as)


# ═══════════════════════════════════════════════════════════════════════════════
#  Statistical tests
# ═══════════════════════════════════════════════════════════════════════════════

def run_statistics(cond_data_by_pid, metric):
    """
    Within-subject comparison across conditions for one metric.

    1. Friedman test on participants who have all 4 conditions.
    2. Pairwise Wilcoxon signed-rank tests (Bonferroni corrected) for every
       condition pair, using participants who have both conditions.

    cond_data_by_pid: {pid: {cond: float_value}}

    Returns dict with keys 'friedman' and 'pairwise'.
    """
    result = {'friedman': None, 'pairwise': {}}
    if not _SCIPY:
        return result

    # ── Friedman ──────────────────────────────────────────────────────────────
    complete_pids = [
        pid for pid in cond_data_by_pid
        if all(
            c in cond_data_by_pid[pid] and not np.isnan(cond_data_by_pid[pid][c])
            for c in CONDITIONS
        )
    ]
    if len(complete_pids) >= 4:
        groups = [
            [cond_data_by_pid[pid][c] for pid in complete_pids]
            for c in CONDITIONS
        ]
        try:
            F, p = sp_stats.friedmanchisquare(*groups)
            result['friedman'] = {
                'F': float(F), 'p': float(p), 'n': len(complete_pids)
            }
        except Exception:
            pass

    # ── Pairwise Wilcoxon (Bonferroni) ────────────────────────────────────────
    pairs   = [(CONDITIONS[i], CONDITIONS[j])
               for i in range(len(CONDITIONS))
               for j in range(i + 1, len(CONDITIONS))]
    n_pairs = len(pairs)

    for c1, c2 in pairs:
        shared = [
            pid for pid in cond_data_by_pid
            if c1 in cond_data_by_pid[pid]
            and c2 in cond_data_by_pid[pid]
            and not np.isnan(cond_data_by_pid[pid][c1])
            and not np.isnan(cond_data_by_pid[pid][c2])
        ]
        if len(shared) < 6:
            result['pairwise'][(c1, c2)] = {
                'n': len(shared), 'W': np.nan, 'p': np.nan, 'p_adj': np.nan
            }
            continue

        a = np.array([cond_data_by_pid[pid][c1] for pid in shared])
        b = np.array([cond_data_by_pid[pid][c2] for pid in shared])
        try:
            W, p     = sp_stats.wilcoxon(a, b)
            p_adj    = min(float(p) * n_pairs, 1.0)
            result['pairwise'][(c1, c2)] = {
                'n': len(shared), 'W': float(W),
                'p': float(p), 'p_adj': p_adj
            }
        except Exception:
            result['pairwise'][(c1, c2)] = {
                'n': len(shared), 'W': np.nan, 'p': np.nan, 'p_adj': np.nan
            }

    return result


# ═══════════════════════════════════════════════════════════════════════════════
#  Per-participant analysis
# ═══════════════════════════════════════════════════════════════════════════════

def analyse_participant(pid, participant_dir, verbose=True):
    """
    Process all (non-training) scenario ingescape CSVs for one participant.

    Generates a per-scenario time-series plot and a per-participant
    condition-comparison bar chart.

    Returns {cond: {metric: mean_value}} (averaged across scenarios per cond).
    """
    scenarios_dir = os.path.join(participant_dir, 'scenarios')
    cleaned_dir   = os.path.join(participant_dir, 'cleaned')
    os.makedirs(cleaned_dir, exist_ok=True)

    files = sorted(glob.glob(os.path.join(scenarios_dir, '*ingescape*.csv')))
    if not files:
        print(f"  No ingescape scenario CSVs found in {scenarios_dir}")
        return {}

    cond_scalars      = defaultdict(list)   # cond → list of scalar metric dicts
    cond_proc_scalars = defaultdict(list)   # cond → list of {proc_key: scalar_dict}

    for filepath in files:
        cond = extract_condition(filepath)
        if cond is None:
            continue

        fname = os.path.basename(filepath)
        if verbose:
            print(f"  ── {fname}")

        data = parse_eyemetrics(filepath)
        if data is None or data['n_frames'] < 10:
            if verbose:
                print(f"    → no eye-tracking data")
            continue

        scalar = compute_scalar_metrics(data)

        pct_valid = (
            float(np.sum(~np.isnan(data['pupil_mm']))) / data['n_frames'] * 100
        )
        if verbose:
            print(f"    {data['n_frames']} frames, {data['duration_s']:.1f} s, "
                  f"{pct_valid:.1f}% valid pupil")

        cond_scalars[cond].append(scalar)

        # Per-procedure metrics
        proc_windows = parse_procedure_windows(filepath)
        proc_metrics = compute_procedure_metrics(data, proc_windows)
        if proc_metrics:
            cond_proc_scalars[cond].append(proc_metrics)

        # Per-scenario time-series plot
        stem      = re.sub(r'_ingescape\.csv$', '', fname, flags=re.IGNORECASE)
        stem      = re.sub(r'\.csv$', '',  stem, flags=re.IGNORECASE)
        plot_path = os.path.join(cleaned_dir, f"{pid}_{stem}_eyemetrics.png")
        w_series  = compute_window_series(data)
        plot_scenario_timeseries(
            w_series, scalar, plot_path,
            title=f"{pid} – {cond} – {stem}  "
                  f"(window={SLIDING_WINDOW_S:.0f} s, step={SLIDE_STEP_S:.0f} s)"
        )

    if not cond_scalars:
        print(f"  No usable scenarios found for {pid}")
        return {}, {}

    # Average scalar metrics across scenarios within each condition
    cond_agg = {}
    for cond, scalar_list in cond_scalars.items():
        agg = {}
        for m in METRICS:
            vals = [s[m] for s in scalar_list if not np.isnan(s.get(m, np.nan))]
            agg[m] = float(np.mean(vals)) if vals else np.nan
        cond_agg[cond] = agg

    # Average procedure metrics across scenarios within each condition
    cond_proc_agg = {}
    for cond, proc_list in cond_proc_scalars.items():
        if not proc_list:
            continue
        proc_agg = {}
        for proc_key in PROC_ORDER:
            agg = {}
            for m in METRICS:
                vals = [
                    d[proc_key][m] for d in proc_list
                    if proc_key in d and not np.isnan(d[proc_key].get(m, np.nan))
                ]
                agg[m] = float(np.mean(vals)) if vals else np.nan
            proc_agg[proc_key] = agg
        cond_proc_agg[cond] = proc_agg

    # Per-participant condition-comparison bar chart
    plot_condition_comparison(
        cond_agg,
        save_as=os.path.join(cleaned_dir, f"{pid}_eyemetrics_by_condition.png"),
        pid=pid,
    )

    return cond_agg, cond_proc_agg


def run_participant(pid):
    """Full analysis for a single participant."""
    participant_dir = os.path.join(HITLS_DIR, pid)
    if not os.path.isdir(participant_dir):
        print(f"Participant directory not found: {participant_dir}")
        return

    print(f"\n{'='*60}")
    print(f"  {pid}")
    print(f"{'='*60}")

    cond_agg, _ = analyse_participant(pid, participant_dir, verbose=True)
    if not cond_agg:
        return

    print(f"\n  ── {pid} — Condition summary ──────────────────────────────")
    for cond in CONDITIONS:
        if cond not in cond_agg:
            continue
        print(f"  {cond}:")
        for m in METRICS:
            v = cond_agg[cond].get(m, np.nan)
            if not np.isnan(v):
                print(f"    {METRIC_LABELS[m]:<37} {v:.3f}")
            else:
                print(f"    {METRIC_LABELS[m]:<37} n/a")


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

    all_data      = {}   # pid → {cond → {metric → value}}
    all_proc_data = {}   # pid → {cond → {proc_key → {metric → value}}}

    for pid in all_pids:
        print(f"\n{'='*60}")
        print(f"  {pid}")
        print(f"{'='*60}")
        cond_agg, cond_proc_agg = analyse_participant(
            pid, os.path.join(HITLS_DIR, pid), verbose=False
        )
        if cond_agg:
            all_data[pid] = cond_agg
        if cond_proc_agg:
            all_proc_data[pid] = cond_proc_agg

    os.makedirs(CMP_DIR, exist_ok=True)

    # ── Per-metric cross-participant box plots ────────────────────────────────
    for metric in METRICS:
        cond_data_by_pid = {
            pid: {
                cond: all_data[pid][cond][metric]
                for cond in all_data[pid]
                if metric in all_data[pid][cond]
            }
            for pid in all_data
        }
        plot_cross_participant_metric(
            metric, cond_data_by_pid,
            save_as=os.path.join(CMP_DIR, f"eyemetrics_{metric}.png"),
        )

    # ── Cross-procedure breakdown chart ──────────────────────────────────────
    if all_proc_data:
        plot_procedure_breakdown(
            all_proc_data,
            save_as=os.path.join(CMP_DIR, 'eyemetrics_by_procedure.png'),
        )

    # ── Cross-participant text report ─────────────────────────────────────────
    report_path = os.path.join(CMP_DIR, 'eyemetrics_report.txt')
    with open(report_path, 'w', encoding='utf-8') as rpt:

        rpt.write("HITLS Eye-Tracking: Eye Metrics Analysis\n")
        rpt.write("=" * 80 + "\n")
        rpt.write(f"Quality threshold : {QUALITY_THRESHOLD}  "
                  f"(filtered_pupil_diameter_quality)\n")
        rpt.write(f"Sliding window    : {SLIDING_WINDOW_S} s  "
                  f"(step {SLIDE_STEP_S} s)\n")
        rpt.write("Pupil diameter    : filtered_pupil_diameter × 1000  (mm)\n")
        rpt.write("Blink/fixation/saccade: unique non-zero event IDs per minute\n")
        rpt.write("Mean fixation dur : mean(frame_count per fixation × IFI) in ms\n")
        rpt.write("Statistics        : Friedman test + pairwise Wilcoxon "
                  "(Bonferroni)\n")
        rpt.write("=" * 80 + "\n")

        for metric in METRICS:
            cond_data_by_pid = {
                pid: {
                    cond: all_data[pid][cond][metric]
                    for cond in all_data[pid]
                    if metric in all_data[pid][cond]
                }
                for pid in all_data
            }

            rpt.write(f"\n\n{'─'*80}\n")
            rpt.write(f"METRIC: {METRIC_LABELS[metric]}\n")
            rpt.write(f"{'─'*80}\n")

            # Descriptive table
            col_w = 16
            rpt.write(f"\n{'Participant':<12}")
            for cond in CONDITIONS:
                rpt.write(f"  {cond:^{col_w}}")
            rpt.write("\n")
            rpt.write(f"{'-'*12}" + f"  {'-'*col_w}" * len(CONDITIONS) + "\n")

            for pid in sorted(all_data):
                rpt.write(f"{pid:<12}")
                for cond in CONDITIONS:
                    v = cond_data_by_pid.get(pid, {}).get(cond, np.nan)
                    if not np.isnan(v):
                        rpt.write(f"  {v:>14.3f}  ")
                    else:
                        rpt.write(f"  {'—':^{col_w}}")
                rpt.write("\n")

            # MEAN ± SD row
            rpt.write(f"\n{'MEAN±SD':<12}")
            for cond in CONDITIONS:
                vals = [
                    cond_data_by_pid[pid][cond]
                    for pid in cond_data_by_pid
                    if cond in cond_data_by_pid[pid]
                    and not np.isnan(cond_data_by_pid[pid][cond])
                ]
                if vals:
                    rpt.write(f"  {np.mean(vals):>6.2f}±{np.std(vals):<7.2f}  ")
                else:
                    rpt.write(f"  {'—':^{col_w}}")
            rpt.write("\n")

            # Statistics
            stats = run_statistics(cond_data_by_pid, metric)

            if stats['friedman']:
                fr = stats['friedman']
                rpt.write(
                    f"\nFriedman test (n={fr['n']} with all 4 conditions): "
                    f"\u03c7\u00b2={fr['F']:.3f}  p={fr['p']:.4f}  "
                    f"{_sig_stars(fr['p'])}\n"
                )
            else:
                rpt.write("\nFriedman test: insufficient data\n")

            if stats['pairwise']:
                rpt.write(f"\nPairwise Wilcoxon signed-rank "
                          f"(Bonferroni, k={len(stats['pairwise'])} pairs):\n")
                rpt.write(
                    f"  {'Pair':<17}  {'n':>4}  {'W':>7}  "
                    f"{'p':>10}  {'p_adj':>10}  {'sig':>4}\n"
                )
                rpt.write(
                    f"  {'-'*17}  {'-'*4}  {'-'*7}  "
                    f"{'-'*10}  {'-'*10}  {'-'*4}\n"
                )
                for (c1, c2), pw in stats['pairwise'].items():
                    pair_str = f"{c1} vs {c2}"
                    w_str  = f"{pw['W']:.0f}" if not np.isnan(pw['W']) else 'n/a'
                    p_str  = f"{pw['p']:.4f}" if not np.isnan(pw['p']) else 'n/a'
                    pa_str = f"{pw['p_adj']:.4f}" if not np.isnan(pw['p_adj']) else 'n/a'
                    rpt.write(
                        f"  {pair_str:<17}  {pw['n']:>4}  {w_str:>7}  "
                        f"{p_str:>10}  {pa_str:>10}  "
                        f"{_sig_stars(pw['p_adj']):>4}\n"
                    )

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
            print("Usage: python3 eyemetrics.py [PXX | ALL]")
            sys.exit(1)
        return

    # ── Interactive menu ───────────────────────────────────────────────────────
    all_pids = _list_participants()
    print("\nHITLS Eye-Tracking Eye Metrics Analysis")
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
