#!/usr/bin/env python3
"""
video_extract.py

For each participant folder (P02, P03, …) under HITLS/:
  1. Reads each scenarios/scenario_*_ingescape.csv.
     – Files with "unfinished" anywhere in the name are skipped.
  2. Takes the first data row timestamp as the scenario start,
     the last data row timestamp as the scenario end.
     Ingescape timestamps are stored in UTC+8; subtracting 8 h gives true UTC.
  3. Finds back_camera/ (or back_cam/) merged_*.mp4 files.
     The filename encodes the start time in UTC+4; subtracting 4 h gives UTC.
  4. Picks the merged video whose time range covers the scenario start.
  5. Extracts the segment with ffmpeg (stream-copy, no re-encode).
  6. Saves as  scenarios/scenario_NN_CODENAME_back.mp4

Usage
-----
  python video_extract.py                    # all participants
  python video_extract.py P02 P05           # specific participants
  python video_extract.py --dry-run         # show offsets without extracting
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

# ─── Configuration ─────────────────────────────────────────────────────────────
HITLS_DIR       = Path(__file__).parent
PARTICIPANT_RE  = re.compile(r'^P\d{2}$')
SCENARIO_RE     = re.compile(r'^scenario_(\d+)_(.+)_ingescape\.csv$', re.IGNORECASE)
MERGED_RE       = re.compile(r'^merged_(\d{8})_(\d{6})_', re.IGNORECASE)
CAMERA_DIRS     = ['back_camera', 'back_cam']

# Ingescape records timestamps in UTC+8 → subtract to reach UTC.
INGESCAPE_OFFSET_S: float = 8 * 3600
# Camera filenames encode time in UTC+4 → subtract to reach UTC.
CAMERA_OFFSET_S:    float = 4 * 3600

# Tolerance when checking whether a scenario falls within a video (seconds).
MATCH_TOLERANCE_S: float = 30.0


# ─── CSV parsing ───────────────────────────────────────────────────────────────
def parse_scenario_csv(path: Path) -> tuple[float, float] | tuple[None, str]:
    """
    Return (start_utc, end_utc) in true UTC epoch seconds, or (None, reason).

    Two formats are supported:

    Old format  – column 1 header: ``timestamp``
        Values are Unix epoch seconds in UTC+8 (INGESCAPE_OFFSET_S applied).

    New format  – column 1 header: ``relative_time_us``
        Values are microseconds from platform start.  A ``TARS Agent /
        session_epoch`` row must be present to anchor relative time to wall
        clock.  The session_epoch value is the UTC+4 wall-clock epoch at the
        moment that row was written (CAMERA_OFFSET_S applied).
    """
    with path.open(newline='', encoding='utf-8', errors='replace') as fh:
        reader = csv.reader(fh, delimiter=';')
        header = next(reader, None)
        if header is None:
            return None, 'empty file'

        col1_name = header[1].strip() if len(header) > 1 else ''

        if col1_name == 'relative_time_us':
            # ── New format ────────────────────────────────────────────────────
            ses_epoch_val: float | None = None
            ses_epoch_rel: float | None = None
            first_rel: float | None = None
            last_rel:  float | None = None

            for row in reader:
                if len(row) < 2:
                    continue
                try:
                    rel = float(row[1])
                except ValueError:
                    continue

                # Detect the session_epoch anchor row
                if (len(row) >= 7
                        and row[2].strip() == 'TARS Agent'
                        and row[3].strip() == 'session_epoch'):
                    try:
                        ses_epoch_val = float(row[6])
                        ses_epoch_rel = rel
                    except (ValueError, IndexError):
                        pass
                    continue

                if first_rel is None:
                    first_rel = rel
                last_rel = rel

            if first_rel is None or last_rel is None:
                return None, 'no data rows found'
            if ses_epoch_val is None:
                return None, 'relative_time_us format but no session_epoch row found'

            # session_epoch is emitted in UTC+4; subtract CAMERA_OFFSET_S to reach UTC.
            base_utc = ses_epoch_val - CAMERA_OFFSET_S - ses_epoch_rel / 1e6
            return (
                base_utc + first_rel / 1e6,
                base_utc + last_rel  / 1e6,
            )

        else:
            # ── Old format (timestamp = UTC+8 epoch seconds) ──────────────────
            start_ts: float | None = None
            end_ts:   float | None = None

            for row in reader:
                if len(row) < 2:
                    continue
                try:
                    ts = float(row[1])
                except ValueError:
                    continue
                if start_ts is None:
                    start_ts = ts
                end_ts = ts

            if start_ts is None or end_ts is None:
                return None, 'no data rows found'

            return (
                start_ts - INGESCAPE_OFFSET_S,
                end_ts   - INGESCAPE_OFFSET_S,
            )


# ─── Video helpers ─────────────────────────────────────────────────────────────
def merged_start_utc(path: Path) -> float | None:
    """Parse true UTC start epoch from merged_YYYYMMDD_HHMMSS_*.mp4."""
    m = MERGED_RE.match(path.name)
    if not m:
        return None
    d, t = m.group(1), m.group(2)
    try:
        dt = datetime(
            int(d[:4]), int(d[4:6]), int(d[6:8]),
            int(t[:2]),  int(t[2:4]),  int(t[4:6]),
            tzinfo=timezone.utc,
        )
    except ValueError:
        return None
    return dt.timestamp() - CAMERA_OFFSET_S


def video_duration(path: Path) -> float | None:
    """Return duration in seconds via ffprobe, or None on failure."""
    try:
        result = subprocess.run(
            [
                'ffprobe', '-v', 'error',
                '-show_entries', 'format=duration',
                '-of', 'json',
                str(path),
            ],
            capture_output=True, text=True, timeout=60,
        )
        return float(json.loads(result.stdout)['format']['duration'])
    except Exception:
        return None


def load_merged_videos(cam_dir: Path) -> list[tuple[float, float, Path]]:
    """
    Return [(start_utc, end_utc, path), …] for every merged_*.mp4 in cam_dir,
    sorted by start time.
    """
    videos: list[tuple[float, float, Path]] = []
    for mp4 in sorted(cam_dir.glob('merged_*.mp4')):
        start = merged_start_utc(mp4)
        if start is None:
            print(f'    [WARN] Cannot parse start time from {mp4.name}; skipping')
            continue
        dur = video_duration(mp4)
        if dur is None:
            print(f'    [WARN] ffprobe failed for {mp4.name}; skipping')
            continue
        videos.append((start, start + dur, mp4))
    return sorted(videos, key=lambda x: x[0])


def extract_segment(
    src: Path,
    start_offset: float,
    duration: float,
    out: Path,
) -> bool:
    """Extract [start_offset … start_offset+duration] from src into out (stream-copy)."""
    out.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [
            'ffmpeg', '-y',
            '-ss', f'{start_offset:.3f}',
            '-i', str(src),
            '-t', f'{duration:.3f}',
            '-c', 'copy',
            str(out),
        ],
        capture_output=True, text=True, timeout=600,
    )
    return result.returncode == 0


# ─── Per-participant logic ──────────────────────────────────────────────────────
def process_participant(part_dir: Path, dry_run: bool) -> None:
    tag = part_dir.name
    print(f'\n{"=" * 62}')
    print(f'  {tag}')
    print(f'{"=" * 62}')

    scenarios_dir = part_dir / 'scenarios'
    if not scenarios_dir.is_dir():
        print('  [SKIP] No scenarios/ directory')
        return

    # Locate camera directory
    cam_dir = next((part_dir / n for n in CAMERA_DIRS if (part_dir / n).is_dir()), None)
    if cam_dir is None:
        print('  [SKIP] No back_camera / back_cam directory found')
        return
    print(f'  Camera dir : {cam_dir.name}')

    # Load merged video index
    merged = load_merged_videos(cam_dir)
    if not merged:
        print('  [SKIP] No usable merged_*.mp4 found')
        return

    for v_start, v_end, v_path in merged:
        print(
            f'  Video : {v_path.name}'
            f'  UTC [{_hms(v_start)} → {_hms(v_end)}'
            f'  {(v_end - v_start) / 60:.0f} min]'
        )

    # Process scenario CSVs
    csv_files = sorted(scenarios_dir.glob('scenario_*_ingescape.csv'))
    if not csv_files:
        print('  [SKIP] No scenario_*_ingescape.csv files')
        return

    for csv_path in csv_files:
        # Skip files that mention "unfinished" anywhere in the name
        if 'unfinished' in csv_path.name.lower():
            print(f'\n  [SKIP] {csv_path.name}  (marked unfinished)')
            continue

        m = SCENARIO_RE.match(csv_path.name)
        if not m:
            continue

        num  = m.group(1)   # e.g. "03"
        name = m.group(2)   # e.g. "BASELINE"

        print(f'\n  ── scenario_{num}_{name} ──')

        times = parse_scenario_csv(csv_path)
        if times[0] is None:
            print(f'    [FAIL] Could not parse timestamps: {times[1]}')
            continue

        s_start, s_end = times
        duration = s_end - s_start

        print(f'    Start UTC : {_ymdhms(s_start)}  ({s_start:.3f})')
        print(f'    End   UTC : {_ymdhms(s_end)}  ({s_end:.3f})')
        print(f'    Duration  : {duration:.1f} s  ({duration / 60:.1f} min)')

        # Find which merged video covers this scenario
        matched = next(
            (
                (vs, ve, vp)
                for vs, ve, vp in merged
                if s_start >= vs - MATCH_TOLERANCE_S and s_start < ve + MATCH_TOLERANCE_S
            ),
            None,
        )

        if matched is None:
            print(f'    [FAIL] Scenario start ({_hms(s_start)} UTC) not covered by any merged video')
            for vs, ve, vp in merged:
                print(f'           {vp.name}: {_hms(vs)} → {_hms(ve)} UTC')
            continue

        vs, ve, vp = matched
        offset = s_start - vs

        # Clamp to video bounds
        if offset < 0:
            print(f'    [WARN] Scenario starts {-offset:.1f} s before video; clamping offset to 0')
            duration += offset
            offset = 0.0

        available = ve - (vs + offset)
        if duration > available + 5:
            print(f'    [WARN] Scenario extends {duration - available:.1f} s past video end; clamping duration')
            duration = max(0.0, available)

        print(f'    Video     : {vp.name}')
        print(f'    Offset    : {offset:.1f} s  →  extract {duration:.1f} s')

        out_name = f'scenario_{num}_{name}_back.mp4'
        out_path = scenarios_dir / out_name

        if out_path.exists():
            print(f'    [SKIP] Already exists: {out_name}')
            continue

        if dry_run:
            print(f'    [DRY-RUN] Would write: {out_name}')
            continue

        print(f'    Extracting → {out_name} …')
        ok = extract_segment(vp, offset, duration, out_path)
        if ok:
            size_mb = out_path.stat().st_size / 1e6
            print(f'    [OK]  {out_name}  ({size_mb:.1f} MB)')
        else:
            print(f'    [FAIL] ffmpeg returned a non-zero exit code')


# ─── Formatting helpers ────────────────────────────────────────────────────────
def _hms(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, tz=timezone.utc).strftime('%H:%M:%S')


def _ymdhms(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S')


# ─── Entry point ───────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('participants', nargs='*', help='Participant IDs to process (e.g. P02 P05). Default: all.')
    parser.add_argument('--dry-run', action='store_true', help='Print what would be done without running ffmpeg.')
    args = parser.parse_args()

    if args.participants:
        requested = {p.upper() for p in args.participants}
        part_dirs = sorted(
            d for d in HITLS_DIR.iterdir()
            if d.is_dir() and PARTICIPANT_RE.match(d.name) and d.name in requested
        )
        missing = requested - {d.name for d in part_dirs}
        if missing:
            print(f'[WARN] Not found: {sorted(missing)}')
    else:
        part_dirs = sorted(d for d in HITLS_DIR.iterdir() if d.is_dir() and PARTICIPANT_RE.match(d.name))

    if not part_dirs:
        print(f'No participant folders found under {HITLS_DIR}')
        sys.exit(1)

    print(f'Participants : {[d.name for d in part_dirs]}')
    if args.dry_run:
        print('Mode         : DRY-RUN (no files will be written)')

    for d in part_dirs:
        process_participant(d, dry_run=args.dry_run)

    print('\nDone.')


if __name__ == '__main__':
    main()
