#!/usr/bin/env python3
"""
camera_merge.py

Scans HITLS/P* participant folders for back_camera / back_cam and
front_camera / front_cam subdirectories.  For each camera folder it:
  1. Parses every clip's absolute start time from the hour-folder name
     and the video filename.
  2. Sorts clips chronologically and groups them into contiguous segments
     (gap ≤ GAP_TOLERANCE_S is treated as continuous; overlaps ≤ OVERLAP_TOL_S
      are also tolerated).
  3. Concatenates each multi-clip segment into one MP4 via ffmpeg's concat
     demuxer (stream-copy, no re-encoding).
     Gaps shorter than BLACK_FILL_MAX_S within a segment are filled with
     precisely-timed silent black frames so the output accurately reflects
     the real recording timeline.
  4. Places the merged file at the root of the camera folder.
  5. Writes a single merge.log per camera folder covering all segments.
  6. Prints a precise report of everything that happened.

Usage
-----
  python camera_merge.py
  python camera_merge.py --gap 30   # custom gap tolerance (s)
python camera_merge.py -p P13 P17
python camera_merge.py --participant P13 P17
python camera_merge.py -p P02          # single still works
python camera_merge.py                 # all participants
Filename conventions supported
  Subfolder  : YYYYYMMMDDDHH   e.g.  2026Y04M13D16H
  Back-cam   : W4{mm}M{ss}S{dur}.mp4   (W4 = camera prefix, ignored)
  Front-cam  :   {mm}M{ss}S{dur}.mp4
  where  mm = minute-in-hour (00-59),  ss = start-second,  dur = duration (s)

"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple

# ─── Configuration ─────────────────────────────────────────────────────────────
HITLS_DIR       = Path(__file__).parent
PARTICIPANT_RE  = re.compile(r'^P\d{2}$')
HOUR_FOLDER_RE  = re.compile(r'^(\d{4})Y(\d{2})M(\d{2})D(\d{2})H$')
CLIP_RE         = re.compile(r'^(?:W\d)?(\d{2})M(\d{2})S(\d+)\.mp4$', re.IGNORECASE)
CAMERA_NAMES    = ['back_camera', 'back_cam', 'front_camera', 'front_cam']

GAP_TOLERANCE_S   = 60.0  # clips with gap ≤ this are grouped into the same segment
OVERLAP_TOL_S     = 1.0   # overlaps up to this are tolerated (not a split)
BLACK_FILL_MAX_S  = 60.0  # intra-segment gaps < this are filled with black frames
CLOCK_TZ_OFFSET_H = -4   # hours to add to UTC for the wall-clock overlay on black filler


# ─── Data model ────────────────────────────────────────────────────────────────
@dataclass
class Clip:
    path:       Path
    abs_start:  float   # UTC epoch seconds
    duration_s: float   # from filename encoding


@dataclass
class SegmentResult:
    clips:       List[Clip]
    status:      str           # 'merged' | 'skipped' | 'single' | 'failed'
    output_path: str
    message:     str
    gap_before:  Optional[float] = None
    overlaps:    List[float] = field(default_factory=list)
    gaps:        List[float] = field(default_factory=list)   # intra-segment (pos=gap, neg=overlap)
    black_fills: List[Tuple[int, float]] = field(default_factory=list)
    props:       Optional[dict] = None
    dropped:     List['Clip'] = field(default_factory=list)  # spurious trailing clips removed


# ─── Parsing helpers ───────────────────────────────────────────────────────────
def hour_folder_epoch(name: str) -> Optional[float]:
    m = HOUR_FOLDER_RE.match(name)
    if not m:
        return None
    yyyy, mo, dd, hh = int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4))
    try:
        return datetime(yyyy, mo, dd, hh, 0, 0, tzinfo=timezone.utc).timestamp()
    except ValueError:
        return None


def parse_clip(filepath: Path, hour_epoch: float) -> Optional[Clip]:
    m = CLIP_RE.match(filepath.name)
    if not m:
        return None
    minute     = int(m.group(1))
    start_sec  = int(m.group(2))
    duration_s = float(m.group(3))
    return Clip(path=filepath,
                abs_start=hour_epoch + minute * 60.0 + start_sec,
                duration_s=duration_s)


def collect_clips(cam_dir: Path) -> List[Clip]:
    clips: List[Clip] = []
    for entry in sorted(cam_dir.iterdir()):
        if not entry.is_dir():
            continue
        epoch = hour_folder_epoch(entry.name)
        if epoch is None:
            continue
        for f in sorted(entry.iterdir()):
            if not f.is_file():
                continue
            clip = parse_clip(f, epoch)
            if clip is not None:
                clips.append(clip)
    clips.sort(key=lambda c: c.abs_start)
    return clips


# ─── Segment detection ─────────────────────────────────────────────────────────
def find_segments(
    clips: List[Clip],
    gap_tol: float = GAP_TOLERANCE_S,
    overlap_tol: float = OVERLAP_TOL_S,
) -> List[Tuple[List[Clip], List[float], List[Clip]]]:
    """Group sorted clips into contiguous segments.

    Returns [(clip_list, overlaps, dropped)] where `dropped` holds any
    spurious trailing clips that were removed because the next clip
    starts at the same (or earlier) time and supersedes them.
    """
    if not clips:
        return []
    segments: List[Tuple[List[Clip], List[float], List[Clip]]] = []
    current:  List[Clip]  = [clips[0]]
    overlaps: List[float] = []
    dropped:  List[Clip]  = []
    for clip in clips[1:]:
        prev = current[-1]
        gap  = clip.abs_start - (prev.abs_start + prev.duration_s)
        if -overlap_tol <= gap <= gap_tol:
            current.append(clip)
            if gap < -0.05:
                overlaps.append(-gap)
        elif gap < -overlap_tol and clip.abs_start <= prev.abs_start:
            # prev is a spurious restart stub: next clip starts at the same
            # (or earlier) time and fully supersedes it.  Drop prev and
            # continue the current segment rather than splitting.
            dropped.append(current.pop())
            current.append(clip)
        else:
            segments.append((current, overlaps, dropped))
            current, overlaps, dropped = [clip], [], []
    segments.append((current, overlaps, dropped))
    return segments


# ─── Video helpers ─────────────────────────────────────────────────────────────
def get_video_props(path: Path) -> Optional[dict]:
    """Return video/audio stream properties needed to generate black filler clips."""
    result = subprocess.run(
        ['ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_streams', str(path)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        return None
    try:
        streams = json.loads(result.stdout).get('streams', [])
    except (ValueError, json.JSONDecodeError):
        return None
    props: dict = {}
    for s in streams:
        if s.get('codec_type') == 'video' and 'width' not in props:
            props['width']   = s.get('width', 1920)
            props['height']  = s.get('height', 1080)
            props['fps']     = s.get('r_frame_rate', '30/1')
            props['pix_fmt'] = s.get('pix_fmt', 'yuv420p')
        elif s.get('codec_type') == 'audio' and 'sample_rate' not in props:
            props['sample_rate'] = str(s.get('sample_rate', '48000'))
            props['channels']    = int(s.get('channels', 2))
    return props if 'width' in props else None


def generate_black_clip(duration_s: float, props: dict, output_path: Path,
                        gap_start_epoch: Optional[float] = None) -> bool:
    """Generate a precisely-timed silent black clip matching the given stream props.
    If gap_start_epoch is provided, a ticking wall-clock overlay is burned in
    (UTC + CLOCK_TZ_OFFSET_H).  Falls back to plain black if drawtext fails."""
    w, h = props['width'], props['height']
    fps  = props['fps']
    # Normalise pix_fmt: lavfi outputs yuv420p; yuvj420p (full-range) can upset
    # the filter pipeline, so always encode the black clip as yuv420p.
    pix  = 'yuv420p'

    def _build_cmd(vf_arg: Optional[str] = None) -> List[str]:
        c = ['ffmpeg', '-y',
             '-f', 'lavfi', '-i', f'color=c=black:size={w}x{h}:rate={fps}']
        if 'sample_rate' in props:
            layout = 'stereo' if props.get('channels', 2) >= 2 else 'mono'
            c += ['-f', 'lavfi', '-i',
                  f'anullsrc=channel_layout={layout}:sample_rate={props["sample_rate"]}']
        c += ['-t', f'{duration_s:.6f}',
              '-c:v', 'libx264', '-preset', 'ultrafast', '-pix_fmt', pix]
        if vf_arg:
            c += ['-vf', vf_arg]
        if 'sample_rate' in props:
            c += ['-c:a', 'aac', '-shortest']
        c.append(str(output_path))
        return c

    vf_drawtext: Optional[str] = None
    if gap_start_epoch is not None:
        # UTC epoch of the gap start, shifted to the display timezone
        base     = int(gap_start_epoch) + CLOCK_TZ_OFFSET_H * 3600
        # date part is static for the gap (gaps < 60 s never cross midnight in practice)
        dt_gap   = datetime.fromtimestamp(base, tz=timezone.utc)
        date_str = dt_gap.strftime('%m/%d/%Y')
        fs       = max(24, h // 18)   # ~60 px at 1080p

        # Build a ticking HH:MM:SS from epoch + pts using drawtext eif() expressions.
        # Escaping rules (subprocess list, no shell):
        #   Inside single-quoted text: \: separates eif args / displays colons;
        #   \, escapes the comma separator in AVFilter option strings.
        hh = f'%{{eif\\:mod(floor(({base}+pts)/3600)\\,24)\\:d\\:2}}'
        mm = f'%{{eif\\:floor(mod({base}+pts\\,3600)/60)\\:d\\:2}}'
        ss = f'%{{eif\\:mod(floor({base}+pts)\\,60)\\:d\\:2}}'
        clock_text = f"'{date_str} {hh}\\:{mm}\\:{ss}'"
        vf_drawtext = (
            f"drawtext=text={clock_text}:x=10:y=10:fontsize={fs}"
            f":fontcolor=white:box=1:boxcolor=black@0.5:boxborderw=5"
        )

    # Attempt with clock overlay first; fall back to plain black if it fails.
    attempts = [vf_drawtext, None] if vf_drawtext else [None]
    for vf in attempts:
        result = subprocess.run(_build_cmd(vf), capture_output=True, text=True, timeout=60)
        if result.returncode == 0:
            return True
        if output_path.exists():
            output_path.unlink()
        if vf is not None:
            print('          ⚠ clock overlay failed, retrying plain black …', flush=True)
    return False


# ─── Log writer ────────────────────────────────────────────────────────────────
def write_camera_log(
    cam_dir: Path,
    participant: str,
    all_clips: List[Clip],
    results: List[SegmentResult],
    gap_tol: float = GAP_TOLERANCE_S,
) -> None:
    """Write a single merge.log in cam_dir covering all segments for this camera folder."""
    log_path = cam_dir / 'merge.log'
    cam_name = cam_dir.name
    now_str  = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
    SEP      = '─' * 72

    merged_count  = sum(1 for r in results if r.status == 'merged')
    skipped_count = sum(1 for r in results if r.status == 'skipped')
    single_count  = sum(1 for r in results if r.status == 'single')
    failed_count  = sum(1 for r in results if r.status == 'failed')
    dropped_total = sum(len(r.dropped) for r in results)

    with open(log_path, 'w', encoding='utf-8') as f:
        f.write(f'Camera Merge Log  —  {participant} / {cam_name}\n')
        f.write(f'  Generated     : {now_str}\n')
        f.write(f'  Gap tolerance : {gap_tol:.1f} s  |  Black fill < {BLACK_FILL_MAX_S:.0f} s\n')
        f.write(f'  Total clips   : {len(all_clips)}\n')
        f.write(f'  Segments      : {len(results)}'
                f'  (merged: {merged_count}, skipped: {skipped_count},'
                f' single: {single_count}, failed: {failed_count})\n')
        if dropped_total:
            f.write(f'  Dropped clips : {dropped_total}'
                    f'  (spurious trailing restart stub(s) removed)\n')
        f.write('\n')

        for seg_i, res in enumerate(results, 1):
            clips     = res.clips
            n         = len(clips)
            t0_ep     = clips[0].abs_start
            t1_ep     = clips[-1].abs_start + clips[-1].duration_s
            t0        = datetime.fromtimestamp(t0_ep, tz=timezone.utc).strftime('%H:%M:%S')
            t1        = datetime.fromtimestamp(t1_ep, tz=timezone.utc).strftime('%H:%M:%S')
            span_s    = t1_ep - t0_ep
            content_s = sum(c.duration_s for c in clips)
            black_s   = sum(d for _, d in res.black_fills)
            filled    = {idx: dur for idx, dur in res.black_fills}
            out_name  = Path(res.output_path).name

            f.write(f'{SEP}\n')
            f.write(f'  Segment {seg_i} of {len(results)}   [{res.status.upper()}]')
            if res.status in ('merged', 'skipped', 'single'):
                f.write(f'   →  {out_name}')
            f.write('\n')
            if res.gap_before is not None:
                f.write(f'  Gap from previous segment : {res.gap_before:.1f} s\n')
            f.write(f'  Wall-clock span : {t0} → {t1}  ({span_s:.1f} s)\n')
            f.write(f'  Content footage : {content_s:.1f} s\n')
            if black_s:
                f.write(f'  Black filler    : {black_s:.2f} s ({len(filled)} gap(s))\n')
            if res.props:
                f.write(f'  Video props     : {res.props["width"]}x{res.props["height"]} '
                        f'{res.props.get("fps","?")} fps  {res.props.get("pix_fmt","?")}\n')
                if 'sample_rate' in res.props:
                    f.write(f'  Audio props     : {res.props["sample_rate"]} Hz  '
                            f'{res.props.get("channels","?")}ch\n')
            if res.status in ('merged', 'skipped') and Path(res.output_path).exists():
                size_mb = Path(res.output_path).stat().st_size / 1_048_576
                f.write(f'  File size       : {size_mb:.1f} MB\n')
            if res.overlaps:
                f.write(f'  Intra-overlaps  : {chr(44).join(f"{o:.2f} s" for o in res.overlaps)}\n')
            if res.dropped:
                f.write(f'  Dropped clips   : {len(res.dropped)}'
                        f'  (superseded by next clip — same start time)\n')
                for dc in res.dropped:
                    dct0 = datetime.fromtimestamp(dc.abs_start, tz=timezone.utc).strftime('%H:%M:%S')
                    dct1 = datetime.fromtimestamp(dc.abs_start + dc.duration_s,
                                                   tz=timezone.utc).strftime('%H:%M:%S')
                    f.write(f'    [drop] {dc.path.name:<55} {dct0} → {dct1}  ({dc.duration_s:.1f} s)\n')
            if res.message and res.status == 'failed':
                f.write(f'  Error           : {res.message}\n')
            f.write('\n')
            f.write(f'  Input clips ({n})\n')
            for idx, clip in enumerate(clips):
                ct0 = datetime.fromtimestamp(clip.abs_start, tz=timezone.utc).strftime('%H:%M:%S')
                ct1 = datetime.fromtimestamp(clip.abs_start + clip.duration_s,
                                              tz=timezone.utc).strftime('%H:%M:%S')
                f.write(f'    [{idx+1:>3}] {clip.path.name:<55} {ct0} → {ct1}  ({clip.duration_s:.1f} s)\n')
                if idx < len(res.gaps):
                    g = res.gaps[idx]
                    if g > 0.05:
                        if idx in filled:
                            f.write(f'           ↕ gap: {g:.2f} s  → filled with {filled[idx]:.2f} s black frames\n')
                        else:
                            f.write(f'           ↕ gap: {g:.2f} s\n')
                    elif g < -0.05:
                        f.write(f'           ↕ overlap: {-g:.2f} s\n')
            f.write('\n')


# ─── Segment merge ─────────────────────────────────────────────────────────────
def merge_segment(clips: List[Clip], cam_dir: Path,
                  gap_tol: float = GAP_TOLERANCE_S) -> SegmentResult:
    """
    Merge a group of clips into one MP4.
    Intra-segment gaps < BLACK_FILL_MAX_S are filled with silent black frames.
    """
    n        = len(clips)
    dur_s    = clips[-1].abs_start + clips[-1].duration_s - clips[0].abs_start
    start_dt = datetime.fromtimestamp(clips[0].abs_start, tz=timezone.utc)
    ts_str   = start_dt.strftime('%Y%m%d_%H%M%S')

    if n == 1:
        return SegmentResult(
            clips=clips, status='single',
            output_path=str(clips[0].path),
            message='single clip – no merge needed',
        )

    out_name = f"merged_{ts_str}_{n}clips.mp4"
    out_path = cam_dir / out_name

    # Per-gap values (positive = gap, negative = overlap) — computed early for the log
    gaps = [
        clips[i].abs_start - (clips[i-1].abs_start + clips[i-1].duration_s)
        for i in range(1, n)
    ]

    if out_path.exists():
        size_mb = out_path.stat().st_size / 1_048_576
        return SegmentResult(
            clips=clips, status='skipped',
            output_path=str(out_path),
            message=f"output already exists – skipped  ({size_mb:.1f} MB)",
            gaps=gaps,
        )

    needs_fill = any(0 < g < BLACK_FILL_MAX_S for g in gaps)

    props: Optional[dict] = None
    if needs_fill:
        props = get_video_props(clips[0].path)
        if props is None:
            print('          ⚠ could not read video props – gaps will not be filled',
                  flush=True)

    tmp_black_clips: List[Path] = []
    black_fills: List[Tuple[int, float]] = []
    try:
        ordered: List[Path] = [clips[0].path]
        for i, gap in enumerate(gaps):
            if props is not None and 0 < gap < BLACK_FILL_MAX_S:
                black_path = cam_dir / f'_black_tmp_{ts_str}_{i}.mp4'
                print(f'        generating {gap:.2f}s black filler …', flush=True)
                if generate_black_clip(gap, props, black_path,
                                       gap_start_epoch=clips[i].abs_start + clips[i].duration_s):
                    ordered.append(black_path)
                    tmp_black_clips.append(black_path)
                    black_fills.append((i, gap))
                else:
                    print('          ⚠ black clip generation failed – gap left unfilled',
                          flush=True)
            ordered.append(clips[i + 1].path)

        tmp_fd, tmp_path = tempfile.mkstemp(suffix='.txt', prefix='cam_concat_')
        try:
            with os.fdopen(tmp_fd, 'w') as f:
                for p in ordered:
                    escaped = str(p.resolve()).replace('\\', '\\\\').replace("'", r"'\''")
                    f.write(f"file '{escaped}'\n")

            filler_note = f', {len(tmp_black_clips)} gap(s) filled' if tmp_black_clips else ''
            print(f'        ffmpeg → {out_name}  ({n} clips, {dur_s:.0f} s{filler_note}) …',
                  flush=True)

            result = subprocess.run(
                ['ffmpeg', '-y', '-f', 'concat', '-safe', '0',
                 '-i', tmp_path, '-c', 'copy', str(out_path)],
                capture_output=True, text=True, timeout=7200,
            )
        except subprocess.TimeoutExpired:
            if out_path.exists():
                out_path.unlink()
            return SegmentResult(
                clips=clips, status='failed',
                output_path=str(out_path),
                message='ffmpeg timed out (> 2 h)',
            )
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

        if result.returncode != 0:
            if out_path.exists():
                out_path.unlink()
            stderr_tail = result.stderr.strip().splitlines()
            err_lines   = '\n'.join(stderr_tail[-8:]) if stderr_tail else '(no stderr)'
            return SegmentResult(
                clips=clips, status='failed',
                output_path=str(out_path),
                message=f"ffmpeg exited {result.returncode}:\n{err_lines}",
            )

        size_mb     = out_path.stat().st_size / 1_048_576
        filler_note = f', {len(tmp_black_clips)} gap(s) filled with black' \
                      if tmp_black_clips else ''
        return SegmentResult(
            clips=clips, status='merged',
            output_path=str(out_path),
            message=f"→ {out_name}  ({size_mb:.1f} MB{filler_note})",
            gaps=gaps, black_fills=black_fills, props=props,
        )

    finally:
        for p in tmp_black_clips:
            try:
                p.unlink()
            except OSError:
                pass


# ─── Camera-folder processor ───────────────────────────────────────────────────
def process_camera(participant: str, cam_dir: Path,
                   gap_tol: float = GAP_TOLERANCE_S) -> dict:
    clips    = collect_clips(cam_dir)
    seg_data = find_segments(clips, gap_tol=gap_tol)
    results: List[SegmentResult] = []

    for i, (seg_clips, overlaps, dropped) in enumerate(seg_data):
        gap_before: Optional[float] = None
        if i > 0:
            prev_clips = seg_data[i - 1][0]
            prev_end   = prev_clips[-1].abs_start + prev_clips[-1].duration_s
            gap_before = seg_clips[0].abs_start - prev_end

        res = merge_segment(seg_clips, cam_dir, gap_tol=gap_tol)
        res.gap_before = gap_before
        res.overlaps   = overlaps
        res.dropped    = dropped
        results.append(res)

    write_camera_log(cam_dir, participant, all_clips=clips,
                    results=results, gap_tol=gap_tol)
    return {
        'participant': participant,
        'camera':      cam_dir.name,
        'clips_total': len(clips),
        'results':     results,
    }


# ─── Entry point ───────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description='Merge camera clips into timed segments with black-frame gap fill.')
    parser.add_argument(
        '--gap', metavar='GAP_S', type=float, default=GAP_TOLERANCE_S,
        help=f'Max gap (s) between clips to still group into one segment '
             f'(default: {GAP_TOLERANCE_S})')
    parser.add_argument(
        '--participant', '-p', metavar='PXX', nargs='+',
        help='Process only these participant(s) (e.g. -p P13 P17). Omit to process all.')
    args    = parser.parse_args()
    gap_tol = args.gap
    only_p  = {v.upper() for v in args.participant} if args.participant else None

    for tool in ('ffmpeg', 'ffprobe'):
        if subprocess.run([tool, '-version'], capture_output=True).returncode != 0:
            sys.exit(f"ERROR: {tool} not found in PATH – aborting.")

    participants = sorted(
        [d for d in HITLS_DIR.iterdir()
         if d.is_dir() and PARTICIPANT_RE.match(d.name)
         and (only_p is None or d.name in only_p)],

        key=lambda p: p.name,
    )
    if not participants:
        if only_p:
            print(f"Participant directory/ies {sorted(only_p)} not found in {HITLS_DIR}.")
        else:
            print("No participant directories (P02, P03 …) found.")
        return

    print(f"Gap tolerance : {gap_tol:.1f} s  |  Black fill < {BLACK_FILL_MAX_S:.0f} s")
    print(f"Participants  : {', '.join(p.name for p in participants)}\n")

    all_reports: List[dict] = []
    for pdir in participants:
        for cam_name in CAMERA_NAMES:
            cam_dir = pdir / cam_name
            if not cam_dir.is_dir():
                continue
            print(f"  Processing {pdir.name}/{cam_name} …", flush=True)
            report = process_camera(pdir.name, cam_dir, gap_tol=gap_tol)
            all_reports.append(report)
            print(f"    {report['clips_total']} clip(s)  →  "
                  f"{len(report['results'])} segment(s)", flush=True)

    # ── Final report ──────────────────────────────────────────────────────────
    SEP  = '═' * 76
    SEP2 = '─' * 76
    ICONS = {'merged': '✓', 'skipped': '↷', 'single': '·', 'failed': '✗'}

    print(f"\n{SEP}")
    print(f"  CAMERA MERGE REPORT")
    print(SEP)

    totals = {k: 0 for k in ICONS}

    for r in all_reports:
        results: List[SegmentResult] = r['results']
        print(f"\n  ▶  {r['participant']}/{r['camera']}"
              f"   ({r['clips_total']} clips, {len(results)} segment(s))")

        for res in results:
            n     = len(res.clips)
            t0    = datetime.fromtimestamp(res.clips[0].abs_start,
                                            tz=timezone.utc).strftime('%H:%M:%S')
            t1_ep = res.clips[-1].abs_start + res.clips[-1].duration_s
            t1    = datetime.fromtimestamp(t1_ep, tz=timezone.utc).strftime('%H:%M:%S')
            ic    = ICONS.get(res.status, '?')
            print(f"     [{ic}] {t0} → {t1}  ({n} clip(s), {t1_ep - res.clips[0].abs_start:.0f} s)")
            for line in res.message.splitlines():
                print(f"          {line}")
            if res.overlaps:
                print(f"          ⚠ intra-segment overlap(s): "
                      f"{', '.join(f'{o:.2f} s' for o in res.overlaps)}")
            if res.dropped:
                names = ', '.join(d.path.name for d in res.dropped)
                print(f"          ✂ dropped {len(res.dropped)} spurious trailing clip(s): {names}")
            if res.gap_before is not None:
                print(f"          ↕ gap from previous segment: {res.gap_before:.1f} s")
            totals[res.status] = totals.get(res.status, 0) + 1

    dropped_total = sum(len(res.dropped) for r in all_reports
                        for res in r['results'])
    print(f"\n{SEP2}")
    print(f"  Segments merged (new):   {totals['merged']}")
    print(f"  Segments skipped:        {totals['skipped']}  (output already existed)")
    print(f"  Single-clip segments:    {totals['single']}  (no merge needed)")
    print(f"  Failed:                  {totals['failed']}")
    if dropped_total:
        print(f"  Spurious clips dropped:  {dropped_total}")
    print(SEP)


if __name__ == '__main__':
    main()
