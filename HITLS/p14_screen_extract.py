#!/usr/bin/env python3
"""
p14_screen_extract.py
Extract per-scenario clips from P14's four screen-recording videos.

Each screen recording has a manually identified anchor:
  eye_tracking_video.mp4   : 12:15  → scenario_02 start
  simu_video.mp4           :  2:20  → first controlFlaps = 0.5 event in scenario_02
  ingescape_video.mp4      : 12:17  → first controlFlaps = 0.5 event in scenario_02
  TARS_interface_video.mp4 :  6:32  → first controlFlaps = 0.5 event in scenario_02

The shared session base_utc is derived from the session_epoch anchor row present
in scenarios 02, 05, 10.  scenario_08 uses the same base_utc.

Usage:
    python p14_screen_extract.py [--dry-run]
"""

import csv
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
HITLS_DIR     = Path(__file__).parent
P14_DIR       = HITLS_DIR / "P14"
SCENARIOS_DIR = P14_DIR / "scenarios"

# ── Constants ─────────────────────────────────────────────────────────────────
CAMERA_OFFSET_S = 4 * 3600   # camera clock is UTC+4

SCENARIO_RE = re.compile(
    r"^(scenario_\d+_[^_]+(?:_[^_]+)*?)_ingescape\.csv$", re.IGNORECASE
)

# ── Screen recordings and their manual anchors ────────────────────────────────
# (filename, output suffix, anchor_seconds_into_video, anchor_event)
# anchor_event: 'scenario_start' = scenario_02 first data row UTC
#               'flaps_05'       = first Aircraft;controlFlaps=0.5 UTC in scenario_02
SCREEN_VIDEOS = [
    ("eye_tracking_video.mp4",   "eye",  12 * 60 + 15, "scenario_start"),
    ("simu_video.mp4",           "simu",  2 * 60 + 20, "flaps_05"),
    ("ingescape_video.mp4",      "inge", 12 * 60 + 17, "flaps_05"),
    ("TARS_interface_video.mp4", "tars",  6 * 60 + 32, "flaps_05"),
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _utc_str(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%H:%M:%S")


def get_base_utc(csv_path: Path) -> float | None:
    """Return base_utc (UTC when relative_time_us = 0) from a session_epoch row."""
    with csv_path.open(newline="", encoding="utf-8", errors="replace") as fh:
        reader = csv.reader(fh, delimiter=";")
        next(reader, None)
        for row in reader:
            if (
                len(row) >= 7
                and row[2].strip() == "TARS Agent"
                and row[3].strip() == "session_epoch"
            ):
                try:
                    epoch_val = float(row[6])
                    epoch_rel = float(row[1])
                    return epoch_val - CAMERA_OFFSET_S - epoch_rel / 1e6
                except ValueError:
                    pass
    return None


def parse_scenario_times(
    csv_path: Path, base_utc: float
) -> tuple[float, float] | tuple[None, None]:
    """Return (start_utc, end_utc) by scanning first and last data rows."""
    first_rel: float | None = None
    last_rel:  float | None = None
    with csv_path.open(newline="", encoding="utf-8", errors="replace") as fh:
        reader = csv.reader(fh, delimiter=";")
        next(reader, None)
        for row in reader:
            if len(row) < 2:
                continue
            # Skip anchor rows
            if len(row) >= 4 and row[2].strip() == "TARS Agent" and row[3].strip() == "session_epoch":
                continue
            # Skip SNAPSHOT rows
            if len(row) >= 3 and row[2].strip() == "SNAPSHOT":
                continue
            try:
                rel = float(row[1])
            except ValueError:
                continue
            if first_rel is None:
                first_rel = rel
            last_rel = rel
    if first_rel is None:
        return None, None
    return base_utc + first_rel / 1e6, base_utc + last_rel / 1e6


def find_flaps_05_utc(csv_path: Path, base_utc: float) -> float | None:
    """Return UTC of first Aircraft;controlFlaps row with value starting with '0.5'."""
    with csv_path.open(newline="", encoding="utf-8", errors="replace") as fh:
        reader = csv.reader(fh, delimiter=";")
        next(reader, None)
        for row in reader:
            if (
                len(row) >= 7
                and row[2].strip() == "Aircraft"
                and row[3].strip() == "controlFlaps"
                and row[6].strip().startswith("0.5")
            ):
                try:
                    return base_utc + float(row[1]) / 1e6
                except ValueError:
                    pass
    return None


def video_duration(path: Path) -> float | None:
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "json", str(path)],
            capture_output=True, text=True, timeout=60,
        )
        return float(json.loads(r.stdout)["format"]["duration"])
    except Exception:
        return None


def extract_segment(
    src: Path, offset: float, duration: float, out: Path
) -> bool:
    out.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        ["ffmpeg", "-y",
         "-ss", f"{offset:.3f}",
         "-i", str(src),
         "-t", f"{duration:.3f}",
         "-c", "copy", str(out)],
        capture_output=True, text=True, timeout=600,
    )
    return result.returncode == 0


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    dry_run = "--dry-run" in sys.argv
    if dry_run:
        print("Mode: DRY-RUN\n")

    # 1. Derive base_utc from any scenario that has a session_epoch row
    base_utc: float | None = None
    anchor_source: str = ""
    for csv_path in sorted(SCENARIOS_DIR.glob("scenario_*_ingescape.csv")):
        b = get_base_utc(csv_path)
        if b is not None:
            base_utc = b
            anchor_source = csv_path.name
            break

    if base_utc is None:
        sys.exit("ERROR: no session_epoch row found in any scenario CSV")
    print(f"base_utc = {base_utc:.3f}  (from {anchor_source})\n")

    # 2. Parse all scenario CSVs
    scenarios: list[tuple[str, str, float, float, Path]] = []
    sc02_csv: Path | None = None

    for csv_path in sorted(SCENARIOS_DIR.glob("scenario_*_ingescape.csv")):
        m = SCENARIO_RE.match(csv_path.name)
        if not m:
            continue
        stem = m.group(1)   # e.g. "scenario_02_TARP-F_24R_alt"
        num  = stem.split("_")[1]
        name = stem[len(f"scenario_{num}_"):]

        start, end = parse_scenario_times(csv_path, base_utc)
        if start is None:
            print(f"  [SKIP] {csv_path.name}: no data rows")
            continue

        scenarios.append((num, name, start, end, csv_path))
        if num == "02":
            sc02_csv = csv_path

    print(f"Scenarios ({len(scenarios)}):")
    for num, name, s, e, _ in scenarios:
        print(f"  scenario_{num}_{name}:  {_utc_str(s)} → {_utc_str(e)}  ({e - s:.0f} s)")
    print()

    # 3. Find anchor event UTCs
    if sc02_csv is None:
        sys.exit("ERROR: scenario_02 CSV not found")

    sc02_start = next((s for n, _, s, _, _ in scenarios if n == "02"), None)
    flaps_05_utc = find_flaps_05_utc(sc02_csv, base_utc)

    if sc02_start is None or flaps_05_utc is None:
        sys.exit("ERROR: could not determine scenario_02 start or flaps_05 event")

    print(f"Anchor events:")
    print(f"  scenario_02 start  : {sc02_start:.3f}  ({_utc_str(sc02_start)})")
    print(f"  controlFlaps = 0.5 : {flaps_05_utc:.3f}  ({_utc_str(flaps_05_utc)})")
    print()

    # 4. Process each screen recording
    for vid_file, suffix, anchor_s, anchor_event in SCREEN_VIDEOS:
        vid_path = P14_DIR / vid_file
        if not vid_path.exists():
            print(f"[SKIP] {vid_file}: file not found\n")
            continue

        print(f"{'=' * 62}")
        print(f"  {vid_file}  (suffix={suffix!r})")
        print(f"  anchor: {anchor_s} s into video  →  {anchor_event}")
        print(f"{'=' * 62}")

        dur = video_duration(vid_path)
        if dur is None:
            print("  [SKIP] ffprobe failed\n")
            continue
        print(f"  Duration: {dur:.0f} s  ({dur / 60:.1f} min)")

        anchor_utc = sc02_start if anchor_event == "scenario_start" else flaps_05_utc
        vid_start_utc = anchor_utc - anchor_s
        vid_end_utc   = vid_start_utc + dur

        print(f"  Implied UTC start: {vid_start_utc:.3f}  ({_utc_str(vid_start_utc)})")
        print(f"  Implied UTC end:   {vid_end_utc:.3f}  ({_utc_str(vid_end_utc)})\n")

        for num, name, s_start, s_end, _ in scenarios:
            s_dur   = s_end - s_start
            offset  = s_start - vid_start_utc
            out_name = f"scenario_{num}_{name}_{suffix}.mp4"
            out_path = SCENARIOS_DIR / out_name

            print(f"  -- scenario_{num}_{name} --")

            if offset < -30:
                print(f"    [FAIL] scenario starts {-offset:.0f} s before video\n")
                continue
            if s_start > vid_end_utc + 30:
                print(f"    [FAIL] scenario starts after video end\n")
                continue

            if offset < 0:
                print(f"    [WARN] clamping offset {offset:.1f} s → 0")
                s_dur += offset
                offset = 0.0

            available = vid_end_utc - (vid_start_utc + offset)
            if s_dur > available + 5:
                print(f"    [WARN] clamping duration {s_dur:.0f} s → {available:.0f} s")
                s_dur = max(0.0, available)

            print(f"    offset={offset:.1f} s   duration={s_dur:.1f} s")

            if out_path.exists():
                size_mb = out_path.stat().st_size / 1e6
                print(f"    [SKIP] already exists ({size_mb:.1f} MB): {out_name}\n")
                continue

            if dry_run:
                print(f"    [DRY-RUN] → {out_name}\n")
                continue

            print(f"    Extracting → {out_name} …", flush=True)
            ok = extract_segment(vid_path, offset, s_dur, out_path)
            if ok:
                size_mb = out_path.stat().st_size / 1e6
                print(f"    [OK]  {out_name}  ({size_mb:.1f} MB)\n")
            else:
                print(f"    [FAIL] ffmpeg error\n")

        print()


if __name__ == "__main__":
    main()
