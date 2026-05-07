#!/usr/bin/env python3
"""pre_create_error_annotations.py

Scans all participant scenario folders under the HITLS directory for CSV
files whose name contains an error type (_alt_, _winds_, or _flaps_).
For each such file it finds the onset timestamp of the corresponding
error task in the TARS Agent checklist data, then writes a pre-populated
annotation JSON file next to the CSV — ready to be loaded by timeline.html.

Error task targets (matched against TARS Agent / current_state JSON):
  alt   → procedure="LINE-UP AND HOLD", task_object="Select Altitude",
           value="PRESET AS CLEARED"   → label "Error: Altitude"
  winds → procedure="LINE-UP AND HOLD", task_object="Winds",
           value="CHECK"               → label "Error: Winds"
  flaps → procedure="BEFORE TAKEOFF",  task_object="FLAPS",
           value="SET FOR TAKEOFF"     → label "Error: Flaps"

Output filename: {scenario_base}_annotations.json  (next to the CSV)
  e.g.  scenario_17_TARS_24L_alt_ingescape.csv
      → scenario_17_TARS_24L_alt_annotations.json

Existing annotation files are skipped unless --force is passed.

Usage:
  python pre_create_error_annotations.py          # dry preview (no files written)
  python pre_create_error_annotations.py --write  # write new annotation files
  python pre_create_error_annotations.py --write --force  # overwrite existing too
"""

import argparse
import csv
import glob
import json
import os
import re
import sys

# ── Task targets per error type ───────────────────────────────────────────────
TARGETS = {
    "alt": {
        "procedure":   "LINE-UP AND HOLD",
        "task_object": "Select Altitude",
        "value":       "PRESET AS CLEARED",
        "label":       "Error: Altitude",
    },
    "winds": {
        "procedure":   "LINE-UP AND HOLD",
        "task_object": "Winds",
        "value":       "CHECK",
        "label":       "Error: Winds",
    },
    "flaps": {
        "procedure":   "BEFORE TAKEOFF",
        "task_object": "FLAPS",
        "value":       "SET FOR TAKEOFF",
        "label":       "Error: Flaps",
    },
}

ERROR_PATTERN = re.compile(r"_(alt|winds|flaps)_")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# ── Helpers ───────────────────────────────────────────────────────────────────

def detect_error_type(filename: str):
    """Return 'alt', 'winds', or 'flaps' if present in filename, else None."""
    m = ERROR_PATTERN.search(filename)
    return m.group(1) if m else None


def parse_csv(csv_path: str):
    """
    Parse a semicolon-delimited ingescape CSV.
    Returns a list of dicts: {ts: float, agent: str, source: str, value: str}.
    The first data row (after the header) defines t0.
    """
    rows = []
    with open(csv_path, newline="", encoding="utf-8", errors="replace") as fh:
        reader = csv.reader(fh, delimiter=";")
        next(reader, None)  # skip header row
        for row in reader:
            if len(row) < 7:
                continue
            try:
                ts = float(row[1])
            except ValueError:
                continue
            rows.append(
                {
                    "ts":     ts,
                    "agent":  row[2],
                    "source": row[3],
                    "value":  row[6],
                }
            )
    return rows


def find_task_onset(rows: list, target: dict):
    """
    Return relSec (float) of the first TARS Agent / current_state row whose
    JSON matches target procedure / task_object / value.
    Returns None if not found.
    """
    if not rows:
        return None
    t0 = rows[0]["ts"]
    for row in rows:
        if row["agent"] != "TARS Agent" or row["source"] != "current_state":
            continue
        try:
            j = json.loads(row["value"])
        except (json.JSONDecodeError, TypeError):
            continue
        if (
            j.get("procedure")   == target["procedure"]
            and j.get("task_object") == target["task_object"]
            and j.get("value")       == target["value"]
        ):
            return round(row["ts"] - t0, 6)
    return None


def annotation_path_for(csv_path: str) -> str:
    """Return the annotation JSON path that matches the timeline.html naming convention."""
    csv_name = os.path.basename(csv_path)
    base = re.sub(r"_ingescape\.csv$", "", csv_name, flags=re.IGNORECASE)
    return os.path.join(os.path.dirname(csv_path), base + "_annotations.json")


def build_payload(csv_name: str, rel_sec: float, label: str) -> str:
    payload = {
        "csv": csv_name,
        "annotations": [
            {
                "id":     1,
                "relSec": rel_sec,
                "label":  label,
            }
        ],
    }
    return json.dumps(payload, indent=2)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--write",  action="store_true",
                        help="Actually write annotation JSON files (default: dry run)")
    parser.add_argument("--force",  action="store_true",
                        help="Overwrite existing annotation files")
    args = parser.parse_args()

    if not args.write:
        print("DRY RUN — pass --write to actually create files.\n")

    pattern   = os.path.join(BASE_DIR, "P*/scenarios/*.csv")
    csv_files = sorted(glob.glob(pattern))

    created  = []
    skipped  = []
    not_found = []

    for csv_path in csv_files:
        fname      = os.path.basename(csv_path)
        error_type = detect_error_type(fname)
        if error_type is None:
            continue

        target      = TARGETS[error_type]
        annot_path  = annotation_path_for(csv_path)
        annot_name  = os.path.basename(annot_path)
        rel_display = os.path.relpath(csv_path, BASE_DIR)

        # Skip if already exists and not forcing
        if os.path.exists(annot_path) and not args.force:
            print(f"  [SKIP ] {rel_display}  →  {annot_name} already exists")
            skipped.append(annot_path)
            continue

        rows    = parse_csv(csv_path)
        rel_sec = find_task_onset(rows, target)

        if rel_sec is None:
            print(f"  [MISS ] {rel_display}  — target task not found in CSV")
            not_found.append(csv_path)
            continue

        payload = build_payload(fname, rel_sec, target["label"])

        if args.write:
            with open(annot_path, "w", encoding="utf-8") as fh:
                fh.write(payload)
            status = "[WRITE]"
        else:
            status = "[WOULD]"

        print(f"  {status} {rel_display}")
        print(f"          → {annot_name}  (relSec={rel_sec:.3f} s, label=\"{target['label']}\")")
        created.append(annot_path)

    print()
    action = "written" if args.write else "would be written"
    print(f"Done.  {len(created)} annotation file(s) {action},  "
          f"{len(skipped)} skipped (already exist),  "
          f"{len(not_found)} CSV(s) where target task was not found.")


if __name__ == "__main__":
    main()
