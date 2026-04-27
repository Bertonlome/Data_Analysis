"""
Extract scenario slices from P0x_ingescape.csv files.

Boundary detection
------------------
The facilitator pauses the simulator between scenarios.
A pause is detected by:  agent == "Aircraft",  source == "paused",  value == "True"
The unpaused stretches of rows are extracted as individual scenarios.

Slug metadata
-------------
The most recent  TARS Agent / allocation_reloaded  event before the scenario end
provides the file slug (e.g. "TARS.csv" -> "TARS") used for pre-classification.

Output: P0x/scenarios/scenario_NN_<slug>_ingescape.csv

Usage:
    python extract_scenarios.py [participant_dirs...]
    python extract_scenarios.py HITLS/P02
    python extract_scenarios.py HITLS/P02 HITLS/P03
    python extract_scenarios.py          # interactive prompt for all P0x dirs
"""

import csv
import json
import re
import sys
from pathlib import Path


# ── helpers ──────────────────────────────────────────────────────────────────

def slugify(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_")


def find_participant_dirs(base: Path) -> list[Path]:
    return sorted(p for p in base.iterdir() if p.is_dir() and re.match(r"P\d+$", p.name))


def find_ingescape_csv(participant_dir: Path) -> Path | None:
    for path in sorted(participant_dir.glob("*_ingescape.csv")):
        if "train_then_test" not in path.name:
            return path
    return None


# ── core extraction ───────────────────────────────────────────────────────────

def extract_scenarios(csv_path: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Scanning {csv_path.name} ...")

    # ── Pass 1: collect pause-state transitions and slug events ──────────────
    # paused_events: (row_idx, is_paused)  — Aircraft / paused
    # slug_events:   (row_idx, slug)       — TARS Agent / allocation_reloaded
    paused_events: list[tuple[int, bool]] = []
    slug_events:   list[tuple[int, str]]  = []
    last_row_idx = 0

    with csv_path.open(newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh, delimiter=";")
        next(reader)  # header
        for row_idx, row in enumerate(reader):
            last_row_idx = row_idx
            if len(row) < 7:
                continue
            agent, src, val_raw = row[2], row[3], row[6]

            if agent == "Aircraft" and src == "paused":
                is_paused = val_raw.strip().lower() in ("true", "1")
                paused_events.append((row_idx, is_paused))

            elif agent == "TARS Agent" and src == "allocation_reloaded":
                try:
                    j = json.loads(val_raw)
                    slug = Path(j.get("csv", "UNKNOWN")).stem
                    slug_events.append((row_idx, slug))
                except json.JSONDecodeError:
                    pass

    # ── Derive scenario boundaries from pause-state transitions ──────────────
    # The file is assumed to start in a running (not paused) state.
    # paused True  → end the current scenario
    # paused False → start a new scenario
    boundaries: list[dict] = []
    paused_state = False  # initial assumption: simulator is running
    in_scenario  = True
    scenario_start = 0

    for row_idx, is_paused in paused_events:
        if is_paused and not paused_state:
            # running → paused: close current scenario
            if in_scenario:
                boundaries.append({"start": scenario_start, "end": row_idx - 1})
                in_scenario = False
            paused_state = True
        elif not is_paused and paused_state:
            # paused → running: open new scenario
            scenario_start = row_idx
            in_scenario    = True
            paused_state   = False

    # Close any scenario still open at EOF
    if in_scenario:
        boundaries.append({"start": scenario_start, "end": last_row_idx})

    if not boundaries:
        print("  No scenarios found (no Aircraft/paused events).")
        return

    # ── Number scenarios and assign slugs ─────────────────────────────────────
    for i, b in enumerate(boundaries):
        b["num"] = i + 1
        # Most-recent allocation_reloaded at or before the scenario's last row
        slug = "UNKNOWN"
        for se_idx, se_slug in slug_events:
            if se_idx <= b["end"]:
                slug = se_slug
            else:
                break
        b["slug"] = slug

    print(f"  Found {len(boundaries)} scenario(s).")

    # ── Pass 2: write slice files ─────────────────────────────────────────────
    boundaries.sort(key=lambda b: b["start"])

    # (start_row, end_row, csv_writer, file_handle)
    out_handles: list[tuple[int, int, object, object]] = []
    written_paths: list[Path] = []

    for b in boundaries:
        fname = f"scenario_{b['num']:02d}_{b['slug']}_ingescape.csv"
        out_path = output_dir / fname
        fout = out_path.open("w", newline="", encoding="utf-8")
        writer = csv.writer(fout, delimiter=";")
        out_handles.append((b["start"], b["end"], writer, fout))
        written_paths.append(out_path)

    # Sequential O(N) pointer scan — advance past finished scenarios
    ts_ranges: list[list[float | None]] = [[None, None] for _ in out_handles]

    with csv_path.open(newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh, delimiter=";")
        header = next(reader)
        for _, _, writer, _ in out_handles:
            writer.writerow(header)

        sci = 0
        n_scenarios = len(out_handles)

        for row_idx, row in enumerate(reader):
            # Advance pointer past scenarios that have ended
            while sci < n_scenarios and row_idx > out_handles[sci][1]:
                sci += 1
            if sci >= n_scenarios:
                break
            s, e, writer, _ = out_handles[sci]
            if s <= row_idx <= e:
                writer.writerow(row)
                if len(row) > 1:
                    try:
                        ts = float(row[1])
                        if ts_ranges[sci][0] is None:
                            ts_ranges[sci][0] = ts
                        ts_ranges[sci][1] = ts
                    except ValueError:
                        pass

    for _, _, _, fout in out_handles:
        fout.close()

    for i, out_path in enumerate(written_paths):
        size_kb = out_path.stat().st_size // 1024
        first_ts, last_ts = ts_ranges[i]
        if first_ts is not None and last_ts is not None:
            duration = last_ts - first_ts
            print(f"    -> {out_path.name}  ({size_kb} KB, {duration:.1f}s)")
        else:
            print(f"    -> {out_path.name}  ({size_kb} KB)")



# ── entry point ───────────────────────────────────────────────────────────────

def _prompt_participant_dirs(all_dirs: list[Path]) -> list[Path]:
    """Interactively ask the user which participant dirs to process."""
    print("Found participant directories:")
    for i, p in enumerate(all_dirs, 1):
        print(f"  [{i}] {p.name}")
    print(f"  [a] All ({len(all_dirs)} participants)")
    print(f"  [q] Quit")

    while True:
        raw = input("\nSelect participant(s) [number(s) comma-separated, a, or q]: ").strip().lower()
        if raw == "q":
            return []
        if raw == "a":
            return all_dirs
        try:
            indices = [int(x.strip()) for x in raw.split(",")]
            chosen = []
            for idx in indices:
                if 1 <= idx <= len(all_dirs):
                    chosen.append(all_dirs[idx - 1])
                else:
                    print(f"  Invalid number: {idx}")
                    break
            else:
                return chosen
        except ValueError:
            print("  Please enter numbers separated by commas, 'a', or 'q'.")


def main() -> None:
    base = Path(__file__).parent

    if len(sys.argv) > 1:
        participant_dirs = [Path(a).resolve() for a in sys.argv[1:]]
    else:
        all_dirs = find_participant_dirs(base)
        if not all_dirs:
            print("No participant directories found.")
            return
        participant_dirs = _prompt_participant_dirs(all_dirs)

    if not participant_dirs:
        print("Nothing to process.")
        return

    for pdir in participant_dirs:
        print(f"\nProcessing {pdir.name} ...")
        csv_path = find_ingescape_csv(pdir)
        if csv_path is None:
            print("  No *_ingescape.csv found, skipping.")
            continue
        extract_scenarios(csv_path, pdir / "scenarios")

    print("\nDone.")


if __name__ == "__main__":
    main()
