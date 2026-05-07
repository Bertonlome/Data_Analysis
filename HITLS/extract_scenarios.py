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
from datetime import datetime, timezone
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


def find_ingescape_logs(participant_dir: Path) -> list[Path]:
    """Return all *_ingescape_backup_*.log files in the participant dir, sorted."""
    return sorted(participant_dir.glob("*_ingescape_backup_*.log"))


# ── log-file parsing ──────────────────────────────────────────────────────────
# Both Ingescape Circle and Recorder logs share the line structure:
#   SOURCE;DD/MM/YYYY;HH:MM:SS.ffffff;LEVEL;FUNCTION;message
# Paused events   → message ends with ".paused to bool 0|1"
# Allocation evts → message contains "allocation_reloaded to string {JSON}"

_LOG_HDR_RE   = re.compile(r'^[^;]+;(\d{2}/\d{2}/\d{4});(\d{2}:\d{2}:\d{2}\.\d+);')
_LOG_PAUSE_RE = re.compile(r'\.paused to bool ([01])\s*$')
_LOG_ALLOC_RE = re.compile(r'allocation_reloaded to string (\{.+\})\s*$')


def _log_ts_to_epoch(date_str: str, time_str: str) -> float | None:
    """Parse 'DD/MM/YYYY' + 'HH:MM:SS.ffffff' into a UTC epoch float."""
    try:
        # Pad microseconds to 6 digits in case they are shorter
        parts = time_str.split('.')
        time_norm = parts[0] + '.' + parts[1].ljust(6, '0')[:6] if len(parts) == 2 else parts[0] + '.000000'
        dt = datetime.strptime(f"{date_str} {time_norm}", "%d/%m/%Y %H:%M:%S.%f")
        return dt.replace(tzinfo=timezone.utc).timestamp()
    except ValueError:
        return None


def parse_log_events(log_path: Path):
    """Yield (line_idx, epoch_s, event_type, value) for pause/allocation events.

    event_type 'paused'     → value is bool (True = paused, False = running)
    event_type 'allocation' → value is slug string (stem of the csv field)
    """
    with log_path.open(encoding='utf-8', errors='replace') as fh:
        for line_idx, line in enumerate(fh):
            m_hdr = _LOG_HDR_RE.match(line)
            if not m_hdr:
                continue
            epoch = _log_ts_to_epoch(m_hdr.group(1), m_hdr.group(2))
            if epoch is None:
                continue
            m_pause = _LOG_PAUSE_RE.search(line)
            if m_pause:
                yield (line_idx, epoch, 'paused', bool(int(m_pause.group(1))))
                continue
            m_alloc = _LOG_ALLOC_RE.search(line)
            if m_alloc:
                try:
                    j = json.loads(m_alloc.group(1))
                    slug = Path(j.get('csv', 'UNKNOWN')).stem
                    yield (line_idx, epoch, 'allocation', slug)
                except json.JSONDecodeError:
                    pass


def extract_scenarios_from_log(log_path: Path, output_dir: Path) -> dict:
    """Detect scenario boundaries in an Ingescape log file.

    Applies the same pause-boundary logic as extract_scenarios() but operates
    on timestamped events instead of CSV rows.  Since the log does not contain
    full data rows, no slice CSVs are written; instead a JSON summary is saved
    to output_dir and the report dict is returned.

    Returns a dict with keys: source_file, scenarios (list of dicts).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Scanning {log_path.name} ...")

    events = list(parse_log_events(log_path))

    paused_events   = [(idx, ep, val) for idx, ep, et, val in events if et == 'paused']
    slug_events     = [(idx, ep, val) for idx, ep, et, val in events if et == 'allocation']
    last_line_idx   = events[-1][0] if events else 0
    last_epoch      = events[-1][1] if events else 0.0

    # ── Same boundary logic as extract_scenarios() ────────────────────────────
    boundaries: list[dict] = []
    paused_state   = False
    in_scenario    = True
    scenario_start_idx   = 0
    scenario_start_epoch = events[0][1] if events else 0.0

    for line_idx, epoch, is_paused in paused_events:
        if is_paused and not paused_state:
            if in_scenario:
                boundaries.append({
                    'start_idx': scenario_start_idx, 'end_idx': line_idx - 1,
                    'start_epoch': scenario_start_epoch, 'end_epoch': epoch,
                })
                in_scenario = False
            paused_state = True
        elif not is_paused and paused_state:
            scenario_start_idx   = line_idx
            scenario_start_epoch = epoch
            in_scenario  = True
            paused_state = False

    if in_scenario:
        boundaries.append({
            'start_idx': scenario_start_idx, 'end_idx': last_line_idx,
            'start_epoch': scenario_start_epoch, 'end_epoch': last_epoch,
        })

    if not boundaries:
        print("  No scenarios found (no paused events).")
        return {'source_file': log_path.name, 'scenarios': []}

    # ── Number scenarios and assign slugs (same logic) ────────────────────────
    for i, b in enumerate(boundaries):
        b['num'] = i + 1
        slug = 'UNKNOWN'
        for se_idx, se_epoch, se_slug in slug_events:
            if se_idx <= b['end_idx']:
                slug = se_slug
            else:
                break
        b['slug']       = slug
        b['duration_s'] = round(b['end_epoch'] - b['start_epoch'], 1)
        b['start_time'] = datetime.utcfromtimestamp(b['start_epoch']).strftime('%H:%M:%S')
        b['end_time']   = datetime.utcfromtimestamp(b['end_epoch']).strftime('%H:%M:%S')

    print(f"  Found {len(boundaries)} scenario(s).")
    for b in boundaries:
        print(f"    -> scenario_{b['num']:02d}_{b['slug']}  "
              f"{b['start_time']} – {b['end_time']}  ({b['duration_s']:.1f} s)")

    # ── Pass 2: write log slice files ─────────────────────────────────────────
    # Derive a short source tag from the log filename (e.g. "backup_1")
    source_tag = re.sub(r'^.*_ingescape_', '', log_path.stem)  # e.g. "backup_1"

    # Build (start_idx, end_idx, file_handle) list
    out_handles: list[tuple[int, int, object]] = []
    written_paths: list[Path] = []
    for b in boundaries:
        fname = f"scenario_{b['num']:02d}_{b['slug']}_{source_tag}.log"
        out_path = output_dir / fname
        out_handles.append((b['start_idx'], b['end_idx'], out_path.open('w', encoding='utf-8')))
        written_paths.append(out_path)

    with log_path.open(encoding='utf-8', errors='replace') as fh:
        sci = 0
        n = len(out_handles)
        for line_idx, line in enumerate(fh):
            while sci < n and line_idx > out_handles[sci][1]:
                sci += 1
            if sci >= n:
                break
            s, e, fout = out_handles[sci]
            if s <= line_idx <= e:
                fout.write(line)

    for _, _, fout in out_handles:
        fout.close()

    for i, out_path in enumerate(written_paths):
        size_kb = out_path.stat().st_size // 1024
        b = boundaries[i]
        print(f"    -> {out_path.name}  ({size_kb} KB, {b['duration_s']:.1f} s)")

    # ── Write JSON summary ─────────────────────────────────────────────────────
    summary_name = log_path.stem + '_summary.json'
    summary_path = output_dir / summary_name
    report = {
        'source_file': log_path.name,
        'scenarios': [
            {'num': b['num'], 'slug': b['slug'],
             'start': b['start_time'], 'end': b['end_time'],
             'duration_s': b['duration_s']}
            for b in boundaries
        ],
    }
    with summary_path.open('w', encoding='utf-8') as fh:
        json.dump(report, fh, indent=2)
    print(f"  Summary → {summary_path.name}")

    return report


# ── comparison ────────────────────────────────────────────────────────────────

def print_scenario_comparison(reports: list[dict]) -> None:
    """Print a side-by-side comparison table of scenarios across multiple sources."""
    if not reports:
        return

    max_scenarios = max(len(r['scenarios']) for r in reports)
    if max_scenarios == 0:
        print("  (no scenarios found in any source)")
        return

    col_w = 26  # width per source column
    header = "  Scenario  " + "".join(f"  {r['source_file'][:col_w]:<{col_w}}" for r in reports)
    print("\n" + "=" * len(header))
    print("  SCENARIO COMPARISON")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for i in range(max_scenarios):
        row = f"  #{i+1:>2}       "
        for r in reports:
            scens = r['scenarios']
            if i < len(scens):
                s = scens[i]
                cell = f"{s['slug']:<12} {s['start']}–{s['end']}"
            else:
                cell = "(missing)"
            row += f"  {cell:<{col_w}}"
        print(row)

    print("=" * len(header))

    # Agreement summary
    print("\n  Agreement between sources:")
    for i in range(max_scenarios):
        slugs = []
        for r in reports:
            scens = r['scenarios']
            slugs.append(scens[i]['slug'] if i < len(scens) else None)
        all_same = len(set(s for s in slugs if s is not None)) <= 1
        marker = "✓" if all_same else "✗"
        print(f"    {marker} Scenario #{i+1:>2}: {' | '.join(str(s) for s in slugs)}")
    print()


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
        header = next(reader)
        # New datalake format uses relative_time_us (µs); old format uses epoch seconds.
        ts_scale = 1e-6 if len(header) > 1 and header[1] == 'relative_time_us' else 1.0
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
        hdr = next(reader)
        for _, _, writer, _ in out_handles:
            writer.writerow(hdr)

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
                        ts = float(row[1]) * ts_scale
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
        scenarios_dir = pdir / "scenarios"
        all_reports: list[dict] = []

        # ── Primary: ingescape CSV ─────────────────────────────────────────────
        csv_path = find_ingescape_csv(pdir)
        if csv_path is None:
            print("  No *_ingescape.csv found.")
        else:
            csv_size = csv_path.stat().st_size
            if csv_size <= csv_path.name.__len__() + 200:  # essentially header-only
                print(f"  {csv_path.name} is empty (header only) — skipping CSV extraction.")
            else:
                extract_scenarios(csv_path, scenarios_dir)

        # ── Fallback / backup: Ingescape log files ────────────────────────────
        log_paths = find_ingescape_logs(pdir)
        if log_paths:
            print(f"  Found {len(log_paths)} backup log file(s).")
            for log_path in log_paths:
                report = extract_scenarios_from_log(log_path, scenarios_dir)
                if report['scenarios']:
                    all_reports.append(report)
        elif csv_path is None:
            print("  No backup log files found either.")

        # ── Comparison across logs ─────────────────────────────────────────────
        if len(all_reports) >= 2:
            print_scenario_comparison(all_reports)
        elif len(all_reports) == 1:
            print(f"\n  Only one log source produced scenarios; no cross-comparison possible.")

    print("\nDone.")


if __name__ == "__main__":
    main()
