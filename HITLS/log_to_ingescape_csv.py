#!/usr/bin/env python3
"""
log_to_ingescape_csv.py

Converts Ingescape .log slice files (produced from P17 backup logs) into the
scenario_NN_<slug>_ingescape.csv format expected by timeline.html.

Supports two log source formats that both use:
  SOURCE;DD/MM/YYYY;HH:MM:SS.ffffff;LEVEL;FUNCTION;message

  ┌─ Ingescape Circle (backup_1) ──────────────────────────────────────────────
  │  model_write lines use UUID-based input identifiers:
  │    set input 4fe7add4-d041-429b-8012-c04a7c2fb3f4.pitch to double 11.9
  │  Agent name resolved via two mechanisms:
  │    1. MAP FROM "A" . "x" TO "<Agent>" . "<source>" lines that immediately
  │       precede the model_write of an input being written.
  │    2. Source-name heuristic lookup table (known per-agent signals in the
  │       ADAIR-POLY study) for direct subscriptions that have no MAP line
  │       (e.g. Aircraft outputs flowing straight into Ingescape Circle).
  └────────────────────────────────────────────────────────────────────────────

  ┌─ Recorder (backup_2) ──────────────────────────────────────────────────────
  │  model_write lines use named agent identifiers directly:
  │    set input Aircraft.paused to bool 0
  │  Agent name and source extracted with no ambiguity.
  └────────────────────────────────────────────────────────────────────────────

Output CSV columns (semicolon-delimited):
  uuid ; timestamp ; agent ; source ; type ; igs_timestamp ; value

  uuid          – row UUID (generated)
  timestamp     – Unix epoch seconds (float)
  agent         – agent name
  source        – output/input name
  type          – Ingescape type integer: 1=bool 2=int 3=double 4=string 5=null
  igs_timestamp – (empty, not available from logs)
  value         – value string

Usage:
  python log_to_ingescape_csv.py HITLS/P17/scenarios/
  python log_to_ingescape_csv.py HITLS/P17/scenarios/scenario_08_TARP-F_backup_1.log
  python log_to_ingescape_csv.py HITLS/P17/scenarios/*.log
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

# ── Ingescape type name → integer code ────────────────────────────────────────
_TYPE_MAP: dict[str, int] = {
    'bool':        1,
    'int':         2,
    'double':      3,
    'string':      4,
    'null':        5,   # "to null string" / "to null impulsion"
    'impulsion':   5,
    'data':        6,
}

# ── Source-name heuristic: known signal names per agent (ADAIR-POLY study) ───
# Used to resolve UUID-based model_write lines in Ingescape Circle logs.
_SOURCE_AGENT: dict[str, str] = {
    # Aircraft
    'pitch':                    'Aircraft',
    'roll':                     'Aircraft',
    'altitude':                 'Aircraft',
    'airspeed':                 'Aircraft',
    'latitude':                 'Aircraft',
    'longitude':                'Aircraft',
    'heading':                  'Aircraft',
    'slip':                     'Aircraft',
    'cabin_altitude':           'Aircraft',
    'verticalSpeed':            'Aircraft',
    'paused':                   'Aircraft',
    'groundspeed':              'Aircraft',
    'controlPitch':             'Aircraft',
    'controlRoll':              'Aircraft',
    'controlYaw':               'Aircraft',
    'master_warning':           'Aircraft',
    'master_caution':           'Aircraft',
    'engine_fire_l':            'Aircraft',
    'engine_fire_r':            'Aircraft',
    'flap_position':            'Aircraft',
    'gear_position':            'Aircraft',
    'spoiler_position':         'Aircraft',
    # TARS Agent
    'current_state':            'TARS Agent',
    'allocation_reloaded':      'TARS Agent',
    'pax_safety':               'TARS Agent',
    'flight_director':          'TARS Agent',
    'speed_mode':               'TARS Agent',
    'heading_mode':             'TARS Agent',
    'autopilot_master':         'TARS Agent',
    'autopilot_heading_set':    'TARS Agent',
    'autopilot_state':          'TARS Agent',
    'yaw_damper':               'TARS Agent',
    'flaps':                    'TARS Agent',
    'altimeter_setting':        'TARS Agent',
    'landing_gear':             'TARS Agent',
    'spoilers':                 'TARS Agent',
    'thrust_reversers':         'TARS Agent',
    'brake_fans':               'TARS Agent',
    # Shared Interface
    'task_approval':            'Shared Interface',
    'task_acknowledged':        'Shared Interface',
    'task_cancelled':           'Shared Interface',
    'task_override':            'Shared Interface',
    'start_procedure':          'Shared Interface',
    'stop_procedure':           'Shared Interface',
    'update_allocation':        'Shared Interface',
    'next_step':                'Shared Interface',
    'previous_step':            'Shared Interface',
    'request_atis':             'Shared Interface',
    'load_csv':                 'Shared Interface',
    'popup_active':             'Shared Interface',
    'force_state_jump':         'Shared Interface',
    'countdown_complete':       'Shared Interface',
    'tts_stop':                 'Shared Interface',
    'tts_unmute':               'Shared Interface',
    'emergency_inject':         'Shared Interface',
    # SmartEyeProBridge
    'eyelid_opening':           'SmartEyeProBridge',
    'an_blink':                 'SmartEyeProBridge',
    'filtered_pupil_diameter':  'SmartEyeProBridge',
    'saccade':                  'SmartEyeProBridge',
    'fixation':                 'SmartEyeProBridge',
    'gaze_point_2d':            'SmartEyeProBridge',
    'gaze_direction_3d':        'SmartEyeProBridge',
    'pupil_diameter_left':      'SmartEyeProBridge',
    'pupil_diameter_right':     'SmartEyeProBridge',
    # RudderTrimAgent
    'trim_rudder':              'RudderTrimAgent',
    'trim_status':              'RudderTrimAgent',
    # Speech / ATC / TTS
    'is_listening':             'Speech_to_Text_Agent',
    'is_speaking':              'TTS_Agent',          # also ATC_Agent; TTS_Agent more common
    'current_text':             'TTS_Agent',
    # Remote_Control_Agent
    'l_eng_fire_button':        'Remote_Control_Agent',
    'r_eng_fire_button':        'Remote_Control_Agent',
    'fire_warn_test':           'Remote_Control_Agent',
    'l_bottle':                 'Remote_Control_Agent',
    'r_bottle':                 'Remote_Control_Agent',
    # PanelListener (catch-all; specific names unknown)
}

# ── Regexes ───────────────────────────────────────────────────────────────────
_LOG_HDR_RE = re.compile(
    r'^([^;]+);(\d{2}/\d{2}/\d{4});(\d{2}:\d{2}:\d{2}\.\d+);[^;]+;([^;]+);(.+)$'
)
# UUID-based model_write (Ingescape Circle)
_UUID_RE  = re.compile(
    r'^set input ([a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12})'
    r'\.(\S+) to (\S+)(?: (.+))?$'
)
# Named-agent model_write (Recorder)
_NAMED_RE = re.compile(
    r'^set input (.+?)\.(\S+) to (\S+)(?: (.+))?$'
)
# MAP FROM "..." . "..." TO "<DestAgent>" . "<DestSource>"
_MAP_RE = re.compile(
    r'MAP FROM ".+?" \. ".+?" TO "(.+?)" \. "(.+?)"'
)


def _ts_to_epoch(date_str: str, time_str: str) -> float | None:
    try:
        parts = time_str.split('.')
        time_norm = parts[0] + '.' + (parts[1].ljust(6, '0')[:6] if len(parts) == 2 else '000000')
        dt = datetime.strptime(f"{date_str} {time_norm}", "%d/%m/%Y %H:%M:%S.%f")
        return dt.replace(tzinfo=timezone.utc).timestamp()
    except ValueError:
        return None


def _type_int(type_str: str) -> int:
    return _TYPE_MAP.get(type_str.lower(), 0)


# ── Format detection ──────────────────────────────────────────────────────────
def _detect_format(log_path: Path) -> str:
    """Return 'circle' (Ingescape Circle) or 'recorder' based on first line."""
    with log_path.open(encoding='utf-8', errors='replace') as fh:
        for line in fh:
            m = _LOG_HDR_RE.match(line.rstrip('\n'))
            if m:
                src = m.group(1).strip()
                if src == 'Recorder':
                    return 'recorder'
                return 'circle'
    return 'circle'


# ── Pass 1 for Ingescape Circle: build UUID → agent name ─────────────────────
def _build_uuid_map(log_path: Path) -> dict[str, str]:
    """
    Scan the log for MAP lines and correlate with subsequent model_write UUID lines
    to infer which UUID belongs to which agent.  Falls back to source-name heuristics
    for UUIDs that appear without a preceding MAP line.
    """
    uuid_agent: dict[str, str] = {}
    # pending: (dest_agent, dest_source) from the most recent MAP line not yet matched
    pending: tuple[str, str] | None = None

    with log_path.open(encoding='utf-8', errors='replace') as fh:
        for line in fh:
            line = line.rstrip('\n')
            m_hdr = _LOG_HDR_RE.match(line)
            if not m_hdr:
                continue
            func = m_hdr.group(4).strip()
            msg  = m_hdr.group(5).strip()

            if func == 'igsMessageOutput':
                m_map = _MAP_RE.search(msg)
                if m_map:
                    pending = (m_map.group(1), m_map.group(2))
                continue

            if func == 'model_write':
                m_uuid = _UUID_RE.match(msg)
                if m_uuid:
                    u      = m_uuid.group(1)
                    source = m_uuid.group(2)
                    if u not in uuid_agent:
                        if pending and pending[1] == source:
                            uuid_agent[u] = pending[0]
                        elif source in _SOURCE_AGENT:
                            uuid_agent[u] = _SOURCE_AGENT[source]
                    pending = None  # consume pending after first model_write
                continue

    return uuid_agent


# ── Core converter ────────────────────────────────────────────────────────────
def convert_log(log_path: Path, out_path: Path) -> dict:
    """
    Convert a single .log slice file to an ingescape CSV.
    Returns a stats dict: {rows, unresolved_uuids}.
    """
    fmt      = _detect_format(log_path)
    is_circle = fmt == 'circle'

    uuid_agent: dict[str, str] = {}
    if is_circle:
        print(f"    Building UUID→agent map …", end=' ', flush=True)
        uuid_agent = _build_uuid_map(log_path)
        print(f"{len(uuid_agent)} UUIDs resolved")

    rows_written = 0
    unresolved: set[str] = set()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with (
        log_path.open(encoding='utf-8', errors='replace') as fh,
        out_path.open('w', newline='', encoding='utf-8') as fout,
    ):
        writer = csv.writer(fout, delimiter=';')
        writer.writerow(['uuid', 'timestamp', 'agent', 'source', 'type', 'igs_timestamp', 'value'])

        for line in fh:
            line = line.rstrip('\n')
            m_hdr = _LOG_HDR_RE.match(line)
            if not m_hdr:
                continue
            func = m_hdr.group(4).strip()
            if func != 'model_write':
                continue
            msg = m_hdr.group(5).strip()

            epoch = _ts_to_epoch(m_hdr.group(2), m_hdr.group(3))
            if epoch is None:
                continue

            agent  = None
            source = None
            typ    = 0
            value  = ''

            if is_circle:
                m = _UUID_RE.match(msg)
                if not m:
                    continue
                uid_str = m.group(1)
                source  = m.group(2)
                typ     = _type_int(m.group(3))
                value   = (m.group(4) or '').strip()
                agent   = uuid_agent.get(uid_str)
                if agent is None:
                    agent = _SOURCE_AGENT.get(source)
                if agent is None:
                    unresolved.add(uid_str)
                    continue
            else:  # recorder
                m = _NAMED_RE.match(msg)
                if not m:
                    continue
                agent  = m.group(1).strip()
                source = m.group(2)
                typ    = _type_int(m.group(3))
                value  = (m.group(4) or '').strip()

            row_uuid = str(uuid.uuid4())
            writer.writerow([row_uuid, f'{epoch:.3f}', agent, source, typ, '', value])
            rows_written += 1

    return {'rows': rows_written, 'unresolved_uuids': unresolved}


# ── Entry point ───────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description='Convert Ingescape .log slices to _ingescape.csv')
    parser.add_argument('paths', nargs='*', metavar='PATH',
                        help='Log files or directories to convert')
    parser.add_argument('--keep', action='store_true',
                        help='Keep source .log files after conversion (default: delete them)')
    args = parser.parse_args()

    if not args.paths:
        print(__doc__)
        sys.exit(0)

    # Collect all .log files from arguments (files or directories)
    log_files: list[Path] = []
    for arg in args.paths:
        p = Path(arg)
        if p.is_dir():
            log_files.extend(sorted(p.glob('**/*.log')))
        elif p.suffix == '.log' and p.is_file():
            log_files.append(p)
        else:
            print(f"  Skipping (not a .log file or directory): {arg}")

    if not log_files:
        print("No .log files found.")
        sys.exit(1)

    print(f"Converting {len(log_files)} log file(s) …\n")

    for log_path in log_files:
        # Output: replace "_backup_N.log" with "_ingescape.csv"
        stem = re.sub(r'_backup_\d+$', '', log_path.stem)
        out_path = log_path.parent / f'{stem}_ingescape.csv'

        print(f"  {log_path.name}")
        print(f"  → {out_path.name}")
        stats = convert_log(log_path, out_path)

        n_unresolved = len(stats['unresolved_uuids'])
        size_kb = out_path.stat().st_size // 1024
        msg = f"    {stats['rows']:,} rows  ({size_kb} KB)"
        if n_unresolved:
            msg += f"  ⚠  {n_unresolved} UUID(s) unresolved (omitted)"
        print(msg)

        if not args.keep:
            log_path.unlink()
            print(f"    deleted {log_path.name}")
        print()

    print("Done.")


if __name__ == '__main__':
    main()
