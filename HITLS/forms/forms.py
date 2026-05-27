"""
forms.py — Run all questionnaire analysis scripts for a selected participant.

Scripts run in order:
  1. pre-experiment-forms   (demographics + PTS)
  2. nasa-tlx               (workload)
  3. oversight-bespoke      (oversight)
  4. perceived-control      (perceived control)
  5. sus                    (usability)
  6. trust-in-automation    (TiA)
  7. trust-risk             (trust & perceived risk VAS)

Usage
-----
  python forms/forms.py           # interactive participant selection
  python forms/forms.py P02       # run directly for participant P02
  python forms/forms.py 3         # run for 3rd participant in the list

Output: reports saved to HITLS/{PID}/cleaned/{PID}_*_report.txt
"""

import os
import sys
import io
import argparse
import importlib.util

FORMS_DIR = os.path.dirname(os.path.abspath(__file__))
HITLS_DIR = os.path.dirname(FORMS_DIR)

SCRIPTS = [
    ("pre-experiment-forms", "pre-experiment-forms.py"),
    ("nasa-tlx",             "nasa-tlx.py"),
    ("oversight-bespoke",    "oversight-bespoke.py"),
    ("perceived-control",    "perceived-control.py"),
    ("sus",                  "sus.py"),
    ("trust-in-automation",  "trust-in-automation.py"),
    ("trust-risk",           "trust-risk.py"),
]


def find_participants():
    return [
        e for e in sorted(os.listdir(HITLS_DIR))
        if os.path.isdir(os.path.join(HITLS_DIR, e))
        and e.startswith("P") and e[1:].isdigit()
    ]


def load_module(name, filename):
    path = os.path.join(FORMS_DIR, filename)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run_with_participant(mod, participant_input):
    """Call mod.main() with stdin pre-filled so the selection prompt is answered."""
    old_stdin = sys.stdin
    sys.stdin = io.StringIO(participant_input + "\n")
    try:
        mod.main()
    finally:
        sys.stdin = old_stdin


def main():
    participants = find_participants()
    if not participants:
        print("No participant folders found.")
        return

    sep = "=" * 78

    # ── Resolve participant(s) from CLI args or interactive prompt ────────────
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("participant", nargs="*", default=[])
    args, _ = parser.parse_known_args()

    def resolve_id(raw):
        """Return a validated participant ID from a raw CLI token."""
        # Strip any leading path (e.g. HITLS/P02 → P02)
        raw = os.path.basename(raw.rstrip("/\\")).strip()
        if raw.isdigit():
            idx = int(raw) - 1
            if not (0 <= idx < len(participants)):
                print(f"Invalid participant number: {raw}")
                sys.exit(1)
            return participants[idx], raw
        pid = raw.upper()
        if pid not in participants:
            print(f"Participant '{pid}' not found.")
            sys.exit(1)
        return pid, pid

    if args.participant:
        selected = [resolve_id(a) for a in args.participant]
    else:
        print(f"\n{sep}")
        print("  HITLS — Questionnaire Analysis Suite")
        print(f"{sep}")
        print("\nAvailable participants:")
        for i, p in enumerate(participants, 1):
            print(f"  {i}. {p}")

        while True:
            choice = input("\nSelect a participant (number or ID, e.g. 1 or P02): ").strip()
            if choice.isdigit():
                idx = int(choice) - 1
                if 0 <= idx < len(participants):
                    selected = [(participants[idx], choice)]
                    break
            elif choice.upper() in participants:
                selected = [(choice.upper(), choice.upper())]
                break
            print(f"  Invalid choice.")

    for participant_id, participant_input in selected:
        print(f"\n{sep}")
        print(f"  Running all questionnaire scripts for {participant_id}")
        print(f"{sep}")

        for label, filename in SCRIPTS:
            print(f"\n{'─'*78}")
            print(f"  ▶  {label}")
            print(f"{'─'*78}")
            try:
                mod = load_module(label, filename)
                run_with_participant(mod, participant_input)
            except Exception as exc:
                print(f"  ✗  {label} failed: {exc}")

        print(f"\n{sep}")
        print(f"  All scripts completed for {participant_id}.")
        print(f"  Reports saved to: HITLS/{participant_id}/cleaned/")
        print(f"{sep}\n")


if __name__ == "__main__":
    main()
