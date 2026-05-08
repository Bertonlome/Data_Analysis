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
"""

import os
import sys
import io
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
                participant_id = participants[idx]
                participant_input = choice
                break
        elif choice.upper() in participants:
            participant_id = choice.upper()
            participant_input = choice.upper()
            break
        print(f"  Invalid choice.")

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
