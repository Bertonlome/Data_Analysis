"""
trust-risk.py — Trust & Risk Visual Analogue Scale (VAS) Processor
===================================================================
Processes two VAS items (0–100 slider) measuring overall trust in the
automation system and perceived risk. Reads from HAT_study.csv
(questionnaire_id = "trust_risk").

Typically invoked by `forms/forms.py`, but can be run interactively.
Output: {PID}/cleaned/{PID}_trust_risk_report.txt
"""

import os
import csv
import json
from collections import defaultdict

HITLS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Two VAS items (0–100) from trust_risk_bespoke questionnaire.
# No aggregation — each item is reported independently per condition.
# Fields: (key, label_en)
ITEMS = [
    ("trust_vas", "How would you rate your level of trust in the system?"),
    ("risk_vas",  "How would you rate the level of perceived risk associated with using the system?"),
]

CONDITION_ORDER = ["baseline_no_system", "TARS", "TARC", "TARP-S", "TARP-F"]

VAS_MIN, VAS_MAX = 0, 100


def find_participants():
    return [
        e for e in sorted(os.listdir(HITLS_DIR))
        if os.path.isdir(os.path.join(HITLS_DIR, e))
        and e.startswith("P") and e[1:].isdigit()
    ]


def find_forms_csv(participant_id):
    folder = os.path.join(HITLS_DIR, participant_id)
    specific = os.path.join(folder, f"{participant_id}_forms.csv")
    if os.path.isfile(specific):
        return specific
    for name in sorted(os.listdir(folder)):
        if "hat_study" in name.lower() and name.endswith(".csv"):
            return os.path.join(folder, name)
    return None


def load_rows(csv_path):
    with open(csv_path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def extract_data(rows):
    """Returns dict {condition: {question_id: int}}."""
    data = defaultdict(dict)
    for row in rows:
        if row["questionnaire_id"].strip() != "trust_risk_bespoke":
            continue
        q = row["question_id"].strip()
        try:
            data[row["condition"].strip()][q] = int(row["value"])
        except ValueError:
            pass
    return dict(data)


def bar(value, width=30):
    filled = int((value / VAS_MAX) * width)
    return f"[{'█' * filled}{'░' * (width - filled)}]"


def build_report(participant_id, all_data):
    lines = []
    sep = "=" * 78

    lines.append(f"\n{sep}")
    lines.append(f"  TRUST & PERCEIVED RISK (VAS) REPORT — {participant_id}")
    lines.append(f"  trust_risk_bespoke — Visual Analogue Scale 0–100")
    lines.append(sep)
    lines.append(
        "\n  Items are reported independently (no aggregation).\n"
        "  Scale: 0 = not at all / no risk  →  100 = completely / extremely high risk\n"
    )

    conditions = [c for c in CONDITION_ORDER if c in all_data]
    conditions += [c for c in sorted(all_data) if c not in CONDITION_ORDER]

    for condition in conditions:
        cdata = all_data[condition]
        lines.append(f"{'─'*78}")
        lines.append(f"  CONDITION: {condition}")
        lines.append(f"{'─'*78}")
        for key, label in ITEMS:
            value = cdata.get(key)
            if value is None:
                lines.append(f"\n  {label}")
                lines.append(f"    Score : N/A")
            else:
                lines.append(f"\n  {label}")
                lines.append(f"    Score : {value:>3} / 100   {bar(value)}")
        lines.append("")

    # ── Summary table ─────────────────────────────────────────────────────────
    lines.append(f"{'─'*78}")
    lines.append("  SUMMARY")
    lines.append(f"{'─'*78}\n")

    col_w = 12
    header = f"  {'Condition':<22}" + "".join(
        f"  {key[:col_w]:>{col_w}}" for key, _ in ITEMS
    )
    lines.append(header)
    lines.append("  " + "─" * (len(header) - 2))

    for condition in conditions:
        cdata = all_data[condition]
        row_str = f"  {condition:<22}"
        for key, _ in ITEMS:
            v = cdata.get(key)
            row_str += f"  {(str(v) + '/100'):>{col_w}}" if v is not None else f"  {'N/A':>{col_w}}"
        lines.append(row_str)

    lines.append(f"\n{sep}\n")
    return "\n".join(lines)


def build_summary(participant_id, all_data):
    conditions = [c for c in CONDITION_ORDER if c in all_data]
    conditions += [c for c in sorted(all_data) if c not in CONDITION_ORDER]

    summary = {"participant": participant_id, "conditions": {}}
    for condition in conditions:
        cdata = all_data[condition]
        summary["conditions"][condition] = {
            key: cdata.get(key) for key, _ in ITEMS
        }
    return summary


def main():
    participants = find_participants()
    if not participants:
        print("No participant folders found in HITLS directory.")
        return

    print("\nAvailable participants:")
    for i, p in enumerate(participants, 1):
        print(f"  {i}. {p}")

    while True:
        choice = input("\nSelect a participant (number or ID, e.g. 1 or P02): ").strip()
        if choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(participants):
                participant_id = participants[idx]
                break
        elif choice.upper() in participants:
            participant_id = choice.upper()
            break
        print(f"  Invalid choice. Please enter a number between 1 and {len(participants)} or a valid ID.")

    csv_path = find_forms_csv(participant_id)
    if csv_path is None:
        print(f"No forms CSV found for {participant_id}.")
        return

    rows = load_rows(csv_path)
    all_data = extract_data(rows)

    if not all_data:
        print(f"No trust/risk VAS data found for {participant_id}.")
        return

    summary = build_summary(participant_id, all_data)
    report  = build_report(participant_id, all_data)
    print(report)

    json_block = (
        "--- MACHINE-READABLE SUMMARY (JSON) ---\n"
        + json.dumps(summary, indent=2, ensure_ascii=False)
        + "\n--- END SUMMARY ---\n"
    )
    out_path = os.path.join(
        HITLS_DIR, participant_id, "cleaned", f"{participant_id}_trust_risk_report.txt"
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(json_block)
        f.write(report)
    print(f"  Report saved to: {os.path.relpath(out_path)}")


if __name__ == "__main__":
    main()
