"""
sus.py — System Usability Scale (SUS) Processor
================================================
Scores the 10-item SUS questionnaire (1–5 Likert, alternating scoring).
Final score scaled to 0–100. Reads from HAT_study.csv
(questionnaire_id = "sus").

Typically invoked by `forms/forms.py`, but can be run interactively.
Output: {PID}/cleaned/{PID}_sus_report.txt
"""

import os
import csv
import json
from collections import defaultdict

HITLS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# SUS items: (key, label, odd=positive scoring rule)
# Odd-numbered items: score = response - 1
# Even-numbered items: score = 5 - response
SUS_ITEMS = [
    ("sus_1",  "I think I would like to use this system frequently.",               True),
    ("sus_2",  "I found the system unnecessarily complex.",                          False),
    ("sus_3",  "I thought the system was easy to use.",                              True),
    ("sus_4",  "I would need support of a technical person to use this system.",     False),
    ("sus_5",  "The various functions were well integrated.",                        True),
    ("sus_6",  "There was too much inconsistency in this system.",                   False),
    ("sus_7",  "Most people would learn to use this system very quickly.",           True),
    ("sus_8",  "I found the system very cumbersome to use.",                         False),
    ("sus_9",  "I felt very confident using the system.",                            True),
    ("sus_10", "I needed to learn a lot before I could get going with this system.", False),
]

# SUS adjective ratings (Bangor et al., 2009)
SUS_ADJECTIVES = [
    (85.5, "Best Imaginable"),
    (72.6, "Excellent"),
    (51.7, "Good"),
    (38.0, "OK"),
    (25.1, "Poor"),
    (0.0,  "Worst Imaginable"),
]

CONDITION_ORDER = ["baseline_no_system", "TARS", "TARC", "TARP-S", "TARP-F"]


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


def extract_sus_data(rows):
    """Returns dict {condition: {sus_key: raw_int}}."""
    data = defaultdict(dict)
    for row in rows:
        if row["questionnaire_id"] != "sus":
            continue
        q = row["question_id"].strip()
        try:
            data[row["condition"].strip()][q] = int(row["value"])
        except ValueError:
            pass
    return dict(data)


def score_sus_item(key, raw, odd):
    """Return 0-4 converted score per SUS rules."""
    return (raw - 1) if odd else (5 - raw)


def compute_sus(condition_data):
    """
    Returns (score_0_100, details) where details = list of
    (key, label, raw, converted, odd).
    Score = sum(converted) * 2.5
    """
    details = []
    for key, label, odd in SUS_ITEMS:
        raw = condition_data.get(key)
        if raw is None:
            continue
        converted = score_sus_item(key, raw, odd)
        details.append((key, label, raw, converted, odd))
    if not details:
        return None, []
    score = sum(d[3] for d in details) * 2.5
    return score, details


def adjective_rating(score):
    for threshold, label in SUS_ADJECTIVES:
        if score >= threshold:
            return label
    return "Worst Imaginable"


def bar_item(raw, converted, width=10):
    """raw bar (1-5 scale), converted bar (0-4 scale)."""
    filled_raw  = int(((raw - 1) / 4) * width)
    filled_conv = int((converted / 4) * width)
    return (
        f"[{'█' * filled_raw}{'░' * (width - filled_raw)}]",
        f"[{'█' * filled_conv}{'░' * (width - filled_conv)}]",
    )


def bar_score(score, width=30):
    filled = int((score / 100) * width)
    return f"[{'█' * filled}{'░' * (width - filled)}]"


def build_report(participant_id, all_data):
    lines = []
    sep = "=" * 78

    lines.append(f"\n{sep}")
    lines.append(f"  SYSTEM USABILITY SCALE (SUS) REPORT — {participant_id}")
    lines.append(sep)
    lines.append(
        "\n  Scoring: odd items = raw−1 | even items = 5−raw  → sum × 2.5 → 0–100\n"
    )

    conditions = [c for c in CONDITION_ORDER if c in all_data]
    conditions += [c for c in sorted(all_data) if c not in CONDITION_ORDER]

    for condition in conditions:
        score, details = compute_sus(all_data[condition])

        lines.append(f"\n{'─'*78}")
        lines.append(f"  CONDITION: {condition}")
        lines.append(f"{'─'*78}\n")

        lines.append(
            f"  {'#':<3}  {'Item':<55} {'Raw':>4}  {'Conv':>5}   {'Raw bar':>12}   {'Conv bar'}"
        )
        lines.append(f"  {'─'*3}  {'─'*55}  {'─'*4}  {'─'*5}   {'─'*12}   {'─'*12}")

        for key, label, raw, converted, odd in details:
            num = key.replace("sus_", "")
            valence = "(+)" if odd else "(–)"
            b_raw, b_conv = bar_item(raw, converted)
            display_label = label if len(label) <= 51 else label[:50] + "…"
            lines.append(
                f"  {num:>2}.  {valence} {display_label:<51} {raw:>4}  {converted:>5}   {b_raw}   {b_conv}"
            )

        if score is not None:
            adj = adjective_rating(score)
            lines.append(f"\n  {'─'*60}")
            lines.append(f"  SUS Score    : {score:.1f} / 100  {bar_score(score)}")
            lines.append(f"  Adjective    : {adj}")

    # ── Summary ──────────────────────────────────────────────────────────────
    lines.append(f"\n\n{'─'*78}")
    lines.append("  SUMMARY")
    lines.append(f"{'─'*78}\n")
    lines.append(f"  {'Condition':<25} {'SUS Score':>12}   {'Adjective Rating'}")
    lines.append(f"  {'─'*25}   {'─'*12}   {'─'*20}")
    for condition in conditions:
        score, _ = compute_sus(all_data[condition])
        if score is not None:
            lines.append(f"  {condition:<25}   {score:>8.1f}/100   {adjective_rating(score)}")
        else:
            lines.append(f"  {condition:<25}   {'N/A':>12}")

    lines.append(f"\n{sep}\n")
    return "\n".join(lines)


def build_summary(participant_id, all_data):
    conditions = [c for c in CONDITION_ORDER if c in all_data]
    conditions += [c for c in sorted(all_data) if c not in CONDITION_ORDER]

    summary = {"participant": participant_id, "conditions": {}}
    for condition in conditions:
        score, details = compute_sus(all_data[condition])
        summary["conditions"][condition] = {
            "sus_score": round(score, 4) if score is not None else None,
            "adjective_rating": adjective_rating(score) if score is not None else None,
            "items": {
                d[0]: {"raw": d[2], "converted": d[3]}
                for d in details
            },
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
    all_data = extract_sus_data(rows)

    if not all_data:
        print(f"No SUS data found for {participant_id}.")
        return

    summary = build_summary(participant_id, all_data)
    report  = build_report(participant_id, all_data)
    print(report)

    json_block = (
        "--- MACHINE-READABLE SUMMARY (JSON) ---\n"
        + json.dumps(summary, indent=2, ensure_ascii=False)
        + "\n--- END SUMMARY ---\n"
    )
    out_path = os.path.join(HITLS_DIR, participant_id, "cleaned", f"{participant_id}_sus_report.txt")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(json_block)
        f.write(report)
    print(f"  Report saved to: {os.path.relpath(out_path)}")


if __name__ == "__main__":
    main()
