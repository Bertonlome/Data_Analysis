"""
oversight-bespoke.py — Oversight Bespoke Scale Processor
=========================================================
Scores a custom bespoke questionnaire measuring perceived oversight
behaviour across automation conditions (5-point Likert items).
Reads from HAT_study.csv (questionnaire_id = "oversight_bespoke").

Typically invoked by `forms/forms.py`, but can be run interactively.
Output: {PID}/cleaned/{PID}_oversight_bespoke_report.txt
"""

import os
import csv
import json
from collections import defaultdict

HITLS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SCALE_MIN = 1
SCALE_MAX = 5

# Item definitions: (key, label, positive_valence, in_extended_set)
# Labels sourced from oversight_bespoke.yaml (label_en).
# in_extended_set=True means the item is marked with * and included only in the
# extended score computation; the base score uses all non-starred items.
ITEMS = [
    ("ob_a2", "A2 – Enough time/attention to oversee",          True,  False),
    ("ob_a3", "A3 – Difficult to follow system actions",         False, False),
    ("ob_b1", "B1 – Actively verified system's actions",         True,  False),
    ("ob_b2", "B2 – No need to monitor closely",                 True,  True),
    ("ob_b3", "B3 – May have missed check opportunities",        False, False),
    ("ob_c1", "C1 – Could detect system errors quickly",         True,  False),
    ("ob_c2", "C2 – Enough time to react to actions",            True,  False),
    ("ob_d1", "D1 – System helped work efficiently",             True,  True),
    ("ob_d2", "D2 – Efficiency came at cost of oversight",       False, True),
]

# Items used in base scoring (no *)
BASE_ITEMS  = [(k, lbl, pos) for k, lbl, pos, ext in ITEMS if not ext]
# Items used in extended scoring (base + *)
EXT_ITEMS   = [(k, lbl, pos) for k, lbl, pos, ext in ITEMS]
# Starred items only (for display)
STAR_ITEMS  = {k for k, lbl, pos, ext in ITEMS if ext}

CONDITION_ORDER = ["TARS", "TARC", "TARP-S", "TARP-F"]


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


def extract_ob_data(rows):
    """
    Returns dict {condition: {question_key: raw_int}}.
    Excludes the attention-check item.
    """
    data = defaultdict(dict)
    for row in rows:
        if row["questionnaire_id"] != "oversight_bespoke":
            continue
        q = row["question_id"].strip()
        if q == "ob_attention_check":
            continue
        try:
            data[row["condition"].strip()][q] = int(row["value"])
        except ValueError:
            pass
    return dict(data)


def score_item(raw, positive):
    """Return scored value (reverse-score negative items)."""
    return raw if positive else (SCALE_MAX + SCALE_MIN - raw)


def compute_ob_score(condition_data, item_list):
    """
    Compute mean scored value over item_list for a single condition.
    Returns (mean, details_list) or (None, []).
    details_list = [(key, label, raw, scored, positive), ...]
    """
    details = []
    for key, label, positive in item_list:
        raw = condition_data.get(key)
        if raw is None:
            continue
        scored = score_item(raw, positive)
        details.append((key, label, raw, scored, positive))
    if not details:
        return None, []
    mean = sum(d[3] for d in details) / len(details)
    return mean, details


def bar_item(raw, scored, width=10):
    filled_raw    = int(((raw    - SCALE_MIN) / (SCALE_MAX - SCALE_MIN)) * width)
    filled_scored = int(((scored - SCALE_MIN) / (SCALE_MAX - SCALE_MIN)) * width)
    b_raw    = f"[{'█' * filled_raw}{'░' * (width - filled_raw)}]"
    b_scored = f"[{'█' * filled_scored}{'░' * (width - filled_scored)}]"
    return b_raw, b_scored


def build_condition_block(condition, condition_data, item_list, score_label):
    lines = []
    mean, details = compute_ob_score(condition_data, item_list)
    header = f"  {'Item':<45} {'Val':>4}   {'Scored':>6}   {'Raw bar':>12}   {'Scored bar'}"
    lines.append(header)
    lines.append(f"  {'─'*45}   {'─'*4}   {'─'*6}   {'─'*12}   {'─'*12}")
    for key, label, raw, scored, positive in details:
        star = "*" if key in STAR_ITEMS else " "
        valence = "(+)" if positive else "(–)"
        b_raw, b_scored = bar_item(raw, scored)
        lines.append(
            f"  {star}{valence} {label:<41} {raw:>4}   {scored:>6}   {b_raw}   {b_scored}"
        )
    if mean is not None:
        lines.append(f"\n  {'─'*60}")
        lines.append(f"  {score_label} Mean Score : {mean:.3f} / {SCALE_MAX:.1f}")
    return lines, mean


def build_report(participant_id, all_data):
    lines = []
    sep = "=" * 72

    lines.append(f"\n{sep}")
    lines.append(f"  OVERSIGHT BESPOKE (OB) REPORT — {participant_id}")
    lines.append(sep)

    lines.append(
        "\n  Items marked (*) are included only in the Extended score.\n"
        "  Negative-valence items (–) are reverse-scored.\n"
    )

    conditions = [c for c in CONDITION_ORDER if c in all_data]
    conditions += [c for c in sorted(all_data) if c not in CONDITION_ORDER]

    for condition in conditions:
        cdata = all_data[condition]

        lines.append(f"\n{'─'*72}")
        lines.append(f"  CONDITION: {condition}")
        lines.append(f"{'─'*72}\n")

        # BASE score
        lines.append("  ── BASE SCORE (items without *) ──────────────────────────────────\n")
        base_lines, base_mean = build_condition_block(
            condition, cdata, BASE_ITEMS, "Base OB"
        )
        lines.extend(base_lines)

        # EXTENDED score
        lines.append("\n  ── EXTENDED SCORE (all items including *) ────────────────────────\n")
        ext_lines, ext_mean = build_condition_block(
            condition, cdata, EXT_ITEMS, "Extended OB"
        )
        lines.extend(ext_lines)

    # ── Summary ──────────────────────────────────────────────────────────────
    lines.append(f"\n\n{'─'*72}")
    lines.append("  SUMMARY")
    lines.append(f"{'─'*72}\n")
    lines.append(f"  {'Condition':<20} {'Base Score (/{MAX})':>22}   {'Extended Score (/{MAX})':>24}".format(MAX=SCALE_MAX))
    lines.append(f"  {'─'*20}   {'─'*22}   {'─'*24}")
    for condition in conditions:
        cdata = all_data[condition]
        base_mean, _ = compute_ob_score(cdata, BASE_ITEMS)
        ext_mean,  _ = compute_ob_score(cdata, EXT_ITEMS)
        base_str = f"{base_mean:.3f} / {SCALE_MAX}" if base_mean is not None else "N/A"
        ext_str  = f"{ext_mean:.3f}  / {SCALE_MAX}" if ext_mean  is not None else "N/A"
        lines.append(f"  {condition:<20}   {base_str:>22}   {ext_str:>24}")

    lines.append(f"\n{sep}\n")
    return "\n".join(lines)


def build_summary(participant_id, all_data):
    conditions = [c for c in CONDITION_ORDER if c in all_data]
    conditions += [c for c in sorted(all_data) if c not in CONDITION_ORDER]

    summary = {"participant": participant_id, "conditions": {}}
    for condition in conditions:
        cdata = all_data[condition]
        base_mean, base_details = compute_ob_score(cdata, BASE_ITEMS)
        ext_mean,  ext_details  = compute_ob_score(cdata, EXT_ITEMS)

        summary["conditions"][condition] = {
            "base_score": {
                "mean": round(base_mean, 4) if base_mean is not None else None,
                "items": {
                    d[0]: {"raw": d[2], "scored": d[3]} for d in base_details
                },
            },
            "extended_score": {
                "mean": round(ext_mean, 4) if ext_mean is not None else None,
                "items": {
                    d[0]: {"raw": d[2], "scored": d[3]} for d in ext_details
                },
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
    all_data = extract_ob_data(rows)

    if not all_data:
        print(f"No oversight_bespoke data found for {participant_id}.")
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
        HITLS_DIR, participant_id, "cleaned", f"{participant_id}_oversight_bespoke_report.txt"
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(json_block)
        f.write(report)
    print(f"  Report saved to: {os.path.relpath(out_path)}")


if __name__ == "__main__":
    main()
