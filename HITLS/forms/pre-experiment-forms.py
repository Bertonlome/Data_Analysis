import os
import csv
import json

HITLS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

LIKERT_SCORE = {
    "Strongly agree": 5,
    "Agree": 4,
    "Neither agree nor disagree": 3,
    "Disagree": 2,
    "Strongly disagree": 1,
}

# PTS item valences: True = positive, False = negative (reverse-scored)
PTS_VALENCE = [True, False, True, True, True, True, True]

PTS_QUESTION_LABELS = [
    "Q1: I usually trust machines until there is a reason not to.",
    "Q2: For the most part, I distrust machines.",
    "Q3: In general, I would rely on a machine to assist me.",
    "Q4: My tendency to trust machines is high.",
    "Q5: It is easy for me to trust machines to do their job.",
    "Q6: I am likely to trust a machine even when I have little knowledge about it.",
    "Q7: I have a generally positive attitude toward advanced automation in aviation.",
]

GENERAL_FIELD_LABELS = [
    ("Timestamp", 0),
    ("Participant ID", 1),
    ("Age", 2),
    ("Gender", 3),
    ("Primary working language", 4),
    ("Dominant hand", 5),
    ("Aeronautical licences held", 6),
    ("Usual type of operations", 7),
    ("Aircraft type mainly operated", 8),
    ("Total flight hours (approx.)", 9),
    ("Total hours as PIC (approx.)", 10),
    ("Years as professional pilot (approx.)", 11),
    ("Days since last flight", 12),
    ("Exposure to highly automated cockpits", 13),
    ("Automation types used in flight", 14),
    ("Perceived familiarity with advanced automation", 15),
]


def find_participants():
    participants = []
    for entry in sorted(os.listdir(HITLS_DIR)):
        full = os.path.join(HITLS_DIR, entry)
        if os.path.isdir(full) and entry.startswith("P") and entry[1:].isdigit():
            participants.append(entry)
    return participants


def find_csv_for_participant(participant_id):
    """
    Returns (csv_path, participant_id) for the best CSV file to use.
    Priority: participant-specific file > shared Formulaire file in participant folder.
    """
    folder = os.path.join(HITLS_DIR, participant_id)
    # Look for a participant-specific file first
    specific = os.path.join(folder, f"{participant_id}_pre_experiment_form.csv")
    if os.path.isfile(specific):
        return specific

    # Fall back to the shared Formulaire file
    formulaire = os.path.join(folder, "Formulaire pré-expérience ADAIR-POLY.csv")
    if os.path.isfile(formulaire):
        return formulaire

    return None


def load_participant_row(csv_path, participant_id):
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        headers = next(reader)
        for row in reader:
            if len(row) > 1 and row[1].strip() == participant_id:
                return headers, row
    return None, None


def score_pts(row):
    scores = []
    raw = []
    for i, (positive) in enumerate(PTS_VALENCE):
        answer = row[16 + i].strip()
        raw.append(answer)
        likert = LIKERT_SCORE.get(answer)
        if likert is None:
            scores.append(None)
        else:
            scores.append(likert if positive else (6 - likert))
    return raw, scores


def bar(answer, width=30):
    """Return a simple text bar proportional to the Likert score."""
    score = LIKERT_SCORE.get(answer, 0)
    filled = int((score / 5) * width)
    return f"[{'█' * filled}{'░' * (width - filled)}] {answer}"


def build_summary(participant_id, row, raw_pts, pts_scores):
    """Return a dict of key scalars suitable for cross-subject aggregation."""
    valid_scores = [s for s in pts_scores if s is not None]
    pts_item_keys = ["Q1", "Q2", "Q3", "Q4", "Q5", "Q6", "Q7"]
    return {
        "participant": participant_id,
        "age": row[2].strip(),
        "gender": row[3].strip(),
        "total_flight_hours": row[9].strip(),
        "pic_hours": row[10].strip(),
        "years_professional_pilot": row[11].strip(),
        "automation_exposure": row[13].strip(),
        "automation_familiarity": row[15].strip(),
        "pts": {
            pts_item_keys[i]: {
                "raw_answer": raw_pts[i],
                "score": pts_scores[i],
            }
            for i in range(len(pts_item_keys))
        },
        "pts_total": sum(valid_scores) if valid_scores else None,
        "pts_mean": round(sum(valid_scores) / len(valid_scores), 4) if valid_scores else None,
        "pts_max": len(valid_scores) * 5 if valid_scores else None,
    }


def build_report(participant_id, headers, row, raw_pts, pts_scores):
    """Return the full report as a string."""
    lines = []
    sep = "=" * 70

    lines.append(f"\n{sep}")
    lines.append(f"  PRE-EXPERIMENT FORM — {participant_id}")
    lines.append(sep)

    lines.append("\n── GENERAL INFORMATION ─────────────────────────────────────────────\n")
    for lbl, idx in GENERAL_FIELD_LABELS:
        value = row[idx].strip() if idx < len(row) else "N/A"
        lines.append(f"  {lbl:<45} {value}")

    lines.append("\n── PROPENSITY TO TRUST SCALE (PTS) ─────────────────────────────────\n")
    lines.append("  (5-point Likert | Q2 is reverse-scored)\n")

    for i, (lbl, answer, score) in enumerate(
        zip(PTS_QUESTION_LABELS, raw_pts, pts_scores)
    ):
        valence_tag = "(–)" if not PTS_VALENCE[i] else "(+)"
        score_str = f"→ score: {score}/5" if score is not None else "→ score: N/A"
        lines.append(f"  {valence_tag} {lbl}")
        lines.append(f"       {bar(answer)}  {score_str}")
        lines.append("")

    valid_scores = [s for s in pts_scores if s is not None]
    if valid_scores:
        total = sum(valid_scores)
        maximum = len(valid_scores) * 5
        mean = total / len(valid_scores)
        lines.append(f"  {'─'*60}")
        lines.append(f"  PTS Total Score : {total} / {maximum}")
        lines.append(f"  PTS Mean Score  : {mean:.2f} / 5.00")
        if mean >= 4.0:
            interp = "High propensity to trust"
        elif mean >= 3.0:
            interp = "Moderate propensity to trust"
        elif mean >= 2.0:
            interp = "Low propensity to trust"
        else:
            interp = "Very low propensity to trust"
        lines.append(f"  Interpretation  : {interp}")
    lines.append(f"\n{sep}\n")
    return "\n".join(lines)


def print_report(participant_id, headers, row, raw_pts, pts_scores):
    summary = build_summary(participant_id, row, raw_pts, pts_scores)
    report = build_report(participant_id, headers, row, raw_pts, pts_scores)
    print(report)

    json_block = (
        "--- MACHINE-READABLE SUMMARY (JSON) ---\n"
        + json.dumps(summary, indent=2, ensure_ascii=False)
        + "\n--- END SUMMARY ---\n"
    )
    out_path = os.path.join(HITLS_DIR, participant_id, "cleaned", f"{participant_id}_pre_experiment_form_report.txt")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(json_block)
        f.write(report)
    print(f"  Report saved to: {os.path.relpath(out_path)}")


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

    csv_path = find_csv_for_participant(participant_id)
    if csv_path is None:
        print(f"No pre-experiment form CSV found for {participant_id}.")
        return

    headers, row = load_participant_row(csv_path, participant_id)
    if row is None:
        print(f"No row found for {participant_id} in {csv_path}.")
        return

    raw_pts, pts_scores = score_pts(row)
    print_report(participant_id, headers, row, raw_pts, pts_scores)


if __name__ == "__main__":
    main()
