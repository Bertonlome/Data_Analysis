import os
import csv
import json
from collections import defaultdict

HITLS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Maps display names used in pairwise comparisons to question_id keys
DISPLAY_TO_KEY = {
    "Mental Demand":    "mental_demand",
    "Physical Demand":  "physical_demand",
    "Temporal Demand":  "temporal_demand",
    "Performance":      "performance",
    "Effort":           "effort",
    "Frustration":      "frustration",
}

SUBSCALE_LABELS = {
    "mental_demand":   "Mental Demand",
    "physical_demand": "Physical Demand",
    "temporal_demand": "Temporal Demand",
    "performance":     "Performance",
    "effort":          "Effort",
    "frustration":     "Frustration",
}

CONDITION_ORDER = [
    "baseline_no_system",
    "TARS",
    "TARC",
    "TARP-S",
    "TARP-F",
]


def find_participants():
    return [
        e for e in sorted(os.listdir(HITLS_DIR))
        if os.path.isdir(os.path.join(HITLS_DIR, e))
        and e.startswith("P") and e[1:].isdigit()
    ]


def find_forms_csv(participant_id):
    folder = os.path.join(HITLS_DIR, participant_id)
    # Priority 1: participant-specific naming P0X_forms.csv
    specific = os.path.join(folder, f"{participant_id}_forms.csv")
    if os.path.isfile(specific):
        return specific
    # Priority 2: any file matching HAT_study*.csv or briefing_export*.csv
    # (HAT_study files contain the same long-form data)
    for name in sorted(os.listdir(folder)):
        lower = name.lower()
        if "hat_study" in lower and name.endswith(".csv"):
            return os.path.join(folder, name)
    return None


def load_rows(csv_path):
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def extract_weights(rows):
    """
    Tally how many times each subscale was chosen across the 15 pairwise
    comparisons (nasa_tlx_subscale_ranking, condition after_familiarization).
    Returns dict {question_key: weight (0-5)}.
    """
    tally = defaultdict(int)
    for row in rows:
        if (row["questionnaire_id"] == "nasa_tlx_subscale_ranking"
                and row["condition"] == "after_familiarization"):
            chosen = row["value"].strip()
            key = DISPLAY_TO_KEY.get(chosen)
            if key:
                tally[key] += 1
    return dict(tally)


def extract_ratings(rows):
    """
    Returns dict {condition: {question_key: rating_float}}.
    """
    ratings = defaultdict(dict)
    for row in rows:
        if row["questionnaire_id"] == "nasa_tlx_evaluation":
            condition = row["condition"].strip()
            q = row["question_id"].strip()
            try:
                ratings[condition][q] = float(row["value"])
            except ValueError:
                pass
    return dict(ratings)


def compute_nasa_tlx(weights, ratings_for_condition):
    """
    NASA-TLX weighted score = sum(weight_i * (rating_i * 5)) / 15
    Ratings are stored as 0-20 interval indices; multiply by 5 to get 0-100.
    Returns (score, per_subscale_detail) or (None, {}) if data missing.
    """
    details = {}
    total_weighted = 0.0
    for key in SUBSCALE_LABELS:
        w = weights.get(key, 0)
        r = ratings_for_condition.get(key)
        if r is None:
            continue
        r100 = r * 5          # convert 0-20 → 0-100
        weighted = w * r100
        total_weighted += weighted
        details[key] = {"weight": w, "rating": r, "rating100": r100, "weighted": weighted}
    if not details:
        return None, {}
    score = total_weighted / 15.0
    return score, details


def bar_rating(value, max_val=20, width=20):
    filled = int((value / max_val) * width)
    return f"[{'█' * filled}{'░' * (width - filled)}] {value:>4.0f}/20"


def bar_weight(value, max_val=5, width=10):
    filled = int((value / max_val) * width) if max_val > 0 else 0
    return f"[{'█' * filled}{'░' * (width - filled)}] {value}/5"


def build_summary(participant_id, weights, all_ratings, conditions):
    """Return a dict of key scalars suitable for cross-subject aggregation."""
    summary = {
        "participant": participant_id,
        "weights": {key: weights.get(key, 0) for key in SUBSCALE_LABELS},
        "conditions": {},
    }
    for condition in conditions:
        score, details = compute_nasa_tlx(weights, all_ratings[condition])
        summary["conditions"][condition] = {
            "nasa_tlx_weighted_score": round(score, 4) if score is not None else None,
            "subscales": {
                key: {
                    "rating_0_20": details[key]["rating"],
                    "rating_0_100": details[key]["rating100"],
                    "weight": details[key]["weight"],
                    "weighted": round(details[key]["weighted"], 4),
                }
                for key in SUBSCALE_LABELS
                if key in details
            },
        }
    return summary


def build_report(participant_id, weights, all_ratings):
    lines = []
    sep = "=" * 72

    lines.append(f"\n{sep}")
    lines.append(f"  NASA-TLX REPORT — {participant_id}")
    lines.append(sep)

    # ── Weights ──────────────────────────────────────────────────────────────
    lines.append("\n── PAIRWISE WEIGHTS (collected after familiarization) ───────────────────\n")
    lines.append(f"  {'Subscale':<20} {'Weight (tally/15 pairs)'}")
    lines.append(f"  {'─'*20}   {'─'*35}")
    for key, label in SUBSCALE_LABELS.items():
        w = weights.get(key, 0)
        lines.append(f"  {label:<20}   {bar_weight(w)}  ({w} selections)")
    lines.append(f"\n  Total tallied: {sum(weights.get(k,0) for k in SUBSCALE_LABELS)} / 15")

    # ── Per-condition scores ──────────────────────────────────────────────────
    conditions = [c for c in CONDITION_ORDER if c in all_ratings]
    # Also include any conditions present but not in our order list
    extra = [c for c in sorted(all_ratings) if c not in CONDITION_ORDER]
    conditions += extra

    for condition in conditions:
        ratings = all_ratings[condition]
        score, details = compute_nasa_tlx(weights, ratings)

        lines.append(f"\n── CONDITION: {condition} {'─' * (55 - len(condition))}\n")
        lines.append(
            f"  {'Subscale':<20} {'Rating':>14}   {'Weight':>7}   {'Weighted':>9}   {'Bar (rating/20)'}"
        )
        lines.append(f"  {'─'*20}   {'─'*14}   {'─'*7}   {'─'*9}   {'─'*25}")

        for key, label in SUBSCALE_LABELS.items():
            if key not in details:
                continue
            d = details[key]
            lines.append(
                f"  {label:<20}   {d['rating']:>4.0f}/20 ({d['rating100']:>3.0f}/100)"
                f"   {d['weight']:>5}/5   {d['weighted']:>7.1f}    {bar_rating(d['rating'])}"
            )

        if score is not None:
            lines.append(f"\n  {'─'*60}")
            lines.append(f"  NASA-TLX Weighted Score : {score:.2f} / 100")
            if score >= 80:
                interp = "Very high workload"
            elif score >= 60:
                interp = "High workload"
            elif score >= 40:
                interp = "Moderate workload"
            elif score >= 20:
                interp = "Low workload"
            else:
                interp = "Very low workload"
            lines.append(f"  Interpretation          : {interp}")

    # ── Summary table ─────────────────────────────────────────────────────────
    lines.append(f"\n── SUMMARY ─────────────────────────────────────────────────────────────\n")
    lines.append(f"  {'Condition':<25} {'NASA-TLX Score':>16}")
    lines.append(f"  {'─'*25}   {'─'*16}")
    for condition in conditions:
        ratings = all_ratings[condition]
        score, _ = compute_nasa_tlx(weights, ratings)
        score_str = f"{score:.2f} / 100" if score is not None else "N/A"
        lines.append(f"  {condition:<25}   {score_str:>16}")

    lines.append(f"\n{sep}\n")
    return "\n".join(lines)


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
    weights = extract_weights(rows)
    all_ratings = extract_ratings(rows)

    if not weights:
        print(f"No pairwise weight data found for {participant_id}.")
        return
    if not all_ratings:
        print(f"No NASA-TLX rating data found for {participant_id}.")
        return

    # Determine ordered conditions list (same logic as build_report)
    conditions = [c for c in CONDITION_ORDER if c in all_ratings]
    conditions += [c for c in sorted(all_ratings) if c not in CONDITION_ORDER]

    summary = build_summary(participant_id, weights, all_ratings, conditions)
    report = build_report(participant_id, weights, all_ratings)
    print(report)

    json_block = (
        "--- MACHINE-READABLE SUMMARY (JSON) ---\n"
        + json.dumps(summary, indent=2, ensure_ascii=False)
        + "\n--- END SUMMARY ---\n"
    )
    out_path = os.path.join(HITLS_DIR, participant_id, "cleaned", f"{participant_id}_nasa_tlx_report.txt")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(json_block)
        f.write(report)
    print(f"  Report saved to: {os.path.relpath(out_path)}")


if __name__ == "__main__":
    main()
