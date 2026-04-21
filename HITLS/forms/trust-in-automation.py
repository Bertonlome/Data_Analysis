import os
import csv
import json
from collections import defaultdict

HITLS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# TiA items as deployed in this study (12-item subset, Körber 2019).
# Subscale assignment and inversion follow the official manual.
# Propensity to Trust is NOT included — measured once in the pre-experiment
# form (PTS) and loaded separately as a trait constant.
#
# Per manual: inverted items are #7(U/P2*), #10(R/C3*), #15(R/C5*), #16(U/P4*).
# Fields: (key, manual_item_num, label, subscale_key, inverted)
TIA_ITEMS = [
    ("kta_01",  1, "The system is capable of interpreting situations correctly.", "reliability_competence",       False),
    ("kta_02",  2, "The system state was always clear to me.",                    "understanding_predictability", False),
    ("kta_03",  6, "The system works reliably.",                                  "reliability_competence",       False),
    ("kta_04",  7, "The system reacts unpredictably.",                            "understanding_predictability", True),
    ("kta_05",  9, "I trust the system.",                                         "trust_in_automation",          False),
    ("kta_06", 10, "A system malfunction is likely.",                             "reliability_competence",       True),
    ("kta_07", 11, "I was able to understand why things happened.",               "understanding_predictability", False),
    ("kta_08", 13, "The system is capable of taking over complicated tasks.",     "reliability_competence",       False),
    ("kta_09", 14, "I can rely on the system.",                                   "trust_in_automation",          False),
    ("kta_10", 15, "The system might make sporadic errors.",                      "reliability_competence",       True),
    ("kta_11", 16, "It is difficult to identify what the system will do next.",   "understanding_predictability", True),
    ("kta_12", 19, "I am confident about the system's capabilities.",             "reliability_competence",       False),
]

SUBSCALES = {
    "reliability_competence":       "Reliability / Competence",
    "understanding_predictability": "Understanding / Predictability",
    "trust_in_automation":          "Trust in Automation",
}

CONDITION_ORDER = ["baseline_no_system", "TARS", "TARC", "TARP-S", "TARP-F"]

SCALE_MIN, SCALE_MAX = 1, 5


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


def extract_familiarity(rows):
    """Return the `used_similar` score (1–5) from system_familiarity/after_briefing, or None."""
    for row in rows:
        if (
            row["questionnaire_id"].strip() == "system_familiarity"
            and row["question_id"].strip() == "used_similar"
        ):
            try:
                return int(row["value"])
            except ValueError:
                return None
    return None


def load_pts_mean(participant_id):
    """Load pts_mean from pre-experiment JSON summary. Returns float or None."""
    report_path = os.path.join(
        HITLS_DIR, participant_id, "cleaned",
        f"{participant_id}_pre_experiment_form_report.txt"
    )
    if not os.path.isfile(report_path):
        return None
    try:
        txt = open(report_path, encoding="utf-8").read()
        start = txt.index("{")
        end = txt.index("--- END SUMMARY ---")
        summary = json.loads(txt[start:end].strip())
        return float(summary["pts_mean"])
    except (ValueError, KeyError):
        return None


def extract_tia_data(rows):
    """Returns dict {condition: {kta_key: raw_int}}."""
    data = defaultdict(dict)
    for row in rows:
        if row["questionnaire_id"] != "trust_in_automation_scale_korber_et_al_2015":
            continue
        q = row["question_id"].strip()
        if q == "kta_attention_check":
            continue
        try:
            data[row["condition"].strip()][q] = int(row["value"])
        except ValueError:
            pass
    return dict(data)


def recode(raw, inverted):
    return (SCALE_MAX + SCALE_MIN - raw) if inverted else raw


def compute_subscales(condition_data, tia_items):
    """
    Returns:
        subscale_results: {subscale_key: {"mean": float, "items": [detail, ...]}}
        global_mean_excl: float — mean of R/C, U/P, TiA subscale means (no PTS)
    detail = (key, item_num, label, raw, recoded, inverted)
    """
    subscale_items = defaultdict(list)

    for key, num, label, sk, inverted in tia_items:
        raw = condition_data.get(key)
        if raw is None:
            continue
        rec = recode(raw, inverted)
        subscale_items[sk].append((key, num, label, raw, rec, inverted))

    subscale_results = {}
    for sk, items in subscale_items.items():
        mean = sum(d[4] for d in items) / len(items)
        subscale_results[sk] = {"mean": mean, "items": items}

    # Global mean excluding PTS (per-condition mean of subscale means)
    sub_means = [res["mean"] for res in subscale_results.values()]
    global_mean_excl = sum(sub_means) / len(sub_means) if sub_means else None
    return subscale_results, global_mean_excl


def bar(value, width=12):
    filled = int(((value - SCALE_MIN) / (SCALE_MAX - SCALE_MIN)) * width)
    return f"[{'█' * filled}{'░' * (width - filled)}]"


def bar_mean(mean, width=20):
    filled = int(((mean - SCALE_MIN) / (SCALE_MAX - SCALE_MIN)) * width)
    return f"[{'█' * filled}{'░' * (width - filled)}]"


def build_report(participant_id, all_data, tia_items, pts_mean, familiarity_score):
    lines = []
    sep = "=" * 78

    lines.append(f"\n{sep}")
    lines.append(f"  TRUST IN AUTOMATION (TiA) REPORT — {participant_id}")
    lines.append(f"  Körber (2019) — 12-item subset")
    lines.append(sep)
    lines.append(
        "\n  Inverted items (*) are recoded: score = 6 − raw.\n"
        "  Each subscale is reported as mean (1–5).\n"
        "  Trait-level subscales (constant across conditions) are shown below\n"
        "  and included in the G.incl global mean alongside condition subscales.\n"
    )

    fam_str = f"{familiarity_score:.1f} / 5   {bar_mean(familiarity_score)}" if familiarity_score is not None else "not available"
    pts_str = f"{pts_mean:.3f} / 5   {bar_mean(pts_mean)}" if pts_mean is not None else "not available"
    lines.append(f"  Familiarity (after_briefing, trait) : {fam_str}")
    lines.append(f"  PTS         (pre-experiment, trait) : {pts_str}\n")

    conditions = [c for c in CONDITION_ORDER if c in all_data]
    conditions += [c for c in sorted(all_data) if c not in CONDITION_ORDER]

    for condition in conditions:
        sub_results, global_excl = compute_subscales(all_data[condition], tia_items)

        traits = [v for v in [familiarity_score, pts_mean] if v is not None]
        if traits and global_excl is not None:
            global_incl = (
                sum(res["mean"] for res in sub_results.values()) + sum(traits)
            ) / (len(sub_results) + len(traits))
        else:
            global_incl = None

        lines.append(f"\n{'─'*78}")
        lines.append(f"  CONDITION: {condition}")
        lines.append(f"{'─'*78}\n")

        for sk, slabel in SUBSCALES.items():
            if sk not in sub_results:
                continue
            result = sub_results[sk]
            lines.append(f"  ── {slabel} ({'─' * (55 - len(slabel))})\n")
            lines.append(
                f"  {'#':>2}  {'inv':>3}  {'Item':<52} {'Raw':>4}  {'Rec':>4}   {'Bar'}"
            )
            lines.append(f"  {'─'*2}  {'─'*3}  {'─'*52}  {'─'*4}  {'─'*4}   {'─'*14}")
            for key, num, label, raw, rec, inverted in result["items"]:
                inv_tag = " (*)" if inverted else "    "
                display = label if len(label) <= 52 else label[:51] + "…"
                lines.append(
                    f"  {num:>2}{inv_tag}  {display:<52} {raw:>4}  {rec:>4}   {bar(rec)}"
                )
            mean = result["mean"]
            lines.append(
                f"\n  {'Subscale mean':>58} {mean:>5.3f} / 5   {bar_mean(mean)}\n"
            )

        lines.append(f"  {'─'*60}")
        if global_excl is not None:
            lines.append(
                f"  Global mean  (excl. traits)        : {global_excl:.3f} / 5   {bar_mean(global_excl)}"
            )
        if global_incl is not None:
            lines.append(
                f"  Global mean  (incl. Fam + PTS)     : {global_incl:.3f} / 5   {bar_mean(global_incl)}"
            )

    # ── Summary table ─────────────────────────────────────────────────────────
    lines.append(f"\n\n{'─'*78}")
    lines.append("  SUMMARY — Subscale Means (1–5)")
    lines.append(f"{'─'*78}\n")

    has_traits = (familiarity_score is not None or pts_mean is not None)
    col_w = 10
    sk_list = list(SUBSCALES.keys())
    sk_headers = [SUBSCALES[sk][:col_w] for sk in sk_list]
    header = (
        f"  {'Condition':<20}"
        + "".join(f"  {h:>{col_w}}" for h in sk_headers)
        + f"  {'G.excl':>{col_w}}"
        + (f"  {'G.incl':>{col_w}}" if has_traits else "")
    )
    lines.append(header)
    lines.append("  " + "─" * (len(header) - 2))

    for condition in conditions:
        sub_results, global_excl = compute_subscales(all_data[condition], tia_items)
        traits = [v for v in [familiarity_score, pts_mean] if v is not None]
        if traits and global_excl is not None:
            global_incl = (
                sum(res["mean"] for res in sub_results.values()) + sum(traits)
            ) / (len(sub_results) + len(traits))
        else:
            global_incl = None

        row_str = f"  {condition:<20}"
        for sk in sk_list:
            if sk in sub_results:
                row_str += f"  {sub_results[sk]['mean']:>{col_w}.3f}"
            else:
                row_str += f"  {'N/A':>{col_w}}"
        ge_str = f"{global_excl:.3f}" if global_excl is not None else "N/A"
        row_str += f"  {ge_str:>{col_w}}"
        if has_traits:
            gi_str = f"{global_incl:.3f}" if global_incl is not None else "N/A"
            row_str += f"  {gi_str:>{col_w}}"
        lines.append(row_str)

    lines.append(f"\n{sep}\n")
    return "\n".join(lines)


def build_summary(participant_id, all_data, tia_items, pts_mean, familiarity_score):
    conditions = [c for c in CONDITION_ORDER if c in all_data]
    conditions += [c for c in sorted(all_data) if c not in CONDITION_ORDER]

    summary = {
        "participant": participant_id,
        "familiarity_after_briefing": familiarity_score,
        "pts_mean_pre_experiment": round(pts_mean, 4) if pts_mean is not None else None,
        "conditions": {},
    }
    for condition in conditions:
        sub_results, global_excl = compute_subscales(all_data[condition], tia_items)
        traits = [v for v in [familiarity_score, pts_mean] if v is not None]
        if traits and global_excl is not None:
            global_incl = (
                sum(res["mean"] for res in sub_results.values()) + sum(traits)
            ) / (len(sub_results) + len(traits))
        else:
            global_incl = None
        summary["conditions"][condition] = {
            "global_mean_excl_traits":  round(global_excl, 4) if global_excl is not None else None,
            "global_mean_incl_traits":  round(global_incl, 4) if global_incl is not None else None,
            "subscales": {
                sk: {
                    "mean": round(res["mean"], 4),
                    "items": {
                        d[0]: {"raw": d[3], "recoded": d[4]}
                        for d in res["items"]
                    },
                }
                for sk, res in sub_results.items()
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
    all_data = extract_tia_data(rows)

    if not all_data:
        print(f"No TiA data found for {participant_id}.")
        return

    familiarity_score = extract_familiarity(rows)
    if familiarity_score is None:
        print(f"  Note: no system_familiarity/used_similar entry found for {participant_id}.")

    pts_mean = load_pts_mean(participant_id)
    if pts_mean is None:
        print(f"  Note: pre-experiment PTS report not found for {participant_id} — global mean will exclude propensity.")

    summary = build_summary(participant_id, all_data, TIA_ITEMS, pts_mean, familiarity_score)
    report  = build_report(participant_id, all_data, TIA_ITEMS, pts_mean, familiarity_score)
    print(report)

    json_block = (
        "--- MACHINE-READABLE SUMMARY (JSON) ---\n"
        + json.dumps(summary, indent=2, ensure_ascii=False)
        + "\n--- END SUMMARY ---\n"
    )
    out_path = os.path.join(
        HITLS_DIR, participant_id, "cleaned", f"{participant_id}_tia_report.txt"
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(json_block)
        f.write(report)
    print(f"  Report saved to: {os.path.relpath(out_path)}")


if __name__ == "__main__":
    main()
