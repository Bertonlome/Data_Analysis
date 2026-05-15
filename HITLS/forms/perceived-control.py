#!/usr/bin/env python3
"""
Perceived Control (PC) questionnaire processor
================================================
4 items, 5-point Likert (1–5), all positive valence (high = better).
Reads from HAT_study.csv (questionnaire_id = "perceived_control").

Output: {pid}_perceived_control_report.txt in {pid}/cleaned/

Typically invoked by `forms/forms.py`, but can be run interactively.
"""

import os, csv, json
from collections import defaultdict

HITLS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCALE_MIN, SCALE_MAX = 1, 5
CONDITION_ORDER = ["TARS", "TARC", "TARP-S", "TARP-F"]

ITEMS = [
    ("pc_01", "I feel in control while using this autonomous system.",                    True),
    ("pc_02", "I feel I can control the way that the autonomous system behaves.",         True),
    ("pc_03", "I have the resources and the ability to make use of this autonomous system.", True),
    ("pc_04", "Team effectiveness in accomplishing the mission.  (1=not effective, 5=very effective)", True),
]

ANCHORS = {
    "pc_01": ["Strongly disagree", "Disagree", "Neutral", "Agree", "Strongly agree"],
    "pc_02": ["Strongly disagree", "Disagree", "Neutral", "Agree", "Strongly agree"],
    "pc_03": ["Strongly disagree", "Disagree", "Neutral", "Agree", "Strongly agree"],
    "pc_04": ["Not effective at all", "Slightly effective", "Neutral", "Quite effective", "Very effective"],
}


# ── CSV helpers ───────────────────────────────────────────────────────────────

def find_participants():
    return [
        e for e in sorted(os.listdir(HITLS_DIR))
        if os.path.isdir(os.path.join(HITLS_DIR, e))
        and e.startswith("P") and e[1:].isdigit()
    ]


def find_forms_csv(pid):
    folder = os.path.join(HITLS_DIR, pid)
    for name in sorted(os.listdir(folder)):
        if "hat_study" in name.lower() and name.endswith(".csv"):
            return os.path.join(folder, name)
    return None


def load_rows(csv_path):
    with open(csv_path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def extract_pc_data(rows):
    """Return {condition: {pc_key: raw_int}}."""
    data = defaultdict(dict)
    for row in rows:
        if row["questionnaire_id"] != "perceived_control":
            continue
        q = row["question_id"].strip()
        try:
            data[row["condition"].strip()][q] = int(row["value"])
        except ValueError:
            pass
    return dict(data)


# ── Scoring ───────────────────────────────────────────────────────────────────

def compute_pc(condition_data):
    """Return (mean, details_list) for a single condition."""
    details = []
    for key, label, positive in ITEMS:
        raw = condition_data.get(key)
        if raw is None:
            continue
        scored = raw if positive else (SCALE_MAX + SCALE_MIN - raw)
        details.append({"key": key, "label": label, "raw": raw, "scored": scored})
    if not details:
        return None, []
    mean = sum(d["scored"] for d in details) / len(details)
    return round(mean, 4), details


# ── Report builder ────────────────────────────────────────────────────────────

def _bar(v, width=10):
    filled = int(((v - SCALE_MIN) / (SCALE_MAX - SCALE_MIN)) * width)
    return f"[{'█' * filled}{'░' * (width - filled)}]"


def build_report(pid, all_data):
    lines = []
    sep = "=" * 72

    lines.append(f"\n{sep}")
    lines.append(f"  PERCEIVED CONTROL (PC) REPORT — {pid}")
    lines.append(sep)

    conditions = [c for c in CONDITION_ORDER if c in all_data]
    conditions += [c for c in sorted(all_data) if c not in CONDITION_ORDER]

    for cond in conditions:
        cdata = all_data[cond]
        lines.append(f"\n{'─'*72}")
        lines.append(f"  CONDITION: {cond}")
        lines.append(f"{'─'*72}\n")
        lines.append(f"  {'Item':<55} {'Raw':>4}   {'Bar'}")
        lines.append(f"  {'─'*55}   {'─'*4}   {'─'*12}")
        for key, label, _ in ITEMS:
            raw = cdata.get(key)
            if raw is None:
                lines.append(f"  {label:<55}   —")
            else:
                lines.append(f"  {label:<55} {raw:>4}   {_bar(raw)}")
        mean, _ = compute_pc(cdata)
        if mean is not None:
            lines.append(f"\n  {'─'*60}")
            lines.append(f"  PC Mean Score : {mean:.3f} / {SCALE_MAX:.1f}")

    return "\n".join(lines)


def build_json(pid, all_data):
    result = {"participant": pid, "conditions": {}}
    for cond, cdata in all_data.items():
        mean, details = compute_pc(cdata)
        result["conditions"][cond] = {
            "mean": mean,
            "items": {
                d["key"]: {"raw": d["raw"], "scored": d["scored"]}
                for d in details
            },
        }
    return result


def write_report(pid, all_data):
    cdir = os.path.join(HITLS_DIR, pid, "cleaned")
    os.makedirs(cdir, exist_ok=True)
    out_path = os.path.join(cdir, f"{pid}_perceived_control_report.txt")

    human_block = build_report(pid, all_data)
    json_obj    = build_json(pid, all_data)
    json_block  = json.dumps(json_obj, indent=2, ensure_ascii=False)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("--- MACHINE-READABLE SUMMARY (JSON) ---\n")
        f.write(json_block)
        f.write("\n--- END SUMMARY ---\n")
        f.write(human_block)

    print(f"  Saved → {out_path}")
    return out_path


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    participants = find_participants()
    for pid in participants:
        csv_path = find_forms_csv(pid)
        if not csv_path:
            print(f"  ⚠  {pid}: no HAT_study CSV found — skipping")
            continue
        rows    = load_rows(csv_path)
        pc_data = extract_pc_data(rows)
        if not pc_data:
            print(f"  ⚠  {pid}: no perceived_control rows found — skipping")
            continue
        write_report(pid, pc_data)


if __name__ == "__main__":
    main()
