# HITLS — Data Analysis Pipeline

Human-in-the-Loop study with four automation conditions:
**TARS** (baseline) · **TARC** (recommended) · **TARP-S** (partial-soft) · **TARP-F** (partial-forced)

Participants: P02 – P18 (some numbers skipped due to exclusions).

---

## Directory Structure

```
HITLS/
├── {PID}/                      # Per-participant data
│   ├── {PID}_ingescape.csv     # Raw Ingescape log
│   ├── {PID}_TARC.csv          # TARC-specific Ingescape log
│   ├── HAT_study.csv           # Questionnaire responses
│   ├── cleaned/                # Generated reports (.txt + JSON blocks)
│   └── scenarios/              # Per-scenario CSVs (sliced from Ingescape)
├── forms/                      # Questionnaire analysis scripts
├── performance/                # Flight performance analysis scripts
├── eye-tracking/               # Eye-tracking quality scripts
├── plots/                      # All cross-participant comparison figures
├── compare_forms.py            # Cross-participant questionnaire comparison
├── allocation.py               # TARC allocation analysis
├── camera_merge.py             # Merge back + front camera clips
├── extract_scenarios.py        # Slice Ingescape CSV into per-scenario files
├── log_to_ingescape_csv.py     # Convert backup logs to Ingescape format
├── pre_create_error_annotations.py  # Scaffold error annotation CSVs
├── transcription.py            # Whisper interview transcription
├── transcript_crosscheck.py    # Interactive transcript correction tool
├── video_cutter.py             # GUI-based video clip trimmer
└── video_extract.py            # Extract scenario clips from merged video
```

---

## Full Analysis Flow

### Step 1 — Merge camera recordings

```bash
python HITLS/camera_merge.py -p P02
```

Merges back-camera and front-camera clips for a participant into a single
continuous timeline. Output: `{PID}/back_camera/` and `{PID}/front_camera/`.

---

### Step 2 — Extract scenario CSVs from Ingescape logs

```bash
python HITLS/extract_scenarios.py HITLS/P02 HITLS/P03 ...
```

Slices the participant's `{PID}_ingescape.csv` into per-scenario CSV files
saved in `{PID}/scenarios/`.

> **P17 only:** First convert the backup log format with:
> ```bash
> python HITLS/log_to_ingescape_csv.py
> ```

---

### Step 3 — Extract scenario video clips

```bash
python HITLS/video_extract.py P02 P05
```

Cuts scenario-length clips from the merged video using timestamps from the
scenario CSV files. Output: `{PID}/scenarios/*.mp4`.

#### Step 3b (optional) — Trim a video clip

```bash
python HITLS/video_cutter.py
```

Opens an interactive GUI to trim or re-cut any clip.

---

### Step 4 — Pre-create error annotation scaffolds

```bash
python HITLS/pre_create_error_annotations.py
# Add --write to save files, --force to overwrite existing ones
```

Generates blank annotation CSV templates for all scenarios that contain an
error condition, ready to be filled in by a coder.

---

### Step 5 — Transcribe participant interviews

```bash
python HITLS/transcription.py
```

Uses OpenAI Whisper to transcribe post-experiment interviews. Runs best on
Windows with a CUDA-enabled GPU (see script header for setup instructions).
Output: `{PID}/{PID}_itw_transcript.json` + `.txt`.

---

### Step 6 — Cross-check and correct transcripts

```bash
python HITLS/transcript_crosscheck.py
```

Interactive side-by-side tool to replay audio while editing transcripts.
Keyboard shortcuts are documented in the script header.

---

### Step 7 — Analyse questionnaires (per participant)

```bash
python HITLS/forms/forms.py P02
# or: python HITLS/forms/forms.py 3   (3rd participant in the list)
```

Runner that executes all seven questionnaire sub-scripts for the selected
participant. Each sub-script can also be run independently (they are invoked
interactively if no participant argument is given):

| Script | Questionnaire | Output report |
|---|---|---|
| `forms/pre-experiment-forms.py` | Demographics + PTS trust propensity | `{PID}_pre_experiment_report.txt` |
| `forms/nasa-tlx.py` | NASA-TLX workload (6 subscales) | `{PID}_nasa_tlx_report.txt` |
| `forms/sus.py` | System Usability Scale (SUS) | `{PID}_sus_report.txt` |
| `forms/trust-in-automation.py` | TiA Körber 12-item (3 subscales) | `{PID}_tia_report.txt` |
| `forms/trust-risk.py` | Trust VAS + Risk VAS (0–100) | `{PID}_trust_risk_report.txt` |
| `forms/oversight-bespoke.py` | Oversight bespoke scale | `{PID}_oversight_bespoke_report.txt` |
| `forms/perceived-control.py` | Perceived Control (4 Likert items) | `{PID}_perceived_control_report.txt` |

All reports are saved in `{PID}/cleaned/` and contain a JSON summary block
delimited by `--- END SUMMARY ---` followed by a human-readable text section.

---

### Step 8 — Analyse flight performance (per participant)

```bash
python HITLS/performance/performance.py P02
# or: python HITLS/performance/performance.py 3
```

Runner that executes all three performance sub-scripts for the selected
participant. Sub-scripts can also be run individually:

| Script | Metrics | Output report |
|---|---|---|
| `performance/aviate_perf.py` | Slip/roll RMSE, airspeed NMAE | `{PID}_aviate_perf_report.txt` |
| `performance/navigate_perf.py` | XTE, ATD, heading & altitude deviation | `{PID}_navigate_perf_report.txt` |
| `performance/time_perf.py` | Scenario duration, task timing | `{PID}_time_perf_report.txt` |

Additionally, `performance/error_perf.py` analyses error detection and
correction and can be run for a single participant or all at once:

```bash
python HITLS/performance/error_perf.py P02   # single participant
python HITLS/performance/error_perf.py A     # all participants
```

---

### Step 9 — Eye-tracking quality check (optional)

```bash
python HITLS/eye-tracking/pupillo.py
```

Fully interactive: select participant, scenario CSV, smoothing window, and
quality threshold. Plots pupil diameter quality and eyelid opening quality
from SmartEyeProBridge data.

---

### Step 10 — Cross-participant comparisons

Run these **after all per-participant reports are generated** (Steps 7–8).
Each compare-type script asks for confirmation before writing or overwriting
output files.

```bash
python HITLS/compare_forms.py            # Questionnaire Likert + box plots
python HITLS/compare_performance.py      # All 3 performance comparisons at once
python HITLS/allocation.py               # TARC allocation similarity analysis

# Or run performance comparisons individually:
python HITLS/performance/compare_aviate.py    # Aviate performance comparison
python HITLS/performance/compare_navigate.py  # Navigate performance comparison
python HITLS/performance/compare_time.py      # Time performance comparison
```

All figures are saved to `HITLS/plots/`.

---

## Data Conventions

- **Timestamps**: Ingescape logs use **UTC+8**; camera filenames use **UTC+4**.
  `extract_scenarios.py` applies the correct offset automatically.
- **Conditions**: Every participant sees all four conditions in counterbalanced
  order. Condition mapping is encoded in the Ingescape scenario filenames.
- **Reports format**: Each `.txt` report file contains a JSON block at the top
  (between `{` / `}` markers) followed by `--- END SUMMARY ---` and then a
  human-readable narrative.
- **Plots directory**: `HITLS/plots/` is created automatically by compare
  scripts if it doesn't exist.

---

## Quick Reference — All Scripts

| Script | Type | Purpose |
|---|---|---|
| `camera_merge.py` | Per-participant | Merge back + front camera clips |
| `extract_scenarios.py` | Per-participant | Slice Ingescape CSV into scenario files |
| `log_to_ingescape_csv.py` | Utility (P17) | Convert backup log to Ingescape format |
| `video_extract.py` | Per-participant | Extract scenario clips from merged video |
| `video_cutter.py` | Utility | GUI clip trimmer |
| `pre_create_error_annotations.py` | Per-participant | Scaffold error annotation CSVs |
| `transcription.py` | Per-participant | Whisper interview transcription |
| `transcript_crosscheck.py` | Per-participant | Interactive transcript correction |
| `forms/forms.py` | Per-participant | Run all questionnaire sub-scripts |
| `performance/performance.py` | Per-participant | Run all performance sub-scripts |
| `performance/error_perf.py` | Per-participant / All | Error detection & correction analysis |
| `eye-tracking/pupillo.py` | Per-participant | Eye-tracking quality visualisation |
| `compare_forms.py` | Compare | Cross-participant questionnaire plots |
| `compare_performance.py` | Compare | All 3 performance comparisons (orchestrator) |
| `allocation.py` | Compare | TARC allocation similarity plots |
| `performance/compare_aviate.py` | Compare | Aviate performance comparison |
| `performance/compare_navigate.py` | Compare | Navigate performance comparison |
| `performance/compare_time.py` | Compare | Time performance comparison |
