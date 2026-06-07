# Model Validation Plan — HITLS vs MITLS

## Implementation Progress

| Step | Module | Status |
|---|---|---|
| 1 | `shared/coverage.py` | ✅ Done |
| 2 | `shared/hitls_loader.py` + `shared/mitls_loader.py` | ✅ Done |
| 3 | `shared/stats.py` | ✅ Done |
| 4 | `jae/jae_validation.py` | ✅ Done |
| 5 | `timing/timing_validation.py` | ✅ Done |
| 6 | `eye_movement/aoi_validation.py` | ✅ Done |
| 7 | `workload/workload_validation.py` | ✅ Done |
| 8 | `run_all.py` | ✅ Done |

---

## Context

This plan describes a framework for validating the cognitive model (ACT-R-based, running inside the MITLS cognitive architecture) against empirical data collected from expert pilots in the HITLS simulation. Both systems ran the same scenario under three task-allocation conditions:

| Model condition | HITLS label | Description |
|---|---|---|
| **C1** | TARS | Manual baseline — human performs all tasks; TARS shows a static checklist only |
| **C2** | TARP-F | Autonomy-centric — all delegatable tasks performed by the agent; agent provides support whenever possible |
| **C3** | TARP-S | Autonomy with monitoring window — identical to C2 but with post-action delays for human crosschecking |

The validation is designed as a **multi-domain quantitative comparison** following Anderson et al.'s (2004) cognitive model validation levels:

1. **Behavioral fit** — does the model reproduce task-level observable outputs (timing, gaze, allocation)?
2. **Functional fit** — does the model's internal state (workload modules, production utilities) correspond to physiological and subjective measures?
3. **Effect replication** — does the model show the same directional trends across conditions that humans show?

---

## Procedure Coverage & Scope

MITLS is built incrementally. **As of June 2026, only two procedures are implemented:**

| Procedure | MITLS status | HITLS status |
|---|---|---|
| BEFORE TAKEOFF (Pre-takeoff checklist) | ✅ Implemented | ✅ Available |
| LINE UP AND HOLD | ✅ Implemented | ✅ Available |
| TAKEOFF | ⬜ Not yet modelled | ✅ Available |
| ENGINE FAILURE | ⬜ Not yet modelled | ✅ Available |
| DECLARE PAN-PAN | ⬜ Not yet modelled | ✅ Available |
| AFTER TAKEOFF | ⬜ Not yet modelled | ✅ Available |

`shared/coverage.py` centralises this coverage map and is the single source of truth. Every analysis module calls `coverage.get_covered_tasks()` to restrict comparisons to the implemented subset. Non-covered tasks are **excluded silently from statistics** but are **listed explicitly in diagnostic outputs** so the cognitive modeller can see what the model is not yet handling.

When MITLS coverage expands (new procedures added), only `coverage.py` needs to be updated; all analysis modules adapt automatically.

---

## Domain Overview

| Validation Domain | MITLS Source | HITLS Source | Key Metrics |
|---|---|---|---|
| **Joint Activity Efficiency** | `output/comparison_summary_metrics.csv` (`JAE_Data_*`) | `compare_performance/time_report.txt` (scenario duration proxy) | JAE per condition, direction match, Friedman χ² |
| **Cognitive Workload** | `output/comparison_summary_metrics.csv` (`Workload_*`) + per-run `workload_analyzer/results_mental_workload.txt` | `HRV/hrv_features_per_scenario.csv`, `compare_forms/nasa_tlx_report.txt` | Spearman ρ, Pearson r, per-condition direction match |
| **Eye Movement / AoI** | per-run `eye_movement/results_eye_movement.txt` | `eye-tracking/AoIs.py` outputs | AoI proportion RMSE, KL divergence, chi-square |
| **Task Timing** | per-run `team-analyzer/task_summary.csv` + `scenario_summary.csv` | `compare_performance/time_report.txt` | Per-task duration RMSE/NMAE, scenario total duration |


---

## Script Architecture

```
model_validation/
├── plan.md                        ← this file
├── run_all.py                     ← orchestrator: coverage report → all modules → summary
├── shared/
│   ├── coverage.py                ← MITLS procedure/task coverage map; get_covered_tasks()
│   ├── hitls_loader.py            ← load and normalize HITLS data (Ingescape CSVs, cleaned reports)
│   ├── mitls_loader.py            ← load and normalize MITLS outputs (CSVs, .txt summaries)
│   └── stats.py                   ← shared statistical helpers (Friedman, Wilcoxon, Holm, r, RMSE)
├── jae/
│   ├── jae_validation.py
│   ├── jae_report.txt             ← Tier 1: publication-ready
│   └── jae_debug_report.txt       ← Tier 2: diagnostic
├── workload/
│   ├── workload_validation.py
│   ├── workload_report.txt
│   └── workload_debug_report.txt
├── eye_movement/
│   ├── aoi_validation.py
│   ├── aoi_report.txt
│   └── aoi_debug_report.txt
├── timing/
│   ├── timing_validation.py
│   ├── timing_report.txt
│   └── timing_debug_report.txt
└── plots/
    ├── pub/                       ← Tier 1: publication-ready figures (clean, labelled)
    └── debug/                     ← Tier 2: diagnostic figures (per-task, participant overlays)
```

---

## Output Tier Design

Every validation module produces two independent output tiers:

### Tier 1 — Publication outputs (`plots/pub/`, `*_report.txt`)
Intended for papers and reports. Characteristics:
- **Coverage-scoped**: only tasks within `coverage.get_covered_tasks()` are included; a header note states the scope (e.g., *"Analysis restricted to BEFORE TAKEOFF and LINE UP AND HOLD procedures"*).
- **Condition-level aggregates**: model value (mean ± CI across 12 repetitions) vs human group mean ± SD.
- **Statistical tests**: Friedman / Wilcoxon / Holm-Bonferroni as appropriate.
- **Clean figures**: publication-style, minimal annotation, shared color scheme.

### Tier 2 — Diagnostic outputs (`plots/debug/`, `*_debug_report.txt`)
Intended for cognitive model debugging and development. Characteristics:
- **Full task inventory**: every task in HITLS is listed; uncovered MITLS tasks are marked `[NOT MODELLED]` and shown as empty slots in figures — making gaps immediately visible.
- **Participant envelope**: each figure overlays the **minimum**, **mean**, and **maximum** individual HITLS participant values as reference bands. The model mean (or per-repetition distribution) is plotted on top so the modeller can see at a glance whether the model is inside/outside the human range and how far it is from the average.
- **Per-task granularity**: one row per `(Task_Object, Task_Value)` pair, not aggregated by procedure or condition — the primary debugging unit.
- **Repetition scatter**: for stochastic metrics (ACT-R noise enabled), all 12 repetitions are shown as individual points alongside the mean, to expose variance issues.
- **Debug annotations**: NMAE flags (|NMAE| > 0.5 highlighted in red), missing data markers, and coverage gaps.

### Shared helper: `HitlsDistribution` (in `stats.py`)
A small dataclass used across all modules:
```python
@dataclass
class HitlsDistribution:
    mean: float
    sd: float
    min_val: float   # worst participant
    max_val: float   # best participant
    pid_min: str     # participant id of worst
    pid_max: str     # participant id of best
    n: int
```
Every diagnostic plot receives a list of `HitlsDistribution` objects (one per task/condition) and renders the envelope automatically via a shared `plot_participant_envelope()` helper.

---

## 1. Shared Utilities (`shared/`)

### `coverage.py`
Single source of truth for MITLS procedure/task coverage:
- `COVERED_PROCEDURES` — set of procedure names implemented in MITLS (initially `{"BEFORE TAKEOFF", "LINE UP AND HOLD"}`)
- `get_covered_tasks(hitls_task_df)` → returns `(covered_df, uncovered_df)` split of a task DataFrame
- `coverage_summary()` → returns a dict `{procedure: "implemented" | "not modelled"}` for all known HITLS procedures
- Used by every analysis module to gate statistics and generate the scope note in Tier 1 reports

### `hitls_loader.py`
Provides functions to load already-computed HITLS analysis outputs:
- `load_nasa_tlx(pids)` → DataFrame (pid × condition, NASA-TLX weighted scores + subscales)
- `load_hrv_features(pids)` → DataFrame from `HITLS/HRV/hrv_features_per_scenario.csv`
- `load_aoi_data(pids)` → per-scenario AoI percentages from `cleaned/` reports
- `load_timing_reports(pids)` → per-task durations from `cleaned/*_time_perf_report.txt`
- `load_error_reports(pids)` → error-detection quadrant data from `cleaned/*_error_perf_report.txt`
- `load_allocation_data(pids)` → task-allocation state sequences from `P0x_TARC.csv`
- `load_ingescape_scenario(path)` → parse a scenario Ingescape CSV into a tidy DataFrame

### `mitls_loader.py`
Loads MITLS cognitive-model outputs:
- `load_comparison_summary()` → from `output/comparison_summary_metrics.csv`; returns DataFrame with one row per condition (C1/C2/C3) and all aggregate columns (JAE, workload, timing, CI, Std)
- `load_convergence(run_number)` → from `output/convergence_analysis_run{N}.csv`; returns convergence table indexed by n_replications
- `load_workload_timeseries(run_dir)` → from `workload_analyzer/results_mental_workload.txt` in a given run folder (tab-delimited, 11 module columns)
- `load_aoi_results(run_dir)` → parse `eye_movement/results_eye_movement.txt` for AoI percentages and dwell times
- `load_task_summary(run_dir)` → from `team-analyzer/task_summary.csv` (per-task FSM/human/TARS durations)
- `load_scenario_summary(run_dir)` → from `team-analyzer/scenario_summary.csv`
- `iter_condition_runs(condition)` → yields all run_dir paths for condition C1/C2/C3 (e.g. all `run_1_results_*` dirs for C1)

### `stats.py`
Centralised statistical utilities (reuse from HITLS codebase conventions):
- `friedman_test(data)` — for within-participant condition comparisons
- `wilcoxon_pairwise(data, conditions)` — pairwise + Holm-Bonferroni
- `rank_biserial_r(stat, n)` — effect size
- `rmse(predicted, observed)`, `nmae(predicted, observed)`, `pearson_r(a, b)`
- `ks_2samp_test(a, b)` — distribution comparison
- `HitlsDistribution` dataclass (mean, sd, min_val, max_val, pid_min, pid_max, n)
- `build_hitls_distributions(df, group_col, value_col)` → list of `HitlsDistribution` per group
- `plot_participant_envelope(ax, distributions, model_values, model_labels)` — shared debug plotting helper: draws human min/mean/max band + shaded SD region, overlays model points/distribution

---

## 2. JAE Validation (`jae/jae_validation.py`)

### What is JAE?

Joint Activity Efficiency (JAE) measures how efficiently the human–agent team completes the shared task. In the model it is defined as:

$$\text{JAE} = \frac{\text{Human\_Active\_Time} + \text{TARS\_Execution\_Time}}{\text{Total\_FSM\_Time}}$$

`Team_Efficiency_Pct = JAE × 100`. Model values from `comparison_summary_metrics.csv`:
- C1 (TARS/Manual): JAE ≈ 0.617 — human does everything, some idle gaps
- C2 (TARP-F/Autonomy): JAE ≈ 0.699 — agent offloads tasks, less idle time
- C3 (TARP-S/Monitoring): JAE ≈ 0.142 — deliberate post-action delays collapse efficiency

The **expected direction** matches the HITLS expectation: C2 > C1 > C3 for raw task throughput, but C3 is intentionally penalised by design (monitoring window).

### HITLS proxy

No direct JAE equivalent exists in HITLS. The closest proxies are:
- **Scenario duration** from `time_report.txt` (shorter = more efficient)
- **Human active-time fraction** from `allocation.py` (proportion of tasks performed by human)
- **Operator idle time** if derivable from Ingescape event gaps

### Analysis steps

1. **Condition-level JAE comparison (model-internal)**
   - Bar chart: JAE_Data_Mean ± CI for C1 / C2 / C3
   - Load `convergence_analysis_run{1,2,3}.csv` and plot JAE stability (converged at 12 reps?)
   - Friedman test on per-repetition JAE values across 3 conditions (12 reps each) → ω²
   - Pairwise Wilcoxon (C1 vs C2, C1 vs C3, C2 vs C3) + Holm-Bonferroni

2. **Direction match against HITLS**
   - Derive HITLS throughput proxy (scenario duration or active-time fraction) per condition
   - Report whether model JAE ranking (C2 > C1 > C3) matches human proxy ranking
   - Note: C3's low JAE is a model design feature (not a failure); validate that the *reason* (coordination overhead spike) is also present in HITLS (longer post-action gaps in TARP-S)

3. **Coordination overhead comparison**
   - Model: `Coordination_Time_Mean / Total_FSM_Time_Mean` per condition
   - HITLS: no direct equivalent — document as model-only characterization
   - Plot: stacked bar (Active % / Coordination % / Idle %) per condition (model vs any HITLS proxy)

### Tier 1 outputs (publication)
- `plots/pub/jae_condition_comparison.png` — bar chart: JAE mean ± CI per condition (C1/C2/C3), significance brackets
- `jae/jae_report.txt` — Friedman χ², pairwise Wilcoxon + Holm p-values, effect sizes ω², direction match statement, scope note

### Tier 2 outputs (diagnostic)
- `plots/debug/jae_convergence_c{1,2,3}.png` — JAE convergence curve: how mean and CI stabilize as n_replications grows (12-rep trace from `convergence_analysis_run{N}.csv`)
- `plots/debug/jae_coordination_breakdown.png` — stacked bar: Active % / Coordination % / Idle % per condition; 12 individual repetition points overlaid
- `jae/jae_debug_report.txt` — per-repetition JAE values for all 3 conditions, coordination overhead breakdown, flag if CI has not converged (rel_precision > 5%)

---

## 3. Workload Validation (`workload/workload_validation.py`)

### Conceptual mapping

ACT-R tracks processing utilization (0–1) per 1-second window across 9 modules grouped into 3 subnetworks:

| ACT-R Subnetwork / Module | Proposed NASA-TLX Subscale | Proposed HRV proxy |
|---|---|---|
| `Perceptual_SubNetwork` (Vision + Audio) | Temporal Demand | — |
| `Cognitive_SubNetwork` (Production + Declarative + Imaginary) | Mental Demand + Effort | RMSSD (inverse) |
| `Motor_SubNetwork` (Motor + Speech) | Physical Demand | — |
| `Overall_Utilization` | **Weighted NASA-TLX total** | SDNN / RMSSD (inverse) |

### Analysis steps

1. **Per-procedure workload comparison**
   - Segment model workload time-series by task events from `task_events.json`
   - Compute mean `Overall_Utilization` per procedure segment
   - Load human NASA-TLX per-condition means (condition ≈ procedure set)
   - Plot side-by-side (model vs human) bump/strip chart over procedure sequence
   - Report Spearman ρ between model procedure-mean utilization and human NASA-TLX total

2. **HRV correlation** (where synchronized data exists)
   - Load `hrv_features_per_scenario.csv` — per-scenario RMSSD (HITLS)
   - Load model `Overall_Utilization` mean per scenario
   - Compute Pearson r (expected: negative — higher utilization → lower RMSSD)
   - Scatter plot: model mean utilization (x) vs human median RMSSD (y), per condition

3. **Condition-level direction test**
   - Aggregate model workload per condition
   - Compare directional ranking to human NASA-TLX condition ranking
   - Report: rank correlation + direction match (%)

4. **Module-level profile visualization**
   - Heatmap: 9 modules × time-steps (model)
   - Overlaid with task event markers from `task_events.json`

### Tier 1 outputs (publication)
- `plots/pub/workload_vs_nasa_tlx.png` — bar/scatter: model overall utilization vs human NASA-TLX total per condition (covered procedures only)
- `plots/pub/workload_vs_hrv.png` — scatter: model utilization vs human median RMSSD, one point per condition
- `workload/workload_report.txt` — Spearman ρ, Pearson r, direction match %, scope note

### Tier 2 outputs (diagnostic)
- `plots/debug/workload_per_task_envelope.png` — **one row per task**: 9-module utilization bar for the model (mean ± repetition spread) vs human min/mean/max envelope from NASA-TLX subscales; uncovered tasks shown as grey `[NOT MODELLED]` rows
- `plots/debug/workload_module_heatmap.png` — 9 modules × time-steps heatmap with task event markers from `task_events.json`
- `plots/debug/workload_per_condition_distribution.png` — per condition: model 12-rep distribution (strip/violin) placed on top of human participant distribution
- `workload/workload_debug_report.txt` — per-task NMAE for each module, flags tasks where model is outside human min–max range

---

## 4. Eye Movement / AoI Validation (`eye_movement/aoi_validation.py`)

### Conceptual mapping

Both MITLS and HITLS define overlapping AoI categories:

| AoI (HITLS label) | AoI (MITLS label) | Notes |
|---|---|---|
| TARS | TARS | Same |
| PFD | PFD | Same |
| ND | ND | Same |
| pedestal | Central_Console | Partial overlap |
| Outside_Window | Outside_Window | Same |
| — | E_W_CAS | tracked in HITLS as ND |

### Analysis steps

1. **AoI proportion comparison**
   - Load HITLS per-participant per-condition AoI percentage (from cleaned reports)
   - Load MITLS `aoi_metrics.csv` AoI percentages
   - Compute per-AoI absolute difference: `|model% − human_mean%|`
   - Bar chart: grouped by AoI, showing human mean ± SD vs model value

2. **Dwell time comparison**
   - HITLS: mean fixation bout duration per AoI per condition
   - MITLS: `Avg_Dwell_Time_s` per AoI
   - RMSE and MAE across AoIs

3. **Distribution similarity**
   - Chi-square goodness-of-fit: model AoI proportion vs human mean proportion (use human SD as expected variance)
   - KL divergence between model and human AoI distributions
   - Visualise as stacked bar (model vs human mean, error bars from human SD)

4. **Scanpath-level comparison** *(stretch goal)*
   - If scanpath sequence data is available from both (MITLS eye-movement-analyzer.py and HITLS SmartEyeProBridge sequences), compute Levenshtein distance on AoI transition sequences

### Tier 1 outputs (publication)
- `plots/pub/aoi_proportions_comparison.png` — grouped bar: human mean ± SD vs model mean per AoI (covered procedures only)
- `plots/pub/aoi_stacked_bar.png` — stacked bar: model vs human mean AoI distribution
- `eye_movement/aoi_report.txt` — per-AoI RMSE, MAE, chi-square GoF, KL divergence, scope note

### Tier 2 outputs (diagnostic)
- `plots/debug/aoi_per_task_envelope.png` — **one row per task in covered procedures**: model AoI % per task vs human min/mean/max participant AoI %; tasks with >2 SD deviation highlighted
- `plots/debug/aoi_dwell_per_task.png` — dwell time: model vs participant envelope per AoI per task
- `plots/debug/aoi_cockpit_overlay_debug.png` — cockpit image overlay with model proportion vs human mean proportion side-by-side per AoI region
- `eye_movement/aoi_debug_report.txt` — per-task, per-AoI: model %, human mean %, human min %, human max %, NMAE, in/out-of-range flag

---

## 5. Task Timing Validation (`timing/timing_validation.py`)

### Conceptual mapping

Both systems share the same TARS checklist procedures (BEFORE TAKEOFF → TAKEOFF → ENG FAILURE → DECLARE PANPAN → AFTER TAKEOFF). Per-task wall-clock durations are available from:
- **MITLS**: `task_summary.csv` — `FSM_Duration_s` (time in state), `Human_Active_Time_s`
- **HITLS**: `time_perf_report.txt` — per-procedure mean durations derived from TARS Agent `current_state` events

### Analysis steps

1. **Per-task duration comparison**
   - Join on `Task_Object` + `Task_Value`
   - Compute RMSE and NMAE: `(model_duration − human_mean_duration) / human_mean_duration`
   - For each task: bar chart showing model vs human mean ± SD
   - Identify outlier tasks (|NMAE| > 0.5)

2. **Scenario total duration**
   - MITLS: `Total_FSM_Time_s` from `scenario_summary.csv`
   - HITLS: scenario duration from `time_perf_report.txt` (BEFORE TAKEOFF start → AFTER TAKEOFF end)
   - Report absolute and % difference

3. **Human vs TARS active time ratio**
   - MITLS: `Total_Human_Active_Time_s / Total_Active_Duration_s`
   - HITLS: derive from video coding, TBD on HITLS scenarios/
   - Compare ratios across conditions

4. **Coordination overhead**
   - MITLS: `Total_Coordination_Time_s / Total_FSM_Time_s`
   - HITLS: derive from video coding, TBD on HITLS scenarios/

### Tier 1 outputs (publication)
- `plots/pub/task_duration_model_vs_human.png` — grouped bar per procedure: model mean ± CI vs human mean ± SD (covered procedures only)
- `plots/pub/scenario_duration_comparison.png` — bar: total scenario duration model vs human mean ± SD per condition
- `timing/timing_report.txt` — per-task NMAE table, scenario duration % error, active-time ratio, scope note

### Tier 2 outputs (diagnostic)
- `plots/debug/timing_per_task_envelope.png` — **one row per task** (all HITLS tasks): model duration (mean + 12-rep strip) vs human min/mean/max; uncovered tasks shown as `[NOT MODELLED]` grey rows with the human distribution still visible — so the modeller immediately sees what remains to implement and how demanding those tasks are
- `plots/debug/timing_nmae_sorted.png` — horizontal bar chart of per-task NMAE sorted by absolute error; tasks with |NMAE| > 0.5 highlighted in red
- `timing/timing_debug_report.txt` — per-task: model mean duration, human mean, human min, human max (with pid), NMAE, in/out-of-range flag; uncovered tasks listed with human stats as implementation targets

---

## 6. Error Detection Validation — *Deferred*

> **Status: not implemented.** Error injection in MITLS C3 is still being updated. This module will be added in a future iteration once the error model is stable. The HITLS `compare_performance/error_report.txt` data is available and waiting.

When implemented, this module will compare ACT-R crosscheck production utilities (`x-7-decide-crosscheck` vs `x-7-no-crosscheck`) against the HITLS 2×2 error-detection matrix (Corrected × Crosschecked), restricted to the procedure(s) that contain the injected error.

---

## 8. Orchestrator (`run_all.py`)

Runs the coverage check then all six validation modules in sequence, collects per-module metrics into a single cross-domain summary table, and writes `model_validation/summary_report.txt`.

Summary table structure:

| Domain | Metric | Value | Interpretation |
|---|---|---|---|
| JAE | Friedman χ² (C1 vs C2 vs C3, model) | — | — |
| JAE | Direction match vs HITLS throughput proxy | — | — |
| Workload | Spearman ρ (utilization vs NASA-TLX) | — | — |
| Workload | Pearson r (utilization vs RMSSD) | — | — |
| Eye Movement | Mean AoI proportion RMSE | — | — |
| Eye Movement | KL divergence | — | — |
| Timing | Per-task duration NMAE (mean) | — | — |
| Timing | Scenario duration % error | — | — |


---

---

## Implementation Notes & Challenges

### 1. Condition alignment
MITLS has 3 conditions (C1/C2/C3 = TARS/TARP-F/TARP-S), each with 12 stochastic repetitions. HITLS has N participants × 3 matching conditions (within-participant, same labels). For validation:
- Align on the 3 shared conditions; use `Run_Name` field in `comparison_summary_metrics.csv` (values `C1`, `C2`, `C3`) as the join key
- Per-repetition data lives in `run_{1,2,3}_results_*` folders; use `iter_condition_runs()` from `mitls_loader.py`
- Error injection in MITLS is deferred; no special handling of C3 trace files is required for now

### 2. Temporal synchronization
MITLS workload uses relative seconds from scenario start; HITLS timestamps are Unix epoch. Both need to be normalized to `t = 0` at scenario start before time-series comparisons.

### 3. Single model vs. population
The model provides a distribution of 12 stochastic repetitions (ACT-R noise); humans provide a distribution of N participants. **Tier 1** reports model mean ± CI against human mean ± SD. **Tier 2** plots the full model repetition distribution against the human min/mean/max envelope — this directly tells the cognitive modeller whether the model's behaviour is realistic (inside the human range), too fast/slow/consistent, or systematically biased. The participant IDs of the min and max human are recorded so the modeller can inspect which type of pilot the model most resembles.

### 4. MITLS multi-condition data location
All per-condition model outputs live under `MITLS/eye_movement/output/`. The `comparison_summary_metrics.csv` file provides C1/C2/C3 aggregated results (12 repetitions each) including JAE, workload, and timing. Individual repetition data is in `run_{1,2,3}_results_N_<timestamp>/` sub-folders. The top-level `MITLS/eye_movement/` analyzer files (e.g. `aoi_metrics.csv`, `workload_summary.csv`) represent only the **last single run** — always prefer the aggregated `output/` data for multi-condition comparisons.

### 5. Reuse of HITLS conventions
All statistical tests, report formats (JSON summary block + human-readable text), and figure styles should replicate the HITLS conventions to maintain consistency.

### 6. Incremental model growth
As new procedures are added to MITLS, only `shared/coverage.py` needs updating. All Tier 2 diagnostic figures automatically shift `[NOT MODELLED]` rows to active rows, making the modeller's progress immediately visible across all domains.

---

## Suggested Implementation Order

1. `shared/coverage.py` — define covered procedures; needed by everything else
2. `shared/hitls_loader.py` + `shared/mitls_loader.py` — data foundations
3. `shared/stats.py` — include `HitlsDistribution`, `build_hitls_distributions()`, `plot_participant_envelope()`
4. `jae/jae_validation.py` — richest, self-contained MITLS data; first end-to-end test of the Tier 1 / Tier 2 split
5. `timing/timing_validation.py` — most direct cross-system comparison; Tier 2 per-task envelope immediately useful for debugging
6. `eye_movement/aoi_validation.py` — distribution comparison; cockpit overlay useful for presentations
7. `workload/workload_validation.py` — requires module-to-subscale mapping discussion before Tier 1
8. `run_all.py` — orchestrator after all modules pass individually
