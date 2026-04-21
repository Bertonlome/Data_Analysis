import os
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── interactive prompts ──────────────────────────────────────────────────────

def list_options(folder: str) -> list[str]:
    return sorted(os.listdir(folder)) if os.path.isdir(folder) else []

def prompt_choice(prompt: str, options: list[str]) -> str:
    print(f"\n{prompt}")
    for i, opt in enumerate(options, 1):
        print(f"  {i}. {opt}")
    while True:
        raw = input("Enter number or name: ").strip()
        if raw.isdigit():
            idx = int(raw) - 1
            if 0 <= idx < len(options):
                return options[idx]
        elif raw in options:
            return raw
        print("  Invalid selection, try again.")

# ── data loading ──────────────────────────────────────────────────────────────

METRICS = [
    "pupil_diameter_quality",
    "filtered_pupil_diameter_quality",
    "eyelid_opening_quality",
]

COLUMNS = ["uuid", "timestamp", "agent", "source", "type", "igs_timestamp", "value"]

def load_metrics(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, sep=";", names=COLUMNS, header=0, low_memory=False)

    df = df[
        (df["agent"] == "SmartEyeProBridge") &
        (df["source"].isin(METRICS))
    ].copy()

    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df.dropna(subset=["timestamp", "value"], inplace=True)

    # Normalise timestamp to seconds from start
    df["time_s"] = df["timestamp"] - df["timestamp"].min()

    return df

# ── plotting ──────────────────────────────────────────────────────────────────

COLORS = {
    "pupil_diameter_quality":          "#1f77b4",
    "filtered_pupil_diameter_quality": "#ff7f0e",
    "eyelid_opening_quality":          "#2ca02c",
}

PUPIL_METRICS = ["pupil_diameter_quality", "filtered_pupil_diameter_quality"]
EYELID_METRICS = ["eyelid_opening_quality"]

def _smooth(sub: pd.DataFrame, window_s: float) -> tuple:
    """Return (x, y) arrays, optionally smoothed with a time-based rolling window."""
    series = pd.Series(
        sub["value"].values,
        index=pd.to_timedelta(sub["time_s"].values, unit="s"),
    ).sort_index()
    if window_s > 0:
        series = series.rolling(f"{window_s}s", min_periods=1).mean()
    return series.index.total_seconds(), series.values

def print_quality_stats(df: pd.DataFrame, threshold: float) -> None:
    print(f"\n── Quality below {threshold} ────────────────────────────────────")
    for metric in METRICS:
        sub = df[df["source"] == metric]
        if sub.empty:
            continue
        pct = (sub["value"] < threshold).mean() * 100
        print(f"  {metric:<42} {pct:6.2f}%")
    print()

def plot_metrics(df: pd.DataFrame, participant: str, condition: str, window_s: float) -> None:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    # ── top: pupil quality ────────────────────────────────────────────────────
    for metric in PUPIL_METRICS:
        sub = df[df["source"] == metric].sort_values("time_s")
        if sub.empty:
            print(f"  Warning: no data found for '{metric}'")
            continue
        x, y = _smooth(sub, window_s)
        ax1.plot(x, y, label=metric, color=COLORS[metric], linewidth=0.8, alpha=0.85)

    win_label = f"  (sliding window: {window_s} s)" if window_s > 0 else ""
    ax1.set_ylabel("Quality (0 – 1)")
    ax1.set_title(f"Pupil Quality — {participant} / {condition}{win_label}")
    ax1.legend(loc="lower left")
    ax1.set_ylim(0, 1.05)
    ax1.grid(True, linestyle="--", alpha=0.4)

    # ── bottom: eyelid quality ────────────────────────────────────────────────
    for metric in EYELID_METRICS:
        sub = df[df["source"] == metric].sort_values("time_s")
        if sub.empty:
            print(f"  Warning: no data found for '{metric}'")
            continue
        x, y = _smooth(sub, window_s)
        ax2.plot(x, y, label=metric, color=COLORS[metric], linewidth=0.8, alpha=0.85)

    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Quality (0 – 1)")
    ax2.set_title(f"Eyelid Opening Quality — {participant} / {condition}{win_label}")
    ax2.legend(loc="lower left")
    ax2.set_ylim(0, 1.05)
    ax2.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()

    out_dir = os.path.join(BASE_DIR, participant)
    out_path = os.path.join(out_dir, f"{condition.replace('.csv','')}_quality.png")
    fig.savefig(out_path, dpi=150)
    print(f"\nPlot saved → {out_path}")
    plt.show()

# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    # Participant
    participants = [d for d in list_options(BASE_DIR) if os.path.isdir(os.path.join(BASE_DIR, d))]
    if not participants:
        print("No participant folders found.")
        return
    participant = prompt_choice("Select participant:", participants)

    # Condition
    p_dir = os.path.join(BASE_DIR, participant)
    conditions = [f for f in list_options(p_dir) if f.endswith(".csv")]
    if not conditions:
        print(f"No CSV files found in {p_dir}")
        return
    condition = prompt_choice("Select condition:", conditions)

    # Sliding window
    while True:
        raw = input("\nSliding-window size in seconds (0 = raw): ").strip()
        try:
            window_s = float(raw)
            if window_s >= 0:
                break
        except ValueError:
            pass
        print("  Please enter a non-negative number.")

    # Quality threshold
    while True:
        raw = input("Quality threshold (default 0.8): ").strip()
        if raw == "":
            threshold = 0.8
            break
        try:
            threshold = float(raw)
            if 0.0 <= threshold <= 1.0:
                break
        except ValueError:
            pass
        print("  Please enter a value between 0 and 1.")

    csv_path = os.path.join(p_dir, condition)
    print(f"\nLoading {csv_path} …")
    df = load_metrics(csv_path)
    print(f"  {len(df)} SmartEyeProBridge quality rows found.")

    if df.empty:
        print("No matching data to plot.")
        return

    print_quality_stats(df, threshold)
    plot_metrics(df, participant, condition, window_s)

if __name__ == "__main__":
    main()
