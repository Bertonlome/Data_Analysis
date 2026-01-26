#!/usr/bin/env python3
"""
Main script to process flight simulation data analysis.

This script:
1. Copies data files from the QN-ACTR-XPlane results directory
2. Runs eye movement analysis
3. Runs trace analysis
4. Runs mental workload analysis

All plots are generated in their respective subdirectories.
"""

import os
import shutil
import subprocess
import sys
import json
import re
from pathlib import Path
from datetime import datetime

# Source directory with the data files
SOURCE_DIR = "/home/cavok3/Desktop/DEV/QN-ACTR-XPlane 2022-03-06/out/production/QN workspace/HMI_1/Results"

# Current working directory (where this script is located)
BASE_DIR = Path(__file__).parent.absolute()

# Create timestamped output directory
TIMESTAMP = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
OUTPUT_DIR = BASE_DIR / "output" / f"results_{TIMESTAMP}"

# Data file mappings: (source_filename, destination_folder, destination_filename)
DATA_FILES = [
    ("results_eye_movement.txt", "eye_movement", "results_eye_movement.txt"),
    ("trace.txt", "trace_analyzer", "trace.txt"),
    ("results_mental_workload.txt", "workload_analyzer", "results_mental_workload.txt"),
]

# Analysis scripts to run (in order)
ANALYSIS_SCRIPTS = [
    ("eye_movement/eye-movement-analyzer.py", "Eye Movement Analysis"),
    ("trace_analyzer/trace-analyzer.py", "Trace Analysis"),
    ("workload_analyzer/workload_analyzer.py", "Mental Workload Analysis"),
    ("team-analyzer/team-analyzer.py", "Team Analysis"),
]


class TaskEvent:
    """Represents a task event from the TARS Agent state transitions"""
    def __init__(self, timestamp, task_object, value, csv_timestamp):
        self.timestamp = timestamp  # Time in trace.txt coordinates
        self.task_object = task_object
        self.value = value
        self.csv_timestamp = csv_timestamp  # Original CSV timestamp
    
    def __repr__(self):
        return f"TaskEvent({self.timestamp:.2f}s, {self.task_object}, {self.value})"


def parse_tars_tasks_from_csv(csv_filepath):
    """
    Parse TARS Agent task events from the CSV file.
    Returns list of TaskEvent objects with CSV timestamps.
    """
    events = []
    
    with open(csv_filepath, 'r', encoding='utf-8') as f:
        header = f.readline()  # Skip header
        
        for line in f:
            parts = line.strip().split(';')
            if len(parts) >= 7:
                agent = parts[2]
                source_type = parts[3]  # This is the event source (e.g., "current_state")
                timestamp_str = parts[1]
                value_str = parts[6]
                
                # Look for TARS Agent current_state events
                if agent == 'TARS Agent' and source_type == 'current_state':
                    try:
                        # The value_str is wrapped in quotes and has escaped quotes inside
                        # Remove outer quotes if present
                        if value_str.startswith('"') and value_str.endswith('"'):
                            value_str = value_str[1:-1]
                        
                        # Unescape double quotes
                        value_str = value_str.replace('""', '"')
                        
                        # Parse the JSON value
                        state_data = json.loads(value_str)
                        task_object = state_data.get('task_object', '')
                        task_value = state_data.get('value', '')
                        csv_timestamp = float(timestamp_str)
                        
                        # Skip IDLE and empty tasks
                        if task_object and task_object != 'Idle':
                            # Create TaskEvent with CSV timestamp (will sync later)
                            event = TaskEvent(csv_timestamp, task_object, task_value, csv_timestamp)
                            events.append(event)
                    except (json.JSONDecodeError, ValueError) as e:
                        # Silently skip problematic lines
                        continue
    
    return events


def find_first_audio_event_time(trace_filepath):
    """
    Find the timestamp of the first audio event in trace.txt.
    Returns the timestamp as a float.
    """
    with open(trace_filepath, 'r', encoding='utf-8') as f:
        for line in f:
            # Look for the first AUDIO SET-BUFFER-CHUNK event
            if 'AUDIO' in line and 'SET-BUFFER-CHUNK' in line:
                # Extract timestamp from the beginning of the line
                match = re.match(r'\s*([\d.]+)\s+AUDIO', line)
                if match:
                    return float(match.group(1))
    return None


def synchronize_task_events(task_events, trace_filepath):
    """
    Synchronize task events from CSV with trace.txt timeline.
    Maps the first task event to the first audio event.
    
    Args:
        task_events: List of TaskEvent objects with CSV timestamps
        trace_filepath: Path to trace.txt file
    
    Returns:
        List of TaskEvent objects with synchronized timestamps in trace.txt coordinates
    """
    if not task_events:
        return []
    
    # Find the first audio event time in trace.txt
    first_audio_time = find_first_audio_event_time(trace_filepath)
    
    if first_audio_time is None:
        print("⚠ Warning: Could not find first audio event in trace.txt")
        return task_events
    
    # Get the CSV timestamp of the first task
    first_task_csv_time = task_events[0].csv_timestamp
    
    # Calculate the time offset
    time_offset = first_audio_time - first_task_csv_time
    
    print(f"\nTime Synchronization:")
    print(f"  First task CSV time: {first_task_csv_time:.3f}")
    print(f"  First audio trace time: {first_audio_time:.3f}")
    print(f"  Time offset: {time_offset:.3f}s\n")
    
    # Create new TaskEvent objects with synchronized timestamps
    synchronized_events = []
    for event in task_events:
        sync_time = event.csv_timestamp + time_offset
        sync_event = TaskEvent(sync_time, event.task_object, event.value, event.csv_timestamp)
        synchronized_events.append(sync_event)
    
    return synchronized_events


def save_task_events_json(task_events, output_path):
    """
    Save task events to a JSON file that can be read by trace-analyzer.py
    """
    task_data = [
        {
            'timestamp': event.timestamp,
            'task_object': event.task_object,
            'value': event.value
        }
        for event in task_events
    ]
    
    with open(output_path, 'w') as f:
        json.dump(task_data, f, indent=2)
    
    print(f"Saved {len(task_events)} synchronized task events to {output_path}")


def print_header(message):
    """Print a formatted header message."""
    print(f"\n{'='*80}")
    print(f"  {message}")
    print(f"{'='*80}\n")


def create_output_directory():
    """Create timestamped output directory structure."""
    print_header("Creating Output Directory")
    
    # Create main output directory and subdirectories
    subdirs = ['eye_movement', 'team-analyzer', 'trace_analyzer', 'workload_analyzer']
    
    for subdir in subdirs:
        output_subdir = OUTPUT_DIR / subdir
        output_subdir.mkdir(parents=True, exist_ok=True)
    
    print(f"Output directory created: {OUTPUT_DIR}")
    print(f"Subdirectories: {', '.join(subdirs)}\n")
    
    return OUTPUT_DIR


def cleanup_old_plots():
    """Remove old plot files from analyzer directories before running new analyses."""
    print_header("STEP 1: Cleaning Up Old Plots")
    
    analyzer_dirs = ['eye_movement', 'trace_analyzer', 'workload_analyzer', 'team-analyzer']
    total_removed = 0
    
    for dir_name in analyzer_dirs:
        dir_path = BASE_DIR / dir_name
        if dir_path.exists():
            # Remove PNG and EPS files (but not background images)
            for ext in ['*.png', '*.eps']:
                for plot_file in dir_path.glob(ext):
                    if 'cockpit_picture' not in plot_file.name:
                        plot_file.unlink()
                        total_removed += 1
    
    if total_removed > 0:
        print(f"Removed {total_removed} old plot files from analyzer directories.\n")
    else:
        print("No old plot files found.\n")


def copy_data_files():
    """Copy data files from source directory to appropriate subdirectories."""
    print_header("STEP 2: Copying Data Files")
    
    source_path = Path(SOURCE_DIR)
    
    # Check if source directory exists
    if not source_path.exists():
        print(f"❌ ERROR: Source directory does not exist:")
        print(f"   {SOURCE_DIR}")
        print(f"\nPlease verify the path and try again.")
        sys.exit(1)
    
    print(f"Source directory: {SOURCE_DIR}\n")
    
    # Copy each data file
    success_count = 0
    for source_file, dest_folder, dest_file in DATA_FILES:
        source_file_path = source_path / source_file
        dest_folder_path = BASE_DIR / dest_folder
        dest_file_path = dest_folder_path / dest_file
        
        # Create destination folder if it doesn't exist
        dest_folder_path.mkdir(parents=True, exist_ok=True)
        
        # Copy the file
        if source_file_path.exists():
            try:
                shutil.copy2(source_file_path, dest_file_path)
                print(f"✓ Copied: {source_file}")
                print(f"  → {dest_folder}/{dest_file}")
                success_count += 1
            except Exception as e:
                print(f"❌ Error copying {source_file}: {e}")
        else:
            print(f"⚠ Warning: {source_file} not found in source directory")
    
    # Handle team-analyzer CSV files separately (use whatever exists locally)
    team_analyzer_dir = BASE_DIR / "team-analyzer"
    if team_analyzer_dir.exists():
        csv_files = list(team_analyzer_dir.glob("*.csv"))
        if csv_files:
            print(f"✓ Using existing team-analyzer CSV file:")
            for csv_file in csv_files:
                print(f"  → team-analyzer/{csv_file.name}")
            success_count += 1
        else:
            print(f"⚠ Warning: No CSV files found in team-analyzer/")
    else:
        print(f"⚠ Warning: team-analyzer/ directory not found")
    
    print(f"\n{success_count}/{len(DATA_FILES) + 1} files ready for analysis.")
    
    if success_count == 0:
        print("\n❌ No files were copied. Aborting analysis.")
        sys.exit(1)
    
    return success_count == len(DATA_FILES) + 1


def run_analysis_script(script_path, description, task_events_file=None):
    """Run an analysis script and report results."""
    print_header(f"Running: {description}")
    
    script_full_path = BASE_DIR / script_path
    
    if not script_full_path.exists():
        print(f"❌ ERROR: Script not found: {script_path}")
        return False
    
    try:
        # Get the Python interpreter from the virtual environment
        venv_python = BASE_DIR / ".venv" / "bin" / "python"
        
        # Use system python if venv doesn't exist
        if not venv_python.exists():
            python_cmd = sys.executable
        else:
            python_cmd = str(venv_python)
        
        # Build command with optional task events file argument
        cmd = [python_cmd, str(script_full_path)]
        if task_events_file:
            cmd.append(task_events_file)
        
        # Run the script with a timeout
        result = subprocess.run(
            cmd,
            cwd=str(BASE_DIR),
            capture_output=True,
            text=True,
            timeout=120  # 2 minute timeout per script
        )
        
        # Print the output
        if result.stdout:
            print(result.stdout)
        
        if result.stderr:
            # Filter out matplotlib warnings
            stderr_lines = result.stderr.split('\n')
            filtered_stderr = [line for line in stderr_lines 
                             if 'UserWarning' not in line 
                             and 'FigureCanvasAgg is non-interactive' not in line
                             and line.strip()]
            if filtered_stderr:
                print("Warnings/Errors:")
                print('\n'.join(filtered_stderr))
        
        if result.returncode == 0:
            print(f"\n✓ {description} completed successfully!")
            return True
        else:
            print(f"\n❌ {description} failed with exit code {result.returncode}")
            return False
            
    except Exception as e:
        print(f"❌ Error running {description}: {e}")
        return False


def copy_results_to_output():
    """Copy source data files and generated plots to output directory."""
    print_header("STEP 4: Copying Results to Output Directory")
    
    # Directories to process
    analysis_dirs = {
        'eye_movement': ['results_eye_movement.txt'],
        'trace_analyzer': ['trace.txt'],
        'workload_analyzer': ['results_mental_workload.txt'],
        'team-analyzer': ['25_01_26 - 16_05_27.csv']
    }
    
    total_files_copied = 0
    
    for dir_name, source_files in analysis_dirs.items():
        source_dir = BASE_DIR / dir_name
        output_subdir = OUTPUT_DIR / dir_name
        
        if not source_dir.exists():
            continue
        
        # Check if any plots were generated (PNG or EPS files)
        png_files = [f for f in source_dir.glob('*.png') if 'cockpit_picture' not in f.name]
        eps_files = [f for f in source_dir.glob('*.eps')]
        has_plots = len(png_files) > 0 or len(eps_files) > 0
        
        # Skip copying if no plots were generated (no analysis results)
        if not has_plots:
            print(f"⊘ Skipping {dir_name}: No analysis results generated")
            continue
        
        # Copy source data files
        for source_file in source_files:
            source_path = source_dir / source_file
            if source_path.exists():
                shutil.copy2(source_path, output_subdir / source_file)
                print(f"✓ Copied source: {dir_name}/{source_file}")
                total_files_copied += 1
        
        # Copy all PNG and EPS files
        for plot_file in png_files:
            shutil.copy2(plot_file, output_subdir / plot_file.name)
            total_files_copied += 1
        
        for plot_file in eps_files:
            shutil.copy2(plot_file, output_subdir / plot_file.name)
            total_files_copied += 1
        
        # Count copied plots
        png_count = len(list(output_subdir.glob('*.png')))
        eps_count = len(list(output_subdir.glob('*.eps')))
        if png_count > 0 or eps_count > 0:
            print(f"  → {dir_name}: {png_count} PNG, {eps_count} EPS plots")
    
    print(f"\nTotal files copied to output: {total_files_copied}")
    print(f"Output location: {OUTPUT_DIR}\n")
    
    return total_files_copied


def run_all_analyses():
    """Run all analysis scripts."""
    print_header("STEP 3: Running Analysis Scripts")
    
    # Parse and synchronize task events for trace analyzer
    task_events_file = None
    csv_path = BASE_DIR / "team-analyzer" / "25_01_26 - 16_05_27.csv"
    trace_path = BASE_DIR / "trace_analyzer" / "trace.txt"
    
    if csv_path.exists() and trace_path.exists():
        print("Preparing task synchronization data...")
        print(f"  CSV file: {csv_path}")
        print(f"  Trace file: {trace_path}")
        
        # Parse tasks from CSV
        task_events = parse_tars_tasks_from_csv(csv_path)
        print(f"  Parsed {len(task_events)} task events from CSV")
        
        # Synchronize with trace.txt timeline
        if task_events:
            sync_events = synchronize_task_events(task_events, trace_path)
            
            # Save to temporary JSON file
            task_events_file = str(BASE_DIR / "task_events.json")
            save_task_events_json(sync_events, task_events_file)
        else:
            print("  No task events found in CSV")
    else:
        print("  CSV or trace file not found, skipping task synchronization")
    
    print()
    
    success_count = 0
    total_scripts = len(ANALYSIS_SCRIPTS)
    
    for script_path, description in ANALYSIS_SCRIPTS:
        # Pass task events file only to trace analyzer
        if "trace-analyzer" in script_path and task_events_file:
            if run_analysis_script(script_path, description, task_events_file):
                success_count += 1
            else:
                print(f"\n⚠ Warning: {description} did not complete successfully.")
                print("Continuing with remaining analyses...\n")
        else:
            if run_analysis_script(script_path, description):
                success_count += 1
            else:
                print(f"\n⚠ Warning: {description} did not complete successfully.")
                print("Continuing with remaining analyses...\n")
    
    # Cleanup temporary file
    if task_events_file and Path(task_events_file).exists():
        Path(task_events_file).unlink()
    
    print_header(f"Analysis Complete: {success_count}/{total_scripts} successful")
    
    return success_count == total_scripts


def main():
    """Main execution function."""
    print_header("Flight Simulation Data Analysis Pipeline")
    print(f"Working directory: {BASE_DIR}")
    print(f"Timestamp: {TIMESTAMP}\n")
    
    # Step 0: Create output directory
    create_output_directory()
    
    # Step 1: Clean up old plots
    cleanup_old_plots()
    
    # Step 2: Copy data files
    all_files_copied = copy_data_files()
    
    if not all_files_copied:
        print("\n⚠ Warning: Not all files were copied. Proceeding anyway...")
    
    # Step 3: Run analyses
    all_analyses_successful = run_all_analyses()
    
    # Step 4: Copy results to output directory
    copy_results_to_output()
    
    # Final summary
    print_header("SUMMARY")
    
    # List generated plot files
    plot_dirs = ["eye_movement", "trace_analyzer", "workload_analyzer", "team-analyzer"]
    total_plots = 0
    
    print("Generated visualizations:\n")
    for plot_dir in plot_dirs:
        plot_path = BASE_DIR / plot_dir
        if plot_path.exists():
            png_files = list(plot_path.glob("*.png"))
            if png_files:
                print(f"{plot_dir}/:")
                for png_file in sorted(png_files):
                    if png_file.name != "cessna_mustang_cockpit_picture.png":  # Skip background image
                        print(f"  ✓ {png_file.name}")
                        total_plots += 1
                print()
    
    print(f"Total plots generated: {total_plots}")
    print(f"\nResults saved to: {OUTPUT_DIR}")
    
    if all_analyses_successful:
        print("\n🎉 All analyses completed successfully!")
        return 0
    else:
        print("\n⚠ Some analyses encountered issues. Please review the output above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
