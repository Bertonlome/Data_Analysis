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
from pathlib import Path

# Source directory with the data files
SOURCE_DIR = "/home/cavok3/Desktop/DEV/QN-ACTR-XPlane 2022-03-06/out/production/QN workspace/HMI_1/Results"

# Current working directory (where this script is located)
BASE_DIR = Path(__file__).parent.absolute()

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
]


def print_header(message):
    """Print a formatted header message."""
    print(f"\n{'='*80}")
    print(f"  {message}")
    print(f"{'='*80}\n")


def copy_data_files():
    """Copy data files from source directory to appropriate subdirectories."""
    print_header("STEP 1: Copying Data Files")
    
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
    
    print(f"\n{success_count}/{len(DATA_FILES)} files copied successfully.")
    
    if success_count == 0:
        print("\n❌ No files were copied. Aborting analysis.")
        sys.exit(1)
    
    return success_count == len(DATA_FILES)


def run_analysis_script(script_path, description):
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
        
        # Run the script with a timeout
        result = subprocess.run(
            [python_cmd, str(script_full_path)],
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


def run_all_analyses():
    """Run all analysis scripts."""
    print_header("STEP 2: Running Analysis Scripts")
    
    success_count = 0
    total_scripts = len(ANALYSIS_SCRIPTS)
    
    for script_path, description in ANALYSIS_SCRIPTS:
        if run_analysis_script(script_path, description):
            success_count += 1
        else:
            print(f"\n⚠ Warning: {description} did not complete successfully.")
            print("Continuing with remaining analyses...\n")
    
    print_header(f"Analysis Complete: {success_count}/{total_scripts} successful")
    
    return success_count == total_scripts


def main():
    """Main execution function."""
    print_header("Flight Simulation Data Analysis Pipeline")
    print(f"Working directory: {BASE_DIR}")
    
    # Step 1: Copy data files
    all_files_copied = copy_data_files()
    
    if not all_files_copied:
        print("\n⚠ Warning: Not all files were copied. Proceeding anyway...")
    
    # Step 2: Run analyses
    all_analyses_successful = run_all_analyses()
    
    # Final summary
    print_header("SUMMARY")
    
    # List generated plot files
    plot_dirs = ["eye_movement", "trace_analyzer", "workload_analyzer"]
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
    
    if all_analyses_successful:
        print("\n🎉 All analyses completed successfully!")
        return 0
    else:
        print("\n⚠ Some analyses encountered issues. Please review the output above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
