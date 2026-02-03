#!/usr/bin/env python3
"""
Cross-Run Comparison Analyzer
Compares task durations and overall scenario times across multiple experimental runs
"""

import csv
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import sys

def parse_csv_file(filepath):
    """Parse the CSV file and extract TARS Agent current_state events"""
    events = []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=';')
        
        for row in reader:
            if len(row) >= 7:
                # Check if this is a TARS Agent current_state event
                if len(row) > 3 and row[2] == "TARS Agent" and row[3] == "current_state":
                    try:
                        # Parse the JSON data in the last column
                        # Note: Python's CSV reader already handles quote escaping
                        value_str = row[6]
                        
                        # Try to parse JSON
                        json_data = json.loads(value_str)
                        
                        event = {
                            'uuid': row[0],
                            'timestamp': float(row[1]),
                            'agent': row[2],
                            'event_type': row[3],
                            'task_object': json_data.get('task_object', ''),
                            'value': json_data.get('value', ''),
                            'procedure': json_data.get('procedure', ''),
                            'category': json_data.get('category', ''),
                        }
                        
                        # Only add if we have the essential fields
                        if event['task_object']:
                            events.append(event)
                    except (json.JSONDecodeError, KeyError, ValueError) as e:
                        # Skip malformed rows
                        continue
    
    return events

def parse_cognitive_task_starts(filepath):
    """Parse the CSV file and extract cognitive model task start events"""
    events = []
    task_start_patterns = [
        'x-6-not-waiting-new-task-both',
        'x-6-not-waiting-new-task-object',
        'x-6-not-waiting-new-task-value'
    ]
    
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=';')
        
        for row in reader:
            if len(row) >= 7:
                if len(row) > 3 and row[2] == "Cognitive_Model" and row[3] == "production_selected":
                    production_name = row[6] if len(row) > 6 else ''
                    if production_name in task_start_patterns:
                        event = {
                            'timestamp': float(row[1]),
                            'production': production_name,
                        }
                        events.append(event)
    
    return events

def parse_cognitive_task_ends(filepath):
    """Parse the CSV file and extract cognitive model task completion events"""
    events = []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=';')
        
        for row in reader:
            if len(row) >= 7:
                if len(row) > 3 and row[2] == "Cognitive_Model" and row[3] == "production_selected":
                    production_name = row[6] if len(row) > 6 else ''
                    if production_name == 'form-task-done':
                        event = {
                            'timestamp': float(row[1]),
                            'production': production_name,
                        }
                        events.append(event)
    
    return events

def parse_all_productions(filepath):
    """Parse all cognitive model production events for coordination analysis"""
    productions = []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=';')
        
        for row in reader:
            if len(row) >= 7:
                if len(row) > 3 and row[2] == "Cognitive_Model" and row[3] == "production_selected":
                    production_name = row[6] if len(row) > 6 else ''
                    if production_name:  # Any production
                        event = {
                            'timestamp': float(row[1]),
                            'production': production_name,
                        }
                        productions.append(event)
    
    return productions

def calculate_coordination_time(productions, start_time, end_time):
    """Calculate total coordination time based on t-i productions"""
    # Filter productions within time range
    task_productions = [
        p for p in productions
        if start_time <= p['timestamp'] < end_time
    ]
    
    # Find coordination sequences (productions starting with 't-i')
    coordination_sequences = []
    current_sequence = None
    
    for j, prod in enumerate(task_productions):
        is_coordination = prod['production'].startswith('t-i')
        
        if is_coordination:
            if current_sequence is None:
                # Start new coordination sequence
                current_sequence = {
                    'start_time': prod['timestamp'],
                    'num_productions': 1
                }
            else:
                # Continue sequence
                current_sequence['num_productions'] += 1
        else:
            # Non-coordination production ends the sequence
            if current_sequence is not None:
                current_sequence['end_time'] = task_productions[j-1]['timestamp']
                current_sequence['duration'] = current_sequence['end_time'] - current_sequence['start_time']
                coordination_sequences.append(current_sequence)
                current_sequence = None
    
    # Handle case where sequence extends to the end
    if current_sequence is not None:
        current_sequence['end_time'] = task_productions[-1]['timestamp']
        current_sequence['duration'] = current_sequence['end_time'] - current_sequence['start_time']
        coordination_sequences.append(current_sequence)
    
    # Calculate total coordination time
    total_coord_time = sum(seq['duration'] for seq in coordination_sequences)
    
    return total_coord_time

def parse_workload_file(filepath):
    """Parse mental workload data from results file"""
    try:
        df = pd.read_csv(filepath, sep='\t')
        # Clean up column names
        df.columns = [col.replace('ClockTime(s)_UtilizationValuesFrom:', 'Time').strip() for col in df.columns]
        return df
    except Exception as e:
        print(f"  Warning: Could not parse workload file: {e}")
        return None

def calculate_workload_statistics(df):
    """Calculate statistics for workload data"""
    stats = {
        'overall': {
            'mean': df['Overall_Utilization'].mean(),
            'median': df['Overall_Utilization'].median(),
            'std': df['Overall_Utilization'].std(),
            'min': df['Overall_Utilization'].min(),
            'max': df['Overall_Utilization'].max(),
            'values': df['Overall_Utilization'].values
        },
        'perceptual': {
            'mean': df['Perceptual_SubNetwork'].mean(),
            'median': df['Perceptual_SubNetwork'].median(),
            'std': df['Perceptual_SubNetwork'].std(),
            'min': df['Perceptual_SubNetwork'].min(),
            'max': df['Perceptual_SubNetwork'].max(),
            'values': df['Perceptual_SubNetwork'].values
        },
        'cognitive': {
            'mean': df['Cognitive_SubNetwork'].mean(),
            'median': df['Cognitive_SubNetwork'].median(),
            'std': df['Cognitive_SubNetwork'].std(),
            'min': df['Cognitive_SubNetwork'].min(),
            'max': df['Cognitive_SubNetwork'].max(),
            'values': df['Cognitive_SubNetwork'].values
        },
        'motor': {
            'mean': df['Motor_SubNetwork'].mean(),
            'median': df['Motor_SubNetwork'].median(),
            'std': df['Motor_SubNetwork'].std(),
            'min': df['Motor_SubNetwork'].min(),
            'max': df['Motor_SubNetwork'].max(),
            'values': df['Motor_SubNetwork'].values
        }
    }
    return stats

def extract_run_data(csv_filepath, run_name):
    """Extract task timing data from a single run"""
    # Parse events
    state_events = parse_csv_file(csv_filepath)
    task_starts = parse_cognitive_task_starts(csv_filepath)
    task_ends = parse_cognitive_task_ends(csv_filepath)
    productions = parse_all_productions(csv_filepath)
    
    if not state_events:
        print(f"Warning: No task events found in {csv_filepath}")
        return None
    
    print(f"Found {len(state_events)} TARS Agent state transitions")
    print(f"Found {len(task_starts)} cognitive task start events")
    print(f"Found {len(task_ends)} cognitive task end events (form-task-done)")
    
    # Find start event - must be "Takeoff clearance"
    start_event = None
    start_idx = 0
    for i, event in enumerate(state_events):
        if event['task_object'] == 'Takeoff clearance':
            start_event = event
            start_idx = i
            break
    
    if not start_event:
        print(f"Warning: No 'Takeoff clearance' event found in {csv_filepath}")
        return None
    
    # End event is the last event (used for timing but excluded from analysis)
    end_event = state_events[-1]
    
    print(f"Using start event: {start_event['task_object']}")
    print(f"Using last event as end marker: {end_event['task_object']}")
    
    # Extract tasks from Takeoff clearance to the end (excluding the end marker itself)
    tasks = state_events[start_idx:-1]  # Exclude CAS (last event)
    
    print(f"Extracted {len(tasks)} tasks from Takeoff clearance to end (excluding {end_event['task_object']})")
    
    # Calculate FSM task durations
    task_durations = []
    for i, task in enumerate(tasks):
        if i < len(tasks) - 1:
            next_timestamp = tasks[i + 1]['timestamp']
        else:
            # Last task duration goes until the end event
            next_timestamp = end_event['timestamp']
        
        duration = next_timestamp - task['timestamp']
        task_durations.append({
            'task_object': task['task_object'],
            'value': task['value'],
            'duration': duration,
        })
    
    # Calculate active cognitive times - only include complete task pairs after start_event
    # Pattern observed: if there's a task before scenario start, then:
    #   End[0] = end of pre-start task
    #   Start[0] = start of task 1, End[1] = end of task 1
    #   Start[1] = start of task 2, End[2] = end of task 2, etc.
    # So Start[i] pairs with End[i+1]
    active_times = []
    
    # Check if first end comes before first start (indicates pre-start task)
    if len(task_starts) > 0 and len(task_ends) > 0:
        if task_ends[0]['timestamp'] < task_starts[0]['timestamp']:
            # Offset pairing: Start[i] pairs with End[i+1]
            for i in range(len(task_starts)):
                end_idx = i + 1
                if end_idx < len(task_ends):
                    start_event_cog = task_starts[i]
                    end_event_cog = task_ends[end_idx]
                    
                    # Only include if start is after scenario start
                    if start_event_cog['timestamp'] >= start_event['timestamp']:
                        if end_event_cog['timestamp'] > start_event_cog['timestamp']:
                            active_duration = end_event_cog['timestamp'] - start_event_cog['timestamp']
                            active_times.append(active_duration)
        else:
            # Normal pairing: Start[i] pairs with End[i]
            for i in range(min(len(task_starts), len(task_ends))):
                start_event_cog = task_starts[i]
                end_event_cog = task_ends[i]
                
                # Only include if start is after scenario start
                if start_event_cog['timestamp'] >= start_event['timestamp']:
                    if end_event_cog['timestamp'] > start_event_cog['timestamp']:
                        active_duration = end_event_cog['timestamp'] - start_event_cog['timestamp']
                        active_times.append(active_duration)
    
    # Overall scenario time
    total_time = end_event['timestamp'] - start_event['timestamp']
    total_active_time = sum(active_times) if active_times else 0
    
    # Calculate total coordination time
    total_coord_time = calculate_coordination_time(productions, start_event['timestamp'], end_event['timestamp'])
    
    # Parse workload data if available
    workload_file = Path(csv_filepath).parent.parent / 'workload_analyzer' / 'results_mental_workload.txt'
    workload_stats = None
    if workload_file.exists():
        df = parse_workload_file(workload_file)
        if df is not None:
            workload_stats = calculate_workload_statistics(df)
    
    return {
        'run_name': run_name,
        'tasks': task_durations,
        'active_times': active_times,
        'total_time': total_time,
        'total_active_time': total_active_time,
        'total_coordination_time': total_coord_time,
        'num_tasks': len(tasks),
        'start_time': start_event['timestamp'],
        'end_time': end_event['timestamp'],
        'workload': workload_stats,
    }

def find_csv_files(output_dir):
    """Find all CSV files in the output subfolders matching results_n_* pattern"""
    import re
    csv_files = []
    output_path = Path(output_dir)
    
    # Pattern to match results_n_timestamp folders (e.g., results_0_2026_02_03_10_33_15)
    results_pattern = re.compile(r'^results_(\d+)_')
    
    for subfolder in output_path.iterdir():
        if subfolder.is_dir() and results_pattern.match(subfolder.name):
            team_analyzer_dir = subfolder / 'team-analyzer'
            if team_analyzer_dir.exists():
                # Find CSV files in the team-analyzer directory
                for csv_file in team_analyzer_dir.glob('*.csv'):
                    csv_files.append({
                        'path': csv_file,
                        'run_name': subfolder.name,
                    })
    
    return csv_files

def create_comparison_plots(runs_data, output_dir):
    """Create comparison plots across all runs"""
    # 1. Overall Scenario Time Comparison
    fig, ax = plt.subplots(figsize=(14, 6))
    
    run_names = [r['run_name'] for r in runs_data]
    total_times = [r['total_time'] for r in runs_data]
    active_times = [r['total_active_time'] for r in runs_data]
    coord_times = [r['total_coordination_time'] for r in runs_data]
    
    x = np.arange(len(run_names))
    width = 0.25
    
    bars1 = ax.bar(x - width, total_times, width, label='Total FSM Time', 
                   color='steelblue', alpha=0.8, edgecolor='black')
    bars2 = ax.bar(x, active_times, width, label='Total Active Cognitive Time', 
                   color='coral', alpha=0.8, edgecolor='black')
    bars3 = ax.bar(x + width, coord_times, width, label='Total Coordination Time',
                   color='mediumseagreen', alpha=0.8, edgecolor='black')
    
    # Add value labels on bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}s',
                   ha='center', va='bottom', fontsize=8)
    
    ax.set_xlabel('Experimental Run', fontsize=12, fontweight='bold')
    ax.set_ylabel('Time (seconds)', fontsize=12, fontweight='bold')
    ax.set_title('Overall Scenario Time Comparison Across Runs', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(run_names, rotation=45, ha='right')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'scenario_time_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig(Path(output_dir) / 'scenario_time_comparison.eps', format='eps', bbox_inches='tight')
    print(f"Saved scenario time comparison plot")
    plt.close()
    
    # 2. Task-by-Task Duration Comparison
    # Get common tasks across all runs
    all_task_objects = set()
    for run in runs_data:
        for task in run['tasks']:
            all_task_objects.add(task['task_object'])
    
    task_objects = sorted(all_task_objects)
    
    # Create grouped bar chart
    fig, ax = plt.subplots(figsize=(16, 10))
    
    x = np.arange(len(task_objects))
    width = 0.8 / len(runs_data)
    
    for i, run in enumerate(runs_data):
        # Create a mapping of task_object to duration
        task_durations_map = {task['task_object']: task['duration'] for task in run['tasks']}
        
        # Get durations in the correct order (0 if task doesn't exist in this run)
        durations = [task_durations_map.get(task_obj, 0) for task_obj in task_objects]
        
        offset = width * (i - len(runs_data)/2 + 0.5)
        bars = ax.bar(x + offset, durations, width, label=run['run_name'], 
                     alpha=0.8, edgecolor='black')
    
    ax.set_xlabel('Task', fontsize=12, fontweight='bold')
    ax.set_ylabel('Duration (seconds)', fontsize=12, fontweight='bold')
    ax.set_title('Task Duration Comparison Across Runs (FSM State Duration)', 
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(task_objects, rotation=45, ha='right', fontsize=9)
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'task_duration_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig(Path(output_dir) / 'task_duration_comparison.eps', format='eps', bbox_inches='tight')
    print(f"Saved task duration comparison plot")
    plt.close()
    
    # 3. Summary Statistics Table
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare table data
    table_data = [['Run Name', 'Total Time (s)', 'Active Time (s)', 'Active %', 
                   'Num Tasks', 'Avg Task Duration (s)']]
    
    for run in runs_data:
        avg_task_duration = run['total_time'] / run['num_tasks'] if run['num_tasks'] > 0 else 0
        active_percentage = (run['total_active_time'] / run['total_time'] * 100) if run['total_time'] > 0 else 0
        
        table_data.append([
            run['run_name'],
            f"{run['total_time']:.2f}",
            f"{run['total_active_time']:.2f}",
            f"{active_percentage:.1f}%",
            str(run['num_tasks']),
            f"{avg_task_duration:.2f}"
        ])
    
    table = ax.table(cellText=table_data, cellLoc='left', loc='center',
                    colWidths=[0.35, 0.12, 0.12, 0.10, 0.10, 0.15])
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # Style header row
    for i in range(len(table_data[0])):
        cell = table[(0, i)]
        cell.set_facecolor('#4472C4')
        cell.set_text_props(weight='bold', color='white')
    
    # Alternate row colors
    for i in range(1, len(table_data)):
        for j in range(len(table_data[0])):
            cell = table[(i, j)]
            if i % 2 == 0:
                cell.set_facecolor('#E7E6E6')
    
    plt.title('Cross-Run Summary Statistics', fontsize=14, fontweight='bold', pad=20)
    plt.savefig(Path(output_dir) / 'summary_statistics.png', dpi=300, bbox_inches='tight')
    plt.savefig(Path(output_dir) / 'summary_statistics.eps', format='eps', bbox_inches='tight')
    print(f"Saved summary statistics table")
    plt.close()
    
    # 4. Workload Comparison - Overall Utilization Box Plot
    runs_with_workload = [r for r in runs_data if r.get('workload') is not None]
    
    if len(runs_with_workload) >= 2:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Box plot for overall utilization
        workload_data = [r['workload']['overall']['values'] for r in runs_with_workload]
        run_labels = [r['run_name'] for r in runs_with_workload]
        
        bp1 = ax1.boxplot(workload_data, labels=run_labels, patch_artist=True,
                          boxprops=dict(facecolor='lightblue', alpha=0.7),
                          medianprops=dict(color='red', linewidth=2),
                          whiskerprops=dict(linewidth=1.5),
                          capprops=dict(linewidth=1.5))
        
        ax1.set_ylabel('Overall Utilization', fontsize=12, fontweight='bold')
        ax1.set_title('Overall Workload Distribution Across Runs', fontsize=14, fontweight='bold')
        ax1.set_xticklabels(run_labels, rotation=45, ha='right')
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Bar chart with means and error bars
        means = [r['workload']['overall']['mean'] for r in runs_with_workload]
        stds = [r['workload']['overall']['std'] for r in runs_with_workload]
        
        x_pos = np.arange(len(run_labels))
        bars = ax2.bar(x_pos, means, yerr=stds, capsize=5, 
                       color='steelblue', alpha=0.7, edgecolor='black')
        
        # Add value labels
        for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
            ax2.text(bar.get_x() + bar.get_width()/2., mean + std + 0.01,
                    f'{mean:.3f}±{std:.3f}',
                    ha='center', va='bottom', fontsize=9)
        
        ax2.set_ylabel('Mean Overall Utilization', fontsize=12, fontweight='bold')
        ax2.set_title('Mean Overall Workload (±SD)', fontsize=14, fontweight='bold')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(run_labels, rotation=45, ha='right')
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(Path(output_dir) / 'workload_overall_comparison.png', dpi=300, bbox_inches='tight')
        plt.savefig(Path(output_dir) / 'workload_overall_comparison.eps', format='eps', bbox_inches='tight')
        print(f"Saved overall workload comparison plot")
        plt.close()
        
        # 5. Workload Comparison - Subnetworks
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        subnetworks = ['perceptual', 'cognitive', 'motor']
        titles = ['Perceptual SubNetwork', 'Cognitive SubNetwork', 'Motor SubNetwork']
        colors = ['lightcoral', 'lightgreen', 'lightskyblue']
        
        for idx, (subnet, title, color) in enumerate(zip(subnetworks, titles, colors)):
            row = idx // 2
            col = idx % 2
            ax = axes[row, col]
            
            # Box plot for this subnetwork
            subnet_data = [r['workload'][subnet]['values'] for r in runs_with_workload]
            
            bp = ax.boxplot(subnet_data, labels=run_labels, patch_artist=True,
                           boxprops=dict(facecolor=color, alpha=0.7),
                           medianprops=dict(color='darkred', linewidth=2),
                           whiskerprops=dict(linewidth=1.5),
                           capprops=dict(linewidth=1.5))
            
            ax.set_ylabel('Utilization', fontsize=11, fontweight='bold')
            ax.set_title(f'{title} Distribution', fontsize=12, fontweight='bold')
            ax.set_xticklabels(run_labels, rotation=45, ha='right')
            ax.grid(True, alpha=0.3, axis='y')
        
        # Fourth subplot: Mean comparison bar chart for all subnetworks
        ax = axes[1, 1]
        
        x_pos = np.arange(len(run_labels))
        width = 0.25
        
        for idx, (subnet, title, color) in enumerate(zip(subnetworks, titles, colors)):
            means = [r['workload'][subnet]['mean'] for r in runs_with_workload]
            offset = (idx - 1) * width
            ax.bar(x_pos + offset, means, width, label=title, 
                   color=color, alpha=0.7, edgecolor='black')
        
        ax.set_ylabel('Mean Utilization', fontsize=11, fontweight='bold')
        ax.set_title('Mean Subnetwork Workload Comparison', fontsize=12, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(run_labels, rotation=45, ha='right')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(Path(output_dir) / 'workload_subnetworks_comparison.png', dpi=300, bbox_inches='tight')
        plt.savefig(Path(output_dir) / 'workload_subnetworks_comparison.eps', format='eps', bbox_inches='tight')
        print(f"Saved subnetwork workload comparison plot")
        plt.close()

def print_summary_report(runs_data):
    """Print a text summary of the comparison"""
    print("\n" + "="*80)
    print("CROSS-RUN COMPARISON SUMMARY")
    print("="*80)
    
    for run in runs_data:
        print(f"\nRun: {run['run_name']}")
        print(f"  Total scenario time: {run['total_time']:.2f}s")
        print(f"  Total active cognitive time: {run['total_active_time']:.2f}s")
        print(f"  Total coordination time: {run['total_coordination_time']:.2f}s")
        active_pct = (run['total_active_time']/run['total_time']*100) if run['total_time'] > 0 else 0
        coord_pct = (run['total_coordination_time']/run['total_time']*100) if run['total_time'] > 0 else 0
        print(f"  Active percentage: {active_pct:.1f}%")
        print(f"  Coordination percentage: {coord_pct:.1f}%")
        print(f"  Number of tasks: {run['num_tasks']}")
        avg_task = (run['total_time']/run['num_tasks']) if run['num_tasks'] > 0 else 0
        print(f"  Average task duration: {avg_task:.2f}s")
        
        # Workload statistics if available
        if run.get('workload'):
            wl = run['workload']
            print(f"  Overall workload (mean±std): {wl['overall']['mean']:.3f}±{wl['overall']['std']:.3f}")
            print(f"    Perceptual: {wl['perceptual']['mean']:.3f}±{wl['perceptual']['std']:.3f}")
            print(f"    Cognitive:  {wl['cognitive']['mean']:.3f}±{wl['cognitive']['std']:.3f}")
            print(f"    Motor:      {wl['motor']['mean']:.3f}±{wl['motor']['std']:.3f}")
    
    # Find fastest and slowest runs (excluding zero-time runs)
    valid_runs = [r for r in runs_data if r['total_time'] > 0]
    if len(valid_runs) >= 2:
        fastest = min(valid_runs, key=lambda r: r['total_time'])
        slowest = max(valid_runs, key=lambda r: r['total_time'])
        
        print("\n" + "-"*80)
        print(f"Fastest run: {fastest['run_name']} ({fastest['total_time']:.2f}s)")
        print(f"Slowest run: {slowest['run_name']} ({slowest['total_time']:.2f}s)")
        time_diff = slowest['total_time'] - fastest['total_time']
        pct_diff = ((slowest['total_time']/fastest['total_time'] - 1) * 100) if fastest['total_time'] > 0 else 0
        print(f"Time difference: {time_diff:.2f}s ({pct_diff:.1f}% slower)")
    print("="*80 + "\n")

def main():
    """Main execution function"""
    # Get the directory containing this script
    script_dir = Path(__file__).parent
    
    print("="*80)
    print("CROSS-RUN COMPARISON ANALYZER")
    print("="*80)
    print(f"Analyzing runs in: {script_dir}")
    print()
    
    # Find all CSV files
    csv_files = find_csv_files(script_dir)
    
    # Filter out specific runs to exclude
    EXCLUDED_RUNS = ['autonomy_centric_run_delays_unreliable_caught']
    csv_files = [cf for cf in csv_files if cf['run_name'] not in EXCLUDED_RUNS]
    
    if not csv_files:
        print("Error: No CSV files found in output subfolders")
        sys.exit(1)
    
    print(f"Found {len(csv_files)} runs to compare:")
    for cf in csv_files:
        print(f"  - {cf['run_name']}: {cf['path']}")
    print()
    
    # Extract data from each run
    runs_data = []
    for cf in csv_files:
        print(f"\nProcessing: {cf['run_name']}")
        print("-" * 80)
        
        run_data = extract_run_data(cf['path'], cf['run_name'])
        if run_data:
            runs_data.append(run_data)
    
    if not runs_data:
        print("Error: No valid run data extracted")
        sys.exit(1)
    
    print("\n" + "="*80)
    print(f"Successfully processed {len(runs_data)} runs")
    print("="*80)
    
    # Print summary report
    print_summary_report(runs_data)
    
    # Create comparison plots
    print("Generating comparison visualizations...")
    create_comparison_plots(runs_data, script_dir)
    
    print("\n" + "="*80)
    print("COMPARISON COMPLETE")
    print("="*80)
    print(f"Output saved to: {script_dir}")
    print("Generated files:")
    print("  - scenario_time_comparison.png/eps")
    print("  - task_duration_comparison.png/eps")
    print("  - summary_statistics.png/eps")
    
    # Check if workload plots were generated
    runs_with_workload = [r for r in runs_data if r.get('workload') is not None]
    if len(runs_with_workload) >= 2:
        print("  - workload_overall_comparison.png/eps")
        print("  - workload_subnetworks_comparison.png/eps")
    
    print("="*80)

if __name__ == "__main__":
    main()
