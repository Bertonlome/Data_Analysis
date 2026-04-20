#!/usr/bin/env python3
"""
Cross-Run Comparison Analyzer
Compares task durations and overall scenario times across multiple experimental runs
"""

import argparse
import csv
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import sys
from scipy import stats

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

def parse_aoi_metrics_file(filepath):
    """Parse AoI metrics CSV file generated by eye-movement-analyzer"""
    try:
        df = pd.read_csv(filepath)
        aoi_metrics = {}
        
        for _, row in df.iterrows():
            aoi_name = row['AoI_Name']
            aoi_metrics[aoi_name] = {
                'fixation_count': int(row['Fixation_Count']),
                'percentage': float(row['Percentage']),
                'avg_dwell_time': float(row['Avg_Dwell_Time_s']),
                'total_dwell_time': float(row['Total_Dwell_Time_s'])
            }
        
        return aoi_metrics
    except Exception as e:
        print(f"  Warning: Could not parse AoI metrics file: {e}")
        return None

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
    
    # Parse AoI metrics if available
    aoi_metrics_file = Path(csv_filepath).parent.parent / 'eye_movement' / 'aoi_metrics.csv'
    aoi_metrics = None
    if aoi_metrics_file.exists():
        aoi_metrics = parse_aoi_metrics_file(aoi_metrics_file)
    
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
        'eye_movement': aoi_metrics,
    }

def find_csv_files(output_dir):
    """Find all CSV files in the output subfolders matching results_n_* pattern and group by run number"""
    import re
    from collections import defaultdict
    
    grouped_files = defaultdict(list)
    output_path = Path(output_dir)
    
    # Pattern to match both:
    # - results_n_timestamp (e.g., results_0_2026_02_03_10_33_15)
    # - run_X_results_n_timestamp (e.g., run_1_results_0_2026_02_03_11_44_30)
    # Use the run_X number if present, otherwise fall back to results_n
    results_pattern = re.compile(r'(?:run_(\d+)_)?results_(\d+)_')
    
    for subfolder in output_path.iterdir():
        match = results_pattern.match(subfolder.name)
        if subfolder.is_dir() and match:
            # Use the first number (run_X) if present, otherwise use second number (results_n)
            run_number = int(match.group(1)) if match.group(1) else int(match.group(2))
            team_analyzer_dir = subfolder / 'team-analyzer'
            if team_analyzer_dir.exists():
                # Find the main CSV file (cassandra_converted.csv)
                csv_file = team_analyzer_dir / 'cassandra_converted.csv'
                if csv_file.exists():
                    grouped_files[run_number].append({
                        'path': csv_file,
                        'folder_name': subfolder.name,
                        'run_folder': subfolder,  # Add the run folder path for task_summary loading
                    })
    
    # Convert to list format with run_number as identifier
    result = []
    for run_num in sorted(grouped_files.keys()):
        result.append({
            'run_number': run_num,
            'files': grouped_files[run_num]
        })
    
    return result


def calculate_confidence_interval(data, confidence=0.95):
    """Calculate mean and 95% confidence interval using t-distribution"""
    if len(data) == 0:
        return 0, 0, 0
    
    data = np.array(data)
    mean = np.mean(data)
    
    if len(data) == 1:
        return mean, 0, 0
    
    sem = stats.sem(data)  # Standard error of the mean
    ci = sem * stats.t.ppf((1 + confidence) / 2, len(data) - 1)  # 95% CI
    
    return mean, ci, sem


def load_task_summary_csv(run_folder):
    """Load task_summary.csv with JAE components from a run folder"""
    task_summary_file = run_folder / 'team-analyzer' / 'task_summary.csv'
    
    if not task_summary_file.exists():
        return None
    
    try:
        df = pd.read_csv(task_summary_file)
        tasks = []
        
        for _, row in df.iterrows():
            tasks.append({
                'task_object': row['Task_Object'],
                'task_value': row['Task_Value'],
                'fsm_duration': float(row['FSM_Duration_s']),
                'human_active_time': float(row['Human_Active_Time_s']),
                'tars_execution_time': float(row['TARS_Execution_Time_s']),
                'active_duration': float(row['Active_Duration_s']),
                'coordination_time': float(row['Coordination_Time_s'])
            })
        
        return tasks
    except Exception as e:
        print(f"  Warning: Could not load task summary CSV: {e}")
        return None


def calculate_jae_data_baseline(all_runs_data):
    """Calculate JAE-Data baseline (minimum Active Duration per task_object across all runs)
    
    Returns:
        Dictionary mapping task_object -> minimum_AD (to be used as ED baseline)
    """
    # Collect all Active Durations per task_object across all runs and repetitions
    task_ads = {}  # task_object -> list of Active Durations
    
    for run in all_runs_data:
        for rep_tasks in run.get('all_repetition_tasks', []):
            for task in rep_tasks:
                task_obj = task['task_object']
                ad = task['active_duration']
                
                if task_obj not in task_ads:
                    task_ads[task_obj] = []
                task_ads[task_obj].append(ad)
    
    # Calculate minimum AD for each task_object (this becomes ED for JAE-Data)
    ed_baseline = {}
    for task_obj, ad_values in task_ads.items():
        ed_baseline[task_obj] = min(ad_values)
    
    return ed_baseline


def calculate_jae_metrics(tasks, ed_baseline):
    """Calculate JAE-Data for each task using the ED baseline
    
    Args:
        tasks: List of task dictionaries with active_duration
        ed_baseline: Dictionary mapping task_object -> ED (minimum AD)
    
    Returns:
        List of tasks with jae_data added
    """
    jae_tasks = []
    
    for task in tasks:
        task_obj = task['task_object']
        ad = task['active_duration']
        
        # Calculate JAE-Data = ED / AD
        if task_obj in ed_baseline and ad > 0:
            ed = ed_baseline[task_obj]
            jae_data = ed / ad
        else:
            jae_data = None
        
        jae_task = task.copy()
        jae_task['jae_data'] = jae_data
        jae_tasks.append(jae_task)
    
    return jae_tasks


def aggregate_jae_data(all_runs_data, ed_baseline):
    """Aggregate JAE metrics across runs with confidence intervals
    
    Returns:
        List of run dictionaries with JAE metrics added
    """
    runs_with_jae = []
    
    for run in all_runs_data:
        # Calculate JAE for each repetition
        jae_values_per_rep = []
        per_task_jae = {}  # task_object -> list of JAE values across repetitions
        
        for rep_tasks in run.get('all_repetition_tasks', []):
            jae_tasks = calculate_jae_metrics(rep_tasks, ed_baseline)
            
            # Collect per-task JAE values
            for task in jae_tasks:
                task_obj = task['task_object']
                if task['jae_data'] is not None:
                    if task_obj not in per_task_jae:
                        per_task_jae[task_obj] = []
                    per_task_jae[task_obj].append(task['jae_data'])
            
            # Calculate scenario-level JAE (mean across tasks)
            valid_jaes = [t['jae_data'] for t in jae_tasks if t['jae_data'] is not None]
            if valid_jaes:
                scenario_jae = np.mean(valid_jaes)
                jae_values_per_rep.append(scenario_jae)
        
        # Calculate mean and CI for scenario-level JAE
        if jae_values_per_rep:
            jae_mean, jae_ci, _ = calculate_confidence_interval(jae_values_per_rep)
            
            # Calculate mean and CI for each task
            task_jae_aggregated = {}
            for task_obj, jae_values in per_task_jae.items():
                mean, ci, _ = calculate_confidence_interval(jae_values)
                task_jae_aggregated[task_obj] = {
                    'mean': mean,
                    'ci': ci,
                    'values': jae_values
                }
            
            run_with_jae = run.copy()
            run_with_jae['jae_data_mean'] = jae_mean
            run_with_jae['jae_data_ci'] = jae_ci
            run_with_jae['jae_data_values'] = jae_values_per_rep
            run_with_jae['jae_per_task'] = task_jae_aggregated  # Add per-task JAE
            runs_with_jae.append(run_with_jae)
    
    return runs_with_jae


def calculate_confidence_interval(data, confidence=0.95):
    """Calculate mean and confidence interval for a dataset"""
    if len(data) == 0:
        return 0, 0, 0
    
    data = np.array(data)
    mean = np.mean(data)
    
    if len(data) == 1:
        return mean, 0, 0
    
    sem = stats.sem(data)  # Standard error of the mean
    ci = sem * stats.t.ppf((1 + confidence) / 2, len(data) - 1)  # 95% CI
    
    return mean, ci, sem


def aggregate_run_data(run_files, run_number, max_n=None):
    """Aggregate data from multiple repetitions of the same experimental run
    
    Args:
        run_files: List of file info dictionaries
        run_number: Run number
        max_n: Maximum number of repetitions to analyze (default: None = all)
    """
    repetitions = []
    all_repetition_tasks = []  # Store all repetition tasks for JAE calculation
    
    # Limit to max_n if specified
    files_to_process = run_files[:max_n] if max_n is not None else run_files
    
    print(f"\nProcessing Run #{run_number} (total: {len(run_files)} repetitions, analyzing: {len(files_to_process)})")
    print("-" * 80)
    
    for file_info in files_to_process:
        print(f"  Loading: {file_info['folder_name']}")
        run_data = extract_run_data(file_info['path'], file_info['folder_name'])
        if run_data:
            repetitions.append(run_data)
            
            # Try to load task summary CSV with JAE components
            run_folder = file_info['run_folder']
            task_summary = load_task_summary_csv(run_folder)
            if task_summary:
                all_repetition_tasks.append(task_summary)
    
    if not repetitions:
        return None
    
    print(f"  Successfully loaded {len(repetitions)} repetitions\n")
    
    # Aggregate metrics across repetitions
    total_times = [r['total_time'] for r in repetitions]
    active_times = [r['total_active_time'] for r in repetitions]
    coord_times = [r['total_coordination_time'] for r in repetitions]
    
    # Calculate means and CIs
    mean_total, ci_total, _ = calculate_confidence_interval(total_times)
    mean_active, ci_active, _ = calculate_confidence_interval(active_times)
    mean_coord, ci_coord, _ = calculate_confidence_interval(coord_times)
    
    # Aggregate task durations (average across repetitions for each task)
    # Assuming all repetitions have the same task sequence
    task_names = [t['task_object'] for t in repetitions[0]['tasks']]
    aggregated_tasks = []
    
    for task_idx, task_name in enumerate(task_names):
        task_durations = []
        for rep in repetitions:
            if task_idx < len(rep['tasks']) and rep['tasks'][task_idx]['task_object'] == task_name:
                task_durations.append(rep['tasks'][task_idx]['duration'])
        
        if task_durations:
            mean_dur, ci_dur, _ = calculate_confidence_interval(task_durations)
            aggregated_tasks.append({
                'task_object': task_name,
                'value': repetitions[0]['tasks'][task_idx]['value'],
                'duration_mean': mean_dur,
                'duration_ci': ci_dur,
                'duration_values': task_durations,
            })
    
    # Aggregate workload data if available
    workload_agg = None
    workload_reps = [r['workload'] for r in repetitions if r.get('workload') is not None]
    
    if workload_reps:
        # Aggregate workload metrics
        overall_means = [w['overall']['mean'] for w in workload_reps]
        perceptual_means = [w['perceptual']['mean'] for w in workload_reps]
        cognitive_means = [w['cognitive']['mean'] for w in workload_reps]
        motor_means = [w['motor']['mean'] for w in workload_reps]
        
        workload_agg = {
            'overall': {
                'mean': np.mean(overall_means),
                'ci': calculate_confidence_interval(overall_means)[1],
                'values': overall_means,
            },
            'perceptual': {
                'mean': np.mean(perceptual_means),
                'ci': calculate_confidence_interval(perceptual_means)[1],
                'values': perceptual_means,
            },
            'cognitive': {
                'mean': np.mean(cognitive_means),
                'ci': calculate_confidence_interval(cognitive_means)[1],
                'values': cognitive_means,
            },
            'motor': {
                'mean': np.mean(motor_means),
                'ci': calculate_confidence_interval(motor_means)[1],
                'values': motor_means,
            },
        }
    
    # Aggregate eye movement data if available
    eye_movement_agg = None
    eye_reps = [r['eye_movement'] for r in repetitions if r.get('eye_movement') is not None]
    
    if eye_reps:
        # Get all unique AoI labels across all repetitions
        all_aois = set()
        for rep in eye_reps:
            all_aois.update(rep.keys())
        
        # Aggregate metrics for each AoI
        aoi_aggregated = {}
        for aoi in all_aois:
            fixation_counts = []
            total_dwells = []
            
            for rep in eye_reps:
                if aoi in rep:
                    fixation_counts.append(rep[aoi]['fixation_count'])
                    total_dwells.append(rep[aoi]['total_dwell_time'])
                else:
                    fixation_counts.append(0)
                    total_dwells.append(0.0)
            
            mean_fix, ci_fix, _ = calculate_confidence_interval(fixation_counts)
            mean_dwell, ci_dwell, _ = calculate_confidence_interval(total_dwells)
            
            aoi_aggregated[aoi] = {
                'fixation_count_mean': mean_fix,
                'fixation_count_ci': ci_fix,
                'fixation_count_values': fixation_counts,
                'total_dwell_mean': mean_dwell,
                'total_dwell_ci': ci_dwell,
                'total_dwell_values': total_dwells,
            }
        
        eye_movement_agg = aoi_aggregated
    
    return {
        'run_number': run_number,
        'run_name': f'C{run_number}',
        'n_repetitions': len(repetitions),
        'total_time_mean': mean_total,
        'total_time_ci': ci_total,
        'total_time_values': total_times,
        'total_active_time_mean': mean_active,
        'total_active_time_ci': ci_active,
        'total_active_time_values': active_times,
        'total_coordination_time_mean': mean_coord,
        'total_coordination_time_ci': ci_coord,
        'total_coordination_time_values': coord_times,
        'tasks': aggregated_tasks,
        'workload': workload_agg,
        'eye_movement': eye_movement_agg,
        'all_repetition_tasks': all_repetition_tasks,  # For JAE calculation
    }


def create_comparison_plots(runs_data, output_dir):
    """Create comparison plots across all runs with 95% confidence intervals"""
    # 1. Overall Scenario Time Comparison with 95% CI - Three separate subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    
    run_names = [r['run_name'] for r in runs_data]
    total_times = [r['total_time_mean'] for r in runs_data]
    total_times_ci = [r['total_time_ci'] for r in runs_data]
    active_times = [r['total_active_time_mean'] for r in runs_data]
    active_times_ci = [r['total_active_time_ci'] for r in runs_data]
    coord_times = [r['total_coordination_time_mean'] for r in runs_data]
    coord_times_ci = [r['total_coordination_time_ci'] for r in runs_data]
    
    x = np.arange(len(run_names))
    
    # Use distinct colors for each metric
    color_fsm = 'steelblue'
    color_active = 'darkorange'
    color_coord = 'mediumseagreen'
    
    # Plot 1: Total FSM Time
    bars1 = ax1.bar(x, total_times, yerr=total_times_ci, capsize=5,
                    color=color_fsm, alpha=0.8, edgecolor='black')
    
    ax1.set_xlabel('Experimental Condition', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Total FSM Time', fontsize=11, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(run_names, rotation=0, ha='center')
    
    # Plot 2: Total Active Cognitive Time
    bars2 = ax2.bar(x, active_times, yerr=active_times_ci, capsize=5,
                    color=color_active, alpha=0.8, edgecolor='black')
    
    ax2.set_xlabel('Experimental Condition', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Total Active Cognitive Time', fontsize=11, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(run_names, rotation=0, ha='center')
    
    # Plot 3: Total Coordination Time
    bars3 = ax3.bar(x, coord_times, yerr=coord_times_ci, capsize=5,
                    color=color_coord, alpha=0.8, edgecolor='black')
    
    ax3.set_xlabel('Experimental Condition', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Total Coordination Time', fontsize=11, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(run_names, rotation=0, ha='center')
    
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
        # Create mappings of task_object to duration mean and CI
        task_durations_map = {task['task_object']: task['duration_mean'] for task in run['tasks']}
        task_ci_map = {task['task_object']: task['duration_ci'] for task in run['tasks']}
        
        # Get durations and CIs in the correct order (0 if task doesn't exist in this run)
        durations = [task_durations_map.get(task_obj, 0) for task_obj in task_objects]
        errors = [task_ci_map.get(task_obj, 0) for task_obj in task_objects]
        
        offset = width * (i - len(runs_data)/2 + 0.5)
        bars = ax.bar(x + offset, durations, width, yerr=errors, capsize=3,
                     label=run['run_name'], alpha=0.8, edgecolor='black')
    
    ax.set_xlabel('Task', fontsize=12, fontweight='bold')
    ax.set_ylabel('Task Duration (seconds)', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(task_objects, rotation=45, ha='right', fontsize=9)
    ax.legend(fontsize=9, loc='upper right')
    #ax.grid(True, alpha=0.3, axis='y')
    
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
                   'Coord Time (s)', 'n']]
    
    for run in runs_data:
        active_percentage = (run['total_active_time_mean'] / run['total_time_mean'] * 100) if run['total_time_mean'] > 0 else 0
        
        table_data.append([
            run['run_name'],
            f"{run['total_time_mean']:.2f}±{run['total_time_ci']:.2f}",
            f"{run['total_active_time_mean']:.2f}±{run['total_active_time_ci']:.2f}",
            f"{active_percentage:.1f}%",
            f"{run['total_coordination_time_mean']:.2f}±{run['total_coordination_time_ci']:.2f}",
            str(run['n_repetitions'])
        ])
    
    table = ax.table(cellText=table_data, cellLoc='left', loc='center',
                    colWidths=[0.20, 0.20, 0.20, 0.10, 0.20, 0.10])
    
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
    
    plt.title('Cross-Condition Summary Statistics', fontsize=14, fontweight='bold', pad=20)
    plt.savefig(Path(output_dir) / 'summary_statistics.png', dpi=300, bbox_inches='tight')
    plt.savefig(Path(output_dir) / 'summary_statistics.eps', format='eps', bbox_inches='tight')
    print(f"Saved summary statistics table")
    plt.close()
    
    # 4. Workload Comparison - Overall Utilization Box Plot
    runs_with_workload = [r for r in runs_data if r.get('workload') is not None]
    
    if len(runs_with_workload) >= 1:
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Bar chart for overall workload with confidence intervals
        workload_means = [r['workload']['overall']['mean'] for r in runs_with_workload]
        workload_cis = [r['workload']['overall']['ci'] for r in runs_with_workload]
        run_labels = [r['run_name'] for r in runs_with_workload]
        
        x = np.arange(len(run_labels))
        bars = ax.bar(x, workload_means, yerr=workload_cis, capsize=5,
                      color='lightblue', alpha=0.8, edgecolor='black')
        
        ax.set_ylabel('Overall Workload (Mean ± 95% CI)', fontsize=12, fontweight='bold')
        ax.set_xlabel('Experimental Condition', fontsize=11, fontweight='bold')
        ax.set_ylim(0, 0.2)
        ax.set_xticks(x)
        ax.set_xticklabels(run_labels, rotation=0, ha='center')
        
        plt.tight_layout()
        plt.savefig(Path(output_dir) / 'workload_overall_comparison.png', dpi=300, bbox_inches='tight')
        plt.savefig(Path(output_dir) / 'workload_overall_comparison.eps', format='eps', bbox_inches='tight')
        print(f"Saved overall workload comparison plot")
        plt.close()
    
    # 5. Eye Movement Comparison - AoI Fixation Count and Dwell Time
    runs_with_eye = [r for r in runs_data if r.get('eye_movement') is not None]
    
    if len(runs_with_eye) >= 1:
        # Get all unique AoIs across all runs
        all_aois = set()
        for run in runs_with_eye:
            all_aois.update(run['eye_movement'].keys())
        
        # Sort AoIs for consistent ordering
        aois = sorted(all_aois)
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Prepare data for plotting
        run_labels = [r['run_name'] for r in runs_with_eye]
        colors = plt.cm.Set3(np.linspace(0, 1, len(run_labels)))
        
        x = np.arange(len(aois))
        width = 0.8 / len(runs_with_eye)
        
        # Plot 1: Number of Fixations per AoI
        for i, run in enumerate(runs_with_eye):
            fixation_counts = []
            fixation_cis = []
            
            for aoi in aois:
                if aoi in run['eye_movement']:
                    fixation_counts.append(run['eye_movement'][aoi]['fixation_count_mean'])
                    fixation_cis.append(run['eye_movement'][aoi]['fixation_count_ci'])
                else:
                    fixation_counts.append(0)
                    fixation_cis.append(0)
            
            offset = width * (i - len(runs_with_eye)/2 + 0.5)
            ax1.bar(x + offset, fixation_counts, width, yerr=fixation_cis, capsize=3,
                   label=run['run_name'], alpha=0.8, edgecolor='black', color=colors[i])
        
        ax1.set_xlabel('Area of Interest', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Number of Fixations', fontsize=12, fontweight='bold')
        ax1.set_title('AoI Fixation Count Comparison (Mean ± 95% CI)', fontsize=14, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(aois, rotation=45, ha='right', fontsize=9)
        if len(runs_with_eye) > 1:
            ax1.legend(fontsize=9, loc='upper right')
        
        # Plot 2: Total Dwell Time per AoI
        for i, run in enumerate(runs_with_eye):
            dwell_times = []
            dwell_cis = []
            
            for aoi in aois:
                if aoi in run['eye_movement']:
                    dwell_times.append(run['eye_movement'][aoi]['total_dwell_mean'])
                    dwell_cis.append(run['eye_movement'][aoi]['total_dwell_ci'])
                else:
                    dwell_times.append(0)
                    dwell_cis.append(0)
            
            offset = width * (i - len(runs_with_eye)/2 + 0.5)
            ax2.bar(x + offset, dwell_times, width, yerr=dwell_cis, capsize=3,
                   label=run['run_name'], alpha=0.8, edgecolor='black', color=colors[i])
        
        ax2.set_xlabel('Area of Interest', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Total Dwell Time (seconds)', fontsize=12, fontweight='bold')
        ax2.set_title('AoI Dwell Time Comparison (Mean ± 95% CI)', fontsize=14, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(aois, rotation=45, ha='right', fontsize=9)
        if len(runs_with_eye) > 1:
            ax2.legend(fontsize=9, loc='upper right')
        
        plt.tight_layout()
        plt.savefig(Path(output_dir) / 'eye_movement_aoi_comparison.png', dpi=300, bbox_inches='tight')
        plt.savefig(Path(output_dir) / 'eye_movement_aoi_comparison.eps', format='eps', bbox_inches='tight')
        print(f"Saved eye movement AoI comparison plot")
        plt.close()
    
    # 6. JAE Per-Task Comparison
    runs_with_jae = [r for r in runs_data if r.get('jae_per_task') is not None]
    
    if len(runs_with_jae) >= 1:
        # Get all unique task objects across all runs
        all_task_objects = set()
        for run in runs_with_jae:
            all_task_objects.update(run['jae_per_task'].keys())
        
        task_objects = sorted(all_task_objects)
        
        # Create grouped bar chart
        fig, ax = plt.subplots(figsize=(16, 10))
        
        x = np.arange(len(task_objects))
        width = 0.8 / len(runs_with_jae) if len(runs_with_jae) > 1 else 0.6
        colors = plt.cm.Set3(np.linspace(0, 1, len(runs_with_jae)))
        
        for i, run in enumerate(runs_with_jae):
            jae_means = []
            jae_cis = []
            
            for task_obj in task_objects:
                if task_obj in run['jae_per_task']:
                    jae_means.append(run['jae_per_task'][task_obj]['mean'])
                    jae_cis.append(run['jae_per_task'][task_obj]['ci'])
                else:
                    jae_means.append(0)
                    jae_cis.append(0)
            
            offset = width * (i - len(runs_with_jae)/2 + 0.5)
            bars = ax.bar(x + offset, jae_means, width, yerr=jae_cis, capsize=3,
                         label=run['run_name'], alpha=0.8, edgecolor='black', color=colors[i])
        
        # Add reference line at JAE = 1.0 (optimal)
        ax.axhline(y=1.0, color='red', linestyle='--', linewidth=2, label='Optimal (JAE=1.0)', alpha=0.7)
        
        ax.set_xlabel('Task', fontsize=12, fontweight='bold')
        ax.set_ylabel('JAE-Data (Joint Activity Efficiency)', fontsize=12, fontweight='bold')
        ax.set_title('Per-Task JAE Comparison (Mean ± 95% CI)', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(task_objects, rotation=45, ha='right', fontsize=9)
        ax.legend(fontsize=9, loc='upper right')
        ax.set_ylim(0, max(1.1, max([max(run['jae_per_task'][t]['mean'] for t in run['jae_per_task']) for run in runs_with_jae]) * 1.1))
        
        plt.tight_layout()
        plt.savefig(Path(output_dir) / 'jae_per_task_comparison.png', dpi=300, bbox_inches='tight')
        plt.savefig(Path(output_dir) / 'jae_per_task_comparison.eps', format='eps', bbox_inches='tight')
        print(f"Saved per-task JAE comparison plot")
        plt.close()
    
    # 7. JAE Global (Scenario-Level) Comparison
    runs_with_jae_global = [r for r in runs_data if r.get('jae_data_mean') is not None]
    
    if len(runs_with_jae_global) >= 1:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        run_names = [r['run_name'] for r in runs_with_jae_global]
        jae_means = [r['jae_data_mean'] for r in runs_with_jae_global]
        jae_cis = [r['jae_data_ci'] for r in runs_with_jae_global]
        
        x = np.arange(len(run_names))
        
        bars = ax.bar(x, jae_means, yerr=jae_cis, capsize=5,
                     color='steelblue', alpha=0.8, edgecolor='black')
        
        # Add reference line at JAE = 1.0 (optimal)
        ax.axhline(y=1.0, color='red', linestyle='--', linewidth=2, label='Optimal (JAE=1.0)', alpha=0.7)
        
        ax.set_xlabel('Experimental Condition', fontsize=12, fontweight='bold')
        ax.set_ylabel('Joint Activity Efficiency', fontsize=12, fontweight='bold')
        #ax.set_title('Scenario-Level JAE Comparison (Mean ± 95% CI)', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(run_names, rotation=0, ha='center')
        ax.legend(fontsize=10, loc='upper right')
        ax.set_ylim(0, max(1.1, max(jae_means) * 1.15))
        
        plt.tight_layout()
        plt.savefig(Path(output_dir) / 'jae_global_comparison.png', dpi=300, bbox_inches='tight')
        plt.savefig(Path(output_dir) / 'jae_global_comparison.eps', format='eps', bbox_inches='tight')
        print(f"Saved global JAE comparison plot")
        plt.close()

def export_summary_metrics_csv(runs_data, output_dir):
    """Export summary metrics to CSV for power analysis"""
    import csv
    
    csv_path = Path(output_dir) / 'comparison_summary_metrics.csv'
    
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Header
        header = ['Run_Name', 'Run_Number', 'N_Repetitions',
                  'Total_FSM_Time_Mean', 'Total_FSM_Time_CI', 'Total_FSM_Time_Std',
                  'Active_Time_Mean', 'Active_Time_CI', 'Active_Time_Std',
                  'Coordination_Time_Mean', 'Coordination_Time_CI', 'Coordination_Time_Std',
                  'Active_Percentage', 'Coordination_Percentage']
        
        # Add workload columns if any run has workload data
        has_workload = any(r.get('workload') is not None for r in runs_data)
        if has_workload:
            header.extend(['Workload_Overall_Mean', 'Workload_Overall_CI', 'Workload_Overall_Std',
                          'Workload_Perceptual_Mean', 'Workload_Perceptual_CI',
                          'Workload_Cognitive_Mean', 'Workload_Cognitive_CI',
                          'Workload_Motor_Mean', 'Workload_Motor_CI'])
        
        # Add JAE columns if any run has JAE data
        has_jae = any(r.get('jae_data_mean') is not None for r in runs_data)
        if has_jae:
            header.extend(['JAE_Data_Mean', 'JAE_Data_CI', 'JAE_Data_Std', 'Team_Efficiency_Pct'])
        
        writer.writerow(header)
        
        # Data rows
        for run in runs_data:
            # Calculate std from values if available
            fsm_std = np.std(run['total_time_values'], ddof=1) if len(run['total_time_values']) > 1 else 0
            active_std = np.std(run['total_active_time_values'], ddof=1) if len(run['total_active_time_values']) > 1 else 0
            coord_std = np.std(run['total_coordination_time_values'], ddof=1) if len(run['total_coordination_time_values']) > 1 else 0
            
            active_pct = (run['total_active_time_mean']/run['total_time_mean']*100) if run['total_time_mean'] > 0 else 0
            coord_pct = (run['total_coordination_time_mean']/run['total_time_mean']*100) if run['total_time_mean'] > 0 else 0
            
            row = [
                run['run_name'],
                run['run_number'],
                run['n_repetitions'],
                run['total_time_mean'],
                run['total_time_ci'],
                fsm_std,
                run['total_active_time_mean'],
                run['total_active_time_ci'],
                active_std,
                run['total_coordination_time_mean'],
                run['total_coordination_time_ci'],
                coord_std,
                active_pct,
                coord_pct
            ]
            
            if has_workload:
                if run.get('workload'):
                    wl = run['workload']
                    wl_overall_std = np.std(wl['overall']['values'], ddof=1) if len(wl['overall']['values']) > 1 else 0
                    row.extend([
                        wl['overall']['mean'], wl['overall']['ci'], wl_overall_std,
                        wl['perceptual']['mean'], wl['perceptual']['ci'],
                        wl['cognitive']['mean'], wl['cognitive']['ci'],
                        wl['motor']['mean'], wl['motor']['ci']
                    ])
                else:
                    row.extend([None] * 9)
            
            if has_jae:
                if run.get('jae_data_mean') is not None:
                    jae_std = np.std(run['jae_data_values'], ddof=1) if len(run['jae_data_values']) > 1 else 0
                    efficiency_pct = run['jae_data_mean'] * 100
                    row.extend([
                        run['jae_data_mean'], run['jae_data_ci'], jae_std, efficiency_pct
                    ])
                else:
                    row.extend([None] * 4)
            
            writer.writerow(row)
    
    print(f"Exported summary metrics to: {csv_path}")

def print_summary_report(runs_data):
    """Print a text summary of the comparison with 95% CI"""
    print("\n" + "="*80)
    print("CROSS-RUN COMPARISON SUMMARY (Mean ± 95% CI)")
    print("="*80)
    
    for run in runs_data:
        print(f"\n{run['run_name']} (n={run['n_repetitions']} repetitions)")
        print("-" * 60)
        print(f"  Total FSM time: {run['total_time_mean']:.2f}s ± {run['total_time_ci']:.2f}s")
        print(f"  Total active cognitive time: {run['total_active_time_mean']:.2f}s ± {run['total_active_time_ci']:.2f}s")
        print(f"  Total coordination time: {run['total_coordination_time_mean']:.2f}s ± {run['total_coordination_time_ci']:.2f}s")
        active_pct = (run['total_active_time_mean']/run['total_time_mean']*100) if run['total_time_mean'] > 0 else 0
        coord_pct = (run['total_coordination_time_mean']/run['total_time_mean']*100) if run['total_time_mean'] > 0 else 0
        print(f"  Active percentage: {active_pct:.1f}%")
        print(f"  Coordination percentage: {coord_pct:.1f}%")
        print(f"  Number of tasks: {len(run['tasks'])}")
        
        # Workload statistics if available
        if run.get('workload'):
            wl = run['workload']
            print(f"  Overall workload (mean±CI): {wl['overall']['mean']:.3f}±{wl['overall']['ci']:.3f}")
            print(f"    Perceptual: {wl['perceptual']['mean']:.3f}±{wl['perceptual']['ci']:.3f}")
            print(f"    Cognitive:  {wl['cognitive']['mean']:.3f}±{wl['cognitive']['ci']:.3f}")
            print(f"    Motor:      {wl['motor']['mean']:.3f}±{wl['motor']['ci']:.3f}")
        
        # JAE statistics if available
        if run.get('jae_data_mean') is not None:
            print(f"  JAE-Data (mean±CI): {run['jae_data_mean']:.3f}±{run['jae_data_ci']:.3f}")
            efficiency_pct = run['jae_data_mean'] * 100
            print(f"    Team efficiency: {efficiency_pct:.1f}% of optimal")
    
    # Find fastest and slowest runs (based on mean times)
    valid_runs = [r for r in runs_data if r['total_time_mean'] > 0]
    if len(valid_runs) >= 2:
        fastest = min(valid_runs, key=lambda r: r['total_time_mean'])
        slowest = max(valid_runs, key=lambda r: r['total_time_mean'])
        
        print("\n" + "-"*80)
        print(f"Fastest run: {fastest['run_name']} ({fastest['total_time_mean']:.2f}±{fastest['total_time_ci']:.2f}s)")
        print(f"Slowest run: {slowest['run_name']} ({slowest['total_time_mean']:.2f}±{slowest['total_time_ci']:.2f}s)")
        time_diff = slowest['total_time_mean'] - fastest['total_time_mean']
        pct_diff = ((slowest['total_time_mean']/fastest['total_time_mean'] - 1) * 100) if fastest['total_time_mean'] > 0 else 0
        print(f"Time difference: {time_diff:.2f}s ({pct_diff:.1f}% slower)")
    print("="*80 + "\n")

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='Cross-Run Comparison Analyzer')
    parser.add_argument('--max-n', type=int, default=None,
                       help='Maximum number of repetitions to analyze per run (default: None = all available)')
    args = parser.parse_args()
    
    # Get the directory containing this script
    script_dir = Path(__file__).parent
    
    print("="*80)
    print("CROSS-RUN COMPARISON ANALYZER WITH 95% CONFIDENCE INTERVALS")
    print("="*80)
    print(f"Analyzing runs in: {script_dir}")
    if args.max_n is not None:
        print(f"Limited to {args.max_n} repetitions per run (--max-n)")
    print()
    
    # Find all CSV files grouped by run number
    grouped_runs = find_csv_files(script_dir)
    
    if not grouped_runs:
        print("Error: No CSV files found in output subfolders")
        sys.exit(1)
    
    print(f"Found {len(grouped_runs)} experimental runs:")
    for group in grouped_runs:
        print(f"  - Run {group['run_number']}: {len(group['files'])} repetitions")
    print()
    
    # Aggregate data for each run (across repetitions)
    runs_data = []
    for group in grouped_runs:
        aggregated = aggregate_run_data(group['files'], group['run_number'], max_n=args.max_n)
        if aggregated:
            runs_data.append(aggregated)
    
    if not runs_data:
        print("Error: No valid run data extracted")
        sys.exit(1)
    
    print("\n" + "="*80)
    print(f"Successfully processed {len(runs_data)} experimental runs")
    print("="*80)
    
    # Calculate JAE-Data baseline and metrics
    print("\nCalculating Joint Activity Efficiency (JAE-Data)...")
    ed_baseline = calculate_jae_data_baseline(runs_data)
    
    if ed_baseline:
        print(f"Established ED baseline for {len(ed_baseline)} unique task types")
        print("\nED Baseline (Minimum Active Duration per task type):")
        for task_obj, ed in sorted(ed_baseline.items()):
            print(f"  {task_obj}: {ed:.3f}s")
        
        # Calculate JAE for each run
        runs_data = aggregate_jae_data(runs_data, ed_baseline)
        print("\nJAE-Data calculated for all runs\n")
    
    # Print summary report
    print_summary_report(runs_data)
    
    # Export summary metrics to CSV for power analysis
    export_summary_metrics_csv(runs_data, script_dir)
    
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
    if len(runs_with_workload) >= 1:
        print("  - workload_overall_comparison.png/eps")
    
    # Check if eye movement plots were generated
    runs_with_eye = [r for r in runs_data if r.get('eye_movement') is not None]
    if len(runs_with_eye) >= 1:
        print("  - eye_movement_aoi_comparison.png/eps")
    
    # Check if JAE plots were generated
    runs_with_jae = [r for r in runs_data if r.get('jae_data_mean') is not None]
    if len(runs_with_jae) >= 1:
        print("  - jae_per_task_comparison.png/eps")
        print("  - jae_global_comparison.png/eps")
    
    print("="*80)

if __name__ == "__main__":
    main()
