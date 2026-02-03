#!/usr/bin/env python3
"""
Team Analyzer for Flight Simulation Data
Analyzes task sequences and timing from TARS Agent state transitions
"""

import csv
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

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
                        json_data = json.loads(row[6])
                        
                        event = {
                            'uuid': row[0],
                            'timestamp': float(row[1]),
                            'agent': row[2],
                            'event_type': row[3],
                            'task_object': json_data.get('task_object', ''),
                            'value': json_data.get('value', ''),
                            'procedure': json_data.get('procedure', ''),
                            'category': json_data.get('category', ''),
                            'human_role': json_data.get('human_role', ''),
                            'autonomy_role': json_data.get('autonomy_role', ''),
                            'callout': json_data.get('callout', ''),
                        }
                        events.append(event)
                    except (json.JSONDecodeError, KeyError) as e:
                        print(f"Warning: Could not parse row: {e}")
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
                # Check if this is a Cognitive_Model production_selected event
                if len(row) > 3 and row[2] == "Cognitive_Model" and row[3] == "production_selected":
                    # Check if the production name matches our patterns
                    production_name = row[6] if len(row) > 6 else ''
                    if production_name in task_start_patterns:
                        event = {
                            'uuid': row[0],
                            'timestamp': float(row[1]),
                            'agent': row[2],
                            'event_type': row[3],
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
                # Check if this is a form-task-done event
                if len(row) > 3 and row[2] == "Cognitive_Model" and row[3] == "production_selected":
                    production_name = row[6] if len(row) > 6 else ''
                    if production_name == 'form-task-done':
                        event = {
                            'uuid': row[0],
                            'timestamp': float(row[1]),
                            'agent': row[2],
                            'event_type': row[3],
                            'production': production_name,
                        }
                        events.append(event)
    
    return events

def parse_all_cognitive_productions(filepath):
    """Parse all cognitive model production_selected events"""
    events = []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=';')
        
        for row in reader:
            if len(row) >= 7:
                # Check if this is a Cognitive_Model production_selected event
                if len(row) > 3 and row[2] == "Cognitive_Model" and row[3] == "production_selected":
                    production_name = row[6] if len(row) > 6 else ''
                    event = {
                        'uuid': row[0],
                        'timestamp': float(row[1]),
                        'agent': row[2],
                        'event_type': row[3],
                        'production': production_name,
                    }
                    events.append(event)
    
    return events

def calculate_coordination_overhead(productions, tasks, end_event):
    """Calculate coordination time for each FSM task/state"""
    task_coordination = []
    
    # Process each task
    for i, task in enumerate(tasks):
        task_start = task['timestamp']
        # Task end is either the next task start or the end event
        if i < len(tasks) - 1:
            task_end = tasks[i + 1]['timestamp']
        else:
            task_end = end_event['timestamp']
        
        # Find all productions during this task
        task_productions = [
            p for p in productions
            if task_start <= p['timestamp'] < task_end
        ]
        
        # Calculate coordination time during this task
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
        
        # Calculate totals for this task
        coord_time = sum(seq['duration'] for seq in coordination_sequences)
        coord_count = sum(seq['num_productions'] for seq in coordination_sequences)
        task_duration = task_end - task_start
        
        task_coordination.append({
            'task_object': task['task_object'],
            'value': task['value'],
            'fsm_duration': task_duration,
            'coordination_time': coord_time,
            'coordination_percentage': (coord_time / task_duration * 100) if task_duration > 0 else 0,
            'num_sequences': len(coordination_sequences),
            'num_productions': coord_count
        })
    
    return task_coordination

def extract_task_sequence(events, start_uuid, end_uuid):
    """Extract tasks between start and end markers"""
    # Find indices
    start_idx = None
    end_idx = None
    
    for i, event in enumerate(events):
        if event['uuid'] == start_uuid:
            start_idx = i
        if event['uuid'] == end_uuid:
            end_idx = i
            
    if start_idx is None:
        raise ValueError(f"Start event not found: {start_uuid}")
    if end_idx is None:
        raise ValueError(f"End event not found: {end_uuid}")
    
    # Extract tasks (from start_idx to end_idx-1, as end event is not a task)
    tasks = events[start_idx:end_idx]
    
    return tasks, events[end_idx]  # Return tasks and end marker

def calculate_task_durations(tasks, end_event):
    """Calculate duration for each task"""
    task_data = []
    
    for i, task in enumerate(tasks):
        # Duration is from this task to the next one (or to end marker if last task)
        if i < len(tasks) - 1:
            next_timestamp = tasks[i + 1]['timestamp']
        else:
            next_timestamp = end_event['timestamp']
        
        duration = next_timestamp - task['timestamp']
        
        task_info = {
            'task_name': f"{task['task_object']}",
            'value': task['value'],
            'full_name': f"{task['task_object']}: {task['value']}",
            'duration': duration,
            'start_time': task['timestamp'],
            'procedure': task['procedure'],
            'category': task['category'],
        }
        task_data.append(task_info)
    
    return task_data

def create_bar_chart(task_data, output_dir):
    """Create a bar chart of task durations"""
    task_names = [t['task_name'] for t in task_data]
    durations = [t['duration'] for t in task_data]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Create bar chart
    bars = ax.barh(task_names, durations, color='steelblue', edgecolor='black', alpha=0.7)
    
    # Customize the plot
    ax.set_xlabel('Duration (seconds)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Task', fontsize=12, fontweight='bold')
    ax.set_title('Task Time Analysis - Team level', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    
    # Add value labels on bars
    for i, (bar, duration) in enumerate(zip(bars, durations)):
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2, 
                f'{duration:.2f}s',
                ha='left', va='center', fontsize=9, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    
    plt.tight_layout()
    
    # Save the plot in PNG and EPS formats
    output_path_png = output_dir / 'task_durations.png'
    output_path_eps = output_dir / 'task_durations.eps'
    plt.savefig(output_path_png, dpi=300, bbox_inches='tight')
    plt.savefig(output_path_eps, format='eps', bbox_inches='tight')
    print(f"Bar chart saved as '{output_path_png}' and '{output_path_eps}'")
    
    return fig

def calculate_active_task_times(cognitive_starts, cognitive_ends, tars_states, start_uuid, end_uuid):
    """Calculate active time on task based on cognitive model engagement"""
    # Find the range of TARS states we're interested in
    start_idx = None
    end_idx = None
    
    for i, state in enumerate(tars_states):
        if state['uuid'] == start_uuid:
            start_idx = i
        if state['uuid'] == end_uuid:
            end_idx = i
    
    if start_idx is None or end_idx is None:
        raise ValueError("Could not find start or end TARS states")
    
    # Get relevant TARS states (include ALL tasks from start to end)
    relevant_states = tars_states[start_idx:end_idx]
    
    # Find cognitive start and end events within the time range
    start_time = tars_states[start_idx]['timestamp']
    end_time = tars_states[end_idx]['timestamp']
    
    relevant_cognitive_starts = [e for e in cognitive_starts 
                                 if start_time <= e['timestamp'] <= end_time]
    relevant_cognitive_ends = [e for e in cognitive_ends 
                               if start_time <= e['timestamp'] <= end_time]
    
    print(f"\nFound {len(relevant_cognitive_starts)} cognitive task start events")
    print(f"Found {len(relevant_cognitive_ends)} cognitive task end events")
    
    # Match cognitive starts to ends
    active_tasks = []
    
    # Handle first task specially - it starts with FSM state, not cognitive event
    first_state = relevant_states[0]
    # Find the first form-task-done event (should be for the first task)
    first_end_event = None
    if relevant_cognitive_ends:
        first_end_event = relevant_cognitive_ends[0]
    
    if first_end_event:
        duration = first_end_event['timestamp'] - first_state['timestamp']
        task_info = {
            'task_name': first_state['task_object'],
            'value': first_state['value'],
            'duration': duration,
            'start_time': first_state['timestamp'],
            'category': first_state['category'],
        }
        active_tasks.append(task_info)
    
    # Process remaining tasks with cognitive start events
    for i, start_event in enumerate(relevant_cognitive_starts):
        # Find which TARS state this cognitive event corresponds to
        # by finding the most recent TARS state before this cognitive event
        corresponding_state = None
        for state in relevant_states[1:]:  # Skip first state (already processed)
            if state['timestamp'] <= start_event['timestamp']:
                corresponding_state = state
            else:
                break
        
        if corresponding_state:
            # Find the corresponding end event (form-task-done)
            # It should be the next one after this start, skipping the first one we already used
            end_event = None
            for end_ev in relevant_cognitive_ends[1:]:  # Skip first end event
                if end_ev['timestamp'] > start_event['timestamp']:
                    end_event = end_ev
                    break
            
            # Calculate duration to the end event (or use end marker if no end found)
            if end_event:
                duration = end_event['timestamp'] - start_event['timestamp']
            else:
                # Fallback: use next start or end marker
                if i < len(relevant_cognitive_starts) - 1:
                    duration = relevant_cognitive_starts[i + 1]['timestamp'] - start_event['timestamp']
                else:
                    duration = end_time - start_event['timestamp']
            
            task_info = {
                'task_name': corresponding_state['task_object'],
                'value': corresponding_state['value'],
                'duration': duration,
                'start_time': start_event['timestamp'],
                'category': corresponding_state['category'],
            }
            active_tasks.append(task_info)
    
    return active_tasks

def create_active_time_bar_chart(task_data, output_dir):
    """Create a bar chart of active cognitive time on tasks"""
    task_names = [t['task_name'] for t in task_data]
    durations = [t['duration'] for t in task_data]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Create bar chart
    bars = ax.barh(task_names, durations, color='coral', edgecolor='black', alpha=0.7)
    
    # Customize the plot
    ax.set_xlabel('Active Duration (seconds)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Task', fontsize=12, fontweight='bold')
    ax.set_title('Active Time on Task - Cognitive Model Engagement', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    
    # Add value labels on bars
    for i, (bar, duration) in enumerate(zip(bars, durations)):
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2, 
                f'{duration:.2f}s',
                ha='left', va='center', fontsize=9, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    
    plt.tight_layout()
    
    # Save the plot in PNG and EPS formats
    output_path_png = output_dir / 'active_task_durations.png'
    output_path_eps = output_dir / 'active_task_durations.eps'
    plt.savefig(output_path_png, dpi=300, bbox_inches='tight')
    plt.savefig(output_path_eps, format='eps', bbox_inches='tight')
    print(f"Active time bar chart saved as '{output_path_png}' and '{output_path_eps}'")
    
    return fig

def print_active_task_summary(task_data, start_time, end_time):
    """Print a summary of active task times"""
    print("\n" + "="*80)
    print("ACTIVE TIME ON TASK ANALYSIS (Cognitive Model Engagement)")
    print("="*80)
    print(f"Start time: {start_time:.3f}")
    print(f"End time: {end_time:.3f}")
    print(f"Total duration: {end_time - start_time:.3f} seconds")
    print(f"Number of tasks: {len(task_data)}")
    print("\n" + "-"*80)
    print(f"{'#':<4} {'Task Object':<30} {'Value':<25} {'Active Time (s)':<15}")
    print("-"*80)
    
    for i, task in enumerate(task_data, 1):
        print(f"{i:<4} {task['task_name']:<30} {task['value']:<25} {task['duration']:<15.3f}")
    
    print("-"*80)
    print(f"{'Total Active Time':<60} {sum(t['duration'] for t in task_data):<15.3f}")
    print("="*80)
    
    # Additional statistics
    print("\nSTATISTICS:")
    durations = [t['duration'] for t in task_data]
    print(f"  Mean active task duration: {np.mean(durations):.3f}s")
    print(f"  Median active task duration: {np.median(durations):.3f}s")
    print(f"  Min active task duration: {np.min(durations):.3f}s")
    print(f"  Max active task duration: {np.max(durations):.3f}s")
    print(f"  Std deviation: {np.std(durations):.3f}s")
    print("="*80 + "\n")

def create_comparison_chart(fsm_tasks, active_tasks, output_dir):
    """Create a comparison chart between FSM time and active cognitive time"""
    # Match tasks by name
    task_names = [t['task_name'] for t in fsm_tasks]
    
    # Create dictionary for quick lookup of active tasks
    active_dict = {t['task_name']: t['duration'] for t in active_tasks}
    
    fsm_durations = []
    active_durations = []
    labels = []
    
    # Process all FSM tasks
    for i, task in enumerate(fsm_tasks, start=1):
        task_name = task['task_name']
        labels.append(task_name)
        fsm_durations.append(task['duration'])
        # Get active duration, default to 0 if not found
        active_durations.append(active_dict.get(task_name, 0))
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Set up bar positions
    y_pos = np.arange(len(labels))
    height = 0.35
    
    # Create bars
    bars1 = ax.barh(y_pos - height/2, fsm_durations, height, 
                    label='FSM Task Time', color='steelblue', edgecolor='black', alpha=0.7)
    bars2 = ax.barh(y_pos + height/2, active_durations, height,
                    label='Active Cognitive Time', color='coral', edgecolor='black', alpha=0.7)
    
    # Customize the plot
    ax.set_xlabel('Duration (seconds)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Task', fontsize=12, fontweight='bold')
    ax.set_title('Task Duration Comparison: FSM Time vs Active Cognitive Time', 
                 fontsize=14, fontweight='bold')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            width = bar.get_width()
            if width > 0:
                ax.text(width, bar.get_y() + bar.get_height()/2, 
                        f'{width:.1f}s',
                        ha='left', va='center', fontsize=8,
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.6))
    
    plt.tight_layout()
    
    # Save the plot in PNG and EPS formats
    output_path_png = output_dir / 'task_duration_comparison.png'
    output_path_eps = output_dir / 'task_duration_comparison.eps'
    plt.savefig(output_path_png, dpi=300, bbox_inches='tight')
    plt.savefig(output_path_eps, format='eps', bbox_inches='tight')
    print(f"Comparison chart saved as '{output_path_png}' and '{output_path_eps}'")
    
    return fig

def print_coordination_summary(task_coordination):
    """Print summary of coordination overhead per task"""
    print("\n" + "="*80)
    print("COORDINATION OVERHEAD PER FSM STATE")
    print("="*80)
    
    print(f"{'#':<4} {'Task Object':<30} {'FSM Time (s)':<14} {'Coord Time (s)':<16} {'Coord %':<10}")
    print("-"*80)
    
    total_fsm = 0
    total_coord = 0
    
    for i, task in enumerate(task_coordination, 1):
        print(f"{i:<4} {task['task_object']:<30} {task['fsm_duration']:<14.3f} "
              f"{task['coordination_time']:<16.3f} {task['coordination_percentage']:<10.1f}")
        total_fsm += task['fsm_duration']
        total_coord += task['coordination_time']
    
    print("-"*80)
    overall_percentage = (total_coord / total_fsm * 100) if total_fsm > 0 else 0
    print(f"{'Total':<4} {'':<30} {total_fsm:<14.3f} {total_coord:<16.3f} {overall_percentage:<10.1f}")
    print("="*80)
    
    print(f"\nSUMMARY:")
    print(f"  Total FSM time: {total_fsm:.3f}s")
    print(f"  Total coordination time: {total_coord:.3f}s")
    print(f"  Overall coordination overhead: {overall_percentage:.1f}% of FSM time")
    
    total_sequences = sum(task['num_sequences'] for task in task_coordination)
    total_productions = sum(task['num_productions'] for task in task_coordination)
    print(f"  Total coordination sequences: {total_sequences}")
    print(f"  Total team-interaction productions: {total_productions}")
    
    if total_sequences > 0:
        print(f"  Mean productions per sequence: {total_productions / total_sequences:.1f}")
    
    print("="*80 + "\n")

def create_coordination_overhead_chart(task_coordination, output_dir):
    """Create a visualization of coordination overhead per task"""
    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.5])
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    
    # Extract data
    task_names = [f"{t['task_object'][:20]}..." if len(t['task_object']) > 20 else t['task_object'] 
                  for t in task_coordination]
    fsm_times = [t['fsm_duration'] for t in task_coordination]
    coord_times = [t['coordination_time'] for t in task_coordination]
    coord_percentages = [t['coordination_percentage'] for t in task_coordination]
    
    # Calculate overall totals
    total_fsm = sum(fsm_times)
    total_coord = sum(coord_times)
    total_task = total_fsm - total_coord
    
    # Chart 1: Overall Pie Chart - Taskwork vs Coordination
    sizes = [total_task, total_coord]
    labels = [f'Taskwork\n{total_task:.2f}s\n({(total_task/total_fsm)*100:.1f}%)',
              f'Coordination\n{total_coord:.2f}s\n({(total_coord/total_fsm)*100:.1f}%)']
    colors = ['#5DADE2', '#F39C12']
    explode = (0, 0.1)
    
    ax1.pie(sizes, explode=explode, labels=labels, colors=colors,
            autopct='', shadow=False, startangle=90, textprops={'fontsize': 11, 'fontweight': 'bold'})
    ax1.set_title('Overall Time Allocation:\nTaskwork vs Coordination', 
                  fontsize=14, fontweight='bold', pad=20)
    
    # Chart 2: Stacked bar chart showing FSM time and coordination time
    x_pos = np.arange(len(task_names))
    task_only_times = [fsm - coord for fsm, coord in zip(fsm_times, coord_times)]
    
    bars1 = ax2.barh(x_pos, task_only_times, color='#5DADE2', edgecolor='black', label='Taskwork', alpha=0.8)
    bars2 = ax2.barh(x_pos, coord_times, left=task_only_times, color='#F39C12', 
                     edgecolor='black', label='Coordination', alpha=0.8)
    
    ax2.set_xlabel('Time (seconds)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Task', fontsize=12, fontweight='bold')
    ax2.set_title('FSM Task Duration: Taskwork vs Coordination', fontsize=14, fontweight='bold')
    ax2.set_yticks(x_pos)
    ax2.set_yticklabels(task_names, fontsize=9)
    ax2.legend(loc='lower right', fontsize=10)
    ax2.grid(axis='x', alpha=0.3, linestyle='--')
    
    # Add percentage labels
    for i, (bar, pct) in enumerate(zip(bars2, coord_percentages)):
        if pct > 0:
            width = bar.get_width()
            left = bar.get_x()
            ax2.text(left + width/2, bar.get_y() + bar.get_height()/2, 
                    f'{pct:.1f}%',
                    ha='center', va='center', fontsize=8, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    # Save the plot in PNG and EPS formats
    output_path_png = output_dir / 'coordination_overhead.png'
    output_path_eps = output_dir / 'coordination_overhead.eps'
    plt.savefig(output_path_png, dpi=300, bbox_inches='tight')
    plt.savefig(output_path_eps, format='eps', bbox_inches='tight')
    print(f"Coordination overhead chart saved as '{output_path_png}' and '{output_path_eps}'")
    plt.close()
    
    # Create separate chart for coordination percentage by task
    fig2, ax = plt.subplots(figsize=(12, 8))
    
    x_pos = np.arange(len(task_names))
    bars = ax.barh(x_pos, coord_percentages, color='#E74C3C', edgecolor='black', alpha=0.7)
    
    ax.set_xlabel('Coordination Overhead (%)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Task', fontsize=12, fontweight='bold')
    ax.set_title('Coordination Overhead as % of FSM Time', fontsize=14, fontweight='bold')
    ax.set_yticks(x_pos)
    ax.set_yticklabels(task_names, fontsize=9)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.set_xlim(0, max(coord_percentages) * 1.2 if coord_percentages else 100)
    
    # Add value labels
    for bar, pct, coord_time in zip(bars, coord_percentages, coord_times):
        if pct > 0:
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2, 
                    f' {pct:.1f}% ({coord_time:.2f}s)',
                    ha='left', va='center', fontsize=8,
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))
    
    plt.tight_layout()
    
    # Save the percentage plot
    output_path_png2 = output_dir / 'coordination_overhead_percentage.png'
    output_path_eps2 = output_dir / 'coordination_overhead_percentage.eps'
    plt.savefig(output_path_png2, dpi=300, bbox_inches='tight')
    plt.savefig(output_path_eps2, format='eps', bbox_inches='tight')
    print(f"Coordination overhead percentage chart saved as '{output_path_png2}' and '{output_path_eps2}'")
    plt.close()
    
    return fig, fig2

def print_task_summary(task_data, start_time, end_time):
    """Print a summary of tasks and their durations"""
    print("\n" + "="*80)
    print("TASK SEQUENCE ANALYSIS")
    print("="*80)
    print(f"Start time: {start_time:.3f}")
    print(f"End time: {end_time:.3f}")
    print(f"Total duration: {end_time - start_time:.3f} seconds")
    print(f"Number of tasks: {len(task_data)}")
    print("\n" + "-"*80)
    print(f"{'#':<4} {'Task Object':<30} {'Value':<25} {'Duration (s)':<15}")
    print("-"*80)
    
    for i, task in enumerate(task_data, 1):
        print(f"{i:<4} {task['task_name']:<30} {task['value']:<25} {task['duration']:<15.3f}")
    
    print("-"*80)
    print(f"{'Total':<60} {sum(t['duration'] for t in task_data):<15.3f}")
    print("="*80)
    
    # Additional statistics
    print("\nSTATISTICS:")
    durations = [t['duration'] for t in task_data]
    print(f"  Mean task duration: {np.mean(durations):.3f}s")
    print(f"  Median task duration: {np.median(durations):.3f}s")
    print(f"  Min task duration: {np.min(durations):.3f}s")
    print(f"  Max task duration: {np.max(durations):.3f}s")
    print(f"  Std deviation: {np.std(durations):.3f}s")
    print("="*80 + "\n")

def main():
    # Setup paths
    script_dir = Path(__file__).parent
    csv_file = script_dir / 'cassandra_converted.csv'
    
    print("="*80)
    print("TEAM ANALYZER - TARS Agent Task Sequence")
    print("="*80)
    print(f"Loading data from: {csv_file}")
    
    # Parse CSV file for TARS states
    tars_events = parse_csv_file(csv_file)
    print(f"Found {len(tars_events)} TARS Agent state transitions")
    
    # Parse CSV file for cognitive task starts
    cognitive_starts = parse_cognitive_task_starts(csv_file)
    print(f"Found {len(cognitive_starts)} cognitive task start events")
    
    # Parse CSV file for cognitive task ends
    cognitive_ends = parse_cognitive_task_ends(csv_file)
    print(f"Found {len(cognitive_ends)} cognitive task end events (form-task-done)")
    
    # Find the first non-Idle event as start boundary (skip Idle task)
    start_idx = 0
    for i, event in enumerate(tars_events):
        if event['task_object'] != 'Idle':
            start_idx = i
            break
    
    # Use the identified start and last TARS events as boundaries
    start_uuid = tars_events[start_idx]['uuid']
    end_uuid = tars_events[-1]['uuid']
    print(f"Using first event as start: {tars_events[start_idx]['task_object']}")
    print(f"Using last event as end: {tars_events[-1]['task_object']}")
    
    # Extract task sequence
    tasks, end_event = extract_task_sequence(tars_events, start_uuid, end_uuid)
    print(f"Extracted {len(tasks)} tasks between markers")
    
    # Calculate FSM-based durations
    fsm_task_data = calculate_task_durations(tasks, end_event)
    
    # Print FSM-based summary
    print_task_summary(fsm_task_data, tasks[0]['timestamp'], end_event['timestamp'])
    
    # Create FSM-based visualization
    print("Generating FSM task duration bar chart...")
    create_bar_chart(fsm_task_data, script_dir)
    
    # Calculate active cognitive time on tasks
    print("\n" + "="*80)
    print("ANALYZING ACTIVE COGNITIVE TIME ON TASKS")
    print("="*80)
    active_task_data = calculate_active_task_times(
        cognitive_starts, cognitive_ends, tars_events, start_uuid, end_uuid
    )
    
    # Print active task summary
    print_active_task_summary(active_task_data, tasks[0]['timestamp'], end_event['timestamp'])
    
    # Create active time visualization
    print("Generating active time bar chart...")
    create_active_time_bar_chart(active_task_data, script_dir)
    
    # Create comparison chart
    print("Generating comparison chart...")
    create_comparison_chart(fsm_task_data, active_task_data, script_dir)
    
    # Calculate coordination overhead per task
    print("\n" + "="*80)
    print("ANALYZING COORDINATION OVERHEAD PER FSM STATE")
    print("="*80)
    all_productions = parse_all_cognitive_productions(csv_file)
    print(f"Found {len(all_productions)} total cognitive productions\n")
    
    task_coordination = calculate_coordination_overhead(all_productions, tasks, end_event)
    
    # Print coordination summary
    print_coordination_summary(task_coordination)
    
    # Create coordination overhead visualization
    print("Generating coordination overhead chart...")
    create_coordination_overhead_chart(task_coordination, script_dir)
    
    print("\n✓ Team analysis completed successfully!")

if __name__ == '__main__':
    main()
