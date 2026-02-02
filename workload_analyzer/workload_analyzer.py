import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import json
import sys

def load_workload_data(filename):
    """
    Load the mental workload data from file.
    """
    df = pd.read_csv(filename, sep='\t')
    
    # Clean up column name (remove the prefix from first column)
    df.columns = [col.replace('ClockTime(s)_UtilizationValuesFrom:', 'Time').strip() for col in df.columns]
    
    return df


def load_task_events(json_filepath):
    """
    Load task events from JSON file.
    Returns list of task dictionaries with timestamp, task_object, and value.
    """
    try:
        with open(json_filepath, 'r') as f:
            tasks = json.load(f)
        return tasks
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Warning: Could not load task events: {e}")
        return []


def add_task_regions(ax, tasks, min_time, max_time):
    """
    Add task regions as shaded background areas to a plot.
    """
    if not tasks:
        return
    
    # Filter tasks: skip first task (Takeoff clearance) AND only include tasks that overlap with plot range
    filtered_tasks = []
    for idx, task in enumerate(tasks):
        task_time = task['timestamp']
        # Get next task time for end boundary
        if idx + 1 < len(tasks):
            next_time = tasks[idx + 1]['timestamp']
        else:
            next_time = max_time + 10  # Extend beyond plot
        
        # Skip first task (Takeoff clearance) and tasks completely outside plot range
        if idx == 0:  # Skip Takeoff clearance
            continue
        # Include task if it overlaps with plot time range
        if task_time <= max_time and next_time >= min_time:
            filtered_tasks.append(task)
    
    # Use alternating colors for task regions
    task_colors = ['#E8F4F8', '#FFF4E6']  # Light blue and light orange
    
    for i, task in enumerate(filtered_tasks):
        task_time = task['timestamp']
        
        # Find the corresponding index in the original tasks list
        orig_idx = tasks.index(task)
        
        # Determine the end time (next task start time from original list)
        if orig_idx + 1 < len(tasks):
            end_time = tasks[orig_idx + 1]['timestamp']
        else:
            end_time = max_time + 10  # Extend beyond plot for last task
        
        # Clip to plot boundaries
        start_time = max(task_time, min_time)
        end_time = min(end_time, max_time)
        
        # Draw shaded region for this task
        if start_time < end_time:
            color = task_colors[i % 2]
            ax.axvspan(start_time, end_time, alpha=0.5, color=color, zorder=1)
            
            # Add task label
            mid_time = (start_time + end_time) / 2
            task_label = f"{task['task_object']}"
            
            # Truncate long labels
            if len(task_label) > 25:
                task_label = task_label[:22] + "..."
            
            # Position text below the x-axis
            ylim = ax.get_ylim()
            y_offset = ylim[0] - (ylim[1] - ylim[0]) * 0.15
            
            ax.text(mid_time, y_offset, task_label,
                   rotation=45, ha='right', va='top',
                   fontsize=8, alpha=0.9, zorder=10)


def plot_perceptual_modules(df, save_as='workload_analyzer/workload_perceptual.png', tasks=None):
    """
    Plot perceptual modules: Vision + Audio
    """
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Add task regions first (as background)
    if tasks:
        add_task_regions(ax, tasks, df['Time'].min(), df['Time'].max())
    
    ax.plot(df['Time'], df['Vision_Module'], 'b-o', label='Vision Module', 
            linewidth=2, markersize=4, alpha=0.7, zorder=3)
    ax.plot(df['Time'], df['Audio_Module'], 'r-s', label='Audio Module', 
            linewidth=2, markersize=4, alpha=0.7, zorder=3)
    ax.plot(df['Time'], df['Perceptual_SubNetwork'], 'g--^', label='Perceptual SubNetwork', 
            linewidth=2.5, markersize=4, alpha=0.9, zorder=3)
    
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Utilization', fontsize=12)
    ax.set_title('Perceptual Modules Workload', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--', zorder=2)
    ax.set_ylim(-0.05, 1.1)
    
    # Extend y-axis to make room for task labels if present
    if tasks:
        ylim = ax.get_ylim()
        ax.set_ylim(ylim[0] - (ylim[1] - ylim[0]) * 0.15, ylim[1])
    
    plt.tight_layout()
    plt.savefig(save_as, dpi=300, bbox_inches='tight')
    eps_path = save_as.replace('.png', '.eps')
    plt.savefig(eps_path, format='eps', bbox_inches='tight')
    print(f"Perceptual modules plot saved as '{save_as}' and '{eps_path}'")
    plt.close()


def plot_cognitive_modules(df, save_as='workload_analyzer/workload_cognitive.png', tasks=None):
    """
    Plot cognitive modules: Production + Declarative + Imaginary
    """
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Add task regions first (as background)
    if tasks:
        add_task_regions(ax, tasks, df['Time'].min(), df['Time'].max())
    
    ax.plot(df['Time'], df['Production_Module'], 'b-o', label='Production Module', 
            linewidth=2, markersize=4, alpha=0.7, zorder=3)
    ax.plot(df['Time'], df['Declarative_Module'], 'r-s', label='Declarative Module', 
            linewidth=2, markersize=4, alpha=0.7, zorder=3)
    ax.plot(df['Time'], df['Imaginary_Module'], 'm-D', label='Imaginary Module', 
            linewidth=2, markersize=4, alpha=0.7, zorder=3)
    ax.plot(df['Time'], df['Cognitive_SubNetwork'], 'g--^', label='Cognitive SubNetwork', 
            linewidth=2.5, markersize=4, alpha=0.9, zorder=3)
    
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Utilization', fontsize=12)
    ax.set_title('Cognitive Modules Workload', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--', zorder=2)
    ax.set_ylim(-0.05, 1.1)
    
    # Extend y-axis to make room for task labels if present
    if tasks:
        ylim = ax.get_ylim()
        ax.set_ylim(ylim[0] - (ylim[1] - ylim[0]) * 0.15, ylim[1])
    
    plt.tight_layout()
    plt.savefig(save_as, dpi=300, bbox_inches='tight')
    eps_path = save_as.replace('.png', '.eps')
    plt.savefig(eps_path, format='eps', bbox_inches='tight')
    print(f"Cognitive modules plot saved as '{save_as}' and '{eps_path}'")
    plt.close()


def plot_motor_modules(df, save_as='workload_analyzer/workload_motor.png', tasks=None):
    """
    Plot motor modules: Motor + Speech
    """
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Add task regions first (as background)
    if tasks:
        add_task_regions(ax, tasks, df['Time'].min(), df['Time'].max())
    
    ax.plot(df['Time'], df['Motor_Module'], 'b-o', label='Motor Module', 
            linewidth=2, markersize=4, alpha=0.7, zorder=3)
    ax.plot(df['Time'], df['Speech_Module'], 'r-s', label='Speech Module', 
            linewidth=2, markersize=4, alpha=0.7, zorder=3)
    ax.plot(df['Time'], df['Motor_SubNetwork'], 'g--^', label='Motor SubNetwork', 
            linewidth=2.5, markersize=4, alpha=0.9, zorder=3)
    
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Utilization', fontsize=12)
    ax.set_title('Motor Modules Workload', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--', zorder=2)
    ax.set_ylim(-0.05, 1.1)
    
    # Extend y-axis to make room for task labels if present
    if tasks:
        ylim = ax.get_ylim()
        ax.set_ylim(ylim[0] - (ylim[1] - ylim[0]) * 0.15, ylim[1])
    
    plt.tight_layout()
    plt.savefig(save_as, dpi=300, bbox_inches='tight')
    eps_path = save_as.replace('.png', '.eps')
    plt.savefig(eps_path, format='eps', bbox_inches='tight')
    print(f"Motor modules plot saved as '{save_as}' and '{eps_path}'")
    plt.close()


def plot_overall_utilization(df, save_as='workload_analyzer/workload_overall.png', tasks=None):
    """
    Plot overall utilization
    """
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Add task regions first (as background)
    if tasks:
        add_task_regions(ax, tasks, df['Time'].min(), df['Time'].max())
    
    ax.plot(df['Time'], df['Overall_Utilization'], 'purple', linewidth=2.5, 
            marker='o', markersize=5, alpha=0.8, label='Overall Utilization', zorder=3)
    
    # Add average line
    avg_util = df['Overall_Utilization'].mean()
    ax.axhline(y=avg_util, color='red', linestyle='--', linewidth=2, 
               label=f'Average: {avg_util:.3f}', alpha=0.7, zorder=3)
    
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Utilization', fontsize=12)
    ax.set_title('Overall Workload Utilization', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--', zorder=2)
    ax.set_ylim(-0.05, max(df['Overall_Utilization'].max() * 1.1, 0.3))
    
    # Extend y-axis to make room for task labels if present
    if tasks:
        ylim = ax.get_ylim()
        ax.set_ylim(ylim[0] - (ylim[1] - ylim[0]) * 0.15, ylim[1])
    
    plt.tight_layout()
    plt.savefig(save_as, dpi=300, bbox_inches='tight')
    eps_path = save_as.replace('.png', '.eps')
    plt.savefig(eps_path, format='eps', bbox_inches='tight')
    print(f"Overall utilization plot saved as '{save_as}' and '{eps_path}'")
    plt.close()


def plot_all_subnetworks(df, save_as='workload_analyzer/workload_all_subnetworks.png', tasks=None):
    """
    Plot all subnetworks together for comparison
    """
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Add task regions first (as background)
    if tasks:
        add_task_regions(ax, tasks, df['Time'].min(), df['Time'].max())
    
    ax.plot(df['Time'], df['Perceptual_SubNetwork'], 'b-o', label='Perceptual SubNetwork', 
            linewidth=2, markersize=4, alpha=0.7, zorder=3)
    ax.plot(df['Time'], df['Cognitive_SubNetwork'], 'r-s', label='Cognitive SubNetwork', 
            linewidth=2, markersize=4, alpha=0.7, zorder=3)
    ax.plot(df['Time'], df['Motor_SubNetwork'], 'g-^', label='Motor SubNetwork', 
            linewidth=2, markersize=4, alpha=0.7, zorder=3)
    ax.plot(df['Time'], df['Overall_Utilization'], 'purple', linewidth=2.5, 
            marker='D', markersize=5, alpha=0.9, label='Overall Utilization', zorder=3)
    
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Utilization', fontsize=12)
    ax.set_title('All SubNetworks and Overall Utilization', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--', zorder=2)
    ax.set_ylim(-0.05, 1.1)
    
    # Extend y-axis to make room for task labels if present
    if tasks:
        ylim = ax.get_ylim()
        ax.set_ylim(ylim[0] - (ylim[1] - ylim[0]) * 0.15, ylim[1])
    
    plt.tight_layout()
    plt.savefig(save_as, dpi=300, bbox_inches='tight')
    eps_path = save_as.replace('.png', '.eps')
    plt.savefig(eps_path, format='eps', bbox_inches='tight')
    print(f"All subnetworks plot saved as '{save_as}' and '{eps_path}'")
    plt.close()


def print_statistics(df):
    """
    Print statistics about the workload data
    """
    print(f"\n{'='*70}")
    print(f"WORKLOAD STATISTICS")
    print(f"{'='*70}")
    print(f"Duration: {df['Time'].min():.1f}s to {df['Time'].max():.1f}s")
    print(f"Total time points: {len(df)}")
    
    print(f"\nPerceptual SubNetwork:")
    print(f"  Mean: {df['Perceptual_SubNetwork'].mean():.3f}")
    print(f"  Max:  {df['Perceptual_SubNetwork'].max():.3f}")
    print(f"  Min:  {df['Perceptual_SubNetwork'].min():.3f}")
    
    print(f"\nCognitive SubNetwork:")
    print(f"  Mean: {df['Cognitive_SubNetwork'].mean():.3f}")
    print(f"  Max:  {df['Cognitive_SubNetwork'].max():.3f}")
    print(f"  Min:  {df['Cognitive_SubNetwork'].min():.3f}")
    
    print(f"\nMotor SubNetwork:")
    print(f"  Mean: {df['Motor_SubNetwork'].mean():.3f}")
    print(f"  Max:  {df['Motor_SubNetwork'].max():.3f}")
    print(f"  Min:  {df['Motor_SubNetwork'].min():.3f}")
    
    print(f"\nOverall Utilization:")
    print(f"  Mean: {df['Overall_Utilization'].mean():.3f}")
    print(f"  Max:  {df['Overall_Utilization'].max():.3f}")
    print(f"  Min:  {df['Overall_Utilization'].min():.3f}")
    
    print(f"\nIndividual Module Peaks:")
    print(f"  Vision:      {df['Vision_Module'].max():.3f}")
    print(f"  Audio:       {df['Audio_Module'].max():.3f}")
    print(f"  Production:  {df['Production_Module'].max():.3f}")
    print(f"  Declarative: {df['Declarative_Module'].max():.3f}")
    print(f"  Imaginary:   {df['Imaginary_Module'].max():.3f}")
    print(f"  Motor:       {df['Motor_Module'].max():.3f}")
    print(f"  Speech:      {df['Speech_Module'].max():.3f}")
    
    print(f"{'='*70}\n")


if __name__ == "__main__":
    # Load the workload data
    print("Loading workload data...")
    df = load_workload_data('workload_analyzer/results_mental_workload.txt')
    
    print(f"Loaded {len(df)} time points.\n")
    
    # Load task events from JSON file if available
    task_events_file = 'task_events.json'
    tasks = load_task_events(task_events_file)
    if tasks:
        print(f"Loaded {len(tasks)} task events for plot annotation\n")
    
    # Print statistics
    print_statistics(df)
    
    # Generate plots
    print("Generating plots...")
    plot_perceptual_modules(df, tasks=tasks)
    plot_cognitive_modules(df, tasks=tasks)
    plot_motor_modules(df, tasks=tasks)
    plot_overall_utilization(df, tasks=tasks)
    plot_all_subnetworks(df, tasks=tasks)
    
    print("\nAll workload visualizations completed successfully!")
