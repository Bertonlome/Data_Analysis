import re
import sys
import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def parse_trace_file(filename):
    """
    Parse the trace file to extract utility values for specific productions
    during conflict resolution.
    """
    times = []
    decide_crosscheck_utilities = []
    no_crosscheck_utilities = []
    decide_crosscheck_u_no_noise = []
    no_crosscheck_u_no_noise = []
    
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    i = 0
    while i < len(lines):
        line = lines[i]
        
        # Look for CONFLICT-RESOLUTION blocks
        if 'PROCEDURAL\tCONFLICT-RESOLUTION' in line:
            # Extract timestamp from the conflict resolution line
            timestamp_match = re.match(r'\s*([\d.]+)\s+PROCEDURAL', line)
            if timestamp_match:
                timestamp = float(timestamp_match.group(1))
                
                # Look ahead for the production rules
                for j in range(i+1, min(i+15, len(lines))):
                    current_line = lines[j]
                    
                    # Check if we found the matching rules line
                    if 'x-7-decide-crosscheck' in current_line and 'x-7-no-crosscheck' in current_line:
                        # Found the block, now extract utilities
                        decide_utility = None
                        no_crosscheck_utility = None
                        decide_u_without_noise = None
                        no_crosscheck_u_without_noise = None
                        
                        # Look for the utility values in the next few lines
                        for k in range(j+1, min(j+10, len(lines))):
                            util_line = lines[k]
                            
                            # Extract x-7-decide-crosscheck utility and U without noise
                            if 'x-7-decide-crosscheck:' in util_line:
                                u_no_noise_match = re.search(r'U without noise:\s*([-\d.]+)', util_line)
                                utility_match = re.search(r'utility:\s*([-\d.]+)', util_line)
                                if u_no_noise_match:
                                    decide_u_without_noise = float(u_no_noise_match.group(1))
                                if utility_match:
                                    decide_utility = float(utility_match.group(1))
                            
                            # Extract x-7-no-crosscheck utility and U without noise
                            elif 'x-7-no-crosscheck:' in util_line:
                                u_no_noise_match = re.search(r'U without noise:\s*([-\d.]+)', util_line)
                                utility_match = re.search(r'utility:\s*([-\d.]+)', util_line)
                                if u_no_noise_match:
                                    no_crosscheck_u_without_noise = float(u_no_noise_match.group(1))
                                if utility_match:
                                    no_crosscheck_utility = float(utility_match.group(1))
                            
                            # Stop if we've reached the end of the block
                            if '________Match and select rule: End________' in util_line:
                                break
                        
                        # If we found all values, add them to our lists
                        if (decide_utility is not None and no_crosscheck_utility is not None and
                            decide_u_without_noise is not None and no_crosscheck_u_without_noise is not None):
                            times.append(timestamp)
                            decide_crosscheck_utilities.append(decide_utility)
                            no_crosscheck_utilities.append(no_crosscheck_utility)
                            decide_crosscheck_u_no_noise.append(decide_u_without_noise)
                            no_crosscheck_u_no_noise.append(no_crosscheck_u_without_noise)
                        
                        break
        
        i += 1
    
    return times, decide_crosscheck_utilities, no_crosscheck_utilities, decide_crosscheck_u_no_noise, no_crosscheck_u_no_noise


def parse_reward_events(filename):
    """
    Parse the trace file to extract reward events (PROPAGATE-REWARD).
    Returns list of tuples: (timestamp, reward_value)
    """
    rewards = []
    
    with open(filename, 'r') as f:
        for line in f:
            # Look for PROPAGATE-REWARD lines
            # Format: timestamp UTILITY PROPAGATE-REWARD value
            if 'UTILITY' in line and 'PROPAGATE-REWARD' in line:
                parts = line.strip().split()
                if len(parts) >= 4:
                    try:
                        timestamp = float(parts[0])
                        reward_value = float(parts[3])
                        rewards.append((timestamp, reward_value))
                    except (ValueError, IndexError):
                        continue
    
    return rewards


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


def plot_utilities(times, decide_crosscheck_utilities, no_crosscheck_utilities, 
                   decide_crosscheck_u_no_noise, no_crosscheck_u_no_noise, rewards, tasks=None):
    """
    Create plots for the utility values over time.
    tasks: Optional list of task dictionaries with 'timestamp', 'task_object', 'value'
    """
    # Plot 1: Combined comparison of utilities with noise
    plt.figure(figsize=(12, 6))
    plt.plot(times, decide_crosscheck_utilities, 'b-o', label='x-7-decide-crosscheck', markersize=4, linewidth=1.5)
    plt.plot(times, no_crosscheck_utilities, 'r-s', label='x-7-no-crosscheck', markersize=4, linewidth=1.5)
    
    # Filter rewards to only those within the time range
    if times:
        min_time = min(times)
        max_time = max(times)
        filtered_rewards = [(t, v) for t, v in rewards if min_time <= t <= max_time]
    else:
        filtered_rewards = rewards
    
    # Add vertical lines for reward events
    has_positive = False
    has_negative = False
    for i, (reward_time, reward_val) in enumerate(filtered_rewards):
        color = 'green' if reward_val > 0 else 'red'
        # Add label only for the first occurrence of each type
        if reward_val > 0 and not has_positive:
            plt.axvline(x=reward_time, color=color, linestyle='--', alpha=0.6, linewidth=1.5, label='Positive Reward')
            has_positive = True
        elif reward_val < 0 and not has_negative:
            plt.axvline(x=reward_time, color=color, linestyle='--', alpha=0.6, linewidth=1.5, label='Negative Reward')
            has_negative = True
        else:
            plt.axvline(x=reward_time, color=color, linestyle='--', alpha=0.6, linewidth=1.5)
    
    plt.xlabel('Time (s)', fontsize=12)
    plt.ylabel('Utility', fontsize=12)
    plt.title('Production Utilities Over Time During Conflict Resolution', fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('trace_analyzer/utility_comparison.png', dpi=300)
    plt.savefig('trace_analyzer/utility_comparison.eps', format='eps')
    print(f"Plot saved as 'utility_comparison.png' and 'utility_comparison.eps'")
    
    # Plot 2: Utility vs U without noise for x-7-decide-crosscheck
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    ax1.plot(times, decide_crosscheck_utilities, 'b-o', label='Utility (with noise)', markersize=5, linewidth=1.5)
    ax1.plot(times, decide_crosscheck_u_no_noise, 'g--^', label='U without noise', markersize=5, linewidth=1.5)
    
    # Add reward event vertical lines (filtered)
    for reward_time, reward_val in filtered_rewards:
        color = 'green' if reward_val > 0 else 'red'
        ax1.axvline(x=reward_time, color=color, linestyle='--', alpha=0.5, linewidth=1.5)
    
    ax1.set_xlabel('Time (s)', fontsize=12)
    ax1.set_ylabel('Utility', fontsize=12)
    ax1.set_title('x-7-decide-crosscheck: Utility vs U without noise', fontsize=12, fontweight='bold')
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(times, no_crosscheck_utilities, 'r-s', label='Utility (with noise)', markersize=5, linewidth=1.5)
    ax2.plot(times, no_crosscheck_u_no_noise, 'orange', linestyle='--', marker='D', 
             label='U without noise', markersize=5, linewidth=1.5)
    
    # Add reward event vertical lines (filtered)
    for reward_time, reward_val in filtered_rewards:
        color = 'green' if reward_val > 0 else 'red'
        ax2.axvline(x=reward_time, color=color, linestyle='--', alpha=0.5, linewidth=1.5)
    
    ax2.set_xlabel('Time (s)', fontsize=12)
    ax2.set_ylabel('Utility', fontsize=12)
    ax2.set_title('x-7-no-crosscheck: Utility vs U without noise', fontsize=12, fontweight='bold')
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('trace_analyzer/utility_with_without_noise.png', dpi=300)
    plt.savefig('trace_analyzer/utility_with_without_noise.eps', format='eps')
    print(f"Plot saved as 'utility_with_without_noise.png' and 'utility_with_without_noise.eps'")
    
    # Plot 3: All four values on one plot with task labels
    fig, ax = plt.subplots(figsize=(16, 8))
    
    ax.plot(times, decide_crosscheck_utilities, 'b-o', label='x-7-decide-crosscheck (with noise)', 
            markersize=5, linewidth=1.5, alpha=0.8, zorder=3)
    ax.plot(times, decide_crosscheck_u_no_noise, 'b--^', label='x-7-decide-crosscheck (no noise)', 
            markersize=5, linewidth=1.5, alpha=0.6, zorder=3)
    ax.plot(times, no_crosscheck_utilities, 'r-s', label='x-7-no-crosscheck (with noise)', 
            markersize=5, linewidth=1.5, alpha=0.8, zorder=3)
    ax.plot(times, no_crosscheck_u_no_noise, 'r--D', label='x-7-no-crosscheck (no noise)', 
            markersize=5, linewidth=1.5, alpha=0.6, zorder=3)
    
    # Add vertical lines for reward events (filtered)
    has_positive = False
    has_negative = False
    for i, (reward_time, reward_val) in enumerate(filtered_rewards):
        color = 'green' if reward_val > 0 else 'red'
        # Add label only for the first occurrence of each type
        if reward_val > 0 and not has_positive:
            ax.axvline(x=reward_time, color=color, linestyle='--', alpha=0.6, linewidth=1.5, 
                      label='Positive Reward', zorder=2)
            has_positive = True
        elif reward_val < 0 and not has_negative:
            ax.axvline(x=reward_time, color=color, linestyle='--', alpha=0.6, linewidth=1.5, 
                      label='Negative Reward', zorder=2)
            has_negative = True
        else:
            ax.axvline(x=reward_time, color=color, linestyle='--', alpha=0.6, linewidth=1.5, zorder=2)
    
    # Add task regions as shaded background areas
    if tasks:
        # Filter tasks to time range of the plot
        if times:
            min_time = min(times)
            max_time = max(times)
            
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
                    ax.axvspan(start_time, end_time, alpha=0.3, color=color, zorder=1)
                    
                    # Add task label
                    mid_time = (start_time + end_time) / 2
                    task_label = f"{task['task_object']}"
                    
                    # Truncate long labels
                    if len(task_label) > 25:
                        task_label = task_label[:22] + "..."
                    
                    # Position text below the x-axis
                    y_pos = ax.get_ylim()[0] - (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.08
                    ax.text(mid_time, y_pos, task_label, 
                           ha='center', va='top', fontsize=8, 
                           rotation=45, style='italic',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                                   edgecolor='gray', alpha=0.8),
                           zorder=4)
    
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Utility', fontsize=12)
    ax.set_title('All Utility Values: With and Without Noise (with Task Timeline)', 
                fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3, zorder=0)
    
    # Add extra space at the bottom for task labels
    plt.subplots_adjust(bottom=0.2)
    
    plt.savefig('trace_analyzer/utility_all_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig('trace_analyzer/utility_all_comparison.eps', format='eps', bbox_inches='tight')
    print(f"Plot saved as 'utility_all_comparison.png' and 'utility_all_comparison.eps'")
    
    plt.show()


def print_statistics(times, decide_crosscheck_utilities, no_crosscheck_utilities,
                     decide_crosscheck_u_no_noise, no_crosscheck_u_no_noise):
    """
    Print basic statistics about the extracted data.
    """
    print(f"\n{'='*70}")
    print(f"TRACE ANALYSIS STATISTICS")
    print(f"{'='*70}")
    print(f"Total conflict resolution events found: {len(times)}")
    print(f"\nx-7-decide-crosscheck:")
    print(f"  Utility (with noise):")
    print(f"    Mean: {sum(decide_crosscheck_utilities)/len(decide_crosscheck_utilities):.4f}")
    print(f"    Min:  {min(decide_crosscheck_utilities):.4f}")
    print(f"    Max:  {max(decide_crosscheck_utilities):.4f}")
    print(f"  U without noise:")
    print(f"    Mean: {sum(decide_crosscheck_u_no_noise)/len(decide_crosscheck_u_no_noise):.4f}")
    print(f"    Min:  {min(decide_crosscheck_u_no_noise):.4f}")
    print(f"    Max:  {max(decide_crosscheck_u_no_noise):.4f}")
    print(f"\nx-7-no-crosscheck:")
    print(f"  Utility (with noise):")
    print(f"    Mean: {sum(no_crosscheck_utilities)/len(no_crosscheck_utilities):.4f}")
    print(f"    Min:  {min(no_crosscheck_utilities):.4f}")
    print(f"    Max:  {max(no_crosscheck_utilities):.4f}")
    print(f"  U without noise:")
    print(f"    Mean: {sum(no_crosscheck_u_no_noise)/len(no_crosscheck_u_no_noise):.4f}")
    print(f"    Min:  {min(no_crosscheck_u_no_noise):.4f}")
    print(f"    Max:  {max(no_crosscheck_u_no_noise):.4f}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    # Parse the trace file
    print("Parsing trace_analyzer/trace.txt...")
    times, decide_utilities, no_crosscheck_utilities, decide_u_no_noise, no_crosscheck_u_no_noise = parse_trace_file('trace_analyzer/trace.txt')
    
    # Parse reward events
    rewards = parse_reward_events('trace_analyzer/trace.txt')
    if rewards:
        print(f"Found {len(rewards)} reward events\n")
    
    # Load task events if provided as command line argument
    tasks = None
    if len(sys.argv) > 1:
        task_events_file = sys.argv[1]
        print(f"Loading task events from {task_events_file}...")
        tasks = load_task_events(task_events_file)
        if tasks:
            print(f"Loaded {len(tasks)} task events for plot annotation\n")
    
    if len(times) > 0:
        print(f"Successfully extracted {len(times)} conflict resolution events.\n")
        
        # Print statistics
        print_statistics(times, decide_utilities, no_crosscheck_utilities, 
                        decide_u_no_noise, no_crosscheck_u_no_noise)
        
        # Create plots
        print("Generating plots...")
        plot_utilities(times, decide_utilities, no_crosscheck_utilities,
                      decide_u_no_noise, no_crosscheck_u_no_noise, rewards, tasks)
    else:
        print("No matching conflict resolution events found in the trace file.")
