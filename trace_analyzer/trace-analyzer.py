import re
import matplotlib.pyplot as plt

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


def plot_utilities(times, decide_crosscheck_utilities, no_crosscheck_utilities, 
                   decide_crosscheck_u_no_noise, no_crosscheck_u_no_noise):
    """
    Create plots for the utility values over time.
    """
    # Plot 1: Combined comparison of utilities with noise
    plt.figure(figsize=(12, 6))
    plt.plot(times, decide_crosscheck_utilities, 'b-o', label='x-7-decide-crosscheck', markersize=4, linewidth=1.5)
    plt.plot(times, no_crosscheck_utilities, 'r-s', label='x-7-no-crosscheck', markersize=4, linewidth=1.5)
    plt.xlabel('Time (s)', fontsize=12)
    plt.ylabel('Utility', fontsize=12)
    plt.title('Production Utilities Over Time During Conflict Resolution', fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('trace_analyzer/utility_comparison.png', dpi=300)
    print(f"Plot saved as 'utility_comparison.png'")
    
    # Plot 2: Utility vs U without noise for x-7-decide-crosscheck
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    ax1.plot(times, decide_crosscheck_utilities, 'b-o', label='Utility (with noise)', markersize=5, linewidth=1.5)
    ax1.plot(times, decide_crosscheck_u_no_noise, 'g--^', label='U without noise', markersize=5, linewidth=1.5)
    ax1.set_xlabel('Time (s)', fontsize=12)
    ax1.set_ylabel('Utility', fontsize=12)
    ax1.set_title('x-7-decide-crosscheck: Utility vs U without noise', fontsize=12, fontweight='bold')
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(times, no_crosscheck_utilities, 'r-s', label='Utility (with noise)', markersize=5, linewidth=1.5)
    ax2.plot(times, no_crosscheck_u_no_noise, 'orange', linestyle='--', marker='D', 
             label='U without noise', markersize=5, linewidth=1.5)
    ax2.set_xlabel('Time (s)', fontsize=12)
    ax2.set_ylabel('Utility', fontsize=12)
    ax2.set_title('x-7-no-crosscheck: Utility vs U without noise', fontsize=12, fontweight='bold')
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('trace_analyzer/utility_with_without_noise.png', dpi=300)
    print(f"Plot saved as 'utility_with_without_noise.png'")
    
    # Plot 3: All four values on one plot
    plt.figure(figsize=(14, 7))
    plt.plot(times, decide_crosscheck_utilities, 'b-o', label='x-7-decide-crosscheck (with noise)', 
             markersize=5, linewidth=1.5, alpha=0.8)
    plt.plot(times, decide_crosscheck_u_no_noise, 'b--^', label='x-7-decide-crosscheck (no noise)', 
             markersize=5, linewidth=1.5, alpha=0.6)
    plt.plot(times, no_crosscheck_utilities, 'r-s', label='x-7-no-crosscheck (with noise)', 
             markersize=5, linewidth=1.5, alpha=0.8)
    plt.plot(times, no_crosscheck_u_no_noise, 'r--D', label='x-7-no-crosscheck (no noise)', 
             markersize=5, linewidth=1.5, alpha=0.6)
    plt.xlabel('Time (s)', fontsize=12)
    plt.ylabel('Utility', fontsize=12)
    plt.title('All Utility Values: With and Without Noise', fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('trace_analyzer/utility_all_comparison.png', dpi=300)
    print(f"Plot saved as 'utility_all_comparison.png'")
    
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
    
    if len(times) > 0:
        print(f"Successfully extracted {len(times)} conflict resolution events.\n")
        
        # Print statistics
        print_statistics(times, decide_utilities, no_crosscheck_utilities, 
                        decide_u_no_noise, no_crosscheck_u_no_noise)
        
        # Create plots
        print("Generating plots...")
        plot_utilities(times, decide_utilities, no_crosscheck_utilities,
                      decide_u_no_noise, no_crosscheck_u_no_noise)
    else:
        print("No matching conflict resolution events found in the trace file.")
