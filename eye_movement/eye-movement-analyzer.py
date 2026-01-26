import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import gaussian_filter
from PIL import Image

# Environment dimensions
ENV_WIDTH = 2555
ENV_HEIGHT = 1374

# Areas of Interest (AoIs) - defined as (x_min, y_min, x_max, y_max)
AOI_TARS = (0, 865, 800, 1374)
AOI_PFD = (1041, 636, 1532, 963)
AOI_E_W_CAS = (1592, 681, 1674, 1038)
AOI_ND = (1719, 682, 2126, 1036)
AOI_CENTRAL_CONSOLE = (1724, 1233, 2282, 1374)
AOI_OUTSIDE_WINDOW = [
    (0, 0, 2555, 344),    # Upper window area
    (0, 0, 924, 852)       # Left window area
]

# AoI names for easy reference
AOI_NAMES = {
    'TARS': AOI_TARS,
    'PFD': AOI_PFD,
    'E_W_CAS': AOI_E_W_CAS,
    'ND': AOI_ND,
    'Central_Console': AOI_CENTRAL_CONSOLE,
    'Outside_Window': AOI_OUTSIDE_WINDOW
}

def parse_eye_movement_file(filename):
    """
    Parse the eye movement data file and extract x, y coordinates.
    """
    x_coords = []
    y_coords = []
    timestamps = []
    labels = []
    
    with open(filename, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 7:
                timestamp = float(parts[0])
                x = int(parts[2])
                y = int(parts[3])
                label = parts[5].strip('"')
                
                timestamps.append(timestamp)
                x_coords.append(x)
                y_coords.append(y)
                labels.append(label)
    
    return np.array(timestamps), np.array(x_coords), np.array(y_coords), labels


def create_heatmap(x_coords, y_coords, width=ENV_WIDTH, height=ENV_HEIGHT, 
                   bins=100, sigma=3):
    """
    Create a 2D heatmap from eye movement coordinates.
    
    Parameters:
    - x_coords: array of x coordinates
    - y_coords: array of y coordinates
    - width: environment width
    - height: environment height
    - bins: number of bins for the histogram (higher = more detail)
    - sigma: gaussian smoothing parameter (higher = more blur)
    """
    # Create 2D histogram
    heatmap, xedges, yedges = np.histogram2d(
        x_coords, y_coords,
        bins=[bins, bins],
        range=[[0, width], [0, height]]
    )
    
    # Apply gaussian filter for smoothing
    heatmap = gaussian_filter(heatmap, sigma=sigma)
    
    return heatmap.T, xedges, yedges


def plot_heatmap(heatmap, save_as='eye_movement_heatmap.png', title='Eye Movement Heatmap'):
    """
    Plot and save the heatmap visualization.
    """
    # Create custom colormap (blue to red through yellow)
    colors = ['#000033', '#0000FF', '#00FFFF', '#FFFF00', '#FF0000']
    n_bins = 256
    cmap = LinearSegmentedColormap.from_list('eye_heatmap', colors, N=n_bins)
    
    fig, ax = plt.subplots(figsize=(16, 9), facecolor='black')
    
    # Plot heatmap
    im = ax.imshow(heatmap, cmap=cmap, aspect='auto', 
                   extent=[0, ENV_WIDTH, ENV_HEIGHT, 0],
                   interpolation='bilinear', alpha=0.85)
    
    # Customize plot
    ax.set_xlim(0, ENV_WIDTH)
    ax.set_ylim(ENV_HEIGHT, 0)
    ax.set_xlabel('X Coordinate (pixels)', fontsize=12, color='white')
    ax.set_ylabel('Y Coordinate (pixels)', fontsize=12, color='white')
    ax.set_title(title, fontsize=16, fontweight='bold', color='white', pad=20)
    
    # Style the axes
    ax.tick_params(colors='white', labelsize=10)
    ax.spines['bottom'].set_color('white')
    ax.spines['top'].set_color('white')
    ax.spines['right'].set_color('white')
    ax.spines['left'].set_color('white')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Fixation Density', rotation=270, labelpad=20, 
                   fontsize=12, color='white')
    cbar.ax.tick_params(colors='white', labelsize=10)
    
    # Make colorbar outline white
    cbar.outline.set_edgecolor('white')
    
    plt.tight_layout()
    plt.savefig(save_as, dpi=300, facecolor='black')
    eps_path = save_as.replace('.png', '.eps')
    plt.savefig(eps_path, format='eps', facecolor='black')
    print(f"Heatmap saved as '{save_as}' and '{eps_path}'")
    
    plt.show()


def plot_scatter_overlay(x_coords, y_coords, timestamps, 
                         save_as='eye_movement_scatter.png'):
    """
    Create a scatter plot showing the sequence of eye movements.
    """
    fig, ax = plt.subplots(figsize=(16, 9), facecolor='white')
    
    # Create scatter plot with color gradient based on time
    scatter = ax.scatter(x_coords, y_coords, c=timestamps, 
                        cmap='viridis', s=20, alpha=0.6, edgecolors='none')
    
    # Customize plot
    ax.set_xlim(0, ENV_WIDTH)
    ax.set_ylim(ENV_HEIGHT, 0)
    ax.set_xlabel('X Coordinate (pixels)', fontsize=12)
    ax.set_ylabel('Y Coordinate (pixels)', fontsize=12)
    ax.set_title('Eye Movement Trajectory (colored by time)', 
                 fontsize=16, fontweight='bold', pad=20)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Time (s)', rotation=270, labelpad=20, fontsize=12)
    
    # Add grid
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(save_as, dpi=300, facecolor='white')
    eps_path = save_as.replace('.png', '.eps')
    plt.savefig(eps_path, format='eps', facecolor='white')
    print(f"Scatter plot saved as '{save_as}' and '{eps_path}'")
    
    plt.show()


def plot_combined_view(heatmap, x_coords, y_coords, 
                       save_as='eye_movement_combined.png'):
    """
    Create a combined view with heatmap and scatter overlay.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 9))
    
    # Left: Heatmap
    colors = ['#000033', '#0000FF', '#00FFFF', '#FFFF00', '#FF0000']
    cmap = LinearSegmentedColormap.from_list('eye_heatmap', colors, N=256)
    
    im1 = ax1.imshow(heatmap, cmap=cmap, aspect='auto',
                     extent=[0, ENV_WIDTH, ENV_HEIGHT, 0],
                     interpolation='bilinear', alpha=0.9)
    ax1.set_xlim(0, ENV_WIDTH)
    ax1.set_ylim(ENV_HEIGHT, 0)
    ax1.set_xlabel('X Coordinate (pixels)', fontsize=12)
    ax1.set_ylabel('Y Coordinate (pixels)', fontsize=12)
    ax1.set_title('Fixation Density Heatmap', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.2, linestyle='--', color='white')
    
    cbar1 = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    cbar1.set_label('Density', rotation=270, labelpad=20, fontsize=11)
    
    # Right: Scatter plot
    scatter = ax2.scatter(x_coords, y_coords, c='red', s=10, alpha=0.4, edgecolors='none')
    ax2.set_xlim(0, ENV_WIDTH)
    ax2.set_ylim(ENV_HEIGHT, 0)
    ax2.set_xlabel('X Coordinate (pixels)', fontsize=12)
    ax2.set_ylabel('Y Coordinate (pixels)', fontsize=12)
    ax2.set_title('All Fixation Points', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--')
    
    plt.suptitle(f'Eye Movement Analysis ({len(x_coords)} fixations)', 
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(save_as, dpi=300)
    eps_path = save_as.replace('.png', '.eps')
    plt.savefig(eps_path, format='eps')
    print(f"Combined plot saved as '{save_as}' and '{eps_path}'")
    plt.close()


def plot_heatmap_on_cockpit(heatmap, background_image_path='eye_movement/cessna_mustang_cockpit_picture.png',
                            save_as='eye_movement_cockpit_overlay.png'):
    """
    Overlay the heatmap on the cockpit picture.
    """
    try:
        # Load the background image
        bg_image = Image.open(background_image_path)
        bg_array = np.array(bg_image)
        
        # Get the actual dimensions of the background image
        bg_height, bg_width = bg_array.shape[:2]
        
        print(f"Background image size: {bg_width}x{bg_height}")
        
        # Create figure
        fig, ax = plt.subplots(figsize=(16, 9))
        
        # Display the background cockpit image
        ax.imshow(bg_array, extent=[0, ENV_WIDTH, ENV_HEIGHT, 0], aspect='auto')
        
        # Create semi-transparent heatmap overlay
        # Use a colormap that's transparent at low values
        colors = [(0, 0, 0, 0), (0, 0, 1, 0.3), (0, 1, 1, 0.5), (1, 1, 0, 0.7), (1, 0, 0, 0.9)]
        n_bins = 256
        cmap = LinearSegmentedColormap.from_list('eye_heatmap_transparent', colors, N=n_bins)
        
        # Overlay the heatmap
        im = ax.imshow(heatmap, cmap=cmap, aspect='auto',
                      extent=[0, ENV_WIDTH, ENV_HEIGHT, 0],
                      interpolation='bilinear', alpha=0.7)
        
        # Customize plot
        ax.set_xlim(0, ENV_WIDTH)
        ax.set_ylim(ENV_HEIGHT, 0)
        ax.set_xlabel('X Coordinate (pixels)', fontsize=12)
        ax.set_ylabel('Y Coordinate (pixels)', fontsize=12)
        ax.set_title('Eye Movement Heatmap on Cockpit View', 
                     fontsize=16, fontweight='bold', pad=20)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Fixation Density', rotation=270, labelpad=20, fontsize=12)
        
        plt.tight_layout()
        plt.savefig(save_as, dpi=300, bbox_inches='tight')
        eps_path = save_as.replace('.png', '.eps')
        plt.savefig(eps_path, format='eps', bbox_inches='tight')
        print(f"Cockpit overlay saved as '{save_as}' and '{eps_path}'")
        plt.close()
        return True
        
    except FileNotFoundError:
        print(f"Warning: Background image '{background_image_path}' not found.")
        print("Skipping cockpit overlay visualization.")
        return False
    except Exception as e:
        print(f"Error creating cockpit overlay: {e}")
        return False


def calculate_dwell_times(timestamps, x_coords, y_coords, threshold=50):
    """
    Calculate dwell time at each fixation point.
    Groups nearby fixations (within threshold pixels) as same location.
    """
    dwell_times = []
    
    for i in range(len(timestamps)):
        if i < len(timestamps) - 1:
            # Time until next fixation
            dwell = timestamps[i + 1] - timestamps[i]
        else:
            # For last fixation, estimate based on average
            if len(timestamps) > 1:
                avg_dwell = (timestamps[-1] - timestamps[0]) / (len(timestamps) - 1)
                dwell = avg_dwell
            else:
                dwell = 0.2  # Default 200ms
        
        dwell_times.append(dwell)
    
    return np.array(dwell_times)


def point_in_rectangle(x, y, rect):
    """
    Check if a point (x, y) is within a rectangle.
    
    Parameters:
    - x, y: point coordinates
    - rect: tuple (x_min, y_min, x_max, y_max)
    
    Returns:
    - True if point is within rectangle, False otherwise
    """
    x_min, y_min, x_max, y_max = rect
    return x_min <= x <= x_max and y_min <= y <= y_max


def classify_fixation_to_aoi(x, y):
    """
    Classify a fixation point to an AoI.
    
    Parameters:
    - x, y: fixation coordinates
    
    Returns:
    - AoI name or 'Other' if not in any defined AoI
    """
    # Check Outside Window first (it's an OR of two rectangles)
    if isinstance(AOI_OUTSIDE_WINDOW, list):
        for rect in AOI_OUTSIDE_WINDOW:
            if point_in_rectangle(x, y, rect):
                return 'Outside_Window'
    
    # Check other AoIs
    for aoi_name, aoi_rect in AOI_NAMES.items():
        if aoi_name == 'Outside_Window':
            continue  # Already checked
        if point_in_rectangle(x, y, aoi_rect):
            return aoi_name
    
    return 'Other'


def calculate_aoi_metrics(x_coords, y_coords, dwell_times):
    """
    Calculate gaze frequency and average dwell time for each AoI.
    
    Parameters:
    - x_coords: array of x coordinates
    - y_coords: array of y coordinates
    - dwell_times: array of dwell times for each fixation
    
    Returns:
    - Dictionary with AoI metrics
    """
    aoi_data = {aoi: {'count': 0, 'total_dwell': 0.0, 'dwell_times': []} 
                for aoi in list(AOI_NAMES.keys()) + ['Other']}
    
    # Classify each fixation and accumulate metrics
    for i, (x, y) in enumerate(zip(x_coords, y_coords)):
        aoi = classify_fixation_to_aoi(x, y)
        aoi_data[aoi]['count'] += 1
        aoi_data[aoi]['total_dwell'] += dwell_times[i]
        aoi_data[aoi]['dwell_times'].append(dwell_times[i])
    
    # Calculate averages
    aoi_metrics = {}
    for aoi, data in aoi_data.items():
        count = data['count']
        avg_dwell = data['total_dwell'] / count if count > 0 else 0
        aoi_metrics[aoi] = {
            'frequency': count,
            'avg_dwell_time': avg_dwell,
            'total_dwell_time': data['total_dwell'],
            'percentage': (count / len(x_coords)) * 100 if len(x_coords) > 0 else 0
        }
    
    return aoi_metrics


def plot_aoi_metrics(aoi_metrics, save_as='eye_movement/aoi_analysis.png'):
    """
    Create visualizations for AoI metrics (gaze frequency and average dwell time).
    
    Parameters:
    - aoi_metrics: dictionary with AoI metrics from calculate_aoi_metrics()
    - save_as: filename to save the plot
    """
    # Filter out AoIs with zero fixations
    aois = [aoi for aoi, metrics in aoi_metrics.items() if metrics['frequency'] > 0]
    frequencies = [aoi_metrics[aoi]['frequency'] for aoi in aois]
    avg_dwell_times = [aoi_metrics[aoi]['avg_dwell_time'] for aoi in aois]
    percentages = [aoi_metrics[aoi]['percentage'] for aoi in aois]
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Define colors for each AoI
    colors = plt.cm.Set3(np.linspace(0, 1, len(aois)))
    
    # Plot 1: Gaze Frequency
    bars1 = ax1.bar(range(len(aois)), frequencies, color=colors, edgecolor='black', linewidth=1.5)
    ax1.set_xlabel('Area of Interest', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Number of Fixations', fontsize=12, fontweight='bold')
    ax1.set_title('Gaze Frequency by AoI', fontsize=14, fontweight='bold', pad=20)
    ax1.set_xticks(range(len(aois)))
    ax1.set_xticklabels(aois, rotation=45, ha='right')
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels on bars
    for i, (bar, freq, pct) in enumerate(zip(bars1, frequencies, percentages)):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{freq}\n({pct:.1f}%)',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Plot 2: Average Dwell Time
    bars2 = ax2.bar(range(len(aois)), avg_dwell_times, color=colors, edgecolor='black', linewidth=1.5)
    ax2.set_xlabel('Area of Interest', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Average Dwell Time (seconds)', fontsize=12, fontweight='bold')
    ax2.set_title('Average Dwell Time by AoI', fontsize=14, fontweight='bold', pad=20)
    ax2.set_xticks(range(len(aois)))
    ax2.set_xticklabels(aois, rotation=45, ha='right')
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels on bars
    for bar, dwell in zip(bars2, avg_dwell_times):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{dwell:.3f}s',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.suptitle(f'Area of Interest (AoI) Analysis - {sum(frequencies)} Total Fixations',
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(save_as, dpi=300, bbox_inches='tight')
    eps_path = save_as.replace('.png', '.eps')
    plt.savefig(eps_path, format='eps', bbox_inches='tight')
    print(f"AoI analysis plot saved as '{save_as}' and '{eps_path}'")
    plt.close()


def print_aoi_statistics(aoi_metrics):
    """
    Print detailed statistics for each AoI.
    
    Parameters:
    - aoi_metrics: dictionary with AoI metrics from calculate_aoi_metrics()
    """
    print(f"\n{'='*80}")
    print(f"AREA OF INTEREST (AoI) ANALYSIS")
    print(f"{'='*80}")
    
    # Sort AoIs by frequency (descending)
    sorted_aois = sorted(aoi_metrics.items(), key=lambda x: x[1]['frequency'], reverse=True)
    
    print(f"{'AoI Name':<20} {'Fixations':<12} {'Percentage':<12} {'Avg Dwell (s)':<15} {'Total Dwell (s)':<15}")
    print(f"{'-'*80}")
    
    for aoi, metrics in sorted_aois:
        if metrics['frequency'] > 0:  # Only show AoIs with fixations
            print(f"{aoi:<20} {metrics['frequency']:<12} {metrics['percentage']:>6.2f}%     "
                  f"{metrics['avg_dwell_time']:>8.3f}        {metrics['total_dwell_time']:>8.3f}")
    
    print(f"{'='*80}\n")


def plot_scanpath_on_cockpit(x_coords, y_coords, timestamps, 
                             background_image_path='eye_movement/cessna_mustang_cockpit_picture.png',
                             save_as='eye_movement_scanpath.png'):
    """
    Create a scan path visualization with arrows showing gaze transitions
    and circles showing fixation points (size proportional to dwell time).
    """
    try:
        # Load the background image
        bg_image = Image.open(background_image_path)
        bg_array = np.array(bg_image)
        
        print(f"Background image size: {bg_array.shape[1]}x{bg_array.shape[0]}")
        
        # Calculate dwell times
        # Calculate and visualize AoI metrics
        print("\nCalculating Area of Interest (AoI) metrics...")
        dwell_times = calculate_dwell_times(timestamps, x_coords, y_coords)
        aoi_metrics = calculate_aoi_metrics(x_coords, y_coords, dwell_times)
        
        # Print AoI statistics
        print_aoi_statistics(aoi_metrics)
        
        # Plot AoI metrics
        print("Creating AoI analysis visualization...")
        plot_aoi_metrics(aoi_metrics, save_as='eye_movement/aoi_analysis.png')
        
        dwell_times = calculate_dwell_times(timestamps, x_coords, y_coords)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(16, 9))
        
        # Display the background cockpit image
        ax.imshow(bg_array, extent=[0, ENV_WIDTH, ENV_HEIGHT, 0], aspect='auto', alpha=0.9)
        
        # Normalize timestamps for color gradient
        time_norm = (timestamps - timestamps.min()) / (timestamps.max() - timestamps.min())
        
        # Create colormap for arrows (gradient over time)
        arrow_cmap = plt.cm.plasma
        
        # Draw arrows between consecutive fixations
        for i in range(len(x_coords) - 1):
            x_start, y_start = x_coords[i], y_coords[i]
            x_end, y_end = x_coords[i + 1], y_coords[i + 1]
            
            # Get color based on time
            color = arrow_cmap(time_norm[i])
            
            # Draw arrow
            ax.annotate('', xy=(x_end, y_end), xytext=(x_start, y_start),
                       arrowprops=dict(arrowstyle='->', lw=2, color=color, alpha=0.6,
                                     connectionstyle="arc3,rad=0.1"))
        
        # Normalize dwell times for circle sizes
        # Use square root to make differences more visible
        min_dwell = np.min(dwell_times)
        max_dwell = np.max(dwell_times)
        if max_dwell > min_dwell:
            size_norm = np.sqrt((dwell_times - min_dwell) / (max_dwell - min_dwell))
        else:
            size_norm = np.ones_like(dwell_times)
        
        # Scale circle sizes (min 50, max 500)
        circle_sizes = 50 + size_norm * 450
        
        # Draw circles at fixation points
        scatter = ax.scatter(x_coords, y_coords, s=circle_sizes, 
                           c=timestamps, cmap='plasma',
                           alpha=0.6, edgecolors='white', linewidths=2,
                           zorder=100)
        
        # Customize plot
        ax.set_xlim(0, ENV_WIDTH)
        ax.set_ylim(ENV_HEIGHT, 0)
        ax.set_xlabel('X Coordinate (pixels)', fontsize=12)
        ax.set_ylabel('Y Coordinate (pixels)', fontsize=12)
        ax.set_title('Eye Movement Scan Path on Cockpit View\n(Circle size = dwell time, Color = time progression)', 
                     fontsize=16, fontweight='bold', pad=20)
        
        # Add colorbar for time
        cbar = plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Time (s)', rotation=270, labelpad=20, fontsize=12)
        
        # Add legend for circle sizes
        # Create dummy scatter plots for legend
        legend_sizes = [50, 200, 500]
        legend_labels = [f'{min_dwell:.2f}s', 
                        f'{(min_dwell + max_dwell)/2:.2f}s', 
                        f'{max_dwell:.2f}s']
        legend_elements = [plt.scatter([], [], s=size, c='gray', alpha=0.6, 
                                      edgecolors='white', linewidths=2,
                                      label=label) 
                          for size, label in zip(legend_sizes, legend_labels)]
        
        legend = ax.legend(handles=legend_elements, 
                          title='Dwell Time', 
                          loc='upper right',
                          framealpha=0.9,
                          fontsize=10)
        legend.get_title().set_fontsize(11)
        legend.get_title().set_fontweight('bold')
        
        plt.tight_layout()
        plt.savefig(save_as, dpi=300, bbox_inches='tight')
        eps_path = save_as.replace('.png', '.eps')
        plt.savefig(eps_path, format='eps', bbox_inches='tight')
        print(f"Scan path visualization saved as '{save_as}' and '{eps_path}'")
        print(f"  Dwell time range: {min_dwell:.3f}s to {max_dwell:.3f}s")
        print(f"  Average dwell time: {np.mean(dwell_times):.3f}s")
        plt.close()
        return True
        
    except FileNotFoundError:
        print(f"Warning: Background image '{background_image_path}' not found.")
        print("Skipping scan path visualization.")
        return False
    except Exception as e:
        print(f"Error creating scan path visualization: {e}")
        return False


def print_statistics(x_coords, y_coords, timestamps, labels):
    """
    Print statistics about the eye movement data.
    """
    print(f"\n{'='*70}")
    print(f"EYE MOVEMENT DATA STATISTICS")
    print(f"{'='*70}")
    print(f"Total fixations: {len(x_coords)}")
    print(f"Duration: {timestamps[0]:.3f}s to {timestamps[-1]:.3f}s ({timestamps[-1] - timestamps[0]:.3f}s)")
    print(f"\nX Coordinates:")
    print(f"  Min:  {np.min(x_coords)} px")
    print(f"  Max:  {np.max(x_coords)} px")
    print(f"  Mean: {np.mean(x_coords):.1f} px")
    print(f"  Std:  {np.std(x_coords):.1f} px")
    print(f"\nY Coordinates:")
    print(f"  Min:  {np.min(y_coords)} px")
    print(f"  Max:  {np.max(y_coords)} px")
    print(f"  Mean: {np.mean(y_coords):.1f} px")
    print(f"  Std:  {np.std(y_coords):.1f} px")
    
    # Count unique locations
    unique_labels = set(labels)
    print(f"\nUnique fixation targets: {len(unique_labels)}")
    
    # Top 10 most frequent fixation targets
    from collections import Counter
    label_counts = Counter(labels)
    print(f"\nTop 10 most fixated targets:")
    for label, count in label_counts.most_common(10):
        percentage = (count / len(labels)) * 100
        print(f"  {label:30s}: {count:4d} ({percentage:5.1f}%)")
    
    print(f"{'='*70}\n")


if __name__ == "__main__":
    # Parse the eye movement data
    print("Parsing eye_movement/results_eye_movement.txt...")
    timestamps, x_coords, y_coords, labels = parse_eye_movement_file('eye_movement/results_eye_movement.txt')
    
    if len(x_coords) > 0:
        print(f"Successfully loaded {len(x_coords)} eye fixation points.\n")
        
        # Print statistics
        print_statistics(x_coords, y_coords, timestamps, labels)
        
        # Create heatmap
        print("Generating heatmap...")
        heatmap, xedges, yedges = create_heatmap(x_coords, y_coords, bins=100, sigma=5)
        
        # Generate visualizations
        print("Creating visualizations...")
        plot_heatmap(heatmap, save_as='eye_movement/eye_movement_heatmap.png')
        plot_scatter_overlay(x_coords, y_coords, timestamps, 
                           save_as='eye_movement/eye_movement_scatter.png')
        plot_combined_view(heatmap, x_coords, y_coords, 
                         save_as='eye_movement/eye_movement_combined.png')
        
        # Create cockpit overlay
        print("\nCreating cockpit overlay...")
        plot_heatmap_on_cockpit(heatmap, 
                               background_image_path='eye_movement/cessna_mustang_cockpit_picture.png',
                               save_as='eye_movement/eye_movement_cockpit_overlay.png')
        
        # Create scan path visualization
        print("\nCreating scan path visualization...")
        plot_scanpath_on_cockpit(x_coords, y_coords, timestamps,
                                background_image_path='eye_movement/cessna_mustang_cockpit_picture.png',
                                save_as='eye_movement/eye_movement_scanpath.png')
        
        print("\nAll visualizations completed successfully!")
    else:
        print("No eye movement data found in the file.")
