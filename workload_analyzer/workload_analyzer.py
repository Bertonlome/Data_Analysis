import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def load_workload_data(filename):
    """
    Load the mental workload data from file.
    """
    df = pd.read_csv(filename, sep='\t')
    
    # Clean up column name (remove the prefix from first column)
    df.columns = [col.replace('ClockTime(s)_UtilizationValuesFrom:', 'Time').strip() for col in df.columns]
    
    return df


def plot_perceptual_modules(df, save_as='workload_analyzer/workload_perceptual.png'):
    """
    Plot perceptual modules: Vision + Audio
    """
    fig, ax = plt.subplots(figsize=(14, 6))
    
    ax.plot(df['Time'], df['Vision_Module'], 'b-o', label='Vision Module', 
            linewidth=2, markersize=4, alpha=0.7)
    ax.plot(df['Time'], df['Audio_Module'], 'r-s', label='Audio Module', 
            linewidth=2, markersize=4, alpha=0.7)
    ax.plot(df['Time'], df['Perceptual_SubNetwork'], 'g--^', label='Perceptual SubNetwork', 
            linewidth=2.5, markersize=4, alpha=0.9)
    
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Utilization', fontsize=12)
    ax.set_title('Perceptual Modules Workload', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_ylim(-0.05, 1.1)
    
    plt.tight_layout()
    plt.savefig(save_as, dpi=300)
    print(f"Perceptual modules plot saved as '{save_as}'")
    plt.close()


def plot_cognitive_modules(df, save_as='workload_analyzer/workload_cognitive.png'):
    """
    Plot cognitive modules: Production + Declarative + Imaginary
    """
    fig, ax = plt.subplots(figsize=(14, 6))
    
    ax.plot(df['Time'], df['Production_Module'], 'b-o', label='Production Module', 
            linewidth=2, markersize=4, alpha=0.7)
    ax.plot(df['Time'], df['Declarative_Module'], 'r-s', label='Declarative Module', 
            linewidth=2, markersize=4, alpha=0.7)
    ax.plot(df['Time'], df['Imaginary_Module'], 'm-D', label='Imaginary Module', 
            linewidth=2, markersize=4, alpha=0.7)
    ax.plot(df['Time'], df['Cognitive_SubNetwork'], 'g--^', label='Cognitive SubNetwork', 
            linewidth=2.5, markersize=4, alpha=0.9)
    
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Utilization', fontsize=12)
    ax.set_title('Cognitive Modules Workload', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_ylim(-0.05, 1.1)
    
    plt.tight_layout()
    plt.savefig(save_as, dpi=300)
    print(f"Cognitive modules plot saved as '{save_as}'")
    plt.close()


def plot_motor_modules(df, save_as='workload_analyzer/workload_motor.png'):
    """
    Plot motor modules: Motor + Speech
    """
    fig, ax = plt.subplots(figsize=(14, 6))
    
    ax.plot(df['Time'], df['Motor_Module'], 'b-o', label='Motor Module', 
            linewidth=2, markersize=4, alpha=0.7)
    ax.plot(df['Time'], df['Speech_Module'], 'r-s', label='Speech Module', 
            linewidth=2, markersize=4, alpha=0.7)
    ax.plot(df['Time'], df['Motor_SubNetwork'], 'g--^', label='Motor SubNetwork', 
            linewidth=2.5, markersize=4, alpha=0.9)
    
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Utilization', fontsize=12)
    ax.set_title('Motor Modules Workload', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_ylim(-0.05, 1.1)
    
    plt.tight_layout()
    plt.savefig(save_as, dpi=300)
    print(f"Motor modules plot saved as '{save_as}'")
    plt.close()


def plot_overall_utilization(df, save_as='workload_analyzer/workload_overall.png'):
    """
    Plot overall utilization
    """
    fig, ax = plt.subplots(figsize=(14, 6))
    
    ax.plot(df['Time'], df['Overall_Utilization'], 'purple', linewidth=2.5, 
            marker='o', markersize=5, alpha=0.8, label='Overall Utilization')
    
    # Add average line
    avg_util = df['Overall_Utilization'].mean()
    ax.axhline(y=avg_util, color='red', linestyle='--', linewidth=2, 
               label=f'Average: {avg_util:.3f}', alpha=0.7)
    
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Utilization', fontsize=12)
    ax.set_title('Overall Workload Utilization', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_ylim(-0.05, max(df['Overall_Utilization'].max() * 1.1, 0.3))
    
    plt.tight_layout()
    plt.savefig(save_as, dpi=300)
    print(f"Overall utilization plot saved as '{save_as}'")
    plt.close()


def plot_all_subnetworks(df, save_as='workload_analyzer/workload_all_subnetworks.png'):
    """
    Plot all subnetworks together for comparison
    """
    fig, ax = plt.subplots(figsize=(14, 7))
    
    ax.plot(df['Time'], df['Perceptual_SubNetwork'], 'b-o', label='Perceptual SubNetwork', 
            linewidth=2, markersize=4, alpha=0.7)
    ax.plot(df['Time'], df['Cognitive_SubNetwork'], 'r-s', label='Cognitive SubNetwork', 
            linewidth=2, markersize=4, alpha=0.7)
    ax.plot(df['Time'], df['Motor_SubNetwork'], 'g-^', label='Motor SubNetwork', 
            linewidth=2, markersize=4, alpha=0.7)
    ax.plot(df['Time'], df['Overall_Utilization'], 'purple', linewidth=2.5, 
            marker='D', markersize=5, alpha=0.9, label='Overall Utilization')
    
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Utilization', fontsize=12)
    ax.set_title('All SubNetworks and Overall Utilization', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_ylim(-0.05, 1.1)
    
    plt.tight_layout()
    plt.savefig(save_as, dpi=300)
    print(f"All subnetworks plot saved as '{save_as}'")
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
    
    # Print statistics
    print_statistics(df)
    
    # Generate plots
    print("Generating plots...")
    plot_perceptual_modules(df)
    plot_cognitive_modules(df)
    plot_motor_modules(df)
    plot_overall_utilization(df)
    plot_all_subnetworks(df)
    
    print("\nAll workload visualizations completed successfully!")
