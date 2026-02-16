#!/usr/bin/env python3
"""
Monte Carlo Convergence Analysis for Flight Simulation Data

This script performs precision-based convergence analysis to determine the number
of simulation replications required for stable estimates. Uses iterative approach
based on Law & Kelton (2000) methodology for determining adequate sample size.

Reference:
    Law, A. M., & Kelton, W. D. (2000). Simulation modeling and analysis (3rd ed.).
    McGraw-Hill.

Usage:
    python convergence_analysis.py [--target-precision 0.05] [--confidence 0.95]
"""

import sys
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# Add parent directory to path to import from compare.py
sys.path.insert(0, str(Path(__file__).parent))
from compare import find_csv_files, aggregate_run_data, calculate_jae_data_baseline, aggregate_jae_data


def progressive_analysis(run_files, run_number, target_precision=0.05, confidence=0.95, max_n=None):
    """
    Progressively analyze simulation replications to assess convergence.
    
    Determines the number of replications needed such that the half-width of the
    confidence interval is no more than a specified percentage of the sample mean.
    Uses iterative t-distribution approach (Law & Kelton, 2000).
    
    Args:
        run_files: List of file info dictionaries for this run
        run_number: Condition number
        target_precision: Target relative precision (e.g., 0.05 for ±5% of mean)
        confidence: Confidence level (e.g., 0.95 for 95% CI)
        max_n: Maximum number of replications to analyze (default: None = all)
    
    Returns:
        Dictionary with progressive statistics and required sample sizes
    """
    n_total = len(run_files)
    
    # Limit analysis to max_n if specified
    if max_n is not None and max_n < n_total:
        n_total = max_n
    
    # Metrics to track
    metrics = {
        'total_fsm_time': {'values': [], 'means': [], 'cis': [], 'stds': []},
        'active_time': {'values': [], 'means': [], 'cis': [], 'stds': []},
        'coordination_time': {'values': [], 'means': [], 'cis': [], 'stds': []},
        'workload_overall': {'values': [], 'means': [], 'cis': [], 'stds': []},
        'jae_data': {'values': [], 'means': [], 'cis': [], 'stds': []},
    }
    
    sample_sizes = list(range(1, n_total + 1))
    
    print(f"\nConvergence Analysis for Condition {run_number}")
    print(f"Total available replications: {len(run_files)}")
    if max_n is not None and max_n < len(run_files):
        print(f"Analyzing up to: {n_total} replications (limited by --max-n)")
    else:
        print(f"Analyzing: {n_total} replications")
    print(f"Target relative precision: ±{target_precision*100:.1f}% of mean")
    print(f"Confidence level: {confidence*100:.0f}%")
    print("="*80)
    
    # Import run data extraction from compare.py
    from compare import extract_run_data, load_task_summary_csv
    
    # Progressively add runs and compute statistics
    for n in sample_sizes:
        subset_files = run_files[:n]
        
        # Extract data for this subset
        repetitions = []
        all_repetition_tasks = []
        
        for file_info in subset_files:
            run_data = extract_run_data(file_info['path'], file_info['folder_name'])
            if run_data:
                repetitions.append(run_data)
                
                # Load task summary for JAE
                run_folder = file_info['run_folder']
                task_summary = load_task_summary_csv(run_folder)
                if task_summary:
                    all_repetition_tasks.append(task_summary)
        
        if not repetitions:
            continue
        
        # Extract values for this subset
        total_times = [r['total_time'] for r in repetitions]
        active_times = [r['total_active_time'] for r in repetitions]
        coord_times = [r['total_coordination_time'] for r in repetitions]
        
        # Calculate statistics
        metrics['total_fsm_time']['values'].append(total_times)
        metrics['total_fsm_time']['means'].append(np.mean(total_times))
        metrics['total_fsm_time']['stds'].append(np.std(total_times, ddof=1) if n > 1 else 0)
        metrics['total_fsm_time']['cis'].append(calculate_ci_width(total_times, confidence))
        
        metrics['active_time']['values'].append(active_times)
        metrics['active_time']['means'].append(np.mean(active_times))
        metrics['active_time']['stds'].append(np.std(active_times, ddof=1) if n > 1 else 0)
        metrics['active_time']['cis'].append(calculate_ci_width(active_times, confidence))
        
        metrics['coordination_time']['values'].append(coord_times)
        metrics['coordination_time']['means'].append(np.mean(coord_times))
        metrics['coordination_time']['stds'].append(np.std(coord_times, ddof=1) if n > 1 else 0)
        metrics['coordination_time']['cis'].append(calculate_ci_width(coord_times, confidence))
        
        # Workload (if available)
        workload_values = []
        for rep in repetitions:
            if rep.get('workload') is not None:
                workload_values.append(rep['workload']['overall']['mean'])
        
        if workload_values:
            metrics['workload_overall']['values'].append(workload_values)
            metrics['workload_overall']['means'].append(np.mean(workload_values))
            metrics['workload_overall']['stds'].append(np.std(workload_values, ddof=1) if len(workload_values) > 1 else 0)
            metrics['workload_overall']['cis'].append(calculate_ci_width(workload_values, confidence))
        else:
            metrics['workload_overall']['values'].append([])
            metrics['workload_overall']['means'].append(None)
            metrics['workload_overall']['stds'].append(None)
            metrics['workload_overall']['cis'].append(None)
        
        # JAE (if available) - needs ED baseline from all data
        if all_repetition_tasks:
            # For progressive JAE, we need to calculate ED baseline from the subset
            temp_run_data = [{
                'all_repetition_tasks': all_repetition_tasks
            }]
            ed_baseline = calculate_jae_data_baseline(temp_run_data)
            
            if ed_baseline:
                jae_values = []
                for rep_tasks in all_repetition_tasks:
                    from compare import calculate_jae_metrics
                    jae_tasks = calculate_jae_metrics(rep_tasks, ed_baseline)
                    valid_jaes = [t['jae_data'] for t in jae_tasks if t['jae_data'] is not None]
                    if valid_jaes:
                        jae_values.append(np.mean(valid_jaes))
                
                if jae_values:
                    metrics['jae_data']['values'].append(jae_values)
                    metrics['jae_data']['means'].append(np.mean(jae_values))
                    metrics['jae_data']['stds'].append(np.std(jae_values, ddof=1) if len(jae_values) > 1 else 0)
                    metrics['jae_data']['cis'].append(calculate_ci_width(jae_values, confidence))
                else:
                    metrics['jae_data']['values'].append([])
                    metrics['jae_data']['means'].append(None)
                    metrics['jae_data']['stds'].append(None)
                    metrics['jae_data']['cis'].append(None)
            else:
                metrics['jae_data']['values'].append([])
                metrics['jae_data']['means'].append(None)
                metrics['jae_data']['stds'].append(None)
                metrics['jae_data']['cis'].append(None)
        else:
            metrics['jae_data']['values'].append([])
            metrics['jae_data']['means'].append(None)
            metrics['jae_data']['stds'].append(None)
            metrics['jae_data']['cis'].append(None)
    
    # Calculate required sample sizes for each metric
    # Using iterative approach based on t-distribution (Law & Kelton, 2000)
    
    required_n = {}
    for metric_name, metric_data in metrics.items():
        if metric_data['means'][-1] is not None and metric_data['stds'][-1] > 0:
            final_mean = metric_data['means'][-1]
            final_std = metric_data['stds'][-1]
            
            # Target margin of error (absolute): E = ε * |mean|
            target_E = target_precision * abs(final_mean)
            
            if target_E == 0:
                # If mean is exactly zero, skip this metric
                continue
            
            # Iterative calculation for required N using t-distribution
            # Starting with z-approximation for initial estimate
            z_score = stats.norm.ppf(1 - (1 - confidence) / 2)
            n_estimate = max(2, int(np.ceil((z_score * final_std / target_E) ** 2)))
            
            # Refine using t-distribution (iterate until convergence)
            # This accounts for the fact that n appears in both sides of the equation
            for _ in range(10):  # Usually converges in 2-3 iterations
                if n_estimate < 2:
                    n_estimate = 2  # Minimum for valid t-distribution
                    break
                t_score = stats.t.ppf(1 - (1 - confidence) / 2, n_estimate - 1)
                n_new = max(2, int(np.ceil((t_score * final_std / target_E) ** 2)))
                if n_new == n_estimate or np.isnan(n_new):
                    break
                n_estimate = n_new
            
            n_required = n_estimate
            
            required_n[metric_name] = {
                'n_required': n_required,
                'current_n': n_total,
                'final_mean': final_mean,
                'final_std': final_std,
                'final_ci_width': metric_data['cis'][-1],
                'target_E': target_E,
                'current_relative_precision': (metric_data['cis'][-1] / abs(final_mean)) if final_mean != 0 else None
            }
    
    return {
        'sample_sizes': sample_sizes,
        'metrics': metrics,
        'required_n': required_n,
        'run_number': run_number,
        'target_precision': target_precision,
        'confidence': confidence
    }


def calculate_ci_width(data, confidence=0.95):
    """Calculate half-width of confidence interval using t-distribution"""
    if len(data) == 0:
        return 0
    if len(data) == 1:
        return 0
    
    sem = stats.sem(data)
    ci_half_width = sem * stats.t.ppf((1 + confidence) / 2, len(data) - 1)
    return ci_half_width


def plot_convergence(analysis_results, output_dir):
    """Create convergence plots showing how metrics stabilize with sample size (Mean ± CI only)"""
    
    sample_sizes = analysis_results['sample_sizes']
    metrics = analysis_results['metrics']
    run_number = analysis_results['run_number']
    
    # Create figure with subplots for each metric
    metric_configs = [
        ('total_fsm_time', 'Total FSM Time (s)', 'FSM Time'),
        ('active_time', 'Active Cognitive Time (s)', 'Active Time'),
        ('coordination_time', 'Coordination Time (s)', 'Coord Time'),
        ('workload_overall', 'Overall Workload', 'Workload'),
        ('jae_data', 'JAE-Data', 'JAE'),
    ]
    
    # Filter out metrics with no data
    active_metrics = [(k, t, s) for k, t, s in metric_configs if any(m is not None for m in metrics[k]['means'])]
    
    n_metrics = len(active_metrics)
    if n_metrics == 0:
        print("No metrics available for convergence plot")
        return
    
    fig, axes = plt.subplots(n_metrics, 1, figsize=(12, 4*n_metrics))
    if n_metrics == 1:
        axes = [axes]
    
    for idx, (metric_key, metric_title, metric_short) in enumerate(active_metrics):
        metric_data = metrics[metric_key]
        
        # Filter out None values
        valid_indices = [i for i, m in enumerate(metric_data['means']) if m is not None]
        if not valid_indices:
            continue
        
        valid_n = [sample_sizes[i] for i in valid_indices]
        valid_means = [metric_data['means'][i] for i in valid_indices]
        valid_cis = [metric_data['cis'][i] for i in valid_indices]
        
        # Mean ± CI convergence
        ax = axes[idx]
        ax.plot(valid_n, valid_means, 'b-', linewidth=2, label='Mean')
        ax.fill_between(valid_n,
                         [m - ci for m, ci in zip(valid_means, valid_cis)],
                         [m + ci for m, ci in zip(valid_means, valid_cis)],
                         alpha=0.3, color='blue', label='95% CI')
        
        # Add horizontal line for final estimate
        if valid_means:
            final_mean = valid_means[-1]
            ax.axhline(y=final_mean, color='red', linestyle='--', alpha=0.5, label='Final Estimate')
        
        ax.set_xlabel('Number of Replications (n)', fontsize=11, fontweight='bold')
        ax.set_ylabel(metric_title, fontsize=11, fontweight='bold')
        ax.set_title(f'{metric_short} Convergence - Mean ± 95% CI', fontsize=12, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = Path(output_dir) / f'convergence_analysis_mean_ci_run{run_number}.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.savefig(plot_path.with_suffix('.eps'), format='eps', bbox_inches='tight')
    print(f"Saved convergence plot: {plot_path}")
    plt.close()


def plot_relative_precision_consolidated(analysis_results, output_dir):
    """Create consolidated plot showing relative precision for all metrics on one graph"""
    
    sample_sizes = analysis_results['sample_sizes']
    metrics = analysis_results['metrics']
    run_number = analysis_results['run_number']
    target_precision = analysis_results['target_precision']
    
    # Metric configurations with display names
    metric_configs = [
        ('total_fsm_time', 'Total FSM Time', '#1f77b4'),      # Blue
        ('active_time', 'Active Cognitive Time', '#ff7f0e'),   # Orange
        ('coordination_time', 'Coordination Time', '#2ca02c'),  # Green
        ('workload_overall', 'Overall Workload', '#d62728'),    # Red
        ('jae_data', 'JAE-Data', '#9467bd'),                    # Purple
    ]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Track if we have any data
    has_data = False
    
    # Plot each metric
    for metric_key, metric_label, color in metric_configs:
        metric_data = metrics[metric_key]
        
        # Filter out None values
        valid_indices = [i for i, m in enumerate(metric_data['means']) if m is not None]
        if not valid_indices:
            continue
        
        has_data = True
        valid_n = [sample_sizes[i] for i in valid_indices]
        valid_means = [metric_data['means'][i] for i in valid_indices]
        valid_cis = [metric_data['cis'][i] for i in valid_indices]
        
        # Calculate relative precision (CI width / mean * 100)
        relative_precision = [(ci / abs(m) * 100) if m != 0 else 0 for m, ci in zip(valid_means, valid_cis)]
        
        # Plot line
        ax.plot(valid_n, relative_precision, linewidth=2.5, marker='o', 
                markersize=6, label=metric_label, color=color, alpha=0.8)
    
    if not has_data:
        print("No metrics available for relative precision plot")
        plt.close()
        return
    
    # Add target line
    target_line = target_precision * 100
    ax.axhline(y=target_line, color='black', linestyle='--', linewidth=2,
               label=f'Target Precision: ±{target_precision*100:.1f}%', zorder=10)
    
    # Shade acceptable region
    y_max = ax.get_ylim()[1]
    ax.fill_between([min(sample_sizes), max(sample_sizes)], 0, target_line,
                    alpha=0.15, color='green', label='Acceptable Precision Zone', zorder=1)
    
    # Formatting
    ax.set_xlabel('Number of Replications (n)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Relative Precision (% of Mean)', fontsize=13, fontweight='bold')
    ax.set_title(f'Precision Convergence - All Metrics (Condition {run_number})', 
                fontsize=14, fontweight='bold', pad=15)
    ax.legend(loc='best', fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.8)
    ax.set_ylim(bottom=0)
    
    # Add note about interpretation
    note_text = f"Lower values indicate better precision\nTarget: CI half-width ≤ {target_precision*100:.1f}% of mean"
    ax.text(0.98, 0.98, note_text, transform=ax.transAxes,
            fontsize=9, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    # Save plot
    plot_path = Path(output_dir) / f'convergence_analysis_precision_run{run_number}.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.savefig(plot_path.with_suffix('.eps'), format='eps', bbox_inches='tight')
    print(f"Saved precision convergence plot: {plot_path}")
    plt.close()


def print_convergence_report(analysis_results):
    """Print detailed convergence analysis report"""
    
    run_number = analysis_results['run_number']
    required_n = analysis_results['required_n']
    target_precision = analysis_results['target_precision']
    confidence = analysis_results['confidence']
    
    print("\n" + "="*80)
    print(f"CONVERGENCE ANALYSIS REPORT - Condition {run_number}")
    print("="*80)
    print(f"Target Relative Precision: ±{target_precision*100:.1f}% of mean")
    print(f"Confidence Level: {confidence*100:.0f}%")
    print(f"Method: Iterative t-distribution (Law & Kelton, 2000)")
    print("="*80)
    
    # Table header
    print(f"\n{'Metric':<25} {'Current n':<12} {'Required n':<12} {'Status':<12} {'Current ±%':<12}")
    print("-"*80)
    
    for metric_name, data in required_n.items():
        metric_display = metric_name.replace('_', ' ').title()
        current_n = data['current_n']
        n_req = data['n_required']
        current_rel_prec = data['current_relative_precision']
        
        if current_rel_prec is not None:
            current_pct = current_rel_prec * 100
        else:
            current_pct = 0
        
        if current_n >= n_req:
            status = "✓ Sufficient"
        else:
            status = f"Need {n_req - current_n} more"
        
        print(f"{metric_display:<25} {current_n:<12} {n_req:<12} {status:<12} ±{current_pct:.1f}%")
    
    print("="*80)
    
    # Recommendations
    max_required = max(data['n_required'] for data in required_n.values())
    current_n = list(required_n.values())[0]['current_n']
    
    print("\nRECOMMENDATIONS:")
    print("-"*80)
    
    if current_n >= max_required:
        print(f"✓ Current number of replications (n={current_n}) is SUFFICIENT for all metrics.")
        print(f"  All metrics achieve ±{target_precision*100:.1f}% relative precision at {confidence*100:.0f}% confidence.")
    else:
        print(f"⚠ Recommend running at least n={max_required} simulation replications total.")
        print(f"  This ensures ±{target_precision*100:.1f}% precision for all metrics.")
        print(f"  Additional replications needed: {max_required - current_n}")
    
    print("\nDETAILED METRICS:")
    print("-"*80)
    for metric_name, data in required_n.items():
        metric_display = metric_name.replace('_', ' ').title()
        print(f"\n{metric_display}:")
        print(f"  Final estimate: {data['final_mean']:.4f} ± {data['final_ci_width']:.4f}")
        print(f"  Standard deviation: {data['final_std']:.4f}")
        print(f"  Target margin of error: ±{data['target_E']:.4f} (±{target_precision*100:.1f}%)")
        if data['current_relative_precision'] is not None:
            print(f"  Current precision: ±{data['current_relative_precision']*100:.2f}% of mean")
        print(f"  Required n: {data['n_required']}")
    
    print("="*80)


def export_convergence_csv(analysis_results, output_dir):
    """Export convergence analysis results to CSV"""
    
    run_number = analysis_results['run_number']
    csv_path = Path(output_dir) / f'convergence_analysis_run{run_number}.csv'
    
    # Create detailed CSV with progressive statistics
    data_rows = []
    sample_sizes = analysis_results['sample_sizes']
    metrics = analysis_results['metrics']
    
    for i, n in enumerate(sample_sizes):
        row = {'n_replications': n}
        
        for metric_name, metric_data in metrics.items():
            if i < len(metric_data['means']) and metric_data['means'][i] is not None:
                row[f'{metric_name}_mean'] = metric_data['means'][i]
                row[f'{metric_name}_ci'] = metric_data['cis'][i]
                row[f'{metric_name}_std'] = metric_data['stds'][i]
                
                # Relative precision
                if metric_data['means'][i] != 0:
                    rel_prec = metric_data['cis'][i] / abs(metric_data['means'][i])
                    row[f'{metric_name}_rel_precision_pct'] = rel_prec * 100
        
        data_rows.append(row)
    
    df = pd.DataFrame(data_rows)
    df.to_csv(csv_path, index=False)
    print(f"Exported convergence data: {csv_path}")
    
    # Also export summary with required sample sizes
    summary_path = Path(output_dir) / f'convergence_summary_run{run_number}.csv'
    summary_rows = []
    
    for metric_name, data in analysis_results['required_n'].items():
        summary_rows.append({
            'Metric': metric_name,
            'Current_N': data['current_n'],
            'Required_N': data['n_required'],
            'Final_Mean': data['final_mean'],
            'Final_Std': data['final_std'],
            'Final_CI_Width': data['final_ci_width'],
            'Target_Margin_Error': data['target_E'],
            'Current_Relative_Precision_Pct': data['current_relative_precision'] * 100 if data['current_relative_precision'] else None,
            'Target_Relative_Precision_Pct': analysis_results['target_precision'] * 100,
            'Status': 'Sufficient' if data['current_n'] >= data['n_required'] else 'Need More'
        })
    
    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(summary_path, index=False)
    print(f"Exported convergence summary: {summary_path}")


def plot_multi_run_precision_comparison(all_analysis_results, output_dir):
    """
    Create a multi-panel figure comparing precision convergence across all runs.
    
    Args:
        all_analysis_results: List of analysis result dictionaries from all runs
        output_dir: Directory to save the plot
    """
    if not all_analysis_results:
        return
    
    n_runs = len(all_analysis_results)
    
    # Create figure with subplots for each run (independent y-axes)
    fig, axes = plt.subplots(1, n_runs, figsize=(7*n_runs, 6), sharey=False)
    
    # Handle single run case
    if n_runs == 1:
        axes = [axes]
    
    # Metric colors
    metric_colors = {
        'total_fsm_time': '#1f77b4',      # Blue
        'active_time': '#ff7f0e',          # Orange
        'coordination_time': '#2ca02c',    # Green
        'workload_overall': '#d62728',     # Red
        'jae_data': '#9467bd'              # Purple
    }
    
    metric_labels = {
        'total_fsm_time': 'Total FSM Time',
        'active_time': 'Active Time',
        'coordination_time': 'Coordination Time',
        'workload_overall': 'Workload',
        'jae_data': 'JAE-Data'
    }
    
    # Plot each run
    for idx, analysis_results in enumerate(all_analysis_results):
        ax = axes[idx]
        run_number = analysis_results['run_number']
        target_precision = analysis_results['target_precision']
        sample_sizes = analysis_results['sample_sizes']
        metrics = analysis_results['metrics']
        
        has_data = False
        
        # Plot each metric
        for metric_name, color in metric_colors.items():
            if metric_name not in metrics:
                continue
            
            metric_data = metrics[metric_name]
            means = metric_data['means']
            cis = metric_data['cis']
            
            # Filter out invalid data
            valid_data = [(n, m, ci) for n, m, ci in zip(sample_sizes, means, cis) 
                         if m != 0 and not np.isnan(m) and not np.isnan(ci)]
            
            if not valid_data:
                continue
            
            has_data = True
            valid_n, valid_means, valid_cis = zip(*valid_data)
            
            # Calculate relative precision (%)
            relative_precision = [(ci / abs(m) * 100) if m != 0 else 0 
                                 for m, ci in zip(valid_means, valid_cis)]
            
            # Plot line
            metric_label = metric_labels.get(metric_name, metric_name)
            ax.plot(valid_n, relative_precision, linewidth=2.5, marker='o', 
                   markersize=6, label=metric_label, color=color, alpha=0.8)
        
        if not has_data:
            continue
        
        # Add target line
        target_line = target_precision * 100
        ax.axhline(y=target_line, color='black', linestyle='--', linewidth=2,
                  label=f'Target: ±{target_precision*100:.1f}%' if idx == 0 else None, 
                  zorder=10)
        
        # Shade acceptable region (PNG only - EPS doesn't handle transparency well)
        y_max = ax.get_ylim()[1]
        shaded_area = ax.fill_between([min(sample_sizes), max(sample_sizes)], 0, target_line,
                                      facecolor='green', edgecolor='none', alpha=0.2, zorder=1)
        
        # Formatting
        ax.set_xlabel('Number of Replications (n)', fontsize=12, fontweight='bold')
        if idx == 0:
            ax.set_ylabel('Relative Precision (% of Mean)', fontsize=12, fontweight='bold')
        ax.set_title(f'Condition {run_number}', fontsize=13, fontweight='bold', pad=10)
        # No grid for cleaner appearance
        ax.set_ylim(bottom=0)
        
        # Legend only on first subplot
        if idx == 0:
            ax.legend(loc='best', fontsize=9, framealpha=0.9)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save PNG with shaded area
    plot_path = Path(output_dir) / 'convergence_analysis_precision_all_runs.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved multi-run precision comparison: {plot_path}")
    
    # Remove shaded areas and save EPS
    for ax in axes:
        for collection in ax.collections[:]:
            collection.remove()
    
    plt.savefig(plot_path.with_suffix('.eps'), format='eps', bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Monte Carlo Convergence Analysis for Simulation Data',
        epilog='Determines required number of simulation replications for stable estimates.'
    )
    parser.add_argument('--target-precision', type=float, default=0.05,
                       help='Target relative precision (default: 0.05 for ±5%% of mean)')
    parser.add_argument('--confidence', type=float, default=0.95,
                       help='Confidence level (default: 0.95 for 95%% CI)')
    parser.add_argument('--run', type=int, default=None,
                       help='Specific run number to analyze (default: analyze all runs)')
    parser.add_argument('--max-n', type=int, default=None,
                       help='Maximum number of replications to analyze (default: None = all available)')
    
    args = parser.parse_args()
    
    # Get output directory
    script_dir = Path(__file__).parent.absolute()
    
    print("="*80)
    print("MONTE CARLO CONVERGENCE ANALYSIS")
    print("="*80)
    print(f"Output directory: {script_dir}")
    print(f"Method: Iterative t-distribution (Law & Kelton, 2000)")
    
    # Find all run files
    all_runs = find_csv_files(script_dir)
    
    if not all_runs:
        print("Error: No run data found")
        return
    
    print(f"\nFound {len(all_runs)} experimental run(s)")
    for run in all_runs:
        print(f"  - Condition {run['run_number']}: {len(run['files'])} replications")
    
    # Analyze specified run or all runs
    if args.run is not None:
        runs_to_analyze = [r for r in all_runs if r['run_number'] == args.run]
        if not runs_to_analyze:
            print(f"Error: Condition {args.run} not found")
            available_runs = [r['run_number'] for r in all_runs]
            print(f"Available runs: {available_runs}")
            return
        print(f"\nAnalyzing only Condition {args.run} (as requested)")
    else:
        runs_to_analyze = all_runs
        print(f"\nAnalyzing all runs")
    
    # Store all analysis results for multi-run comparison
    all_analysis_results = []
    
    for run_info in runs_to_analyze:
        run_number = run_info['run_number']
        run_files = run_info['files']
        
        # Perform progressive analysis
        analysis_results = progressive_analysis(
            run_files, 
            run_number,
            target_precision=args.target_precision,
            confidence=args.confidence,
            max_n=args.max_n
        )
        
        # Store for multi-run comparison
        all_analysis_results.append(analysis_results)
        
        # Print report
        print_convergence_report(analysis_results)
        
        # Create individual plots
        plot_convergence(analysis_results, script_dir)
        plot_relative_precision_consolidated(analysis_results, script_dir)
        
        # Export CSV
        export_convergence_csv(analysis_results, script_dir)
    
    # Create multi-run comparison plot if analyzing multiple runs
    if len(all_analysis_results) > 1:
        plot_multi_run_precision_comparison(all_analysis_results, script_dir)
    
    print("\n" + "="*80)
    print("CONVERGENCE ANALYSIS COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
