#!/usr/bin/env python3
"""
Power Analysis for Flight Simulation Data

This script performs precision-based power analysis to determine required sample size
for stable confidence intervals. It progressively analyzes runs to show convergence.

Usage:
    python power.py [--target-precision 0.05] [--confidence 0.95]
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


def progressive_analysis(run_files, run_number, target_precision=0.05, confidence=0.95):
    """
    Progressively analyze runs to show convergence of estimates.
    
    Args:
        run_files: List of file info dictionaries for this run
        run_number: Run number
        target_precision: Target relative precision (e.g., 0.05 for ±5%)
        confidence: Confidence level (e.g., 0.95 for 95% CI)
    
    Returns:
        Dictionary with progressive statistics and required sample sizes
    """
    n_total = len(run_files)
    
    # Metrics to track
    metrics = {
        'total_fsm_time': {'values': [], 'means': [], 'cis': [], 'stds': []},
        'active_time': {'values': [], 'means': [], 'cis': [], 'stds': []},
        'coordination_time': {'values': [], 'means': [], 'cis': [], 'stds': []},
        'workload_overall': {'values': [], 'means': [], 'cis': [], 'stds': []},
        'jae_data': {'values': [], 'means': [], 'cis': [], 'stds': []},
    }
    
    sample_sizes = list(range(1, n_total + 1))
    
    print(f"\nProgressive Analysis for Run {run_number}")
    print(f"Total available repetitions: {n_total}")
    print(f"Target relative precision: ±{target_precision*100:.1f}%")
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
    z_score = stats.norm.ppf(1 - (1 - confidence) / 2)  # For 95% CI, z ≈ 1.96
    
    required_n = {}
    for metric_name, metric_data in metrics.items():
        if metric_data['means'][-1] is not None and metric_data['stds'][-1] > 0:
            final_mean = metric_data['means'][-1]
            final_std = metric_data['stds'][-1]
            
            # Target margin of error (absolute)
            target_E = target_precision * final_mean
            
            # Required N: N ≈ (z * s / E)²
            n_required = (z_score * final_std / target_E) ** 2
            
            required_n[metric_name] = {
                'n_required': int(np.ceil(n_required)),
                'current_n': n_total,
                'final_mean': final_mean,
                'final_std': final_std,
                'final_ci_width': metric_data['cis'][-1],
                'target_E': target_E,
                'current_relative_precision': (metric_data['cis'][-1] / final_mean) if final_mean > 0 else None
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
    """Calculate half-width of confidence interval"""
    if len(data) == 0:
        return 0
    if len(data) == 1:
        return 0
    
    sem = stats.sem(data)
    ci_half_width = sem * stats.t.ppf((1 + confidence) / 2, len(data) - 1)
    return ci_half_width


def plot_convergence(analysis_results, output_dir):
    """Create convergence plots showing how metrics stabilize with sample size"""
    
    sample_sizes = analysis_results['sample_sizes']
    metrics = analysis_results['metrics']
    run_number = analysis_results['run_number']
    target_precision = analysis_results['target_precision']
    
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
    
    fig, axes = plt.subplots(n_metrics, 2, figsize=(16, 4*n_metrics))
    if n_metrics == 1:
        axes = axes.reshape(1, -1)
    
    for idx, (metric_key, metric_title, metric_short) in enumerate(active_metrics):
        metric_data = metrics[metric_key]
        
        # Filter out None values
        valid_indices = [i for i, m in enumerate(metric_data['means']) if m is not None]
        if not valid_indices:
            continue
        
        valid_n = [sample_sizes[i] for i in valid_indices]
        valid_means = [metric_data['means'][i] for i in valid_indices]
        valid_cis = [metric_data['cis'][i] for i in valid_indices]
        valid_stds = [metric_data['stds'][i] for i in valid_indices]
        
        # Left plot: Mean ± CI convergence
        ax1 = axes[idx, 0]
        ax1.plot(valid_n, valid_means, 'b-', linewidth=2, label='Mean')
        ax1.fill_between(valid_n,
                         [m - ci for m, ci in zip(valid_means, valid_cis)],
                         [m + ci for m, ci in zip(valid_means, valid_cis)],
                         alpha=0.3, color='blue', label='95% CI')
        
        # Add horizontal line for final estimate
        if valid_means:
            final_mean = valid_means[-1]
            ax1.axhline(y=final_mean, color='red', linestyle='--', alpha=0.5, label='Final Estimate')
        
        ax1.set_xlabel('Number of Repetitions (n)', fontsize=11, fontweight='bold')
        ax1.set_ylabel(metric_title, fontsize=11, fontweight='bold')
        ax1.set_title(f'{metric_short} Convergence - Mean ± 95% CI', fontsize=12, fontweight='bold')
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)
        
        # Right plot: Relative precision convergence
        ax2 = axes[idx, 1]
        
        # Calculate relative precision (CI width / mean)
        relative_precision = [(ci / m * 100) if m > 0 else 0 for m, ci in zip(valid_means, valid_cis)]
        
        ax2.plot(valid_n, relative_precision, 'g-', linewidth=2, marker='o', markersize=4)
        
        # Add target line
        target_line = target_precision * 100
        ax2.axhline(y=target_line, color='red', linestyle='--', linewidth=2,
                   label=f'Target: ±{target_precision*100:.1f}%')
        
        # Shade acceptable region
        ax2.fill_between([min(valid_n), max(valid_n)], 0, target_line,
                        alpha=0.2, color='green', label='Acceptable Precision')
        
        ax2.set_xlabel('Number of Repetitions (n)', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Relative Precision (% of mean)', fontsize=11, fontweight='bold')
        ax2.set_title(f'{metric_short} - CI Half-Width as % of Mean', fontsize=12, fontweight='bold')
        ax2.legend(loc='best')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(bottom=0)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = Path(output_dir) / f'power_analysis_convergence_run{run_number}.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.savefig(plot_path.with_suffix('.eps'), format='eps', bbox_inches='tight')
    print(f"Saved convergence plot: {plot_path}")
    plt.close()


def print_power_analysis_report(analysis_results):
    """Print detailed power analysis report"""
    
    run_number = analysis_results['run_number']
    required_n = analysis_results['required_n']
    target_precision = analysis_results['target_precision']
    confidence = analysis_results['confidence']
    
    print("\n" + "="*80)
    print(f"POWER ANALYSIS REPORT - Run {run_number}")
    print("="*80)
    print(f"Target Relative Precision: ±{target_precision*100:.1f}% of mean")
    print(f"Confidence Level: {confidence*100:.0f}%")
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
        print(f"✓ Current sample size (n={current_n}) is SUFFICIENT for all metrics.")
        print(f"  All metrics achieve ±{target_precision*100:.1f}% relative precision at {confidence*100:.0f}% confidence.")
    else:
        print(f"⚠ Recommend running at least n={max_required} repetitions total.")
        print(f"  This ensures ±{target_precision*100:.1f}% precision for all metrics.")
        print(f"  Additional runs needed: {max_required - current_n}")
    
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


def export_power_analysis_csv(analysis_results, output_dir):
    """Export power analysis results to CSV"""
    
    run_number = analysis_results['run_number']
    csv_path = Path(output_dir) / f'power_analysis_run{run_number}.csv'
    
    # Create detailed CSV with progressive statistics
    data_rows = []
    sample_sizes = analysis_results['sample_sizes']
    metrics = analysis_results['metrics']
    
    for i, n in enumerate(sample_sizes):
        row = {'n_repetitions': n}
        
        for metric_name, metric_data in metrics.items():
            if i < len(metric_data['means']) and metric_data['means'][i] is not None:
                row[f'{metric_name}_mean'] = metric_data['means'][i]
                row[f'{metric_name}_ci'] = metric_data['cis'][i]
                row[f'{metric_name}_std'] = metric_data['stds'][i]
                
                # Relative precision
                if metric_data['means'][i] > 0:
                    rel_prec = metric_data['cis'][i] / metric_data['means'][i]
                    row[f'{metric_name}_rel_precision_pct'] = rel_prec * 100
        
        data_rows.append(row)
    
    df = pd.DataFrame(data_rows)
    df.to_csv(csv_path, index=False)
    print(f"Exported power analysis data: {csv_path}")
    
    # Also export summary with required sample sizes
    summary_path = Path(output_dir) / f'power_analysis_summary_run{run_number}.csv'
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
    print(f"Exported power analysis summary: {summary_path}")


def main():
    parser = argparse.ArgumentParser(description='Power Analysis for Flight Simulation Data')
    parser.add_argument('--target-precision', type=float, default=0.05,
                       help='Target relative precision (default: 0.05 for ±5%%)')
    parser.add_argument('--confidence', type=float, default=0.95,
                       help='Confidence level (default: 0.95 for 95%% CI)')
    parser.add_argument('--run', type=int, default=None,
                       help='Specific run number to analyze (default: analyze all runs)')
    
    args = parser.parse_args()
    
    # Get output directory
    script_dir = Path(__file__).parent.absolute()
    
    print("="*80)
    print("PRECISION-BASED POWER ANALYSIS")
    print("="*80)
    print(f"Output directory: {script_dir}")
    
    # Find all run files
    all_runs = find_csv_files(script_dir)
    
    if not all_runs:
        print("Error: No run data found")
        return
    
    print(f"\nFound {len(all_runs)} experimental run(s)")
    for run in all_runs:
        print(f"  - Run {run['run_number']}: {len(run['files'])} repetitions")
    
    # Analyze specified run or all runs
    if args.run is not None:
        runs_to_analyze = [r for r in all_runs if r['run_number'] == args.run]
        if not runs_to_analyze:
            print(f"Error: Run {args.run} not found")
            available_runs = [r['run_number'] for r in all_runs]
            print(f"Available runs: {available_runs}")
            return
        print(f"\nAnalyzing only Run {args.run} (as requested)")
    else:
        runs_to_analyze = all_runs
        print(f"\nAnalyzing all runs")
    
    for run_info in runs_to_analyze:
        run_number = run_info['run_number']
        run_files = run_info['files']
        
        # Perform progressive analysis
        analysis_results = progressive_analysis(
            run_files, 
            run_number,
            target_precision=args.target_precision,
            confidence=args.confidence
        )
        
        # Print report
        print_power_analysis_report(analysis_results)
        
        # Create plots
        plot_convergence(analysis_results, script_dir)
        
        # Export CSV
        export_power_analysis_csv(analysis_results, script_dir)
    
    print("\n" + "="*80)
    print("POWER ANALYSIS COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
