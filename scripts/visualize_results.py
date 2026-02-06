#!/usr/bin/env python3
"""
Script to visualize experiment results.

This script generates plots and visualizations from experiment results,
including comparison charts, performance metrics, and trend analysis.

Usage:
    python scripts/visualize_results.py --results ./results/experiment/results.json
    python scripts/visualize_results.py --results-dir ./results --compare
    python scripts/visualize_results.py --results results.json --output-dir ./plots
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import defaultdict

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


def load_results(results_path: Path) -> Dict[str, Any]:
    """Load results from JSON file."""
    with open(results_path, 'r') as f:
        return json.load(f)


def plot_resolution_comparison(
    results: Dict[str, Any],
    output_path: Path,
    title: str = "Resolution Rate Comparison"
) -> None:
    """
    Plot resolution rate comparison across methods.
    
    Args:
        results: Results dictionary
        output_path: Path to save plot
        title: Plot title
    """
    methods = []
    resolution_rates = []
    
    # Extract resolution rates
    if 'results' in results:
        for method_name, method_results in results['results'].items():
            if isinstance(method_results, dict) and 'summary' in method_results:
                methods.append(method_name)
                resolution_rates.append(method_results['summary'].get('resolution_rate', 0))
    
    if not methods:
        logger.warning("No resolution data found")
        return
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = sns.color_palette("husl", len(methods))
    bars = ax.bar(methods, resolution_rates, color=colors)
    
    ax.set_ylabel('Resolution Rate (%)', fontsize=12)
    ax.set_xlabel('Method', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_ylim(0, max(resolution_rates) * 1.2)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.,
            height,
            f'{height:.1f}%',
            ha='center',
            va='bottom',
            fontsize=10
        )
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved resolution comparison plot to {output_path}")


def plot_pass_at_k(
    results: Dict[str, Any],
    output_path: Path,
    title: str = "Pass@k Comparison"
) -> None:
    """
    Plot Pass@k metrics for different methods.
    
    Args:
        results: Results dictionary
        output_path: Path to save plot
        title: Plot title
    """
    methods = []
    pass_at_k_data = defaultdict(list)
    k_values = []
    
    # Extract Pass@k data
    if 'results' in results:
        for method_name, method_results in results['results'].items():
            if isinstance(method_results, dict) and 'summary' in method_results:
                methods.append(method_name)
                pass_at_k = method_results['summary'].get('pass_at_k', {})
                for k, rate in pass_at_k.items():
                    pass_at_k_data[method_name].append(rate)
                    if k not in k_values:
                        k_values.append(k)
    
    if not methods or not k_values:
        logger.warning("No Pass@k data found")
        return
    
    k_values = sorted([int(k) for k in k_values])
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for method in methods:
        rates = pass_at_k_data[method]
        ax.plot(k_values[:len(rates)], rates, marker='o', label=method, linewidth=2)
    
    ax.set_xlabel('k', fontsize=12)
    ax.set_ylabel('Pass@k (%)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved Pass@k plot to {output_path}")


def plot_token_usage(
    results: Dict[str, Any],
    output_path: Path,
    title: str = "Token Usage Comparison"
) -> None:
    """
    Plot token usage comparison across methods.
    
    Args:
        results: Results dictionary
        output_path: Path to save plot
        title: Plot title
    """
    methods = []
    prompt_tokens = []
    completion_tokens = []
    
    # Extract token usage
    if 'results' in results:
        for method_name, method_results in results['results'].items():
            if isinstance(method_results, dict) and 'summary' in method_results:
                token_usage = method_results['summary'].get('token_usage', {})
                methods.append(method_name)
                prompt_tokens.append(token_usage.get('prompt_tokens', 0) / 1000)  # Convert to thousands
                completion_tokens.append(token_usage.get('completion_tokens', 0) / 1000)
    
    if not methods:
        logger.warning("No token usage data found")
        return
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(methods))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, prompt_tokens, width, label='Prompt Tokens', color='skyblue')
    bars2 = ax.bar(x + width/2, completion_tokens, width, label='Completion Tokens', color='lightcoral')
    
    ax.set_ylabel('Tokens (thousands)', fontsize=12)
    ax.set_xlabel('Method', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45, ha='right')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved token usage plot to {output_path}")


def plot_resolution_by_repo(
    results: Dict[str, Any],
    output_path: Path,
    title: str = "Resolution Rate by Repository"
) -> None:
    """
    Plot resolution rate breakdown by repository.
    
    Args:
        results: Results dictionary
        output_path: Path to save plot
        title: Plot title
    """
    repo_stats = defaultdict(lambda: defaultdict(lambda: {'resolved': 0, 'total': 0}))
    
    # Extract per-repo statistics
    if 'results' in results:
        for method_name, method_results in results['results'].items():
            if isinstance(method_results, dict) and 'results' in method_results:
                for result in method_results['results']:
                    repo = result.get('repo', 'unknown')
                    resolved = result.get('resolved', False)
                    
                    repo_stats[repo][method_name]['total'] += 1
                    if resolved:
                        repo_stats[repo][method_name]['resolved'] += 1
    
    if not repo_stats:
        logger.warning("No per-repo data found")
        return
    
    # Prepare data for plotting
    repos = sorted(repo_stats.keys())
    methods = sorted(set(
        method for repo_data in repo_stats.values() for method in repo_data.keys()
    ))
    
    # Create grouped bar chart
    fig, ax = plt.subplots(figsize=(14, 8))
    
    x = np.arange(len(repos))
    width = 0.8 / len(methods)
    
    colors = sns.color_palette("husl", len(methods))
    
    for i, method in enumerate(methods):
        rates = []
        for repo in repos:
            stats = repo_stats[repo][method]
            rate = (stats['resolved'] / stats['total'] * 100) if stats['total'] > 0 else 0
            rates.append(rate)
        
        offset = width * (i - len(methods) / 2 + 0.5)
        ax.bar(x + offset, rates, width, label=method, color=colors[i])
    
    ax.set_ylabel('Resolution Rate (%)', fontsize=12)
    ax.set_xlabel('Repository', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(repos, rotation=45, ha='right')
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved resolution by repo plot to {output_path}")


def plot_heatmap(
    results: Dict[str, Any],
    output_path: Path,
    title: str = "Method Performance Heatmap"
) -> None:
    """
    Plot performance heatmap across methods and metrics.
    
    Args:
        results: Results dictionary
        output_path: Path to save plot
        title: Plot title
    """
    methods = []
    metrics_data = defaultdict(list)
    
    # Extract metrics
    if 'results' in results:
        for method_name, method_results in results['results'].items():
            if isinstance(method_results, dict) and 'summary' in method_results:
                methods.append(method_name)
                summary = method_results['summary']
                
                metrics_data['Resolution'].append(summary.get('resolution_rate', 0))
                metrics_data['Localization'].append(summary.get('localization_accuracy', 0))
                metrics_data['CodeBLEU'].append(summary.get('codebleu_score', 0) * 100)
    
    if not methods:
        logger.warning("No metric data found")
        return
    
    # Create heatmap data
    metric_names = list(metrics_data.keys())
    data = np.array([metrics_data[m] for m in metric_names])
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    im = ax.imshow(data, cmap='YlGnBu', aspect='auto', vmin=0, vmax=100)
    
    ax.set_xticks(np.arange(len(methods)))
    ax.set_yticks(np.arange(len(metric_names)))
    ax.set_xticklabels(methods, rotation=45, ha='right')
    ax.set_yticklabels(metric_names)
    
    # Add text annotations
    for i in range(len(metric_names)):
        for j in range(len(methods)):
            text = ax.text(
                j, i, f'{data[i, j]:.1f}',
                ha="center", va="center", color="black", fontsize=10
            )
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    plt.colorbar(im, ax=ax, label='Score (%)')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved heatmap to {output_path}")


def plot_comparison_table(
    results: Dict[str, Any],
    output_path: Path
) -> None:
    """
    Create a comparison table visualization.
    
    Args:
        results: Results dictionary
        output_path: Path to save plot
    """
    methods = []
    table_data = []
    
    # Extract data
    if 'results' in results:
        for method_name, method_results in results['results'].items():
            if isinstance(method_results, dict) and 'summary' in method_results:
                methods.append(method_name)
                summary = method_results['summary']
                
                row = [
                    method_name,
                    f"{summary.get('resolution_rate', 0):.1f}%",
                    f"{summary.get('localization_accuracy', 0):.1f}%",
                    f"{summary.get('codebleu_score', 0):.3f}",
                    f"{summary.get('token_usage', {}).get('total_tokens', 0) / 1000:.1f}K"
                ]
                table_data.append(row)
    
    if not table_data:
        logger.warning("No data for table")
        return
    
    # Create table
    fig, ax = plt.subplots(figsize=(12, len(table_data) * 0.5 + 2))
    ax.axis('tight')
    ax.axis('off')
    
    columns = ['Method', 'Resolution', 'Localization', 'CodeBLEU', 'Tokens']
    
    table = ax.table(
        cellText=table_data,
        colLabels=columns,
        cellLoc='center',
        loc='center',
        colWidths=[0.25, 0.15, 0.15, 0.15, 0.15]
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style header
    for i in range(len(columns)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Alternate row colors
    for i in range(1, len(table_data) + 1):
        for j in range(len(columns)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')
    
    plt.title('Method Comparison Summary', fontsize=14, fontweight='bold', pad=20)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved comparison table to {output_path}")


def compare_multiple_experiments(
    results_dir: Path,
    output_dir: Path
) -> None:
    """
    Compare results from multiple experiments.
    
    Args:
        results_dir: Directory containing result files
        output_dir: Directory to save plots
    """
    # Find all result files
    result_files = list(results_dir.glob('**/results.json'))
    
    if not result_files:
        logger.warning(f"No results files found in {results_dir}")
        return
    
    logger.info(f"Found {len(result_files)} result files")
    
    # Load and aggregate results
    all_results = {}
    for result_file in result_files:
        exp_name = result_file.parent.name
        try:
            results = load_results(result_file)
            all_results[exp_name] = results
        except Exception as e:
            logger.warning(f"Failed to load {result_file}: {e}")
    
    # Create comparison plots
    plot_multi_experiment_comparison(all_results, output_dir)


def plot_multi_experiment_comparison(
    all_results: Dict[str, Dict],
    output_dir: Path
) -> None:
    """
    Plot comparison across multiple experiments.
    
    Args:
        all_results: Dictionary mapping experiment names to results
        output_dir: Directory to save plots
    """
    # Aggregate resolution rates
    exp_names = []
    method_names = set()
    resolution_data = defaultdict(dict)
    
    for exp_name, results in all_results.items():
        exp_names.append(exp_name)
        if 'results' in results:
            for method_name, method_results in results['results'].items():
                method_names.add(method_name)
                if isinstance(method_results, dict) and 'summary' in method_results:
                    resolution_data[method_name][exp_name] = method_results['summary'].get('resolution_rate', 0)
    
    if not exp_names or not method_names:
        logger.warning("No data for multi-experiment comparison")
        return
    
    # Create grouped bar chart
    fig, ax = plt.subplots(figsize=(14, 8))
    
    x = np.arange(len(exp_names))
    width = 0.8 / len(method_names)
    
    colors = sns.color_palette("husl", len(method_names))
    method_list = sorted(method_names)
    
    for i, method in enumerate(method_list):
        rates = [resolution_data[method].get(exp, 0) for exp in exp_names]
        offset = width * (i - len(method_list) / 2 + 0.5)
        ax.bar(x + offset, rates, width, label=method, color=colors[i])
    
    ax.set_ylabel('Resolution Rate (%)', fontsize=12)
    ax.set_xlabel('Experiment', fontsize=12)
    ax.set_title('Resolution Rate Across Experiments', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(exp_names, rotation=45, ha='right')
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    output_path = output_dir / 'multi_experiment_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved multi-experiment comparison to {output_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Visualize experiment results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate all plots for a result file
  python scripts/visualize_results.py --results ./results/exp/results.json
  
  # Generate specific plots
  python scripts/visualize_results.py --results results.json --plots resolution,pass_at_k
  
  # Compare multiple experiments
  python scripts/visualize_results.py --results-dir ./results --compare
  
  # Custom output directory
  python scripts/visualize_results.py --results results.json --output-dir ./plots
        """
    )
    
    parser.add_argument(
        '--results', '-r',
        type=str,
        help='Path to results JSON file'
    )
    
    parser.add_argument(
        '--results-dir', '-d',
        type=str,
        help='Directory containing multiple result files'
    )
    
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default='./plots',
        help='Output directory for plots'
    )
    
    parser.add_argument(
        '--plots', '-p',
        type=str,
        default='all',
        help='Comma-separated list of plots to generate (all, resolution, pass_at_k, token_usage, heatmap, table)'
    )
    
    parser.add_argument(
        '--compare', '-c',
        action='store_true',
        help='Compare multiple experiments'
    )
    
    parser.add_argument(
        '--format', '-f',
        type=str,
        default='png',
        choices=['png', 'pdf', 'svg'],
        help='Output format'
    )
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Parse plot types
    plot_types = [p.strip() for p in args.plots.split(',')]
    generate_all = 'all' in plot_types
    
    if args.compare and args.results_dir:
        # Compare multiple experiments
        compare_multiple_experiments(Path(args.results_dir), output_dir)
    
    elif args.results:
        # Generate plots for single result file
        results_path = Path(args.results)
        if not results_path.exists():
            logger.error(f"Results file not found: {results_path}")
            sys.exit(1)
        
        results = load_results(results_path)
        
        # Generate requested plots
        if generate_all or 'resolution' in plot_types:
            plot_resolution_comparison(
                results,
                output_dir / f'resolution_comparison.{args.format}'
            )
        
        if generate_all or 'pass_at_k' in plot_types:
            plot_pass_at_k(
                results,
                output_dir / f'pass_at_k.{args.format}'
            )
        
        if generate_all or 'token_usage' in plot_types:
            plot_token_usage(
                results,
                output_dir / f'token_usage.{args.format}'
            )
        
        if generate_all or 'heatmap' in plot_types:
            plot_heatmap(
                results,
                output_dir / f'performance_heatmap.{args.format}'
            )
        
        if generate_all or 'table' in plot_types:
            plot_comparison_table(
                results,
                output_dir / f'comparison_table.{args.format}'
            )
        
        if generate_all or 'by_repo' in plot_types:
            plot_resolution_by_repo(
                results,
                output_dir / f'resolution_by_repo.{args.format}'
            )
        
        logger.info(f"All plots saved to {output_dir}")
    
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()
