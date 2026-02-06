#!/usr/bin/env python3
"""
Cost Estimation Tool for SWE-bench Comparison Framework.

This script estimates the cost of running experiments based on:
- Dataset size (number of instances)
- Model pricing
- Estimated tokens per instance

Usage:
    python scripts/estimate_cost.py --config configs/full_comparison.yaml
    python scripts/estimate_cost.py --dataset verified --model gpt-5-mini --instances 500
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Any, Optional

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from swe_bench_framework.evaluation.metrics.token_usage import TokenUsageMetric


def get_dataset_size(dataset_name: str) -> int:
    """Get the number of instances in a dataset."""
    sizes = {
        'lite': 300,
        'verified': 500,
        'full': 2294,
    }
    return sizes.get(dataset_name.lower(), 0)


def get_model_pricing(model_name: str) -> Dict[str, float]:
    """Get pricing for a model."""
    pricing = TokenUsageMetric.DEFAULT_PRICING
    
    # Find matching pricing
    for key, price in pricing.items():
        if key in model_name.lower():
            return price
    
    # Default pricing
    return {'prompt': 0.01, 'completion': 0.03}


def estimate_cost(
    num_instances: int,
    model: str,
    avg_prompt_tokens: int = 2000,
    avg_completion_tokens: int = 1000,
    num_methods: int = 1,
    iterations_per_instance: int = 1,
) -> Dict[str, Any]:
    """
    Estimate the cost of running experiments.
    
    Args:
        num_instances: Number of instances to process
        model: Model name
        avg_prompt_tokens: Average prompt tokens per instance
        avg_completion_tokens: Average completion tokens per instance
        num_methods: Number of methods being compared
        iterations_per_instance: Average iterations per instance (for iterative methods)
    
    Returns:
        Cost estimation dictionary
    """
    pricing = get_model_pricing(model)
    
    # Calculate per-instance cost
    prompt_cost_per_1k = pricing['prompt']
    completion_cost_per_1k = pricing['completion']
    
    prompt_cost = (avg_prompt_tokens / 1000) * prompt_cost_per_1k
    completion_cost = (avg_completion_tokens / 1000) * completion_cost_per_1k
    cost_per_instance = (prompt_cost + completion_cost) * iterations_per_instance
    
    # Calculate totals
    total_instances = num_instances * num_methods
    total_cost = cost_per_instance * num_instances * num_methods
    total_prompt_tokens = avg_prompt_tokens * total_instances * iterations_per_instance
    total_completion_tokens = avg_completion_tokens * total_instances * iterations_per_instance
    
    return {
        'model': model,
        'pricing': pricing,
        'num_instances': num_instances,
        'num_methods': num_methods,
        'total_instances': total_instances,
        'iterations_per_instance': iterations_per_instance,
        'avg_prompt_tokens': avg_prompt_tokens,
        'avg_completion_tokens': avg_completion_tokens,
        'cost_per_instance': cost_per_instance,
        'total_cost': total_cost,
        'total_prompt_tokens': total_prompt_tokens,
        'total_completion_tokens': total_completion_tokens,
        'total_tokens': total_prompt_tokens + total_completion_tokens,
    }


def format_currency(amount: float) -> str:
    """Format amount as currency."""
    if amount < 0.01:
        return f"${amount:.6f}"
    elif amount < 1:
        return f"${amount:.4f}"
    else:
        return f"${amount:.2f}"


def print_estimate(estimate: Dict[str, Any]):
    """Print cost estimate in a formatted way."""
    print("\n" + "=" * 70)
    print("COST ESTIMATION REPORT")
    print("=" * 70)
    
    print(f"\n[DATASET & METHODS]")
    print(f"  Model: {estimate['model']}")
    print(f"  Instances per method: {estimate['num_instances']}")
    print(f"  Number of methods: {estimate['num_methods']}")
    print(f"  Total instances: {estimate['total_instances']}")
    print(f"  Iterations per instance: {estimate['iterations_per_instance']}")
    
    print(f"\n[PRICING] (per 1K tokens):")
    print(f"  Prompt: {format_currency(estimate['pricing']['prompt'])}")
    print(f"  Completion: {format_currency(estimate['pricing']['completion'])}")
    
    print(f"\n[TOKEN USAGE] (avg per instance):")
    print(f"  Prompt: {estimate['avg_prompt_tokens']:,} tokens")
    print(f"  Completion: {estimate['avg_completion_tokens']:,} tokens")
    print(f"  Total: {estimate['avg_prompt_tokens'] + estimate['avg_completion_tokens']:,} tokens")
    
    print(f"\n[COST BREAKDOWN]")
    print(f"  Cost per instance: {format_currency(estimate['cost_per_instance'])}")
    print(f"  -" * 35)
    print(f"  TOTAL ESTIMATED COST: {format_currency(estimate['total_cost'])}")
    print(f"  -" * 35)
    
    print(f"\n[TOTAL TOKEN USAGE]")
    print(f"  Prompt: {estimate['total_prompt_tokens']:,} tokens")
    print(f"  Completion: {estimate['total_completion_tokens']:,} tokens")
    print(f"  Total: {estimate['total_tokens']:,} tokens")
    
    # Add warnings for high costs
    if estimate['total_cost'] > 100:
        print("\n[!] WARNING: Estimated cost exceeds $100!")
    elif estimate['total_cost'] > 50:
        print("\n[!] WARNING: Estimated cost exceeds $50!")
    elif estimate['total_cost'] > 10:
        print("\n[!] NOTE: Estimated cost exceeds $10.")
    
    print("\n" + "=" * 70)
    print("[TIPS] To reduce cost:")
    print("  * Use a smaller dataset (lite=300, verified=500, full=2294)")
    print("  * Use gpt-5-mini instead of gpt-5 or gpt-5-pro")
    print("  * Reduce max_iterations for agentic methods")
    print("  * Use instance_ids to test on specific instances first")
    print("=" * 70 + "\n")


def load_config(config_path: str) -> Optional[Dict[str, Any]]:
    """Load configuration from YAML file."""
    try:
        import yaml
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"Error loading config: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description='Estimate cost of running SWE-bench experiments',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Estimate cost from config file
    python scripts/estimate_cost.py --config configs/full_comparison.yaml
    
    # Estimate cost for specific dataset and model
    python scripts/estimate_cost.py --dataset verified --model gpt-5-mini
    
    # Estimate with custom parameters
    python scripts/estimate_cost.py --dataset lite --model gpt-5-pro \\
        --prompt-tokens 3000 --completion-tokens 1500 --methods 3
        """
    )
    
    parser.add_argument('--config', '-c', type=str,
                        help='Path to configuration file')
    parser.add_argument('--dataset', '-d', type=str, 
                        choices=['lite', 'verified', 'full'],
                        help='Dataset name')
    parser.add_argument('--model', '-m', type=str, default='gpt-5-mini',
                        help='Model name (default: gpt-5-mini)')
    parser.add_argument('--instances', '-n', type=int,
                        help='Number of instances (overrides dataset)')
    parser.add_argument('--methods', type=int, default=1,
                        help='Number of methods to compare (default: 1)')
    parser.add_argument('--iterations', '-i', type=int, default=1,
                        help='Average iterations per instance (default: 1)')
    parser.add_argument('--prompt-tokens', '-p', type=int, default=2000,
                        help='Average prompt tokens per instance (default: 2000)')
    parser.add_argument('--completion-tokens', '-t', type=int, default=1000,
                        help='Average completion tokens per instance (default: 1000)')
    
    args = parser.parse_args()
    
    # Determine number of instances
    num_instances = None
    
    if args.config:
        config = load_config(args.config)
        if config:
            # Get dataset from config
            dataset = config.get('dataset', 'lite')
            num_instances = get_dataset_size(dataset)
            
            # Count methods
            methods_config = config.get('methods', {})
            args.methods = len(methods_config) if methods_config else 1
            
            # Try to get model from config
            for method_name, method_cfg in methods_config.items():
                if 'model' in method_cfg:
                    args.model = method_cfg['model']
                    break
            
            print(f"Loaded config: {args.config}")
            print(f"  Dataset: {dataset} ({num_instances} instances)")
            print(f"  Methods: {args.methods}")
            print(f"  Model: {args.model}")
    
    if args.instances:
        num_instances = args.instances
    elif args.dataset and not num_instances:
        num_instances = get_dataset_size(args.dataset)
    
    if not num_instances:
        print("Error: Please specify --dataset, --instances, or --config")
        sys.exit(1)
    
    # Generate estimate
    estimate = estimate_cost(
        num_instances=num_instances,
        model=args.model,
        avg_prompt_tokens=args.prompt_tokens,
        avg_completion_tokens=args.completion_tokens,
        num_methods=args.methods,
        iterations_per_instance=args.iterations,
    )
    
    print_estimate(estimate)


if __name__ == '__main__':
    main()
