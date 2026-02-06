#!/usr/bin/env python3
"""
Minimal Experiment Script

This script demonstrates the minimal code needed to run a comparison
experiment between two context gathering methods on SWE-bench.

Usage:
    python minimal_experiment.py

Requirements:
    - swe_bench_framework installed
    - OPENAI_API_KEY environment variable set
    - Docker installed and running
"""

import os
import sys
from pathlib import Path

# Add the framework to path (if not installed)
# sys.path.insert(0, '/path/to/swe_bench_framework')

from swe_bench_framework import ExperimentOrchestrator
from swe_bench_framework.config import ExperimentConfig


def main():
    """Run a minimal comparison experiment."""
    
    # Check API key
    if not os.getenv('OPENAI_API_KEY'):
        print("Error: OPENAI_API_KEY environment variable not set")
        print("Please set it with: export OPENAI_API_KEY='your-key'")
        sys.exit(1)
    
    # Create output directory
    output_dir = Path('./results/minimal_experiment')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Define configuration programmatically
    config_dict = {
        'experiment': {
            'name': 'minimal_comparison',
            'description': 'Minimal comparison of BM25 vs AutoCodeRover',
            'output_dir': str(output_dir),
            'random_seed': 42
        },
        'dataset': {
            'name': 'swe-bench-lite',
            'split': 'test',
            'filter': {
                'max_instances': 10  # Small subset for quick testing
            }
        },
        'llm': {
            'provider': 'openai',
            'model': 'gpt-4-turbo-preview',
            'temperature': 0.0,
            'max_tokens': 4096,
            'api_key': '${OPENAI_API_KEY}'
        },
        'methods': [
            {
                'name': 'bm25',
                'type': 'rag',
                'enabled': True,
                'config': {
                    'top_k': 20
                }
            },
            {
                'name': 'autocoderover',
                'type': 'agentic',
                'enabled': True,
                'config': {
                    'max_iterations': 20
                }
            }
        ],
        'patch_generation': {
            'strategy': 'direct',
            'max_attempts': 3
        },
        'evaluation': {
            'sandbox': {
                'type': 'docker',
                'timeout': 300
            },
            'metrics': [
                'resolution_rate',
                'localization_accuracy',
                'token_usage'
            ]
        },
        'logging': {
            'level': 'INFO',
            'format': 'structured',
            'outputs': [
                {'type': 'file', 'path': str(output_dir / 'logs')},
                {'type': 'stdout'}
            ]
        }
    }
    
    # Save configuration to file
    import yaml
    config_path = output_dir / 'config.yaml'
    with open(config_path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False)
    print(f"Configuration saved to: {config_path}")
    
    # Load configuration
    config = ExperimentConfig.from_yaml(str(config_path))
    print(f"Loaded configuration: {config.experiment.name}")
    
    # Create orchestrator
    orchestrator = ExperimentOrchestrator(config)
    print("Experiment orchestrator created")
    
    # Run experiment
    print("\n" + "="*60)
    print("RUNNING EXPERIMENT")
    print("="*60 + "\n")
    
    try:
        report = orchestrator.run()
    except KeyboardInterrupt:
        print("\n\nExperiment interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nExperiment failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Print results
    print("\n" + "="*60)
    print("EXPERIMENT RESULTS")
    print("="*60)
    
    for method_name, metrics in report.summary.items():
        print(f"\n{method_name}:")
        print(f"  Resolution Rate: {metrics.get('resolution_rate', 0):.2%}")
        print(f"  Avg Recall@5: {metrics.get('avg_recall@5', 0):.2%}")
        print(f"  Avg Tokens: {metrics.get('avg_tokens', 0):.0f}")
        print(f"  Total Cost: ${metrics.get('total_cost', 0):.2f}")
    
    # Save detailed report
    report_path = output_dir / 'report.json'
    report.save(str(report_path))
    print(f"\nDetailed report saved to: {report_path}")
    
    # Generate comparison visualization
    try:
        figures_dir = output_dir / 'figures'
        figures_dir.mkdir(exist_ok=True)
        report.generate_visualizations(str(figures_dir))
        print(f"Visualizations saved to: {figures_dir}")
    except Exception as e:
        print(f"Could not generate visualizations: {e}")
    
    print("\n" + "="*60)
    print("EXPERIMENT COMPLETE")
    print("="*60)
    
    return report


if __name__ == '__main__':
    main()
