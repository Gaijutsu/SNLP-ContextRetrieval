#!/usr/bin/env python3
"""
Simple script to run a SWE-bench comparison experiment.

This script provides a straightforward way to run experiments
without using the full CLI interface.

Usage:
    python run_experiment.py --config configs/basic_comparison.yaml
    python run_experiment.py --config configs/minimal.yaml --output-dir ./my_results
    python run_experiment.py --config config.yaml --instances django-1234,flask-5678
"""

import argparse
import logging
import sys
from pathlib import Path

# Add framework to path
sys.path.insert(0, str(Path(__file__).parent / 'swe_bench_framework'))

from swe_bench_framework.config.loader import ConfigLoader
from swe_bench_framework import (
    SWEBenchLiteLoader,
    SWEBenchVerifiedLoader,
    SWEBenchFullLoader,
    SWEBenchEvaluator,
)
from swe_bench_framework.utils.repo_manager import RepoManager


def setup_logging(log_level: str = 'INFO') -> None:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
        ]
    )


def main():
    """Main entry point for running experiments."""
    parser = argparse.ArgumentParser(
        description='Run SWE-bench comparison experiment',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run from config file
  python run_experiment.py --config configs/basic_comparison.yaml
  
  # Run with custom output directory
  python run_experiment.py --config config.yaml --output-dir ./results
  
  # Run on specific instances only
  python run_experiment.py --config config.yaml --instances django-1234,flask-5678
  
  # Run with verbose logging
  python run_experiment.py --config config.yaml --verbose
        """
    )
    
    parser.add_argument(
        '--config', '-c',
        type=str,
        required=True,
        help='Path to configuration file (YAML or JSON)'
    )
    
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        help='Output directory (overrides config)'
    )
    
    parser.add_argument(
        '--name', '-n',
        type=str,
        help='Experiment name (overrides config)'
    )
    
    parser.add_argument(
        '--instances',
        type=str,
        help='Comma-separated list of instance IDs to run'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be run without executing'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = 'DEBUG' if args.verbose else 'INFO'
    setup_logging(log_level)
    
    logger = logging.getLogger(__name__)
    logger.info(f"Starting experiment runner")
    logger.info(f"Config file: {args.config}")
    
    # Load configuration
    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        sys.exit(1)
    
    config_loader = ConfigLoader()
    
    try:
        experiment_config = config_loader.load(config_path)
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        sys.exit(1)
    
    # Apply command-line overrides
    if args.name:
        experiment_config['experiment_name'] = args.name
        logger.info(f"Experiment name (override): {args.name}")
    
    if args.output_dir:
        experiment_config['output_dir'] = args.output_dir
        logger.info(f"Output directory (override): {args.output_dir}")
    
    if args.instances:
        instance_ids = [i.strip() for i in args.instances.split(',')]
        experiment_config['instance_ids'] = instance_ids
        logger.info(f"Running on instances: {instance_ids}")
    
    # Log configuration summary
    logger.info(f"Experiment: {experiment_config.get('experiment_name', 'unnamed')}")
    logger.info(f"Output directory: {experiment_config.get('output_dir', './results')}")
    logger.info(f"Dataset: {experiment_config.get('dataset', 'lite')}")
    logger.info(f"Methods: {list(experiment_config.get('methods', {}).keys())}")
    
    # Dry run
    if args.dry_run:
        import json
        print("\nDry run - would execute with configuration:")
        print(json.dumps(experiment_config, indent=2))
        return
    
    # Create orchestrator
    try:
        orchestrator = config_loader.create_orchestrator(experiment_config)
    except Exception as e:
        logger.error(f"Failed to create orchestrator: {e}")
        sys.exit(1)
    
    # Create dataset loader
    dataset_name = experiment_config.get('dataset', 'lite')
    logger.info(f"Loading dataset: {dataset_name}")
    
    if dataset_name == 'lite':
        dataset_loader = SWEBenchLiteLoader()
    elif dataset_name == 'verified':
        dataset_loader = SWEBenchVerifiedLoader()
    elif dataset_name == 'full':
        dataset_loader = SWEBenchFullLoader()
    else:
        logger.error(f"Unknown dataset: {dataset_name}")
        sys.exit(1)
    
    # Filter instances if specified
    if 'instance_ids' in experiment_config:
        instance_ids = experiment_config['instance_ids']
        logger.info(f"Filtering to {len(instance_ids)} instances")
        # Note: The actual filtering would depend on the dataset loader implementation
    
    # Create evaluator
    sandbox_type = experiment_config.get('sandbox', 'local')
    logger.info(f"Using sandbox: {sandbox_type}")
    evaluator = SWEBenchEvaluator(sandbox_type=sandbox_type)
    
    # Create repo provider
    repos_dir = experiment_config.get('repos_dir', './repos')
    repo_manager = RepoManager(repos_dir)
    repo_provider = lambda instance: repo_manager.get_repo_path(instance)
    
    # Run experiment
    logger.info("Starting experiment...")
    
    try:
        results = orchestrator.run(
            dataset_loader=dataset_loader,
            evaluator=evaluator,
            repo_provider=repo_provider
        )
        
        # Print summary
        orchestrator.print_summary()
        
        logger.info(f"Experiment completed successfully!")
        logger.info(f"Results saved to: {orchestrator.output_dir}")
        
    except KeyboardInterrupt:
        logger.warning("Experiment interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Experiment failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
