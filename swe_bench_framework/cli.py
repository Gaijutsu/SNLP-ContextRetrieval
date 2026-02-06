#!/usr/bin/env python3
"""
Command-line interface for the SWE-bench Comparison Framework.

This module provides a CLI for running experiments, comparing methods,
evaluating predictions, building RAG indexes, and generating reports.

Usage:
    python -m swe_bench_framework.cli run --config config.yaml
    python -m swe_bench_framework.cli compare --configs config1.yaml config2.yaml
    python -m swe_bench_framework.cli evaluate --predictions predictions.json
    python -m swe_bench_framework.cli index --repo-path /path/to/repo
    python -m swe_bench_framework.cli report --results results.json
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Optional, List
from datetime import datetime

import click
import yaml

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from swe_bench_framework import (
    ExperimentOrchestrator,
    ExperimentBuilder,
    OrchestratorConfig,
    SWEBenchLiteLoader,
    SWEBenchVerifiedLoader,
    SWEBenchFullLoader,
    SWEBenchEvaluator,
    ReportGenerator,
)
from swe_bench_framework.config.loader import ConfigLoader
from swe_bench_framework.utils.repo_manager import RepoManager
from swe_bench_framework.patch_generators.direct_generator import DirectPatchGenerator
from swe_bench_framework.patch_generators.iterative_generator import IterativePatchGenerator


def _register_method_from_config(orchestrator, method_name: str, method_config: dict) -> None:
    """Register a method from configuration.
    
    Args:
        orchestrator: The experiment orchestrator
        method_name: Name of the method
        method_config: Configuration for the method
    """
    import logging
    logger = logging.getLogger(__name__)
    
    method_type = method_config.get('type', 'agentic')
    model = method_config.get('model', 'gpt-5-mini')
    
    # model_config may be nested inside 'config' (from ConfigLoader) or at top level
    config = method_config.get('config', {})
    model_config = config.get('model_config', {}) if isinstance(config, dict) else {}
    if not model_config and 'model_config' in method_config:
        model_config = method_config['model_config']
    
    # Build LLM config
    llm_config = {
        'model': model,
        'temperature': model_config.get('temperature', 1.0),
        'max_tokens': model_config.get('max_tokens', 2048),
    }
    
    logger.info(f"[DEBUG] llm_config: {llm_config}")
    
    # Create generator based on type
    if method_type == 'agentic':
        # Agentic methods use iterative generator
        generator_config = {
            'max_iterations': method_config.get('agentic_config', {}).get('max_iterations', 5),
            'feedback_threshold': 0.7,
        }
        generator = IterativePatchGenerator(llm_config, generator_config)
    elif method_type == 'rag':
        # RAG methods use direct generator
        generator_config = {
            'max_retries': 1,
            'validate_syntax': True,
        }
        generator = DirectPatchGenerator(llm_config, generator_config)
    else:
        # Default to direct generator
        generator_config = {
            'max_retries': 1,
            'validate_syntax': True,
        }
        generator = DirectPatchGenerator(llm_config, generator_config)
    
    # Context gatherer - simplified for now
    context_gatherer = None
    
    # Register with orchestrator
    orchestrator.register_method(
        name=method_name,
        patch_generator=generator,
        context_gatherer=context_gatherer,
        method_config=method_config
    )


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@click.group()
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose output')
@click.option('--quiet', '-q', is_flag=True, help='Suppress non-error output')
@click.pass_context
def cli(ctx: click.Context, verbose: bool, quiet: bool) -> None:
    """
    SWE-bench Comparison Framework CLI.
    
    A tool for comparing agentic exploration vs RAG methods
    for automated software patching on SWE-bench.
    
    Use --help with any command for more information.
    """
    # Ensure context object exists
    ctx.ensure_object(dict)
    
    # Set log level
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        ctx.obj['log_level'] = 'DEBUG'
    elif quiet:
        logging.getLogger().setLevel(logging.WARNING)
        ctx.obj['log_level'] = 'WARNING'
    else:
        ctx.obj['log_level'] = 'INFO'


@cli.command()
@click.option(
    '--config', '-c',
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help='Path to configuration file (YAML or JSON)'
)
@click.option(
    '--output-dir', '-o',
    type=click.Path(path_type=Path),
    help='Output directory (overrides config)'
)
@click.option(
    '--name', '-n',
    type=str,
    help='Experiment name (overrides config)'
)
@click.option(
    '--instances',
    type=str,
    help='Comma-separated list of instance IDs to run (e.g., "django-1234,flask-5678")'
)
@click.option(
    '--dry-run',
    is_flag=True,
    help='Show what would be run without executing'
)
@click.pass_context
def run(
    ctx: click.Context,
    config: Path,
    output_dir: Optional[Path],
    name: Optional[str],
    instances: Optional[str],
    dry_run: bool
) -> None:
    """
    Run an experiment from a configuration file.
    
    Examples:
        \b
        # Run experiment from config
        python -m swe_bench_framework.cli run --config configs/basic.yaml
        
        \b
        # Run with custom output directory
        python -m swe_bench_framework.cli run --config config.yaml --output-dir ./results
        
        \b
        # Run on specific instances only
        python -m swe_bench_framework.cli run --config config.yaml --instances "django-1234,flask-5678"
    """
    logger.info(f"Loading configuration from {config}")
    
    # Load configuration
    config_loader = ConfigLoader()
    experiment_config = config_loader.load(config)
    
    # Apply overrides
    if name:
        experiment_config['experiment_name'] = name
    if output_dir:
        experiment_config['output_dir'] = str(output_dir)
    if instances:
        instance_ids = [i.strip() for i in instances.split(',')]
        experiment_config['instance_ids'] = instance_ids
    
    # Log configuration
    logger.info(f"Experiment: {experiment_config.get('experiment_name', 'unnamed')}")
    logger.info(f"Output directory: {experiment_config.get('output_dir', './results')}")
    logger.info(f"Methods: {list(experiment_config.get('methods', {}).keys())}")
    
    def log_mem(label):
        try:
            import psutil, os
            mem = psutil.Process(os.getpid()).memory_info().rss / 1024**2
            logger.info(f"[MEMORY] {label}: {mem:.0f} MB")
        except: pass
    
    log_mem("Before dry_run check")
    
    if dry_run:
        click.echo("\nDry run - would execute with configuration:")
        click.echo(json.dumps(experiment_config, indent=2))
        return
    
    # Create orchestrator from config
    log_mem("Before orchestrator creation")
    orchestrator = config_loader.create_orchestrator(experiment_config)
    log_mem("After orchestrator creation")
    
    # Register methods from config
    logger.info("Registering methods...")
    methods_config = experiment_config.get('methods', {})
    if not methods_config:
        logger.warning("No methods configured in the experiment!")
    
    for method_name, method_config in methods_config.items():
        logger.info(f"Setting up method: {method_name}")
        _register_method_from_config(orchestrator, method_name, method_config)
    
    # Create dataset loader
    logger.info("Creating dataset loader...")
    log_mem("Before dataset loader")
    dataset_name = experiment_config.get('dataset', 'lite')
    if dataset_name == 'lite':
        dataset_loader = SWEBenchLiteLoader()
    elif dataset_name == 'verified':
        dataset_loader = SWEBenchVerifiedLoader()
    elif dataset_name == 'full':
        dataset_loader = SWEBenchFullLoader()
    else:
        raise click.BadParameter(f"Unknown dataset: {dataset_name}")
    log_mem("After dataset loader")
    
    # Filter instances if specified
    if 'instance_ids' in experiment_config:
        logger.info("Applying instance_ids filter...")
        dataset_loader = dataset_loader.with_filter(
            instance_ids=experiment_config['instance_ids']
        )
        log_mem("After filter")
    elif 'max_instances' in experiment_config:
        logger.info(f"Applying max_instances filter: {experiment_config['max_instances']}")
        dataset_loader = dataset_loader.with_filter(
            max_instances=experiment_config['max_instances']
        )
        log_mem("After filter")
    
    # Create evaluator
    logger.info("Creating evaluator...")
    log_mem("Before evaluator")
    sandbox_type = experiment_config.get('sandbox', 'local')
    evaluator_config = {
        'use_docker': sandbox_type == 'docker',
        'docker': {},
        'compute_codebleu': False,  # DISABLED - memory intensive
        'compute_localization': False,  # DISABLED - memory intensive
    }
    evaluator = SWEBenchEvaluator(evaluator_config)
    log_mem("After evaluator")
    
    # Create repo provider
    repo_manager = RepoManager(experiment_config.get('repos_dir', './repos'))
    repo_provider = lambda instance: repo_manager.get_repo_path(instance)
    
    # Run experiment
    try:
        results = orchestrator.run(
            dataset_loader=dataset_loader,
            evaluator=evaluator,
            repo_provider=repo_provider
        )
        
        # Print summary
        orchestrator.print_summary()
        
        click.echo(f"\nResults saved to: {orchestrator.output_dir}")
        
    except Exception as e:
        logger.error(f"Experiment failed: {e}", exc_info=True)
        raise click.ClickException(str(e))


@cli.command()
@click.option(
    '--configs', '-c',
    type=click.Path(exists=True, path_type=Path),
    multiple=True,
    required=True,
    help='Configuration files to compare'
)
@click.option(
    '--output-dir', '-o',
    type=click.Path(path_type=Path),
    default='./comparison_results',
    help='Output directory for comparison results'
)
@click.option(
    '--parallel',
    is_flag=True,
    help='Run experiments in parallel'
)
@click.pass_context
def compare(
    ctx: click.Context,
    configs: List[Path],
    output_dir: Path,
    parallel: bool
) -> None:
    """
    Compare multiple methods across different configurations.
    
    This command runs multiple experiments and generates a comparison report.
    
    Examples:
        \b
        # Compare two configurations
        python -m swe_bench_framework.cli compare --configs config1.yaml config2.yaml
        
        \b
        # Compare with custom output
        python -m swe_bench_framework.cli compare -c config1.yaml -c config2.yaml -o ./comparison
    """
    logger.info(f"Comparing {len(configs)} configurations")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_results = []
    
    for config_path in configs:
        logger.info(f"Running experiment from {config_path}")
        
        # Load and run experiment
        config_loader = ConfigLoader()
        experiment_config = config_loader.load(config_path)
        
        # Create unique output directory for this experiment
        exp_name = experiment_config.get('experiment_name', config_path.stem)
        exp_output_dir = output_dir / exp_name
        experiment_config['output_dir'] = str(exp_output_dir)
        
        try:
            orchestrator = config_loader.create_orchestrator(experiment_config)
            
            # Create dataset loader
            dataset_name = experiment_config.get('dataset', 'lite')
            if dataset_name == 'lite':
                dataset_loader = SWEBenchLiteLoader()
            elif dataset_name == 'verified':
                dataset_loader = SWEBenchVerifiedLoader()
            else:
                dataset_loader = SWEBenchFullLoader()
            
            # Create evaluator
            evaluator = SWEBenchEvaluator(
                sandbox_type=experiment_config.get('sandbox', 'local')
            )
            
            # Create repo provider
            repo_manager = RepoManager(experiment_config.get('repos_dir', './repos'))
            repo_provider = lambda instance: repo_manager.get_repo_path(instance)
            
            # Run experiment
            results = orchestrator.run(
                dataset_loader=dataset_loader,
                evaluator=evaluator,
                repo_provider=repo_provider
            )
            
            all_results.append({
                'name': exp_name,
                'config': str(config_path),
                'results': results,
                'summary': orchestrator.get_summary()
            })
            
        except Exception as e:
            logger.error(f"Experiment {exp_name} failed: {e}", exc_info=True)
            all_results.append({
                'name': exp_name,
                'config': str(config_path),
                'error': str(e)
            })
    
    # Generate comparison report
    report_generator = ReportGenerator()
    comparison_report = report_generator.generate_comparison_report(all_results)
    
    # Save comparison report
    report_path = output_dir / 'comparison_report.json'
    with open(report_path, 'w') as f:
        json.dump(comparison_report, f, indent=2)
    
    # Generate markdown report
    md_report_path = output_dir / 'comparison_report.md'
    report_generator.save_report(comparison_report, str(md_report_path), 'markdown')
    
    click.echo(f"\nComparison report saved to: {output_dir}")
    
    # Print summary
    click.echo("\n" + "=" * 60)
    click.echo("Comparison Summary")
    click.echo("=" * 60)
    for result in all_results:
        if 'error' in result:
            click.echo(f"\n{result['name']}: FAILED - {result['error']}")
        else:
            summary = result['summary']
            click.echo(f"\n{result['name']}:")
            for method, method_result in summary.get('results', {}).items():
                click.echo(f"  {method}: {method_result.get('resolution_rate', 0):.2f}% resolution")


@cli.command()
@click.option(
    '--predictions', '-p',
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help='Path to predictions file (JSON)'
)
@click.option(
    '--dataset',
    type=click.Choice(['lite', 'verified', 'full']),
    default='lite',
    help='Dataset to evaluate against'
)
@click.option(
    '--output', '-o',
    type=click.Path(path_type=Path),
    help='Output file for evaluation results'
)
@click.option(
    '--sandbox',
    type=click.Choice(['local', 'docker']),
    default='local',
    help='Sandbox type for evaluation'
)
@click.pass_context
def evaluate(
    ctx: click.Context,
    predictions: Path,
    dataset: str,
    output: Optional[Path],
    sandbox: str
) -> None:
    """
    Evaluate existing predictions against SWE-bench.
    
    This command evaluates a predictions file and generates metrics.
    
    Examples:
        \b
        # Evaluate predictions
        python -m swe_bench_framework.cli evaluate --predictions preds.json
        
        \b
        # Evaluate with specific dataset
        python -m swe_bench_framework.cli evaluate -p preds.json --dataset verified
    """
    logger.info(f"Evaluating predictions from {predictions}")
    
    # Load predictions
    with open(predictions, 'r') as f:
        predictions_data = json.load(f)
    
    # Load dataset
    if dataset == 'lite':
        dataset_loader = SWEBenchLiteLoader()
    elif dataset == 'verified':
        dataset_loader = SWEBenchVerifiedLoader()
    else:
        dataset_loader = SWEBenchFullLoader()
    
    # Create evaluator
    evaluator = SWEBenchEvaluator(sandbox_type=sandbox)
    
    # Evaluate predictions
    try:
        results = evaluator.evaluate_predictions(predictions_data, dataset_loader)
        
        # Calculate metrics
        total = len(results)
        resolved = sum(1 for r in results if r.resolved)
        resolution_rate = (resolved / total * 100) if total > 0 else 0
        
        click.echo("\n" + "=" * 60)
        click.echo("Evaluation Results")
        click.echo("=" * 60)
        click.echo(f"Total instances: {total}")
        click.echo(f"Resolved: {resolved}")
        click.echo(f"Resolution rate: {resolution_rate:.2f}%")
        
        # Save results if output specified
        if output:
            output_results = {
                'predictions_file': str(predictions),
                'dataset': dataset,
                'total': total,
                'resolved': resolved,
                'resolution_rate': resolution_rate,
                'results': [r.to_dict() for r in results]
            }
            
            with open(output, 'w') as f:
                json.dump(output_results, f, indent=2)
            
            click.echo(f"\nResults saved to: {output}")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=True)
        raise click.ClickException(str(e))


@cli.command()
@click.option(
    '--repo-path', '-r',
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help='Path to repository to index'
)
@click.option(
    '--output-dir', '-o',
    type=click.Path(path_type=Path),
    help='Output directory for index'
)
@click.option(
    '--index-type',
    type=click.Choice(['bm25', 'dense', 'hybrid']),
    default='hybrid',
    help='Type of index to build'
)
@click.option(
    '--embedding-model',
    type=str,
    default='sentence-transformers/all-MiniLM-L6-v2',
    help='Embedding model for dense index'
)
@click.option(
    '--chunk-size',
    type=int,
    default=1000,
    help='Size of code chunks'
)
@click.option(
    '--chunk-overlap',
    type=int,
    default=200,
    help='Overlap between chunks'
)
@click.pass_context
def index(
    ctx: click.Context,
    repo_path: Path,
    output_dir: Optional[Path],
    index_type: str,
    embedding_model: str,
    chunk_size: int,
    chunk_overlap: int
) -> None:
    """
    Build RAG index for a repository.
    
    This command builds a searchable index for RAG-based methods.
    
    Examples:
        \b
        # Build hybrid index
        python -m swe_bench_framework.cli index --repo-path /path/to/repo
        
        \b
        # Build dense index with custom model
        python -m swe_bench_framework.cli index -r /path/to/repo --index-type dense \\
            --embedding-model sentence-transformers/all-mpnet-base-v2
    """
    logger.info(f"Building {index_type} index for {repo_path}")
    
    # Set default output directory
    if output_dir is None:
        output_dir = Path('./indexes') / repo_path.name / index_type
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Import index builders
        from swe_bench_framework.context_gatherers.index_builder import (
            BM25IndexBuilder,
            DenseIndexBuilder,
            HybridIndexBuilder
        )
        
        # Create appropriate builder
        if index_type == 'bm25':
            builder = BM25IndexBuilder(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        elif index_type == 'dense':
            builder = DenseIndexBuilder(
                model_name=embedding_model,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
        else:  # hybrid
            builder = HybridIndexBuilder(
                model_name=embedding_model,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
        
        # Build index
        click.echo(f"Building {index_type} index...")
        index_path = builder.build(repo_path, output_dir)
        
        click.echo(f"\nIndex built successfully!")
        click.echo(f"Index saved to: {index_path}")
        
        # Print index stats
        stats = builder.get_stats()
        click.echo("\nIndex Statistics:")
        for key, value in stats.items():
            click.echo(f"  {key}: {value}")
        
    except Exception as e:
        logger.error(f"Index building failed: {e}", exc_info=True)
        raise click.ClickException(str(e))


@cli.command()
@click.option(
    '--results', '-r',
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help='Path to results file or directory'
)
@click.option(
    '--output-dir', '-o',
    type=click.Path(path_type=Path),
    default='./reports',
    help='Output directory for reports'
)
@click.option(
    '--formats', '-f',
    type=str,
    default='json,markdown,html',
    help='Comma-separated list of output formats'
)
@click.option(
    '--template',
    type=click.Path(exists=True, path_type=Path),
    help='Custom report template file'
)
@click.pass_context
def report(
    ctx: click.Context,
    results: Path,
    output_dir: Path,
    formats: str,
    template: Optional[Path]
) -> None:
    """
    Generate report from experiment results.
    
    This command generates formatted reports from experiment results.
    
    Examples:
        \b
        # Generate report from results
        python -m swe_bench_framework.cli report --results results.json
        
        \b
        # Generate multiple formats
        python -m swe_bench_framework.cli report -r results.json -f json,markdown,html
    """
    logger.info(f"Generating report from {results}")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Parse formats
    format_list = [f.strip() for f in formats.split(',')]
    
    try:
        # Load results
        with open(results, 'r') as f:
            results_data = json.load(f)
        
        # Create report generator
        report_generator = ReportGenerator()
        
        # Load custom template if provided
        if template:
            report_generator.load_template(template)
        
        # Generate report
        report_data = report_generator.generate_comparison_report(results_data)
        
        # Save in each format
        for fmt in format_list:
            output_path = output_dir / f"report.{fmt}"
            report_generator.save_report(report_data, str(output_path), fmt)
            click.echo(f"Generated {fmt} report: {output_path}")
        
        click.echo(f"\nReports saved to: {output_dir}")
        
    except Exception as e:
        logger.error(f"Report generation failed: {e}", exc_info=True)
        raise click.ClickException(str(e))


@cli.command()
@click.option(
    '--output', '-o',
    type=click.Path(path_type=Path),
    default='config.yaml',
    help='Output file for generated config'
)
@click.option(
    '--template',
    type=click.Choice(['minimal', 'basic', 'full', 'rag', 'agentic']),
    default='basic',
    help='Configuration template to use'
)
@click.pass_context
def init(
    ctx: click.Context,
    output: Path,
    template: str
) -> None:
    """
    Initialize a new configuration file.
    
    This command creates a starter configuration file.
    
    Examples:
        \b
        # Create basic config
        python -m swe_bench_framework.cli init
        
        \b
        # Create minimal config
        python -m swe_bench_framework.cli init --template minimal -o minimal.yaml
    """
    templates = {
        'minimal': {
            'experiment_name': 'my_experiment',
            'output_dir': './results',
            'dataset': 'lite',
            'methods': {
                'agentless': {
                    'type': 'agentic',
                    'strategy': 'agentless',
                    'model': 'gpt-5-mini'
                }
            },
            'sandbox': 'local'
        },
        'basic': {
            'experiment_name': 'basic_comparison',
            'output_dir': './results',
            'dataset': 'lite',
            'methods': {
                'agentless': {
                    'type': 'agentic',
                    'strategy': 'agentless',
                    'model': 'gpt-5-mini'
                },
                'hybrid_rag': {
                    'type': 'rag',
                    'retrieval': 'hybrid',
                    'model': 'gpt-5-mini',
                    'top_k': 10
                }
            },
            'sandbox': 'local',
            'max_workers': 1
        },
        'full': {
            'experiment_name': 'full_comparison',
            'output_dir': './results',
            'dataset': 'verified',
            'methods': {
                'autocoderover': {
                    'type': 'agentic',
                    'strategy': 'autocoderover',
                    'model': 'gpt-5-mini'
                },
                'swe_agent': {
                    'type': 'agentic',
                    'strategy': 'swe_agent',
                    'model': 'gpt-5-mini'
                },
                'agentless': {
                    'type': 'agentic',
                    'strategy': 'agentless',
                    'model': 'gpt-5-mini'
                },
                'bm25_rag': {
                    'type': 'rag',
                    'retrieval': 'bm25',
                    'model': 'gpt-5-mini',
                    'top_k': 10
                },
                'dense_rag': {
                    'type': 'rag',
                    'retrieval': 'dense',
                    'model': 'gpt-5-mini',
                    'top_k': 10
                },
                'hybrid_rag': {
                    'type': 'rag',
                    'retrieval': 'hybrid',
                    'model': 'gpt-5-mini',
                    'top_k': 10
                }
            },
            'sandbox': 'docker',
            'max_workers': 4
        },
        'rag': {
            'experiment_name': 'rag_comparison',
            'output_dir': './results',
            'dataset': 'lite',
            'methods': {
                'bm25_rag': {
                    'type': 'rag',
                    'retrieval': 'bm25',
                    'model': 'gpt-5-mini',
                    'top_k': 10
                },
                'dense_rag': {
                    'type': 'rag',
                    'retrieval': 'dense',
                    'model': 'gpt-5-mini',
                    'top_k': 10
                },
                'hybrid_rag': {
                    'type': 'rag',
                    'retrieval': 'hybrid',
                    'model': 'gpt-5-mini',
                    'top_k': 10
                }
            },
            'sandbox': 'local'
        },
        'agentic': {
            'experiment_name': 'agentic_comparison',
            'output_dir': './results',
            'dataset': 'lite',
            'methods': {
                'autocoderover': {
                    'type': 'agentic',
                    'strategy': 'autocoderover',
                    'model': 'gpt-5-mini'
                },
                'swe_agent': {
                    'type': 'agentic',
                    'strategy': 'swe_agent',
                    'model': 'gpt-5-mini'
                },
                'agentless': {
                    'type': 'agentic',
                    'strategy': 'agentless',
                    'model': 'gpt-5-mini'
                }
            },
            'sandbox': 'local'
        }
    }
    
    config = templates.get(template, templates['basic'])
    
    # Write config file
    with open(output, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    click.echo(f"Created {template} configuration: {output}")


@cli.command()
@click.pass_context
def version(ctx: click.Context) -> None:
    """Show version information."""
    from swe_bench_framework import __version__
    click.echo(f"SWE-bench Comparison Framework v{__version__}")


if __name__ == '__main__':
    cli()
