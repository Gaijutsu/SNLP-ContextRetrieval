"""
Experiment orchestrator for the SWE-bench comparison framework.

This module provides the main entry point for running experiments that compare
multiple methods on the SWE-bench dataset.
"""

import json
import time
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass, field

from .runner import ExperimentRunner, ExperimentConfig, ExperimentResult
from ..dataset.loader import DatasetLoader
from ..patch_generators.base import PatchGenerator
from ..evaluation.base import Evaluator
from ..evaluation.report_generator import ReportGenerator


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class OrchestratorConfig:
    """Configuration for the experiment orchestrator."""
    experiment_name: str
    output_dir: str
    max_workers_per_method: int = 1
    save_intermediate: bool = True
    generate_report: bool = True
    report_formats: List[str] = field(default_factory=lambda: ['json', 'markdown'])
    log_level: str = 'INFO'
    tracking: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MethodConfig:
    """Configuration for a single method."""
    name: str
    patch_generator: PatchGenerator
    context_gatherer: Optional[Any] = None
    config: Dict[str, Any] = field(default_factory=dict)


class ExperimentOrchestrator:
    """
    Main orchestrator for running comparison experiments.
    
    This class manages the full experiment lifecycle:
    1. Load dataset
    2. Run multiple methods
    3. Collect results
    4. Generate comparison reports
    
    Example:
        orchestrator = ExperimentOrchestrator(config)
        
        # Register methods
        orchestrator.register_method("Method A", method_a_generator, method_a_gatherer)
        orchestrator.register_method("Method B", method_b_generator, method_b_gatherer)
        
        # Run experiment
        results = orchestrator.run(
            dataset_loader=loader,
            evaluator=evaluator,
            repo_provider=repo_provider
        )
        
        # Generate report
        report = orchestrator.generate_report()
    """
    
    def __init__(self, config: OrchestratorConfig):
        """
        Initialize the experiment orchestrator.
        
        Args:
            config: Orchestrator configuration
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(getattr(logging, config.log_level))
        
        # Registered methods
        self._methods: Dict[str, MethodConfig] = {}
        
        # Results storage
        self._results: Dict[str, ExperimentResult] = {}
        
        # Report generator
        self._report_generator = ReportGenerator()
        
        # Create output directory
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Experiment tracking
        self._tracking_enabled = False
        self._tracking_backend = None
        self._tracking_run = None
    
    def _init_tracking(self, tracking_config: Dict[str, Any]) -> None:
        """Initialize experiment tracking (MLflow or W&B).
        
        Args:
            tracking_config: Tracking configuration from logging config
        """
        self.logger.info(f"[TRACKING] Initializing with config: {tracking_config}")
        
        if not tracking_config:
            self.logger.info("[TRACKING] No tracking config provided")
            return
        
        if not tracking_config.get('enabled', False):
            self.logger.info("[TRACKING] Tracking is disabled")
            return
        
        backend = tracking_config.get('backend', 'mlflow')
        experiment_name = tracking_config.get('experiment_name') or self.config.experiment_name
        
        self.logger.info(f"[TRACKING] Backend: {backend}, Experiment: {experiment_name}")
        
        try:
            if backend == 'mlflow':
                import mlflow
                uri = tracking_config.get('uri', 'http://localhost:5000')
                self.logger.info(f"[TRACKING] Connecting to MLflow at {uri}")
                mlflow.set_tracking_uri(uri)
                mlflow.set_experiment(experiment_name)
                self._tracking_run = mlflow.start_run(run_name=self.config.experiment_name)
                self._tracking_backend = 'mlflow'
                self._tracking_enabled = True
                self.logger.info(f"[TRACKING] MLflow tracking initialized successfully. Run ID: {self._tracking_run.info.run_id}")
                
            elif backend == 'wandb':
                import wandb
                wandb_config = tracking_config.get('wandb', {})
                self._tracking_run = wandb.init(
                    project=wandb_config.get('project', 'swe-bench-comparison'),
                    entity=wandb_config.get('entity'),
                    name=self.config.experiment_name,
                    config={'experiment_name': self.config.experiment_name}
                )
                self._tracking_backend = 'wandb'
                self._tracking_enabled = True
                self.logger.info(f"Weights & Biases tracking initialized")
                
        except ImportError as e:
            self.logger.warning(f"Tracking backend '{backend}' not available: {e}")
        except Exception as e:
            self.logger.warning(f"Failed to initialize tracking: {e}")
    
    def _log_metrics(self, method_name: str, result: ExperimentResult) -> None:
        """Log metrics to tracking backend.
        
        Args:
            method_name: Name of the method
            result: Experiment result
        """
        if not self._tracking_enabled:
            return
        
        try:
            # Basic metrics
            metrics = {
                f'{method_name}/resolution_rate': result.resolution_rate,
                f'{method_name}/duration': result.duration,
                f'{method_name}/total_instances': len(result.results),
                f'{method_name}/errors': len(result.errors),
            }
            
            # Aggregate token usage and calculate cost
            total_prompt_tokens = 0
            total_completion_tokens = 0
            total_cost = 0.0
            model_stats = {}
            
            for patch_result in result.patch_results:
                token_usage = getattr(patch_result, 'token_usage', {})
                if token_usage:
                    prompt_tokens = token_usage.get('prompt_tokens', 0)
                    completion_tokens = token_usage.get('completion_tokens', 0)
                    total_prompt_tokens += prompt_tokens
                    total_completion_tokens += completion_tokens
                    
                    # Track by model for cost calculation
                    model = getattr(patch_result, 'model', 'unknown')
                    if model not in model_stats:
                        model_stats[model] = {'prompt': 0, 'completion': 0}
                    model_stats[model]['prompt'] += prompt_tokens
                    model_stats[model]['completion'] += completion_tokens
            
            # Calculate cost using TokenUsageMetric pricing
            from ..evaluation.metrics.token_usage import TokenUsageMetric
            pricing = TokenUsageMetric.DEFAULT_PRICING
            
            for model, stats in model_stats.items():
                # Find matching pricing
                model_pricing = None
                for key, price in pricing.items():
                    if key in model.lower():
                        model_pricing = price
                        break
                
                if model_pricing:
                    prompt_cost = (stats['prompt'] / 1000) * model_pricing['prompt']
                    completion_cost = (stats['completion'] / 1000) * model_pricing['completion']
                    total_cost += prompt_cost + completion_cost
            
            # Add token and cost metrics
            metrics[f'{method_name}/prompt_tokens'] = total_prompt_tokens
            metrics[f'{method_name}/completion_tokens'] = total_completion_tokens
            metrics[f'{method_name}/total_tokens'] = total_prompt_tokens + total_completion_tokens
            metrics[f'{method_name}/estimated_cost_usd'] = round(total_cost, 4)
            
            self.logger.info(f"[TRACKING] Method {method_name}: {total_prompt_tokens + total_completion_tokens} tokens, ${total_cost:.4f} USD")
            
            if self._tracking_backend == 'mlflow':
                import mlflow
                # Log to parent run (for easy comparison across methods)
                # Use method index as step for time-series visualization
                method_index = list(self._methods.keys()).index(method_name)
                mlflow.log_metrics(metrics, step=method_index)
                # Also log as nested run (for isolation)
                with mlflow.start_run(run_name=method_name, nested=True):
                    mlflow.log_metrics(metrics)
                    mlflow.log_param('method', method_name)
                    mlflow.log_param('total_instances', len(result.results))
                    mlflow.log_param('errors', len(result.errors))
                    # Log cost as a separate metric for easy filtering
                    mlflow.log_metric('estimated_cost_usd', round(total_cost, 4))
                    mlflow.log_metric('total_tokens', total_prompt_tokens + total_completion_tokens)
                self.logger.info(f"[TRACKING] Logged metrics to MLflow for {method_name}")
                    
            elif self._tracking_backend == 'wandb':
                import wandb
                wandb.log(metrics)
                wandb.log({'estimated_cost_usd': round(total_cost, 4)})
                
        except Exception as e:
            self.logger.warning(f"Failed to log metrics: {e}")
    
    def _close_tracking(self) -> None:
        """Close tracking backend."""
        if not self._tracking_enabled:
            return
        
        try:
            if self._tracking_backend == 'mlflow':
                import mlflow
                mlflow.end_run()
            elif self._tracking_backend == 'wandb':
                import wandb
                wandb.finish()
        except Exception as e:
            self.logger.warning(f"Failed to close tracking: {e}")
        finally:
            self._tracking_enabled = False
    
    def register_method(
        self,
        name: str,
        patch_generator: PatchGenerator,
        context_gatherer: Optional[Any] = None,
        method_config: Dict[str, Any] = None
    ) -> None:
        """
        Register a method for comparison.
        
        Args:
            name: Method name
            patch_generator: Patch generator for the method
            context_gatherer: Optional context gatherer
            method_config: Additional method configuration
        """
        self._methods[name] = MethodConfig(
            name=name,
            patch_generator=patch_generator,
            context_gatherer=context_gatherer,
            config=method_config or {}
        )
        
        self.logger.info(f"Registered method: {name}")
    
    def run(
        self,
        dataset_loader: DatasetLoader,
        evaluator: Evaluator,
        repo_provider: Callable[[Any], str]
    ) -> Dict[str, ExperimentResult]:
        """
        Run the comparison experiment.
        
        Args:
            dataset_loader: Loader for the dataset
            evaluator: Evaluator for patches
            repo_provider: Function to get repo path for an instance
            
        Returns:
            Dictionary mapping method name to experiment results
        """
        self.logger.info(f"Starting experiment: {self.config.experiment_name}")
        self.logger.info(f"Number of methods: {len(self._methods)}")
        self.logger.info(f"Methods to run: {list(self._methods.keys())}")
        
        # Initialize experiment tracking
        self._init_tracking(self.config.tracking)
        
        start_time = time.time()
        
        # Run each method
        for method_idx, (method_name, method_config) in enumerate(self._methods.items(), 1):
            self.logger.info(f"[{method_idx}/{len(self._methods)}] Running method: {method_name}")
            
            # Create progress callback for MLflow logging
            parent_run_id = self._tracking_run.info.run_id if self._tracking_run else None
            def make_progress_callback(name, parent_id):
                def callback(current, total, metrics):
                    if self._tracking_enabled and self._tracking_backend == 'mlflow' and parent_id:
                        try:
                            import mlflow
                            # Log to the parent MLflow run (not nested)
                            step_metrics = {
                                f'{name}/progress': current / total * 100,
                                f'{name}/resolved_count': metrics['resolved'],
                                f'{name}/running_resolution_rate': metrics['resolution_rate'],
                            }
                            # Log instance-specific metrics (duration, tokens, cost)
                            if 'instance_duration' in metrics:
                                step_metrics[f'{name}/instance_duration'] = metrics['instance_duration']
                            if 'instance_prompt_tokens' in metrics:
                                step_metrics[f'{name}/instance_prompt_tokens'] = metrics['instance_prompt_tokens']
                                step_metrics[f'{name}/instance_completion_tokens'] = metrics['instance_completion_tokens']
                                step_metrics[f'{name}/instance_total_tokens'] = metrics['instance_total_tokens']
                            if 'instance_cost_usd' in metrics:
                                step_metrics[f'{name}/instance_cost_usd'] = metrics['instance_cost_usd']
                            
                            # Use current as step for time-series view
                            mlflow.log_metrics(step_metrics, step=current, run_id=parent_id)
                            
                            # Also log to a separate run for instance-level tracking
                            # This creates a table view of all instances
                            instance_id = metrics.get('instance_id', f'step_{current}')
                            with mlflow.start_run(run_name=f"{name}_{instance_id}", nested=True):
                                mlflow.log_params({
                                    'method': name,
                                    'instance_id': instance_id,
                                    'model': metrics.get('model', 'unknown'),
                                })
                                mlflow.log_metrics({
                                    'resolved': 1 if metrics.get('instance_resolved') else 0,
                                    'duration': metrics.get('instance_duration', 0),
                                    'prompt_tokens': metrics.get('instance_prompt_tokens', 0),
                                    'completion_tokens': metrics.get('instance_completion_tokens', 0),
                                    'total_tokens': metrics.get('instance_total_tokens', 0),
                                    'cost_usd': metrics.get('instance_cost_usd', 0),
                                })
                        except Exception as e:
                            pass  # Silently ignore tracking errors during progress
                return callback
            
            # Create experiment runner
            runner_config = ExperimentConfig(
                method_name=method_name,
                max_workers=self.config.max_workers_per_method,
                save_intermediate=self.config.save_intermediate,
                intermediate_dir=str(self.output_dir / 'intermediate'),
                log_level=self.config.log_level,
                progress_callback=make_progress_callback(method_name, parent_run_id)
            )
            
            runner = ExperimentRunner(runner_config)
            
            # Run experiment
            result = runner.run(
                dataset_loader=dataset_loader,
                patch_generator=method_config.patch_generator,
                evaluator=evaluator,
                repo_provider=repo_provider,
                context_gatherer=method_config.context_gatherer
            )
            
            self._results[method_name] = result
            
            # Log results
            self.logger.info(
                f"Method {method_name} completed: "
                f"{result.resolution_rate:.2f}% resolution rate"
            )
            
            # Log to tracking backend
            self._log_metrics(method_name, result)
        
        end_time = time.time()
        
        self.logger.info(f"Experiment completed in {end_time - start_time:.2f}s")
        
        # Save results
        self._save_results()
        
        # Generate report if configured
        if self.config.generate_report:
            self.generate_report()
        
        # Close tracking
        self._close_tracking()
        
        return self._results
    
    def generate_report(self) -> Dict[str, Any]:
        """
        Generate comparison report.
        
        Returns:
            Report dictionary
        """
        self.logger.info("Generating comparison report")
        
        # Add results to report generator
        for method_name, result in self._results.items():
            self._report_generator.add_method_results(
                method_name=method_name,
                results=result.results,
                config=self._methods[method_name].config
            )
        
        # Generate report
        report = self._report_generator.generate_comparison_report()
        
        # Save in requested formats
        for format in self.config.report_formats:
            output_path = self.output_dir / f"report.{format}"
            self._report_generator.save_report(report, str(output_path), format)
            self.logger.info(f"Saved {format} report to {output_path}")
        
        return report
    
    def _save_results(self) -> None:
        """Save experiment results."""
        results_file = self.output_dir / 'results.json'
        
        # Convert results to serializable format
        serializable_results = {}
        for method_name, result in self._results.items():
            serializable_results[method_name] = {
                'summary': result.to_dict(),
                'results': [r.to_dict() for r in result.results],
                'patch_results': [p.to_dict() for p in result.patch_results],
                'errors': result.errors
            }
        
        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        self.logger.info(f"Saved results to {results_file}")
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get experiment summary.
        
        Returns:
            Summary dictionary
        """
        summary = {
            'experiment_name': self.config.experiment_name,
            'num_methods': len(self._methods),
            'methods': list(self._methods.keys()),
            'results': {}
        }
        
        for method_name, result in self._results.items():
            summary['results'][method_name] = {
                'resolution_rate': result.resolution_rate,
                'num_instances': len(result.results),
                'num_errors': len(result.errors),
                'duration': result.duration
            }
        
        return summary
    
    def print_summary(self) -> None:
        """Print experiment summary to console."""
        summary = self.get_summary()
        
        print("\n" + "=" * 60)
        print(f"Experiment: {summary['experiment_name']}")
        print("=" * 60)
        
        print(f"\nMethods: {', '.join(summary['methods'])}")
        print("\nResults:")
        print("-" * 60)
        
        for method_name, result in summary['results'].items():
            print(f"\n{method_name}:")
            print(f"  Resolution Rate: {result['resolution_rate']:.2f}%")
            print(f"  Instances: {result['num_instances']}")
            print(f"  Errors: {result['num_errors']}")
            print(f"  Duration: {result['duration']:.2f}s")
        
        print("\n" + "=" * 60)


class BatchOrchestrator:
    """
    Orchestrator for running multiple experiments in batch.
    
    Useful for running experiments with different configurations
    or on different datasets.
    """
    
    def __init__(self, base_config: OrchestratorConfig):
        """
        Initialize the batch orchestrator.
        
        Args:
            base_config: Base configuration for all experiments
        """
        self.base_config = base_config
        self._experiments: List[Dict[str, Any]] = []
        self._results: List[Dict[str, Any]] = []
    
    def add_experiment(
        self,
        name: str,
        dataset_loader: DatasetLoader,
        methods: Dict[str, MethodConfig],
        evaluator: Evaluator,
        repo_provider: Callable[[Any], str],
        config_overrides: Dict[str, Any] = None
    ) -> None:
        """
        Add an experiment to the batch.
        
        Args:
            name: Experiment name
            dataset_loader: Dataset loader
            methods: Dictionary of methods
            evaluator: Evaluator
            repo_provider: Repo provider function
            config_overrides: Configuration overrides
        """
        self._experiments.append({
            'name': name,
            'dataset_loader': dataset_loader,
            'methods': methods,
            'evaluator': evaluator,
            'repo_provider': repo_provider,
            'config_overrides': config_overrides or {}
        })
    
    def run_all(self) -> List[Dict[str, Any]]:
        """
        Run all experiments in the batch.
        
        Returns:
            List of experiment results
        """
        for exp_config in self._experiments:
            logger.info(f"Running experiment: {exp_config['name']}")
            
            # Create orchestrator with overrides
            config = self._merge_configs(self.base_config, exp_config['config_overrides'])
            config.experiment_name = exp_config['name']
            
            orchestrator = ExperimentOrchestrator(config)
            
            # Register methods
            for method_name, method_config in exp_config['methods'].items():
                orchestrator.register_method(
                    method_name,
                    method_config.patch_generator,
                    method_config.context_gatherer,
                    method_config.config
                )
            
            # Run experiment
            results = orchestrator.run(
                dataset_loader=exp_config['dataset_loader'],
                evaluator=exp_config['evaluator'],
                repo_provider=exp_config['repo_provider']
            )
            
            # Store results
            self._results.append({
                'name': exp_config['name'],
                'results': results,
                'summary': orchestrator.get_summary()
            })
        
        return self._results
    
    def _merge_configs(
        self,
        base: OrchestratorConfig,
        overrides: Dict[str, Any]
    ) -> OrchestratorConfig:
        """Merge configuration overrides."""
        base_dict = {
            'experiment_name': base.experiment_name,
            'output_dir': base.output_dir,
            'max_workers_per_method': base.max_workers_per_method,
            'save_intermediate': base.save_intermediate,
            'generate_report': base.generate_report,
            'report_formats': base.report_formats,
            'log_level': base.log_level
        }
        
        base_dict.update(overrides)
        
        return OrchestratorConfig(**base_dict)
    
    def save_batch_results(self, output_path: str) -> None:
        """
        Save batch results to file.
        
        Args:
            output_path: Path to save results
        """
        with open(output_path, 'w') as f:
            json.dump(self._results, f, indent=2, default=str)
        
        logger.info(f"Saved batch results to {output_path}")


class ExperimentBuilder:
    """
    Builder for constructing experiments declaratively.
    
    Example:
        builder = ExperimentBuilder()
        
        experiment = (builder
            .with_name("My Experiment")
            .with_output_dir("./results")
            .with_dataset('lite')
            .with_method("RAG", rag_generator, rag_gatherer)
            .with_method("Agentic", agent_generator, agent_gatherer)
            .with_evaluator('swe_bench')
            .build())
        
        results = experiment.run()
    """
    
    def __init__(self):
        self.name = "experiment"
        self.output_dir = "./results"
        self.dataset_name = "lite"
        self.methods: Dict[str, MethodConfig] = {}
        self.evaluator_type = "swe_bench"
        self.max_workers = 1
    
    def with_name(self, name: str) -> 'ExperimentBuilder':
        """Set experiment name."""
        self.name = name
        return self
    
    def with_output_dir(self, output_dir: str) -> 'ExperimentBuilder':
        """Set output directory."""
        self.output_dir = output_dir
        return self
    
    def with_dataset(self, dataset_name: str) -> 'ExperimentBuilder':
        """Set dataset name."""
        self.dataset_name = dataset_name
        return self
    
    def with_method(
        self,
        name: str,
        generator: PatchGenerator,
        gatherer: Optional[Any] = None
    ) -> 'ExperimentBuilder':
        """Add a method."""
        self.methods[name] = MethodConfig(
            name=name,
            patch_generator=generator,
            context_gatherer=gatherer
        )
        return self
    
    def with_evaluator(self, evaluator_type: str) -> 'ExperimentBuilder':
        """Set evaluator type."""
        self.evaluator_type = evaluator_type
        return self
    
    def with_parallel(self, max_workers: int) -> 'ExperimentBuilder':
        """Enable parallel execution."""
        self.max_workers = max_workers
        return self
    
    def build(self) -> ExperimentOrchestrator:
        """Build the experiment orchestrator."""
        config = OrchestratorConfig(
            experiment_name=self.name,
            output_dir=self.output_dir,
            max_workers_per_method=self.max_workers
        )
        
        orchestrator = ExperimentOrchestrator(config)
        
        for method_name, method_config in self.methods.items():
            orchestrator.register_method(
                method_name,
                method_config.patch_generator,
                method_config.context_gatherer,
                method_config.config
            )
        
        return orchestrator
