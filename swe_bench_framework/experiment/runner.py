"""
Experiment runner for the SWE-bench comparison framework.

This module provides utilities for running a single method on a dataset,
with support for parallel execution and progress tracking.
"""

import time
import logging
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple
from dataclasses import dataclass, field
from concurrent.futures import ProcessPoolExecutor, as_completed

from ..dataset.loader import SWEInstance, DatasetLoader
from ..patch_generators.base import PatchGenerator, PatchResult
from ..evaluation.base import Evaluator, EvaluationResult


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    """Configuration for an experiment run."""
    method_name: str
    max_workers: int = 1
    timeout: Optional[int] = None
    continue_on_error: bool = True
    save_intermediate: bool = True
    intermediate_dir: Optional[str] = None
    log_level: str = 'INFO'
    progress_callback: Optional[Callable] = None  # Callback(current, total, metrics)


@dataclass
class ExperimentResult:
    """Result of an experiment run."""
    method_name: str
    results: List[EvaluationResult]
    patch_results: List[PatchResult]
    config: ExperimentConfig
    start_time: float
    end_time: float
    errors: List[Dict[str, Any]] = field(default_factory=list)
    
    @property
    def duration(self) -> float:
        """Get experiment duration in seconds."""
        return self.end_time - self.start_time
    
    @property
    def resolution_rate(self) -> float:
        """Get resolution rate as percentage."""
        if not self.results:
            return 0.0
        resolved = sum(1 for r in self.results if r.resolved)
        return (resolved / len(self.results)) * 100
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'method_name': self.method_name,
            'duration': self.duration,
            'resolution_rate': self.resolution_rate,
            'num_instances': len(self.results),
            'num_errors': len(self.errors),
            'config': {
                'max_workers': self.config.max_workers,
                'timeout': self.config.timeout
            }
        }


class ExperimentRunner:
    """
    Runner for executing a single method on a dataset.
    
    This class handles:
    - Loading instances from the dataset
    - Running patch generation for each instance
    - Evaluating generated patches
    - Collecting and aggregating results
    - Saving intermediate results
    
    Example:
        runner = ExperimentRunner(config)
        
        result = runner.run(
            dataset_loader=loader,
            patch_generator=generator,
            evaluator=evaluator,
            repo_provider=repo_provider
        )
        
        print(f"Resolution rate: {result.resolution_rate:.2f}%")
    """
    
    def __init__(self, config: ExperimentConfig):
        """
        Initialize the experiment runner.
        
        Args:
            config: Experiment configuration
        """
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{config.method_name}")
        self.logger.setLevel(getattr(logging, config.log_level))
        
        self._results: List[EvaluationResult] = []
        self._patch_results: List[PatchResult] = []
        self._errors: List[Dict[str, Any]] = []
    
    def run(
        self,
        dataset_loader: DatasetLoader,
        patch_generator: PatchGenerator,
        evaluator: Evaluator,
        repo_provider: Callable[[SWEInstance], str],
        context_gatherer: Optional[Any] = None
    ) -> ExperimentResult:
        """
        Run the experiment.
        
        Args:
            dataset_loader: Loader for the dataset
            patch_generator: Generator for patches
            evaluator: Evaluator for patches
            repo_provider: Function to get repo path for an instance
            context_gatherer: Optional context gatherer
            
        Returns:
            ExperimentResult with all results
        """
        start_time = time.time()
        
        self.logger.info(f"Starting experiment: {self.config.method_name}")
        
        # Initialize components with stage logging and memory tracking
        self.logger.info("[1/4] Initializing dataset loader...")
        self._log_memory("Before dataset init")
        dataset_loader.initialize()
        self._log_memory("After dataset init")
        self.logger.info("[1/4] Dataset loader initialized")
        
        self.logger.info("[2/4] Initializing patch generator...")
        self._log_memory("Before patch gen init")
        patch_generator.initialize()
        self._log_memory("After patch gen init")
        self.logger.info("[2/4] Patch generator initialized")
        
        self.logger.info("[3/4] Initializing evaluator...")
        self._log_memory("Before evaluator init")
        evaluator.initialize()
        self._log_memory("After evaluator init")
        self.logger.info("[3/4] Evaluator initialized")
        
        if context_gatherer:
            self.logger.info("[4/4] Initializing context gatherer...")
            self._log_memory("Before context gatherer init")
            context_gatherer.initialize()
            self._log_memory("After context gatherer init")
            self.logger.info("[4/4] Context gatherer initialized")
        
        try:
            # Load instances - use streaming when possible
            self.logger.info("Loading instances...")
            self._log_memory("Before loading instances")
            
            # Check for filters
            instance_ids = getattr(dataset_loader, 'filter', {}).get('instance_ids')
            
            # Always use load_all which now efficiently handles streaming
            instances = dataset_loader.load_all()
            self._log_memory("After loading instances")
            self.logger.info(f"Loaded {len(instances)} instances")
            
            # Run experiment
            if self.config.max_workers > 1:
                results, patch_results = self._run_parallel(
                    instances, patch_generator, evaluator, repo_provider, context_gatherer
                )
            else:
                results, patch_results = self._run_sequential(
                    instances, patch_generator, evaluator, repo_provider, context_gatherer
                )
            
            self._results = results
            self._patch_results = patch_results
            
        finally:
            # Cleanup
            if context_gatherer:
                context_gatherer.cleanup()
            evaluator.cleanup()
            patch_generator.cleanup()
            dataset_loader.cleanup()
        
        end_time = time.time()
        
        self.logger.info(f"Experiment completed in {end_time - start_time:.2f}s")
        self.logger.info(f"Resolution rate: {self.resolution_rate:.2f}%")
        
        return ExperimentResult(
            method_name=self.config.method_name,
            results=self._results,
            patch_results=self._patch_results,
            config=self.config,
            start_time=start_time,
            end_time=end_time,
            errors=self._errors
        )
    
    def _run_sequential(
        self,
        instances: List[SWEInstance],
        patch_generator: PatchGenerator,
        evaluator: Evaluator,
        repo_provider: Callable[[SWEInstance], str],
        context_gatherer: Optional[Any]
    ) -> Tuple[List[EvaluationResult], List[PatchResult]]:
        """
        Run experiment sequentially.
        
        Args:
            instances: List of instances to process
            patch_generator: Patch generator
            evaluator: Evaluator
            repo_provider: Repo provider function
            context_gatherer: Context gatherer
            
        Returns:
            Tuple of (evaluation results, patch results)
        """
        results = []
        patch_results = []
        total = len(instances)
        
        self.logger.info(f"Processing {total} instances sequentially")
        
        for i, instance in enumerate(instances):
            progress_pct = (i / total) * 100 if total > 0 else 0
            progress_bar = self._make_progress_bar(i, total, width=30)
            
            self.logger.info(f"[{progress_bar}] {i+1}/{total} ({progress_pct:.1f}%) - Processing {instance.instance_id}")
            
            try:
                eval_result, patch_result = self._process_instance(
                    instance, patch_generator, evaluator, repo_provider, context_gatherer
                )
                
                results.append(eval_result)
                patch_results.append(patch_result)
                
                # Calculate running metrics for progress callback
                resolved_count = sum(1 for r in results if r.resolved)
                
                # Get token usage for this instance
                token_usage = getattr(patch_result, 'token_usage', {})
                prompt_tokens = token_usage.get('prompt_tokens', 0)
                completion_tokens = token_usage.get('completion_tokens', 0)
                
                # Get duration for this instance
                instance_duration = getattr(eval_result, 'execution_time', 0)
                
                # Calculate cost for this instance
                model = getattr(patch_result, 'model', 'unknown')
                from ..evaluation.metrics.token_usage import TokenUsageMetric
                pricing = TokenUsageMetric.DEFAULT_PRICING
                instance_cost = 0.0
                for key, price in pricing.items():
                    if key in model.lower():
                        prompt_cost = (prompt_tokens / 1000) * price['prompt']
                        completion_cost = (completion_tokens / 1000) * price['completion']
                        instance_cost = prompt_cost + completion_cost
                        break
                
                metrics = {
                    'resolved': resolved_count,
                    'total': len(results),
                    'resolution_rate': (resolved_count / len(results) * 100) if results else 0,
                    # Instance-specific metrics
                    'instance_id': instance.instance_id,
                    'instance_resolved': eval_result.resolved,
                    'instance_duration': instance_duration,
                    'instance_prompt_tokens': prompt_tokens,
                    'instance_completion_tokens': completion_tokens,
                    'instance_total_tokens': prompt_tokens + completion_tokens,
                    'instance_cost_usd': round(instance_cost, 6),
                    'model': model,
                }
                
                # Log success status with detailed metrics
                status = "RESOLVED" if eval_result.resolved else "FAILED"
                cost_str = f"${metrics['instance_cost_usd']:.4f}" if metrics['instance_cost_usd'] > 0 else "N/A"
                tokens_str = f"{metrics['instance_total_tokens']} tokens" if metrics['instance_total_tokens'] > 0 else "N/A tokens"
                duration_str = f"{metrics['instance_duration']:.1f}s" if metrics['instance_duration'] > 0 else "N/A"
                self.logger.info(f"  -> {status} | {tokens_str} | {cost_str} | {duration_str}")
                
                # Call progress callback if provided
                if self.config.progress_callback:
                    self.config.progress_callback(i + 1, total, metrics)
                
                # Save intermediate results
                if self.config.save_intermediate:
                    self._save_intermediate(results, patch_results)
                
            except Exception as e:
                self.logger.error(f"  -> ERROR: {e}")
                self._errors.append({
                    'instance_id': instance.instance_id,
                    'error': str(e),
                    'timestamp': time.time()
                })
                
                if not self.config.continue_on_error:
                    raise
        
        # Final progress
        progress_bar = self._make_progress_bar(total, total, width=30)
        self.logger.info(f"[{progress_bar}] {total}/{total} (100.0%) - Complete")
        
        return results, patch_results
    
    def _log_memory(self, label: str) -> None:
        """Log current memory usage."""
        try:
            import psutil
            import os
            process = psutil.Process(os.getpid())
            mem_mb = process.memory_info().rss / 1024 / 1024
            self.logger.info(f"[MEMORY] {label}: {mem_mb:.1f} MB")
        except ImportError:
            pass  # psutil not available
    
    def _make_progress_bar(self, current: int, total: int, width: int = 30) -> str:
        """Create a text progress bar."""
        if total == 0:
            return "█" * width
        filled = int(width * current / total)
        empty = width - filled
        return "█" * filled + "░" * empty
    
    def _run_sequential_generator(
        self,
        instances: Iterator[SWEInstance],
        patch_generator: PatchGenerator,
        evaluator: Evaluator,
        repo_provider: Callable[[SWEInstance], str],
        context_gatherer: Optional[Any]
    ) -> Tuple[List[EvaluationResult], List[PatchResult]]:
        """
        Run experiment sequentially using a generator (memory efficient).
        
        Args:
            instances: Iterator of instances to process
            patch_generator: Patch generator
            evaluator: Evaluator
            repo_provider: Repo provider function
            context_gatherer: Context gatherer
            
        Returns:
            Tuple of (evaluation results, patch results)
        """
        results = []
        patch_results = []
        i = 0
        
        self.logger.info("Processing instances (streaming mode, no progress bar available)")
        
        for instance in instances:
            i += 1
            self.logger.info(f"[{i}] Processing {instance.instance_id}")
            
            try:
                eval_result, patch_result = self._process_instance(
                    instance, patch_generator, evaluator, repo_provider, context_gatherer
                )
                
                results.append(eval_result)
                patch_results.append(patch_result)
                
                # Log success status
                status = "RESOLVED" if eval_result.resolved else "FAILED"
                self.logger.info(f"  -> {status} (patch: {'valid' if patch_result.success else 'invalid'})")
                
                # Save intermediate results periodically
                if self.config.save_intermediate and i % 5 == 0:
                    self._save_intermediate(results, patch_results)
                    self.logger.info(f"  -> Saved intermediate results ({i} processed)")
                
            except Exception as e:
                self.logger.error(f"  -> ERROR: {e}")
                self._errors.append({
                    'instance_id': instance.instance_id,
                    'error': str(e),
                    'timestamp': time.time()
                })
                
                if not self.config.continue_on_error:
                    raise
        
        self.logger.info(f"Complete - processed {i} instances")
        
        # Final save
        if self.config.save_intermediate:
            self._save_intermediate(results, patch_results)
        
        return results, patch_results
    
    def _run_parallel(
        self,
        instances: List[SWEInstance],
        patch_generator: PatchGenerator,
        evaluator: Evaluator,
        repo_provider: Callable[[SWEInstance], str],
        context_gatherer: Optional[Any]
    ) -> Tuple[List[EvaluationResult], List[PatchResult]]:
        """
        Run experiment in parallel.
        
        Args:
            instances: List of instances to process
            patch_generator: Patch generator
            evaluator: Evaluator
            repo_provider: Repo provider function
            context_gatherer: Context gatherer
            
        Returns:
            Tuple of (evaluation results, patch results)
        """
        results = []
        patch_results = []
        
        with ProcessPoolExecutor(max_workers=self.config.max_workers) as executor:
            # Submit all tasks
            future_to_instance = {
                executor.submit(
                    self._process_instance_wrapper,
                    instance,
                    patch_generator,
                    evaluator,
                    repo_provider,
                    context_gatherer
                ): instance
                for instance in instances
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_instance):
                instance = future_to_instance[future]
                
                try:
                    eval_result, patch_result = future.result(
                        timeout=self.config.timeout
                    )
                    results.append(eval_result)
                    patch_results.append(patch_result)
                    
                except Exception as e:
                    self.logger.error(f"Error processing {instance.instance_id}: {e}")
                    self._errors.append({
                        'instance_id': instance.instance_id,
                        'error': str(e),
                        'timestamp': time.time()
                    })
        
        return results, patch_results
    
    def _process_instance(
        self,
        instance: SWEInstance,
        patch_generator: PatchGenerator,
        evaluator: Evaluator,
        repo_provider: Callable[[SWEInstance], str],
        context_gatherer: Optional[Any]
    ) -> Tuple[EvaluationResult, PatchResult]:
        """
        Process a single instance.
        
        Args:
            instance: Instance to process
            patch_generator: Patch generator
            evaluator: Evaluator
            repo_provider: Repo provider function
            context_gatherer: Context gatherer
            
        Returns:
            Tuple of (evaluation result, patch result)
        """
        # Get repository path
        repo_path = repo_provider(instance)
        
        # Gather context if gatherer provided
        context_bundle = None
        if context_gatherer:
            context_bundle = context_gatherer.gather_context(instance, repo_path)
        
        # Generate patch
        patch_result = patch_generator.generate_patch(context_bundle, instance)
        
        # Evaluate patch
        eval_result = evaluator.evaluate(patch_result, instance, repo_path)
        
        return eval_result, patch_result
    
    def _process_instance_wrapper(
        self,
        instance: SWEInstance,
        patch_generator: PatchGenerator,
        evaluator: Evaluator,
        repo_provider: Callable[[SWEInstance], str],
        context_gatherer: Optional[Any]
    ) -> Tuple[EvaluationResult, PatchResult]:
        """Wrapper for parallel processing."""
        return self._process_instance(
            instance, patch_generator, evaluator, repo_provider, context_gatherer
        )
    
    def _save_intermediate(
        self,
        results: List[EvaluationResult],
        patch_results: List[PatchResult]
    ) -> None:
        """Save intermediate results."""
        if not self.config.intermediate_dir:
            return
        
        import json
        
        intermediate_dir = Path(self.config.intermediate_dir)
        intermediate_dir.mkdir(parents=True, exist_ok=True)
        
        # Save results
        results_file = intermediate_dir / f"{self.config.method_name}_results.jsonl"
        with open(results_file, 'w') as f:
            for result in results:
                f.write(json.dumps(result.to_dict()) + '\n')
        
        # Save patch results
        patches_file = intermediate_dir / f"{self.config.method_name}_patches.jsonl"
        with open(patches_file, 'w') as f:
            for patch in patch_results:
                f.write(json.dumps(patch.to_dict()) + '\n')
    
    @property
    def resolution_rate(self) -> float:
        """Get current resolution rate."""
        if not self._results:
            return 0.0
        resolved = sum(1 for r in self._results if r.resolved)
        return (resolved / len(self._results)) * 100


class SingleInstanceRunner:
    """
    Runner for processing a single instance.
    
    Useful for debugging and testing.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def run(
        self,
        instance: SWEInstance,
        patch_generator: PatchGenerator,
        evaluator: Evaluator,
        repo_path: str,
        context_gatherer: Optional[Any] = None
    ) -> Tuple[EvaluationResult, PatchResult]:
        """
        Run on a single instance.
        
        Args:
            instance: Instance to process
            patch_generator: Patch generator
            evaluator: Evaluator
            repo_path: Path to repository
            context_gatherer: Optional context gatherer
            
        Returns:
            Tuple of (evaluation result, patch result)
        """
        self.logger.info(f"Processing single instance: {instance.instance_id}")
        
        # Initialize
        patch_generator.initialize()
        evaluator.initialize()
        
        if context_gatherer:
            context_gatherer.initialize()
        
        try:
            # Gather context
            context_bundle = None
            if context_gatherer:
                context_bundle = context_gatherer.gather_context(instance, repo_path)
            
            # Generate patch
            self.logger.info("Generating patch...")
            patch_result = patch_generator.generate_patch(context_bundle, instance)
            
            # Evaluate patch
            self.logger.info("Evaluating patch...")
            eval_result = evaluator.evaluate(patch_result, instance, repo_path)
            
            return eval_result, patch_result
            
        finally:
            # Cleanup
            if context_gatherer:
                context_gatherer.cleanup()
            evaluator.cleanup()
            patch_generator.cleanup()
