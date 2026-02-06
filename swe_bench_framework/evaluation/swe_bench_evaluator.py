"""
SWE-bench evaluator for the comparison framework.

This module implements the main evaluation logic for SWE-bench instances,
including patch application, test execution, and metric computation.
"""

import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from .base import Evaluator, EvaluationResult, TestResults, Metric
from .docker_sandbox import DockerSandbox, LocalSandbox
from ..patch_generators.base import PatchResult
from ..context_gatherers.base import ContextBundle
from ..dataset.loader import SWEInstance


class SWEBenchEvaluator(Evaluator):
    """
    Standard SWE-bench evaluation pipeline.
    
    This evaluator:
    1. Applies the generated patch in a sandboxed environment
    2. Runs FAIL_TO_PASS and PASS_TO_PASS tests
    3. Computes all evaluation metrics
    4. Returns a comprehensive EvaluationResult
    
    Example:
        evaluator = SWEBenchEvaluator(config)
        evaluator.initialize()
        
        result = evaluator.evaluate(patch_result, instance, repo_path)
        
        if result.resolved:
            print(f"Instance resolved!")
        print(f"Tests passed: {len(result.tests_passed)}")
        print(f"Localization: {result.localization_accuracy}")
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the SWE-bench evaluator.
        
        Args:
            config: Configuration dictionary
                - docker: Docker sandbox configuration
                - use_docker: Whether to use Docker (default: True)
                - compute_codebleu: Whether to compute CodeBLEU (default: True)
                - compute_localization: Whether to compute localization (default: True)
                - metrics: List of metric configurations
        """
        super().__init__(config)
        
        self.use_docker = config.get('use_docker', True)
        
        # Initialize sandbox
        sandbox_config = config.get('docker', {})
        if self.use_docker:
            self.sandbox = DockerSandbox(sandbox_config)
        else:
            self.sandbox = LocalSandbox(sandbox_config)
        
        # Metric computation flags
        self.compute_codebleu_flag = config.get('compute_codebleu', True)
        self.compute_localization_flag = config.get('compute_localization', True)
        
        # Initialize metrics
        self._init_metrics()
    
    def _init_metrics(self) -> None:
        """Initialize metric collectors."""
        from .metrics.resolution import ResolutionMetric
        from .metrics.localization import LocalizationMetric
        from .metrics.codebleu import CodeBLEUMetric
        from .metrics.token_usage import TokenUsageMetric
        
        # Create metrics in a local variable (not self._metrics to avoid infinite loop)
        metrics = [
            ResolutionMetric(),
            LocalizationMetric(),
            CodeBLEUMetric(),
            TokenUsageMetric()
        ]
        
        # Register metrics with evaluator (register_metric appends to self._metrics)
        for metric in metrics:
            self.register_metric(metric)
    
    def initialize(self) -> None:
        """Initialize the evaluator and sandbox."""
        import logging
        logger = logging.getLogger(__name__)
        
        super().initialize()
        
        sandbox_type = "Docker" if self.use_docker else "Local"
        logger.info(f"Initializing {sandbox_type} sandbox")
        self.sandbox.initialize()
        logger.info(f"Sandbox initialized")
    
    def cleanup(self) -> None:
        """Cleanup resources."""
        self.sandbox.cleanup()
        super().cleanup()
    
    def evaluate(
        self,
        patch_result: PatchResult,
        instance: SWEInstance,
        repo_path: str
    ) -> EvaluationResult:
        """
        Evaluate a generated patch against a SWE-bench instance.
        
        Args:
            patch_result: The result from patch generation
            instance: The SWE-bench instance
            repo_path: Path to the repository
            
        Returns:
            EvaluationResult with all metrics
        """
        start_time = time.time()
        instance_id = instance.instance_id
        
        # Handle failed patch generation
        if not patch_result.success or not patch_result.patch_content:
            return EvaluationResult(
                instance_id=instance_id,
                resolved=False,
                patch_applied=False,
                tests_passed=[],
                tests_failed=getattr(instance, 'failed_tests', []),
                localization_accuracy={},
                codebleu_score=0.0,
                execution_time=time.time() - start_time,
                metadata={
                    'error': patch_result.error_message or "No patch generated",
                    'generation_attempts': patch_result.attempts
                }
            )
        
        # 1. Apply patch in sandbox
        patch_applied = self.sandbox.apply_patch(
            patch_result.patch_content,
            repo_path,
            instance=instance
        )
        
        if not patch_applied:
            return EvaluationResult(
                instance_id=instance_id,
                resolved=False,
                patch_applied=False,
                tests_passed=[],
                tests_failed=getattr(instance, 'failed_tests', []),
                localization_accuracy={},
                codebleu_score=0.0,
                execution_time=time.time() - start_time,
                metadata={'error': 'Failed to apply patch'}
            )
        
        # 2. Run tests
        test_results = self.sandbox.run_tests(instance, repo_path)
        
        # 3. Check if resolved
        resolved = self._check_resolved(test_results, instance)
        
        # 4. Compute localization accuracy if context available
        localization_accuracy = {}
        if self.compute_localization_flag and hasattr(patch_result, 'context_bundle'):
            context_bundle = getattr(patch_result, 'context_bundle', None)
            if context_bundle:
                localization_accuracy = self.compute_localization_accuracy(
                    context_bundle,
                    getattr(instance, 'modified_files', []),
                    getattr(instance, 'modified_methods', [])
                )
        
        # 5. Compute CodeBLEU if gold patch available
        codebleu_score = 0.0
        if self.compute_codebleu_flag and hasattr(instance, 'patch'):
            gold_patch = getattr(instance, 'patch', None)
            if gold_patch:
                codebleu_score = self._compute_codebleu(
                    patch_result.patch_content,
                    gold_patch
                )
        
        execution_time = time.time() - start_time
        
        return EvaluationResult(
            instance_id=instance_id,
            resolved=resolved,
            patch_applied=True,
            tests_passed=test_results.passed,
            tests_failed=test_results.failed,
            localization_accuracy=localization_accuracy,
            codebleu_score=codebleu_score,
            execution_time=execution_time,
            metadata={
                'test_errors': test_results.errors,
                'test_skipped': test_results.skipped,
                'test_execution_time': test_results.execution_time,
                'generation_time': patch_result.generation_time,
                'generation_attempts': patch_result.attempts,
                'token_usage': patch_result.token_usage
            }
        )
    
    def _check_resolved(
        self,
        test_results: TestResults,
        instance: SWEInstance
    ) -> bool:
        """
        Check if the instance is resolved based on test results.
        
        An instance is resolved if:
        1. All FAIL_TO_PASS tests pass
        2. All PASS_TO_PASS tests still pass
        
        Args:
            test_results: Results from running tests
            instance: The SWE-bench instance
            
        Returns:
            True if resolved
        """
        fail_to_pass = set(getattr(instance, 'failed_tests', []))
        pass_to_pass = set(getattr(instance, 'passed_tests', []))
        
        passed = set(test_results.passed)
        failed = set(test_results.failed)
        
        # All FAIL_TO_PASS tests must pass
        fail_to_pass_resolved = fail_to_pass.issubset(passed)
        
        # All PASS_TO_PASS tests must still pass
        pass_to_pass_preserved = pass_to_pass.issubset(passed) and \
                                 not bool(pass_to_pass & failed)
        
        return fail_to_pass_resolved and pass_to_pass_preserved
    
    def compute_localization_accuracy(
        self,
        context_bundle: ContextBundle,
        gold_files: List[str],
        gold_functions: List[str]
    ) -> Dict[str, float]:
        """
        Compute localization accuracy metrics.
        
        Computes Recall@K and Hit@K for files, functions, and snippets.
        
        Args:
            context_bundle: The gathered context
            gold_files: List of gold-standard modified files
            gold_functions: List of gold-standard modified functions
            
        Returns:
            Dictionary with localization metrics
        """
        from .metrics.localization import LocalizationMetric
        
        metric = LocalizationMetric()
        
        # Handle missing context bundle
        if context_bundle is None:
            return {'localization_score': 0.0, 'error': 'No context provided'}
        
        # Extract predicted files and functions from context
        predicted_files = []
        predicted_functions = []
        
        for chunk in context_bundle.chunks:
            if chunk.source_file and chunk.source_file not in predicted_files:
                predicted_files.append(chunk.source_file)
            
            # Try to extract function names from metadata
            if hasattr(chunk, 'metadata') and chunk.metadata:
                func_name = chunk.metadata.get('function_name')
                if func_name and func_name not in predicted_functions:
                    predicted_functions.append(func_name)
        
        # Compute metrics
        results = {}
        
        # File-level metrics
        for k in [1, 3, 5, 10]:
            results[f'file_recall@{k}'] = metric.compute_recall_at_k(
                gold_files, predicted_files, k
            )
            results[f'file_hit@{k}'] = metric.compute_hit_at_k(
                gold_files, predicted_files, k
            )
        
        # Function-level metrics
        if gold_functions:
            for k in [1, 3, 5, 10]:
                results[f'function_recall@{k}'] = metric.compute_recall_at_k(
                    gold_functions, predicted_functions, k
                )
                results[f'function_hit@{k}'] = metric.compute_hit_at_k(
                    gold_functions, predicted_functions, k
                )
        
        # Overall localization score
        results['localization_score'] = results.get('file_recall@5', 0.0)
        
        return results
    
    def _compute_codebleu(self, generated_patch: str, gold_patch: str) -> float:
        """
        Compute CodeBLEU score between generated and gold patches.
        
        Args:
            generated_patch: The generated patch
            gold_patch: The gold reference patch
            
        Returns:
            CodeBLEU score (0.0-1.0)
        """
        from .metrics.codebleu import CodeBLEUMetric
        
        metric = CodeBLEUMetric()
        return metric.compute_codebleu(generated_patch, gold_patch)
    
    def evaluate_batch(
        self,
        patch_results: List[PatchResult],
        instances: List[SWEInstance],
        repo_paths: List[str]
    ) -> List[EvaluationResult]:
        """
        Evaluate multiple patches in batch.
        
        Args:
            patch_results: List of patch results
            instances: List of SWE-bench instances
            repo_paths: List of repository paths
            
        Returns:
            List of evaluation results
        """
        results = []
        for patch, instance, repo_path in zip(patch_results, instances, repo_paths):
            result = self.evaluate(patch, instance, repo_path)
            results.append(result)
        return results
    
    def get_aggregate_stats(
        self,
        results: List[EvaluationResult]
    ) -> Dict[str, Any]:
        """
        Compute aggregate statistics from multiple evaluation results.
        
        Args:
            results: List of evaluation results
            
        Returns:
            Dictionary with aggregate statistics
        """
        if not results:
            return {}
        
        total = len(results)
        resolved = sum(1 for r in results if r.resolved)
        patch_applied = sum(1 for r in results if r.patch_applied)
        
        # Average metrics
        avg_codebleu = sum(r.codebleu_score for r in results) / total
        avg_execution_time = sum(r.execution_time for r in results) / total
        
        # Localization metrics (average across all results)
        localization_keys = set()
        for r in results:
            localization_keys.update(r.localization_accuracy.keys())
        
        avg_localization = {}
        for key in localization_keys:
            values = [r.localization_accuracy.get(key, 0.0) for r in results]
            avg_localization[f'avg_{key}'] = sum(values) / len(values)
        
        return {
            'total_instances': total,
            'resolved': resolved,
            'unresolved': total - resolved,
            'patch_applied': patch_applied,
            'patch_apply_rate': (patch_applied / total) * 100 if total > 0 else 0,
            'resolution_rate': (resolved / total) * 100 if total > 0 else 0,
            'avg_codebleu': avg_codebleu,
            'avg_execution_time': avg_execution_time,
            'localization': avg_localization
        }


class LightweightEvaluator(Evaluator):
    """
    Lightweight evaluator that doesn't require Docker.
    
    This evaluator performs basic checks without running actual tests,
    useful for quick validation during development.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.check_syntax = config.get('check_syntax', True)
        self.check_format = config.get('check_format', True)
    
    def evaluate(
        self,
        patch_result: PatchResult,
        instance: SWEInstance,
        repo_path: str
    ) -> EvaluationResult:
        """
        Perform lightweight evaluation.
        
        Checks:
        1. Patch was generated
        2. Patch has correct format
        3. Patch can be parsed
        """
        start_time = time.time()
        
        # Check if patch was generated
        if not patch_result.success or not patch_result.patch_content:
            return EvaluationResult(
                instance_id=instance.instance_id,
                resolved=False,
                patch_applied=False,
                tests_passed=[],
                tests_failed=[],
                localization_accuracy={},
                codebleu_score=0.0,
                execution_time=time.time() - start_time,
                metadata={'error': 'No patch generated'}
            )
        
        patch = patch_result.patch_content
        
        # Check format
        format_valid = True
        if self.check_format:
            format_valid = self._check_patch_format(patch)
        
        # Check syntax
        syntax_valid = True
        if self.check_syntax:
            syntax_valid = self._check_patch_syntax(patch)
        
        execution_time = time.time() - start_time
        
        return EvaluationResult(
            instance_id=instance.instance_id,
            resolved=format_valid and syntax_valid,  # Lightweight "resolution"
            patch_applied=format_valid,
            tests_passed=[],
            tests_failed=[],
            localization_accuracy={},
            codebleu_score=0.0,
            execution_time=execution_time,
            metadata={
                'format_valid': format_valid,
                'syntax_valid': syntax_valid,
                'patch_length': len(patch)
            }
        )
    
    def _check_patch_format(self, patch: str) -> bool:
        """Check if patch has correct unified diff format."""
        # Check for required headers
        has_minus = '---' in patch
        has_plus = '+++' in patch
        has_hunk = '@@' in patch
        
        return has_minus and has_plus
    
    def _check_patch_syntax(self, patch: str) -> bool:
        """Check patch syntax by parsing hunks."""
        lines = patch.split('\n')
        in_hunk = False
        
        for line in lines:
            if line.startswith('@@'):
                in_hunk = True
                # Check hunk header format
                if not re.match(r'@@ -\d+(,\d+)? \+\d+(,\d+)? @@', line):
                    return False
            elif in_hunk:
                # Check line prefixes
                if line and not line.startswith((' ', '-', '+', '\\')):
                    return False
        
        return True
    
    def compute_localization_accuracy(
        self,
        context_bundle: ContextBundle,
        gold_files: List[str],
        gold_functions: List[str]
    ) -> Dict[str, float]:
        """Compute localization accuracy (same as full evaluator)."""
        from .metrics.localization import LocalizationMetric
        
        # Handle missing context bundle
        if context_bundle is None:
            return {'file_recall@5': 0.0, 'file_hit@5': 0.0, 'error': 'No context provided'}
        
        metric = LocalizationMetric()
        
        predicted_files = [chunk.source_file for chunk in context_bundle.chunks]
        
        return {
            'file_recall@5': metric.compute_recall_at_k(gold_files, predicted_files, 5),
            'file_hit@5': metric.compute_hit_at_k(gold_files, predicted_files, 5)
        }
