"""
Resolution metric for the SWE-bench comparison framework.

This module implements the primary evaluation metric: % Resolved (resolution rate),
which measures the percentage of instances where the generated patch successfully
resolves the issue.
"""

from typing import Any, Dict, List, Optional
from collections import defaultdict

from .base import Metric, AggregateMetric


class ResolutionMetric(AggregateMetric):
    """
    Metric for computing resolution rate (% Resolved).
    
    The resolution rate is the primary metric for SWE-bench evaluation,
    measuring the percentage of instances where:
    1. All FAIL_TO_PASS tests pass (issue is fixed)
    2. All PASS_TO_PASS tests still pass (no regressions)
    
    Example:
        metric = ResolutionMetric()
        result = metric.compute(patch_result, instance, evaluation_result)
        # result = {'resolved': True, 'resolution_rate': 100.0}
        
        # Aggregate across multiple evaluations
        metric.add_value(1.0 if evaluation_result.resolved else 0.0)
        print(f"Overall resolution rate: {metric.get_mean() * 100:.2f}%")
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the resolution metric.
        
        Args:
            config: Configuration dictionary
                - track_by_repo: Track resolution by repository (default: True)
                - track_by_difficulty: Track resolution by difficulty (default: False)
        """
        super().__init__(config)
        self.track_by_repo = config.get('track_by_repo', True) if config else True
        self.track_by_difficulty = config.get('track_by_difficulty', False) if config else False
        
        # Per-repository tracking
        self._repo_stats: Dict[str, Dict[str, int]] = defaultdict(
            lambda: {'resolved': 0, 'total': 0}
        )
        
        # Per-difficulty tracking
        self._difficulty_stats: Dict[str, Dict[str, int]] = defaultdict(
            lambda: {'resolved': 0, 'total': 0}
        )
    
    def compute(
        self,
        patch_result: 'PatchResult',
        instance: 'SWEInstance',
        evaluation_result: 'EvaluationResult'
    ) -> Dict[str, Any]:
        """
        Compute resolution metrics.
        
        Args:
            patch_result: The result from patch generation
            instance: The SWE-bench instance
            evaluation_result: The evaluation result
            
        Returns:
            Dictionary with resolution metrics
        """
        resolved = evaluation_result.resolved
        
        # Track for aggregation
        self.add_value(1.0 if resolved else 0.0)
        
        # Track by repository
        if self.track_by_repo:
            repo = getattr(instance, 'repo', 'unknown')
            self._repo_stats[repo]['total'] += 1
            if resolved:
                self._repo_stats[repo]['resolved'] += 1
        
        # Track by difficulty
        if self.track_by_difficulty:
            difficulty = self._estimate_difficulty(instance)
            self._difficulty_stats[difficulty]['total'] += 1
            if resolved:
                self._difficulty_stats[difficulty]['resolved'] += 1
        
        return {
            'resolved': resolved,
            'resolution_rate': 100.0 if resolved else 0.0,
            'patch_applied': evaluation_result.patch_applied
        }
    
    def get_name(self) -> str:
        """Get the metric name."""
        return 'resolution'
    
    def _estimate_difficulty(self, instance: 'SWEInstance') -> str:
        """
        Estimate difficulty of an instance.
        
        Args:
            instance: The SWE-bench instance
            
        Returns:
            Difficulty level ('easy', 'medium', 'hard')
        """
        # Simple heuristics based on problem statement
        problem = getattr(instance, 'problem_statement', '').lower()
        
        # Check for complexity indicators
        hard_indicators = ['refactor', 'architecture', 'redesign', 'complex', 'multiple']
        easy_indicators = ['typo', 'simple', 'minor', 'documentation', 'comment']
        
        for indicator in hard_indicators:
            if indicator in problem:
                return 'hard'
        
        for indicator in easy_indicators:
            if indicator in problem:
                return 'easy'
        
        return 'medium'
    
    def get_repo_stats(self) -> Dict[str, Dict[str, float]]:
        """
        Get resolution statistics by repository.
        
        Returns:
            Dictionary mapping repo to stats
        """
        stats = {}
        for repo, counts in self._repo_stats.items():
            total = counts['total']
            resolved = counts['resolved']
            stats[repo] = {
                'total': total,
                'resolved': resolved,
                'resolution_rate': (resolved / total * 100) if total > 0 else 0.0
            }
        return stats
    
    def get_difficulty_stats(self) -> Dict[str, Dict[str, float]]:
        """
        Get resolution statistics by difficulty.
        
        Returns:
            Dictionary mapping difficulty to stats
        """
        stats = {}
        for difficulty, counts in self._difficulty_stats.items():
            total = counts['total']
            resolved = counts['resolved']
            stats[difficulty] = {
                'total': total,
                'resolved': resolved,
                'resolution_rate': (resolved / total * 100) if total > 0 else 0.0
            }
        return stats
    
    def get_aggregate_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive aggregate summary.
        
        Returns:
            Dictionary with overall and per-category statistics
        """
        total = self.get_count()
        resolved_count = int(sum(self._values))
        
        summary = {
            'overall': {
                'total': total,
                'resolved': resolved_count,
                'unresolved': total - resolved_count,
                'resolution_rate': self.get_mean() * 100
            }
        }
        
        if self.track_by_repo:
            summary['by_repo'] = self.get_repo_stats()
        
        if self.track_by_difficulty:
            summary['by_difficulty'] = self.get_difficulty_stats()
        
        return summary
    
    def reset(self) -> None:
        """Reset all statistics."""
        super().reset()
        self._repo_stats = defaultdict(lambda: {'resolved': 0, 'total': 0})
        self._difficulty_stats = defaultdict(lambda: {'resolved': 0, 'total': 0})


class PassAtKMetric(Metric):
    """
    Metric for computing pass@k (resolution within k attempts).
    
    pass@k measures whether at least one of k generated patches
    successfully resolves the issue.
    
    Example:
        metric = PassAtKMetric(k_values=[1, 3, 5])
        # For each instance, call compute with multiple attempts
        result = metric.compute_with_attempts(instance_id, [patch1, patch2, patch3])
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the pass@k metric.
        
        Args:
            config: Configuration dictionary
                - k_values: List of k values to compute (default: [1, 3, 5])
        """
        super().__init__(config)
        self.k_values = config.get('k_values', [1, 3, 5]) if config else [1, 3, 5]
        
        # Track results per instance
        self._instance_results: Dict[str, List[bool]] = {}
    
    def compute(
        self,
        patch_result: 'PatchResult',
        instance: 'SWEInstance',
        evaluation_result: 'EvaluationResult'
    ) -> Dict[str, Any]:
        """
        Compute pass@k for a single attempt.
        
        Note: For proper pass@k calculation, use compute_with_attempts()
        to provide multiple attempts for the same instance.
        """
        instance_id = instance.instance_id
        resolved = evaluation_result.resolved
        
        # Track this attempt
        if instance_id not in self._instance_results:
            self._instance_results[instance_id] = []
        self._instance_results[instance_id].append(resolved)
        
        # Compute pass@k based on attempts so far
        return self._compute_pass_at_k(instance_id)
    
    def compute_with_attempts(
        self,
        instance_id: str,
        resolved_attempts: List[bool]
    ) -> Dict[str, Any]:
        """
        Compute pass@k given multiple attempts.
        
        Args:
            instance_id: Unique instance identifier
            resolved_attempts: List of resolution results for each attempt
            
        Returns:
            Dictionary with pass@k values
        """
        self._instance_results[instance_id] = resolved_attempts
        return self._compute_pass_at_k(instance_id)
    
    def _compute_pass_at_k(self, instance_id: str) -> Dict[str, Any]:
        """Compute pass@k for an instance."""
        attempts = self._instance_results.get(instance_id, [])
        
        results = {}
        for k in self.k_values:
            if len(attempts) >= k:
                # At least one of first k attempts resolved
                pass_at_k = any(attempts[:k])
                results[f'pass@{k}'] = pass_at_k
            else:
                results[f'pass@{k}'] = None  # Not enough attempts
        
        results['num_attempts'] = len(attempts)
        results['any_resolved'] = any(attempts)
        
        return results
    
    def get_name(self) -> str:
        """Get the metric name."""
        return 'pass_at_k'
    
    def get_aggregate_pass_at_k(self) -> Dict[str, float]:
        """
        Get aggregate pass@k across all instances.
        
        Returns:
            Dictionary with pass@k percentages
        """
        results = {}
        
        for k in self.k_values:
            passed = sum(
                1 for attempts in self._instance_results.values()
                if len(attempts) >= k and any(attempts[:k])
            )
            total = sum(
                1 for attempts in self._instance_results.values()
                if len(attempts) >= k
            )
            
            results[f'pass@{k}'] = (passed / total * 100) if total > 0 else 0.0
        
        return results
    
    def reset(self) -> None:
        """Reset the metric."""
        super().reset()
        self._instance_results = {}


class PatchApplyRateMetric(Metric):
    """
    Metric for computing patch application rate.
    
    Measures the percentage of generated patches that can be
    successfully applied to the codebase.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self._applied = 0
        self._total = 0
    
    def compute(
        self,
        patch_result: 'PatchResult',
        instance: 'SWEInstance',
        evaluation_result: 'EvaluationResult'
    ) -> Dict[str, Any]:
        """Compute patch apply rate."""
        applied = evaluation_result.patch_applied
        
        self._total += 1
        if applied:
            self._applied += 1
        
        return {
            'patch_applied': applied,
            'apply_rate': (self._applied / self._total * 100) if self._total > 0 else 0.0
        }
    
    def get_name(self) -> str:
        """Get the metric name."""
        return 'patch_apply_rate'
    
    def get_apply_rate(self) -> float:
        """Get overall patch apply rate."""
        return (self._applied / self._total * 100) if self._total > 0 else 0.0
    
    def reset(self) -> None:
        """Reset the metric."""
        super().reset()
        self._applied = 0
        self._total = 0


# Forward reference imports
from ...patch_generators.base import PatchResult  # noqa: E402
from ...dataset.loader import SWEInstance  # noqa: E402
from ..base import EvaluationResult  # noqa: E402
