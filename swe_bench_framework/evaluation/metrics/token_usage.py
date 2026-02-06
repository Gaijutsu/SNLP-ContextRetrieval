"""
Token usage metrics for the SWE-bench comparison framework.

This module implements metrics for tracking token consumption during
patch generation, enabling cost analysis and efficiency comparisons.
"""

from typing import Any, Dict, List, Optional
from collections import defaultdict

from .base import Metric, AggregateMetric


class TokenUsageMetric(AggregateMetric):
    """
    Metric for tracking token consumption.
    
    Tracks:
    - Prompt tokens (input to LLM)
    - Completion tokens (output from LLM)
    - Total tokens
    - Estimated cost (if pricing available)
    
    Example:
        metric = TokenUsageMetric()
        result = metric.compute(patch_result, instance, evaluation_result)
        # result = {'total_tokens': 1500, 'prompt_tokens': 1000, ...}
        
        # Get aggregate stats
        print(f"Avg tokens: {metric.get_mean_total_tokens()}")
    """
    
    # Default pricing per 1K tokens (approximate)
    DEFAULT_PRICING = {
        'gpt-5': {'prompt': 0.005, 'completion': 0.015},
        'gpt-5-mini': {'prompt': 0.00025, 'completion': 0.002},
        'gpt-5-pro': {'prompt': 0.01, 'completion': 0.03},
        'gpt-4o': {'prompt': 0.0025, 'completion': 0.01},
        'gpt-4o-mini': {'prompt': 0.00015, 'completion': 0.0006},
        'claude-4.5-sonnet': {'prompt': 0.003, 'completion': 0.015},
        'claude-4.5': {'prompt': 0.003, 'completion': 0.015},
    }
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the token usage metric.
        
        Args:
            config: Configuration dictionary
                - pricing: Custom pricing dictionary
                - track_by_model: Track usage by model (default: True)
                - track_by_instance: Track per-instance usage (default: False)
        """
        super().__init__(config)
        config = config or {}
        
        self.pricing = config.get('pricing', self.DEFAULT_PRICING)
        self.track_by_model = config.get('track_by_model', True)
        self.track_by_instance = config.get('track_by_instance', False)
        
        # Aggregate tracking
        self._total_prompt_tokens = 0
        self._total_completion_tokens = 0
        self._total_tokens = 0
        self._estimated_cost = 0.0
        
        # Per-model tracking
        self._model_stats: Dict[str, Dict[str, int]] = defaultdict(
            lambda: {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0, 'count': 0}
        )
        
        # Per-instance tracking
        self._instance_stats: Dict[str, Dict[str, Any]] = {}
    
    def compute(
        self,
        patch_result: 'PatchResult',
        instance: 'SWEInstance',
        evaluation_result: 'EvaluationResult'
    ) -> Dict[str, Any]:
        """
        Compute token usage metrics.
        
        Args:
            patch_result: The result from patch generation
            instance: The SWE-bench instance
            evaluation_result: The evaluation result
            
        Returns:
            Dictionary with token usage metrics
        """
        token_usage = getattr(patch_result, 'token_usage', {})
        
        prompt_tokens = token_usage.get('prompt_tokens', 0)
        completion_tokens = token_usage.get('completion_tokens', 0)
        total_tokens = token_usage.get('total_tokens', prompt_tokens + completion_tokens)
        
        # Update aggregates
        self._total_prompt_tokens += prompt_tokens
        self._total_completion_tokens += completion_tokens
        self._total_tokens += total_tokens
        
        # Get model name
        model = self._extract_model_name(patch_result)
        
        # Estimate cost
        cost = self._estimate_cost(model, prompt_tokens, completion_tokens)
        self._estimated_cost += cost
        
        # Track by model
        if self.track_by_model:
            self._model_stats[model]['prompt_tokens'] += prompt_tokens
            self._model_stats[model]['completion_tokens'] += completion_tokens
            self._model_stats[model]['total_tokens'] += total_tokens
            self._model_stats[model]['count'] += 1
        
        # Track by instance
        if self.track_by_instance:
            instance_id = instance.instance_id
            self._instance_stats[instance_id] = {
                'prompt_tokens': prompt_tokens,
                'completion_tokens': completion_tokens,
                'total_tokens': total_tokens,
                'estimated_cost': cost,
                'model': model
            }
        
        return {
            'prompt_tokens': prompt_tokens,
            'completion_tokens': completion_tokens,
            'total_tokens': total_tokens,
            'estimated_cost': cost,
            'model': model
        }
    
    def _extract_model_name(self, patch_result: 'PatchResult') -> str:
        """Extract model name from patch result metadata."""
        metadata = getattr(patch_result, 'metadata', {})
        model = metadata.get('model', 'unknown')
        
        # Clean up model name for pricing lookup
        if '/' in model:
            model = model.split('/')[-1]
        
        return model
    
    def _estimate_cost(
        self,
        model: str,
        prompt_tokens: int,
        completion_tokens: int
    ) -> float:
        """
        Estimate cost based on token usage.
        
        Args:
            model: Model name
            prompt_tokens: Number of prompt tokens
            completion_tokens: Number of completion tokens
            
        Returns:
            Estimated cost in USD
        """
        # Find matching pricing
        model_pricing = None
        for key, pricing in self.pricing.items():
            if key in model.lower():
                model_pricing = pricing
                break
        
        if not model_pricing:
            # Use default pricing
            model_pricing = {'prompt': 0.01, 'completion': 0.03}
        
        prompt_cost = (prompt_tokens / 1000) * model_pricing['prompt']
        completion_cost = (completion_tokens / 1000) * model_pricing['completion']
        
        return prompt_cost + completion_cost
    
    def get_name(self) -> str:
        """Get the metric name."""
        return 'token_usage'
    
    def get_total_tokens(self) -> int:
        """Get total tokens consumed."""
        return self._total_tokens
    
    def get_total_prompt_tokens(self) -> int:
        """Get total prompt tokens."""
        return self._total_prompt_tokens
    
    def get_total_completion_tokens(self) -> int:
        """Get total completion tokens."""
        return self._total_completion_tokens
    
    def get_estimated_cost(self) -> float:
        """Get estimated total cost."""
        return self._estimated_cost
    
    def get_average_tokens_per_instance(self) -> float:
        """Get average tokens per instance."""
        count = len(self._instance_stats) if self.track_by_instance else self.get_count()
        return self._total_tokens / count if count > 0 else 0.0
    
    def get_model_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get per-model statistics."""
        stats = {}
        for model, data in self._model_stats.items():
            count = data['count']
            stats[model] = {
                'total_prompt_tokens': data['prompt_tokens'],
                'total_completion_tokens': data['completion_tokens'],
                'total_tokens': data['total_tokens'],
                'instance_count': count,
                'avg_tokens_per_instance': data['total_tokens'] / count if count > 0 else 0
            }
        return stats
    
    def get_aggregate_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive token usage summary.
        
        Returns:
            Dictionary with aggregate statistics
        """
        count = self.get_count()
        
        summary = {
            'overall': {
                'total_prompt_tokens': self._total_prompt_tokens,
                'total_completion_tokens': self._total_completion_tokens,
                'total_tokens': self._total_tokens,
                'estimated_cost_usd': round(self._estimated_cost, 4),
                'instance_count': count,
                'avg_tokens_per_instance': round(self.get_average_tokens_per_instance(), 2),
                'avg_prompt_tokens': round(self._total_prompt_tokens / count, 2) if count > 0 else 0,
                'avg_completion_tokens': round(self._total_completion_tokens / count, 2) if count > 0 else 0
            }
        }
        
        if self.track_by_model:
            summary['by_model'] = self.get_model_stats()
        
        return summary
    
    def reset(self) -> None:
        """Reset all statistics."""
        super().reset()
        self._total_prompt_tokens = 0
        self._total_completion_tokens = 0
        self._total_tokens = 0
        self._estimated_cost = 0.0
        self._model_stats = defaultdict(
            lambda: {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0, 'count': 0}
        )
        self._instance_stats = {}


class CostEfficiencyMetric(Metric):
    """
    Metric for computing cost efficiency.
    
    Measures cost per resolved instance, providing insight into
the economic efficiency of different methods.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self._total_cost = 0.0
        self._resolved_count = 0
        self._total_count = 0
    
    def compute(
        self,
        patch_result: 'PatchResult',
        instance: 'SWEInstance',
        evaluation_result: 'EvaluationResult'
    ) -> Dict[str, Any]:
        """Compute cost efficiency metrics."""
        # Get cost from token usage
        token_usage = getattr(patch_result, 'token_usage', {})
        
        # Estimate cost (simplified)
        total_tokens = token_usage.get('total_tokens', 0)
        estimated_cost = (total_tokens / 1000) * 0.02  # Approximate cost
        
        self._total_cost += estimated_cost
        self._total_count += 1
        
        if evaluation_result.resolved:
            self._resolved_count += 1
        
        cost_per_resolved = (
            self._total_cost / self._resolved_count
            if self._resolved_count > 0 else float('inf')
        )
        
        return {
            'estimated_cost': estimated_cost,
            'total_cost': self._total_cost,
            'cost_per_resolved': cost_per_resolved,
            'resolved_count': self._resolved_count
        }
    
    def get_name(self) -> str:
        """Get the metric name."""
        return 'cost_efficiency'
    
    def reset(self) -> None:
        """Reset the metric."""
        super().reset()
        self._total_cost = 0.0
        self._resolved_count = 0
        self._total_count = 0


class IterationEfficiencyMetric(Metric):
    """
    Metric for tracking iteration efficiency.
    
    Measures how many iterations were needed to generate a valid patch.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self._iteration_counts: List[int] = []
        self._success_at_iteration: Dict[int, int] = defaultdict(int)
    
    def compute(
        self,
        patch_result: 'PatchResult',
        instance: 'SWEInstance',
        evaluation_result: 'EvaluationResult'
    ) -> Dict[str, Any]:
        """Compute iteration efficiency metrics."""
        attempts = getattr(patch_result, 'attempts', 1)
        self._iteration_counts.append(attempts)
        
        if evaluation_result.resolved:
            self._success_at_iteration[attempts] += 1
        
        return {
            'attempts': attempts,
            'avg_attempts': sum(self._iteration_counts) / len(self._iteration_counts),
            'max_attempts': max(self._iteration_counts) if self._iteration_counts else 0
        }
    
    def get_name(self) -> str:
        """Get the metric name."""
        return 'iteration_efficiency'
    
    def get_success_distribution(self) -> Dict[int, int]:
        """Get distribution of successful resolutions by iteration."""
        return dict(self._success_at_iteration)
    
    def reset(self) -> None:
        """Reset the metric."""
        super().reset()
        self._iteration_counts = []
        self._success_at_iteration = defaultdict(int)


# Forward reference imports
from ...patch_generators.base import PatchResult  # noqa: E402
from ...dataset.loader import SWEInstance  # noqa: E402
from ..base import EvaluationResult  # noqa: E402
