"""
Base classes for evaluation metrics in the SWE-bench comparison framework.

This module defines the abstract base class for all metrics, providing
a unified interface for computing various evaluation metrics.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class Metric(ABC):
    """
    Abstract base class for evaluation metrics.
    
    Metrics compute specific aspects of evaluation results, such as
    resolution rate, localization accuracy, or code quality scores.
    
    Example:
        class MyMetric(Metric):
            def compute(self, patch_result, instance, evaluation_result):
                return {'my_metric': some_value}
            
            def get_name(self):
                return 'my_metric'
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the metric.
        
        Args:
            config: Configuration for the metric
        """
        self.config = config or {}
        self.name = self.__class__.__name__
        self._state: Dict[str, Any] = {}
    
    @abstractmethod
    def compute(
        self,
        patch_result: 'PatchResult',
        instance: 'SWEInstance',
        evaluation_result: 'EvaluationResult'
    ) -> Dict[str, Any]:
        """
        Compute the metric.
        
        Args:
            patch_result: The result from patch generation
            instance: The SWE-bench instance
            evaluation_result: The evaluation result
            
        Returns:
            Dictionary with metric values
        """
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Get the metric name."""
        pass
    
    def reset(self) -> None:
        """Reset the metric state. Override if needed."""
        self._state = {}
    
    def get_state(self) -> Dict[str, Any]:
        """Get current metric state."""
        return self._state.copy()
    
    def aggregate(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Aggregate multiple metric results.
        
        Args:
            results: List of metric results
            
        Returns:
            Aggregated result
        """
        if not results:
            return {}
        
        # Default: compute mean for numeric values
        aggregated = {}
        keys = results[0].keys()
        
        for key in keys:
            values = [r[key] for r in results if key in r]
            if values and all(isinstance(v, (int, float)) for v in values):
                aggregated[f'{key}_mean'] = sum(values) / len(values)
                aggregated[f'{key}_min'] = min(values)
                aggregated[f'{key}_max'] = max(values)
            else:
                aggregated[key] = values
        
        return aggregated


class AggregateMetric(Metric):
    """
    Metric that aggregates values across multiple evaluations.
    
    This is useful for metrics that need to track state across
    multiple evaluations, such as average resolution rate.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self._values: List[float] = []
    
    def add_value(self, value: float) -> None:
        """Add a value to the aggregation."""
        self._values.append(value)
    
    def get_mean(self) -> float:
        """Get mean of all values."""
        if not self._values:
            return 0.0
        return sum(self._values) / len(self._values)
    
    def get_count(self) -> int:
        """Get count of values."""
        return len(self._values)
    
    def reset(self) -> None:
        """Reset the aggregation."""
        super().reset()
        self._values = []


# Forward reference imports
from ...patch_generators.base import PatchResult  # noqa: E402
from ...dataset.loader import SWEInstance  # noqa: E402
from ..base import EvaluationResult  # noqa: E402
