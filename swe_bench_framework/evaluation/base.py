"""
Base classes for evaluation in the SWE-bench comparison framework.

This module defines the abstract base class and data structures for all
evaluation strategies, ensuring a unified interface across different evaluators.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from datetime import datetime


@dataclass
class EvaluationResult:
    """
    Result of evaluating a patch.
    
    Attributes:
        instance_id: Unique identifier for the SWE-bench instance
        resolved: Whether the patch successfully resolved the issue
        patch_applied: Whether the patch was successfully applied
        tests_passed: List of test names that passed
        tests_failed: List of test names that failed
        localization_accuracy: Dictionary with localization metrics
        codebleu_score: CodeBLEU similarity score (0.0-1.0)
        execution_time: Time taken for evaluation in seconds
        metadata: Additional metadata about the evaluation
    """
    instance_id: str
    resolved: bool
    patch_applied: bool
    tests_passed: List[str] = field(default_factory=list)
    tests_failed: List[str] = field(default_factory=list)
    localization_accuracy: Dict[str, float] = field(default_factory=dict)
    codebleu_score: float = 0.0
    execution_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate the result after initialization."""
        # Ensure lists are not None
        if self.tests_passed is None:
            self.tests_passed = []
        if self.tests_failed is None:
            self.tests_failed = []
        if self.localization_accuracy is None:
            self.localization_accuracy = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert EvaluationResult to dictionary for serialization."""
        return {
            'instance_id': self.instance_id,
            'resolved': self.resolved,
            'patch_applied': self.patch_applied,
            'tests_passed': self.tests_passed,
            'tests_failed': self.tests_failed,
            'localization_accuracy': self.localization_accuracy,
            'codebleu_score': self.codebleu_score,
            'execution_time': self.execution_time,
            'metadata': self.metadata,
            'timestamp': datetime.now().isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EvaluationResult':
        """Create EvaluationResult from dictionary."""
        # Remove timestamp if present
        data_copy = {k: v for k, v in data.items() if k != 'timestamp'}
        return cls(**data_copy)
    
    @property
    def resolution_rate(self) -> float:
        """Get resolution as a percentage (100.0 if resolved, 0.0 otherwise)."""
        return 100.0 if self.resolved else 0.0
    
    @property
    def total_tests(self) -> int:
        """Get total number of tests run."""
        return len(self.tests_passed) + len(self.tests_failed)
    
    @property
    def test_pass_rate(self) -> float:
        """Get percentage of tests that passed."""
        total = self.total_tests
        if total == 0:
            return 0.0
        return (len(self.tests_passed) / total) * 100.0


@dataclass
class TestResults:
    """Results from running tests."""
    passed: List[str] = field(default_factory=list)
    failed: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    skipped: List[str] = field(default_factory=list)
    execution_time: float = 0.0
    raw_output: str = ""
    
    @property
    def all_passed(self) -> bool:
        """Check if all tests passed."""
        return len(self.failed) == 0 and len(self.errors) == 0
    
    @property
    def total(self) -> int:
        """Get total number of tests."""
        return len(self.passed) + len(self.failed) + len(self.errors) + len(self.skipped)


class Evaluator(ABC):
    """
    Abstract base class for evaluation strategies.
    
    This class defines the interface that all evaluators must implement,
    including the main evaluate() method and localization accuracy computation.
    
    Example usage:
        evaluator = SWEBenchEvaluator(config)
        result = evaluator.evaluate(patch_result, instance, repo_path)
        if result.resolved:
            print(f"Instance {result.instance_id} resolved!")
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the evaluator.
        
        Args:
            config: Configuration for the evaluator
        """
        self.config = config
        self.name = self.__class__.__name__
        self._initialized = False
        self._metrics: List['Metric'] = []
    
    @abstractmethod
    def evaluate(
        self,
        patch_result: 'PatchResult',
        instance: 'SWEInstance',
        repo_path: str
    ) -> EvaluationResult:
        """
        Evaluate a generated patch.
        
        Args:
            patch_result: The result from patch generation
            instance: The SWE-bench instance
            repo_path: Path to the repository
            
        Returns:
            EvaluationResult with all metrics
            
        Raises:
            RuntimeError: If the evaluator has not been initialized
        """
        if not self._initialized:
            raise RuntimeError(f"{self.name} must be initialized before evaluation")
        pass
    
    @abstractmethod
    def compute_localization_accuracy(
        self,
        context_bundle: 'ContextBundle',
        gold_files: List[str],
        gold_functions: List[str]
    ) -> Dict[str, float]:
        """
        Compute localization accuracy metrics.
        
        Args:
            context_bundle: The gathered context
            gold_files: List of gold-standard modified files
            gold_functions: List of gold-standard modified functions
            
        Returns:
            Dictionary with localization metrics (e.g., Recall@K, Hit@K)
        """
        pass
    
    def initialize(self) -> None:
        """
        Initialize the evaluator. Override for any setup needed.
        
        This method should be called before the first call to evaluate().
        """
        self._initialized = True
    
    def cleanup(self) -> None:
        """
        Cleanup any resources. Override for custom cleanup.
        
        This method should be called when the evaluator is no longer needed.
        """
        self._initialized = False
    
    def register_metric(self, metric: 'Metric') -> None:
        """
        Register a metric collector.
        
        Args:
            metric: Metric instance to register
        """
        self._metrics.append(metric)
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about evaluation.
        
        Returns:
            Dictionary with evaluator statistics
        """
        return {
            'name': self.name,
            'initialized': self._initialized,
            'num_metrics': len(self._metrics),
            'config': self.config
        }


class Metric(ABC):
    """
    Abstract base class for evaluation metrics.
    
    Metrics compute specific aspects of evaluation results, such as
    resolution rate, localization accuracy, or code quality scores.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the metric.
        
        Args:
            config: Configuration for the metric
        """
        self.config = config or {}
        self.name = self.__class__.__name__
    
    @abstractmethod
    def compute(
        self,
        patch_result: 'PatchResult',
        instance: 'SWEInstance',
        evaluation_result: EvaluationResult
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
        pass


# Forward reference imports
from ..patch_generators.base import PatchResult  # noqa: E402
from ..context_gatherers.base import ContextBundle  # noqa: E402
from ..dataset.loader import SWEInstance  # noqa: E402
