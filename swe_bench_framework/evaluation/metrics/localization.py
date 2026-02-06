"""
Localization metrics for the SWE-bench comparison framework.

This module implements fault localization metrics including Recall@K and Hit@K
for files, functions, and code snippets.
"""

from typing import Any, Dict, List, Optional, Set, Tuple
from collections import defaultdict

from .base import Metric, AggregateMetric


class LocalizationMetric(AggregateMetric):
    """
    Metric for computing localization accuracy (Recall@K, Hit@K).
    
    Localization metrics measure how well a method identifies the correct
    locations that need to be modified to fix an issue.
    
    Metrics computed:
    - Recall@K: Percentage of gold elements in top-K predictions
    - Hit@K: Binary indicator if any gold element appears in top-K
    
    Example:
        metric = LocalizationMetric(k_values=[1, 3, 5, 10])
        
        # Compute file-level metrics
        recall_at_5 = metric.compute_recall_at_k(gold_files, predicted_files, 5)
        hit_at_5 = metric.compute_hit_at_k(gold_files, predicted_files, 5)
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the localization metric.
        
        Args:
            config: Configuration dictionary
                - k_values: List of K values to compute (default: [1, 3, 5, 10])
                - levels: List of levels to track ('file', 'function', 'snippet')
        """
        super().__init__(config)
        self.k_values = config.get('k_values', [1, 3, 5, 10]) if config else [1, 3, 5, 10]
        self.levels = config.get('levels', ['file', 'function', 'snippet']) if config else ['file', 'function', 'snippet']
        
        # Track results per level
        self._file_results: List[Dict[str, Any]] = []
        self._function_results: List[Dict[str, Any]] = []
        self._snippet_results: List[Dict[str, Any]] = []
    
    def compute(
        self,
        patch_result: 'PatchResult',
        instance: 'SWEInstance',
        evaluation_result: 'EvaluationResult'
    ) -> Dict[str, Any]:
        """
        Compute localization metrics.
        
        Args:
            patch_result: The result from patch generation
            instance: The SWE-bench instance
            evaluation_result: The evaluation result
            
        Returns:
            Dictionary with localization metrics
        """
        # Get gold standard from instance
        gold_files = getattr(instance, 'modified_files', [])
        gold_functions = getattr(instance, 'modified_methods', [])
        
        # Get predictions from context bundle if available
        predicted_files = []
        predicted_functions = []
        
        context_bundle = getattr(patch_result, 'context_bundle', None)
        if context_bundle:
            for chunk in context_bundle.chunks:
                if chunk.source_file and chunk.source_file not in predicted_files:
                    predicted_files.append(chunk.source_file)
                
                # Extract function names from metadata
                if hasattr(chunk, 'metadata') and chunk.metadata:
                    func_name = chunk.metadata.get('function_name')
                    if func_name and func_name not in predicted_functions:
                        predicted_functions.append(func_name)
        
        results = {}
        
        # Compute file-level metrics
        if 'file' in self.levels and gold_files:
            for k in self.k_values:
                results[f'file_recall@{k}'] = self.compute_recall_at_k(
                    gold_files, predicted_files, k
                )
                results[f'file_hit@{k}'] = self.compute_hit_at_k(
                    gold_files, predicted_files, k
                )
        
        # Compute function-level metrics
        if 'function' in self.levels and gold_functions:
            for k in self.k_values:
                results[f'function_recall@{k}'] = self.compute_recall_at_k(
                    gold_functions, predicted_functions, k
                )
                results[f'function_hit@{k}'] = self.compute_hit_at_k(
                    gold_functions, predicted_functions, k
                )
        
        # Store results for aggregation
        self._file_results.append({
            'gold': gold_files,
            'predicted': predicted_files
        })
        
        if gold_functions:
            self._function_results.append({
                'gold': gold_functions,
                'predicted': predicted_functions
            })
        
        return results
    
    def compute_recall_at_k(
        self,
        gold: List[str],
        predicted: List[str],
        k: int
    ) -> float:
        """
        Compute Recall@K.
        
        Recall@K = (Number of gold elements in top-K predictions) / (Total gold elements)
        
        Args:
            gold: List of gold-standard elements
            predicted: List of predicted elements (ordered by confidence)
            k: Number of top predictions to consider
            
        Returns:
            Recall@K as a percentage (0.0-100.0)
        """
        if not gold:
            return 0.0
        
        # Get top-K predictions
        top_k = predicted[:k]
        
        # Count gold elements in top-K
        gold_set = set(gold)
        top_k_set = set(top_k)
        
        correct = len(gold_set & top_k_set)
        
        return (correct / len(gold_set)) * 100.0
    
    def compute_hit_at_k(
        self,
        gold: List[str],
        predicted: List[str],
        k: int
    ) -> float:
        """
        Compute Hit@K.
        
        Hit@K = 1 if any gold element appears in top-K predictions, else 0
        
        Args:
            gold: List of gold-standard elements
            predicted: List of predicted elements (ordered by confidence)
            k: Number of top predictions to consider
            
        Returns:
            Hit@K as a binary value (0.0 or 100.0)
        """
        if not gold:
            return 0.0
        
        # Get top-K predictions
        top_k = predicted[:k]
        
        # Check if any gold element is in top-K
        gold_set = set(gold)
        top_k_set = set(top_k)
        
        return 100.0 if gold_set & top_k_set else 0.0
    
    def compute_mean_reciprocal_rank(
        self,
        gold: List[str],
        predicted: List[str]
    ) -> float:
        """
        Compute Mean Reciprocal Rank (MRR).
        
        MRR = (1 / |Q|) * sum(1 / rank_i) for each query
        
        Args:
            gold: List of gold-standard elements
            predicted: List of predicted elements (ordered by confidence)
            
        Returns:
            MRR value (0.0-1.0)
        """
        if not gold:
            return 0.0
        
        reciprocal_ranks = []
        
        for gold_item in gold:
            try:
                rank = predicted.index(gold_item) + 1  # 1-indexed
                reciprocal_ranks.append(1.0 / rank)
            except ValueError:
                # Gold item not in predictions
                reciprocal_ranks.append(0.0)
        
        return sum(reciprocal_ranks) / len(gold)
    
    def compute_top_k_accuracy(
        self,
        gold: List[str],
        predicted: List[str],
        k: int
    ) -> float:
        """
        Compute top-K accuracy (all gold elements must be in top-K).
        
        Args:
            gold: List of gold-standard elements
            predicted: List of predicted elements
            k: Number of top predictions to consider
            
        Returns:
            Accuracy as percentage (0.0 or 100.0)
        """
        if not gold:
            return 0.0
        
        top_k = set(predicted[:k])
        gold_set = set(gold)
        
        return 100.0 if gold_set.issubset(top_k) else 0.0
    
    def get_name(self) -> str:
        """Get the metric name."""
        return 'localization'
    
    def get_aggregate_stats(self) -> Dict[str, Any]:
        """
        Get aggregate localization statistics.
        
        Returns:
            Dictionary with aggregate statistics
        """
        stats = {}
        
        # File-level aggregate
        if self._file_results:
            stats['file'] = self._aggregate_level(self._file_results)
        
        # Function-level aggregate
        if self._function_results:
            stats['function'] = self._aggregate_level(self._function_results)
        
        return stats
    
    def _aggregate_level(
        self,
        results: List[Dict[str, List[str]]]
    ) -> Dict[str, float]:
        """Aggregate metrics for a level."""
        aggregated = {}
        
        for k in self.k_values:
            recall_values = [
                self.compute_recall_at_k(r['gold'], r['predicted'], k)
                for r in results
            ]
            hit_values = [
                self.compute_hit_at_k(r['gold'], r['predicted'], k)
                for r in results
            ]
            
            aggregated[f'recall@{k}_mean'] = sum(recall_values) / len(recall_values)
            aggregated[f'hit@{k}_mean'] = sum(hit_values) / len(hit_values)
        
        # MRR
        mrr_values = [
            self.compute_mean_reciprocal_rank(r['gold'], r['predicted'])
            for r in results
        ]
        aggregated['mrr_mean'] = sum(mrr_values) / len(mrr_values)
        
        return aggregated
    
    def reset(self) -> None:
        """Reset the metric."""
        super().reset()
        self._file_results = []
        self._function_results = []
        self._snippet_results = []


class ExactMatchMetric(Metric):
    """
    Metric for computing exact match accuracy.
    
    Measures whether the predicted files/functions exactly match
the gold standard.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self._exact_matches = 0
        self._total = 0
    
    def compute(
        self,
        patch_result: 'PatchResult',
        instance: 'SWEInstance',
        evaluation_result: 'EvaluationResult'
    ) -> Dict[str, Any]:
        """Compute exact match."""
        gold_files = set(getattr(instance, 'modified_files', []))
        
        # Get predicted files
        predicted_files = set()
        context_bundle = getattr(patch_result, 'context_bundle', None)
        if context_bundle:
            for chunk in context_bundle.chunks:
                if chunk.source_file:
                    predicted_files.add(chunk.source_file)
        
        # Check exact match
        exact_match = gold_files == predicted_files
        
        self._total += 1
        if exact_match:
            self._exact_matches += 1
        
        return {
            'exact_match': exact_match,
            'exact_match_rate': (self._exact_matches / self._total * 100)
        }
    
    def get_name(self) -> str:
        """Get the metric name."""
        return 'exact_match'
    
    def reset(self) -> None:
        """Reset the metric."""
        super().reset()
        self._exact_matches = 0
        self._total = 0


class LineLevelLocalizationMetric(Metric):
    """
    Metric for line-level localization accuracy.
    
    Measures how accurately the predicted line ranges match
the actual modified lines.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.tolerance = config.get('tolerance', 3) if config else 3
    
    def compute(
        self,
        patch_result: 'PatchResult',
        instance: 'SWEInstance',
        evaluation_result: 'EvaluationResult'
    ) -> Dict[str, Any]:
        """
        Compute line-level localization accuracy.
        
        This requires parsing the gold patch to extract modified line ranges.
        """
        gold_patch = getattr(instance, 'patch', None)
        if not gold_patch:
            return {'line_accuracy': 0.0}
        
        # Parse gold patch to get modified lines
        gold_lines = self._parse_patch_lines(gold_patch)
        
        # Get predicted line ranges from context
        predicted_ranges = []
        context_bundle = getattr(patch_result, 'context_bundle', None)
        if context_bundle:
            for chunk in context_bundle.chunks:
                if chunk.start_line and chunk.end_line:
                    predicted_ranges.append((chunk.start_line, chunk.end_line))
        
        # Compute overlap
        if not gold_lines or not predicted_ranges:
            return {'line_accuracy': 0.0}
        
        overlap = self._compute_line_overlap(gold_lines, predicted_ranges)
        
        return {
            'line_accuracy': overlap * 100,
            'gold_lines': len(gold_lines),
            'predicted_ranges': len(predicted_ranges)
        }
    
    def _parse_patch_lines(self, patch: str) -> Set[int]:
        """Parse patch to extract modified line numbers."""
        lines = set()
        current_line = 0
        
        for line in patch.split('\n'):
            if line.startswith('@@'):
                # Parse hunk header: @@ -start,count +start,count @@
                match = __import__('re').match(r'@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@', line)
                if match:
                    current_line = int(match.group(2))
            elif line.startswith('+') and not line.startswith('+++'):
                lines.add(current_line)
                current_line += 1
            elif line.startswith(' ') or line.startswith('-'):
                if not line.startswith('---'):
                    current_line += 1
        
        return lines
    
    def _compute_line_overlap(
        self,
        gold_lines: Set[int],
        predicted_ranges: List[Tuple[int, int]]
    ) -> float:
        """Compute overlap between gold lines and predicted ranges."""
        # Create set of all lines in predicted ranges
        predicted_lines = set()
        for start, end in predicted_ranges:
            predicted_lines.update(range(start, end + 1))
        
        # Compute overlap with tolerance
        covered_gold = set()
        for gold_line in gold_lines:
            for pred_line in predicted_lines:
                if abs(gold_line - pred_line) <= self.tolerance:
                    covered_gold.add(gold_line)
                    break
        
        return len(covered_gold) / len(gold_lines) if gold_lines else 0.0
    
    def get_name(self) -> str:
        """Get the metric name."""
        return 'line_level_localization'


# Forward reference imports
from ...patch_generators.base import PatchResult  # noqa: E402
from ...dataset.loader import SWEInstance  # noqa: E402
from ..base import EvaluationResult  # noqa: E402
