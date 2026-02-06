"""
Report generator for the SWE-bench comparison framework.

This module provides utilities for generating comparison reports between
different methods, including statistical analysis and visualization preparation.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict

from .base import EvaluationResult


@dataclass
class MethodResults:
    """Results for a single method."""
    method_name: str
    results: List[EvaluationResult]
    config: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def total_instances(self) -> int:
        """Get total number of instances."""
        return len(self.results)
    
    @property
    def resolved_count(self) -> int:
        """Get number of resolved instances."""
        return sum(1 for r in self.results if r.resolved)
    
    @property
    def resolution_rate(self) -> float:
        """Get resolution rate as percentage."""
        if not self.results:
            return 0.0
        return (self.resolved_count / self.total_instances) * 100


class ReportGenerator:
    """
    Generator for comparison reports between methods.
    
    This class provides methods for:
    - Aggregating results from multiple methods
    - Computing statistical comparisons
    - Generating JSON/Markdown reports
    - Preparing data for visualization
    
    Example:
        generator = ReportGenerator()
        
        # Add results from different methods
        generator.add_method_results("Method A", method_a_results)
        generator.add_method_results("Method B", method_b_results)
        
        # Generate report
        report = generator.generate_comparison_report()
        generator.save_report(report, "output/report.json")
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the report generator.
        
        Args:
            config: Configuration dictionary
                - output_format: Output format ('json', 'markdown', 'html')
                - include_confidence_intervals: Include confidence intervals
                - confidence_level: Confidence level (default: 0.95)
        """
        self.config = config or {}
        self.output_format = self.config.get('output_format', 'json')
        self.include_confidence_intervals = self.config.get(
            'include_confidence_intervals', True
        )
        self.confidence_level = self.config.get('confidence_level', 0.95)
        
        # Store results by method
        self._method_results: Dict[str, MethodResults] = {}
        
        # Report metadata
        self._report_metadata = {
            'generated_at': datetime.now().isoformat(),
            'version': '1.0.0'
        }
    
    def add_method_results(
        self,
        method_name: str,
        results: List[EvaluationResult],
        config: Dict[str, Any] = None,
        metadata: Dict[str, Any] = None
    ) -> None:
        """
        Add results for a method.
        
        Args:
            method_name: Name of the method
            results: List of evaluation results
            config: Method configuration
            metadata: Additional metadata
        """
        self._method_results[method_name] = MethodResults(
            method_name=method_name,
            results=results,
            config=config or {},
            metadata=metadata or {}
        )
    
    def generate_comparison_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive comparison report.
        
        Returns:
            Dictionary containing the full comparison report
        """
        if not self._method_results:
            return {'error': 'No results added'}
        
        report = {
            'metadata': self._report_metadata,
            'summary': self._generate_summary(),
            'method_comparison': self._generate_method_comparison(),
            'statistical_analysis': self._generate_statistical_analysis(),
            'detailed_results': self._generate_detailed_results(),
            'visualization_data': self._generate_visualization_data()
        }
        
        return report
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate overall summary."""
        methods = list(self._method_results.keys())
        
        summary = {
            'num_methods': len(methods),
            'methods': methods,
            'total_instances': self._get_total_instances(),
            'best_method': self._get_best_method(),
            'resolution_rates': {
                name: results.resolution_rate
                for name, results in self._method_results.items()
            }
        }
        
        return summary
    
    def _generate_method_comparison(self) -> Dict[str, Any]:
        """Generate method-by-method comparison."""
        comparison = {}
        
        for method_name, method_results in self._method_results.items():
            comparison[method_name] = {
                'total_instances': method_results.total_instances,
                'resolved': method_results.resolved_count,
                'unresolved': method_results.total_instances - method_results.resolved_count,
                'resolution_rate': round(method_results.resolution_rate, 2),
                'patch_apply_rate': self._compute_patch_apply_rate(method_results),
                'avg_execution_time': self._compute_avg_execution_time(method_results),
                'avg_codebleu': self._compute_avg_codebleu(method_results),
                'localization': self._compute_avg_localization(method_results),
                'config': method_results.config
            }
        
        return comparison
    
    def _generate_statistical_analysis(self) -> Dict[str, Any]:
        """Generate statistical analysis."""
        analysis = {
            'significance_tests': self._run_significance_tests(),
            'confidence_intervals': self._compute_confidence_intervals(),
            'effect_sizes': self._compute_effect_sizes()
        }
        
        return analysis
    
    def _run_significance_tests(self) -> Dict[str, Any]:
        """Run statistical significance tests between methods."""
        from itertools import combinations
        
        methods = list(self._method_results.keys())
        tests = {}
        
        for method1, method2 in combinations(methods, 2):
            test_name = f"{method1}_vs_{method2}"
            tests[test_name] = self._compare_two_methods(method1, method2)
        
        return tests
    
    def _compare_two_methods(
        self,
        method1: str,
        method2: str
    ) -> Dict[str, Any]:
        """Compare two methods statistically."""
        results1 = self._method_results[method1]
        results2 = self._method_results[method2]
        
        # Get resolution outcomes (1 for resolved, 0 for not)
        outcomes1 = [1 if r.resolved else 0 for r in results1.results]
        outcomes2 = [1 if r.resolved else 0 for r in results2.results]
        
        # Chi-square test for proportions
        try:
            from scipy import stats
            
            # Create contingency table
            table = [
                [sum(outcomes1), len(outcomes1) - sum(outcomes1)],
                [sum(outcomes2), len(outcomes2) - sum(outcomes2)]
            ]
            
            chi2, p_value, dof, expected = stats.chi2_contingency(table)
            
            return {
                'method1_resolution_rate': results1.resolution_rate,
                'method2_resolution_rate': results2.resolution_rate,
                'difference': results1.resolution_rate - results2.resolution_rate,
                'chi2_statistic': chi2,
                'p_value': p_value,
                'significant': p_value < 0.05,
                'significance_level': '***' if p_value < 0.001 else ('**' if p_value < 0.01 else ('*' if p_value < 0.05 else 'ns'))
            }
        except ImportError:
            # Fallback without scipy
            return {
                'method1_resolution_rate': results1.resolution_rate,
                'method2_resolution_rate': results2.resolution_rate,
                'difference': results1.resolution_rate - results2.resolution_rate,
                'note': 'scipy not available for statistical testing'
            }
    
    def _compute_confidence_intervals(self) -> Dict[str, Any]:
        """Compute confidence intervals for resolution rates."""
        intervals = {}
        
        for method_name, method_results in self._method_results.items():
            n = method_results.total_instances
            k = method_results.resolved_count
            p = k / n if n > 0 else 0
            
            # Wilson score interval (only valid for n >= 1)
            if n < 1:
                intervals[method_name] = {
                    'point_estimate': 0.0,
                    'lower_bound': 0.0,
                    'upper_bound': 0.0,
                    'confidence_level': 0.95,
                    'note': 'Insufficient data for confidence interval'
                }
                continue
            
            z = 1.96  # 95% confidence
            
            denominator = 1 + z**2 / n
            center = (p + z**2 / (2 * n)) / denominator
            margin = z * ((p * (1 - p) / n + z**2 / (4 * n**2)) ** 0.5) / denominator
            
            intervals[method_name] = {
                'point_estimate': round(p * 100, 2),
                'lower_bound': round(max(0, center - margin) * 100, 2),
                'upper_bound': round(min(1, center + margin) * 100, 2),
                'confidence_level': 0.95
            }
        
        return intervals
    
    def _compute_effect_sizes(self) -> Dict[str, Any]:
        """Compute effect sizes between methods."""
        from itertools import combinations
        
        effect_sizes = {}
        
        for method1, method2 in combinations(self._method_results.keys(), 2):
            results1 = self._method_results[method1]
            results2 = self._method_results[method2]
            
            # Cohen's h for proportions
            p1 = results1.resolution_rate / 100
            p2 = results2.resolution_rate / 100
            
            # Arcsine transformation
            import math
            h = 2 * (math.asin(p1**0.5) - math.asin(p2**0.5))
            
            effect_sizes[f"{method1}_vs_{method2}"] = {
                'cohens_h': round(h, 4),
                'interpretation': self._interpret_effect_size(abs(h))
            }
        
        return effect_sizes
    
    def _interpret_effect_size(self, h: float) -> str:
        """Interpret Cohen's h effect size."""
        if h < 0.2:
            return 'negligible'
        elif h < 0.5:
            return 'small'
        elif h < 0.8:
            return 'medium'
        else:
            return 'large'
    
    def _generate_detailed_results(self) -> Dict[str, Any]:
        """Generate detailed per-instance results."""
        detailed = {}
        
        for method_name, method_results in self._method_results.items():
            detailed[method_name] = [
                {
                    'instance_id': r.instance_id,
                    'resolved': r.resolved,
                    'patch_applied': r.patch_applied,
                    'tests_passed': len(r.tests_passed),
                    'tests_failed': len(r.tests_failed),
                    'codebleu_score': r.codebleu_score,
                    'execution_time': r.execution_time
                }
                for r in method_results.results
            ]
        
        return detailed
    
    def _generate_visualization_data(self) -> Dict[str, Any]:
        """Generate data for visualization."""
        viz_data = {
            'resolution_comparison': self._get_resolution_comparison_data(),
            'localization_comparison': self._get_localization_comparison_data(),
            'time_comparison': self._get_time_comparison_data(),
            'per_instance_comparison': self._get_per_instance_comparison_data()
        }
        
        return viz_data
    
    def _get_resolution_comparison_data(self) -> Dict[str, Any]:
        """Get data for resolution rate comparison chart."""
        return {
            'labels': list(self._method_results.keys()),
            'values': [
                results.resolution_rate
                for results in self._method_results.values()
            ]
        }
    
    def _get_localization_comparison_data(self) -> Dict[str, Any]:
        """Get data for localization comparison chart."""
        data = {}
        
        for method_name, method_results in self._method_results.items():
            # Aggregate localization metrics
            loc_metrics = defaultdict(list)
            
            for result in method_results.results:
                for key, value in result.localization_accuracy.items():
                    loc_metrics[key].append(value)
            
            # Compute averages
            data[method_name] = {
                key: sum(values) / len(values) if values else 0
                for key, values in loc_metrics.items()
            }
        
        return data
    
    def _get_time_comparison_data(self) -> Dict[str, Any]:
        """Get data for execution time comparison."""
        return {
            'labels': list(self._method_results.keys()),
            'avg_times': [
                self._compute_avg_execution_time(results)
                for results in self._method_results.values()
            ]
        }
    
    def _get_per_instance_comparison_data(self) -> Dict[str, Any]:
        """Get per-instance comparison data."""
        # Get all unique instance IDs
        all_instances = set()
        for results in self._method_results.values():
            all_instances.update(r.instance_id for r in results.results)
        
        # Build comparison table
        comparison = []
        for instance_id in sorted(all_instances):
            row = {'instance_id': instance_id}
            
            for method_name, method_results in self._method_results.items():
                result = next(
                    (r for r in method_results.results if r.instance_id == instance_id),
                    None
                )
                
                if result:
                    row[method_name] = {
                        'resolved': result.resolved,
                        'patch_applied': result.patch_applied,
                        'codebleu': result.codebleu_score
                    }
                else:
                    row[method_name] = None
            
            comparison.append(row)
        
        return comparison
    
    def _get_total_instances(self) -> int:
        """Get total number of unique instances across all methods."""
        all_instances = set()
        for results in self._method_results.values():
            all_instances.update(r.instance_id for r in results.results)
        return len(all_instances)
    
    def _get_best_method(self) -> Optional[str]:
        """Get the method with highest resolution rate."""
        if not self._method_results:
            return None
        
        return max(
            self._method_results.items(),
            key=lambda x: x[1].resolution_rate
        )[0]
    
    def _compute_patch_apply_rate(self, method_results: MethodResults) -> float:
        """Compute patch apply rate for a method."""
        if not method_results.results:
            return 0.0
        
        applied = sum(1 for r in method_results.results if r.patch_applied)
        return (applied / len(method_results.results)) * 100
    
    def _compute_avg_execution_time(self, method_results: MethodResults) -> float:
        """Compute average execution time for a method."""
        if not method_results.results:
            return 0.0
        
        times = [r.execution_time for r in method_results.results]
        return sum(times) / len(times)
    
    def _compute_avg_codebleu(self, method_results: MethodResults) -> float:
        """Compute average CodeBLEU score for a method."""
        if not method_results.results:
            return 0.0
        
        scores = [r.codebleu_score for r in method_results.results]
        return sum(scores) / len(scores)
    
    def _compute_avg_localization(
        self,
        method_results: MethodResults
    ) -> Dict[str, float]:
        """Compute average localization metrics for a method."""
        if not method_results.results:
            return {}
        
        # Aggregate all localization metrics
        all_metrics = defaultdict(list)
        
        for result in method_results.results:
            for key, value in result.localization_accuracy.items():
                all_metrics[key].append(value)
        
        # Compute averages
        return {
            key: round(sum(values) / len(values), 2) if values else 0.0
            for key, values in all_metrics.items()
        }
    
    def save_report(
        self,
        report: Dict[str, Any],
        output_path: str,
        format: str = 'json'
    ) -> None:
        """
        Save report to file.
        
        Args:
            report: The report dictionary
            output_path: Path to save the report
            format: Output format ('json', 'markdown')
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == 'json':
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2)
        
        elif format == 'markdown':
            markdown = self._convert_to_markdown(report)
            with open(output_path, 'w') as f:
                f.write(markdown)
        
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def _convert_to_markdown(self, report: Dict[str, Any]) -> str:
        """Convert report to Markdown format."""
        lines = []
        
        # Title
        lines.append("# SWE-bench Comparison Report\n")
        
        # Metadata
        lines.append(f"**Generated:** {report['metadata']['generated_at']}\n")
        
        # Summary
        summary = report['summary']
        lines.append("## Summary\n")
        lines.append(f"- **Methods Compared:** {summary['num_methods']}")
        lines.append(f"- **Total Instances:** {summary['total_instances']}")
        lines.append(f"- **Best Method:** {summary['best_method']}\n")
        
        # Method Comparison
        lines.append("## Method Comparison\n")
        lines.append("| Method | Instances | Resolved | Rate (%) | Avg Time (s) |")
        lines.append("|--------|-----------|----------|----------|-------------|")
        
        for method, data in report['method_comparison'].items():
            lines.append(
                f"| {method} | {data['total_instances']} | "
                f"{data['resolved']} | {data['resolution_rate']:.2f} | "
                f"{data['avg_execution_time']:.2f} |"
            )
        
        lines.append("")
        
        # Statistical Analysis
        lines.append("## Statistical Analysis\n")
        
        for test_name, test_result in report['statistical_analysis']['significance_tests'].items():
            lines.append(f"### {test_name}\n")
            lines.append(f"- Difference: {test_result.get('difference', 'N/A'):.2f}%")
            lines.append(f"- P-value: {test_result.get('p_value', 'N/A')}")
            lines.append(f"- Significant: {test_result.get('significant', 'N/A')}\n")
        
        return '\n'.join(lines)
    
    def reset(self) -> None:
        """Reset the report generator."""
        self._method_results = {}
        self._report_metadata = {
            'generated_at': datetime.now().isoformat(),
            'version': '1.0.0'
        }
