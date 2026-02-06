"""
Evaluation module for the SWE-bench comparison framework.
"""

from .base import Evaluator, EvaluationResult, TestResults, Metric
from .swe_bench_evaluator import SWEBenchEvaluator, LightweightEvaluator
from .docker_sandbox import DockerSandbox, LocalSandbox, SandboxConfig
from .report_generator import ReportGenerator, MethodResults

__all__ = [
    'Evaluator',
    'EvaluationResult',
    'TestResults',
    'Metric',
    'SWEBenchEvaluator',
    'LightweightEvaluator',
    'DockerSandbox',
    'LocalSandbox',
    'SandboxConfig',
    'ReportGenerator',
    'MethodResults',
]
