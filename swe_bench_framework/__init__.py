"""
SWE-bench Comparison Framework

A modular framework for comparing agentic exploration vs RAG methods
for automated software patching on the SWE-bench benchmark.
"""

__version__ = "1.0.0"

# Core abstractions
from .dataset.loader import SWEInstance, DatasetLoader
from .patch_generators.base import PatchGenerator, PatchResult
from .evaluation.base import Evaluator, EvaluationResult, Metric
from .context_gatherers.base import ContextGatherer, ContextBundle, ContextChunk

# Dataset loaders
from .dataset.swe_bench_loader import (
    SWEBenchLoader,
    SWEBenchLiteLoader,
    SWEBenchVerifiedLoader,
    SWEBenchFullLoader
)

# Patch generators
from .patch_generators.direct_generator import DirectPatchGenerator
from .patch_generators.iterative_generator import IterativePatchGenerator

# Evaluators
from .evaluation.swe_bench_evaluator import SWEBenchEvaluator
from .evaluation.docker_sandbox import DockerSandbox, LocalSandbox

# Metrics
from .evaluation.metrics.resolution import ResolutionMetric, PassAtKMetric
from .evaluation.metrics.localization import LocalizationMetric
from .evaluation.metrics.codebleu import CodeBLEUMetric
from .evaluation.metrics.token_usage import TokenUsageMetric
from .evaluation.metrics.semantic_entropy import SemanticEntropyMetric

# Experiment
from .experiment.orchestrator import ExperimentOrchestrator, ExperimentBuilder, OrchestratorConfig
from .experiment.runner import ExperimentRunner, ExperimentConfig

# Report generation
from .evaluation.report_generator import ReportGenerator

__all__ = [
    # Version
    '__version__',
    
    # Core abstractions
    'SWEInstance',
    'DatasetLoader',
    'PatchGenerator',
    'PatchResult',
    'Evaluator',
    'EvaluationResult',
    'Metric',
    'ContextGatherer',
    'ContextBundle',
    'ContextChunk',
    
    # Dataset loaders
    'SWEBenchLoader',
    'SWEBenchLiteLoader',
    'SWEBenchVerifiedLoader',
    'SWEBenchFullLoader',
    
    # Patch generators
    'DirectPatchGenerator',
    'IterativePatchGenerator',
    
    # Evaluators
    'SWEBenchEvaluator',
    'DockerSandbox',
    'LocalSandbox',
    
    # Metrics
    'ResolutionMetric',
    'PassAtKMetric',
    'LocalizationMetric',
    'CodeBLEUMetric',
    'TokenUsageMetric',
    'SemanticEntropyMetric',
    
    # Experiment
    'ExperimentOrchestrator',
    'ExperimentBuilder',
    'OrchestratorConfig',
    'ExperimentRunner',
    'ExperimentConfig',
    
    # Report
    'ReportGenerator',
]
