"""
Configuration management for the SWE-bench comparison framework.

This module provides configuration loading, validation, and default
configuration management for experiments.
"""

from .schema import (
    ExperimentConfig,
    DatasetConfig,
    LLMConfig,
    MethodConfig,
    PatchGenerationConfig,
    EvaluationConfig,
    LoggingConfig,
    ExperimentTrackingConfig,
    SandboxConfig,
    RAGConfig,
    AgenticConfig,
    RateLimitConfig,
)

from .loader import load_config, load_config_from_dict
from .defaults import get_default_config, get_minimal_config

__all__ = [
    # Schema classes
    "ExperimentConfig",
    "DatasetConfig",
    "LLMConfig",
    "MethodConfig",
    "PatchGenerationConfig",
    "EvaluationConfig",
    "LoggingConfig",
    "ExperimentTrackingConfig",
    "SandboxConfig",
    "RAGConfig",
    "AgenticConfig",
    "RateLimitConfig",
    # Loader functions
    "load_config",
    "load_config_from_dict",
    # Default configs
    "get_default_config",
    "get_minimal_config",
]
