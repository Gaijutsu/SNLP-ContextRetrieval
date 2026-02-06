"""
Core module for SWE-bench comparison framework.

This module contains the core abstractions, data models, and interfaces
that form the foundation of the framework.
"""

from .data_models import (
    SWEInstance,
    ContextType,
    ContextChunk,
    ContextBundle,
    PatchResult,
    EvaluationResult,
    SearchResult,
)

from .interfaces import (
    ContextGatherer,
    PatchGenerator,
    Evaluator,
    RepositoryIndexer,
)

from .exceptions import (
    FrameworkError,
    ConfigurationError,
    GatheringError,
    GenerationError,
    EvaluationError,
    ValidationError,
    RepositoryError,
    LLMError,
)

from .registry import (
    ComponentRegistry,
    register_gatherer,
    register_generator,
    register_evaluator,
    register_indexer,
    create_gatherer,
    create_generator,
    create_evaluator,
    create_indexer,
)

__all__ = [
    # Data models
    "SWEInstance",
    "ContextType",
    "ContextChunk",
    "ContextBundle",
    "PatchResult",
    "EvaluationResult",
    "SearchResult",
    # Interfaces
    "ContextGatherer",
    "PatchGenerator",
    "Evaluator",
    "RepositoryIndexer",
    # Exceptions
    "FrameworkError",
    "ConfigurationError",
    "GatheringError",
    "GenerationError",
    "EvaluationError",
    "ValidationError",
    "RepositoryError",
    "LLMError",
    # Registry
    "ComponentRegistry",
    "register_gatherer",
    "register_generator",
    "register_evaluator",
    "register_indexer",
    "create_gatherer",
    "create_generator",
    "create_evaluator",
    "create_indexer",
]
