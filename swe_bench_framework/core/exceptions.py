"""
Custom exceptions for the SWE-bench comparison framework.

This module defines all custom exceptions used throughout the framework
for proper error handling and debugging.
"""

from typing import Any, Dict, Optional


class FrameworkError(Exception):
    """
    Base exception for all framework errors.
    
    This is the root exception class that all other framework exceptions
    inherit from. It provides common functionality for error tracking.
    
    Attributes:
        message: Error message
        details: Additional error details
        error_code: Optional error code for categorization
    """
    
    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        error_code: Optional[str] = None
    ):
        super().__init__(message)
        self.message = message
        self.details = details or {}
        self.error_code = error_code
    
    def __str__(self) -> str:
        if self.error_code:
            return f"[{self.error_code}] {self.message}"
        return self.message
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for logging."""
        return {
            'error_type': self.__class__.__name__,
            'message': self.message,
            'details': self.details,
            'error_code': self.error_code,
        }


class ConfigurationError(FrameworkError):
    """
    Exception raised for configuration-related errors.
    
    This includes invalid configuration files, missing required fields,
    or invalid configuration values.
    
    Example:
        >>> raise ConfigurationError(
        ...     "Missing required field: 'llm.model'",
        ...     details={'config_file': 'config.yaml'}
        ... )
    """
    
    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            message,
            details=details,
            error_code="CONFIG_ERROR"
        )


class GatheringError(FrameworkError):
    """
    Exception raised during context gathering.
    
    This includes errors from agentic exploration, RAG retrieval,
    or any other context gathering operation.
    
    Example:
        >>> raise GatheringError(
        ...     "Failed to retrieve context",
        ...     details={
        ...         'instance_id': 'django-1234',
        ...         'gatherer': 'BM25Retriever',
        ...         'error': 'Index not found'
        ...     }
        ... )
    """
    
    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            message,
            details=details,
            error_code="GATHER_ERROR"
        )


class GenerationError(FrameworkError):
    """
    Exception raised during patch generation.
    
    This includes LLM errors, invalid patch formats, or generation failures.
    
    Example:
        >>> raise GenerationError(
        ...     "Failed to generate valid patch",
        ...     details={
        ...         'instance_id': 'django-1234',
        ...         'attempts': 3,
        ...         'last_error': 'Patch syntax error'
        ...     }
        ... )
    """
    
    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            message,
            details=details,
            error_code="GENERATE_ERROR"
        )


class EvaluationError(FrameworkError):
    """
    Exception raised during patch evaluation.
    
    This includes test execution failures, sandbox errors, or
    metric computation errors.
    
    Example:
        >>> raise EvaluationError(
        ...     "Test execution failed",
        ...     details={
        ...         'instance_id': 'django-1234',
        ...         'test_command': 'pytest tests/',
        ...         'exit_code': 1
        ...     }
        ... )
    """
    
    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            message,
            details=details,
            error_code="EVAL_ERROR"
        )


class ValidationError(FrameworkError):
    """
    Exception raised for validation failures.
    
    This includes invalid patch formats, syntax errors, or
    failed validation checks.
    
    Example:
        >>> raise ValidationError(
        ...     "Patch syntax error",
        ...     details={
        ...         'line': 42,
        ...         'error': 'Invalid hunk header'
        ...     }
        ... )
    """
    
    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            message,
            details=details,
            error_code="VALIDATION_ERROR"
        )


class RepositoryError(FrameworkError):
    """
    Exception raised for repository-related errors.
    
    This includes cloning failures, checkout errors, or index building failures.
    
    Example:
        >>> raise RepositoryError(
        ...     "Failed to checkout commit",
        ...     details={
        ...         'repo': 'django/django',
        ...         'commit': 'abc123',
        ...         'error': 'Commit not found'
        ...     }
        ... )
    """
    
    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            message,
            details=details,
            error_code="REPO_ERROR"
        )


class LLMError(FrameworkError):
    """
    Exception raised for LLM-related errors.
    
    This includes API errors, rate limiting, or model errors.
    
    Example:
        >>> raise LLMError(
        ...     "API rate limit exceeded",
        ...     details={
        ...         'provider': 'openai',
        ...         'model': 'gpt-5-mini',
        ...         'retry_after': 60
        ...     }
        ... )
    """
    
    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            message,
            details=details,
            error_code="LLM_ERROR"
        )


class SandboxError(FrameworkError):
    """
    Exception raised for sandbox-related errors.
    
    This includes Docker errors, container failures, or test execution errors.
    
    Example:
        >>> raise SandboxError(
        ...     "Docker container failed to start",
        ...     details={
        ...         'image': 'swe-bench/sandbox:latest',
        ...         'error': 'Image not found'
        ...     }
        ... )
    """
    
    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            message,
            details=details,
            error_code="SANDBOX_ERROR"
        )


class IndexError(FrameworkError):
    """
    Exception raised for index-related errors.
    
    This includes index building failures, search errors, or loading errors.
    
    Example:
        >>> raise IndexError(
        ...     "Failed to build BM25 index",
        ...     details={
        ...         'indexer': 'BM25Indexer',
        ...         'num_documents': 1000,
        ...         'error': 'Out of memory'
        ...     }
        ... )
    """
    
    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            message,
            details=details,
            error_code="INDEX_ERROR"
        )


class DatasetError(FrameworkError):
    """
    Exception raised for dataset-related errors.
    
    This includes loading failures, invalid data, or missing fields.
    
    Example:
        >>> raise DatasetError(
        ...     "Failed to load SWE-bench dataset",
        ...     details={
        ...         'dataset': 'swe-bench-lite',
        ...         'split': 'test',
        ...         'error': 'File not found'
        ...     }
        ... )
    """
    
    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            message,
            details=details,
            error_code="DATASET_ERROR"
        )


class TimeoutError(FrameworkError):
    """
    Exception raised when an operation times out.
    
    Example:
        >>> raise TimeoutError(
        ...     "Patch generation timed out",
        ...     details={
        ...         'operation': 'generate_patch',
        ...         'timeout': 300,
        ...         'elapsed': 300.5
        ...     }
        ... )
    """
    
    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            message,
            details=details,
            error_code="TIMEOUT_ERROR"
        )


class NotInitializedError(FrameworkError):
    """
    Exception raised when a component is used before initialization.
    
    Example:
        >>> raise NotInitializedError(
        ...     "ContextGatherer not initialized",
        ...     details={
        ...         'component': 'BM25Gatherer',
        ...         'method': 'gather_context'
        ...     }
        ... )
    """
    
    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            message,
            details=details,
            error_code="NOT_INITIALIZED"
        )


class RegistryError(FrameworkError):
    """
    Exception raised for registry-related errors.
    
    This includes registration failures or lookup errors.
    
    Example:
        >>> raise RegistryError(
        ...     "Component not found in registry",
        ...     details={
        ...         'component_type': 'gatherer',
        ...         'name': 'unknown_gatherer'
        ...     }
        ... )
    """
    
    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            message,
            details=details,
            error_code="REGISTRY_ERROR"
        )
