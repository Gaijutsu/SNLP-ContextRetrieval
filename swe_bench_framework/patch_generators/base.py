"""
Base classes for patch generation in the SWE-bench comparison framework.

This module defines the abstract base class and data structures for all
patch generation strategies, ensuring a unified interface across different
approaches (direct, iterative, edit-script based).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from datetime import datetime


@dataclass
class PatchResult:
    """
    Result of patch generation.
    
    Attributes:
        instance_id: Unique identifier for the SWE-bench instance
        patch_content: The generated patch in unified diff format, or None if failed
        success: Whether the patch generation was successful
        generation_time: Time taken to generate the patch in seconds
        attempts: Number of attempts made to generate a valid patch
        token_usage: Dictionary with token consumption details (prompt, completion, total)
        error_message: Error message if generation failed
        intermediate_steps: List of intermediate steps for iterative methods
        confidence_score: Confidence score for the generated patch (0.0-1.0)
        metadata: Additional metadata about the generation process
    """
    instance_id: str
    patch_content: Optional[str]
    success: bool
    generation_time: float
    attempts: int
    token_usage: Dict[str, int] = field(default_factory=dict)
    error_message: Optional[str] = None
    intermediate_steps: List[Dict[str, Any]] = field(default_factory=list)
    confidence_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate the result after initialization."""
        if self.success and self.patch_content is None:
            raise ValueError("Successful patch result must have patch_content")
        if not self.success and self.error_message is None:
            self.error_message = "Unknown error occurred during patch generation"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert PatchResult to dictionary for serialization."""
        return {
            'instance_id': self.instance_id,
            'patch_content': self.patch_content,
            'success': self.success,
            'generation_time': self.generation_time,
            'attempts': self.attempts,
            'token_usage': self.token_usage,
            'error_message': self.error_message,
            'intermediate_steps': self.intermediate_steps,
            'confidence_score': self.confidence_score,
            'metadata': self.metadata,
            'timestamp': datetime.now().isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PatchResult':
        """Create PatchResult from dictionary."""
        # Remove timestamp if present (not part of constructor)
        data_copy = {k: v for k, v in data.items() if k != 'timestamp'}
        return cls(**data_copy)


class PatchGenerator(ABC):
    """
    Abstract base class for all patch generation strategies.
    
    This class defines the interface that all patch generators must implement,
    including direct LLM generation, iterative refinement, and edit-script based
    approaches.
    
    Example usage:
        generator = DirectPatchGenerator(llm_config, generator_config)
        result = generator.generate_patch(context_bundle, instance)
        if result.success:
            print(f"Generated patch in {result.generation_time:.2f}s")
    """
    
    def __init__(self, llm_config: Dict[str, Any], config: Dict[str, Any]):
        """
        Initialize the patch generator.
        
        Args:
            llm_config: Configuration for the LLM (model, temperature, etc.)
            config: Generator-specific configuration
        """
        self.llm_config = llm_config
        self.config = config
        self.name = self.__class__.__name__
        self._initialized = False
    
    @abstractmethod
    def generate_patch(
        self,
        context_bundle: 'ContextBundle',
        instance: 'SWEInstance'
    ) -> PatchResult:
        """
        Generate a patch given context and problem statement.
        
        Args:
            context_bundle: Gathered context containing relevant code snippets
            instance: The SWE-bench instance with problem statement
            
        Returns:
            PatchResult containing the generated patch or error information
            
        Raises:
            RuntimeError: If the generator has not been initialized
        """
        if not self._initialized:
            raise RuntimeError(f"{self.name} must be initialized before generating patches")
        pass
    
    @abstractmethod
    def validate_patch(self, patch_content: str, repo_path: str) -> bool:
        """
        Validate that a patch is syntactically correct and can be applied.
        
        Args:
            patch_content: The patch in unified diff format
            repo_path: Path to the repository to validate against
            
        Returns:
            True if the patch is valid, False otherwise
        """
        pass
    
    def initialize(self) -> None:
        """
        Initialize the generator. Override for any setup needed.
        
        This method should be called before the first call to generate_patch().
        """
        self._initialized = True
    
    def cleanup(self) -> None:
        """
        Cleanup any resources. Override for custom cleanup.
        
        This method should be called when the generator is no longer needed.
        """
        self._initialized = False
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about patch generation.
        
        Returns:
            Dictionary with generator statistics
        """
        return {
            'name': self.name,
            'initialized': self._initialized,
            'llm_model': self.llm_config.get('model', 'unknown'),
            'config': self.config
        }


# Type imports for forward references
from ..context_gatherers.base import ContextBundle  # noqa: E402
from ..dataset.loader import SWEInstance  # noqa: E402
