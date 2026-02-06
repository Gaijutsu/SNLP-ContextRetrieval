"""
Base classes for context gathering in the SWE-bench comparison framework.

This module defines the abstract base class and data structures for all
context gathering strategies, both agentic and RAG-based.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from enum import Enum


class ContextType(Enum):
    """Types of context that can be gathered."""
    FILE_CONTENT = "file_content"
    CLASS_DEFINITION = "class_definition"
    FUNCTION_DEFINITION = "function_definition"
    IMPORT_DEPENDENCY = "import_dependency"
    TEST_CONTEXT = "test_context"
    ERROR_CONTEXT = "error_context"
    REPO_STRUCTURE = "repo_structure"


@dataclass
class ContextChunk:
    """
    A single piece of context.
    
    Attributes:
        content: The actual content
        source_file: Path to the source file
        context_type: Type of context
        start_line: Starting line number
        end_line: Ending line number
        relevance_score: Relevance score (0.0-1.0)
        metadata: Additional metadata
    """
    content: str
    source_file: str
    context_type: ContextType
    start_line: int
    end_line: int
    relevance_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_prompt_format(self) -> str:
        """Convert chunk to format suitable for prompts."""
        return f"### {self.source_file} (lines {self.start_line}-{self.end_line})\n```{self.content}\n```"


@dataclass
class ContextBundle:
    """
    Bundle of context gathered for a patch generation task.
    
    Attributes:
        instance_id: Unique identifier for the instance
        problem_statement: The problem description
        chunks: List of context chunks
        repo_structure: Repository structure information
        gathered_at: Timestamp of when context was gathered
        gatherer_type: Type of gatherer used
        token_count: Estimated token count
        metadata: Additional metadata
    """
    instance_id: str
    problem_statement: str
    chunks: List[ContextChunk]
    repo_structure: Dict[str, Any] = field(default_factory=dict)
    gathered_at: str = ""
    gatherer_type: str = ""
    token_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_prompt_context(
        self,
        max_tokens: int = 8000,
        priority_order: Optional[List[ContextType]] = None
    ) -> str:
        """
        Convert context bundle to string for LLM prompt.
        
        Args:
            max_tokens: Maximum tokens for context
            priority_order: Priority order for context types
            
        Returns:
            Formatted context string
        """
        from ..patch_generators.prompts.builders import PromptBuilder
        
        builder = PromptBuilder(max_tokens=max_tokens)
        return builder.build_context(self, priority_order)


class ContextGatherer(ABC):
    """
    Abstract base class for all context gathering strategies.
    
    Both agentic and RAG methods implement this interface.
    
    Example:
        class MyGatherer(ContextGatherer):
            def gather_context(self, instance, repo_path):
                # Gather context
                return ContextBundle(...)
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the context gatherer.
        
        Args:
            config: Configuration for the gatherer
        """
        self.config = config
        self.name = self.__class__.__name__
        self._initialized = False
    
    @abstractmethod
    def gather_context(
        self,
        instance: 'SWEInstance',
        repo_path: str
    ) -> ContextBundle:
        """
        Gather context for a given SWE-bench instance.
        
        Args:
            instance: The SWE-bench instance
            repo_path: Path to the repository checkout
            
        Returns:
            ContextBundle containing all gathered context
        """
        pass
    
    def initialize(self, repo_path: str) -> None:
        """
        Initialize any resources needed.
        
        Args:
            repo_path: Path to the repository
        """
        self._initialized = True
    
    def cleanup(self) -> None:
        """Cleanup resources."""
        self._initialized = False
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about context gathering.
        
        Returns:
            Dictionary with statistics
        """
        return {
            'name': self.name,
            'initialized': self._initialized
        }


# Forward reference imports
from ..dataset.loader import SWEInstance  # noqa: E402
