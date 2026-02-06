"""
Data models for the SWE-bench comparison framework.

This module defines all dataclasses used throughout the framework,
including SWE-bench instances, context bundles, and results.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
import json


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
class SWEInstance:
    """
    Represents a SWE-bench instance.
    
    This dataclass contains all information about a single SWE-bench task,
    including the problem statement, test information, and ground truth.
    
    Attributes:
        instance_id: Unique identifier for the instance
        repo: Repository name (e.g., "django/django")
        base_commit: Commit hash to checkout
        problem_statement: Description of the issue to fix
        hints_text: Optional hints for solving the issue
        test_patch: Patch containing test changes
        patch: Gold patch (ground truth solution)
        failed_tests: List of tests that fail before the fix
        passed_tests: List of tests that pass before the fix
        modified_files: Ground truth list of files modified by the fix
        modified_methods: Ground truth list of methods modified by the fix
    """
    instance_id: str
    repo: str
    base_commit: str
    problem_statement: str
    hints_text: Optional[str] = None
    test_patch: str = ""
    patch: str = ""  # Gold patch
    failed_tests: List[str] = field(default_factory=list)
    passed_tests: List[str] = field(default_factory=list)
    modified_files: List[str] = field(default_factory=list)
    modified_methods: List[str] = field(default_factory=list)
    
    def get_gold_context(self) -> Dict[str, List[str]]:
        """
        Get ground truth context for evaluation.
        
        Returns:
            Dictionary with 'files' and 'methods' keys containing
            ground truth modified files and methods.
        """
        return {
            'files': self.modified_files,
            'methods': self.modified_methods
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert instance to dictionary."""
        return {
            'instance_id': self.instance_id,
            'repo': self.repo,
            'base_commit': self.base_commit,
            'problem_statement': self.problem_statement,
            'hints_text': self.hints_text,
            'test_patch': self.test_patch,
            'patch': self.patch,
            'failed_tests': self.failed_tests,
            'passed_tests': self.passed_tests,
            'modified_files': self.modified_files,
            'modified_methods': self.modified_methods,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SWEInstance':
        """Create instance from dictionary."""
        return cls(
            instance_id=data['instance_id'],
            repo=data['repo'],
            base_commit=data['base_commit'],
            problem_statement=data['problem_statement'],
            hints_text=data.get('hints_text'),
            test_patch=data.get('test_patch', ''),
            patch=data.get('patch', ''),
            failed_tests=data.get('failed_tests', []),
            passed_tests=data.get('passed_tests', []),
            modified_files=data.get('modified_files', []),
            modified_methods=data.get('modified_methods', []),
        )


@dataclass
class ContextChunk:
    """
    A single piece of context gathered for patch generation.
    
    Attributes:
        content: The actual content of the context
        source_file: Path to the source file
        context_type: Type of context (function, class, etc.)
        start_line: Starting line number (1-indexed)
        end_line: Ending line number (1-indexed)
        relevance_score: Score indicating relevance to the problem (0.0-1.0)
        metadata: Additional metadata about the chunk
    """
    content: str
    source_file: str
    context_type: ContextType
    start_line: int
    end_line: int
    relevance_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate the context chunk after initialization."""
        if self.start_line < 1:
            raise ValueError(f"start_line must be >= 1, got {self.start_line}")
        if self.end_line < self.start_line:
            raise ValueError(
                f"end_line ({self.end_line}) must be >= start_line ({self.start_line})"
            )
        if not 0.0 <= self.relevance_score <= 1.0:
            raise ValueError(
                f"relevance_score must be between 0.0 and 1.0, got {self.relevance_score}"
            )
    
    def get_line_count(self) -> int:
        """Get the number of lines in this chunk."""
        return self.end_line - self.start_line + 1
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert chunk to dictionary."""
        return {
            'content': self.content,
            'source_file': self.source_file,
            'context_type': self.context_type.value,
            'start_line': self.start_line,
            'end_line': self.end_line,
            'relevance_score': self.relevance_score,
            'metadata': self.metadata,
        }


@dataclass
class ContextBundle:
    """
    Bundle of context gathered for a patch generation task.
    
    This is the unified format for context from both agentic and RAG methods.
    
    Attributes:
        instance_id: ID of the SWE-bench instance
        problem_statement: The problem statement
        chunks: List of context chunks
        repo_structure: Dictionary describing repository structure
        gathered_at: ISO format timestamp when context was gathered
        gatherer_type: Type of gatherer used (e.g., "autocoderover", "hybrid_rag")
        token_count: Total token count of all chunks
        metadata: Additional metadata about the gathering process
    """
    instance_id: str
    problem_statement: str
    chunks: List[ContextChunk]
    repo_structure: Dict[str, Any] = field(default_factory=dict)
    gathered_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    gatherer_type: str = "unknown"
    token_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate the context bundle after initialization."""
        if not self.instance_id:
            raise ValueError("instance_id cannot be empty")
    
    def to_prompt_context(
        self, 
        max_tokens: int = 8000,
        priority_order: Optional[List[ContextType]] = None
    ) -> str:
        """
        Convert context bundle to string for LLM prompt.
        
        Args:
            max_tokens: Maximum number of tokens to include
            priority_order: Order of priority for context types
            
        Returns:
            Formatted context string ready for LLM prompt
        """
        if priority_order is None:
            priority_order = [
                ContextType.ERROR_CONTEXT,
                ContextType.FUNCTION_DEFINITION,
                ContextType.CLASS_DEFINITION,
                ContextType.TEST_CONTEXT,
                ContextType.REPO_STRUCTURE,
                ContextType.FILE_CONTENT,
                ContextType.IMPORT_DEPENDENCY,
            ]
        
        # Sort chunks by priority (lower index = higher priority) and relevance
        type_priority = {t: i for i, t in enumerate(priority_order)}
        sorted_chunks = sorted(
            self.chunks,
            key=lambda c: (type_priority.get(c.context_type, 999), -c.relevance_score)
        )
        
        # Build context string
        context_parts = []
        current_tokens = 0
        
        for chunk in sorted_chunks:
            # Estimate tokens (rough approximation: 4 chars per token)
            chunk_tokens = len(chunk.content) // 4
            
            if current_tokens + chunk_tokens > max_tokens:
                # Try to truncate if it's a file content chunk
                if chunk.context_type == ContextType.FILE_CONTENT:
                    remaining_tokens = max_tokens - current_tokens
                    if remaining_tokens > 100:  # Only include if meaningful
                        truncated_content = chunk.content[:remaining_tokens * 4]
                        context_parts.append(self._format_chunk(
                            chunk, truncated_content
                        ))
                break
            
            context_parts.append(self._format_chunk(chunk))
            current_tokens += chunk_tokens
        
        return "\n\n".join(context_parts)
    
    def _format_chunk(self, chunk: ContextChunk, content: Optional[str] = None) -> str:
        """Format a single chunk for the prompt."""
        content = content or chunk.content
        return (
            f"### {chunk.source_file} (lines {chunk.start_line}-{chunk.end_line})\n"
            f"```{content}\n```"
        )
    
    def get_chunks_by_type(self, context_type: ContextType) -> List[ContextChunk]:
        """Get all chunks of a specific type."""
        return [c for c in self.chunks if c.context_type == context_type]
    
    def get_top_k_chunks(self, k: int) -> List[ContextChunk]:
        """Get top k chunks by relevance score."""
        return sorted(self.chunks, key=lambda c: -c.relevance_score)[:k]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert bundle to dictionary."""
        return {
            'instance_id': self.instance_id,
            'problem_statement': self.problem_statement,
            'chunks': [c.to_dict() for c in self.chunks],
            'repo_structure': self.repo_structure,
            'gathered_at': self.gathered_at,
            'gatherer_type': self.gatherer_type,
            'token_count': self.token_count,
            'metadata': self.metadata,
        }


@dataclass
class PatchResult:
    """
    Result of patch generation.
    
    Attributes:
        instance_id: ID of the SWE-bench instance
        patch_content: The generated patch (unified diff format)
        success: Whether patch generation was successful
        generation_time: Time taken to generate the patch (seconds)
        attempts: Number of attempts made
        token_usage: Dictionary with token usage statistics
        error_message: Error message if generation failed
        intermediate_steps: List of intermediate steps taken
        confidence_score: Confidence score for the patch (0.0-1.0)
        context_bundle: Reference to the context bundle used
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
    context_bundle: Optional[ContextBundle] = None
    
    def __post_init__(self):
        """Validate the patch result after initialization."""
        if not self.instance_id:
            raise ValueError("instance_id cannot be empty")
        if not 0.0 <= self.confidence_score <= 1.0:
            raise ValueError(
                f"confidence_score must be between 0.0 and 1.0, got {self.confidence_score}"
            )
    
    def get_total_tokens(self) -> int:
        """Get total token usage."""
        return (
            self.token_usage.get('prompt_tokens', 0) +
            self.token_usage.get('completion_tokens', 0)
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
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
        }


@dataclass
class EvaluationResult:
    """
    Result of evaluating a patch.
    
    Attributes:
        instance_id: ID of the SWE-bench instance
        resolved: Whether the patch resolves the issue
        patch_applied: Whether the patch was successfully applied
        tests_passed: List of tests that passed
        tests_failed: List of tests that failed
        localization_accuracy: Dictionary with localization metrics
        codebleu_score: CodeBLEU score comparing to gold patch
        execution_time: Time taken for evaluation (seconds)
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
        """Validate the evaluation result after initialization."""
        if not self.instance_id:
            raise ValueError("instance_id cannot be empty")
        if not 0.0 <= self.codebleu_score <= 1.0:
            raise ValueError(
                f"codebleu_score must be between 0.0 and 1.0, got {self.codebleu_score}"
            )
    
    def get_test_pass_rate(self) -> float:
        """Get the percentage of tests that passed."""
        total = len(self.tests_passed) + len(self.tests_failed)
        if total == 0:
            return 0.0
        return len(self.tests_passed) / total
    
    def get_recall_at_k(self, k: int) -> float:
        """Get recall@k for file localization."""
        key = f'recall@{k}'
        return self.localization_accuracy.get(key, 0.0)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
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
        }


@dataclass
class SearchResult:
    """
    Result from index search in RAG methods.
    
    Attributes:
        content: The retrieved content
        file_path: Path to the source file
        start_line: Starting line number
        end_line: Ending line number
        score: Relevance score
        metadata: Additional metadata
    """
    content: str
    file_path: str
    start_line: int
    end_line: int
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_context_chunk(self, context_type: ContextType = ContextType.FILE_CONTENT) -> ContextChunk:
        """Convert search result to context chunk."""
        return ContextChunk(
            content=self.content,
            source_file=self.file_path,
            context_type=context_type,
            start_line=self.start_line,
            end_line=self.end_line,
            relevance_score=self.score,
            metadata=self.metadata,
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            'content': self.content,
            'file_path': self.file_path,
            'start_line': self.start_line,
            'end_line': self.end_line,
            'score': self.score,
            'metadata': self.metadata,
        }
