"""
Prompt builders for patch generation in the SWE-bench comparison framework.

This module provides utilities for building prompts from ContextBundle objects,
including token budget management and context prioritization.
"""

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable
from enum import Enum

from .templates import PromptTemplates


class ContextType(Enum):
    """Types of context that can be gathered, ordered by priority."""
    ERROR_CONTEXT = "error_context"           # Highest priority
    FUNCTION_DEFINITION = "function_definition"
    CLASS_DEFINITION = "class_definition"
    TEST_CONTEXT = "test_context"
    REPO_STRUCTURE = "repo_structure"
    FILE_CONTENT = "file_content"
    IMPORT_DEPENDENCY = "import_dependency"   # Lowest priority


@dataclass
class TokenBudget:
    """
    Token budget configuration for prompt building.
    
    Attributes:
        max_tokens: Maximum total tokens allowed
        reserved_tokens: Tokens reserved for prompt structure
        context_tokens: Tokens available for context content
        problem_tokens: Tokens allocated for problem statement
    """
    max_tokens: int = 8000
    reserved_tokens: int = 500
    problem_tokens: int = 1000
    
    @property
    def context_tokens(self) -> int:
        """Calculate available tokens for context."""
        return self.max_tokens - self.reserved_tokens - self.problem_tokens


class TokenCounter:
    """
    Token counter for estimating token usage.
    
    Uses a simple approximation (4 characters per token) which is
    reasonably accurate for English text and code.
    """
    
    # Average characters per token (conservative estimate)
    CHARS_PER_TOKEN = 4
    
    def __init__(self, tokenizer: Optional[str] = None):
        """
        Initialize the token counter.
        
        Args:
            tokenizer: Optional tokenizer name (for future use with tiktoken)
        """
        self.tokenizer = tokenizer
    
    def count(self, text: str) -> int:
        """
        Estimate the number of tokens in text.
        
        Args:
            text: The text to count tokens for
            
        Returns:
            Estimated token count
        """
        if not text:
            return 0
        return len(text) // self.CHARS_PER_TOKEN
    
    def count_batch(self, texts: List[str]) -> List[int]:
        """
        Count tokens for multiple texts.
        
        Args:
            texts: List of texts to count
            
        Returns:
            List of token counts
        """
        return [self.count(t) for t in texts]


@dataclass
class ContextChunk:
    """A single piece of context with metadata."""
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
    """Bundle of context gathered for a patch generation task."""
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
        builder = PromptBuilder(max_tokens=max_tokens)
        return builder.build_context(self, priority_order)


class PromptBuilder:
    """
    Builder for constructing prompts from ContextBundle objects.
    
    This class handles:
    - Token budget management
    - Context prioritization
    - Prompt formatting
    """
    
    # Default priority order for context types
    DEFAULT_PRIORITY_ORDER = [
        ContextType.ERROR_CONTEXT,
        ContextType.FUNCTION_DEFINITION,
        ContextType.CLASS_DEFINITION,
        ContextType.TEST_CONTEXT,
        ContextType.REPO_STRUCTURE,
        ContextType.FILE_CONTENT,
        ContextType.IMPORT_DEPENDENCY
    ]
    
    def __init__(
        self,
        max_tokens: int = 8000,
        token_counter: Optional[TokenCounter] = None,
        templates: Optional[PromptTemplates] = None
    ):
        """
        Initialize the prompt builder.
        
        Args:
            max_tokens: Maximum tokens for the prompt
            token_counter: Optional custom token counter
            templates: Optional custom prompt templates
        """
        self.budget = TokenBudget(max_tokens=max_tokens)
        self.token_counter = token_counter or TokenCounter()
        self.templates = templates or PromptTemplates()
    
    def build(
        self,
        context_bundle: ContextBundle,
        problem_statement: str,
        hints_text: Optional[str] = None,
        priority_order: Optional[List[ContextType]] = None,
        custom_instructions: Optional[str] = None
    ) -> str:
        """
        Build a complete prompt from context bundle.
        
        Args:
            context_bundle: The gathered context
            problem_statement: The problem description
            hints_text: Optional hints
            priority_order: Priority order for context types
            custom_instructions: Optional custom instructions
            
        Returns:
            Complete formatted prompt
        """
        # Build context section within token budget
        context_content = self.build_context(context_bundle, priority_order)
        
        # Build full prompt using templates
        return self.templates.build_full_prompt(
            problem_statement=problem_statement,
            context_content=context_content,
            hints_text=hints_text,
            custom_instructions=custom_instructions
        )
    
    def build_context(
        self,
        context_bundle: ContextBundle,
        priority_order: Optional[List[ContextType]] = None
    ) -> str:
        """
        Build context section within token budget.
        
        Args:
            context_bundle: The gathered context
            priority_order: Priority order for context types
            
        Returns:
            Formatted context string
        """
        if priority_order is None:
            priority_order = self.DEFAULT_PRIORITY_ORDER
        
        # Handle missing context bundle gracefully
        if context_bundle is None:
            return "# Context\nNo context provided. Please analyze the problem statement and provide a fix based on your understanding.\n"
        
        # Sort chunks by priority and relevance
        sorted_chunks = self._sort_chunks(context_bundle.chunks, priority_order)
        
        # Build context within token budget
        context_parts = []
        remaining_tokens = self.budget.context_tokens
        
        for chunk in sorted_chunks:
            chunk_text = chunk.to_prompt_format()
            chunk_tokens = self.token_counter.count(chunk_text)
            
            if chunk_tokens <= remaining_tokens:
                context_parts.append(chunk_text)
                remaining_tokens -= chunk_tokens
            else:
                # Try to truncate file content chunks
                if chunk.context_type == ContextType.FILE_CONTENT:
                    truncated = self._truncate_chunk(chunk, remaining_tokens)
                    if truncated:
                        context_parts.append(truncated)
                break
        
        # Add repository structure if there's room
        if context_bundle.repo_structure and remaining_tokens > 100:
            structure_text = self._format_repo_structure(
                context_bundle.repo_structure,
                remaining_tokens
            )
            if structure_text:
                context_parts.insert(0, structure_text)
        
        return '\n\n'.join(context_parts)
    
    def _sort_chunks(
        self,
        chunks: List[ContextChunk],
        priority_order: List[ContextType]
    ) -> List[ContextChunk]:
        """
        Sort chunks by priority and relevance score.
        
        Args:
            chunks: List of context chunks
            priority_order: Priority order for context types
            
        Returns:
            Sorted list of chunks
        """
        # Create priority map
        priority_map = {ct: i for i, ct in enumerate(priority_order)}
        
        def sort_key(chunk: ContextChunk) -> tuple:
            priority = priority_map.get(chunk.context_type, len(priority_order))
            # Higher relevance score should come first (hence negative)
            return (priority, -chunk.relevance_score)
        
        return sorted(chunks, key=sort_key)
    
    def _truncate_chunk(
        self,
        chunk: ContextChunk,
        max_tokens: int
    ) -> Optional[str]:
        """
        Truncate a chunk to fit within token budget.
        
        Args:
            chunk: The chunk to truncate
            max_tokens: Maximum tokens allowed
            
        Returns:
            Truncated chunk text or None if can't truncate
        """
        # Reserve tokens for header
        header = f"### {chunk.source_file} (lines {chunk.start_line}-{chunk.end_line})\n```\n"
        footer = "\n```"
        
        header_tokens = self.token_counter.count(header)
        footer_tokens = self.token_counter.count(footer)
        
        available_tokens = max_tokens - header_tokens - footer_tokens
        if available_tokens < 50:  # Minimum content size
            return None
        
        # Truncate content (rough approximation)
        available_chars = available_tokens * self.token_counter.CHARS_PER_TOKEN
        truncated_content = chunk.content[:available_chars]
        
        # Add truncation indicator
        if len(chunk.content) > available_chars:
            truncated_content += "\n... [truncated]"
        
        return f"{header}{truncated_content}{footer}"
    
    def _format_repo_structure(
        self,
        repo_structure: Dict[str, Any],
        max_tokens: int
    ) -> Optional[str]:
        """
        Format repository structure within token budget.
        
        Args:
            repo_structure: Repository structure information
            max_tokens: Maximum tokens allowed
            
        Returns:
            Formatted structure string or None
        """
        header = "### Repository Structure\n"
        
        # Build structure string
        parts = []
        if 'files' in repo_structure:
            files = repo_structure['files']
            if isinstance(files, list) and files:
                parts.append(f"Key files: {', '.join(files[:10])}")
        
        if 'directories' in repo_structure:
            dirs = repo_structure['directories']
            if isinstance(dirs, list) and dirs:
                parts.append(f"Directories: {', '.join(dirs[:5])}")
        
        structure_text = header + '\n'.join(parts)
        
        if self.token_counter.count(structure_text) <= max_tokens:
            return structure_text
        return None
    
    def estimate_tokens(
        self,
        context_bundle: ContextBundle,
        problem_statement: str,
        hints_text: Optional[str] = None
    ) -> Dict[str, int]:
        """
        Estimate token usage for a prompt.
        
        Args:
            context_bundle: The gathered context
            problem_statement: The problem description
            hints_text: Optional hints
            
        Returns:
            Dictionary with token estimates
        """
        context_tokens = sum(
            self.token_counter.count(chunk.to_prompt_format())
            for chunk in context_bundle.chunks
        )
        
        problem_tokens = self.token_counter.count(problem_statement)
        hints_tokens = self.token_counter.count(hints_text) if hints_text else 0
        
        # Estimate template overhead
        template_overhead = self.token_counter.count(PromptTemplates.SYSTEM_PROMPT)
        template_overhead += self.token_counter.count(PromptTemplates.INSTRUCTIONS_TEMPLATE)
        
        total_tokens = (
            context_tokens + problem_tokens + hints_tokens + template_overhead
        )
        
        return {
            'context_tokens': context_tokens,
            'problem_tokens': problem_tokens,
            'hints_tokens': hints_tokens,
            'template_overhead': template_overhead,
            'total_estimate': total_tokens,
            'max_tokens': self.budget.max_tokens,
            'within_budget': total_tokens <= self.budget.max_tokens
        }


class AdaptivePromptBuilder(PromptBuilder):
    """
    Adaptive prompt builder that adjusts strategy based on context.
    
    This builder can dynamically adjust token allocation based on:
    - Problem complexity
    - Context density
    - Available information
    """
    
    def __init__(
        self,
        max_tokens: int = 8000,
        token_counter: Optional[TokenCounter] = None,
        templates: Optional[PromptTemplates] = None,
        adaptive_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the adaptive prompt builder.
        
        Args:
            max_tokens: Maximum tokens for the prompt
            token_counter: Optional custom token counter
            templates: Optional custom prompt templates
            adaptive_config: Configuration for adaptive behavior
        """
        super().__init__(max_tokens, token_counter, templates)
        self.adaptive_config = adaptive_config or {}
    
    def build(
        self,
        context_bundle: ContextBundle,
        problem_statement: str,
        hints_text: Optional[str] = None,
        priority_order: Optional[List[ContextType]] = None,
        custom_instructions: Optional[str] = None
    ) -> str:
        """
        Build prompt with adaptive token allocation.
        
        Adjusts token allocation based on problem characteristics.
        """
        # Analyze problem complexity
        complexity = self._analyze_complexity(problem_statement, context_bundle)
        
        # Adjust budget based on complexity
        if complexity == 'high':
            # Allocate more tokens to context for complex problems
            self.budget.problem_tokens = 800
            self.budget.reserved_tokens = 400
        elif complexity == 'low':
            # Simpler problems need less context
            self.budget.problem_tokens = 1200
            self.budget.reserved_tokens = 300
        
        return super().build(
            context_bundle, problem_statement, hints_text,
            priority_order, custom_instructions
        )
    
    def _analyze_complexity(
        self,
        problem_statement: str,
        context_bundle: ContextBundle
    ) -> str:
        """
        Analyze problem complexity.
        
        Args:
            problem_statement: The problem description
            context_bundle: The gathered context
            
        Returns:
            Complexity level ('low', 'medium', 'high')
        """
        # Simple heuristics for complexity
        indicators = {
            'high': ['multiple', 'refactor', 'architecture', 'redesign', 'complex'],
            'low': ['typo', 'simple', 'minor', 'documentation', 'comment']
        }
        
        problem_lower = problem_statement.lower()
        
        # Check for complexity indicators
        for indicator in indicators['high']:
            if indicator in problem_lower:
                return 'high'
        
        for indicator in indicators['low']:
            if indicator in problem_lower:
                return 'low'
        
        # Check context size (if context bundle provided)
        if context_bundle is not None:
            num_chunks = len(context_bundle.chunks)
            if num_chunks > 10:
                return 'high'
            elif num_chunks < 3:
                return 'low'
        
        return 'medium'
