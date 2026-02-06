"""Base classes for RAG (Retrieval-Augmented Generation) context gathering.

This module provides the foundational abstractions for RAG-based context gathering
in the SWE-bench comparison framework.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple
from datetime import datetime
import logging

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class CodeChunk:
    """Represents a chunk of code from the repository.
    
    Attributes:
        content: The actual code content.
        file_path: Path to the source file.
        start_line: Starting line number (1-indexed).
        end_line: Ending line number (1-indexed).
        chunk_type: Type of chunk (function, class, method, etc.).
        name: Name of the function/class/method if applicable.
        imports: List of imports included with this chunk.
        metadata: Additional metadata about the chunk.
    """
    content: str
    file_path: str
    start_line: int
    end_line: int
    chunk_type: str = "code"  # function, class, method, module, etc.
    name: Optional[str] = None
    imports: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate chunk data."""
        if self.start_line > self.end_line:
            raise ValueError(f"start_line ({self.start_line}) must be <= end_line ({self.end_line})")
        if not self.content.strip():
            logger.warning(f"Empty chunk created for {self.file_path}")


@dataclass
class SearchResult:
    """Result from a retrieval search.
    
    Attributes:
        chunk: The retrieved code chunk.
        score: Relevance score (higher is better).
        rank: Rank position in results (1-indexed).
        source: Which retriever produced this result.
    """
    chunk: CodeChunk
    score: float
    rank: int
    source: str = "unknown"
    
    def to_context_chunk(self, context_type: str = "retrieved_code") -> 'ContextChunk':
        """Convert SearchResult to ContextChunk for the framework."""
        from ..base import ContextChunk as BaseContextChunk, ContextType
        
        # Import here to avoid circular imports
        from ..base import ContextChunk as BaseContextChunk, ContextType
        
        # Map chunk_type to ContextType
        type_mapping = {
            "function": ContextType.FUNCTION_DEFINITION,
            "method": ContextType.FUNCTION_DEFINITION,
            "class": ContextType.CLASS_DEFINITION,
            "import": ContextType.IMPORT_DEPENDENCY,
            "test": ContextType.TEST_CONTEXT,
            "error": ContextType.ERROR_CONTEXT,
            "module": ContextType.FILE_CONTENT,
        }
        ctx_type = type_mapping.get(self.chunk.chunk_type, ContextType.FILE_CONTENT)
        
        return BaseContextChunk(
            content=self.chunk.content,
            source_file=self.chunk.file_path,
            context_type=ctx_type,
            start_line=self.chunk.start_line,
            end_line=self.chunk.end_line,
            relevance_score=self.score,
            metadata={
                **self.chunk.metadata,
                "chunk_name": self.chunk.name,
                "chunk_type": self.chunk.chunk_type,
                "imports": self.chunk.imports,
                "search_rank": self.rank,
                "search_source": self.source,
            }
        )


@dataclass
class RAGContextBundle:
    """Context bundle specific to RAG gathering.
    
    This extends the framework's ContextBundle with RAG-specific metadata.
    """
    instance_id: str
    problem_statement: str
    chunks: List['ContextChunk']
    repo_structure: Dict[str, Any]
    gathered_at: str
    gatherer_type: str
    token_count: int
    retrieval_metadata: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_framework_bundle(self) -> 'ContextBundle':
        """Convert to the framework's standard ContextBundle."""
        from ..base import ContextBundle
        
        return ContextBundle(
            instance_id=self.instance_id,
            problem_statement=self.problem_statement,
            chunks=self.chunks,
            repo_structure=self.repo_structure,
            gathered_at=self.gathered_at,
            gatherer_type=self.gatherer_type,
            token_count=self.token_count,
            metadata={
                **self.metadata,
                "rag_metadata": self.retrieval_metadata,
            }
        )


class BaseRAGComponent(ABC):
    """Base class for all RAG components.
    
    Provides common functionality for chunkers, indexers, retrievers, etc.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the RAG component.
        
        Args:
            config: Configuration dictionary for the component.
        """
        self.config = config
        self.name = self.__class__.__name__
        self._initialized = False
        self._stats: Dict[str, Any] = {
            "calls": 0,
            "errors": 0,
            "total_time": 0.0,
        }
    
    @abstractmethod
    def initialize(self, **kwargs) -> None:
        """Initialize the component with any required resources."""
        pass
    
    @abstractmethod
    def cleanup(self) -> None:
        """Cleanup any resources used by the component."""
        pass
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about component usage."""
        return self._stats.copy()
    
    def _update_stats(self, duration: float, error: bool = False) -> None:
        """Update component statistics."""
        self._stats["calls"] += 1
        self._stats["total_time"] += duration
        if error:
            self._stats["errors"] += 1
    
    def is_initialized(self) -> bool:
        """Check if the component has been initialized."""
        return self._initialized


class BaseRAGGatherer(BaseRAGComponent):
    """Base class for RAG-based context gatherers.
    
    This class extends the framework's ContextGatherer interface with
    RAG-specific functionality. It orchestrates the entire RAG pipeline:
    chunking → indexing → retrieval → context assembly.
    
    Example:
        ```python
        config = {
            "chunker": {"type": "ast", "include_imports": True},
            "indexer": {"type": "hybrid", "bm25_weight": 0.7},
            "retriever": {"top_k": 20, "use_reranker": True},
        }
        gatherer = RAGGatherer(config)
        gatherer.initialize(repo_path="/path/to/repo")
        context = gatherer.gather_context(instance, repo_path)
        ```
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the RAG gatherer.
        
        Args:
            config: Configuration dictionary containing settings for:
                - chunker: Chunking configuration
                - indexer: Indexing configuration
                - retriever: Retrieval configuration
                - query_generator: Query generation configuration
                - context_assembly: Context assembly configuration
        """
        super().__init__(config)
        self.chunker = None
        self.indexer = None
        self.retriever = None
        self.query_generator = None
        self._repo_path: Optional[str] = None
        self._index_path: Optional[str] = None
    
    @abstractmethod
    def gather_context(
        self,
        instance: 'SWEInstance',
        repo_path: str
    ) -> 'ContextBundle':
        """Gather context for a given SWE-bench instance using RAG.
        
        This method orchestrates the full RAG pipeline:
        1. Generate search queries from the problem statement
        2. Retrieve relevant code chunks using the configured retriever
        3. Assemble context bundle with token budgeting
        
        Args:
            instance: The SWE-bench instance containing problem statement.
            repo_path: Path to the repository checkout.
            
        Returns:
            ContextBundle containing all gathered context.
        """
        pass
    
    def build_index(self, repo_path: str, output_path: str) -> None:
        """Build the search index for a repository.
        
        This is typically called once per repository and the index
        is reused across multiple instances.
        
        Args:
            repo_path: Path to the repository to index.
            output_path: Path where the index should be saved.
        """
        if self.chunker is None or self.indexer is None:
            raise RuntimeError("Chunker and indexer must be initialized before building index")
        
        logger.info(f"Building index for {repo_path} -> {output_path}")
        
        # Step 1: Chunk the repository
        chunks = self.chunker.chunk_repository(repo_path)
        logger.info(f"Created {len(chunks)} chunks from repository")
        
        # Step 2: Build index from chunks
        self.indexer.build_index(chunks, output_path)
        logger.info(f"Index saved to {output_path}")
        
        self._index_path = output_path
    
    def load_index(self, index_path: str) -> None:
        """Load a pre-built index.
        
        Args:
            index_path: Path to the saved index.
        """
        if self.indexer is None:
            raise RuntimeError("Indexer must be initialized before loading index")
        
        logger.info(f"Loading index from {index_path}")
        self.indexer.load_index(index_path)
        self._index_path = index_path
    
    def is_index_built(self, repo_path: str) -> bool:
        """Check if an index exists for the given repository.
        
        Args:
            repo_path: Path to the repository.
            
        Returns:
            True if a valid index exists, False otherwise.
        """
        import os
        
        if self._index_path and os.path.exists(self._index_path):
            return True
        
        # Check for default index location
        default_index = os.path.join(repo_path, ".rag_index")
        if os.path.exists(default_index):
            self._index_path = default_index
            return True
        
        return False
    
    def get_index_path(self) -> Optional[str]:
        """Get the path to the current index."""
        return self._index_path