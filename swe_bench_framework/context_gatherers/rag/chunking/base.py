"""Base class for code chunking strategies.

Code chunkers split repository code into manageable pieces for indexing and retrieval.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Any, Iterator
from pathlib import Path
import logging

from ..base import CodeChunk, BaseRAGComponent

logger = logging.getLogger(__name__)


class CodeChunker(BaseRAGComponent, ABC):
    """Abstract base class for code chunking strategies.
    
    Chunkers are responsible for splitting repository code into semantic units
    that can be indexed and retrieved. Different strategies include:
    - AST-based: Split by functions, classes, methods
    - Sliding window: Fixed-size chunks with overlap
    - Hierarchical: Multi-level chunking
    
    Example:
        ```python
        chunker = ASTChunker(config={"include_imports": True})
        chunks = chunker.chunk_repository("/path/to/repo")
        for chunk in chunks:
            print(f"{chunk.file_path}:{chunk.start_line}-{chunk.end_line}")
        ```
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the chunker.
        
        Args:
            config: Configuration dictionary with common options:
                - file_extensions: List of file extensions to process (default: [".py"])
                - exclude_patterns: List of glob patterns to exclude
                - max_file_size: Maximum file size in bytes to process
                - min_chunk_size: Minimum chunk size in characters
        """
        super().__init__(config)
        self.file_extensions = config.get("file_extensions", [".py"])
        self.exclude_patterns = config.get("exclude_patterns", [
            "**/test*",
            "**/tests/**",
            "**/__pycache__/**",
            "**/*.pyc",
            "**/node_modules/**",
            "**/.git/**",
            "**/build/**",
            "**/dist/**",
            "**/.venv/**",
            "**/venv/**",
        ])
        self.max_file_size = config.get("max_file_size", 1024 * 1024)  # 1MB
        self.min_chunk_size = config.get("min_chunk_size", 50)  # characters
    
    def initialize(self, **kwargs) -> None:
        """Initialize the chunker."""
        self._initialized = True
        logger.info(f"Initialized {self.name}")
    
    def cleanup(self) -> None:
        """Cleanup resources."""
        self._initialized = False
    
    @abstractmethod
    def chunk_file(self, file_path: str, content: Optional[str] = None) -> List[CodeChunk]:
        """Chunk a single file.
        
        Args:
            file_path: Path to the file to chunk.
            content: Optional file content (if already loaded).
            
        Returns:
            List of CodeChunk objects.
        """
        pass
    
    def chunk_repository(self, repo_path: str) -> List[CodeChunk]:
        """Chunk all files in a repository.
        
        Args:
            repo_path: Path to the repository root.
            
        Returns:
            List of all CodeChunk objects from the repository.
        """
        import fnmatch
        import os
        
        repo_path = Path(repo_path)
        all_chunks = []
        processed_files = 0
        skipped_files = 0
        
        logger.info(f"Starting repository chunking: {repo_path}")
        
        for file_path in self._iter_files(repo_path):
            try:
                # Check exclude patterns
                rel_path = str(file_path.relative_to(repo_path))
                if self._should_exclude(rel_path):
                    skipped_files += 1
                    continue
                
                # Check file size
                if file_path.stat().st_size > self.max_file_size:
                    logger.warning(f"Skipping large file: {rel_path}")
                    skipped_files += 1
                    continue
                
                # Chunk the file
                chunks = self.chunk_file(str(file_path))
                all_chunks.extend(chunks)
                processed_files += 1
                
                if processed_files % 100 == 0:
                    logger.info(f"Processed {processed_files} files, created {len(all_chunks)} chunks")
                    
            except Exception as e:
                logger.error(f"Error chunking {file_path}: {e}")
                skipped_files += 1
                continue
        
        logger.info(
            f"Repository chunking complete: {processed_files} files processed, "
            f"{skipped_files} files skipped, {len(all_chunks)} chunks created"
        )
        
        self._update_stats(0.0)
        return all_chunks
    
    def _iter_files(self, repo_path: Path) -> Iterator[Path]:
        """Iterate over all files in the repository matching extensions."""
        for ext in self.file_extensions:
            yield from repo_path.rglob(f"*{ext}")
    
    def _should_exclude(self, rel_path: str) -> bool:
        """Check if a file should be excluded based on patterns."""
        import fnmatch
        
        for pattern in self.exclude_patterns:
            if fnmatch.fnmatch(rel_path, pattern):
                return True
        return False
    
    def _read_file(self, file_path: str) -> str:
        """Read file content with proper encoding handling."""
        encodings = ["utf-8", "latin-1", "cp1252"]
        
        for encoding in encodings:
            try:
                with open(file_path, "r", encoding=encoding) as f:
                    return f.read()
            except UnicodeDecodeError:
                continue
        
        raise ValueError(f"Could not decode file: {file_path}")
    
    def _filter_small_chunks(self, chunks: List[CodeChunk]) -> List[CodeChunk]:
        """Filter out chunks that are too small."""
        return [c for c in chunks if len(c.content.strip()) >= self.min_chunk_size]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get chunking statistics."""
        stats = super().get_stats()
        stats.update({
            "file_extensions": self.file_extensions,
            "exclude_patterns": len(self.exclude_patterns),
            "max_file_size": self.max_file_size,
            "min_chunk_size": self.min_chunk_size,
        })
        return stats