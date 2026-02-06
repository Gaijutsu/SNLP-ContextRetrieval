"""Sliding window code chunking implementation.

This module provides a simple sliding window approach to chunking code,
useful for files that can't be parsed by AST or for language-agnostic chunking.
"""

from typing import List, Dict, Optional, Any, Iterator
import logging

from .base import CodeChunker
from ..base import CodeChunk

logger = logging.getLogger(__name__)


class SlidingWindowChunker(CodeChunker):
    """Chunk code using a sliding window approach.
    
    This chunker creates fixed-size chunks with configurable overlap.
    It's language-agnostic and works with any text file, making it useful
    for files that can't be parsed by AST-based chunkers.
    
    The chunker attempts to break at natural boundaries (newlines, functions)
    when possible to preserve semantic coherence.
    
    Example:
        ```python
        chunker = SlidingWindowChunker(config={
            "chunk_size": 1000,
            "chunk_overlap": 200,
            "respect_boundaries": True
        })
        chunks = chunker.chunk_file("example.py")
        ```
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the sliding window chunker.
        
        Args:
            config: Configuration dictionary with options:
                - chunk_size: Target size of each chunk in characters (default: 1000)
                - chunk_overlap: Overlap between consecutive chunks (default: 200)
                - respect_boundaries: Try to break at line boundaries (default: True)
                - boundary_chars: Characters that indicate good break points
                - min_chunk_size: Minimum chunk size to keep (default: 100)
                - line_based: Use line count instead of character count (default: False)
                - lines_per_chunk: Lines per chunk if line_based (default: 50)
                - line_overlap: Line overlap if line_based (default: 10)
        """
        super().__init__(config)
        self.chunk_size = config.get("chunk_size", 1000)
        self.chunk_overlap = config.get("chunk_overlap", 200)
        self.respect_boundaries = config.get("respect_boundaries", True)
        self.boundary_chars = config.get("boundary_chars", ["\n\n", "\n", ". ", "; "])
        self.min_chunk_size = config.get("min_chunk_size", 100)
        self.line_based = config.get("line_based", False)
        self.lines_per_chunk = config.get("lines_per_chunk", 50)
        self.line_overlap = config.get("line_overlap", 10)
    
    def chunk_file(self, file_path: str, content: Optional[str] = None) -> List[CodeChunk]:
        """Chunk a file using sliding window.
        
        Args:
            file_path: Path to the file to chunk.
            content: Optional file content (if already loaded).
            
        Returns:
            List of CodeChunk objects.
        """
        if content is None:
            content = self._read_file(file_path)
        
        if not content.strip():
            return []
        
        if self.line_based:
            return self._chunk_by_lines(content, file_path)
        else:
            return self._chunk_by_chars(content, file_path)
    
    def _chunk_by_lines(self, content: str, file_path: str) -> List[CodeChunk]:
        """Chunk file by line count."""
        lines = content.split("\n")
        chunks = []
        
        start_line = 0
        chunk_id = 0
        
        while start_line < len(lines):
            end_line = min(start_line + self.lines_per_chunk, len(lines))
            
            # Extract chunk lines
            chunk_lines = lines[start_line:end_line]
            chunk_content = "\n".join(chunk_lines)
            
            # Create chunk if it meets minimum size
            if len(chunk_content.strip()) >= self.min_chunk_size:
                chunk = CodeChunk(
                    content=chunk_content,
                    file_path=file_path,
                    start_line=start_line + 1,  # 1-indexed
                    end_line=end_line,
                    chunk_type="sliding_window",
                    name=f"chunk_{chunk_id}",
                    imports=[],
                    metadata={
                        "chunk_id": chunk_id,
                        "chunk_method": "line_based",
                        "line_count": end_line - start_line,
                    }
                )
                chunks.append(chunk)
                chunk_id += 1
            
            # Move window
            start_line += self.lines_per_chunk - self.line_overlap
            
            # Prevent infinite loop
            if start_line >= end_line:
                start_line = end_line
        
        return chunks
    
    def _chunk_by_chars(self, content: str, file_path: str) -> List[CodeChunk]:
        """Chunk file by character count with overlap."""
        chunks = []
        start_pos = 0
        chunk_id = 0
        
        # Track line numbers
        lines = content.split("\n")
        line_starts = [0]
        for line in lines[:-1]:
            line_starts.append(line_starts[-1] + len(line) + 1)  # +1 for newline
        
        def pos_to_line(pos: int) -> int:
            """Convert character position to line number."""
            for i, line_start in enumerate(line_starts):
                if line_start > pos:
                    return i
            return len(line_starts)
        
        while start_pos < len(content):
            end_pos = min(start_pos + self.chunk_size, len(content))
            
            # Try to find a good boundary if respecting boundaries
            if self.respect_boundaries and end_pos < len(content):
                best_break = end_pos
                for boundary in self.boundary_chars:
                    # Look for boundary within last 20% of chunk
                    search_start = start_pos + int(self.chunk_size * 0.8)
                    pos = content.rfind(boundary, search_start, end_pos + len(boundary))
                    if pos > search_start and pos < best_break:
                        best_break = pos + len(boundary)
                end_pos = best_break
            
            # Extract chunk content
            chunk_content = content[start_pos:end_pos]
            
            # Create chunk if it meets minimum size
            if len(chunk_content.strip()) >= self.min_chunk_size:
                start_line = pos_to_line(start_pos)
                end_line = pos_to_line(end_pos)
                
                chunk = CodeChunk(
                    content=chunk_content,
                    file_path=file_path,
                    start_line=start_line,
                    end_line=end_line,
                    chunk_type="sliding_window",
                    name=f"chunk_{chunk_id}",
                    imports=[],
                    metadata={
                        "chunk_id": chunk_id,
                        "chunk_method": "char_based",
                        "char_count": len(chunk_content),
                    }
                )
                chunks.append(chunk)
                chunk_id += 1
            
            # Move window with overlap
            start_pos = end_pos - self.chunk_overlap
            
            # Prevent infinite loop
            if start_pos >= end_pos:
                start_pos = end_pos
        
        return chunks
    
    def chunk_text(self, text: str, source: str = "unknown") -> List[CodeChunk]:
        """Chunk arbitrary text (not from a file).
        
        Args:
            text: Text content to chunk.
            source: Source identifier for the chunks.
            
        Returns:
            List of CodeChunk objects.
        """
        if self.line_based:
            return self._chunk_by_lines(text, source)
        else:
            return self._chunk_by_chars(text, source)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get chunking statistics."""
        stats = super().get_stats()
        stats.update({
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "respect_boundaries": self.respect_boundaries,
            "line_based": self.line_based,
            "lines_per_chunk": self.lines_per_chunk if self.line_based else None,
            "line_overlap": self.line_overlap if self.line_based else None,
        })
        return stats