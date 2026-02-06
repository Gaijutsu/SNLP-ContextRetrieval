"""Code chunking module for RAG.

This module provides different strategies for splitting code into
semantic chunks for indexing and retrieval.
"""

from .base import CodeChunker
from .ast_chunker import ASTChunker
from .sliding_chunker import SlidingWindowChunker

__all__ = [
    "CodeChunker",
    "ASTChunker",
    "SlidingWindowChunker",
]