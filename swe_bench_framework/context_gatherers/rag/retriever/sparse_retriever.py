"""Sparse retrieval implementation using BM25.

This module provides BM25-based sparse retrieval for code search.
"""

from typing import List, Optional
import logging

from .base import Retriever
from ..indexer.bm25_indexer import BM25Indexer
from ..base import SearchResult

logger = logging.getLogger(__name__)


class BM25Retriever(Retriever):
    """BM25-based sparse retriever.
    
    Uses BM25 algorithm for keyword-based retrieval. BM25 is effective for:
    - Exact keyword matching
    - Finding code with specific function/class names
    - Matching technical terms and identifiers
    
    Example:
        ```python
        indexer = BM25Indexer(config={"k1": 1.5, "b": 0.75})
        indexer.build_index(chunks, "/path/to/index")
        
        retriever = BM25Retriever(indexer, config={"top_k": 20})
        results = retriever.retrieve("handle json decode error")
        ```
    """
    
    def __init__(self, indexer: BM25Indexer, config: dict):
        """Initialize the BM25 retriever.
        
        Args:
            indexer: BM25Indexer instance.
            config: Configuration dictionary with options:
                - top_k: Number of results to retrieve (default: 10)
                - min_score: Minimum BM25 score threshold (default: 0.0)
                - deduplicate: Remove duplicate chunks (default: True)
        """
        super().__init__(indexer, config)
        self.bm25_indexer = indexer
    
    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[SearchResult]:
        """Retrieve relevant code chunks using BM25.
        
        Args:
            query: Search query string.
            top_k: Override default top_k if provided.
            
        Returns:
            List of SearchResult objects sorted by BM25 score.
        """
        k = top_k or self.top_k
        
        # Search the index
        results = self.bm25_indexer.search(query, k)
        
        # Apply filtering
        results = self._filter_results(results)
        
        # Limit to top_k
        results = results[:k]
        
        return results
    
    def retrieve_with_scores(
        self,
        query: str,
        top_k: Optional[int] = None
    ) -> tuple[List[SearchResult], list[float]]:
        """Retrieve results with their raw BM25 scores.
        
        Args:
            query: Search query string.
            top_k: Override default top_k if provided.
            
        Returns:
            Tuple of (results, raw_scores).
        """
        results = self.retrieve(query, top_k)
        scores = [r.score for r in results]
        return results, scores


class MultiFieldBM25Retriever(BM25Retriever):
    """BM25 retriever with field-specific boosting.
    
    Allows different weights for different fields (content, name, imports, etc.)
    for more precise retrieval.
    
    Example:
        ```python
        retriever = MultiFieldBM25Retriever(
            indexer,
            config={
                "top_k": 20,
                "field_weights": {"name": 3.0, "content": 1.0, "imports": 0.5}
            }
        )
        ```
    """
    
    def __init__(self, indexer: BM25Indexer, config: dict):
        """Initialize the multi-field retriever.
        
        Args:
            indexer: BM25Indexer instance.
            config: Configuration dictionary with options:
                - field_weights: Dict of field names to weights
                - name_boost: Boost for name field (default: 3.0)
                - content_boost: Boost for content field (default: 1.0)
        """
        super().__init__(indexer, config)
        
        # Set up field weights
        self.field_weights = config.get("field_weights", {})
        if not self.field_weights:
            self.field_weights = {
                "name": config.get("name_boost", 3.0),
                "content": config.get("content_boost", 1.0),
                "imports": config.get("imports_boost", 0.5),
            }
    
    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[SearchResult]:
        """Retrieve with field-aware scoring.
        
        This implementation first retrieves with standard BM25, then
        adjusts scores based on field weights.
        """
        k = top_k or self.top_k
        
        # Get initial results
        results = self.bm25_indexer.search(query, k * 2)  # Get more for re-ranking
        
        # Adjust scores based on field weights
        for result in results:
            boost = 1.0
            
            # Apply name boost if query matches chunk name
            if result.chunk.name and query.lower() in result.chunk.name.lower():
                boost *= self.field_weights.get("name", 3.0)
            
            # Apply import boost if query matches imports
            if result.chunk.imports:
                import_text = " ".join(result.chunk.imports).lower()
                if any(term.lower() in import_text for term in query.split()):
                    boost *= self.field_weights.get("imports", 0.5)
            
            result.score *= boost
        
        # Re-sort by adjusted scores
        results.sort(key=lambda r: r.score, reverse=True)
        
        # Apply filtering
        results = self._filter_results(results)
        
        return results[:k]