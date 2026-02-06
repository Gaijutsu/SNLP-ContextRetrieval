"""Base class for retrieval strategies.

Retrievers search indexes and return relevant code chunks.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Any, Callable
import logging

from ..base import SearchResult, CodeChunk, BaseRAGComponent
from ..indexer.base import RepositoryIndexer

logger = logging.getLogger(__name__)


class Retriever(BaseRAGComponent, ABC):
    """Abstract base class for retrieval strategies.
    
    Retrievers are responsible for:
    1. Searching indexes to find relevant code
    2. Post-processing and filtering results
    3. Potentially re-ranking results
    
    Different retrieval strategies include:
    - Sparse: BM25-based keyword matching
    - Dense: Vector similarity search
    - Hybrid: Combination of multiple methods
    
    Example:
        ```python
        retriever = BM25Retriever(indexer, config={"top_k": 20})
        results = retriever.retrieve("how to handle errors")
        ```
    """
    
    def __init__(self, indexer: RepositoryIndexer, config: Dict[str, Any]):
        """Initialize the retriever.
        
        Args:
            indexer: The indexer to search.
            config: Configuration dictionary with options:
                - top_k: Number of results to retrieve (default: 10)
                - min_score: Minimum score threshold (default: 0.0)
                - deduplicate: Remove duplicate chunks (default: True)
                - filter_fn: Optional filter function for results
        """
        super().__init__(config)
        self.indexer = indexer
        self.top_k = config.get("top_k", 10)
        self.min_score = config.get("min_score", 0.0)
        self.deduplicate = config.get("deduplicate", True)
        self.filter_fn: Optional[Callable[[SearchResult], bool]] = config.get("filter_fn")
    
    @abstractmethod
    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[SearchResult]:
        """Retrieve relevant code chunks for a query.
        
        Args:
            query: Search query string.
            top_k: Override default top_k if provided.
            
        Returns:
            List of SearchResult objects sorted by relevance.
        """
        pass
    
    def batch_retrieve(self, queries: List[str], top_k: Optional[int] = None) -> List[List[SearchResult]]:
        """Retrieve for multiple queries.
        
        Args:
            queries: List of search query strings.
            top_k: Override default top_k if provided.
            
        Returns:
            List of search results for each query.
        """
        return [self.retrieve(q, top_k) for q in queries]
    
    def _filter_results(self, results: List[SearchResult]) -> List[SearchResult]:
        """Filter results based on configuration.
        
        Args:
            results: Raw search results.
            
        Returns:
            Filtered results.
        """
        # Filter by minimum score
        if self.min_score > 0:
            results = [r for r in results if r.score >= self.min_score]
        
        # Apply custom filter
        if self.filter_fn:
            results = [r for r in results if self.filter_fn(r)]
        
        # Deduplicate
        if self.deduplicate:
            results = self._deduplicate_results(results)
        
        return results
    
    def _deduplicate_results(self, results: List[SearchResult]) -> List[SearchResult]:
        """Remove duplicate chunks from results.
        
        Args:
            results: Search results potentially containing duplicates.
            
        Returns:
            Deduplicated results (keeps highest scoring duplicate).
        """
        seen: Dict[str, SearchResult] = {}
        
        for result in results:
            # Create unique key for chunk
            key = f"{result.chunk.file_path}:{result.chunk.start_line}:{result.chunk.end_line}"
            
            if key not in seen or result.score > seen[key].score:
                seen[key] = result
        
        # Return in original order (sorted by score)
        return list(seen.values())
    
    def _rerank_results(
        self,
        results: List[SearchResult],
        reranker: 'Reranker'
    ) -> List[SearchResult]:
        """Re-rank results using a reranker.
        
        Args:
            results: Results to re-rank.
            reranker: Reranker instance.
            
        Returns:
            Re-ranked results.
        """
        return reranker.rerank(results)
    
    def initialize(self, **kwargs) -> None:
        """Initialize the retriever."""
        if not self.indexer.is_index_loaded():
            logger.warning("Indexer does not have a loaded index")
        self._initialized = True
        logger.info(f"Initialized {self.name}")
    
    def cleanup(self) -> None:
        """Cleanup resources."""
        self._initialized = False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get retriever statistics."""
        stats = super().get_stats()
        stats.update({
            "top_k": self.top_k,
            "min_score": self.min_score,
            "deduplicate": self.deduplicate,
            "indexer_type": self.indexer.name if self.indexer else None,
        })
        return stats


class Reranker(ABC):
    """Abstract base class for result re-rankers.
    
    Re-rankers take initial retrieval results and re-score them
    for better accuracy, typically using cross-encoders.
    """
    
    @abstractmethod
    def rerank(self, results: List[SearchResult], query: Optional[str] = None) -> List[SearchResult]:
        """Re-rank search results.
        
        Args:
            results: Initial search results.
            query: Optional query for context-aware re-ranking.
            
        Returns:
            Re-ranked results.
        """
        pass
    
    @abstractmethod
    def rerank_batch(
        self,
        results_list: List[List[SearchResult]],
        queries: List[str]
    ) -> List[List[SearchResult]]:
        """Re-rank multiple result sets.
        
        Args:
            results_list: List of search result lists.
            queries: Corresponding queries.
            
        Returns:
            Re-ranked results for each query.
        """
        pass