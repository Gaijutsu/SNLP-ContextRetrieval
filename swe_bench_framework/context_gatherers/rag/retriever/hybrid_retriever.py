"""Hybrid retrieval implementation combining multiple retrievers.

This module provides hybrid retrieval using multiple strategies and
fusing their results.
"""

from typing import List, Dict, Optional, Any
import logging

from .base import Retriever
from .sparse_retriever import BM25Retriever
from .dense_retriever import DenseRetriever
from ..indexer.hybrid_indexer import HybridIndexer
from ..base import SearchResult

logger = logging.getLogger(__name__)


class HybridRetriever(Retriever):
    """Hybrid retriever combining multiple retrieval strategies.
    
    Combines results from multiple retrievers (BM25, Dense, etc.) using
    fusion methods like RRF or weighted sum for improved recall and
    precision.
    
    Example:
        ```python
        bm25_retriever = BM25Retriever(bm25_indexer, config={"top_k": 20})
        dense_retriever = DenseRetriever(dense_indexer, config={"top_k": 20})
        
        hybrid = HybridRetriever(
            [bm25_retriever, dense_retriever],
            config={
                "fusion_method": "rrf",
                "rrf_k": 60,
                "weights": [0.5, 0.5]
            }
        )
        results = hybrid.retrieve("error handling function")
        ```
    """
    
    FUSION_METHODS = ["rrf", "weighted_sum", "interleave"]
    
    def __init__(self, retrievers: List[Retriever], config: dict):
        """Initialize the hybrid retriever.
        
        Args:
            retrievers: List of retriever instances to combine.
            config: Configuration dictionary with options:
                - fusion_method: "rrf", "weighted_sum", or "interleave" (default: "rrf")
                - rrf_k: RRF constant for rrf method (default: 60)
                - weights: List of weights for each retriever (default: equal weights)
                - top_k: Number of final results (default: 10)
                - initial_k: Initial retrieval count per retriever (default: top_k * 3)
        """
        # Use first retriever's indexer as placeholder
        super().__init__(retrievers[0].indexer if retrievers else None, config)
        
        self.retrievers = retrievers
        self.fusion_method = config.get("fusion_method", "rrf")
        self.rrf_k = config.get("rrf_k", 60)
        self.top_k = config.get("top_k", 10)
        self.initial_k = config.get("initial_k", self.top_k * 3)
        
        # Set up weights
        self.weights = config.get("weights")
        if self.weights is None:
            self.weights = [1.0 / len(retrievers)] * len(retrievers) if retrievers else []
        
        # Validate
        if self.fusion_method not in self.FUSION_METHODS:
            raise ValueError(f"Unknown fusion method: {self.fusion_method}")
        
        if len(self.weights) != len(retrievers):
            raise ValueError("Number of weights must match number of retrievers")
    
    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[SearchResult]:
        """Retrieve using all retrievers and fuse results.
        
        Args:
            query: Search query string.
            top_k: Override default top_k if provided.
            
        Returns:
            Fused search results.
        """
        k = top_k or self.top_k
        
        # Retrieve from all retrievers
        all_results = []
        for retriever in self.retrievers:
            results = retriever.retrieve(query, self.initial_k)
            all_results.append(results)
        
        # Fuse results
        if self.fusion_method == "rrf":
            return self._fuse_rrf(all_results, k)
        elif self.fusion_method == "weighted_sum":
            return self._fuse_weighted_sum(all_results, k)
        else:
            return self._fuse_interleave(all_results, k)
    
    def _fuse_rrf(
        self,
        results_list: List[List[SearchResult]],
        top_k: int
    ) -> List[SearchResult]:
        """Fuse results using Reciprocal Rank Fusion.
        
        Args:
            results_list: List of result lists from each retriever.
            top_k: Number of top results to return.
            
        Returns:
            Fused results.
        """
        from collections import defaultdict
        
        def get_key(r: SearchResult) -> str:
            return f"{r.chunk.file_path}:{r.chunk.start_line}:{r.chunk.end_line}"
        
        # Calculate weighted RRF scores
        scores: dict[str, tuple[float, 'CodeChunk']] = defaultdict(lambda: (0.0, None))
        
        for weight, results in zip(self.weights, results_list):
            for rank, result in enumerate(results, start=1):
                key = get_key(result)
                score = weight * (1.0 / (self.rrf_k + rank))
                current_score, _ = scores[key]
                scores[key] = (current_score + score, result.chunk)
        
        # Sort by score
        sorted_results = sorted(scores.items(), key=lambda x: x[1][0], reverse=True)
        
        # Create SearchResult objects
        final_results = []
        for rank, (key, (score, chunk)) in enumerate(sorted_results[:top_k], start=1):
            final_results.append(SearchResult(
                chunk=chunk,
                score=score,
                rank=rank,
                source="hybrid_rrf"
            ))
        
        return final_results
    
    def _fuse_weighted_sum(
        self,
        results_list: List[List[SearchResult]],
        top_k: int
    ) -> List[SearchResult]:
        """Fuse results using weighted sum of normalized scores.
        
        Args:
            results_list: List of result lists from each retriever.
            top_k: Number of top results to return.
            
        Returns:
            Fused results.
        """
        def get_key(r: SearchResult) -> str:
            return f"{r.chunk.file_path}:{r.chunk.start_line}:{r.chunk.end_line}"
        
        def normalize_scores(results: List[SearchResult]) -> Dict[str, float]:
            if not results:
                return {}
            max_score = max(r.score for r in results) if results else 1.0
            min_score = min(r.score for r in results) if results else 0.0
            score_range = max_score - min_score if max_score > min_score else 1.0
            
            return {
                get_key(r): (r.score - min_score) / score_range
                for r in results
            }
        
        # Normalize and combine scores
        all_keys = set()
        normalized_scores = []
        
        for results in results_list:
            norm = normalize_scores(results)
            normalized_scores.append(norm)
            all_keys.update(norm.keys())
        
        # Calculate weighted sum
        combined_scores: dict[str, tuple[float, Optional['CodeChunk']]] = {}
        
        for key in all_keys:
            score = sum(
                weight * norm.get(key, 0)
                for weight, norm in zip(self.weights, normalized_scores)
            )
            
            # Find the chunk
            chunk = None
            for results in results_list:
                for r in results:
                    if get_key(r) == key:
                        chunk = r.chunk
                        break
                if chunk:
                    break
            
            combined_scores[key] = (score, chunk)
        
        # Sort and create results
        sorted_results = sorted(
            combined_scores.items(),
            key=lambda x: x[1][0],
            reverse=True
        )[:top_k]
        
        final_results = []
        for rank, (key, (score, chunk)) in enumerate(sorted_results, start=1):
            if chunk:
                final_results.append(SearchResult(
                    chunk=chunk,
                    score=score,
                    rank=rank,
                    source="hybrid_weighted"
                ))
        
        return final_results
    
    def _fuse_interleave(
        self,
        results_list: List[List[SearchResult]],
        top_k: int
    ) -> List[SearchResult]:
        """Fuse results by weighted interleaving.
        
        Args:
            results_list: List of result lists.
            top_k: Number of top results to return.
            
        Returns:
            Interleaved results.
        """
        seen = set()
        final_results = []
        
        # Calculate how many to take from each retriever based on weights
        counts = [int(w * top_k * 2) for w in self.weights]
        
        # Interleave results
        max_pos = max(len(r) for r in results_list) if results_list else 0
        
        for pos in range(max_pos):
            for i, (results, count) in enumerate(zip(results_list, counts)):
                if pos < len(results) and len(final_results) < top_k:
                    result = results[pos]
                    key = f"{result.chunk.file_path}:{result.chunk.start_line}:{result.chunk.end_line}"
                    
                    if key not in seen:
                        seen.add(key)
                        final_results.append(result)
        
        # Update ranks
        for rank, result in enumerate(final_results, start=1):
            result.rank = rank
            result.source = "hybrid_interleave"
        
        return final_results[:top_k]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get retriever statistics."""
        stats = super().get_stats()
        stats.update({
            "fusion_method": self.fusion_method,
            "rrf_k": self.rrf_k,
            "weights": self.weights,
            "num_retrievers": len(self.retrievers),
            "retriever_stats": [r.get_stats() for r in self.retrievers],
        })
        return stats


class HybridIndexerRetriever(Retriever):
    """Retriever that uses a HybridIndexer directly.
    
    This is a convenience wrapper around HybridIndexer that provides
    the Retriever interface.
    
    Example:
        ```python
        indexer = HybridIndexer(config={"bm25_weight": 0.5, "dense_weight": 0.5})
        indexer.build_index(chunks, "/path/to/index")
        
        retriever = HybridIndexerRetriever(indexer, config={"top_k": 20})
        results = retriever.retrieve("error handling")
        ```
    """
    
    def __init__(self, indexer: HybridIndexer, config: dict):
        """Initialize the retriever.
        
        Args:
            indexer: HybridIndexer instance.
            config: Configuration dictionary.
        """
        super().__init__(indexer, config)
        self.hybrid_indexer = indexer
    
    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[SearchResult]:
        """Retrieve using the hybrid indexer.
        
        Args:
            query: Search query string.
            top_k: Override default top_k if provided.
            
        Returns:
            Search results.
        """
        k = top_k or self.top_k
        
        results = self.hybrid_indexer.search(query, k)
        results = self._filter_results(results)
        
        return results