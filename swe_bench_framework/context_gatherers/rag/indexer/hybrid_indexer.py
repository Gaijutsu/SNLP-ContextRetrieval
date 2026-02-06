"""Hybrid indexing implementation combining BM25 and Dense retrieval.

This module provides hybrid retrieval using Reciprocal Rank Fusion (RRF)
to combine results from BM25 and dense embedding indexes.
"""

from typing import List, Dict, Optional, Any, Tuple
from pathlib import Path
import logging

from .base import RepositoryIndexer
from .bm25_indexer import BM25Indexer
from .dense_indexer import DenseIndexer
from ..base import CodeChunk, SearchResult

logger = logging.getLogger(__name__)


class HybridIndexer(RepositoryIndexer):
    """Hybrid indexer combining BM25 and Dense retrieval with RRF.
    
    Hybrid retrieval combines sparse (BM25) and dense (embedding) methods
    to leverage strengths of both approaches:
    - BM25: Good at exact keyword matching
    - Dense: Good at semantic similarity
    
    Uses Reciprocal Rank Fusion (RRF) to combine results:
    RRF_Score(d) = Σ(1 / (k + rank_i))
    
    Where k is a constant (typically 60) and rank_i is the position of
    document d in result list i.
    
    Example:
        ```python
        indexer = HybridIndexer(config={
            "bm25_weight": 0.5,
            "dense_weight": 0.5,
            "rrf_k": 60,
            "bm25": {"k1": 1.5, "b": 0.75},
            "dense": {"model": "jina-embeddings-v2-base-code"}
        })
        indexer.build_index(chunks, "/path/to/index")
        results = indexer.search("error handling", top_k=10)
        ```
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the hybrid indexer.
        
        Args:
            config: Configuration dictionary with options:
                - bm25_weight: Weight for BM25 scores in fusion (default: 0.5)
                - dense_weight: Weight for dense scores in fusion (default: 0.5)
                - rrf_k: RRF constant k (default: 60)
                - top_k_multiplier: Multiplier for initial retrieval (default: 3)
                - bm25: Configuration for BM25 indexer
                - dense: Configuration for dense indexer
                - fusion_method: "rrf" or "weighted_sum" (default: "rrf")
        """
        super().__init__(config)
        
        # Fusion configuration
        self.bm25_weight = config.get("bm25_weight", 0.5)
        self.dense_weight = config.get("dense_weight", 0.5)
        self.rrf_k = config.get("rrf_k", 60)
        self.top_k_multiplier = config.get("top_k_multiplier", 3)
        self.fusion_method = config.get("fusion_method", "rrf")
        
        # Create sub-indexers
        bm25_config = config.get("bm25", {})
        dense_config = config.get("dense", {})
        
        self.bm25_indexer = BM25Indexer(bm25_config)
        self.dense_indexer = DenseIndexer(dense_config)
        
        # Validate weights
        total_weight = self.bm25_weight + self.dense_weight
        if abs(total_weight - 1.0) > 0.01:
            logger.warning(f"Weights don't sum to 1.0 ({total_weight}), normalizing")
            self.bm25_weight /= total_weight
            self.dense_weight /= total_weight
    
    def build_index(self, chunks: List[CodeChunk], output_path: str) -> None:
        """Build hybrid index from code chunks.
        
        Args:
            chunks: List of CodeChunk objects to index.
            output_path: Path where the index should be saved.
        """
        logger.info(f"Building hybrid index with {len(chunks)} chunks")
        
        self._chunks = chunks
        
        # Build both sub-indexes
        Path(output_path).mkdir(parents=True, exist_ok=True)
        
        bm25_path = Path(output_path) / "bm25"
        dense_path = Path(output_path) / "dense"
        
        bm25_path.mkdir(exist_ok=True)
        dense_path.mkdir(exist_ok=True)
        
        self.bm25_indexer.build_index(chunks, str(bm25_path))
        self.dense_indexer.build_index(chunks, str(dense_path))
        
        # Save metadata
        self.save_metadata(output_path)
        self.save_chunks(output_path)
        
        self._index_metadata = {
            "bm25_weight": self.bm25_weight,
            "dense_weight": self.dense_weight,
            "rrf_k": self.rrf_k,
            "fusion_method": self.fusion_method,
            "chunk_count": len(chunks),
        }
        
        logger.info(f"Hybrid index built and saved to {output_path}")
    
    def load_index(self, index_path: str) -> None:
        """Load a pre-built hybrid index.
        
        Args:
            index_path: Path to the saved index directory.
        """
        bm25_path = Path(index_path) / "bm25"
        dense_path = Path(index_path) / "dense"
        
        if not bm25_path.exists() or not dense_path.exists():
            raise FileNotFoundError(f"Hybrid index not found at {index_path}")
        
        # Load both sub-indexes
        self.bm25_indexer.load_index(str(bm25_path))
        self.dense_indexer.load_index(str(dense_path))
        
        # Load chunks and metadata
        self.load_chunks(index_path)
        metadata = self.load_metadata(index_path)
        
        if metadata:
            config = metadata.get("config", {})
            self.bm25_weight = config.get("bm25_weight", self.bm25_weight)
            self.dense_weight = config.get("dense_weight", self.dense_weight)
            self.rrf_k = config.get("rrf_k", self.rrf_k)
            self.fusion_method = config.get("fusion_method", self.fusion_method)
        
        self._initialized = True
        logger.info(f"Loaded hybrid index with {len(self._chunks)} chunks")
    
    def search(self, query: str, top_k: int = 10) -> List[SearchResult]:
        """Search the hybrid index.
        
        Args:
            query: Search query string.
            top_k: Number of top results to return.
            
        Returns:
            List of SearchResult objects sorted by fused score.
        """
        # Retrieve more results initially for better fusion
        initial_k = top_k * self.top_k_multiplier
        
        # Search both indexes
        bm25_results = self.bm25_indexer.search(query, initial_k)
        dense_results = self.dense_indexer.search(query, initial_k)
        
        # Fuse results
        if self.fusion_method == "rrf":
            return self._fuse_rrf(bm25_results, dense_results, top_k)
        else:
            return self._fuse_weighted_sum(bm25_results, dense_results, top_k)
    
    def _fuse_rrf(
        self,
        bm25_results: List[SearchResult],
        dense_results: List[SearchResult],
        top_k: int
    ) -> List[SearchResult]:
        """Fuse results using Reciprocal Rank Fusion.
        
        RRF_Score(d) = Σ(1 / (k + rank_i))
        
        Args:
            bm25_results: Results from BM25 indexer.
            dense_results: Results from dense indexer.
            top_k: Number of top results to return.
            
        Returns:
            Fused search results.
        """
        # Create a unique key for each chunk
        def get_chunk_key(result: SearchResult) -> str:
            return f"{result.chunk.file_path}:{result.chunk.start_line}:{result.chunk.end_line}"
        
        # Calculate RRF scores
        rrf_scores: Dict[str, Tuple[float, CodeChunk]] = {}
        
        # Add BM25 contributions
        for rank, result in enumerate(bm25_results, start=1):
            key = get_chunk_key(result)
            score = self.bm25_weight * (1.0 / (self.rrf_k + rank))
            if key in rrf_scores:
                rrf_scores[key] = (rrf_scores[key][0] + score, result.chunk)
            else:
                rrf_scores[key] = (score, result.chunk)
        
        # Add dense contributions
        for rank, result in enumerate(dense_results, start=1):
            key = get_chunk_key(result)
            score = self.dense_weight * (1.0 / (self.rrf_k + rank))
            if key in rrf_scores:
                rrf_scores[key] = (rrf_scores[key][0] + score, result.chunk)
            else:
                rrf_scores[key] = (score, result.chunk)
        
        # Sort by RRF score
        sorted_results = sorted(
            rrf_scores.items(),
            key=lambda x: x[1][0],
            reverse=True
        )[:top_k]
        
        # Create search results
        results = []
        for rank, (key, (score, chunk)) in enumerate(sorted_results, start=1):
            result = SearchResult(
                chunk=chunk,
                score=score,
                rank=rank,
                source="hybrid_rrf"
            )
            results.append(result)
        
        return results
    
    def _fuse_weighted_sum(
        self,
        bm25_results: List[SearchResult],
        dense_results: List[SearchResult],
        top_k: int
    ) -> List[SearchResult]:
        """Fuse results using weighted sum of normalized scores.
        
        Args:
            bm25_results: Results from BM25 indexer.
            dense_results: Results from dense indexer.
            top_k: Number of top results to return.
            
        Returns:
            Fused search results.
        """
        def get_chunk_key(result: SearchResult) -> str:
            return f"{result.chunk.file_path}:{result.chunk.start_line}:{result.chunk.end_line}"
        
        # Normalize scores for each result set
        def normalize_scores(results: List[SearchResult]) -> Dict[str, float]:
            if not results:
                return {}
            max_score = max(r.score for r in results) if results else 1.0
            min_score = min(r.score for r in results) if results else 0.0
            score_range = max_score - min_score if max_score > min_score else 1.0
            
            return {
                get_chunk_key(r): (r.score - min_score) / score_range
                for r in results
            }
        
        bm25_norm = normalize_scores(bm25_results)
        dense_norm = normalize_scores(dense_results)
        
        # Combine scores
        all_keys = set(bm25_norm.keys()) | set(dense_norm.keys())
        combined_scores: Dict[str, Tuple[float, Optional[CodeChunk]]] = {}
        
        for key in all_keys:
            score = (
                self.bm25_weight * bm25_norm.get(key, 0) +
                self.dense_weight * dense_norm.get(key, 0)
            )
            
            # Get the chunk from either result set
            chunk = None
            for r in bm25_results + dense_results:
                if get_chunk_key(r) == key:
                    chunk = r.chunk
                    break
            
            combined_scores[key] = (score, chunk)
        
        # Sort and create results
        sorted_results = sorted(
            combined_scores.items(),
            key=lambda x: x[1][0],
            reverse=True
        )[:top_k]
        
        results = []
        for rank, (key, (score, chunk)) in enumerate(sorted_results, start=1):
            if chunk:
                result = SearchResult(
                    chunk=chunk,
                    score=score,
                    rank=rank,
                    source="hybrid_weighted"
                )
                results.append(result)
        
        return results
    
    def batch_search(self, queries: List[str], top_k: int = 10) -> List[List[SearchResult]]:
        """Search for multiple queries at once.
        
        Args:
            queries: List of search query strings.
            top_k: Number of top results per query.
            
        Returns:
            List of search results for each query.
        """
        return [self.search(q, top_k) for q in queries]
    
    def search_with_components(
        self,
        query: str,
        top_k: int = 10
    ) -> Tuple[List[SearchResult], List[SearchResult], List[SearchResult]]:
        """Search and return both component results and fused results.
        
        Args:
            query: Search query string.
            top_k: Number of top results to return.
            
        Returns:
            Tuple of (bm25_results, dense_results, fused_results).
        """
        initial_k = top_k * self.top_k_multiplier
        
        bm25_results = self.bm25_indexer.search(query, initial_k)
        dense_results = self.dense_indexer.search(query, initial_k)
        
        if self.fusion_method == "rrf":
            fused_results = self._fuse_rrf(bm25_results, dense_results, top_k)
        else:
            fused_results = self._fuse_weighted_sum(bm25_results, dense_results, top_k)
        
        return bm25_results[:top_k], dense_results[:top_k], fused_results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get indexer statistics."""
        stats = super().get_stats()
        stats.update({
            "bm25_weight": self.bm25_weight,
            "dense_weight": self.dense_weight,
            "rrf_k": self.rrf_k,
            "fusion_method": self.fusion_method,
            "top_k_multiplier": self.top_k_multiplier,
            "bm25_stats": self.bm25_indexer.get_stats(),
            "dense_stats": self.dense_indexer.get_stats(),
        })
        return stats