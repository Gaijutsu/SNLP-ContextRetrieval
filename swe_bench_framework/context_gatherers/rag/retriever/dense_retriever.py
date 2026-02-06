"""Dense retrieval implementation using embeddings.

This module provides vector-based dense retrieval for code search.
"""

from typing import List, Optional
import logging

from .base import Retriever
from ..indexer.dense_indexer import DenseIndexer
from ..base import SearchResult

logger = logging.getLogger(__name__)


class DenseRetriever(Retriever):
    """Dense embedding-based retriever.
    
    Uses vector similarity for semantic retrieval. Dense retrieval is effective for:
    - Semantic similarity matching
    - Finding conceptually related code
    - Handling paraphrases and synonyms
    
    Example:
        ```python
        indexer = DenseIndexer(config={"model": "jina-embeddings-v2-base-code"})
        indexer.build_index(chunks, "/path/to/index")
        
        retriever = DenseRetriever(indexer, config={"top_k": 20})
        results = retriever.retrieve("function to parse json data")
        ```
    """
    
    def __init__(self, indexer: DenseIndexer, config: dict):
        """Initialize the dense retriever.
        
        Args:
            indexer: DenseIndexer instance.
            config: Configuration dictionary with options:
                - top_k: Number of results to retrieve (default: 10)
                - min_score: Minimum similarity threshold (default: 0.0)
                - deduplicate: Remove duplicate chunks (default: True)
        """
        super().__init__(indexer, config)
        self.dense_indexer = indexer
    
    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[SearchResult]:
        """Retrieve relevant code chunks using dense embeddings.
        
        Args:
            query: Search query string.
            top_k: Override default top_k if provided.
            
        Returns:
            List of SearchResult objects sorted by similarity score.
        """
        k = top_k or self.top_k
        
        # Search the index
        results = self.dense_indexer.search(query, k)
        
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
        """Retrieve results with their raw similarity scores.
        
        Args:
            query: Search query string.
            top_k: Override default top_k if provided.
            
        Returns:
            Tuple of (results, raw_scores).
        """
        results = self.retrieve(query, top_k)
        scores = [r.score for r in results]
        return results, scores
    
    def retrieve_similar_to_chunk(
        self,
        chunk: 'CodeChunk',
        top_k: Optional[int] = None,
        exclude_same_file: bool = True
    ) -> List[SearchResult]:
        """Retrieve chunks similar to a given chunk.
        
        Args:
            chunk: Reference chunk.
            top_k: Number of results to retrieve.
            exclude_same_file: Exclude chunks from the same file.
            
        Returns:
            List of similar chunks.
        """
        # Use chunk content as query
        query = chunk.content
        
        results = self.retrieve(query, top_k=top_k or self.top_k)
        
        if exclude_same_file:
            results = [
                r for r in results
                if r.chunk.file_path != chunk.file_path
            ]
        
        return results


class MultiQueryDenseRetriever(DenseRetriever):
    """Dense retriever that expands queries for better coverage.
    
    Generates multiple query variants and combines results for
    improved recall.
    
    Example:
        ```python
        retriever = MultiQueryDenseRetriever(
            indexer,
            config={
                "top_k": 20,
                "query_expansion": True,
                "expansion_variants": 3
            }
        )
        ```
    """
    
    def __init__(self, indexer: DenseIndexer, config: dict):
        """Initialize the multi-query retriever.
        
        Args:
            indexer: DenseIndexer instance.
            config: Configuration dictionary with options:
                - query_expansion: Enable query expansion (default: True)
                - expansion_variants: Number of variants to generate (default: 3)
                - fusion_method: "rrf" or "interleave" (default: "rrf")
        """
        super().__init__(indexer, config)
        self.query_expansion = config.get("query_expansion", True)
        self.expansion_variants = config.get("expansion_variants", 3)
        self.fusion_method = config.get("fusion_method", "rrf")
    
    def _expand_query(self, query: str) -> List[str]:
        """Generate query variants.
        
        Args:
            query: Original query.
            
        Returns:
            List of query variants.
        """
        variants = [query]
        
        if not self.query_expansion:
            return variants
        
        # Simple expansion strategies
        # 1. Remove stop words
        stop_words = {"the", "a", "an", "is", "are", "was", "were", 
                      "be", "been", "being", "have", "has", "had",
                      "do", "does", "did", "will", "would", "could",
                      "should", "may", "might", "must", "shall",
                      "can", "need", "dare", "ought", "used", "to",
                      "of", "in", "for", "on", "with", "at", "by",
                      "from", "as", "into", "through", "during",
                      "before", "after", "above", "below", "between",
                      "under", "again", "further", "then", "once"}
        
        words = query.split()
        content_words = [w for w in words if w.lower() not in stop_words]
        if content_words and len(content_words) < len(words):
            variants.append(" ".join(content_words))
        
        # 2. Extract code-like terms (camelCase, snake_case)
        import re
        code_terms = re.findall(r'[a-zA-Z_][a-zA-Z0-9_]*', query)
        if code_terms and len(code_terms) >= 2:
            variants.append(" ".join(code_terms))
        
        # 3. Focus on action words
        action_words = [w for w in words if w.lower() in {
            "fix", "handle", "process", "parse", "convert", "transform",
            "validate", "check", "verify", "get", "set", "create",
            "build", "generate", "compute", "calculate", "update",
            "delete", "remove", "add", "insert", "find", "search"
        }]
        if action_words:
            variants.append(" ".join(action_words + code_terms[:3]))
        
        return variants[:self.expansion_variants]
    
    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[SearchResult]:
        """Retrieve with query expansion.
        
        Args:
            query: Search query string.
            top_k: Override default top_k if provided.
            
        Returns:
            Combined results from all query variants.
        """
        k = top_k or self.top_k
        
        # Expand query
        queries = self._expand_query(query)
        
        if len(queries) == 1:
            # No expansion, use standard retrieval
            return super().retrieve(query, k)
        
        # Retrieve for each variant
        all_results = []
        for q in queries:
            results = self.dense_indexer.search(q, k)
            all_results.append(results)
        
        # Fuse results
        if self.fusion_method == "rrf":
            return self._fuse_rrf(all_results, k)
        else:
            return self._fuse_interleave(all_results, k)
    
    def _fuse_rrf(
        self,
        results_list: List[List[SearchResult]],
        top_k: int,
        rrf_k: int = 60
    ) -> List[SearchResult]:
        """Fuse results using Reciprocal Rank Fusion.
        
        Args:
            results_list: List of result lists from different queries.
            top_k: Number of top results to return.
            rrf_k: RRF constant.
            
        Returns:
            Fused results.
        """
        from collections import defaultdict
        
        def get_key(r: SearchResult) -> str:
            return f"{r.chunk.file_path}:{r.chunk.start_line}:{r.chunk.end_line}"
        
        # Calculate RRF scores
        scores: dict[str, tuple[float, 'CodeChunk']] = defaultdict(lambda: (0.0, None))
        
        for results in results_list:
            for rank, result in enumerate(results, start=1):
                key = get_key(result)
                score = 1.0 / (rrf_k + rank)
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
                source="dense_multiquery"
            ))
        
        return final_results
    
    def _fuse_interleave(
        self,
        results_list: List[List[SearchResult]],
        top_k: int
    ) -> List[SearchResult]:
        """Fuse results by interleaving.
        
        Args:
            results_list: List of result lists.
            top_k: Number of top results to return.
            
        Returns:
            Interleaved results.
        """
        seen = set()
        final_results = []
        
        # Interleave results
        max_len = max(len(r) for r in results_list) if results_list else 0
        
        for i in range(max_len):
            for results in results_list:
                if i < len(results):
                    result = results[i]
                    key = f"{result.chunk.file_path}:{result.chunk.start_line}:{result.chunk.end_line}"
                    
                    if key not in seen:
                        seen.add(key)
                        final_results.append(result)
                        
                        if len(final_results) >= top_k:
                            break
            
            if len(final_results) >= top_k:
                break
        
        # Update ranks
        for rank, result in enumerate(final_results, start=1):
            result.rank = rank
            result.source = "dense_multiquery"
        
        return final_results