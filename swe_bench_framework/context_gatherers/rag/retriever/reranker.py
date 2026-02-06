"""Cross-encoder re-ranking implementation.

This module provides cross-encoder based re-ranking for improving
the precision of retrieval results.
"""

from typing import List, Optional, Any
import logging
import numpy as np

from .base import Reranker
from ..base import SearchResult

logger = logging.getLogger(__name__)


class CrossEncoderReranker(Reranker):
    """Cross-encoder based re-ranker.
    
    Cross-encoders process query and document together for more accurate
    relevance scoring than bi-encoders. They are typically used to re-rank
    the top-k results from initial retrieval.
    
    Popular models:
    - cross-encoder/ms-marco-MiniLM-L-6-v2 (fast, good accuracy)
    - cross-encoder/ms-marco-MiniLM-L-12-v2 (slower, better accuracy)
    - BAAI/bge-reranker-base (good for code)
    
    Usage pattern:
    1. Use BM25/Dense for fast initial retrieval (top 50-100)
    2. Apply cross-encoder for precise re-ranking (top 5-10)
    
    Example:
        ```python
        reranker = CrossEncoderReranker(
            model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
            config={"batch_size": 32}
        )
        
        # Initial retrieval
        initial_results = retriever.retrieve("error handling", top_k=50)
        
        # Re-rank
        final_results = reranker.rerank(initial_results, query="error handling", top_k=10)
        ```
    """
    
    # Recommended models
    RECOMMENDED_MODELS = {
        "ms-marco-MiniLM-L-6-v2": "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "ms-marco-MiniLM-L-12-v2": "cross-encoder/ms-marco-MiniLM-L-12-v2",
        "ms-marco-electra-base": "cross-encoder/ms-marco-electra-base",
        "bge-reranker-base": "BAAI/bge-reranker-base",
        "bge-reranker-large": "BAAI/bge-reranker-large",
    }
    
    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        config: Optional[dict] = None
    ):
        """Initialize the cross-encoder reranker.
        
        Args:
            model_name: Name of the cross-encoder model.
            config: Configuration dictionary with options:
                - batch_size: Batch size for inference (default: 32)
                - max_length: Maximum sequence length (default: 512)
                - device: Device for inference (default: "cpu")
                - normalize_scores: Normalize scores to [0, 1] (default: True)
        """
        self.config = config or {}
        self.model_name = model_name
        self.batch_size = self.config.get("batch_size", 32)
        self.max_length = self.config.get("max_length", 512)
        self.device = self.config.get("device", "cpu")
        self.normalize_scores = self.config.get("normalize_scores", True)
        
        # Lazy loading of model
        self._model = None
        self._tokenizer = None
    
    def _load_model(self):
        """Lazy load the cross-encoder model."""
        if self._model is not None:
            return
        
        try:
            from sentence_transformers import CrossEncoder
        except ImportError:
            raise ImportError(
                "sentence-transformers is required for CrossEncoderReranker. "
                "Install with: pip install sentence-transformers"
            )
        
        logger.info(f"Loading cross-encoder model: {self.model_name}")
        self._model = CrossEncoder(
            self.model_name,
            max_length=self.max_length,
            device=self.device
        )
    
    def rerank(
        self,
        results: List[SearchResult],
        query: Optional[str] = None,
        top_k: Optional[int] = None
    ) -> List[SearchResult]:
        """Re-rank search results using cross-encoder.
        
        Args:
            results: Initial search results to re-rank.
            query: Query string (required for re-ranking).
            top_k: Number of top results to return.
            
        Returns:
            Re-ranked search results.
        """
        if not results:
            return []
        
        if query is None:
            logger.warning("No query provided for re-ranking, returning original results")
            return results
        
        # Load model if needed
        self._load_model()
        
        # Prepare pairs for cross-encoder
        pairs = []
        for result in results:
            # Create text pair (query, document)
            doc_text = self._format_chunk_for_reranking(result)
            pairs.append((query, doc_text))
        
        # Score pairs in batches
        all_scores = []
        for i in range(0, len(pairs), self.batch_size):
            batch_pairs = pairs[i:i + self.batch_size]
            batch_scores = self._model.predict(batch_pairs)
            all_scores.extend(batch_scores)
        
        # Normalize scores if configured
        if self.normalize_scores and all_scores:
            min_score = min(all_scores)
            max_score = max(all_scores)
            score_range = max_score - min_score if max_score > min_score else 1.0
            all_scores = [(s - min_score) / score_range for s in all_scores]
        
        # Update results with new scores
        for result, score in zip(results, all_scores):
            result.score = float(score)
        
        # Re-sort by new scores
        results.sort(key=lambda r: r.score, reverse=True)
        
        # Update ranks
        for rank, result in enumerate(results, start=1):
            result.rank = rank
            result.source = f"reranked_{result.source}"
        
        # Return top_k
        if top_k:
            results = results[:top_k]
        
        return results
    
    def rerank_batch(
        self,
        results_list: List[List[SearchResult]],
        queries: List[str]
    ) -> List[List[SearchResult]]:
        """Re-rank multiple result sets.
        
        Args:
            results_list: List of search result lists.
            queries: Corresponding queries for each result list.
            
        Returns:
            Re-ranked results for each query.
        """
        if len(results_list) != len(queries):
            raise ValueError("Number of result lists must match number of queries")
        
        return [
            self.rerank(results, query)
            for results, query in zip(results_list, queries)
        ]
    
    def _format_chunk_for_reranking(self, result: SearchResult) -> str:
        """Format a code chunk for cross-encoder input.
        
        Args:
            result: Search result containing the chunk.
            
        Returns:
            Formatted text for the cross-encoder.
        """
        chunk = result.chunk
        parts = []
        
        # Add file path and name
        if chunk.name:
            parts.append(f"File: {chunk.file_path}, Name: {chunk.name}")
        else:
            parts.append(f"File: {chunk.file_path}")
        
        # Add chunk type
        if chunk.chunk_type:
            parts.append(f"Type: {chunk.chunk_type}")
        
        # Add imports if available
        if chunk.imports:
            parts.append(f"Imports: {', '.join(chunk.imports[:5])}")
        
        # Add content (truncated if needed)
        content = chunk.content
        max_content_len = self.max_length - 100  # Reserve space for metadata
        if len(content) > max_content_len:
            content = content[:max_content_len] + "..."
        parts.append(f"Content:\n{content}")
        
        return "\n".join(parts)


class LLMReranker(Reranker):
    """LLM-based re-ranker for high-precision ranking.
    
    Uses an LLM to score the relevance of each result. More expensive
    than cross-encoders but can provide better understanding of
    complex queries.
    
    Example:
        ```python
        reranker = LLMReranker(
            llm_config={"model": "gpt-5-mini", "temperature": 0.0},
            config={"max_results": 10}
        )
        final_results = reranker.rerank(initial_results, query="error handling")
        ```
    """
    
    def __init__(
        self,
        llm_config: dict,
        config: Optional[dict] = None
    ):
        """Initialize the LLM reranker.
        
        Args:
            llm_config: Configuration for the LLM.
            config: Configuration dictionary with options:
                - max_results: Maximum results to re-rank (default: 10)
                - prompt_template: Custom prompt template
        """
        self.llm_config = llm_config
        self.config = config or {}
        self.max_results = self.config.get("max_results", 10)
        self.prompt_template = self.config.get("prompt_template", self._default_prompt())
        
        self._llm_client = None
    
    def _default_prompt(self) -> str:
        """Get the default ranking prompt."""
        return """Rate the relevance of the following code snippet to the query.

Query: {query}

Code:
```
{code}
```

Rate from 0 to 10 where:
- 0: Completely irrelevant
- 5: Somewhat relevant
- 10: Highly relevant, directly addresses the query

Respond with only a number from 0 to 10."""
    
    def _load_llm(self):
        """Lazy load the LLM client."""
        if self._llm_client is not None:
            return
        
        # This is a placeholder - actual implementation would depend on
        # the LLM provider (OpenAI, Anthropic, etc.)
        logger.warning("LLM reranker requires LLM client implementation")
        self._llm_client = None
    
    def rerank(
        self,
        results: List[SearchResult],
        query: Optional[str] = None,
        top_k: Optional[int] = None
    ) -> List[SearchResult]:
        """Re-rank using LLM scoring.
        
        Args:
            results: Initial search results.
            query: Query string.
            top_k: Number of top results to return.
            
        Returns:
            Re-ranked results.
        """
        if not results or not query:
            return results
        
        self._load_llm()
        
        if self._llm_client is None:
            logger.warning("LLM client not available, returning original results")
            return results
        
        # Limit to max_results for efficiency
        results = results[:self.max_results]
        
        # Score each result
        for result in results:
            score = self._score_result(result, query)
            result.score = score / 10.0  # Normalize to [0, 1]
        
        # Re-sort
        results.sort(key=lambda r: r.score, reverse=True)
        
        # Update ranks
        for rank, result in enumerate(results, start=1):
            result.rank = rank
            result.source = f"llm_reranked_{result.source}"
        
        if top_k:
            results = results[:top_k]
        
        return results
    
    def _score_result(self, result: SearchResult, query: str) -> float:
        """Score a single result using LLM.
        
        Args:
            result: Search result to score.
            query: Query string.
            
        Returns:
            Score from 0 to 10.
        """
        # Placeholder implementation
        # Actual implementation would call LLM API
        prompt = self.prompt_template.format(
            query=query,
            code=result.chunk.content[:1000]
        )
        
        # This is a placeholder - actual implementation would:
        # response = self._llm_client.complete(prompt)
        # score = float(response.strip())
        
        logger.debug(f"LLM scoring prompt: {prompt[:100]}...")
        return 5.0  # Placeholder score
    
    def rerank_batch(
        self,
        results_list: List[List[SearchResult]],
        queries: List[str]
    ) -> List[List[SearchResult]]:
        """Re-rank multiple result sets."""
        return [
            self.rerank(results, query)
            for results, query in zip(results_list, queries)
        ]


class CascadeReranker(Reranker):
    """Cascade re-ranker that applies multiple re-rankers in sequence.
    
    Applies re-rankers from fastest to slowest, progressively
    narrowing down the candidate set.
    
    Example:
        ```python
        cascade = CascadeReranker([
            (CrossEncoderReranker("ms-marco-MiniLM-L-6-v2"), 50, 20),
            (CrossEncoderReranker("bge-reranker-large"), 20, 5),
        ])
        final_results = cascade.rerank(initial_results, query="error handling")
        ```
    """
    
    def __init__(self, stages: List[tuple[Reranker, int, int]]):
        """Initialize the cascade reranker.
        
        Args:
            stages: List of (reranker, input_k, output_k) tuples.
        """
        self.stages = stages
    
    def rerank(
        self,
        results: List[SearchResult],
        query: Optional[str] = None,
        top_k: Optional[int] = None
    ) -> List[SearchResult]:
        """Apply cascade re-ranking.
        
        Args:
            results: Initial search results.
            query: Query string.
            top_k: Final number of results (overrides last stage).
            
        Returns:
            Re-ranked results.
        """
        current_results = results
        
        for reranker, input_k, output_k in self.stages:
            # Limit to input_k
            current_results = current_results[:input_k]
            
            # Apply reranker
            current_results = reranker.rerank(current_results, query, output_k)
        
        # Apply final top_k if specified
        if top_k:
            current_results = current_results[:top_k]
        
        return current_results
    
    def rerank_batch(
        self,
        results_list: List[List[SearchResult]],
        queries: List[str]
    ) -> List[List[SearchResult]]:
        """Re-rank multiple result sets."""
        return [
            self.rerank(results, query)
            for results, query in zip(results_list, queries)
        ]