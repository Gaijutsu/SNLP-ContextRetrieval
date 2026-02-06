"""Main RAG Gatherer implementation for the SWE-bench framework.

This module provides the main RAGGatherer class that orchestrates the entire
RAG pipeline: chunking → indexing → retrieval → context assembly.
"""

import os
import time
import logging
from typing import List, Dict, Optional, Any, Type
from datetime import datetime
from pathlib import Path

from .base import (
    BaseRAGGatherer,
    CodeChunk,
    SearchResult,
    RAGContextBundle,
)
from .chunking.base import CodeChunker
from .chunking.ast_chunker import ASTChunker
from .chunking.sliding_chunker import SlidingWindowChunker
from .indexer.base import RepositoryIndexer
from .indexer.bm25_indexer import BM25Indexer
from .indexer.dense_indexer import DenseIndexer
from .indexer.hybrid_indexer import HybridIndexer
from .retriever.base import Retriever, Reranker
from .retriever.sparse_retriever import BM25Retriever
from .retriever.dense_retriever import DenseRetriever
from .retriever.hybrid_retriever import HybridRetriever, HybridIndexerRetriever
from .retriever.reranker import CrossEncoderReranker
from .query_generator import (
    QueryGenerator,
    MultiStrategyQueryGenerator,
    GeneratedQuery,
)

# Import framework base classes
from ..base import ContextChunk, ContextBundle, ContextType, SWEInstance

logger = logging.getLogger(__name__)


class RAGGatherer(BaseRAGGatherer):
    """Main RAG-based context gatherer for SWE-bench.
    
    This class orchestrates the complete RAG pipeline:
    1. **Chunking**: Split repository code into semantic chunks
    2. **Indexing**: Build searchable indexes (BM25, Dense, or Hybrid)
    3. **Query Generation**: Convert problem statements to search queries
    4. **Retrieval**: Search indexes for relevant code
    5. **Re-ranking**: Optionally re-rank results for better precision
    6. **Context Assembly**: Build ContextBundle within token budget
    
    The gatherer supports multiple retrieval strategies and can be configured
    for different trade-offs between speed and accuracy.
    
    Example:
        ```python
        config = {
            "chunker": {"type": "ast", "include_imports": True},
            "indexer": {"type": "hybrid", "bm25_weight": 0.7},
            "retriever": {"top_k": 20, "use_reranker": True},
            "query_generator": {"strategies": ["keywords", "code_symbols"]},
            "context_assembly": {"max_tokens": 8000},
        }
        
        gatherer = RAGGatherer(config)
        gatherer.initialize(repo_path="/path/to/repo")
        
        # Build index if needed
        if not gatherer.is_index_built(repo_path):
            gatherer.build_index(repo_path, index_path)
        else:
            gatherer.load_index(index_path)
        
        # Gather context
        context = gatherer.gather_context(instance, repo_path)
        ```
    """
    
    # Supported chunker types
    CHUNKERS = {
        "ast": ASTChunker,
        "sliding": SlidingWindowChunker,
    }
    
    # Supported indexer types
    INDEXERS = {
        "bm25": BM25Indexer,
        "dense": DenseIndexer,
        "hybrid": HybridIndexer,
    }
    
    # Supported retriever types
    RETRIEVERS = {
        "bm25": BM25Retriever,
        "dense": DenseRetriever,
        "hybrid": HybridIndexerRetriever,
    }
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the RAG gatherer.
        
        Args:
            config: Configuration dictionary with sections:
                - chunker: Chunking configuration (type, options)
                - indexer: Indexing configuration (type, options)
                - retriever: Retrieval configuration (type, options)
                - reranker: Re-ranking configuration (optional)
                - query_generator: Query generation configuration
                - context_assembly: Context assembly configuration
        """
        super().__init__(config)
        
        # Extract configuration sections
        self.chunker_config = config.get("chunker", {"type": "ast"})
        self.indexer_config = config.get("indexer", {"type": "hybrid"})
        self.retriever_config = config.get("retriever", {"top_k": 20})
        self.reranker_config = config.get("reranker", None)
        self.query_gen_config = config.get("query_generator", {})
        self.context_config = config.get("context_assembly", {"max_tokens": 8000})
        
        # Initialize components (lazy loading)
        self._chunker: Optional[CodeChunker] = None
        self._indexer: Optional[RepositoryIndexer] = None
        self._retriever: Optional[Retriever] = None
        self._reranker: Optional[Reranker] = None
        self._query_generator: Optional[QueryGenerator] = None
        
        # Statistics
        self._gather_stats: Dict[str, Any] = {
            "instances_processed": 0,
            "total_chunks_retrieved": 0,
            "total_tokens_in_context": 0,
            "avg_retrieval_time": 0.0,
        }
    
    def _get_chunker(self) -> CodeChunker:
        """Get or create the chunker."""
        if self._chunker is None:
            chunker_type = self.chunker_config.get("type", "ast")
            chunker_class = self.CHUNKERS.get(chunker_type, ASTChunker)
            self._chunker = chunker_class(self.chunker_config)
            self._chunker.initialize()
        return self._chunker
    
    def _get_indexer(self) -> RepositoryIndexer:
        """Get or create the indexer."""
        if self._indexer is None:
            indexer_type = self.indexer_config.get("type", "hybrid")
            indexer_class = self.INDEXERS.get(indexer_type, HybridIndexer)
            self._indexer = indexer_class(self.indexer_config)
            self._indexer.initialize()
        return self._indexer
    
    def _get_retriever(self) -> Retriever:
        """Get or create the retriever."""
        if self._retriever is None:
            indexer = self._get_indexer()
            
            # Check if using hybrid indexer directly
            if isinstance(indexer, HybridIndexer):
                self._retriever = HybridIndexerRetriever(indexer, self.retriever_config)
            else:
                retriever_type = self.retriever_config.get("type")
                if retriever_type == "bm25" and isinstance(indexer, BM25Indexer):
                    self._retriever = BM25Retriever(indexer, self.retriever_config)
                elif retriever_type == "dense" and isinstance(indexer, DenseIndexer):
                    self._retriever = DenseRetriever(indexer, self.retriever_config)
                else:
                    # Auto-select based on indexer type
                    if isinstance(indexer, BM25Indexer):
                        self._retriever = BM25Retriever(indexer, self.retriever_config)
                    elif isinstance(indexer, DenseIndexer):
                        self._retriever = DenseRetriever(indexer, self.retriever_config)
                    else:
                        raise ValueError(f"Cannot create retriever for indexer type: {type(indexer)}")
            
            self._retriever.initialize()
        return self._retriever
    
    def _get_reranker(self) -> Optional[Reranker]:
        """Get or create the reranker."""
        if self._reranker is None and self.reranker_config:
            reranker_type = self.reranker_config.get("type", "cross_encoder")
            
            if reranker_type == "cross_encoder":
                model_name = self.reranker_config.get(
                    "model",
                    "cross-encoder/ms-marco-MiniLM-L-6-v2"
                )
                self._reranker = CrossEncoderReranker(model_name, self.reranker_config)
            else:
                logger.warning(f"Unknown reranker type: {reranker_type}")
        
        return self._reranker
    
    def _get_query_generator(self) -> QueryGenerator:
        """Get or create the query generator."""
        if self._query_generator is None:
            generator_type = self.query_gen_config.get("type", "multi_strategy")
            
            if generator_type == "multi_strategy":
                self._query_generator = MultiStrategyQueryGenerator(self.query_gen_config)
            elif generator_type == "simple":
                from .query_generator import SimpleQueryGenerator
                self._query_generator = SimpleQueryGenerator()
            elif generator_type == "keywords":
                from .query_generator import KeywordOnlyQueryGenerator
                self._query_generator = KeywordOnlyQueryGenerator(self.query_gen_config)
            else:
                self._query_generator = MultiStrategyQueryGenerator(self.query_gen_config)
        
        return self._query_generator
    
    def initialize(self, repo_path: str) -> None:
        """Initialize the gatherer for a repository.
        
        Args:
            repo_path: Path to the repository.
        """
        self._repo_path = repo_path
        
        # Initialize components
        _ = self._get_chunker()
        _ = self._get_indexer()
        _ = self._get_query_generator()
        
        self._initialized = True
        logger.info(f"Initialized RAGGatherer for {repo_path}")
    
    def gather_context(
        self,
        instance: SWEInstance,
        repo_path: str
    ) -> ContextBundle:
        """Gather context for a SWE-bench instance using RAG.
        
        This method orchestrates the full RAG pipeline:
        1. Generate search queries from problem statement
        2. Retrieve relevant code chunks
        3. Optionally re-rank results
        4. Assemble context bundle with token budgeting
        
        Args:
            instance: The SWE-bench instance.
            repo_path: Path to the repository checkout.
            
        Returns:
            ContextBundle containing gathered context.
        """
        start_time = time.time()
        
        if not self._initialized:
            self.initialize(repo_path)
        
        # Ensure index is loaded
        if not self._get_indexer().is_index_loaded():
            raise RuntimeError("Index not loaded. Call load_index() or build_index() first.")
        
        logger.info(f"Gathering context for instance: {instance.instance_id}")
        
        # Step 1: Generate queries
        queries = self._generate_queries(instance.problem_statement)
        logger.info(f"Generated {len(queries)} queries")
        
        # Step 2: Retrieve for each query
        all_results = self._retrieve_for_queries(queries)
        logger.info(f"Retrieved {len(all_results)} total results")
        
        # Step 3: Re-rank if configured
        reranker = self._get_reranker()
        if reranker:
            all_results = self._rerank_results(all_results, queries[0].query, reranker)
            logger.info(f"Re-ranked to {len(all_results)} results")
        
        # Step 4: Assemble context bundle
        context_bundle = self._assemble_context(
            all_results,
            instance,
            repo_path
        )
        
        # Update statistics
        duration = time.time() - start_time
        self._update_gather_stats(len(all_results), context_bundle.token_count, duration)
        
        logger.info(
            f"Context gathering complete: {len(context_bundle.chunks)} chunks, "
            f"{context_bundle.token_count} tokens, {duration:.2f}s"
        )
        
        return context_bundle
    
    def _generate_queries(self, problem_statement: str) -> List[GeneratedQuery]:
        """Generate search queries from problem statement."""
        generator = self._get_query_generator()
        return generator.generate(problem_statement)
    
    def _retrieve_for_queries(
        self,
        queries: List[GeneratedQuery]
    ) -> List[SearchResult]:
        """Retrieve results for all queries and merge."""
        retriever = self._get_retriever()
        
        # Collect results from all queries
        all_results: List[SearchResult] = []
        seen_chunks: set = set()
        
        for query in queries:
            results = retriever.retrieve(query.query, top_k=self.retriever_config.get("top_k", 20))
            
            # Apply query weight to scores
            for result in results:
                result.score *= query.weight
                
                # Deduplicate
                chunk_key = f"{result.chunk.file_path}:{result.chunk.start_line}:{result.chunk.end_line}"
                if chunk_key not in seen_chunks:
                    seen_chunks.add(chunk_key)
                    all_results.append(result)
        
        # Sort by score
        all_results.sort(key=lambda r: r.score, reverse=True)
        
        return all_results
    
    def _rerank_results(
        self,
        results: List[SearchResult],
        query: str,
        reranker: Reranker
    ) -> List[SearchResult]:
        """Re-rank results using the configured reranker."""
        rerank_k = self.reranker_config.get("rerank_k", 50)
        top_k = self.reranker_config.get("top_k", 20)
        
        # Limit to rerank_k for efficiency
        results_to_rerank = results[:rerank_k]
        
        # Re-rank
        reranked = reranker.rerank(results_to_rerank, query, top_k=top_k)
        
        return reranked
    
    def _assemble_context(
        self,
        results: List[SearchResult],
        instance: SWEInstance,
        repo_path: str
    ) -> ContextBundle:
        """Assemble context bundle from search results."""
        max_tokens = self.context_config.get("max_tokens", 8000)
        max_chunks = self.context_config.get("max_chunks", 50)
        
        # Convert search results to context chunks
        context_chunks = []
        for result in results[:max_chunks]:
            context_chunk = result.to_context_chunk()
            context_chunks.append(context_chunk)
        
        # Build repository structure info
        repo_structure = self._build_repo_structure(repo_path)
        
        # Calculate token count (approximate)
        token_count = self._estimate_tokens(context_chunks)
        
        # Trim to fit token budget if needed
        if token_count > max_tokens:
            context_chunks, token_count = self._trim_to_budget(
                context_chunks,
                max_tokens
            )
        
        # Create context bundle
        bundle = ContextBundle(
            instance_id=instance.instance_id,
            problem_statement=instance.problem_statement,
            chunks=context_chunks,
            repo_structure=repo_structure,
            gathered_at=datetime.utcnow().isoformat(),
            gatherer_type="RAGGatherer",
            token_count=token_count,
            metadata={
                "num_results": len(results),
                "num_chunks": len(context_chunks),
                "retrieval_config": self.retriever_config,
            }
        )
        
        return bundle
    
    def _build_repo_structure(self, repo_path: str) -> Dict[str, Any]:
        """Build repository structure information."""
        structure = {
            "root": repo_path,
            "files": [],
            "directories": [],
        }
        
        try:
            for root, dirs, files in os.walk(repo_path):
                # Skip common non-source directories
                dirs[:] = [
                    d for d in dirs
                    if d not in ['.git', '__pycache__', 'node_modules', '.venv', 'venv']
                ]
                
                rel_root = os.path.relpath(root, repo_path)
                if rel_root != '.':
                    structure["directories"].append(rel_root)
                
                for file in files:
                    if file.endswith('.py'):
                        rel_path = os.path.join(rel_root, file) if rel_root != '.' else file
                        structure["files"].append(rel_path)
        except Exception as e:
            logger.warning(f"Error building repo structure: {e}")
        
        return structure
    
    def _estimate_tokens(self, chunks: List[ContextChunk]) -> int:
        """Estimate token count for context chunks.
        
        Uses a simple approximation: ~4 characters per token on average.
        """
        total_chars = sum(len(c.content) for c in chunks)
        return total_chars // 4
    
    def _trim_to_budget(
        self,
        chunks: List[ContextChunk],
        max_tokens: int
    ) -> tuple[List[ContextChunk], int]:
        """Trim chunks to fit within token budget.
        
        Prioritizes chunks by relevance score and context type.
        """
        # Sort by priority: relevance score, then context type
        type_priority = {
            ContextType.ERROR_CONTEXT: 0,
            ContextType.FUNCTION_DEFINITION: 1,
            ContextType.CLASS_DEFINITION: 2,
            ContextType.TEST_CONTEXT: 3,
            ContextType.FILE_CONTENT: 4,
            ContextType.IMPORT_DEPENDENCY: 5,
            ContextType.REPO_STRUCTURE: 6,
        }
        
        chunks.sort(key=lambda c: (
            type_priority.get(c.context_type, 99),
            -c.relevance_score
        ))
        
        # Select chunks until budget is exhausted
        selected = []
        current_tokens = 0
        
        for chunk in chunks:
            chunk_tokens = len(chunk.content) // 4
            
            if current_tokens + chunk_tokens <= max_tokens:
                selected.append(chunk)
                current_tokens += chunk_tokens
            else:
                # Try to truncate if it's a file content chunk
                if chunk.context_type == ContextType.FILE_CONTENT:
                    remaining = max_tokens - current_tokens
                    if remaining > 200:  # Minimum chunk size
                        truncated_content = chunk.content[:remaining * 4]
                        chunk.content = truncated_content + "\n... (truncated)"
                        selected.append(chunk)
                        current_tokens += remaining
                
                break
        
        return selected, current_tokens
    
    def _update_gather_stats(
        self,
        num_results: int,
        token_count: int,
        duration: float
    ) -> None:
        """Update gathering statistics."""
        self._gather_stats["instances_processed"] += 1
        self._gather_stats["total_chunks_retrieved"] += num_results
        self._gather_stats["total_tokens_in_context"] += token_count
        
        # Update average
        n = self._gather_stats["instances_processed"]
        old_avg = self._gather_stats["avg_retrieval_time"]
        self._gather_stats["avg_retrieval_time"] = (
            (old_avg * (n - 1) + duration) / n
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get gathering statistics."""
        stats = super().get_stats()
        stats.update(self._gather_stats)
        
        # Add component stats
        if self._chunker:
            stats["chunker"] = self._chunker.get_stats()
        if self._indexer:
            stats["indexer"] = self._indexer.get_stats()
        if self._retriever:
            stats["retriever"] = self._retriever.get_stats()
        
        return stats
    
    def cleanup(self) -> None:
        """Cleanup resources."""
        if self._chunker:
            self._chunker.cleanup()
        if self._indexer:
            self._indexer.cleanup()
        if self._retriever:
            self._retriever.cleanup()
        
        self._initialized = False
        logger.info("RAGGatherer cleaned up")


def create_rag_gatherer(
    retrieval_type: str = "hybrid",
    embedding_model: Optional[str] = None,
    use_reranker: bool = True,
    max_tokens: int = 8000
) -> RAGGatherer:
    """Factory function to create a pre-configured RAG gatherer.
    
    Args:
        retrieval_type: Type of retrieval - "bm25", "dense", or "hybrid"
        embedding_model: Embedding model for dense retrieval
        use_reranker: Whether to use cross-encoder re-ranking
        max_tokens: Maximum tokens in context
        
    Returns:
        Configured RAGGatherer instance.
    """
    config = {
        "chunker": {"type": "ast", "include_imports": True},
        "indexer": {"type": retrieval_type},
        "retriever": {"top_k": 20},
        "query_generator": {"strategies": ["keywords", "code_symbols", "error_patterns"]},
        "context_assembly": {"max_tokens": max_tokens},
    }
    
    # Configure embedding model if specified
    if embedding_model and retrieval_type in ("dense", "hybrid"):
        config["indexer"]["model"] = embedding_model
    
    # Configure reranker
    if use_reranker:
        config["reranker"] = {
            "type": "cross_encoder",
            "model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
            "rerank_k": 50,
            "top_k": 20,
        }
    
    return RAGGatherer(config)