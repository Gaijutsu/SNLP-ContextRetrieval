"""RAG (Retrieval-Augmented Generation) module for SWE-bench framework.

This module provides RAG-based context gathering for automated software patching.
It includes components for:

- **Chunking**: Splitting code into semantic units (AST-based, sliding window)
- **Indexing**: Building searchable indexes (BM25, Dense, Hybrid)
- **Retrieval**: Searching indexes for relevant code
- **Re-ranking**: Improving result precision with cross-encoders
- **Query Generation**: Converting issue descriptions to search queries

Example usage:
    ```python
    from swe_bench_framework.context_gatherers.rag import RAGGatherer
    
    config = {
        "chunker": {"type": "ast", "include_imports": True},
        "indexer": {"type": "hybrid", "bm25_weight": 0.7},
        "retriever": {"top_k": 20, "use_reranker": True},
    }
    
    gatherer = RAGGatherer(config)
    gatherer.initialize(repo_path="/path/to/repo")
    gatherer.build_index(repo_path, index_path)
    
    context = gatherer.gather_context(instance, repo_path)
    ```
"""

# Base classes
from .base import (
    CodeChunk,
    SearchResult,
    RAGContextBundle,
    BaseRAGComponent,
    BaseRAGGatherer,
)

# Chunking
from .chunking.base import CodeChunker
from .chunking.ast_chunker import ASTChunker
from .chunking.sliding_chunker import SlidingWindowChunker

# Indexing
from .indexer.base import RepositoryIndexer, Tokenizer, CodeTokenizer, SimpleTokenizer
from .indexer.bm25_indexer import BM25Indexer
from .indexer.dense_indexer import DenseIndexer, EmbeddingModel
from .indexer.hybrid_indexer import HybridIndexer

# Retrieval
from .retriever.base import Retriever, Reranker
from .retriever.sparse_retriever import BM25Retriever, MultiFieldBM25Retriever
from .retriever.dense_retriever import DenseRetriever, MultiQueryDenseRetriever
from .retriever.hybrid_retriever import HybridRetriever, HybridIndexerRetriever
from .retriever.reranker import (
    CrossEncoderReranker,
    LLMReranker,
    CascadeReranker,
)

# Query Generation
from .query_generator import (
    QueryGenerator,
    MultiStrategyQueryGenerator,
    SimpleQueryGenerator,
    KeywordOnlyQueryGenerator,
    GeneratedQuery,
)

# Main Gatherer
from .rag_gatherer import RAGGatherer, create_rag_gatherer

__all__ = [
    # Base classes
    "CodeChunk",
    "SearchResult",
    "RAGContextBundle",
    "BaseRAGComponent",
    "BaseRAGGatherer",
    
    # Chunking
    "CodeChunker",
    "ASTChunker",
    "SlidingWindowChunker",
    
    # Indexing
    "RepositoryIndexer",
    "Tokenizer",
    "CodeTokenizer",
    "SimpleTokenizer",
    "BM25Indexer",
    "DenseIndexer",
    "EmbeddingModel",
    "HybridIndexer",
    
    # Retrieval
    "Retriever",
    "Reranker",
    "BM25Retriever",
    "MultiFieldBM25Retriever",
    "DenseRetriever",
    "MultiQueryDenseRetriever",
    "HybridRetriever",
    "HybridIndexerRetriever",
    "CrossEncoderReranker",
    "LLMReranker",
    "CascadeReranker",
    
    # Query Generation
    "QueryGenerator",
    "MultiStrategyQueryGenerator",
    "SimpleQueryGenerator",
    "KeywordOnlyQueryGenerator",
    "GeneratedQuery",
    
    # Main Gatherer
    "RAGGatherer",
    "create_rag_gatherer",
]

__version__ = "0.1.0"


def register_with_factory():
    """Register RAG components with the framework's gatherer factory."""
    try:
        from .. import register_gatherer
        register_gatherer("rag", RAGGatherer)
        register_gatherer("rag_bm25", lambda config: RAGGatherer({
            **config,
            "indexer": {"type": "bm25"},
            "retriever": {"type": "bm25"},
        }))
        register_gatherer("rag_dense", lambda config: RAGGatherer({
            **config,
            "indexer": {"type": "dense", "model": "all-MiniLM-L6-v2"},
            "retriever": {"type": "dense"},
        }))
        register_gatherer("rag_hybrid", lambda config: RAGGatherer({
            **config,
            "indexer": {"type": "hybrid"},
            "retriever": {"type": "hybrid"},
        }))
    except ImportError:
        # Factory not available, skip registration
        pass


# Auto-register on import
register_with_factory()