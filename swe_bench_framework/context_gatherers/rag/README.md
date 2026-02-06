# RAG (Retrieval-Augmented Generation) Module

This module provides RAG-based context gathering for the SWE-bench comparison framework.

## Overview

The RAG module implements retrieval-augmented generation for automated software patching.
It combines multiple retrieval strategies (BM25, Dense, Hybrid) with intelligent query
generation and context assembly.

## Architecture

```
RAG Pipeline:
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Query Generator│────▶│   Retriever     │────▶│   Re-ranker     │
│                 │     │  (BM25/Dense)   │     │ (Cross-Encoder) │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                                                          │
                              ┌───────────────────────────┘
                              ▼
                    ┌─────────────────┐
                    │ Context Assembly│
                    │ (Token Budget)  │
                    └─────────────────┘
```

## Components

### 1. Chunking (`chunking/`)

**Base Classes:**
- `CodeChunker`: Abstract base for all chunkers

**Implementations:**
- `ASTChunker`: Python AST-based chunking (functions, classes, methods)
- `SlidingWindowChunker`: Fixed-size sliding window chunking

### 2. Indexing (`indexer/`)

**Base Classes:**
- `RepositoryIndexer`: Abstract base for all indexers
- `Tokenizer`, `CodeTokenizer`: Text tokenization

**Implementations:**
- `BM25Indexer`: Sparse retrieval with BM25 algorithm
- `DenseIndexer`: Dense retrieval with FAISS and embeddings
- `HybridIndexer`: BM25 + Dense with Reciprocal Rank Fusion

### 3. Retrieval (`retriever/`)

**Base Classes:**
- `Retriever`: Abstract base for all retrievers
- `Reranker`: Abstract base for re-rankers

**Implementations:**
- `BM25Retriever`: BM25-based retrieval
- `DenseRetriever`: Dense embedding retrieval
- `HybridRetriever`: Multi-strategy fusion
- `CrossEncoderReranker`: Cross-encoder re-ranking

### 4. Query Generation (`query_generator.py`)

- `QueryGenerator`: Abstract base
- `MultiStrategyQueryGenerator`: Multiple query strategies
- `KeywordOnlyQueryGenerator`: Keyword extraction
- `SimpleQueryGenerator`: Pass-through

### 5. Main Gatherer (`rag_gatherer.py`)

- `RAGGatherer`: Main orchestrator class
- `create_rag_gatherer()`: Factory function

## Usage

### Basic Usage

```python
from swe_bench_framework.context_gatherers.rag import RAGGatherer

config = {
    "chunker": {"type": "ast", "include_imports": True},
    "indexer": {"type": "hybrid", "bm25_weight": 0.7},
    "retriever": {"top_k": 20},
    "query_generator": {"strategies": ["keywords", "code_symbols"]},
    "context_assembly": {"max_tokens": 8000},
}

gatherer = RAGGatherer(config)
gatherer.initialize(repo_path="/path/to/repo")

# Build index (one-time)
gatherer.build_index(repo_path, "/path/to/index")

# Or load existing index
gatherer.load_index("/path/to/index")

# Gather context
context = gatherer.gather_context(instance, repo_path)
```

### Pre-configured Gatherers

```python
from swe_bench_framework.context_gatherers.rag import create_rag_gatherer

# BM25-only
gatherer = create_rag_gatherer(retrieval_type="bm25")

# Dense with embeddings
gatherer = create_rag_gatherer(
    retrieval_type="dense",
    embedding_model="jinaai/jina-embeddings-v2-base-code"
)

# Hybrid with re-ranking
gatherer = create_rag_gatherer(
    retrieval_type="hybrid",
    use_reranker=True,
    max_tokens=8000
)
```

## Configuration Options

### Chunker Options

```python
{
    "type": "ast",  # or "sliding"
    "include_imports": True,
    "include_docstrings": True,
    "separate_class_methods": False,
    "min_function_lines": 2,
    # For sliding:
    "chunk_size": 1000,
    "chunk_overlap": 200,
}
```

### Indexer Options

**BM25:**
```python
{
    "type": "bm25",
    "k1": 1.5,
    "b": 0.75,
    "tokenizer": "code",
    "field_weights": {"name": 3.0, "content": 1.0}
}
```

**Dense:**
```python
{
    "type": "dense",
    "model": "all-MiniLM-L6-v2",
    "index_type": "IndexFlatIP",  # or "IndexIVFFlat", "IndexHNSWFlat"
    "normalize": True,
    "batch_size": 32,
}
```

**Hybrid:**
```python
{
    "type": "hybrid",
    "bm25_weight": 0.5,
    "dense_weight": 0.5,
    "rrf_k": 60,
    "fusion_method": "rrf",  # or "weighted_sum"
    "bm25": {...},
    "dense": {...},
}
```

### Retriever Options

```python
{
    "top_k": 20,
    "min_score": 0.0,
    "deduplicate": True,
    "initial_k": 60,  # For hybrid
}
```

### Re-ranker Options

```python
{
    "type": "cross_encoder",
    "model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "rerank_k": 50,
    "top_k": 20,
    "batch_size": 32,
}
```

## Query Generation Strategies

The `MultiStrategyQueryGenerator` supports:

1. **Keywords**: Extract important keywords from problem statement
2. **Code Symbols**: Extract function/class names, snake_case, camelCase
3. **Error Patterns**: Extract exception types and error messages
4. **Natural Language**: Reformulate as search-friendly queries
5. **Action Focus**: Focus on action words (fix, handle, add, etc.)

## Dependencies

Required packages:
```
rank-bm25          # For BM25 indexing
faiss-cpu          # For dense indexing (or faiss-gpu)
sentence-transformers  # For embeddings
numpy              # For numerical operations
```

Optional packages:
```
transformers       # For Jina embeddings
tiktoken           # For token counting
```

## File Structure

```
rag/
├── __init__.py              # Module exports
├── base.py                  # Base classes (CodeChunk, SearchResult, etc.)
├── query_generator.py       # Query generation strategies
├── rag_gatherer.py          # Main RAGGatherer implementation
├── README.md                # This file
├── chunking/
│   ├── __init__.py
│   ├── base.py              # CodeChunker base class
│   ├── ast_chunker.py       # AST-based chunking
│   └── sliding_chunker.py   # Sliding window chunking
├── indexer/
│   ├── __init__.py
│   ├── base.py              # RepositoryIndexer base class
│   ├── bm25_indexer.py      # BM25 indexing
│   ├── dense_indexer.py     # Dense/FAISS indexing
│   └── hybrid_indexer.py    # Hybrid indexing with RRF
└── retriever/
    ├── __init__.py
    ├── base.py              # Retriever base classes
    ├── sparse_retriever.py  # BM25 retrieval
    ├── dense_retriever.py   # Dense retrieval
    ├── hybrid_retriever.py  # Hybrid retrieval
    └── reranker.py          # Cross-encoder re-ranking
```

## Performance Considerations

1. **Index Building**: One-time cost per repository, reusable across instances
2. **BM25**: Fast retrieval, good for exact keyword matching
3. **Dense**: Slower retrieval, better semantic matching
4. **Hybrid**: Best of both, moderate speed
5. **Re-ranking**: Adds latency but improves precision

## References

- BM25: Robertson et al., "Okapi at TREC-3"
- Dense Retrieval: Karpukhin et al., "Dense Passage Retrieval"
- Reciprocal Rank Fusion: Cormack et al., "Reciprocal Rank Fusion"
- Cross-Encoders: Nogueira et al., "Multi-Stage Document Ranking"