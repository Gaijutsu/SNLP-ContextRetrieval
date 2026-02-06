"""Retrieval module for RAG.

This module provides different retrieval strategies and re-ranking:
- Sparse: BM25-based keyword retrieval
- Dense: Vector similarity retrieval
- Hybrid: Combination of multiple strategies
- Re-ranking: Cross-encoder based result re-ranking
"""

from .base import Retriever, Reranker
from .sparse_retriever import BM25Retriever, MultiFieldBM25Retriever
from .dense_retriever import DenseRetriever, MultiQueryDenseRetriever
from .hybrid_retriever import HybridRetriever, HybridIndexerRetriever
from .reranker import CrossEncoderReranker, LLMReranker, CascadeReranker

__all__ = [
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
]