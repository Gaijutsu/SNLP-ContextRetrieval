"""BM25 indexing implementation for code retrieval.

This module provides BM25-based sparse retrieval using the rank_bm25 library.
"""

from typing import List, Dict, Optional, Any
from pathlib import Path
import pickle
import logging

try:
    from rank_bm25 import BM25Okapi
    HAS_BM25 = True
except ImportError:
    HAS_BM25 = False
    BM25Okapi = None

from .base import RepositoryIndexer, CodeTokenizer
from ..base import CodeChunk, SearchResult

logger = logging.getLogger(__name__)


class BM25Indexer(RepositoryIndexer):
    """BM25-based sparse retrieval indexer.
    
    BM25 (Best Matching 25) is a probabilistic retrieval function that ranks
    documents based on query term frequency, inverse document frequency, and
    document length normalization.
    
    Formula: Score = Σ(IDF × adjusted_TF)
    
    Parameters:
    - k1: Controls term frequency saturation (higher = more occurrences matter)
    - b: Controls length normalization (0-1, higher = more penalty)
    
    Example:
        ```python
        indexer = BM25Indexer(config={"k1": 1.5, "b": 0.75})
        indexer.build_index(chunks, "/path/to/index")
        results = indexer.search("error handling function", top_k=10)
        ```
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the BM25 indexer.
        
        Args:
            config: Configuration dictionary with options:
                - k1: BM25 k1 parameter (default: 1.5)
                - b: BM25 b parameter (default: 0.75)
                - epsilon: BM25 epsilon parameter (default: 0.25)
                - tokenizer: Tokenizer type - "code" or "simple" (default: "code")
                - field_weights: Weights for different fields (content, name, imports)
        """
        super().__init__(config)
        
        if not HAS_BM25:
            raise ImportError(
                "rank_bm25 is required for BM25Indexer. "
                "Install with: pip install rank-bm25"
            )
        
        self.k1 = config.get("k1", 1.5)
        self.b = config.get("b", 0.75)
        self.epsilon = config.get("epsilon", 0.25)
        
        # Initialize tokenizer
        tokenizer_type = config.get("tokenizer", "code")
        if tokenizer_type == "code":
            self.tokenizer = CodeTokenizer(lowercase=True, split_camel_case=True)
        else:
            self.tokenizer = CodeTokenizer(lowercase=True, split_camel_case=False)
        
        # Field weights for BM25F-style weighting
        self.field_weights = config.get("field_weights", {
            "content": 1.0,
            "name": 3.0,  # Function/class names are more important
            "imports": 0.5,
        })
        
        self._tokenized_chunks: List[List[str]] = []
    
    def build_index(self, chunks: List[CodeChunk], output_path: str) -> None:
        """Build BM25 index from code chunks.
        
        Args:
            chunks: List of CodeChunk objects to index.
            output_path: Path where the index should be saved.
        """
        logger.info(f"Building BM25 index with {len(chunks)} chunks")
        
        self._chunks = chunks
        self._tokenized_chunks = []
        
        # Tokenize each chunk
        for chunk in chunks:
            # Combine weighted fields
            tokens = []
            
            # Content tokens
            content_tokens = self.tokenizer.tokenize(chunk.content)
            tokens.extend(content_tokens * int(self.field_weights["content"]))
            
            # Name tokens (boosted)
            if chunk.name:
                name_tokens = self.tokenizer.tokenize(chunk.name)
                tokens.extend(name_tokens * int(self.field_weights["name"]))
            
            # Import tokens
            if chunk.imports:
                import_text = " ".join(chunk.imports)
                import_tokens = self.tokenizer.tokenize(import_text)
                tokens.extend(import_tokens * int(self.field_weights["imports"]))
            
            self._tokenized_chunks.append(tokens)
        
        # Build BM25 index
        self._index = BM25Okapi(
            self._tokenized_chunks,
            k1=self.k1,
            b=self.b
        )
        
        # Save index
        Path(output_path).mkdir(parents=True, exist_ok=True)
        self._save_index(output_path)
        self.save_chunks(output_path)
        self.save_metadata(output_path)
        
        self._index_metadata = {
            "k1": self.k1,
            "b": self.b,
            "chunk_count": len(chunks),
            "tokenizer": "code",
        }
        
        logger.info(f"BM25 index built and saved to {output_path}")
    
    def _save_index(self, output_path: str) -> None:
        """Save the BM25 index to disk."""
        index_file = Path(output_path) / f"{self.index_name}_bm25.pkl"
        
        with open(index_file, "wb") as f:
            pickle.dump({
                "index": self._index,
                "tokenized_chunks": self._tokenized_chunks,
                "k1": self.k1,
                "b": self.b,
            }, f)
        
        logger.info(f"Saved BM25 index to {index_file}")
    
    def load_index(self, index_path: str) -> None:
        """Load a pre-built BM25 index.
        
        Args:
            index_path: Path to the saved index directory.
        """
        index_file = Path(index_path) / f"{self.index_name}_bm25.pkl"
        
        if not index_file.exists():
            raise FileNotFoundError(f"BM25 index not found: {index_file}")
        
        with open(index_file, "rb") as f:
            data = pickle.load(f)
        
        self._index = data["index"]
        self._tokenized_chunks = data["tokenized_chunks"]
        self.k1 = data.get("k1", self.k1)
        self.b = data.get("b", self.b)
        
        # Load chunks
        self.load_chunks(index_path)
        
        # Load metadata
        self.load_metadata(index_path)
        
        self._initialized = True
        logger.info(f"Loaded BM25 index with {len(self._chunks)} chunks")
    
    def search(self, query: str, top_k: int = 10) -> List[SearchResult]:
        """Search the BM25 index.
        
        Args:
            query: Search query string.
            top_k: Number of top results to return.
            
        Returns:
            List of SearchResult objects sorted by BM25 score.
        """
        if self._index is None:
            raise RuntimeError("Index not loaded. Call load_index() first.")
        
        # Tokenize query
        query_tokens = self.tokenizer.tokenize(query)
        
        if not query_tokens:
            logger.warning("Empty query after tokenization")
            return []
        
        # Get scores
        scores = self._index.get_scores(query_tokens)
        
        # Get top-k indices
        import numpy as np
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        # Create search results
        results = []
        for rank, idx in enumerate(top_indices, start=1):
            if scores[idx] > 0:  # Only include positive scores
                chunk = self._chunks[idx]
                result = SearchResult(
                    chunk=chunk,
                    score=float(scores[idx]),
                    rank=rank,
                    source="bm25"
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
    
    def get_token_stats(self) -> Dict[str, Any]:
        """Get statistics about tokenization."""
        if not self._tokenized_chunks:
            return {}
        
        token_counts = [len(tokens) for tokens in self._tokenized_chunks]
        
        return {
            "total_chunks": len(self._tokenized_chunks),
            "total_tokens": sum(token_counts),
            "avg_tokens_per_chunk": sum(token_counts) / len(token_counts),
            "min_tokens": min(token_counts),
            "max_tokens": max(token_counts),
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get indexer statistics."""
        stats = super().get_stats()
        stats.update({
            "k1": self.k1,
            "b": self.b,
            "field_weights": self.field_weights,
            "token_stats": self.get_token_stats(),
        })
        return stats