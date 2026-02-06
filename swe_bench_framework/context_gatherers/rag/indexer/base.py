"""Base class for repository indexing strategies.

Indexers build and manage searchable indexes from code chunks.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Any
from pathlib import Path
import pickle
import json
import logging

from ..base import CodeChunk, SearchResult, BaseRAGComponent

logger = logging.getLogger(__name__)


class RepositoryIndexer(BaseRAGComponent, ABC):
    """Abstract base class for repository indexing strategies.
    
    Indexers are responsible for:
    1. Building searchable indexes from code chunks
    2. Saving and loading indexes to/from disk
    3. Searching indexes to retrieve relevant code
    
    Different indexing strategies include:
    - BM25: Sparse retrieval using term frequency
    - Dense: Vector-based retrieval using embeddings
    - Hybrid: Combination of multiple strategies
    
    Example:
        ```python
        indexer = BM25Indexer(config={"k1": 1.5, "b": 0.75})
        indexer.build_index(chunks, "/path/to/index")
        indexer.load_index("/path/to/index")
        results = indexer.search("how to handle errors", top_k=10)
        ```
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the indexer.
        
        Args:
            config: Configuration dictionary with options:
                - index_dir: Directory for index storage
                - cache_enabled: Whether to cache indexes (default: True)
                - index_name: Name of the index file
        """
        super().__init__(config)
        self.index_dir = config.get("index_dir", ".rag_index")
        self.cache_enabled = config.get("cache_enabled", True)
        self.index_name = config.get("index_name", "index")
        self._index: Optional[Any] = None
        self._chunks: List[CodeChunk] = []
        self._index_metadata: Dict[str, Any] = {}
    
    @abstractmethod
    def build_index(self, chunks: List[CodeChunk], output_path: str) -> None:
        """Build index from code chunks.
        
        Args:
            chunks: List of CodeChunk objects to index.
            output_path: Path where the index should be saved.
        """
        pass
    
    @abstractmethod
    def load_index(self, index_path: str) -> None:
        """Load a pre-built index.
        
        Args:
            index_path: Path to the saved index.
        """
        pass
    
    @abstractmethod
    def search(self, query: str, top_k: int = 10) -> List[SearchResult]:
        """Search the index for relevant code chunks.
        
        Args:
            query: Search query string.
            top_k: Number of top results to return.
            
        Returns:
            List of SearchResult objects sorted by relevance.
        """
        pass
    
    def save_metadata(self, output_path: str) -> None:
        """Save index metadata to disk.
        
        Args:
            output_path: Directory where metadata should be saved.
        """
        metadata_path = Path(output_path) / f"{self.index_name}_metadata.json"
        
        metadata = {
            "indexer_type": self.name,
            "config": self.config,
            "chunk_count": len(self._chunks),
            "index_metadata": self._index_metadata,
        }
        
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2, default=str)
        
        logger.info(f"Saved metadata to {metadata_path}")
    
    def load_metadata(self, index_path: str) -> Dict[str, Any]:
        """Load index metadata from disk.
        
        Args:
            index_path: Directory containing the metadata file.
            
        Returns:
            Metadata dictionary.
        """
        metadata_path = Path(index_path) / f"{self.index_name}_metadata.json"
        
        if not metadata_path.exists():
            logger.warning(f"Metadata file not found: {metadata_path}")
            return {}
        
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        
        self._index_metadata = metadata.get("index_metadata", {})
        return metadata
    
    def save_chunks(self, output_path: str) -> None:
        """Save chunks to disk for later retrieval.
        
        Args:
            output_path: Directory where chunks should be saved.
        """
        chunks_path = Path(output_path) / f"{self.index_name}_chunks.pkl"
        
        with open(chunks_path, "wb") as f:
            pickle.dump(self._chunks, f)
        
        logger.info(f"Saved {len(self._chunks)} chunks to {chunks_path}")
    
    def load_chunks(self, index_path: str) -> List[CodeChunk]:
        """Load chunks from disk.
        
        Args:
            index_path: Directory containing the chunks file.
            
        Returns:
            List of CodeChunk objects.
        """
        chunks_path = Path(index_path) / f"{self.index_name}_chunks.pkl"
        
        if not chunks_path.exists():
            logger.warning(f"Chunks file not found: {chunks_path}")
            return []
        
        with open(chunks_path, "rb") as f:
            self._chunks = pickle.load(f)
        
        logger.info(f"Loaded {len(self._chunks)} chunks from {chunks_path}")
        return self._chunks
    
    def get_chunk_by_index(self, idx: int) -> Optional[CodeChunk]:
        """Get a chunk by its index.
        
        Args:
            idx: Chunk index.
            
        Returns:
            CodeChunk if found, None otherwise.
        """
        if 0 <= idx < len(self._chunks):
            return self._chunks[idx]
        return None
    
    def get_index_size(self) -> int:
        """Get the number of chunks in the index."""
        return len(self._chunks)
    
    def is_index_loaded(self) -> bool:
        """Check if an index is currently loaded."""
        return self._index is not None and len(self._chunks) > 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get indexer statistics."""
        stats = super().get_stats()
        stats.update({
            "index_loaded": self.is_index_loaded(),
            "chunk_count": len(self._chunks),
            "index_metadata": self._index_metadata,
        })
        return stats
    
    def initialize(self, **kwargs) -> None:
        """Initialize the indexer."""
        self._initialized = True
        logger.info(f"Initialized {self.name}")
    
    def cleanup(self) -> None:
        """Cleanup resources."""
        self._index = None
        self._chunks = []
        self._initialized = False


class Tokenizer(ABC):
    """Abstract base class for text tokenizers used in indexing.
    
    Tokenizers convert text into tokens for indexing and search.
    """
    
    @abstractmethod
    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into a list of tokens.
        
        Args:
            text: Text to tokenize.
            
        Returns:
            List of tokens.
        """
        pass


class CodeTokenizer(Tokenizer):
    """Tokenizer optimized for code.
    
    This tokenizer handles code-specific tokenization:
    - Preserves camelCase and snake_case identifiers
    - Handles operators and punctuation
    - Normalizes whitespace
    """
    
    def __init__(self, lowercase: bool = True, split_camel_case: bool = True):
        """Initialize the code tokenizer.
        
        Args:
            lowercase: Convert tokens to lowercase.
            split_camel_case: Split camelCase identifiers.
        """
        self.lowercase = lowercase
        self.split_camel_case = split_camel_case
        import re
        self._token_pattern = re.compile(r'[a-zA-Z_][a-zA-Z0-9_]*|\d+|[{}()\[\];,.=+\-*/<>!&|]+')
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenize code text."""
        import re
        
        # Find all tokens
        tokens = self._token_pattern.findall(text)
        
        result = []
        for token in tokens:
            # Split camelCase if enabled
            if self.split_camel_case:
                # Split on camelCase boundaries
                parts = re.sub('([A-Z][a-z]+)', r' \1', re.sub('([A-Z]+)', r' \1', token)).split()
                result.extend(parts)
            else:
                result.append(token)
        
        # Normalize
        if self.lowercase:
            result = [t.lower() for t in result]
        
        # Filter out very short tokens
        result = [t for t in result if len(t) > 1 or t.isdigit()]
        
        return result


class SimpleTokenizer(Tokenizer):
    """Simple whitespace-based tokenizer."""
    
    def __init__(self, lowercase: bool = True):
        self.lowercase = lowercase
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenize by whitespace."""
        if self.lowercase:
            text = text.lower()
        return text.split()