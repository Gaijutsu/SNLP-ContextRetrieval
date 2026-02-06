"""Dense indexing implementation using embeddings and FAISS.

This module provides vector-based dense retrieval using code embeddings
and FAISS for efficient similarity search.
"""

from typing import List, Dict, Optional, Any, Callable
from pathlib import Path
import pickle
import logging
import numpy as np

from .base import RepositoryIndexer
from ..base import CodeChunk, SearchResult

logger = logging.getLogger(__name__)


class EmbeddingModel(ABC):
    """Abstract base class for embedding models."""
    
    @abstractmethod
    def encode(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Encode texts into embeddings.
        
        Args:
            texts: List of texts to encode.
            batch_size: Batch size for encoding.
            
        Returns:
            Array of embeddings with shape (len(texts), embedding_dim).
        """
        pass
    
    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        """Get the embedding dimension."""
        pass


class SentenceTransformerModel(EmbeddingModel):
    """SentenceTransformer-based embedding model."""
    
    def __init__(self, model_name: str, device: str = "cpu"):
        """Initialize the model.
        
        Args:
            model_name: Name of the sentence-transformer model.
            device: Device to use ("cpu", "cuda", etc.).
        """
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers is required. "
                "Install with: pip install sentence-transformers"
            )
        
        self.model = SentenceTransformer(model_name, device=device)
        self._embedding_dim = self.model.get_sentence_embedding_dimension()
    
    def encode(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Encode texts into embeddings."""
        return self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True
        )
    
    @property
    def embedding_dim(self) -> int:
        """Get the embedding dimension."""
        return self._embedding_dim


class JinaEmbeddingModel(EmbeddingModel):
    """Jina AI code embedding model."""
    
    def __init__(self, model_name: str = "jinaai/jina-embeddings-v2-base-code", device: str = "cpu"):
        """Initialize the Jina model.
        
        Args:
            model_name: Name of the Jina model.
            device: Device to use.
        """
        try:
            from transformers import AutoModel
        except ImportError:
            raise ImportError(
                "transformers is required for Jina models. "
                "Install with: pip install transformers"
            )
        
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        self.model.to(device)
        self.model.eval()
        self.device = device
        self._embedding_dim = self.model.config.hidden_size
    
    def encode(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Encode texts into embeddings."""
        import torch
        
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            # Tokenize
            inputs = self.model.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=8192,
                return_tensors="pt"
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Encode
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Mean pooling
                embeddings = outputs.last_hidden_state.mean(dim=1)
                all_embeddings.append(embeddings.cpu().numpy())
        
        return np.vstack(all_embeddings)
    
    @property
    def embedding_dim(self) -> int:
        """Get the embedding dimension."""
        return self._embedding_dim


class DenseIndexer(RepositoryIndexer):
    """Dense embedding-based indexer using FAISS.
    
    Dense retrieval uses neural encoders to convert code and queries into
    dense vector representations, enabling semantic similarity search.
    
    Supports multiple embedding models:
    - Jina embeddings (jina-embeddings-v2-base-code)
    - CodeBERT (microsoft/codebert-base)
    - Sentence transformers (all-MiniLM-L6-v2, etc.)
    
    Example:
        ```python
        indexer = DenseIndexer(config={
            "model": "jinaai/jina-embeddings-v2-base-code",
            "index_type": "IndexFlatIP"
        })
        indexer.build_index(chunks, "/path/to/index")
        results = indexer.search("how to handle errors", top_k=10)
        ```
    """
    
    # Supported index types
    INDEX_TYPES = {
        "IndexFlatIP": "Exact search with inner product",
        "IndexFlatL2": "Exact search with L2 distance",
        "IndexIVFFlat": "Inverted file index (faster, approximate)",
        "IndexHNSWFlat": "Hierarchical NSW graph (fast, approximate)",
    }
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the dense indexer.
        
        Args:
            config: Configuration dictionary with options:
                - model: Embedding model name (default: "all-MiniLM-L6-v2")
                - model_type: Model type - "sentence_transformer", "jina", "codebert"
                - index_type: FAISS index type (default: "IndexFlatIP")
                - nlist: Number of clusters for IVF index (default: 100)
                - nprobe: Number of clusters to search (default: 10)
                - device: Device for encoding (default: "cpu")
                - batch_size: Batch size for encoding (default: 32)
                - normalize: Normalize embeddings (default: True)
        """
        super().__init__(config)
        
        # Model configuration
        self.model_name = config.get("model", "all-MiniLM-L6-v2")
        self.model_type = config.get("model_type", "sentence_transformer")
        self.device = config.get("device", "cpu")
        self.batch_size = config.get("batch_size", 32)
        self.normalize = config.get("normalize", True)
        
        # Index configuration
        self.index_type = config.get("index_type", "IndexFlatIP")
        self.nlist = config.get("nlist", 100)  # For IVF
        self.nprobe = config.get("nprobe", 10)  # For IVF
        
        # Initialize embedding model
        self._embedding_model: Optional[EmbeddingModel] = None
        
        # FAISS index
        self._faiss_index: Optional[Any] = None
        self._embeddings: Optional[np.ndarray] = None
        
        # Check FAISS availability
        try:
            import faiss
            self._faiss = faiss
            self._has_faiss = True
        except ImportError:
            logger.warning("FAISS not available. Dense indexing will not work.")
            self._has_faiss = False
            self._faiss = None
    
    def _get_embedding_model(self) -> EmbeddingModel:
        """Get or create the embedding model."""
        if self._embedding_model is None:
            if self.model_type == "jina":
                self._embedding_model = JinaEmbeddingModel(self.model_name, self.device)
            else:
                self._embedding_model = SentenceTransformerModel(self.model_name, self.device)
        return self._embedding_model
    
    def _encode_chunks(self, chunks: List[CodeChunk]) -> np.ndarray:
        """Encode chunks into embeddings."""
        model = self._get_embedding_model()
        
        # Prepare texts for encoding
        texts = []
        for chunk in chunks:
            # Combine content with name for better semantic matching
            text_parts = []
            if chunk.name:
                text_parts.append(chunk.name)
            text_parts.append(chunk.content)
            texts.append("\n".join(text_parts))
        
        # Encode in batches
        logger.info(f"Encoding {len(texts)} chunks with {self.model_name}")
        embeddings = model.encode(texts, batch_size=self.batch_size)
        
        # Normalize if configured
        if self.normalize:
            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        return embeddings
    
    def _create_faiss_index(self, embeddings: np.ndarray) -> Any:
        """Create FAISS index from embeddings."""
        if not self._has_faiss:
            raise RuntimeError("FAISS is required for dense indexing")
        
        dim = embeddings.shape[1]
        
        if self.index_type == "IndexFlatIP":
            index = self._faiss.IndexFlatIP(dim)
        elif self.index_type == "IndexFlatL2":
            index = self._faiss.IndexFlatL2(dim)
        elif self.index_type == "IndexIVFFlat":
            quantizer = self._faiss.IndexFlatIP(dim)
            index = self._faiss.IndexIVFFlat(quantizer, dim, self.nlist)
            index.train(embeddings)
        elif self.index_type == "IndexHNSWFlat":
            index = self._faiss.IndexHNSWFlat(dim, 32)
        else:
            raise ValueError(f"Unsupported index type: {self.index_type}")
        
        index.add(embeddings)
        
        # Set nprobe for IVF indexes
        if hasattr(index, 'nprobe'):
            index.nprobe = self.nprobe
        
        return index
    
    def build_index(self, chunks: List[CodeChunk], output_path: str) -> None:
        """Build dense index from code chunks.
        
        Args:
            chunks: List of CodeChunk objects to index.
            output_path: Path where the index should be saved.
        """
        if not self._has_faiss:
            raise RuntimeError("FAISS is required for dense indexing")
        
        logger.info(f"Building dense index with {len(chunks)} chunks")
        
        self._chunks = chunks
        
        # Encode chunks
        self._embeddings = self._encode_chunks(chunks)
        
        # Create FAISS index
        self._faiss_index = self._create_faiss_index(self._embeddings)
        
        # Save index
        Path(output_path).mkdir(parents=True, exist_ok=True)
        self._save_index(output_path)
        self.save_chunks(output_path)
        self.save_metadata(output_path)
        
        self._index_metadata = {
            "model": self.model_name,
            "model_type": self.model_type,
            "index_type": self.index_type,
            "embedding_dim": self._embeddings.shape[1],
            "chunk_count": len(chunks),
            "normalize": self.normalize,
        }
        
        logger.info(f"Dense index built and saved to {output_path}")
    
    def _save_index(self, output_path: str) -> None:
        """Save the dense index to disk."""
        if not self._has_faiss:
            raise RuntimeError("FAISS is required")
        
        index_file = Path(output_path) / f"{self.index_name}_dense.faiss"
        embeddings_file = Path(output_path) / f"{self.index_name}_embeddings.npy"
        
        # Save FAISS index
        self._faiss.write_index(self._faiss_index, str(index_file))
        
        # Save embeddings
        np.save(embeddings_file, self._embeddings)
        
        logger.info(f"Saved dense index to {index_file}")
    
    def load_index(self, index_path: str) -> None:
        """Load a pre-built dense index.
        
        Args:
            index_path: Path to the saved index directory.
        """
        if not self._has_faiss:
            raise RuntimeError("FAISS is required for dense indexing")
        
        index_file = Path(index_path) / f"{self.index_name}_dense.faiss"
        embeddings_file = Path(index_path) / f"{self.index_name}_embeddings.npy"
        
        if not index_file.exists():
            raise FileNotFoundError(f"Dense index not found: {index_file}")
        
        # Load FAISS index
        self._faiss_index = self._faiss.read_index(str(index_file))
        
        # Load embeddings
        if embeddings_file.exists():
            self._embeddings = np.load(embeddings_file)
        
        # Load chunks
        self.load_chunks(index_path)
        
        # Load metadata
        metadata = self.load_metadata(index_path)
        if metadata:
            self.model_name = metadata.get("config", {}).get("model", self.model_name)
            self.index_type = metadata.get("config", {}).get("index_type", self.index_type)
        
        self._initialized = True
        logger.info(f"Loaded dense index with {len(self._chunks)} chunks")
    
    def search(self, query: str, top_k: int = 10) -> List[SearchResult]:
        """Search the dense index.
        
        Args:
            query: Search query string.
            top_k: Number of top results to return.
            
        Returns:
            List of SearchResult objects sorted by similarity.
        """
        if self._faiss_index is None:
            raise RuntimeError("Index not loaded. Call load_index() first.")
        
        # Encode query
        model = self._get_embedding_model()
        query_embedding = model.encode([query], batch_size=1)
        
        # Normalize if configured
        if self.normalize:
            query_embedding = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)
        
        # Search
        scores, indices = self._faiss_index.search(query_embedding.astype('float32'), top_k)
        
        # Create search results
        results = []
        for rank, (idx, score) in enumerate(zip(indices[0], scores[0]), start=1):
            if idx >= 0 and idx < len(self._chunks):  # Valid index
                chunk = self._chunks[idx]
                result = SearchResult(
                    chunk=chunk,
                    score=float(score),
                    rank=rank,
                    source="dense"
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
        if self._faiss_index is None:
            raise RuntimeError("Index not loaded. Call load_index() first.")
        
        # Encode queries
        model = self._get_embedding_model()
        query_embeddings = model.encode(queries, batch_size=self.batch_size)
        
        # Normalize if configured
        if self.normalize:
            query_embeddings = query_embeddings / np.linalg.norm(query_embeddings, axis=1, keepdims=True)
        
        # Search
        scores, indices = self._faiss_index.search(query_embeddings.astype('float32'), top_k)
        
        # Create results
        all_results = []
        for query_idx in range(len(queries)):
            results = []
            for rank, (idx, score) in enumerate(zip(indices[query_idx], scores[query_idx]), start=1):
                if idx >= 0 and idx < len(self._chunks):
                    chunk = self._chunks[idx]
                    result = SearchResult(
                        chunk=chunk,
                        score=float(score),
                        rank=rank,
                        source="dense"
                    )
                    results.append(result)
            all_results.append(results)
        
        return all_results
    
    def get_embedding_stats(self) -> Dict[str, Any]:
        """Get statistics about embeddings."""
        if self._embeddings is None:
            return {}
        
        return {
            "embedding_dim": self._embeddings.shape[1],
            "num_embeddings": self._embeddings.shape[0],
            "avg_norm": float(np.mean(np.linalg.norm(self._embeddings, axis=1))),
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get indexer statistics."""
        stats = super().get_stats()
        stats.update({
            "model": self.model_name,
            "model_type": self.model_type,
            "index_type": self.index_type,
            "device": self.device,
            "normalize": self.normalize,
            "embedding_stats": self.get_embedding_stats(),
        })
        return stats