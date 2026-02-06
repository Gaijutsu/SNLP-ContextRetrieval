# Context Gatherers API Documentation

This document describes the Context Gatherer API, which provides a unified interface for both agentic exploration and RAG-based context gathering methods.

---

## Overview

The `ContextGatherer` interface is the core abstraction for all context gathering strategies. Both agentic methods (like AutoCodeRover, SWE-agent) and RAG methods (like BM25, Dense retrieval) implement this interface, enabling fair comparison using identical downstream components.

```python
from swe_bench_framework.context_gatherers import ContextGatherer

class MyGatherer(ContextGatherer):
    def gather_context(self, instance, repo_path):
        # Implementation
        pass
    
    def initialize(self, repo_path):
        pass
    
    def cleanup(self):
        pass
```

---

## ContextGatherer Interface

### Class Definition

```python
class ContextGatherer(ABC):
    """
    Abstract base class for all context gathering strategies.
    
    Both agentic and RAG methods implement this interface, providing
    a unified way to gather context for patch generation.
    
    Example:
        >>> gatherer = MyContextGatherer(config)
        >>> gatherer.initialize(repo_path)
        >>> context = gatherer.gather_context(instance, repo_path)
        >>> gatherer.cleanup()
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the context gatherer.
        
        Args:
            config: Configuration dictionary for the gatherer
        """
        self.config = config
        self.name = self.__class__.__name__
        self._initialized = False
```

### Abstract Methods

#### `gather_context()`

```python
@abstractmethod
def gather_context(
    self,
    instance: SWEInstance,
    repo_path: str
) -> ContextBundle:
    """
    Gather context for a given SWE-bench instance.
    
    This is the main method that all context gathering strategies
    must implement. It should return a ContextBundle containing
    all relevant context for patch generation.
    
    Args:
        instance: The SWE-bench instance containing problem statement
        repo_path: Path to the repository checkout
        
    Returns:
        ContextBundle containing all gathered context
        
    Raises:
        GatheringError: If context gathering fails
    """
    pass
```

#### `initialize()`

```python
@abstractmethod
def initialize(self, repo_path: str) -> None:
    """
    Initialize any resources needed (indexes, models, etc.).
    
    This method should be called before gather_context() and is
    used to set up any resources that are reused across multiple
    instances from the same repository.
    
    Args:
        repo_path: Path to the repository
        
    Raises:
        RepositoryError: If repository initialization fails
    """
    pass
```

#### `cleanup()`

```python
@abstractmethod
def cleanup(self) -> None:
    """
    Cleanup resources.
    
    This method should be called after all instances have been
    processed to release any resources held by the gatherer.
    """
    pass
```

### Utility Methods

#### `get_stats()`

```python
def get_stats(self) -> Dict[str, Any]:
    """
    Return statistics about context gathering.
    
    Returns:
        Dictionary with statistics (implementation-specific)
    """
    return {}
```

#### `is_initialized()`

```python
def is_initialized(self) -> bool:
    """Check if the gatherer has been initialized."""
    return self._initialized
```

---

## Data Models

### ContextBundle

```python
@dataclass
class ContextBundle:
    """
    Bundle of context gathered for a patch generation task.
    
    This is the unified output format for all context gathering methods.
    It contains all relevant context needed for patch generation.
    """
    
    instance_id: str
    """Unique identifier for the SWE-bench instance."""
    
    problem_statement: str
    """The problem statement from the issue."""
    
    chunks: List[ContextChunk]
    """List of context chunks gathered."""
    
    repo_structure: Dict[str, Any]
    """Repository structure information."""
    
    gathered_at: str
    """ISO timestamp when context was gathered."""
    
    gatherer_type: str
    """Type of gatherer used (e.g., 'agentic', 'rag')."""
    
    token_count: int
    """Total token count of all chunks."""
    
    metadata: Dict[str, Any] = None
    """Additional metadata (implementation-specific)."""
    
    def to_prompt_context(self, max_tokens: int = 8000) -> str:
        """
        Convert context bundle to string for LLM prompt.
        
        Args:
            max_tokens: Maximum tokens to include
            
        Returns:
            Formatted context string
        """
        # Implementation: sort by relevance, truncate to fit token limit
        pass
```

### ContextChunk

```python
@dataclass
class ContextChunk:
    """
    A single piece of context.
    
    Represents a code snippet or other contextual information
    gathered from the repository.
    """
    
    content: str
    """The actual content (code, text, etc.)."""
    
    source_file: str
    """Path to the source file."""
    
    context_type: ContextType
    """Type of context (see ContextType enum)."""
    
    start_line: int
    """Starting line number in the source file."""
    
    end_line: int
    """Ending line number in the source file."""
    
    relevance_score: float = 0.0
    """Relevance score (0.0-1.0, higher is more relevant)."""
    
    metadata: Dict[str, Any] = None
    """Additional metadata (e.g., AST node type, embedding)."""
```

### ContextType

```python
class ContextType(Enum):
    """Types of context that can be gathered."""
    
    FILE_CONTENT = "file_content"
    """General file content."""
    
    CLASS_DEFINITION = "class_definition"
    """Class definition."""
    
    FUNCTION_DEFINITION = "function_definition"
    """Function or method definition."""
    
    IMPORT_DEPENDENCY = "import_dependency"
    """Import statements."""
    
    TEST_CONTEXT = "test_context"
    """Test file content."""
    
    ERROR_CONTEXT = "error_context"
    """Error messages, stack traces."""
    
    REPO_STRUCTURE = "repo_structure"
    """Repository structure information."""
```

---

## Agentic Methods

### BaseAgenticGatherer

```python
class BaseAgenticGatherer(ContextGatherer):
    """
    Base class for agentic context gathering methods.
    
    Provides common functionality for agent-based exploration,
    including tool management and ReAct loop support.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.environment: Optional[AgentEnvironment] = None
        self.max_iterations = config.get('max_iterations', 50)
    
    def initialize(self, repo_path: str) -> None:
        """Initialize the agent environment."""
        self.environment = AgentEnvironment(repo_path, self.config)
        self._initialized = True
    
    def cleanup(self) -> None:
        """Cleanup agent environment."""
        if self.environment:
            self.environment.cleanup()
        self._initialized = False
    
    @abstractmethod
    def _run_exploration(self, instance: SWEInstance) -> List[ContextChunk]:
        """
        Run the agent exploration loop.
        
        Args:
            instance: The SWE-bench instance
            
        Returns:
            List of gathered context chunks
        """
        pass
```

### AutoCodeRover Gatherer

```python
class AutoCodeRoverGatherer(BaseAgenticGatherer):
    """
    AutoCodeRover-style context gathering.
    
    Uses AST-based code search with spectrum-based fault localization.
    Implements stratified retrieval with hierarchical search.
    
    Key Features:
    - AST-based code navigation
    - Spectrum-based fault localization (SBFL)
    - Stratified retrieval strategy
    - Iterative search refinement
    
    Example:
        >>> config = {
        ...     'max_iterations': 50,
        ...     'sbfl': {'enabled': True, 'formula': 'ochiai'},
        ...     'retrieval': {
        ...         'strategy': 'stratified',
        ...         'layers': [
        ...             {'type': 'file', 'top_k': 5},
        ...             {'type': 'class', 'top_k': 10},
        ...             {'type': 'function', 'top_k': 20}
        ...         ]
        ...     }
        ... }
        >>> gatherer = AutoCodeRoverGatherer(config)
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.sbfl_enabled = config.get('sbfl', {}).get('enabled', True)
        self.sbfl_formula = config.get('sbfl', {}).get('formula', 'ochiai')
    
    def gather_context(self, instance: SWEInstance, repo_path: str) -> ContextBundle:
        """
        Gather context using AutoCodeRover strategy.
        
        Process:
        1. Extract keywords from problem statement
        2. Run SBFL if test suite available
        3. Stratified search: file -> class -> function
        4. Iterative refinement based on results
        5. Convert gathered info to ContextBundle
        """
        # Implementation
        pass
```

**Configuration Options:**

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `max_iterations` | int | 50 | Maximum exploration iterations |
| `sbfl.enabled` | bool | True | Enable spectrum-based fault localization |
| `sbfl.formula` | str | "ochiai" | SBFL formula (ochiai, tarantula, jaccard) |
| `retrieval.strategy` | str | "stratified" | Retrieval strategy |
| `retrieval.layers` | list | [...] | Hierarchical search layers |

### SWE-agent Gatherer

```python
class SWEAgentGatherer(BaseAgenticGatherer):
    """
    SWE-agent-style context gathering.
    
    Uses Agent-Computer Interface (ACI) with ReAct reasoning.
    Provides purpose-built tools optimized for LM agents.
    
    Key Features:
    - Agent-Computer Interface (ACI)
    - ReAct-style reasoning loop
    - Specialized file viewer
    - Linter integration
    - Test execution
    
    Example:
        >>> config = {
        ...     'max_iterations': 100,
        ...     'aci': {
        ...         'enable_linter': True,
        ...         'enable_test_runner': True,
        ...         'file_viewer_lines': 100
        ...     },
        ...     'react': {
        ...         'max_thought_length': 500
        ...     }
        ... }
        >>> gatherer = SWEAgentGatherer(config)
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.aci_config = config.get('aci', {})
        self.react_config = config.get('react', {})
    
    def gather_context(self, instance: SWEInstance, repo_path: str) -> ContextBundle:
        """
        Gather context using SWE-agent strategy.
        
        Process:
        1. Initialize ACI environment
        2. ReAct loop: Thought -> Action -> Observation
        3. Execute ACI commands (view, search, edit, test)
        4. Maintain conversation history
        5. Convert to ContextBundle
        """
        # Implementation
        pass
```

**Configuration Options:**

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `max_iterations` | int | 100 | Maximum ReAct iterations |
| `aci.enable_linter` | bool | True | Enable linter integration |
| `aci.enable_test_runner` | bool | True | Enable test execution |
| `aci.file_viewer_lines` | int | 100 | Lines shown by file viewer |
| `react.max_thought_length` | int | 500 | Max length of thoughts |

### Agentless Gatherer

```python
class AgentlessGatherer(BaseAgenticGatherer):
    """
    Agentless-style context gathering.
    
    Uses hierarchical localization without agent scaffolding.
    Three-phase process: localization -> repair -> validation.
    
    Key Features:
    - Hierarchical localization (file -> class/function -> location)
    - No agent autonomy (fixed workflow)
    - Multiple patch generation
    - Patch filtering and re-ranking
    
    Example:
        >>> config = {
        ...     'localization': {
        ...         'strategy': 'hierarchical',
        ...         'file_top_k': 5,
        ...         'function_top_k': 10
        ...     },
        ...     'repair': {
        ...         'num_samples': 10,
        ...         'temperature': 0.7
        ...     }
        ... }
        >>> gatherer = AgentlessGatherer(config)
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.localization_config = config.get('localization', {})
        self.repair_config = config.get('repair', {})
    
    def gather_context(self, instance: SWEInstance, repo_path: str) -> ContextBundle:
        """
        Gather context using Agentless strategy.
        
        Process:
        1. File-level localization (identify suspicious files)
        2. Class/function-level localization
        3. Pinpoint exact edit locations
        4. Gather context around identified locations
        5. Convert to ContextBundle
        """
        # Implementation
        pass
```

**Configuration Options:**

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `localization.strategy` | str | "hierarchical" | Localization strategy |
| `localization.file_top_k` | int | 5 | Top-K files to consider |
| `localization.function_top_k` | int | 10 | Top-K functions to consider |
| `repair.num_samples` | int | 10 | Number of patch samples |
| `repair.temperature` | float | 0.7 | Temperature for sampling |

---

## RAG Methods

### BaseRAGGatherer

```python
class BaseRAGGatherer(ContextGatherer):
    """
    Base class for RAG-based context gathering methods.
    
    Provides common functionality for retrieval-based methods,
    including index management and query processing.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.indexer: Optional[RepositoryIndexer] = None
        self.retriever: Optional[Retriever] = None
        self.top_k = config.get('top_k', 20)
    
    def initialize(self, repo_path: str) -> None:
        """Initialize indexer and build/load index."""
        self.indexer = self._create_indexer()
        index_path = self._get_index_path(repo_path)
        
        if os.path.exists(index_path):
            self.indexer.load_index(index_path)
        else:
            self.indexer.build_index(repo_path, index_path)
        
        self.retriever = self._create_retriever(self.indexer)
        self._initialized = True
    
    def cleanup(self) -> None:
        """Cleanup indexer and retriever."""
        if self.indexer:
            self.indexer.cleanup()
        self._initialized = False
    
    @abstractmethod
    def _create_indexer(self) -> RepositoryIndexer:
        """Create the appropriate indexer."""
        pass
    
    @abstractmethod
    def _create_retriever(self, indexer: RepositoryIndexer) -> Retriever:
        """Create the appropriate retriever."""
        pass
```

### BM25 Gatherer

```python
class BM25Gatherer(BaseRAGGatherer):
    """
    BM25-based sparse retrieval context gathering.
    
    Uses probabilistic retrieval with term frequency weighting.
    Fast and interpretable, good for keyword-based matching.
    
    Key Features:
    - BM25 probabilistic scoring
    - Fast retrieval from large codebases
    - Keyword-based matching
    - Configurable parameters (k1, b)
    
    Example:
        >>> config = {
        ...     'indexing': {
        ...         'chunking': {'strategy': 'ast', 'max_chunk_size': 500},
        ...         'bm25': {'k1': 1.5, 'b': 0.75}
        ...     },
        ...     'retrieval': {'top_k': 20}
        ... }
        >>> gatherer = BM25Gatherer(config)
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.bm25_config = config.get('indexing', {}).get('bm25', {})
        self.k1 = self.bm25_config.get('k1', 1.5)
        self.b = self.bm25_config.get('b', 0.75)
    
    def gather_context(self, instance: SWEInstance, repo_path: str) -> ContextBundle:
        """
        Gather context using BM25 retrieval.
        
        Process:
        1. Extract query from problem statement
        2. Search BM25 index
        3. Retrieve top-K chunks
        4. Convert to ContextBundle
        """
        query = self._extract_query(instance.problem_statement)
        results = self.retriever.retrieve(query, top_k=self.top_k)
        chunks = self._results_to_chunks(results)
        
        return ContextBundle(
            instance_id=instance.instance_id,
            problem_statement=instance.problem_statement,
            chunks=chunks,
            repo_structure=self._get_repo_structure(),
            gathered_at=datetime.utcnow().isoformat(),
            gatherer_type='rag_bm25',
            token_count=sum(c.token_count for c in chunks),
            metadata={'query': query, 'k1': self.k1, 'b': self.b}
        )
```

**Configuration Options:**

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `indexing.chunking.strategy` | str | "ast" | Chunking strategy |
| `indexing.chunking.max_chunk_size` | int | 500 | Max chunk size |
| `indexing.bm25.k1` | float | 1.5 | BM25 term frequency parameter |
| `indexing.bm25.b` | float | 0.75 | BM25 length normalization parameter |
| `retrieval.top_k` | int | 20 | Number of results to retrieve |

### Dense Gatherer

```python
class DenseGatherer(BaseRAGGatherer):
    """
    Dense embedding-based context gathering.
    
    Uses neural embeddings for semantic similarity search.
    Captures semantic meaning beyond exact keyword matching.
    
    Key Features:
    - Neural embeddings (code-specific models)
    - Semantic similarity search
    - Cross-lingual capabilities
    - Configurable embedding models
    
    Example:
        >>> config = {
        ...     'indexing': {
        ...         'chunking': {'strategy': 'ast', 'max_chunk_size': 500},
        ...         'embeddings': {
        ...             'model': 'jinaai/jina-embeddings-v2-base-code',
        ...             'batch_size': 32,
        ...             'device': 'cuda'
        ...         }
        ...     },
        ...     'retrieval': {
        ...         'top_k': 20,
        ...         'similarity_metric': 'cosine'
        ...     }
        ... }
        >>> gatherer = DenseGatherer(config)
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.embedding_config = config.get('indexing', {}).get('embeddings', {})
        self.model_name = self.embedding_config.get(
            'model', 
            'jinaai/jina-embeddings-v2-base-code'
        )
    
    def gather_context(self, instance: SWEInstance, repo_path: str) -> ContextBundle:
        """
        Gather context using dense retrieval.
        
        Process:
        1. Encode problem statement as query embedding
        2. Search vector index
        3. Retrieve top-K similar chunks
        4. Convert to ContextBundle
        """
        query = self._extract_query(instance.problem_statement)
        query_embedding = self._encode_query(query)
        results = self.retriever.retrieve(
            query_embedding, 
            top_k=self.top_k
        )
        chunks = self._results_to_chunks(results)
        
        return ContextBundle(
            instance_id=instance.instance_id,
            problem_statement=instance.problem_statement,
            chunks=chunks,
            repo_structure=self._get_repo_structure(),
            gathered_at=datetime.utcnow().isoformat(),
            gatherer_type='rag_dense',
            token_count=sum(c.token_count for c in chunks),
            metadata={'query': query, 'model': self.model_name}
        )
```

**Configuration Options:**

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `indexing.chunking.strategy` | str | "ast" | Chunking strategy |
| `indexing.chunking.max_chunk_size` | int | 500 | Max chunk size |
| `indexing.embeddings.model` | str | "jinaai/jina-embeddings-v2-base-code" | Embedding model |
| `indexing.embeddings.batch_size` | int | 32 | Batch size for encoding |
| `indexing.embeddings.device` | str | "cuda" | Device (cuda/cpu) |
| `retrieval.top_k` | int | 20 | Number of results to retrieve |
| `retrieval.similarity_metric` | str | "cosine" | Similarity metric |

### Hybrid Gatherer

```python
class HybridGatherer(BaseRAGGatherer):
    """
    Hybrid BM25 + Dense context gathering.
    
    Combines sparse (BM25) and dense retrieval with
    Reciprocal Rank Fusion (RRF) for best of both approaches.
    
    Key Features:
    - Combined BM25 and dense retrieval
    - Reciprocal Rank Fusion (RRF)
    - Optional cross-encoder re-ranking
    - Configurable weighting
    
    Example:
        >>> config = {
        ...     'indexing': {
        ...         'chunking': {'strategy': 'ast', 'max_chunk_size': 500},
        ...         'embeddings': {
        ...             'model': 'jinaai/jina-embeddings-v2-base-code',
        ...             'batch_size': 32
        ...         },
        ...         'bm25': {'k1': 1.5, 'b': 0.75}
        ...     },
        ...     'retrieval': {
        ...         'sparse_weight': 0.3,
        ...         'dense_weight': 0.7,
        ...         'rrf_k': 60,
        ...         'top_k': 20,
        ...         'reranker': {
        ...             'enabled': True,
        ...             'model': 'cross-encoder/ms-marco-MiniLM-L-6-v2',
        ...             'top_k': 10
        ...         }
        ...     }
        ... }
        >>> gatherer = HybridGatherer(config)
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        retrieval_config = config.get('retrieval', {})
        self.sparse_weight = retrieval_config.get('sparse_weight', 0.3)
        self.dense_weight = retrieval_config.get('dense_weight', 0.7)
        self.rrf_k = retrieval_config.get('rrf_k', 60)
        self.reranker_config = retrieval_config.get('reranker', {})
    
    def gather_context(self, instance: SWEInstance, repo_path: str) -> ContextBundle:
        """
        Gather context using hybrid retrieval.
        
        Process:
        1. Extract query from problem statement
        2. Search BM25 index
        3. Search dense index
        4. Fuse results with RRF
        5. Optional: re-rank with cross-encoder
        6. Convert to ContextBundle
        """
        query = self._extract_query(instance.problem_statement)
        
        # Retrieve from both indexes
        sparse_results = self.sparse_retriever.retrieve(query, top_k=self.top_k)
        dense_results = self.dense_retriever.retrieve(query, top_k=self.top_k)
        
        # Fuse with RRF
        fused_results = self._reciprocal_rank_fusion(
            sparse_results, 
            dense_results,
            k=self.rrf_k
        )
        
        # Optional re-ranking
        if self.reranker_config.get('enabled', False):
            fused_results = self.reranker.rerank(
                query, 
                fused_results,
                top_k=self.reranker_config.get('top_k', 10)
            )
        
        chunks = self._results_to_chunks(fused_results)
        
        return ContextBundle(
            instance_id=instance.instance_id,
            problem_statement=instance.problem_statement,
            chunks=chunks,
            repo_structure=self._get_repo_structure(),
            gathered_at=datetime.utcnow().isoformat(),
            gatherer_type='rag_hybrid',
            token_count=sum(c.token_count for c in chunks),
            metadata={
                'query': query,
                'sparse_weight': self.sparse_weight,
                'dense_weight': self.dense_weight,
                'rrf_k': self.rrf_k
            }
        )
    
    def _reciprocal_rank_fusion(
        self, 
        sparse_results: List[SearchResult],
        dense_results: List[SearchResult],
        k: int = 60
    ) -> List[SearchResult]:
        """
        Fuse results using Reciprocal Rank Fusion.
        
        RRF Score = Σ(1 / (k + rank_i))
        
        Args:
            sparse_results: Results from sparse retrieval
            dense_results: Results from dense retrieval
            k: RRF constant (typically 60)
            
        Returns:
            Fused and re-ranked results
        """
        scores = {}
        
        for rank, result in enumerate(sparse_results, start=1):
            doc_id = result.doc_id
            weight = self.sparse_weight * (1 / (k + rank))
            scores[doc_id] = scores.get(doc_id, 0) + weight
        
        for rank, result in enumerate(dense_results, start=1):
            doc_id = result.doc_id
            weight = self.dense_weight * (1 / (k + rank))
            scores[doc_id] = scores.get(doc_id, 0) + weight
        
        # Sort by fused score
        sorted_ids = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        # Return re-ranked results
        return [self._get_result_by_id(doc_id) for doc_id, _ in sorted_ids]
```

**Configuration Options:**

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `indexing.chunking.strategy` | str | "ast" | Chunking strategy |
| `indexing.embeddings.model` | str | "jinaai/jina-embeddings-v2-base-code" | Embedding model |
| `indexing.bm25.k1` | float | 1.5 | BM25 k1 parameter |
| `indexing.bm25.b` | float | 0.75 | BM25 b parameter |
| `retrieval.sparse_weight` | float | 0.3 | Weight for sparse retrieval |
| `retrieval.dense_weight` | float | 0.7 | Weight for dense retrieval |
| `retrieval.rrf_k` | int | 60 | RRF constant |
| `retrieval.top_k` | int | 20 | Number of results |
| `retrieval.reranker.enabled` | bool | True | Enable re-ranking |
| `retrieval.reranker.model` | str | "cross-encoder/..." | Cross-encoder model |

---

## Factory and Registration

### Creating Gatherers

```python
from swe_bench_framework.context_gatherers import ContextGathererFactory

# Create by name
gatherer = ContextGathererFactory.create('bm25', config)

# Create by type
gatherer = ContextGathererFactory.create_by_type('rag', 'dense', config)
```

### Registering Custom Gatherers

```python
from swe_bench_framework.context_gatherers import ContextGathererFactory
from swe_bench_framework.core.registry import register_gatherer

# Define custom gatherer
class MyCustomGatherer(ContextGatherer):
    def gather_context(self, instance, repo_path):
        # Implementation
        pass
    
    def initialize(self, repo_path):
        pass
    
    def cleanup(self):
        pass

# Register
register_gatherer('my_custom', MyCustomGatherer)

# Use in config
methods:
  - name: "my_custom"
    type: "custom"
    enabled: true
```

---

## Best Practices

### 1. Always Initialize Before Use

```python
gatherer = MyGatherer(config)
gatherer.initialize(repo_path)  # Required!
try:
    context = gatherer.gather_context(instance, repo_path)
finally:
    gatherer.cleanup()  # Always cleanup
```

### 2. Handle Errors Gracefully

```python
from swe_bench_framework.core.exceptions import GatheringError

try:
    context = gatherer.gather_context(instance, repo_path)
except GatheringError as e:
    logger.error(f"Context gathering failed: {e}")
    context = ContextBundle(
        instance_id=instance.instance_id,
        problem_statement=instance.problem_statement,
        chunks=[],
        repo_structure={},
        gathered_at=datetime.utcnow().isoformat(),
        gatherer_type='failed',
        token_count=0,
        metadata={'error': str(e)}
    )
```

### 3. Use Appropriate Context Types

```python
# Tag chunks with appropriate types
chunk = ContextChunk(
    content=function_code,
    source_file="module.py",
    context_type=ContextType.FUNCTION_DEFINITION,  # Not FILE_CONTENT
    start_line=10,
    end_line=50,
    relevance_score=0.95
)
```

### 4. Compute Accurate Token Counts

```python
from swe_bench_framework.utils.token_counter import TokenCounter

counter = TokenCounter('cl100k_base')
token_count = counter.count(chunk.content)
```
