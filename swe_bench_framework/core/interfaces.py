"""
Abstract base classes and interfaces for the SWE-bench comparison framework.

This module defines the core interfaces that all components must implement,
ensuring a consistent API across different methods.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .data_models import (
        SWEInstance,
        ContextBundle,
        ContextChunk,
        PatchResult,
        EvaluationResult,
        SearchResult,
    )


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
    
    @abstractmethod
    def gather_context(
        self,
        instance: 'SWEInstance',
        repo_path: str
    ) -> 'ContextBundle':
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
    
    @abstractmethod
    def cleanup(self) -> None:
        """
        Cleanup resources.
        
        This method should be called after all instances have been
        processed to release any resources held by the gatherer.
        """
        pass
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Return statistics about context gathering.
        
        Returns:
            Dictionary with statistics (implementation-specific)
        """
        return {}
    
    def is_initialized(self) -> bool:
        """Check if the gatherer has been initialized."""
        return self._initialized


class PatchGenerator(ABC):
    """
    Abstract base class for all patch generation strategies.
    
    This interface defines how patches are generated from gathered context.
    Implementations can use different strategies like direct generation,
    iterative refinement, or structured edit scripts.
    
    Example:
        >>> generator = MyPatchGenerator(llm_config, config)
        >>> result = generator.generate_patch(context_bundle, instance)
        >>> if result.success:
        ...     print(f"Generated patch: {result.patch_content}")
    """
    
    def __init__(self, llm_config: Dict[str, Any], config: Dict[str, Any]):
        """
        Initialize the patch generator.
        
        Args:
            llm_config: Configuration for the LLM (model, temperature, etc.)
            config: Configuration for the generation strategy
        """
        self.llm_config = llm_config
        self.config = config
        self.name = self.__class__.__name__
    
    @abstractmethod
    def generate_patch(
        self,
        context_bundle: 'ContextBundle',
        instance: 'SWEInstance'
    ) -> 'PatchResult':
        """
        Generate a patch given context and problem statement.
        
        This is the main method that all patch generation strategies
        must implement. It should return a PatchResult containing
        the generated patch or error information.
        
        Args:
            context_bundle: Gathered context from a ContextGatherer
            instance: The SWE-bench instance
            
        Returns:
            PatchResult containing the generated patch or error
            
        Raises:
            GenerationError: If patch generation fails
        """
        pass
    
    @abstractmethod
    def validate_patch(self, patch_content: str, repo_path: str) -> bool:
        """
        Validate that a patch is syntactically correct.
        
        Args:
            patch_content: The patch content to validate
            repo_path: Path to the repository
            
        Returns:
            True if the patch is valid, False otherwise
        """
        pass
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Return statistics about patch generation.
        
        Returns:
            Dictionary with statistics (implementation-specific)
        """
        return {}


class Evaluator(ABC):
    """
    Abstract base class for evaluation strategies.
    
    This interface defines how patches are evaluated against the
    SWE-bench test suite and how metrics are computed.
    
    Example:
        >>> evaluator = MyEvaluator(config)
        >>> result = evaluator.evaluate(patch_result, instance, repo_path)
        >>> print(f"Resolved: {result.resolved}")
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the evaluator.
        
        Args:
            config: Configuration dictionary for the evaluator
        """
        self.config = config
        self.name = self.__class__.__name__
    
    @abstractmethod
    def evaluate(
        self,
        patch_result: 'PatchResult',
        instance: 'SWEInstance',
        repo_path: str
    ) -> 'EvaluationResult':
        """
        Evaluate a generated patch.
        
        This is the main method that all evaluators must implement.
        It should apply the patch, run tests, and compute all metrics.
        
        Args:
            patch_result: The result from patch generation
            instance: The SWE-bench instance
            repo_path: Path to the repository
            
        Returns:
            EvaluationResult with all metrics
            
        Raises:
            EvaluationError: If evaluation fails
        """
        pass
    
    @abstractmethod
    def compute_localization_accuracy(
        self,
        context_bundle: 'ContextBundle',
        gold_files: List[str],
        gold_functions: List[str]
    ) -> Dict[str, float]:
        """
        Compute localization accuracy metrics.
        
        This method computes metrics like recall@k for file and
        function localization accuracy.
        
        Args:
            context_bundle: The gathered context
            gold_files: Ground truth list of modified files
            gold_functions: Ground truth list of modified functions
            
        Returns:
            Dictionary with localization metrics (e.g., recall@1, recall@5)
        """
        pass
    
    def compute_codebleu(
        self,
        generated_patch: str,
        gold_patch: str
    ) -> float:
        """
        Compute CodeBLEU score between generated and gold patch.
        
        Args:
            generated_patch: The generated patch
            gold_patch: The gold patch
            
        Returns:
            CodeBLEU score (0.0-1.0)
        """
        # Default implementation - subclasses can override
        # This is a placeholder; real implementation would use
        # the CodeBLEU metric computation
        return 0.0


class RepositoryIndexer(ABC):
    """
    Abstract base class for repository indexing strategies.
    
    This interface is used by RAG methods to build and query
    indexes of repository code.
    
    Example:
        >>> indexer = BM25Indexer(config)
        >>> indexer.build_index(repo_path, output_path)
        >>> indexer.load_index(output_path)
        >>> results = indexer.search("query", top_k=10)
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the repository indexer.
        
        Args:
            config: Configuration dictionary for the indexer
        """
        self.config = config
        self.name = self.__class__.__name__
        self.index_path: Optional[str] = None
        self._built = False
    
    @abstractmethod
    def build_index(self, repo_path: str, output_path: str) -> None:
        """
        Build index from repository.
        
        This method should create an index from the repository
        and save it to the specified output path.
        
        Args:
            repo_path: Path to the repository
            output_path: Path where the index should be saved
            
        Raises:
            RepositoryError: If index building fails
        """
        pass
    
    @abstractmethod
    def load_index(self, index_path: str) -> None:
        """
        Load existing index.
        
        Args:
            index_path: Path to the saved index
            
        Raises:
            RepositoryError: If index loading fails
        """
        pass
    
    @abstractmethod
    def search(self, query: str, top_k: int = 10) -> List['SearchResult']:
        """
        Search the index.
        
        Args:
            query: Search query string
            top_k: Number of results to return
            
        Returns:
            List of search results
            
        Raises:
            RepositoryError: If search fails
        """
        pass
    
    def is_built(self) -> bool:
        """Check if the index has been built."""
        return self._built
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Return statistics about the index.
        
        Returns:
            Dictionary with statistics (implementation-specific)
        """
        return {}


class Chunker(ABC):
    """
    Abstract base class for code chunking strategies.
    
    This interface is used by RAG methods to split repository
    code into chunks for indexing.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the chunker.
        
        Args:
            config: Configuration dictionary for the chunker
        """
        self.config = config
        self.name = self.__class__.__name__
    
    @abstractmethod
    def chunk_file(self, file_path: str, content: str) -> List[Dict[str, Any]]:
        """
        Chunk a single file.
        
        Args:
            file_path: Path to the file
            content: File content
            
        Returns:
            List of chunks, each with 'content', 'start_line', 'end_line'
        """
        pass
    
    @abstractmethod
    def chunk_repository(self, repo_path: str) -> List[Dict[str, Any]]:
        """
        Chunk an entire repository.
        
        Args:
            repo_path: Path to the repository
            
        Returns:
            List of chunks from all files
        """
        pass


class Retriever(ABC):
    """
    Abstract base class for retrieval strategies.
    
    This interface is used by RAG methods to retrieve relevant
    code chunks from an index.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the retriever.
        
        Args:
            config: Configuration dictionary for the retriever
        """
        self.config = config
        self.name = self.__class__.__name__
    
    @abstractmethod
    def retrieve(
        self,
        query: str,
        top_k: int = 10
    ) -> List['SearchResult']:
        """
        Retrieve relevant chunks for a query.
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of search results
        """
        pass
    
    @abstractmethod
    def batch_retrieve(
        self,
        queries: List[str],
        top_k: int = 10
    ) -> List[List['SearchResult']]:
        """
        Retrieve results for multiple queries.
        
        Args:
            queries: List of search queries
            top_k: Number of results per query
            
        Returns:
            List of result lists, one per query
        """
        pass


class Reranker(ABC):
    """
    Abstract base class for reranking strategies.
    
    This interface is used to rerank retrieved results for better
    relevance.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the reranker.
        
        Args:
            config: Configuration dictionary for the reranker
        """
        self.config = config
        self.name = self.__class__.__name__
    
    @abstractmethod
    def rerank(
        self,
        query: str,
        results: List['SearchResult'],
        top_k: int = 10
    ) -> List['SearchResult']:
        """
        Rerank search results.
        
        Args:
            query: Original search query
            results: Results to rerank
            top_k: Number of results to return
            
        Returns:
            Reranked results
        """
        pass


class LLMClient(ABC):
    """
    Abstract base class for LLM clients.
    
    This interface provides a unified way to interact with different
    LLM providers (OpenAI, Anthropic, local models, etc.).
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the LLM client.
        
        Args:
            config: Configuration dictionary with API keys, model, etc.
        """
        self.config = config
        self.name = self.__class__.__name__
        self.model = config.get('model', 'gpt-5-mini')
        self.temperature = config.get('temperature', 0.0)
        self.max_tokens = config.get('max_tokens', 4096)
    
    @abstractmethod
    def generate(
        self,
        prompt: str,
        **kwargs
    ) -> str:
        """
        Generate text from a prompt.
        
        Args:
            prompt: Input prompt
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text
        """
        pass
    
    @abstractmethod
    def chat(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> str:
        """
        Generate response from a conversation.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            **kwargs: Additional generation parameters
            
        Returns:
            Generated response
        """
        pass
    
    @abstractmethod
    def get_token_count(self, text: str) -> int:
        """
        Get the token count for a text.
        
        Args:
            text: Text to count tokens for
            
        Returns:
            Number of tokens
        """
        pass
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Return statistics about API usage.
        
        Returns:
            Dictionary with usage statistics
        """
        return {}


class Metric(ABC):
    """
    Abstract base class for evaluation metrics.
    
    This interface defines how metrics are computed from evaluation results.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the metric.
        
        Args:
            config: Configuration dictionary for the metric
        """
        self.config = config
        self.name = self.__class__.__name__
    
    @abstractmethod
    def compute(
        self,
        patch_result: 'PatchResult',
        instance: 'SWEInstance',
        evaluation_result: 'EvaluationResult'
    ) -> Dict[str, float]:
        """
        Compute the metric.
        
        Args:
            patch_result: The patch generation result
            instance: The SWE-bench instance
            evaluation_result: The evaluation result
            
        Returns:
            Dictionary with metric values
        """
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Get the name of the metric."""
        pass
    
    def aggregate(self, results: List[Dict[str, float]]) -> Dict[str, float]:
        """
        Aggregate metric values across multiple instances.
        
        Args:
            results: List of metric dictionaries
            
        Returns:
            Aggregated metric values
        """
        if not results:
            return {}
        
        # Default implementation: compute mean for each key
        aggregated = {}
        keys = results[0].keys()
        
        for key in keys:
            values = [r[key] for r in results if key in r]
            if values:
                aggregated[f'{key}_mean'] = sum(values) / len(values)
                aggregated[f'{key}_count'] = len(values)
        
        return aggregated
