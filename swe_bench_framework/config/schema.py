"""
Configuration schema for the SWE-bench comparison framework.

This module defines all configuration dataclasses used to configure
experiments, methods, and components.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


@dataclass
class RateLimitConfig:
    """
    Rate limiting configuration for LLM APIs.
    
    Attributes:
        requests_per_minute: Maximum requests per minute
        tokens_per_minute: Maximum tokens per minute
        max_retries: Maximum number of retries on rate limit
        retry_delay: Initial delay between retries (seconds)
        exponential_backoff: Whether to use exponential backoff
    """
    requests_per_minute: int = 60
    tokens_per_minute: int = 150000
    max_retries: int = 3
    retry_delay: float = 1.0
    exponential_backoff: bool = True


@dataclass
class LLMConfig:
    """
    LLM configuration for the framework.
    
    Attributes:
        provider: LLM provider (openai, anthropic, local, azure)
        model: Model name (e.g., gpt-5-mini)
        temperature: Sampling temperature (0.0-2.0)
        max_tokens: Maximum tokens to generate
        top_p: Nucleus sampling parameter
        api_key: API key (can use ${ENV_VAR} syntax)
        api_base: Custom API base URL
        rate_limit: Rate limiting configuration
        additional_params: Additional provider-specific parameters
    """
    provider: str = "openai"
    model: str = "gpt-5-mini"
    temperature: float = 0.0
    max_tokens: int = 4096
    top_p: float = 1.0
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    rate_limit: RateLimitConfig = field(default_factory=RateLimitConfig)
    additional_params: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError(
                f"temperature must be between 0.0 and 2.0, got {self.temperature}"
            )
        if self.max_tokens < 1:
            raise ValueError(f"max_tokens must be >= 1, got {self.max_tokens}")


@dataclass
class DatasetConfig:
    """
    Dataset configuration for experiments.
    
    Attributes:
        name: Dataset name (e.g., swe-bench-lite, swe-bench-full)
        split: Dataset split (test, train, dev)
        filter_repos: List of repositories to include (None = all)
        max_instances: Maximum number of instances to process (None = all)
        instance_ids: Specific instance IDs to process (None = all)
        cache_dir: Directory for caching dataset
    """
    name: str = "swe-bench-lite"
    split: str = "test"
    filter_repos: Optional[List[str]] = None
    max_instances: Optional[int] = None
    instance_ids: Optional[List[str]] = None
    cache_dir: Optional[str] = None


@dataclass
class AgenticConfig:
    """
    Configuration for agentic context gathering methods.
    
    Attributes:
        max_iterations: Maximum number of agent iterations
        tools: List of tools to enable
        enable_linter: Whether to enable linter feedback
        enable_test_runner: Whether to enable test execution
        max_thought_length: Maximum length of agent thoughts
        action_timeout: Timeout for individual actions (seconds)
    """
    max_iterations: int = 50
    tools: List[str] = field(default_factory=lambda: [
        "search_class",
        "search_method",
        "view_file",
        "grep",
        "run_test",
    ])
    enable_linter: bool = True
    enable_test_runner: bool = True
    max_thought_length: int = 500
    action_timeout: float = 30.0


@dataclass
class RAGConfig:
    """
    Configuration for RAG context gathering methods.
    
    Attributes:
        indexing: Indexing configuration
        retrieval: Retrieval configuration
        context_assembly: Context assembly configuration
    """
    
    @dataclass
    class IndexingConfig:
        """Indexing configuration."""
        chunking_strategy: str = "ast"  # ast, sliding, or hybrid
        max_chunk_size: int = 500
        overlap: int = 50
        embedding_model: str = "jina-embeddings-v2-base-code"
        embedding_batch_size: int = 32
        bm25_k1: float = 1.5
        bm25_b: float = 0.75
    
    @dataclass
    class RetrievalConfig:
        """Retrieval configuration."""
        sparse_weight: float = 0.3
        dense_weight: float = 0.7
        rrf_k: int = 60
        top_k: int = 20
        reranker_enabled: bool = True
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
        reranker_top_k: int = 10
    
    @dataclass
    class ContextAssemblyConfig:
        """Context assembly configuration."""
        max_tokens: int = 8000
        priority: List[str] = field(default_factory=lambda: [
            "error_context",
            "function_definition",
            "class_definition",
            "test_context",
        ])
    
    indexing: IndexingConfig = field(default_factory=IndexingConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    context_assembly: ContextAssemblyConfig = field(
        default_factory=ContextAssemblyConfig
    )


@dataclass
class MethodConfig:
    """
    Configuration for a single method in the comparison.
    
    Attributes:
        name: Method name (e.g., autocoderover, hybrid_rag)
        type: Method type (agentic or rag)
        enabled: Whether this method is enabled
        config: Method-specific configuration
    """
    name: str
    type: str  # "agentic" or "rag"
    enabled: bool = True
    config: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.type not in ("agentic", "rag"):
            raise ValueError(f"type must be 'agentic' or 'rag', got {self.type}")


@dataclass
class PatchGenerationConfig:
    """
    Configuration for patch generation.
    
    Attributes:
        strategy: Generation strategy (direct, iterative, edit_script)
        max_attempts: Maximum number of generation attempts
        syntax_check: Whether to check patch syntax
        test_before_submit: Whether to test before submitting
        prompt_template: Custom prompt template name
        max_context_tokens: Maximum context tokens for prompt
    """
    strategy: str = "direct"
    max_attempts: int = 3
    syntax_check: bool = True
    test_before_submit: bool = False
    prompt_template: Optional[str] = None
    max_context_tokens: int = 8000
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        valid_strategies = ("direct", "iterative", "edit_script")
        if self.strategy not in valid_strategies:
            raise ValueError(
                f"strategy must be one of {valid_strategies}, got {self.strategy}"
            )


@dataclass
class SandboxConfig:
    """
    Configuration for the evaluation sandbox.
    
    Attributes:
        type: Sandbox type (docker, local)
        image: Docker image name
        timeout: Test execution timeout (seconds)
        memory_limit: Memory limit for container
        cpu_limit: CPU limit for container
        network_disabled: Whether to disable network access
    """
    type: str = "docker"
    image: str = "swe-bench/sandbox:latest"
    timeout: float = 300.0
    memory_limit: str = "4g"
    cpu_limit: Optional[str] = None
    network_disabled: bool = True


@dataclass
class EvaluationConfig:
    """
    Configuration for evaluation.
    
    Attributes:
        sandbox: Sandbox configuration
        metrics: List of metrics to compute
        localization_k_values: K values for localization metrics
        compute_codebleu: Whether to compute CodeBLEU
    """
    sandbox: SandboxConfig = field(default_factory=SandboxConfig)
    metrics: List[str] = field(default_factory=lambda: [
        "resolution_rate",
        "localization_accuracy",
        "codebleu",
        "token_usage",
    ])
    localization_k_values: List[int] = field(default_factory=lambda: [1, 3, 5, 10])
    compute_codebleu: bool = True


@dataclass
class ExperimentTrackingConfig:
    """
    Configuration for experiment tracking.
    
    Attributes:
        enabled: Whether tracking is enabled
        backend: Tracking backend (mlflow, wandb, tensorboard)
        uri: Tracking server URI
        experiment_name: Name of the experiment
    """
    enabled: bool = True
    backend: str = "mlflow"
    uri: Optional[str] = None
    experiment_name: Optional[str] = None


@dataclass
class LoggingConfig:
    """
    Configuration for logging.
    
    Attributes:
        level: Log level (DEBUG, INFO, WARNING, ERROR)
        format: Log format (structured, text)
        file_path: Path to log file (None = no file logging)
        stdout: Whether to log to stdout
        experiment_tracking: Experiment tracking configuration
    """
    level: str = "INFO"
    format: str = "structured"
    file_path: Optional[str] = None
    stdout: bool = True
    experiment_tracking: ExperimentTrackingConfig = field(
        default_factory=ExperimentTrackingConfig
    )
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        valid_levels = ("DEBUG", "INFO", "WARNING", "ERROR")
        if self.level not in valid_levels:
            raise ValueError(
                f"level must be one of {valid_levels}, got {self.level}"
            )
        valid_formats = ("structured", "text")
        if self.format not in valid_formats:
            raise ValueError(
                f"format must be one of {valid_formats}, got {self.format}"
            )


@dataclass
class ExperimentConfig:
    """
    Main configuration for an experiment.
    
    This is the top-level configuration that contains all other
    configuration sections.
    
    Attributes:
        name: Experiment name
        description: Experiment description
        output_dir: Output directory for results
        random_seed: Random seed for reproducibility
        dataset: Dataset configuration
        llm: LLM configuration
        methods: List of method configurations
        patch_generation: Patch generation configuration
        evaluation: Evaluation configuration
        logging: Logging configuration
        tracking: Experiment tracking configuration (top-level)
    """
    name: str
    description: str = ""
    output_dir: str = "./results"
    random_seed: int = 42
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    methods: List[MethodConfig] = field(default_factory=list)
    patch_generation: PatchGenerationConfig = field(
        default_factory=PatchGenerationConfig
    )
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    tracking: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if not self.name:
            raise ValueError("Experiment name cannot be empty")
    
    def get_enabled_methods(self) -> List[MethodConfig]:
        """Get list of enabled methods."""
        return [m for m in self.methods if m.enabled]
    
    def get_method_by_name(self, name: str) -> Optional[MethodConfig]:
        """Get a method configuration by name."""
        for method in self.methods:
            if method.name == name:
                return method
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'name': self.name,
            'description': self.description,
            'output_dir': self.output_dir,
            'random_seed': self.random_seed,
            'dataset': self._dataclass_to_dict(self.dataset),
            'llm': self._dataclass_to_dict(self.llm),
            'methods': [self._dataclass_to_dict(m) for m in self.methods],
            'patch_generation': self._dataclass_to_dict(self.patch_generation),
            'evaluation': self._dataclass_to_dict(self.evaluation),
            'logging': self._dataclass_to_dict(self.logging),
        }
    
    @staticmethod
    def _dataclass_to_dict(obj: Any) -> Dict[str, Any]:
        """Convert a dataclass to dictionary."""
        if hasattr(obj, '__dataclass_fields__'):
            result = {}
            for key in obj.__dataclass_fields__:
                value = getattr(obj, key)
                if hasattr(value, '__dataclass_fields__'):
                    result[key] = ExperimentConfig._dataclass_to_dict(value)
                elif isinstance(value, list):
                    result[key] = [
                        ExperimentConfig._dataclass_to_dict(v) 
                        if hasattr(v, '__dataclass_fields__') else v
                        for v in value
                    ]
                else:
                    result[key] = value
            return result
        return obj
