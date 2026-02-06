"""
Default configurations for the SWE-bench comparison framework.

This module provides default configuration presets for common scenarios.
"""

from typing import Dict, Any

from .schema import (
    ExperimentConfig,
    DatasetConfig,
    LLMConfig,
    MethodConfig,
    PatchGenerationConfig,
    EvaluationConfig,
    LoggingConfig,
    RateLimitConfig,
    SandboxConfig,
    ExperimentTrackingConfig,
)


def get_default_config(name: str = "default_experiment") -> ExperimentConfig:
    """
    Get a default experiment configuration.
    
    This configuration includes:
    - SWE-bench-lite dataset
    - GPT-5 Mini as the LLM
    - Direct patch generation strategy
    - Standard evaluation metrics
    
    Args:
        name: Experiment name
        
    Returns:
        Default experiment configuration
    """
    return ExperimentConfig(
        name=name,
        description="Default SWE-bench comparison experiment",
        output_dir="./results/default",
        random_seed=42,
        dataset=DatasetConfig(
            name="swe-bench-lite",
            split="test",
        ),
        llm=LLMConfig(
            provider="openai",
            model="gpt-5-mini",
            temperature=0.0,
            max_tokens=4096,
            rate_limit=RateLimitConfig(
                requests_per_minute=60,
                tokens_per_minute=150000,
            ),
        ),
        methods=[
            MethodConfig(
                name="hybrid_rag",
                type="rag",
                enabled=True,
                config={
                    "indexing": {
                        "chunking_strategy": "ast",
                        "max_chunk_size": 500,
                        "embedding_model": "jina-embeddings-v2-base-code",
                    },
                    "retrieval": {
                        "sparse_weight": 0.3,
                        "dense_weight": 0.7,
                        "top_k": 20,
                        "reranker_enabled": True,
                    },
                },
            ),
        ],
        patch_generation=PatchGenerationConfig(
            strategy="direct",
            max_attempts=3,
            syntax_check=True,
            max_context_tokens=8000,
        ),
        evaluation=EvaluationConfig(
            sandbox=SandboxConfig(
                type="docker",
                timeout=300.0,
                memory_limit="4g",
            ),
            metrics=[
                "resolution_rate",
                "localization_accuracy",
                "codebleu",
                "token_usage",
            ],
            localization_k_values=[1, 3, 5, 10],
        ),
        logging=LoggingConfig(
            level="INFO",
            format="structured",
            stdout=True,
            experiment_tracking=ExperimentTrackingConfig(
                enabled=True,
                backend="mlflow",
            ),
        ),
    )


def get_minimal_config(name: str = "minimal_experiment") -> ExperimentConfig:
    """
    Get a minimal experiment configuration for quick testing.
    
    This configuration is suitable for testing and development,
    with minimal resource requirements.
    
    Args:
        name: Experiment name
        
    Returns:
        Minimal experiment configuration
    """
    return ExperimentConfig(
        name=name,
        description="Minimal SWE-bench experiment for testing",
        output_dir="./results/minimal",
        random_seed=42,
        dataset=DatasetConfig(
            name="swe-bench-lite",
            split="test",
            max_instances=10,  # Only process 10 instances
        ),
        llm=LLMConfig(
            provider="openai",
            model="gpt-5-mini",
            temperature=0.0,
            max_tokens=2048,
        ),
        methods=[
            MethodConfig(
                name="simple_rag",
                type="rag",
                enabled=True,
                config={
                    "retrieval": {
                        "top_k": 10,
                    },
                },
            ),
        ],
        patch_generation=PatchGenerationConfig(
            strategy="direct",
            max_attempts=1,
            syntax_check=True,
            max_context_tokens=4000,
        ),
        evaluation=EvaluationConfig(
            sandbox=SandboxConfig(
                type="docker",
                timeout=120.0,
                memory_limit="2g",
            ),
            metrics=["resolution_rate"],
            localization_k_values=[1, 5],
        ),
        logging=LoggingConfig(
            level="DEBUG",
            format="text",
            stdout=True,
            experiment_tracking=ExperimentTrackingConfig(
                enabled=False,
            ),
        ),
    )


def get_agentic_comparison_config(name: str = "agentic_comparison") -> ExperimentConfig:
    """
    Get a configuration for comparing agentic methods.
    
    This configuration includes multiple agentic methods for comparison.
    
    Args:
        name: Experiment name
        
    Returns:
        Configuration for agentic comparison
    """
    config = get_default_config(name)
    config.description = "Comparison of agentic methods on SWE-bench"
    config.methods = [
        MethodConfig(
            name="autocoderover",
            type="agentic",
            enabled=True,
            config={
                "max_iterations": 50,
                "tools": ["search_class", "search_method", "view_file", "grep"],
                "enable_linter": True,
                "enable_test_runner": True,
            },
        ),
        MethodConfig(
            name="swe_agent",
            type="agentic",
            enabled=True,
            config={
                "max_iterations": 100,
                "enable_linter": True,
                "enable_test_runner": True,
                "max_thought_length": 500,
            },
        ),
    ]
    return config


def get_rag_comparison_config(name: str = "rag_comparison") -> ExperimentConfig:
    """
    Get a configuration for comparing RAG methods.
    
    This configuration includes multiple RAG methods for comparison.
    
    Args:
        name: Experiment name
        
    Returns:
        Configuration for RAG comparison
    """
    config = get_default_config(name)
    config.description = "Comparison of RAG methods on SWE-bench"
    config.methods = [
        MethodConfig(
            name="bm25_rag",
            type="rag",
            enabled=True,
            config={
                "indexing": {
                    "chunking_strategy": "sliding",
                    "bm25_k1": 1.5,
                    "bm25_b": 0.75,
                },
                "retrieval": {
                    "top_k": 20,
                    "reranker_enabled": False,
                },
            },
        ),
        MethodConfig(
            name="dense_rag",
            type="rag",
            enabled=True,
            config={
                "indexing": {
                    "chunking_strategy": "ast",
                    "embedding_model": "jina-embeddings-v2-base-code",
                },
                "retrieval": {
                    "top_k": 20,
                    "reranker_enabled": False,
                },
            },
        ),
        MethodConfig(
            name="hybrid_rag",
            type="rag",
            enabled=True,
            config={
                "indexing": {
                    "chunking_strategy": "ast",
                    "embedding_model": "jina-embeddings-v2-base-code",
                },
                "retrieval": {
                    "sparse_weight": 0.3,
                    "dense_weight": 0.7,
                    "top_k": 20,
                    "reranker_enabled": True,
                },
            },
        ),
    ]
    return config


def get_full_comparison_config(name: str = "full_comparison") -> ExperimentConfig:
    """
    Get a configuration for full comparison (agentic + RAG).
    
    This configuration includes both agentic and RAG methods
    for comprehensive comparison.
    
    Args:
        name: Experiment name
        
    Returns:
        Configuration for full comparison
    """
    config = get_default_config(name)
    config.description = "Full comparison of agentic and RAG methods"
    config.output_dir = "./results/full_comparison"
    config.methods = [
        # Agentic methods
        MethodConfig(
            name="autocoderover",
            type="agentic",
            enabled=True,
            config={
                "max_iterations": 50,
                "tools": ["search_class", "search_method", "view_file", "grep"],
            },
        ),
        MethodConfig(
            name="swe_agent",
            type="agentic",
            enabled=True,
            config={
                "max_iterations": 100,
                "enable_linter": True,
            },
        ),
        # RAG methods
        MethodConfig(
            name="bm25_rag",
            type="rag",
            enabled=True,
            config={
                "retrieval": {"top_k": 20, "reranker_enabled": False},
            },
        ),
        MethodConfig(
            name="hybrid_rag",
            type="rag",
            enabled=True,
            config={
                "retrieval": {
                    "sparse_weight": 0.3,
                    "dense_weight": 0.7,
                    "top_k": 20,
                    "reranker_enabled": True,
                },
            },
        ),
    ]
    return config


def get_preset_config(preset_name: str) -> ExperimentConfig:
    """
    Get a preset configuration by name.
    
    Args:
        preset_name: Name of the preset (default, minimal, agentic, rag, full)
        
    Returns:
        Preset experiment configuration
        
    Raises:
        ValueError: If the preset name is not recognized
    """
    presets = {
        "default": get_default_config,
        "minimal": get_minimal_config,
        "agentic": get_agentic_comparison_config,
        "rag": get_rag_comparison_config,
        "full": get_full_comparison_config,
    }
    
    if preset_name not in presets:
        available = ", ".join(presets.keys())
        raise ValueError(
            f"Unknown preset: {preset_name}. Available: {available}"
        )
    
    return presets[preset_name]()


def list_presets() -> Dict[str, str]:
    """
    List available configuration presets.
    
    Returns:
        Dictionary mapping preset names to descriptions
    """
    return {
        "default": "Default configuration with hybrid RAG",
        "minimal": "Minimal configuration for testing (10 instances)",
        "agentic": "Comparison of agentic methods",
        "rag": "Comparison of RAG methods (BM25, Dense, Hybrid)",
        "full": "Full comparison of agentic and RAG methods",
    }
