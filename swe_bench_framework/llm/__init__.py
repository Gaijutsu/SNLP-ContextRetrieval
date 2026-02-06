"""
LLM integration for the SWE-bench comparison framework.

This module provides LLM clients for different providers (OpenAI, Anthropic, etc.)
and token counting utilities.
"""

from .base import LLMClient
from .token_counter import TokenCounter

# Optional imports - these may fail if dependencies are not installed
try:
    from .openai_client import OpenAIClient
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

__all__ = [
    "LLMClient",
    "TokenCounter",
]

if OPENAI_AVAILABLE:
    __all__.append("OpenAIClient")


def create_llm_client(config: dict) -> LLMClient:
    """
    Create an LLM client from configuration.
    
    Args:
        config: Configuration dictionary with 'provider' and other settings
        
    Returns:
        LLM client instance
        
    Raises:
        ValueError: If the provider is not supported
        ImportError: If required dependencies are not installed
    """
    provider = config.get('provider', 'openai')
    
    if provider == 'openai':
        if not OPENAI_AVAILABLE:
            raise ImportError(
                "OpenAI client requires the 'openai' package. "
                "Install it with: pip install openai"
            )
        return OpenAIClient(config)
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")
