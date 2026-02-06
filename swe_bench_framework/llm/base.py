"""
Base LLM client for the SWE-bench comparison framework.

This module defines the abstract base class for LLM clients.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class LLMClient(ABC):
    """
    Abstract base class for LLM clients.
    
    This interface provides a unified way to interact with different
    LLM providers (OpenAI, Anthropic, local models, etc.).
    
    Example:
        >>> client = OpenAIClient({'model': 'gpt-5-mini', 'api_key': '...'})
        >>> response = client.generate("What is Python?")
        >>> print(response)
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the LLM client.
        
        Args:
            config: Configuration dictionary with API keys, model, etc.
                Required keys depend on the provider.
        """
        self.config = config
        self.name = self.__class__.__name__
        self.model = config.get('model', 'gpt-5-mini')
        self.temperature = config.get('temperature', 0.0)
        self.max_tokens = config.get('max_tokens', 4096)
        self.top_p = config.get('top_p', 1.0)
        
        # Usage tracking
        self._total_calls = 0
        self._total_tokens = 0
        self._prompt_tokens = 0
        self._completion_tokens = 0
    
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
            **kwargs: Additional generation parameters that override
                     the default configuration
            
        Returns:
            Generated text
            
        Raises:
            LLMError: If generation fails
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
            messages: List of message dictionaries with 'role' and 'content'.
                Roles can be: 'system', 'user', 'assistant'
            **kwargs: Additional generation parameters
            
        Returns:
            Generated response
            
        Raises:
            LLMError: If generation fails
            
        Example:
            >>> messages = [
            ...     {'role': 'system', 'content': 'You are a helpful assistant.'},
            ...     {'role': 'user', 'content': 'What is Python?'}
            ... ]
            >>> response = client.chat(messages)
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
    
    def count_tokens_in_messages(self, messages: List[Dict[str, str]]) -> int:
        """
        Count tokens in a list of messages.
        
        Args:
            messages: List of message dictionaries
            
        Returns:
            Total token count
        """
        total = 0
        for message in messages:
            content = message.get('content', '')
            total += self.get_token_count(content)
        return total
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Return statistics about API usage.
        
        Returns:
            Dictionary with usage statistics
        """
        return {
            'total_calls': self._total_calls,
            'total_tokens': self._total_tokens,
            'prompt_tokens': self._prompt_tokens,
            'completion_tokens': self._completion_tokens,
            'model': self.model,
        }
    
    def reset_stats(self) -> None:
        """Reset usage statistics."""
        self._total_calls = 0
        self._total_tokens = 0
        self._prompt_tokens = 0
        self._completion_tokens = 0
    
    def _update_stats(
        self,
        prompt_tokens: int = 0,
        completion_tokens: int = 0
    ) -> None:
        """
        Update usage statistics.
        
        Args:
            prompt_tokens: Number of prompt tokens
            completion_tokens: Number of completion tokens
        """
        self._total_calls += 1
        self._prompt_tokens += prompt_tokens
        self._completion_tokens += completion_tokens
        self._total_tokens += prompt_tokens + completion_tokens
    
    def _merge_kwargs(self, **kwargs) -> Dict[str, Any]:
        """
        Merge provided kwargs with default configuration.
        
        Args:
            **kwargs: Override parameters
            
        Returns:
            Merged configuration dictionary
        """
        merged = {
            'model': self.model,
            'temperature': self.temperature,
            'max_tokens': self.max_tokens,
            'top_p': self.top_p,
        }
        merged.update(self.config.get('additional_params', {}))
        merged.update(kwargs)
        return merged


class MockLLMClient(LLMClient):
    """
    Mock LLM client for testing.
    
    This client returns predetermined responses and is useful for
    testing without making actual API calls.
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        responses: Optional[List[str]] = None
    ):
        """
        Initialize the mock client.
        
        Args:
            config: Configuration dictionary
            responses: List of responses to return in sequence
        """
        super().__init__(config)
        self.responses = responses or ["Mock response"]
        self._response_index = 0
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate a mock response."""
        response = self.responses[self._response_index % len(self.responses)]
        self._response_index += 1
        
        # Simulate token usage
        prompt_tokens = len(prompt.split())
        completion_tokens = len(response.split())
        self._update_stats(prompt_tokens, completion_tokens)
        
        return response
    
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Generate a mock chat response."""
        return self.generate("", **kwargs)
    
    def get_token_count(self, text: str) -> int:
        """Estimate token count."""
        return len(text.split())
