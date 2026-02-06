"""
OpenAI LLM client for the SWE-bench comparison framework.

This module provides an LLM client implementation for OpenAI's API.
"""

import time
from typing import Any, Dict, List, Optional
import logging

try:
    import openai
    from openai import OpenAI, RateLimitError, APIError, APITimeoutError
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

from .base import LLMClient
from ..core.exceptions import LLMError
from ..utils.token_utils import get_tokenizer_for_model, TokenCounter

logger = logging.getLogger(__name__)


class OpenAIClient(LLMClient):
    """
    OpenAI LLM client with rate limiting support.
    
    This client provides access to OpenAI's models (GPT-5, GPT-4o, etc.)
    with built-in rate limiting and retry logic.
    
    Example:
        >>> config = {
        ...     'model': 'gpt-5-mini',
        ...     'api_key': 'your-api-key',
        ...     'temperature': 0.0,
        ... }
        >>> client = OpenAIClient(config)
        >>> response = client.generate("What is Python?")
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the OpenAI client.
        
        Args:
            config: Configuration dictionary with:
                - model: Model name (default: gpt-5-mini)
                - api_key: OpenAI API key
                - api_base: Custom API base URL (optional)
                - temperature: Sampling temperature (default: 0.0)
                - max_tokens: Maximum tokens (default: 4096)
                - top_p: Nucleus sampling (default: 1.0)
                - rate_limit: Rate limiting configuration
                
        Raises:
            ImportError: If the openai package is not installed
        """
        if not OPENAI_AVAILABLE:
            raise ImportError(
                "OpenAI client requires the 'openai' package. "
                "Install it with: pip install openai"
            )
        
        super().__init__(config)
        
        # Initialize OpenAI client
        api_key = config.get('api_key')
        api_base = config.get('api_base')
        
        client_kwargs = {}
        if api_key:
            client_kwargs['api_key'] = api_key
        if api_base:
            client_kwargs['base_url'] = api_base
        
        self.client = OpenAI(**client_kwargs)
        
        # Rate limiting configuration
        rate_limit_config = config.get('rate_limit', {})
        self.requests_per_minute = rate_limit_config.get('requests_per_minute', 60)
        self.tokens_per_minute = rate_limit_config.get('tokens_per_minute', 150000)
        self.max_retries = rate_limit_config.get('max_retries', 3)
        self.retry_delay = rate_limit_config.get('retry_delay', 1.0)
        self.exponential_backoff = rate_limit_config.get('exponential_backoff', True)
        
        # Rate limiting state
        self._request_times: List[float] = []
        self._token_usage: List[tuple] = []  # (timestamp, tokens)
        
        # Initialize token counter
        encoding_name = get_tokenizer_for_model(self.model)
        self.token_counter = TokenCounter(encoding_name)
    
    def _wait_for_rate_limit(self, estimated_tokens: int = 0) -> None:
        """
        Wait if necessary to respect rate limits.
        
        Args:
            estimated_tokens: Estimated tokens for the request
        """
        now = time.time()
        
        # Clean up old entries (older than 1 minute)
        cutoff = now - 60
        self._request_times = [t for t in self._request_times if t > cutoff]
        self._token_usage = [(t, tokens) for t, tokens in self._token_usage if t > cutoff]
        
        # Check request rate limit
        if len(self._request_times) >= self.requests_per_minute:
            sleep_time = 60 - (now - self._request_times[0])
            if sleep_time > 0:
                logger.debug(f"Rate limit: sleeping for {sleep_time:.2f}s (requests)")
                time.sleep(sleep_time)
        
        # Check token rate limit
        current_tokens = sum(tokens for _, tokens in self._token_usage)
        if current_tokens + estimated_tokens > self.tokens_per_minute:
            sleep_time = 60 - (now - self._token_usage[0][0])
            if sleep_time > 0:
                logger.debug(f"Rate limit: sleeping for {sleep_time:.2f}s (tokens)")
                time.sleep(sleep_time)
    
    def _record_request(self, tokens: int = 0) -> None:
        """
        Record a request for rate limiting.
        
        Args:
            tokens: Number of tokens used
        """
        now = time.time()
        self._request_times.append(now)
        if tokens > 0:
            self._token_usage.append((now, tokens))
    
    def _make_request_with_retry(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> Any:
        """
        Make an API request with retry logic.
        
        Args:
            messages: List of message dictionaries
            **kwargs: Additional API parameters
            
        Returns:
            API response
            
        Raises:
            LLMError: If all retries fail
        """
        delay = self.retry_delay
        
        for attempt in range(self.max_retries + 1):
            try:
                # Estimate tokens for rate limiting
                estimated_tokens = self.count_tokens_in_messages(messages)
                self._wait_for_rate_limit(estimated_tokens)
                
                # Make the request
                # Use max_completion_tokens for newer OpenAI API (GPT-4, GPT-5, o1, o3 models)
                max_tokens_value = kwargs.get('max_tokens', self.max_tokens)
                response = self.client.chat.completions.create(
                    model=kwargs.get('model', self.model),
                    messages=messages,
                    temperature=kwargs.get('temperature', self.temperature),
                    max_completion_tokens=max_tokens_value,
                    top_p=kwargs.get('top_p', self.top_p),
                )
                
                # Record the request
                total_tokens = response.usage.total_tokens if response.usage else 0
                self._record_request(total_tokens)
                
                return response
                
            except RateLimitError as e:
                if attempt < self.max_retries:
                    logger.warning(f"Rate limit hit, retrying in {delay}s...")
                    time.sleep(delay)
                    if self.exponential_backoff:
                        delay *= 2
                else:
                    raise LLMError(
                        "Rate limit exceeded after all retries",
                        details={'error': str(e), 'retries': self.max_retries}
                    )
                    
            except APITimeoutError as e:
                if attempt < self.max_retries:
                    logger.warning(f"API timeout, retrying in {delay}s...")
                    time.sleep(delay)
                    if self.exponential_backoff:
                        delay *= 2
                else:
                    raise LLMError(
                        "API timeout after all retries",
                        details={'error': str(e), 'retries': self.max_retries}
                    )
                    
            except APIError as e:
                raise LLMError(
                    f"OpenAI API error: {e}",
                    details={'error': str(e), 'code': e.code if hasattr(e, 'code') else None}
                )
                
            except Exception as e:
                raise LLMError(
                    f"Unexpected error calling OpenAI API: {e}",
                    details={'error': str(e)}
                )
        
        # Should not reach here
        raise LLMError("All retries failed")
    
    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate text from a prompt.
        
        Args:
            prompt: Input prompt
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text
        """
        messages = [{'role': 'user', 'content': prompt}]
        return self.chat(messages, **kwargs)
    
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """
        Generate response from a conversation.
        
        Args:
            messages: List of message dictionaries
            **kwargs: Additional generation parameters
            
        Returns:
            Generated response
        """
        merged_kwargs = self._merge_kwargs(**kwargs)
        
        response = self._make_request_with_retry(messages, **merged_kwargs)
        
        # Extract content
        content = response.choices[0].message.content or ""
        
        # Update stats
        if response.usage:
            self._update_stats(
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens
            )
        
        return content
    
    def get_token_count(self, text: str) -> int:
        """
        Get the token count for a text.
        
        Args:
            text: Text to count tokens for
            
        Returns:
            Number of tokens
        """
        return self.token_counter.count(text)
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Return statistics about API usage.
        
        Returns:
            Dictionary with usage statistics
        """
        stats = super().get_stats()
        stats.update({
            'provider': 'openai',
            'encoding': self.token_counter.encoding_name,
        })
        return stats


class AzureOpenAIClient(OpenAIClient):
    """
    Azure OpenAI LLM client.
    
    This client extends OpenAIClient to work with Azure OpenAI Service.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Azure OpenAI client.
        
        Args:
            config: Configuration dictionary with:
                - azure_endpoint: Azure OpenAI endpoint
                - api_key: Azure OpenAI API key
                - api_version: API version (default: 2024-02-01)
                - deployment_name: Model deployment name
        """
        if not OPENAI_AVAILABLE:
            raise ImportError(
                "Azure OpenAI client requires the 'openai' package. "
                "Install it with: pip install openai"
            )
        
        # Initialize base without calling parent __init__
        self.config = config
        self.name = self.__class__.__name__
        self.model = config.get('deployment_name', 'gpt-5-mini')
        self.temperature = config.get('temperature', 0.0)
        self.max_tokens = config.get('max_tokens', 4096)
        self.top_p = config.get('top_p', 1.0)
        
        # Usage tracking
        self._total_calls = 0
        self._total_tokens = 0
        self._prompt_tokens = 0
        self._completion_tokens = 0
        
        # Initialize Azure OpenAI client
        azure_endpoint = config.get('azure_endpoint')
        api_key = config.get('api_key')
        api_version = config.get('api_version', '2024-02-01')
        
        if not azure_endpoint:
            raise LLMError("Azure OpenAI endpoint is required")
        
        self.client = openai.AzureOpenAI(
            azure_endpoint=azure_endpoint,
            api_key=api_key,
            api_version=api_version,
        )
        
        # Rate limiting configuration
        rate_limit_config = config.get('rate_limit', {})
        self.requests_per_minute = rate_limit_config.get('requests_per_minute', 60)
        self.tokens_per_minute = rate_limit_config.get('tokens_per_minute', 150000)
        self.max_retries = rate_limit_config.get('max_retries', 3)
        self.retry_delay = rate_limit_config.get('retry_delay', 1.0)
        self.exponential_backoff = rate_limit_config.get('exponential_backoff', True)
        
        # Rate limiting state
        self._request_times: List[float] = []
        self._token_usage: List[tuple] = []
        
        # Initialize token counter
        encoding_name = get_tokenizer_for_model(self.model)
        self.token_counter = TokenCounter(encoding_name)
