"""
Token counter for LLM operations.

This module provides token counting functionality for various LLM models.
"""

from typing import List, Optional
import logging

try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False

logger = logging.getLogger(__name__)


class TokenCounter:
    """
    Token counter for LLM models.
    
    This class provides token counting functionality using tiktoken
    when available, with fallback estimation for when it's not.
    
    Example:
        >>> counter = TokenCounter('cl100k_base')
        >>> count = counter.count("Hello, world!")
        >>> print(count)
        4
    """
    
    # Mapping of model names to encoding names
    MODEL_TO_ENCODING = {
        # GPT-5 models
        'gpt-5': 'cl100k_base',
        'gpt-5-mini': 'cl100k_base',
        'gpt-5-pro': 'cl100k_base',
        # GPT-4o models (still current)
        'gpt-4o': 'o200k_base',
        'gpt-4o-mini': 'o200k_base',
        # Claude models (use cl100k_base as approximation)
        'claude-4.5-sonnet': 'cl100k_base',
        'claude-4.5': 'cl100k_base',
    }
    
    # Average characters per token for estimation
    CHARS_PER_TOKEN = 4.0
    
    def __init__(self, encoding_name: Optional[str] = None, model: Optional[str] = None):
        """
        Initialize the token counter.
        
        Args:
            encoding_name: Name of the tiktoken encoding to use
            model: Model name (used to determine encoding if encoding_name not provided)
        """
        if encoding_name:
            self.encoding_name = encoding_name
        elif model:
            self.encoding_name = self.get_encoding_for_model(model)
        else:
            self.encoding_name = 'cl100k_base'
        
        self._encoding = None
        
        if TIKTOKEN_AVAILABLE:
            try:
                self._encoding = tiktoken.get_encoding(self.encoding_name)
                logger.debug(f"Loaded encoding: {self.encoding_name}")
            except Exception as e:
                logger.warning(
                    f"Failed to load encoding {self.encoding_name}: {e}. "
                    "Using fallback estimation."
                )
        else:
            logger.debug("tiktoken not available, using fallback estimation")
    
    def count(self, text: str) -> int:
        """
        Count tokens in text.
        
        Args:
            text: Text to count tokens for
            
        Returns:
            Number of tokens
        """
        if not text:
            return 0
        
        if self._encoding is not None:
            try:
                return len(self._encoding.encode(text))
            except Exception as e:
                logger.warning(f"Tokenization failed: {e}. Using fallback.")
        
        # Fallback: estimate based on characters
        return self._estimate_tokens(text)
    
    def count_batch(self, texts: List[str]) -> List[int]:
        """
        Count tokens in multiple texts.
        
        Args:
            texts: List of texts to count
            
        Returns:
            List of token counts
        """
        return [self.count(t) for t in texts]
    
    def truncate(self, text: str, max_tokens: int, truncation_side: str = "right") -> str:
        """
        Truncate text to fit within token limit.
        
        Args:
            text: Text to truncate
            max_tokens: Maximum number of tokens
            truncation_side: Which side to truncate ('left' or 'right')
            
        Returns:
            Truncated text
        """
        if not text:
            return text
        
        if self.count(text) <= max_tokens:
            return text
        
        if self._encoding is not None:
            try:
                tokens = self._encoding.encode(text)
                if len(tokens) <= max_tokens:
                    return text
                
                if truncation_side == "left":
                    truncated_tokens = tokens[-max_tokens:]
                else:
                    truncated_tokens = tokens[:max_tokens]
                
                return self._encoding.decode(truncated_tokens)
            except Exception as e:
                logger.warning(f"Truncation failed: {e}. Using fallback.")
        
        # Fallback: estimate based on characters
        return self._estimate_truncate(text, max_tokens, truncation_side)
    
    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count based on character count."""
        return int(len(text) / self.CHARS_PER_TOKEN)
    
    def _estimate_truncate(self, text: str, max_tokens: int, truncation_side: str) -> str:
        """Estimate truncation based on character count."""
        max_chars = int(max_tokens * self.CHARS_PER_TOKEN)
        
        if truncation_side == "left":
            return text[-max_chars:]
        else:
            return text[:max_chars]
    
    @classmethod
    def get_encoding_for_model(cls, model_name: str) -> str:
        """
        Get the appropriate encoding for a model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Name of the encoding to use
        """
        # Try exact match first
        if model_name in cls.MODEL_TO_ENCODING:
            return cls.MODEL_TO_ENCODING[model_name]
        
        # Try prefix match
        for prefix, encoding in cls.MODEL_TO_ENCODING.items():
            if model_name.startswith(prefix):
                return encoding
        
        # Default to cl100k_base for unknown models
        logger.warning(
            f"Unknown model {model_name}, defaulting to cl100k_base encoding"
        )
        return 'cl100k_base'
    
    @classmethod
    def for_model(cls, model_name: str) -> 'TokenCounter':
        """
        Create a token counter for a specific model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            TokenCounter configured for the model
        """
        return cls(model=model_name)


def count_tokens(text: str, model: str = "gpt-5-mini") -> int:
    """
    Count tokens in text for a specific model.
    
    Args:
        text: Text to count
        model: Model name
        
    Returns:
        Number of tokens
    """
    counter = TokenCounter.for_model(model)
    return counter.count(text)


def truncate_to_tokens(
    text: str,
    max_tokens: int,
    model: str = "gpt-5-mini",
    truncation_side: str = "right"
) -> str:
    """
    Truncate text to fit within token limit for a specific model.
    
    Args:
        text: Text to truncate
        max_tokens: Maximum number of tokens
        model: Model name
        truncation_side: Which side to truncate
        
    Returns:
        Truncated text
    """
    counter = TokenCounter.for_model(model)
    return counter.truncate(text, max_tokens, truncation_side)
