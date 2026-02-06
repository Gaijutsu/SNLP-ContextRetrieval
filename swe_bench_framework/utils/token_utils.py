"""
Token counting utilities for the SWE-bench comparison framework.

This module provides token counting functionality using tiktoken,
with support for various tokenizers and truncation operations.
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
    Token counter using tiktoken.
    
    This class provides token counting functionality for various
    OpenAI tokenizers, with fallback estimation when tiktoken
    is not available.
    
    Example:
        >>> counter = TokenCounter('cl100k_base')
        >>> count = counter.count("Hello, world!")
        >>> print(count)
        4
        >>> truncated = counter.truncate("Long text...", max_tokens=10)
    """
    
    # Average characters per token for different languages
    CHARS_PER_TOKEN = {
        'english': 4.0,
        'code': 3.5,
        'default': 4.0,
    }
    
    def __init__(self, encoding_name: str = "cl100k_base"):
        """
        Initialize the token counter.
        
        Args:
            encoding_name: Name of the tiktoken encoding to use.
                Common values: 'cl100k_base' (GPT-5, Claude 4.5), 'o200k_base' (GPT-4o)
        """
        self.encoding_name = encoding_name
        self._encoding = None
        
        if TIKTOKEN_AVAILABLE:
            try:
                self._encoding = tiktoken.get_encoding(encoding_name)
                logger.debug(f"Loaded encoding: {encoding_name}")
            except Exception as e:
                logger.warning(
                    f"Failed to load encoding {encoding_name}: {e}. "
                    "Using fallback estimation."
                )
        else:
            logger.warning(
                "tiktoken not available. Using fallback estimation. "
                "Install with: pip install tiktoken"
            )
    
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
    
    def truncate(
        self,
        text: str,
        max_tokens: int,
        truncation_side: str = "right"
    ) -> str:
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
    
    def truncate_batch(
        self,
        texts: List[str],
        max_tokens: int,
        truncation_side: str = "right"
    ) -> List[str]:
        """
        Truncate multiple texts to fit within token limit.
        
        Args:
            texts: List of texts to truncate
            max_tokens: Maximum number of tokens per text
            truncation_side: Which side to truncate
            
        Returns:
            List of truncated texts
        """
        return [self.truncate(t, max_tokens, truncation_side) for t in texts]
    
    def split_into_chunks(
        self,
        text: str,
        chunk_size: int,
        overlap: int = 0
    ) -> List[str]:
        """
        Split text into chunks of specified token size.
        
        Args:
            text: Text to split
            chunk_size: Maximum tokens per chunk
            overlap: Number of overlapping tokens between chunks
            
        Returns:
            List of text chunks
        """
        if not text:
            return []
        
        if self._encoding is not None:
            try:
                tokens = self._encoding.encode(text)
                chunks = []
                start = 0
                
                while start < len(tokens):
                    end = min(start + chunk_size, len(tokens))
                    chunk_tokens = tokens[start:end]
                    chunks.append(self._encoding.decode(chunk_tokens))
                    start = end - overlap if overlap > 0 else end
                    
                    # Avoid infinite loop if overlap >= chunk_size
                    if start >= end:
                        start = end
                
                return chunks
            except Exception as e:
                logger.warning(f"Chunking failed: {e}. Using fallback.")
        
        # Fallback: estimate based on characters
        return self._estimate_split(text, chunk_size, overlap)
    
    def _estimate_tokens(self, text: str, text_type: str = "default") -> int:
        """
        Estimate token count based on character count.
        
        Args:
            text: Text to estimate
            text_type: Type of text (english, code, default)
            
        Returns:
            Estimated token count
        """
        chars_per_token = self.CHARS_PER_TOKEN.get(text_type, 4.0)
        return int(len(text) / chars_per_token)
    
    def _estimate_truncate(
        self,
        text: str,
        max_tokens: int,
        truncation_side: str
    ) -> str:
        """
        Estimate truncation based on character count.
        
        Args:
            text: Text to truncate
            max_tokens: Maximum tokens
            truncation_side: Which side to truncate
            
        Returns:
            Truncated text
        """
        chars_per_token = self.CHARS_PER_TOKEN['default']
        max_chars = int(max_tokens * chars_per_token)
        
        if truncation_side == "left":
            return text[-max_chars:]
        else:
            return text[:max_chars]
    
    def _estimate_split(
        self,
        text: str,
        chunk_size: int,
        overlap: int
    ) -> List[str]:
        """
        Estimate chunking based on character count.
        
        Args:
            text: Text to split
            chunk_size: Maximum tokens per chunk
            overlap: Overlapping tokens
            
        Returns:
            List of chunks
        """
        chars_per_token = self.CHARS_PER_TOKEN['default']
        chunk_chars = int(chunk_size * chars_per_token)
        overlap_chars = int(overlap * chars_per_token)
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = min(start + chunk_chars, len(text))
            chunks.append(text[start:end])
            start = end - overlap_chars if overlap_chars > 0 else end
            
            if start >= end:
                start = end
        
        return chunks
    
    def get_stats(self) -> dict:
        """
        Get statistics about the token counter.
        
        Returns:
            Dictionary with statistics
        """
        return {
            'encoding_name': self.encoding_name,
            'encoding_loaded': self._encoding is not None,
            'tiktoken_available': TIKTOKEN_AVAILABLE,
        }


# Convenience functions for common operations

def count_tokens(text: str, encoding_name: str = "cl100k_base") -> int:
    """
    Count tokens in text using specified encoding.
    
    Args:
        text: Text to count
        encoding_name: Name of the encoding to use
        
    Returns:
        Number of tokens
    """
    counter = TokenCounter(encoding_name)
    return counter.count(text)


def truncate_to_tokens(
    text: str,
    max_tokens: int,
    encoding_name: str = "cl100k_base",
    truncation_side: str = "right"
) -> str:
    """
    Truncate text to fit within token limit.
    
    Args:
        text: Text to truncate
        max_tokens: Maximum number of tokens
        encoding_name: Name of the encoding to use
        truncation_side: Which side to truncate
        
    Returns:
        Truncated text
    """
    counter = TokenCounter(encoding_name)
    return counter.truncate(text, max_tokens, truncation_side)


def split_text_into_chunks(
    text: str,
    chunk_size: int,
    overlap: int = 0,
    encoding_name: str = "cl100k_base"
) -> List[str]:
    """
    Split text into chunks of specified token size.
    
    Args:
        text: Text to split
        chunk_size: Maximum tokens per chunk
        overlap: Number of overlapping tokens
        encoding_name: Name of the encoding to use
        
    Returns:
        List of text chunks
    """
    counter = TokenCounter(encoding_name)
    return counter.split_into_chunks(text, chunk_size, overlap)


def get_tokenizer_for_model(model_name: str) -> str:
    """
    Get the appropriate tokenizer encoding for a model.
    
    Args:
        model_name: Name of the model
        
    Returns:
        Name of the encoding to use
    """
    model_to_encoding = {
        # GPT-5 models
        'gpt-5': 'cl100k_base',
        'gpt-5-mini': 'cl100k_base',
        'gpt-5-pro': 'cl100k_base',
        # GPT-4o models
        'gpt-4o': 'o200k_base',
        'gpt-4o-mini': 'o200k_base',
        # Claude models (use cl100k_base as approximation)
        'claude-4.5-sonnet': 'cl100k_base',
        'claude-4.5': 'cl100k_base',
    }
    
    # Try exact match first
    if model_name in model_to_encoding:
        return model_to_encoding[model_name]
    
    # Try prefix match
    for prefix, encoding in model_to_encoding.items():
        if model_name.startswith(prefix):
            return encoding
    
    # Default to cl100k_base for unknown models
    logger.warning(
        f"Unknown model {model_name}, defaulting to cl100k_base encoding"
    )
    return 'cl100k_base'
