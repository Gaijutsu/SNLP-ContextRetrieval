"""Query generation for RAG-based code retrieval.

This module provides query generation strategies for converting issue
descriptions into effective search queries.
"""

import re
from typing import List, Dict, Optional, Any, Set
from dataclasses import dataclass
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)


@dataclass
class GeneratedQuery:
    """A generated search query with metadata."""
    query: str
    query_type: str  # keyword, natural_language, code_symbols, etc.
    weight: float = 1.0  # Importance weight
    source: str = "generated"  # How it was generated


class QueryGenerator(ABC):
    """Abstract base class for query generation strategies."""
    
    @abstractmethod
    def generate(self, problem_statement: str) -> List[GeneratedQuery]:
        """Generate search queries from a problem statement.
        
        Args:
            problem_statement: The issue/problem description.
            
        Returns:
            List of generated queries.
        """
        pass


class MultiStrategyQueryGenerator(QueryGenerator):
    """Query generator using multiple strategies.
    
    Generates queries using various strategies and combines them for
    comprehensive coverage:
    - Keyword extraction
    - Code symbol extraction
    - Natural language reformulation
    - Error pattern extraction
    
    Example:
        ```python
        generator = MultiStrategyQueryGenerator(config={
            "strategies": ["keywords", "code_symbols", "error_patterns"],
            "max_queries": 5
        })
        queries = generator.generate("JSON decode error in parse_data function")
        ```
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the query generator.
        
        Args:
            config: Configuration dictionary with options:
                - strategies: List of strategies to use
                - max_queries: Maximum number of queries to generate (default: 5)
                - include_original: Include original problem statement (default: True)
                - keyword_boost: Boost for keyword queries (default: 1.5)
        """
        self.config = config
        self.strategies = config.get("strategies", [
            "keywords",
            "code_symbols",
            "error_patterns",
            "natural_language"
        ])
        self.max_queries = config.get("max_queries", 5)
        self.include_original = config.get("include_original", True)
        self.keyword_boost = config.get("keyword_boost", 1.5)
    
    def generate(self, problem_statement: str) -> List[GeneratedQuery]:
        """Generate queries using multiple strategies.
        
        Args:
            problem_statement: The issue/problem description.
            
        Returns:
            List of generated queries.
        """
        all_queries = []
        
        # Include original if configured
        if self.include_original:
            all_queries.append(GeneratedQuery(
                query=problem_statement,
                query_type="original",
                weight=1.0,
                source="original"
            ))
        
        # Apply each strategy
        for strategy in self.strategies:
            if strategy == "keywords":
                queries = self._extract_keywords(problem_statement)
                all_queries.extend(queries)
            elif strategy == "code_symbols":
                queries = self._extract_code_symbols(problem_statement)
                all_queries.extend(queries)
            elif strategy == "error_patterns":
                queries = self._extract_error_patterns(problem_statement)
                all_queries.extend(queries)
            elif strategy == "natural_language":
                queries = self._reformulate_natural_language(problem_statement)
                all_queries.extend(queries)
            elif strategy == "action_focus":
                queries = self._action_focus(problem_statement)
                all_queries.extend(queries)
        
        # Deduplicate and limit
        all_queries = self._deduplicate_queries(all_queries)
        all_queries = all_queries[:self.max_queries]
        
        return all_queries
    
    def _extract_keywords(self, text: str) -> List[GeneratedQuery]:
        """Extract important keywords from the problem statement.
        
        Args:
            text: Problem statement text.
            
        Returns:
            List of keyword-based queries.
        """
        # Common stop words to remove
        stop_words = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
            'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
            'would', 'could', 'should', 'may', 'might', 'must', 'shall',
            'can', 'need', 'dare', 'ought', 'used', 'to', 'of', 'in',
            'for', 'on', 'with', 'at', 'by', 'from', 'as', 'into',
            'through', 'during', 'before', 'after', 'above', 'below',
            'between', 'under', 'again', 'further', 'then', 'once',
            'here', 'there', 'when', 'where', 'why', 'how', 'all',
            'each', 'few', 'more', 'most', 'other', 'some', 'such',
            'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than',
            'too', 'very', 'just', 'and', 'but', 'if', 'or', 'because',
            'until', 'while', 'this', 'that', 'these', 'those', 'i',
            'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves',
            'you', 'your', 'yours', 'yourself', 'yourselves', 'he',
            'him', 'his', 'himself', 'she', 'her', 'hers', 'herself',
            'it', 'its', 'itself', 'they', 'them', 'their', 'theirs',
            'themselves', 'what', 'which', 'who', 'whom', 'whose',
            'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was',
            'were', 'being', 'been', 'have', 'has', 'had', 'do', 'does',
            'did', 'will', 'would', 'shall', 'should', 'may', 'might',
            'can', 'could', 'must', 'ought', 'need', 'dare', 'used',
            'it', 'its', 'itself', 'they', 'them', 'their', 'theirs'
        }
        
        # Extract words
        words = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', text)
        
        # Filter out stop words and short words
        keywords = [
            w for w in words
            if w.lower() not in stop_words and len(w) > 2
        ]
        
        # Get unique keywords while preserving order
        seen = set()
        unique_keywords = []
        for kw in keywords:
            kw_lower = kw.lower()
            if kw_lower not in seen:
                seen.add(kw_lower)
                unique_keywords.append(kw)
        
        if not unique_keywords:
            return []
        
        # Create queries
        queries = []
        
        # Top keywords query
        top_keywords = unique_keywords[:8]
        queries.append(GeneratedQuery(
            query=" ".join(top_keywords),
            query_type="keywords",
            weight=self.keyword_boost,
            source="keyword_extraction"
        ))
        
        # Action + object query
        action_words = [w for w in unique_keywords if w.lower() in {
            'fix', 'handle', 'process', 'parse', 'convert', 'transform',
            'validate', 'check', 'verify', 'get', 'set', 'create',
            'build', 'generate', 'compute', 'calculate', 'update',
            'delete', 'remove', 'add', 'insert', 'find', 'search',
            'load', 'save', 'read', 'write', 'send', 'receive',
            'encode', 'decode', 'serialize', 'deserialize', 'format'
        }]
        
        if action_words and len(unique_keywords) > 1:
            # Combine action with top non-action keywords
            other_keywords = [w for w in unique_keywords[:5] if w not in action_words]
            if other_keywords:
                action_query = f"{action_words[0]} {' '.join(other_keywords[:2])}"
                queries.append(GeneratedQuery(
                    query=action_query,
                    query_type="action_keywords",
                    weight=self.keyword_boost * 1.2,
                    source="keyword_extraction"
                ))
        
        return queries
    
    def _extract_code_symbols(self, text: str) -> List[GeneratedQuery]:
        """Extract code symbols (function names, class names, etc.).
        
        Args:
            text: Problem statement text.
            
        Returns:
            List of code symbol queries.
        """
        queries = []
        
        # Extract snake_case identifiers
        snake_case = re.findall(r'\b[a-z][a-z0-9_]*[a-z0-9]\b', text)
        snake_case = [s for s in snake_case if '_' in s or len(s) > 5]
        
        # Extract camelCase/PascalCase identifiers
        camel_case = re.findall(r'\b[a-zA-Z][a-zA-Z0-9]*[a-z][A-Z][a-zA-Z0-9]*\b', text)
        
        # Extract dot notation (module.function)
        dot_notation = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)+\b', text)
        
        # Combine all symbols
        all_symbols = list(dict.fromkeys(snake_case + camel_case + dot_notation))
        
        if all_symbols:
            # Create symbol-focused query
            symbol_query = " ".join(all_symbols[:5])
            queries.append(GeneratedQuery(
                query=symbol_query,
                query_type="code_symbols",
                weight=1.3,
                source="symbol_extraction"
            ))
            
            # Create function search query
            func_symbols = [s for s in all_symbols if '_' in s or any(c.isupper() for c in s[1:])]
            if func_symbols:
                queries.append(GeneratedQuery(
                    query=f"def {func_symbols[0]}",
                    query_type="function_search",
                    weight=1.4,
                    source="symbol_extraction"
                ))
        
        return queries
    
    def _extract_error_patterns(self, text: str) -> List[GeneratedQuery]:
        """Extract error patterns and exception types.
        
        Args:
            text: Problem statement text.
            
        Returns:
            List of error pattern queries.
        """
        queries = []
        
        # Extract exception types
        exception_pattern = r'\b([A-Z][a-zA-Z]*Error|Exception|Warning|AssertionError)\b'
        exceptions = re.findall(exception_pattern, text)
        
        # Extract error messages
        error_message_pattern = r'["\']([^"\']*(?:error|exception|failed)[^"\']*)["\']'
        error_messages = re.findall(error_message_pattern, text, re.IGNORECASE)
        
        # Extract traceback patterns
        traceback_pattern = r'File "([^"]+)", line (\d+)'
        tracebacks = re.findall(traceback_pattern, text)
        
        if exceptions:
            # Query for exception handling
            exc_query = f"{exceptions[0]} handle catch"
            queries.append(GeneratedQuery(
                query=exc_query,
                query_type="error_handling",
                weight=1.4,
                source="error_extraction"
            ))
            
            # Query for raise statement
            raise_query = f"raise {exceptions[0]}"
            queries.append(GeneratedQuery(
                query=raise_query,
                query_type="error_source",
                weight=1.3,
                source="error_extraction"
            ))
        
        if error_messages:
            # Query with error message keywords
            msg_keywords = " ".join(error_messages[0].split()[:5])
            queries.append(GeneratedQuery(
                query=msg_keywords,
                query_type="error_message",
                weight=1.2,
                source="error_extraction"
            ))
        
        if tracebacks:
            # Query for specific file
            file_path = tracebacks[0][0]
            file_name = file_path.split('/')[-1].replace('.py', '')
            queries.append(GeneratedQuery(
                query=f"{file_name} error",
                query_type="file_error",
                weight=1.3,
                source="error_extraction"
            ))
        
        return queries
    
    def _reformulate_natural_language(self, text: str) -> List[GeneratedQuery]:
        """Reformulate problem statement into search-friendly natural language.
        
        Args:
            text: Problem statement text.
            
        Returns:
            List of natural language queries.
        """
        queries = []
        
        # Remove markdown formatting
        clean_text = re.sub(r'```[\s\S]*?```', '', text)  # Code blocks
        clean_text = re.sub(r'`([^`]+)`', r'\1', clean_text)  # Inline code
        clean_text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', clean_text)  # Links
        
        # Extract the first sentence or paragraph
        sentences = re.split(r'[.!?]+', clean_text)
        first_sentence = sentences[0].strip() if sentences else clean_text
        
        if len(first_sentence) > 20:
            queries.append(GeneratedQuery(
                query=first_sentence[:200],
                query_type="natural_language",
                weight=1.0,
                source="reformulation"
            ))
        
        # Create "how to" query
        how_to_pattern = r'(?:how|what|when|where)\s+(?:to|do|is|are|does|can|should)\s+(.+)'
        match = re.search(how_to_pattern, text, re.IGNORECASE)
        if match:
            how_to_query = f"how to {match.group(1)[:100]}"
            queries.append(GeneratedQuery(
                query=how_to_query,
                query_type="how_to",
                weight=1.1,
                source="reformulation"
            ))
        
        return queries
    
    def _action_focus(self, text: str) -> List[GeneratedQuery]:
        """Create action-focused queries.
        
        Args:
            text: Problem statement text.
            
        Returns:
            List of action-focused queries.
        """
        queries = []
        
        # Common action words in bug reports
        action_patterns = [
            (r'(?:should|needs? to|must)\s+(\w+)', 'requirement'),
            (r'(?:fix|correct|repair)\s+(\w+)', 'fix'),
            (r'(?:add|implement|create)\s+(\w+)', 'add'),
            (r'(?:remove|delete|eliminate)\s+(\w+)', 'remove'),
            (r'(?:update|change|modify)\s+(\w+)', 'update'),
            (r'(?:handle|catch|process)\s+(\w+)', 'handle'),
        ]
        
        for pattern, action_type in action_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                action_query = f"{action_type} {' '.join(matches[:3])}"
                queries.append(GeneratedQuery(
                    query=action_query,
                    query_type="action_focus",
                    weight=1.2,
                    source="action_extraction"
                ))
                break  # Only take the first match type
        
        return queries
    
    def _deduplicate_queries(self, queries: List[GeneratedQuery]) -> List[GeneratedQuery]:
        """Remove duplicate queries.
        
        Args:
            queries: List of queries.
            
        Returns:
            Deduplicated list (keeps highest weight duplicate).
        """
        seen: Dict[str, GeneratedQuery] = {}
        
        for query in queries:
            key = query.query.lower().strip()
            if key not in seen or query.weight > seen[key].weight:
                seen[key] = query
        
        # Sort by weight descending
        return sorted(seen.values(), key=lambda q: q.weight, reverse=True)


class SimpleQueryGenerator(QueryGenerator):
    """Simple query generator that just uses the problem statement.
    
    This is a baseline generator that doesn't do any processing.
    """
    
    def generate(self, problem_statement: str) -> List[GeneratedQuery]:
        """Generate a single query from the problem statement."""
        return [GeneratedQuery(
            query=problem_statement,
            query_type="original",
            weight=1.0,
            source="simple"
        )]


class KeywordOnlyQueryGenerator(QueryGenerator):
    """Query generator that only extracts keywords.
    
    Useful when you want simple, keyword-focused queries.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.max_keywords = config.get("max_keywords", 10)
    
    def generate(self, problem_statement: str) -> List[GeneratedQuery]:
        """Generate keyword-only queries."""
        # Extract words
        words = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', problem_statement)
        
        # Filter stop words
        stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                      'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                      'would', 'could', 'should', 'may', 'might', 'must', 'shall',
                      'can', 'need', 'dare', 'ought', 'used', 'to', 'of', 'in',
                      'for', 'on', 'with', 'at', 'by', 'from', 'as', 'into'}
        
        keywords = [w for w in words if w.lower() not in stop_words and len(w) > 2]
        
        # Get unique keywords
        seen = set()
        unique_keywords = []
        for kw in keywords:
            kw_lower = kw.lower()
            if kw_lower not in seen:
                seen.add(kw_lower)
                unique_keywords.append(kw)
        
        if not unique_keywords:
            return [GeneratedQuery(
                query=problem_statement,
                query_type="original",
                weight=1.0,
                source="fallback"
            )]
        
        return [GeneratedQuery(
            query=" ".join(unique_keywords[:self.max_keywords]),
            query_type="keywords",
            weight=1.0,
            source="keyword_extraction"
        )]