"""
CodeBLEU metric for the SWE-bench comparison framework.

This module implements the CodeBLEU metric, which measures code similarity
using n-gram matching, syntax matching, and data-flow matching.

Reference: CodeBLEU: a Method for Automatic Evaluation of Code Synthesis
"""

import re
import ast
from typing import Any, Dict, List, Optional, Set, Tuple
from collections import Counter

from .base import Metric


class CodeBLEUMetric(Metric):
    """
    Metric for computing CodeBLEU score.
    
    CodeBLEU combines four components:
    1. N-gram match: Textual similarity
    2. Weighted n-gram match: Syntax-aware similarity
    3. Syntax match: AST structural similarity
    4. Data-flow match: Semantic similarity via data flow
    
    Example:
        metric = CodeBLEUMetric(alpha=0.25, beta=0.25, gamma=0.25, delta=0.25)
        score = metric.compute_codebleu(generated_patch, gold_patch)
        # score is between 0.0 and 1.0
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the CodeBLEU metric.
        
        Args:
            config: Configuration dictionary
                - alpha: Weight for n-gram match (default: 0.25)
                - beta: Weight for weighted n-gram match (default: 0.25)
                - gamma: Weight for syntax match (default: 0.25)
                - delta: Weight for data-flow match (default: 0.25)
                - ngram_size: Size of n-grams (default: 4)
        """
        super().__init__(config)
        config = config or {}
        
        self.alpha = config.get('alpha', 0.25)
        self.beta = config.get('beta', 0.25)
        self.gamma = config.get('gamma', 0.25)
        self.delta = config.get('delta', 0.25)
        self.ngram_size = config.get('ngram_size', 4)
        
        # Validate weights sum to 1
        total_weight = self.alpha + self.beta + self.gamma + self.delta
        if abs(total_weight - 1.0) > 0.001:
            # Normalize weights
            self.alpha /= total_weight
            self.beta /= total_weight
            self.gamma /= total_weight
            self.delta /= total_weight
    
    def compute(
        self,
        patch_result: 'PatchResult',
        instance: 'SWEInstance',
        evaluation_result: 'EvaluationResult'
    ) -> Dict[str, Any]:
        """
        Compute CodeBLEU metric.
        
        Args:
            patch_result: The result from patch generation
            instance: The SWE-bench instance
            evaluation_result: The evaluation result
            
        Returns:
            Dictionary with CodeBLEU score and components
        """
        generated_patch = patch_result.patch_content
        gold_patch = getattr(instance, 'patch', None)
        
        if not generated_patch or not gold_patch:
            return {
                'codebleu': 0.0,
                'ngram_match': 0.0,
                'weighted_ngram_match': 0.0,
                'syntax_match': 0.0,
                'dataflow_match': 0.0
            }
        
        codebleu = self.compute_codebleu(generated_patch, gold_patch)
        
        return {
            'codebleu': codebleu,
            'ngram_match': self._last_ngram_match,
            'weighted_ngram_match': self._last_weighted_ngram_match,
            'syntax_match': self._last_syntax_match,
            'dataflow_match': self._last_dataflow_match
        }
    
    def compute_codebleu(
        self,
        generated: str,
        reference: str
    ) -> float:
        """
        Compute CodeBLEU score between generated and reference code.
        
        Args:
            generated: Generated code/patch
            reference: Reference/gold code/patch
            
        Returns:
            CodeBLEU score (0.0-1.0)
        """
        # Extract code from patches (remove diff markers)
        gen_code = self._extract_code_from_patch(generated)
        ref_code = self._extract_code_from_patch(reference)
        
        # Tokenize
        gen_tokens = self._tokenize(gen_code)
        ref_tokens = self._tokenize(ref_code)
        
        if not gen_tokens or not ref_tokens:
            return 0.0
        
        # Compute components
        self._last_ngram_match = self._compute_ngram_match(gen_tokens, ref_tokens)
        self._last_weighted_ngram_match = self._compute_weighted_ngram_match(
            gen_tokens, ref_tokens
        )
        self._last_syntax_match = self._compute_syntax_match(gen_code, ref_code)
        self._last_dataflow_match = self._compute_dataflow_match(gen_code, ref_code)
        
        # Combine components
        codebleu = (
            self.alpha * self._last_ngram_match +
            self.beta * self._last_weighted_ngram_match +
            self.gamma * self._last_syntax_match +
            self.delta * self._last_dataflow_match
        )
        
        return codebleu
    
    def _extract_code_from_patch(self, patch: str) -> str:
        """Extract code from unified diff patch."""
        lines = []
        for line in patch.split('\n'):
            if line.startswith('+') and not line.startswith('+++'):
                lines.append(line[1:])
            elif line.startswith(' ') and not line.startswith(' @@'):
                lines.append(line[1:])
        return '\n'.join(lines)
    
    def _tokenize(self, code: str) -> List[str]:
        """Tokenize code into tokens."""
        # Simple tokenization
        # Remove comments and strings for cleaner tokenization
        code = re.sub(r'#.*', '', code)
        code = re.sub(r'""".*?"""', '""""""', code, flags=re.DOTALL)
        code = re.sub(r"'''.*?'''", "''''''", code, flags=re.DOTALL)
        code = re.sub(r'"[^"]*"', '""', code)
        code = re.sub(r"'[^']*'", "''", code)
        
        # Split on whitespace and punctuation
        tokens = re.findall(r'\w+|[^\w\s]', code)
        return [t for t in tokens if t.strip()]
    
    def _get_ngrams(self, tokens: List[str], n: int) -> List[Tuple[str, ...]]:
        """Get n-grams from tokens."""
        return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
    
    def _compute_ngram_match(
        self,
        gen_tokens: List[str],
        ref_tokens: List[str]
    ) -> float:
        """
        Compute n-gram match score (BLEU-like).
        
        Args:
            gen_tokens: Generated code tokens
            ref_tokens: Reference code tokens
            
        Returns:
            N-gram match score (0.0-1.0)
        """
        scores = []
        
        for n in range(1, self.ngram_size + 1):
            gen_ngrams = Counter(self._get_ngrams(gen_tokens, n))
            ref_ngrams = Counter(self._get_ngrams(ref_tokens, n))
            
            if not gen_ngrams or not ref_ngrams:
                continue
            
            # Count matching n-grams
            matches = sum((gen_ngrams & ref_ngrams).values())
            total = sum(gen_ngrams.values())
            
            if total > 0:
                scores.append(matches / total)
        
        # Geometric mean of n-gram scores
        if not scores:
            return 0.0
        
        # Add brevity penalty
        bp = min(1.0, len(gen_tokens) / len(ref_tokens)) if ref_tokens else 0.0
        
        import math
        geo_mean = math.exp(sum(math.log(s) for s in scores if s > 0) / len(scores))
        
        return bp * geo_mean
    
    def _compute_weighted_ngram_match(
        self,
        gen_tokens: List[str],
        ref_tokens: List[str]
    ) -> float:
        """
        Compute weighted n-gram match (syntax-aware).
        
        Keywords and syntax elements get higher weights.
        
        Args:
            gen_tokens: Generated code tokens
            ref_tokens: Reference code tokens
            
        Returns:
            Weighted n-gram match score (0.0-1.0)
        """
        # Define keyword weights
        python_keywords = {
            'def', 'class', 'if', 'else', 'elif', 'for', 'while', 'try',
            'except', 'finally', 'with', 'return', 'yield', 'import',
            'from', 'as', 'pass', 'break', 'continue', 'raise', 'assert'
        }
        
        def get_weight(token: str) -> float:
            if token in python_keywords:
                return 2.0
            elif token.isidentifier():
                return 1.5
            else:
                return 1.0
        
        # Apply weights to tokens
        weighted_gen = []
        weighted_ref = []
        
        for token in gen_tokens:
            weight = get_weight(token)
            weighted_gen.extend([token] * int(weight))
        
        for token in ref_tokens:
            weight = get_weight(token)
            weighted_ref.extend([token] * int(weight))
        
        return self._compute_ngram_match(weighted_gen, weighted_ref)
    
    def _compute_syntax_match(self, gen_code: str, ref_code: str) -> float:
        """
        Compute syntax match score using AST.
        
        Args:
            gen_code: Generated code
            ref_code: Reference code
            
        Returns:
            Syntax match score (0.0-1.0)
        """
        try:
            gen_ast = ast.parse(gen_code)
            ref_ast = ast.parse(ref_code)
            
            # Compare AST structures
            gen_nodes = self._get_ast_nodes(gen_ast)
            ref_nodes = self._get_ast_nodes(ref_ast)
            
            if not gen_nodes or not ref_nodes:
                return 0.0
            
            # Compute similarity based on node type sequences
            matches = sum(1 for g, r in zip(gen_nodes, ref_nodes) if g == r)
            max_len = max(len(gen_nodes), len(ref_nodes))
            
            return matches / max_len if max_len > 0 else 0.0
            
        except SyntaxError:
            # If code can't be parsed, fall back to token-based matching
            gen_tokens = self._tokenize(gen_code)
            ref_tokens = self._tokenize(ref_code)
            return self._compute_ngram_match(gen_tokens, ref_tokens)
    
    def _get_ast_nodes(self, tree: ast.AST) -> List[str]:
        """Get list of AST node types."""
        nodes = []
        for node in ast.walk(tree):
            nodes.append(type(node).__name__)
        return nodes
    
    def _compute_dataflow_match(self, gen_code: str, ref_code: str) -> float:
        """
        Compute data-flow match score.
        
        This is a simplified version that compares variable usage patterns.
        
        Args:
            gen_code: Generated code
            ref_code: Reference code
            
        Returns:
            Data-flow match score (0.0-1.0)
        """
        try:
            gen_vars = self._extract_variable_flow(gen_code)
            ref_vars = self._extract_variable_flow(ref_code)
            
            if not gen_vars or not ref_vars:
                return 0.0
            
            # Compare variable flow patterns
            gen_patterns = set(gen_vars.items())
            ref_patterns = set(ref_vars.items())
            
            intersection = len(gen_patterns & ref_patterns)
            union = len(gen_patterns | ref_patterns)
            
            return intersection / union if union > 0 else 0.0
            
        except Exception:
            return 0.0
    
    def _extract_variable_flow(self, code: str) -> Dict[str, Set[str]]:
        """
        Extract variable flow patterns from code.
        
        Returns a dictionary mapping variable names to the operations
        performed on them.
        
        Args:
            code: Source code
            
        Returns:
            Dictionary of variable flows
        """
        flows = {}
        
        try:
            tree = ast.parse(code)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Name):
                    var_name = node.id
                    if var_name not in flows:
                        flows[var_name] = set()
                    
                    # Determine context
                    parent = getattr(node, 'parent', None)
                    if isinstance(parent, ast.Assign):
                        flows[var_name].add('assign')
                    elif isinstance(parent, ast.Call):
                        flows[var_name].add('call')
                    elif isinstance(parent, ast.Attribute):
                        flows[var_name].add('attribute')
                    else:
                        flows[var_name].add('use')
                        
        except SyntaxError:
            pass
        
        return flows
    
    def get_name(self) -> str:
        """Get the metric name."""
        return 'codebleu'


class SimpleCodeSimilarityMetric(Metric):
    """
    Simple code similarity metric using basic string comparison.
    
    This is a lightweight alternative to CodeBLEU for quick comparisons.
    """
    
    def compute(
        self,
        patch_result: 'PatchResult',
        instance: 'SWEInstance',
        evaluation_result: 'EvaluationResult'
    ) -> Dict[str, Any]:
        """Compute simple similarity metrics."""
        generated = patch_result.patch_content or ""
        gold = getattr(instance, 'patch', "")
        
        if not generated or not gold:
            return {'similarity': 0.0, 'line_similarity': 0.0}
        
        # Character-level similarity
        char_sim = self._compute_char_similarity(generated, gold)
        
        # Line-level similarity
        line_sim = self._compute_line_similarity(generated, gold)
        
        return {
            'similarity': char_sim,
            'line_similarity': line_sim
        }
    
    def _compute_char_similarity(self, s1: str, s2: str) -> float:
        """Compute character-level similarity."""
        if not s1 or not s2:
            return 0.0
        
        # Use Levenshtein distance approximation
        len1, len2 = len(s1), len(s2)
        max_len = max(len1, len2)
        
        if max_len == 0:
            return 1.0
        
        # Simple approximation: common characters / total characters
        common = sum(1 for c in s1 if c in s2)
        return common / max_len
    
    def _compute_line_similarity(self, s1: str, s2: str) -> float:
        """Compute line-level similarity."""
        lines1 = set(s1.split('\n'))
        lines2 = set(s2.split('\n'))
        
        if not lines1 or not lines2:
            return 0.0
        
        intersection = len(lines1 & lines2)
        union = len(lines1 | lines2)
        
        return intersection / union if union > 0 else 0.0
    
    def get_name(self) -> str:
        """Get the metric name."""
        return 'simple_similarity'


# Forward reference imports
from ...patch_generators.base import PatchResult  # noqa: E402
from ...dataset.loader import SWEInstance  # noqa: E402
from ..base import EvaluationResult  # noqa: E402
