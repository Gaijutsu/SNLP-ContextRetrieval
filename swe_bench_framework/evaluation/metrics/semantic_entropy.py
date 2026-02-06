"""
Semantic entropy metric for the SWE-bench comparison framework.

This module implements semantic entropy computation, which measures the
uncertainty in generated patches by clustering multiple generations by
semantic meaning.

Reference: Semantic Entropy Probes: Robust and Cheap Hallucination Detection
in LLMs (https://arxiv.org/abs/2406.15927)
"""

import re
import hashlib
from typing import Any, Dict, List, Optional, Set, Tuple
from collections import defaultdict
from dataclasses import dataclass

from .base import Metric


@dataclass
class PatchCluster:
    """Cluster of semantically similar patches."""
    cluster_id: int
    representative: str
    patches: List[str]
    size: int


class SemanticEntropyMetric(Metric):
    """
    Metric for computing semantic entropy of generated patches.
    
    Semantic entropy is computed by:
    1. Generating multiple patches for the same input
    2. Clustering patches by semantic similarity
    3. Computing entropy across semantic clusters
    
    Higher entropy indicates more uncertainty/variability in generations.
    
    Example:
        metric = SemanticEntropyMetric(num_samples=5, clustering_method='ast')
        
        # Generate multiple patches
        patches = generate_multiple_patches(context, instance, n=5)
        
        # Compute entropy
        entropy = metric.compute_semantic_entropy(patches)
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the semantic entropy metric.
        
        Args:
            config: Configuration dictionary
                - num_samples: Number of patches to generate (default: 5)
                - clustering_method: Method for clustering ('ast', 'token', 'line')
                - temperature: Temperature for generation diversity (default: 0.7)
                - min_cluster_size: Minimum patches per cluster (default: 1)
        """
        super().__init__(config)
        config = config or {}
        
        self.num_samples = config.get('num_samples', 5)
        self.clustering_method = config.get('clustering_method', 'ast')
        self.temperature = config.get('temperature', 0.7)
        self.min_cluster_size = config.get('min_cluster_size', 1)
        
        # Cache for embeddings (if using embedding-based clustering)
        self._embedding_cache: Dict[str, List[float]] = {}
    
    def compute(
        self,
        patch_result: 'PatchResult',
        instance: 'SWEInstance',
        evaluation_result: 'EvaluationResult'
    ) -> Dict[str, Any]:
        """
        Compute semantic entropy metric.
        
        Note: This requires multiple patch generations. If only one patch
        is available, it returns a default value.
        
        Args:
            patch_result: The result from patch generation
            instance: The SWE-bench instance
            evaluation_result: The evaluation result
            
        Returns:
            Dictionary with semantic entropy metrics
        """
        # Check if we have intermediate steps (multiple attempts)
        intermediate_steps = getattr(patch_result, 'intermediate_steps', [])
        
        patches = []
        
        # Collect patches from intermediate steps
        for step in intermediate_steps:
            patch = step.get('patch_content')
            if patch:
                patches.append(patch)
        
        # Add final patch
        if patch_result.patch_content:
            patches.append(patch_result.patch_content)
        
        if len(patches) < 2:
            # Not enough patches for meaningful entropy calculation
            return {
                'semantic_entropy': 0.0,
                'num_patches': len(patches),
                'num_clusters': len(patches),
                'diversity_score': 0.0,
                'note': 'Insufficient patches for entropy calculation'
            }
        
        # Compute semantic entropy
        entropy, clusters = self.compute_semantic_entropy(patches)
        
        # Compute diversity score
        diversity = self._compute_diversity(patches, clusters)
        
        return {
            'semantic_entropy': entropy,
            'num_patches': len(patches),
            'num_clusters': len(clusters),
            'diversity_score': diversity,
            'cluster_sizes': [c.size for c in clusters],
            'dominant_cluster_size': max(c.size for c in clusters) if clusters else 0
        }
    
    def compute_semantic_entropy(
        self,
        patches: List[str]
    ) -> Tuple[float, List[PatchCluster]]:
        """
        Compute semantic entropy from multiple patches.
        
        Args:
            patches: List of generated patches
            
        Returns:
            Tuple of (entropy, clusters)
        """
        if not patches:
            return 0.0, []
        
        # Cluster patches by semantic similarity
        clusters = self._cluster_patches(patches)
        
        # Compute entropy
        entropy = self._compute_entropy_from_clusters(clusters, len(patches))
        
        return entropy, clusters
    
    def _cluster_patches(self, patches: List[str]) -> List[PatchCluster]:
        """
        Cluster patches by semantic similarity.
        
        Args:
            patches: List of patches to cluster
            
        Returns:
            List of patch clusters
        """
        if self.clustering_method == 'ast':
            return self._cluster_by_ast(patches)
        elif self.clustering_method == 'token':
            return self._cluster_by_tokens(patches)
        elif self.clustering_method == 'line':
            return self._cluster_by_lines(patches)
        else:
            # Default to line-based clustering
            return self._cluster_by_lines(patches)
    
    def _cluster_by_ast(self, patches: List[str]) -> List[PatchCluster]:
        """
        Cluster patches by AST structure similarity.
        
        This extracts the AST node sequence from each patch and groups
        patches with similar AST structures.
        """
        clusters: Dict[str, List[str]] = defaultdict(list)
        
        for patch in patches:
            ast_signature = self._extract_ast_signature(patch)
            clusters[ast_signature].append(patch)
        
        return [
            PatchCluster(
                cluster_id=i,
                representative=patches[0],
                patches=patches,
                size=len(patches)
            )
            for i, (_, patches) in enumerate(clusters.items())
        ]
    
    def _extract_ast_signature(self, patch: str) -> str:
        """Extract AST-based signature from patch."""
        try:
            import ast
            
            # Extract code from patch
            code = self._extract_code_from_patch(patch)
            
            # Parse AST
            tree = ast.parse(code)
            
            # Get node type sequence
            node_types = [type(node).__name__ for node in ast.walk(tree)]
            
            # Create signature from sorted unique node types
            unique_nodes = sorted(set(node_types))
            return '|'.join(unique_nodes)
            
        except Exception:
            # Fall back to simple hash
            return hashlib.md5(patch.encode()).hexdigest()[:16]
    
    def _cluster_by_tokens(self, patches: List[str]) -> List[PatchCluster]:
        """
        Cluster patches by token sequence similarity.
        
        Groups patches with similar token sequences.
        """
        clusters: Dict[str, List[str]] = defaultdict(list)
        
        for patch in patches:
            token_signature = self._extract_token_signature(patch)
            clusters[token_signature].append(patch)
        
        return [
            PatchCluster(
                cluster_id=i,
                representative=patches[0],
                patches=patches,
                size=len(patches)
            )
            for i, (_, patches) in enumerate(clusters.items())
        ]
    
    def _extract_token_signature(self, patch: str) -> str:
        """Extract token-based signature from patch."""
        # Tokenize
        tokens = self._tokenize(patch)
        
        # Create signature from token types (keywords, identifiers, etc.)
        token_types = []
        for token in tokens:
            if token in self._get_python_keywords():
                token_types.append(f'KW:{token}')
            elif token.isidentifier():
                token_types.append('ID')
            elif token.isdigit():
                token_types.append('NUM')
            else:
                token_types.append(f'SYM:{token}')
        
        return '|'.join(token_types)
    
    def _cluster_by_lines(self, patches: List[str]) -> List[PatchCluster]:
        """
        Cluster patches by modified line similarity.
        
        Groups patches that modify similar sets of lines.
        """
        clusters: Dict[str, List[str]] = defaultdict(list)
        
        for patch in patches:
            line_signature = self._extract_line_signature(patch)
            clusters[line_signature].append(patch)
        
        return [
            PatchCluster(
                cluster_id=i,
                representative=patches[0],
                patches=patches,
                size=len(patches)
            )
            for i, (_, patches) in enumerate(clusters.items())
        ]
    
    def _extract_line_signature(self, patch: str) -> str:
        """Extract line-based signature from patch."""
        # Extract modified lines (lines starting with + or -)
        modified_lines = []
        for line in patch.split('\n'):
            if line.startswith('+') and not line.startswith('+++'):
                # Normalize: remove whitespace, lowercase
                normalized = re.sub(r'\s+', '', line[1:].lower())
                modified_lines.append(f'+{normalized}')
            elif line.startswith('-') and not line.startswith('---'):
                normalized = re.sub(r'\s+', '', line[1:].lower())
                modified_lines.append(f'-{normalized}')
        
        # Create signature
        return '|'.join(sorted(modified_lines))
    
    def _compute_entropy_from_clusters(
        self,
        clusters: List[PatchCluster],
        total_patches: int
    ) -> float:
        """
        Compute entropy from cluster distribution.
        
        H = -sum(p_i * log(p_i)) where p_i is the probability of cluster i
        
        Args:
            clusters: List of clusters
            total_patches: Total number of patches
            
        Returns:
            Entropy value
        """
        import math
        
        if not clusters or total_patches == 0:
            return 0.0
        
        entropy = 0.0
        
        for cluster in clusters:
            p = cluster.size / total_patches
            if p > 0:
                entropy -= p * math.log(p)
        
        # Normalize by log(N) to get value between 0 and 1
        if len(clusters) > 1:
            entropy /= math.log(len(clusters))
        
        return entropy
    
    def _compute_diversity(
        self,
        patches: List[str],
        clusters: List[PatchCluster]
    ) -> float:
        """
        Compute diversity score based on cluster distribution.
        
        Higher diversity means patches are more spread across clusters.
        
        Args:
            patches: List of patches
            clusters: List of clusters
            
        Returns:
            Diversity score (0.0-1.0)
        """
        if not patches or not clusters:
            return 0.0
        
        # Use normalized entropy as diversity
        total = len(patches)
        
        # Gini-Simpson diversity index
        diversity = 1.0
        for cluster in clusters:
            p = cluster.size / total
            diversity -= p * p
        
        return diversity
    
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
        """Tokenize code."""
        import re
        
        # Remove comments
        code = re.sub(r'#.*', '', code)
        code = re.sub(r'""".*?"""', '', code, flags=re.DOTALL)
        
        # Split on whitespace and punctuation
        tokens = re.findall(r'\w+|[^\w\s]', code)
        return [t for t in tokens if t.strip()]
    
    def _get_python_keywords(self) -> Set[str]:
        """Get Python keywords."""
        return {
            'def', 'class', 'if', 'else', 'elif', 'for', 'while', 'try',
            'except', 'finally', 'with', 'return', 'yield', 'import',
            'from', 'as', 'pass', 'break', 'continue', 'raise', 'assert',
            'lambda', 'global', 'nonlocal', 'del', 'in', 'is', 'not', 'or', 'and'
        }
    
    def get_name(self) -> str:
        """Get the metric name."""
        return 'semantic_entropy'
    
    def reset(self) -> None:
        """Reset the metric."""
        super().reset()
        self._embedding_cache = {}


class PerplexityMetric(Metric):
    """
    Metric for computing perplexity of generated patches.
    
    Perplexity measures how well a probability model predicts a sample.
    Lower perplexity indicates higher confidence.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self._perplexities: List[float] = []
    
    def compute(
        self,
        patch_result: 'PatchResult',
        instance: 'SWEInstance',
        evaluation_result: 'EvaluationResult'
    ) -> Dict[str, Any]:
        """
        Compute perplexity metric.
        
        Note: This requires log probabilities from the LLM, which may not
        always be available.
        """
        # Try to get log probabilities from metadata
        metadata = getattr(patch_result, 'metadata', {})
        logprobs = metadata.get('logprobs', [])
        
        if not logprobs:
            return {
                'perplexity': None,
                'avg_logprob': None,
                'note': 'Log probabilities not available'
            }
        
        # Compute perplexity
        import math
        
        avg_logprob = sum(logprobs) / len(logprobs)
        perplexity = math.exp(-avg_logprob)
        
        self._perplexities.append(perplexity)
        
        return {
            'perplexity': perplexity,
            'avg_logprob': avg_logprob,
            'min_logprob': min(logprobs),
            'max_logprob': max(logprobs)
        }
    
    def get_name(self) -> str:
        """Get the metric name."""
        return 'perplexity'
    
    def get_average_perplexity(self) -> float:
        """Get average perplexity across all computations."""
        return sum(self._perplexities) / len(self._perplexities) if self._perplexities else 0.0
    
    def reset(self) -> None:
        """Reset the metric."""
        super().reset()
        self._perplexities = []


# Forward reference imports
from ...patch_generators.base import PatchResult  # noqa: E402
from ...dataset.loader import SWEInstance  # noqa: E402
from ..base import EvaluationResult  # noqa: E402
