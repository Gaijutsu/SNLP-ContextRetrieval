# Evaluation API Documentation

This document describes the Evaluation API, which provides interfaces and implementations for evaluating generated patches and computing metrics.

---

## Overview

The `Evaluator` interface defines how patches are evaluated against the SWE-bench test suite. The framework includes multiple metric implementations for comprehensive evaluation.

```python
from swe_bench_framework.evaluation import Evaluator, SWEBenchEvaluator

evaluator = SWEBenchEvaluator(config)
result = evaluator.evaluate(patch_result, instance, repo_path)
print(f"Resolved: {result.resolved}")
```

---

## Evaluator Interface

### Class Definition

```python
class Evaluator(ABC):
    """
    Abstract base class for evaluation strategies.
    
    This interface defines how patches are evaluated against the
    SWE-bench test suite and how metrics are computed.
    
    Example:
        >>> evaluator = MyEvaluator(config)
        >>> result = evaluator.evaluate(patch_result, instance, repo_path)
        >>> print(f"Resolved: {result.resolved}")
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the evaluator.
        
        Args:
            config: Configuration dictionary for the evaluator
        """
        self.config = config
        self.name = self.__class__.__name__
```

### Abstract Methods

#### `evaluate()`

```python
@abstractmethod
def evaluate(
    self,
    patch_result: PatchResult,
    instance: SWEInstance,
    repo_path: str
) -> EvaluationResult:
    """
    Evaluate a generated patch.
    
    This is the main method that all evaluators must implement.
    It should apply the patch, run tests, and compute all metrics.
    
    Args:
        patch_result: The result from patch generation
        instance: The SWE-bench instance
        repo_path: Path to the repository
        
    Returns:
        EvaluationResult with all metrics
        
    Raises:
        EvaluationError: If evaluation fails
    """
    pass
```

#### `compute_localization_accuracy()`

```python
@abstractmethod
def compute_localization_accuracy(
    self,
    context_bundle: ContextBundle,
    gold_files: List[str],
    gold_functions: List[str]
) -> Dict[str, float]:
    """
    Compute localization accuracy metrics.
    
    This method computes metrics like recall@k for file and
    function localization accuracy.
    
    Args:
        context_bundle: The gathered context
        gold_files: Ground truth list of modified files
        gold_functions: Ground truth list of modified functions
        
    Returns:
        Dictionary with localization metrics (e.g., recall@1, recall@5)
    """
    pass
```

### Utility Methods

#### `compute_codebleu()`

```python
def compute_codebleu(
    self,
    generated_patch: str,
    gold_patch: str
) -> float:
    """
    Compute CodeBLEU score between generated and gold patch.
    
    Args:
        generated_patch: The generated patch
        gold_patch: The gold patch
        
    Returns:
        CodeBLEU score (0.0-1.0)
    """
    # Default implementation - subclasses can override
    return 0.0
```

---

## Data Models

### EvaluationResult

```python
@dataclass
class EvaluationResult:
    """
    Result of evaluating a patch.
    
    Contains all metrics and information about the evaluation.
    """
    
    instance_id: str
    """Unique identifier for the SWE-bench instance."""
    
    resolved: bool
    """Whether the patch successfully resolved the issue."""
    
    patch_applied: bool
    """Whether the patch was successfully applied."""
    
    tests_passed: List[str]
    """List of test names that passed."""
    
    tests_failed: List[str]
    """List of test names that failed."""
    
    localization_accuracy: Dict[str, float]
    """Localization accuracy metrics (recall@k, etc.)."""
    
    codebleu_score: float
    """CodeBLEU score comparing to gold patch."""
    
    execution_time: float
    """Time taken for evaluation (seconds)."""
    
    metadata: Dict[str, Any] = None
    """Additional metadata (error messages, logs, etc.)."""
```

---

## SWE-bench Evaluator

```python
class SWEBenchEvaluator(Evaluator):
    """
    Standard SWE-bench evaluation pipeline.
    
    This evaluator implements the official SWE-bench evaluation protocol:
    1. Apply patch in Docker sandbox
    2. Run test suite
    3. Check FAIL_TO_PASS and PASS_TO_PASS tests
    4. Compute all metrics
    
    Example:
        >>> config = {
        ...     'sandbox': {
        ...         'type': 'docker',
        ...         'timeout': 300,
        ...         'memory_limit': '4g'
        ...     },
        ...     'metrics': [
        ...         'resolution_rate',
        ...         'localization_accuracy',
        ...         'codebleu',
        ...         'token_usage'
        ...     ]
        ... }
        >>> evaluator = SWEBenchEvaluator(config)
        >>> result = evaluator.evaluate(patch_result, instance, repo_path)
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.sandbox = DockerSandbox(config.get('sandbox', {}))
        self.metrics_collectors = self._create_metrics_collectors(
            config.get('metrics', [])
        )
    
    def evaluate(
        self,
        patch_result: PatchResult,
        instance: SWEInstance,
        repo_path: str
    ) -> EvaluationResult:
        """
        Evaluate a patch using SWE-bench protocol.
        
        Process:
        1. Check if patch exists
        2. Apply patch in Docker sandbox
        3. Run test suite
        4. Compute metrics
        5. Return evaluation result
        """
        start_time = time.time()
        
        # Check if patch exists
        if not patch_result.patch_content:
            return EvaluationResult(
                instance_id=instance.instance_id,
                resolved=False,
                patch_applied=False,
                tests_passed=[],
                tests_failed=instance.failed_tests,
                localization_accuracy={},
                codebleu_score=0.0,
                execution_time=0.0,
                metadata={'error': 'No patch generated'}
            )
        
        # Apply patch
        applied = self.sandbox.apply_patch(
            patch_result.patch_content,
            repo_path
        )
        
        if not applied:
            return EvaluationResult(
                instance_id=instance.instance_id,
                resolved=False,
                patch_applied=False,
                tests_passed=[],
                tests_failed=instance.failed_tests,
                localization_accuracy={},
                codebleu_score=0.0,
                execution_time=time.time() - start_time,
                metadata={'error': 'Failed to apply patch'}
            )
        
        # Run tests
        test_results = self.sandbox.run_tests(instance, repo_path)
        
        # Check if resolved
        resolved = self._check_resolved(test_results, instance)
        
        # Compute localization accuracy
        localization = self.compute_localization_accuracy(
            patch_result.context_bundle,
            instance.modified_files,
            instance.modified_methods
        )
        
        # Compute CodeBLEU
        codebleu = self.compute_codebleu(
            patch_result.patch_content,
            instance.patch
        )
        
        # Collect additional metrics
        metadata = {}
        for collector in self.metrics_collectors:
            metrics = collector.compute(patch_result, instance, test_results)
            metadata.update(metrics)
        
        return EvaluationResult(
            instance_id=instance.instance_id,
            resolved=resolved,
            patch_applied=True,
            tests_passed=test_results.passed,
            tests_failed=test_results.failed,
            localization_accuracy=localization,
            codebleu_score=codebleu,
            execution_time=time.time() - start_time,
            metadata=metadata
        )
    
    def _check_resolved(
        self,
        test_results: TestResults,
        instance: SWEInstance
    ) -> bool:
        """
        Check if the issue is resolved.
        
        Resolution criteria:
        1. All FAIL_TO_PASS tests pass
        2. All PASS_TO_PASS tests pass
        """
        # Check FAIL_TO_PASS tests
        for test in instance.failed_tests:
            if test not in test_results.passed:
                return False
        
        # Check PASS_TO_PASS tests
        for test in instance.passed_tests:
            if test in test_results.failed:
                return False
        
        return True
    
    def compute_localization_accuracy(
        self,
        context_bundle: ContextBundle,
        gold_files: List[str],
        gold_functions: List[str]
    ) -> Dict[str, float]:
        """
        Compute localization accuracy metrics.
        
        Returns:
            Dictionary with recall@k for files and functions
        """
        results = {}
        
        # Extract predicted files and functions from context
        predicted_files = self._extract_predicted_files(context_bundle)
        predicted_functions = self._extract_predicted_functions(context_bundle)
        
        # Compute Recall@k for different k values
        k_values = self.config.get('localization', {}).get('k_values', [1, 3, 5, 10])
        
        for k in k_values:
            # File-level Recall@k
            file_recall = self._compute_recall_at_k(
                predicted_files[:k],
                gold_files
            )
            results[f'file_recall@{k}'] = file_recall
            
            # Function-level Recall@k
            function_recall = self._compute_recall_at_k(
                predicted_functions[:k],
                gold_functions
            )
            results[f'function_recall@{k}'] = function_recall
        
        # Hit@k (binary indicator)
        for k in k_values:
            file_hit = self._compute_hit_at_k(predicted_files[:k], gold_files)
            results[f'file_hit@{k}'] = float(file_hit)
            
            function_hit = self._compute_hit_at_k(
                predicted_functions[:k],
                gold_functions
            )
            results[f'function_hit@{k}'] = float(function_hit)
        
        return results
    
    def _compute_recall_at_k(
        self,
        predicted: List[str],
        gold: List[str]
    ) -> float:
        """Compute Recall@k."""
        if not gold:
            return 1.0
        
        predicted_set = set(predicted)
        gold_set = set(gold)
        
        correct = len(predicted_set & gold_set)
        return correct / len(gold_set)
    
    def _compute_hit_at_k(
        self,
        predicted: List[str],
        gold: List[str]
    ) -> bool:
        """Compute Hit@k (binary indicator)."""
        predicted_set = set(predicted)
        gold_set = set(gold)
        
        return len(predicted_set & gold_set) > 0
    
    def compute_codebleu(
        self,
        generated_patch: str,
        gold_patch: str
    ) -> float:
        """
        Compute CodeBLEU score.
        
        CodeBLEU combines:
        - N-gram match (textual similarity)
        - Weighted n-gram match (syntax importance)
        - Syntax AST match (structural similarity)
        - Data-flow match (semantic similarity)
        """
        from swe_bench_framework.evaluation.metrics.codebleu import compute_codebleu
        
        return compute_codebleu(generated_patch, gold_patch)
```

**Configuration Options:**

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `sandbox.type` | str | "docker" | Sandbox type (docker) |
| `sandbox.timeout` | int | 300 | Test execution timeout (seconds) |
| `sandbox.memory_limit` | str | "4g" | Memory limit for sandbox |
| `metrics` | list | ["resolution_rate"] | Metrics to compute |
| `localization.k_values` | list | [1, 3, 5, 10] | K values for Recall@k |

---

## Metrics

### Resolution Metric

```python
class ResolutionMetric(Metric):
    """
    Primary metric: percentage of issues resolved.
    
    This is the gold standard metric for automated program repair.
    An issue is considered resolved if:
    1. All FAIL_TO_PASS tests pass
    2. All PASS_TO_PASS tests still pass
    """
    
    def compute(
        self,
        patch_result: PatchResult,
        instance: SWEInstance,
        evaluation_result: EvaluationResult
    ) -> Dict[str, float]:
        """
        Compute resolution metric.
        
        Returns:
            Dictionary with 'resolved' (0.0 or 1.0)
        """
        return {
            'resolved': 1.0 if evaluation_result.resolved else 0.0,
            'patch_applied': 1.0 if evaluation_result.patch_applied else 0.0
        }
    
    def get_name(self) -> str:
        return 'resolution'
    
    def aggregate(self, results: List[Dict[str, float]]) -> Dict[str, float]:
        """Aggregate resolution rate across instances."""
        if not results:
            return {}
        
        resolved_count = sum(r.get('resolved', 0) for r in results)
        applied_count = sum(r.get('patch_applied', 0) for r in results)
        
        return {
            'resolution_rate': resolved_count / len(results),
            'patch_apply_rate': applied_count / len(results),
            'total_instances': len(results)
        }
```

### Localization Metric

```python
class LocalizationMetric(Metric):
    """
    Localization accuracy metrics.
    
    Measures how accurately the method identified the
    files and functions that need to be modified.
    """
    
    def compute(
        self,
        patch_result: PatchResult,
        instance: SWEInstance,
        evaluation_result: EvaluationResult
    ) -> Dict[str, float]:
        """
        Compute localization metrics.
        
        Returns:
            Dictionary with recall@k and hit@k for files and functions
        """
        return evaluation_result.localization_accuracy
    
    def get_name(self) -> str:
        return 'localization'
    
    def aggregate(self, results: List[Dict[str, float]]) -> Dict[str, float]:
        """Aggregate localization metrics across instances."""
        if not results:
            return {}
        
        aggregated = {}
        
        # Collect all metric keys
        keys = set()
        for r in results:
            keys.update(r.keys())
        
        # Compute mean for each metric
        for key in keys:
            values = [r[key] for r in results if key in r]
            if values:
                aggregated[f'avg_{key}'] = sum(values) / len(values)
        
        return aggregated
```

### CodeBLEU Metric

```python
class CodeBLEUMetric(Metric):
    """
    CodeBLEU metric for code similarity.
    
    CodeBLEU combines multiple similarity measures:
    - N-gram match (textual similarity)
    - Weighted n-gram match (syntax importance)
    - Syntax AST match (structural similarity)
    - Data-flow match (semantic similarity)
    
    Reference: "CodeBLEU: a Method for Automatic Evaluation of Code Synthesis"
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.alpha = config.get('alpha', 0.25)  # N-gram weight
        self.beta = config.get('beta', 0.25)    # Weighted n-gram weight
        self.gamma = config.get('gamma', 0.25)  # Syntax weight
        self.delta = config.get('delta', 0.25)  # Data-flow weight
    
    def compute(
        self,
        patch_result: PatchResult,
        instance: SWEInstance,
        evaluation_result: EvaluationResult
    ) -> Dict[str, float]:
        """
        Compute CodeBLEU score.
        
        Returns:
            Dictionary with 'codebleu' score (0.0-1.0)
        """
        if not patch_result.patch_content or not instance.patch:
            return {'codebleu': 0.0}
        
        score = compute_codebleu(
            patch_result.patch_content,
            instance.patch,
            alpha=self.alpha,
            beta=self.beta,
            gamma=self.gamma,
            delta=self.delta
        )
        
        return {'codebleu': score}
    
    def get_name(self) -> str:
        return 'codebleu'
    
    def aggregate(self, results: List[Dict[str, float]]) -> Dict[str, float]:
        """Aggregate CodeBLEU scores."""
        if not results:
            return {}
        
        scores = [r.get('codebleu', 0) for r in results]
        return {
            'avg_codebleu': sum(scores) / len(scores),
            'median_codebleu': sorted(scores)[len(scores) // 2],
            'min_codebleu': min(scores),
            'max_codebleu': max(scores)
        }
```

### Token Usage Metric

```python
class TokenUsageMetric(Metric):
    """
    Token usage and cost metric.
    
    Tracks API usage for cost analysis.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.cost_per_1k_input = config.get('cost_per_1k_input', 0.01)
        self.cost_per_1k_output = config.get('cost_per_1k_output', 0.03)
    
    def compute(
        self,
        patch_result: PatchResult,
        instance: SWEInstance,
        evaluation_result: EvaluationResult
    ) -> Dict[str, float]:
        """
        Compute token usage metrics.
        
        Returns:
            Dictionary with token counts and estimated cost
        """
        token_usage = patch_result.token_usage
        
        input_tokens = token_usage.get('input', 0)
        output_tokens = token_usage.get('output', 0)
        total_tokens = token_usage.get('total', input_tokens + output_tokens)
        
        # Estimate cost
        input_cost = (input_tokens / 1000) * self.cost_per_1k_input
        output_cost = (output_tokens / 1000) * self.cost_per_1k_output
        total_cost = input_cost + output_cost
        
        return {
            'input_tokens': float(input_tokens),
            'output_tokens': float(output_tokens),
            'total_tokens': float(total_tokens),
            'estimated_cost': total_cost
        }
    
    def get_name(self) -> str:
        return 'token_usage'
    
    def aggregate(self, results: List[Dict[str, float]]) -> Dict[str, float]:
        """Aggregate token usage across instances."""
        if not results:
            return {}
        
        total_tokens = [r.get('total_tokens', 0) for r in results]
        costs = [r.get('estimated_cost', 0) for r in results]
        
        return {
            'avg_tokens': sum(total_tokens) / len(total_tokens),
            'total_tokens': sum(total_tokens),
            'avg_cost': sum(costs) / len(costs),
            'total_cost': sum(costs)
        }
```

### Semantic Entropy Metric

```python
class SemanticEntropyMetric(Metric):
    """
    Semantic entropy for uncertainty detection.
    
    Measures the diversity of generated patches for the same input.
    High entropy indicates uncertainty or hallucination.
    
    Reference: "Detecting Hallucinations in Large Language Models Using
    Semantic Entropy" (Farquhar et al., 2024)
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.num_samples = config.get('num_samples', 5)
        self.embedding_model = config.get(
            'embedding_model',
            'all-MiniLM-L6-v2'
        )
        self.clustering_threshold = config.get('clustering_threshold', 0.9)
        self._load_embedding_model()
    
    def compute(
        self,
        patch_result: PatchResult,
        instance: SWEInstance,
        evaluation_result: EvaluationResult
    ) -> Dict[str, float]:
        """
        Compute semantic entropy.
        
        Process:
        1. Generate multiple patches for same context
        2. Embed each patch
        3. Cluster by semantic similarity
        4. Compute entropy across clusters
        
        Returns:
            Dictionary with 'semantic_entropy' and 'num_clusters'
        """
        # This requires regenerating patches with sampling
        # For efficiency, we can use cached samples
        
        if not hasattr(patch_result, 'sampled_patches'):
            return {'semantic_entropy': 0.0, 'num_clusters': 1}
        
        patches = patch_result.sampled_patches
        
        if len(patches) < 2:
            return {'semantic_entropy': 0.0, 'num_clusters': 1}
        
        # Embed patches
        embeddings = [self._embed_patch(p) for p in patches]
        
        # Cluster by similarity
        clusters = self._cluster_embeddings(
            embeddings,
            threshold=self.clustering_threshold
        )
        
        # Compute entropy
        cluster_probs = [len(c) / len(patches) for c in clusters]
        entropy = -sum(p * math.log(p) for p in cluster_probs if p > 0)
        
        return {
            'semantic_entropy': entropy,
            'num_clusters': len(clusters),
            'max_cluster_size': max(len(c) for c in clusters),
            'normalized_entropy': entropy / math.log(len(clusters)) if clusters else 0
        }
    
    def _embed_patch(self, patch: str) -> np.ndarray:
        """Embed patch using sentence transformer."""
        return self.model.encode(patch)
    
    def _cluster_embeddings(
        self,
        embeddings: List[np.ndarray],
        threshold: float
    ) -> List[List[int]]:
        """Cluster embeddings by cosine similarity."""
        n = len(embeddings)
        clusters = []
        assigned = set()
        
        for i in range(n):
            if i in assigned:
                continue
            
            cluster = [i]
            assigned.add(i)
            
            for j in range(i + 1, n):
                if j in assigned:
                    continue
                
                similarity = cosine_similarity(
                    embeddings[i].reshape(1, -1),
                    embeddings[j].reshape(1, -1)
                )[0, 0]
                
                if similarity >= threshold:
                    cluster.append(j)
                    assigned.add(j)
            
            clusters.append(cluster)
        
        return clusters
    
    def get_name(self) -> str:
        return 'semantic_entropy'
    
    def aggregate(self, results: List[Dict[str, float]]) -> Dict[str, float]:
        """Aggregate semantic entropy across instances."""
        if not results:
            return {}
        
        entropies = [r.get('semantic_entropy', 0) for r in results]
        num_clusters = [r.get('num_clusters', 1) for r in results]
        
        return {
            'avg_semantic_entropy': sum(entropies) / len(entropies),
            'avg_num_clusters': sum(num_clusters) / len(num_clusters),
            'high_entropy_instances': sum(1 for e in entropies if e > 1.0)
        }
```

**Configuration Options:**

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `num_samples` | int | 5 | Number of patch samples to generate |
| `embedding_model` | str | "all-MiniLM-L6-v2" | Model for embedding patches |
| `clustering_threshold` | float | 0.9 | Cosine similarity threshold for clustering |

---

## Docker Sandbox

```python
class DockerSandbox:
    """
    Docker-based sandbox for test execution.
    
    Provides isolated environment for running tests
    with configurable resource limits.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.timeout = config.get('timeout', 300)
        self.memory_limit = config.get('memory_limit', '4g')
        self.image = config.get('image', 'swe-bench/sandbox:latest')
        self.cache_level = config.get('cache_level', 'env')
    
    def apply_patch(self, patch_content: str, repo_path: str) -> bool:
        """
        Apply a patch to the repository.
        
        Args:
            patch_content: Unified diff patch content
            repo_path: Path to the repository
            
        Returns:
            True if patch was applied successfully
        """
        try:
            result = subprocess.run(
                ['git', 'apply', '-'],
                input=patch_content,
                cwd=repo_path,
                capture_output=True,
                text=True,
                timeout=30
            )
            return result.returncode == 0
        except Exception:
            return False
    
    def run_tests(
        self,
        instance: SWEInstance,
        repo_path: str
    ) -> TestResults:
        """
        Run test suite for the instance.
        
        Args:
            instance: SWE-bench instance with test information
            repo_path: Path to the repository
            
        Returns:
            TestResults with passed and failed tests
        """
        # Build test command
        test_cmd = instance.test_cmd
        
        # Run in Docker container
        container = self._create_container(repo_path)
        
        try:
            result = container.exec_run(
                test_cmd,
                timeout=self.timeout
            )
            
            # Parse test output
            passed, failed = self._parse_test_output(result.output)
            
            return TestResults(
                passed=passed,
                failed=failed,
                return_code=result.exit_code,
                output=result.output
            )
        finally:
            container.stop()
            container.remove()
```

---

## Factory and Registration

### Creating Evaluators

```python
from swe_bench_framework.evaluation import EvaluatorFactory

# Create by name
evaluator = EvaluatorFactory.create('swe_bench', config)

# Create from config
config = {'type': 'swe_bench', 'sandbox': {'timeout': 300}}
evaluator = EvaluatorFactory.create_from_config(config)
```

### Registering Custom Metrics

```python
from swe_bench_framework.evaluation import Metric
from swe_bench_framework.core.registry import register_metric

# Define custom metric
class MyCustomMetric(Metric):
    def compute(self, patch_result, instance, evaluation_result):
        # Your computation
        return {'my_metric': value}
    
    def get_name(self):
        return 'my_metric'

# Register
register_metric('my_metric', MyCustomMetric)

# Use in config
evaluation:
  metrics:
    - resolution_rate
    - my_metric
```

---

## Best Practices

### 1. Always Use Docker for Evaluation

```python
# Good
evaluator = SWEBenchEvaluator({
    'sandbox': {'type': 'docker', 'timeout': 300}
})

# Not recommended (may not be reproducible)
evaluator = SWEBenchEvaluator({
    'sandbox': {'type': 'local'}
})
```

### 2. Handle Evaluation Failures Gracefully

```python
try:
    result = evaluator.evaluate(patch_result, instance, repo_path)
except EvaluationError as e:
    logger.error(f"Evaluation failed: {e}")
    result = EvaluationResult(
        instance_id=instance.instance_id,
        resolved=False,
        patch_applied=False,
        tests_passed=[],
        tests_failed=instance.failed_tests,
        localization_accuracy={},
        codebleu_score=0.0,
        execution_time=0.0,
        metadata={'error': str(e)}
    )
```

### 3. Compute Multiple Metrics

```python
evaluation:
  metrics:
    - resolution_rate      # Primary metric
    - localization_accuracy # Diagnostic
    - codebleu             # Code quality
    - token_usage          # Cost analysis
    - semantic_entropy     # Uncertainty
```

### 4. Report Confidence Intervals

```python
from scipy import stats

# Compute confidence interval for resolution rate
results = [r['resolved'] for r in evaluation_results]
n = len(results)
p = sum(results) / n

# 95% confidence interval
ci = stats.proportion_confint(sum(results), n, alpha=0.05, method='wilson')

print(f"Resolution Rate: {p:.2%} (95% CI: {ci[0]:.2%} - {ci[1]:.2%})")
```
