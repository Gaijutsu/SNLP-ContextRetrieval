# Usage Guide

This guide covers how to use the SWE-bench Comparison Framework for running experiments, configuring methods, and analyzing results.

---

## Running Experiments

### Basic Experiment

Run an experiment using a configuration file:

```bash
python -m swe_bench_framework run --config config.yaml
```

### Command-Line Options

```bash
python -m swe_bench_framework run \
    --config config.yaml \
    --output-dir ./results \
    --workers 4 \
    --instances 100 \
    --methods autocoderover bm25
```

| Option | Description | Default |
|--------|-------------|---------|
| `--config` | Path to configuration file | Required |
| `--output-dir` | Directory for results | From config |
| `--workers` | Number of parallel workers | 4 |
| `--instances` | Max instances to evaluate | All |
| `--methods` | Methods to run (override config) | All enabled |
| `--resume` | Resume from previous run | False |
| `--dry-run` | Validate config without running | False |

### Programmatic Usage

```python
from swe_bench_framework import ExperimentOrchestrator
from swe_bench_framework.config import ExperimentConfig

# Load configuration
config = ExperimentConfig.from_yaml('config.yaml')

# Create orchestrator
orchestrator = ExperimentOrchestrator(config)

# Run experiment
report = orchestrator.run()

# Access results
for method_name, metrics in report.summary.items():
    print(f"{method_name}: {metrics['resolution_rate']:.2%}")
```

---

## Configuration File Format

### Basic Structure

```yaml
# experiment_config.yaml

# Experiment metadata
experiment:
  name: "my_experiment"
  description: "Comparing agentic vs RAG methods"
  output_dir: "./results"
  random_seed: 42

# Dataset configuration
dataset:
  name: "swe-bench-lite"  # or "swe-bench-verified", "swe-bench-full"
  split: "test"
  filter:
    repos: ["django", "scikit-learn"]  # Optional: filter by repository
    max_instances: 100  # Optional: limit number of instances

# LLM configuration (shared across methods)
llm:
  provider: "openai"  # or "anthropic", "azure", "local"
  model: "gpt-4-turbo-preview"
  temperature: 0.0
  max_tokens: 4096
  top_p: 1.0
  api_key: "${OPENAI_API_KEY}"  # Environment variable
  rate_limit:
    requests_per_minute: 60
    tokens_per_minute: 150000

# Methods to compare
methods:
  - name: "autocoderover"
    type: "agentic"
    enabled: true
    config:
      max_iterations: 50
      tools:
        - search_class
        - search_method
        - view_file
        - grep
        - run_test

  - name: "bm25"
    type: "rag"
    enabled: true
    config:
      top_k: 20
      k1: 1.5
      b: 0.75

# Patch generation configuration
patch_generation:
  strategy: "direct"  # or "iterative", "edit_script"
  max_attempts: 3
  validation:
    syntax_check: true
    test_before_submit: false

# Evaluation configuration
evaluation:
  sandbox:
    type: "docker"
    timeout: 300
    memory_limit: "4g"
  metrics:
    - resolution_rate
    - localization_accuracy
    - codebleu
    - token_usage
    - semantic_entropy
  localization:
    k_values: [1, 3, 5, 10]

# Logging configuration
logging:
  level: "INFO"  # DEBUG, INFO, WARNING, ERROR
  format: "structured"  # or "text"
  outputs:
    - type: "file"
      path: "./logs"
    - type: "stdout"
  experiment_tracking:
    enabled: true
    backend: "mlflow"  # or "wandb"
    uri: "${MLFLOW_TRACKING_URI}"
```

### Method-Specific Configuration

#### AutoCodeRover

```yaml
methods:
  - name: "autocoderover"
    type: "agentic"
    enabled: true
    config:
      max_iterations: 50
      retrieval:
        strategy: "stratified"
        layers:
          - type: "file"
            top_k: 5
          - type: "class"
            top_k: 10
          - type: "function"
            top_k: 20
      sbfl:
        enabled: true
        formula: "ochiai"
      patch_validation:
        max_attempts: 3
        linter_enabled: true
```

#### SWE-agent

```yaml
methods:
  - name: "swe_agent"
    type: "agentic"
    enabled: true
    config:
      max_iterations: 100
      aci:
        enable_linter: true
        enable_test_runner: true
        file_viewer_lines: 100
      react:
        max_thought_length: 500
        max_observation_length: 2000
```

#### Agentless

```yaml
methods:
  - name: "agentless"
    type: "agentic"
    enabled: true
    config:
      localization:
        strategy: "hierarchical"
        file_top_k: 5
        function_top_k: 10
      repair:
        num_samples: 10
        temperature: 0.7
      validation:
        regression_testing: true
        majority_voting: true
```

#### BM25 RAG

```yaml
methods:
  - name: "bm25"
    type: "rag"
    enabled: true
    config:
      indexing:
        chunking:
          strategy: "ast"
          max_chunk_size: 500
          overlap: 50
        bm25:
          k1: 1.5
          b: 0.75
      retrieval:
        top_k: 20
        query_expansion: true
```

#### Dense RAG

```yaml
methods:
  - name: "dense"
    type: "rag"
    enabled: true
    config:
      indexing:
        chunking:
          strategy: "ast"
          max_chunk_size: 500
        embeddings:
          model: "jinaai/jina-embeddings-v2-base-code"
          batch_size: 32
          device: "cuda"  # or "cpu"
      retrieval:
        top_k: 20
        similarity_metric: "cosine"
```

#### Hybrid RAG

```yaml
methods:
  - name: "hybrid"
    type: "rag"
    enabled: true
    config:
      indexing:
        chunking:
          strategy: "ast"
          max_chunk_size: 500
        embeddings:
          model: "jinaai/jina-embeddings-v2-base-code"
          batch_size: 32
        bm25:
          k1: 1.5
          b: 0.75
      retrieval:
        sparse_weight: 0.3
        dense_weight: 0.7
        rrf_k: 60
        top_k: 20
        reranker:
          enabled: true
          model: "cross-encoder/ms-marco-MiniLM-L-6-v2"
          top_k: 10
```

---

## Command-Line Interface

### Main Commands

```bash
# Run experiment
python -m swe_bench_framework run --config config.yaml

# Generate report from results
python -m swe_bench_framework report --results-dir ./results

# Validate configuration
python -m swe_bench_framework validate --config config.yaml

# List available methods
python -m swe_bench_framework list-methods

# List available metrics
python -m swe_bench_framework list-metrics

# Clean up Docker resources
python -m swe_bench_framework cleanup
```

### Run Command Options

```bash
python -m swe_bench_framework run \
    --config config.yaml \           # Configuration file (required)
    --output-dir ./results \         # Output directory
    --workers 4 \                    # Parallel workers
    --instances 100 \                # Max instances
    --methods autocoderover bm25 \   # Methods to run
    --repos django flask \           # Filter by repository
    --resume \                       # Resume from checkpoint
    --dry-run                        # Validate only
```

### Report Command Options

```bash
python -m swe_bench_framework report \
    --results-dir ./results \        # Results directory
    --output report.html \           # Output file
    --format html \                  # Format: html, markdown, json
    --compare \                      # Generate comparison charts
    --significance                   # Statistical significance tests
```

---

## Example Workflows

### Workflow 1: Quick Comparison (Development)

Quickly compare two methods on a small subset:

```yaml
# quick_compare.yaml
experiment:
  name: "quick_compare"
  output_dir: "./results/quick"

dataset:
  name: "swe-bench-lite"
  filter:
    max_instances: 20  # Small subset

llm:
  provider: "openai"
  model: "gpt-3.5-turbo"  # Cheaper model for testing
  temperature: 0.0
  api_key: "${OPENAI_API_KEY}"

methods:
  - name: "bm25"
    type: "rag"
    enabled: true
    config:
      top_k: 10

  - name: "autocoderover"
    type: "agentic"
    enabled: true
    config:
      max_iterations: 20  # Fewer iterations

evaluation:
  sandbox:
    timeout: 120  # Shorter timeout
  metrics:
    - resolution_rate
```

Run:
```bash
python -m swe_bench_framework run --config quick_compare.yaml --workers 2
```

### Workflow 2: Full Evaluation

Comprehensive evaluation on SWE-bench Verified:

```yaml
# full_eval.yaml
experiment:
  name: "full_evaluation"
  output_dir: "./results/full"

dataset:
  name: "swe-bench-verified"

llm:
  provider: "openai"
  model: "gpt-4-turbo-preview"
  temperature: 0.0
  api_key: "${OPENAI_API_KEY}"

methods:
  - name: "bm25"
    type: "rag"
    enabled: true

  - name: "dense"
    type: "rag"
    enabled: true

  - name: "hybrid"
    type: "rag"
    enabled: true

  - name: "autocoderover"
    type: "agentic"
    enabled: true

  - name: "swe_agent"
    type: "agentic"
    enabled: true

  - name: "agentless"
    type: "agentic"
    enabled: true

evaluation:
  sandbox:
    timeout: 300
  metrics:
    - resolution_rate
    - localization_accuracy
    - codebleu
    - token_usage
    - semantic_entropy
```

Run:
```bash
python -m swe_bench_framework run --config full_eval.yaml --workers 8
```

### Workflow 3: Ablation Study

Study the effect of a specific parameter:

```python
# ablation_study.py
from swe_bench_framework import ExperimentOrchestrator
from swe_bench_framework.config import ExperimentConfig
import yaml

# Base configuration
base_config = {
    'experiment': {'name': 'ablation', 'output_dir': './results/ablation'},
    'dataset': {'name': 'swe-bench-lite', 'filter': {'max_instances': 50}},
    'llm': {'provider': 'openai', 'model': 'gpt-4', 'temperature': 0.0},
}

# Test different top_k values
for top_k in [5, 10, 20, 50]:
    config = base_config.copy()
    config['methods'] = [{
        'name': f'bm25_k{top_k}',
        'type': 'rag',
        'enabled': True,
        'config': {'top_k': top_k}
    }]
    
    # Save config
    with open(f'config_k{top_k}.yaml', 'w') as f:
        yaml.dump(config, f)
    
    # Run experiment
    exp_config = ExperimentConfig.from_yaml(f'config_k{top_k}.yaml')
    orchestrator = ExperimentOrchestrator(exp_config)
    report = orchestrator.run()
    
    print(f"top_k={top_k}: {report.summary[f'bm25_k{top_k}']['resolution_rate']:.2%}")
```

### Workflow 4: Resume Interrupted Run

If an experiment is interrupted, resume from checkpoint:

```bash
python -m swe_bench_framework run \
    --config config.yaml \
    --resume \
    --output-dir ./results/interrupted_run
```

### Workflow 5: Custom Method Evaluation

Evaluate a custom context gatherer:

```python
# custom_method.py
from swe_bench_framework.context_gatherers import ContextGatherer, ContextBundle
from swe_bench_framework.core.registry import register_gatherer

class MyCustomGatherer(ContextGatherer):
    def gather_context(self, instance, repo_path):
        # Your implementation
        return ContextBundle(...)
    
    def initialize(self, repo_path):
        pass
    
    def cleanup(self):
        pass

# Register
register_gatherer('my_custom', MyCustomGatherer)

# Run experiment
from swe_bench_framework import ExperimentOrchestrator
from swe_bench_framework.config import ExperimentConfig

config = ExperimentConfig.from_yaml('config_with_custom.yaml')
orchestrator = ExperimentOrchestrator(config)
report = orchestrator.run()
```

---

## Analyzing Results

### Accessing Results Programmatically

```python
import json

# Load results
with open('results/experiment/results.json', 'r') as f:
    results = json.load(f)

# Access per-instance results
for instance_result in results['instance_results']:
    instance_id = instance_result['instance_id']
    method = instance_result['method']
    resolved = instance_result['resolved']
    print(f"{instance_id} ({method}): {'✓' if resolved else '✗'}")

# Access summary statistics
for method, metrics in results['summary'].items():
    print(f"\n{method}:")
    print(f"  Resolution Rate: {metrics['resolution_rate']:.2%}")
    print(f"  Avg Recall@5: {metrics['avg_recall@5']:.2%}")
    print(f"  Avg CodeBLEU: {metrics['avg_codebleu']:.3f}")
```

### Generating Comparison Reports

```bash
# HTML report with charts
python -m swe_bench_framework report \
    --results-dir ./results \
    --output comparison.html \
    --format html \
    --compare

# Markdown report
python -m swe_bench_framework report \
    --results-dir ./results \
    --output comparison.md \
    --format markdown

# JSON for further analysis
python -m swe_bench_framework report \
    --results-dir ./results \
    --output comparison.json \
    --format json
```

### Statistical Analysis

```python
import scipy.stats as stats

# Compare two methods
method1_results = [r['resolved'] for r in results if r['method'] == 'autocoderover']
method2_results = [r['resolved'] for r in results if r['method'] == 'bm25']

# McNemar's test (paired binary outcome)
from statsmodels.stats.contingency_tables import mcnemar

# Create contingency table
both_correct = sum(m1 and m2 for m1, m2 in zip(method1_results, method2_results))
method1_only = sum(m1 and not m2 for m1, m2 in zip(method1_results, method2_results))
method2_only = sum(not m1 and m2 for m1, m2 in zip(method1_results, method2_results))
both_wrong = sum(not m1 and not m2 for m1, m2 in zip(method1_results, method2_results))

table = [[both_correct, method1_only], [method2_only, both_wrong]]
result = mcnemar(table, exact=True)
print(f"McNemar's test p-value: {result.pvalue}")
```

---

## Best Practices

### 1. Start Small

- Begin with SWE-bench Lite (300 instances)
- Use a subset (20-50 instances) for development
- Scale up once everything works

### 2. Version Control Configurations

```bash
# Save experiment configurations
git add configs/
git commit -m "Add experiment: comparing BM25 vs AutoCodeRover"

# Tag releases
git tag -a v1.0 -m "Full evaluation on SWE-bench Verified"
```

### 3. Monitor Resource Usage

```bash
# Monitor Docker disk usage
docker system df

# Monitor memory usage
htop

# Check API rate limits
python -m swe_bench_framework check-limits
```

### 4. Use Caching

```yaml
evaluation:
  sandbox:
    cache_level: "env"  # Speeds up repeated evaluations
```

### 5. Enable Comprehensive Logging

```yaml
logging:
  level: "INFO"
  format: "structured"
  outputs:
    - type: "file"
      path: "./logs"
```

---

## FAQ

### Q: How long does an experiment take?

**A**: Depends on:
- Number of instances
- Number of methods
- LLM API speed
- Docker evaluation time

Rough estimates (with 4 workers):
- 50 instances, 2 methods: ~2-4 hours
- 300 instances (Lite), 6 methods: ~12-24 hours
- 500 instances (Verified), 6 methods: ~24-48 hours

### Q: How much does it cost?

**A**: Depends on:
- LLM model used
- Number of instances
- Method efficiency

Rough estimates (GPT-4):
- SWE-bench Lite (300 instances): $50-200
- SWE-bench Verified (500 instances): $100-400

### Q: Can I use local models?

**A**: Yes! Configure in your config:
```yaml
llm:
  provider: "local"
  model: "meta-llama/Llama-2-70b"
  url: "http://localhost:8000/v1"
```

### Q: How do I add a new method?

**A**: See the [Architecture Documentation](architecture.md#extension-points) for details on extending the framework.

### Q: Can I run on cloud infrastructure?

**A**: Yes, the framework supports:
- Modal cloud evaluation
- Custom cloud providers via Docker
- Distributed execution

### Q: How do I debug failures?

**A**: 
1. Check logs in `./logs/`
2. Enable DEBUG logging: `logging: { level: "DEBUG" }`
3. Run single instance: `--instances sympy__sympy-20590`
4. Check Docker logs: `docker logs <container_id>`

### Q: What if Docker runs out of space?

**A**:
```bash
# Clean up Docker
docker system prune -a

# Increase Docker disk size in Docker Desktop settings
# Or use external volume for cache
```

---

## Troubleshooting

See the [Installation Guide](installation.md#troubleshooting) for common issues and solutions.
