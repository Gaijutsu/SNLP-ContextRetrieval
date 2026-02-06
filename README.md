# SWE-bench Comparison Framework

A modular framework for comparing **agentic exploration** vs **RAG (Retrieval-Augmented Generation)** methods for automated software patching on the [SWE-bench](https://www.swebench.com/) benchmark.

## Overview

This framework enables systematic comparison of different approaches to automated software engineering:

- **Agentic Methods**: AutoCodeRover, SWE-agent, Agentless
- **RAG Methods**: BM25, Dense Embeddings, Hybrid Retrieval
- **Hybrid Methods**: Combining agentic exploration with RAG context

## Features

- **Modular Architecture**: Easy to extend with new methods and metrics
- **Multiple Datasets**: Support for SWE-bench Lite, Verified, and Full
- **Comprehensive Metrics**: Resolution rate, Pass@k, localization, CodeBLEU, token usage
- **Flexible Evaluation**: Local or Docker-based sandbox execution
- **Rich Reporting**: JSON, Markdown, HTML, and CSV reports with visualizations
- **Experiment Tracking**: Optional MLflow and Weights & Biases integration

## Installation

### Quick Install

```bash
# Clone the repository
git clone https://github.com/your-org/swe-bench-comparison.git
cd swe-bench-comparison

# Install the package
pip install -e .
`
# Or install with optional dependencies
pip install -e ".[all]"
```

### Development Install

```bash
# Install with development dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

### Environment Setup

```bash
# Copy environment template
cp .env.example .env

# Edit .env with your API keys
nano .env
```

Required environment variables:
- `OPENAI_API_KEY` - for GPT-4/GPT-3.5 models
- `ANTHROPIC_API_KEY` - for Claude models (optional)

## Quick Start

### 1. Initialize a Configuration

```bash
# Create a basic configuration
swe-compare init --template basic -o my_config.yaml

# Or use one of the provided templates
cp configs/basic_comparison.yaml my_config.yaml
```

### 2. Run an Experiment

```bash
# Run from config file
swe-compare run --config my_config.yaml

# Or use the simple runner
python run_experiment.py --config my_config.yaml
```

### 3. Generate Report

```bash
# Generate report from results
swe-compare report --results ./results/my_experiment/results.json
```

## Configuration

### Example: Basic Comparison

```yaml
experiment_name: "basic_comparison"
output_dir: "./results/basic_comparison"
dataset: "lite"

methods:
  agentless:
    type: "agentic"
    strategy: "agentless"
    model: "gpt-4"
    
  hybrid_rag:
    type: "rag"
    retrieval: "hybrid"
    model: "gpt-4"
    rag_config:
      top_k: 10
      embedding_model: "sentence-transformers/all-MiniLM-L6-v2"

evaluation:
  metrics:
    - "resolution"
    - "pass_at_k"
    - "token_usage"
```

See `configs/` directory for more examples:
- `minimal.yaml` - Quick testing on a few instances
- `basic_comparison.yaml` - Agentless vs Hybrid RAG
- `rag_only.yaml` - Compare different RAG configurations
- `agentic_only.yaml` - Compare different agentic methods
- `full_comparison.yaml` - Comprehensive comparison of all methods

## CLI Commands

### Run Experiment

```bash
# Run from config
swe-compare run --config config.yaml

# Run with overrides
swe-compare run --config config.yaml --name "my_run" --output-dir ./results

# Run on specific instances
swe-compare run --config config.yaml --instances "django-1234,flask-5678"

# Dry run (show what would be executed)
swe-compare run --config config.yaml --dry-run
```

### Compare Multiple Methods

```bash
# Compare multiple configurations
swe-compare compare --configs config1.yaml config2.yaml --output-dir ./comparison
```

### Evaluate Predictions

```bash
# Evaluate existing predictions
swe-compare evaluate --predictions predictions.json --dataset lite

# Save evaluation results
swe-compare evaluate --predictions preds.json --output eval_results.json
```

### Build RAG Index

```bash
# Build hybrid index for a repository
swe-compare index --repo-path /path/to/repo --index-type hybrid

# Build with custom embedding model
swe-compare index -r /path/to/repo --index-type dense \
    --embedding-model sentence-transformers/all-mpnet-base-v2
```

### Generate Report

```bash
# Generate report in multiple formats
swe-compare report --results results.json --formats json,markdown,html

# Use custom template
swe-compare report --results results.json --template custom_template.html
```

## Python API

### Basic Usage

```python
from swe_bench_framework import (
    ExperimentOrchestrator,
    OrchestratorConfig,
    SWEBenchLiteLoader,
    SWEBenchEvaluator,
)

# Create orchestrator
config = OrchestratorConfig(
    experiment_name="my_experiment",
    output_dir="./results"
)
orchestrator = ExperimentOrchestrator(config)

# Register methods
orchestrator.register_method(
    name="My Method",
    patch_generator=my_generator,
    context_gatherer=my_gatherer
)

# Run experiment
results = orchestrator.run(
    dataset_loader=SWEBenchLiteLoader(),
    evaluator=SWEBenchEvaluator(),
    repo_provider=lambda inst: f"./repos/{inst.repo}"
)

# Generate report
orchestrator.generate_report()
```

### Using the Builder Pattern

```python
from swe_bench_framework import ExperimentBuilder

# Build experiment declaratively
experiment = (ExperimentBuilder()
    .with_name("My Experiment")
    .with_output_dir("./results")
    .with_dataset("lite")
    .with_method("RAG", rag_generator, rag_gatherer)
    .with_method("Agentic", agent_generator, agent_gatherer)
    .with_parallel(max_workers=4)
    .build())

# Run experiment
results = experiment.run(
    dataset_loader=SWEBenchLiteLoader(),
    evaluator=SWEBenchEvaluator(),
    repo_provider=repo_provider
)
```

## Project Structure

```
swe_bench_framework/
├── cli.py                 # Command-line interface
├── config/                # Configuration management
│   ├── loader.py
│   └── validator.py
├── core/                  # Core abstractions
│   ├── base.py
│   └── types.py
├── dataset/               # Dataset loaders
│   ├── loader.py
│   └── swe_bench_loader.py
├── patch_generators/      # Patch generation methods
│   ├── base.py
│   ├── agentic/
│   └── rag/
├── context_gatherers/     # Context gathering methods
│   ├── base.py
│   ├── agentic_explorer.py
│   └── retrieval/
├── evaluation/            # Evaluation and metrics
│   ├── base.py
│   ├── swe_bench_evaluator.py
│   └── metrics/
├── experiment/            # Experiment orchestration
│   ├── orchestrator.py
│   └── runner.py
└── utils/                 # Utility functions
    ├── repo_manager.py
    └── logging.py

configs/                   # Example configurations
├── minimal.yaml
├── basic_comparison.yaml
├── rag_only.yaml
├── agentic_only.yaml
└── full_comparison.yaml

scripts/                   # Utility scripts
├── download_swe_bench.py
├── build_indexes.py
└── visualize_results.py
```

## Supported Methods

### Agentic Methods

| Method | Description | Paper |
|--------|-------------|-------|
| AutoCodeRover | Structured exploration with search | [arXiv:2403.16292](https://arxiv.org/abs/2403.16292) |
| SWE-agent | Full agent-based approach | [arXiv:2405.15793](https://arxiv.org/abs/2405.15793) |
| Agentless | Agentic without explicit agent | [arXiv:2407.01489](https://arxiv.org/abs/2407.01489) |

### RAG Methods

| Method | Description |
|--------|-------------|
| BM25 | Traditional keyword-based retrieval |
| Dense | Neural embedding-based retrieval |
| Hybrid | Combines BM25 and dense retrieval |

## Evaluation Metrics

- **Resolution Rate**: Percentage of instances successfully resolved
- **Pass@k**: Pass rate at k attempts
- **Localization**: Accuracy of bug localization
- **CodeBLEU**: Code similarity metric
- **Token Usage**: Number of tokens consumed
- **Semantic Entropy**: Diversity of generated patches

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development Setup

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run with coverage
pytest --cov=swe_bench_framework

# Format code
black swe_bench_framework/

# Type checking
mypy swe_bench_framework/
```

## Acknowledgments

- [SWE-bench](https://www.swebench.com/) - The benchmark for software engineering
- [AutoCodeRover](https://github.com/nus-apr/auto-code-rover) - Agentic approach
- [SWE-agent](https://github.com/princeton-nlp/SWE-agent) - Agent-based approach
- [Agentless](https://github.com/princeton-nlp/Agentless) - Agentless approach

## Troubleshooting

### Common Issues

**Issue**: `ModuleNotFoundError: No module named 'swe_bench_framework'`

**Solution**: Install the package in editable mode:
```bash
pip install -e .
```

**Issue**: `API key not found`

**Solution**: Set your API keys in the environment:
```bash
export OPENAI_API_KEY=your_key_here
```

**Issue**: `Docker not found`

**Solution**: Either install Docker or use local sandbox:
```yaml
sandbox: "local"  # in your config file
```
