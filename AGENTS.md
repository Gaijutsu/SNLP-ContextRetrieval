# SWE-bench Comparison Framework - Agent Guide

This document provides essential information for AI coding agents working on the SWE-bench Comparison Framework project.

---

## Project Overview

The **SWE-bench Comparison Framework** is a modular Python framework for comparing **agentic exploration** vs **RAG (Retrieval-Augmented Generation)** methods for automated software patching on the [SWE-bench](https://www.swebench.com/) benchmark.

### Core Purpose

The framework enables systematic comparison of different approaches to automated software engineering:

- **Agentic Methods**: AutoCodeRover, SWE-agent, Agentless - use iterative exploration with tools
- **RAG Methods**: BM25, Dense Embeddings, Hybrid Retrieval - use static indexing and retrieval
- **Hybrid Methods**: Combining agentic exploration with RAG context

### Key Features

- Modular architecture with clean interfaces
- Support for SWE-bench Lite (300 instances), Verified (500 instances), and Full (2,294 instances)
- Comprehensive metrics: Resolution rate, Pass@k, localization accuracy, CodeBLEU, token usage
- Flexible evaluation: Local or Docker-based sandbox execution
- Rich reporting: JSON, Markdown, HTML, and CSV reports
- Optional experiment tracking via MLflow and Weights & Biases

---

## Technology Stack

### Core Dependencies

| Category | Libraries |
|----------|-----------|
| **Configuration** | pydantic>=2.0.0, pyyaml>=6.0 |
| **LLM Integration** | tiktoken>=0.5.0, OpenAI/Anthropic APIs |
| **ML/Embeddings** | torch>=2.0.0, transformers>=4.30.0, sentence-transformers>=2.2.0 |
| **Retrieval** | faiss-cpu>=1.7.4, rank-bm25>=0.2.2, scikit-learn>=1.3.0 |
| **Code Analysis** | tree-sitter>=0.20.0, jedi>=0.19.0, ast-decompiler>=0.7.0 |
| **Evaluation** | codebleu>=0.6.0, sacrebleu>=2.3.0, nltk>=3.8.0 |
| **CLI** | click>=8.0.0 |

### Python Version

- **Minimum**: Python 3.8
- **Recommended**: Python 3.10+

---

## Project Structure

```
swe_bench_framework/          # Main package
├── __init__.py               # Package exports
├── cli.py                    # Command-line interface (Click)
├── core/                     # Core abstractions
│   ├── interfaces.py         # Abstract base classes (ContextGatherer, PatchGenerator, etc.)
│   ├── data_models.py        # Dataclasses (SWEInstance, ContextBundle, PatchResult, etc.)
│   ├── exceptions.py         # Custom exceptions
│   └── registry.py           # Component registry
├── config/                   # Configuration management
│   ├── loader.py             # YAML/JSON config loading
│   ├── schema.py             # Configuration dataclasses
│   ├── validator.py          # Config validation
│   └── defaults.py           # Default configurations
├── context_gatherers/        # Context gathering implementations
│   ├── base.py               # ContextGatherer ABC
│   ├── agentic/              # Agentic methods (AutoCodeRover, SWE-agent, Agentless)
│   │   ├── base.py
│   │   ├── autocoderover.py
│   │   ├── swe_agent.py
│   │   ├── agentless.py
│   │   ├── environment.py    # Agent execution environment
│   │   └── tools/            # Agent tools (search, view, edit, test)
│   └── rag/                  # RAG methods (BM25, Dense, Hybrid)
│       ├── base.py
│       ├── rag_gatherer.py
│       ├── indexer/          # Index builders (BM25, Dense, Hybrid)
│       ├── retriever/        # Retrieval implementations
│       └── chunking/         # Code chunking strategies
├── patch_generators/         # Patch generation
│   ├── base.py               # PatchGenerator ABC
│   ├── direct_generator.py   # Single-pass LLM generation
│   ├── iterative_generator.py # Multi-turn refinement
│   └── prompts/              # Prompt templates and builders
├── evaluation/               # Evaluation and metrics
│   ├── base.py               # Evaluator ABC
│   ├── swe_bench_evaluator.py # SWE-bench evaluation
│   ├── docker_sandbox.py     # Docker/local sandbox
│   ├── metrics/              # Metric implementations
│   └── report_generator.py   # Report generation
├── dataset/                  # Dataset loaders
│   ├── loader.py             # Base dataset loader
│   └── swe_bench_loader.py   # SWE-bench specific loaders
├── experiment/               # Experiment orchestration
│   ├── orchestrator.py       # Main experiment orchestrator
│   └── runner.py             # Experiment runner
├── llm/                      # LLM clients
│   ├── base.py               # LLMClient ABC
│   ├── openai_client.py      # OpenAI integration
│   └── token_counter.py      # Token counting utilities
└── utils/                    # Utilities
    ├── file_utils.py
    ├── git_utils.py
    └── token_utils.py

configs/                      # Example configurations
├── minimal.yaml              # Quick testing (2 instances)
├── basic_comparison.yaml     # Agentless vs Hybrid RAG
├── rag_only.yaml             # RAG comparisons
├── agentic_only.yaml         # Agentic comparisons
└── full_comparison.yaml      # All methods

scripts/                      # Utility scripts
├── download_swe_bench.py     # Dataset downloader
├── build_indexes.py          # RAG index builder
└── visualize_results.py      # Results visualization

docs/                         # Documentation
├── architecture.md           # Architecture documentation
├── usage.md                  # Usage guide
├── methodology.md            # Research methodology
├── installation.md           # Installation instructions
└── index.md                  # Main documentation

run_experiment.py             # Simple experiment runner script
setup.py                      # Package setup
requirements.txt              # Dependencies
.env.example                  # Environment variables template
```

---

## Build and Installation

### Quick Install

```bash
# Install in editable mode
pip install -e .

# Or with all optional dependencies
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
# Required: OPENAI_API_KEY (for GPT-4/GPT-3.5)
# Optional: ANTHROPIC_API_KEY (for Claude models)
```

---

## Running Experiments

### CLI Commands

```bash
# Run experiment from config
swe-compare run --config configs/basic_comparison.yaml

# Run with overrides
swe-compare run --config config.yaml --name "my_run" --output-dir ./results

# Run on specific instances
swe-compare run --config config.yaml --instances "django-1234,flask-5678"

# Dry run (show what would be executed)
swe-compare run --config config.yaml --dry-run

# Compare multiple configurations
swe-compare compare --configs config1.yaml config2.yaml --output-dir ./comparison

# Evaluate existing predictions
swe-compare evaluate --predictions preds.json --dataset lite

# Build RAG index
swe-compare index --repo-path /path/to/repo --index-type hybrid

# Generate report
swe-compare report --results results.json --formats json,markdown,html

# Initialize config from template
swe-compare init --template basic -o my_config.yaml
```

### Alternative: Using Python Script

```bash
python run_experiment.py --config configs/basic_comparison.yaml
```

### Programmatic Usage

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

---

## Code Style Guidelines

### Python Style

- Follow **PEP 8** style guidelines
- Use **black** for code formatting (line length: 88 characters)
- Use **type hints** for all function signatures
- Use **docstrings** for all public classes and methods (Google style)

### Code Formatting Commands

```bash
# Format code
black swe_bench_framework/

# Type checking
mypy swe_bench_framework/

# Linting
flake8 swe_bench_framework/
```

### Naming Conventions

| Element | Convention | Example |
|---------|------------|---------|
| Classes | PascalCase | `ContextGatherer`, `PatchResult` |
| Functions/Methods | snake_case | `gather_context()`, `generate_patch()` |
| Variables | snake_case | `context_bundle`, `instance_id` |
| Constants | UPPER_SNAKE_CASE | `MAX_ITERATIONS`, `DEFAULT_TOP_K` |
| Private methods | _leading_underscore | `_format_chunk()`, `_validate()` |
| Abstract classes | ABC suffix or Base prefix | `ContextGatherer(ABC)`, `BaseRetriever` |

### Import Style

```python
# Standard library imports
import json
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

# Third-party imports
import click
import yaml
from pydantic import BaseModel

# Local imports
from .base import ContextGatherer
from ..core.data_models import ContextBundle, SWEInstance
```

---

## Testing

### Current Status

The project does **not** currently have a `tests/` directory. Testing is primarily done through:

1. **Manual testing** via example configurations in `configs/`
2. **Integration testing** through actual experiment runs
3. **Dry-run mode** to validate configurations

### Running Tests (if available)

```bash
# Run pytest (if tests exist)
pytest

# Run with coverage
pytest --cov=swe_bench_framework
```

### Testing Best Practices

When adding new features:

1. Test with `minimal.yaml` config first (2 instances)
2. Use `--dry-run` flag to validate without executing
3. Run on a small subset before full evaluation
4. Check logs in `./logs/` or output directory

---

## Key Interfaces

### 1. ContextGatherer (Abstract Base Class)

```python
class ContextGatherer(ABC):
    """Abstract base class for all context gathering strategies."""
    
    @abstractmethod
    def gather_context(self, instance: SWEInstance, repo_path: str) -> ContextBundle:
        """Gather context for a given SWE-bench instance."""
        pass
    
    @abstractmethod
    def initialize(self, repo_path: str) -> None:
        """Initialize any resources needed."""
        pass
    
    @abstractmethod
    def cleanup(self) -> None:
        """Cleanup resources."""
        pass
```

### 2. PatchGenerator (Abstract Base Class)

```python
class PatchGenerator(ABC):
    """Abstract base class for all patch generation strategies."""
    
    @abstractmethod
    def generate_patch(self, context_bundle: ContextBundle, 
                       instance: SWEInstance) -> PatchResult:
        """Generate a patch given context and problem statement."""
        pass
    
    @abstractmethod
    def validate_patch(self, patch_content: str, repo_path: str) -> bool:
        """Validate that a patch is syntactically correct."""
        pass
```

### 3. Evaluator (Abstract Base Class)

```python
class Evaluator(ABC):
    """Abstract base class for evaluation strategies."""
    
    @abstractmethod
    def evaluate(self, patch_result: PatchResult, instance: SWEInstance,
                 repo_path: str) -> EvaluationResult:
        """Evaluate a generated patch."""
        pass
```

---

## Configuration Schema

### Basic Configuration Structure

```yaml
experiment_name: "my_experiment"
output_dir: "./results"
dataset: "lite"  # Options: lite, verified, full

# Methods to compare
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

sandbox: "local"  # Options: local, docker
max_workers: 1
```

See `configs/` directory for complete examples.

---

## Security Considerations

### API Keys

- Store API keys in `.env` file (never commit to version control)
- Use environment variable syntax in configs: `${OPENAI_API_KEY}`
- The `.env` file is listed in `.gitignore`

### Code Execution

- All test execution happens in Docker containers (when using `sandbox: docker`)
- Local sandbox (`sandbox: local`) runs tests directly on host - use with caution
- Containers can be configured with restricted network access

### Resource Limits

Docker containers have configurable memory and CPU limits:

```yaml
evaluation:
  sandbox:
    type: "docker"
    timeout: 300.0
    memory_limit: "4g"
    cpu_limit: "2"
```

---

## Common Development Tasks

### Adding a New Context Gatherer

1. Create a new class inheriting from `ContextGatherer` in `context_gatherers/`
2. Implement `gather_context()`, `initialize()`, and `cleanup()` methods
3. Register in the component registry if needed
4. Add configuration schema to `config/schema.py`
5. Create an example config in `configs/`

### Adding a New Metric

1. Create a new class inheriting from `Metric` in `evaluation/metrics/`
2. Implement `compute()` and `get_name()` methods
3. Register in the component registry
4. Add to default metrics list if applicable

### Modifying Configuration

1. Update `config/schema.py` for new configuration options
2. Update `config/loader.py` for loading logic
3. Update example configs in `configs/`
4. Update documentation in `docs/`

---

## Troubleshooting

### Common Issues

**Issue**: `ModuleNotFoundError: ModuleNotFoundError: No module named 'swe_bench_framework'`

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

### Debug Mode

Enable verbose logging:
```bash
swe-compare --verbose run --config config.yaml
```

Or in Python:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

---

## Additional Resources

- **README.md**: Project overview and quick start
- **docs/architecture.md**: Detailed architecture documentation
- **docs/usage.md**: Comprehensive usage guide
- **docs/methodology.md**: Research methodology and experimental design
- **docs/installation.md**: Installation instructions

---

## License

MIT License - See [LICENSE](LICENSE) for details.
