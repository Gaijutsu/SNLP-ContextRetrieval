# SWE-bench Comparison Framework Documentation

## Overview

The SWE-bench Comparison Framework is a modular, extensible research platform for comparing **agentic exploration** versus **RAG (Retrieval-Augmented Generation)** methods for automated software patching on the [SWE-bench](https://www.swebench.com/) benchmark.

This framework enables researchers to:

- **Fairly compare** different context gathering strategies (agentic vs RAG)
- **Evaluate** patch generation methods using standardized metrics
- **Extend** the framework with new methods and metrics
- **Reproduce** experiments with comprehensive logging and tracking

---

## Key Features

### Unified Abstraction Layer
Both agentic and RAG methods implement the same `ContextGatherer` interface, ensuring fair comparison by using identical patch generation and evaluation pipelines.

### Comprehensive Evaluation
The framework computes multiple metrics:
- **Resolution Rate**: Primary metric (% of issues successfully resolved)
- **Localization Accuracy**: Recall@k for file and function identification
- **CodeBLEU**: Code similarity to human-written patches
- **Semantic Entropy**: Uncertainty detection in generated patches
- **Token Usage**: Cost analysis for each method

### Pluggable Architecture
Easily add new methods by implementing standard interfaces:
- `ContextGatherer`: For context gathering strategies
- `PatchGenerator`: For patch generation approaches
- `Evaluator`: For custom evaluation metrics

### Experiment Tracking
Built-in support for:
- Structured logging (JSON format)
- MLflow and Weights & Biases integration
- Comprehensive result storage and reporting

---

## Quick Start Guide

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd swe_bench_framework

# Install dependencies
pip install -r requirements.txt

# Set up API keys
export OPENAI_API_KEY="your-key-here"
export ANTHROPIC_API_KEY="your-key-here"
```

### 2. Run a Simple Comparison

```python
from swe_bench_framework import ExperimentOrchestrator
from swe_bench_framework.config import ExperimentConfig

# Load configuration
config = ExperimentConfig.from_yaml('config.yaml')

# Run experiment
orchestrator = ExperimentOrchestrator(config)
report = orchestrator.run()

# Print results
print(f"Resolution Rate: {report.summary['method_name']['resolution_rate']:.2%}")
```

### 3. Using the CLI

```bash
# Run experiment from config
python -m swe_bench_framework run --config config.yaml

# Generate report
python -m swe_bench_framework report --results results/
```

---

## Framework Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    SWE-bench Comparison Framework                │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────────┐    ┌─────────────────────────────────────┐ │
│  │  Configuration  │───▶│         Experiment Orchestrator      │ │
│  └─────────────────┘    └─────────────────────────────────────┘ │
│                                    │                             │
│           ┌────────────────────────┼────────────────────────┐    │
│           ▼                        ▼                        ▼    │
│  ┌─────────────────┐    ┌─────────────────┐    ┌──────────────┐ │
│  │ ContextGatherer │    │ PatchGenerator  │    │  Evaluator   │ │
│  │   Interface     │    │   Interface     │    │  Interface   │ │
│  └────────┬────────┘    └────────┬────────┘    └──────┬───────┘ │
│           │                      │                    │         │
│     ┌─────┴─────┐          ┌─────┴─────┐              │         │
│     ▼           ▼          ▼           ▼              │         │
│  ┌──────┐   ┌──────┐   ┌──────┐   ┌──────┐           │         │
│  │Agentic│   │ RAG  │   │Direct│   │Iter. │           │         │
│  │Methods│   │Methods│   │ Gen  │   │ Gen  │           │         │
│  └──────┘   └──────┘   └──────┘   └──────┘           │         │
│                                                       │         │
│  ┌────────────────────────────────────────────────────┘         │
│  │                    Metrics Collection                         │
│  │  - Resolution Rate  - Localization Acc.  - CodeBLEU          │
│  │  - Token Usage      - Semantic Entropy                      │
│  └──────────────────────────────────────────────────────────────┘
└─────────────────────────────────────────────────────────────────┘
```

---

## Documentation Structure

| Document | Description |
|----------|-------------|
| [Architecture](architecture.md) | System architecture, component diagrams, and design decisions |
| [Methodology](methodology.md) | Research methodology, hypothesis, and evaluation principles |
| [Installation](installation.md) | Prerequisites, installation steps, and configuration |
| [Usage](usage.md) | Running experiments, CLI reference, and example workflows |
| [API Reference](api/) | Detailed API documentation for all interfaces |
| [Examples](examples/) | Configuration examples and sample code |

---

## Supported Methods

### Agentic Exploration Methods

| Method | Description | Key Features |
|--------|-------------|--------------|
| **AutoCodeRover** | AST-based code search with spectrum-based fault localization | Stratified retrieval, SBFL integration |
| **SWE-agent** | Agent-Computer Interface (ACI) with ReAct reasoning | Purpose-built tools, linter integration |
| **Agentless** | Hierarchical localization without agent scaffolding | Three-phase process, patch filtering |

### RAG Methods

| Method | Description | Key Features |
|--------|-------------|--------------|
| **BM25** | Sparse retrieval with probabilistic scoring | Fast, keyword-based, interpretable |
| **Dense** | Neural embedding-based retrieval | Semantic similarity, cross-lingual |
| **Hybrid** | Combined BM25 + Dense with Reciprocal Rank Fusion | Best of both approaches |

---

## Research Context

This framework is designed to test the hypothesis:

> **Agentic exploration produces higher signal-to-noise ratio in context gathering compared to RAG methods, leading to better patch generation for complex software issues.**

The framework enables systematic comparison by:
1. Controlling for LLM model and temperature
2. Using identical patch generation strategies
3. Applying standardized evaluation metrics
4. Ensuring reproducible experimental conditions

---

## Citation

If you use this framework in your research, please cite:

```bibtex
@software{swe_bench_comparison_framework,
  title = {SWE-bench Comparison Framework: Agentic Exploration vs RAG},
  year = {2025},
  url = {<repository-url>}
}
```

---

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## Support

For questions, issues, or contributions:
- Open an issue on GitHub
- Contact the maintainers
- Check the [FAQ](usage.md#faq) section
