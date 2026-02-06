# Architecture Documentation

## System Architecture Overview

The SWE-bench Comparison Framework follows a layered architecture with clean separation of concerns and well-defined interfaces. This document describes the architecture, component interactions, and design decisions.

---

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                           SWE-bench Comparison Framework                            │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                     │
│  ┌─────────────────┐     ┌─────────────────────────────────────────────────────┐   │
│  │  Configuration  │────▶│              Experiment Orchestrator                │   │
│  │     Module      │     │  (manages experiment lifecycle, parallelization)    │   │
│  └─────────────────┘     └──────────────────────────┬──────────────────────────┘   │
│                                                     │                               │
│                         ┌──────────────────────────┼──────────────────────────┐    │
│                         │                          │                          │    │
│                         ▼                          ▼                          ▼    │
│  ┌─────────────────────────────────────────────────────────────────────────────┐  │
│  │                        Context Gathering Layer                               │  │
│  │  ┌─────────────────────┐              ┌─────────────────────────────────┐   │  │
│  │  │   Agentic Methods   │              │         RAG Methods             │   │  │
│  │  │  ┌───────────────┐  │              │  ┌─────────────────────────┐    │   │  │
│  │  │  │  AutoCodeRover│  │              │  │   Repository Indexer    │    │   │  │
│  │  │  │  (AST Search) │  │              │  │  - BM25 Index           │    │   │  │
│  │  │  └───────────────┘  │              │  │  - Dense Embeddings     │    │   │  │
│  │  │  ┌───────────────┐  │              │  │  - Hybrid Index         │    │   │  │
│  │  │  │   SWE-agent   │  │              │  └─────────────────────────┘    │   │  │
│  │  │  │  (ReAct ACI)  │  │              │  ┌─────────────────────────┐    │   │  │
│  │  │  └───────────────┘  │              │  │   Retriever Pipeline    │    │   │  │
│  │  │  ┌───────────────┐  │              │  │  - Query Generator      │    │   │  │
│  │  │  │   Agentless   │  │              │  │  - Multi-strategy       │    │   │  │
│  │  │  │ (Hierarchical)│  │              │  │  - Re-ranking           │    │   │  │
│  │  │  └───────────────┘  │              │  └─────────────────────────┘    │   │  │
│  │  └─────────────────────┘              └─────────────────────────────────┘   │  │
│  │                                                                              │  │
│  │  ┌─────────────────────────────────────────────────────────────────────────┐ │  │
│  │  │              ContextGatherer Interface (Unified Abstraction)            │ │  │
│  │  │     gather_context(instance: SWEInstance) -> ContextBundle              │ │  │
│  │  └─────────────────────────────────────────────────────────────────────────┘ │  │
│  └─────────────────────────────────────────────────────────────────────────────┘  │
│                                    │                                              │
│                                    ▼                                              │
│  ┌─────────────────────────────────────────────────────────────────────────────┐  │
│  │                        Patch Generation Layer                                │  │
│  │                                                                              │  │
│  │  ┌─────────────────────────────────────────────────────────────────────┐    │  │
│  │  │              PatchGenerator Interface (Unified Abstraction)         │    │  │
│  │  │  generate_patch(context: ContextBundle, instance: SWEInstance)      │    │  │
│  │  │                    -> PatchResult                                   │    │  │
│  │  └─────────────────────────────────────────────────────────────────────┘    │  │
│  │                                                                              │  │
│  │  ┌─────────────────────┐  ┌─────────────────────┐  ┌─────────────────────┐  │  │
│  │  │   Direct LLM        │  │   Multi-turn        │  │   Edit Script       │  │  │
│  │  │   Generation        │  │   Refinement        │  │   Generation        │  │  │
│  │  │   (single-pass)     │  │   (iterative)       │  │   (structured)      │  │  │
│  │  └─────────────────────┘  └─────────────────────┘  └─────────────────────┘  │  │
│  └─────────────────────────────────────────────────────────────────────────────┘  │
│                                    │                                              │
│                                    ▼                                              │
│  ┌─────────────────────────────────────────────────────────────────────────────┐  │
│  │                        Evaluation Layer                                      │  │
│  │                                                                              │  │
│  │  ┌─────────────────────┐  ┌─────────────────────┐  ┌─────────────────────┐  │  │
│  │  │   Patch Validator   │  │   Test Executor     │  │   Metrics Collector │  │  │
│  │  │  - Syntax Check     │  │  - Docker Sandbox   │  │  - Resolution Rate  │  │  │
│  │  │  - Context Verify   │  │  - Test Runner      │  │  - Localization Acc │  │  │
│  │  └─────────────────────┘  └─────────────────────┘  │  - CodeBLEU         │  │  │
│  │                                                     │  - Token Usage      │  │  │
│  │                                                     └─────────────────────┘  │  │
│  └─────────────────────────────────────────────────────────────────────────────┘  │
│                                    │                                              │
│                                    ▼                                              │
│  ┌─────────────────────────────────────────────────────────────────────────────┐  │
│  │                        Experiment Tracking Layer                             │  │
│  │  ┌─────────────────────┐  ┌─────────────────────┐  ┌─────────────────────┐  │  │
│  │  │   Results Store     │  │   Logging System    │  │   Report Generator  │  │  │
│  │  │  (JSON/Database)    │  │  (structured logs)  │  │  (comparative)      │  │  │
│  │  └─────────────────────┘  └─────────────────────┘  └─────────────────────┘  │  │
│  └─────────────────────────────────────────────────────────────────────────────┘  │
│                                                                                     │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Core Interfaces

The framework is built around five core interfaces that define contracts between components:

### 1. ContextGatherer

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

**Purpose**: Unifies agentic and RAG methods under a single interface.

**Implementations**:
- `AutoCodeRoverGatherer`: AST-based search with iterative exploration
- `SWEAgentGatherer`: ReAct-based agent with ACI tools
- `AgentlessGatherer`: Hierarchical localization workflow
- `BM25Gatherer`: Sparse retrieval using BM25
- `DenseGatherer`: Neural embedding-based retrieval
- `HybridGatherer`: Combined BM25 + Dense with RRF

### 2. PatchGenerator

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

**Purpose**: Defines how patches are generated from gathered context.

**Implementations**:
- `DirectPatchGenerator`: Single-pass LLM generation
- `IterativePatchGenerator`: Multi-turn refinement with feedback
- `EditScriptGenerator`: Structured edit script generation

### 3. Evaluator

```python
class Evaluator(ABC):
    """Abstract base class for evaluation strategies."""
    
    @abstractmethod
    def evaluate(self, patch_result: PatchResult, instance: SWEInstance,
                 repo_path: str) -> EvaluationResult:
        """Evaluate a generated patch."""
        pass
    
    @abstractmethod
    def compute_localization_accuracy(self, context_bundle: ContextBundle,
                                      gold_files: List[str],
                                      gold_functions: List[str]) -> Dict[str, float]:
        """Compute localization accuracy metrics."""
        pass
```

**Purpose**: Standardizes patch evaluation and metric computation.

**Implementations**:
- `SWEBenchEvaluator`: Standard SWE-bench evaluation pipeline

### 4. RepositoryIndexer

```python
class RepositoryIndexer(ABC):
    """Abstract base class for repository indexing strategies."""
    
    @abstractmethod
    def build_index(self, repo_path: str, output_path: str) -> None:
        """Build index from repository."""
        pass
    
    @abstractmethod
    def load_index(self, index_path: str) -> None:
        """Load existing index."""
        pass
    
    @abstractmethod
    def search(self, query: str, top_k: int = 10) -> List[SearchResult]:
        """Search the index."""
        pass
```

**Purpose**: Enables different indexing strategies for RAG methods.

**Implementations**:
- `BM25Indexer`: BM25-based sparse retrieval
- `DenseIndexer`: Neural embedding-based retrieval
- `HybridIndexer`: Combined BM25 + Dense with RRF

### 5. Metric

```python
class Metric(ABC):
    """Abstract base class for evaluation metrics."""
    
    @abstractmethod
    def compute(self, patch_result: PatchResult, instance: SWEInstance,
                evaluation_result: EvaluationResult) -> Dict[str, float]:
        """Compute the metric."""
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Get the name of the metric."""
        pass
```

**Purpose**: Defines how metrics are computed from evaluation results.

**Implementations**:
- `ResolutionMetric`: % of issues resolved
- `LocalizationMetric`: Recall@k for file/function localization
- `CodeBLEUMetric`: Code similarity score
- `TokenUsageMetric`: API cost tracking
- `SemanticEntropyMetric`: Uncertainty detection

---

## Data Flow

### Main Pipeline Flow

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              MAIN PIPELINE FLOW                                  │
└─────────────────────────────────────────────────────────────────────────────────┘

  ┌─────────────────┐
  │  Load Config    │
  │  (YAML/JSON)    │
  └────────┬────────┘
           │
           ▼
  ┌─────────────────┐     ┌──────────────────────────────────────────────────────┐
  │  Load SWE-bench │────▶│  For each instance:                                  │
  │  Instances      │     │  ┌──────────────────────────────────────────────────┐│
  └────────┬────────┘     │  │  1. CHECKOUT REPOSITORY                          ││
           │              │  │     - Clone repo                                 ││
           │              │  │     - Checkout to base_commit                    ││
           │              │  │     - Apply test patch                           ││
           │              │  └──────────────────────────────────────────────────┘│
           │              │                         │                            │
           │              │                         ▼                            │
           │              │  ┌──────────────────────────────────────────────────┐│
           │              │  │  2. CONTEXT GATHERING                            ││
           │              │  │                                                  ││
           │              │  │  Agentic Path:          RAG Path:                ││
           │              │  │  ┌─────────────┐       ┌─────────────────────┐   ││
           │              │  │  │ Initialize  │       │ Load Index          │   ││
           │              │  │  │ Environment │       │ (BM25/Dense/Hybrid) │   ││
           │              │  │  └──────┬──────┘       └──────────┬──────────┘   ││
           │              │  │         │                         │              ││
           │              │  │         ▼                         ▼              ││
           │              │  │  ┌─────────────┐       ┌─────────────────────┐   ││
           │              │  │  │ Execute     │       │ Query Generation    │   ││
           │              │  │  │ Tool Loop   │       │ (from problem stmt) │   ││
           │              │  │  │ (ReAct)     │       └──────────┬──────────┘   ││
           │              │  │  └──────┬──────┘                  │              ││
           │              │  │         │                         ▼              ││
           │              │  │         │              ┌─────────────────────┐   ││
           │              │  │         │              │ Retrieve Top-K      │   ││
           │              │  │         │              │ (with re-ranking)   │   ││
           │              │  │         │              └──────────┬──────────┘   ││
           │              │  │         │                         │              ││
           │              │  │         ▼                         ▼              ││
           │              │  │  ┌───────────────────────────────────────┐      ││
           │              │  │  │ Convert to ContextBundle              │      ││
           │              │  │  │ (unified format for both paths)       │      ││
           │              │  │  └───────────────────┬───────────────────┘      ││
           │              │  └──────────────────────┼──────────────────────────┘│
           │              │                         │                           │
           │              │                         ▼                           │
           │              │  ┌──────────────────────────────────────────────────┐│
           │              │  │  3. PATCH GENERATION                             ││
           │              │  │     - Build prompt from ContextBundle            ││
           │              │  │     - Call LLM (single/multi-turn)               ││
           │              │  │     - Extract and validate patch                 ││
           │              │  │     - Return PatchResult                         ││
           │              │  └──────────────────────┬───────────────────────────┘│
           │              │                         │                            │
           │              │                         ▼                            │
           │              │  ┌──────────────────────────────────────────────────┐│
           │              │  │  4. EVALUATION                                   ││
           │              │  │     - Apply patch in Docker sandbox              ││
           │              │  │     - Run tests                                  ││
           │              │  │     - Compute metrics                            ││
           │              │  │     - Return EvaluationResult                    ││
           │              │  └──────────────────────┬───────────────────────────┘│
           │              │                         │                            │
           │              │                         ▼                            │
           │              │  ┌──────────────────────────────────────────────────┐│
           │              │  │  5. LOG RESULTS                                  ││
           │              │  │     - Save to results store                      ││
           │              │  │     - Update metrics collector                   ││
           │              │  └──────────────────────────────────────────────────┘│
           │              └──────────────────────────────────────────────────────┘
           │
           ▼
  ┌─────────────────┐
  │  Generate       │
  │  Report         │
  └─────────────────┘
```

---

## Component Interactions

### Agentic Methods Flow

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  SWEInstance    │────▶│  AgentEnvironment│────▶│  Tool Registry  │
│  (problem stmt) │     │  (state + tools) │     │  (search, view) │
└─────────────────┘     └────────┬────────┘     └─────────────────┘
                                 │
                                 ▼
                        ┌─────────────────┐
                        │  ReAct Loop     │
                        │  (thought-act-  │
                        │   observe)      │
                        └────────┬────────┘
                                 │
                                 ▼
                        ┌─────────────────┐
                        │  ContextBundle  │
                        │  (unified fmt)  │
                        └─────────────────┘
```

### RAG Methods Flow

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  SWEInstance    │────▶│ Query Generator │────▶│  Repository     │
│  (problem stmt) │     │  (extract terms)│     │  Indexer        │
└─────────────────┘     └─────────────────┘     └────────┬────────┘
                                                         │
                                 ┌───────────────────────┼───────┐
                                 ▼                       ▼       ▼
                        ┌─────────────────┐     ┌──────────────┐ ┌──────────┐
                        │  BM25 Search    │     │ Dense Search │ │ Reranker │
                        │  (sparse)       │     │ (embeddings) │ │          │
                        └────────┬────────┘     └──────┬───────┘ └────┬─────┘
                                 │                      │              │
                                 └──────────────────────┼──────────────┘
                                                        ▼
                                               ┌─────────────────┐
                                               │  ContextBundle  │
                                               │  (unified fmt)  │
                                               └─────────────────┘
```

---

## Design Decisions

### 1. Unified ContextGatherer Interface

**Decision**: Both agentic and RAG methods implement the same interface.

**Rationale**:
- Enables fair comparison using identical downstream components
- Simplifies the architecture by reducing special cases
- Makes it easy to add new methods

**Trade-offs**:
- Agentic methods may have richer internal state that's lost in ContextBundle
- Some agentic-specific metrics may be harder to extract

### 2. Separation of Context Gathering and Patch Generation

**Decision**: Context gathering and patch generation are separate components.

**Rationale**:
- Allows mixing and matching different strategies
- Enables studying the contribution of each component
- Simplifies testing and debugging

**Trade-offs**:
- Some methods (like end-to-end trained models) may not fit this separation
- Additional overhead in converting between formats

### 3. Docker-Based Evaluation

**Decision**: All evaluation happens in Docker containers.

**Rationale**:
- Ensures reproducibility across different environments
- Isolates potentially harmful code execution
- Matches SWE-bench official evaluation protocol

**Trade-offs**:
- Slower than native execution
- Requires Docker installation and significant disk space
- Resource overhead for container management

### 4. Configuration-Driven Experiments

**Decision**: Experiments are configured via YAML files.

**Rationale**:
- Makes experiments reproducible and version-controllable
- Reduces code changes for parameter tuning
- Enables easy sharing of experiment configurations

**Trade-offs**:
- Less flexible than code-based configuration
- Requires learning the configuration schema

### 5. Pluggable Metrics

**Decision**: Metrics are computed by pluggable Metric classes.

**Rationale**:
- Easy to add new metrics without changing core code
- Metrics can be selectively enabled/disabled
- Supports both simple and complex metric computations

**Trade-offs**:
- Some metrics may require expensive recomputation
- Metric dependencies need to be managed

---

## Package Structure

```
swe_bench_framework/
├── __init__.py
├── __version__.py
├── cli.py                          # Command-line interface
│
├── core/                           # Core abstractions
│   ├── __init__.py
│   ├── interfaces.py               # All abstract base classes
│   ├── data_models.py              # Dataclasses (SWEInstance, etc.)
│   ├── exceptions.py               # Custom exceptions
│   └── registry.py                 # Component registry
│
├── config/                         # Configuration management
│   ├── __init__.py
│   ├── loader.py                   # Config loading
│   ├── schema.py                   # Configuration dataclasses
│   ├── validator.py                # Config validation
│   └── defaults.py                 # Default configurations
│
├── context_gatherers/              # Context gathering implementations
│   ├── __init__.py
│   ├── base.py                     # ContextGatherer ABC
│   ├── factory.py                  # Factory for creating gatherers
│   │
│   ├── agentic/                    # Agentic methods
│   │   ├── __init__.py
│   │   ├── base.py                 # BaseAgenticGatherer
│   │   ├── autocoderover.py
│   │   ├── swe_agent.py
│   │   ├── agentless.py
│   │   ├── environment.py          # AgentEnvironment
│   │   └── tools/
│   │       ├── __init__.py
│   │       ├── base.py
│   │       ├── file_tools.py
│   │       ├── search_tools.py
│   │       ├── ast_tools.py
│   │       └── execution_tools.py
│   │
│   └── rag/                        # RAG methods
│       ├── __init__.py
│       ├── base.py                 # BaseRAGGatherer
│       ├── factory.py
│       │
│       ├── indexer/
│       │   ├── __init__.py
│       │   ├── base.py
│       │   ├── bm25_indexer.py
│       │   ├── dense_indexer.py
│       │   └── hybrid_indexer.py
│       │
│       ├── retriever/
│       │   ├── __init__.py
│       │   ├── base.py
│       │   ├── sparse_retriever.py
│       │   ├── dense_retriever.py
│       │   ├── hybrid_retriever.py
│       │   └── reranker.py
│       │
│       └── chunking/
│           ├── __init__.py
│           ├── base.py
│           ├── ast_chunker.py
│           └── sliding_chunker.py
│
├── patch_generators/               # Patch generation
│   ├── __init__.py
│   ├── base.py
│   ├── factory.py
│   ├── direct_generator.py
│   ├── iterative_generator.py
│   ├── edit_script_generator.py
│   └── prompts/
│       ├── __init__.py
│       ├── templates.py
│       └── builders.py
│
├── evaluation/                     # Evaluation
│   ├── __init__.py
│   ├── base.py
│   ├── swe_bench_evaluator.py
│   ├── docker_sandbox.py
│   ├── metrics/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── resolution.py
│   │   ├── localization.py
│   │   ├── codebleu.py
│   │   ├── token_usage.py
│   │   └── semantic_entropy.py
│   └── report_generator.py
│
├── llm/                            # LLM integration
│   ├── __init__.py
│   ├── base.py
│   ├── openai_client.py
│   ├── anthropic_client.py
│   ├── local_client.py
│   └── token_counter.py
│
├── dataset/                        # Dataset handling
│   ├── __init__.py
│   ├── loader.py
│   └── swe_bench_loader.py
│
├── experiment/                     # Experiment orchestration
│   ├── __init__.py
│   ├── orchestrator.py
│   ├── runner.py
│   ├── parallel.py
│   └── cache.py
│
├── logging/                        # Logging and tracking
│   ├── __init__.py
│   ├── logger.py
│   ├── structured_logger.py
│   └── trackers/
│       ├── __init__.py
│       ├── base.py
│       ├── mlflow_tracker.py
│       └── wandb_tracker.py
│
└── utils/                          # Utilities
    ├── __init__.py
    ├── file_utils.py
    ├── git_utils.py
    ├── ast_utils.py
    └── token_utils.py
```

---

## Extension Points

The framework provides several extension points for customization:

### Adding a New Context Gatherer

```python
from swe_bench_framework.context_gatherers import ContextGatherer, ContextBundle

class MyCustomGatherer(ContextGatherer):
    def gather_context(self, instance, repo_path):
        # Your implementation
        return ContextBundle(...)
    
    def initialize(self, repo_path):
        # Setup code
        pass
    
    def cleanup(self):
        # Cleanup code
        pass

# Register
from swe_bench_framework.core.registry import register_gatherer
register_gatherer('my_custom', MyCustomGatherer)
```

### Adding a New Metric

```python
from swe_bench_framework.evaluation import Metric

class MyMetric(Metric):
    def compute(self, patch_result, instance, evaluation_result):
        # Your computation
        return {'my_metric': value}
    
    def get_name(self):
        return 'my_metric'

# Register
from swe_bench_framework.core.registry import register_metric
register_metric('my_metric', MyMetric)
```

---

## Performance Considerations

### Memory Usage

- **Repository Indexing**: Large repositories may require significant memory for embeddings
- **Context Bundles**: Multiple instances in memory can accumulate
- **Docker Images**: Each repository may have its own Docker image

**Mitigations**:
- Use batch processing for large datasets
- Implement context bundle serialization
- Clean up Docker images after evaluation

### Computation Time

- **Agentic Methods**: Slower due to iterative exploration (minutes per instance)
- **RAG Methods**: Faster retrieval but may require index building (seconds per instance)
- **Evaluation**: Docker-based evaluation adds overhead

**Mitigations**:
- Parallelize instance processing
- Cache indexes across runs
- Use cloud-based evaluation (Modal)

---

## Security Considerations

1. **Code Execution**: All test execution happens in Docker containers
2. **API Keys**: Stored in environment variables, never in code
3. **Network Access**: Containers can be configured with restricted network access
4. **Resource Limits**: Docker containers have memory and CPU limits
