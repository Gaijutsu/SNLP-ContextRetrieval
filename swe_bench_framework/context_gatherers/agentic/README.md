# Agentic Exploration Module

This module implements agentic (exploration-based) context gathering methods for the SWE-bench comparison framework. These methods use iterative exploration with tools to navigate and understand codebases, in contrast to RAG-based methods that rely on pre-built indexes.

## Overview

Agentic exploration methods simulate how a human developer would explore a codebase to understand and fix an issue:
1. Read the problem statement
2. Search for relevant files
3. Navigate through code
4. Gather context iteratively
5. Build understanding before generating a patch

## Implemented Methods

### 1. AutoCodeRoverGatherer

**Paper**: [AutoCodeRover: Autonomous Program Improvement](https://arxiv.org/abs/2404.05427)

**Approach**: Stratified retrieval with AST-based code search

**Key Features**:
- **AST-Based Code Search**: Uses abstract syntax tree analysis for accurate code retrieval
- **Stratified Retrieval**: Hierarchical strategy (file → class → function)
- **Two-Phase Architecture**:
  - Phase 1: Context retrieval using search APIs
  - Phase 2: Context refinement and detailed gathering
- **Spectrum-Based Fault Localization (SBFL)**: Leverages test execution information when available

**Configuration**:
```python
config = {
    'max_iterations': 50,
    'max_files_per_layer': 5,      # Top files to consider
    'max_classes_per_layer': 10,   # Top classes to consider
    'max_functions_per_layer': 20, # Top functions to consider
    'use_sbfl': True               # Enable test-based localization
}
```

**Usage**:
```python
from context_gatherers import create_gatherer

gatherer = create_gatherer('autocoderover', config)
bundle = gatherer.gather_context(instance, repo_path)
```

---

### 2. SWEAgentGatherer

**Paper**: [SWE-agent: Agent-Computer Interfaces Enable Automated Software Engineering](https://arxiv.org/abs/2405.15793)

**Approach**: ReAct pattern with Agent-Computer Interface (ACI)

**Key Features**:
- **ReAct Pattern**: Interleaved reasoning and action (Thought → Action → Observation)
- **Agent-Computer Interface (ACI)**: Purpose-built tools optimized for LM agents
- **Limited Output**: Tools return controlled amounts of information to avoid context overflow
- **Iterative Decision Making**: Agent decides next action based on observations

**Configuration**:
```python
config = {
    'max_iterations': 100,
    'max_thought_length': 500,       # Max characters for thoughts
    'enable_completion_detection': True,  # Auto-detect when done
    'min_iterations': 3              # Minimum iterations before stopping
}
```

**Usage**:
```python
from context_gatherers import create_gatherer

gatherer = create_gatherer('swe_agent', config)
bundle = gatherer.gather_context(instance, repo_path)
```

---

### 3. AgentlessGatherer

**Paper**: [Agentless: Demystifying LLM-based Software Engineering Agents](https://arxiv.org/abs/2407.01489)

**Approach**: Hierarchical localization with fixed workflow

**Key Features**:
- **No Complex Agent Scaffolding**: Simple, predictable workflow
- **Three-Phase Process**:
  1. File selection based on keyword matching
  2. Class/method localization within selected files
  3. Edit location pinpointing
- **Fixed Workflow**: No autonomous decision-making, more reliable
- **Cost Efficient**: Lower computational overhead

**Configuration**:
```python
config = {
    'max_iterations': 50,
    'max_files': 5,                  # Maximum files to select
    'max_classes_per_file': 3,       # Max classes per file
    'max_methods_per_class': 5,      # Max methods per class
    'context_window_lines': 10       # Lines of context around locations
}
```

**Usage**:
```python
from context_gatherers import create_gatherer

gatherer = create_gatherer('agentless', config)
bundle = gatherer.gather_context(instance, repo_path)
```

---

## Tools

The agentic module provides a comprehensive set of tools for code exploration:

### File Tools

#### ViewFileTool
View file contents with line limits (ACI-style).
```python
view_tool = ViewFileTool(repo_path)
obs = view_tool.execute({
    'path': 'src/module.py',
    'offset': 0,      # Starting line
    'limit': 100      # Max lines to view
})
```

#### GrepTool
Search file contents using regex patterns.
```python
grep_tool = GrepTool(repo_path)
obs = grep_tool.execute({
    'pattern': 'def calculate',
    'path': 'src/',           # Optional: restrict search
    'file_pattern': '*.py',   # Optional: file filter
    'max_results': 50
})
```

#### FindFileTool
Find files by name pattern.
```python
find_tool = FindFileTool(repo_path)
obs = find_tool.execute({
    'pattern': 'test_',
    'max_results': 20
})
```

### Search Tools

#### SearchClassTool
Find class definitions using AST parsing.
```python
search_class = SearchClassTool(repo_path)
obs = search_class.execute({
    'class_name': 'Calculator',
    'exact_match': False,
    'max_results': 10
})
```

#### SearchMethodTool
Find method/function definitions using AST parsing.
```python
search_method = SearchMethodTool(repo_path)
obs = search_method.execute({
    'method_name': 'calculate',
    'class_name': 'Calculator',  # Optional: restrict to class
    'file_path': 'src/calc.py',  # Optional: restrict to file
    'max_results': 10
})
```

### AST Tools

#### GetClassHierarchyTool
Get class inheritance hierarchy.
```python
hierarchy_tool = GetClassHierarchyTool(repo_path)
obs = hierarchy_tool.execute({
    'class_name': 'AdvancedCalculator',
    'include_ancestors': True,
    'include_descendants': False
})
```

#### GetCallGraphTool
Get function call relationships.
```python
callgraph_tool = GetCallGraphTool(repo_path)
obs = callgraph_tool.execute({
    'function_name': 'process_data',
    'include_callees': True,
    'include_callers': True
})
```

### Execution Tools

#### RunTestTool
Run tests in the repository.
```python
test_tool = RunTestTool(repo_path)
obs = test_tool.execute({
    'test_path': 'tests/',      # Optional: specific test path
    'test_name': 'test_calc',   # Optional: specific test name
    'verbose': True,
    'timeout': 120
})
```

#### LinterTool
Run Python linter for syntax checking.
```python
linter_tool = LinterTool(repo_path)
obs = linter_tool.execute({
    'file_path': 'src/module.py',
    'linter': 'flake8',  # or 'pylint', auto-detected if not specified
    'max_issues': 20
})
```

---

## Architecture

### Base Classes

#### BaseAgenticGatherer
Abstract base class for all agentic gatherers. Provides:
- Environment management
- State tracking
- Tool registration
- Context bundle assembly

#### AgentEnvironment
Core infrastructure for agentic exploration:
- Tool registry
- State management
- Action execution
- Context conversion

#### AgentState
Tracks exploration state:
- Viewed files
- Search history
- Action history
- Found symbols
- Context chunks

### Data Flow

```
SWEInstance (problem statement)
    ↓
AgenticGatherer.explore()
    ↓
AgentEnvironment.execute_action() → Tool.execute()
    ↓
Observation
    ↓
Process observation → Update state
    ↓
Repeat until complete
    ↓
_state_to_context_bundle() → ContextBundle
```

---

## Usage Examples

### Basic Usage

```python
from context_gatherers import create_gatherer, SWEInstance

# Create instance
instance = SWEInstance(
    instance_id='django-1234',
    repo='django/django',
    base_commit='abc123',
    problem_statement='The Model.save() method fails when...',
    modified_files=['django/db/models/base.py'],
    modified_methods=['Model.save']
)

# Create and run gatherer
gatherer = create_gatherer('agentless', {
    'max_iterations': 30,
    'max_files': 5
})

bundle = gatherer.gather_context(instance, '/path/to/repo')

# Access results
print(f"Gathered {len(bundle.chunks)} context chunks")
print(f"Token count: {bundle.token_count}")

# Get stats
stats = gatherer.get_stats()
print(f"Iterations: {stats['total_iterations']}")
print(f"Files viewed: {stats['files_viewed']}")
```

### Custom Agent Implementation

```python
from context_gatherers.agentic import BaseAgenticGatherer

class MyCustomGatherer(BaseAgenticGatherer):
    def explore(self, instance):
        # Custom exploration logic
        state = self.environment.state
        
        # Search for relevant files
        obs = self.execute_tool('grep', 
            pattern='buggy_function',
            max_results=10
        )
        
        # View found files
        if obs.success:
            for result in obs.metadata.get('results', []):
                file_path = result['file']
                self.execute_tool('view_file', 
                    path=file_path,
                    limit=50
                )
        
        # Mark complete
        state.mark_complete('Exploration complete')

# Register the custom gatherer
from context_gatherers import register_gatherer_type
register_gatherer_type('my_custom', MyCustomGatherer)
```

### Using Tools Directly

```python
from context_gatherers.agentic.tools import (
    ViewFileTool,
    SearchClassTool,
    GrepTool
)

repo_path = '/path/to/repo'

# View a file
view_tool = ViewFileTool(repo_path)
obs = view_tool.execute({'path': 'src/main.py', 'limit': 50})
print(obs.output)

# Search for a class
search_tool = SearchClassTool(repo_path)
obs = search_tool.execute({'class_name': 'MyClass'})
for result in obs.metadata.get('results', []):
    print(f"Found {result['class_name']} in {result['file']}")

# Grep for patterns
grep_tool = GrepTool(repo_path)
obs = grep_tool.execute({'pattern': 'TODO|FIXME'})
print(obs.output)
```

---

## Configuration Reference

### Common Configuration Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `max_iterations` | int | 50 | Maximum exploration iterations |
| `timeout` | int | 300 | Maximum time in seconds |

### AutoCodeRover-Specific

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `max_files_per_layer` | int | 5 | Max files at file layer |
| `max_classes_per_layer` | int | 10 | Max classes at class layer |
| `max_functions_per_layer` | int | 20 | Max functions at function layer |
| `use_sbfl` | bool | True | Enable test-based localization |

### SWE-agent-Specific

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `max_thought_length` | int | 500 | Max characters for thoughts |
| `enable_completion_detection` | bool | True | Auto-detect completion |
| `min_iterations` | int | 3 | Minimum iterations |

### Agentless-Specific

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `max_files` | int | 5 | Maximum files to select |
| `max_classes_per_file` | int | 3 | Max classes per file |
| `max_methods_per_class` | int | 5 | Max methods per class |
| `context_window_lines` | int | 10 | Context lines around locations |

---

## Performance Comparison

Based on research findings:

| Method | SWE-bench Lite | Cost/Issue | Complexity |
|--------|---------------|------------|------------|
| Agentless | 27-32% | $0.34-0.70 | Low |
| AutoCodeRover | 19-26% | ~$0.43 | Medium |
| SWE-agent | ~20% | - | High |

---

## Testing

Run the test suite:

```bash
cd /mnt/okcomputer/output/swe_bench_framework
python test_agentic.py
```

The test suite covers:
- Gatherer creation and configuration
- Tool functionality
- End-to-end exploration
- Context bundle generation

---

## Extending

### Adding a New Tool

```python
from context_gatherers.agentic.environment import Tool, Observation

class MyTool(Tool):
    def __init__(self, repo_path):
        super().__init__('my_tool', 'Description of my tool')
        self.repo_path = repo_path
    
    def execute(self, parameters):
        # Tool implementation
        result = do_something(parameters)
        
        return Observation(
            output=result,
            metadata={'key': 'value'}
        )
```

### Adding a New Gatherer

```python
from context_gatherers.agentic import BaseAgenticGatherer

class MyGatherer(BaseAgenticGatherer):
    def explore(self, instance):
        # Your exploration logic
        pass

# Register
from context_gatherers import register_gatherer_type
register_gatherer_type('my_gatherer', MyGatherer)
```

---

## References

1. **AutoCodeRover**: Zhang et al., "AutoCodeRover: Autonomous Program Improvement", arXiv:2404.05427
2. **SWE-agent**: Yang et al., "SWE-agent: Agent-Computer Interfaces Enable Automated Software Engineering", arXiv:2405.15793
3. **Agentless**: Xia et al., "Agentless: Demystifying LLM-based Software Engineering Agents", arXiv:2407.01489
