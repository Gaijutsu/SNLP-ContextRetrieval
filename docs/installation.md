# Installation Guide

This guide covers the installation and configuration of the SWE-bench Comparison Framework.

---

## Prerequisites

### System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **Python** | 3.9 | 3.10+ |
| **RAM** | 16 GB | 32 GB+ |
| **Disk Space** | 150 GB | 500 GB+ |
| **CPU Cores** | 4 | 8+ |
| **OS** | Linux/macOS | Linux (Ubuntu 20.04+) |

### Required Software

1. **Python 3.9+**
   ```bash
   python --version  # Should show 3.9 or higher
   ```

2. **Docker**
   - Required for SWE-bench evaluation
   - [Installation Guide](https://docs.docker.com/get-docker/)
   ```bash
   docker --version  # Verify installation
   ```

3. **Git**
   ```bash
   git --version  # Verify installation
   ```

### Optional Software

- **CUDA** (for GPU-accelerated embeddings): Version 11.8+
- **NVIDIA Docker Runtime** (for GPU support in containers)

---

## Installation Steps

### Step 1: Clone the Repository

```bash
git clone <repository-url>
cd swe_bench_framework
```

### Step 2: Create a Virtual Environment

**Using venv:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

**Using conda:**
```bash
conda create -n swe_bench python=3.10
conda activate swe_bench
```

### Step 3: Install Dependencies

**Basic Installation:**
```bash
pip install -r requirements.txt
```

**Development Installation (with testing tools):**
```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

**Editable Installation (for development):**
```bash
pip install -e .
```

### Step 4: Verify Installation

```bash
python -c "from swe_bench_framework import ExperimentOrchestrator; print('Installation successful!')"
```

### Step 5: Test Docker Setup

```bash
# Test Docker installation
docker run hello-world

# Check Docker disk usage
docker system df

# Ensure sufficient disk space (minimum 120GB free)
df -h
```

---

## Configuration Setup

### Directory Structure

Create the following directory structure for experiments:

```
swe_bench_experiments/
├── configs/              # Experiment configurations
├── results/              # Experiment results
├── logs/                 # Log files
├── indexes/              # Repository indexes (for RAG methods)
└── cache/                # Cache directory
```

```bash
mkdir -p swe_bench_experiments/{configs,results,logs,indexes,cache}
```

### Configuration File

Create a basic configuration file at `swe_bench_experiments/configs/basic.yaml`:

```yaml
# Experiment metadata
experiment:
  name: "my_first_experiment"
  description: "Basic comparison of agentic vs RAG methods"
  output_dir: "./swe_bench_experiments/results"
  random_seed: 42

# Dataset configuration
dataset:
  name: "swe-bench-lite"
  split: "test"
  filter:
    max_instances: 50  # Start with small subset

# LLM configuration
llm:
  provider: "openai"
  model: "gpt-4-turbo-preview"
  temperature: 0.0
  max_tokens: 4096
  api_key: "${OPENAI_API_KEY}"

# Methods to compare
methods:
  - name: "bm25"
    type: "rag"
    enabled: true
    config:
      top_k: 20
      
  - name: "autocoderover"
    type: "agentic"
    enabled: true
    config:
      max_iterations: 30

# Evaluation configuration
evaluation:
  sandbox:
    type: "docker"
    timeout: 300
  metrics:
    - resolution_rate
    - localization_accuracy
    - token_usage
```

---

## API Key Configuration

### OpenAI

1. Get your API key from [OpenAI Platform](https://platform.openai.com/api-keys)

2. Set environment variable:
   ```bash
   export OPENAI_API_KEY="sk-..."
   ```

3. Add to `.bashrc` or `.zshrc` for persistence:
   ```bash
   echo 'export OPENAI_API_KEY="sk-..."' >> ~/.bashrc
   source ~/.bashrc
   ```

### Anthropic

1. Get your API key from [Anthropic Console](https://console.anthropic.com/)

2. Set environment variable:
   ```bash
   export ANTHROPIC_API_KEY="sk-ant-..."
   ```

3. Add to `.bashrc` or `.zshrc`:
   ```bash
   echo 'export ANTHROPIC_API_KEY="sk-ant-..."' >> ~/.bashrc
   source ~/.bashrc
   ```

### Azure OpenAI (Optional)

```bash
export AZURE_OPENAI_API_KEY="..."
export AZURE_OPENAI_ENDPOINT="https://..."
export AZURE_OPENAI_API_VERSION="2024-02-01"
```

### Local Models (Optional)

For using local models with vLLM or similar:

```bash
export LOCAL_LLM_URL="http://localhost:8000/v1"
export LOCAL_LLM_MODEL="meta-llama/Llama-2-70b"
```

---

## Advanced Configuration

### Docker Configuration

**Increase Docker Resources:**

1. Open Docker Desktop settings
2. Go to Resources
3. Set:
   - CPUs: 8+
   - Memory: 16 GB+
   - Swap: 4 GB
   - Disk image size: 120 GB+

**Configure Docker for Linux:**

```bash
# Add user to docker group (avoid sudo)
sudo usermod -aG docker $USER
newgrp docker

# Verify
docker run hello-world
```

### Python Path Configuration

If installing in development mode:

```bash
# Add to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Or use .env file
echo "PYTHONPATH=$(pwd)" > .env
```

### Cache Configuration

Configure caching to speed up repeated evaluations:

```yaml
evaluation:
  sandbox:
    type: "docker"
    cache_level: "env"  # Options: none, base, env, instance
```

| Cache Level | Description | Storage | Speed |
|-------------|-------------|---------|-------|
| `none` | No caching | ~120GB | Slowest |
| `base` | Cache base images only | ~120GB | Slow |
| `env` | Cache base + environment | ~100GB | Moderate |
| `instance` | Cache everything | ~2TB | Fastest |

---

## Verification

### Test Installation

Run the test suite to verify installation:

```bash
# Run all tests
pytest

# Run specific test modules
pytest tests/unit/test_context_gatherers.py
pytest tests/unit/test_patch_generators.py
pytest tests/integration/test_end_to_end.py
```

### Test API Keys

```python
# test_api.py
import os
from swe_bench_framework.llm import OpenAIClient, AnthropicClient

# Test OpenAI
openai_key = os.getenv("OPENAI_API_KEY")
if openai_key:
    client = OpenAIClient({"api_key": openai_key, "model": "gpt-4"})
    response = client.generate("Hello, world!")
    print(f"OpenAI test: {response[:50]}...")
else:
    print("OpenAI API key not found")

# Test Anthropic
anthropic_key = os.getenv("ANTHROPIC_API_KEY")
if anthropic_key:
    client = AnthropicClient({"api_key": anthropic_key, "model": "claude-3-sonnet"})
    response = client.generate("Hello, world!")
    print(f"Anthropic test: {response[:50]}...")
else:
    print("Anthropic API key not found")
```

### Test Docker Setup

```bash
# Pull a test image
docker pull python:3.10-slim

# Run a test container
docker run --rm python:3.10-slim python -c "print('Docker works!')"
```

---

## Troubleshooting

### Common Issues

#### Issue: `ModuleNotFoundError: No module named 'swe_bench_framework'`

**Solution:**
```bash
# Ensure you're in the correct directory
cd swe_bench_framework

# Install in editable mode
pip install -e .

# Verify Python path
python -c "import sys; print(sys.path)"
```

#### Issue: `docker: permission denied`

**Solution:**
```bash
# Add user to docker group
sudo usermod -aG docker $USER

# Apply changes
newgrp docker

# Or use sudo (not recommended for regular use)
sudo docker ...
```

#### Issue: `API key not found`

**Solution:**
```bash
# Verify environment variable is set
echo $OPENAI_API_KEY

# Set it if missing
export OPENAI_API_KEY="your-key-here"

# Add to shell profile for persistence
echo 'export OPENAI_API_KEY="your-key-here"' >> ~/.bashrc
```

#### Issue: `Insufficient disk space`

**Solution:**
```bash
# Check disk usage
df -h

# Clean Docker resources
docker system prune -a

# Clean up unused images
docker image prune -a

# Check Docker disk usage
docker system df
```

#### Issue: `Out of memory`

**Solution:**
```bash
# Reduce number of workers in config
evaluation:
  max_workers: 2  # Reduce from default

# Or run sequentially
python -m swe_bench_framework run --config config.yaml --workers 1
```

#### Issue: `Embedding model download fails`

**Solution:**
```bash
# Pre-download models
python -c "
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('jinaai/jina-embeddings-v2-base-code')
"

# Or use local cache directory
export SENTENCE_TRANSFORMERS_HOME="/path/to/cache"
```

### Getting Help

If you encounter issues not covered here:

1. Check the [GitHub Issues](<repository-url>/issues)
2. Review the [FAQ](usage.md#faq)
3. Open a new issue with:
   - Error message
   - Steps to reproduce
   - Environment details (OS, Python version, etc.)

---

## Next Steps

After successful installation:

1. **Read the [Usage Guide](usage.md)** for running experiments
2. **Review [Configuration Examples](examples/)** for different scenarios
3. **Explore the [API Documentation](api/)** for extending the framework
