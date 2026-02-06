# Research Methodology

## Overview

This document describes the research methodology for comparing agentic exploration versus RAG methods for automated software patching on SWE-bench. The methodology is designed to ensure fair, reproducible, and statistically valid comparisons.

---

## Research Questions

### Primary Research Question

**Does agentic exploration produce higher signal-to-noise ratio in context gathering compared to RAG methods, leading to better patch generation for complex software issues?**

### Secondary Research Questions

1. **Localization Accuracy**: Do agentic methods more accurately identify the files and functions that need to be modified?

2. **Context Efficiency**: How does the token efficiency (signal-to-noise ratio) compare between methods?

3. **Issue Complexity**: Does the advantage of agentic methods increase with issue complexity?

4. **Cost-Effectiveness**: What is the trade-off between performance and computational cost?

5. **Generalization**: Do findings generalize across different repository types and issue categories?

---

## Hypothesis Formulation

### Main Hypothesis (H1)

> **H1**: Agentic exploration methods achieve higher signal-to-noise ratio in context gathering compared to RAG methods, resulting in higher resolution rates on SWE-bench.

**Signal-to-Noise Ratio Definition**:
```
Signal-to-Noise Ratio = (Relevant Context Tokens) / (Total Context Tokens)
```

Where:
- **Relevant Context**: Code chunks that are part of the gold patch or directly related to the issue
- **Total Context**: All code chunks included in the context window

### Sub-Hypotheses

**H1a (Localization)**: Agentic methods achieve higher Recall@k for file and function localization.

**H1b (Efficiency)**: Agentic methods achieve higher signal-to-noise ratio at equivalent context window sizes.

**H1c (Complexity)**: The performance gap between agentic and RAG methods increases with issue complexity (measured by number of files modified, issue description length, etc.).

### Null Hypothesis (H0)

> **H0**: There is no significant difference in signal-to-noise ratio or resolution rates between agentic exploration and RAG methods.

---

## Experimental Design

### Independent Variables

| Variable | Levels | Description |
|----------|--------|-------------|
| **Context Gathering Method** | 6 levels | AutoCodeRover, SWE-agent, Agentless, BM25, Dense, Hybrid |
| **Patch Generation Strategy** | 2 levels | Direct, Iterative |
| **LLM Model** | 2+ levels | GPT-4, Claude-3.5 (controlled variable) |

### Dependent Variables

| Variable | Type | Measurement |
|----------|------|-------------|
| **Resolution Rate** | Binary (aggregated as %) | Pass/Fail on SWE-bench test suite |
| **Localization Recall@k** | Continuous | % of gold files/functions in top-k predictions |
| **Signal-to-Noise Ratio** | Continuous | Relevant tokens / Total tokens |
| **CodeBLEU Score** | Continuous | 0.0-1.0 similarity to gold patch |
| **Semantic Entropy** | Continuous | Uncertainty in generated patches |
| **Token Usage** | Integer | Total tokens consumed |
| **Execution Time** | Continuous | Wall-clock time per instance |

### Controlled Variables

| Variable | Control Strategy |
|----------|------------------|
| **LLM Temperature** | Fixed at 0.0 for deterministic generation |
| **Context Window Size** | Fixed at 8,000 tokens for all methods |
| **Patch Generation Strategy** | Same generator used for all context gatherers |
| **Evaluation Environment** | Docker-based SWE-bench harness |
| **Random Seed** | Fixed for reproducibility |

---

## Evaluation Metrics Explained

### Primary Metric: Resolution Rate

**Definition**: Percentage of instances where the generated patch successfully resolves the issue.

**Calculation**:
```
Resolution Rate = (Instances Resolved / Instances Attempted) × 100%
```

**Success Criteria**:
1. All FAIL_TO_PASS tests pass (issue is fixed)
2. All PASS_TO_PASS tests pass (no regressions)

**Why This Matters**: Resolution rate is the gold standard for automated program repair. It directly measures whether the patch fixes the issue without breaking existing functionality.

### Secondary Metrics

#### 1. Localization Accuracy (Recall@k)

**File-level Recall@k**:
```
Recall@k = (Gold files in top-k predictions / Total gold files) × 100%
```

**Function-level Recall@k**:
```
Recall@k = (Gold functions in top-k predictions / Total gold functions) × 100%
```

**Why This Matters**: Good localization is necessary but not sufficient for successful patching. Measuring it separately helps understand where methods succeed or fail.

#### 2. Signal-to-Noise Ratio

**Calculation**:
```
SNR = (Number of relevant chunks) / (Total chunks retrieved)
```

A chunk is considered **relevant** if:
- It's from a file modified in the gold patch
- It contains a function modified in the gold patch
- It's directly referenced by the issue description

**Why This Matters**: Higher SNR means the LLM receives more useful information relative to distractions, potentially leading to better patches.

#### 3. CodeBLEU

**Components**:
- N-gram match (textual similarity)
- Weighted n-gram match (syntax importance)
- Syntax AST match (structural similarity)
- Data-flow match (semantic similarity)

**Range**: 0.0-1.0 (higher is more similar to gold patch)

**Why This Matters**: While not a substitute for functional correctness, CodeBLEU measures how "human-like" the generated patches are.

#### 4. Semantic Entropy

**Calculation**:
1. Generate N patches for the same context (N=5-10)
2. Cluster patches by semantic equivalence
3. Compute entropy across clusters

```
Semantic Entropy = -Σ(p_i × log(p_i))
```

Where p_i is the proportion of patches in cluster i.

**Why This Matters**: High semantic entropy indicates uncertainty or hallucination, which may correlate with incorrect patches.

#### 5. Token Usage

**Components**:
- Input tokens (context + prompt)
- Output tokens (generated patch)
- Total tokens = Input + Output

**Why This Matters**: Token usage directly translates to API cost. Understanding cost-effectiveness is crucial for practical deployment.

---

## Fair Comparison Principles

### 1. Identical LLM Configuration

All methods use the same:
- LLM model (e.g., GPT-4-turbo)
- Temperature (0.0)
- Max tokens (4096)
- Top-p (1.0)

### 2. Identical Patch Generation

The same `PatchGenerator` implementation is used for all context gatherers:
```python
# All methods use this same generator
generator = DirectPatchGenerator(llm_config, gen_config)

# For each method
for gatherer in [autocoderover, swe_agent, bm25, dense, hybrid]:
    context = gatherer.gather_context(instance, repo_path)
    patch_result = generator.generate_patch(context, instance)
```

### 3. Identical Evaluation

All patches are evaluated using the same:
- Docker sandbox environment
- SWE-bench test harness
- Timeout settings (300 seconds)
- Test execution protocol

### 4. Comparable Context Windows

All methods are configured to use the same context window size (8,000 tokens). This ensures:
- Fair comparison of information density
- Consistent LLM input size
- Comparable API costs

### 5. Multiple Runs for Stability

Each method is run multiple times (minimum 3) with different random seeds to:
- Account for LLM stochasticity
- Compute confidence intervals
- Ensure statistical significance

---

## Dataset Selection

### SWE-bench Variants

| Variant | Size | Use Case |
|---------|------|----------|
| **SWE-bench Lite** | 300 instances | Development, quick iteration |
| **SWE-bench Verified** | 500 instances | Primary evaluation |
| **SWE-bench Full** | 2,294 instances | Comprehensive evaluation |

### Recommended Splits

**Development Phase**:
- Use SWE-bench Lite
- Sample 50-100 instances for rapid iteration

**Primary Evaluation**:
- Use SWE-bench Verified
- Full 500 instances for reliable results

**Extended Analysis**:
- Use SWE-bench Full
- Stratified sampling by repository for generalization analysis

### Stratification Strategy

To ensure representative sampling, stratify by:
1. **Repository**: Different codebases have different characteristics
2. **Issue Complexity**: Number of files modified
3. **Issue Type**: Bug fix vs. feature implementation

---

## Statistical Analysis Plan

### 1. Descriptive Statistics

For each method, report:
- Mean and standard deviation for continuous metrics
- Proportions with confidence intervals for binary metrics
- Distribution plots for understanding variance

### 2. Comparative Analysis

**Pairwise Comparisons**:
- McNemar's test for resolution rates (paired binary outcome)
- Paired t-test for continuous metrics (e.g., CodeBLEU)
- Wilcoxon signed-rank test for non-normal distributions

**Multiple Comparisons**:
- ANOVA for comparing multiple methods
- Tukey's HSD for post-hoc pairwise comparisons
- Bonferroni correction for multiple hypothesis testing

### 3. Effect Size

Report effect sizes to quantify practical significance:
- Cohen's d for continuous metrics
- Odds ratio for binary outcomes
- Confidence intervals for all estimates

### 4. Correlation Analysis

Explore relationships between metrics:
- Pearson/Spearman correlation between localization and resolution
- Regression analysis to identify significant predictors

### 5. Subgroup Analysis

Analyze performance across different subgroups:
- By repository
- By issue complexity
- By issue type

---

## Threats to Validity

### Internal Validity

| Threat | Mitigation |
|--------|------------|
| **Implementation Bias** | Use reference implementations, code review |
| **Configuration Differences** | Standardize all hyperparameters |
| **Random Variation** | Multiple runs with different seeds |
| **Measurement Error** | Use established metrics, validate implementations |

### External Validity

| Threat | Mitigation |
|--------|------------|
| **Dataset Bias** | Use multiple SWE-bench variants |
| **Language Limitation** | Primarily Python; acknowledge limitation |
| **Temporal Validity** | Use SWE-bench-Live for recent issues |

### Construct Validity

| Threat | Mitigation |
|--------|------------|
| **Metric Appropriateness** | Use multiple complementary metrics |
| **Test Quality** | Use SWE-bench Verified with human-validated tests |
| **Patch Correctness** | Execution-based evaluation, not just similarity |

### Statistical Conclusion Validity

| Threat | Mitigation |
|--------|------------|
| **Low Statistical Power** | Use sufficient sample size (500+ instances) |
| **Violated Assumptions** | Use non-parametric tests when appropriate |
| **Multiple Comparisons** | Apply Bonferroni or FDR correction |

---

## Reproducibility Checklist

### Code Reproducibility

- [ ] All code is version controlled (Git)
- [ ] Exact commit hash is documented
- [ ] All dependencies are pinned (requirements.txt)
- [ ] Random seeds are set and documented
- [ ] Configuration files are included

### Data Reproducibility

- [ ] Dataset version is specified (e.g., SWE-bench Verified v1)
- [ ] Data preprocessing steps are documented
- [ ] Train/validation/test splits are fixed

### Experimental Reproducibility

- [ ] All hyperparameters are documented
- [ ] Hardware specifications are recorded
- [ ] Software versions are documented (Python, Docker, etc.)
- [ ] API versions are recorded (OpenAI, Anthropic)

### Results Reproducibility

- [ ] All results are saved with metadata
- [ ] Raw predictions are preserved
- [ ] Evaluation logs are stored
- [ ] Statistical analysis code is included

---

## Expected Outcomes

### Scenario 1: Agentic Methods Superior

If agentic methods significantly outperform RAG methods:
- **Implication**: Dynamic exploration is valuable for complex software issues
- **Recommendation**: Invest in better agent architectures and tools
- **Follow-up**: Study which agentic components contribute most

### Scenario 2: RAG Methods Competitive

If RAG methods achieve comparable performance:
- **Implication**: Simple retrieval may be sufficient for many issues
- **Recommendation**: Focus on improving retrieval quality and efficiency
- **Follow-up**: Identify when agentic exploration is truly needed

### Scenario 3: Hybrid Methods Best

If hybrid approaches (combining agentic and RAG) perform best:
- **Implication**: Both approaches have complementary strengths
- **Recommendation**: Develop adaptive methods that combine both
- **Follow-up**: Study optimal integration strategies

---

## Publication Plan

### Target Venues

1. **ICSE/ASE/FSE**: Top software engineering venues
2. **NeurIPS/ICML/ICLR**: ML venues with software engineering track
3. **arXiv**: Preprint for rapid dissemination

### Artifacts to Release

1. **Code**: Full framework implementation (GitHub)
2. **Data**: All experimental results and predictions
3. **Configurations**: All experiment configurations
4. **Logs**: Structured logs for analysis
5. **Models**: Any trained models or indexes

### Evaluation by Others

- Provide clear installation and usage instructions
- Include example configurations for replication
- Offer support through GitHub issues
- Consider benchmark submission for standardized comparison
